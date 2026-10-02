from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, JUMP_ACTIONS, Action
from .recovery import recover_decision
from .state import MarioSnapshot


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    jump_needed_probability: float | None = None
    danger_score: float | None = None
    jump_intent: str | None = None
    jump_intent_probabilities: Mapping[str, float] = field(default_factory=dict)
    selection: Mapping[str, Any] | None = None


class Policy(Protocol):
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


class TypeSafePolicy:
    def __init__(self) -> None:
        try:
            from typesafe_sdk import Choice, Score, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._Choice = Choice
        self._Score = Score
        self._client = TypeSafeClient()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _answer(response: Any, question_id: str, typed_collection: str) -> Any:
        collection = getattr(response, typed_collection, None)
        if collection is not None and question_id in collection:
            return collection[question_id]
        answers = getattr(response, "answers", None)
        if answers is not None and question_id in answers:
            return answers[question_id]
        raise KeyError(f"TypeSafe response omitted {question_id!r}")

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        prediction = snapshot.to_state().get("prediction")
        if prediction and prediction.get("status") == "available":
            return self._choose_with_forecast(snapshot, actions)
        # Filter only immediate requests. A future request can land before application.
        candidates = tuple(
            action
            for action in actions
            if snapshot.frames_until_action > 0
            or snapshot.grounded
            or snapshot.dy >= 0
            or action not in JUMP_ACTIONS
        ) or tuple(actions)
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in candidates}
        state = snapshot.to_state()
        delay = snapshot.frames_until_action
        landing = state["trajectory"]["landing_projection"]
        landing_time = landing.get("frames")
        horizon = delay + snapshot.decision_horizon_frames
        landing_context = {
            "assumption": "Reference only: constant observed relative velocity and enemy height; "
            "does not simulate candidate buttons, walls or ceilings. A false risk flag is not safety.",
            "application_frame": delay,
            "cycle_end_frame": horizon,
            "touchdown_frame_estimate": landing_time,
            "touchdown_during_requested_cycle": (
                landing_time is not None and delay < landing_time <= horizon
            ),
            "enemies": [],
        }
        if landing_time is not None:
            for enemy in state["hazard"].get("upcoming_enemies", []):
                velocity = enemy["relative_velocity_x"]
                at_landing = enemy["distance_pixels"] + velocity * landing_time
                at_end = enemy["distance_pixels"] + velocity * horizon
                vertical = enemy["vertical_offset_pixels"] + landing["standing_y"] - snapshot.y
                clearance = 16 + max(0, -velocity) * 8
                landing_context["enemies"].append(
                    {
                        "kind": enemy["kind"],
                        "distance_at_touchdown_pixels": at_landing,
                        "distance_at_cycle_end_pixels": at_end,
                        "vertical_offset_at_touchdown_pixels": vertical,
                        "clearance_needed_pixels": clearance,
                        "inadequate_clearance_after_touchdown": bool(
                            delay < landing_time <= horizon
                            and abs(vertical) <= 16
                            and at_landing >= -16
                            and at_end < clearance
                        ),
                    }
                )
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Choose Mario's next controller action to reach the flag alive.",
                    "timing": f"Your action begins after {snapshot.frames_until_action} emulator frames "
                    f"and then lasts {snapshot.decision_horizon_frames} frames. "
                    "First predict Mario's state after reaction_timing.scheduled_action runs for "
                    "frames_until_action frames. scheduled_first_frame_action is the actual first "
                    "button input, including any release needed before re-pressing A. Remaining "
                    "frames use scheduled_action. The observed player, terrain and hazards describe "
                    "NOW; all rules below and your chosen action concern the FUTURE application "
                    "state. With zero frames_until_action, act on the observed state immediately. "
                    "A fast API response does not apply early. A slow response pauses the emulator "
                    "at the boundary; it does not add more movement. Landing does not end the cycle. "
                    "Plan across landing and takeoff, apex, approaching enemies and pit edges. "
                    "A currently falling Mario may be grounded when your jump is applied; a "
                    "currently grounded Mario may already be rising under the scheduled jump. "
                    "Do not choose a jump that requires ground if the scheduled action will walk "
                    "off the edge first. Account for uncertainty in the predicted landing.",
                    "landing_reference": landing_context,
                    "priority_0_landing_safety": "Before repeating a jump or continuing forward, "
                    "check landing_reference for the COMPLETE next cycle. This airborne braking "
                    "rule applies only when touchdown_during_requested_cycle=true: landing must "
                    "occur AFTER this answer starts, not during the already committed action. "
                    "An enemy already under Mario may be stomped and disappear before application; "
                    "do not brake for an obsolete predicted touchdown after a stomp/bounce. "
                    "If an enemy has "
                    "inadequate_clearance_after_touchdown=true AND distance_at_touchdown_pixels>0 "
                    "AND visible terrain supports braking, choose LEFT early to leave space "
                    "before that enemy. This takes priority over repeating a jump, running forward "
                    "or precision-step instructions. The threat can occur AFTER a safe touchdown: "
                    "false landing_collision_predicted or false vertical_overlap_now does not "
                    "make the remaining grounded portion safe. Do not brake back toward an enemy "
                    "with distance_at_touchdown_pixels<=0 or over a pit. These reference estimates "
                    "assume constant velocity and do not simulate the chosen control input.",
                    "priority_0_precision_step": "If trajectory.precision_target_cleared is true, "
                    "and remains applicable at the predicted action boundary, choose RIGHT then: "
                    "release A to shorten the jump and land on the specified "
                    "precision_landing_target before the gap. This takes precedence over holding "
                    "a normal ascent. Only after landing on that step should you jump across "
                    "the gap. Extra height here overshoots the intermediate platform.",
                    "priority_1_jump_continuity": "When rising with A held, CONTINUE the forward "
                    "jump: right_jump after right_jump, or right_run_jump after right_run_jump. "
                    "Choosing right, right_run, noop or left RELEASES A and cuts the jump short. "
                    "Do not release A merely because Mario is already airborne or currently above "
                    "an enemy. Preserve height until the apex unless deliberately shortening the "
                    "jump to land on a specific visible platform. There is no automatic jump hold.",
                    "priority_2_takeoff": "On ground, jump over nearby blocking obstacles or gaps "
                    "with a reachable landing. If jump_must_start_this_decision is true, choose "
                    "a forward jump at application time if ground support is still available. "
                    "Also jump forward for rear_contact_imminent if predicted grounded. "
                    "For enemies prefer a running jump at the takeoff window; "
                    "do not start early just because an enemy is visible far away. For stairs, "
                    "land on the next step before committing across the gap. An already missed "
                    "deadline is urgent: braking does not instantly stop Mario.",
                    "priority_3_progress": "On clear supported ground with no urgent takeoff, "
                    "prefer right_run to build forward momentum. Use right for deliberate slower "
                    "precision movement, not as the default. When blocked, jump onto the obstacle "
                    "rather than repeatedly walking into it.",
                    "priority_4_landing": "When falling and hazard.landing_collision_predicted is true "
                    "with landing_braking_supported=true AND landing_threat_from_behind=false, "
                    "and braking will still help at application time, choose LEFT briefly to land before "
                    "the enemy, then jump again from ground. If landing_threat_from_behind=true, "
                    "choose right_run to move away; do NOT brake toward an enemy behind or below. "
                    "Do not wait for vertical_overlap_now: "
                    "the enemy can be harmless now but collide at landing. Otherwise, when falling, "
                    "A cannot start another jump. Keep "
                    "forward movement toward visible support; release A for the next takeoff. "
                    "Brake only to reach a specific landing, not just because a prediction is "
                    "unknown. Compare player.center_x with landing_surfaces safe_center intervals. "
                    "Landing projections are approximate and ignore ceilings and walls. Missing "
                    "predictions during ascent are normal, not evidence that the jump is unsafe.",
                    "facts": "Grounded jump macros release a previously held A for one frame "
                    "before re-pressing; one-frame macros need another jump decision to press. "
                    "Terrain preview distances are rebased but check their age. Unknown terrain "
                    "is not clear ground. Check all upcoming enemies and the next landing.",
                },
                criteria=criteria,
            ),
            "jump_intent": self._Choice(
                instructions="What should the jump button do at the FUTURE action application time? "
                "First predict frames_until_action frames under scheduled_action (using "
                "scheduled_first_frame_action for the first frame). Landing does not replan early. "
                "Use player.jump_phase, can_start_jump, terrain and trajectory. If precision_target_cleared, release A to land on the intermediate step. This is an "
                "independent diagnostic, not another controller action. Do not assume access "
                "to the next_action answer.",
                criteria={
                    "start": "Begin a new jump from confirmed ground, allowing release/repress.",
                    "hold": "Continue holding A during ascent to gain necessary height.",
                    "release": "Release A to shorten ascent or prepare for the next grounded jump.",
                    "none": "No change to the jump button is needed; do not initiate a jump.",
                },
            ),
            "danger": self._Score(
                instructions="Estimate risk of death or damage within "
                "reaction_timing.total_reaction_horizon_frames if scheduled_action continues "
                "(or recent_control.action for the initial zero-delay request). "
                "Use 0 for safe supported travel, 1 for a plausible threat requiring a "
                "correction, and 2 for imminent collision or an unsupported fall. This is a "
                "risk score, not a calibrated probability. Treat uncertain terrain cautiously.",
                criteria=[
                    "Safe supported movement throughout the reaction horizon",
                    "Plausible collision or unsupported landing unless control changes",
                    "Imminent death or damage within the reaction horizon",
                ],
            ),
        }
        # Only add the corrective question context when the requested cycle can
        # actually affect this landing. An obsolete landing/stomp prediction must
        # not bias a later action; ordinary states retain the established questions.
        actionable_landing_risk = any(
            enemy["inadequate_clearance_after_touchdown"]
            and enemy["distance_at_touchdown_pixels"] > 0
            for enemy in landing_context["enemies"]
        )
        if not actionable_landing_risk:
            questions["next_action"].instructions.pop("landing_reference")
            questions["next_action"].instructions.pop("priority_0_landing_safety")
        if state["hazard"]["overhead_threats"]:
            questions["next_action"].instructions["question"] = (
                "An overhead enemy may drop into Mario's rising path. Choose the next controller "
                "action to pass below it alive. Prefer RIGHT to release A on supported ground."
            )
            questions["next_action"].instructions["priority_1_jump_continuity"] = (
                "Suspend normal jump continuity for this overhead hazard. Holding A can raise "
                "Mario into the enemy; release A with RIGHT while moving below it. Only preserve "
                "jump height if needed to avoid an actual pit. An elevated enemy is not a "
                "ground-level enemy to jump over."
            )
            questions["next_action"].instructions["priority_0_overhead_escape"] = (
                "hazard.overhead_threats identifies an enemy above Mario that is falling or "
                "near a platform edge and may drop into his path during the reaction horizon. "
                "Negative relative_y means ABOVE; positive vertical_velocity_y means the enemy "
                "is falling; positive relative_velocity_y means the vertical gap is closing. "
                "A platform-edge warning anticipates a possible fall, not a certain collision. "
                "Predict the committed action first. If it would carry Mario underneath this "
                "enemy, prefer RIGHT to release A, cut ascent and pass underneath on supported "
                "ground. Do not jump up into an enemy above or keep holding A into it when a wall "
                "blocks horizontal escape. This takes priority over jump continuity, obstacle "
                "jumping and landing braking. Do not automatically retreat LEFT toward an "
                "enemy slightly behind. Check ground support: preserve a necessary gap crossing "
                "if releasing A would cause a pit fall. Resume normal jumping after the "
                "overhead path is clear. Horizontal extrapolation ignores walls and controls."
            )
            questions["jump_intent"].instructions += (
                " For hazard.overhead_threats, release A to avoid rising into a dropping enemy "
                "when there is supported ground below; continuing ascent is not mandatory."
            )
            for action in JUMP_ACTIONS:
                if action.value in questions["next_action"].criteria:
                    questions["next_action"].criteria[action.value] += (
                        " CAUTION in this state: continuing upward can collide with the overhead "
                        "enemy. Prefer releasing A unless a pit requires this jump height."
                    )
        gap_window = state["terrain"].get("gap_takeoff_window")
        if gap_window and gap_window["must_jump_this_cycle"]:
            questions["next_action"].instructions["question"] = (
                "This is the last safe cycle to start a jump before the visible pit. "
                "Choose a forward jump at application time if the landing is reachable."
            )
            questions["next_action"].instructions["priority_0_gap_takeoff"] = (
                "terrain.gap_takeoff_window predicts that ground support is available at action "
                "application but continuing to run for the whole cycle passes the safe takeoff "
                "edge. Choose RIGHT_RUN_JUMP now for a reachable crossing; waiting another "
                "decision can leave Mario airborne and unable to jump. gap_ahead=false describes "
                "only the current tile distance and does not cancel this deadline. This takes "
                "priority over routine running and passing underneath overhead enemies. If the "
                "jump route is blocked or the landing unreachable, brake before the edge instead "
                "of running off it. Evaluate the visible landing surfaces and enemies."
            )
            questions["jump_intent"].instructions += (
                " terrain.gap_takeoff_window.must_jump_this_cycle means this is the last safe "
                "cycle to START a grounded jump for a reachable pit crossing."
            )
        reach = state["terrain"].get("jump_reach_reference")
        if reach and reach["approach_before_jumping"]:
            questions["next_action"].instructions["question"] = (
                "A jump from the predicted application position at the current speed is too "
                "short, or its far landing is not yet visible. Choose RIGHT to move closer on the visible supported "
                "approach before jumping."
            )
            questions["next_action"].instructions["priority_0_jump_distance"] = (
                "terrain.jump_reach_reference compares the full takeoff-to-landing distance "
                "and landing height, not just pit width. At the future application state, "
                "repeating a jump from this platform is too early or lacks a confirmed landing. "
                "Choose RIGHT without A to approach, allow a short drop to the visible lower "
                "floor, then jump from confirmed support closer to the gap. This takes priority "
                "over repeating the previous jump. Use deliberate RIGHT rather than racing "
                "across a short approach before landing. Do not jump in midair after walking off "
                "a step. Re-evaluate at each boundary; do not walk past the safe takeoff edge. "
                "The estimate ignores collisions and acceleration. A running jump may travel "
                "farther, but pressing B does not instantly grant maximum speed or prove reach."
            )
            questions["jump_intent"].instructions += (
                " If jump_reach_reference.approach_before_jumping is true, release A and move "
                "closer on the supported approach before the next grounded takeoff."
            )
            questions["next_action"].instructions["priority_2_takeoff"] = (
                "Defer the next grounded takeoff here: a visible platform under Mario is not "
                "automatically a good launch point. Move closer with RIGHT on the supported "
                "approach, check other hazards, and re-evaluate the far landing before jumping."
            )
            if Action.RIGHT.value in questions["next_action"].criteria:
                questions["next_action"].criteria[Action.RIGHT.value] = (
                    "Recommended supported approach: move right and release A, reduce distance "
                    "to the far landing, then take off closer to the gap from confirmed ground."
                )
            for action in JUMP_ACTIONS:
                if action.value in questions["next_action"].criteria:
                    questions["next_action"].criteria[action.value] = (
                        "Start/hold A now. A fresh takeoff here is premature for the current "
                        "speed or unconfirmed far landing; prefer supported RIGHT approach. "
                        "Only jump if another verified route or immediate hazard requires it."
                    )
        pipe_plants = state["hazard"].get("pipe_plants", [])
        if pipe_plants:
            waiting = any(p["wait_before_pipe_jump"] for p in pipe_plants)
            questions["next_action"].instructions["priority_0_pipe_plant"] = (
                "Piranha plants cannot be stomped. hazard.pipe_plants reports their engine phase: "
                "extended/rising/retracting are still exposed; only hidden means fully inside "
                "the pipe. Before jumping onto an occupied pipe, wait on supported ground until "
                "the plant is hidden. Use NOOP when stopped safely beside the pipe; use LEFT "
                "briefly to brake forward momentum if needed, checking rear enemies and support. "
                "NOOP does not instantly stop motion. Do not repeat jumps into the exposed plant "
                "or keep running into its mouth. Stay beside the pipe during the wait; once "
                "hidden, proximity within 33 pixels prevents it starting to emerge while you "
                "remain close. Then jump forward onto/over the pipe and preserve required ascent. "
                "The hidden pause is finite if Mario moves away. Never wait motionless in midair "
                "or over a pit: if already above the pipe, continue toward safe support. "
                "Check the FUTURE application state after scheduled_action; timers are estimates."
            )
            if waiting:
                questions["next_action"].instructions["question"] = (
                    "An exposed piranha plant occupies the pipe ahead. Choose an action to wait "
                    "safely beside the pipe until it fully retracts, then jump onto it."
                )
                questions[
                    "jump_intent"
                ].instructions += " Do not start a pipe jump while wait_before_pipe_jump is true; wait for hidden."
            if any(p["wait_before_pipe_jump"] and p["distance_pixels"] <= 48 for p in pipe_plants):
                normal = questions["next_action"].instructions
                questions["next_action"].instructions = {
                    "question": "Wait safely beside the occupied pipe until the piranha plant is "
                    "fully hidden. Prefer NOOP to release A and let Mario land beside the pipe.",
                    "timing": normal["timing"],
                    "plant_safety": normal["priority_0_pipe_plant"],
                    "waiting": "If below the exposed plant, do not start or continue jumping "
                    "toward it. Release A and forward drive to land on the supported floor beside "
                    "the pipe. Remaining stationary here is intentional. NOOP while stopped; "
                    "LEFT only when braking is necessary and rear terrain/enemies allow it. "
                    "A pipe blocking horizontal movement does not justify jumping while exposed. "
                    "If already above the plant with ample clearance at application, continue "
                    "across toward safe support. Never wait over a pit. Resume forward jumping "
                    "when the plant is observed hidden, not merely retracting.",
                }
                descriptions = {
                    Action.NOOP: "Release all buttons; land and wait beside the occupied pipe. "
                    "Preferred when safely stopped below its rim. Does not instantly stop inertia.",
                    Action.LEFT: "Brake or move left if needed to avoid entering the exposed plant; "
                    "check rear enemies and ground support.",
                    Action.RIGHT: "Move right without jumping; only for a safe approach or escape.",
                    Action.RIGHT_RUN: "Run right without jumping; can carry Mario into danger.",
                    Action.JUMP: "Press A without horizontal input. Unsafe below an exposed plant; "
                    "wait for it to hide before starting a pipe jump.",
                    Action.RIGHT_JUMP: "Move right and hold A. Only continue if already safely "
                    "above the plant; otherwise release A and wait below the pipe rim.",
                    Action.RIGHT_RUN_JUMP: "Run right and hold A. Only continue if already safely "
                    "above the plant; otherwise release A and wait below the pipe rim.",
                }
                questions["next_action"].criteria = {
                    action.value: descriptions[action] for action in candidates
                }
        if state["terrain"].get("stair_approach"):
            timing = questions["next_action"].instructions["timing"]
            stair_guidance = (
                "terrain.stair_approach predicts supported ground at application, but a rising "
                "staircase is still more than 48 pixels ahead at walking speed. Do not bounce "
                "immediately after landing: a slow early jump peaks before reaching the stairs "
                "and descends into enemies higher up. Prefer RIGHT without A for this supported "
                "approach cycle; re-evaluate the jump once closer. Check ALL enemies, including "
                "a second enemy above/behind the first: stomping one is not a safe route through "
                "the group. If an immediate enemy makes approaching unsafe and a jump is needed, "
                "use RIGHT_RUN_JUMP from the grounded takeoff, not a slow jump. Pressing B only "
                "after takeoff cannot be relied on to repair a short jump. This guidance applies "
                "to the predicted grounded application, not a jump already in progress."
            )
            questions["next_action"].instructions = {
                "question": "Walk RIGHT toward the distant staircase before starting a new jump. "
                "The predicted application is grounded and the next walking cycle is supported. "
                "Do not immediately repeat the previous slow jump after landing.",
                "timing": timing,
                "stair_approach": stair_guidance,
            }
            for action in candidates:
                if action == Action.RIGHT:
                    questions["next_action"].criteria[action.value] = (
                        "Preferred: release A and walk right for the supported approach cycle. "
                        "Get closer before jumping; the staircase is still too far away."
                    )
                elif action in JUMP_ACTIONS:
                    questions["next_action"].criteria[action.value] = (
                        "A fresh jump here is premature for these stairs. Only a verified "
                        "immediate enemy requiring takeoff justifies jumping now; use running "
                        "takeoff then, rather than repeating a slow walking jump."
                    )
            questions["jump_intent"].instructions = (
                "The next action starts after landing before distant stairs. Release A if held, "
                "otherwise none, while approaching on supported ground. Start only for an "
                "immediate hazard requiring a grounded running jump."
            )
        exit_pipe = state["terrain"].get("side_exit_pipe")
        if exit_pipe:
            timing = questions["next_action"].instructions["timing"]
            questions["next_action"].instructions = {
                "question": "Enter the sideways exit pipe to continue this stage. Choose the "
                "action that aligns Mario with its LEFT-facing mouth and walks RIGHT into it.",
                "timing": timing,
                "pipe_entry": "terrain.side_exit_pipe identifies an actual horizontal pipe mouth, "
                "not an obstacle to jump over. Release A. When entering=true, use RIGHT without "
                "A and allow the automatic area transition to finish. Entering the pipe is not "
                "stage completion; after the transition continue toward the flag.",
                "alignment": "Compare predicted application x/y with mouth_x and entry_standing_y. "
                "If above the entry floor AND over/right of the mouth, Mario is on top of the "
                "pipe: walking right hits its vertical shaft and repeated jumps cannot help. "
                "When approach_supported=true, use LEFT without A to retreat off the left lip "
                "toward retreat_target_x. Account for the already scheduled movement: once it "
                "will reach that target, use RIGHT to brake leftward drift and descend onto "
                "the supported entry floor. If already left of the mouth, release A and land "
                "on that floor, then walk RIGHT into the mouth while grounded. Do not wait on "
                "top of the pipe or jump onto its rim again. Small temporary backward progress "
                "is necessary to get down from the rim. Check rear hazards/support before "
                "retreating; unsupported approach is not permission to walk into a pit.",
            }
            descriptions = {
                Action.RIGHT: "Release A; walk right into the pipe at entry floor height, or "
                "brake leftward drift after retreating off its lip. Continue during entry.",
                Action.LEFT: "Release A and retreat off the pipe's left lip when above the mouth; "
                "aim for retreat_target_x with supported ground and no rear enemy.",
                Action.NOOP: "Release buttons and descend if already safely left of the mouth. "
                "Does not stop inertia; standing still on top of the pipe cannot enter it.",
                Action.RIGHT_RUN: "Move right fast without A; less precise than RIGHT for entry.",
                Action.JUMP: "Press A: moves away from the entry floor; avoid at this pipe.",
                Action.RIGHT_JUMP: "Jump right: climbs onto the pipe or hits its shaft; avoid.",
                Action.RIGHT_RUN_JUMP: "Run/jump right: overshoots the entry or hits the shaft; avoid.",
            }
            questions["next_action"].criteria = {
                action.value: descriptions[action] for action in candidates
            }
            questions["jump_intent"].instructions = (
                "At a sideways exit pipe, release A to descend to its entry floor and walk "
                "right inside. Choose release if A is held, otherwise none. Do not start or "
                "hold a jump to overcome the pipe shaft. Allow automatic entry to finish."
            )
        return self._request(state, questions, candidates)

    def _choose_with_forecast(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        state = snapshot.to_state()
        candidates = tuple(actions)
        descriptions = {
            Action.NOOP: "Release all buttons and coast; inertia remains.",
            Action.RIGHT: "Move right with A released.",
            Action.RIGHT_RUN: "Run right with A released.",
            Action.RIGHT_JUMP: "Move right holding A.",
            Action.RIGHT_RUN_JUMP: "Run right holding A.",
            Action.JUMP: "Hold A without direction; horizontal inertia remains.",
            Action.LEFT: "Brake rightward momentum, then move left; release A.",
        }
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Choose the next controller action to reach the flag alive.",
                    "timing": "prediction.committed_future is the state AFTER the already "
                    "scheduled buttons finish. Your answer starts then and lasts exactly "
                    "action_cycle_frames. All forecast frames are offsets from observation. "
                    "You cannot change the committed prefix, even with a fast response.",
                    "evidence": "Compare prediction.action_forecasts for ALL allowed actions. "
                    "Native engine rollouts include acceleration/braking, jumps, walls, pits, "
                    "all enemies, turns, stomps and subsequent bounces. Prefer these outcomes "
                    "over legacy constant-velocity hazard flags or generic jump-continuity "
                    "rules. A safe terrain surface only confirms support, not enemy clearance.",
                    "continuations": "Each candidate lasts one cycle. The repeat branch repeats "
                    "that candidate at subsequent boundaries; run_jump switches to forward "
                    "running jumps at the NEXT boundary. Both simulate release/repress. "
                    "Survived_horizon is conditional on that sequence, not proof that any later "
                    "choice is safe. Compare corresponding branches, not just the "
                    "first cycle. Do not reject early braking because repeating LEFT forever "
                    "is bad when LEFT then run_jump safely clears the enemy group.",
                    "priority": "Avoid predicted death or damage when another candidate has "
                    "a survivable continuation. Start the correction now even if collision "
                    "occurs several cycles later: late braking may be ineffective. Compare "
                    "enemy_encounters rows (see encounter_columns and encounter_objects) ahead, behind and "
                    "above; a stomp of the first enemy is not safety from the second. Among "
                    "survivable routes prefer forward progress with useful clearance. Do not "
                    "wait or retreat indefinitely if a forward route survives. If every "
                    "sampled continuation fails, these continuations are not exhaustive; "
                    "choose the action preserving the best escape opportunity using terrain "
                    "and contact timing. Unknown/terminated is not safe.",
                    "navigation": "Use terrain for route intent beyond the finite horizon. "
                    "At side_exit_pipe, descend to its entry floor and walk right into its "
                    "mouth; climbing its shaft does not advance. Plants cannot be stomped. "
                    "Enemy origin distances are approximate references; native death/damage "
                    "outcomes are stronger evidence than a distance threshold. Only your "
                    "next_action controls the game; diagnostic answers do not change buttons.",
                    "interactions": "interactables describes useful objects, not enemies. For "
                    "a springboard below/beside Mario, land on its top and use its bounce to "
                    "gain height. If running jumps hit a tall wall, briefly retreat or release "
                    "horizontal input to align over the spring, then jump with the launch. "
                    "Compare forecast max_height_y and forward progress over the longer path. "
                    "For plants, zero velocity does not mean permanently stationary: compare "
                    "plant phase/timer fields and wait_for_clearance. Wait only on supported "
                    "ground, checking other enemies; resume when hidden, not just retracting.",
                    "recovery": "When recovery.active is true, repeated actions have failed to "
                    "make progress. Seek a different route or interaction, including temporary "
                    "retreat, walking into a pipe, landing on a spring, or releasing A. Check "
                    "action_frames to avoid the same ineffective attempts. A survivable loop "
                    "with no progress is not a solution. Prefer a candidate whose forecast "
                    "actually escapes; do not endlessly defer the useful action to a future "
                    "continuation. The bounded recovery selector can sample a safe alternative "
                    "to a repeated proposal and records the original proposal separately.",
                },
                criteria={a.value: descriptions[a] for a in candidates},
            ),
            "jump_intent": self._Choice(
                instructions="Describe the desired jump input at prediction.committed_future, "
                "using the action forecasts and terrain. This is an independent diagnostic, "
                "not another controller. Grounded jump macros rearm held A for one frame.",
                criteria={
                    "start": "Start a grounded jump.",
                    "hold": "Maintain jump ascent.",
                    "release": "Release A to cut height or prepare takeoff.",
                    "none": "No jump input needed.",
                },
            ),
            "danger": self._Score(
                instructions="Estimate danger of continuing current movement using committed_future "
                "and the matching repeat action forecast. 0 means no predicted damage in the "
                "finite horizon, 1 means an avoidable threat, 2 means imminent or committed "
                "death/damage. This is a diagnostic risk score, not a probability.",
                criteria=["No predicted harm", "Threat requiring correction", "Imminent harm"],
            ),
        }
        return self._request(state, questions, candidates)

    def _request(
        self, state: dict[str, Any], questions: dict[str, Any], candidates: Sequence[Action]
    ) -> Decision:
        started = time.perf_counter()
        response = self._client.system_one(state=state, questions=questions)
        latency_ms = (time.perf_counter() - started) * 1000

        action_answer = self._answer(response, "next_action", "choices")
        jump_answer = self._answer(response, "jump_intent", "choices")
        danger_answer = self._answer(response, "danger", "scores")
        action = Action(str(action_answer.choice))
        if action not in candidates:
            raise ValueError(f"TypeSafe action {action.value!r} is not allowed")
        probabilities = {
            str(key): float(value) for key, value in dict(action_answer.probabilities).items()
        }
        decision = Decision(
            action=action,
            confidence=float(action_answer.confidence),
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=sum(
                float(jump_answer.probabilities.get(intent, 0.0)) for intent in ("start", "hold")
            ),
            jump_intent=str(jump_answer.choice),
            jump_intent_probabilities=dict(jump_answer.probabilities),
            danger_score=float(danger_answer.score),
        )
        return recover_decision(state, decision, tuple(candidates))


class HeuristicPolicy:
    """Offline smoke-test policy; not intended as the Mario benchmark baseline."""

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        allowed = set(actions)
        action = Action.RIGHT_RUN if Action.RIGHT_RUN in allowed else actions[0]
        if snapshot.stalled_steps >= 2 and Action.RIGHT_RUN_JUMP in allowed:
            action = Action.RIGHT_RUN_JUMP
        return Decision(
            action=action,
            confidence=1.0,
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
        )
