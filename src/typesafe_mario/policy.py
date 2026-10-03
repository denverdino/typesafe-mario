from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, JUMP_ACTIONS, Action
from .observation import model_state
from .risk import assess_risk
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
        state = model_state(snapshot)
        candidates = tuple(actions)
        if not candidates:
            raise ValueError("At least one controller action is required")
        risk_control = assess_risk(state.get("prediction"), candidates, state=state)
        candidates = tuple(Action(a) for a in risk_control["candidate_actions"])
        state["risk_control"] = risk_control
        prediction = state.get("prediction")
        if prediction and prediction.get("continuation_search"):
            # Keep complete trajectories in local diagnostics, but send concise
            # first-cycle evidence and conditional plans to the action classifier.
            state["prediction"] = {
                k: prediction[k]
                for k in (
                    "backend",
                    "version",
                    "status",
                    "observation_frame",
                    "application_frame",
                    "action_cycle_frames",
                    "committed_future",
                    "committed_interactions",
                    "plan_followup",
                    "continuation_search",
                    "observed_interactions",
                    "observed_boundary",
                    "navigation_goal",
                    "enemy_crossing",
                    "recent_motion",
                    "actor_forecasts",
                    "validation",
                )
                if k in prediction
            }
            state["prediction"]["action_forecasts"] = [
                {
                    k: b[k]
                    for k in (
                        "action",
                        "first_cycle",
                        "first_landing",
                        "first_cycle_risk",
                        "contact_exposure",
                        "control_risk",
                        "continuation",
                        "hazards",
                    )
                    if k in b
                }
                for b in prediction["action_forecasts"]
            ]
        criteria = {a.value: ACTION_DESCRIPTIONS[a] for a in candidates}
        for branch in state.get("prediction", {}).get("action_forecasts", ()):
            name, plan = branch["action"], branch.get("continuation")
            if name in criteria and plan:
                criteria[name] += (
                    f" Estimated continuation: {plan['status']}; progress {plan['progress']} px; "
                    f"utility {plan['score']} (higher favors progress with fewer close contacts); "
                    f"failure {plan['failure']}; warnings {plan['warnings']}; "
                    f"sequence {plan['actions']}."
                )
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Choose Mario's next controller action to reach the flag alive.",
                    "timing": "Current player/terrain/actors are observations. Your action starts "
                    "after reaction_timing.frames_until_action and lasts action_horizon_frames. "
                    "prediction.committed_future estimates the application state after the already "
                    "scheduled input; it is uncertain and cannot be changed by your answer.",
                    "evidence": "prediction.action_forecasts use approximate dynamics fitted to "
                    "real observed motion. No emulator trials or hidden future information are "
                    "available. Compare first_cycle, position ranges, visible enemy contact risks "
                    "and continuation warnings for all candidates. Ranges have no "
                    "guaranteed coverage. no_contact_predicted is not proof of safety. A nominal "
                    "landing or clear path is not a verified outcome. Enemy turns, stomps, springs, "
                    "pipe entry and newly appearing actors can invalidate the estimates.",
                    "interactions": "observed_interactions identifies Goombas inferred defeated "
                    "from an actual bounce, stationary body and score increase. Their bodies "
                    "can remain in visible_actors temporarily; local forecasts exclude them. "
                    "This does not establish a future stomp or eliminate Koopa shell hazards. "
                    "continuation.interactions may model a conditional Goomba bounce to check "
                    "later enemies and terrain. possible_stomp/uncertain_bounce remain warnings: "
                    "neither that stomp nor survival is established. Koopa shell interactions stop "
                    "the estimate; do not assume a shell disappears. "
                    "A top-down stomp on a Koopa makes a stationary shell, not a defeated enemy. "
                    "observed_interactions and actor_forecasts.state distinguish inferred "
                    "stationary_shell and moving_shell from ordinary Koopas using actual stomp "
                    "and motion evidence. A future possible stomp is not that evidence. "
                    "Land on supported ground before considering walking into a confirmed stationary "
                    "shell to kick it horizontally away from Mario. Never deliberately walk into "
                    "a moving shell or a merely motionless unconfirmed Koopa. A moving shell "
                    "can bounce back from walls or pipes and kill Mario: inspect both outbound "
                    "and return paths, leave headroom to jump over the return, and reobserve "
                    "after kicking. kick_forecasts are conditional trajectories with an explicit "
                    "speed prior; their frame offsets start at a hypothetical kick, not now. "
                    "A kick branch remains uncertain and does not prove the shell clears enemies.",
                    "landing_window": "first_landing and continuation.landings report nominal "
                    "landing positions, enemy body gaps, braking margins and time until another "
                    "action can apply. Check all enemies beyond the first one before takeoff. "
                    "landing_contact_window means the predicted landing leaves insufficient "
                    "reaction margin; a second jump cannot be assumed immediate. A low ceiling "
                    "can prevent jump clearance. unresolved_flight means the search has not "
                    "established a completed landing even if ground exists below. These are "
                    "approximate margins, not safety guarantees. uncertain_wall_contact means "
                    "a descending corner contact may not stop horizontal momentum: do not use "
                    "that predicted wall as a brake before an enemy. uncertain_takeoff means "
                    "the scheduled input predicts landing near the application boundary: A may "
                    "be pressed while still airborne and remain held through landing without "
                    "starting a jump. Inspect committed_future.landing_timing; prefer an "
                    "eligible route that does not depend on this unconfirmed takeoff.",
                    "plan_continuity": "prediction.plan_followup rechecks the remaining route "
                    "from the previously selected action against the latest observations. When "
                    "its next step remains eligible and the rechecked route is complete and "
                    "warning-free, favor following through on run-up and takeoff instead of "
                    "repeatedly postponing them. New contact, landing or terrain warnings take "
                    "precedence. committed_interactions are possible contacts during the already "
                    "scheduled input: a conditional stomp can leave Mario bouncing rather than "
                    "grounded when your answer applies. If committed_future.valid_for_application "
                    "is false, its position is where projection stopped, not a valid application "
                    "state; do not infer a new grounded jump from it.",
                    "motion_evidence": "recent_motion contains actual executed input segments "
                    "and observed positions. Use them to check whether a prior action produced "
                    "progress or braking. observed_boundary, when present, infers a left movement "
                    "limit from repeated actual left input at the visible edge. Retreat cannot "
                    "create more separation beyond that estimated limit. "
                    "actor_forecasts gives each visible enemy's velocity "
                    "estimated from contiguous observations and three approximate future positions. "
                    "Trajectory frame and hazard possible_contact_frame are offsets from observation_frame. "
                    "Hazards assume repeating that input; a different continuation may avoid them. "
                    "validation reports past first-cycle prediction errors and heuristic range coverage; "
                    "large errors or few samples call for extra separation, not confidence in a close pass.",
                    "continuation": "Use continuation to compare feasible action sequences: "
                    "prefer complete estimated_viable plans without warnings, then uncertain "
                    "plans, and predicted_failure last; compare utility within that risk tier. "
                    "Do not buy extra progress by skimming enemies when a clear plan exists. Warnings describe "
                    "uncertainty; a minor edge touch above visible floor need not prevent progress. "
                    "The control_risk fields repeat one input and may warn of avoidable later hazards. "
                    "Only the first cycle will be executed before replanning. "
                    "Grounded jump and swimming stroke inputs rearm A for one frame when needed. "
                    "Do not interpret later samples as a committed route or a guaranteed landing.",
                    "risk_control": "Prioritize survival and room to brake over forward speed. "
                    "risk_control explains candidate filtering using bounded action continuations. "
                    "Excluded actions have a contact, fall or visibility warning when a lower-risk "
                    "alternative exists. Remaining actions are not certified safe. Nominal body "
                    "support can be uncertain near a gap when position ranges cross its edge; "
                    "uncertain_landing also reports that grounded support margin. Nominal body "
                    "contact is stronger evidence than overlap of uncertainty ranges alone. "
                    "If no lower-risk alternative can be identified, all remain available: "
                    "compare earliest contact, braking and "
                    "visible support. contact_exposure compares nominal overlapping frames, "
                    "summed overlap area and final overlap within the executable cycle. When all "
                    "inputs predict contact, favor reducing these exposures rather than progress "
                    "before contact; this does not establish survival. Jumping beneath a low ceiling can cause an early landing "
                    "in front of an enemy; holding A cannot jump again until release/repress. "
                    "Do not repeat a forward jump just because jump_intent suggests holding A.",
                    "navigation": "Favor forward progress while retaining room to react to visible "
                    "hazards. prediction.enemy_crossing identifies observed approaching walkers "
                    "on supported open ground. Use that space to prepare a directional jump across "
                    "the group, checking headroom and the landing beyond all enemies. A brief run-up "
                    "or reposition can help; repeated waiting/retreat can let enemies consume the "
                    "jumping space. Do not postpone the same future jump every decision or retreat "
                    "under low bricks merely because the first cycle is clear. The hint never "
                    "overrides contact, gap or uncertain-landing evidence. "
                    "When prediction.navigation_goal identifies a ceiling-sealed platform, "
                    "retreat and descend toward its lower-floor target; continuation utility then "
                    "measures progress toward that local goal, so negative horizontal progress can "
                    "be useful. It is a geometric proposal, not a safety guarantee. "
                    "terrain.blocks are visible solid rectangles in world coordinates, "
                    "x rightward and y upward, on the same support plane as player.y. Mario spans "
                    "x..x+16 and y..y+height_pixels. Beyond terrain.known_x is unknown. At a visible "
                    "side_exit_pipe, align with entry_standing_y and walk right into the mouth. "
                    "A springboard may enable a higher jump, but its bounce is not modelled. "
                    "Plants may emerge or retreat; no internal timers are known. Obtain fresh "
                    "observations before treating a previously occupied pipe as clear.",
                    "traversal": "For a visible gap, keep A held through ascent until enough "
                    "horizontal distance is covered to reach the far support. Releasing A early "
                    "shortens the jump and can make a previously feasible crossing fail. "
                    "Prefer right_run_jump when a gap needs distance, and a directional jump "
                    "when a wall or pipe blocks walking. Near a ceiling compare the actual "
                    "predicted contact/landing, not just the jump button. An enemy on a high "
                    "platform may fall into the path below; vertical separation is temporary.",
                    "recovery": "recovery.active reports no meaningful forward progress. Consider "
                    "releasing A, braking, a brief retreat or a different visible route. Avoid "
                    "endlessly repeating an ineffective action; backward movement can be useful. "
                    "Geometric route cues are advisory. Observed stationary wall pushes can "
                    "exclude inputs that still "
                    "produce no motion when a complete clear alternative exists. Choose a "
                    "different useful input now; do not postpone the same planned jump forever. "
                    "recovery.local_motion records small back-and-forth loops as well as waiting. "
                    "When a clear immediate climb is available, another retreat/wait cycle is "
                    "not progress; take the useful first step now. "
                    "Your selected next_action is used "
                    "unchanged; no route or safety selector replaces it.",
                },
                criteria=criteria,
            ),
            "jump_intent": self._Choice(
                instructions="Describe desired jump/stroke input at estimated application time. "
                "This is diagnostic only; next_action controls the buttons.",
                criteria={
                    "start": "Start a grounded jump or swim stroke.",
                    "hold": "Continue ascent or repeated swimming strokes.",
                    "release": "Release A to descend or prepare another jump.",
                    "none": "No jump input needed.",
                },
            ),
            "danger": self._Score(
                instructions="Assess observed and estimated danger, accounting for unknowns. "
                "This is a qualitative diagnostic, not a calibrated collision probability.",
                criteria=["No immediate threat observed", "Possible threat", "Imminent threat"],
            ),
        }
        if snapshot.swimming:
            questions["next_action"].instructions["swimming"] = (
                "Use repeated A strokes for upward movement, release A to sink; inertia remains. "
                "Swimming does not require landing. The approximate water model may be wrong; "
                "check observed depth changes and visible enemies every cycle."
            )
            for a in candidates:
                questions["next_action"].criteria[a.value] = (
                    "Swim stroke with this directional input: " + a.value
                    if a in JUMP_ACTIONS
                    else "Release A and drift/sink with input: " + a.value
                )
        decision = self._request(state, questions, candidates)
        if risk_control["status"] in ("filtered", "no_lower_risk_option"):
            decision = replace(decision, selection=risk_control)
        return decision

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
        return decision


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
