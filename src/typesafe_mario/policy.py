from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, Action
from .state import MarioSnapshot


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    jump_needed_probability: float | None = None
    danger_score: float | None = None


class Policy(Protocol):
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


class TypeSafePolicy:
    def __init__(self) -> None:
        try:
            from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._Choice = Choice
        self._Noul = Noul
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
        state = snapshot.to_state()
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in actions}
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Which controller macro should Mario commit to next?",
                    "goal": snapshot.goal,
                    "scoring": (
                        "Actively use reliable `opportunities` to collect coins or stomp a "
                        "stompable enemy when doing so preserves survival and progress. "
                        "Compare a controlled jump, running jump and brief slowdown. "
                        "A stompable type does not mean this jump will succeed: require "
                        "descending top-contact geometry and a supported route after the bounce; "
                        "consider nearby enemies and gaps. Distances and horizontal overlap "
                        "are not a stomp prediction. Relative Y in opportunities is positive down. "
                        "Judge the opportunity at macro application under `committed_control`, "
                        "not only at the observation frame. If `scoring.pursuit.allow_extra_effort` "
                        "is false, stop spending extra movement or time on score; incidental "
                        "collection during forward progress is still useful. This reflects low "
                        "or unknown remaining time or prolonged failure to improve best progress. "
                        "Never move left to chase missed rewards or wait to farm enemies. "
                        "LEFT remains available for evasion and recovery. Preserve forward "
                        "gap-crossing controls. Abandon scoring when landing geometry is "
                        "uncertain, progress is blocked or an immediate threat needs attention. "
                        "Unknown or truncated observations do not establish safety. "
                        "Confirmed counts are lower bounds; score changes alone do not prove a "
                        "particular collection or stomp. Choose one of the supplied macros."
                    ),
                    "reward_blocks": (
                        "Actively try to hit reachable `coin_block` or `powerup_block` "
                        "opportunities from below "
                        "when `scoring.pursuit.allow_extra_effort` is true and the approach and "
                        "landing are safe. Hitting reveals their reward; touching the side or "
                        "landing on top does not release it. Use `head_bump_geometry`: its horizontal "
                        "interval is [block left minus head X, block right minus head X); "
                        "it contains zero when the head is under the block. Positive "
                        "`rise_to_block_bottom_pixels` is how far the current head probe is "
                        "below the underside. This is current geometry, not a hit prediction. "
                        "Choose an approach and rising jump so the head enters that interval "
                        "at underside contact, accounting for horizontal speed, rising/falling "
                        "phase and committed controls before the chosen macro starts. "
                        "Approach with right at controlled speed; prefer right_jump when "
                        "the forward rising path can hit the underside. When already aligned "
                        "and moving slowly, use jump to hit overhead. Holding jump while rising "
                        "can preserve height. Jump/noop do not instantly cancel momentum, "
                        "and jumping too early may carry Mario onto the top or past the block. "
                        "Do not run blindly past a nearby reachable reward block. "
                        "After a hit, let Mario descend and reassess. If a multi-coin brick "
                        "is observed again, another grounded jump is useful only while the "
                        "pursuit budget and safety permit; do not wait through an unknown or "
                        "active block bounce, chase an empty/disappeared block, or backtrack. "
                        "Resume forward progress when a target is passed, unreachable, unsafe, "
                        "or extra effort is disabled. Never interrupt a gap crossing for a "
                        "block. Unknown head geometry does not establish a reachable hit."
                    ),
                    "powerups": (
                        "A `powerup_block` releases a mushroom or fire flower when hit from "
                        "below, determined by Mario's status at the hit; "
                        "expected_powerup_if_hit_now is conditional, not an already spawned item. "
                        "When safe, prioritize becoming big while small, or getting a fire flower "
                        "while big, over extra coins. Plan to collect the released item as part "
                        "of the same forward route. A block hit alone does not grant the upgrade. "
                        "After the hit, use the actual `powerup` opportunity: mushrooms move, "
                        "flowers stay in place. These are friendly touch targets, not stomp "
                        "targets or hazards. While emergence_phase is emerging the item is "
                        "still rising from the block; a null collision box means contact "
                        "geometry is not yet available. A brief safe slowdown or positioning "
                        "jump can keep it within reach while it emerges. Compare Mario's and "
                        "the item's screen collision boxes, relative Y and horizontal velocity "
                        "at action application; approach or jump onto the supporting block "
                        "when needed to touch the item above it. Do not jump under the now-empty "
                        "block again. Avoid overshooting, but do not backtrack for a missed "
                        "item. Never enter an enemy, pit or unknown landing route for an upgrade. "
                        "Respect the existing pursuit time/progress budget; once exhausted or "
                        "the item is no longer observed, resume forward navigation. Already "
                        "fiery Mario needs no further upgrade, so only collect incidentally. "
                        "Verify the actual player.powerup_status; revealing or losing sight "
                        "of an item does not prove collection."
                    ),
                    "timing": (
                        f"The selected macro runs for {snapshot.decision_horizon_frames} emulator "
                        "frames in the next cycle. Simulation pauses at cycle boundaries until "
                        "inference completes; waiting does not advance game time."
                    ),
                    "geometry": (
                        "Use `terrain.observation_reliability`. While airborne, prefer "
                        "`terrain.last_grounded_preview` over low-reliability current geometry. "
                        "A trusted obstacle or gap within three tiles requires a forward jump. "
                        "Judge historical gap proximity from its world coordinates and current "
                        "`player.x`, not its old tile distance."
                    ),
                    "trajectory": (
                        "Use `trajectory`. If `crossing_known_gap` is true, preserve forward "
                        "speed and keep a forward jump held while rising. Do not switch to noop "
                        "or left over a gap."
                    ),
                    "stall": (
                        "If `episode.stalled_frames` is increasing and "
                        "`recent_control.outcome` is blocked, the current non-jump action failed."
                    ),
                    "enemy_timing": (
                        "Use `hazard` projections, not distance alone. Contact estimates assume "
                        "unchanged horizontal velocities; they do not simulate acceleration or "
                        "committed controls, and describe horizontal overlap, not confirmed "
                        "collision. The 8-frame clearance allowance and urgency flags are "
                        "heuristics, including when referenced by action criteria. Judge jump "
                        "timing using vertical geometry and `committed_control`. "
                        "A false urgency flag or unknown landing estimate does not establish "
                        "safety. "
                        "`will_land_before_contact` compares estimates; it does not establish "
                        "whether the enemy will be cleared. Use "
                        "`upcoming_enemies` and their spacing to avoid landing on a second or "
                        "third enemy hidden behind the nearest one."
                    ),
                    "delay": (
                        "The selected macro starts after "
                        "`committed_control.frames_before_selected_action`. Distance projections "
                        "use the total reaction horizon, including the selected macro's duration. "
                        "A first-landing estimate does not guarantee Mario can still jump when "
                        "the selected macro starts."
                    ),
                },
                criteria=criteria,
            ),
            "jump_needed": self._Noul(
                instructions=(
                    "Do trusted `terrain`, projected `hazard`, `trajectory`, and "
                    "`player.jump_phase` indicate that a jump should begin or remain "
                    "held during the next action cycle? Treat hazard urgency flags as heuristic "
                    "evidence, checking collision risk and takeoff conditions under committed "
                    "controls. Also count a trusted obstacle/gap within three tiles, immediate "
                    "collision risk, or a rising jump over a known gap as yes. Also count a "
                    "jump for reliable coin or stomp `opportunities`, including an aligned "
                    "upward head hit on a coin_block or powerup_block using head_bump_geometry, "
                    "or a safe jump to touch an observed mushroom/fire flower, when it fits "
                    "`scoring.pursuit` and preserves safety and forward progress. A scoring "
                    "opportunity alone does not require jumping if contact or the landing "
                    "route is uncertain."
                )
            ),
            "danger": self._Score(
                instructions="How dangerous is Mario's immediate situation?",
                criteria=[
                    "Safe open movement",
                    "Potential obstacle or enemy soon",
                    "Immediate collision, fall, or enemy threat",
                ],
            ),
        }
        # Keep specialized approach instructions out of ordinary navigation and
        # expired pursuit budgets. Jev still selects every action itself.
        if not state["scoring"]["pursuit"]["allow_extra_effort"] or not any(
            item["kind"] in ("coin_block", "powerup_block")
            and item["head_bump_geometry"] is not None
            for item in state["opportunities"]["items"]
        ):
            questions["next_action"].instructions.pop("reward_blocks")
        if not state["scoring"]["pursuit"]["allow_extra_effort"] or not any(
            item["kind"] in ("powerup_block", "powerup") for item in state["opportunities"]["items"]
        ):
            questions["next_action"].instructions.pop("powerups")
        started = time.perf_counter()
        response = self._client.system_one(state=state, questions=questions)
        latency_ms = (time.perf_counter() - started) * 1000

        action_answer = self._answer(response, "next_action", "choices")
        jump_answer = self._answer(response, "jump_needed", "nouls")
        danger_answer = self._answer(response, "danger", "scores")
        action = Action(str(action_answer.choice))
        probabilities = {
            str(key): float(value) for key, value in dict(action_answer.probabilities).items()
        }
        return Decision(
            action=action,
            confidence=float(action_answer.confidence),
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=float(jump_answer.noul),
            danger_score=float(danger_answer.score),
        )


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
