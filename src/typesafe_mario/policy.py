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
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in actions}
        questions = {
            "next_action": self._Choice(
                instructions={
                    "question": "Which controller macro should Mario commit to next?",
                    "goal": "Advance toward the stage flag while avoiding death.",
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
                    "`player.jump_phase` indicate that a forward jump should begin or remain "
                    "held during the next action cycle? Treat hazard urgency flags as heuristic "
                    "evidence, checking collision risk and takeoff conditions under committed "
                    "controls. Also count a trusted obstacle/gap within three tiles, immediate "
                    "collision risk, or a rising jump over a known gap as yes."
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
        started = time.perf_counter()
        response = self._client.system_one(state=snapshot.to_state(), questions=questions)
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
