from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .actions import Action
from .questions import build_questions
from .requests import DecisionRequest, PreparedDecision
from .state import MarioSnapshot
from .tactical import build_tactical_state


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
    provider_metadata: Mapping[str, str] = field(default_factory=dict)


class Policy(Protocol):
    def prepare(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> PreparedDecision: ...

    def resolve(self, prepared: PreparedDecision) -> Decision: ...


class TypeSafePolicy:
    def __init__(self, *, diagnostic_questions: bool = False) -> None:
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._client = TypeSafeClient()
        self.diagnostic_questions = diagnostic_questions

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
        return self.resolve(self.prepare(snapshot, actions))

    def prepare(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> PreparedDecision:
        tactical = build_tactical_state(snapshot, actions)
        request = DecisionRequest.capture(
            state=tactical.state,
            questions=build_questions(
                tactical.state,
                tactical.candidates,
                diagnostics=getattr(self, "diagnostic_questions", False),
            ),
        )
        return PreparedDecision(snapshot, tactical.candidates, request)

    def resolve(self, prepared: PreparedDecision) -> Decision:
        if prepared.request is None:
            raise ValueError("TypeSafe requires a captured provider request")
        payload = prepared.request.to_payload()
        risk_control = payload["state"]["risk_control"]
        decision = self._request(payload, prepared.candidates)
        if risk_control["status"] in ("filtered", "no_lower_risk_option"):
            decision = replace(decision, selection=risk_control)
        return decision

    def _request(self, payload: dict[str, Any], candidates: Sequence[Action]) -> Decision:
        started = time.perf_counter()
        response = self._client.system_one(**payload)
        latency_ms = (time.perf_counter() - started) * 1000

        action_answer = self._answer(response, "next_action", "choices")
        jump_answer = (
            self._answer(response, "jump_intent", "choices")
            if "jump_intent" in payload["questions"]
            else None
        )
        danger_answer = (
            self._answer(response, "danger", "scores") if "danger" in payload["questions"] else None
        )
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
            )
            if jump_answer is not None
            else None,
            jump_intent=str(jump_answer.choice) if jump_answer is not None else None,
            jump_intent_probabilities=dict(jump_answer.probabilities)
            if jump_answer is not None
            else {},
            danger_score=float(danger_answer.score) if danger_answer is not None else None,
            provider_metadata={
                key: value
                for key in ("model", "request_id")
                if isinstance(value := getattr(response, key, None), str)
            },
        )
        return decision


class HeuristicPolicy:
    """Offline smoke-test policy; not intended as the Mario benchmark baseline."""

    def prepare(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> PreparedDecision:
        if not actions:
            raise ValueError("At least one controller action is required")
        return PreparedDecision(snapshot, tuple(actions))

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        return self.resolve(self.prepare(snapshot, actions))

    def resolve(self, prepared: PreparedDecision) -> Decision:
        snapshot, actions = prepared.snapshot, prepared.candidates
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
