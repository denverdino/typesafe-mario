"""Run evidence, separated from scheduling and controller execution."""

import json
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from time import monotonic
from typing import Any
from uuid import uuid4

from .actions import Action
from .analysis import compare_position
from .observation import model_state, observe
from .policy import Decision
from .questions import PROMPT_VERSION
from .requests import REQUEST_SCHEMA_VERSION, PreparedDecision
from .state import MarioSnapshot
from .tactical import STATE_VERSION


class EpisodeLog:
    """Attribute every executed frame to its active decision, including wait frames."""

    def __init__(self, log: Any, run_config: dict[str, Any] | None = None) -> None:
        self.log = log
        self.run_config = run_config or {}
        self.episode_id = uuid4().hex
        self.active: dict[str, Any] | None = None
        self.total_reward = 0.0
        self.total_frames = 0
        self.ended = False
        self.started = monotonic()
        self.recent_frames = deque(maxlen=64)
        self.request_indices: set[int] = set()
        self.forecast: dict = {}
        self.branch: dict = {}
        self.recovering = False

    def write(self, record: dict[str, Any]) -> None:
        self.log.write(
            json.dumps(
                {
                    "schema_version": 2,
                    "episode_id": self.episode_id,
                    "run_config": self.run_config,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "elapsed_ms": round((monotonic() - self.started) * 1000, 3),
                    **record,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self.log.flush()

    def begin(
        self,
        index: int,
        observed: MarioSnapshot,
        decision: Decision,
        applied: MarioSnapshot,
        delay_frames: int,
        prepared: PreparedDecision | None = None,
    ) -> None:
        request = prepared.request if prepared is not None else None
        self.forecast = observed.prediction or {}
        self.branch = next(
            (
                b
                for b in self.forecast.get("action_forecasts", ())
                if b["action"] == decision.action.value
            ),
            {},
        )
        self.active = {
            "record_type": "decision",
            "decision": index,
            "state": request.to_payload()["state"] if request else model_state(observed),
            "request_id": self.request_id(index) if request else None,
            "debug_state": observed.to_debug_state(include_model_prediction=False),
            "state_text": observed.to_text(),
            "action": decision.action.value,
            "confidence": decision.confidence,
            "selection": dict(decision.selection) if decision.selection is not None else None,
            "probabilities": dict(decision.probabilities),
            "jump_needed_probability": decision.jump_needed_probability,
            "jump_intent": decision.jump_intent,
            "jump_intent_probabilities": dict(decision.jump_intent_probabilities),
            "danger_score": decision.danger_score,
            "latency_ms": decision.latency_ms,
            "provider_metadata": dict(decision.provider_metadata),
            "reward": 0.0,
            "prediction_comparison": {
                "application": compare_position(
                    self.forecast.get("committed_future"), {"x": applied.x, "y": applied.y}
                ),
            },
            "execution": {
                "start_frame": self.total_frames,
                "apply_at_frame": applied.frame_index,
                "frames": 0,
                "actions": [],
                "observed_events": [],
                "response_delay_frames": delay_frames,
                "start_state": applied.to_state(),
            },
        }

    def request_id(self, index: int) -> str:
        return f"{self.episode_id}:{index}"

    def record_request(self, index: int, prepared: PreparedDecision) -> None:
        if prepared.request is None:
            return
        self.request_indices.add(index)
        self.write(
            {
                "record_type": "policy_request",
                "status": "prepared",
                "decision": index,
                "request_id": self.request_id(index),
                "timing": prepared.timing.to_state(),
                "candidate_actions": [a.value for a in prepared.candidates],
                "request": {
                    "schema_version": REQUEST_SCHEMA_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "state_version": STATE_VERSION,
                    "sha256": prepared.request.sha256,
                    "payload": prepared.request.to_payload(),
                },
            }
        )

    def record_outcome(self, index: int, status: str, **details) -> None:
        if index in self.request_indices:
            self.write(
                {
                    "record_type": "request_outcome",
                    "request_id": self.request_id(index),
                    "decision": index,
                    "status": status,
                    **details,
                }
            )

    def discard(self, index: int, future, reason: str) -> None:
        if index not in self.request_indices:
            future.cancel()
            return
        # Do not claim network delivery or completion when abandoning a running call.
        if future.cancel():
            completion = "cancelled_before_running"
        elif future.done():
            completion = "failed" if future.exception() is not None else "returned"
        else:
            completion = "still_running"
        self.record_outcome(index, "discarded", reason=reason, completion_state=completion)

    def step(self, action: Action, reward: float, snapshot: MarioSnapshot | None = None) -> None:
        self.total_frames += 1
        self.total_reward += reward
        if self.active is not None:
            if self.active["execution"]["frames"] == 0:
                self.record_outcome(
                    self.active["decision"],
                    "applied",
                    observed_at_frame=self.active["execution"]["apply_at_frame"],
                    action=self.active["action"],
                    actual_first_action=action.value,
                )
            self.active["reward"] += reward
            self.active["execution"]["frames"] += 1
            self.active["execution"]["actions"].append(action.value)
        if snapshot is not None:
            o = observe(snapshot)
            self.recent_frames.append(
                {
                    "observed_at_frame": o.frame,
                    "execution_frame": self.total_frames,
                    "actual_action": action.value,
                    "x": o.x,
                    "y": o.y,
                    "vx": o.vx,
                    "vy": o.vy,
                    "grounded": o.grounded,
                    "actors": [asdict(a) for a in o.actors],
                    "terminal": o.terminal,
                }
            )
            if self.active is not None:
                events = self.active["execution"]["observed_events"]
                if snapshot.landed_this_frame:
                    events.append(
                        {
                            "kind": "landed",
                            "observed_at_frame": o.frame,
                            "x": o.x,
                            "y": o.y,
                            "actual_action": action.value,
                        }
                    )
                if snapshot.recovery.get("active") and not self.recovering:
                    self.active["recovery_event"] = {
                        "observed_at_frame": o.frame,
                    }
            self.recovering = bool(snapshot.recovery.get("active"))

    def finish(self, snapshot: MarioSnapshot, terminated: bool, truncated: bool) -> None:
        if self.active is not None:
            self.active["prediction_comparison"]["executed_cycle"] = compare_position(
                self.branch.get("first_cycle"),
                {"x": snapshot.x, "y": snapshot.y},
                eligible=self.active["execution"]["frames"]
                == self.forecast.get("action_cycle_frames")
                and self.forecast.get("committed_future", {}).get("valid_for_application", True),
            )
            expected_landing = self.branch.get("first_landing")
            actual_landings = [
                e for e in self.active["execution"]["observed_events"] if e["kind"] == "landed"
            ]
            if actual_landings or "recovery_event" in self.active:
                self.active["execution"]["event_trace"] = list(self.recent_frames)
            self.active["prediction_comparison"]["landing"] = {
                "forecast": expected_landing,
                "forecast_frame_reference": "offset_from_observed_at_frame",
                "observed_at_frame": self.forecast.get("observed_at_frame"),
                "actual": actual_landings,
            }
            self.write(
                {
                    **self.active,
                    "result_state": snapshot.to_state(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            self.active = None

    def end(self, snapshot: MarioSnapshot, terminated: bool, truncated: bool, reason: str) -> None:
        if self.ended:
            return
        if self.active is not None and self.active["execution"]["frames"] == 0:
            self.record_outcome(
                self.active["decision"],
                "discarded",
                reason=reason,
                completion_state="returned_without_execution",
            )
        self.finish(snapshot, terminated, truncated)
        self.write(
            {
                "record_type": "episode_end",
                "reason": reason,
                "result_state": snapshot.to_state(),
                "reward": self.total_reward,
                "frames": self.total_frames,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "recent_frames": list(self.recent_frames),
            }
        )
        self.ended = True
