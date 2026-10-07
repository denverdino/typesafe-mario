"""Diagnostic evidence only; never supplies observations or controls to the policy."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic
from typing import Any, TextIO

from .actions import ACTION_TO_INDEX, Action
from .scoring import ScoringSummary


class RunTrace:
    def __init__(self, stream: TextIO, **metadata: Any) -> None:
        self.stream = stream
        self.started = monotonic()
        self.episode = -1
        self.frame = 0
        self.ended = True
        self.pending: int | None = None
        self.active: int | None = None
        self.request_frame = self.apply_frame = 0
        self.pending_frame = 0
        self.summary = ScoringSummary()
        packages = {}
        for name in (
            "typesafe-mario",
            "gym-super-mario-bros",
            "nes-py",
            "gymnasium",
            "typesafe-sdk",
        ):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        sources = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }
        self.write(
            "metadata",
            schema_version=2,
            python=platform.python_version(),
            packages=packages,
            source_sha256=sources,
            frame_convention="Frame N is after N env.step calls; apply at N first steps N+1.",
            **metadata,
        )

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "episode": self.episode,
            "frame": self.frame,
            "elapsed_ms": round((monotonic() - self.started) * 1000, 3),
            **fields,
        }
        # Gym telemetry may contain NumPy scalar values; no environment variables
        # or policy internals (which could contain credentials) are serialized.
        self.stream.write(
            json.dumps(record, default=lambda value: value.item(), separators=(",", ":")) + "\n"
        )
        if event != "frame":
            self.stream.flush()

    @staticmethod
    def evidence(info: Any, ram: Any, snapshot: Any) -> dict[str, Any]:
        return {
            "info": dict(info),
            "ram_hex": bytes(ram).hex() if ram is not None else None,
            "observation": {
                "grounded": snapshot.grounded,
                "airborne_frames": snapshot.airborne_frames,
                "vertical_motion": snapshot.vertical_motion,
                "dx": snapshot.dx,
                "dy": snapshot.dy,
            },
        }

    def begin(self, info: Any, ram: Any, snapshot: Any, *, seed: int | None) -> None:
        self.episode += 1
        self.frame = 0
        self.ended = False
        self.pending = self.active = None
        self.summary.begin(snapshot)
        self.write("episode_start", seed=seed, **self.evidence(info, ram, snapshot))

    def request(self, index: int, snapshot: Any, committed: Action, horizon: int) -> None:
        self.pending = index
        self.pending_frame = self.frame
        self.write(
            "request",
            decision=index,
            state=snapshot.to_state(),
            committed_decision=self.active,
            committed_action=committed.value,
            expected_apply_frame=self.frame + horizon,
        )

    def apply(self, index: int, action: Action) -> None:
        self.active = index
        self.request_frame = self.pending_frame
        self.apply_frame = self.frame
        self.summary.mark_interval(self.frame)
        self.pending = None
        self.write("apply", decision=index, request_frame=self.request_frame, action=action.value)

    def step(
        self,
        action: Action,
        info: Any,
        ram: Any,
        snapshot: Any,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.frame += 1
        self.summary.observe(snapshot)
        self.write(
            "frame",
            decision=self.active,
            action=action.value,
            action_index=ACTION_TO_INDEX[action],
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            **self.evidence(info, ram, snapshot),
        )
        if snapshot.scoring is not None:
            for event in snapshot.scoring.events:
                fields = asdict(event)
                kind = fields.pop("kind")
                fields.pop("frame")
                self.write(kind, decision=self.active, **fields)

    def scoring_result(self) -> dict[str, Any]:
        return self.summary.interval_result()

    def execution(self) -> dict[str, int]:
        return {
            "episode": self.episode,
            "request_frame": self.request_frame,
            "apply_frame": self.apply_frame,
            "end_frame": self.frame,
            "executed_frames": self.frame - self.apply_frame,
        }

    def end(self, reason: str) -> None:
        if not self.ended:
            self.write(
                "episode_end",
                reason=reason,
                active_decision=self.active,
                pending_decision=self.pending,
                summary=self.summary.episode_result(),
            )
            self.ended = True
