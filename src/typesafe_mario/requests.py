"""Captured SDK arguments, isolated from mutable observations and transport objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .actions import Action
from .state import MarioSnapshot
from .timing import DecisionTiming

REQUEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DecisionRequest:
    payload_json: str

    @classmethod
    def capture(cls, *, state: dict, questions: dict) -> DecisionRequest:
        return cls(
            json.dumps(
                {"state": state, "questions": questions},
                allow_nan=False,
                separators=(",", ":"),
            )
        )

    def to_payload(self) -> dict:
        """Each consumer gets a fresh copy of the exact captured SDK arguments."""
        return json.loads(self.payload_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedDecision:
    snapshot: MarioSnapshot
    candidates: tuple[Action, ...]
    request: DecisionRequest | None = None

    @property
    def timing(self) -> DecisionTiming:
        return DecisionTiming(
            self.snapshot.frame_index,
            self.snapshot.frames_until_action,
            self.snapshot.decision_horizon_frames,
        )
