"""Observed progress memory; records evidence, never selects controller inputs."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProgressMemory:
    level: tuple[int, int, int] | None = None
    anchor_x: int = 0
    progress_frame: int = 0
    history: list[tuple[int, int, int, str | None]] = field(default_factory=list)

    def update(
        self, frame: int, level: tuple[int, int, int], x: int, y: int, action: str | None
    ) -> dict[str, Any]:
        # Wrapping past the left edge can report x=65535; castle warps can
        # also move backwards without changing the area. Neither is a stall.
        # Compare adjacent observations, so sustained ordinary retreat still
        # counts as no progress. Allow ample motion per elapsed emulator frame.
        discontinuity = bool(
            self.history and abs(x - self.history[-1][1]) > 32 * max(1, frame - self.history[-1][0])
        )
        if level != self.level or discontinuity or x >= self.anchor_x + 4:
            self.level, self.anchor_x, self.progress_frame = level, x, frame
            self.history.clear()
        self.history.append((frame, x, y, action))
        self.history = self.history[-96:]
        age = frame - self.progress_frame
        stationary = []
        for observed in reversed(self.history):
            if (
                observed[0] != frame - len(stationary)
                or abs(observed[1] - x) > 2
                or abs(observed[2] - y) > 2
            ):
                break
            stationary.append(observed)
        # Measure a contiguous spatial envelope, not distance from just the
        # latest position: alternating inputs can keep resetting stillness.
        local = []
        min_x = max_x = x
        min_y = max_y = y
        for observed in reversed(self.history):
            _, ox, oy, _ = observed
            bounds = min(min_x, ox), max(max_x, ox), min(min_y, oy), max(max_y, oy)
            if (
                observed[0] != frame - len(local)
                or bounds[1] - bounds[0] > 8
                or bounds[3] - bounds[2] > 4
            ):
                break
            min_x, max_x, min_y, max_y = bounds
            local.append(observed)
        return {
            "active": age >= 48,
            "no_progress_frames": age,
            "anchor_x": self.anchor_x,
            "recent_max_height": max(h[2] for h in self.history),
            "action_frames": dict(Counter(h[3] for h in self.history if h[3] is not None)),
            "stationary_frames": max(0, len(stationary) - 1),
            "stationary_action_frames": dict(
                Counter(h[3] for h in stationary[:-1] if h[3] is not None)
            ),
            "local_motion": {
                "frames": max(0, len(local) - 1),
                "x_range": [min_x, max_x],
                "y_range": [min_y, max_y],
                "action_frames": dict(Counter(h[3] for h in local[:-1] if h[3] is not None)),
            },
            "reason": "no_meaningful_forward_progress" if age >= 48 else None,
        }
