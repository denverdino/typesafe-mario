"""Frame-based enemy identities and motion estimates, independent of the camera."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnemyMeasurement:
    slot: int
    kind_id: int
    kind: str
    x: int
    y: int
    engine_state: int
    plant: dict[str, Any] | None = None


@dataclass(frozen=True)
class EnemyTrack:
    id: str
    measurement: EnemyMeasurement
    samples: tuple[tuple[int, int, int], ...]
    last_seen_frame: int
    observed: bool = True

    @property
    def velocity(self) -> tuple[float | None, float | None]:
        if len(self.samples) < 2:
            return None, None
        first, last = self.samples[0], self.samples[-1]
        dt = last[0] - first[0]
        return (last[1] - first[1]) / dt, (last[2] - first[2]) / dt

    def to_state(self) -> dict[str, Any]:
        m = self.measurement
        vx, vy = self.velocity
        return {
            "id": self.id,
            "slot": m.slot,
            "kind": m.kind,
            "engine_state": m.engine_state,
            "position": {"x": m.x, "y": m.y},
            "velocity": {"x": vx, "y": vy},
            "motion": "unknown"
            if vy is None
            else "falling"
            if vy > 0
            else "rising"
            if vy < 0
            else "horizontal"
            if vx
            else "stationary",
            "observed": self.observed,
            "last_seen_frame": self.last_seen_frame,
            "sample_span_frames": self.samples[-1][0] - self.samples[0][0],
            "confidence": "low" if not self.observed or vx is None else "estimated",
            "position_quantization_pixels": 1,
            **({"plant": dict(m.plant)} if m.plant is not None else {}),
            "role": "interactable" if m.kind in {"springboard", "flagpole"} else "enemy",
        }


@dataclass
class EnemyTracker:
    epoch: int = 0
    level: tuple[int, int, int] | None = None
    generation: int = 0
    tracks: dict[int, EnemyTrack] = field(default_factory=dict)

    def update(
        self,
        frame: int,
        level: tuple[int, int, int],
        measurements: list[EnemyMeasurement] | None,
    ) -> tuple[EnemyTrack, ...]:
        from dataclasses import replace

        if level != self.level:
            self.tracks.clear()
            self.level = level
        if measurements is None:
            self.tracks = {
                slot: replace(t, observed=False)
                for slot, t in self.tracks.items()
                if frame - t.last_seen_frame <= 8
            }
            return tuple(self.tracks.values())
        updated = {}
        for m in measurements:
            old = self.tracks.get(m.slot)
            new_identity = (
                old is None
                or old.measurement.kind_id != m.kind_id
                or abs(old.measurement.x - m.x) > 32
                or abs(old.measurement.y - m.y) > 32
            )
            if new_identity:
                self.generation += 1
                identity = f"{self.epoch}:{level}:{m.slot}:{self.generation}"
                samples = ()
            else:
                identity, samples = old.id, old.samples
                vx, vy = old.velocity
                dx, dy = m.x - old.measurement.x, m.y - old.measurement.y
                if old.measurement.engine_state != m.engine_state or not old.observed:
                    samples = ()
                elif (vx is not None and vx * dx < 0) or (vy is not None and vy * dy < 0):
                    samples = samples[-1:]
            samples = tuple(s for s in samples if frame - s[0] <= 8) + ((frame, m.x, m.y),)
            updated[m.slot] = EnemyTrack(identity, m, samples, frame)
        self.tracks = updated
        return tuple(updated.values())
