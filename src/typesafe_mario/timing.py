"""Absolute frames use the observation clock, including checkpoint-restored runs."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionTiming:
    observed_at_frame: int
    action_delay_frames: int
    action_cycle_frames: int

    def __post_init__(self):
        for name, minimum in (
            ("observed_at_frame", 0),
            ("action_delay_frames", 0),
            ("action_cycle_frames", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def apply_at_frame(self) -> int:
        return self.observed_at_frame + self.action_delay_frames

    def to_state(self) -> dict[str, int]:
        return {**asdict(self), "apply_at_frame": self.apply_at_frame}
