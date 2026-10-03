"""Bounded physics priors calibrated only from real consecutive observations."""

from collections import defaultdict, deque
from math import isfinite
from statistics import fmean

from .actions import JUMP_ACTIONS, Action
from .observation import Observation

# Deliberately approximate priors, not a reconstruction of NES engine state.
PRIORS = {
    "walk_accel": (0.12, 0.01, 0.5),
    "run_accel": (0.18, 0.01, 0.5),
    "brake_accel": (0.24, 0.01, 0.8),
    "air_accel": (0.08, 0.01, 0.3),
    "air_walk_speed": (1.6, 1.4, 2.0),
    "friction": (0.08, 0.0, 0.4),
    "gravity_hold": (0.18, 0.02, 0.5),
    "gravity_hold_fast": (0.18, 0.02, 0.5),
    "gravity_fall": (0.5, 0.1, 0.9),
    "jump_speed": (5.0, 2.0, 7.0),
    "jump_speed_fast": (6.0, 2.0, 7.0),
    "stomp_speed": (4.0, 1.0, 8.0),
    "swim_gravity": (0.12, 0.02, 0.3),
    "swim_impulse": (2.5, 0.5, 5.0),
}


def direction(action: Action) -> int:
    return -1 if action == Action.LEFT else 1 if action.value.startswith("right") else 0


def position_fit(values):
    """End-of-step velocity and acceleration from nine evenly spaced positions."""
    assert len(values) == 9
    xs = [x - values[0] for x in values]
    acceleration = sum((3 * (i - 4) ** 2 - 20) * x for i, x in enumerate(xs)) / 462
    middle_velocity = sum((i - 4) * x for i, x in enumerate(xs)) / 60
    return middle_velocity + 3.5 * acceleration, acceleration


class Dynamics:
    def __init__(self, *, calibrated=None):
        self.calibrated = dict(calibrated or {})
        for name, value in self.calibrated.items():
            if (
                name not in PRIORS
                or not isfinite(value)
                or not PRIORS[name][1] <= value <= PRIORS[name][2]
            ):
                raise ValueError(f"Invalid calibrated parameter: {name}")
        self.samples = defaultdict(lambda: deque(maxlen=64))

    def value(self, name: str) -> float:
        prior, lo, hi = PRIORS[name]
        if name in self.calibrated:
            # Controlled experiments are the stable reference. Gameplay samples
            # remain diagnostics; they must not silently overwrite a measured
            # profile with contact-contaminated estimates.
            return self.calibrated[name]
        values = self.samples.get(name, ())
        return max(lo, min(hi, (prior * 8 + sum(values)) / (8 + len(values))))

    def fitted(self):
        """Fit once per forecast; hypothetical steps never refit history."""
        return Parameters({name: self.value(name) for name in PRIORS})

    def add(self, name: str, value: float):
        _, lo, hi = PRIORS[name]
        # Pixel quantization produces zero and occasionally negative estimates
        # of positive acceleration. Filtering them by the physical parameter
        # bounds biases the learned mean upward; clamp the fitted value instead.
        if name not in (
            "jump_speed",
            "jump_speed_fast",
            "stomp_speed",
            "swim_impulse",
            "air_walk_speed",
        ):
            lo, hi = -1.2, 1.2
        if name in ("gravity_hold", "gravity_hold_fast", "gravity_fall"):
            # During release, two quantized position differences can drop by
            # two pixels even when physical acceleration is below one. Rejecting
            # only those large downward samples biases landing estimates late.
            hi = 2.2
        if lo <= value <= hi:
            self.samples[name].append(value)

    @staticmethod
    def contact(o: Observation) -> bool:
        return any(
            block.left - 18 <= o.x <= block.right + 2
            and block.bottom <= o.y + o.height
            and block.top > o.y + 1
            for block in o.blocks
        ) or any(abs(actor.x - o.x) < 24 and abs(actor.y - o.y) < 32 for actor in o.actors)

    def learn_air(self, history):
        """Fit acceleration to nine actual positions, without differencing pixel speeds."""
        if len(history) < 9:
            return
        window = list(history)[-9:]
        first, last = window[0], window[-1]
        if last.previous_action is None:
            return
        d = direction(Action(last.previous_action))
        if not d or any(
            h.frame != first.frame + i
            or h.level != last.level
            or h.grounded
            or h.swimming
            or not h.motion_reliable
            or h.terminal
            or (i > 0 and h.previous_action != last.previous_action)
            or self.contact(h)
            for i, h in enumerate(window)
        ):
            return
        # Least-squares x(t)=c+v*t+a*t²/2, centered on t=4.
        # Integer position rounding is shared across all samples; selecting
        # individual 0/1 velocity pairs would censor the accelerating phase.
        end_velocity, acceleration = position_fit([h.x for h in window])
        start_velocity = end_velocity - 8 * acceleration
        if (
            abs(acceleration) < 0.025
            and 1.4 < d * start_velocity < 2.0
            and 1.4 < d * end_velocity < 2.0
        ):
            # A sustained plateau identifies walking-flight speed, whereas
            # accelerating and reversing windows identify acceleration only.
            self.add("air_walk_speed", d * (last.x - first.x) / 8)
        # B alone does not establish a running flight. Exclude windows near
        # the walking cap, where constant speed cannot identify acceleration.
        if max(d * start_velocity, d * end_velocity) < 1.4:
            self.add("air_accel", d * acceleration)

    def learn_held_gravity(self, history, *, fast_jump):
        if len(history) < 9 or fast_jump is None:
            return
        window = list(history)[-9:]
        first, last = window[0], window[-1]
        if any(
            h.frame != first.frame + i
            or h.level != last.level
            or h.grounded
            or h.swimming
            or not h.motion_reliable
            or h.terminal
            or h.previous_action not in JUMP_ACTIONS
            or self.contact(h)
            for i, h in enumerate(window)
        ):
            return
        velocity, acceleration = position_fit([h.y for h in window])
        # A window crossing the apex mixes held-ascent and falling gravity.
        # Stay above the one-pixel quantization band at both ends.
        if min(velocity, velocity - 8 * acceleration) > 1:
            self.add("gravity_hold_fast" if fast_jump else "gravity_hold", -acceleration)

    def learn_friction(self, history):
        if len(history) < 9:
            return
        window = list(history)[-9:]
        first, last = window[0], window[-1]
        if any(
            h.frame != first.frame + i
            or h.level != last.level
            or not h.grounded
            or h.swimming
            or not h.motion_reliable
            or h.terminal
            or (i > 0 and h.previous_action != "noop")
            or self.contact(h)
            for i, h in enumerate(window)
        ):
            return
        velocity, acceleration = position_fit([h.x for h in window])
        start_velocity = velocity - 8 * acceleration
        # Once stopped, position rounding does not identify a deceleration.
        # Require motion in one direction at both ends, without selecting
        # individual nonzero pixel deltas (which censors slow-moving samples).
        if velocity * start_velocity > 0 and min(abs(velocity), abs(start_velocity)) > 0.25:
            self.add("friction", -acceleration if velocity > 0 else acceleration)

    def learn(self, previous: Observation, current: Observation, *, fast_jump: bool | None = None):
        a, b = previous, current
        if (
            b.frame != a.frame + 1
            or a.level != b.level
            or a.swimming != b.swimming
            or not (a.motion_reliable and b.motion_reliable)
            or a.terminal
            or b.terminal
            or b.previous_action is None
        ):
            return
        action = Action(b.previous_action)
        # Do not fit acceleration at walls/ceilings or during contact with actors.
        if self.contact(a) or self.contact(b):
            return
        d = direction(action)
        dv = b.vx - a.vx
        # A sharp horizontal stop is collision evidence, not a friction sample.
        if abs(dv) <= 1.2 and a.grounded and b.grounded and not b.swimming:
            if d and d * a.vx < 0:
                self.add("brake_accel", dv * d)
            elif d and abs(a.vx) < (2.3 if "run" in action.value else 1.3):
                name = "run_accel" if "run" in action.value else "walk_accel"
                self.add(name, dv * d)
        pressed = action in JUMP_ACTIONS and a.previous_action not in JUMP_ACTIONS
        if b.swimming:
            if pressed and b.vy > 0:
                self.add("swim_impulse", b.vy)
            elif not pressed and not b.grounded:
                self.add("swim_gravity", a.vy - b.vy)
        elif a.grounded and not b.grounded and action in JUMP_ACTIONS and b.vy > 0:
            fast = abs(a.vx) >= 2 if fast_jump is None else fast_jump
            self.add("jump_speed_fast" if fast else "jump_speed", b.vy)
        elif not a.grounded and not b.grounded and b.vy - a.vy <= 1.2 and b.vy > -4.5:
            name = "gravity_hold" if a.vy > 0 and action in JUMP_ACTIONS else "gravity_fall"
            if name == "gravity_hold":
                return  # Held ascent is fitted from phase-consistent positions.
            self.add(name, a.vy - b.vy)

    def summary(self) -> dict:
        return {
            "samples": sum(map(len, self.samples.values())),
            "source": "actual_consecutive_observations_only",
            "parameters": {
                name: {
                    "estimate": round(self.value(name), 3),
                    "calibrated_prior": self.calibrated.get(name),
                    "samples": len(self.samples.get(name, ())),
                    "observed_mean": round(fmean(self.samples[name]), 3)
                    if self.samples.get(name)
                    else None,
                }
                for name in PRIORS
            },
        }


class Parameters:
    def __init__(self, values):
        self.value = values.__getitem__
