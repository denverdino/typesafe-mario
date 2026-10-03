"""Causal, observation-only estimates. This module cannot advance a game emulator.

Ranges are heuristic uncertainty envelopes, not calibrated probabilities or safety
certificates. Every path is conditional on holding its requested action; Jev must
reconsider that assumption when fresh observations arrive.
"""

from collections import deque
from dataclasses import dataclass, replace
from itertools import pairwise
from math import ceil
from time import perf_counter

from .actions import JUMP_ACTIONS, Action, first_frame_action
from .actor_motion import WALKERS, actor_path
from .continuation import MAX_SEARCH_FRAMES, Node, actor_contacts, continuations
from .dynamics import Dynamics, direction, position_fit
from .landing import landing_window, reaction_tail_frames
from .observation import Observation
from .routing import enemy_crossing_geometry, underpass_goal
from .shells import ObservedShells, kick_forecasts, possible_ground_kick
from .timing import DecisionTiming

BACKEND = "observation_dynamics"
VERSION = 54
MAX_HORIZON = MAX_SEARCH_FRAMES


@dataclass
class Motion:
    x: float
    y: float
    vx: float
    vy: float
    grounded: bool
    previous_action: str | None
    fast_jump: bool = False
    bouncing: bool = False
    jump_pending: bool = False
    left_boundary: float | None = None


def _advance(
    m: Motion, action: Action, o: Observation, dynamics: Dynamics
) -> tuple[Motion, list[str]]:
    """One inexpensive approximate physics step against observed geometry."""
    s = Motion(
        m.x,
        m.y,
        m.vx,
        m.vy,
        m.grounded,
        m.previous_action,
        m.fast_jump,
        m.bouncing,
        left_boundary=m.left_boundary,
    )
    events = []
    d = direction(action)
    if d:
        name = (
            "air_accel"
            if not s.grounded
            else "brake_accel"
            if d * s.vx < 0
            else "run_accel"
            if "run" in action.value
            else "walk_accel"
        )
        accel = dynamics.value(name)
        cap = 2.0 if o.swimming else 3.0 if "run" in action.value else 1.6
        if not s.grounded and not o.swimming:
            # Observed flight speed regime persists across button changes.
            # Pressing B after walking takeoff cannot create a running flight.
            cap = 3.0 if s.fast_jump else dynamics.value("air_walk_speed")
        # Releasing B does not instantly erase existing momentum.
        s.vx += d * accel if d * s.vx < cap else 0
        s.vx = max(-3.5, min(3.5, s.vx))
    elif s.grounded:
        s.vx = (1 if s.vx >= 0 else -1) * max(0, abs(s.vx) - dynamics.value("friction"))
    pressed = action in JUMP_ACTIONS and (s.previous_action not in JUMP_ACTIONS or m.jump_pending)
    if o.swimming:
        s.vy = (
            dynamics.value("swim_impulse")
            if pressed
            else max(-2.5, s.vy - dynamics.value("swim_gravity"))
        )
        s.grounded = False
    elif pressed and s.grounded and m.jump_pending:
        s.fast_jump = abs(m.vx) >= 2
        s.vy = dynamics.value("jump_speed_fast" if s.fast_jump else "jump_speed")
        s.grounded = False
    elif not s.grounded:
        name = (
            "gravity_hold"
            if s.vy > 0 and action in JUMP_ACTIONS and not s.bouncing
            else "gravity_fall"
        )
        if name == "gravity_hold" and s.fast_jump:
            name = "gravity_hold_fast"
        gravity = dynamics.value(name)
        s.vy = max(-5.0, s.vy - gravity)
    nx, ny = s.x + s.vx, s.y + s.vy
    if s.left_boundary is not None and nx < s.left_boundary:
        nx, s.vx = s.left_boundary, 0
        events.append("observed_left_boundary")
    for b in o.blocks:
        if s.y < b.top - 0.01 and s.y + o.height > b.bottom + 0.01:
            # Resolve same-frame upward clearance before a side stop. Using
            # only the old height can invent a collision just as Mario clears
            # a pipe/brick top, erasing the horizontal momentum of the flight.
            if s.vy > 0 and ny >= b.top:
                continue
            # Repeated observed pipe contacts put the wall at x+14. The sprite
            # is16px wide, but its horizontal collision body has a2px inset.
            right_hit = s.vx > 0 and s.x + 2 < b.left and nx + 14 > b.left
            left_hit = s.vx < 0 and s.x + 14 > b.right and nx + 2 < b.right
            side_x = b.left + 1 if right_hit else b.right - 1
            bottom = b.bottom
            if (right_hit or left_hit) and s.vy < 0 and ny < bottom:
                # Tiles can describe one solid face in several vertical rows.
                # Trace to its exposed underside, not an internal row seam.
                while True:
                    lower = min(
                        (
                            c.bottom
                            for c in o.blocks
                            if c.bottom < bottom <= c.top and c.left <= side_x < c.right
                        ),
                        default=bottom,
                    )
                    if lower == bottom:
                        break
                    bottom = lower
            if (right_hit or left_hit) and s.vy < 0 and ny < bottom:
                # A descending body can skim an underside rather than stop
                # at its side. The full sprite box overstates side contact.
                # Keep momentum and uncertainty; never rely on this corner
                # to brake before an enemy. Full-height walls still stop us.
                events.append("uncertain_wall_contact")
            elif right_hit:
                nx, s.vx = min(nx, b.left - 14), 0
                events.append("wall_contact")
            elif left_hit:
                nx, s.vx = max(nx, b.right - 2), 0
                events.append("wall_contact")
    s.grounded = False
    for b in o.blocks:
        if nx + 13 <= b.left or nx + 3 >= b.right:
            continue
        if s.vy <= 0 and s.y >= b.top - 0.01 and ny <= b.top:
            # Observed edge departures lose jump support before the full sprite
            # leaves the floor. Use an inset foot span with a pixel of margin;
            # otherwise releasing A at an edge invents a second takeoff.
            if nx + 11 <= b.left or nx + 5 >= b.right:
                events.append("marginal_support")
                continue
            ny, s.vy, s.grounded = max(ny, b.top), 0, True
            if not m.grounded:
                events.append("possible_landing")
        elif (
            s.vy > 0
            and s.y < b.bottom
            and ny + o.height > b.bottom
            and b.left <= (s.x if s.y + o.height > b.bottom else nx) + 8 < b.right
        ):
            # Telemetry and our approximate body box may already overlap the
            # underside. Resolve an existing head overlap before moving out.
            # A sprite edge grazing a brick is not a head collision: recorded
            # small-Mario jumps continue rising when the head center clears it.
            ny, s.vy = min(ny, b.bottom - o.height), 0
            events.append("ceiling_contact")
    if s.grounded:
        s.fast_jump = False
        s.bouncing = False
        # Observed fresh A first moves on support; ascent starts on the next
        # held step. A press while leaving the edge cannot create a takeoff.
        s.jump_pending = pressed and not m.jump_pending
    elif m.grounded and not pressed and not o.swimming:
        s.fast_jump = abs(m.vx) >= 2
    s.x, s.y, s.previous_action = nx, ny, action.value
    return s, events


def _estimate(m: Motion, frame: int, *, uncertain: bool) -> dict:
    # Time and unmodelled transitions broaden intervals; no statistical coverage claim.
    ex = 1 + 0.2 * frame + 0.01 * frame * frame
    ey = 1 + 0.3 * frame + 0.015 * frame * frame
    if uncertain:
        ex *= 1.5
        ey *= 1.5
    return {
        "frame": frame,
        "x": round(m.x, 2),
        "y": round(m.y, 2),
        "vx": round(m.vx, 2),
        "vy": round(m.vy, 2),
        "grounded_estimate": m.grounded,
        "x_range": [
            round(max(m.x - ex, m.left_boundary) if m.left_boundary is not None else m.x - ex, 2),
            round(m.x + ex, 2),
        ],
        "y_range": [round(m.y - ey, 2), round(m.y + ey, 2)],
    }


def _include_motion_variants(estimate, variants):
    """Keep topology-changing speed sensitivity inside the reported envelope."""
    for motion in variants:
        other = _estimate(motion, estimate["frame"], uncertain=False)
        for key in ("x_range", "y_range"):
            estimate[key] = [
                min(estimate[key][0], other[key][0]),
                max(estimate[key][1], other[key][1]),
            ]


def _lower_floor_catches(m, estimate, action, o, parameters):
    """Check an uncertain step-down using visible geometry and unchanged input."""
    if action in JUMP_ACTIONS or not any(32 <= b.top < m.y - 1 for b in o.blocks):
        return False
    landings = []
    for x in estimate["x_range"]:
        sample = replace(m, x=x)
        caught = False
        for _ in range(32):
            sample, _ = _advance(sample, action, o, parameters)
            if sample.y < 32 or sample.x < o.known_x[0] or sample.x + 16 > o.known_x[1]:
                break
            if (
                sample.grounded
                and sample.y < m.y - 1
                and any(
                    abs(b.top - sample.y) < 0.1 and b.left + 4 <= sample.x + 8 <= b.right - 4
                    for b in o.blocks
                )
            ):
                caught = True
                landings.append(sample)
                break
        if not caught:
            return False
    return any(
        all(
            abs(b.top - landing.y) < 0.1 and b.left + 4 <= landing.x + 8 <= b.right - 4
            for landing in landings
        )
        for b in o.blocks
    )


class Predictor:
    """Per-episode learning and validation, updated only by real observations."""

    def __init__(self, *, physics_parameters=None):
        self.physics_parameters = dict(physics_parameters or {})
        self.dynamics = Dynamics(calibrated=self.physics_parameters)
        self.history = deque(maxlen=16)
        self.pending = []
        self.errors = deque(maxlen=256)
        self.pending_stomps = {}
        self.defeated_goombas = {}
        self.fast_jump = None
        self.bouncing = False
        self.left_boundary = None
        self.shells = ObservedShells()
        self.selected_plan = None

    def _observed_interactions(self, a, b):
        actors = {actor.identity: actor for actor in b.actors}
        self.defeated_goombas = {
            identity: position
            for identity, position in self.defeated_goombas.items()
            if identity in actors
            and abs(actors[identity].x - position[0]) <= 2
            and abs(actors[identity].y - position[1]) <= 2
        }
        if b.frame != a.frame + 1 or not (a.motion_reliable and b.motion_reliable):
            self.pending_stomps.clear()
            self.defeated_goombas.clear()
            return
        # A real bounce is evidence; a hypothetical stomp is not. Scoreboard
        # updates may follow a few frames later. Never infer a defeated Koopa:
        # its shell remains an interaction hazard.
        if a.vy < 0 < b.vy and not a.grounded and not b.grounded and not a.swimming:
            possible = [
                actor
                for actor in a.actors
                if a.x + 14 > actor.x + 2
                and a.x + 2 < actor.x + 14
                and actor.y + 7 <= a.y <= actor.y + 24
            ]
            if (
                len(possible) == 1
                and possible[0].kind == "goomba"
                and not any(
                    actor.kind == "springboard" and abs(actor.x - a.x) < 24 for actor in a.actors
                )
            ):
                actor = possible[0]
                self.pending_stomps[actor.identity] = (b.frame, actor.x, actor.y, a.score, b.vy)
        remaining = {}
        for identity, (frame, x, y, score, bounce_speed) in self.pending_stomps.items():
            actor = actors.get(identity)
            if not actor or b.frame - frame > 8 or abs(actor.x - x) > 2 or abs(actor.y - y) > 2:
                continue
            if b.score >= score + 100 and b.frame > frame:
                self.defeated_goombas[identity] = (actor.x, actor.y)
                self.dynamics.add("stomp_speed", bounce_speed)
                self.bouncing = not b.grounded and all(
                    not h.grounded and not h.swimming for h in self.history if h.frame >= frame
                )
            else:
                remaining[identity] = (frame, x, y, score, bounce_speed)
        self.pending_stomps = remaining

    def observe(self, observation: Observation):
        o = observation
        last = self.history[-1] if self.history else None
        if last and o == last:
            return
        if last and (
            o.level != last.level
            or o.frame <= last.frame
            or abs(o.x - last.x) > 32 * max(1, o.frame - last.frame)
            or abs(o.y - last.y) > 32 * max(1, o.frame - last.frame)
        ):
            self.__init__(physics_parameters=self.physics_parameters)
            last = None
        if (
            not o.motion_reliable
            or o.terminal
            or o.known_x is None
            or (last and (o.frame != last.frame + 1 or o.known_x != last.known_x))
            or (self.left_boundary is not None and o.x < self.left_boundary)
        ):
            self.left_boundary = None
        if last:
            if o.frame != last.frame + 1 or not o.motion_reliable or o.terminal:
                self.selected_plan = None
            elif self.selected_plan:
                plan = self.selected_plan
                step = o.frame - 1 - plan["apply_at_frame"]
                index = step // plan["cycle"]
                if 0 <= index < len(plan["actions"]):
                    expected = plan["actions"][index]
                    if step % plan["cycle"] == 0:
                        expected = first_frame_action(
                            expected,
                            grounded=last.grounded,
                            previous_action=last.previous_action,
                            swimming=last.swimming,
                        )
                    if o.previous_action != expected.value:
                        self.selected_plan = None
            self.shells.observe(last, o)
            self._observed_interactions(last, o)
            if (
                o.frame != last.frame + 1
                or not (last.motion_reliable and o.motion_reliable)
                or last.swimming != o.swimming
            ) or o.grounded:
                self.fast_jump = None
                self.bouncing = False
            elif last.grounded and not o.swimming:
                takeoff_vx = last.vx
                ground_history = list(self.history)[-9:]
                if len(ground_history) == 9 and all(
                    h.frame == last.frame - 8 + i
                    and h.level == last.level
                    and h.grounded
                    and not h.swimming
                    and h.motion_reliable
                    and not h.terminal
                    and not self.dynamics.contact(h)
                    and h.previous_action is not None
                    and last.previous_action is not None
                    and (
                        i == 0
                        or (
                            direction(Action(h.previous_action))
                            == direction(Action(last.previous_action))
                            and ("run" in h.previous_action) == ("run" in last.previous_action)
                        )
                    )
                    for i, h in enumerate(ground_history)
                ):
                    fitted_vx, acceleration = position_fit([h.x for h in ground_history])
                    if abs(fitted_vx) <= 3.5 and abs(acceleration) <= 0.8:
                        takeoff_vx = fitted_vx
                self.fast_jump = abs(takeoff_vx) >= 2
            a, b = last, o
            # Integer pixel deltas quantize acceleration to 0/1 px/frame².
            # Overlapping three-frame velocity windows recover subpixel trends
            # using past positions only. Never smooth across input/mode changes.
            if len(self.history) >= 4:
                recent = list(self.history)[-4:] + [o]
                if all(
                    h.frame == recent[0].frame + i
                    and h.grounded == o.grounded
                    and h.swimming == o.swimming
                    and h.previous_action == o.previous_action
                    and h.motion_reliable
                    and not self.dynamics.contact(h)
                    for i, h in enumerate(recent)
                ):
                    a = replace(last, vx=(last.x - recent[0].x) / 3, vy=(last.y - recent[0].y) / 3)
                    b = replace(o, vx=(o.x - recent[1].x) / 3, vy=(o.y - recent[1].y) / 3)
            self.dynamics.learn(a, b, fast_jump=self.fast_jump)
            self.dynamics.learn_air([*self.history, o])
            self.dynamics.learn_friction([*self.history, o])
            self.dynamics.learn_held_gravity(
                [*self.history, o], fast_jump=None if self.bouncing else self.fast_jump
            )
            remaining = []
            for target, start, action, estimate in self.pending:
                actual = (
                    first_frame_action(
                        action,
                        grounded=last.grounded,
                        previous_action=last.previous_action,
                        swimming=last.swimming,
                    )
                    if o.frame == start + 1
                    else action
                )
                if (
                    o.frame != last.frame + 1
                    or o.previous_action != actual.value
                    or not (last.motion_reliable and o.motion_reliable)
                ):
                    continue
                if o.frame == target:
                    self.errors.append(
                        (
                            abs(o.x - estimate["x"]),
                            abs(o.y - estimate["y"]),
                            estimate["x_range"][0] <= o.x <= estimate["x_range"][1]
                            and estimate["y_range"][0] <= o.y <= estimate["y_range"][1],
                        )
                    )
                elif o.frame < target:
                    remaining.append((target, start, action, estimate))
            self.pending = remaining
        self.history.append(o)
        recent = list(self.history)[-5:]
        if (
            len(recent) == 5
            and o.known_x
            and all(
                h.frame == o.frame - 4 + i
                and h.level == o.level
                and h.known_x == o.known_x
                and h.motion_reliable
                and not h.terminal
                and h.previous_action == "left"
                and h.x == o.known_x[0]
                for i, h in enumerate(recent)
            )
        ):
            # Repeated real input establishes a local positional constraint.
            # It supplies neither offscreen tiles nor a global camera rule.
            self.left_boundary = o.x

    def calibration(self) -> dict:
        return self.dynamics.summary()

    def validation(self) -> dict:
        n = len(self.errors)
        return {
            "samples": n,
            "scope": "executed_first_cycles_only",
            "mean_absolute_x_error": round(sum(e[0] for e in self.errors) / n, 2) if n else None,
            "mean_absolute_y_error": round(sum(e[1] for e in self.errors) / n, 2) if n else None,
            "range_coverage": round(sum(e[2] for e in self.errors) / n, 3) if n else None,
        }

    def expect(self, prediction: dict, action: Action, *, apply_at_frame: int):
        """Register only the action actually selected and about to execute."""
        if (
            prediction.get("backend") != BACKEND
            or prediction.get("apply_at_frame") != apply_at_frame
        ):
            return
        branch = next(
            (f for f in prediction["action_forecasts"] if f["action"] == action.value), None
        )
        if branch is not None:
            plan = branch.get("continuation", {})
            self.selected_plan = {
                "apply_at_frame": apply_at_frame,
                "cycle": prediction["action_cycle_frames"],
                "actions": tuple(Action(a) for a in plan.get("actions", ())),
            }
            self.pending.append(
                (
                    apply_at_frame + prediction["action_cycle_frames"],
                    apply_at_frame,
                    action,
                    branch["first_cycle"],
                )
            )

    def _enemy_motion(self, o: Observation, *, fall_gravity=None) -> dict:
        result = {}
        # Never consume observations from a future frame, even if asked to forecast
        # a historical observation. Only a contiguous visible track supplies velocity.
        history = [h for h in self.history if h.frame <= o.frame and h.level == o.level]
        for actor in o.actors:
            samples = [(o.frame, actor)]
            for h in reversed(history):
                if h.frame >= o.frame:
                    continue
                old = next((a for a in h.actors if a.identity == actor.identity), None)
                if (
                    old is None
                    or old.kind != actor.kind
                    or samples[-1][0] - h.frame != 1
                    or abs(samples[-1][1].x - old.x) > 16
                    or abs(samples[-1][1].y - old.y) > 16
                ):
                    break
                samples.append((h.frame, old))
                if len(samples) >= 9:
                    break
            if len(samples) > 1:
                f, a = samples[-1]
                dt = o.frame - f
                vx = (actor.x - a.x) / dt
                steps = [new.x - old.x for (_, new), (_, old) in pairwise(samples)]
                if actor.kind in WALKERS and min(steps) < 0 < max(steps):
                    # Pixel quantization can leave zero steps after a turn. Use
                    # the latest nonzero direction without cancelling speed
                    # against motion before the observed reversal.
                    latest = next(dx for dx in steps if dx)
                    vx = sum(abs(dx) for dx in steps) / dt * (1 if latest > 0 else -1)
                vy = (actor.y - a.y) / dt
                if actor.kind in WALKERS and len(samples) == 9:
                    heights = [a.y for _, a in reversed(samples)]
                    if all(a >= b for a, b in pairwise(heights)) and heights[0] > heights[-1]:
                        fitted_vy, acceleration = position_fit(heights)
                        if -5 <= fitted_vy <= 0 and -0.65 <= acceleration <= -0.05:
                            vy = fitted_vy
                            if fall_gravity is not None:
                                fall_gravity[actor.identity] = -acceleration
                result[actor.identity] = (vx, vy, True)
            else:
                result[actor.identity] = (0, 0, False)
        return result

    def _recent_motion(self, o: Observation) -> list[dict]:
        history = [h for h in self.history if h.frame <= o.frame and h.level == o.level]
        segments = []
        for a, b in pairwise(history):
            if b.frame != a.frame + 1 or not (a.motion_reliable and b.motion_reliable):
                segments.clear()
                continue
            if not segments or segments[-1]["action"] != b.previous_action:
                segments.append(
                    {
                        "start_frame": a.frame,
                        "action": b.previous_action,
                        "start": {"x": a.x, "y": a.y},
                    }
                )
            segments[-1].update(end_frame=b.frame, end={"x": b.x, "y": b.y, "grounded": b.grounded})
        return segments[-4:]

    def forecast(
        self,
        o: Observation,
        actions,
        *,
        cycle: int,
        delay: int = 0,
        scheduled_action: Action | None = None,
        scheduled_first_frame_action: Action | None = None,
    ) -> dict:
        timing = DecisionTiming(o.frame, delay, cycle)
        if delay and scheduled_action is None:
            raise ValueError("Positive cycle, nonnegative delay and committed buttons are required")
        if self.history and o.frame < self.history[-1].frame:
            raise ValueError(
                "Cannot forecast a historical frame using subsequently learned dynamics"
            )
        started = perf_counter()
        actions = tuple(actions)
        shells = self.shells.confirmed(o) if self.history and self.history[-1] == o else {}
        result = {
            "backend": BACKEND,
            "version": VERSION,
            "status": "estimated",
            **timing.to_state(),
            "frame_reference": "offset_from_observed_at_frame",
            "action_forecasts": [],
            "assumptions": "Approximate observation-calibrated motion and visible static tiles. "
            "Each candidate repeats at cycle boundaries. No emulator trials, hidden timers "
            "or future spawns. Ranges are heuristic, not calibrated confidence intervals. "
            "Contacts, stomps, bounces and transitions can invalidate a trajectory.",
            "calibration": self.calibration(),
            "validation": self.validation(),
            "recent_motion": self._recent_motion(o),
            "observed_interactions": [
                {
                    "actor_id": identity,
                    "inference": "defeated_goomba",
                    "evidence": "observed_bounce_stationary_actor_score_gain",
                }
                for identity in self.defeated_goombas
            ],
        }
        if cycle > MAX_HORIZON or delay > MAX_HORIZON or o.terminal:
            result.update(
                status="unavailable", reason="terminal_or_horizon_out_of_bounds", compute_ms=0.0
            )
            return result
        result["observed_interactions"].extend(
            {
                "actor_id": identity,
                "inference": shell.state,
                "evidence": "observed_koopa_stomp_then_stationary"
                if shell.state == "stationary_shell"
                else "observed_shell_then_motion",
            }
            for identity, shell in shells.items()
        )
        vx = o.vx
        recent = list(self.history)[-5:]
        if (
            len(recent) == 5
            and recent[-1] == o
            and all(
                h.frame == o.frame - 4 + i
                and h.grounded == o.grounded
                and h.swimming == o.swimming
                and h.previous_action == o.previous_action
                and h.motion_reliable
                and not self.dynamics.contact(h)
                for i, h in enumerate(recent)
            )
        ):
            vx = (o.x - recent[0].x) / 4
        velocity_history = list(self.history)[-9:]
        takeoff_window = (
            not o.grounded
            and o.previous_action in JUMP_ACTIONS
            and not self.bouncing
            and not any(not a.grounded and b.grounded for a, b in pairwise(velocity_history))
        )
        if (
            len(velocity_history) == 9
            and velocity_history[-1] == o
            and all(
                h.frame == o.frame - 8 + i
                and h.level == o.level
                and (h.grounded == o.grounded or takeoff_window)
                and h.swimming == o.swimming
                and (i == 0 or h.previous_action == o.previous_action)
                and h.motion_reliable
                and not h.terminal
                and not self.dynamics.contact(h)
                for i, h in enumerate(velocity_history)
            )
        ):
            # The short displacement average describes past midpoint speed.
            # Fit the current end-of-step speed during a continuous input so
            # accelerating approaches and braking retreats do not inherit it.
            # A takeoff changes vertical mode, not horizontal continuity; keep
            # that window rather than falling back to its stale midpoint speed.
            fitted_vx, acceleration = position_fit([h.x for h in velocity_history])
            displacement = o.x - velocity_history[0].x
            if (
                abs(fitted_vx) <= 3.5
                and abs(acceleration) <= 0.8
                and any(h.x != o.x for h in recent)
                and (o.previous_action != "noop" or fitted_vx * displacement >= 0)
            ):
                vx = fitted_vx
        vy = o.vy
        if (
            not self.bouncing
            and len(recent) == 5
            and recent[-1] == o
            and all(
                h.frame == o.frame - 4 + i
                and h.level == o.level
                and not h.grounded
                and not h.swimming
                and h.vy > 0
                and h.previous_action in JUMP_ACTIONS
                and h.motion_reliable
                and not self.dynamics.contact(h)
                for i, h in enumerate(recent)
            )
        ):
            # A single integer height delta can misclassify a feasible jump.
            # Average four observed displacements, then bring that midpoint
            # velocity forward using the fitted held-ascent deceleration.
            gravity = self.dynamics.value("gravity_hold_fast" if self.fast_jump else "gravity_hold")
            vy = max(0, (o.y - recent[0].y) / 4 - 1.5 * gravity)
        vertical_history = list(self.history)[-9:]
        if (
            vy == 0
            and not self.bouncing
            and not o.grounded
            and not o.swimming
            and len(vertical_history) == 9
            and vertical_history[-1] == o
            and all(
                h.frame == o.frame - 8 + i
                and h.level == o.level
                and not h.grounded
                and not h.swimming
                and h.motion_reliable
                and h.previous_action in JUMP_ACTIONS
                and not self.dynamics.contact(h)
                for i, h in enumerate(vertical_history)
            )
            and all(a.y <= b.y for a, b in pairwise(vertical_history))
        ):
            # A zero integer height delta can still be a subpixel ascent. Keep
            # that measured phase rather than switching to fall gravity early.
            fitted_vy, _ = position_fit([h.y for h in vertical_history])
            if 0 < fitted_vy < 1:
                vy = fitted_vy
        motion = Motion(
            o.x, o.y, vx, vy, o.grounded, o.previous_action, bool(self.fast_jump), self.bouncing
        )
        variant_speeds = []
        if (
            delay
            and o.grounded
            and not o.swimming
            and o.motion_reliable
            and o.previous_action is not None
            and direction(Action(o.previous_action)) * direction(scheduled_action) < 0
        ):
            # Pixel positions do not resolve instantaneous speed while changing
            # direction. A one-pixel/frame error can decide whether a jump hits
            # a low ceiling. Propagate both speeds through the committed input
            # and executable cycles; these are model paths, never emulator trials.
            variant_speeds = [max(-3.5, min(3.5, vx + dv)) for dv in (-1, 1)]
            result["motion_sensitivity"] = {
                "reason": "ground_input_reversal",
                "initial_vx_range": variant_speeds,
                "assumption": "Heuristic one-pixel/frame speed sensitivity, not a confidence bound. "
                "Different ceiling/wall outcomes invalidate a clear-route comparison.",
            }
        if len(self.history) >= 2 and self.history[-1] == o:
            previous = self.history[-2]
            motion.jump_pending = bool(
                o.grounded
                and o.motion_reliable
                and previous.motion_reliable
                and previous.level == o.level
                and previous.frame + 1 == o.frame
                and o.previous_action in JUMP_ACTIONS
                and previous.previous_action not in JUMP_ACTIONS
            )
        if self.history and self.history[-1] == o:
            motion.left_boundary = self.left_boundary
        if motion.left_boundary is not None:
            result["observed_boundary"] = {
                "left_x": motion.left_boundary,
                "evidence": "repeated_left_input_without_horizontal_motion_at_visible_edge",
                "assumption": "Applies while the observed viewport is unchanged; motion remains estimated.",
            }
        motion_variants = [replace(motion, vx=speed) for speed in variant_speeds]
        parameters = self.dynamics.fitted()
        fall_gravity = {}
        enemy_motion = self._enemy_motion(o, fall_gravity=fall_gravity)
        for identity, shell in shells.items():
            enemy_motion[identity] = (shell.vx, shell.vy, shell.velocity_known)
        active_actors = tuple(
            replace(a, kind=shells[a.identity].state) if a.identity in shells else a
            for a in o.actors
            if a.identity not in self.defeated_goombas
        )
        actor_observation = replace(o, actors=active_actors)
        # Build once for both the immutable input prefix and candidate paths.
        actor_paths = {
            a.identity: actor_path(
                a,
                enemy_motion[a.identity],
                o,
                delay + MAX_HORIZON + reaction_tail_frames(cycle, parameters.value("brake_accel")),
                gravity=fall_gravity.get(a.identity, 0.25),
            )
            for a in active_actors
        }
        committed = Node(motion)
        committed_warnings = set()
        committed_events = []
        last_landing = None
        for frame in range(delay):
            actual = (
                (
                    scheduled_first_frame_action
                    or first_frame_action(
                        scheduled_action,
                        grounded=motion.grounded,
                        previous_action=motion.previous_action,
                        swimming=o.swimming,
                    )
                )
                if frame == 0
                else scheduled_action
            )
            previous = motion
            motion, events = _advance(motion, actual, o, parameters)
            for i, variant in enumerate(motion_variants):
                variant_action = (
                    (
                        scheduled_first_frame_action
                        or first_frame_action(
                            scheduled_action,
                            grounded=variant.grounded,
                            previous_action=variant.previous_action,
                            swimming=o.swimming,
                        )
                    )
                    if frame == 0
                    else scheduled_action
                )
                motion_variants[i], variant_events = _advance(
                    variant, variant_action, o, parameters
                )
                if {"ceiling_contact", "wall_contact"} & set(variant_events) != {
                    "ceiling_contact",
                    "wall_contact",
                } & set(events):
                    committed_warnings.add("uncertain_wall_contact")
                    committed_events.append("uncertain_wall_contact")
            if "possible_landing" in events:
                envelope = _estimate(previous, frame + 1, uncertain=not o.motion_reliable)
                vertical_margin = (envelope["y_range"][1] - envelope["y_range"][0]) / 2
                last_landing = {
                    "frame": frame + 1,
                    "timing_margin_frames": ceil(vertical_margin / max(1, abs(previous.vy))) + 1,
                }
            committed_events.extend(events)
            if "uncertain_wall_contact" in events:
                committed_warnings.add("uncertain_wall_contact")
            committed.elapsed = frame + 1
            motion = actor_contacts(
                previous,
                motion,
                committed,
                committed_warnings,
                actor_observation,
                actor_paths,
                frame + 1,
                parameters,
            )
            if committed.stopped:
                break
        result["committed_future"] = _estimate(motion, delay, uncertain=not o.motion_reliable)
        _include_motion_variants(result["committed_future"], motion_variants)
        uncertain_takeoff = bool(
            not o.swimming
            and motion.grounded
            and last_landing
            and delay - last_landing["frame"] <= last_landing["timing_margin_frames"]
        )
        if last_landing:
            result["committed_future"]["landing_timing"] = {
                **last_landing,
                "uncertain_takeoff": uncertain_takeoff,
                "assumption": "Heuristic timing margin; landing near application may occur "
                "after A is pressed, so holding A does not establish a new jump.",
            }
        committed_uncertain = bool(committed.interactions or committed.stopped)
        if committed_uncertain:
            result["committed_interactions"] = list(committed.interactions)
            result["committed_future"].update(
                conditional=True,
                projected_frames=committed.elapsed,
                contact_failure=committed.failure,
                valid_for_application=not committed.stopped,
            )
            committed_warnings.add("uncertain_committed_interaction")
        # A jump outlives three eight-frame decisions. Resolve its landing before
        # calling a release/retreat safer merely because it falls after the cutoff.
        horizon = min(MAX_HORIZON, max(48, cycle * 4))
        result["horizon_after_application_frames"] = horizon
        crossing = enemy_crossing_geometry(replace(o, actors=active_actors), motion, enemy_motion)
        if crossing:
            result["enemy_crossing"] = crossing
        runup = motion.grounded and any(
            0 <= b.left - motion.x <= 48
            and b.top - motion.y >= 48
            and any(
                c.left == b.left and c.bottom <= motion.y + o.height and c.top > motion.y
                for c in o.blocks
            )
            for b in o.blocks
        )
        runup = runup or crossing is not None
        navigation_goal = underpass_goal(o, motion)
        if navigation_goal:
            result["navigation_goal"] = navigation_goal
        plan_horizon = max(horizon, 80) if runup or navigation_goal else horizon
        sample_frames = sorted(
            {delay, delay + min(cycle, plan_horizon), delay + min(2 * cycle, plan_horizon)}
        )
        result["actor_forecasts"] = []
        for actor in active_actors:
            avx, avy, known = enemy_motion[actor.identity]
            samples = []
            for frame in sample_frames:
                x, y, supported = actor_paths[actor.identity][frame]
                samples.append(
                    {
                        "frame": frame,
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "uncertainty_margin_pixels": round(
                            1 + frame * (0.35 if supported else 2), 2
                        ),
                        "motion_uncertain": not supported,
                    }
                )
            result["actor_forecasts"].append(
                {
                    "id": actor.identity,
                    "kind": actor.kind,
                    "state": actor.kind,
                    "observed_velocity": {
                        "x": round(avx, 3),
                        "y": round(avy, 3),
                        "available": known,
                    },
                    "trajectory": samples,
                }
            )
            if actor.kind == "stationary_shell":
                result["actor_forecasts"][-1]["kick_forecasts"] = kick_forecasts(
                    actor, o, plan_horizon
                )
        for action in actions:
            m = replace(motion)
            variants = [replace(v) for v in motion_variants]
            unknowns = {"model_error", "unseen_spawns"}
            if o.swimming:
                unknowns.add("swimming_dynamics")
            hazards = {}
            conditional_shells = set()
            events = set(committed_events)
            trajectory = []
            possible_fall = False
            first_landing = None
            unsupported_descent = False
            contact_exposure = {"frames": 0, "overlap_area_sum": 0.0, "end_overlap_area": 0.0}
            flight_committed = not o.swimming and (not motion.grounded or action in JUMP_ACTIONS)
            control_risk = {
                "through_frame": delay + min(horizon, 2 * cycle),
                "possible_contact": False,
                "nominal_contact": False,
                "possible_fall": False,
                "uncertain_landing": False,
                "unobserved_terrain": False,
                "unreliable_motion": not o.motion_reliable,
                "uncertain_wall_contact": "uncertain_wall_contact" in committed_events,
                "uncertain_committed_interaction": committed_uncertain,
                "uncertain_takeoff": uncertain_takeoff and action in JUMP_ACTIONS,
                "landing_contact_window": False,
            }
            for step in range(horizon):
                actual = (
                    first_frame_action(
                        action,
                        grounded=m.grounded,
                        previous_action=m.previous_action,
                        swimming=o.swimming,
                    )
                    if step % cycle == 0
                    else action
                )
                m, emitted = _advance(m, actual, o, parameters)
                frame = delay + step + 1
                if frame <= control_risk["through_frame"]:
                    for i, variant in enumerate(variants):
                        variant_action = (
                            first_frame_action(
                                action,
                                grounded=variant.grounded,
                                previous_action=variant.previous_action,
                                swimming=o.swimming,
                            )
                            if step % cycle == 0
                            else action
                        )
                        variants[i], variant_events = _advance(
                            variant, variant_action, o, parameters
                        )
                        if {"ceiling_contact", "wall_contact"} & set(variant_events) != {
                            "ceiling_contact",
                            "wall_contact",
                        } & set(emitted):
                            control_risk["uncertain_wall_contact"] = True
                            unknowns.add("uncertain_wall_contact")
                if "uncertain_wall_contact" in emitted:
                    unknowns.add("uncertain_wall_contact")
                    if frame <= control_risk["through_frame"]:
                        control_risk["uncertain_wall_contact"] = True
                events.update(emitted)
                if "possible_landing" in emitted and not o.swimming:
                    landing = landing_window(
                        m,
                        actor_observation,
                        actor_paths,
                        frame,
                        step + 1,
                        cycle,
                        parameters,
                        committed.removed_actors,
                    )
                    if first_landing is None:
                        first_landing = landing
                    if frame <= control_risk["through_frame"] or flight_committed:
                        control_risk["landing_contact_window"] |= any(
                            e["contact_before_escape"] or not e["reaction_window_complete"]
                            for e in landing["enemies"]
                        )
                estimate = _estimate(m, frame, uncertain=not o.motion_reliable)
                if frame <= control_risk["through_frame"]:
                    _include_motion_variants(estimate, variants)
                if (
                    m.grounded
                    and not o.swimming
                    and frame <= control_risk["through_frame"]
                    and o.known_x is not None
                    and o.known_x[0] <= estimate["x_range"][0] + 8
                    and estimate["x_range"][1] + 8 < o.known_x[1]
                    and not any(
                        32 <= b.top <= m.y + 1
                        and b.left <= estimate["x_range"][0] + 8
                        and estimate["x_range"][1] + 8 < b.right
                        for b in o.blocks
                    )
                    and not _lower_floor_catches(m, estimate, actual, o, parameters)
                ):
                    # A nominal grounded endpoint can still overrun a gap.
                    # Preserve the executable-cycle warning even when a later
                    # nominal continuation plans to jump or brake at that edge.
                    # A visible lower floor spanning the range remains a refuge.
                    control_risk["uncertain_landing"] = True
                    unknowns.add("uncertain_support")
                if (
                    ("possible_landing" in emitted or "marginal_support" in emitted)
                    and (frame <= control_risk["through_frame"] or flight_committed)
                    and not any(
                        abs(b.top - m.y) < 0.1 and b.left + 4 <= m.x + 8 <= b.right - 4
                        for b in o.blocks
                    )
                ):
                    # A missed edge is not itself a fall hazard when a visible
                    # lower floor supports the entire body. Keep the diagnostic
                    # without forbidding a climb solely for a marginal top touch.
                    lower_floor = any(
                        b.top < m.y - 1 and b.left <= m.x and m.x + 16 <= b.right for b in o.blocks
                    )
                    control_risk["uncertain_landing"] |= not lower_floor
                    unknowns.add("marginal_landing")
                if (
                    o.known_x is None
                    or estimate["x_range"][0] < o.known_x[0]
                    or estimate["x_range"][1] + 16 > o.known_x[1]
                ):
                    unknowns.add("unobserved_terrain")
                    if frame <= control_risk["through_frame"]:
                        control_risk["unobserved_terrain"] = True
                if m.y < 32 and not m.grounded:
                    possible_fall = True
                    if frame <= control_risk["through_frame"] or (
                        flight_committed and o.known_x and o.known_x[0] <= m.x + 8 < o.known_x[1]
                    ):
                        control_risk["possible_fall"] = True
                support_below = any(
                    b.top <= m.y + 1 and b.left <= m.x + 8 < b.right for b in o.blocks
                )
                if (
                    not o.swimming
                    and not m.grounded
                    and m.vy < 0
                    and not support_below
                    and (frame <= control_risk["through_frame"] or flight_committed)
                    and o.known_x is not None
                    and o.known_x[0] <= m.x + 8 < o.known_x[1]
                ):
                    # A descending body over unsupported space is already a
                    # warning; waiting for y<32 is too late to preserve a choice.
                    unsupported_descent = True
                if m.grounded or support_below:
                    unsupported_descent = False
                if "possible_landing" in emitted:
                    # The unavoidable flight is over. Later hypothetical steps
                    # are replannable; do not hard-filter on an optional departure.
                    flight_committed = False
                overlap_area = 0.0
                for actor in active_actors:
                    if actor.identity in committed.removed_actors:
                        continue
                    if actor.identity in conditional_shells:
                        # Contact may have launched this shell. Its old static
                        # path no longer describes a later collision; retain
                        # uncertainty and use the separate kick/return advisory.
                        if frame <= control_risk["through_frame"]:
                            control_risk["uncertain_shell_kick"] = True
                        continue
                    ex, ey, known = actor_paths[actor.identity][frame]
                    if step < cycle and actor.kind not in ("springboard", "flagpole"):
                        overlap_area += max(0, min(m.x + 16, ex + 16) - max(m.x, ex)) * max(
                            0, min(m.y + o.height, ey + 16) - max(m.y, ey)
                        )
                    spread = 1 + frame * (0.35 if known else 2)
                    if not known:
                        unknowns.add("enemy_motion_unknown")
                    if (
                        estimate["x_range"][0] < ex + 16 + spread
                        and estimate["x_range"][1] + 16 > ex - spread
                        and estimate["y_range"][0] < ey + 16 + spread
                        and estimate["y_range"][1] + o.height > ey - spread
                    ):
                        if actor.kind in ("springboard", "flagpole"):
                            unknowns.add("unmodelled_interaction")
                        else:
                            if possible_ground_kick(m, actor, ex, ey):
                                unknowns.add("possible_shell_kick")
                                conditional_shells.add(actor.identity)
                                if frame <= control_risk["through_frame"]:
                                    control_risk["uncertain_shell_kick"] = True
                                hazards.setdefault(
                                    actor.identity,
                                    {
                                        "kind": actor.kind,
                                        "id": actor.identity,
                                        "possible_contact_frame": frame,
                                        "interaction": "possible_shell_kick",
                                    },
                                )
                                continue
                            if frame <= control_risk["through_frame"]:
                                control_risk["possible_contact"] = True
                                if (
                                    m.x < ex + 16
                                    and m.x + 16 > ex
                                    and m.y < ey + 16
                                    and m.y + o.height > ey
                                ):
                                    control_risk["nominal_contact"] = True
                            hazards.setdefault(
                                actor.identity,
                                {
                                    "kind": actor.kind,
                                    "id": actor.identity,
                                    "possible_contact_frame": frame,
                                },
                            )
                            unknowns.add("collision_response_unknown")
                if step < cycle:
                    contact_exposure["frames"] += int(overlap_area > 0)
                    contact_exposure["overlap_area_sum"] += overlap_area
                    contact_exposure["end_overlap_area"] = overlap_area
                if step + 1 == cycle:
                    first_cycle = estimate
                    first_cycle_risk = {
                        **control_risk,
                        "through_frame": frame,
                        # Being above a gap during descent is uncertainty, not
                        # a failed landing within this cycle. Keep it explicit;
                        # the resolved flight still owns possible_fall below.
                        "unsupported_descent": unsupported_descent,
                    }
                if (step + 1) % 4 == 0 or step + 1 in (cycle, horizon):
                    trajectory.append(estimate)
            control_risk["possible_fall"] |= unsupported_descent
            risk = (
                "possible_contact"
                if hazards
                else "unknown"
                if "unobserved_terrain" in unknowns
                else "possible_fall"
                if possible_fall
                else "no_contact_predicted"
            )
            result["action_forecasts"].append(
                {
                    "action": action.value,
                    "first_cycle": first_cycle,
                    "first_landing": first_landing,
                    "end": estimate,
                    "risk": risk,
                    "confidence": "low"
                    if result["calibration"]["samples"] < 16 or len(unknowns) > 2
                    else "estimated",
                    "unknowns": sorted(unknowns),
                    "hazards": list(hazards.values()),
                    "events": sorted(events),
                    "trajectory": trajectory,
                    "control_risk": control_risk,
                    "first_cycle_risk": first_cycle_risk,
                    "contact_exposure": {k: round(v, 2) for k, v in contact_exposure.items()},
                }
            )
        followup = ()
        if self.selected_plan and self.history and self.history[-1] == o:
            plan = self.selected_plan
            elapsed = o.frame + delay - plan["apply_at_frame"]
            if (
                o.frame >= plan["apply_at_frame"]
                and cycle == plan["cycle"]
                and elapsed >= cycle
                and elapsed % cycle == 0
            ):
                index = elapsed // cycle
                scheduled_index = (o.frame - plan["apply_at_frame"]) // cycle
                if not delay or (
                    scheduled_index < len(plan["actions"])
                    and scheduled_action == plan["actions"][scheduled_index]
                ):
                    followup = plan["actions"][index:]
                    if not all(a in actions for a in followup):
                        followup = ()
        plans, expansions = continuations(
            motion,
            replace(o, actors=active_actors),
            actions,
            cycle,
            delay,
            plan_horizon,
            parameters,
            actor_paths,
            _advance,
            runup=runup,
            navigation_goal=navigation_goal,
            initial_warnings=committed_warnings,
            uncertain_takeoff=uncertain_takeoff,
            removed_actors=committed.removed_actors,
            landing_horizon=min(MAX_HORIZON, plan_horizon + 2 * cycle),
            followup=followup,
        )
        if followup:
            result["plan_followup"] = {
                "action": followup[0].value,
                "revalidated": True,
                **plans[followup[0].value]["followup"],
                "assumption": "Remainder of the selected action's earlier plan, rechecked "
                "from current observations. Advisory only; new risks override it.",
            }
        for branch in result["action_forecasts"]:
            branch["continuation"] = plans[branch["action"]]
        result["continuation_search"] = {
            "method": "bounded_observation_model_beam",
            "expanded_cycles": expansions,
            "runup_considered": runup,
            "assumption": "Conditional sequences; only the first action is eligible for execution. "
            "Replan from real observations. No emulator trials or learned future labels.",
        }
        result["compute_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
