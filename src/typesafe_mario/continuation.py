"""Bounded action-sequence search inside the approximate observation model.

No game/environment interface is accepted here. Search estimates are never fed
back into dynamics learning. A sequence is a conditional suggestion, not a route
that the controller commits to execute.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from .actions import JUMP_ACTIONS, Action, first_frame_action
from .landing import landing_window
from .shells import possible_ground_kick

BEAM_WIDTH = 3
MAX_SEARCH_FRAMES = 96
STOMPABLE = frozenset({"goomba", "green_koopa", "red_koopa"})


@dataclass
class Node:
    motion: Any
    actions: tuple = ()
    elapsed: int = 0
    penalty: float = 0
    warnings: frozenset = field(default_factory=frozenset)
    failure: str | None = None
    stopped: bool = False
    supported_frames: int = 0
    progress_sum: float = 0
    goal_progress_sum: float = 0
    removed_actors: frozenset = field(default_factory=frozenset)
    interactions: tuple = ()
    landed: bool = False
    landings: tuple = ()


def actor_contacts(previous, m, n, warnings, o, actor_paths, frame, dynamics):
    """Apply conditional contact responses to a local forecast node only."""
    possible_bounce = None
    for actor in o.actors:
        if actor.kind == "flagpole" or actor.identity in n.removed_actors:
            continue
        ex, ey, known = actor_paths[actor.identity][frame]
        horizontal = m.x + 16 > ex and m.x < ex + 16
        vertical = m.y < ey + 16 and m.y + o.height > ey
        if horizontal and vertical:
            if possible_ground_kick(m, actor, ex, ey):
                warnings.update(("possible_shell_kick", "unsupported_interaction"))
                n.interactions += (
                    {
                        "actor_id": actor.identity,
                        "frame": frame,
                        "inference": "conditional_shell_kick",
                    },
                )
                n.stopped = True
            elif actor.kind == "springboard":
                warnings.add("unsupported_interaction")
                n.penalty += 24
                n.stopped = True
            elif actor.kind in STOMPABLE and m.vy < 0 and previous.y >= ey + 14:
                warnings.add("possible_stomp")
                n.penalty += 24
                if actor.kind == "goomba" and possible_bounce is None:
                    possible_bounce = (actor.identity, ey)
                else:
                    if actor.kind in {"green_koopa", "red_koopa"}:
                        warnings.add("possible_stationary_shell")
                        n.interactions += (
                            {
                                "actor_id": actor.identity,
                                "frame": frame,
                                "inference": "conditional_koopa_stomp",
                            },
                        )
                    # Reobserve after a hypothetical stomp before
                    # assuming a kickable shell; retain other hazards.
                    n.stopped = True
            else:
                n.failure, n.stopped = "nominal_contact", True
        elif abs(m.x - ex) < 24 + (0 if known else min(32, 2 * frame)) and abs(
            m.y + o.height / 2 - (ey + 8)
        ) < 24 + (0 if known else min(32, 2 * frame)):
            warnings.add("close_contact")
            n.penalty += 0.6 if known else 1.2
    if possible_bounce is not None and not n.stopped:
        identity, ey = possible_bounce
        speed = dynamics.value("stomp_speed")
        m = replace(m, y=ey + 16, vy=speed, grounded=False, bouncing=True)
        n.removed_actors = n.removed_actors | {identity}
        n.interactions += (
            {
                "actor_id": identity,
                "frame": frame,
                "inference": "conditional_goomba_bounce",
                "speed": round(speed, 2),
            },
        )
        warnings.add("uncertain_bounce")
    return m


def continuations(
    motion,
    o,
    actions,
    cycle,
    delay,
    horizon,
    dynamics,
    actor_paths,
    advance,
    *,
    runup=False,
    navigation_goal=None,
    initial_warnings=(),
    uncertain_takeoff=False,
    removed_actors=frozenset(),
    landing_horizon=None,
    followup=(),
):
    """Retain a small, spatially diverse beam for each possible first action."""
    # Controller cadence changes opportunities to replan, not the amount of
    # physical future needed to finish a jump. The caller supplies that budget.
    horizon = min(horizon, MAX_SEARCH_FRAMES)
    if landing_horizon is not None:
        landing_horizon = min(landing_horizon, horizon + 2 * cycle)
    expansions = 0

    def distance_to_goal(m):
        if m.grounded and abs(m.y - navigation_goal["y"]) < 1:
            # Reaching the lower floor completes the descent; do not reward
            # delaying it merely to finish closer to an arbitrary x waypoint.
            return 0
        return abs(m.x - navigation_goal["x"]) + abs(m.y - navigation_goal["y"])

    def goal_progress(m):
        return distance_to_goal(motion) - distance_to_goal(m)

    def terminal_warnings(n):
        warnings = set(n.warnings)
        m = n.motion
        if (n.elapsed >= horizon or n.stopped) and not o.swimming and not m.grounded:
            # A floor somewhere below does not establish landing time, enemy
            # clearance or a chance to jump again. Otherwise a late takeoff
            # can push its risky landing beyond the search horizon each time.
            warnings.add("unresolved_flight")
        return warnings

    def rank(n):
        # Avoid a predicted failure before optimizing progress. Unknown terrain
        # and actor interactions are penalized, never certified as clear space.
        # Progress along the path rewards acting now. An endpoint-only score
        # can perpetually postpone a jump to make its landing match the horizon.
        duration = max(horizon, n.elapsed)
        utility = n.motion.x - motion.x + 0.5 * n.progress_sum / duration - n.penalty
        if navigation_goal:
            utility = goal_progress(n.motion) + 0.5 * n.goal_progress_sum / duration - n.penalty
        if not n.motion.grounded and not o.swimming:
            support = any(
                b.top <= n.motion.y + 1 and b.left <= n.motion.x + 8 < b.right for b in o.blocks
            )
            if not support:
                utility -= 50
        return (
            n.failure is None,
            "unobserved_terrain" not in n.warnings,
            n.elapsed if n.failure else 0,
            not terminal_warnings(n),
            utility,
        )

    def extend(node, action):
        nonlocal expansions
        expansions += 1
        n = replace(node, actions=node.actions + (action,))
        if n.stopped:
            return n
        m = n.motion
        warnings = set(n.warnings)
        if action in JUMP_ACTIONS and m.grounded:
            n.penalty += 2  # Prefer a simple walk when it achieves the same route.
        for i in range(min(cycle, horizon - n.elapsed)):
            previous = m
            actual = (
                first_frame_action(
                    action,
                    grounded=m.grounded,
                    previous_action=m.previous_action,
                    swimming=o.swimming,
                )
                if i == 0
                else action
            )
            m, events = advance(m, actual, o, dynamics)
            n.landed |= "possible_landing" in events
            if "uncertain_wall_contact" in events:
                warnings.add("uncertain_wall_contact")
            n.progress_sum += m.x - motion.x
            if navigation_goal:
                n.goal_progress_sum += goal_progress(m)
            n.elapsed += 1
            frame = delay + n.elapsed
            if "possible_landing" in events and not o.swimming:
                # A nominal landing of an already committed flight near a pit
                # is not a robust refuge. Use
                # the same horizontal error envelope as the repeated-input
                # forecast, including the already committed input delay.
                # Visible lower support can catch a stair/platform edge miss.
                spread = 1 + 0.2 * frame + 0.01 * frame * frame
                # Later optional flights are replanned after fresh observations;
                # applying an ever-growing envelope to them would forbid useful
                # run-ups and climbs simply for being far in the search horizon.
                if (
                    not motion.grounded
                    and not n.landings
                    and not any(
                        32 <= b.top <= m.y + 1
                        and b.left <= m.x + 8 - spread
                        and m.x + 8 + spread < b.right
                        for b in o.blocks
                    )
                ):
                    warnings.add("uncertain_landing")
                landing = landing_window(
                    m, o, actor_paths, frame, n.elapsed, cycle, dynamics, n.removed_actors
                )
                n.landings += (landing,)
                if any(
                    e["contact_before_escape"] or not e["reaction_window_complete"]
                    for e in landing["enemies"]
                ):
                    warnings.add("landing_contact_window")
            if o.known_x is None or m.x < o.known_x[0] or m.x + 16 > o.known_x[1]:
                warnings.add("unobserved_terrain")
                n.penalty += 40
                n.stopped = True
            elif m.y < 32 and not m.grounded and not o.swimming:
                n.failure, n.stopped = "possible_fall", True
            if ("possible_landing" in events or "marginal_support" in events) and not any(
                abs(b.top - m.y) < 0.1 and b.left + 4 <= m.x + 8 <= b.right - 4 for b in o.blocks
            ):
                warnings.add("marginal_landing")
                n.penalty += 12
            centered_support = m.grounded and any(
                abs(b.top - m.y) < 0.1 and b.left + 4 <= m.x + 8 <= b.right - 4 for b in o.blocks
            )
            n.supported_frames = n.supported_frames + 1 if centered_support else 0
            if n.supported_frames >= 2:
                warnings.discard("marginal_landing")
            m = actor_contacts(previous, m, n, warnings, o, actor_paths, frame, dynamics)
            if n.stopped:
                break
        n.motion, n.warnings = m, frozenset(warnings)
        return n

    result = {}

    def settle(n):
        while n.elapsed < horizon and not n.stopped:
            n = extend(n, Action.NOOP)
        return n

    for first in actions:
        warnings = frozenset(initial_warnings) | (
            {"uncertain_bounce"} if motion.bouncing else set()
        )
        if uncertain_takeoff and first in JUMP_ACTIONS:
            warnings |= {"uncertain_takeoff"}
        beam = [
            extend(Node(replace(motion), warnings=warnings, removed_actors=removed_actors), first)
        ]
        # Preserve simple refuges outside the progress beam. Greedy pruning of
        # waiting/braking states is not evidence that the first input must fail.
        refuges = []
        followup_node = None
        if followup and followup[0] == first:
            n = beam[0]
            for action in followup[1:]:
                if n.elapsed >= horizon or n.stopped:
                    break
                n = extend(n, action)
            followup_node = settle(n) if Action.NOOP in actions else n
            refuges.append(followup_node)
        for refuge in dict.fromkeys((first, Action.NOOP, Action.LEFT)):
            if refuge not in actions:
                continue
            n = beam[0]
            while n.elapsed < horizon and not n.stopped:
                n = extend(n, refuge)
            refuges.append(n)
        if navigation_goal and {Action.LEFT, Action.RIGHT, Action.NOOP} <= set(actions):
            # Keep a descent attempt across the brief uncertain edge departure;
            # a beam that favors warning-free nodes can otherwise prune it
            # before the visible lower floor is reached. It remains model-only.
            n = beam[0]
            while n.elapsed < horizon and not n.stopped:
                dx = n.motion.x - navigation_goal["x"]
                a = Action.LEFT if dx > 2 else Action.RIGHT if dx < -2 else Action.NOOP
                n = extend(n, a)
            refuges.append(n)
        if runup and {Action.LEFT, Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP} <= set(actions):
            # Generic sequences cross the temporary loss of progress required
            # to accelerate before a high wall. Every step uses visible geometry
            # and fitted dynamics; there are no level-specific coordinates.
            for retreat in range(3):
                for acceleration in range(1, 4):
                    n = beam[0]
                    sequence = [Action.LEFT] * retreat + [Action.RIGHT_RUN] * acceleration
                    for a in sequence + [Action.RIGHT_RUN_JUMP] * ((horizon + cycle - 1) // cycle):
                        if n.elapsed >= horizon or n.stopped:
                            break
                        n = extend(n, a)
                        if n.landed and n.motion.grounded and Action.NOOP in actions:
                            refuges.append(settle(n))
                    refuges.append(n)
        for _ in range(1, (horizon + cycle - 1) // cycle):
            candidates = [
                child
                for n in beam
                for child in ([n] if n.stopped else [extend(n, a) for a in actions])
            ]
            landings = [n for n in candidates if n.landed and n.motion.grounded and not n.stopped]
            if landings and Action.NOOP in actions:
                # Keep a completed landing as well as the beam's forward
                # flight. Repeating A can otherwise launch again at the last
                # boundary and hide the useful landing behind the horizon.
                refuges.append(settle(max(landings, key=rank)))
            beam, seen = [], set()
            for n in sorted(candidates, key=rank, reverse=True):
                m = n.motion
                key = (
                    round((m.x - motion.x) / 4),
                    round((m.y - motion.y) / 4),
                    round(m.vx),
                    round(m.vy),
                    m.previous_action in JUMP_ACTIONS,
                    m.fast_jump,
                    m.bouncing,
                    m.jump_pending,
                    n.removed_actors,
                    n.failure,
                    n.stopped,
                )
                if key not in seen:
                    seen.add(key)
                    beam.append(n)
                    if len(beam) == BEAM_WIDTH:
                        break
        # Spend a small, bounded tail on finishing an existing flight, never
        # on starting another jump. This preserves useful high-wall run-ups
        # without calling an airborne endpoint a completed landing.
        endings = []
        if landing_horizon and Action.NOOP in actions:
            search_horizon = horizon
            horizon = landing_horizon
            for n in beam + refuges:
                if not n.stopped and not n.motion.grounded and n.elapsed >= search_horizon:
                    endings.append(settle(n))
            horizon = search_horizon
        best = max(beam + refuges + endings, key=rank)
        if runup and best.motion.x <= motion.x + 2:
            partial = [
                n
                for n in beam + refuges + endings
                if not n.failure
                and terminal_warnings(n) == {"unresolved_flight"}
                and n.motion.x > motion.x + 8
                and any(
                    b.top <= n.motion.y and b.left <= n.motion.x + 8 < b.right for b in o.blocks
                )
            ]
            if partial:
                # Show an unfinished climb instead of a stationary dead end,
                # but keep its warning: it is never a clear landing refuge.
                best = max(partial, key=rank)
        m = best.motion
        warnings = terminal_warnings(best)
        if not o.motion_reliable:
            warnings.add("unreliable_motion")
        result[first.value] = {
            "status": "predicted_failure"
            if best.failure
            else "uncertain"
            if warnings
            else "estimated_viable",
            "actions": [a.value for a in best.actions],
            "evaluated_frames": best.elapsed,
            "horizon_frames": max(horizon, best.elapsed),
            "failure": best.failure,
            "warnings": sorted(warnings),
            "interactions": list(best.interactions),
            "landings": list(best.landings),
            "progress": round(m.x - motion.x, 2),
            "mean_progress": round(best.progress_sum / max(1, best.elapsed), 2),
            "score": round(rank(best)[-1], 2),
            "end": {"x": round(m.x, 2), "y": round(m.y, 2), "grounded_estimate": m.grounded},
        }
        if followup_node is not None:
            result[first.value]["followup"] = {
                "actions": [a.value for a in followup_node.actions],
                "warnings": sorted(terminal_warnings(followup_node)),
                "failure": followup_node.failure,
                "evaluated_frames": followup_node.elapsed,
                "horizon_frames": horizon,
            }
    return result, expansions
