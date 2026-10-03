"""Geometric hints from parsed terrain; no controller selection."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import MarioSnapshot


def enemy_crossing_geometry(observation, motion, enemy_motion) -> dict[str, Any] | None:
    """Recognize room to prepare a jump over observed approaching walkers."""
    o, m = observation, motion
    if o.swimming or not o.motion_reliable or not m.grounded or o.known_x is None:
        return None
    actors = sorted(
        (
            a
            for a in o.actors
            if a.kind in {"goomba", "green_koopa", "red_koopa"}
            and m.x + 16 <= a.x <= m.x + 128
            and abs(a.y + 8 - m.y) <= 16
            and enemy_motion[a.identity][2]
            and enemy_motion[a.identity][0] < -0.1
        ),
        key=lambda a: a.x,
    )
    if not actors:
        return None
    left, right = min(o.x, m.x), actors[-1].x + 32
    if not (o.known_x[0] <= left and right <= o.known_x[1]):
        return None
    floor = next(
        (b for b in o.blocks if abs(b.top - m.y) <= 1 and b.left <= left and b.right >= right),
        None,
    )
    # This minimum clearance only enables searching; each trajectory must still
    # resolve ceiling contacts, every actor and its eventual landing itself.
    obstructing = [b for b in o.blocks if b.top > m.y + 1 and b.bottom < m.y + o.height + 64]
    if floor is None or any(b.right > left and b.left < right for b in obstructing):
        return None
    rear_edge = max([floor.left, o.known_x[0], *(b.right for b in obstructing if b.right <= left)])
    return {
        "kind": "approaching_enemies_on_open_ground",
        "actor_ids": [a.identity for a in actors],
        "observed_group_x": [actors[0].x, actors[-1].x + 16],
        "open_ground_x": [rear_edge, right],
        "standing_y": floor.top,
        "intent": "Prepare a directional jump across the approaching group while open ground "
        "and headroom remain. Compare run-up and held-jump sequences; avoid repeatedly waiting "
        "or retreating toward lower clearance. Geometry does not establish a safe crossing.",
    }


def underpass_goal(observation, motion) -> dict[str, Any] | None:
    """Infer a local descent target when a visible roof seals the forward wall."""
    o, m = observation, motion
    if o.swimming or not o.motion_reliable or o.known_x is None:
        return None
    for platform in o.blocks:
        target_x = platform.left - 16
        if not (
            o.known_x[0] <= target_x
            and platform.left - 32 <= m.x < platform.right
            and platform.bottom < m.y + o.height
            and platform.top <= m.y + o.height
        ):
            continue
        for wall in o.blocks:
            if not (m.x + 2 <= wall.left <= m.x + 128 and wall.bottom <= platform.top < wall.top):
                continue
            for ceiling in o.blocks:
                if not (
                    ceiling.left <= platform.left
                    and ceiling.right >= wall.left + 16
                    and ceiling.bottom >= platform.top + o.height
                    and m.y + o.height <= ceiling.bottom + 1
                ):
                    continue
                height = platform.top
                for block in sorted(o.blocks, key=lambda b: b.bottom):
                    if block.left <= wall.left + 8 < block.right and block.bottom <= height:
                        height = max(height, block.top)
                if height < ceiling.bottom:
                    continue
                for floor in o.blocks:
                    if not (
                        floor.left <= target_x
                        and floor.right >= wall.right
                        and 32 <= floor.top < platform.bottom - o.height
                    ):
                        continue
                    if any(
                        b.left < target_x + 14
                        and b.right > target_x + 2
                        and b.bottom < platform.top + o.height
                        and b.top > floor.top
                        for b in o.blocks
                    ):
                        continue
                    if any(
                        b.left < wall.right
                        and b.right > target_x
                        and b.bottom < floor.top + o.height
                        and b.top > floor.top
                        for b in o.blocks
                    ):
                        continue
                    return {
                        "kind": "descend_below_blocked_platform",
                        "x": target_x,
                        "y": floor.top,
                        "clear_below_y": platform.bottom,
                        "evidence": "visible_wall_reaches_overhead_ceiling_with_lower_route",
                        "intent": "Retreat past the platform edge and descend to the visible lower floor; "
                        "then reassess forward travel. Geometry is not verified reachability or safety.",
                    }
    return None


def staging_geometry(snapshot: "MarioSnapshot") -> dict[str, Any] | None:
    """Find a tall wall with lower stepping surfaces on its approach side.

    Geometry proposes a goal only; it does not establish reachability or safety.
    Requiring a wall's left edge ahead excludes the overhead ceiling surface.
    """
    if snapshot.swimming or snapshot.side_exit_pipe or snapshot.player_state != 8:
        return None
    surfaces = snapshot.navigation_features()["landing_surfaces"]
    for wall in sorted(surfaces, key=lambda t: t["start_x"]):
        if not snapshot.x - 16 <= wall["start_x"] <= snapshot.x + 128:
            continue
        approach = [
            t
            for t in surfaces
            if wall["start_x"] - 128 <= t["start_x"] < t["end_x"] <= wall["start_x"]
        ]
        if not approach:
            continue
        floor = min(t["standing_y"] for t in approach)
        if wall["standing_y"] - floor < 96:
            continue
        steps = [
            t
            for t in approach
            if floor + 32 <= t["standing_y"] < wall["standing_y"]
            and wall["standing_y"] - t["standing_y"] <= 80
            and t["end_x"] >= snapshot.x - 96
        ]
        if not steps:
            continue
        return {
            "obstacle_start_x": wall["start_x"],
            "obstacle_end_x": wall["end_x"],
            "obstacle_standing_y": wall["standing_y"],
            "approach_floor_y": floor,
            "staging_surfaces": [
                {k: t[k] for k in ("start_x", "end_x", "standing_y")} for t in steps
            ],
            "intent": "Land on an intermediate surface before launching over the tall obstacle. "
            "Do not trade that support for a lower dead end. Geometry alone is not safety.",
        }
    return None


def pipe_entry_geometry(snapshot: "MarioSnapshot") -> dict[str, Any] | None:
    if snapshot.swimming or snapshot.player_state != 8 or not snapshot.side_exit_pipe:
        return None
    return {"kind": "side_exit_pipe", **snapshot.side_exit_pipe}
