"""Collision-map geometry and explicitly approximate landing projections."""

from collections.abc import Sequence
from math import ceil
from typing import Any


def jump_reach_reference(
    surfaces: Sequence[dict[str, Any]],
    *,
    x: float,
    y: int,
    speed: float,
    horizon: int,
    gap_start: int,
    gap_end: int | None,
) -> dict[str, Any] | None:
    """Conservative planning reference, not a collision or candidate-action simulator."""
    targets = [s for s in surfaces if s["start_x"] == gap_end]
    if (gap_end is not None and not targets) or speed <= 0:
        return None
    # A high suspended block must not hide an available lower landing.
    target = min(targets, key=lambda s: s["standing_y"]) if targets else None
    height, velocity = float(y), 5.0
    flight_frames = None
    # Approximate fully held small-Mario arc, calibrated against logged jumps.
    # Air acceleration and walls/ceilings are deliberately not treated as certain.
    for frame in range(1, 81) if target is not None else ():
        previous = height
        height += velocity
        if velocity < 0 and height <= target["standing_y"] <= previous:
            flight_frames = frame
            break
        velocity = max(-5.0, velocity - (0.18 if velocity > 0 else 0.5625))
    required = target["safe_center_min_x"] - (x + 8) if target is not None else None
    reach = speed * flight_frames if flight_frames is not None else None
    end_center = x + 8 + speed * horizon
    # A continuous same/lower walkway is required; a nearby raised block is not
    # walkable just because its top is a visible surface.
    cursor = x + 8
    for s in sorted(surfaces, key=lambda s: s["start_x"]):
        if y - 64 <= s["standing_y"] <= y and s["start_x"] <= cursor < s["end_x"]:
            cursor = max(cursor, s["end_x"])
    blocked = any(
        s["standing_y"] > y
        and s["standing_y"] <= y + 32
        and s["start_x"] < end_center
        and s["end_x"] > x + 8
        for s in surfaces
    )
    supported = cursor >= end_center and end_center <= gap_start - 8 and not blocked
    # Walking down a step needs enough lower floor to land AND reach the next
    # decision boundary. A continuous horizontal interval alone is insufficient.
    for upper in surfaces:
        edge = upper["end_x"]
        if not (x + 8 <= edge <= end_center and upper["standing_y"] == y):
            continue
        lower = next(
            (
                s
                for s in sorted(surfaces, key=lambda s: -s["standing_y"])
                if s["start_x"] <= edge < s["end_x"] and y - 64 <= s["standing_y"] < y
            ),
            None,
        )
        if lower is None:
            supported = False
            continue
        drop, falling_speed, fall_frames = 0.0, 0.0, 0
        while drop < y - lower["standing_y"]:
            falling_speed = min(5.0, falling_speed + 0.5625)
            drop += falling_speed
            fall_frames += 1
        edge_frames = ceil((edge - (x + 8)) / speed)
        next_boundary = ceil((edge_frames + fall_frames) / horizon) * horizon
        grounded_center = edge + speed * (next_boundary - edge_frames)
        if grounded_center > lower["safe_center_max_x"]:
            supported = False
    reachable = reach >= required if reach is not None and required is not None else None
    return {
        "assumption": "Approximate fully held jump: initial up speed 5, ascent gravity 0.18, "
        "descent gravity 0.5625 and fall cap 5 px/frame. Constant observed horizontal speed; "
        "ignores acceleration, walls, ceilings and enemies. Reachability is NOT a guarantee.",
        "takeoff_x_estimate": round(x, 2),
        "takeoff_y_estimate": y,
        "horizontal_speed_pixels_per_frame": speed,
        "landing_surface": target,
        "required_distance_pixels": round(required, 2) if required is not None else None,
        "landing_height_difference_pixels": target["standing_y"] - y
        if target is not None
        else None,
        "current_speed_jump": {
            "estimated_flight_frames": flight_frames,
            "estimated_reach_pixels": round(reach, 2) if reach is not None else None,
            "can_reach": reachable,
        },
        "approach_supported": supported,
        "approach_before_jumping": not reachable and supported,
    }


def is_support_tile(tile: int) -> bool:
    # SMB CheckForCoinMTiles, ChkInvisibleMTiles and CheckForClimbMTiles:
    # https://gist.github.com/1wErt3r/4048722
    climb_thresholds = (0x24, 0x6D, 0x8A, 0xC6)
    return tile not in (0, 0xC2, 0xC3, 0x5F, 0x60, 0xC5) and tile < climb_thresholds[tile >> 6]


def navigation(
    rows: Sequence[str], *, x: int, y: int, full_height: bool = False, screen_y: int = 0
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "geometry_available": False,
        "obstacle_ahead": False,
        "obstacle_distance_tiles": None,
        "obstacle_height_tiles": 0,
        "gap_ahead": False,
        "gap_distance_tiles": None,
        "gap_width_tiles_visible": 0,
        "gap_end_visible": False,
        "gap_start_x": None,
        "gap_end_x": None,
        "obstacle_start_x": None,
        "clear_forward_tiles": 0,
        "landing_surfaces": [],
        "summary": "Support is unknown; do not assume the forward path is clear.",
    }
    if not rows:
        return result
    if full_height:
        mario_col = 2
        # SMB's reported screen position is 32 pixels above the support plane.
        support_start = max(0, screen_y // 16)
        support_columns = {(x + offset) // 16 - (x // 16 - 2) for offset in (3, 12)}
        origin_x = (x // 16 - 2) * 16
        for row in range(len(rows)):
            column = 0
            while column < len(rows[row]):
                exposed = rows[row][column] == "#" and (row == 0 or rows[row - 1][column] != "#")
                if not exposed:
                    column += 1
                    continue
                start = column
                while (
                    column < len(rows[row])
                    and rows[row][column] == "#"
                    and (row == 0 or rows[row - 1][column] != "#")
                ):
                    column += 1
                left, right = origin_x + start * 16, origin_x + column * 16
                result["landing_surfaces"].append(
                    {
                        "start_x": left,
                        "end_x": right,
                        "safe_center_min_x": left + 8,
                        "safe_center_max_x": right - 8,
                        "standing_y": 255 - row * 16,
                        "height_above_player_pixels": 255 - row * 16 - y,
                    }
                )
    else:
        position = next(((i, row.index("M")) for i, row in enumerate(rows) if "M" in row), None)
        if position is None:
            return result
        mario_row, mario_col = position
        support_start = mario_row + 1
        support_columns = {mario_col}
        origin_x = (x // 16 - mario_col) * 16
    ground_row = next(
        (
            r
            for r in range(support_start, len(rows))
            if any(rows[r][column] == "#" for column in support_columns)
        ),
        None,
    )
    if ground_row is None:
        return result
    result["geometry_available"] = True
    gap_finished = False
    for column in range(mario_col + 1, len(rows[0])):
        distance = column - mario_col
        height = 0
        row = ground_row - 1
        while row >= 0 and rows[row][column] == "#":
            height += 1
            row -= 1
        if height and result["obstacle_distance_tiles"] is None:
            result.update(
                obstacle_distance_tiles=distance,
                obstacle_height_tiles=height,
                obstacle_start_x=origin_x + column * 16,
            )
        # A lower floor is a possible landing, not a bottomless pit.
        supported = any(rows[r][column] == "#" for r in range(ground_row, len(rows)))
        if not supported and result["gap_distance_tiles"] is None:
            result.update(gap_distance_tiles=distance, gap_start_x=origin_x + column * 16)
        if result["gap_distance_tiles"] is not None and not gap_finished:
            if supported:
                result.update(gap_end_visible=True, gap_end_x=origin_x + column * 16)
                gap_finished = True
            else:
                result["gap_width_tiles_visible"] += 1
        if result["obstacle_distance_tiles"] is None and result["gap_distance_tiles"] is None:
            result["clear_forward_tiles"] = distance
    result["gap_ahead"] = (
        result["gap_distance_tiles"] is not None and result["gap_distance_tiles"] <= 3
    )
    result["obstacle_ahead"] = (
        result["obstacle_distance_tiles"] is not None and result["obstacle_distance_tiles"] <= 3
    )
    result["summary"] = (
        f"Obstacle distance={result['obstacle_distance_tiles']}; "
        f"gap distance={result['gap_distance_tiles']}; "
        f"visible contiguous gap width={result['gap_width_tiles_visible']}."
    )
    return result


def landing_projection(
    surfaces: Sequence[dict[str, Any]], *, x: int, y: int, dx: float, dy: int, grounded: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "confidence": "unknown",
        "assumption": "Constant horizontal velocity; descending only. "
        "Approximate gravity 0.5 px/frame², fall speed capped at 5 px/frame. "
        "Ignores walls, enemies and moving platforms; never a safety guarantee.",
        "frames": None,
        "x": None,
        "standing_y": None,
        "within_safe_surface": None,
    }
    if grounded or dy >= 0 or not surfaces:
        return result
    height, velocity = float(y), float(dy)
    for frame in range(1, 61):
        velocity = max(-5.0, velocity - 0.5)
        next_height = height + velocity
        projected_x = x + dx * frame
        center_x = projected_x + 8
        for surface in sorted(surfaces, key=lambda s: s["standing_y"], reverse=True):
            if (
                next_height <= surface["standing_y"] <= height
                and surface["start_x"] <= center_x < surface["end_x"]
            ):
                result.update(
                    confidence="approximate",
                    frames=frame,
                    x=projected_x,
                    standing_y=surface["standing_y"],
                    within_safe_surface=(
                        surface["safe_center_min_x"] <= center_x <= surface["safe_center_max_x"]
                    ),
                )
                return result
        height = next_height
        if height < 0:
            break
    result.update(confidence="approximate", within_safe_surface=False)
    return result
