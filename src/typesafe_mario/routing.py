"""Geometric staging goals and selection of native-verified obstacle routes."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .actions import Action

if TYPE_CHECKING:
    from .policy import Decision
    from .state import MarioSnapshot


def staging_geometry(snapshot: "MarioSnapshot") -> dict[str, Any] | None:
    """Find a tall wall with lower stepping surfaces on its approach side.

    Geometry proposes a goal only; native rollouts establish reachability and safety.
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


def select_staging_route(
    state: dict[str, Any], decision: "Decision", actions: tuple[Action, ...]
) -> "Decision | None":
    """Revalidate the whole braking/launch route each cycle, before random recovery."""
    routes = []
    for forecast in state.get("prediction", {}).get("action_forecasts", []):
        action = Action(forecast["action"])
        if action not in actions:
            continue
        for name, branch in forecast["continuations"].items():
            route = branch.get("route", {})
            if (
                (route.get("obstacle_cleared") or route.get("pipe_entered"))
                and branch.get("risk") == "safe"
                and branch.get("continuation_complete", False)
            ):
                routes.append((route["clear_frame"], action, name, route))
    if not routes:
        return None
    _, action, name, route = min(
        routes, key=lambda item: (item[0], -decision.probabilities.get(item[1].value, 0))
    )
    return replace(
        decision,
        action=action,
        confidence=decision.probabilities.get(action.value, 0),
        selection={
            "mode": "pipe_entry_route" if route.get("pipe_entered") else "staging_route",
            "reason": "native_verified_route",
            "proposed_action": decision.action.value,
            "proposed_confidence": decision.confidence,
            "selected_action": action.value,
            "continuation": name,
            "route": route,
        },
    )
