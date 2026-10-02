"""Action-conditioned forecasts. The provider, not this module, selects an action.

Native rollouts have privileged engine access, including future spawns. All frame
offsets are relative to the observation. Paths are conditional on explicit buttons;
surviving a finite path is not a guarantee about a different future action sequence.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from math import hypot
from time import perf_counter
from typing import Any

from .actions import ACTION_TO_INDEX, Action, first_frame_action
from .checkpoint import Checkpoint
from .routing import pipe_entry_geometry, staging_geometry
from .state import MarioSnapshot, MarioStateParser

# These pairs request identical buttons at every cycle, including A rearming.
# Reuse only within one candidate/application state and the same fixed horizon.
# Conditional waits and staging routes deliberately remain separate simulations.
_IDENTICAL_CONTINUATIONS = {
    (Action.RIGHT_RUN_JUMP, "run_jump"): "repeat",
    (Action.RIGHT_RUN_JUMP, "repeat_twice_then_jump"): "repeat",
    (Action.RIGHT, "walk"): "repeat",
    (Action.NOOP, "repeat_twice_then_jump"): "release_then_jump",
}


def _player(s: MarioSnapshot) -> dict[str, Any]:
    return {
        "x": s.x,
        "screen_y": s.screen_y,
        "height_y": s.y,
        "vx": s.horizontal_speed_exact if s.horizontal_speed_exact is not None else s.dx,
        "grounded": s.grounded,
        "swimming": s.swimming,
        "jump_phase": s.jump_phase,
        "powerup_status": s.status,
    }


def _sample(frame: int, s: MarioSnapshot, action: Action | None) -> dict[str, Any]:
    return {
        "frame": frame,
        "action": action.value if action else None,
        "player": _player(s),
        "enemies": [
            {
                "id": t.id,
                "kind": t.measurement.kind,
                "x": t.measurement.x,
                "y": t.measurement.y,
                "engine_state": t.measurement.engine_state,
                **({"plant": t.measurement.plant} if t.measurement.plant is not None else {}),
            }
            for t in s.enemy_tracks
            if t.observed
        ],
    }


def _step(
    env: Any, parser: MarioStateParser, action: Action, *, include_geometry: bool = False
) -> tuple[MarioSnapshot, bool]:
    _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
    s = parser.parse(
        info,
        env.unwrapped.ram,
        previous_action=action.value,
        previous_reward=float(reward),
        include_geometry=include_geometry,
    )
    return s, bool(terminated or truncated or s.dead or s.clear)


def _outcome(s: MarioSnapshot, ended: bool) -> str:
    if s.dead:
        return "death"
    if s.clear:
        return "stage_clear"
    return "terminated" if ended else "survived_horizon"


def _risk(s: MarioSnapshot, ended: bool, events: list[dict[str, Any]]) -> str:
    if s.dead or any(e["type"] == "powerup_change" and e["to"] == "small" for e in events):
        return "unsafe"
    return "unknown" if ended and not s.clear else "safe"


def _nearby_plants(s: MarioSnapshot):
    return [
        t
        for t in s.enemy_tracks
        if t.observed
        and t.measurement.plant is not None
        and 0 <= t.measurement.x - s.x <= 96
        and s.y < t.measurement.plant.get("pipe_top_y", 0) + 16
    ]


def _phase_events(previous: MarioSnapshot, current: MarioSnapshot, frame: int):
    old = {t.id: (t.measurement.plant or {}).get("phase") for t in previous.enemy_tracks}
    return [
        {"frame": frame, "type": "plant_phase", "id": t.id, "phase": t.measurement.plant["phase"]}
        for t in current.enemy_tracks
        if t.measurement.plant and old.get(t.id) != t.measurement.plant.get("phase")
    ]


def _rollout(
    env: Any,
    parser: MarioStateParser,
    s: MarioSnapshot,
    action: Action,
    continuation: str,
    *,
    delay: int,
    cycle: int,
    horizon: int,
    brake_cycles: int | None = None,
    route_goal: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encounters: dict[str, dict[str, Any]] = {}
    landings, events = [], []
    samples = [_sample(delay, s, None)]
    ended = False
    application_x = s.x
    cycle_end_player = None
    terminal_frame = None
    max_x, max_height = s.x, s.y
    first_cycle_max_x, first_cycle_max_height = s.x, s.y
    origin_level = (s.world, s.stage, s.area)
    wait_targets = {t.id for t in _nearby_plants(s)}
    wait_complete = continuation != "wait_for_clearance"
    # Do not label a fall safe merely because the death flag is one frame beyond
    # the nominal horizon. Finish the current flight, with a bounded extension.
    landing_resolved = False
    swimming_resolved = False
    staging_landed = False
    obstacle_mounted = False
    route_clear_frame = None
    pipe_route = route_goal is not None and route_goal.get("kind") == "side_exit_pipe"
    for offset in range(horizon + 48):
        requested = action
        if brake_cycles is not None:
            requested = (
                Action.LEFT
                if offset < brake_cycles * cycle
                else Action.RIGHT
                if pipe_route
                else Action.RIGHT_RUN_JUMP
            )
        elif offset >= cycle:
            requested = Action.RIGHT_RUN_JUMP
            if continuation == "repeat" or (
                continuation == "repeat_twice_then_jump" and offset < 2 * cycle
            ):
                requested = action
            elif continuation == "walk":
                requested = Action.RIGHT
            elif continuation == "release_then_jump" and offset < 2 * cycle:
                requested = Action.NOOP
            elif continuation == "hold_jump_then_run" and offset < 3 * cycle:
                requested = Action.JUMP
            elif continuation == "wait_for_clearance" and not wait_complete:
                remaining = [t for t in s.enemy_tracks if t.id in wait_targets]
                if (
                    offset % cycle == 0
                    and s.grounded
                    and all((t.measurement.plant or {}).get("phase") == "hidden" for t in remaining)
                ):
                    wait_complete = True
                    events.append({"frame": delay + offset + 1, "type": "wait_complete"})
                else:
                    requested = Action.NOOP
        actual = (
            first_frame_action(
                requested,
                grounded=s.grounded,
                previous_action=s.previous_action,
                swimming=s.swimming,
            )
            if offset % cycle == 0
            else requested
        )
        previous = s
        s, ended = _step(env, parser, actual)
        frame = delay + offset + 1
        if pipe_route and (s.world, s.stage, s.area) == origin_level:
            if route_clear_frame is None and s.player_state == 2:
                route_clear_frame = frame
        elif route_goal is not None and (s.world, s.stage, s.area) == origin_level:
            staging_landed |= s.grounded and any(
                t["start_x"] <= s.x + 8 <= t["end_x"] and s.y == t["standing_y"]
                for t in route_goal["staging_surfaces"]
            )
            obstacle_mounted |= (
                s.x >= route_goal["obstacle_start_x"] and s.y >= route_goal["obstacle_standing_y"]
            )
            if (
                route_clear_frame is None
                and obstacle_mounted
                and s.x >= route_goal["obstacle_end_x"] + 8
            ):
                route_clear_frame = frame
        if (s.world, s.stage, s.area) == origin_level:
            max_x, max_height = max(max_x, s.x), max(max_height, s.y)
            if offset < cycle:
                first_cycle_max_x = max(first_cycle_max_x, s.x)
                first_cycle_max_height = max(first_cycle_max_height, s.y)
        if (s.world, s.stage, s.area) != (previous.world, previous.stage, previous.area):
            events.append(
                {"frame": frame, "type": "area_transition", "to": [s.world, s.stage, s.area]}
            )
        events.extend(_phase_events(previous, s, frame))
        for track in s.enemy_tracks:
            if not track.observed:
                continue
            m = track.measurement
            dx, dy = m.x - s.x, m.y - s.screen_y
            distance = hypot(dx, dy)
            if (
                track.id not in encounters
                or distance < encounters[track.id]["origin_distance_pixels"]
            ):
                encounters[track.id] = {
                    "id": track.id,
                    "kind": m.kind,
                    "role": "interactable" if m.kind in {"springboard", "flagpole"} else "enemy",
                    "closest_frame": frame,
                    "dx": dx,
                    "dy": dy,
                    "origin_distance_pixels": round(distance, 2),
                }
        if s.landed_this_frame:
            landings.append(
                {
                    "frame": frame,
                    "x": s.x,
                    "height_y": s.y,
                    "enemy_offsets": [
                        {
                            "id": t.id,
                            "dx": t.measurement.x - s.x,
                            "dy": t.measurement.y - s.screen_y,
                        }
                        for t in s.enemy_tracks
                        if t.observed
                    ],
                }
            )
        if previous.dy < 0 and s.dy > 0 and not previous.grounded:
            events.append({"frame": frame, "type": "upward_bounce"})
        if previous.status != s.status:
            events.append(
                {"frame": frame, "type": "powerup_change", "from": previous.status, "to": s.status}
            )
        if offset + 1 == cycle:
            cycle_end_player = _player(s)
        landing_resolved = offset + 1 >= horizon and s.grounded
        # Swimming need not end on a floor. Still extend an unresolved descent:
        # coasting into a seabed opening must not hide death just after the horizon.
        swimming_resolved = offset + 1 >= horizon and s.swimming and s.dy >= 0 and s.y > 32
        finished = ended or landing_resolved or swimming_resolved or offset + 1 == horizon + 48
        if frame % 4 == 0 or finished or offset + 1 == cycle or offset + 1 == horizon:
            samples.append(_sample(frame, s, actual))
        if ended:
            terminal_frame = frame
        if finished:
            break
    outcome = _outcome(s, ended)
    summary = {
        "outcome": outcome,
        "risk": _risk(s, ended, events),
        "evaluated_through_frame": frame,
        "terminal_frame": terminal_frame,
        "cycle_end_player": cycle_end_player,
        "end_player": _player(s),
        "progress_pixels": s.x - application_x,
        "max_forward_progress_pixels": max_x - application_x,
        "max_height_y": max_height,
        "first_cycle": {
            "max_x": first_cycle_max_x,
            "max_height_y": first_cycle_max_height,
        },
        "continuation_complete": wait_complete,
        "landing_resolved": landing_resolved,
        "swimming_resolved": swimming_resolved,
        "landings": landings,
        "events": events,
        "enemy_encounters": sorted(encounters.values(), key=lambda e: e["closest_frame"]),
    }
    if (
        not wait_complete or (not ended and not (landing_resolved or swimming_resolved))
    ) and summary["risk"] == "safe":
        summary["risk"] = "unknown"
    if route_goal is not None:
        summary["route"] = {
            "brake_cycles": brake_cycles,
            "clear_frame": route_clear_frame,
        }
        if pipe_route:
            summary["route"].update(
                mouth_x=route_goal["mouth_x"],
                pipe_entered=route_clear_frame is not None,
            )
        else:
            summary["route"].update(
                obstacle_start_x=route_goal["obstacle_start_x"],
                obstacle_end_x=route_goal["obstacle_end_x"],
                staging_landed=staging_landed,
                obstacle_cleared=route_clear_frame is not None,
            )
    trace = {"action": action.value, "continuation": continuation, "samples": samples}
    return summary, trace


def forecast_actions(
    env: Any,
    parser: MarioStateParser,
    snapshot: MarioSnapshot,
    actions: Sequence[Action],
) -> MarioSnapshot:
    """Read-only with respect to the live episode, including when simulation fails."""
    base = getattr(env, "unwrapped", env)
    if not all(callable(getattr(base, name, None)) for name in ("dump_state", "load_state")):
        return replace(
            snapshot,
            prediction={
                "status": "unavailable",
                "reason": "native snapshot API unavailable",
                "action_forecasts": [],
            },
            prediction_traces=(),
        )
    started = perf_counter()
    # Never recursively capture previous bulky forecasts in every branch.
    observed = replace(snapshot, prediction=None, prediction_traces=())
    original = Checkpoint.capture(env, parser, observed)
    delay, cycle = snapshot.frames_until_action, snapshot.decision_horizon_frames
    recovery = snapshot.recovery_features().get("active", False)
    spring_nearby = any(
        t.measurement.kind == "springboard"
        and t.observed
        and abs(t.measurement.x - snapshot.x) <= 128
        for t in snapshot.enemy_tracks
    )
    horizon = max(96 if recovery or spring_nearby else 48, cycle * 2)
    if cycle < 1 or delay < 0:
        raise ValueError("Prediction requires a positive cycle and nonnegative delay")
    forecasts, traces = [], []
    try:
        _, simulated_parser, s = original.restore(env)
        ended = False
        prefix_events = []
        prefix = [_sample(0, s, None)]
        for offset in range(delay):
            if snapshot.scheduled_action is None:
                raise ValueError("A committed prefix requires scheduled_action")
            action = Action(snapshot.scheduled_action)
            if offset == 0:
                action = (
                    Action(snapshot.scheduled_first_frame_action)
                    if (snapshot.scheduled_first_frame_action is not None)
                    else first_frame_action(
                        action,
                        grounded=s.grounded,
                        previous_action=s.previous_action,
                        swimming=s.swimming,
                    )
                )
            previous = s
            s, ended = _step(env, simulated_parser, action, include_geometry=True)
            prefix_events.extend(_phase_events(previous, s, offset + 1))
            if previous.status != s.status:
                prefix_events.append(
                    {
                        "frame": offset + 1,
                        "type": "powerup_change",
                        "from": previous.status,
                        "to": s.status,
                    }
                )
            if (offset + 1) % 4 == 0 or ended or offset + 1 == delay:
                prefix.append(_sample(offset + 1, s, action))
            if ended:
                break
        committed = {
            "player": _player(s),
            "outcome": _outcome(s, ended),
            "risk": _risk(s, ended, prefix_events),
            "events": prefix_events,
            "frames": offset + 1 if delay else 0,
            "enemies": [t.to_state() for t in s.enemy_tracks],
        }
        traces.append({"phase": "committed", "samples": prefix})
        if not ended:
            application = Checkpoint.capture(env, simulated_parser, s)
            route_goal = staging_geometry(s) or pipe_entry_geometry(s)
            entry_route = bool(route_goal and route_goal.get("kind") == "side_exit_pipe")
            continuations = ["repeat", "run_jump"]
            if recovery or spring_nearby:
                continuations += [
                    "walk",
                    "release_then_jump",
                    "repeat_twice_then_jump",
                    "hold_jump_then_run",
                ]
            plants = [
                t for t in _nearby_plants(s) if (t.measurement.plant or {}).get("phase") != "hidden"
            ]
            wait_horizon = horizon
            if plants:
                continuations.append("wait_for_clearance")
                wait_horizon = min(
                    192,
                    max(
                        horizon,
                        max(
                            (t.measurement.plant or {}).get("estimated_frames_until_hidden") or 144
                            for t in plants
                        )
                        + cycle * 2
                        + 48,
                    ),
                )
            for action in actions:
                branches = {}
                simulated = {}
                for continuation in continuations:
                    equivalent = _IDENTICAL_CONTINUATIONS.get((action, continuation))
                    if equivalent in simulated:
                        summary, trace = deepcopy(simulated[equivalent])
                        trace["continuation"] = continuation
                    else:
                        _, branch_parser, branch_snapshot = application.restore(env)
                        summary, trace = _rollout(
                            env,
                            branch_parser,
                            branch_snapshot,
                            action,
                            continuation,
                            delay=delay,
                            cycle=cycle,
                            horizon=wait_horizon
                            if continuation == "wait_for_clearance"
                            else horizon,
                        )
                    simulated[continuation] = summary, trace
                    branches[continuation] = summary
                    traces.append(trace)
                # Sample sustained braking, not just one cycle then an immediate
                # forward jump. Restrict this extra search to staging geometry.
                route_lengths = (
                    (
                        range(1, 7)
                        if action == Action.LEFT
                        else (0,)
                        if action == (Action.RIGHT if entry_route else Action.RIGHT_RUN_JUMP)
                        else ()
                    )
                    if route_goal
                    else ()
                )
                for braking in route_lengths:
                    _, branch_parser, branch_snapshot = application.restore(env)
                    name = f"{'entry' if entry_route else 'staging'}_brake_{braking}"
                    summary, trace = _rollout(
                        env,
                        branch_parser,
                        branch_snapshot,
                        action,
                        name,
                        delay=delay,
                        cycle=cycle,
                        horizon=max(128, (braking + 12) * cycle),
                        brake_cycles=braking,
                        route_goal=route_goal,
                    )
                    branches[name] = summary
                    traces.append(trace)
                forecasts.append({"action": action.value, "continuations": branches})
        prediction = {
            "status": "available",
            "backend": "native_checkpoint",
            "version": 5,
            "observation_frame": snapshot.frame_index,
            "application_frame": delay,
            "action_cycle_frames": cycle,
            "horizon_after_application_frames": horizon,
            "airborne_extension_limit_frames": 48,
            "wait_horizon_limit_frames": 192,
            "staging_brake_limit_cycles": 6,
            "assumptions": (
                "Frame offsets start at observation. Native engine with privileged state, "
                "including future spawns. Candidates start AFTER committed_future. Each lasts "
                "one action cycle; repeat repeats that action, run_jump uses right_run_jump "
                "at subsequent cycle boundaries. Jump release/repress matches execution. "
                "walk uses RIGHT after the first cycle; release_then_jump uses NOOP for the "
                "second cycle then run-jumps; repeat_twice_then_jump repeats the candidate "
                "for two cycles then run-jumps; hold_jump_then_run holds JUMP for cycles two "
                "and three then run-jumps. wait_for_clearance uses NOOP after the first cycle "
                "until nearby plants are observed hidden and Mario grounded at a boundary, "
                "then run-jumps. It may use a longer horizon, capped at 192 frames; incomplete "
                "waits are unknown. Events show phase changes; these are conditional plans, "
                "not commitments to future actions. Re-evaluate at each actual cycle. "
                "When tall-obstacle staging geometry is present, staging_brake_N holds LEFT "
                "for N cycles then uses running jumps (N=0 starts running jumps immediately). "
                "These routes cover at least 128 frames and N+12 cycles, plus the airborne "
                "extension. Route evidence records intermediate landings and actual obstacle "
                "clearance; a high arc alone is not clearance. "
                "At dry sideways exits, entry_brake_N instead follows braking with RIGHT "
                "without A and records actual pipe-entry animation. "
                "Airborne paths extend up to 48 extra frames until landing or termination; "
                "unresolved land flights are unknown, not safe. In water A actions rearm at "
                "each cycle even without landing, producing repeated strokes. Swimming needs "
                "no landing: a non-sinking endpoint above the bottom danger zone resolves it; "
                "unresolved descents still extend and remain unknown unless resolved. "
                "Safe means only surviving the "
                "stated finite button sequence without observed "
                "damage; it is not safety under arbitrary follow-up choices. Enemy distances "
                "are sprite-origin references, not collision-box guarantees."
            ),
            "committed_future": committed,
            "action_forecasts": forecasts,
            "compute_ms": round((perf_counter() - started) * 1000, 2),
        }
        return replace(snapshot, prediction=prediction, prediction_traces=tuple(traces))
    finally:
        original.restore(env)
