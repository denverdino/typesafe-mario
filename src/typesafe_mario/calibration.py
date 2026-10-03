"""Fit physics from controlled, real observations without a policy provider.

Each probe is a fresh actual episode, never a restored or branched emulator
snapshot. The fitted model receives positions and visible geometry only.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, Action, first_frame_action
from .dynamics import PRIORS, Dynamics, direction, position_fit
from .observation import observe

PROFILE_VERSION = 1


def _usable_window(window):
    first, last = window[0], window[-1]
    return all(
        h.frame == first.frame + i
        and h.level == last.level
        and not h.swimming
        and not h.terminal
        and h.motion_reliable
        and h.known_x is not None
        and not Dynamics.contact(h)
        for i, h in enumerate(window)
    )


def fit_profile(probes, *, env_id):
    """Pure fitting: no environment, RAM, API, or hypothetical observations."""
    samples = defaultdict(list)
    for history in probes:
        fast_jump = False
        for i, o in enumerate(history):
            if i:
                a = history[i - 1]
                if (
                    a.grounded
                    and not o.grounded
                    and o.y > a.y
                    and o.previous_action in JUMP_ACTIONS
                    and _usable_window([a, o])
                ):
                    speed = a.vx
                    ground = history[max(0, i - 9) : i]
                    if (
                        len(ground) == 9
                        and _usable_window(ground)
                        and all(h.grounded for h in ground)
                    ):
                        speed, _ = position_fit([h.x for h in ground])
                    fast_jump = abs(speed) >= 2
                    samples["jump_speed_fast" if fast_jump else "jump_speed"].append(o.y - a.y)
            if i < 8:
                continue
            window = history[i - 8 : i + 1]
            first = window[0]
            if not _usable_window(window):
                continue
            vx, ax = position_fit([h.x for h in window])
            vy, ay = position_fit([h.y for h in window])
            same_input = all(h.previous_action == o.previous_action for h in window[1:])
            if same_input and o.previous_action and all(h.grounded for h in window):
                d = direction(Action(o.previous_action))
                start = vx - 8 * ax
                if not d and start * vx > 0 and min(abs(start), abs(vx)) > 0.25:
                    samples["friction"].append(-ax if vx > 0 else ax)
                elif d and d * start < -0.25 and d * vx < -0.1:
                    samples["brake_accel"].append(d * ax)
                elif (
                    d
                    and d * start >= -0.1
                    and d * vx < (2.8 if "run" in o.previous_action else 1.5)
                ):
                    samples["run_accel" if "run" in o.previous_action else "walk_accel"].append(
                        d * ax
                    )
            if all(not h.grounded for h in window):
                held = all(h.previous_action in JUMP_ACTIONS for h in window[1:])
                released = all(h.previous_action not in JUMP_ACTIONS for h in window[1:])
                if held and min(vy, vy - 8 * ay) > 1:
                    samples["gravity_hold_fast" if fast_jump else "gravity_hold"].append(-ay)
                elif (
                    (released or all(a.y > b.y for a, b in pairwise(window)))
                    and -4.3 < vy < 4.3
                    and abs(vy - 8 * ay) < 4.3
                ):
                    samples["gravity_fall"].append(-ay)
                if same_input and o.previous_action:
                    d = direction(Action(o.previous_action))
                    if d and max(abs(vx), abs(vx - 8 * ax)) < 1.4:
                        samples["air_accel"].append(d * ax)
                    elif d and abs(ax) < 0.025 and 1.4 < d * vx < 2:
                        samples["air_walk_speed"].append(d * (o.x - first.x) / 8)
    parameters = {}
    for name, values in samples.items():
        _, lo, hi = PRIORS[name]
        if len(values) < 2:
            continue
        value = median(values)
        if lo <= value <= hi:
            parameters[name] = {
                "value": round(value, 6),
                "samples": len(values),
                "median_absolute_deviation": round(median(abs(v - value) for v in values), 6),
            }
    return {
        "schema_version": PROFILE_VERSION,
        "env_id": env_id,
        "source": "controlled_actual_observations",
        "parameters": parameters,
        "uncalibrated": sorted(set(PRIORS) - parameters.keys()),
        "sample_note": "Overlapping fitting windows; sample counts are not independent trials.",
    }


def validate_profile(probes, parameters):
    """Compare eight-frame predictions on held-out real inputs, with fitting frozen."""
    from .prediction import Predictor

    errors = {"default": [], "calibrated": []}
    coverage = defaultdict(int)
    for history in probes:
        settled = next((i for i, h in enumerate(history) if h.grounded), len(history))
        models = {"default": Predictor(), "calibrated": Predictor(physics_parameters=parameters)}
        for i, o in enumerate(history):
            for model in models.values():
                model.observe(o)
            if i < max(8, settled) or i + 8 >= len(history):
                continue
            future = history[i + 1 : i + 9]
            action = Action(future[0].previous_action)
            if i % 4 and action == o.previous_action:
                continue
            if not _usable_window([o, *future]) or any(h.previous_action != action for h in future):
                continue
            if (
                first_frame_action(action, grounded=o.grounded, previous_action=o.previous_action)
                != action
            ):
                continue
            for name, model in models.items():
                model.dynamics = Dynamics(calibrated=parameters if name == "calibrated" else None)
                forecast = model.forecast(o, (action,), cycle=8)
                end = forecast["action_forecasts"][0]["first_cycle"]
                errors[name].append((abs(end["x"] - future[-1].x), abs(end["y"] - future[-1].y)))
            coverage[("ground_" if o.grounded else "air_") + action.value] += 1
    result = {
        "samples": len(errors["default"]),
        "horizon_frames": 8,
        "source": "held_out_actual_probes",
        "learning_during_validation": False,
        "coverage_by_input_and_phase": dict(coverage),
    }
    for name, values in errors.items():
        result[name] = {
            "mean_absolute_x_error": round(fmean(v[0] for v in values), 3) if values else None,
            "mean_absolute_y_error": round(fmean(v[1] for v in values), 3) if values else None,
            "max_position_error": round(max(max(v) for v in values), 3) if values else None,
        }
    return result


def load_profile(path: Path, *, env_id):
    data = path.read_bytes()
    profile = json.loads(data)
    if profile.get("schema_version") != PROFILE_VERSION or profile.get("env_id") != env_id:
        raise ValueError("Physics profile version or environment does not match")
    values = {name: entry["value"] for name, entry in profile["parameters"].items()}
    Dynamics(calibrated=values)  # Validate finite values and physical bounds.
    return values, {"path": str(path), "sha256": sha256(data).hexdigest(), "parameters": values}


def _probe_room(o):
    if o.terminal or o.swimming or not o.motion_reliable or o.known_x is None:
        return False
    # Stop before an approaching actor or unknown floor makes probing unsafe.
    if any(abs(a.x - o.x) < 72 for a in o.actors if a.kind != "flagpole"):
        return False
    return any(b.top <= o.y + 1 and b.left <= o.x - 8 and o.x + 32 < b.right for b in o.blocks)


def collect_calibration(*, env_id, seed, output: Path):
    from .runner import create_mario_env
    from .state import MarioStateParser

    # Excite one regime at a time. Repeated vertical probes measure jump
    # impulse independently; no action is selected by trying alternative futures.
    probes = [
        ("walk_coast_brake", [(Action.RIGHT, 16), (Action.NOOP, 12), (Action.LEFT, 12)]),
        ("run_brake", [(Action.RIGHT_RUN, 16), (Action.LEFT, 14)]),
        ("held_jump", [(Action.JUMP, 48)]),
        ("held_jump_repeat", [(Action.JUMP, 48)]),
        ("released_jump", [(Action.JUMP, 8), (Action.NOOP, 32)]),
        ("air_steering", [(Action.JUMP, 8), (Action.RIGHT_JUMP, 12), (Action.LEFT, 12)]),
        ("fast_jump", [(Action.RIGHT_RUN, 24), (Action.JUMP, 16)]),
        ("fast_jump_repeat", [(Action.RIGHT_RUN, 24), (Action.JUMP, 16)]),
        (
            "validation_walk",
            [(Action.LEFT, 8), (Action.RIGHT, 16), (Action.NOOP, 9), (Action.LEFT, 12)],
        ),
        (
            "validation_run",
            [(Action.RIGHT, 6), (Action.NOOP, 4), (Action.RIGHT_RUN, 19), (Action.LEFT, 14)],
        ),
        ("validation_jump", [(Action.RIGHT, 10), (Action.NOOP, 8), (Action.JUMP, 40)]),
        (
            "validation_release",
            [(Action.RIGHT, 5), (Action.NOOP, 4), (Action.JUMP, 11), (Action.NOOP, 23)],
        ),
        ("validation_fast_jump", [(Action.RIGHT_RUN, 27), (Action.JUMP, 20)]),
        ("validation_steering", [(Action.JUMP, 10), (Action.RIGHT_JUMP, 16), (Action.LEFT, 8)]),
    ]
    env = create_mario_env(env_id, render_mode="rgb_array")
    records, histories = [], []
    try:
        for name, phases in probes:
            _, info = env.reset(seed=seed)
            parser = MarioStateParser()
            snapshot = parser.parse(info, env.unwrapped.ram)
            history = [observe(snapshot)]

            def step(action, parser=parser, history=history):
                nonlocal snapshot
                _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
                snapshot = parser.parse(
                    info, env.unwrapped.ram, previous_action=action, previous_reward=reward
                )
                history.append(observe(snapshot))
                return terminated or truncated or snapshot.dead

            reason = "completed"
            for _ in range(80):
                if snapshot.grounded:
                    break
                if step(Action.NOOP):
                    reason = "terminal"
                    break
            if not snapshot.grounded:
                reason = "no_grounded_start"
            else:
                for action, count in phases:
                    for index in range(count):
                        if not _probe_room(history[-1]):
                            reason = "insufficient_visible_space"
                            break
                        actual = (
                            first_frame_action(
                                action,
                                grounded=snapshot.grounded,
                                previous_action=snapshot.previous_action,
                            )
                            if index == 0
                            else action
                        )
                        if step(actual):
                            reason = "terminal"
                            break
                    if reason != "completed":
                        break
            histories.append(history)
            records.append(
                {"name": name, "reason": reason, "observations": [asdict(o) for o in history]}
            )
            print(f"Calibration {name}: {len(history) - 1} actual frames, {reason}")
    finally:
        env.close()
    training = [
        h
        for h, (name, _) in zip(histories, probes, strict=True)
        if not name.startswith("validation_")
    ]
    validation = [
        h for h, (name, _) in zip(histories, probes, strict=True) if name.startswith("validation_")
    ]
    profile = fit_profile(training, env_id=env_id)
    profile["validation"] = validate_profile(
        validation, {name: p["value"] for name, p in profile["parameters"].items()}
    )
    profile["seed"] = seed
    profile["probes"] = [
        {"name": r["name"], "reason": r["reason"], "frames": len(r["observations"]) - 1}
        for r in records
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix(".observations.json")
    trace.write_text(json.dumps({"env_id": env_id, "seed": seed, "probes": records}) + "\n")
    profile["observations_file"] = str(trace)
    output.write_text(json.dumps(profile, indent=2) + "\n")
    return profile
