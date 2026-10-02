import json
from dataclasses import replace
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.policy import Decision
from typesafe_mario.prediction import forecast_actions
from typesafe_mario.recovery import recover_decision
from typesafe_mario.runner import _first_frame_action, create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture(params=[0, 1])
def approach(request):
    pytest.importorskip("gym_super_mario_bros")
    data = json.loads((Path(__file__).parent / "fixtures/world4-2-staging.json").read_text())
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser = MarioStateParser()
        s = parser.parse(info, env.unwrapped.ram)
        for name, count in data["episodes"][request.param]["buttons"]:
            for _ in range(count):
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(name)])
                s = parser.parse(
                    info, env.unwrapped.ram, previous_action=name, previous_reward=reward
                )
        yield env, parser, s
    finally:
        env.close()


def test_tall_obstacle_exposes_staging_surfaces_without_gap_trigger(approach):
    _, _, s = approach
    route = s.to_state()["terrain"]["staging_route"]
    assert route["obstacle_start_x"] == 2880
    assert route["obstacle_standing_y"] == 191
    assert any(t["standing_y"] == 127 for t in route["staging_surfaces"])
    assert "staging_route" not in replace(s, swimming=True).to_state()["terrain"]
    translated = replace(s, x=s.x + 512, world=8, stage=1)
    assert translated.to_state()["terrain"]["staging_route"]["obstacle_start_x"] == 3392
    from typesafe_mario.routing import staging_geometry

    assert staging_geometry(replace(s, side_exit_pipe={"mouth_x": 2880})) is None
    assert staging_geometry(replace(s, collision_grid=tuple("." * 11 for _ in range(13)))) is None


def test_native_route_preserves_staging_and_clears_logged_wall(approach):
    env, parser, s = approach
    predicted = forecast_actions(env, parser, s, tuple(Action))
    proposal = Decision(Action.RIGHT_RUN_JUMP, 0.9, {"right_run_jump": 0.9, "left": 0.01}, 0)
    chosen = recover_decision(predicted.to_state(), proposal, tuple(Action))
    assert chosen.selection["mode"] == "staging_route"
    assert chosen.action == Action.LEFT
    name = chosen.selection["continuation"]
    branch = next(f for f in predicted.prediction["action_forecasts"] if f["action"] == "left")
    branch = branch["continuations"][name]
    assert branch["risk"] == "safe"
    assert branch["route"]["obstacle_cleared"]
    assert branch["route"]["staging_landed"]
    assert any(p["height_y"] in (127, 143) for p in branch["landings"])
    # Reproduce the actual simulated macro sequence, not a mocked predictor.
    trace = next(t for t in predicted.prediction_traces if t.get("continuation") == name)
    assert trace["samples"]
    brake_cycles = branch["route"]["brake_cycles"]
    for frame in range(branch["evaluated_through_frame"]):
        requested = Action.LEFT if frame < brake_cycles * 8 else Action.RIGHT_RUN_JUMP
        action = _first_frame_action(s, requested) if frame % 8 == 0 else requested
        _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
        s = parser.parse(info, env.unwrapped.ram, previous_action=action, previous_reward=reward)
        assert not (terminated or truncated or s.dead)
    assert (s.x, s.y) == (branch["end_player"]["x"], branch["end_player"]["height_y"])


def test_recovery_prefers_new_supported_progress_to_returning_to_same_wall():
    state = {
        "player": {"x": 2842, "y": 79},
        "recovery": {
            "active": True,
            "anchor_x": 2866,
            "recent_max_height": 143,
            "action_frames": {},
        },
        "prediction": {"committed_future": {"player": {"x": 2842}}, "action_forecasts": []},
    }
    for action, advance, peak, end_x, end_y in [
        ("jump", 24, 207, 2866, 79),
        ("left", 0, 143, 2816, 143),
    ]:
        state["prediction"]["action_forecasts"].append(
            {
                "action": action,
                "continuations": {
                    "repeat": {
                        "risk": "safe",
                        "max_forward_progress_pixels": advance,
                        "max_height_y": peak,
                        "end_player": {"x": end_x, "height_y": end_y, "grounded": True},
                        "events": [],
                    }
                },
            }
        )
    proposal = Decision(Action.JUMP, 0.9, {"jump": 0.9, "left": 0.1}, 0)
    assert recover_decision(state, proposal, (Action.JUMP, Action.LEFT)).action == Action.LEFT


def test_replanned_route_survives_committed_cycles_without_random_recovery(approach):
    env, parser, s = approach
    actions = (Action.LEFT, Action.RIGHT_RUN_JUMP)
    proposal = Decision(Action.RIGHT_RUN_JUMP, 0.9, {"right_run_jump": 0.9, "left": 0.01}, 0)
    first = forecast_actions(env, parser, s, actions)
    current = recover_decision(first.to_state(), proposal, actions).action
    for _ in range(28):
        request = replace(
            s,
            frames_until_action=8,
            scheduled_action=current.value,
            scheduled_first_frame_action=_first_frame_action(s, current).value,
        )
        predicted = forecast_actions(env, parser, request, actions)
        following = recover_decision(predicted.to_state(), proposal, actions).action
        for frame in range(8):
            actual = _first_frame_action(s, current) if frame == 0 else current
            _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[actual])
            s = parser.parse(
                info, env.unwrapped.ram, previous_action=actual, previous_reward=reward
            )
            assert not (terminated or truncated or s.dead)
        committed = predicted.prediction["committed_future"]["player"]
        assert (s.x, s.y) == (committed["x"], committed["height_y"])
        if s.x > 2920:
            break
        current = following
    assert s.x > 2920


def test_staging_selector_rejects_incomplete_unsafe_or_disallowed_routes():
    from typesafe_mario.routing import select_staging_route

    proposal = Decision(Action.RIGHT, 1, {"right": 1}, 0)
    branch = {
        "risk": "safe",
        "continuation_complete": True,
        "route": {"obstacle_cleared": True, "clear_frame": 80},
    }
    state = {
        "prediction": {
            "action_forecasts": [{"action": "left", "continuations": {"staging_brake_3": branch}}]
        }
    }
    assert select_staging_route(state, proposal, (Action.RIGHT,)) is None
    for change in ({"risk": "unsafe"}, {"risk": "unknown"}, {"continuation_complete": False}):
        original = branch.copy()
        branch.update(change)
        assert select_staging_route(state, proposal, tuple(Action)) is None
        branch.update(original)


def test_route_clearance_does_not_compare_coordinates_across_areas(monkeypatch):
    from typesafe_mario import prediction

    start = MarioStateParser().parse({"x_pos": 100, "y_pos": 79, "area": 1})
    arrived = replace(start, x=500, y=200, area=2, grounded=True)
    monkeypatch.setattr(prediction, "_step", lambda *args: (arrived, False))
    goal = {
        "obstacle_start_x": 160,
        "obstacle_end_x": 192,
        "obstacle_standing_y": 175,
        "staging_surfaces": [],
    }
    branch, _ = prediction._rollout(
        None,
        None,
        start,
        Action.RIGHT_RUN_JUMP,
        "staging_brake_0",
        delay=0,
        cycle=8,
        horizon=16,
        brake_cycles=0,
        route_goal=goal,
    )
    assert not branch["route"]["obstacle_cleared"]


def test_verified_multi_cycle_retreat_enters_side_pipe_instead_of_rejumping():
    pytest.importorskip("gym_super_mario_bros")
    data = json.loads((Path(__file__).parent / "fixtures/world4-2-exit.json").read_text())
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser = MarioStateParser()
        s = parser.parse(info, env.unwrapped.ram)
        for name, count in data["buttons"]:
            for _ in range(count):
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(name)])
                s = parser.parse(
                    info, env.unwrapped.ram, previous_action=name, previous_reward=reward
                )
        assert (s.x, s.y) == (3010, 177)
        predicted = forecast_actions(env, parser, s, tuple(Action))
        proposal = Decision(Action.RIGHT_RUN_JUMP, 1, {"right_run_jump": 1}, 0)
        choice = recover_decision(predicted.to_state(), proposal, tuple(Action))
        assert choice.selection["mode"] == "pipe_entry_route"
        assert choice.action == Action.LEFT
        assert choice.selection["route"]["pipe_entered"]
        assert choice.selection["route"]["brake_cycles"] == 3
        entered = False
        for frame in range(128):
            a = Action.LEFT if frame < 24 else Action.RIGHT
            _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[a])
            s = parser.parse(info, env.unwrapped.ram, previous_action=a, previous_reward=reward)
            entered |= s.player_state == 2
            assert not (terminated or truncated or s.dead)
        assert entered
    finally:
        env.close()
