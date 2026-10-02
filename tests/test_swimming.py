"""Native regressions for the underwater stall in run-20261002T023404.251790Z."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.prediction import forecast_actions
from typesafe_mario.runner import _first_frame_action, create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def underwater(request):
    pytest.importorskip("gym_super_mario_bros")
    filename = getattr(request, "param", "world2-2-swimming.json")
    fixture = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())
    env = create_mario_env(fixture["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=fixture["seed"])
        parser = MarioStateParser()
        s = parser.parse(info, env.unwrapped.ram)
        for action, count in fixture["buttons"]:
            for _ in range(count):
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(action)])
                s = parser.parse(
                    info, env.unwrapped.ram, previous_action=action, previous_reward=reward
                )
        if filename == "world2-2-swimming.json":
            assert (s.x, s.y) == (1060, 167)
        yield env, parser, s
    finally:
        env.close()


def test_swim_macro_escapes_logged_pit_and_matches_forecast(underwater):
    env, parser, s = underwater
    predicted = forecast_actions(env, parser, s, (Action.RIGHT_JUMP,))
    branch = predicted.prediction["action_forecasts"][0]["continuations"]["repeat"]
    assert branch["risk"] == "safe"
    assert not branch["end_player"]["grounded"]
    assert branch["swimming_resolved"]
    end = branch["evaluated_through_frame"]
    for frame in range(96):
        action = _first_frame_action(s, Action.RIGHT_JUMP) if frame % 8 == 0 else Action.RIGHT_JUMP
        _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
        s = parser.parse(
            info, env.unwrapped.ram, previous_action=action.value, previous_reward=reward
        )
        assert not (terminated or truncated or s.dead)
        if frame + 1 == end:
            assert (s.x, s.y) == (branch["end_player"]["x"], branch["end_player"]["height_y"])
    assert s.x >= 1200
    assert s.y >= 200


def test_sinking_without_strokes_is_still_unsafe(underwater):
    env, parser, s = underwater
    predicted = forecast_actions(env, parser, s, (Action.LEFT,))
    branch = predicted.prediction["action_forecasts"][0]["continuations"]["repeat"]
    assert branch["risk"] == "unsafe"
    assert branch["outcome"] == "death"


@pytest.mark.parametrize("underwater", ["world2-2-exit.json"], indirect=True)
def test_water_exit_is_recognized_and_release_enters_it(underwater):
    env, parser, s = underwater
    pipe = s.to_state()["terrain"]["side_exit_pipe"]
    assert pipe["mouth_x"] == 3024
    assert pipe["entry_standing_y"] == 143
    assert pipe["above_entry"]
    for _ in range(40):
        _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[Action.RIGHT])
        s = parser.parse(info, env.unwrapped.ram, previous_action="right", previous_reward=reward)
        assert not (terminated or truncated or s.dead)
        if not s.swimming:
            break
    assert not s.swimming
    assert not s.to_state()["player"]["swimming"]


def test_water_state_uses_engine_flag_and_clears_on_land():
    ram = bytearray(0x800)
    ram[0x704] = 1
    ram[0x1D] = 1
    parser = MarioStateParser()
    s = parser.parse({"world": 2, "stage": 2, "y_pos": 150}, ram, previous_action="jump")
    state = s.to_state()
    assert state["player"]["swimming"]
    assert state["player"]["can_start_jump"]
    assert state["player"]["jump_requires_release"]
    assert state["trajectory"]["landing_projection"]["confidence"] == "not_applicable"
    assert "gap_takeoff_window" not in state["terrain"]
    ram[0x704] = 0
    land = parser.parse({"world": 2, "stage": 2, "y_pos": 150}, ram, previous_action="jump")
    assert not land.to_state()["player"]["swimming"]
    assert _first_frame_action(land, Action.JUMP) == Action.JUMP
    assert not parser.parse({"world": 2, "stage": 2}).swimming


def test_swimming_committed_prefix_rearms_like_execution(underwater):
    env, parser, s = underwater
    s = replace(s, frames_until_action=8, scheduled_action="right_jump")
    predicted = forecast_actions(env, parser, s, (Action.RIGHT_JUMP,))
    for frame in range(8):
        action = _first_frame_action(s, Action.RIGHT_JUMP) if frame == 0 else Action.RIGHT_JUMP
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[action])
        s = parser.parse(
            info, env.unwrapped.ram, previous_action=action.value, previous_reward=reward
        )
    committed = predicted.prediction["committed_future"]["player"]
    assert (s.x, s.y) == (committed["x"], committed["height_y"])
    assert s.y > 167


def test_falling_swimmer_keeps_stroke_actions_without_prediction():
    from test_policy import Provider, policy

    ram = bytearray(0x800)
    ram[0x704] = 1
    ram[0x1D] = 1
    s = replace(MarioStateParser().parse({}, ram), dy=-3, grounded=False)
    provider = Provider("right_jump")
    decision = policy(provider).choose(s, tuple(Action))
    assert decision.action == Action.RIGHT_JUMP
    assert set(provider.request["questions"]["next_action"].criteria) == {a.value for a in Action}
