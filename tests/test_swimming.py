"""Native regressions for the underwater stall in run-20261002T023404.251790Z."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from prediction_helpers import predicted

from typesafe_mario.actions import ACTION_TO_INDEX, Action
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


def test_observed_water_forecasts_compare_strokes_without_guaranteeing_survival(underwater):
    _, _, s = underwater
    result = predicted(s, (Action.RIGHT_JUMP, Action.RIGHT))
    stroke, release = result.prediction["action_forecasts"]
    assert stroke["first_cycle"]["y"] > release["first_cycle"]["y"]
    assert all("swimming_dynamics" in f["unknowns"] for f in (stroke, release))
    assert all(f["risk"] not in ("safe", "unsafe") for f in (stroke, release))


def test_water_committed_prefix_is_an_estimate_with_uncertainty(underwater):
    _, _, s = underwater
    result = predicted(replace(s, frames_until_action=8, scheduled_action="right_jump"))
    future = result.prediction["committed_future"]
    assert future["frame"] == 8
    assert future["y_range"][0] < future["y"] < future["y_range"][1]
