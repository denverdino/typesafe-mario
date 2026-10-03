"""World 1-4 regressions reduced from run-20261002T021302.140176Z."""

from dataclasses import replace

import pytest
from prediction_helpers import predicted

from typesafe_mario.actions import ACTION_TO_INDEX, Action, first_frame_action
from typesafe_mario.runner import create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def castle():
    pytest.importorskip("gym_super_mario_bros")
    env = create_mario_env("SuperMarioBros-1-4-v0", render_mode="rgb_array")
    try:
        _, info = env.reset(seed=123)
        parser = MarioStateParser(decision_horizon_frames=8)
        s = parser.parse(info, env.unwrapped.ram)
        # The repeated restart path holds A through the descending entrance stairs.
        for frame in range(48):
            action = (
                first_frame_action(
                    Action.RIGHT_RUN_JUMP, grounded=s.grounded, previous_action=s.previous_action
                )
                if frame % 8 == 0
                else Action.RIGHT_RUN_JUMP
            )
            _, reward, _, _, info = env.step(ACTION_TO_INDEX[action])
            s = parser.parse(
                info, env.unwrapped.ram, previous_action=action.value, previous_reward=reward
            )
        assert (s.x, s.y) == (124, 160)
        yield (
            env,
            parser,
            replace(
                s,
                frames_until_action=8,
                scheduled_action="right_run_jump",
                scheduled_first_frame_action="right_run_jump",
            ),
        )
    finally:
        env.close()


def test_castle_forecasts_are_estimates_without_claiming_actual_death_frame(castle):
    _, _, s = castle
    result = predicted(s, (Action.RIGHT_RUN_JUMP, Action.LEFT))
    for f in result.prediction["action_forecasts"]:
        assert "terminal_frame" not in f and "outcome" not in f
        assert "model_error" in f["unknowns"]
        assert f["first_cycle"]["frame"] == 16
        assert f["confidence"] == "low"


def test_castle_risk_filter_preserves_the_selected_eligible_action(castle):
    from test_policy import Provider, policy

    _, _, s = castle
    result = predicted(s)
    choice = policy(Provider("left")).choose(result, tuple(Action))
    assert choice.action == Action.LEFT
    assert choice.selection["method"] == "observation_risk_filter"
