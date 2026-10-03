import json
from dataclasses import replace
from pathlib import Path

import pytest
from prediction_helpers import assess_forecast

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.runner import create_mario_env
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


def test_staging_prediction_has_no_verified_route_or_automatic_override(approach):
    from prediction_helpers import predicted
    from test_policy import Provider, policy

    from typesafe_mario.observation import model_state

    _, _, s = approach
    result = predicted(s)
    assert model_state(result)["terrain"]["blocks"]
    assert all("route" not in f for f in result.prediction["action_forecasts"])
    requested = Action(assess_forecast(result.prediction)["candidate_actions"][-1])
    choice = policy(Provider(requested.value)).choose(result, tuple(Action))
    assert choice.action == requested
    if choice.selection is not None:
        assert requested.value in choice.selection["candidate_actions"]
