import json
from pathlib import Path

import pytest
from prediction_helpers import predicted

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.runner import create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def interaction_scene(request):
    pytest.importorskip("gym_super_mario_bros")
    data = json.loads((Path(__file__).parent / "fixtures/world2-1-spring.json").read_text())
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser = MarioStateParser()
        snapshot = parser.parse(info, env.unwrapped.ram)
        frame = 0
        for name, count in data["buttons"]:
            for _ in range(count):
                if frame == request.param:
                    yield env, parser, snapshot
                    return
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(name)])
                snapshot = parser.parse(
                    info, env.unwrapped.ram, previous_action=name, previous_reward=reward
                )
                frame += 1
    finally:
        env.close()


@pytest.mark.parametrize("interaction_scene", [1600], indirect=True)
def test_spring_forecast_expresses_uncertainty_without_claiming_a_clearing_route(interaction_scene):
    _, _, s = interaction_scene
    result = predicted(s)
    assert all("route" not in f for f in result.prediction["action_forecasts"])
    assert all(f["risk"] != "safe" for f in result.prediction["action_forecasts"])
    from typesafe_mario.observation import model_state

    assert any(a["kind"] == "springboard" for a in model_state(result)["visible_actors"])


@pytest.mark.parametrize("interaction_scene", [312], indirect=True)
def test_plant_prediction_does_not_use_engine_countdown(interaction_scene):
    from typesafe_mario.observation import model_state

    _, _, s = interaction_scene
    result = predicted(s)
    text = json.dumps(model_state(result))
    assert "estimated_frames_until_hidden" not in text
    assert "wait_complete" not in text
    assert "engine_state" not in text
    assert "wait_for_clearance" not in text
    assert len(model_state(result)["prediction"]["action_forecasts"]) == 7
