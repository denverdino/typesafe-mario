import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.runner import create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def scene(request):
    pytest.importorskip("gym_super_mario_bros")
    fixture = json.loads((Path(__file__).parent / "fixtures/world2-1-collision.json").read_text())
    env = create_mario_env(fixture["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=fixture["seed"])
        parser = MarioStateParser()
        s = parser.parse(info, env.unwrapped.ram)
        frame = 0
        for action, count in fixture["buttons"]:
            for _ in range(count):
                if frame == getattr(request, "param", 296):
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
                    return
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(action)])
                s = parser.parse(
                    info, env.unwrapped.ram, previous_action=action, previous_reward=reward
                )
                frame += 1
    finally:
        env.close()


def forecast(*args, **kwargs):
    from typesafe_mario.prediction import forecast_actions

    return forecast_actions(*args, **kwargs)


def test_forecasts_show_collision_and_early_braking_across_multiple_enemies(scene):
    env, p, s = scene
    result = forecast(env, p, s, (Action.RIGHT_RUN_JUMP, Action.LEFT))
    prediction = result.prediction
    assert prediction["committed_future"]["player"]["x"] == 558
    assert prediction["application_frame"] == 8
    by_action = {f["action"]: f for f in prediction["action_forecasts"]}
    run = by_action["right_run_jump"]["continuations"]["repeat"]
    assert run["outcome"] == "death"
    assert run["terminal_frame"] == 36
    assert run["end_player"]["x"] == 642
    assert len(run["enemy_encounters"]) >= 2
    assert len([e for e in run["enemy_encounters"] if e["kind"] == "goomba"]) == 2
    brake = by_action["left"]["continuations"]["run_jump"]
    assert brake["outcome"] == "survived_horizon"
    assert brake["end_player"]["x"] > 642
    assert brake["landings"]
    assert "prediction_traces" not in result.to_state()
    assert result.to_debug_state()["prediction_traces"]


def test_forecasting_restores_ram_parser_rewards_and_time_limit(scene):
    from typesafe_mario.checkpoint import Checkpoint

    env, p, s = scene
    cp = Checkpoint.capture(env, p, s)
    old_parser = copy.deepcopy(vars(p))
    ram = bytes(env.unwrapped.ram)
    before = cp.layer_state
    forecast(env, p, s, tuple(Action))
    after = Checkpoint.capture(env, p, s)
    assert bytes(env.unwrapped.ram) == ram
    assert after.layer_state == before
    assert vars(p) == old_parser
    # Actual next-frame transition and reward must also remain identical.
    actual = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP])
    actual_ram = bytes(env.unwrapped.ram)
    cp.restore(env)
    expected = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP])
    assert actual[1:] == expected[1:]
    assert bytes(env.unwrapped.ram) == actual_ram


def test_simulation_error_restores_before_propagating(scene, monkeypatch):
    env, p, s = scene
    ram = bytes(env.unwrapped.ram)
    original = env.step

    def failing(action):
        original(action)
        raise RuntimeError("simulation failed")

    monkeypatch.setattr(env, "step", failing)
    with pytest.raises(RuntimeError, match="simulation failed"):
        forecast(env, p, s, (Action.RIGHT,))
    assert bytes(env.unwrapped.ram) == ram


def test_missing_native_backend_reports_unknown():
    from types import SimpleNamespace

    p = MarioStateParser()
    s = p.parse({})
    result = forecast(SimpleNamespace(unwrapped=SimpleNamespace()), p, s, tuple(Action))
    assert result.to_state()["prediction"]["status"] == "unavailable"
    assert result.to_state()["prediction"]["action_forecasts"] == []


@pytest.mark.parametrize("scene", [280], indirect=True)
def test_candidate_rearms_jump_and_forecasts_exact_first_cycle(scene):
    env, p, s = scene
    assert s.grounded and s.previous_action == "right_run_jump"
    s = replace(s, frames_until_action=0, scheduled_action=None, scheduled_first_frame_action=None)
    result = forecast(env, p, s, (Action.RIGHT_RUN_JUMP,))
    branch = result.prediction["action_forecasts"][0]["continuations"]["repeat"]
    assert branch["cycle_end_player"]["x"] == 510
    assert branch["cycle_end_player"]["height_y"] == 112


@pytest.mark.parametrize("scene", [328], indirect=True)
def test_death_in_committed_prefix_cannot_be_fixed_by_future_action(scene):
    env, p, s = scene
    result = forecast(env, p, s, tuple(Action))
    assert result.prediction["committed_future"]["outcome"] == "death"
    assert result.prediction["committed_future"]["frames"] == 4
    assert result.prediction["action_forecasts"] == []


@pytest.mark.parametrize("scene", [328], indirect=True)
@pytest.mark.parametrize("delay", [0, 8])
def test_damage_is_visible_in_prefix_or_candidate_even_when_mario_survives(scene, delay):
    env, p, s = scene
    env.unwrapped.ram[0x756] = 1  # Tall Mario can survive the recorded collision.
    env.unwrapped.ram[0x754] = 0
    s = replace(s, status="tall", frames_until_action=delay)
    result = forecast(env, p, s, (Action.RIGHT_RUN_JUMP,))
    scope = (
        result.prediction["committed_future"]
        if delay
        else (result.prediction["action_forecasts"][0]["continuations"]["repeat"])
    )
    assert scope.get("risk") == "unsafe"
    player = scope["player"] if delay else scope["end_player"]
    assert player["powerup_status"] == "small"
    assert {"frame": 4, "type": "powerup_change", "from": "tall", "to": "small"} in scope["events"]
    assert env.unwrapped.ram[0x756] == 1


def test_forecast_prefix_matches_observed_execution_at_application(scene):
    env, p, s = scene
    result = forecast(env, p, s, (Action.LEFT,))
    for _ in range(8):
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP])
        applied = p.parse(
            info, env.unwrapped.ram, previous_action="right_run_jump", previous_reward=reward
        )
    predicted = result.prediction["committed_future"]
    assert predicted["player"]["x"] == applied.x
    assert predicted["player"]["height_y"] == applied.y
    assert predicted["enemies"] == [t.to_state() for t in applied.enemy_tracks]
