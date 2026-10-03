import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from prediction_helpers import assess_forecast

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


def test_prediction_has_no_emulator_access_or_side_effects(scene, monkeypatch):
    from prediction_helpers import predicted

    env, parser, snapshot = scene
    ram, history = bytes(env.unwrapped.ram), copy.deepcopy(vars(parser))

    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction must not advance or snapshot the emulator")

    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr(env.unwrapped, "dump_state", forbidden)
    monkeypatch.setattr(env.unwrapped, "load_state", forbidden)
    result = predicted(snapshot)
    assert result.prediction["backend"] == "observation_dynamics"
    assert bytes(env.unwrapped.ram) == ram and vars(parser) == history
    assert all(f["risk"] not in ("safe", "unsafe") for f in result.prediction["action_forecasts"])


def test_causal_validation_waits_for_actual_selected_actions(scene):
    from prediction_helpers import predicted

    from typesafe_mario.observation import observe
    from typesafe_mario.prediction import Predictor

    env, parser, snapshot = scene
    p = Predictor()
    p.observe(observe(snapshot))
    result = predicted(snapshot, predictor=p).prediction
    assert p.validation()["samples"] == 0
    for _ in range(8):
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP])
        snapshot = parser.parse(
            info, env.unwrapped.ram, previous_action="right_run_jump", previous_reward=reward
        )
        p.observe(observe(snapshot))
    p.expect(result, Action.LEFT, apply_at_frame=snapshot.frame_index)
    for _ in range(8):
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action.LEFT])
        snapshot = parser.parse(
            info, env.unwrapped.ram, previous_action="left", previous_reward=reward
        )
        p.observe(observe(snapshot))
    assert p.validation()["samples"] == 1
    assert p.validation()["mean_absolute_x_error"] >= 0


def test_grounded_held_jump_releases_then_represses_in_estimate():
    from test_observation_prediction import floor_scene

    from typesafe_mario.prediction import Predictor

    p = Predictor()
    held = floor_scene(previous_action="jump", vx=0)
    released = replace(held, previous_action="noop")
    a = p.forecast(held, (Action.JUMP,), cycle=8)["action_forecasts"][0]["first_cycle"]
    b = p.forecast(released, (Action.JUMP,), cycle=8)["action_forecasts"][0]["first_cycle"]
    assert 79 < a["y"] < b["y"]


def test_forecasts_do_not_need_native_snapshot_support():
    from prediction_helpers import predicted

    result = predicted(MarioStateParser().parse({}))
    assert result.prediction["status"] == "estimated"
    assert all("unobserved_terrain" in f["unknowns"] for f in result.prediction["action_forecasts"])


def test_recorded_first_enemy_failure_warns_early_and_resolves_ceiling():
    from typesafe_mario.observation import observe
    from typesafe_mario.prediction import Predictor

    pytest.importorskip("gym_super_mario_bros")
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/world1-1-ceiling-risk.json").read_text()
    )
    env = create_mario_env(fixture["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=fixture["seed"])
        parser, predictor = MarioStateParser(), Predictor()
        s = parser.parse(info, env.unwrapped.ram)
        predictor.observe(observe(s))
        forecasts = {}
        for action, count in fixture["buttons"]:
            for _ in range(count):
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(action)])
                s = parser.parse(
                    info, env.unwrapped.ram, previous_action=action, previous_reward=reward
                )
                predictor.observe(observe(s))
                if s.frame_index in (80, 88):
                    forecasts[s.frame_index] = predictor.forecast(
                        observe(s),
                        tuple(Action),
                        cycle=8,
                        delay=8,
                        scheduled_action=Action.RIGHT_RUN_JUMP,
                    )
        assert s.dead and s.x == 315  # Exact recorded execution, no counterfactual trials.
        assert assess_forecast(forecasts[80])["candidate_actions"] == ["left"]
        assert forecasts[88]["committed_future"]["y"] < 117
        forward = next(
            f for f in forecasts[88]["action_forecasts"] if f["action"] == "right_run_jump"
        )
        assert forward["control_risk"]["possible_contact"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "name,frame,kept,rejected",
    [
        ("world1-1-gap-release", 664, "right_run_jump", "right"),
        # Falling phase is uncertain: keep braking, reject the nominal running
        # collision. Walking followed by braking is no longer certified worse.
        ("world1-2-falling-enemy", 1072, "left", "right_run"),
    ],
)
def test_recorded_failure_context_keeps_a_preventive_action(name, frame, kept, rejected):
    from typesafe_mario.observation import observe
    from typesafe_mario.prediction import Predictor

    pytest.importorskip("gym_super_mario_bros")
    fixture = json.loads((Path(__file__).parent / f"fixtures/{name}.json").read_text())
    buttons = [Action(a) for a, count in fixture["buttons"] for _ in range(count)]
    env = create_mario_env(fixture["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=fixture["seed"])
        parser, predictor = MarioStateParser(), Predictor()
        s = parser.parse(info, env.unwrapped.ram)
        predictor.observe(observe(s))
        for action in buttons[:frame]:
            _, reward, _, _, info = env.step(ACTION_TO_INDEX[action])
            s = parser.parse(
                info, env.unwrapped.ram, previous_action=action, previous_reward=reward
            )
            predictor.observe(observe(s))
        result = predictor.forecast(
            observe(s), tuple(Action), cycle=8, delay=8, scheduled_action=buttons[frame]
        )
        assert kept in assess_forecast(result)["candidate_actions"]
        assert rejected not in assess_forecast(result)["candidate_actions"]
    finally:
        env.close()


def test_recorded_quantized_apices_do_not_make_visible_wall_unclimbable():
    from typesafe_mario.observation import observe
    from typesafe_mario.prediction import Predictor

    pytest.importorskip("gym_super_mario_bros")
    fixture = json.loads((Path(__file__).parent / "fixtures/world1-2-held-apex.json").read_text())
    env = create_mario_env(fixture["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=fixture["seed"])
        parser, p = MarioStateParser(), Predictor()
        s = parser.parse(info, env.unwrapped.ram)
        p.observe(observe(s))
        for name, count in fixture["buttons"]:
            for _ in range(count):
                a = Action(name)
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[a])
                s = parser.parse(info, env.unwrapped.ram, previous_action=a, previous_reward=reward)
                p.observe(observe(s))
        assert (s.x, s.y) == (471, 79)
        assert p.dynamics.value("gravity_hold") < 0.24
        result = p.forecast(
            observe(s), tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN
        )
        plan = next(
            b["continuation"] for b in result["action_forecasts"] if b["action"] == "right_jump"
        )
        assert plan["progress"] > 32
        assert plan["status"] == "estimated_viable"
    finally:
        env.close()
