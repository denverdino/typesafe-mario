import json
from dataclasses import replace
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.prediction import forecast_actions
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
def test_stuck_forecast_finds_retreat_spring_launch_and_wall_crossing(interaction_scene):
    env, parser, snapshot = interaction_scene
    result = forecast_actions(env, parser, snapshot, (Action.RIGHT_RUN_JUMP, Action.LEFT))
    p = result.prediction
    assert p["horizon_after_application_frames"] >= 96
    forecasts = {f["action"]: f["continuations"] for f in p["action_forecasts"]}
    assert forecasts["right_run_jump"]["repeat"]["max_forward_progress_pixels"] <= 2
    escape = forecasts["left"]["run_jump"]
    assert escape["risk"] == "safe"
    assert escape["end_player"]["x"] > 3072
    assert escape["max_height_y"] > 239
    assert any(o["kind"] == "springboard" for o in result.to_state()["interactables"])


@pytest.mark.parametrize("interaction_scene", [1600], indirect=True)
def test_deferred_jump_height_cannot_keep_a_repeated_idle_action_forever(interaction_scene):
    from typesafe_mario.policy import Decision
    from typesafe_mario.recovery import recover_decision

    env, parser, snapshot = interaction_scene
    for _ in range(120):
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action.NOOP])
        snapshot = parser.parse(
            info, env.unwrapped.ram, previous_action="noop", previous_reward=reward
        )
    assert snapshot.x == 3026
    result = forecast_actions(env, parser, snapshot, tuple(Action))
    assert result.to_state()["recovery"]["active"]
    proposal = Decision(Action.NOOP, 0.9, {"noop": 0.9, "left": 0.02}, 1)
    chosen = recover_decision(result.to_state(), proposal, tuple(Action))
    assert chosen.action != Action.NOOP
    assert chosen.selection["mode"] == "recovery"


@pytest.mark.parametrize("interaction_scene", [1600], indirect=True)
def test_model_forecasts_bound_size_without_losing_candidate_outcomes(interaction_scene):
    env, parser, snapshot = interaction_scene
    result = forecast_actions(env, parser, snapshot, tuple(Action))
    model = result.to_state()
    assert len(json.dumps(model)) < 24000
    assert len(model["prediction"]["action_forecasts"]) == len(Action)
    for raw, compact in zip(
        result.prediction["action_forecasts"], model["prediction"]["action_forecasts"], strict=True
    ):
        assert raw["action"] == compact["action"]
        assert raw["continuations"].keys() == compact["continuations"].keys()
        for name, branch in raw["continuations"].items():
            for key in (
                "risk",
                "outcome",
                "max_forward_progress_pixels",
                "max_height_y",
                "continuation_complete",
            ):
                assert compact["continuations"][name][key] == branch[key]
            rows = compact["continuations"][name]["enemy_encounters"]
            assert len(rows) == len(branch["enemy_encounters"])
            for encounter, row in zip(branch["enemy_encounters"], rows, strict=True):
                obj = model["prediction"]["encounter_objects"][row[0]]
                assert obj["id"] == encounter["id"]
                assert obj["kind"] == encounter["kind"]
                assert row[1:] == [encounter[k] for k in ("closest_frame", "dx", "dy")]
    assert result.to_debug_state()["prediction_details"] == result.prediction


@pytest.mark.parametrize("interaction_scene", [312], indirect=True)
def test_wait_forecast_observes_hidden_phase_before_launch_after_48_frames(interaction_scene):
    env, parser, snapshot = interaction_scene
    result = forecast_actions(env, parser, snapshot, (Action.NOOP,))
    branches = result.prediction["action_forecasts"][0]["continuations"]
    assert "wait_for_clearance" in branches
    wait = branches["wait_for_clearance"]
    events = wait["events"]
    hidden = next(
        e["frame"] for e in events if e["type"] == "plant_phase" and e["phase"] == "hidden"
    )
    launch = next(e["frame"] for e in events if e["type"] == "wait_complete")
    assert 48 < hidden <= launch
    assert wait["continuation_complete"]
    trace = next(
        t for t in result.prediction_traces if t.get("continuation") == "wait_for_clearance"
    )
    assert all(
        sample["action"] in (None, "noop")
        for sample in trace["samples"]
        if sample["frame"] < launch
    )


@pytest.mark.parametrize("interaction_scene", [312], indirect=True)
def test_wait_timeout_is_unknown_not_an_infinite_safe_plan(interaction_scene, monkeypatch):
    env, parser, snapshot = interaction_scene
    # Freeze only the plant's engine timers/position; real emulator still moves Mario.
    original = env.step
    initial = bytes(env.unwrapped.ram)

    def frozen_plant(action):
        result = original(action)
        for address in [0xCF, 0x58, 0xA0, 0x78A]:
            env.unwrapped.ram[address] = initial[address]
        return result

    monkeypatch.setattr(env, "step", frozen_plant)
    result = forecast_actions(env, parser, replace(snapshot, dx=0), (Action.NOOP,))
    wait = result.prediction["action_forecasts"][0]["continuations"]["wait_for_clearance"]
    assert wait["risk"] == "unknown"
    assert not wait["continuation_complete"]
    assert wait["evaluated_through_frame"] <= 192
