from dataclasses import replace

import pytest
from test_observation_prediction import floor_scene


def test_ground_acceleration_fit_uses_positions_and_excludes_collisions():
    from typesafe_mario.calibration import fit_profile
    from typesafe_mario.observation import Block

    probe = [
        floor_scene(frame=f, x=40 + 0.05 * f * (f + 1), vx=0.1 * f, previous_action="right_run")
        for f in range(21)
    ]
    profile = fit_profile([probe], env_id="test")
    assert profile["parameters"]["run_accel"]["value"] == pytest.approx(0.1)
    collided = [replace(o, blocks=(*o.blocks, Block(o.x + 12, o.x + 28, 79, 143))) for o in probe]
    assert "run_accel" not in fit_profile([collided], env_id="test")["parameters"]


def test_profile_seeds_prediction_and_survives_episode_reset_without_fake_samples():
    from typesafe_mario.prediction import Predictor

    p = Predictor(physics_parameters={"run_accel": 0.1})
    assert p.dynamics.value("run_accel") == pytest.approx(0.1)
    assert not p.dynamics.samples
    p.dynamics.add("run_accel", 0.4)
    assert p.dynamics.value("run_accel") == pytest.approx(0.1)
    p.observe(floor_scene(frame=10))
    p.observe(floor_scene(frame=0))
    assert p.dynamics.value("run_accel") == pytest.approx(0.1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 100])
def test_invalid_calibration_cannot_override_physical_bounds(value):
    from typesafe_mario.dynamics import Dynamics

    with pytest.raises(ValueError):
        Dynamics(calibrated={"run_accel": value})


def test_profile_rejects_a_different_environment(tmp_path):
    import json

    from typesafe_mario.calibration import load_profile

    path = tmp_path / "physics.json"
    path.write_text(json.dumps({"schema_version": 1, "env_id": "other", "parameters": {}}))
    with pytest.raises(ValueError, match="environment"):
        load_profile(path, env_id="test")


def test_calibration_cli_never_constructs_a_provider(tmp_path):
    from unittest.mock import patch

    from typesafe_mario.cli import main

    with (
        patch("typesafe_mario.calibration.collect_calibration", return_value={}) as collect,
        patch("typesafe_mario.cli.TypeSafePolicy", side_effect=AssertionError("API constructed")),
    ):
        assert main(["calibrate", "--env", "test", "--output", str(tmp_path / "p.json")]) == 0
    assert collect.call_args.kwargs["env_id"] == "test"


def test_validation_compares_frozen_models_on_unseen_actual_motion():
    from typesafe_mario.calibration import validate_profile

    heldout = [
        floor_scene(frame=f, x=40 + 0.05 * f * (f + 1), vx=0.1 * f, previous_action="right_run")
        for f in range(25)
    ]
    result = validate_profile([heldout], {"run_accel": 0.1})
    assert result["samples"] > 0
    assert (
        result["calibrated"]["mean_absolute_x_error"] < result["default"]["mean_absolute_x_error"]
    )


def test_validation_includes_input_onset_and_excludes_entry_animation():
    from typesafe_mario.calibration import validate_profile

    onset = [
        floor_scene(
            frame=f,
            x=40 + 0.05 * max(0, f - 8) * max(0, f - 7),
            vx=0.1 * max(0, f - 8),
            previous_action="noop" if f <= 8 else "right_run",
        )
        for f in range(17)
    ]
    assert validate_profile([onset], {"run_accel": 0.1})["samples"] == 1
    entry = [
        floor_scene(frame=f, x=40, y=200 - 5 * f, vy=-5, grounded=False, previous_action="noop")
        for f in range(17)
    ]
    assert validate_profile([entry], {})["samples"] == 0


@pytest.mark.parametrize("bad", ["gap", "level", "unreliable", "terminal", "swimming"])
def test_jump_impulse_rejects_invalid_observation_pairs(bad):
    from typesafe_mario.calibration import fit_profile

    probes = []
    for _ in range(2):
        a = floor_scene(frame=0, x=40, vx=0, previous_action="noop")
        b = replace(a, frame=1, y=84, vy=5, grounded=False, previous_action="jump")
        b = replace(
            b,
            **{
                "gap": {"frame": 100},
                "level": {"level": (2, 1, 1)},
                "unreliable": {"motion_reliable": False},
                "terminal": {"terminal": True},
                "swimming": {"swimming": True},
            }[bad],
        )
        probes.append([a, b])
    assert "jump_speed" not in fit_profile(probes, env_id="test")["parameters"]
