import pytest

from typesafe_mario.state import MarioStateParser


@pytest.mark.parametrize("positions", [(2, 65535, 65528, 8), (900, 902, 0, 8)])
def test_coordinate_wrap_or_same_area_warp_rebases_progress(positions):
    p = MarioStateParser()
    for x in positions:
        p.parse({"x_pos": x, "y_pos": 79})
    for x in range(9, 210):
        s = p.parse({"x_pos": x, "y_pos": 79})
    recovery = s.to_state()["recovery"]
    assert not recovery["active"]
    assert 205 <= recovery["anchor_x"] <= 209
    assert recovery["no_progress_frames"] < 4
    # A real stall after the rebase must still activate recovery.
    for _ in range(60):
        s = p.parse({"x_pos": 209, "y_pos": 79})
    assert s.to_state()["recovery"]["active"]


def test_gradual_retreat_does_not_reset_stall_deadline():
    p = MarioStateParser()
    for x in range(500, 299, -2):
        s = p.parse({"x_pos": x, "y_pos": 79}, previous_action="left")
    recovery = s.to_state()["recovery"]
    assert recovery["active"]
    assert recovery["anchor_x"] == 500
    assert recovery["no_progress_frames"] == 100


def test_stationary_wall_input_is_distinct_from_waiting_or_a_jump_in_progress():
    p = MarioStateParser()
    for _ in range(64):
        s = p.parse({"x_pos": 100, "y_pos": 79}, previous_action="right")
    assert s.recovery["stationary_frames"] >= 48
    assert s.recovery["stationary_action_frames"]["right"] >= 48
    s = p.parse({"x_pos": 100, "y_pos": 85}, previous_action="jump")
    assert s.recovery["stationary_frames"] == 0
    assert s.recovery["stationary_action_frames"] == {}


def test_repeated_jumps_and_small_horizontal_oscillation_trigger_recovery():
    p = MarioStateParser()
    for frame in range(80):
        s = p.parse(
            {"x_pos": 100 + frame % 3, "y_pos": 79 + frame % 32}, previous_action="right_run_jump"
        )
    recovery = s.to_state().get("recovery", {})
    assert recovery.get("active") is True
    assert recovery["no_progress_frames"] >= 48
    assert recovery["action_frames"]["right_run_jump"] >= 48


def test_meaningful_progress_and_area_transition_clear_recovery():
    p = MarioStateParser()
    for _ in range(60):
        s = p.parse({"x_pos": 100, "y_pos": 79})
    assert s.to_state()["recovery"]["active"]
    s = p.parse({"x_pos": 109, "y_pos": 79})
    assert not s.to_state()["recovery"]["active"]
    for _ in range(60):
        s = p.parse({"x_pos": 109, "y_pos": 79})
    s = p.parse({"x_pos": 10, "y_pos": 79, "area": 2})
    assert not s.to_state()["recovery"]["active"]
    p.reset()
    assert not p.parse({"x_pos": 100}).to_state()["recovery"]["active"]


def test_supported_plant_wait_is_intentional_but_not_unbounded():
    from dataclasses import replace

    from test_piranha import plant_frame

    s = plant_frame()
    s = replace(
        s,
        previous_action="noop",
        recovery={
            "active": True,
            "no_progress_frames": 80,
            "action_frames": {"noop": 80},
        },
    )
    assert not s.to_state()["recovery"]["active"]
    assert s.to_state()["recovery"]["intentional_wait"]
    s = replace(s, recovery={**s.recovery, "no_progress_frames": 200})
    assert s.to_state()["recovery"]["active"]
