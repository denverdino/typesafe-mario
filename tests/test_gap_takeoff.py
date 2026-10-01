from dataclasses import replace

import pytest

from typesafe_mario.state import MarioStateParser


def approach(x=1230, *, speed=3, delay=8, horizon=8):
    ram = bytearray(0x800)
    for tile_x in range(x // 16 - 2, x // 16 + 9):
        if not 80 <= tile_x < 83:  # Gap x=1280..1328.
            address = 0x500 + ((tile_x // 16) % 2) * 208 + 11 * 16 + tile_x % 16
            ram[address] = 0x54
    s = MarioStateParser(decision_horizon_frames=horizon).parse(
        {"x_pos": x, "y_pos": 79, "y_pixel": 176}, ram
    )
    return replace(s, dx=speed, frames_until_action=delay, scheduled_action="right_run")


def test_gap_requires_jump_before_next_cycle_even_when_current_gap_ahead_is_false():
    s = approach()
    assert s.navigation_features()["gap_ahead"] is False
    window = s.gap_takeoff_features()
    assert window["must_jump_this_cycle"] is True
    assert window["predicted_application_center_x"] == 1262
    assert window["predicted_cycle_end_center_x"] == 1286
    assert window["safe_takeoff_center_max_x"] == 1272
    assert s.to_state()["terrain"]["gap_takeoff_window"] == window


def test_previous_cycle_is_not_urged_to_jump_prematurely():
    assert approach(1206).gap_takeoff_features() is None


def test_next_boundary_exactly_on_safe_edge_can_wait_one_cycle():
    assert approach(1216).gap_takeoff_features() is None
    assert approach(1240).gap_takeoff_features()["must_jump_this_cycle"] is True


def test_missed_window_does_not_promise_a_grounded_jump():
    window = approach(1254).gap_takeoff_features()
    assert window["must_jump_this_cycle"] is False
    assert window["safe_takeoff_window_missed"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"grounded": False},
        {"scheduled_action": "right_run_jump"},
        {"scheduled_action": "left"},
        {"dx": 0},
        {"dx": -2},
        {"collision_grid": (), "local_grid": ()},
    ],
)
def test_no_takeoff_window_for_airborne_jumping_reversing_or_unknown_support(changes):
    assert replace(approach(), **changes).gap_takeoff_features() is None


def test_timing_uses_action_frames_and_speed_not_api_latency():
    immediate = approach(1230, delay=0)
    assert immediate.gap_takeoff_features() is None
    assert approach(1230, delay=0, horizon=12).gap_takeoff_features()["must_jump_this_cycle"]
    assert approach(speed=1).gap_takeoff_features() is None
    assert (
        replace(approach(), last_response_delay_frames=100).gap_takeoff_features()
        == approach().gap_takeoff_features()
    )
