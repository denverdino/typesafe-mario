from dataclasses import replace

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.observation import observe
from typesafe_mario.prediction import Predictor
from typesafe_mario.state import MarioStateParser
from typesafe_mario.tactical import build_tactical_state


def scene(*, enemy_start=150, enemy_end=149, player_start=100, player_end=102, y=176, y_start=None):
    parser, predictor = MarioStateParser(), Predictor()
    parser.parse({"x_pos": player_start, "left_x_pos": 100, "y_pos": 79})
    for player_x, enemy_x, enemy_y in (
        (player_start, enemy_start, y if y_start is None else y_start),
        (player_end, enemy_end, y),
    ):
        ram = bytearray(2048)
        ram[0xF], ram[0x16], ram[0xB6], ram[0xCF] = 1, 6, 1, enemy_y
        ram[0x6E], ram[0x87] = divmod(enemy_x, 256)
        snapshot = parser.parse(
            {"x_pos": player_x, "left_x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram
        )
        predictor.observe(observe(snapshot))
    forecast = predictor.forecast(observe(snapshot), (Action.NOOP,), cycle=8)
    return replace(snapshot, prediction=forecast)


def assessment(snapshot):
    return build_tactical_state(snapshot, (Action.NOOP,)).state["actor_assessments"][0]


@pytest.mark.parametrize(
    "positions,direction,trend,closing",
    [
        ({}, "left", "closing", 3),
        ({"enemy_start": 50, "enemy_end": 51}, "right", "opening", -1),
        ({"enemy_end": 152}, "right", "unchanged", 0),
        ({"enemy_end": 150}, "no_observed_displacement", "closing", 2),
    ],
)
def test_relative_trend_accounts_for_mario_motion_and_enemy_side(
    positions, direction, trend, closing
):
    result = assessment(scene(**positions))
    assert result["horizontal_motion"] == direction
    assert result["horizontal_separation"]["trend"] == trend
    assert result["horizontal_separation"]["closing_speed_px_per_frame"] == closing


def test_relative_position_uses_current_world_coordinates_and_upward_y():
    snapshot = scene(y=160)
    result = assessment(snapshot)
    assert result["id"] == observe(snapshot).actors[0].identity
    assert result["relative_position"] == {"x": 47, "y": 16}


@pytest.mark.parametrize("y_start,direction", [(177, "up"), (175, "down")])
def test_vertical_motion_converts_screen_y_to_upward_world_y(y_start, direction):
    assert assessment(scene(y_start=y_start))["vertical_motion"] == direction


@pytest.mark.parametrize("missing", ["prediction", "velocity", "identity", "frame", "status"])
def test_unavailable_or_unmatched_evidence_does_not_claim_stationary(missing):
    snapshot = scene()
    if missing == "prediction":
        snapshot = replace(snapshot, prediction=None)
    elif missing == "velocity":
        snapshot.prediction["actor_forecasts"][0]["observed_velocity"]["available"] = False
    elif missing == "identity":
        snapshot.prediction["actor_forecasts"][0]["id"] = "old-slot-generation"
    elif missing == "frame":
        del snapshot.prediction["observed_at_frame"]
    else:
        snapshot.prediction["status"] = "unavailable"
    result = assessment(snapshot)
    assert result["horizontal_motion"] == "unknown"
    assert result["vertical_motion"] == "unknown"
    assert result["horizontal_separation"] == {
        "trend": "unknown",
        "closing_speed_px_per_frame": None,
    }


def test_unreliable_mario_motion_cannot_establish_relative_trend():
    result = assessment(replace(scene(), dx=100))
    assert result["horizontal_motion"] == "left"
    assert result["horizontal_separation"]["trend"] == "unknown"


def test_same_horizontal_position_does_not_report_a_time_to_contact():
    result = assessment(scene(enemy_start=103, enemy_end=102))
    assert result["horizontal_separation"]["trend"] == "unknown"
    assert result["horizontal_separation"]["closing_speed_px_per_frame"] is None


def test_ram_direction_does_not_change_tactical_summary():
    snapshot = scene()
    altered = replace(
        snapshot,
        enemy_tracks=tuple(
            replace(
                track,
                measurement=replace(track.measurement, engine_direction_raw=1, engine_state=4),
            )
            for track in snapshot.enemy_tracks
        ),
    )
    assert assessment(snapshot) == assessment(altered)
