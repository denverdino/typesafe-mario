import json

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.observation import model_state, observe
from typesafe_mario.prediction import Predictor
from typesafe_mario.state import MarioStateParser


def frame(
    parser,
    *,
    x=150,
    y=176,
    y_high=1,
    kind=6,
    active=1,
    direction=2,
    engine_state=0,
    camera=0,
    hidden=False,
    ram_present=True,
):
    ram = bytearray(0x800)
    ram[0x0F], ram[0x16], ram[0x1E], ram[0x46] = active, kind, engine_state, direction
    ram[0x6E], ram[0x87] = divmod(x, 256)
    ram[0xB6], ram[0xCF] = y_high, y
    if hidden:
        ram[0x10], ram[0x17], ram[0x6F], ram[0x88], ram[0xB7], ram[0xD0] = 1, 6, 3, 0, 1, 176
    return parser.parse(
        {"x_pos": camera + 100, "left_x_pos": 100, "y_pos": 79, "y_pixel": 176},
        ram if ram_present else None,
    )


@pytest.mark.parametrize(
    "kind,name",
    [
        (0x0A, "grey_cheep_cheep"),
        (0x0B, "red_cheep_cheep"),
        (0x0E, "green_paratroopa_jump"),
        (0x0F, "red_paratroopa"),
        (0x10, "green_paratroopa_fly"),
        (0x14, "flying_cheep_cheep"),
    ],
)
def test_fish_and_paratroopas_have_distinct_observed_types(kind, name):
    s = frame(MarioStateParser(), kind=kind)
    assert s.enemies[0].kind == name
    assert observe(s).actors[0].kind == name


@pytest.mark.parametrize("y_high", [0, 2, 255])
def test_vertical_page_outside_screen_cannot_reappear_via_low_byte(y_high):
    s = frame(MarioStateParser(), y_high=y_high)
    assert not observe(s).actors
    assert not model_state(s)["visible_actors"]
    assert s.enemy_tracks[0].measurement.y == 176  # Raw diagnostic evidence remains.


@pytest.mark.parametrize(
    "interruption",
    [
        {"active": 0},
        {"x": 400},
        {"y_high": 2},
        {"ram_present": False},
    ],
)
def test_visible_identity_restarts_after_disappearance(interruption):
    p = MarioStateParser()
    old = observe(frame(p)).actors[0]
    assert not observe(frame(p, **interruption)).actors
    new = observe(frame(p)).actors[0]
    assert new.identity != old.identity


def test_teleport_breaks_identity_and_observed_velocity():
    p, predictor = MarioStateParser(), Predictor()
    old = observe(frame(p))
    predictor.observe(old)
    new = observe(frame(p, x=190))
    predictor.observe(new)
    assert new.actors[0].identity != old.actors[0].identity
    result = predictor.forecast(new, (Action.NOOP,), cycle=8)
    assert not result["actor_forecasts"][0]["observed_velocity"]["available"]


def test_direction_and_state_are_diagnostic_without_changing_visible_identity():
    p = MarioStateParser()
    first = observe(frame(p, active=2, direction=1))
    s = frame(p, active=2, direction=2, engine_state=4)
    second = observe(s)
    assert first.actors == second.actors
    raw = s.enemy_tracks[0].to_state()
    assert raw.get("engine_direction_raw") == 2
    assert raw.get("active_flag") == 2
    assert raw["engine_state"] == 4
    text = json.dumps(model_state(s))
    assert "engine_direction" not in text and "engine_state" not in text


def test_world_page_and_camera_scroll_do_not_break_continuous_identity():
    p = MarioStateParser()
    first = observe(frame(p, x=255, camera=100)).actors[0]
    second = observe(frame(p, x=256, camera=101)).actors[0]
    assert first.identity == second.identity
    assert second.x == 256
    assert second.y == 79


def test_hidden_slot_activity_does_not_change_visible_generation():
    a, b = MarioStateParser(), MarioStateParser()
    for kwargs in ({}, {"active": 0}, {}):
        visible = observe(frame(a, **kwargs))
        with_hidden = observe(frame(b, hidden=True, **kwargs))
        assert visible.actors == with_hidden.actors


def test_parser_reset_invalidates_visible_identity():
    p = MarioStateParser()
    old = observe(frame(p)).actors[0].identity
    p.reset()
    assert observe(frame(p)).actors[0].identity != old
