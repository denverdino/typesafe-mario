from dataclasses import replace

from typesafe_mario.state import MarioStateParser


def plant_frame(*, y=136, speed=255, moving=0, timer=40, kind=13):
    ram = bytearray(0x800)
    ram[0xF], ram[0x16], ram[0x87], ram[0xCF] = 1, kind, 152, y
    ram[0x417], ram[0x434], ram[0x58], ram[0xA0], ram[0x78A] = 136, 160, speed, moving, timer
    return MarioStateParser().parse({"x_pos": 130, "y_pos": 79, "y_pixel": 176}, ram)


def test_piranha_id_and_extended_phase_are_identified_from_ram():
    e = plant_frame().enemies[0]
    assert e.kind == "piranha_plant"
    assert e.plant["phase"] == "extended"
    assert e.plant["estimated_frames_until_hidden"] == 88
    assert e.plant["pipe_top_y"] == 127
    assert e.plant["emergence_suppressed_by_proximity"] is False


def test_retracting_is_not_hidden_even_when_current_frame_has_no_vertical_movement():
    s = plant_frame(y=148, speed=1, moving=1, timer=0)
    assert s.enemies[0].vertical_velocity_y is None
    assert s.enemies[0].plant["phase"] == "retracting"
    assert s.enemies[0].plant["estimated_frames_until_hidden"] == 24
    assert s.to_state()["hazard"]["pipe_plants"][0]["wait_before_pipe_jump"] is True


def test_hidden_plant_releases_wait_and_reports_pause_without_promising_safe_landing():
    s = plant_frame(y=160, speed=1, timer=64)
    plant = s.to_state()["hazard"]["pipe_plants"][0]
    assert plant["phase"] == "hidden"
    assert plant["hidden_pause_frames_remaining"] == 64
    assert plant["wait_before_pipe_jump"] is False


def test_rising_plant_remains_a_wait_hazard():
    e = plant_frame(y=148, moving=1, timer=0).enemies[0]
    assert e.plant["phase"] == "rising"
    assert e.plant["estimated_frames_until_hidden"] == 136


def test_committed_emergence_is_not_hidden_before_its_first_movement_frame():
    s = plant_frame(y=160, speed=255, moving=1, timer=0)
    e = s.enemies[0]
    assert e.plant["phase"] == "rising"
    assert e.plant["emergence_suppressed_by_proximity"] is False
    assert s.to_state()["hazard"]["pipe_plants"][0]["wait_before_pipe_jump"] is True


def test_spiny_is_not_a_piranha_and_invalid_limits_are_unknown():
    e = plant_frame(kind=0x12).enemies[0]
    assert e.kind == "spiny"
    assert e.plant is None
    e = plant_frame(y=100).enemies[0]
    assert e.plant["phase"] == "unknown"


def test_pipe_wait_does_not_stop_mario_already_above_or_past_the_plant():
    s = plant_frame()
    above = replace(s, y=159)
    assert above.to_state()["hazard"]["pipe_plants"][0]["wait_before_pipe_jump"] is False
    behind = replace(s, enemies=(replace(s.enemies[0], dx_pixels=-40),))
    assert behind.to_state()["hazard"].get("pipe_plants", []) == []
