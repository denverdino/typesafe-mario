from dataclasses import replace

from typesafe_mario.state import MarioStateParser


def stair_snapshot(x=2046, y=89):
    ram = bytearray(0x800)
    ram[0x57], ram[0x1D] = 28, 2
    for left, right, row in [
        (2016, 2128, 11),
        (2128, 2144, 10),
        (2144, 2160, 9),
        (2160, 2176, 8),
        (2176, 2208, 7),
    ]:
        for tx in range(left // 16, right // 16):
            for r in range(row, 13):
                ram[0x500 + ((tx // 16) % 2) * 208 + r * 16 + tx % 16] = 0x54
    s = MarioStateParser().parse({"x_pos": x, "y_pos": y, "y_pixel": 255 - y}, ram)
    return replace(s, dx=2, dy=-5, frames_until_action=8, scheduled_action="right_jump")


def test_landing_before_distant_stairs_prompts_supported_approach():
    s = stair_snapshot()
    cue = s.to_state()["terrain"]["stair_approach"]
    assert cue["application_x_estimate"] == 2060
    assert cue["first_step_x"] == 2128
    assert cue["distance_at_application_pixels"] == 68
    assert cue["approach_supported"] is True


def test_second_approach_cycle_but_no_delay_once_within_jump_window():
    s = replace(stair_snapshot(2060, 79), grounded=True, dy=0, scheduled_action="right")
    assert s.to_state()["terrain"]["stair_approach"]["distance_at_application_pixels"] == 54
    s = replace(stair_snapshot(2074, 79), grounded=True, dy=0, scheduled_action="right")
    assert "stair_approach" not in s.to_state()["terrain"]


def test_does_not_cut_active_jump_or_delay_without_supported_application():
    s = stair_snapshot()
    for variant in [
        replace(s, dy=3),
        replace(s, frames_until_action=0),
        replace(s, grounded=True, y=79),
        replace(s, collision_grid=()),
    ]:
        assert "stair_approach" not in variant.to_state()["terrain"]


def test_flat_ground_and_single_obstacle_do_not_trigger():
    s = stair_snapshot()
    for grid in [
        tuple("." * 11 for _ in range(13)),
        tuple("." * 11 for _ in range(11)) + ("###########",) * 2,
    ]:
        assert "stair_approach" not in replace(s, collision_grid=grid).to_state()["terrain"]
