from dataclasses import replace

from typesafe_mario.state import MarioStateParser


def pipe_ram(mouth_x=2656, row=6):
    ram = bytearray(0x800)

    def put(tx, r, value):
        ram[0x500 + ((tx // 16) % 2) * 208 + r * 16 + tx % 16] = value

    tx = mouth_x // 16
    put(tx, row, 0x1C)
    put(tx, row + 1, 0x1F)
    for col in range(tx - 5, tx + 6):
        put(col, row + 2, 0x52)
    return ram


def parse(ram, x=2674, y=159, player_state=8):
    return MarioStateParser().parse(
        {"x_pos": x, "y_pos": y, "world": 1, "stage": 2, "area": 3, "player_state": player_state},
        ram,
    )


def test_side_pipe_reports_entry_floor_and_retreat_from_top():
    s = parse(pipe_ram())
    pipe = s.to_state()["terrain"]["side_exit_pipe"]
    assert pipe["mouth_x"] == 2656
    assert pipe["entry_standing_y"] == 127
    assert pipe["retreat_target_x"] == 2632
    assert pipe["above_entry"] is True
    assert pipe["over_mouth"] is True
    assert pipe["approach_supported"] is True
    assert replace(s, x=2630, y=127).to_state()["terrain"]["side_exit_pipe"]["above_entry"] is False


def test_side_pipe_coordinates_follow_tiles_and_vertical_row_not_level_position():
    pipe = parse(pipe_ram(320, 9), x=280, y=79).to_state()["terrain"]["side_exit_pipe"]
    assert pipe["mouth_x"] == 320
    assert pipe["entry_standing_y"] == 79
    assert pipe["over_mouth"] is False


def test_non_pipe_walls_vertical_pipes_and_incomplete_pairs_do_not_trigger():
    for top, bottom in [(0x52, 0x52), (0x10, 0x11), (0x1C, 0)]:
        ram = pipe_ram()
        tx = 2656 // 16
        base = 0x500 + ((tx // 16) % 2) * 208 + tx % 16
        ram[base + 6 * 16], ram[base + 7 * 16] = top, bottom
        assert "side_exit_pipe" not in parse(ram).to_state()["terrain"]
    assert "side_exit_pipe" not in parse(None).to_state()["terrain"]


def test_entry_animation_is_visible_even_if_mouth_leaves_local_scan():
    s = parse(None, player_state=2)
    assert s.to_state()["terrain"]["side_exit_pipe"]["entering"] is True
    assert "side_exit_pipe" not in parse(None, player_state=8).to_state()["terrain"]


def test_missing_approach_floor_does_not_claim_safe_retreat():
    ram = pipe_ram()
    tx = 2656 // 16 - 1
    ram[0x500 + ((tx // 16) % 2) * 208 + 8 * 16 + tx % 16] = 0
    assert parse(ram).to_state()["terrain"]["side_exit_pipe"]["approach_supported"] is False
