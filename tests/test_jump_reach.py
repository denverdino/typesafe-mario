from typesafe_mario.terrain import jump_reach_reference


def surface(left, right, y):
    return {
        "start_x": left,
        "end_x": right,
        "safe_center_min_x": left + 8,
        "safe_center_max_x": right - 8,
        "standing_y": y,
    }


def test_early_step_jump_falls_short_but_ground_approach_is_available():
    surfaces = [surface(1840, 1872, 111), surface(1872, 1920, 79), surface(1952, 1984, 127)]
    plan = jump_reach_reference(
        surfaces, x=1850, y=111, speed=1.75, horizon=8, gap_start=1920, gap_end=1952
    )
    assert plan["current_speed_jump"]["can_reach"] is False
    assert plan["required_distance_pixels"] == 102
    assert plan["approach_before_jumping"] is True
    assert plan["approach_supported"] is True


def test_closer_takeoff_can_reach_the_higher_landing():
    surfaces = [surface(1872, 1920, 79), surface(1952, 1984, 127)]
    plan = jump_reach_reference(
        surfaces, x=1892, y=79, speed=1.75, horizon=8, gap_start=1920, gap_end=1952
    )
    assert plan["current_speed_jump"]["can_reach"] is True
    assert plan["approach_before_jumping"] is False


def test_reach_accounts_for_landing_height():
    base = [surface(1840, 1920, 79)]
    low = jump_reach_reference(
        base + [surface(1952, 1984, 79)],
        x=1892,
        y=79,
        speed=1.75,
        horizon=8,
        gap_start=1920,
        gap_end=1952,
    )
    high = jump_reach_reference(
        base + [surface(1952, 1984, 143)],
        x=1892,
        y=79,
        speed=1.75,
        horizon=8,
        gap_start=1920,
        gap_end=1952,
    )
    assert (
        high["current_speed_jump"]["estimated_reach_pixels"]
        < low["current_speed_jump"]["estimated_reach_pixels"]
    )


def test_short_jump_does_not_authorize_walking_off_edge_or_through_wall():
    for surfaces in [
        [surface(1840, 1872, 111), surface(1952, 1984, 127)],
        [surface(1840, 1872, 111), surface(1872, 1920, 159), surface(1952, 1984, 127)],
    ]:
        plan = jump_reach_reference(
            surfaces, x=1864, y=111, speed=1.75, horizon=8, gap_start=1920, gap_end=1952
        )
        assert plan["approach_supported"] is False
        assert plan["approach_before_jumping"] is False


def test_missing_landing_does_not_invent_reachability():
    assert (
        jump_reach_reference(
            [surface(1872, 1920, 79)],
            x=1892,
            y=79,
            speed=1.75,
            horizon=8,
            gap_start=1920,
            gap_end=1952,
        )
        is None
    )


def landing_approach_snapshot():
    from dataclasses import replace

    from typesafe_mario.state import MarioStateParser

    ram = bytearray(0x800)
    ram[0x57], ram[0x1D] = 28, 1  # 1.75 px/frame, airborne.
    for left, right, row in [(1792, 1840, 11), (1840, 1872, 9), (1872, 1920, 11), (1952, 1984, 8)]:
        for tile in range(left // 16, right // 16):
            for solid_row in range(row, 13):
                ram[0x500 + ((tile // 16) % 2) * 208 + solid_row * 16 + tile % 16] = 0x54
    s = MarioStateParser().parse({"x_pos": 1836, "y_pos": 145, "y_pixel": 110}, ram)
    return replace(s, dx=2, dy=-5, frames_until_action=8, scheduled_action="right_jump")


def test_future_takeoff_after_landing_uses_fractional_speed_not_rounded_frame_delta():
    s = landing_approach_snapshot()
    plan = s.jump_reach_features()
    assert s.horizontal_speed_exact == 1.75
    assert plan["takeoff_x_estimate"] == 1850
    assert plan["takeoff_y_estimate"] == 111
    assert plan["approach_before_jumping"] is True
    assert s.to_state()["terrain"]["jump_reach_reference"] == plan


def test_does_not_replan_takeoff_before_landing_or_during_committed_ground_jump():
    from dataclasses import replace

    s = landing_approach_snapshot()
    assert replace(s, frames_until_action=4).jump_reach_features() is None
    assert replace(s, grounded=True, y=111).jump_reach_features() is None


def test_landing_prediction_also_uses_fractional_horizontal_speed():
    from dataclasses import replace

    # One-frame rounding to 1 px would incorrectly predict missing the step.
    s = replace(landing_approach_snapshot(), x=1821, dx=1)
    # Regenerate the collision window for this x (one tile earlier).
    rows = tuple("." + row[:-1] for row in s.collision_grid)
    s = replace(s, collision_grid=rows)
    plan = s.jump_reach_features()
    assert plan is not None
    assert plan["takeoff_x_estimate"] == 1835


def test_flat_floor_keeps_existing_gap_deadline_instead_of_step_approach_rule():
    from dataclasses import replace

    s = landing_approach_snapshot()
    s = replace(
        s,
        x=1880,
        y=79,
        grounded=True,
        frames_until_action=0,
        collision_grid=tuple(row[3:] + "..." for row in s.collision_grid),
    )
    assert s.jump_reach_features() is None


def test_overhead_platform_does_not_hide_a_reachable_lower_landing():
    base = [surface(1840, 1920, 79), surface(1952, 1984, 79)]
    args = {"x": 1870, "y": 79, "speed": 1.75, "horizon": 8, "gap_start": 1920, "gap_end": 1952}
    lower = jump_reach_reference(base, **args)
    with_overhead = jump_reach_reference(base + [surface(1952, 1984, 159)], **args)
    assert lower["current_speed_jump"]["can_reach"]
    assert with_overhead["current_speed_jump"]["can_reach"]


def test_short_lower_floor_has_no_room_to_land_and_replan_after_a_drop():
    plan = jump_reach_reference(
        [surface(1840, 1872, 143), surface(1872, 1904, 79), surface(1968, 2000, 159)],
        x=1860,
        y=143,
        speed=1.75,
        horizon=8,
        gap_start=1904,
        gap_end=1968,
    )
    assert not plan["approach_supported"]
    assert not plan["approach_before_jumping"]


def test_unseen_landing_can_prompt_safe_approach_without_inventing_a_distance():
    plan = jump_reach_reference(
        [surface(1840, 1872, 111), surface(1872, 1920, 79)],
        x=1840,
        y=111,
        speed=1.75,
        horizon=8,
        gap_start=1920,
        gap_end=None,
    )
    assert plan["landing_surface"] is None
    assert plan["required_distance_pixels"] is None
    assert plan["approach_before_jumping"]
