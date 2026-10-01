from dataclasses import replace

from typesafe_mario.state import MarioStateParser


def snapshot():
    return MarioStateParser().parse({"x_pos": 100, "y_pos": 80, "y_pixel": 80})


def test_separate_gaps_are_not_combined():
    s = replace(snapshot(), local_grid=("...........", "..M........", "###.#.#####"))
    assert s.navigation_features()["gap_width_tiles_visible"] == 1


def test_missing_support_is_unknown_not_clear():
    s = replace(snapshot(), local_grid=("...........", "..M........", "..........."))
    terrain = s.navigation_features()
    assert terrain["geometry_available"] is False
    assert terrain["clear_forward_tiles"] == 0


def test_grounded_preview_is_rebased_and_has_age():
    ram = bytearray(0x800)
    # Floor at screen y=208, two-tile gap starting at x=144.
    for c in range(16):
        if c not in (9, 10):
            ram[0x500 + 11 * 16 + c] = 1
    p = MarioStateParser()
    p.parse({"x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram)
    ram[0x1D] = 1
    s = p.parse({"x_pos": 116, "y_pos": 89, "y_pixel": 166}, ram)
    preview = s.to_state()["terrain"]["last_grounded_preview"]
    assert preview["gap_start_x"] == 144
    assert preview["gap_distance_pixels"] == 28
    assert preview["age_frames"] == 1
    assert s.gap_width_at_commit == 2
    # Unknown geometry at a greater height cannot expand the committed gap.
    s = p.parse({"x_pos": 132, "y_pos": 159, "y_pixel": 96}, ram)
    assert s.gap_width_at_commit == 2


def test_lower_floor_is_a_landing_surface_not_a_pit():
    ram = bytearray(0x800)
    for c in range(16):
        ram[0x500 + 11 * 16 + c] = 1
    # Mario stands on a pipe above continuous ground.
    for row in (7, 8, 9, 10):
        ram[0x500 + row * 16 + 6] = 1
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 143, "y_pixel": 112}, ram)
    terrain = s.to_state()["terrain"]
    assert terrain["gap_distance_tiles"] is None
    assert any(surface["standing_y"] == 79 for surface in terrain["landing_surfaces"])


def test_airborne_ram_state_overrides_nearby_support():
    ram = bytearray(0x800)
    ram[0x1D] = 1
    for c in range(16):
        ram[0x500 + 11 * 16 + c] = 1
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram)
    assert not s.grounded
    assert s.to_state()["player"]["can_start_jump"] is False


def test_expired_enemy_slot_has_no_spurious_velocity():
    p = MarioStateParser()
    ram = bytearray(0x800)
    ram[0xF], ram[0x16], ram[0x87] = 1, 6, 140
    p.parse({"x_pos": 100}, ram)
    ram[0xF] = 0
    p.parse({"x_pos": 102}, ram)
    ram[0xF], ram[0x87] = 1, 240
    s = p.parse({"x_pos": 104}, ram)
    assert s.enemies[0].relative_velocity_x == 0


def test_vertical_separation_does_not_force_grounded_takeoff():
    from typesafe_mario.state import EnemyObservation

    s = replace(snapshot(), grounded=True, enemies=(EnemyObservation(0, 6, "goomba", 48, -80, -4),))
    h = s.threat_features()
    assert h["jump_must_start_this_decision"] is False
    assert h["contact_within_reaction_horizon"] is False


def test_descending_landing_uses_visible_platform_not_fixed_airtime():
    from typesafe_mario.terrain import landing_projection

    surface = {
        "start_x": 96,
        "end_x": 144,
        "safe_center_min_x": 104,
        "safe_center_max_x": 136,
        "standing_y": 79,
    }
    result = landing_projection([surface], x=100, y=94, dx=2, dy=-5, grounded=False)
    assert result["frames"] == 3
    assert result["x"] == 106
    assert result["within_safe_surface"] is True
    assert result["confidence"] == "approximate"
    missed = landing_projection([surface], x=140, y=94, dx=2, dy=-5, grounded=False)
    assert missed["within_safe_surface"] is False
    rising = landing_projection([surface], x=100, y=94, dx=2, dy=3, grounded=False)
    assert rising["confidence"] == "unknown"
    assert rising["frames"] is None


def test_coins_hidden_blocks_and_climbables_are_not_landing_surfaces():
    ram = bytearray(0x800)
    for column, tile in enumerate((0xC2, 0xC3, 0x5F, 0x60, 0x24), start=6):
        ram[0x500 + 11 * 16 + column] = tile
    terrain = (
        MarioStateParser()
        .parse({"x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram)
        .to_state()["terrain"]
    )
    assert terrain["landing_surfaces"] == []
    assert terrain["geometry_available"] is False


def test_elevated_enemy_cannot_hide_an_imminent_same_level_enemy():
    from typesafe_mario.state import EnemyObservation

    s = replace(
        snapshot(),
        grounded=True,
        enemies=(
            EnemyObservation(0, 6, "goomba", 10, -80, -4),
            EnemyObservation(1, 6, "goomba", 48, 8, -4),
        ),
    )
    hazard = s.threat_features()
    assert hazard["nearest_enemy_distance_pixels"] == 48
    assert hazard["contact_within_reaction_horizon"] is True
    assert hazard["jump_must_start_this_decision"] is True
    elevated = replace(s, enemies=s.enemies[:1]).threat_features()
    assert elevated["takeoff_window_already_missed"] is False
    assert elevated["estimated_contact_frames"] is None


def test_horizontal_overlap_is_urgent_even_without_a_velocity_history():
    from typesafe_mario.state import EnemyObservation

    s = replace(snapshot(), grounded=True, enemies=(EnemyObservation(0, 6, "goomba", 8, 8),))
    assert s.threat_features()["contact_within_reaction_horizon"] is True


def test_ram_confirmed_landing_takes_precedence_over_last_frame_descent():
    ram = bytearray(0x800)
    for column in range(16):
        ram[0x500 + 11 * 16 + column] = 1
    p = MarioStateParser()
    ram[0x1D] = 2
    p.parse({"x_pos": 100, "y_pos": 84, "y_pixel": 171}, ram, previous_action="right_jump")
    ram[0x1D] = 0
    landed = p.parse({"x_pos": 102, "y_pos": 79, "y_pixel": 176}, ram, previous_action="right_jump")
    assert landed.grounded
    assert landed.to_state()["player"]["jump_requires_release"]


def test_landing_safe_interval_uses_player_center_not_sprite_origin():
    from typesafe_mario.terrain import landing_projection

    surface = {
        "start_x": 96,
        "end_x": 144,
        "safe_center_min_x": 104,
        "safe_center_max_x": 136,
        "standing_y": 79,
    }
    left = landing_projection([surface], x=96, y=84, dx=0, dy=-5, grounded=False)
    right = landing_projection([surface], x=136, y=84, dx=0, dy=-5, grounded=False)
    assert left["within_safe_surface"] is True
    assert right["within_safe_surface"] is False


def test_grounding_uses_ram_when_only_the_right_foot_has_support():
    ram = bytearray(0x800)
    ram[0x500 + 11 * 16 + 7] = 0x54
    s = MarioStateParser().parse({"x_pos": 110, "y_pos": 79, "y_pixel": 176}, ram)
    assert s.grounded
    assert s.to_state()["terrain"]["geometry_available"] is True


def test_active_zero_type_enemy_is_a_koopa_not_an_empty_slot():
    ram = bytearray(0x800)
    ram[0xF], ram[0x16], ram[0x87], ram[0xCF] = 1, 0, 140, 184
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram)
    assert len(s.enemies) == 1
    assert s.enemies[0].kind == "green_koopa"
    assert s.threat_features()["enemy_ahead"] is True


def test_slow_ascent_is_still_rising_and_does_not_suggest_jump_release():
    ram = bytearray(0x800)
    ram[0x1D] = 1
    p = MarioStateParser()
    p.parse({"x_pos": 100, "y_pos": 130, "y_pixel": 125}, ram)
    s = p.parse({"x_pos": 102, "y_pos": 131, "y_pixel": 124}, ram, previous_action="right_jump")
    assert s.jump_phase == "rising"


def test_previous_progress_does_not_hide_a_current_stall():
    s = replace(
        snapshot(),
        grounded=True,
        previous_action="right_run",
        action_frames=50,
        action_progress=100,
        stalled_steps=12,
    )
    assert s.control_outcome() == "blocked"


def test_falling_jump_predicts_collision_with_second_enemy_at_landing():
    from typesafe_mario.state import EnemyObservation

    ram = bytearray(0x800)
    for c in range(16):
        ram[0x500 + 11 * 16 + c] = 0x54
    ram[0x1D] = 1
    base = MarioStateParser().parse({"x_pos": 150, "y_pos": 132, "y_pixel": 123}, ram)
    s = replace(
        base,
        dx=3,
        dy=-5,
        grounded=False,
        enemies=(
            EnemyObservation(0, 6, "goomba", 23, 61, -4),
            EnemyObservation(1, 6, "goomba", 46, 61, -4),
        ),
    )
    hazard = s.threat_features()
    assert hazard["vertical_overlap_now"] is False
    assert hazard["landing_collision_predicted"] is True
    assert hazard["landing_braking_supported"] is True
    assert hazard["landing_threats"][0]["projected_distance_pixels"] == 2
    assert hazard["landing_threats"][0]["slot"] == 1
    elevated = replace(s, enemies=(replace(s.enemies[1], dy_pixels=-80),))
    assert elevated.threat_features()["landing_collision_predicted"] is False


def test_landing_risk_distinguishes_a_rear_enemy_from_one_ahead():
    from typesafe_mario.state import EnemyObservation

    ram = bytearray(0x800)
    for c in range(16):
        ram[0x500 + 11 * 16 + c] = 0x54
    ram[0x1D] = 1
    s = MarioStateParser().parse({"x_pos": 150, "y_pos": 105, "y_pixel": 150}, ram)
    s = replace(s, dx=1, dy=-5, enemies=(EnemyObservation(2, 6, "goomba", -12, 34, 0),))
    hazard = s.threat_features()
    assert hazard["landing_collision_predicted"]
    assert hazard["landing_threat_from_behind"]
    assert hazard["landing_threats"][0]["current_distance_pixels"] == -12
    landed = replace(s, grounded=True, y=79, enemies=(replace(s.enemies[0], dy_pixels=8),))
    assert landed.threat_features()["rear_contact_imminent"]


def test_short_step_before_gap_remains_a_target_until_landing():
    ram = bytearray(0x800)
    for c in range(16):
        if c not in (9, 10):
            ram[0x500 + 11 * 16 + c] = 0x54
    for c in (7, 8):
        ram[0x500 + 10 * 16 + c] = 0x54
    p = MarioStateParser()
    grounded = p.parse({"x_pos": 100, "y_pos": 79, "y_pixel": 176}, ram)
    target = grounded.to_state()["trajectory"]["precision_landing_target"]
    assert target["start_x"] == 112
    assert target["standing_y"] == 95
    ram[0x1D] = 1
    ascending = p.parse(
        {"x_pos": 105, "y_pos": 110, "y_pixel": 145}, ram, previous_action="right_jump"
    )
    assert ascending.to_state()["trajectory"]["precision_target_cleared"] is True
    assert ascending.to_state()["trajectory"]["precision_landing_target"] == target
    ram[0x1D] = 0
    landed = p.parse({"x_pos": 120, "y_pos": 95, "y_pixel": 160}, ram)
    assert landed.to_state()["trajectory"]["precision_landing_target"] is None


def test_precision_target_stops_applying_after_mario_passes_it():
    target = {"start_x": 112, "end_x": 144, "safe_center_max_x": 136, "standing_y": 95}
    s = replace(snapshot(), grounded=False, x=151, y=110, precision_landing_target=target)
    trajectory = s.to_state()["trajectory"]
    assert not trajectory["precision_target_cleared"]
    assert trajectory["precision_target_missed"]


def test_running_approach_selects_top_step_before_visible_gap():
    ram = bytearray(0x800)
    # A four-step staircase, with a gap starting 114 pixels ahead.
    for c in range(16):
        if c not in (9, 10):
            ram[0x500 + 11 * 16 + c] = 0x54
    for c, top in ((4, 10), (5, 9), (6, 8), (7, 7), (8, 7)):
        for row in range(top, 11):
            ram[0x500 + row * 16 + c] = 0x54
    p = MarioStateParser()
    s = p.parse({"x_pos": 30, "y_pos": 79, "y_pixel": 176}, ram)
    target = s.precision_landing_target
    assert target["start_x"] == 112
    assert target["standing_y"] == 143
    ram[0x1D] = 1
    s = p.parse({"x_pos": 78, "y_pos": 147, "y_pixel": 108}, ram)
    assert s.to_state()["trajectory"]["precision_target_cleared"]


def test_reaction_window_uses_planned_frames_not_previous_network_delay():
    from typesafe_mario.state import EnemyObservation

    s = replace(
        snapshot(),
        grounded=True,
        frames_until_action=8,
        last_response_delay_frames=0,
        enemies=(EnemyObservation(0, 6, "goomba", 76, 8, -3),),
    )
    assert s.threat_features()["takeoff_deadline_frames"] == 4
    assert s.threat_features()["jump_must_start_this_decision"]
    later = replace(s, last_response_delay_frames=30)
    assert later.threat_features() == s.threat_features()
    assert later.to_state()["reaction_timing"]["total_reaction_horizon_frames"] == 16
