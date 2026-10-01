from dataclasses import replace

from typesafe_mario.state import EnemyObservation, MarioStateParser


def enemy_frame(parser, *, mario_screen=100, enemy_screen=50, active=1, kind=6):
    ram = bytearray(0x800)
    ram[0xF], ram[0x16], ram[0x87], ram[0xCF] = active, kind, 110, enemy_screen
    return parser.parse({"x_pos": 100, "y_pos": 255 - mario_screen, "y_pixel": mario_screen}, ram)


def test_enemy_fall_is_distinguished_from_mario_rising():
    p = MarioStateParser()
    first = enemy_frame(p)
    assert first.enemies[0].vertical_velocity_y is None
    rising_mario = enemy_frame(p, mario_screen=96)
    assert rising_mario.enemies[0].vertical_velocity_y == 0
    assert rising_mario.enemies[0].relative_velocity_y == 4
    falling_enemy = enemy_frame(p, mario_screen=92, enemy_screen=53)
    assert falling_enemy.enemies[0].vertical_velocity_y == 3
    assert falling_enemy.enemies[0].relative_velocity_y == 7


def test_enemy_motion_history_resets_for_inactive_or_reused_slots():
    p = MarioStateParser()
    enemy_frame(p)
    enemy_frame(p, active=0)
    assert enemy_frame(p, enemy_screen=140).enemies[0].vertical_velocity_y is None
    assert enemy_frame(p, kind=0, enemy_screen=40).enemies[0].vertical_velocity_y is None
    p.reset()
    assert enemy_frame(p).enemies[0].vertical_velocity_y is None


def test_screen_wrap_is_unknown_vertical_motion_not_a_fast_fall():
    p = MarioStateParser()
    enemy_frame(p, enemy_screen=250)
    assert enemy_frame(p, enemy_screen=2).enemies[0].vertical_velocity_y is None


def test_falling_enemy_above_and_slightly_behind_remains_a_model_threat():
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 130, "y_pixel": 125})
    s = replace(
        s,
        frames_until_action=8,
        grounded=False,
        dy=3,
        scheduled_action="right_jump",
        enemies=(
            EnemyObservation(
                1, 6, "goomba", -1, -44, 0, vertical_velocity_y=3, relative_velocity_y=6
            ),
        ),
    )
    threats = s.to_state()["hazard"]["overhead_threats"]
    assert len(threats) == 1
    assert threats[0]["slot"] == 1
    assert threats[0]["relative_x_pixels"] == -1
    assert threats[0]["falling"] is True


def test_elevated_enemy_at_platform_edge_warns_before_fall_starts():
    # Platform x=112..144, standing y=159. Enemy's center is near its left edge.
    ram = bytearray(0x800)
    for c in (7, 8):
        ram[0x500 + 6 * 16 + c] = 1
    for c in range(16):
        ram[0x500 + 11 * 16 + c] = 1
    s = MarioStateParser().parse({"x_pos": 80, "y_pos": 79, "y_pixel": 176}, ram)
    s = replace(
        s,
        frames_until_action=8,
        dx=3,
        scheduled_action="right_run_jump",
        enemies=(
            EnemyObservation(
                0, 6, "goomba", 32, -72, -3, vertical_velocity_y=0, relative_velocity_y=0
            ),
        ),
    )
    threats = s.to_state()["hazard"]["overhead_threats"]
    assert len(threats) == 1
    assert threats[0]["near_platform_edge"] is True
    assert threats[0]["falling"] is False
    red_koopa = replace(s.enemies[0], kind_id=3, kind="red_koopa")
    assert replace(s, enemies=(red_koopa,)).overhead_threat_features() == []
    assert (
        replace(s, enemies=(replace(red_koopa, vertical_velocity_y=2),)).overhead_threat_features()[
            0
        ]["falling"]
        is True
    )


def test_ground_enemies_and_distant_overhead_enemies_do_not_trigger_overhead_rule():
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 79})
    s = replace(
        s,
        enemies=(
            EnemyObservation(0, 6, "goomba", 20, 8, -3, vertical_velocity_y=0),
            EnemyObservation(1, 6, "goomba", 200, -40, -1, vertical_velocity_y=3),
        ),
    )
    assert s.to_state()["hazard"]["overhead_threats"] == []


def test_enemy_already_passed_before_application_does_not_cut_a_later_jump():
    s = MarioStateParser().parse({"x_pos": 100, "y_pos": 130})
    s = replace(
        s,
        frames_until_action=8,
        enemies=(EnemyObservation(0, 6, "goomba", 0, -60, -4, vertical_velocity_y=3),),
    )
    assert s.overhead_threat_features() == []


def test_enemy_projected_below_mario_before_application_is_not_an_overhead_escape():
    s = MarioStateParser().parse({"x_pos": 1200, "y_pos": 112})
    s = replace(
        s,
        frames_until_action=8,
        enemies=(
            EnemyObservation(
                0, 6, "goomba", 60, -51, -3, vertical_velocity_y=4, relative_velocity_y=9
            ),
        ),
    )
    assert s.overhead_threat_features() == []
