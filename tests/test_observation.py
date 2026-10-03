import json
from dataclasses import replace

from typesafe_mario.state import MarioStateParser
from typesafe_mario.tracking import EnemyMeasurement, EnemyTrack


def scene():
    s = MarioStateParser().parse({"x_pos": 232, "y_pos": 79, "left_x_pos": 232})
    actors = tuple(
        EnemyTrack(str(i), EnemyMeasurement(i, 6, "goomba", x, 176, 99), ((0, x, 176),), 0)
        for i, x in enumerate((210, 300))
    )
    return replace(s, enemy_tracks=actors, collision_grid=tuple(["." * 11] * 11 + ["#" * 11] * 2))


def test_observation_clips_tiles_and_actors_to_current_view():
    from typesafe_mario.observation import observe

    o = observe(scene())
    assert [a.x for a in o.actors] == [210]
    assert o.known_x == (192, 256)
    assert all(192 <= b.left < b.right <= 256 for b in o.blocks)


def test_hidden_engine_details_do_not_change_observations_or_provider_state():
    from typesafe_mario.observation import model_state, observe

    s = scene()
    altered = replace(
        s,
        horizontal_speed_exact=9876,
        enemy_tracks=tuple(
            replace(
                t,
                measurement=replace(
                    t.measurement, engine_state=444, plant={"estimated_frames_until_hidden": 12345}
                ),
            )
            for t in s.enemy_tracks
        ),
        prediction={"backend": "native_checkpoint", "secret_future": 56789},
    )
    assert observe(s) == observe(altered)
    assert model_state(s) == model_state(altered)
    text = json.dumps(model_state(altered))
    assert "engine_state" not in text and "estimated_frames_until_hidden" not in text
    assert "secret_future" not in text and "9876" not in text


def test_missing_viewport_does_not_claim_ram_geometry_or_actors_visible():
    from typesafe_mario.observation import observe

    o = observe(replace(scene(), camera_left_x=None))
    assert o.known_x is None and not o.blocks and not o.actors


def test_coordinate_discontinuity_is_not_an_enormous_velocity():
    from typesafe_mario.observation import observe

    o = observe(replace(scene(), dx=-65530))
    assert o.vx == 0
    assert not o.motion_reliable


def test_viewport_origin_accounts_for_camera_scrolling():
    s = MarioStateParser().parse({"x_pos": 2000, "left_x_pos": 112})
    assert s.camera_left_x == 1888


def test_plant_outside_observed_geometry_is_not_exposed_from_ram():
    from typesafe_mario.observation import observe

    s = scene()
    plant = EnemyTrack(
        "plant", EnemyMeasurement(0, 13, "piranha_plant", 100, 200, 0), ((0, 100, 200),), 0
    )
    assert not observe(replace(s, enemy_tracks=(plant,))).actors


def test_parser_exposes_visible_retreat_floor_without_offscreen_tiles():
    from typesafe_mario.observation import observe

    ram = bytearray(2048)
    # Both pages are solid; the observation must still clip to the camera.
    ram[0x500:0x6A0] = bytes([0x54]) * 416
    s = MarioStateParser().parse({"x_pos": 520, "y_pos": 79, "left_x_pos": 112}, ram)
    o = observe(s)
    assert o.known_x == (408, 664)
    assert any(b.left == 408 and b.right == 664 for b in o.blocks)
    assert all(408 <= b.left < b.right <= 664 for b in o.blocks)
