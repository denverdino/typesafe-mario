from dataclasses import replace
from itertools import pairwise

from prediction_helpers import assess_forecast

from typesafe_mario.actions import Action
from typesafe_mario.observation import Actor, Block, Observation


def floor_scene(**changes):
    base = Observation(
        0,
        (1, 1, 1),
        80,
        79,
        1,
        0,
        True,
        False,
        "right",
        True,
        (0, 256),
        (Block(0, 256, 63, 79),),
        (),
    )
    return replace(base, **changes)


def forecast(o=None, actions=tuple(Action), **kwargs):
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = o or floor_scene()
    p.observe(o)
    return p, p.forecast(o, actions, cycle=8, **kwargs)


def test_candidates_are_observation_only_estimates_with_no_safety_promises():
    _, result = forecast()
    assert result["backend"] == "observation_dynamics"
    assert len(result["action_forecasts"]) == 7
    by_action = {f["action"]: f for f in result["action_forecasts"]}
    assert by_action["right"]["first_cycle"]["x"] > by_action["left"]["first_cycle"]["x"]
    assert by_action["right_jump"]["first_cycle"]["y"] > 79
    assert all(f["risk"] not in ("safe", "unsafe") for f in by_action.values())
    assert all(f["confidence"] in ("low", "estimated") for f in by_action.values())
    assert all(
        f["first_cycle"]["x_range"][0] <= f["first_cycle"]["x"] <= f["first_cycle"]["x_range"][1]
        for f in by_action.values()
    )


def test_committed_input_is_predicted_before_candidate_and_not_changeable():
    _, result = forecast(delay=8, scheduled_action=Action.RIGHT)
    assert result["committed_future"]["x"] > 80
    assert result["apply_at_frame"] == 8
    assert all(f["first_cycle"]["frame"] == 16 for f in result["action_forecasts"])


def test_unknown_geometry_and_unseen_future_remain_unknown():
    _, result = forecast(floor_scene(known_x=None, blocks=()))
    assert all(f["risk"] == "unknown" for f in result["action_forecasts"])
    assert all("unobserved_terrain" in f["unknowns"] for f in result["action_forecasts"])


def test_visible_wall_blocks_nominal_horizontal_motion():
    o = floor_scene(blocks=(Block(0, 256, 63, 79), Block(112, 128, 79, 143)))
    _, r = forecast(o, (Action.RIGHT_RUN,))
    assert r["action_forecasts"][0]["end"]["x"] <= 98
    assert "wall_contact" in r["action_forecasts"][0]["events"]


def test_falling_lands_on_observed_floor_and_ceiling_limits_jump():
    _, r = forecast(floor_scene(y=95, vy=-3, grounded=False), (Action.NOOP,))
    assert r["action_forecasts"][0]["first_cycle"]["y"] == 79
    o = floor_scene(blocks=(Block(0, 256, 63, 79), Block(64, 128, 111, 127)))
    _, r = forecast(o, (Action.JUMP,))
    assert "ceiling_contact" in r["action_forecasts"][0]["events"]
    assert max(s["y"] for s in r["action_forecasts"][0]["trajectory"]) <= 95


def test_water_strokes_and_release_have_different_depth_estimates():
    _, r = forecast(floor_scene(y=150, swimming=True, grounded=False), (Action.JUMP, Action.NOOP))
    assert (
        r["action_forecasts"][0]["first_cycle"]["y"] > r["action_forecasts"][1]["first_cycle"]["y"]
    )
    assert "swimming_dynamics" in r["action_forecasts"][0]["unknowns"]


def test_new_enemy_has_uncertain_motion_and_cannot_be_certified_clear():
    _, r = forecast(floor_scene(actors=(Actor("0:goomba", "goomba", 110, 79),)), (Action.RIGHT,))
    f = r["action_forecasts"][0]
    assert f["hazards"] and f["hazards"][0]["kind"] == "goomba"
    assert "enemy_motion_unknown" in f["unknowns"]


def test_provider_evidence_uses_actual_motion_and_separate_visible_enemy_tracks():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    for frame in range(5):
        o = floor_scene(
            frame=frame,
            x=80 + frame,
            actors=(
                Actor("g", "goomba", 160 - frame, 71),
                Actor("k", "green_koopa", 220 + frame, 71),
            ),
        )
        p.observe(o)
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    motions = r["recent_motion"]
    assert motions[-1]["action"] == "right"
    assert motions[-1]["start"] == {"x": 80, "y": 79}
    assert motions[-1]["end"]["x"] == 84
    assert motions[-1]["end_frame"] == 4
    actors = {a["id"]: a for a in r["actor_forecasts"]}
    assert actors["g"]["observed_velocity"]["x"] == -1
    assert actors["k"]["observed_velocity"]["x"] == 1
    assert actors["g"]["trajectory"][0]["frame"] == 8
    assert actors["g"]["trajectory"][0]["x"] == 148
    assert len(actors["g"]["trajectory"]) <= 3
    # No hypothetical motion may be written into the next request's history.
    assert p.forecast(o, tuple(Action), cycle=8)["recent_motion"] == motions


def test_learning_uses_real_observations_only_and_forecasts_do_not_train():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    for frame in range(6):
        p.observe(floor_scene(frame=frame, x=80 + 0.1 * frame * frame, vx=0.2 * frame))
    before = p.calibration()
    assert before["samples"] > 0
    o = floor_scene(frame=5, x=82.5, vx=1)
    p.forecast(o, tuple(Action), cycle=8)
    p.forecast(o, tuple(Action), cycle=8)
    p.observe(o)
    assert p.calibration() == before
    p.observe(replace(o, frame=6, x=65535))
    assert p.calibration()["samples"] == 0


def test_observed_jump_calibrates_impulse_but_collision_does_not_calibrate_acceleration():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    p.observe(floor_scene(vx=2))
    p.observe(floor_scene(frame=1, x=80, vx=0))
    assert p.calibration()["samples"] == 0
    p.observe(floor_scene(frame=2, x=80, vx=0, y=84, vy=5, grounded=False, previous_action="jump"))
    assert p.calibration()["parameters"]["jump_speed"]["samples"] == 1


def test_only_executed_candidate_is_validated_and_restart_drops_pending():
    p, r = forecast(actions=(Action.RIGHT, Action.LEFT))
    p.expect(r, Action.RIGHT, apply_at_frame=0)
    for frame in range(1, 9):
        p.observe(floor_scene(frame=frame, x=80 + frame, previous_action="right"))
    assert p.validation()["samples"] == 1
    assert p.validation()["mean_absolute_x_error"] >= 0
    # An intervening different action invalidates the comparison.
    r = p.forecast(floor_scene(frame=8, x=88), (Action.RIGHT,), cycle=8)
    p.expect(r, Action.RIGHT, apply_at_frame=8)
    for frame in range(9, 17):
        p.observe(floor_scene(frame=frame, x=80 + frame, previous_action="left"))
    assert p.validation()["samples"] == 1


def test_pixel_quantized_motion_can_calibrate_acceleration():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    last_x = 80
    for frame in range(9):
        x = 80 + round(0.1 * frame * frame)
        p.observe(floor_scene(frame=frame, x=x, vx=x - last_x, previous_action="right_run"))
        last_x = x
    assert p.calibration()["parameters"]["run_accel"]["samples"] > 0


def test_future_history_is_not_available_to_historical_prediction():
    import pytest

    from typesafe_mario.prediction import Predictor

    p = Predictor()
    p.observe(floor_scene(frame=10))
    with pytest.raises(ValueError, match="historical"):
        p.forecast(floor_scene(frame=0), tuple(Action), cycle=8)


def test_calibration_does_not_discard_zero_quantized_gravity_samples():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    last_y = 150
    for frame in range(17):
        y = 150 + round(3 * frame - 0.09 * frame * frame)
        p.observe(
            floor_scene(
                frame=frame,
                x=80 + frame,
                y=y,
                vy=y - last_y,
                grounded=False,
                previous_action="right_jump",
            )
        )
        last_y = y
    estimate = p.calibration()["parameters"]["gravity_hold"]["estimate"]
    assert 0.1 <= estimate <= 0.26


def test_zero_acceleration_measurements_are_retained_without_bias():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    for value in [0, 1 / 3] * 32:
        model.add("gravity_hold", value)
    assert 0.16 <= model.value("gravity_hold") <= 0.2
    assert model.summary()["parameters"]["gravity_hold"]["samples"] == 64


def test_integer_velocity_fluctuations_do_not_bias_gravity_upward():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    history = []
    for frame in range(41):
        a = floor_scene(
            frame=frame,
            y=150 + round(4.5 * frame),
            vy=4 if frame % 2 else 5,
            grounded=False,
            previous_action="right_jump",
        )
        history.append(a)
        model.learn_held_gravity(history, fast_jump=False)
    assert model.summary()["parameters"]["gravity_hold"]["samples"] > 20
    assert model.value("gravity_hold") < 0.1


def test_quantized_release_descent_does_not_bias_gravity_downward():
    from typesafe_mario.prediction import Predictor

    # Observed release segment from a real run, translated away from its level
    # coordinates. Pixel deltas briefly decrease by two without any collision.
    heights = [147, 149, 152, 153, 154, 154, 154, 152, 150, 146, 142, 138, 133]
    p = Predictor()
    for frame, y in enumerate(heights):
        p.observe(
            floor_scene(
                frame=frame,
                x=80 + 3 * frame,
                y=y,
                vy=y - heights[frame - 1] if frame else 3,
                grounded=False,
                previous_action="noop" if frame else "right_run_jump",
            )
        )
    gravity = p.calibration()["parameters"]["gravity_fall"]["estimate"]
    assert 0.57 <= gravity <= 0.65


def test_visible_walker_outside_known_tiles_is_not_predicted_to_fall_into_unknown_void():
    from typesafe_mario.actor_motion import actor_path

    o = floor_scene(known_x=(0, 160), blocks=(Block(0, 160, 63, 79),))
    actor = Actor("g", "goomba", 160, 71)
    path = actor_path(actor, (-0.625, 0, True), o, 32)
    assert all(y == 71 for _, y, _ in path)
    assert not path[1][2]


def test_fast_and_stationary_takeoffs_use_distinct_observed_impulses():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    model = Dynamics()
    for frame in range(20):
        a = floor_scene(frame=frame * 2, vx=3, previous_action="right_run")
        model.learn(
            a,
            replace(
                a, frame=a.frame + 1, y=85, vy=6, grounded=False, previous_action="right_run_jump"
            ),
        )
    assert model.value("jump_speed") == 5
    assert model.value("jump_speed_fast") >= 5.8
    o = floor_scene()
    slow, _ = _advance(Motion(80, 79, 0, 0, True, "noop"), Action.RIGHT_RUN_JUMP, o, model)
    fast, _ = _advance(Motion(80, 79, 3, 0, True, "right_run"), Action.RIGHT_RUN_JUMP, o, model)
    slow, _ = _advance(slow, Action.RIGHT_RUN_JUMP, o, model)
    fast, _ = _advance(fast, Action.RIGHT_RUN_JUMP, o, model)
    assert fast.vy > slow.vy + 0.5


def test_running_at_air_speed_limit_does_not_learn_zero_acceleration():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    history = []
    for frame in range(64):
        a = floor_scene(
            frame=frame,
            x=80 + 3 * frame,
            y=150,
            vx=3,
            vy=2,
            grounded=False,
            previous_action="right_run_jump",
        )
        history.append(a)
        model.learn_air(history)
    assert model.summary()["parameters"]["air_accel"]["samples"] == 0


def test_observed_wall_contact_does_not_shift_player_back_two_pixels():
    from typesafe_mario.prediction import Predictor

    o = floor_scene(
        x=594, vx=0, blocks=(Block(0, 800, 63, 79), Block(608, 640, 79, 127)), known_x=(560, 736)
    )
    r = Predictor().forecast(o, (Action.RIGHT,), cycle=8)
    assert r["action_forecasts"][0]["first_cycle"]["x"] == 594


def test_only_observed_scored_goomba_stomp_removes_a_lingering_hazard():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(
        x=100,
        y=83,
        vx=0,
        vy=-6,
        grounded=False,
        previous_action="right_jump",
        actors=(Actor("g", "goomba", 100, 71), Actor("k", "green_koopa", 180, 71)),
    )
    p.observe(o)
    bounced = replace(o, frame=1, y=87, vy=4)
    p.observe(bounced)
    before = p.forecast(bounced, tuple(Action), cycle=8)
    assert before["observed_interactions"] == []
    assert p.calibration()["parameters"]["stomp_speed"]["samples"] == 0
    confirmed = replace(bounced, frame=2, y=90, vy=3, score=100)
    p.observe(confirmed)
    assert list(p.dynamics.samples["stomp_speed"]) == [4]
    after = p.forecast(confirmed, tuple(Action), cycle=8)
    assert after["observed_interactions"] == [
        {
            "actor_id": "g",
            "inference": "defeated_goomba",
            "evidence": "observed_bounce_stationary_actor_score_gain",
        }
    ]
    assert all(h["id"] != "g" for b in after["action_forecasts"] for h in b["hazards"])
    moved = replace(confirmed, frame=3, actors=(Actor("g", "goomba", 104, 71),))
    p.observe(moved)
    assert list(p.dynamics.samples["stomp_speed"]) == [4]
    assert p.forecast(moved, tuple(Action), cycle=8)["observed_interactions"] == []


def test_score_gain_without_bounce_cannot_remove_a_live_enemy():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(actors=(Actor("g", "goomba", 110, 71),))
    p.observe(o)
    o = replace(o, frame=1, score=100)
    p.observe(o)
    r = p.forecast(o, tuple(Action), cycle=8)
    assert r["observed_interactions"] == []


def test_bounce_near_overlapping_koopa_cannot_establish_goomba_defeat():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(
        x=100,
        y=83,
        vy=-6,
        grounded=False,
        actors=(Actor("g", "goomba", 100, 71), Actor("k", "green_koopa", 104, 71)),
    )
    p.observe(o)
    p.observe(replace(o, frame=1, y=87, vy=4))
    o = replace(o, frame=2, y=90, vy=3, score=100)
    p.observe(o)
    result = p.forecast(o, tuple(Action), cycle=8)
    assert result["observed_interactions"] == []
    assert {h["id"] for b in result["action_forecasts"] for h in b["hazards"]} == {"g", "k"}


def test_confirmed_defeat_does_not_survive_an_unobserved_slot_reuse():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(x=100, y=83, vy=-6, grounded=False, actors=(Actor("g", "goomba", 100, 71),))
    p.observe(o)
    p.observe(replace(o, frame=1, y=87, vy=4))
    o = replace(o, frame=2, y=90, vy=3, score=100)
    p.observe(o)
    assert p.forecast(o, tuple(Action), cycle=8)["observed_interactions"]
    o = replace(o, frame=200)
    p.observe(o)
    result = p.forecast(o, tuple(Action), cycle=8)
    assert not result["observed_interactions"]
    assert any(h["id"] == "g" for b in result["action_forecasts"] for h in b["hazards"])


def test_rearming_jump_can_leave_the_last_support_pixel_before_takeoff():
    # Actual from-reset run: x1098,y79 -> release A -> x1100,y79 ->
    # repress A -> x1102,y79. No upward impulse occurred at this edge.
    from typesafe_mario.actions import first_frame_action
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    o = floor_scene(
        x=1098,
        vx=1.75,
        previous_action="right_jump",
        blocks=(Block(1040, 1104, 63, 79), Block(1136, 1216, 63, 79)),
    )
    m = Motion(o.x, o.y, o.vx, o.vy, o.grounded, o.previous_action)
    release = first_frame_action(
        Action.RIGHT_JUMP, grounded=True, previous_action=m.previous_action
    )
    m, _ = _advance(m, release, o, Dynamics())
    m, _ = _advance(m, Action.RIGHT_JUMP, o, Dynamics())
    assert m.y <= 79


def test_pressing_run_during_a_walking_flight_does_not_train_zero_acceleration():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    # Observed long flight: B is held, but speed stays at the walking limit.
    # Its zero delta does not identify acceleration below the limit.
    a = floor_scene(y=140, vx=1.75, vy=1, grounded=False, previous_action="right_run_jump")
    history = [a]
    for frame in range(64):
        b = replace(a, frame=a.frame + 1, x=a.x + 1.75)
        model.learn(a, b)
        history.append(b)
        model.learn_air(history)
        a = b
    assert model.summary()["parameters"]["air_accel"]["samples"] == 0


def test_fast_takeoff_gravity_does_not_reduce_estimated_standing_jump_height():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(x=80, vx=3, previous_action="right_run")
    p.observe(o)
    for frame, y in enumerate((85, 91, 97, 102, 107, 112, 117, 121, 126, 130, 133), 1):
        o = replace(
            o,
            frame=frame,
            x=o.x + 3,
            y=y,
            vy=y - o.y,
            grounded=False,
            previous_action="right_run_jump",
        )
        p.observe(o)
    params = p.calibration()["parameters"]
    assert params["gravity_hold"]["samples"] == 0
    assert params["gravity_hold_fast"]["samples"] > 0


def test_run_pressed_after_walking_takeoff_cannot_invent_extra_air_speed():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    o = floor_scene(vx=1.75, previous_action="right")
    m = Motion(o.x, o.y, o.vx, 0, True, "right")
    m, _ = _advance(m, Action.RIGHT_JUMP, o, Dynamics())
    m, _ = _advance(m, Action.RIGHT_JUMP, o, Dynamics())
    assert not m.grounded
    start_x = m.x
    for _ in range(8):
        m, _ = _advance(m, Action.RIGHT_RUN_JUMP, o, Dynamics())
    assert m.x - start_x == 14
    assert m.vx == 1.75


def test_unseen_takeoff_does_not_identify_jump_gravity_class():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(vx=3)
    p.observe(o)
    o = replace(o, frame=100, y=140, vx=0, vy=4, grounded=False, previous_action="right_run_jump")
    p.observe(o)
    p.observe(replace(o, frame=101, y=143.5, vy=3.5))
    params = p.calibration()["parameters"]
    assert params["gravity_hold_fast"]["samples"] == 0
    assert params["gravity_hold"]["samples"] == 0


def test_landing_then_walking_off_does_not_reuse_a_previous_fast_flight():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, Predictor, _advance

    o = floor_scene(x=80, vx=3, previous_action="right_run")
    p = Predictor()
    p.observe(o)
    p.observe(replace(o, frame=1, y=85, vy=6, grounded=False, previous_action="right_run_jump"))
    p.observe(replace(o, frame=2, y=79, vy=-6))
    p.observe(replace(o, frame=3, vx=1.75, previous_action="right"))
    o = replace(
        o, frame=4, y=78, vy=-1, vx=1.75, grounded=False, blocks=(), previous_action="right_run"
    )
    p.observe(o)
    result = p.forecast(o, (Action.RIGHT_RUN,), cycle=8)
    assert result["action_forecasts"][0]["first_cycle"]["vx"] == 1.75

    floor = floor_scene(x=100, blocks=(Block(0, 112, 63, 79),))
    m = Motion(100, 80, 0, -2, False, "noop", True)
    m, _ = _advance(m, Action.NOOP, floor, Dynamics())
    assert m.grounded
    m.vx = 1.75
    for _ in range(8):
        m, _ = _advance(m, Action.RIGHT, floor, Dynamics())
    assert not m.grounded
    m, _ = _advance(m, Action.RIGHT_RUN, floor, Dynamics())
    assert m.vx == 1.75


def test_flush_ceiling_stop_does_not_train_gravity():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    previous_y = 88
    for frame, y in enumerate([90, 92, 93, 94, 95]):
        p.observe(
            floor_scene(
                frame=frame,
                y=y,
                vy=y - previous_y,
                grounded=False,
                previous_action="right_jump",
                blocks=(Block(64, 128, 111, 127),),
            )
        )
        previous_y = y
    before = p.calibration()
    p.observe(
        floor_scene(
            frame=5,
            y=95,
            vy=0,
            grounded=False,
            previous_action="right_jump",
            blocks=(Block(64, 128, 111, 127),),
        )
    )
    assert p.calibration() == before


def test_unreliable_motion_invalidates_pending_validation():
    p, r = forecast(actions=(Action.RIGHT,))
    p.expect(r, Action.RIGHT, apply_at_frame=0)
    for frame in range(1, 9):
        p.observe(floor_scene(frame=frame, x=100 + frame, motion_reliable=frame != 1))
    assert p.validation()["samples"] == 0


def test_rising_observation_partly_inside_ceiling_cannot_predict_flying_through_it():
    # Log 083125: frame 88, x261/y117, rising 5 px/frame under the first brick.
    o = floor_scene(
        x=261,
        y=117,
        vx=3,
        vy=5,
        grounded=False,
        previous_action="right_run_jump",
        known_x=(224, 400),
        blocks=(Block(256, 272, 127, 143), Block(320, 400, 127, 143), Block(224, 400, 63, 79)),
        actors=(Actor("0:goomba", "goomba", 330, 71),),
    )
    _, r = forecast(o, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)
    assert r["committed_future"]["y"] < 117
    branch = next(f for f in r["action_forecasts"] if f["action"] == "right_run_jump")
    assert "ceiling_contact" in branch["events"]
    assert branch["risk"] == "possible_contact"


def test_partial_pipe_overlap_blocks_walking_but_allows_jumping_over_it():
    # Real pipe contact at x434 overlaps the approximate x..x+16 body by 2 px.
    o = floor_scene(
        x=434,
        vx=0,
        known_x=(400, 576),
        blocks=(Block(448, 480, 95, 111), Block(448, 480, 79, 95), Block(400, 576, 63, 79)),
    )
    _, result = forecast(o, (Action.RIGHT, Action.RIGHT_RUN_JUMP))
    walk, jump = result["action_forecasts"]
    assert walk["end"]["x"] <= 434
    assert "wall_contact" in walk["events"]
    assert max(point["y"] for point in jump["trajectory"]) > 111
    assert jump["end"]["x"] > 434


def test_enemy_leaving_visible_platform_can_fall_into_player_path():
    from typesafe_mario.prediction import Predictor

    o = floor_scene(
        frame=1,
        x=100,
        vx=1,
        blocks=(Block(0, 256, 63, 79), Block(160, 224, 151, 167)),
        actors=(Actor("0:goomba", "goomba", 150, 159),),
    )
    p = Predictor()
    p.observe(replace(o, frame=0, actors=(Actor("0:goomba", "goomba", 151, 159),)))
    p.observe(o)
    result = p.forecast(o, (Action.RIGHT,), cycle=8)
    assert result["action_forecasts"][0]["control_risk"]["possible_contact"]


def test_forecast_velocity_uses_past_pixel_motion_without_rounding_up_whole_jump():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    last_x = 100
    for frame, x in enumerate((100, 101, 102, 103, 105)):
        o = floor_scene(
            frame=frame, x=x, vx=x - last_x, grounded=False, y=150, vy=0, previous_action="noop"
        )
        p.observe(o)
        last_x = x
    result = p.forecast(o, (Action.NOOP,), cycle=8)
    assert result["committed_future"]["vx"] == 1.25
    assert result["action_forecasts"][0]["first_cycle"]["x"] == 115


def test_tiny_edge_overlap_is_not_a_confident_landing():
    o = floor_scene(
        x=1122,
        y=82,
        vx=2,
        vy=-5,
        grounded=False,
        known_x=(1072, 1248),
        blocks=(Block(1136, 1248, 63, 79),),
    )
    _, result = forecast(o, (Action.RIGHT,))
    assert result["action_forecasts"][0]["control_risk"]["uncertain_landing"]


def test_airborne_reverse_input_cannot_use_ground_braking_strength():
    _, r = forecast(floor_scene(vx=3, y=130, vy=2, grounded=False), (Action.LEFT,))
    assert r["action_forecasts"][0]["first_cycle"]["vx"] > 2


def test_air_reverse_learning_does_not_contaminate_ground_braking():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    a = floor_scene(vx=2, y=130, vy=2, grounded=False, previous_action="left")
    b = replace(a, frame=1, vx=1.95, vy=1.5)
    model.learn(a, b)
    model.learn_air([replace(a, frame=f, x=a.x + 2 * f - 0.025 * f * (f + 1)) for f in range(9)])
    assert model.summary()["parameters"]["brake_accel"]["samples"] == 0
    assert model.summary()["parameters"]["air_accel"]["samples"] == 1


def test_delayed_observed_takeoff_can_calibrate_held_jump_impulse():
    from typesafe_mario.dynamics import Dynamics

    model = Dynamics()
    a = floor_scene(previous_action="right_jump")
    model.learn(a, replace(a, frame=1, y=83, vy=4, grounded=False))
    assert model.summary()["parameters"]["jump_speed"]["samples"] == 1


def test_forecast_geometry_is_independent_of_world_number_and_absolute_x():
    from typesafe_mario.prediction import Predictor

    a = floor_scene(
        x=80,
        actors=(Actor("0:goomba", "goomba", 150, 79),),
        blocks=(Block(0, 112, 63, 79), Block(144, 256, 63, 79)),
    )
    b = replace(
        a,
        level=(7, 4, 2),
        x=a.x + 2048,
        known_x=(2048, 2304),
        blocks=tuple(replace(t, left=t.left + 2048, right=t.right + 2048) for t in a.blocks),
        actors=tuple(replace(t, x=t.x + 2048) for t in a.actors),
    )
    before = Predictor().forecast(a, tuple(Action), cycle=8)
    after = Predictor().forecast(b, tuple(Action), cycle=8)
    assert assess_forecast(before) == assess_forecast(after)
    for left, right in zip(before["action_forecasts"], after["action_forecasts"], strict=True):
        assert left["control_risk"] == right["control_risk"]
        assert left["hazards"] == right["hazards"]
        assert abs(right["end"]["x"] - left["end"]["x"] - 2048) < 0.01
        assert right["end"]["y"] == left["end"]["y"]


def test_capped_quantized_air_motion_does_not_erase_acceleration():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    x, vx, last_x = 300.0, 0.8, 300
    for frame in range(201):
        d = 1 if (max(0, frame - 1) // 32) % 2 == 0 else -1
        if frame:
            vx = max(-1.6, min(1.6, vx + 0.1 * d))
            x += vx
        pixel_x = round(x)
        p.observe(
            floor_scene(
                frame=frame,
                x=pixel_x,
                vx=pixel_x - last_x,
                y=180,
                vy=0,
                grounded=False,
                previous_action="right_jump" if d == 1 else "left",
                blocks=(),
                actors=(),
                known_x=(0, 1000),
            )
        )
        last_x = pixel_x
    assert abs(p.dynamics.value("air_accel") - 0.1) < 0.025


def test_air_fit_rejects_input_changes_contacts_and_broken_observation_windows():
    from typesafe_mario.dynamics import Dynamics

    window = [
        floor_scene(
            frame=f, x=80 + 0.05 * f * (f + 1), y=180, grounded=False, previous_action="right_jump"
        )
        for f in range(9)
    ]
    valid = Dynamics()
    valid.learn_air(window)
    assert abs(valid.samples["air_accel"][0] - 0.1) < 1e-8
    for changed in (
        replace(window[4], frame=9),
        replace(window[4], previous_action="left"),
        replace(window[4], grounded=True),
        replace(window[4], swimming=True),
        replace(window[4], motion_reliable=False),
        replace(window[4], terminal=True),
        replace(window[4], actors=(Actor("g", "goomba", 82, 180),)),
    ):
        broken = window.copy()
        broken[4] = changed
        model = Dynamics()
        model.learn_air(broken)
        assert not model.samples["air_accel"]


def test_quantized_apex_retains_observed_ascent_before_switching_to_fall():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    last_y = 130
    for frame, y in enumerate((132, 134, 136, 137, 139, 140, 141, 142, 142)):
        o = floor_scene(
            frame=frame,
            x=811,
            y=y,
            vx=0,
            vy=y - last_y,
            grounded=False,
            previous_action="right_jump",
        )
        p.observe(o)
        last_y = y
    result = p.forecast(
        o, (Action.RIGHT_JUMP,), cycle=8, delay=8, scheduled_action=Action.RIGHT_JUMP
    )
    assert result["committed_future"]["y"] >= 130
    assert result["committed_future"]["y"] <= 143


def test_held_gravity_window_rejects_release_unknown_takeoff_and_apex():
    from typesafe_mario.dynamics import Dynamics

    window = [
        floor_scene(
            frame=f, x=80, y=79 + 5 * f - 0.1 * f * f, grounded=False, previous_action="right_jump"
        )
        for f in range(9)
    ]
    model = Dynamics()
    model.learn_held_gravity(window, fast_jump=False)
    assert abs(model.samples["gravity_hold"][0] - 0.2) < 1e-8
    model = Dynamics()
    model.learn_held_gravity(window, fast_jump=None)
    assert not model.samples["gravity_hold"]
    released = window.copy()
    released[4] = replace(released[4], previous_action="right")
    model.learn_held_gravity(released, fast_jump=False)
    assert not model.samples["gravity_hold"]
    apex = [replace(h, y=142 + round(0.5 * h.frame - 0.1 * h.frame * h.frame)) for h in window]
    model.learn_held_gravity(apex, fast_jump=False)
    assert not model.samples["gravity_hold"]


def test_observed_bounce_phase_survives_forecasting_and_resets_on_landing():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(
        x=100,
        y=83,
        vx=0,
        vy=-6,
        grounded=False,
        previous_action="jump",
        actors=(Actor("g", "goomba", 100, 71),),
    )
    p.observe(o)
    p.observe(replace(o, frame=1, y=87, vy=4))
    confirmed = replace(o, frame=2, y=90, vy=3, score=100)
    p.observe(confirmed)
    result = p.forecast(confirmed, (Action.JUMP,), cycle=8)
    branch = result["action_forecasts"][0]
    assert branch["first_cycle"]["vy"] < 0
    assert "uncertain_bounce" in branch["continuation"]["warnings"]
    landed = replace(confirmed, frame=3, y=79, vy=0, grounded=True)
    p.observe(landed)
    result = p.forecast(landed, (Action.NOOP,), cycle=8)
    assert "uncertain_bounce" not in result["action_forecasts"][0]["continuation"]["warnings"]


def test_delayed_stomp_score_does_not_reclassify_a_subsequent_regular_jump():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(
        x=100,
        y=83,
        vx=0,
        vy=-6,
        grounded=False,
        previous_action="jump",
        actors=(Actor("g", "goomba", 100, 71),),
    )
    p.observe(o)
    p.observe(replace(o, frame=1, y=87, vy=4))
    p.observe(replace(o, frame=2, y=79, vy=0, grounded=True, previous_action="noop"))
    regular_jump = replace(o, frame=3, y=84, vy=5, score=100)
    p.observe(regular_jump)
    result = p.forecast(regular_jump, (Action.JUMP,), cycle=8)
    assert result["observed_interactions"]
    assert "uncertain_bounce" not in result["action_forecasts"][0]["continuation"]["warnings"]


def test_fresh_jump_on_landing_frame_carries_into_next_frame():
    from typesafe_mario.prediction import Predictor

    o = floor_scene(
        x=2393,
        y=131,
        vx=1.75,
        vy=-4,
        grounded=False,
        previous_action="noop",
        known_x=(2270, 2526),
        blocks=(
            Block(2400, 2448, 111, 127),
            Block(2416, 2448, 127, 143),
            Block(2270, 2448, 63, 79),
        ),
    )
    result = Predictor().forecast(
        o, (Action.NOOP,), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP
    )
    assert result["committed_future"]["y"] >= 150


def test_jump_pressed_earlier_in_flight_does_not_wait_indefinitely_for_landing():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    o = floor_scene()
    m = Motion(80, 120, 0, -3, False, "noop")
    heights = []
    for _ in range(24):
        m, _ = _advance(m, Action.JUMP, o, Dynamics())
        heights.append(m.y)
    assert m.grounded
    assert all(b <= a for a, b in pairwise(heights))


def test_observed_left_edge_stall_keeps_retreat_in_view_and_checks_approaching_enemy():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    for frame in range(8):
        o = floor_scene(
            frame=frame,
            x=400,
            vx=0,
            previous_action="left",
            known_x=(400, 656),
            blocks=(Block(400, 656, 63, 79),),
            actors=(Actor("g", "goomba", 432 - frame, 71),),
        )
        p.observe(o)
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.LEFT)
    assert r["committed_future"]["x"] == 400
    left = next(b for b in r["action_forecasts"] if b["action"] == "left")
    assert left["first_cycle"]["x"] == 400
    assert not left["first_cycle_risk"]["unobserved_terrain"]
    assert left["first_cycle_risk"]["nominal_contact"]
    assert left["continuation"]["failure"] == "nominal_contact"


def test_left_boundary_needs_actual_pushing_and_expires_with_view_or_history_change():
    from typesafe_mario.prediction import Predictor

    for change in ({"known_x": (390, 646)}, {"frame": 30}, {"motion_reliable": False}):
        p = Predictor()
        for frame in range(8):
            o = floor_scene(
                frame=frame,
                x=400,
                vx=0,
                previous_action="left",
                known_x=(400, 656),
                blocks=(Block(390, 656, 63, 79),),
            )
            p.observe(o)
        changed = replace(o, **({"frame": 8} | change))
        p.observe(changed)
        r = p.forecast(changed, (Action.LEFT,), cycle=8)
        assert r["action_forecasts"][0]["first_cycle"]["x"] < 400
    p = Predictor()
    for frame in range(8):
        o = floor_scene(frame=frame, x=0, vx=0, previous_action="noop")
        p.observe(o)
    assert p.forecast(o, (Action.LEFT,), cycle=8)["action_forecasts"][0]["first_cycle"]["x"] < 0


def test_quantized_held_ascent_does_not_force_aborting_a_gap_jump():
    from typesafe_mario.prediction import Predictor

    # Actual v29 1-1 frames732..736. Last integer delta3 understates the
    # ascent rate; held A reaches y133 at744 in the recorded execution.
    p = Predictor()
    for frame, x, y, vy in (
        (732, 1061, 93, 4),
        (733, 1062, 98, 5),
        (734, 1064, 102, 4),
        (735, 1066, 106, 4),
        (736, 1068, 109, 3),
    ):
        o = floor_scene(
            frame=frame,
            x=x,
            y=y,
            vx=2,
            vy=vy,
            grounded=False,
            previous_action="right_run_jump",
            known_x=(952, 1208),
            blocks=(Block(952, 1104, 63, 79), Block(1136, 1208, 63, 79)),
        )
        p.observe(o)
    result = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)
    assert abs(result["committed_future"]["y"] - 133) < 2
    assert "right_run_jump" in assess_forecast(result)["candidate_actions"]


def test_short_low_speed_releases_do_not_inflate_friction_from_pixel_rounding():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    x, previous_x, frame = 100.0, 100, 0
    for _ in range(30):
        velocity = 0.7
        for step in range(16):
            action = "right" if step < 8 else "noop"
            if action == "noop":
                velocity = max(0, velocity - 0.06)
            x += velocity
            o = floor_scene(
                frame=frame,
                x=int(x),
                vx=int(x) - previous_x,
                previous_action=action,
                known_x=(0, 10000),
                blocks=(Block(0, 10000, 63, 79),),
            )
            p.observe(o)
            previous_x, frame = int(x), frame + 1
    # Actual deceleration is0.06. Sparse subpixel motion may retain the0.08
    # prior, but must not infer the0.4 stop rate produced by censored0/1 deltas.
    assert 0.04 <= p.dynamics.value("friction") <= 0.09


def test_friction_is_learned_from_contiguous_grounded_positions_in_both_directions():
    from typesafe_mario.prediction import Predictor

    for direction in (-1, 1):
        p = Predictor()
        x, previous_x = 200.0, 200
        for frame in range(19):
            x += direction * (3 - 0.12 * frame)
            o = floor_scene(
                frame=frame,
                x=int(x),
                vx=int(x) - previous_x,
                previous_action="noop",
                known_x=(0, 1000),
                blocks=(Block(0, 1000, 63, 79),),
            )
            p.observe(o)
            previous_x = int(x)
        assert 0.10 < p.dynamics.value("friction") < 0.14


def test_fresh_ground_jump_first_moves_on_support_then_takes_off():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    o = floor_scene(vx=0, previous_action="noop")
    first, _ = _advance(Motion(o.x, o.y, 0, 0, True, "noop"), Action.JUMP, o, Dynamics())
    assert first.y == 79 and first.grounded
    second, _ = _advance(first, Action.JUMP, o, Dynamics())
    assert second.y > 79 and not second.grounded
    released, _ = _advance(first, Action.NOOP, o, Dynamics())
    assert released.y == 79 and released.grounded


def test_jump_pressed_on_last_support_pixel_cannot_invent_takeoff_after_departure():
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion, _advance

    # Actual v32 1-2: x1274→1277 on first A frame, still y79 but support lost.
    o = floor_scene(x=1274, vx=3, blocks=(Block(1128, 1280, 63, 79),))
    first, _ = _advance(Motion(o.x, o.y, 3, 0, True, "right"), Action.RIGHT_RUN_JUMP, o, Dynamics())
    assert first.x == 1277 and first.y == 79 and not first.grounded
    second, _ = _advance(first, Action.RIGHT_RUN_JUMP, o, Dynamics())
    assert second.y < 79 and not second.grounded


def test_observed_fresh_ground_press_is_carried_into_continued_input():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    o = floor_scene(vx=0, previous_action="noop")
    p.observe(o)
    pressed = replace(o, frame=1, previous_action="jump")
    p.observe(pressed)
    result = p.forecast(
        pressed,
        (Action.NOOP,),
        cycle=8,
        delay=1,
        scheduled_action=Action.JUMP,
        scheduled_first_frame_action=Action.JUMP,
    )
    assert result["committed_future"]["y"] > 79


def test_accelerating_position_window_estimates_current_speed_not_past_average():
    from typesafe_mario.prediction import Predictor

    # Independent constant-acceleration motion: v starts at0.4 and gains0.1
    # each frame. At frame8, v=1.2; releasing horizontal input in flight should
    # carry approximately9.6pixels over the next8frames, in either direction.
    for sign, action in ((1, "right"), (-1, "left")):
        p = Predictor()
        previous_x = 100
        for frame in range(9):
            x = round(100 + sign * (0.4 * frame + 0.05 * frame * (frame + 1)))
            o = floor_scene(
                frame=frame,
                x=x,
                vx=x - previous_x,
                y=170,
                vy=0,
                grounded=False,
                previous_action=action,
            )
            p.observe(o)
            previous_x = x
        r = p.forecast(o, (Action.NOOP,), cycle=8)
        travel = sign * (r["action_forecasts"][0]["first_cycle"]["x"] - o.x)
        assert 8.5 < travel < 10.7


def test_position_fit_does_not_invent_reversal_after_coasting_to_stop():
    from typesafe_mario.prediction import Predictor

    for sign in (1, -1):
        p = Predictor()
        previous_x = 100
        for frame, displacement in enumerate((0, 1, 2, 3, 3, 3, 3, 3, 3)):
            x = 100 + sign * displacement
            o = floor_scene(frame=frame, x=x, vx=x - previous_x, previous_action="noop")
            p.observe(o)
            previous_x = x
        r = p.forecast(o, (Action.NOOP,), cycle=8)
        assert r["committed_future"]["vx"] == 0
        assert r["action_forecasts"][0]["first_cycle"]["x"] == o.x


def test_quantized_takeoff_speed_does_not_turn_walking_flight_into_running_flight():
    from typesafe_mario.prediction import Predictor

    p = Predictor()
    # Actual v37 takeoff trace, translated: the last grounded delta is2, but
    # its surrounding positions show speed below2. Airborne right+B then
    # remains about1.75px/frame rather than accelerating toward3.
    xs = (100, 101, 102, 103, 104, 105, 107, 108, 110, 111, 113, 115, 117, 118, 120, 122)
    ys = (79,) * 9 + (84, 89, 93, 98, 102, 106, 109)
    for frame, (x, y) in enumerate(zip(xs, ys, strict=True)):
        o = floor_scene(
            frame=frame,
            x=x,
            y=y,
            vx=x - xs[max(0, frame - 1)],
            vy=y - ys[max(0, frame - 1)],
            grounded=frame < 9,
            previous_action="right_run" if frame < 8 else "right_run_jump",
        )
        p.observe(o)
    r = p.forecast(o, (Action.RIGHT_RUN_JUMP,), cycle=8)
    assert r["action_forecasts"][0]["first_cycle"]["vx"] < 2
    assert r["calibration"]["parameters"]["jump_speed"]["samples"] == 1
    assert r["calibration"]["parameters"]["jump_speed_fast"]["samples"] == 0
