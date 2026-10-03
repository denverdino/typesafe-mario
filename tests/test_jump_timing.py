"""Regressions from the three episodes in run-20261003T001239.415148Z."""

from dataclasses import replace

import pytest
from prediction_helpers import assess_forecast
from test_observation_prediction import floor_scene

from typesafe_mario.actions import Action
from typesafe_mario.dynamics import Dynamics
from typesafe_mario.observation import Actor, Block
from typesafe_mario.prediction import Motion, Predictor, _advance


@pytest.mark.parametrize("offset", [0, 1200])
@pytest.mark.parametrize("tiled", [False, True])
def test_descending_under_brick_edge_does_not_invent_a_horizontal_stop(offset, tiled):
    # Episode 3, f928→936: (813,136)→(827,99), still moving right.
    o = floor_scene(
        x=offset + 813,
        y=136,
        vx=1.67,
        vy=-3,
        grounded=False,
        known_x=(offset + 750, offset + 1006),
        blocks=(Block(offset + 750, offset + 1006, 63, 79),)
        + (
            tuple(Block(offset + 832, offset + 864, y, y + 16) for y in range(191, 126, -16))
            if tiled
            else (Block(offset + 832, offset + 864, 127, 207),)
        ),
    )
    m = Motion(o.x, o.y, o.vx, o.vy, False, "right_jump")
    events = []
    for _ in range(8):
        m, emitted = _advance(m, Action.RIGHT, o, Dynamics())
        events.extend(emitted)
    assert abs(m.x - (offset + 827)) < 2
    assert m.vx > 1.5
    assert "uncertain_wall_contact" in events


def test_underside_uncertainty_cannot_be_a_clear_braking_refuge():
    o = floor_scene(
        x=813,
        y=136,
        vx=1.67,
        vy=-3,
        grounded=False,
        known_x=(750, 1006),
        blocks=(Block(750, 1006, 63, 79), Block(832, 864, 127, 207)),
        actors=(Actor("g", "goomba", 852, 71),),
    )
    p = Predictor()
    p.observe(replace(o, frame=0, actors=(replace(o.actors[0], x=852.625),)))
    o = replace(o, frame=1)
    p.observe(o)
    r = p.forecast(o, (Action.LEFT,), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    assert r["committed_future"]["x"] > 825
    assert r["action_forecasts"][0]["control_risk"]["uncertain_wall_contact"]
    assert assess_forecast(r)["status"] == "no_lower_risk_option"


def test_stomp_during_committed_input_does_not_invent_grounded_application():
    # Episode 1 f1104: an actual stomp occurred before the next answer applied.
    o = floor_scene(
        x=800,
        y=92,
        vx=1.5,
        vy=-5,
        grounded=False,
        known_x=(720, 976),
        previous_action="right_run_jump",
        blocks=(Block(720, 976, 63, 79),),
        actors=(Actor("g1", "goomba", 805, 71), Actor("g2", "goomba", 844, 71)),
    )
    p = Predictor()
    p.observe(replace(o, frame=0, actors=tuple(replace(a, x=a.x + 0.625) for a in o.actors)))
    o = replace(o, frame=1)
    p.observe(o)
    before = p.calibration()
    result = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)
    assert not result["committed_future"]["grounded_estimate"]
    assert result["committed_interactions"][0]["actor_id"] == "g1"
    assert assess_forecast(result)["status"] == "no_lower_risk_option"
    assert all(
        b["first_cycle_risk"]["uncertain_committed_interaction"] for b in result["action_forecasts"]
    )
    assert {a["id"] for a in result["actor_forecasts"]} == {"g1", "g2"}
    assert result["observed_interactions"] == []
    assert p.calibration() == before
    assert not p.defeated_goombas


def test_floor_below_unfinished_flight_does_not_prove_a_landing():
    from typesafe_mario.continuation import continuations

    o = floor_scene(x=80, y=120, vy=3, grounded=False, previous_action="jump")
    plans, _ = continuations(
        Motion(o.x, o.y, 0, 3, False, "jump"), o, (Action.JUMP,), 8, 0, 8, Dynamics(), {}, _advance
    )
    assert "unresolved_flight" in plans["jump"]["warnings"]
    assert plans["jump"]["status"] != "estimated_viable"


@pytest.mark.parametrize("enemy_x, unsafe", [(134, True), (190, False)])
def test_landing_reports_room_to_brake_or_jump_before_the_next_enemy(enemy_x, unsafe):
    o = floor_scene(
        x=100, y=80, vx=1.6, vy=-5, grounded=False, actors=(Actor("g", "goomba", enemy_x, 71),)
    )
    p = Predictor()
    p.observe(replace(o, frame=0, actors=(replace(o.actors[0], x=enemy_x + 0.625),)))
    o = replace(o, frame=1)
    p.observe(o)
    r = p.forecast(o, (Action.RIGHT,), cycle=8)
    branch = r["action_forecasts"][0]
    landing = branch["first_landing"]
    assert landing["frame"] == 1
    assert landing["frames_until_next_action"] == 7
    margin = landing["enemies"][0]
    assert margin["actor_id"] == "g"
    assert margin["braking_distance_pixels"] > 0
    assert margin["jump_clearance_frames"] >= 3
    assert margin["contact_before_escape"] is unsafe
    assert branch["control_risk"]["landing_contact_window"] is unsafe
    assert ("landing_contact_window" in branch["continuation"]["warnings"]) is unsafe


def test_only_selected_plan_is_carried_forward_and_rechecked():
    o = floor_scene(vx=0, blocks=(Block(0, 256, 63, 79), Block(128, 160, 79, 111)))
    p = Predictor()
    p.observe(o)
    first = p.forecast(o, tuple(Action), cycle=8)
    selected = next(b for b in first["action_forecasts"] if b["action"] == "right_run")
    assert "plan_followup" not in p.forecast(o, tuple(Action), cycle=8)
    p.expect(first, Action.RIGHT_RUN, apply_at_frame=0)
    result = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN)
    followup = result["plan_followup"]
    assert followup["action"] == selected["continuation"]["actions"][1]
    assert followup["revalidated"]
    # A discontinuous observation must not preserve a stale route.
    p.observe(replace(o, frame=40))
    assert "plan_followup" not in p.forecast(replace(o, frame=40), tuple(Action), cycle=8)


def test_rechecked_plan_keeps_new_enemy_risk_and_cannot_override_action():
    from test_policy import Provider, policy

    from typesafe_mario.state import MarioStateParser

    o = floor_scene(vx=0, blocks=(Block(0, 256, 63, 79), Block(128, 160, 79, 111)))
    p = Predictor()
    p.observe(o)
    first = p.forecast(o, tuple(Action), cycle=8)
    p.expect(first, Action.RIGHT_RUN, apply_at_frame=0)
    # A newly observed nearby enemy invalidates the clear route, while the
    # actually selected run input is already committed for seven more frames.
    o = replace(o, frame=1, previous_action="right_run", actors=(Actor("g", "goomba", 94, 71),))
    p.observe(o)
    result = p.forecast(o, tuple(Action), cycle=8, delay=7, scheduled_action=Action.RIGHT_RUN)
    followup = result["plan_followup"]
    assert followup["failure"] or followup["warnings"]
    assert assess_forecast(result)["status"] == "no_lower_risk_option"
    provider = Provider("left")
    decision = policy(provider).choose(
        replace(
            MarioStateParser().parse({}),
            prediction=result,
            frame_index=result["observed_at_frame"],
            frames_until_action=result["action_delay_frames"],
            decision_horizon_frames=result["action_cycle_frames"],
        ),
        tuple(Action),
    )
    sent = provider.request["state"]["prediction"]
    assert sent["plan_followup"] == followup
    assert sent["committed_future"]["valid_for_application"] is False
    assert decision.action == Action.LEFT


def test_simultaneous_contact_during_queue_cannot_remove_the_second_enemy():
    o = floor_scene(
        x=100,
        y=92,
        vx=0,
        vy=-5,
        grounded=False,
        actors=(Actor("g", "goomba", 100, 71), Actor("k", "green_koopa", 100, 71)),
    )
    p = Predictor()
    p.observe(o)
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.NOOP)
    assert r["committed_future"]["valid_for_application"] is False
    assert assess_forecast(r)["status"] == "no_lower_risk_option"
    assert r["observed_interactions"] == []
    assert {a["id"] for a in r["actor_forecasts"]} == {"g", "k"}


def test_settling_tail_is_bounded_by_two_actual_control_cycles():
    o = floor_scene(x=80, y=120, vx=0, vy=3, grounded=False, previous_action="noop")
    r = Predictor().forecast(o, (Action.NOOP,), cycle=2)
    plan = r["action_forecasts"][0]["continuation"]
    assert 48 <= plan["evaluated_frames"] <= 52  # 48 physical frames plus two control cycles.
    assert plan["end"]["grounded_estimate"]


def test_settling_tail_progress_is_averaged_over_evaluated_frames():
    from typesafe_mario.continuation import continuations

    o = floor_scene(x=100, y=120, vx=1, vy=0, grounded=False, previous_action="noop")
    plans, _ = continuations(
        Motion(100, 120, 1, 0, False, "noop"),
        o,
        (Action.NOOP,),
        8,
        0,
        8,
        Dynamics(),
        {},
        _advance,
        landing_horizon=16,
    )
    plan = plans["noop"]
    assert plan["evaluated_frames"] == 16
    # 13 airborne x increments of1, then grounded increments .92,.84,.76.
    assert plan["mean_progress"] == pytest.approx(8.45)
