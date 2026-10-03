import json
from pathlib import Path

from prediction_helpers import assess_forecast

from typesafe_mario.actions import Action
from typesafe_mario.dynamics import Dynamics
from typesafe_mario.observation import Actor, Block, Observation
from typesafe_mario.prediction import Predictor


def recorded_case(name):
    data = json.loads((Path(__file__).parent / "fixtures/world1-2-v42-timing.json").read_text())[
        "cases"
    ][name]
    p = Predictor()
    for h in data["history"]:
        h = dict(
            h,
            level=tuple(h["level"]),
            known_x=tuple(h["known_x"]),
            blocks=tuple(Block(**b) for b in h["blocks"]),
            actors=tuple(Actor(**a) for a in h["actors"]),
        )
        o = Observation(**h)
        p.observe(o)
    p.dynamics = Dynamics()
    for key, values in data["samples"].items():
        for v in values:
            p.dynamics.add(key, v)
    p.fast_jump, p.bouncing = data["fast_jump"], data["bouncing"]
    return (
        p,
        o,
        {"cycle": 8, "delay": 8, "scheduled_action": Action(data["timing"]["scheduled_action"])},
    )


def test_late_repeated_jump_landing_cannot_force_wait_over_a_clear_current_jump():
    p, o, timing = recorded_case("episode2_frame912")
    r = p.forecast(o, tuple(Action), **timing)
    jump = next(b for b in r["action_forecasts"] if b["action"] == "jump")
    assert jump["continuation"]["failure"] is None
    assert not any(v is True for v in jump["first_cycle_risk"].values())
    assert "jump" in assess_forecast(r)["candidate_actions"]
    # A late uncertain landing is not cleared; it remains comparable to a
    # waiting action that itself already has possible-contact warnings.
    assert "landing_contact_window" in assess_forecast(r)["warnings"]["jump"]


def test_jump_at_predicted_landing_boundary_is_not_a_guaranteed_takeoff():
    p, o, timing = recorded_case("episode1_frame240")
    r = p.forecast(o, tuple(Action), **timing)
    jump = next(b for b in r["action_forecasts"] if b["action"] == "right_run_jump")
    # Nominal support appears exactly at f248; actual support arrived at f250.
    assert r["committed_future"]["grounded_estimate"]
    assert jump["first_cycle_risk"]["uncertain_takeoff"]
    assert "uncertain_takeoff" in jump["continuation"]["warnings"]
    assert assess_forecast(r)["status"] != "unrestricted"


def test_observed_enemy_turn_does_not_average_away_its_current_direction():
    p, o, timing = recorded_case("episode1_frame248")
    actor = p.forecast(o, tuple(Action), **timing)["actor_forecasts"][0]
    # Actual x450,449,449,448,447,448,449,449: the Goomba has turned right.
    assert 0.5 <= actor["observed_velocity"]["x"] <= 1
    assert actor["trajectory"][0]["x"] > 452


def test_current_landing_warning_cannot_be_discharged_by_a_later_clear_route():
    from typesafe_mario.risk import assess_risk

    p, o, timing = recorded_case("episode2_frame912")
    r = p.forecast(o, tuple(Action), **timing)
    jump = next(b for b in r["action_forecasts"] if b["action"] == "jump")
    jump["first_cycle_risk"]["landing_contact_window"] = True
    report = assess_risk(r, tuple(Action))
    assert "jump" not in report.get("repeated_input_warnings", {})
    assert "landing_contact_window" in report["warnings"]["jump"]


def test_uncertain_takeoff_is_preserved_when_a_continuation_fallback_is_used():
    from typesafe_mario.risk import assess_risk

    p, o, timing = recorded_case("episode1_frame240")
    r = p.forecast(o, (Action.JUMP, Action.NOOP), **timing)
    # Force both repeated-input projections to warn. Only NOOP has a valid
    # clear continuation; even a nominal clear JUMP plan cannot erase its prefix.
    for b in r["action_forecasts"]:
        b["control_risk"]["possible_contact"] = True
        b["continuation"].update(status="estimated_viable", warnings=[], failure=None)
    jump = r["action_forecasts"][0]
    jump["continuation"]["failure"] = "contact"
    report = assess_risk(r, (Action.JUMP, Action.NOOP))
    assert report["basis"] == "bounded_action_continuations"
    assert report["candidate_actions"] == ["noop"]
    assert "uncertain_takeoff" in report["warnings"]["jump"]


def test_settled_landing_and_observed_ground_do_not_block_a_new_jump():
    from test_observation_prediction import floor_scene

    for o, delay in [(floor_scene(vx=0), 0), (floor_scene(vx=0, y=80, vy=-4, grounded=False), 8)]:
        p = Predictor()
        p.observe(o)
        r = p.forecast(o, tuple(Action), cycle=8, delay=delay, scheduled_action=Action.NOOP)
        assert r["committed_future"]["grounded_estimate"]
        jump = next(b for b in r["action_forecasts"] if b["action"] == "jump")
        assert not jump["first_cycle_risk"]["uncertain_takeoff"]
        assert "uncertain_takeoff" not in jump["continuation"]["warnings"]
        assert jump["first_cycle"]["y"] > o.y


def test_enemy_turn_velocity_is_symmetric_and_does_not_invent_stationary_motion():
    from dataclasses import replace

    from test_observation_prediction import floor_scene

    for positions, expected in [
        ([50, 49, 49, 48, 47, 48, 49, 49], 5 / 7),
        ([50, 51, 51, 52, 53, 52, 51, 51], -5 / 7),
        ([50] * 8, 0),
        ([50, 49, 49, 48, 48, 47, 47, 46], -4 / 7),
    ]:
        p = Predictor()
        for frame, x in enumerate(positions):
            o = replace(floor_scene(), frame=frame, actors=(Actor("g", "goomba", x, 71),))
            p.observe(o)
        vx, _, known = p._enemy_motion(o)["g"]
        assert known
        assert abs(vx - expected) < 1e-9


def test_provider_receives_takeoff_uncertainty_and_keeps_its_selected_action():
    from dataclasses import replace

    from test_policy import Provider, policy

    from typesafe_mario.state import MarioStateParser

    p, o, timing = recorded_case("episode1_frame240")
    r = p.forecast(o, tuple(Action), **timing)
    chosen = assess_forecast(r)["candidate_actions"][0]
    provider = Provider(chosen)
    decision = policy(provider).choose(
        replace(
            MarioStateParser().parse({}),
            prediction=r,
            frame_index=r["observed_at_frame"],
            frames_until_action=r["action_delay_frames"],
            decision_horizon_frames=r["action_cycle_frames"],
        ),
        tuple(Action),
    )
    sent = provider.request["state"]["prediction"]
    assert sent["committed_future"]["landing_timing"]["uncertain_takeoff"]
    state = provider.request["state"]
    assert "uncertain_takeoff" in state["risk_control"]["excluded_actions"]["right_run_jump"]
    assert "right_run_jump" not in {b["action"] for b in state["action_assessments"]}
    jump = next(b for b in r["action_forecasts"] if b["action"] == "right_run_jump")
    assert "uncertain_takeoff" in jump["continuation"]["warnings"]
    assert decision.action.value == chosen


def test_fallback_cannot_erase_landing_margin_with_a_close_or_incomplete_plan():
    from copy import deepcopy

    from typesafe_mario.risk import assess_risk

    p, o, timing = recorded_case("episode2_frame912")
    original = p.forecast(o, (Action.JUMP, Action.NOOP), **timing)
    for incomplete in (False, True):
        r = deepcopy(original)
        jump, noop = r["action_forecasts"]
        jump["control_risk"] = {"through_frame": 24, "landing_contact_window": True}
        jump["first_cycle_risk"] = {"landing_contact_window": False}
        jump["continuation"].update(failure=None, status="estimated_viable", warnings=[])
        if incomplete:
            jump["continuation"]["evaluated_frames"] = 8
        else:
            jump["continuation"]["warnings"] = ["close_contact"]
        noop["control_risk"] = {"through_frame": 24, "nominal_contact": True}
        noop["continuation"]["failure"] = "nominal_contact"
        report = assess_risk(r, (Action.JUMP, Action.NOOP))
        assert "landing_contact_window" in report["warnings"]["jump"]
        assert report["status"] == "no_lower_risk_option"
