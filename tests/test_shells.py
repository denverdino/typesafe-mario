from dataclasses import replace

import pytest
from test_observation_prediction import floor_scene

from typesafe_mario.actions import Action
from typesafe_mario.actor_motion import actor_path
from typesafe_mario.observation import Actor, Block
from typesafe_mario.prediction import Predictor
from typesafe_mario.risk import assess_risk


def observed_shell():
    p = Predictor()
    actor = Actor("k", "green_koopa", 100, 71)
    for frame, y, vy in ((0, 94, -4), (1, 98, 4), (2, 102, 4), (3, 105, 3)):
        o = floor_scene(frame=frame, x=100, y=y, vy=vy, grounded=False, actors=(actor,))
        p.observe(o)
    return p, o


def test_actual_stomp_then_stationary_koopa_is_retained_as_shell():
    p, o = observed_shell()
    r = p.forecast(o, tuple(Action), cycle=8)
    actor = r["actor_forecasts"][0]
    assert actor.get("state") == "stationary_shell"
    assert r["observed_interactions"][0]["inference"] == "stationary_shell"
    assert actor["id"] == "k"
    assert actor["kick_forecasts"]["right"]["assumption"]
    # An ordinary motionless turtle has no observed stomp evidence.
    fresh = Predictor()
    for frame in range(5):
        ordinary = floor_scene(frame=frame, x=60, actors=o.actors)
        fresh.observe(ordinary)
    assert (
        fresh.forecast(ordinary, tuple(Action), cycle=8)["actor_forecasts"][0].get("state")
        != "stationary_shell"
    )


def test_observed_shell_motion_uses_post_kick_velocity_and_wall_rebound():
    p, o = observed_shell()
    moved = replace(
        o,
        frame=4,
        actors=(replace(o.actors[0], x=104),),
        blocks=o.blocks + (Block(136, 152, 79, 127),),
    )
    p.observe(moved)
    r = p.forecast(moved, tuple(Action), cycle=8)
    a = r["actor_forecasts"][0]
    assert a.get("state") == "moving_shell"
    assert a["observed_velocity"]["x"] == 4
    assert a["trajectory"][-1]["x"] < 104


@pytest.mark.parametrize("change", [{"frame": 10}, {"actors": ()}, {"motion_reliable": False}])
def test_shell_inference_expires_on_observation_loss(change):
    p, o = observed_shell()
    lost = replace(o, frame=4, **{k: v for k, v in change.items() if k != "frame"})
    if "frame" in change:
        lost = replace(lost, frame=change["frame"])
    p.observe(lost)
    assert not p.forecast(lost, tuple(Action), cycle=8)["observed_interactions"]


def test_moving_shell_bounces_from_visible_pipe_and_remains_a_hazard():
    o = floor_scene(blocks=(Block(0, 256, 63, 79), Block(136, 152, 79, 127)))
    path = actor_path(Actor("k", "moving_shell", 100, 71), (4, 0, True), o, 20)
    assert max(x for x, _, _ in path) <= 120
    assert path[-1][0] < 100


@pytest.mark.parametrize("speed", [0, 2])
def test_ground_contact_with_confirmed_shell_is_an_uncertain_kick_not_proven_death(speed):
    p, o = observed_shell()
    for frame in range(4, 17):
        y = max(79, 105 - (frame - 3) * 3)
        o = replace(
            o,
            frame=frame,
            x=max(80, 100 - (frame - 3) * 2),
            y=y,
            vy=-3 if y > 79 else 0,
            vx=-2 if frame < 13 else 0,
            grounded=y == 79,
            previous_action="noop",
        )
        p.observe(o)
    if speed:
        o = replace(o, frame=17, x=o.x + speed, vx=speed, previous_action="right")
        p.observe(o)
    before = p.shells.confirmed(o)["k"].state
    r = p.forecast(o, (Action.RIGHT,), cycle=8)
    branch = r["action_forecasts"][0]
    assert branch["control_risk"]["uncertain_shell_kick"]
    assert not branch["control_risk"]["nominal_contact"]
    assert branch["continuation"]["failure"] is None
    assert "possible_shell_kick" in branch["continuation"]["warnings"]
    assert branch["continuation"]["evaluated_frames"] < branch["continuation"]["horizon_frames"]
    assert p.shells.confirmed(o)["k"].state == before == "stationary_shell"


def test_hypothetical_koopa_stomp_does_not_create_observed_shell_evidence():
    p = Predictor()
    o = floor_scene(
        x=100, y=100, vy=-3, vx=0, grounded=False, actors=(Actor("k", "green_koopa", 100, 71),)
    )
    p.observe(o)
    r = p.forecast(o, (Action.NOOP,), cycle=8)
    assert "possible_stationary_shell" in r["action_forecasts"][0]["continuation"]["warnings"]
    assert r["observed_interactions"] == []
    assert not p.shells.confirmed(o)


def test_continuation_cannot_discharge_possible_kick_in_executable_cycle():
    p, o = observed_shell()
    o = replace(
        o,
        frame=4,
        x=81,
        y=79,
        vx=1.5,
        vy=0,
        grounded=True,
        previous_action="noop",
        actors=o.actors + (Actor("g", "goomba", 46, 71),),
    )
    p.observe(o)
    prediction = p.forecast(o, tuple(Action), cycle=8)
    jump = next(b for b in prediction["action_forecasts"] if b["action"] == "jump")
    assert jump["first_cycle_risk"]["uncertain_shell_kick"]
    risk = assess_risk(prediction, (Action.JUMP, Action.RIGHT_JUMP))
    assert "uncertain_shell_kick" in risk["warnings"]["jump"]
    assert risk["status"] == "no_lower_risk_option"


def test_possible_shell_kick_does_not_hide_another_enemy():
    p, o = observed_shell()
    o = replace(
        o,
        frame=4,
        x=81,
        y=79,
        vx=2,
        vy=0,
        grounded=True,
        previous_action="right",
        actors=o.actors + (Actor("g", "goomba", 94, 71),),
    )
    p.observe(o)
    branch = p.forecast(o, (Action.RIGHT,), cycle=8)["action_forecasts"][0]
    assert branch["control_risk"]["uncertain_shell_kick"]
    assert branch["control_risk"]["nominal_contact"]
    assert {h["id"] for h in branch["hazards"]} == {"k", "g"}


def test_slow_motion_invalidates_kickable_shell_inference():
    p, o = observed_shell()
    o = replace(o, frame=4, actors=(replace(o.actors[0], x=101),))
    p.observe(o)
    assert not p.shells.confirmed(o)
    actor = p.forecast(o, (Action.NOOP,), cycle=8)["actor_forecasts"][0]
    assert actor["state"] == "green_koopa"
    assert "kick_forecasts" not in actor


def test_shell_evidence_and_return_warning_reach_provider():
    from test_policy import Provider, policy

    from typesafe_mario.state import MarioStateParser

    p, o = observed_shell()
    o = replace(o, frame=4, blocks=o.blocks + (Block(136, 152, 79, 127),))
    p.observe(o)
    prediction = p.forecast(o, tuple(Action), cycle=8)
    provider = Provider("jump")
    snapshot = replace(MarioStateParser().parse({}), prediction=prediction, frame_index=o.frame)
    decision = policy(provider).choose(snapshot, tuple(Action))
    sent = provider.request["state"]["prediction"]
    actor = sent["actor_forecasts"][0]
    assert actor["state"] == "stationary_shell"
    assert sent["observed_interactions"][0]["actor_id"] == "k"
    right = actor["kick_forecasts"]["right"]
    assert 0 < right["first_rebound_frame"] <= right["return_near_kick_frame"]
    assert decision.action == Action.JUMP
