"""Recorded failures from run-20261003T033909.643156Z."""

import json
from pathlib import Path

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.dynamics import Dynamics
from typesafe_mario.observation import Actor, Block, Observation
from typesafe_mario.prediction import Predictor


def recorded_case(name):
    case = json.loads((Path(__file__).parent / "fixtures/world1-1-v44-failures.json").read_text())[
        "cases"
    ][name]
    p = Predictor()
    for h in case["history"]:
        o = Observation(
            **dict(
                h,
                level=tuple(h["level"]),
                known_x=tuple(h["known_x"]),
                blocks=tuple(Block(**b) for b in h["blocks"]),
                actors=tuple(Actor(**a) for a in h["actors"]),
            )
        )
        p.observe(o)
    p.dynamics = Dynamics()
    for key, samples in case["samples"].items():
        for value in samples:
            p.dynamics.add(key, value)
    p.fast_jump, p.bouncing = case["fast_jump"], case["bouncing"]
    return p, o, case


def test_brick_edge_does_not_invent_ceiling_collision_in_the_committed_jump():
    p, o, case = recorded_case("episode1_decision160")
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.JUMP)
    # Native held-A trace: y109 -> y132. v44 instead predicted y95.7,
    # reversing a still-rising jump merely because the body grazes a brick.
    assert abs(r["committed_future"]["y"] - case["expected_next"]["y"]) < 3
    assert r["committed_future"]["vy"] > 0


@pytest.mark.parametrize("case_name", ["episode1_decision159", "episode3_decision160"])
def test_directional_jump_past_brick_edge_remains_available(case_name):
    p, o, case = recorded_case(case_name)
    r = p.forecast(
        o,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["timing"]["scheduled_action"]),
    )
    # Both native counterfactuals cross the enemy pair onto the next block.
    assert "right_run_jump" in r["risk_control"]["candidate_actions"]


def test_gap_refuge_cannot_discharge_edge_uncertainty_with_a_nominal_landing():
    p, o, _ = recorded_case("episode2_decision105")
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)
    # The original NOOP->LEFT plan nominally lands only four pixels inside
    # support. Native replay overshoots it and falls; immediate LEFT returns
    # to the near bank. A nominal continuation must not erase that margin.
    assert "left" in r["risk_control"]["candidate_actions"]
    assert "noop" not in r["risk_control"]["candidate_actions"]
    noop = next(b for b in r["action_forecasts"] if b["action"] == "noop")
    assert "uncertain_landing" in noop["continuation"]["warnings"]
    assert "uncertain_landing" in r["risk_control"]["excluded_actions"]["noop"]
