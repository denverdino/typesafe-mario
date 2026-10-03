"""Reproduce a logged stall using observations only, without an emulator/API."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.observation import Block, Observation
from typesafe_mario.prediction import Predictor
from typesafe_mario.risk import assess_risk


@pytest.fixture
def stalled():
    fixture = json.loads((Path(__file__).parent / "fixtures/left-wall-stall.json").read_text())
    state = fixture["state"]
    player = state["player"]
    o = Observation(
        state["observed_at_frame"],
        (1, 1, 1),
        player["x"],
        player["y"],
        0,
        0,
        True,
        False,
        "left",
        True,
        tuple(state["terrain"]["known_x"]),
        tuple(Block(**b) for b in state["terrain"]["blocks"]),
        (),
    )
    prediction = Predictor(physics_parameters=fixture["parameters"]).forecast(
        o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.LEFT
    )
    return state, prediction


def test_logged_left_wall_stall_must_offer_movement_instead_of_more_stationary_input(stalled):
    state, prediction = stalled
    report = assess_risk(prediction, tuple(Action), state=state)
    assert "left" not in report["candidate_actions"]
    assert "noop" not in report["candidate_actions"]
    assert {"right_jump", "right_run_jump"} & set(report["candidate_actions"])


@pytest.mark.parametrize("condition", ["brief", "no_wall", "unsafe_alternatives"])
def test_left_wall_recovery_requires_observed_stall_and_clear_alternative(stalled, condition):
    state, prediction = deepcopy(stalled)
    if condition == "brief":
        state["recovery"]["stationary_frames"] = 8
        state["recovery"]["stationary_action_frames"] = {"left": 8}
    elif condition == "no_wall":
        state["terrain"]["blocks"] = []
    else:
        for branch in prediction["action_forecasts"]:
            if branch["action"] not in {"left", "noop"}:
                branch["first_cycle_risk"]["nominal_contact"] = True
                branch["control_risk"]["nominal_contact"] = True
    assert "left" in assess_risk(prediction, tuple(Action), state=state)["candidate_actions"]
