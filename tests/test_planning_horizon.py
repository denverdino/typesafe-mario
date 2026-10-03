"""Planning time is a physical horizon, independent of controller cadence."""

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.observation import Block, Observation
from typesafe_mario.prediction import Predictor
from typesafe_mario.risk import assess_risk


@pytest.mark.parametrize("cycle", [2, 4, 8])
def test_shorter_control_cycle_preserves_requested_planning_time(cycle):
    o = Observation(
        0,
        (1, 2, 3),
        100,
        79,
        0,
        0,
        True,
        False,
        "noop",
        True,
        (0, 400),
        (Block(0, 400, 63, 79),),
        (),
    )
    r = Predictor().forecast(o, (Action.NOOP,), cycle=cycle)
    plan = r["action_forecasts"][0]["continuation"]
    assert plan["horizon_frames"] == 48
    assert plan["evaluated_frames"] == 48


@pytest.mark.parametrize("offset", [0, 1200])
def test_four_frame_control_can_finish_a_visible_column_climb(offset):
    # Visible geometry from the World 1-2 stall. No RAM or executable replay.
    o = Observation(
        0,
        (1, 2, 3),
        offset + 354,
        79,
        0,
        0,
        True,
        False,
        "noop",
        True,
        (offset + 234, offset + 490),
        tuple(
            Block(offset + left, offset + right, bottom, top)
            for left, right, bottom, top in (
                (234, 490, 63, 79),
                (234, 490, 239, 255),
                (336, 352, 79, 127),
                (368, 384, 79, 143),
                (400, 416, 79, 143),
                (432, 448, 79, 127),
            )
        ),
        (),
        height=32,
    )
    r = Predictor().forecast(o, tuple(Action), cycle=4)
    branch = next(b for b in r["action_forecasts"] if b["action"] == "right_run_jump")
    plan = branch["continuation"]
    assert plan["end"]["x"] > o.x + 16
    assert plan["end"]["grounded_estimate"]
    assert plan["status"] == "estimated_viable"
    assert "right_run_jump" in assess_risk(r, tuple(Action))["candidate_actions"]
