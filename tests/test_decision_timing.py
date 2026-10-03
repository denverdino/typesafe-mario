from dataclasses import replace

import pytest
from test_observation_prediction import floor_scene

from typesafe_mario.actions import Action
from typesafe_mario.observation import model_state
from typesafe_mario.prediction import Predictor
from typesafe_mario.state import MarioStateParser


def test_prediction_uses_absolute_application_frame_at_nonzero_observation():
    p = Predictor()
    o = floor_scene(frame=120)
    p.observe(o)
    result = p.forecast(o, (Action.RIGHT,), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    assert result.get("observed_at_frame") == 120
    assert result.get("apply_at_frame") == 128
    assert result.get("action_delay_frames") == 8
    assert result.get("frame_reference") == "offset_from_observed_at_frame"
    assert result["action_forecasts"][0]["first_cycle"]["frame"] == 16
    p.expect(result, Action.RIGHT, apply_at_frame=127)
    assert not p.pending
    p.expect(result, Action.RIGHT, apply_at_frame=128)
    assert p.pending[0][0] == 136
    assert p.pending[0][1] == 128


def test_provider_observation_exposes_same_absolute_clock_after_resume():
    s = replace(
        MarioStateParser().parse({}),
        frame_index=432,
        frames_until_action=8,
        decision_horizon_frames=8,
        scheduled_action="right",
    )
    state = model_state(s)
    assert state.get("observed_at_frame") == 432
    timing = state["reaction_timing"]
    assert timing.get("apply_at_frame") == 440
    assert timing.get("action_delay_frames") == 8
    assert timing.get("action_cycle_frames") == 8


@pytest.mark.parametrize("frame,delay,cycle", [(-1, 0, 8), (0, -1, 8), (0, 0, 0), (0, 1.5, 8)])
def test_invalid_frame_contract_is_rejected(frame, delay, cycle):
    from typesafe_mario.timing import DecisionTiming

    with pytest.raises(ValueError):
        DecisionTiming(frame, delay, cycle)
