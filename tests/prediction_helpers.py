from dataclasses import replace

from typesafe_mario.actions import Action
from typesafe_mario.observation import observe
from typesafe_mario.prediction import Predictor


def predicted(snapshot, actions=tuple(Action), predictor=None):
    p = predictor or Predictor()
    o = observe(snapshot)
    if predictor is None:
        p.observe(o)
    result = p.forecast(
        o,
        actions,
        cycle=snapshot.decision_horizon_frames,
        delay=snapshot.frames_until_action,
        scheduled_action=Action(snapshot.scheduled_action) if snapshot.scheduled_action else None,
        scheduled_first_frame_action=Action(snapshot.scheduled_first_frame_action)
        if snapshot.scheduled_first_frame_action
        else None,
    )
    return replace(snapshot, prediction=result)


def assess_forecast(prediction):
    """Exercise arbitration separately from the predictor, for its forecasted actions."""
    from typesafe_mario.risk import assess_risk

    return assess_risk(
        prediction, tuple(Action(b["action"]) for b in prediction["action_forecasts"])
    )
