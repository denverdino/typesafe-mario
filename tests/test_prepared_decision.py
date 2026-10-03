import json
from dataclasses import replace

import pytest
from test_observation_prediction import floor_scene
from test_policy import Provider, policy

from typesafe_mario.actions import Action
from typesafe_mario.prediction import Predictor
from typesafe_mario.state import MarioStateParser


def snapshot_with_prediction():
    return replace(
        MarioStateParser().parse({}),
        prediction=Predictor().forecast(floor_scene(), tuple(Action), cycle=8),
        recovery={"active": True, "stationary_frames": 60},
    )


def test_preparation_arbitrates_once_with_recovery_and_predictor_only_produces_evidence(
    monkeypatch,
):
    from typesafe_mario import risk

    calls = []
    original = risk.assess_risk

    def assess(prediction, actions, *, state=None):
        calls.append(state)
        return original(prediction, actions, state=state)

    monkeypatch.setattr(risk, "assess_risk", assess)
    s = snapshot_with_prediction()
    assert "risk_control" not in s.prediction
    prepared = policy(Provider()).prepare(s, tuple(Action))
    assert len(calls) == 1
    assert calls[0]["recovery"]["stationary_frames"] == 60
    payload = prepared.request.to_payload()
    assert list(payload["questions"]["next_action"]["criteria"]) == [
        a.value for a in prepared.candidates
    ]
    assert payload["state"]["risk_control"]["candidate_actions"] == [
        a.value for a in prepared.candidates
    ]


def test_captured_request_is_compact_and_survives_source_and_transport_mutation():
    provider = Provider()
    p = policy(provider)
    s = snapshot_with_prediction()
    prepared = p.prepare(s, tuple(Action))
    payload = prepared.request.to_payload()
    assert "action_forecasts" not in payload["state"]["prediction"]
    assert "trajectory" not in payload["state"]["action_assessments"][0]
    assert "trajectory" in s.prediction["action_forecasts"][0]
    captured = json.dumps(payload, sort_keys=True)
    s.prediction["action_forecasts"].clear()
    payload["state"]["player"]["x"] = -999
    decision = p.resolve(prepared)
    assert decision.action == Action.RIGHT
    assert json.dumps(provider.request, sort_keys=True) == captured
    provider.request["questions"].clear()
    assert json.dumps(prepared.request.to_payload(), sort_keys=True) == captured


def test_legal_response_is_checked_against_prepared_candidates():
    p = policy(Provider("left"))
    prepared = p.prepare(MarioStateParser().parse({}), (Action.RIGHT,))
    with pytest.raises(ValueError, match="allowed"):
        p.resolve(prepared)


def test_nonfinite_state_is_rejected_before_provider_dispatch():
    provider = Provider()
    p = policy(provider)
    with pytest.raises(ValueError):
        p.prepare(replace(MarioStateParser().parse({}), x=float("nan")), tuple(Action))
    assert provider.request is None


@pytest.mark.parametrize(
    "field,value", [("observed_at_frame", 10), ("apply_at_frame", 8), ("action_cycle_frames", 4)]
)
def test_stale_or_inconsistent_forecast_is_rejected_before_dispatch(field, value):
    provider = Provider()
    s = snapshot_with_prediction()
    s.prediction[field] = value
    with pytest.raises(ValueError, match="forecast timing"):
        policy(provider).prepare(s, tuple(Action))
    assert provider.request is None
