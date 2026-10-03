from types import SimpleNamespace

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.policy import TypeSafePolicy
from typesafe_mario.state import MarioStateParser


class Provider:
    def __init__(self, choice="right"):
        self.choice = choice
        self.request = None

    def system_one(self, **request):
        self.request = request
        return SimpleNamespace(
            choices={
                "next_action": SimpleNamespace(
                    choice=self.choice, confidence=0.8, probabilities={self.choice: 1.0}
                ),
                "jump_intent": SimpleNamespace(
                    choice="release",
                    confidence=0.8,
                    probabilities={"start": 0.1, "hold": 0.2, "release": 0.6, "none": 0.1},
                ),
            },
            scores={"danger": SimpleNamespace(score=0.4)},
        )


def policy(provider):
    p = TypeSafePolicy.__new__(TypeSafePolicy)
    p._client = provider
    return p


def test_latency_only_measures_api_excluding_observation_preparation(monkeypatch):
    from typesafe_mario import policy as policy_module
    from typesafe_mario import tactical as module

    clock = [0.0]
    original = module.model_state

    def slow_state(snapshot):
        clock[0] += 5
        return original(snapshot)

    class TimedProvider(Provider):
        def system_one(self, **request):
            clock[0] += 0.25
            return super().system_one(**request)

    monkeypatch.setattr(module, "model_state", slow_state)
    monkeypatch.setattr(policy_module.time, "perf_counter", lambda: clock[0])
    assert (
        policy(TimedProvider()).choose(MarioStateParser().parse({}), tuple(Action)).latency_ms
        == 250
    )


def test_diagnostic_jump_answer_does_not_override_controller_choice():
    provider = Provider()
    s = MarioStateParser(decision_horizon_frames=3).parse({"x_pos": 100})
    p = policy(provider)
    p.diagnostic_questions = True
    d = p.choose(s, tuple(Action))
    assert d.action == Action.RIGHT and d.jump_intent == "release"
    assert d.jump_needed_probability == pytest.approx(0.3)
    assert provider.request["state"]["reaction_timing"]["action_cycle_frames"] == 3


def test_provider_cannot_select_disallowed_action():
    with pytest.raises(ValueError, match="allowed"):
        policy(Provider("left")).choose(MarioStateParser().parse({}), (Action.RIGHT,))


@pytest.mark.parametrize("risk", ["possible_contact", "possible_fall", "unknown"])
def test_forecasts_and_recovery_do_not_replace_provider_selection(risk):
    from dataclasses import replace

    s = replace(
        MarioStateParser().parse({}),
        recovery={"active": True, "action_frames": {"right": 80}},
        prediction={
            "backend": "observation_dynamics",
            "action_forecasts": [
                {"action": "right", "risk": risk},
                {"action": "left", "risk": "no_contact_predicted"},
            ],
        },
    )
    provider = Provider("right")
    d = policy(provider).choose(s, tuple(Action))
    assert d.action == Action.RIGHT and d.selection is None
    assert all(
        a["execution"]["risk"] == "unavailable"
        for a in provider.request["state"]["action_assessments"]
    )
    assert s.prediction["action_forecasts"][0]["risk"] == risk


def test_no_prediction_still_uses_visible_observations_and_keeps_future_jump_choices():
    from dataclasses import replace

    s = replace(
        MarioStateParser().parse({}),
        grounded=False,
        dy=-3,
        frames_until_action=8,
        scheduled_action="right",
    )
    provider = Provider("right_jump")
    assert policy(provider).choose(s, tuple(Action)).action == Action.RIGHT_JUMP
    assert set(provider.request["questions"]["next_action"]["criteria"]) == {
        a.value for a in Action
    }
    assert "enemy_tracks" not in provider.request["state"]


def test_native_forecast_cannot_leak_back_into_provider_input():
    from dataclasses import replace

    s = replace(
        MarioStateParser().parse({}),
        prediction={"backend": "native_checkpoint", "future_death_frame": 22},
    )
    provider = Provider()
    policy(provider).choose(s, tuple(Action))
    assert "prediction" not in provider.request["state"]


def test_provider_gets_compact_plans_without_mutating_diagnostic_forecasts():
    from dataclasses import replace

    from test_observation_prediction import floor_scene

    from typesafe_mario.prediction import Predictor

    prediction = Predictor().forecast(floor_scene(), tuple(Action), cycle=8)
    prediction["observed_interactions"] = [{"actor_id": "g", "inference": "defeated_goomba"}]
    s = replace(MarioStateParser().parse({}), prediction=prediction)
    provider = Provider("right")
    policy(provider).choose(s, tuple(Action))
    sent = provider.request["state"]["prediction"]
    assessments = provider.request["state"]["action_assessments"]
    assert "action_forecasts" not in sent
    assert "trajectory" not in assessments[0]
    assert "continuation" in assessments[0]
    assert sent["recent_motion"] == prediction["recent_motion"]
    assert sent["actor_forecasts"] == prediction["actor_forecasts"]
    assert sent["validation"] == prediction["validation"]
    assert (
        assessments[0]["execution"]["predicted_end"]
        == prediction["action_forecasts"][0]["first_cycle"]
    )
    assert sent["observed_interactions"] == [{"actor_id": "g", "inference": "defeated_goomba"}]
    assert "trajectory" in prediction["action_forecasts"][0]
    assert (
        assessments[0]["continuation"]["actions"]
        == prediction["action_forecasts"][0]["continuation"]["actions"]
    )


def test_open_crossing_evidence_reaches_provider_without_replacing_its_action():
    from dataclasses import replace

    from test_continuation import approaching_group_scene

    p, o = approaching_group_scene()
    prediction = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.NOOP)
    snapshot = replace(
        MarioStateParser().parse({"x_pos": 100, "y_pos": 79}),
        grounded=True,
        frame_index=o.frame,
        frames_until_action=8,
        scheduled_action="noop",
        prediction=prediction,
    )
    provider = Provider("right_jump")
    decision = policy(provider).choose(snapshot, tuple(Action))
    sent = provider.request["state"]["prediction"]["enemy_crossing"]
    assert sent["actor_ids"] == ["0", "1", "2"]
    assert sent["open_ground_x"] == [0, 237]
    assert decision.action == Action.RIGHT_JUMP
