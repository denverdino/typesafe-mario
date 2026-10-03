from types import SimpleNamespace

from test_policy import Provider, policy

from typesafe_mario.actions import Action
from typesafe_mario.state import MarioStateParser


def test_main_decision_does_not_require_independent_diagnostic_answers():
    class ActionOnlyProvider(Provider):
        def system_one(self, **request):
            assert list(request["questions"]) == ["next_action"]
            return SimpleNamespace(
                choices={
                    "next_action": SimpleNamespace(
                        choice="right", confidence=0.8, probabilities={"right": 1.0}
                    )
                }
            )

    decision = policy(ActionOnlyProvider()).choose(MarioStateParser().parse({}), tuple(Action))
    assert decision.action == Action.RIGHT
    assert decision.jump_intent is None and decision.danger_score is None


def test_diagnostics_can_be_requested_explicitly():
    p = policy(Provider())
    p.diagnostic_questions = True
    result = p.choose(MarioStateParser().parse({}), tuple(Action))
    assert result.jump_intent == "release" and result.danger_score == 0.4
    assert set(p._client.request["questions"]) == {"next_action", "jump_intent", "danger"}
