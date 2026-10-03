import copy
import json
from pathlib import Path

import pytest

from typesafe_mario.actions import Action

CASES = json.loads((Path(__file__).parent / "fixtures/world1-2-v46-failures.json").read_text())[
    "cases"
]


def assess(case):
    from typesafe_mario.assessments import build_action_assessments

    return build_action_assessments(case["final_state"]["prediction"], tuple(Action))


def test_contact_during_committed_input_invalidates_all_later_action_comparisons():
    result = assess(CASES[0])
    assert all(a["execution"]["risk"] == "contact_before_application" for a in result)
    assert all(a["continuation"]["usable_for_comparison"] is False for a in result)


def test_conditional_stomp_cannot_erase_executable_cycle_body_contact():
    result = next(a for a in assess(CASES[1]) if a["action"] == "right_run_jump")
    assert result["execution"]["risk"] == "predicted_failure"
    assert "nominal_contact" in result["execution"]["warnings"]
    assert result["continuation"]["conditional"] is True
    assert result["continuation"]["status"] == "uncertain"


def test_assessment_only_contains_allowed_actions_and_does_not_mutate_forecast():
    from typesafe_mario.assessments import build_action_assessments

    prediction = copy.deepcopy(CASES[1]["final_state"]["prediction"])
    before = copy.deepcopy(prediction)
    result = build_action_assessments(prediction, (Action.LEFT,))
    assert [r["action"] for r in result] == ["left"]
    assert prediction == before


@pytest.mark.parametrize("prediction", [None, {"status": "unavailable"}])
def test_missing_forecast_is_unknown_not_safe(prediction):
    from typesafe_mario.assessments import build_action_assessments

    result = build_action_assessments(prediction, (Action.NOOP,))[0]
    assert result["execution"]["risk"] == "unavailable"
    assert result["continuation"]["usable_for_comparison"] is False


def test_contacts_keep_actor_identity_and_separate_executable_and_later_windows():
    from typesafe_mario.assessments import build_action_assessments

    prediction = {
        "status": "estimated",
        "action_forecasts": [
            {
                "action": "right",
                "first_cycle_risk": {"through_frame": 16, "possible_contact": True},
                "control_risk": {"through_frame": 24, "possible_contact": True},
                "hazards": [
                    {"id": "front", "kind": "goomba", "possible_contact_frame": 12},
                    {"id": "returning", "kind": "moving_shell", "possible_contact_frame": 22},
                    {"id": "distant", "kind": "goomba", "possible_contact_frame": 40},
                ],
            }
        ],
    }
    a = build_action_assessments(prediction, (Action.RIGHT,))[0]
    assert a["execution"]["contacts"] == [prediction["action_forecasts"][0]["hazards"][0]]
    assert [c["id"] for c in a["held_input"]["contacts"]] == ["front", "returning"]
    a["execution"]["contacts"][0]["id"] = "mutated"
    assert prediction["action_forecasts"][0]["hazards"][0]["id"] == "front"
