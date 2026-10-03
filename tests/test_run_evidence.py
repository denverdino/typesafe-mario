import io
import json
from dataclasses import replace

from typesafe_mario.actions import Action
from typesafe_mario.policy import Decision
from typesafe_mario.runner import EpisodeLog
from typesafe_mario.state import MarioStateParser


def record_cycle(frames=2, *, invalid_application=False):
    parser = MarioStateParser(decision_horizon_frames=2)
    s = parser.parse({"x_pos": 100, "y_pos": 79})
    prediction = {
        "backend": "observation_dynamics",
        "observed_at_frame": 0,
        "action_cycle_frames": 2,
        "committed_future": {"x": 99, "y": 79, "x_range": [98, 101], "y_range": [78, 80]},
        "action_forecasts": [
            {
                "action": "right",
                "first_cycle": {"x": 103, "y": 79, "x_range": [102, 104], "y_range": [78, 80]},
            }
        ],
    }
    observed = replace(s, prediction=prediction)
    if invalid_application:
        prediction["committed_future"]["valid_for_application"] = False
    output = io.StringIO()
    log = EpisodeLog(output)
    log.begin(0, observed, Decision(Action.RIGHT, 1, {}, 0), s, 0)
    for i in range(frames):
        s = parser.parse({"x_pos": 102 + i * 2, "y_pos": 79})
        log.step(Action.RIGHT, 1, s)
    log.end(s, False, False, "decision_limit")
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_each_decision_reports_prediction_error_and_preserves_one_full_forecast():
    decision, end = record_cycle()
    assert decision["prediction_comparison"]["application"]["absolute_error"] == {"x": 1, "y": 0}
    comparison = decision["prediction_comparison"]["executed_cycle"]
    assert comparison["absolute_error"] == {"x": 1, "y": 0}
    assert comparison["within_position_range"] is True
    assert "prediction" not in decision["debug_state"]["model_state"]
    assert decision["debug_state"]["prediction_details"]["action_forecasts"]
    assert end["recent_frames"][-1]["observed_at_frame"] == 2
    assert end["recent_frames"][-1]["actual_action"] == "right"


def test_partial_cycle_does_not_compare_early_position_with_full_cycle_forecast():
    decision, _ = record_cycle(frames=1)
    assert decision["prediction_comparison"]["executed_cycle"]["status"] == "not_comparable"


def test_invalid_application_also_invalidates_full_cycle_error_and_legacy_summary():
    from typesafe_mario.analysis import decision_comparisons

    decision, _ = record_cycle(invalid_application=True)
    assert decision["prediction_comparison"]["executed_cycle"]["status"] == "not_comparable"
    del decision["prediction_comparison"]
    assert decision_comparisons(decision)["executed_cycle"]["status"] == "not_comparable"


def test_event_trace_is_bounded_and_landing_preserves_held_button_evidence():
    parser = MarioStateParser(decision_horizon_frames=80)
    s = parser.parse({"x_pos": 100, "y_pos": 79})
    output = io.StringIO()
    log = EpisodeLog(output)
    log.begin(0, s, Decision(Action.JUMP, 1, {}, 0), s, 0)
    for i in range(80):
        s = replace(s, frame_index=i + 1, landed_this_frame=i == 76)
        log.step(Action.JUMP, 0, s)
    log.end(s, False, False, "decision_limit")
    decision, end = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(end["recent_frames"]) == 64
    trace = decision["execution"]["event_trace"]
    assert len(trace) == 64 and trace[-1]["observed_at_frame"] == 80
    assert decision["execution"]["observed_events"][0]["observed_at_frame"] == 77
    assert all(frame["actual_action"] == "jump" for frame in trace[-4:])


def test_position_comparison_rejects_invalid_committed_projection():
    from typesafe_mario.analysis import compare_position

    assert (
        compare_position({"x": 12, "y": 79, "valid_for_application": False}, {"x": 100, "y": 79})[
            "status"
        ]
        == "not_comparable"
    )
