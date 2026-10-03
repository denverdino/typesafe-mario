"""Build the provider projection and arbitrate its candidate set exactly once."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from . import risk
from .actions import Action
from .assessments import build_action_assessments, comparison_status
from .observation import model_state
from .state import MarioSnapshot

STATE_VERSION = 51


@dataclass(frozen=True)
class TacticalState:
    state: dict
    candidates: tuple[Action, ...]


def _actor_assessments(state: dict) -> list[dict]:
    """Summarize current visible evidence; this is not a collision forecast."""
    prediction = state.get("prediction") or {}
    forecasts = (
        {actor["id"]: actor for actor in prediction.get("actor_forecasts", ())}
        if prediction.get("observed_at_frame") == state["observed_at_frame"]
        and prediction.get("status") == "estimated"
        else {}
    )
    player = state["player"]
    assessments = []
    for actor in state["visible_actors"]:
        dx, dy = actor["x"] - player["x"], actor["y"] - player["y"]
        velocity = forecasts.get(actor["identity"], {}).get("observed_velocity", {})
        horizontal, vertical = "unknown", "unknown"
        trend, closing = "unknown", None
        known = velocity.get("available") is True
        if known:
            vx, vy = velocity["x"], velocity["y"]
            horizontal = "right" if vx > 0 else "left" if vx < 0 else "no_observed_displacement"
            vertical = "up" if vy > 0 else "down" if vy < 0 else "no_observed_displacement"
            if player["motion_reliable"] and dx != 0:
                closing = round(
                    (player["horizontal_speed_px_per_frame"] - vx) * (1 if dx > 0 else -1),
                    3,
                )
                trend = "closing" if closing > 0 else "opening" if closing < 0 else "unchanged"
        assessments.append(
            {
                "id": actor["identity"],
                "relative_position": {"x": dx, "y": dy},
                "motion_evidence": "observed_velocity_estimate" if known else "unavailable",
                "horizontal_motion": horizontal,
                "vertical_motion": vertical,
                "horizontal_separation": {
                    "trend": trend,
                    "closing_speed_px_per_frame": closing,
                },
            }
        )
    return assessments


def build_tactical_state(snapshot: MarioSnapshot, actions: Sequence[Action]) -> TacticalState:
    state = deepcopy(model_state(snapshot))
    state["schema_version"] = STATE_VERSION
    prediction = state.get("prediction")
    if prediction:
        timing = state["reaction_timing"]
        for key in (
            "observed_at_frame",
            "apply_at_frame",
            "action_delay_frames",
            "action_cycle_frames",
        ):
            if key in prediction and prediction[key] != timing[key]:
                raise ValueError(f"Inconsistent forecast timing: {key}")
    state["actor_assessments"] = _actor_assessments(state)
    candidates = tuple(actions)
    if not candidates:
        raise ValueError("At least one controller action is required")
    risk_control = risk.assess_risk(state.get("prediction"), candidates, state=state)
    candidates = tuple(Action(a) for a in risk_control["candidate_actions"])
    state["risk_control"] = risk_control
    assessments = build_action_assessments(prediction, candidates)
    state["action_assessments"] = assessments
    state["decision_context"] = {
        "allowed_actions": [a.value for a in candidates],
        "comparison_status": comparison_status(assessments, risk_control["status"]),
        "evidence_priority": ["execution", "held_input", "conditional_continuation"],
        "trajectory_frames": "offset_from_observed_at_frame",
        "continuation_score_scope": "Compare only within comparable execution-risk conditions",
    }
    if prediction:
        # Keep complete trajectories in local diagnostics, but send concise
        # first-cycle evidence and conditional plans to the action classifier.
        state["prediction"] = {
            k: prediction[k]
            for k in (
                "backend",
                "version",
                "status",
                "observed_at_frame",
                "apply_at_frame",
                "action_delay_frames",
                "frame_reference",
                "action_cycle_frames",
                "committed_future",
                "committed_interactions",
                "plan_followup",
                "continuation_search",
                "observed_interactions",
                "observed_boundary",
                "navigation_goal",
                "enemy_crossing",
                "recent_motion",
                "motion_sensitivity",
                "actor_forecasts",
                "validation",
            )
            if k in prediction
        }
    return TacticalState(state, candidates)
