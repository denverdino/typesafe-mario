"""One provider-facing interpretation of forecasts, without changing eligibility."""

from copy import deepcopy


def _warnings(risk):
    return [key for key, value in risk.items() if value is True]


def _contacts(branch, through_frame):
    return deepcopy(
        [
            contact
            for contact in branch.get("hazards", ())
            if through_frame is not None and contact["possible_contact_frame"] <= through_frame
        ]
    )


def build_action_assessments(prediction: dict | None, candidates) -> list[dict]:
    prediction = prediction or {}
    branches = {b["action"]: b for b in prediction.get("action_forecasts", ())}
    committed = prediction.get("committed_future", {})
    blocked = committed.get("valid_for_application") is False
    assessments = []
    for action in candidates:
        branch = branches.get(str(action), {})
        prefix = branch.get("first_cycle_risk", {})
        warnings = _warnings(prefix)
        if blocked:
            risk = (
                "contact_before_application"
                if committed.get("contact_failure") == "nominal_contact"
                else "invalid_application_state"
            )
        elif not prefix or prediction.get("status") != "estimated":
            risk = "unavailable"
        elif prefix.get("nominal_contact") or prefix.get("possible_fall"):
            risk = "predicted_failure"
        elif warnings:
            risk = "uncertain"
        else:
            risk = "no_warning_estimated"
        plan = branch.get("continuation", {})
        conditional = bool(plan.get("interactions")) or any(
            w in plan.get("warnings", ())
            for w in (
                "possible_stomp",
                "uncertain_bounce",
                "possible_stationary_shell",
                "possible_shell_kick",
            )
        )
        assessments.append(
            {
                "action": str(action),
                "execution": {
                    "risk": risk,
                    "warnings": warnings,
                    "through_frame_offset": prefix.get("through_frame"),
                    "predicted_end": deepcopy(branch.get("first_cycle")),
                    "contact_exposure": deepcopy(branch.get("contact_exposure")),
                    "contacts": _contacts(branch, prefix.get("through_frame")),
                    "first_landing": deepcopy(branch.get("first_landing")),
                },
                "held_input": {
                    "warnings": _warnings(branch.get("control_risk", {})),
                    "through_frame_offset": branch.get("control_risk", {}).get("through_frame"),
                    "contacts": _contacts(
                        branch, branch.get("control_risk", {}).get("through_frame")
                    ),
                },
                "continuation": {
                    **deepcopy(
                        {
                            k: plan[k]
                            for k in (
                                "status",
                                "actions",
                                "evaluated_frames",
                                "horizon_frames",
                                "failure",
                                "warnings",
                                "progress",
                                "score",
                                "landings",
                                "interactions",
                            )
                            if k in plan
                        }
                    ),
                    "conditional": conditional,
                    "usable_for_comparison": bool(plan) and not blocked,
                },
            }
        )
    return assessments


def comparison_status(assessments: list[dict], filter_status: str) -> str:
    risks = {a["execution"]["risk"] for a in assessments}
    if risks & {"contact_before_application", "invalid_application_state"}:
        return "invalid_application_state"
    if risks == {"predicted_failure"}:
        return "all_executable_cycles_predict_failure"
    if risks & {"unavailable"} or any(
        set(a["execution"]["warnings"]) & {"unobserved_terrain", "unreliable_motion"}
        for a in assessments
    ):
        return "insufficient_evidence"
    if filter_status == "no_lower_risk_option":
        return "incomparable_risks"
    return "eligible_candidates_not_verified_safe"
