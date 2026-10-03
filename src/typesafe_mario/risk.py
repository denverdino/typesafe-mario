"""Conservative candidate eligibility from observation-only forecast warnings."""

from .actions import Action


def assess_risk(prediction: dict | None, actions, *, state: dict | None = None) -> dict:
    names = [Action(a).value for a in actions]
    report = {
        "method": "observation_risk_filter",
        "status": "unavailable",
        "candidate_actions": names,
        "excluded_actions": {},
        "assumption": "Lower estimated risk, not verified safety. Check contacts and terrain "
        "through two control cycles after application, including room to brake. "
        "No emulator trials or automatic replacement of the provider's answer.",
    }
    if not prediction or prediction.get("backend") != "observation_dynamics":
        return report
    branches = {b["action"]: b for b in prediction.get("action_forecasts", ())}
    if not names or any(not branches.get(a, {}).get("control_risk") for a in names):
        return report
    reasons = {}
    for name in names:
        risk = branches[name]["control_risk"]
        reasons[name] = [
            k
            for k in (
                "possible_contact",
                "nominal_contact",
                "possible_fall",
                "uncertain_landing",
                "unobserved_terrain",
                "unreliable_motion",
                "uncertain_shell_kick",
                "uncertain_wall_contact",
                "uncertain_committed_interaction",
                "uncertain_takeoff",
                "landing_contact_window",
            )
            if risk.get(k)
        ]
    lower_risk = [name for name in names if not reasons[name]]
    if not lower_risk:
        # Broad uncertainty envelopes can cover every option. Preserve the
        # distinction between a possible overlap and a nominal body collision;
        # incomplete terrain or unresolved falls cannot qualify as a refuge.
        lower_risk = [name for name in names if reasons[name] == ["possible_contact"]]
        if not any(branches[name]["control_risk"].get("nominal_contact") for name in names):
            lower_risk = []
    report["through_frame"] = branches[names[0]]["control_risk"]["through_frame"]
    plans = {name: branches[name].get("continuation") for name in names}

    def complete_clear(plan):
        return bool(
            plan
            and plan["status"] == "estimated_viable"
            and not plan["failure"]
            and not plan["warnings"]
            and plan["evaluated_frames"]
            >= plan.get("horizon_frames", prediction.get("horizon_after_application_frames", 48))
        )

    for name, plan in plans.items():
        prefix = branches[name].get("first_cycle_risk")
        if (
            reasons[name]
            and set(reasons[name])
            <= {"possible_fall", "uncertain_landing", "landing_contact_window"}
            and complete_clear(plan)
            and prefix
            and not any(v is True for v in prefix.values())
        ):
            # Continuation contact checks use nominal bodies, not the wider
            # uncertainty envelopes. They cannot discharge a body-contact warning.
            # A late repeated-input landing margin can be discharged only by a
            # complete clear route whose executable first cycle is also clear.
            report.setdefault("repeated_input_warnings", {})[name] = reasons[name]
            reasons[name] = []
            if name not in lower_risk:
                lower_risk.append(name)
    if not lower_risk and all(plans.values()) and any(p["failure"] for p in plans.values()):
        # Repeating the first input forever can manufacture an avoidable risk.
        # Compare bounded, replannable continuations when they are available.
        # Retain warning-only paths; close calls/stomps are never a safety proof.
        plan_reasons = {
            name: ([plan["failure"]] if plan["failure"] else [])
            + [
                w
                for w in plan["warnings"]
                if w
                in (
                    "unresolved_flight",
                    "unreliable_motion",
                    "marginal_landing",
                    "uncertain_landing",
                    "unobserved_terrain",
                    "possible_stomp",
                    "unsupported_interaction",
                    "uncertain_wall_contact",
                    "uncertain_committed_interaction",
                    "uncertain_takeoff",
                    "landing_contact_window",
                )
            ]
            for name, plan in plans.items()
        }
        for name in names:
            prefix = branches[name].get("first_cycle_risk")
            if branches[name]["control_risk"].get("landing_contact_window") and not (
                complete_clear(plans[name])
                and prefix
                and not any(v is True for v in prefix.values())
            ):
                # The fallback must obey the same discharge rule as above.
                # A close call or unfinished route cannot clear a landing margin.
                plan_reasons[name].append("landing_contact_window")
            for warning in (
                "possible_fall",
                "unobserved_terrain",
                "unreliable_motion",
                "uncertain_landing",
                "uncertain_shell_kick",
                "uncertain_wall_contact",
                "uncertain_committed_interaction",
                "uncertain_takeoff",
                "landing_contact_window",
            ):
                if branches[name].get("first_cycle_risk", {}).get(warning):
                    # A later nominal route cannot undo an executable-cycle
                    # fall or supply missing terrain/motion evidence, establish
                    # support, or rule out a shell launch during this cycle.
                    plan_reasons[name].append(warning)
        lower_risk = [name for name in names if not plan_reasons[name]]
        if lower_risk:
            reasons = plan_reasons
            report["basis"] = "bounded_action_continuations"
            report["assumption"] = (
                "Candidate has an estimated lower-risk continuation, not verified safety. "
                "Only the first action is executed before observing and replanning. "
                "Unknown terrain, unresolved flight and model error remain limitations."
            )
            report["through_frame"] = max(
                plans[n]["evaluated_frames"] for n in names
            ) + prediction.get("application_frame", 0)
    if not lower_risk and all(
        branches[name]["control_risk"].get("nominal_contact")
        and branches[name].get("contact_exposure")
        for name in names
    ):
        # When no clear input exists, compare the whole executable cycle rather
        # than progress before the search's first contact. This is a mitigation
        # estimate, never a prediction that an overlapping body survives.
        for name in names:
            prefix = branches[name].get("first_cycle_risk", {})
            for warning in (
                "possible_fall",
                "uncertain_landing",
                "unobserved_terrain",
                "unreliable_motion",
                "uncertain_wall_contact",
                "uncertain_committed_interaction",
                "uncertain_takeoff",
                "landing_contact_window",
            ):
                if prefix.get(warning) and warning not in reasons[name]:
                    reasons[name].append(warning)
        comparable = [
            name for name in names if set(reasons[name]) <= {"possible_contact", "nominal_contact"}
        ]
        keys = ("frames", "overlap_area_sum", "end_overlap_area")

        def dominates(a, b):
            av, bv = branches[a]["contact_exposure"], branches[b]["contact_exposure"]
            return all(av[k] <= bv[k] for k in keys) and any(av[k] < bv[k] for k in keys)

        frontier = [n for n in comparable if not any(dominates(a, n) for a in comparable)]
        if len(frontier) < len(comparable):
            lower_risk = frontier
            report["basis"] = "contact_exposure_dominance"
            report["assumption"] = (
                "Every input predicts nominal contact. Retain inputs not dominated in first-cycle "
                "contact duration, summed overlap area and final overlap. This estimates reducing "
                "exposure, not survival; collision response and model errors remain unknown."
            )
            for n in comparable:
                if n not in frontier:
                    reasons[n].append("dominated_contact_exposure")
    # Immediate clearance is not enough when another eligible input has a
    # complete warning-free continuation. Do not trade that margin for the
    # extra distance of a future close call or unresolved interaction.
    if lower_risk and all(plans[name] for name in lower_risk):
        clear_plans = [name for name in lower_risk if complete_clear(plans[name])]
        if clear_plans and len(clear_plans) < len(lower_risk):
            for name in lower_risk:
                if name not in clear_plans:
                    plan = plans[name]
                    reasons[name] += ["continuation_" + w for w in plan["warnings"]]
                    if plan["failure"]:
                        reasons[name].append("continuation_" + plan["failure"])
                    if not reasons[name]:
                        reasons[name].append("continuation_incomplete")
            lower_risk = clear_plans
            report["continuation_preference"] = "complete_warning_free_estimate"
    if state and lower_risk:
        recovery, player = state.get("recovery", {}), state.get("player", {})
        stationary_inputs = recovery.get("stationary_action_frames", {})
        x, y = player.get("x", 0), player.get("y", 0)
        committed = prediction.get("committed_future", {})
        local = recovery.get("local_motion", {})
        near_wall = any(
            x + 8 <= b["left"] <= x + 32
            and b["bottom"] < y + player.get("height_pixels", 16)
            and b["top"] > y
            for b in state.get("terrain", {}).get("blocks", ())
        )
        if (
            recovery.get("active")
            and local.get("frames", 0) >= 64
            and sum(local.get("action_frames", {}).get(a, 0) for a in ("right", "right_run"))
            >= 16
            and player.get("grounded")
            and committed.get("grounded_estimate")
            and local["x_range"][0] - 8 <= committed.get("x", x) <= local["x_range"][1] + 8
            and abs(committed.get("y", y) - y) <= 2
            and near_wall
        ):
            # A completed future route cannot justify postponing its first
            # useful movement forever. Prefer a clear step out of this observed
            # loop now, but only if it also finishes on support beyond the stall.
            escapes = [
                name
                for name in lower_risk
                if complete_clear(plans[name])
                and branches[name].get("first_cycle_risk")
                and not any(v is True for v in branches[name]["first_cycle_risk"].values())
                and (
                    branches[name]["first_cycle"]["x"] > local["x_range"][1] + 4
                    or branches[name]["first_cycle"]["y"] > local["y_range"][1] + 4
                )
                and plans[name].get("end", {}).get("grounded_estimate")
                and plans[name]["end"]["x"] > recovery.get("anchor_x", x) + 4
            ]
            if escapes:
                excluded = [name for name in lower_risk if name not in escapes]
                for name in excluded:
                    reasons[name].append("observed_local_loop")
                lower_risk = escapes
                report["recovery_filter"] = {
                    "reason": "bounded_motion_with_clear_escape",
                    "local_motion": local,
                    "excluded_actions": excluded,
                }
        wall = any(
            x + 12 <= b["left"] <= x + 18
            and b["bottom"] < y + player.get("height_pixels", 16)
            and b["top"] > y
            for b in state.get("terrain", {}).get("blocks", ())
        )
        if (
            recovery.get("active")
            and recovery.get("stationary_frames", 0) >= 24
            and sum(stationary_inputs.get(a, 0) for a in ("right", "right_run")) >= 16
            and player.get("grounded")
            and committed.get("grounded_estimate")
            and abs(committed.get("x", x) - x) <= 2
            and abs(committed.get("y", y) - y) <= 2
            and wall
        ):
            moving = [
                name
                for name in lower_risk
                if complete_clear(plans[name])
                and branches[name].get("first_cycle_risk")
                and not any(v is True for v in branches[name]["first_cycle_risk"].values())
                and (
                    abs(branches[name]["first_cycle"]["x"] - x) >= 4
                    or abs(branches[name]["first_cycle"]["y"] - y) >= 4
                )
            ]
            stationary = [
                name
                for name in lower_risk
                if abs(branches[name]["first_cycle"]["x"] - x) <= 2
                and abs(branches[name]["first_cycle"]["y"] - y) <= 2
                and branches[name]["first_cycle"]["grounded_estimate"]
            ]
            if moving and stationary:
                for name in stationary:
                    reasons[name].append("observed_wall_stall")
                lower_risk = [name for name in lower_risk if name not in stationary]
                report["recovery_filter"] = {
                    "reason": "repeated_wall_input_without_motion",
                    "stationary_frames": recovery["stationary_frames"],
                    "excluded_actions": stationary,
                }
        if (
            "noop" in lower_risk
            and recovery.get("active")
            and recovery.get("stationary_frames", 0) >= 48
            and stationary_inputs.get("noop", 0) >= 40
            and player.get("grounded")
            and committed.get("grounded_estimate")
            and abs(committed.get("x", x) - x) <= 2
            and abs(committed.get("y", y) - y) <= 2
            and any(
                name != "noop"
                and complete_clear(plans[name])
                and branches[name].get("first_cycle_risk")
                and not any(v is True for v in branches[name]["first_cycle_risk"].values())
                and (
                    abs(branches[name]["first_cycle"]["x"] - x) >= 4
                    or abs(branches[name]["first_cycle"]["y"] - y) >= 4
                )
                for name in lower_risk
            )
        ):
            # Repeated waiting can indefinitely postpone the very movement
            # that makes its continuation score attractive. Reconsider it only
            # when actual idle history and a clear moving estimate both exist.
            lower_risk.remove("noop")
            reasons["noop"].append("prolonged_idle_with_clear_alternative")
            report["recovery_filter"] = {
                "reason": "prolonged_idle_with_clear_alternative",
                "stationary_frames": recovery["stationary_frames"],
                "excluded_actions": ["noop"],
            }
    report["warnings"] = reasons
    if not lower_risk:
        report.update(status="no_lower_risk_option", warnings=reasons)
    elif len(lower_risk) == len(names):
        report["status"] = "unrestricted"
    else:
        report.update(
            status="filtered",
            candidate_actions=lower_risk,
            excluded_actions={name: r for name, r in reasons.items() if name not in lower_risk},
        )
    return report
