"""Offline run summaries. Reads additive schema-2 logs without a game or provider."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def compare_position(estimate: dict | None, actual: dict, *, eligible=True) -> dict:
    if not eligible or not estimate or not estimate.get("valid_for_application", True):
        return {"status": "not_comparable"}
    if not all(k in estimate and k in actual for k in ("x", "y")):
        return {"status": "unavailable"}
    covered = None
    if all(k + "_range" in estimate for k in ("x", "y")):
        covered = all(
            estimate[k + "_range"][0] <= actual[k] <= estimate[k + "_range"][1] for k in ("x", "y")
        )
    return {
        "status": "compared",
        "predicted": {k: estimate[k] for k in ("x", "y")},
        "actual": {k: actual[k] for k in ("x", "y")},
        "absolute_error": {k: round(abs(actual[k] - estimate[k]), 3) for k in ("x", "y")},
        "within_position_range": covered,
    }


def decision_comparisons(record: dict) -> dict:
    if "prediction_comparison" in record:
        return record["prediction_comparison"]
    state, execution = record.get("state", {}), record.get("execution", {})
    prediction = state.get("prediction", {})
    branch = next(
        (b for b in prediction.get("action_forecasts", ()) if b["action"] == record.get("action")),
        {},
    )
    return {
        "application": compare_position(
            prediction.get("committed_future"), execution.get("start_state", {}).get("player", {})
        ),
        "executed_cycle": compare_position(
            branch.get("first_cycle"),
            record.get("result_state", {}).get("player", {}),
            eligible=execution.get("frames") == prediction.get("action_cycle_frames")
            and "action_cycle_frames" in prediction
            and prediction.get("committed_future", {}).get("valid_for_application", True),
        ),
    }


def _aggregate(comparisons):
    samples = [c for c in comparisons if c.get("status") == "compared"]
    coverage = [
        c["within_position_range"] for c in samples if c.get("within_position_range") is not None
    ]
    return {
        "samples": len(samples),
        "mean_absolute_error": {
            k: round(mean(c["absolute_error"][k] for c in samples), 3) if samples else None
            for k in ("x", "y")
        },
        "range_samples": len(coverage),
        "range_coverage": round(mean(coverage), 3) if coverage else None,
    }


def summarize_run(path: Path) -> dict:
    episodes = defaultdict(list)
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                episodes[row["episode_id"]].append(row)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid run record at {path}:{line_no}") from exc
    summaries = []
    for identity, rows in episodes.items():
        decisions = [r for r in rows if r["record_type"] == "decision"]
        requests = [r for r in rows if r["record_type"] == "policy_request"]
        ends = [r for r in rows if r["record_type"] == "episode_end"]
        last = ends[-1] if ends else (decisions[-1] if decisions else {})
        result = last.get("result_state", {})
        applied = {
            r.get("request_id") for r in decisions if r.get("execution", {}).get("frames", 0) > 0
        }
        finalized = {r.get("request_id") for r in decisions}
        applied.update(
            r.get("request_id")
            for r in rows
            if r["record_type"] == "request_outcome" and r.get("status") == "applied"
        )
        comparisons = [decision_comparisons(d) for d in decisions]
        latency = [d["latency_ms"] for d in decisions if d.get("latency_ms") is not None]
        sizes = [
            len(json.dumps(r["request"]["payload"], separators=(",", ":")).encode())
            for r in requests
            if "request" in r
        ]
        summaries.append(
            {
                "episode_id": identity,
                "run_config": rows[0].get("run_config", {}),
                "outcome": last.get("reason", "incomplete") if ends else "incomplete",
                "stage_clear": bool(ends and result.get("episode", {}).get("stage_clear")),
                "decisions": len(decisions),
                "requests": len(requests),
                "frames": sum(d.get("execution", {}).get("frames", 0) for d in decisions),
                "final_x": result.get("player", {}).get("x"),
                "furthest_x": max(
                    (
                        d.get("result_state", {})
                        .get("episode", {})
                        .get(
                            "best_progress", d.get("result_state", {}).get("player", {}).get("x", 0)
                        )
                        for d in decisions
                    ),
                    default=None,
                ),
                "unapplied_request_ids": [
                    r["request_id"]
                    for r in requests
                    if r.get("request_id") and r["request_id"] not in applied
                ],
                "unfinalized_applied_request_ids": sorted(applied - finalized - {None}),
                "frame_count_scope": "finalized_decision_records",
                "request_outcomes": dict(
                    Counter(r["status"] for r in rows if r["record_type"] == "request_outcome")
                ),
                "risk_statuses": dict(
                    Counter(
                        d.get("state", {}).get("risk_control", {}).get("status", "unavailable")
                        for d in decisions
                    )
                ),
                "single_candidate_decisions": sum(
                    len(d.get("state", {}).get("risk_control", {}).get("candidate_actions", ()))
                    == 1
                    for d in decisions
                ),
                "recovery_decisions": sum(
                    bool(d.get("state", {}).get("recovery", {}).get("active")) for d in decisions
                ),
                "median_api_latency_ms": round(median(latency), 2) if latency else None,
                "median_request_bytes": median(sizes) if sizes else None,
                "prediction_comparisons": {
                    key: _aggregate(c.get(key, {}) for c in comparisons)
                    for key in ("application", "executed_cycle")
                },
            }
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "episodes": summaries,
        "cleared_episodes": sum(e["stage_clear"] for e in summaries),
    }


def format_summary(run: dict) -> str:
    lines = [f"{run['path']}: {run['cleared_episodes']}/{len(run['episodes'])} episodes cleared"]
    for e in run["episodes"]:
        lines.append(
            f"  {e['episode_id']}: {e['outcome']}; decisions={e['decisions']}, "
            f"frames={e['frames']}, furthest_x={e['furthest_x']}, final_x={e['final_x']}"
        )
        lines.append(
            f"    risk={e['risk_statuses']}; recovery={e['recovery_decisions']}; "
            f"unapplied_requests={len(e['unapplied_request_ids'])}"
        )
        for key, comparison in e["prediction_comparisons"].items():
            lines.append(
                f"    {key}: samples={comparison['samples']}, "
                f"MAE={comparison['mean_absolute_error']}, "
                f"range_coverage={comparison['range_coverage']}"
            )
    return "\n".join(lines)
