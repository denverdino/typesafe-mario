"""Progress memory and bounded, forecast-safe exploration after repeated failure."""

from collections import Counter
from dataclasses import dataclass, field, replace
from random import Random
from typing import TYPE_CHECKING, Any

from .actions import Action

if TYPE_CHECKING:
    from .policy import Decision


@dataclass
class ProgressMemory:
    level: tuple[int, int, int] | None = None
    anchor_x: int = 0
    progress_frame: int = 0
    history: list[tuple[int, int, int, str | None]] = field(default_factory=list)

    def update(
        self, frame: int, level: tuple[int, int, int], x: int, y: int, action: str | None
    ) -> dict[str, Any]:
        if level != self.level or x >= self.anchor_x + 4:
            self.level, self.anchor_x, self.progress_frame = level, x, frame
            self.history.clear()
        self.history.append((frame, x, y, action))
        self.history = self.history[-96:]
        age = frame - self.progress_frame
        return {
            "active": age >= 48,
            "no_progress_frames": age,
            "anchor_x": self.anchor_x,
            "recent_max_height": max(h[2] for h in self.history),
            "action_frames": dict(Counter(h[3] for h in self.history if h[3] is not None)),
            "reason": "no_meaningful_forward_progress" if age >= 48 else None,
        }


def recover_decision(
    state: dict[str, Any], decision: "Decision", actions: tuple[Action, ...]
) -> "Decision":
    recovery = state.get("recovery", {})
    if not recovery.get("active"):
        return decision
    prediction = state.get("prediction", {})
    counts = recovery.get("action_frames", {})
    cycle = prediction.get("action_cycle_frames", 8)
    # A different proposal already explores; avoid replacing it merely for diversity.
    repeated = counts.get(decision.action.value, 0) >= 2 * cycle
    options = {}
    immediate_progress = False
    for forecast in prediction.get("action_forecasts", []):
        action = Action(forecast["action"])
        if action not in actions:
            continue
        safe = [b for b in forecast["continuations"].values() if b.get("risk") == "safe"]
        if not safe:
            continue
        if action == decision.action:
            first = forecast.get("first_cycle", safe[0].get("first_cycle", {}))
            immediate_progress = (
                first.get("max_x", 0) >= recovery.get("anchor_x", 0) + 4
                or first.get("max_height_y", 0) > recovery.get("recent_max_height", 0) + 4
            )

        def utility(branch):
            if branch.get("outcome") == "stage_clear":
                return 10000
            advance = branch.get("max_forward_progress_pixels", branch.get("progress_pixels", 0))
            height = max(0, branch.get("max_height_y", 0) - recovery.get("recent_max_height", 0))
            transition = any(e["type"] == "area_transition" for e in branch.get("events", []))
            return advance + min(32, height / 2) + 64 * transition

        options[action] = max(utility(b) for b in safe)
    if not options:
        return decision
    best = max(options.values())
    proposed_utility = options.get(decision.action, float("-inf"))
    if best > proposed_utility + 8:
        pool = [a for a, score in options.items() if score >= best - 4]
        reason = "forecast_progress_alternative"
    elif repeated and not immediate_progress:
        pool = [a for a in options if a != decision.action]
        reason = "try_undertried_safe_action"
    else:
        return decision
    if not pool:
        return decision
    weights = [
        (max(0, decision.probabilities.get(a.value, 0)) + 0.02) ** 0.5
        / (1 + counts.get(a.value, 0) / cycle) ** 2
        for a in pool
    ]
    seed = str((state.get("level"), prediction.get("observation_frame"), counts))
    selected = Random(seed).choices(pool, weights=weights, k=1)[0]
    return replace(
        decision,
        action=selected,
        confidence=decision.probabilities.get(selected.value, 0),
        selection={
            "mode": "recovery",
            "reason": reason,
            "proposed_action": decision.action.value,
            "proposed_confidence": decision.confidence,
            "selected_action": selected.value,
            "alternatives": {a.value: options[a] for a in pool},
        },
    )
