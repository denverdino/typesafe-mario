from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    NOOP = "noop"
    RIGHT = "right"
    RIGHT_JUMP = "right_jump"
    RIGHT_RUN = "right_run"
    RIGHT_RUN_JUMP = "right_run_jump"
    JUMP = "jump"
    LEFT = "left"


# These indices match gym_super_mario_bros.actions.SIMPLE_MOVEMENT.
ACTION_TO_INDEX: dict[Action, int] = {
    Action.NOOP: 0,
    Action.RIGHT: 1,
    Action.RIGHT_JUMP: 2,
    Action.RIGHT_RUN: 3,
    Action.RIGHT_RUN_JUMP: 4,
    Action.JUMP: 5,
    Action.LEFT: 6,
}

JUMP_ACTIONS = frozenset({Action.RIGHT_JUMP, Action.RIGHT_RUN_JUMP, Action.JUMP})

JUMP_RELEASE_ACTION: dict[Action, Action] = {
    Action.RIGHT_JUMP: Action.RIGHT,
    Action.RIGHT_RUN_JUMP: Action.RIGHT_RUN,
    Action.JUMP: Action.NOOP,
}


def first_frame_action(
    action: Action, *, grounded: bool, previous_action: str | None, swimming: bool = False
) -> Action:
    """Rearm A at a macro boundary, identically in execution and prediction."""
    if action in JUMP_ACTIONS and (grounded or swimming) and previous_action in JUMP_ACTIONS:
        return JUMP_RELEASE_ACTION[action]
    return action


ACTION_DESCRIPTIONS: dict[Action, str] = {
    Action.NOOP: "Release all buttons; existing momentum can continue.",
    Action.RIGHT: "Hold RIGHT; release A and B.",
    Action.RIGHT_JUMP: "Hold RIGHT and A; release B. Apply the boundary A-rearm rule.",
    Action.RIGHT_RUN: "Hold RIGHT and B; release A.",
    Action.RIGHT_RUN_JUMP: "Hold RIGHT, B and A. Apply the boundary A-rearm rule.",
    Action.JUMP: "Hold A with no direction or B; momentum can continue. Apply the boundary A-rearm rule.",
    Action.LEFT: "Hold LEFT; release A and B. Brake rightward momentum before moving left.",
}
