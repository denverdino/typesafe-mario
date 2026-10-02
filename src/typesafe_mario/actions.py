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


def first_frame_action(action: Action, *, grounded: bool, previous_action: str | None) -> Action:
    """Rearm A at a macro boundary, identically in execution and prediction."""
    if action in JUMP_ACTIONS and grounded and previous_action in JUMP_ACTIONS:
        return JUMP_RELEASE_ACTION[action]
    return action


ACTION_DESCRIPTIONS: dict[Action, str] = {
    Action.NOOP: "Release all controls; coast. Only for deliberate waiting, not a rising jump.",
    Action.RIGHT: "Walk right for precision positioning; releases A and shortens a rising jump. Choose this when precision_target_cleared to land on the intermediate step. "
    "Do not use to continue an ascent that still needs height.",
    Action.RIGHT_JUMP: "Continue holding A and right throughout a walking jump's ascent, "
    "or jump onto a nearby obstacle from ground. Keeps necessary jump height.",
    Action.RIGHT_RUN: "Default forward advance on clear supported ground: hold right and B. "
    "Releases A; not for maintaining a rising jump or missing an urgent takeoff.",
    Action.RIGHT_RUN_JUMP: "Continue holding a running jump during ascent, or take off now "
    "to clear an enemy/gap at its takeoff window. Preserve forward speed and jump height.",
    Action.JUMP: "Jump or hold A without direction when forward motion must be limited. "
    "Existing horizontal momentum can continue.",
    Action.LEFT: "Brake rightward momentum then move left, to reach a specific safe position. "
    "Releases A; braking takes time and cannot instantly evade contact.",
}
