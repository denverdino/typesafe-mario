from collections import Counter

import pytest

from typesafe_mario.state import MarioStateParser


@pytest.mark.parametrize("positions", [(2, 65535, 65528, 8), (900, 902, 0, 8)])
def test_coordinate_wrap_or_same_area_warp_rebases_progress(positions):
    p = MarioStateParser()
    for x in positions:
        p.parse({"x_pos": x, "y_pos": 79})
    for x in range(9, 210):
        s = p.parse({"x_pos": x, "y_pos": 79})
    recovery = s.to_state()["recovery"]
    assert not recovery["active"]
    assert 205 <= recovery["anchor_x"] <= 209
    assert recovery["no_progress_frames"] < 4
    # A real stall after the rebase must still activate recovery.
    for _ in range(60):
        s = p.parse({"x_pos": 209, "y_pos": 79})
    assert s.to_state()["recovery"]["active"]


def test_gradual_retreat_does_not_reset_stall_deadline():
    p = MarioStateParser()
    for x in range(500, 299, -2):
        s = p.parse({"x_pos": x, "y_pos": 79}, previous_action="left")
    recovery = s.to_state()["recovery"]
    assert recovery["active"]
    assert recovery["anchor_x"] == 500
    assert recovery["no_progress_frames"] == 100


def test_repeated_jumps_and_small_horizontal_oscillation_trigger_recovery():
    p = MarioStateParser()
    for frame in range(80):
        s = p.parse(
            {"x_pos": 100 + frame % 3, "y_pos": 79 + frame % 32}, previous_action="right_run_jump"
        )
    recovery = s.to_state().get("recovery", {})
    assert recovery.get("active") is True
    assert recovery["no_progress_frames"] >= 48
    assert recovery["action_frames"]["right_run_jump"] >= 48


def test_meaningful_progress_and_area_transition_clear_recovery():
    p = MarioStateParser()
    for _ in range(60):
        s = p.parse({"x_pos": 100, "y_pos": 79})
    assert s.to_state()["recovery"]["active"]
    s = p.parse({"x_pos": 109, "y_pos": 79})
    assert not s.to_state()["recovery"]["active"]
    for _ in range(60):
        s = p.parse({"x_pos": 109, "y_pos": 79})
    s = p.parse({"x_pos": 10, "y_pos": 79, "area": 2})
    assert not s.to_state()["recovery"]["active"]
    p.reset()
    assert not p.parse({"x_pos": 100}).to_state()["recovery"]["active"]


def test_supported_plant_wait_is_intentional_but_not_unbounded():
    from dataclasses import replace

    from test_piranha import plant_frame

    s = plant_frame()
    s = replace(
        s,
        previous_action="noop",
        recovery={
            "active": True,
            "no_progress_frames": 80,
            "action_frames": {"noop": 80},
        },
    )
    assert not s.to_state()["recovery"]["active"]
    assert s.to_state()["recovery"]["intentional_wait"]
    s = replace(s, recovery={**s.recovery, "no_progress_frames": 200})
    assert s.to_state()["recovery"]["active"]


def test_safe_undertried_actions_can_override_repeated_high_probability_choice():
    from typesafe_mario.actions import Action
    from typesafe_mario.policy import Decision
    from typesafe_mario.recovery import recover_decision

    state = {
        "level": {"world": 2, "stage": 1, "area": 1},
        "recovery": {
            "active": True,
            "no_progress_frames": 80,
            "action_frames": {"right_run_jump": 80},
        },
        "prediction": {"observation_frame": 100, "action_forecasts": []},
    }
    for name, risk in [
        ("right_run_jump", "safe"),
        ("left", "safe"),
        ("noop", "safe"),
        ("right", "unsafe"),
    ]:
        state["prediction"]["action_forecasts"].append(
            {
                "action": name,
                "continuations": {
                    "repeat": {
                        "risk": risk,
                        "progress_pixels": 0,
                        "max_forward_progress_pixels": 0,
                        "events": [],
                    }
                },
            }
        )
    proposal = Decision(
        Action.RIGHT_RUN_JUMP,
        0.9,
        {"right_run_jump": 0.9, "left": 0.04, "noop": 0.03, "right": 0.03},
        1,
    )
    chosen = Counter()
    for frame in range(100, 140):
        state["prediction"]["observation_frame"] = frame
        result = recover_decision(state, proposal, tuple(Action))
        chosen[result.action] += 1
        assert result.action in (Action.LEFT, Action.NOOP)
        assert result.selection["proposed_action"] == "right_run_jump"
        assert result.probabilities == proposal.probabilities
        assert result == recover_decision(state, proposal, tuple(Action))
    assert len(chosen) == 2
    # A repeated action that actually gains new height next cycle must be allowed
    # to finish a useful launch; merely having a high future height is insufficient.
    state["prediction"]["action_forecasts"][0]["first_cycle"] = {"max_height_y": 240}
    state["recovery"]["recent_max_height"] = 180
    assert recover_decision(state, proposal, tuple(Action)) == proposal
    state["recovery"]["active"] = False
    assert recover_decision(state, proposal, tuple(Action)) == proposal


def test_recovery_does_not_sample_unsafe_or_unknown_alternatives():
    from typesafe_mario.actions import Action
    from typesafe_mario.policy import Decision
    from typesafe_mario.recovery import recover_decision

    s = {
        "recovery": {"active": True, "action_frames": {"right": 80}},
        "prediction": {
            "action_forecasts": [
                {"action": "left", "continuations": {"repeat": {"risk": "unsafe"}}},
                {"action": "noop", "continuations": {"repeat": {"risk": "unknown"}}},
            ]
        },
    }
    proposed = Decision(Action.RIGHT, 0.8, {"right": 0.8}, 1)
    assert recover_decision(s, proposed, tuple(Action)).action == Action.RIGHT
