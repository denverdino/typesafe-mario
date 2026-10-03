"""Replay the observed jump/release/wait loop at the paired staircase."""

import json
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, JUMP_ACTIONS, Action
from typesafe_mario.recovery import ProgressMemory
from typesafe_mario.risk import assess_risk


def recorded_case():
    return json.loads(
        (Path(__file__).parent / "fixtures/world1-1-interrupted-ascent.json").read_text()
    )


def loop_state():
    data = recorded_case()
    memory = ProgressMemory()
    for frame, x, y, action in data["observed_history"]:
        recovery = memory.update(frame, (1, 1, 1), x, y, action)
    state = data["state"]
    state["recovery"] = recovery
    return state


def test_vertical_motion_does_not_erase_horizontal_loop_evidence():
    recovery = loop_state()["recovery"]
    assert recovery["stationary_frames"] < 8
    assert recovery["local_motion"]["frames"] < 8
    assert recovery["horizontal_motion"]["frames"] >= 64
    assert recovery["horizontal_motion"]["x_range"] == [2226, 2226]
    assert recovery["horizontal_motion"]["y_range"][1] >= 130


@pytest.mark.parametrize("change", ["progress", "retreat", "gap", "warp", "area"])
def test_horizontal_loop_requires_contiguous_local_observations(change):
    memory = ProgressMemory()
    for frame in range(96):
        memory.update(frame, (1, 1, 1), 100, 79 + frame % 32, "right_jump")
    frame, x, level = 96, 100, (1, 1, 1)
    if change == "progress":
        x = 110
    elif change == "retreat":
        x = 80
    elif change == "gap":
        frame = 110
    elif change == "warp":
        x = 0
    else:
        level = (1, 1, 2)
    assert memory.update(frame, level, x, 79, "left")["horizontal_motion"]["frames"] == 0


def test_recorded_ascent_retains_clear_holds_and_explains_release_exclusion():
    state = loop_state()
    report = assess_risk(state["prediction"], tuple(Action), state=state)
    assert set(report["candidate_actions"]) == JUMP_ACTIONS
    assert report["recovery_filter"]["reason"] == "interrupted_ascent_with_clear_landing"
    assert "interrupted_ascent_with_clear_landing" in report["excluded_actions"]["noop"]


@pytest.mark.parametrize(
    "condition",
    [
        "brief",
        "descending",
        "grounded",
        "swimming",
        "released",
        "invalid",
        "no_wall",
        "contact",
        "fall",
        "ceiling",
        "unknown",
        "no_landing",
        "release_escape",
    ],
)
def test_recovery_does_not_require_holding_a_without_evidence(condition):
    state = loop_state()
    prediction = state["prediction"]
    committed = prediction["committed_future"]
    if condition == "brief":
        state["recovery"]["horizontal_motion"]["frames"] = 32
    elif condition == "descending":
        committed["vy"] = -2
    elif condition == "grounded":
        committed["grounded_estimate"] = True
    elif condition == "swimming":
        state["player"]["swimming"] = True
    elif condition == "released":
        state["reaction_timing"]["scheduled_action"] = "noop"
    elif condition == "invalid":
        committed["valid_for_application"] = False
    elif condition == "no_wall":
        state["terrain"]["blocks"] = []
    else:
        for branch in prediction["action_forecasts"]:
            if branch["action"] in JUMP_ACTIONS:
                if condition in {"contact", "fall", "unknown"}:
                    warning = {
                        "contact": "possible_contact",
                        "fall": "possible_fall",
                        "unknown": "unobserved_terrain",
                    }[condition]
                    branch["control_risk"][warning] = True
                    branch["first_cycle_risk"][warning] = True
                elif condition == "ceiling":
                    branch["continuation"]["warnings"] = ["uncertain_wall_contact"]
                elif condition == "no_landing":
                    branch["continuation"]["landings"] = []
            elif condition == "release_escape":
                branch["continuation"]["end"]["x"] = 2250
    report = assess_risk(prediction, tuple(Action), state=state)
    assert "noop" in report["candidate_actions"]
    assert (
        report.get("recovery_filter", {}).get("reason") != "interrupted_ascent_with_clear_landing"
    )


@pytest.mark.parametrize("action", [Action.NOOP, Action.RIGHT_JUMP])
def test_native_replay_distinguishes_releasing_from_finishing_the_jump(action):
    from dataclasses import replace

    from typesafe_mario.observation import observe
    from typesafe_mario.prediction import Predictor
    from typesafe_mario.runner import create_mario_env
    from typesafe_mario.state import MarioStateParser
    from typesafe_mario.tactical import build_tactical_state

    pytest.importorskip("gym_super_mario_bros")
    data = recorded_case()
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser, predictor = MarioStateParser(), Predictor()
        snapshot = parser.parse(info, env.unwrapped.ram)
        predictor.observe(observe(snapshot))
        frames = 0
        for name, count in data["buttons"]:
            for _ in range(count):
                _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action(name)])
                frames += 1
                if action == Action.RIGHT_JUMP:
                    snapshot = parser.parse(
                        info, env.unwrapped.ram, previous_action=name, previous_reward=reward
                    )
                    predictor.observe(observe(snapshot))
                    if frames == data["state"]["observed_at_frame"]:
                        forecast = predictor.forecast(
                            observe(snapshot),
                            tuple(Action),
                            cycle=8,
                            delay=8,
                            scheduled_action=Action.RIGHT_JUMP,
                        )
                        tactical = build_tactical_state(
                            replace(
                                snapshot,
                                prediction=forecast,
                                frames_until_action=8,
                                scheduled_action="right_jump",
                            ),
                            tuple(Action),
                        )
                        assert Action.NOOP not in tactical.candidates
                        assert Action.RIGHT_JUMP in tactical.candidates
                        assert tactical.state["recovery"]["horizontal_motion"]["frames"] >= 64
        assert (info["x_pos"], info["y_pos"]) == (2226, 132)
        for frame in range(48):
            _, _, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
            assert not terminated and not truncated
            if action == Action.RIGHT_JUMP and frame == 15:
                assert info["y_pos"] == 143 and int(env.unwrapped.ram[0x1D]) == 0
        if action == Action.NOOP:
            assert (info["x_pos"], info["y_pos"]) == (2226, 79)
        else:
            assert info["x_pos"] >= 2290
    finally:
        env.close()
