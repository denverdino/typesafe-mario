"""The final staircase's right / wait / left / wait cycle must be escapable."""

import json
from pathlib import Path

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.recovery import ProgressMemory
from typesafe_mario.risk import assess_risk


def loop_memory(frames=96):
    memory = ProgressMemory()
    # Repeated four-pixel shuffling is not continuous stillness. The phase
    # deliberately ends at an extreme, where the old stationary counter resets.
    for frame in range(frames):
        phase = frame % 32
        x = 2990 + min(phase, 32 - phase) // 4
        action = ("right", "noop", "left", "noop")[phase // 8]
        recovery = memory.update(frame, (1, 1, 1), x, 191, action)
    return memory, recovery


def test_small_back_and_forth_motion_preserves_evidence_of_a_local_loop():
    memory, recovery = loop_memory()
    assert recovery["active"]
    assert recovery["stationary_frames"] < 48
    assert recovery["local_motion"]["frames"] >= 64
    assert recovery["local_motion"]["x_range"] == [2990, 2994]
    assert recovery["local_motion"]["y_range"] == [191, 191]
    # A jump, real run-up, observation gap or warp must invalidate the loop.
    for frame, x, y in ((96, 2990, 201), (96, 2970, 191), (110, 2990, 191), (96, 0, 79)):
        memory, _ = loop_memory()
        assert memory.update(frame, (1, 1, 1), x, y, "left")["local_motion"]["frames"] < 64


def recorded_cases():
    return json.loads(
        (Path(__file__).parent / "fixtures/world1-1-stair-loop.json").read_text()
    )["cases"]


@pytest.mark.parametrize("case", recorded_cases(), ids=lambda c: str(c["frame"]))
def test_recorded_stair_loop_offers_an_immediate_clear_jump(case):
    state = case["state"]
    _, state["recovery"] = loop_memory()
    report = assess_risk(state["prediction"], tuple(Action), state=state)
    assert "right_jump" in report["candidate_actions"]
    assert not {"noop", "right", "right_run", "left"} & set(report["candidate_actions"])


@pytest.mark.parametrize("warning", ["possible_fall", "possible_contact", "uncertain_takeoff"])
def test_loop_recovery_cannot_discharge_jump_risks(warning):
    state = recorded_cases()[0]["state"]
    _, state["recovery"] = loop_memory()
    for branch in state["prediction"]["action_forecasts"]:
        if "jump" in branch["action"]:
            branch["first_cycle_risk"][warning] = True
            branch["control_risk"][warning] = True
    report = assess_risk(state["prediction"], tuple(Action), state=state)
    assert "noop" in report["candidate_actions"]
    assert "right_jump" not in report["candidate_actions"]


def test_brief_shuffling_does_not_force_a_jump():
    state = recorded_cases()[0]["state"]
    _, state["recovery"] = loop_memory(32)
    report = assess_risk(state["prediction"], tuple(Action), state=state)
    assert "noop" in report["candidate_actions"]
    assert "left" in report["candidate_actions"]


def test_native_replay_climbs_out_of_the_recorded_loop():
    from dataclasses import replace

    from typesafe_mario.actions import ACTION_TO_INDEX, first_frame_action
    from typesafe_mario.observation import model_state, observe
    from typesafe_mario.prediction import Predictor
    from typesafe_mario.runner import create_mario_env
    from typesafe_mario.state import MarioStateParser

    pytest.importorskip("gym_super_mario_bros")
    data = json.loads(
        (Path(__file__).parent / "fixtures/world1-1-stair-loop.json").read_text()
    )
    buttons = [Action(a) for a, count in data["buttons"] for _ in range(count)]
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser, predictor = MarioStateParser(), Predictor()
        snapshot = parser.parse(info, env.unwrapped.ram)
        predictor.observe(observe(snapshot))

        def step(action):
            nonlocal snapshot
            _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
            snapshot = parser.parse(
                info, env.unwrapped.ram, previous_action=action, previous_reward=reward
            )
            predictor.observe(observe(snapshot))
            assert not terminated and not truncated and not snapshot.dead

        for action in buttons[:2256]:
            step(action)
        assert (snapshot.x, snapshot.y) == (2992, 191)
        forecast = predictor.forecast(
            observe(snapshot), tuple(Action), cycle=8, delay=8, scheduled_action=buttons[2256]
        )
        state = model_state(replace(snapshot, prediction=forecast))
        report = assess_risk(forecast, tuple(Action), state=state)
        assert "right_jump" in report["candidate_actions"]
        assert not {"noop", "right", "left"} & set(report["candidate_actions"])
        for action in buttons[2256:2264]:
            step(action)
        plan = next(b for b in forecast["action_forecasts"] if b["action"] == "right_jump")
        # Offline verification only: execute the estimated escape in the real
        # emulator. Runtime still replans and sends each choice to the provider.
        for name in plan["continuation"]["actions"]:
            action = Action(name)
            first = first_frame_action(
                action, grounded=snapshot.grounded, previous_action=snapshot.previous_action
            )
            for frame in range(8):
                step(first if frame == 0 else action)
        assert snapshot.x > 3004
        assert snapshot.y == 207
        assert snapshot.grounded
    finally:
        env.close()
