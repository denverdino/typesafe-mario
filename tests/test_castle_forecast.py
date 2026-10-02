"""World 1-4 regressions reduced from run-20261002T021302.140176Z."""

from dataclasses import replace

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action, first_frame_action
from typesafe_mario.prediction import forecast_actions
from typesafe_mario.runner import create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def castle():
    pytest.importorskip("gym_super_mario_bros")
    env = create_mario_env("SuperMarioBros-1-4-v0", render_mode="rgb_array")
    try:
        _, info = env.reset(seed=123)
        parser = MarioStateParser(decision_horizon_frames=8)
        s = parser.parse(info, env.unwrapped.ram)
        # The repeated restart path holds A through the descending entrance stairs.
        for frame in range(48):
            action = (
                first_frame_action(
                    Action.RIGHT_RUN_JUMP, grounded=s.grounded, previous_action=s.previous_action
                )
                if frame % 8 == 0
                else Action.RIGHT_RUN_JUMP
            )
            _, reward, _, _, info = env.step(ACTION_TO_INDEX[action])
            s = parser.parse(
                info, env.unwrapped.ram, previous_action=action.value, previous_reward=reward
            )
        assert (s.x, s.y) == (124, 160)
        yield (
            env,
            parser,
            replace(
                s,
                frames_until_action=8,
                scheduled_action="right_run_jump",
                scheduled_first_frame_action="right_run_jump",
            ),
        )
    finally:
        env.close()


def test_airborne_horizon_does_not_hide_death_one_frame_later(castle):
    env, parser, s = castle
    result = forecast_actions(env, parser, s, (Action.RIGHT_RUN_JUMP, Action.LEFT))
    branch = result.to_state()["prediction"]["action_forecasts"][0]["continuations"]["repeat"]
    assert branch["outcome"] == "death"
    assert branch["risk"] == "unsafe"
    assert branch["terminal_frame"] == 57  # 8 committed + 49 candidate frames.
    brake = result.prediction["action_forecasts"][1]["continuations"]["repeat"]
    assert brake["risk"] == "safe"
    assert brake["end_player"]["grounded"]


def test_safe_release_route_is_used_before_stall_recovery(castle):
    from test_policy import Provider, policy

    env, parser, s = castle
    # Observation 56; committed cycle ends at x=172, descending toward the pit lip.
    for _ in range(8):
        _, reward, _, _, info = env.step(ACTION_TO_INDEX[Action.RIGHT_RUN_JUMP])
        s = parser.parse(
            info, env.unwrapped.ram, previous_action="right_run_jump", previous_reward=reward
        )
    s = replace(
        s,
        frames_until_action=8,
        scheduled_action="right_run_jump",
        scheduled_first_frame_action="right_run_jump",
    )
    result = forecast_actions(env, parser, s, tuple(Action))
    assert not result.to_state()["recovery"]["active"]
    decision = policy(Provider("right_run_jump")).choose(result, tuple(Action))
    assert decision.action in (Action.NOOP, Action.RIGHT, Action.RIGHT_RUN, Action.LEFT)
    assert decision.selection["mode"] == "forecast_safety"
    assert decision.selection["proposed_action"] == "right_run_jump"
    assert decision.probabilities == {"right_run_jump": 1.0}


def test_unresolved_flight_is_bounded_and_unknown(monkeypatch):
    from typesafe_mario import prediction

    s = replace(MarioStateParser().parse({}), grounded=False)
    calls = []

    def airborne_step(env, parser, action):
        calls.append(action)
        return s, False

    monkeypatch.setattr(prediction, "_step", airborne_step)
    branch, trace = prediction._rollout(
        None, None, s, Action.JUMP, "repeat", delay=8, cycle=8, horizon=48
    )
    assert len(calls) == 96
    assert branch["risk"] == "unknown"
    assert not branch["landing_resolved"]
    assert trace["samples"][-1]["frame"] == 104


@pytest.mark.parametrize(
    "proposed_risks,alternative_risk,allowed,expected",
    [
        (["unsafe", "unsafe"], "safe", (Action.RIGHT_RUN_JUMP, Action.RIGHT), Action.RIGHT),
        (
            ["unsafe", "unknown"],
            "safe",
            (Action.RIGHT_RUN_JUMP, Action.RIGHT),
            Action.RIGHT_RUN_JUMP,
        ),
        (["unsafe", "safe"], "safe", (Action.RIGHT_RUN_JUMP, Action.RIGHT), Action.RIGHT_RUN_JUMP),
        (["unsafe"], "unknown", (Action.RIGHT_RUN_JUMP, Action.RIGHT), Action.RIGHT_RUN_JUMP),
        (["unsafe"], "unsafe", (Action.RIGHT_RUN_JUMP, Action.RIGHT), Action.RIGHT_RUN_JUMP),
        (["unsafe"], "safe", (Action.RIGHT_RUN_JUMP,), Action.RIGHT_RUN_JUMP),
        ([], "safe", (Action.RIGHT_RUN_JUMP, Action.RIGHT), Action.RIGHT_RUN_JUMP),
    ],
)
def test_guard_requires_unsafe_proposal_and_allowed_safe_alternative(
    proposed_risks, alternative_risk, allowed, expected
):
    from typesafe_mario.policy import Decision
    from typesafe_mario.recovery import recover_decision

    state = {
        "recovery": {"active": False},
        "prediction": {
            "action_forecasts": [
                {
                    "action": "right_run_jump",
                    "continuations": {
                        str(i): {"risk": risk} for i, risk in enumerate(proposed_risks)
                    },
                },
                {"action": "right", "continuations": {"run_jump": {"risk": alternative_risk}}},
            ]
        },
    }
    proposed = Decision(Action.RIGHT_RUN_JUMP, 0.9, {"right_run_jump": 0.9, "right": 0.1}, 0)
    result = recover_decision(state, proposed, allowed)
    assert result.action == expected
    if expected == proposed.action:
        assert result is proposed
    else:
        assert result.confidence == 0.1
        assert result.selection["proposed_confidence"] == 0.9
