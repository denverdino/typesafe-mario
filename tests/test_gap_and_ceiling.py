"""Recorded failures: do not mistake uncertain braking for a clear refuge."""

import json
from pathlib import Path

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.assessments import build_action_assessments
from typesafe_mario.dynamics import Dynamics
from typesafe_mario.observation import Actor, Block, Observation
from typesafe_mario.prediction import Predictor
from typesafe_mario.risk import assess_risk


def recorded_cases():
    return json.loads(
        (Path(__file__).parent / "fixtures/world1-1-gap-and-ceiling.json").read_text()
    )


def braking_case():
    return recorded_cases()["cases"][1]["checkpoints"][0]["prediction"]


def recorded_predictor(episode, decision):
    case = next(
        c
        for c in recorded_cases()["cases"][episode - 1]["checkpoints"]
        if c["decision"] == decision
    )
    predictor = Predictor()
    for h in case["history"]:
        observation = Observation(
            **dict(
                h,
                level=tuple(h["level"]),
                known_x=tuple(h["known_x"]),
                actors=tuple(Actor(**a) for a in h["actors"]),
                blocks=tuple(Block(**b) for b in h["blocks"]),
            )
        )
        predictor.observe(observation)
    predictor.dynamics = Dynamics()
    for name, values in case["samples"].items():
        for value in values:
            predictor.dynamics.add(name, value)
    predictor.fast_jump, predictor.bouncing = case["fast_jump"], case["bouncing"]
    return predictor, observation, case


def test_uncertain_braking_cannot_be_the_only_option_over_a_known_gap():
    prediction = braking_case()
    report = assess_risk(prediction, tuple(Action))
    assert {"left", "right_jump", "right_run_jump"} <= set(report["candidate_actions"])
    assert "uncertain_landing" in report["warnings"]["right_jump"]
    assert report["basis"] == "comparable_landing_uncertainty"


@pytest.mark.parametrize("episode,decision", [(2, 163), (4, 92)])
def test_rebuilt_forecast_keeps_the_native_gap_crossing_available(episode, decision):
    predictor, observation, case = recorded_predictor(episode, decision)
    prediction = predictor.forecast(
        observation,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["scheduled_action"]),
    )
    assert "right_jump" in assess_risk(prediction, tuple(Action))["candidate_actions"]
    assert abs(prediction["committed_future"]["x"] - case["actual_application"]["x"]) < 3


def test_walking_flight_speed_is_calibrated_from_saturated_observed_motion():
    from dataclasses import replace

    from test_observation_prediction import floor_scene

    model = Dynamics()
    history = []
    for frame in range(80):
        history.append(
            replace(
                floor_scene(),
                frame=frame,
                x=100 + int(1.75 * frame),
                y=150,
                vy=0,
                grounded=False,
                previous_action="right_jump",
            )
        )
        model.learn_air(history)
    assert 1.7 < model.value("air_walk_speed") < 1.8


@pytest.mark.parametrize("condition", ["accelerating", "reversing", "gap", "contact"])
def test_speed_cap_learning_rejects_non_plateau_or_discontinuous_motion(condition):
    from dataclasses import replace

    from test_observation_prediction import floor_scene

    history = [
        replace(
            floor_scene(),
            frame=f,
            x=100 + int(1.75 * f),
            y=150,
            grounded=False,
            previous_action="right_jump",
        )
        for f in range(9)
    ]
    if condition == "accelerating":
        history = [replace(h, x=100 + int(0.8 * f + 0.06 * f * f)) for f, h in enumerate(history)]
    elif condition == "reversing":
        history = [replace(h, previous_action="left") for h in history]
    elif condition == "gap":
        history[3] = replace(history[3], frame=2)
    else:
        history[-1] = replace(history[-1], actors=(Actor("a", "goomba", 114, 150),))
    model = Dynamics()
    model.learn_air(history)
    assert model.summary()["parameters"]["air_walk_speed"]["samples"] == 0


def test_descending_over_gap_before_visible_landing_is_uncertain_not_failed():
    predictor, observation, case = recorded_predictor(1, 148)
    prediction = predictor.forecast(
        observation,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["scheduled_action"]),
    )
    assessment = build_action_assessments(prediction, [Action.RIGHT_JUMP])[0]
    assert assessment["execution"]["risk"] == "uncertain"
    assert "unsupported_descent" in assessment["execution"]["warnings"]
    assert assessment["execution"]["first_landing"]["y"] == 79


def test_reversing_input_beneath_brick_cannot_claim_a_clear_running_jump():
    predictor, observation, case = recorded_predictor(3, 156)
    prediction = predictor.forecast(
        observation,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["scheduled_action"]),
    )
    report = assess_risk(prediction, tuple(Action))
    assert report["candidate_actions"] != ["right_run_jump"]
    branch = next(b for b in prediction["action_forecasts"] if b["action"] == "right_run_jump")
    assert branch["control_risk"]["uncertain_wall_contact"]
    assert prediction["committed_future"]["x_range"][0] <= case["actual_application"]["x"]


@pytest.mark.parametrize("episode,decision", [(5, 36), (5, 37), (6, 245)])
def test_rising_past_top_edge_does_not_invent_a_horizontal_stop(episode, decision):
    predictor, observation, case = recorded_predictor(episode, decision)
    prediction = predictor.forecast(
        observation,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["scheduled_action"]),
    )
    branch = next(b for b in prediction["action_forecasts"] if b["action"] == case["action"])
    assert abs(branch["first_cycle"]["x"] - case["actual_end"]["x"]) < 5


def test_reversal_without_overhead_geometry_does_not_invent_wall_uncertainty():
    from dataclasses import replace

    predictor, observation, case = recorded_predictor(3, 156)
    observation = replace(observation, blocks=tuple(b for b in observation.blocks if b.top <= 79))
    prediction = predictor.forecast(
        observation,
        tuple(Action),
        cycle=8,
        delay=8,
        scheduled_action=Action(case["scheduled_action"]),
    )
    assert not any(
        b["control_risk"]["uncertain_wall_contact"] for b in prediction["action_forecasts"]
    )


@pytest.mark.parametrize("episode,expected_x", [(1, 1485), (2, 1437), (4, 1446)])
def test_native_counterfactual_reaches_opposite_bank(episode, expected_x):
    from typesafe_mario.actions import ACTION_TO_INDEX
    from typesafe_mario.observation import observe
    from typesafe_mario.runner import create_mario_env
    from typesafe_mario.state import MarioStateParser

    pytest.importorskip("gym_super_mario_bros")
    data = recorded_cases()
    case = data["cases"][episode - 1]
    frame = case["checkpoints"][0]["observed_at_frame"] + 8
    buttons = [Action(a) for a, n in case["buttons"] for _ in range(n)]
    env = create_mario_env(data["env_id"], render_mode="rgb_array")
    try:
        _, info = env.reset(seed=data["seed"])
        parser, predictor = MarioStateParser(), Predictor()
        predictor.observe(observe(parser.parse(info, env.unwrapped.ram)))
        for f, action in enumerate(buttons[:frame], 1):
            _, reward, _, _, info = env.step(ACTION_TO_INDEX[action])
            if episode == 4:
                snapshot = parser.parse(
                    info, env.unwrapped.ram, previous_action=action, previous_reward=reward
                )
                predictor.observe(observe(snapshot))
                if f == frame - 8:
                    forecast = predictor.forecast(
                        observe(snapshot),
                        tuple(Action),
                        cycle=8,
                        delay=8,
                        scheduled_action=Action.RIGHT_JUMP,
                    )
                    forward = next(
                        b for b in forecast["action_forecasts"] if b["action"] == "right_jump"
                    )
                    assert forward["continuation"]["failure"] is None
                    assert "right_jump" in assess_risk(forecast, tuple(Action))["candidate_actions"]
        for _ in range(48):
            _, _, terminated, truncated, info = env.step(ACTION_TO_INDEX[Action.RIGHT_JUMP])
            assert not terminated and not truncated
        assert info["x_pos"] == expected_x and info["y_pos"] == 79
    finally:
        env.close()


@pytest.mark.parametrize("warning", ["possible_contact", "possible_fall", "unobserved_terrain"])
def test_landing_comparison_does_not_discharge_executable_cycle_hazards(warning):
    prediction = braking_case()
    branch = next(b for b in prediction["action_forecasts"] if b["action"] == "right_jump")
    branch["first_cycle_risk"][warning] = True
    report = assess_risk(prediction, tuple(Action))
    assert "right_jump" not in report["candidate_actions"]


def test_a_complete_clear_braking_route_still_excludes_uncertain_landings():
    prediction = braking_case()
    branch = next(b for b in prediction["action_forecasts"] if b["action"] == "left")
    branch["continuation"]["warnings"] = []
    assert assess_risk(prediction, tuple(Action))["candidate_actions"] == ["left"]


@pytest.mark.parametrize("change", ["incomplete", "contact", "conditional", "grounded"])
def test_landing_comparison_requires_a_resolved_ongoing_flight(change):
    prediction = braking_case()
    if change == "grounded":
        prediction["committed_future"]["grounded_estimate"] = True
    elif change == "conditional":
        prediction["committed_future"]["conditional"] = True
    else:
        for branch in prediction["action_forecasts"]:
            if branch["action"] in {"right_jump", "right_run_jump", "jump"}:
                if change == "incomplete":
                    branch["continuation"]["evaluated_frames"] = 8
                else:
                    branch["continuation"]["warnings"].append("possible_stomp")
    assert assess_risk(prediction, tuple(Action))["candidate_actions"] == ["left"]
