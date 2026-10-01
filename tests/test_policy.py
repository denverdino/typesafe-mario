from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, Score

from typesafe_mario.actions import Action
from typesafe_mario.policy import TypeSafePolicy
from typesafe_mario.state import MarioStateParser


class Provider:
    def __init__(self, choice="right"):
        self.choice = choice
        self.request = None

    def system_one(self, **request):
        self.request = request
        return SimpleNamespace(
            choices={
                "next_action": SimpleNamespace(
                    choice=self.choice, confidence=0.8, probabilities={self.choice: 1.0}
                ),
                "jump_intent": SimpleNamespace(
                    choice="release",
                    confidence=0.8,
                    probabilities={"start": 0.1, "hold": 0.2, "release": 0.6, "none": 0.1},
                ),
            },
            scores={"danger": SimpleNamespace(score=0.4)},
        )


def policy(provider):
    p = TypeSafePolicy.__new__(TypeSafePolicy)
    p._Choice, p._Score, p._client = Choice, Score, provider
    return p


def test_jump_intent_is_parsed_without_overriding_controller_choice():
    provider = Provider()
    snapshot = MarioStateParser(decision_horizon_frames=3).parse({"x_pos": 100})
    decision = policy(provider).choose(snapshot, tuple(Action))
    assert decision.action == Action.RIGHT
    assert decision.jump_intent == "release"
    assert decision.jump_needed_probability == pytest.approx(0.3)
    assert set(provider.request["questions"]) == {"next_action", "jump_intent", "danger"}
    assert provider.request["state"]["reaction_timing"]["action_horizon_frames"] == 3


def test_provider_cannot_return_an_action_outside_supplied_candidates():
    with pytest.raises(ValueError, match="allowed"):
        policy(Provider("left")).choose(MarioStateParser().parse({}), (Action.RIGHT,))


def test_falling_offers_movement_choices_without_redundant_jump_buttons():
    from dataclasses import replace

    provider = Provider()
    falling = replace(MarioStateParser().parse({}), grounded=False, dy=-3, jump_phase="falling")
    result = policy(provider).choose(falling, tuple(Action))
    assert result.action == Action.RIGHT
    assert set(provider.request["questions"]["next_action"].criteria) == {
        "right",
        "right_run",
        "left",
        "noop",
    }
    rising = replace(falling, dy=1, jump_phase="rising")
    policy(provider).choose(rising, tuple(Action))
    assert "right_run_jump" in provider.request["questions"]["next_action"].criteria


def test_provider_cannot_return_a_filtered_falling_jump():
    from dataclasses import replace

    falling = replace(MarioStateParser().parse({}), grounded=False, dy=-3, jump_phase="falling")
    with pytest.raises(ValueError, match="allowed"):
        policy(Provider("right_jump")).choose(falling, tuple(Action))


def test_future_decision_keeps_jump_available_when_mario_may_land_before_application():
    from dataclasses import replace

    provider = Provider("right_jump")
    falling = replace(
        MarioStateParser().parse({}),
        grounded=False,
        dy=-3,
        jump_phase="falling",
        frames_until_action=8,
        scheduled_action="right",
        scheduled_first_frame_action="right",
    )
    result = policy(provider).choose(falling, tuple(Action))
    assert result.action == Action.RIGHT_JUMP
    timing = provider.request["state"]["reaction_timing"]
    assert timing["frames_until_action"] == 8
    assert timing["total_reaction_horizon_frames"] == 16


def landing_request_state():
    from dataclasses import replace

    snapshot = replace(
        MarioStateParser().parse({"x_pos": 1464, "y_pos": 132}),
        frames_until_action=8,
        grounded=False,
        dy=-5,
    )
    state = snapshot.to_state()
    state["trajectory"]["landing_projection"].update(frames=11, standing_y=79)
    state["hazard"]["upcoming_enemies"] = [
        {
            "kind": "goomba",
            "distance_pixels": 82,
            "relative_velocity_x": -3,
            "vertical_offset_pixels": 61,
        },
        {
            "kind": "goomba",
            "distance_pixels": 12,
            "relative_velocity_x": -3,
            "vertical_offset_pixels": -35,
        },
    ]
    return SimpleNamespace(
        frames_until_action=8,
        decision_horizon_frames=8,
        grounded=False,
        dy=-5,
        y=132,
        to_state=lambda: state,
    )


def test_questions_distinguish_safe_touchdown_from_insufficient_clearance_after_cycle():
    provider = Provider()
    policy(provider).choose(landing_request_state(), tuple(Action))
    reference = provider.request["questions"]["next_action"].instructions["landing_reference"]
    assert reference["application_frame"] == 8
    assert reference["cycle_end_frame"] == 16
    assert reference["touchdown_frame_estimate"] == 11
    enemy, elevated = reference["enemies"]
    assert enemy["distance_at_touchdown_pixels"] == 49
    assert enemy["distance_at_cycle_end_pixels"] == 34
    assert enemy["clearance_needed_pixels"] == 40
    assert enemy["inadequate_clearance_after_touchdown"] is True
    assert elevated["distance_at_cycle_end_pixels"] == -36
    assert elevated["inadequate_clearance_after_touchdown"] is False


@pytest.mark.parametrize(
    ("touchdown", "distance", "velocity", "height_offset", "expected"),
    [
        (11, 82, -3, 61, True),  # Safe at touchdown, too close after the cycle.
        (11, 12, -3, 61, False),  # Enemy already passed before landing.
        (20, 82, -3, 61, False),  # Touchdown is outside this action cycle.
        (11, 82, -3, 80, False),  # Enemy is on another vertical level.
        (11, 88, -3, 61, False),  # Exactly the conservative clearance margin remains.
    ],
)
def test_landing_reference_respects_time_direction_height_and_clearance(
    touchdown, distance, velocity, height_offset, expected
):
    snapshot = landing_request_state()
    state = snapshot.to_state()
    state["trajectory"]["landing_projection"]["frames"] = touchdown
    state["hazard"]["upcoming_enemies"] = [
        {
            "kind": "goomba",
            "distance_pixels": distance,
            "relative_velocity_x": velocity,
            "vertical_offset_pixels": height_offset,
        }
    ]
    provider = Provider()
    policy(provider).choose(snapshot, tuple(Action))
    instructions = provider.request["questions"]["next_action"].instructions
    assert ("landing_reference" in instructions) is expected


def test_unknown_landing_does_not_invent_enemy_clearance():
    snapshot = landing_request_state()
    snapshot.to_state()["trajectory"]["landing_projection"]["frames"] = None
    provider = Provider()
    policy(provider).choose(snapshot, tuple(Action))
    assert "landing_reference" not in provider.request["questions"]["next_action"].instructions


def test_landing_before_application_cannot_trigger_airborne_braking_instruction():
    snapshot = landing_request_state()
    snapshot.to_state()["trajectory"]["landing_projection"]["frames"] = 3
    provider = Provider()
    policy(provider).choose(snapshot, tuple(Action))
    assert "landing_reference" not in provider.request["questions"]["next_action"].instructions


def test_overhead_questions_are_local_to_the_threat_without_overriding_model_action():
    from dataclasses import replace

    from typesafe_mario.actions import ACTION_DESCRIPTIONS
    from typesafe_mario.state import EnemyObservation

    ordinary = MarioStateParser().parse({"x_pos": 100, "y_pos": 130})
    threatened = replace(
        ordinary,
        enemies=(EnemyObservation(0, 6, "goomba", -1, -44, 0, vertical_velocity_y=3),),
    )
    provider = Provider("right_run_jump")
    p = policy(provider)
    original_descriptions = dict(ACTION_DESCRIPTIONS)
    # Questions may recommend a release, but only the provider selects the action.
    assert p.choose(threatened, tuple(Action)).action == Action.RIGHT_RUN_JUMP
    assert "priority_0_overhead_escape" in provider.request["questions"]["next_action"].instructions
    assert provider.request["state"]["hazard"]["overhead_threats"][0]["relative_x_pixels"] == -1
    p.choose(ordinary, tuple(Action))
    assert (
        "priority_0_overhead_escape"
        not in provider.request["questions"]["next_action"].instructions
    )
    assert provider.request["questions"]["next_action"].criteria == {
        action.value: text for action, text in original_descriptions.items()
    }
    assert ACTION_DESCRIPTIONS == original_descriptions


def test_pipe_wait_guidance_preserves_provider_control_and_is_scoped_to_nearby_plant():
    from dataclasses import replace

    from typesafe_mario.actions import ACTION_DESCRIPTIONS
    from typesafe_mario.state import EnemyObservation

    ordinary = MarioStateParser().parse({"x_pos": 100, "y_pos": 79})
    plant = EnemyObservation(
        0,
        13,
        "piranha_plant",
        22,
        -40,
        plant={"phase": "extended", "pipe_top_y": 127},
    )
    provider = Provider("right_run_jump")
    p = policy(provider)
    assert (
        p.choose(replace(ordinary, enemies=(plant,)), tuple(Action)).action == Action.RIGHT_RUN_JUMP
    )
    question = provider.request["questions"]["next_action"]
    assert "waiting" in question.instructions
    assert "priority_1_jump_continuity" not in question.instructions
    assert set(question.criteria) == {a.value for a in Action}
    hidden = replace(plant, plant={"phase": "hidden", "pipe_top_y": 127})
    p.choose(replace(ordinary, enemies=(hidden,)), tuple(Action))
    assert "waiting" not in provider.request["questions"]["next_action"].instructions
    p.choose(ordinary, tuple(Action))
    assert "priority_0_pipe_plant" not in provider.request["questions"]["next_action"].instructions
    assert provider.request["questions"]["next_action"].criteria == {
        a.value: text for a, text in ACTION_DESCRIPTIONS.items()
    }


def test_exit_pipe_replaces_obstacle_jump_guidance_without_overriding_model_choice():
    from dataclasses import replace

    provider = Provider("left")
    snapshot = replace(
        MarioStateParser().parse({"x_pos": 2674, "y_pos": 159}),
        side_exit_pipe={
            "mouth_x": 2656,
            "entry_standing_y": 127,
            "retreat_target_x": 2632,
            "approach_supported": True,
        },
        frames_until_action=8,
        scheduled_action="right_jump",
        scheduled_first_frame_action="right",
    )
    result = policy(provider).choose(snapshot, tuple(Action))
    assert result.action == Action.LEFT
    request = provider.request
    assert request["state"]["terrain"]["side_exit_pipe"]["over_mouth"] is True
    assert request["state"]["reaction_timing"]["frames_until_action"] == 8
    question = request["questions"]["next_action"]
    assert "priority_3_progress" not in question.instructions
    assert "priority_1_jump_continuity" not in question.instructions
    assert "timing" in question.instructions
    assert set(question.criteria) == {action.value for action in Action}
    policy(provider).choose(replace(snapshot, side_exit_pipe=None), tuple(Action))
    assert "priority_3_progress" in provider.request["questions"]["next_action"].instructions
