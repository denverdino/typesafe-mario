"""Observed falls must not retain the certainty of the preceding level walk."""

import json
from pathlib import Path

import pytest
from test_observation_prediction import floor_scene

from typesafe_mario.actor_motion import actor_path
from typesafe_mario.observation import Actor, Block


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/observed-enemy-falls.json").read_text())["cases"],
    ids=lambda c: str(c["decision"]),
)
def test_leaving_visible_support_invalidates_confidence_in_level_velocity(case):
    actor = Actor(**case["actor"])
    terrain = case["terrain"]
    scene = floor_scene(
        known_x=tuple(terrain["known_x"]),
        blocks=tuple(Block(**b) for b in terrain["blocks"]),
        actors=(actor,),
    )
    velocity = case["velocity"]
    path = actor_path(actor, (velocity["x"], velocity["y"], True), scene, 24)
    # The real observed heights were 178 and 103, respectively. Knowledge of
    # the preceding horizontal velocity does not establish the fall's phase.
    assert not path[24][2]


def test_walking_on_continuous_visible_support_preserves_motion_evidence():
    actor = Actor("g", "goomba", 160, 71)
    path = actor_path(actor, (-0.625, 0, True), floor_scene(), 24)
    assert all(known and y == 71 for _, y, known in path)


def test_continuation_accounts_for_uncertain_fall_height_before_body_overlap():
    from typesafe_mario.continuation import Node, actor_contacts
    from typesafe_mario.dynamics import Dynamics
    from typesafe_mario.prediction import Motion

    actor = Actor("falling", "goomba", 100, 135)
    scene = floor_scene(actors=(actor,))
    motion = Motion(100, 110, 0, 1, False, "jump")
    node = Node(motion)
    warnings = set()
    # At +24 the nominal actor has reached the floor; the log shows a real
    # actor can still be 32px higher, overlapping this otherwise clear jump.
    actor_contacts(
        motion,
        motion,
        node,
        warnings,
        scene,
        {"falling": [(100, 71, False)] * 25},
        24,
        Dynamics(),
    )
    assert warnings
    assert node.failure is None  # An uncertain contact is not a proven death.


@pytest.mark.parametrize("gravity", [0.125, 0.25])
def test_falling_actor_forecast_uses_observed_acceleration_and_current_velocity(gravity):
    from typesafe_mario.actions import Action
    from typesafe_mario.prediction import Predictor

    predictor = Predictor()
    for frame in range(9):
        scene = floor_scene(
            frame=frame,
            actors=(Actor("g", "goomba", 200, 200 - gravity * frame * (frame + 1) / 2),),
        )
        predictor.observe(scene)
    prediction = predictor.forecast(scene, (Action.NOOP,), cycle=8)
    sample = next(p for p in prediction["actor_forecasts"][0]["trajectory"] if p["frame"] == 8)
    # Independent trajectories at frame 16; no emulator state or future
    # observations are supplied to the predictor.
    expected = {0.125: 183, 0.25: 166}[gravity]
    assert abs(sample["y"] - expected) < 2
