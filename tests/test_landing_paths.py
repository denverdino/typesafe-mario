"""Fast landing regressions using local geometry only; no emulator or provider."""

from dataclasses import replace

import pytest

from typesafe_mario.actor_motion import actor_path
from typesafe_mario.dynamics import Dynamics
from typesafe_mario.landing import landing_window, reaction_tail_frames
from typesafe_mario.observation import Actor, Block, Observation
from typesafe_mario.prediction import Motion


def landing_scene(actor, blocks=()):
    return Observation(
        0,
        (1, 2, 3),
        100,
        79,
        0,
        0,
        True,
        False,
        "noop",
        True,
        (0, 256),
        (Block(0, 256, 63, 79), *blocks),
        (actor,),
    )


@pytest.mark.parametrize("offset", [0, 1200])
def test_wall_return_hits_before_next_action_even_when_shell_initially_moves_away(offset):
    actor = Actor("shell", "moving_shell", 125, 71)
    o = landing_scene(actor, (Block(144, 160, 79, 127),))
    o = replace(
        o,
        x=o.x + offset,
        known_x=tuple(x + offset for x in o.known_x),
        blocks=tuple(replace(b, left=b.left + offset, right=b.right + offset) for b in o.blocks),
        actors=(replace(actor, x=actor.x + offset),),
    )
    path = actor_path(o.actors[0], (4, 0, True), o, 24)
    result = landing_window(
        Motion(o.x, 79, 0, 0, True, "noop"),
        o,
        {actor.identity: path},
        0,
        1,
        8,
        Dynamics(),
    )
    enemy = result["enemies"][0]
    assert result["frames_until_next_action"] == 7
    assert enemy["contact_in_frames"] == 5
    assert enemy["contact_before_escape"] is True


@pytest.mark.parametrize("elapsed", [1, 8])
def test_enemy_currently_above_landing_can_fall_into_reaction_window(elapsed):
    actor = Actor("falling", "goomba", 100, 111)
    o = landing_scene(actor)
    path = actor_path(actor, (0, -5, True), o, 24)
    result = landing_window(
        Motion(100, 79, 0, 0, True, "noop"),
        o,
        {actor.identity: path},
        0,
        elapsed,
        8,
        Dynamics(),
    )
    assert result["enemies"][0]["contact_before_escape"] is True


def test_departing_shell_without_wall_is_not_a_return_threat():
    actor = Actor("shell", "moving_shell", 125, 71)
    o = landing_scene(actor)
    result = landing_window(
        Motion(100, 79, 0, 0, True, "noop"),
        o,
        {actor.identity: actor_path(actor, (4, 0, True), o, 24)},
        0,
        1,
        8,
        Dynamics(),
    )
    assert result["enemies"][0]["contact_before_escape"] is False


def test_actor_one_tile_above_support_is_not_cleared_by_ground_jump_formula():
    actor = Actor("falling-shell", "moving_shell", 128, 87)
    o = landing_scene(actor)
    result = landing_window(
        Motion(100, 79, 0, 0, True, "noop"),
        o,
        {actor.identity: actor_path(actor, (-4, 0, True), o, 32)},
        0,
        8,
        8,
        Dynamics(),
    )
    enemy = result["enemies"][0]
    assert enemy["contact_in_frames"] == 4
    assert enemy["jump_clearance_frames"] is None
    assert enemy["contact_before_escape"] is True


def test_braking_already_underway_does_not_require_waiting_for_next_action():
    actor = Actor("g", "goomba", 125, 71)
    o = landing_scene(actor)
    result = landing_window(
        Motion(100, 79, 2, 0, True, "left"),
        o,
        {actor.identity: actor_path(actor, (0, 0, True), o, 32)},
        0,
        1,
        8,
        Dynamics(),
    )
    enemy = result["enemies"][0]
    assert enemy["contact_in_frames"] == 5
    assert enemy["braking_contact_in_frames"] is None
    assert enemy["contact_before_escape"] is False


def test_truncated_path_does_not_claim_a_complete_reaction_window():
    actor = Actor("g", "goomba", 118, 71)
    o = landing_scene(actor)
    result = landing_window(
        Motion(100, 79, 2, 0, True, "right"),
        o,
        {actor.identity: [(118, 71, True)] * 97},
        95,
        95,
        8,
        Dynamics(),
    )
    enemy = result["enemies"][0]
    assert enemy.get("reaction_window_complete") is False
    assert enemy["motion_uncertain"] is True


@pytest.mark.parametrize("cycle", [2, 4, 8, 16])
@pytest.mark.parametrize("brake", [0.01, 0.24, 0.8])
def test_reaction_padding_covers_landing_at_end_of_forecast(cycle, brake):
    from typesafe_mario.dynamics import PRIORS, Parameters

    actor = Actor("g", "goomba", 118, 71)
    o = landing_scene(actor)
    parameters = Parameters({**{k: v[0] for k, v in PRIORS.items()}, "brake_accel": brake})
    path = actor_path(actor, (0, 0, True), o, 96 + reaction_tail_frames(cycle, brake))
    result = landing_window(
        Motion(100, 79, 3.5, 0, True, "right"),
        o,
        {actor.identity: path},
        96,
        96,
        cycle,
        parameters,
    )
    enemy = result["enemies"][0]
    assert enemy["reaction_window_complete"] is True
    assert enemy["contact_in_frames"] == 1
    assert enemy["contact_before_escape"] is True
