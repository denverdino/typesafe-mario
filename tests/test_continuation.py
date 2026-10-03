from dataclasses import replace

import pytest
from prediction_helpers import assess_forecast
from test_observation_prediction import floor_scene

from typesafe_mario.actions import Action
from typesafe_mario.observation import Actor, Block
from typesafe_mario.prediction import Predictor


def paths(o, actions=tuple(Action)):
    p = Predictor()
    p.observe(o)
    return p.forecast(o, actions, cycle=8)


def test_walk_then_jump_can_resolve_a_future_wall_without_changing_first_action():
    o = floor_scene(vx=0, blocks=(Block(0, 256, 63, 79), Block(128, 160, 79, 111)))
    r = paths(o)
    branch = next(b for b in r["action_forecasts"] if b["action"] == "right_run")
    plan = branch["continuation"]
    assert plan["actions"][0] == "right_run"
    assert any("jump" in a for a in plan["actions"][1:])
    assert plan["end"]["x"] > branch["end"]["x"] + 16
    assert plan["status"] == "estimated_viable"


def test_search_does_not_discard_the_hold_needed_to_clear_a_high_visible_wall():
    r = paths(floor_scene(vx=0, blocks=(Block(0, 256, 63, 79), Block(128, 176, 79, 143))))
    b = next(b for b in r["action_forecasts"] if b["action"] == "right_run_jump")
    # Repeated hold crosses fully onto the high platform even at walking-flight
    # speed; distance beyond the platform is not needed to establish this route.
    assert b["end"]["x"] > 144 and b["end"]["y"] >= 143
    assert b["continuation"]["end"]["x"] > 160


def test_earlier_progress_is_preferred_to_postponing_the_same_jump_forever():
    # A single obstacle keeps the two routes comparable: the multi-step scene
    # can require an extra failed jump/retry for one first action.
    o = floor_scene(
        x=322,
        vx=0,
        known_x=(288, 544),
        blocks=(
            Block(288, 544, 63, 79),
            Block(336, 352, 79, 111),
        ),
    )
    r = Predictor().forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    plans = {b["action"]: b["continuation"] for b in r["action_forecasts"]}
    assert plans["right_run_jump"]["score"] > plans["right"]["score"]


@pytest.mark.parametrize("offset", [0, 1000])
def test_ceiling_sealed_platform_values_retreat_to_visible_lower_route(offset):
    o = floor_scene(
        x=110 + offset,
        y=143,
        vx=0,
        previous_action="right",
        known_x=(offset, 256 + offset),
        blocks=(
            Block(offset, 256 + offset, 63, 79),
            Block(60 + offset, 156 + offset, 127, 143),
            Block(124 + offset, 156 + offset, 143, 207),
            Block(60 + offset, 156 + offset, 207, 223),
        ),
    )
    r = paths(o)
    assert r.get("navigation_goal", {}).get("kind") == "descend_below_blocked_platform"
    plans = {b["action"]: b["continuation"] for b in r["action_forecasts"]}
    left = plans["left"]
    assert left["end"]["y"] == 79
    assert left["status"] == "estimated_viable"
    assert left["score"] > plans["right_jump"]["score"]
    assert "left" in assess_forecast(r)["candidate_actions"]
    below = replace(o, y=79)
    assert not paths(below).get("navigation_goal")
    # Open sky or no visible lower support cannot establish this detour.
    assert not paths(replace(o, blocks=o.blocks[:-1])).get("navigation_goal")
    assert not paths(replace(o, blocks=o.blocks[1:])).get("navigation_goal")
    blocked_below = replace(o, blocks=(*o.blocks, Block(124 + offset, 156 + offset, 79, 127)))
    assert not paths(blocked_below).get("navigation_goal")
    assert not paths(replace(o, y=223)).get("navigation_goal")


@pytest.mark.parametrize(
    "wall",
    [
        (Block(736, 768, 79, 143),),
        (
            Block(736, 800, 79, 95),
            Block(736, 784, 95, 111),
            Block(736, 768, 111, 127),
            Block(736, 752, 127, 143),
        ),
    ],
)
def test_high_wall_can_be_approached_with_a_modelled_runup(wall):
    o = floor_scene(x=722, vx=0, known_x=(672, 864), blocks=(Block(672, 864, 63, 79), *wall))
    p = Predictor()
    for _ in range(64):
        p.dynamics.add("gravity_hold", 0.25)
    r = p.forecast(o, tuple(Action), cycle=8)
    plan = next(b["continuation"] for b in r["action_forecasts"] if b["action"] == "left")
    # The run-up crosses the high face. A settling tail can descend onto a
    # lower stair; an unfinished flight must remain explicitly uncertain.
    assert plan["end"]["x"] + 8 > 740
    assert plan["end"]["y"] >= min(b.top for b in wall)
    if not plan["end"]["grounded_estimate"]:
        assert "unresolved_flight" in plan["warnings"]
    assert "right_run" in plan["actions"] and "right_run_jump" in plan["actions"]


def test_visible_fall_is_not_a_viable_continuation():
    o = floor_scene(x=100, y=100, vx=0, vy=-3, grounded=False, blocks=())
    r = paths(o)
    assert all(b["continuation"]["status"] == "predicted_failure" for b in r["action_forecasts"])


def test_plan_is_translation_invariant_and_does_not_learn_from_estimates():
    o = floor_scene(blocks=(Block(0, 256, 63, 79), Block(128, 160, 79, 111)))
    p = Predictor()
    before = p.calibration()
    r = p.forecast(o, tuple(Action), cycle=8)
    assert p.calibration() == before
    offset = 1701
    shifted = replace(
        o,
        x=o.x + offset,
        known_x=tuple(x + offset for x in o.known_x),
        blocks=tuple(replace(b, left=b.left + offset, right=b.right + offset) for b in o.blocks),
    )
    r2 = paths(shifted)
    for a, b in zip(r["action_forecasts"], r2["action_forecasts"], strict=True):
        assert a["continuation"]["actions"] == b["continuation"]["actions"]
        assert a["continuation"]["progress"] == b["continuation"]["progress"]


def test_unknown_terrain_is_not_certified_as_viable():
    r = paths(floor_scene(known_x=None, blocks=()))
    assert all(b["continuation"]["status"] != "estimated_viable" for b in r["action_forecasts"])


def test_continuation_never_uses_disallowed_buttons():
    r = paths(floor_scene(), (Action.RIGHT, Action.LEFT))
    assert all(
        set(b["continuation"]["actions"]) <= {"right", "left"} for b in r["action_forecasts"]
    )


def test_beam_keeps_waiting_refuge_under_ceiling_in_front_of_enemy():
    o = floor_scene(
        vx=0,
        previous_action="noop",
        blocks=(Block(0, 256, 63, 79), Block(0, 256, 95, 111)),
        actors=(Actor("g", "goomba", 160, 71),),
    )
    r = paths(o)
    plan = r["action_forecasts"][0]["continuation"]
    assert plan["failure"] is None


def test_stomp_before_end_of_action_is_not_a_warning_free_refuge():
    o = floor_scene(
        x=80,
        y=100,
        vx=1,
        vy=-3,
        grounded=False,
        previous_action="noop",
        actors=(Actor("g", "goomba", 100, 71),),
    )
    r = paths(o, (Action.NOOP, Action.RIGHT))
    assert assess_forecast(r)["candidate_actions"] != ["right"]


def test_continuation_can_finish_a_marginal_touch_on_top_of_narrow_column():
    o = floor_scene(
        x=354,
        vx=0,
        previous_action="noop",
        known_x=(320, 496),
        blocks=(
            Block(320, 496, 63, 79),
            Block(336, 352, 79, 127),
            Block(368, 384, 79, 143),
            Block(400, 416, 79, 143),
        ),
    )
    r = paths(o)
    forward = next(b for b in r["action_forecasts"] if b["action"] == "right_run_jump")
    assert forward["continuation"]["end"]["x"] > 360
    assert "marginal_landing" not in forward["continuation"]["warnings"]
    assert "right_run_jump" in assess_forecast(r)["candidate_actions"]


def test_nominal_side_collision_and_possible_stomp_are_distinguished():
    side = paths(floor_scene(x=80, vx=3, actors=(Actor("g", "goomba", 99, 71),)), (Action.RIGHT,))
    assert side["action_forecasts"][0]["continuation"]["status"] == "predicted_failure"
    above = paths(
        floor_scene(
            x=100, y=100, vx=0, vy=-3, grounded=False, actors=(Actor("g", "goomba", 100, 71),)
        ),
        (Action.NOOP,),
    )
    plan = above["action_forecasts"][0]["continuation"]
    assert "possible_stomp" in plan["warnings"]
    assert plan["status"] != "predicted_failure"


def test_spring_contact_stops_continuation_instead_of_inventing_stable_support():
    o = floor_scene(
        x=100,
        y=120,
        vx=0,
        vy=-3,
        grounded=False,
        blocks=(Block(0, 256, 63, 79), Block(100, 116, 79, 95)),
        actors=(Actor("spring", "springboard", 100, 87),),
    )
    plan = paths(o, (Action.NOOP,))["action_forecasts"][0]["continuation"]
    assert plan["status"] == "uncertain"
    assert "unsupported_interaction" in plan["warnings"]
    assert plan["evaluated_frames"] < 48


def test_possible_goomba_stomp_continues_without_certifying_survival_or_learning():
    o = floor_scene(
        x=100, y=100, vx=0, vy=-3, grounded=False, actors=(Actor("g", "goomba", 100, 71),)
    )
    p = Predictor()
    p.observe(o)
    before = p.calibration()
    plan = p.forecast(o, (Action.NOOP,), cycle=8)["action_forecasts"][0]["continuation"]
    assert plan["evaluated_frames"] == 48
    assert plan["status"] == "uncertain"
    assert "possible_stomp" in plan["warnings"]
    assert plan["interactions"][0]["actor_id"] == "g"
    assert p.calibration() == before


def test_bounce_path_checks_second_enemy_after_possible_stomp():
    o = floor_scene(
        x=100,
        y=100,
        vx=1.6,
        vy=-3,
        grounded=False,
        actors=(Actor("g", "goomba", 100, 71), Actor("p", "piranha_plant", 132, 95)),
    )
    plan = paths(o, (Action.RIGHT,))["action_forecasts"][0]["continuation"]
    assert plan["interactions"][0]["actor_id"] == "g"
    assert plan["failure"] == "nominal_contact"
    assert plan["evaluated_frames"] > plan["interactions"][0]["frame"]


def test_koopa_stomp_does_not_remove_shell_or_invent_a_bounce_route():
    o = floor_scene(
        x=100, y=100, vx=0, vy=-3, grounded=False, actors=(Actor("k", "green_koopa", 100, 71),)
    )
    plan = paths(o, (Action.NOOP,))["action_forecasts"][0]["continuation"]
    assert plan["evaluated_frames"] < 48
    assert plan["interactions"][0]["inference"] == "conditional_koopa_stomp"
    assert "possible_stationary_shell" in plan["warnings"]
    assert "possible_stomp" in plan["warnings"]


def test_simultaneous_enemy_contacts_do_not_erase_the_other_hazard():
    o = floor_scene(
        x=100,
        y=100,
        vx=0,
        vy=-3,
        grounded=False,
        actors=(Actor("g", "goomba", 100, 71), Actor("p", "piranha_plant", 100, 71)),
    )
    plan = paths(o, (Action.NOOP,))["action_forecasts"][0]["continuation"]
    assert plan["failure"] == "nominal_contact"
    assert plan["interactions"] == []


def approaching_group_scene(offset=0, *, ceiling=239, approaching=True, gap=False):
    p = Predictor()
    for frame in range(9):
        o = floor_scene(
            frame=frame,
            x=offset + 100,
            vx=0,
            previous_action="noop",
            known_x=(offset, offset + 300),
            blocks=(
                Block(offset, offset + (130 if gap else 300), 63, 79),
                Block(offset, offset + 300, ceiling, ceiling + 16),
            ),
            actors=tuple(
                Actor(str(i), "goomba", offset + x + frame * (-0.625 if approaching else 0.625), 71)
                for i, x in enumerate((164, 186, 210))
            ),
        )
        p.observe(o)
    return p, o


@pytest.mark.parametrize("offset", [0, 1300])
def test_open_enemy_approach_searches_runup_and_crossing_before_retreat(offset):
    p, o = approaching_group_scene(offset)
    before = p.calibration()
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.NOOP)
    assert r.get("enemy_crossing", {}).get("actor_ids") == ["0", "1", "2"]
    assert r["continuation_search"]["runup_considered"]
    assert all(b["continuation"]["horizon_frames"] == 80 for b in r["action_forecasts"])
    assert p.calibration() == before


@pytest.mark.parametrize("changes", [{"ceiling": 127}, {"gap": True}, {"approaching": False}])
def test_enemy_crossing_hint_requires_open_supported_approach(changes):
    p, o = approaching_group_scene(**changes)
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.NOOP)
    assert "enemy_crossing" not in r
    assert not r["continuation_search"]["runup_considered"]
