from dataclasses import replace

import pytest
from prediction_helpers import assess_forecast
from test_observation_prediction import floor_scene
from test_policy import Provider, policy

from typesafe_mario.actions import Action
from typesafe_mario.observation import Actor, Block
from typesafe_mario.prediction import Predictor
from typesafe_mario.risk import assess_risk
from typesafe_mario.state import MarioStateParser


def approach():
    # The failed run at frame 80: jump is already committed beneath a low brick.
    return floor_scene(
        frame=80,
        x=237,
        vx=3,
        previous_action="right_run",
        known_x=(192, 368),
        blocks=(Block(256, 272, 127, 143), Block(320, 368, 127, 143), Block(192, 368, 63, 79)),
        actors=(Actor("0:goomba", "goomba", 335, 71),),
    )


def prediction(o=None, actions=tuple(Action)):
    o = o or approach()
    p = Predictor()
    p.observe(
        replace(
            o,
            frame=o.frame - 1,
            x=o.x - 3,
            actors=tuple(replace(a, x=a.x + 0.625) for a in o.actors),
        )
    )
    p.observe(o)
    return p.forecast(o, actions, cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)


def test_brake_remains_available_before_ceiling_jump_lands_near_enemy():
    report = assess_forecast(prediction())
    assert report["status"] == "filtered"
    assert report["candidate_actions"] == ["left"]
    assert "possible_contact" in report["excluded_actions"]["right_run_jump"]


def test_all_risky_keeps_choices_and_reports_no_lower_risk_option():
    o = replace(approach(), actors=(Actor("0:goomba", "goomba", 267, 79),))
    report = assess_forecast(prediction(o))
    assert report["status"] == "no_lower_risk_option"
    assert set(report["candidate_actions"]) == {a.value for a in Action}
    assert not report["excluded_actions"]


def test_missing_terrain_is_not_evidence_that_retreat_is_lower_risk():
    report = assess_forecast(prediction(replace(approach(), known_x=None, blocks=())))
    assert report["status"] == "no_lower_risk_option"
    assert set(report["candidate_actions"]) == {a.value for a in Action}


def test_provider_receives_filtered_choices_and_its_answer_is_executed_unchanged():
    provider = Provider("left")
    snapshot = replace(
        MarioStateParser().parse({}),
        prediction=prediction(),
        frame_index=80,
        frames_until_action=8,
        scheduled_action="right_run_jump",
    )
    decision = policy(provider).choose(snapshot, tuple(Action))
    assert set(provider.request["questions"]["next_action"]["criteria"]) == {"left"}
    assert decision.action == Action.LEFT
    assert decision.selection["method"] == "observation_risk_filter"
    assert "right_run_jump" in decision.selection["excluded_actions"]


def test_disallowed_lower_risk_action_cannot_empty_the_provider_choices():
    provider = Provider("right_run_jump")
    snapshot = replace(
        MarioStateParser().parse({}),
        prediction=prediction(),
        frame_index=80,
        frames_until_action=8,
        scheduled_action="right_run_jump",
    )
    decision = policy(provider).choose(snapshot, (Action.RIGHT_RUN_JUMP,))
    assert decision.action == Action.RIGHT_RUN_JUMP
    assert set(provider.request["questions"]["next_action"]["criteria"]) == {"right_run_jump"}
    assert decision.selection["status"] == "no_lower_risk_option"


def test_falling_above_known_gap_is_flagged_before_reaching_bottom_of_screen():
    o = floor_scene(
        frame=80, x=100, y=190, vx=0, vy=-1, grounded=False, known_x=(0, 256), blocks=(), actors=()
    )
    result = prediction(o, (Action.NOOP,))
    branch = result["action_forecasts"][0]
    assert branch["first_cycle"]["y"] > 32
    assert branch["control_risk"]["possible_fall"]


def test_visible_landing_after_short_gap_does_not_filter_every_forward_action():
    o = floor_scene(
        x=100,
        y=111,
        vx=0,
        vy=0,
        previous_action="noop",
        known_x=(0, 400),
        actors=(),
        blocks=(Block(0, 112, 95, 111), Block(128, 400, 63, 79), Block(0, 400, 135, 151)),
    )
    result = Predictor().forecast(o, tuple(Action), cycle=8)
    assert "right" in assess_forecast(result)["candidate_actions"]
    branch = next(f for f in result["action_forecasts"] if f["action"] == "right")
    assert branch["end"]["grounded_estimate"]
    assert not branch["control_risk"]["possible_fall"]


@pytest.mark.parametrize("offset", [0, 1200])
def test_grounded_gap_approach_requires_support_for_position_uncertainty(offset):
    o = floor_scene(
        x=46 + offset,
        vx=2,
        known_x=(offset, 300 + offset),
        blocks=(
            Block(offset, 80 + offset, 63, 79),
            Block(128 + offset, 300 + offset, 63, 79),
            Block(16 + offset, 80 + offset, 127, 143),
        ),
    )
    result = Predictor().forecast(o, tuple(Action), cycle=8, delay=4, scheduled_action=Action.RIGHT)
    branches = {b["action"]: b for b in result["action_forecasts"]}
    assert branches["right"]["first_cycle"]["grounded_estimate"]
    assert branches["right"]["first_cycle_risk"]["uncertain_landing"]
    assert "right" not in assess_forecast(result)["candidate_actions"]
    assert "left" in assess_forecast(result)["candidate_actions"]
    limited = assess_risk(result, (Action.RIGHT, Action.RIGHT_RUN))
    assert limited["status"] == "no_lower_risk_option"
    assert "uncertain_landing" in limited["warnings"]["right"]
    # The same upper-edge uncertainty can be caught by a visible lower floor.
    supported = replace(o, blocks=(*o.blocks, Block(offset, 300 + offset, 31, 47)))
    result = Predictor().forecast(
        supported, tuple(Action), cycle=8, delay=4, scheduled_action=Action.RIGHT
    )
    right = next(b for b in result["action_forecasts"] if b["action"] == "right")
    assert not right["first_cycle_risk"]["uncertain_landing"]


def test_uncertain_contact_does_not_reenable_nominal_collision_when_retreat_is_better():
    # Later live failure: descending toward a ground enemy, with room to brake left.
    o = floor_scene(
        frame=504,
        x=655,
        y=136,
        vx=-1,
        vy=-3,
        grounded=False,
        known_x=(608, 784),
        previous_action="jump",
        blocks=(Block(608, 784, 63, 79),),
        actors=(Actor("0:goomba", "goomba", 681, 71),),
    )
    p = Predictor()
    p.observe(
        replace(o, frame=503, x=656, y=139, actors=(Actor("0:goomba", "goomba", 681.625, 71),))
    )
    p.observe(o)
    result = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    branches = {f["action"]: f for f in result["action_forecasts"]}
    assert all(f["control_risk"]["possible_contact"] for f in branches.values())
    assert "left" in assess_forecast(result)["candidate_actions"]
    assert "right" not in assess_forecast(result)["candidate_actions"]


def test_airborne_crossing_to_visible_support_is_not_an_unresolved_fall():
    o = floor_scene(
        x=95,
        y=200,
        vx=2,
        vy=-1,
        grounded=False,
        known_x=(0, 256),
        blocks=(Block(0, 96, 63, 79), Block(128, 256, 63, 79)),
    )
    result = Predictor().forecast(o, (Action.RIGHT_JUMP,), cycle=8)
    branch = result["action_forecasts"][0]
    assert not branch["end"]["grounded_estimate"]
    assert branch["end"]["x"] > 128
    assert not branch["control_risk"]["possible_fall"]


def test_risk_filter_does_not_cut_a_gap_jump_short_at_forecast_boundary():
    o = floor_scene(
        x=1028,
        y=112,
        vx=3,
        vy=5,
        grounded=False,
        previous_action="right_run_jump",
        known_x=(992, 1156),
        blocks=(Block(992, 1104, 63, 79), Block(1136, 1156, 63, 79)),
    )
    p = Predictor()
    for _ in range(64):
        p.dynamics.add("gravity_hold", 0.3)
        p.dynamics.add("gravity_fall", 0.62)
        p.dynamics.add("air_accel", 0.032)
    result = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN_JUMP)
    assert "right_run_jump" in assess_forecast(result)["candidate_actions"]
    assert "left" not in assess_forecast(result)["candidate_actions"]


def test_completed_flight_does_not_commit_optional_later_departure():
    o = floor_scene(x=80, y=100, vx=3, vy=-3, grounded=False, blocks=(Block(0, 144, 63, 79),))
    result = Predictor().forecast(o, (Action.RIGHT, Action.LEFT), cycle=8)
    assert "right" in assess_forecast(result)["candidate_actions"]
    right = result["action_forecasts"][0]
    assert right["first_cycle"]["grounded_estimate"]
    assert not right["control_risk"]["possible_fall"]


def test_late_marginal_landing_cannot_clear_committed_flight_uncertainty():
    o = floor_scene(x=80, y=200, vx=2, vy=-1, grounded=False, blocks=(Block(148, 256, 63, 79),))
    result = Predictor().forecast(o, (Action.NOOP,), cycle=8)
    assert result["action_forecasts"][0]["control_risk"]["uncertain_landing"]


def test_pipe_edge_touch_above_visible_floor_does_not_forbid_forward_jumps():
    o = floor_scene(
        x=594,
        vx=0,
        known_x=(560, 736),
        previous_action="right_run",
        blocks=(Block(608, 640, 79, 127), Block(560, 736, 63, 79)),
    )
    p = Predictor()
    for _ in range(64):
        p.dynamics.add("gravity_hold", 0.252)
        p.dynamics.add("air_accel", 0.052)
        p.dynamics.add("jump_speed", 5.18)
    r = p.forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT_RUN)
    assert "right_run_jump" in assess_forecast(r)["candidate_actions"]


def test_jump_rearming_frame_does_not_end_the_flight_before_takeoff():
    o = floor_scene(
        x=70, vx=3, previous_action="right_jump", known_x=(0, 400), blocks=(Block(0, 144, 63, 79),)
    )
    result = Predictor().forecast(o, (Action.RIGHT_JUMP,), cycle=8)
    assert result["action_forecasts"][0]["control_risk"]["possible_fall"]


def test_clear_complete_plan_is_preferred_to_faster_future_close_contact():
    # Reduced v20 decision152: both first inputs pass immediate checks, but
    # running gains more distance by skimming enemies whereas jumping has a
    # complete contact-free continuation.
    prediction = {
        "backend": "observation_dynamics",
        "action_cycle_frames": 8,
        "horizon_after_application_frames": 48,
        "action_forecasts": [
            {
                "action": "right_run",
                "control_risk": {"through_frame": 24},
                "continuation": {
                    "status": "uncertain",
                    "evaluated_frames": 48,
                    "failure": None,
                    "warnings": ["close_contact"],
                    "score": 143.1,
                },
            },
            {
                "action": "jump",
                "control_risk": {"through_frame": 24},
                "continuation": {
                    "status": "estimated_viable",
                    "evaluated_frames": 48,
                    "failure": None,
                    "warnings": [],
                    "score": 81.83,
                },
            },
        ],
    }
    report = assess_risk(prediction, (Action.RIGHT_RUN, Action.JUMP))
    assert report["candidate_actions"] == ["jump"]
    assert "continuation_close_contact" in report["excluded_actions"]["right_run"]
    # A truncated nominally clear prefix cannot justify that exclusion.
    prediction["action_forecasts"][1]["continuation"]["evaluated_frames"] = 8
    assert assess_risk(prediction, (Action.RIGHT_RUN, Action.JUMP))["candidate_actions"] == [
        "right_run",
        "jump",
    ]


def test_optional_risky_followup_cannot_disqualify_a_clear_waiting_refuge():
    o = floor_scene(
        x=80,
        vx=0,
        previous_action="noop",
        known_x=(0, 400),
        blocks=(Block(0, 400, 63, 79), Block(0, 400, 95, 111)),
        actors=(Actor("g", "goomba", 200, 71),),
    )
    result = Predictor().forecast(o, tuple(Action), cycle=8)
    assert "noop" in assess_forecast(result)["candidate_actions"]
    plan = next(b["continuation"] for b in result["action_forecasts"] if b["action"] == "noop")
    assert not plan["warnings"]


def test_unresolved_final_flight_cannot_hide_a_complete_waiting_refuge():
    o = floor_scene(
        x=80, vx=0, previous_action="noop", known_x=(0, 400), blocks=(Block(0, 160, 63, 79),)
    )
    result = Predictor().forecast(o, tuple(Action), cycle=8)
    assert "noop" in assess_forecast(result)["candidate_actions"]
    plan = next(b["continuation"] for b in result["action_forecasts"] if b["action"] == "noop")
    assert not plan["warnings"]


def stalled_step_scene():
    # Visible geometry at the user's x2402 stall, followed by a gap.
    return floor_scene(
        x=2402,
        y=127,
        vx=0,
        known_x=(2277, 2533),
        previous_action="right",
        blocks=(
            Block(2416, 2448, 127, 143),
            Block(2400, 2448, 111, 127),
            Block(2384, 2448, 95, 111),
            Block(2368, 2448, 79, 95),
            Block(2277, 2448, 63, 79),
            Block(2480, 2533, 63, 79),
            Block(2480, 2496, 127, 143),
            Block(2480, 2512, 111, 127),
            Block(2480, 2528, 95, 111),
            Block(2480, 2533, 79, 95),
        ),
    )


def test_safe_first_cycle_and_complete_changed_inputs_can_avoid_repeated_jump_fall():
    r = Predictor().forecast(
        stalled_step_scene(), tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT
    )
    assert "right_jump" in assess_forecast(r)["candidate_actions"]
    jump = next(b for b in r["action_forecasts"] if b["action"] == "right_jump")
    assert jump["control_risk"]["possible_fall"]
    assert not jump["first_cycle_risk"]["possible_fall"]
    # A clear later plan cannot erase an immediate contact warning.
    jump["first_cycle_risk"]["possible_contact"] = True
    assert "right_jump" not in assess_risk(r, tuple(Action))["candidate_actions"]


@pytest.mark.parametrize("actors", [(), (Actor("far", "goomba", 2280, 71),)])
def test_observed_wall_stall_preserves_alternatives_instead_of_postponing_jump(actors):
    from typesafe_mario.observation import model_state
    from typesafe_mario.recovery import ProgressMemory

    o = replace(stalled_step_scene(), actors=actors)
    memory = ProgressMemory()
    for frame in range(64):
        recovery = memory.update(frame, o.level, int(o.x), int(o.y), "right")
    r = Predictor().forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.RIGHT)
    snapshot = replace(
        MarioStateParser().parse({"x_pos": o.x, "y_pos": o.y}),
        grounded=True,
        recovery=recovery,
        prediction=r,
    )
    state = model_state(snapshot)
    state["visible_actors"] = [
        {"identity": a.identity, "kind": a.kind, "x": a.x, "y": a.y} for a in actors
    ]
    state["terrain"] = {
        "blocks": [
            {"left": b.left, "right": b.right, "bottom": b.bottom, "top": b.top} for b in o.blocks
        ]
    }
    report = assess_risk(r, tuple(Action), state=state)
    assert "right" not in report["candidate_actions"]
    assert "right_run" not in report["candidate_actions"]
    assert "noop" not in report["candidate_actions"]
    assert "jump" in report["candidate_actions"]
    assert "right_jump" in report["candidate_actions"]
    state["recovery"]["stationary_action_frames"] = {"noop": 63}
    assert "right" in assess_risk(r, tuple(Action), state=state)["candidate_actions"]


def test_later_centered_landing_does_not_erase_immediate_edge_uncertainty():
    o = floor_scene(
        x=118,
        y=82,
        vx=2,
        vy=-5,
        grounded=False,
        known_x=(0, 256),
        blocks=(Block(128, 256, 63, 79),),
    )
    result = Predictor().forecast(o, (Action.RIGHT, Action.NOOP), cycle=8)
    assert all(b["first_cycle_risk"]["uncertain_landing"] for b in result["action_forecasts"])
    report = assess_forecast(result)
    assert report["status"] == "no_lower_risk_option"
    assert all("uncertain_landing" in w for w in report["warnings"].values())


def test_all_contact_options_prefer_reducing_exposure_over_running_into_plant():
    o = floor_scene(
        x=1833.5,
        y=111,
        vx=0.3,
        previous_action="noop",
        known_x=(1700, 1956),
        blocks=(Block(1824, 1856, 79, 111), Block(1700, 1956, 63, 79)),
        actors=(Actor("p", "piranha_plant", 1848, 103),),
    )
    result = Predictor().forecast(o, tuple(Action), cycle=8)
    branches = {b["action"]: b for b in result["action_forecasts"]}
    assert all(b["first_cycle_risk"]["nominal_contact"] for b in branches.values())
    report = assess_forecast(result)
    assert report["basis"] == "contact_exposure_dominance"
    assert "right_run" not in report["candidate_actions"]
    assert "jump" in report["candidate_actions"]
    assert report["warnings"]["jump"]  # Lower exposure is not a safety claim.
    assert (
        branches["jump"]["contact_exposure"]["frames"]
        < branches["right_run"]["contact_exposure"]["frames"]
    )


def test_exposure_reduction_never_qualifies_an_unobserved_or_falling_escape():
    actions = (Action.JUMP, Action.RIGHT)
    result = {
        "backend": "observation_dynamics",
        "action_forecasts": [
            {
                "action": a.value,
                "control_risk": {
                    "through_frame": 16,
                    "possible_contact": True,
                    "nominal_contact": True,
                },
                "contact_exposure": {
                    "frames": frames,
                    "overlap_area_sum": frames * 10,
                    "end_overlap_area": frames,
                },
            }
            for a, frames in zip(actions, (1, 8), strict=True)
        ],
    }
    for warning in (
        "possible_fall",
        "unobserved_terrain",
        "uncertain_landing",
        "unreliable_motion",
    ):
        for scope in ("control_risk", "first_cycle_risk"):
            risk = result["action_forecasts"][0].setdefault(scope, {})
            risk[warning] = True
            report = assess_risk(result, actions)
            assert report["status"] == "no_lower_risk_option"
            assert warning in report["warnings"]["jump"]
            del risk[warning]


def test_prolonged_idle_reconsiders_waiting_when_clear_movement_is_available():
    o = floor_scene(x=80, vx=0, previous_action="noop")
    r = Predictor().forecast(o, tuple(Action), cycle=8, delay=8, scheduled_action=Action.NOOP)
    state = {
        "player": {"x": 80, "y": 79, "grounded": True},
        "recovery": {
            "active": True,
            "stationary_frames": 95,
            "stationary_action_frames": {"noop": 95},
        },
    }
    report = assess_risk(r, tuple(Action), state=state)
    assert "noop" not in report["candidate_actions"]
    assert "right" in report["candidate_actions"]
    # Brief intentional waits and waits with no clear moving route remain valid.
    state["recovery"]["stationary_frames"] = 16
    assert "noop" in assess_risk(r, tuple(Action), state=state)["candidate_actions"]
    state["recovery"]["stationary_frames"] = 95
    for branch in r["action_forecasts"]:
        if branch["action"] != "noop":
            branch["continuation"]["warnings"] = ["close_contact"]
            branch["continuation"]["status"] = "uncertain"
    assert "noop" in assess_risk(r, tuple(Action), state=state)["candidate_actions"]


@pytest.mark.parametrize("warning", ["possible_fall", "unobserved_terrain", "unreliable_motion"])
def test_continuation_fallback_cannot_erase_executable_cycle_terrain_risk(warning):
    # All repeated-input paths warn; the search finds a later nominal refuge.
    # That refuge must not clear uncertainty in the input that executes NOW.
    actions = (Action.RIGHT_JUMP, Action.NOOP, Action.LEFT)
    r = Predictor().forecast(floor_scene(vx=0), actions, cycle=8)
    branches = {b["action"]: b for b in r["action_forecasts"]}
    for branch in branches.values():
        branch["control_risk"]["possible_fall"] = True
    branches["right_jump"]["first_cycle_risk"][warning] = True
    branches["left"]["continuation"]["failure"] = "possible_fall"
    # Keep NOOP in the fallback comparison rather than the early discharge path.
    branches["noop"]["control_risk"]["possible_contact"] = True
    report = assess_risk(r, actions)
    assert report["basis"] == "bounded_action_continuations"
    assert report["candidate_actions"] == ["noop"]
    assert warning in report["warnings"]["right_jump"]
