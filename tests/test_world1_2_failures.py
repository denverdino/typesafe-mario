"""Self-contained actual-input replay; no provider calls or original artifacts needed."""

import json
from pathlib import Path

import pytest

from typesafe_mario.actions import ACTION_TO_INDEX, Action
from typesafe_mario.runner import create_mario_env

CASES = json.loads((Path(__file__).parent / "fixtures/world1-2-v46-failures.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=["koopa_braking", "held_jump_landing"])
def test_recorded_input_reproduces_failure_boundaries(case):
    pytest.importorskip("gym_super_mario_bros")
    env = create_mario_env(case["env_id"], render_mode="rgb_array")
    try:
        env.reset(seed=case["seed"])
        frames, final_grounded = 0, []
        for cycle in case["cycles"]:
            for name in cycle["actions"]:
                _, _, terminated, _, info = env.step(ACTION_TO_INDEX[Action(name)])
                frames += 1
                if cycle is case["cycles"][-1]:
                    final_grounded.append(int(env.unwrapped.ram[0x1D]) == 0)
            assert [info["x_pos"], info["y_pos"]] == cycle["end_xy"]
        assert frames == case["expected_frames"]
        assert terminated and info["death"] and not info.get("clear")
        if case["final_action"] == "right_run_jump":
            assert final_grounded == [False, False, False, True, True, False]
            assert case["cycles"][-1]["actions"] == ["right_run_jump"] * 6
    finally:
        env.close()
