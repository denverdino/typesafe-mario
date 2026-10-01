import copy
import json
from dataclasses import replace

import pytest

from typesafe_mario.actions import Action
from typesafe_mario.checkpoint import Checkpoint, checkpoint_from_log
from typesafe_mario.policy import Decision
from typesafe_mario.runner import EpisodeLog, _advance_frame, _unwrap_ram, create_mario_env
from typesafe_mario.state import MarioStateParser


@pytest.fixture
def env():
    pytest.importorskip("gym_super_mario_bros")
    e = create_mario_env("SuperMarioBros-1-2-v0", render_mode="rgb_array")
    yield e
    e.close()


def advance(env, parser, snapshot, action, frames=8):
    trace = []
    for frame in range(frames):
        _, reward, terminated, truncated, snapshot, _actual = _advance_frame(
            env, parser, snapshot, action, new_decision=frame == 0, delay_frames=8
        )
        trace.append(
            (reward, terminated, truncated, snapshot.to_debug_state(), bytes(_unwrap_ram(env)))
        )
        if terminated or truncated:
            break
    return snapshot, trace


def test_snapshot_restores_emulator_parser_reward_and_time_limit(env):
    _, info = env.reset(seed=123)
    parser = MarioStateParser()
    snapshot = parser.parse(info, _unwrap_ram(env))
    snapshot, _ = advance(env, parser, snapshot, Action.RIGHT_RUN, 24)
    checkpoint = Checkpoint.capture(env, parser, snapshot)
    snapshot, expected = advance(env, parser, snapshot, Action.RIGHT_RUN_JUMP, 24)
    env.unwrapped.done = True
    env.env._elapsed_steps = env.env._max_episode_steps
    _, restored_parser, restored_snapshot = checkpoint.restore(env)
    _, actual = advance(env, restored_parser, restored_snapshot, Action.RIGHT_RUN_JUMP, 24)
    assert actual == expected
    # Restoring twice must not consume/mutate the saved parser history.
    _, restored_parser, restored_snapshot = checkpoint.restore(env)
    _, again = advance(env, restored_parser, restored_snapshot, Action.RIGHT_RUN_JUMP, 24)
    assert again == expected


def recorded_log(env, path):
    _, info = env.reset(seed=123)
    parser = MarioStateParser()
    s = parser.parse(info, _unwrap_ram(env))
    with path.open("w") as f:
        recorder = EpisodeLog(
            f,
            {
                "env_id": "SuperMarioBros-1-2-v0",
                "seed": 123,
                "frames_per_decision": 8,
                "control_mode": "fixed_frame_lookahead",
            },
        )
        for i, a in enumerate((Action.RIGHT_RUN, Action.RIGHT_RUN_JUMP, Action.RIGHT_JUMP)):
            recorder.begin(i, replace(s, frames_until_action=8), Decision(a, 1, {}, 0), s, 8)
            for frame in range(8):
                _, reward, t, tr, s, actual = _advance_frame(
                    env, parser, s, a, new_decision=frame == 0, delay_frames=8
                )
                recorder.step(actual, reward)
            recorder.finish(s, t, tr)
    return s, copy.deepcopy(parser), bytes(_unwrap_ram(env))


def test_log_replay_recovers_frame_history_without_provider_calls(env, tmp_path):
    path = tmp_path / "run.jsonl"
    expected, parser, ram = recorded_log(env, path)
    cp = checkpoint_from_log(
        env, path, decision=3, env_id="SuperMarioBros-1-2-v0", frames_per_decision=8
    )
    _, restored_parser, actual = cp.restore(env)
    assert actual.to_debug_state() == expected.to_debug_state()
    assert bytes(_unwrap_ram(env)) == ram
    assert vars(restored_parser) == vars(parser)
    assert cp.provenance["before_decision"] == 3
    assert cp.provenance["replayed_frames"] == 24


def test_replay_rejects_wrong_environment_horizon_ambiguous_episode_and_drift(env, tmp_path):
    path = tmp_path / "run.jsonl"
    recorded_log(env, path)
    defaults = {"decision": 2, "env_id": "SuperMarioBros-1-2-v0", "frames_per_decision": 8}
    for override in [
        {"env_id": "SuperMarioBros-1-1-v0"},
        {"frames_per_decision": 4},
        {"decision": -1},
        {"decision": 99},
    ]:
        with pytest.raises(ValueError):
            checkpoint_from_log(env, path, **(defaults | override))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[0]["result_state"]["player"]["x"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="replay diverged"):
        checkpoint_from_log(env, path, **defaults)
    rows[1]["episode_id"] = "second"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="episode"):
        checkpoint_from_log(env, path, **defaults)


def test_resumed_log_can_be_replayed_again_with_its_parent_checkpoint(env, tmp_path):
    from typesafe_mario.policy import HeuristicPolicy
    from typesafe_mario.runner import run_episode

    source = tmp_path / "source.jsonl"
    recorded_log(env, source)
    resumed = run_episode(
        env_id="SuperMarioBros-1-2-v0",
        policy=HeuristicPolicy(),
        frames_per_decision=8,
        max_decisions=2,
        seed=999,
        artifacts_dir=tmp_path / "resumed",
        display="none",
        resume_log=source,
        resume_decision=2,
    )
    rows = [json.loads(line) for line in resumed.read_text().splitlines()]
    expected = rows[-1]["result_state"]["player"]
    cp = checkpoint_from_log(
        env, resumed, decision=2, env_id="SuperMarioBros-1-2-v0", frames_per_decision=8
    )
    _, _, s = cp.restore(env)
    assert (s.x, s.y) == (expected["x"], expected["y"])
    assert cp.provenance["replayed_frames"] == 32
    assert rows[0]["state"]["player"]["horizontal_speed_px_per_frame"] != 0
    assert rows[0]["execution"]["response_delay_frames"] == 0
    assert rows[0]["run_config"]["seed"] == 123


def test_checkpoint_replay_uses_explicit_episode(env, tmp_path):
    path = tmp_path / "run.jsonl"
    recorded_log(env, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    first = rows[0]["episode_id"]
    second = copy.deepcopy(rows)
    for row in second:
        row["episode_id"] = "another"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows + second))
    cp = checkpoint_from_log(
        env,
        path,
        decision=2,
        episode_id=first,
        env_id="SuperMarioBros-1-2-v0",
        frames_per_decision=8,
    )
    assert cp.provenance["source_episode"] == first


def test_checkpoint_ancestry_cycles_fail_without_replay(env, tmp_path):
    path = tmp_path / "run.jsonl"
    recorded_log(env, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["run_config"]["resume"] = {
        "source_log": str(path),
        "source_episode": rows[0]["episode_id"],
        "before_decision": 0,
    }
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Cyclic"):
        checkpoint_from_log(
            env, path, decision=2, env_id="SuperMarioBros-1-2-v0", frames_per_decision=8
        )


def test_finished_snapshot_cannot_be_used_as_retry_point(env):
    _, info = env.reset(seed=123)
    parser = MarioStateParser()
    s = parser.parse(info, _unwrap_ram(env))
    for finished in [replace(s, dead=True), replace(s, clear=True)]:
        with pytest.raises(ValueError, match="finished"):
            Checkpoint.capture(env, parser, finished)
