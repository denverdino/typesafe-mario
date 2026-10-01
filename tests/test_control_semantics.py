import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typesafe_mario.actions import Action
from typesafe_mario.policy import Decision
from typesafe_mario.runner import run_episode


class MovingEnv:
    def __init__(self, terminal_frame=None):
        self.frame = 0
        self.actions = []
        self.terminal_frame = terminal_frame
        self.ram = bytearray(0x800)
        for c in range(16):
            self.ram[0x500 + 5 * 16 + c] = 1

    def info(self):
        return {
            "x_pos": 100 + self.frame * 2,
            "y_pos": 80,
            "y_pixel": 80,
            "death": self.frame == self.terminal_frame,
        }

    def reset(self, **kwargs):
        return None, self.info()

    def step(self, action):
        self.actions.append(action)
        self.frame += 1
        return None, float(action), self.frame == self.terminal_frame, False, self.info()

    def close(self):
        pass


class RecordingPolicy:
    def __init__(self):
        self.snapshots = []

    def choose(self, snapshot, actions):
        self.snapshots.append(snapshot)
        return Decision(Action.RIGHT_JUMP, 1.0, {"right_jump": 1.0}, 0.0)


def run(env, policy):
    with (
        TemporaryDirectory() as folder,
        patch("typesafe_mario.runner.create_mario_env", return_value=env),
    ):
        path = run_episode(
            env_id="test",
            policy=policy,
            frames_per_decision=8,
            max_decisions=2,
            seed=123,
            artifacts_dir=Path(folder),
            display="none",
        )
        return [json.loads(line) for line in path.read_text().splitlines()]


def test_headless_observes_each_frame_and_rearms_grounded_jump():
    env, policy = MovingEnv(), RecordingPolicy()
    run(env, policy)
    assert policy.snapshots[1].x == 100
    assert policy.snapshots[1].to_state()["reaction_timing"]["frames_until_action"] == 8
    assert env.actions == [2] * 8 + [1] + [2] * 7


def test_logs_final_state_and_actual_execution_on_death():
    rows = run(MovingEnv(3), RecordingPolicy())
    decision = rows[0]
    assert decision["reward"] == 6.0
    assert decision["result_state"]["episode"]["dead"] is True
    assert decision["execution"]["frames"] == 3
    assert decision["execution"]["actions"] == ["right_jump"] * 3
    assert rows[-1]["record_type"] == "episode_end"
    assert rows[-1]["reason"] == "death"
    assert rows[-1]["episode_id"] == decision["episode_id"]


def test_run_metadata_identifies_settings_for_comparisons():
    rows = run(MovingEnv(), RecordingPolicy())
    assert rows[-1]["run_config"]["frames_per_decision"] == 8
    assert rows[-1]["run_config"]["env_id"] == "test"
    assert rows[-1]["run_config"]["seed"] == 123
    assert rows[-1]["run_config"]["display"] == "none"


def test_headless_landing_preserves_fixed_frame_cycle():
    class LandingEnv(MovingEnv):
        def __init__(self):
            super().__init__()
            self.ram[0x1D] = 1

        def step(self, action):
            result = super().step(action)
            if self.frame == 3:
                self.ram[0x1D] = 0
            return result

    env, policy = LandingEnv(), RecordingPolicy()
    rows = run(env, policy)
    assert policy.snapshots[1].x == 100
    assert not policy.snapshots[1].grounded
    assert rows[0]["execution"]["frames"] == 8
    assert env.actions == [2] * 8 + [1] + [2] * 7
