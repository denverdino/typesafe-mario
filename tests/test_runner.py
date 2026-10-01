from __future__ import annotations

import io
import json
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from typesafe_mario.actions import Action
from typesafe_mario.dashboard import DashboardCommand
from typesafe_mario.policy import Decision
from typesafe_mario.runner import _run_realtime_dashboard, run_episode
from typesafe_mario.state import MarioStateParser


class FakeEnv:
    def __init__(self) -> None:
        self.actions: list[int] = []
        self.frame = 0
        self.closed = False
        self.ram = bytearray(0x0800)
        for column in range(16):
            self.ram[0x0500 + 5 * 16 + column] = 1

    def info(self) -> dict[str, int]:
        return {
            "x_pos": 100 + self.frame * 2,
            "y_pos": 80,
            "y_pixel": 80,
            "life": 2,
            "time": 390,
        }

    def reset(self, **kwargs):
        self.frame = 0
        return self.frame, self.info()

    def step(self, action):
        self.actions.append(action)
        self.frame += 1
        return self.frame, 1.0, False, False, self.info()

    def close(self):
        self.closed = True


def decision(action: Action, danger_score: float | None = 0.25) -> Decision:
    return Decision(
        action=action,
        confidence=1.0,
        probabilities={action.value: 1.0},
        latency_ms=2000.0,
        danger_score=danger_score,
    )


class DashboardRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = FakeEnv()
        self.dashboard = Mock()
        self.policy = Mock()
        self.log = io.StringIO()
        self.requests = []
        self.frames_per_decision = 3
        # Control API completion without network calls, sleeps, or thread races.
        executor_class = self.enterContext(patch("typesafe_mario.runner.ThreadPoolExecutor"))
        executor_class.return_value.__enter__.return_value.submit.side_effect = self.submit

    def submit(self, choose, snapshot, actions):
        future = Future()
        future.set_running_or_notify_cancel()
        self.requests.append((snapshot, future))
        return future

    def run_dashboard(self, max_decisions=2, **options):
        frame, info = self.env.reset()
        _run_realtime_dashboard(
            env=self.env,
            dashboard=self.dashboard,
            policy=self.policy,
            parser=MarioStateParser(decision_horizon_frames=self.frames_per_decision),
            frame=frame,
            info=info,
            log=self.log,
            frames_per_decision=self.frames_per_decision,
            max_decisions=max_decisions,
            screenshot_path=None,
            **options,
        )

    def complete_ready(self, frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
        self.assertLess(self.dashboard.draw.call_count, 40)
        for _, future in self.requests:
            if not future.done():
                future.set_result(decision(Action.RIGHT_JUMP))
        return DashboardCommand.QUIT if run_ended else DashboardCommand.CONTINUE

    def test_slow_response_freezes_at_frame_boundary_without_extending_action(self):
        seen = [[], []]

        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            self.assertLess(self.dashboard.draw.call_count, 30)
            if waiting:
                index = len(self.requests) - 1
                seen[index].append(frame)
                if len(seen[index]) == 6:
                    self.requests[index][1].set_result(decision((Action.RIGHT, Action.LEFT)[index]))
            return DashboardCommand.QUIT if run_ended else DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        self.run_dashboard()
        self.assertEqual(seen, [[0] * 6, [1, 2, 3, 3, 3, 3]])
        self.assertEqual(self.env.actions, [1] * 3 + [6] * 3)
        rows = [json.loads(l) for l in self.log.getvalue().splitlines()]
        self.assertEqual([r["execution"]["frames"] for r in rows[:-1]], [3, 3])
        self.assertEqual(rows[1]["state"]["player"]["x"], 100)
        self.assertEqual(rows[1]["execution"]["start_state"]["player"]["x"], 106)
        self.assertEqual(rows[1]["execution"]["response_delay_frames"], 3)

    def test_early_response_waits_until_boundary_and_next_request_starts_with_action(self):
        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            self.assertLess(self.dashboard.draw.call_count, 20)
            for i, (_, future) in enumerate(self.requests):
                if not future.done():
                    future.set_result(decision((Action.RIGHT, Action.LEFT)[i]))
            return DashboardCommand.QUIT if run_ended else DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        self.run_dashboard()
        self.assertEqual(self.env.actions, [1] * 3 + [6] * 3)
        self.assertEqual([s.x for s, _ in self.requests], [100, 100])
        timing = self.requests[1][0].to_state()["reaction_timing"]
        self.assertEqual(timing["frames_until_action"], 3)
        self.assertEqual(timing["scheduled_action"], "right")
        self.assertEqual(timing["scheduled_first_frame_action"], "right")

    def test_landing_keeps_fixed_cycle_and_can_jump_again_at_boundary(self):
        self.frames_per_decision = 8
        self.env.ram[0x1D] = 1
        step = self.env.step

        def landing(action):
            result = step(action)
            if self.env.frame == 3:
                self.env.ram[0x1D] = 0
            return result

        self.env.step = landing
        self.dashboard.draw.side_effect = self.complete_ready
        self.run_dashboard(max_decisions=3)
        self.assertEqual([s.x for s, _ in self.requests], [100, 100, 116])
        self.assertEqual(self.env.actions, [2] * 8 + [1] + [2] * 7 + [1] + [2] * 7)
        timing = self.requests[2][0].to_state()["reaction_timing"]
        self.assertEqual(timing["scheduled_first_frame_action"], "right")

    def test_one_frame_jump_cycles_repress_after_release(self):
        self.frames_per_decision = 1
        self.dashboard.draw.side_effect = self.complete_ready
        self.run_dashboard(max_decisions=3)
        self.assertEqual(self.env.actions, [2, 1, 2])

    def test_restart_discards_old_response_and_resets_cycle(self):
        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            n = self.dashboard.draw.call_count
            self.assertLess(n, 20)
            if n == 1:
                return DashboardCommand.RESTART
            if n == 2:
                self.requests[0][1].set_result(decision(Action.RIGHT))
                self.requests[1][1].set_result(decision(Action.LEFT))
            return DashboardCommand.QUIT if run_ended else DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        self.run_dashboard(max_decisions=1)
        self.assertEqual(self.env.actions, [6] * 3)
        rows = [json.loads(l) for l in self.log.getvalue().splitlines()]
        ends = [r for r in rows if r["record_type"] == "episode_end"]
        self.assertEqual([r["reason"] for r in ends], ["restart", "decision_limit"])
        self.assertNotEqual(ends[0]["episode_id"], ends[1]["episode_id"])

    def test_restart_restores_checkpoint_history_instead_of_resetting_level(self):
        from dataclasses import replace

        checkpoint = Mock()

        def restore(env):
            env.frame = 25
            parser = MarioStateParser(decision_horizon_frames=3)
            snapshot = parser.parse(env.info(), env.ram)
            return env.frame, parser, replace(snapshot, dx=4, previous_action="right")

        checkpoint.restore.side_effect = restore
        ended = 0

        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            nonlocal ended
            self.assertLess(self.dashboard.draw.call_count, 20)
            for _, future in self.requests:
                if not future.done():
                    future.set_result(decision(Action.RIGHT))
            if run_ended:
                ended += 1
                return DashboardCommand.RESTART if ended == 1 else DashboardCommand.QUIT
            return DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        with patch.object(self.env, "reset", wraps=self.env.reset) as reset:
            self.run_dashboard(max_decisions=1, retry_checkpoint=checkpoint)
        reset.assert_called_once()  # Initial test setup only; neither retry calls reset.
        self.assertEqual(checkpoint.restore.call_count, 2)
        self.assertEqual(
            [(s.x, s.dx, s.frames_until_action) for s, _ in self.requests],
            [(150, 4, 0), (150, 4, 0)],
        )
        rows = [json.loads(line) for line in self.log.getvalue().splitlines()]
        decisions = [row for row in rows if row["record_type"] == "decision"]
        self.assertEqual([row["execution"]["start_frame"] for row in decisions], [0, 0])
        self.assertEqual([row["result_state"]["player"]["x"] for row in decisions], [156, 156])

    def test_quit_while_waiting_does_not_advance(self):
        self.dashboard.draw.return_value = DashboardCommand.QUIT
        self.run_dashboard()
        self.assertEqual(self.env.actions, [])
        self.assertEqual(json.loads(self.log.getvalue())["reason"], "quit")

    def test_late_response_after_death_is_never_applied(self):
        step = self.env.step

        def dying(action):
            frame, _, _, _, info = step(action)
            info["death"] = frame == 2
            return frame, float(action), frame == 2, False, info

        self.env.step = dying

        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            self.assertLess(self.dashboard.draw.call_count, 20)
            if len(self.requests) == 1 and not self.requests[0][1].done():
                self.requests[0][1].set_result(decision(Action.RIGHT))
            if run_ended:
                self.requests[-1][1].set_result(decision(Action.LEFT))
                return DashboardCommand.QUIT
            return DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        self.run_dashboard()
        rows = [json.loads(l) for l in self.log.getvalue().splitlines()]
        self.assertEqual(self.env.actions, [1, 1])
        self.assertEqual(rows[0]["reward"], 2)
        self.assertEqual(rows[-1]["reason"], "death")

    def test_api_error_logs_failure_and_closes_resources(self):
        def draw(*args, **kwargs):
            self.requests[0][1].set_exception(RuntimeError("API request failed"))
            return DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        with TemporaryDirectory() as artifacts:
            with (
                patch("typesafe_mario.runner.create_mario_env", return_value=self.env),
                patch("typesafe_mario.runner.LiveDashboard", return_value=self.dashboard),
                self.assertRaisesRegex(RuntimeError, "API request failed"),
            ):
                run_episode(
                    env_id="test",
                    policy=self.policy,
                    frames_per_decision=3,
                    max_decisions=1,
                    seed=123,
                    artifacts_dir=Path(artifacts),
                )
            rows = [
                json.loads(l)
                for l in next(Path(artifacts).glob("*.jsonl")).read_text().splitlines()
            ]
            self.assertEqual(rows[-1]["reason"], "policy_error")
        self.assertEqual(self.env.actions, [])
        self.assertTrue(self.env.closed)
        self.dashboard.close.assert_called_once()
        self.policy.close.assert_called_once()

    def test_redraw_does_not_change_paused_or_ended_state(self):
        paused, ended = [], []

        def draw(frame, snapshot, active_decision, *, waiting, run_ended, **kwargs):
            self.assertLess(self.dashboard.draw.call_count, 20)
            if waiting:
                paused.append(snapshot.to_state())
                if len(paused) == 4:
                    self.requests[0][1].set_result(decision(Action.RIGHT))
            if run_ended:
                ended.append(snapshot.to_state())
                if len(ended) == 3:
                    return DashboardCommand.QUIT
            return DashboardCommand.CONTINUE

        self.dashboard.draw.side_effect = draw
        self.run_dashboard(max_decisions=1)
        self.assertEqual(paused, [paused[0]] * 4)
        self.assertEqual(ended, [ended[0]] * 3)
        self.assertEqual(ended[0]["recent_control"]["frames_observed"], 3)
