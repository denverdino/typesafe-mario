from __future__ import annotations

import io
import json
import unittest
from concurrent.futures import Future
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typesafe_mario.actions import Action
from typesafe_mario.dashboard import DashboardCommand
from typesafe_mario.policy import Decision
from typesafe_mario.runner import _next_frame_action, run_episode
from typesafe_mario.state import MarioStateParser


class FrameEnvironment:
    """A deterministic moving scene; each step advances exactly one game frame."""

    def __init__(self, *, terminal_frame: int | None = None, airborne: bool = True) -> None:
        self.frame = 0
        self.actions: list[int] = []
        self.terminal_frame = terminal_frame
        self.airborne = airborne
        self.closed = False
        self.ram = bytearray(0x0800)
        self.ram[0x000E] = 8
        self.ram[0x00B5] = self.ram[0x00B6] = 1
        self.ram[0x000F] = 1
        self.ram[0x0016] = 6
        self.ram[0x0087] = 240
        self.ram[0x00CF] = 176
        for page in range(2):
            for column in range(16):
                self.ram[0x0500 + page * 208 + 11 * 16 + column] = 1

    def info(self):
        self.ram[0x001D] = int(self.airborne and self.frame > 0)
        self.ram[0x00CE] = 176 - (self.frame * 3 if self.airborne else 0)
        return {
            "x_pos": 40 + self.frame * 3,
            "y_pos": 79 + (self.frame * 3 if self.airborne else 0),
            "y_pixel": 176 - (self.frame * 3 if self.airborne else 0),
            "life": 2,
            "time": 400,
            "death": self.frame == self.terminal_frame,
        }

    def reset(self, **kwargs):
        self.frame = 0
        return None, self.info()

    def step(self, action):
        self.actions.append(action)
        self.frame += 1
        return None, 1.0, self.frame == self.terminal_frame, False, self.info()

    def close(self):
        self.closed = True


class ScoringEnvironment(FrameEnvironment):
    def info(self):
        info = super().info()
        coins = sum(self.frame >= frame for frame in (5, 10, 13))
        info.update(score=coins * 200, coins=coins)
        return info


class RecordingPolicy:
    def __init__(self, action: Action | None = None) -> None:
        self.snapshots = []
        self.closed = False
        self.action = action

    def choose(self, snapshot, actions):
        action = self.action or (Action.RIGHT_RUN, Action.LEFT)[len(self.snapshots) % 2]
        self.snapshots.append(snapshot)
        return Decision(action, 1.0, {action.value: 1.0}, latency_ms=1000.0)

    def close(self):
        self.closed = True


class LandingEnvironment(FrameEnvironment):
    """A descent that lands inside an active eight-frame cycle at frame 13."""

    def info(self):
        info = super().info()
        y = 79 if self.frame == 0 or self.frame >= 13 else 79 + min(self.frame, 20 - self.frame) * 3
        info.update(y_pos=y, y_pixel=255 - y)
        self.ram[0xCE] = 255 - y
        self.ram[0x1D] = 1 if 0 < self.frame < 13 else 0
        return info


class DelayedJumpEnvironment(FrameEnvironment):
    """Land at 15; consume the new A submitted at 16 on frame 17."""

    def info(self):
        info = super().info()
        if self.frame == 0 or 15 <= self.frame <= 16:
            y = 79
        elif self.frame < 15:
            y = 79 + min(self.frame, 22 - self.frame) * 3
        else:
            y = 79 + (self.frame - 16) * 3
        info.update(y_pos=y, y_pixel=255 - y)
        self.ram[0xCE] = 255 - y
        self.ram[0x1D] = 0 if y == 79 else 1
        return info

    def step(self, action):
        self.ram[0x0D] = self.ram[0x0A]
        self.ram[0x0A] = self.ram[0x6FC] & 0xC0
        self.ram[0x6FC] = (0, 0x01, 0x81, 0x41, 0xC1, 0x80, 0x02)[action]
        return super().step(action)


class ControlledExecutor:
    """Resolve inference on explicit UI ticks, without sleeps or network access."""

    def __init__(self, delay_ticks=0):
        self.tick = 0
        self.delay_ticks = delay_ticks
        self.pending = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        future = Future()
        result = function(*args)
        if self.delay_ticks == 0:
            future.set_result(result)
        else:
            self.pending.append((self.tick + self.delay_ticks, future, result))
        return future

    def advance(self):
        self.tick += 1
        for tick, future, result in self.pending:
            if self.tick >= tick and not future.done():
                future.set_result(result)


class RecordingDashboard:
    def __init__(self, env, executor, restart_tick=None):
        self.env = env
        self.executor = executor
        self.draws = []
        self.closed = False
        self.restart_tick = restart_tick

    def draw(self, frame, snapshot, decision, **kwargs):
        self.draws.append((self.env.frame, snapshot, kwargs))
        self.executor.advance()
        if len(self.draws) == self.restart_tick:
            return DashboardCommand.RESTART
        if kwargs["run_ended"] or len(self.draws) > 100:
            return DashboardCommand.QUIT
        return DashboardCommand.CONTINUE

    def close(self):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def run_scene(
        self,
        *,
        display="dashboard",
        delay_ticks=0,
        horizon=8,
        terminal_frame=None,
        restart_tick=None,
        airborne=True,
        action=None,
        max_decisions=2,
        environment=None,
    ):
        env = environment or FrameEnvironment(terminal_frame=terminal_frame, airborne=airborne)
        policy = RecordingPolicy(action)
        executor = ControlledExecutor(delay_ticks)
        dashboard = RecordingDashboard(env, executor, restart_tick)
        with (
            TemporaryDirectory() as temporary,
            patch("typesafe_mario.runner.create_mario_env", return_value=env),
            patch("typesafe_mario.runner.LiveDashboard", return_value=dashboard),
            patch("typesafe_mario.runner.ThreadPoolExecutor", return_value=executor),
            redirect_stdout(io.StringIO()),
        ):
            path = run_episode(
                env_id="test",
                policy=policy,
                frames_per_decision=horizon,
                max_decisions=max_decisions,
                seed=123,
                artifacts_dir=Path(temporary),
                display=display,
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            trace_path = path.with_suffix(".trace.jsonl")
            env.trace = (
                [json.loads(line) for line in trace_path.read_text().splitlines()]
                if trace_path.exists()
                else []
            )
        return env, policy, dashboard, records

    def test_scoring_includes_warmup_and_partial_terminal_cycle(self):
        env, _, _, records = self.run_scene(environment=ScoringEnvironment(terminal_frame=13))
        result = records[0]["scoring_result"]
        self.assertEqual((result["start_frame"], result["end_frame"]), (8, 13))
        self.assertEqual(result["score_gain"], 400)
        self.assertEqual(result["confirmed_coins"], 2)
        self.assertEqual(env.trace[-1]["summary"]["net_score_gain"], 600)
        self.assertEqual(env.trace[-1]["summary"]["confirmed_coins_collected"], 3)

    def test_paused_inference_does_not_duplicate_scoring(self):
        env, _, _, _ = self.run_scene(
            delay_ticks=12, environment=ScoringEnvironment(terminal_frame=13)
        )
        self.assertEqual(env.trace[-1]["summary"]["confirmed_coins_collected"], 3)
        self.assertEqual(len([e for e in env.trace if e["event"] == "coin_collected"]), 3)

    def test_scoring_result_does_not_mutate_request_snapshot(self):
        _, policy, _, records = self.run_scene(environment=ScoringEnvironment(terminal_frame=13))
        self.assertEqual(records[0]["state"]["scoring"]["score"], 0)
        self.assertEqual(policy.snapshots[0].scoring.score, 0)
        self.assertEqual(records[0]["scoring_result"]["score_gain"], 400)

    def test_restart_clears_episode_summary(self):
        env, _, _, _ = self.run_scene(
            restart_tick=11, environment=ScoringEnvironment(terminal_frame=13)
        )
        ends = [e for e in env.trace if e["event"] == "episode_end"]
        self.assertEqual([e["summary"]["confirmed_coins_collected"] for e in ends], [2, 3])

    def test_restart_flushes_partial_active_decision_once(self):
        _, _, _, records = self.run_scene(
            restart_tick=11, environment=ScoringEnvironment(terminal_frame=13)
        )
        self.assertEqual(
            [
                (
                    r["execution"]["episode"],
                    r["scoring_result"]["start_frame"],
                    r["scoring_result"]["end_frame"],
                )
                for r in records
            ],
            [(0, 8, 11), (1, 8, 13)],
        )
        self.assertEqual([r["scoring_result"]["score_gain"] for r in records], [200, 400])

    def test_quit_flushes_partial_active_decision_once(self):
        class QuitDashboard(RecordingDashboard):
            def draw(self, *args, **kwargs):
                result = super().draw(*args, **kwargs)
                return DashboardCommand.QUIT if self.env.frame == 11 else result

        with patch(f"{__name__}.RecordingDashboard", QuitDashboard):
            env, _, _, records = self.run_scene(environment=ScoringEnvironment(terminal_frame=13))
        self.assertEqual(env.trace[-1]["reason"], "quit")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scoring_result"]["end_frame"], 11)
        self.assertEqual(records[0]["scoring_result"]["score_gain"], 200)

    def test_error_flushes_partial_active_decision_once(self):
        class FailingEnvironment(ScoringEnvironment):
            def step(self, action):
                if self.frame == 11:
                    raise RuntimeError("synthetic step failure")
                return super().step(action)

        env = FailingEnvironment()
        with (
            TemporaryDirectory() as temporary,
            patch("typesafe_mario.runner.create_mario_env", return_value=env),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic step failure"):
                run_episode(
                    env_id="test",
                    policy=RecordingPolicy(),
                    frames_per_decision=8,
                    max_decisions=2,
                    seed=123,
                    artifacts_dir=Path(temporary),
                    display="none",
                )
            path = next(
                p for p in Path(temporary).glob("*.jsonl") if not p.name.endswith(".trace.jsonl")
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scoring_result"]["end_frame"], 11)
        self.assertEqual(records[0]["scoring_result"]["score_gain"], 200)

    def test_boundary_preserves_a_new_jump_until_the_game_consumes_it(self):
        for macro, held in (
            (Action.RIGHT_JUMP, 2),
            (Action.RIGHT_RUN_JUMP, 4),
            (Action.JUMP, 5),
        ):
            for display, delay in (("dashboard", 0), ("dashboard", 12), ("none", 0), ("game", 0)):
                with self.subTest(macro=macro, display=display, delay=delay):
                    env, policy, _, records = self.run_scene(
                        environment=DelayedJumpEnvironment(),
                        action=macro,
                        display=display,
                        delay_ticks=delay,
                        max_decisions=3,
                    )
                    # Frames 16..18: keep the new A through the cycle boundary
                    # and through the first observation of actual ascent.
                    self.assertEqual(env.actions[15:18], [held, held, held])
                    context = policy.snapshots[2].to_state()["committed_control"]
                    self.assertEqual(context["first_frame_action"], macro.value)
                    self.assertEqual(context["frames_before_selected_action"], 8)
                    self.assertEqual(len(env.actions), 32)
                    self.assertEqual([r["execution"]["executed_frames"] for r in records], [8] * 3)

    def test_boundary_distinguishes_pending_jump_from_a_consumed_hold(self):
        # RAM observed around run 130400 episode 1 frames 680..682:
        # submitted A is initially pending, then consumed on the next step.
        for saved, processed, expected in (
            (0x81, 0x00, Action.RIGHT_JUMP),
            (0xC1, 0x40, Action.RIGHT_JUMP),
            (0x81, 0x80, Action.RIGHT),
            (0xC1, 0xC0, Action.RIGHT),
            (0x01, 0x00, Action.RIGHT),
        ):
            with self.subTest(saved=saved, processed=processed):
                ram = bytearray(0x800)
                ram[0x0E] = 8
                ram[0x6FC], ram[0x0A] = saved, processed
                snapshot = MarioStateParser().parse(
                    {"x_pos": 1272, "y_pos": 79}, ram, previous_action="right_jump"
                )
                self.assertEqual(
                    _next_frame_action(Action.RIGHT_JUMP, snapshot, cycle_start=True), expected
                )

    def test_jump_rearms_on_descent_and_presses_on_midcycle_landing(self):
        for macro, held, released in (
            (Action.RIGHT_JUMP, 2, 1),
            (Action.RIGHT_RUN_JUMP, 4, 3),
            (Action.JUMP, 5, 0),
        ):
            with self.subTest(macro=macro):
                env, _, _, records = self.run_scene(environment=LandingEnvironment(), action=macro)
                self.assertEqual(env.actions[:8], [0] * 8)
                self.assertEqual(env.actions[8:16], [held] * 3 + [released] * 2 + [held] * 3)
                self.assertEqual([r["execution"]["executed_frames"] for r in records], [8, 8])

    def test_control_context_matches_current_cycle_not_recent_input(self):
        _, policy, _, records = self.run_scene()
        first = policy.snapshots[0].to_state()["committed_control"]
        second = policy.snapshots[1].to_state()["committed_control"]
        self.assertEqual(first["action"], "noop")
        self.assertEqual(first["first_frame_action"], "noop")
        self.assertEqual(first["frames_before_selected_action"], 8)
        self.assertEqual(second["action"], "right_run")
        self.assertEqual(second["first_frame_action"], "right_run")
        self.assertEqual(policy.snapshots[1].previous_action, "noop")
        self.assertEqual(records[1]["state"]["committed_control"], second)
        self.assertEqual(policy.snapshots[0].to_state()["committed_control"], first)

    def test_control_context_exposes_release_frame_and_jump_rearm(self):
        _, policy, _, _ = self.run_scene(airborne=False, action=Action.RIGHT_JUMP, max_decisions=3)
        context = policy.snapshots[2].to_state()["committed_control"]
        self.assertEqual(context["action"], "right_jump")
        self.assertEqual(context["first_frame_action"], "right")
        self.assertTrue(context["release_jump_while_falling"])
        self.assertTrue(context["press_jump_on_landing"])

    def test_paused_restart_context_returns_to_bootstrap(self):
        _, policy, _, _ = self.run_scene(delay_ticks=12, restart_tick=10)
        context = policy.snapshots[1].to_state()["committed_control"]
        self.assertEqual(context["action"], "noop")
        self.assertEqual(context["frames_before_selected_action"], 8)

    def test_trace_links_request_apply_and_actual_frames_without_changing_model_input(self):
        env, policy, _, records = self.run_scene()
        metadata = env.trace[0]
        self.assertEqual(metadata["event"], "metadata")
        self.assertEqual(metadata["env_id"], "test")
        self.assertEqual(metadata["frames_per_decision"], 8)
        self.assertEqual(len(metadata["source_sha256"]["state.py"]), 64)
        requests = [event for event in env.trace if event["event"] == "request"]
        applies = [event for event in env.trace if event["event"] == "apply"]
        frames = [event for event in env.trace if event["event"] == "frame"]
        self.assertEqual([event["frame"] for event in requests], [0, 8])
        self.assertEqual([event["frame"] for event in applies], [8, 16])
        self.assertEqual([event["action_index"] for event in frames], env.actions)
        self.assertEqual(requests[1]["state"], policy.snapshots[1].to_state())
        self.assertEqual(requests[1]["committed_action"], "right_run")
        self.assertEqual(requests[1]["expected_apply_frame"], 16)
        self.assertEqual(frames[-1]["ram_hex"], bytes(env.ram).hex())
        self.assertEqual(frames[-1]["info"], env.info())
        self.assertEqual(
            records[0]["execution"],
            {
                "episode": 0,
                "request_frame": 0,
                "apply_frame": 8,
                "end_frame": 16,
                "executed_frames": 8,
            },
        )
        self.assertEqual(env.trace[-1]["reason"], "max_decisions")

    def test_trace_captures_retrigger_release_and_waits_without_extra_frames(self):
        env, _, _, _ = self.run_scene(delay_ticks=12, airborne=False, action=Action.RIGHT_JUMP)
        frames = [event for event in env.trace if event["event"] == "frame"]
        self.assertEqual([event["frame"] for event in frames], list(range(1, 25)))
        self.assertEqual([event["action_index"] for event in frames], env.actions)
        self.assertEqual(frames[16]["action"], "right")
        self.assertEqual(frames[16]["decision"], 1)

    def test_trace_retains_bootstrap_terminal_and_unapplied_request(self):
        env, _, _, records = self.run_scene(terminal_frame=3)
        self.assertEqual(records, [])
        requests = [event for event in env.trace if event["event"] == "request"]
        self.assertEqual(len(requests), 1)
        self.assertFalse(any(event["event"] == "apply" for event in env.trace))
        self.assertEqual(env.trace[-1]["event"], "episode_end")
        self.assertEqual(env.trace[-1]["frame"], 3)
        self.assertEqual(env.trace[-1]["pending_decision"], 0)
        self.assertEqual(env.trace[-1]["reason"], "death")

    def test_trace_distinguishes_restarts_and_partial_action_termination(self):
        env, _, _, records = self.run_scene(restart_tick=10, terminal_frame=11)
        starts = [event for event in env.trace if event["event"] == "episode_start"]
        ends = [event for event in env.trace if event["event"] == "episode_end"]
        self.assertEqual([event["episode"] for event in starts], [0, 1])
        self.assertEqual([event["reason"] for event in ends], ["restart", "death"])
        self.assertEqual([event["frame"] for event in ends], [10, 11])
        self.assertEqual(starts[0]["seed"], 123)
        self.assertIsNone(starts[1]["seed"])
        self.assertEqual(records[-1]["execution"]["episode"], 1)
        self.assertEqual(records[-1]["execution"]["executed_frames"], 3)

    def test_trace_keeps_failed_request_and_closes_resources_on_inference_error(self):
        env = FrameEnvironment()
        policy = RecordingPolicy()
        with (
            TemporaryDirectory() as temporary,
            patch("typesafe_mario.runner.create_mario_env", return_value=env),
            patch.object(policy, "choose", side_effect=RuntimeError("inference failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run_episode(
                    env_id="test",
                    policy=policy,
                    frames_per_decision=8,
                    max_decisions=2,
                    seed=123,
                    artifacts_dir=Path(temporary),
                    display="none",
                )
            trace_path = next(Path(temporary).glob("*.trace.jsonl"))
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        self.assertTrue(env.closed)
        self.assertTrue(policy.closed)
        self.assertEqual(events[-2]["event"], "error")
        self.assertEqual(events[-2]["error_type"], "RuntimeError")
        self.assertEqual(events[-1]["pending_decision"], 0)
        self.assertEqual(events[-1]["reason"], "error")
        self.assertEqual(events[-1]["frame"], 8)

    def test_early_inference_changes_action_only_at_cycle_boundary(self):
        for display in ("dashboard", "none", "game"):
            with self.subTest(display=display):
                env, policy, _, _ = self.run_scene(display=display)
                self.assertEqual(env.actions, [0] * 8 + [3] * 8 + [6] * 8)
                self.assertTrue(env.closed)
                self.assertTrue(policy.closed)

    def test_slow_inference_pauses_frames_and_parser_until_result(self):
        env, policy, dashboard, _ = self.run_scene(delay_ticks=12)
        paused = dashboard.draws[7:12]
        self.assertEqual([frame for frame, _, _ in paused], [8] * 5)
        first = paused[0][1]
        self.assertEqual(first.airborne_frames, 8)
        for _, snapshot, _ in paused:
            self.assertEqual(snapshot, first)
        self.assertEqual(env.actions, [0] * 8 + [3] * 8 + [6] * 8)
        self.assertEqual(policy.snapshots[1].last_response_delay_frames, 8)

    def test_all_modes_report_per_frame_velocity_and_elapsed_frames(self):
        for display in ("dashboard", "none", "game"):
            with self.subTest(display=display):
                _, policy, _, _ = self.run_scene(display=display)
                state = policy.snapshots[1].to_state()
                self.assertEqual(state["player"]["horizontal_speed_px_per_frame"], 3)
                self.assertEqual(state["player"]["vertical_speed_px_per_frame"], 3)
                self.assertEqual(state["trajectory"]["airborne_frames"], 8)
                self.assertEqual(state["hazard"]["relative_velocity_x"], -3)

    def test_configured_cycle_length_is_respected(self):
        env, policy, _, _ = self.run_scene(horizon=4)
        self.assertEqual(env.actions, [0] * 4 + [3] * 4 + [6] * 4)
        self.assertEqual(policy.snapshots[1].last_response_delay_frames, 4)

    def test_terminal_frame_stops_a_partial_cycle(self):
        for display in ("dashboard", "none"):
            with self.subTest(display=display):
                env, _, _, _ = self.run_scene(display=display, terminal_frame=3)
                self.assertEqual(env.frame, 3)
                self.assertTrue(env.closed)

    def test_restart_while_paused_discards_old_result_and_resets_frame_budget(self):
        env, policy, dashboard, _ = self.run_scene(delay_ticks=12, restart_tick=10)
        self.assertEqual(env.actions, [0] * 16 + [6] * 8 + [3] * 8)
        self.assertEqual(policy.snapshots[1].x, 40)
        self.assertEqual(policy.snapshots[1].airborne_frames, 0)
        self.assertEqual(dashboard.draws[10][0], 1)

    def test_log_records_executed_action_reward_and_partial_cycle_termination(self):
        for display in ("dashboard", "none"):
            with self.subTest(display=display):
                _, _, _, records = self.run_scene(display=display, terminal_frame=11)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["action"], "right_run")
                self.assertEqual(records[0]["reward"], 3.0)
                self.assertTrue(records[0]["terminated"])

    def test_grounded_jump_releases_only_a_previously_held_button(self):
        for display in ("dashboard", "none", "game"):
            with self.subTest(display=display):
                env, _, _, _ = self.run_scene(
                    display=display, airborne=False, action=Action.RIGHT_JUMP
                )
                self.assertEqual(env.actions, [0] * 8 + [2] * 8 + [1] + [2] * 7)

    def test_one_frame_cycles_can_press_and_retrigger_jump(self):
        for display in ("dashboard", "none", "game"):
            with self.subTest(display=display):
                env, _, _, _ = self.run_scene(
                    display=display,
                    horizon=1,
                    airborne=False,
                    action=Action.RIGHT_JUMP,
                    max_decisions=3,
                )
                self.assertEqual(env.actions, [0, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
