from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .actions import ACTION_TO_INDEX, Action, first_frame_action
from .checkpoint import Checkpoint, checkpoint_from_log
from .dashboard import DashboardCommand, LiveDashboard
from .observation import model_state, observe
from .policy import Decision, Policy
from .prediction import BACKEND, VERSION, Predictor
from .state import MarioSnapshot, MarioStateParser


def _unwrap_ram(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        ram = getattr(current, "ram", None)
        if ram is not None:
            return ram
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return None


def create_mario_env(env_id: str, render_mode: str = "human") -> Any:
    try:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise RuntimeError(
            'Mario dependencies are missing. Install with: pip install -e ".[mario]"'
        ) from exc

    env = gym.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


class EpisodeLog:
    """Attribute every executed frame to its active decision, including wait frames."""

    def __init__(self, log: Any, run_config: dict[str, Any] | None = None) -> None:
        self.log = log
        self.run_config = run_config or {}
        self.episode_id = uuid4().hex
        self.active: dict[str, Any] | None = None
        self.total_reward = 0.0
        self.total_frames = 0
        self.ended = False

    def write(self, record: dict[str, Any]) -> None:
        self.log.write(
            json.dumps(
                {
                    "schema_version": 2,
                    "episode_id": self.episode_id,
                    "run_config": self.run_config,
                    **record,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self.log.flush()

    def begin(
        self,
        index: int,
        observed: MarioSnapshot,
        decision: Decision,
        applied: MarioSnapshot,
        delay_frames: int,
    ) -> None:
        self.active = {
            "record_type": "decision",
            "decision": index,
            "state": model_state(observed),
            "debug_state": observed.to_debug_state(),
            "state_text": observed.to_text(),
            "action": decision.action.value,
            "confidence": decision.confidence,
            "selection": dict(decision.selection) if decision.selection is not None else None,
            "probabilities": dict(decision.probabilities),
            "jump_needed_probability": decision.jump_needed_probability,
            "jump_intent": decision.jump_intent,
            "jump_intent_probabilities": dict(decision.jump_intent_probabilities),
            "danger_score": decision.danger_score,
            "latency_ms": decision.latency_ms,
            "reward": 0.0,
            "execution": {
                "start_frame": self.total_frames,
                "frames": 0,
                "actions": [],
                "response_delay_frames": delay_frames,
                "start_state": applied.to_state(),
            },
        }

    def step(self, action: Action, reward: float) -> None:
        self.total_frames += 1
        self.total_reward += reward
        if self.active is not None:
            self.active["reward"] += reward
            self.active["execution"]["frames"] += 1
            self.active["execution"]["actions"].append(action.value)

    def finish(self, snapshot: MarioSnapshot, terminated: bool, truncated: bool) -> None:
        if self.active is not None:
            self.write(
                {
                    **self.active,
                    "result_state": snapshot.to_state(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            self.active = None

    def end(self, snapshot: MarioSnapshot, terminated: bool, truncated: bool, reason: str) -> None:
        if self.ended:
            return
        self.finish(snapshot, terminated, truncated)
        self.write(
            {
                "record_type": "episode_end",
                "reason": reason,
                "result_state": snapshot.to_state(),
                "reward": self.total_reward,
                "frames": self.total_frames,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        self.ended = True


def _end_reason(snapshot: MarioSnapshot, terminated: bool, truncated: bool) -> str:
    if snapshot.clear:
        return "stage_clear"
    if snapshot.dead:
        return "death"
    if truncated:
        return "truncated"
    return "terminated" if terminated else "decision_limit"


def _first_frame_action(snapshot: MarioSnapshot, action: Action) -> Action:
    return first_frame_action(
        action,
        grounded=snapshot.grounded,
        previous_action=snapshot.previous_action,
        swimming=snapshot.swimming,
    )


def _advance_frame(
    env: Any,
    parser: MarioStateParser,
    snapshot: MarioSnapshot,
    action: Action,
    *,
    new_decision: bool,
    delay_frames: int,
):
    if new_decision:
        action = _first_frame_action(snapshot, action)
    frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
    snapshot = parser.parse(
        info,
        _unwrap_ram(env),
        previous_action=action.value,
        previous_reward=float(reward),
        previous_response_delay_frames=delay_frames,
    )
    return frame, float(reward), terminated, truncated, snapshot, action


def _run_realtime_dashboard(
    *,
    env: Any,
    dashboard: LiveDashboard | None,
    policy: Policy,
    parser: MarioStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
    run_config: dict[str, Any] | None = None,
    retry_checkpoint: Checkpoint | None = None,
) -> None:
    """Fixed emulator-frame cycles, with one future action being planned in parallel."""
    actions = tuple(Action)
    active_decision: Decision | None = None
    pending: Future[Decision] | None = None
    pending_snapshot: MarioSnapshot | None = None
    pending_index = 0
    next_index = 0
    cycle_frames = 0
    pending_request_frame = 0
    previous_response_delay_frames = 0
    recorder = EpisodeLog(log, run_config)
    screenshot_saved = False
    terminated = truncated = False
    if retry_checkpoint is not None:
        frame, parser, snapshot = retry_checkpoint.restore(env)
    else:
        snapshot = parser.parse(info, _unwrap_ram(env), previous_response_delay_frames=0)
    predictor = Predictor()
    predictor.observe(observe(snapshot))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="typesafe-jev") as executor:
        while True:
            decision_updated = False
            cycle_complete = active_decision is None or cycle_frames >= frames_per_decision
            run_ended = bool(snapshot.dead or snapshot.clear or terminated or truncated)
            # Headless/game modes can block here; the dashboard keeps drawing while waiting.
            if (
                not run_ended
                and cycle_complete
                and pending is not None
                and (pending.done() or dashboard is None)
            ):
                try:
                    decision = pending.result()
                except Exception:
                    recorder.end(snapshot, terminated, truncated, "policy_error")
                    raise
                pending = None
                recorder.finish(snapshot, terminated, truncated)
                active_decision = decision
                previous_response_delay_frames = recorder.total_frames - pending_request_frame
                recorder.begin(
                    pending_index,
                    pending_snapshot,
                    decision,
                    snapshot,
                    previous_response_delay_frames,
                )
                predictor.expect(
                    pending_snapshot.prediction or {},
                    decision.action,
                    application_frame=snapshot.frame_index,
                )
                print(
                    f"#{pending_index:04d} observed_x={pending_snapshot.x:04d} "
                    f"applied_x={snapshot.x:04d} action={decision.action.value:<15} "
                    f"confidence={decision.confidence:.2f} latency={decision.latency_ms:.0f}ms"
                )
                cycle_frames = 0
                cycle_complete = False
                decision_updated = True

            run_ended = run_ended or (
                cycle_complete and pending is None and next_index >= max_decisions
            )
            if not run_ended and pending is None and next_index < max_decisions:
                # The next response applies only AFTER the newly selected action finishes.
                # Bootstrap is the sole zero-lookahead request, with the emulator paused.
                remaining = frames_per_decision - cycle_frames if active_decision else 0
                scheduled = active_decision.action if active_decision else None
                pending_snapshot = replace(
                    snapshot,
                    frames_until_action=remaining,
                    scheduled_action=scheduled,
                    scheduled_first_frame_action=(
                        _first_frame_action(snapshot, scheduled) if scheduled else None
                    ),
                )
                pending_index = next_index
                pending_request_frame = recorder.total_frames
                try:
                    prediction = predictor.forecast(
                        observe(pending_snapshot),
                        actions,
                        cycle=frames_per_decision,
                        delay=remaining,
                        scheduled_action=scheduled,
                        scheduled_first_frame_action=pending_snapshot.scheduled_first_frame_action,
                    )
                    pending_snapshot = replace(pending_snapshot, prediction=prediction)
                except Exception:
                    recorder.end(snapshot, terminated, truncated, "prediction_error")
                    raise
                pending = executor.submit(policy.choose, pending_snapshot, actions)
                next_index += 1

            if not run_ended and active_decision is not None and not cycle_complete:
                frame, reward, terminated, truncated, snapshot, actual = _advance_frame(
                    env,
                    parser,
                    snapshot,
                    active_decision.action,
                    new_decision=decision_updated,
                    delay_frames=previous_response_delay_frames,
                )
                recorder.step(actual, reward)
                predictor.observe(observe(snapshot))
                cycle_frames += 1
                if cycle_frames == frames_per_decision:
                    recorder.finish(snapshot, terminated, truncated)

            run_ended = bool(
                run_ended
                or snapshot.dead
                or snapshot.clear
                or terminated
                or truncated
                or (
                    cycle_frames >= frames_per_decision
                    and pending is None
                    and next_index >= max_decisions
                )
            )
            if run_ended:
                recorder.end(
                    snapshot, terminated, truncated, _end_reason(snapshot, terminated, truncated)
                )
            if dashboard is None:
                if run_ended:
                    break
                continue

            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=recorder.total_reward,
                waiting=pending is not None and not pending.done(),
                run_ended=run_ended,
            )
            if screenshot_path is not None and active_decision is not None and not screenshot_saved:
                dashboard.save(screenshot_path)
                screenshot_saved = True
            if command == DashboardCommand.QUIT:
                recorder.end(snapshot, terminated, truncated, "quit")
                break
            if command == DashboardCommand.RESTART:
                recorder.end(snapshot, terminated, truncated, "restart")
                recorder = EpisodeLog(log, run_config)
                if pending is not None:
                    pending.cancel()
                if retry_checkpoint is not None:
                    frame, parser, snapshot = retry_checkpoint.restore(env)
                else:
                    frame, info = env.reset()
                    parser.reset()
                    snapshot = parser.parse(
                        info, _unwrap_ram(env), previous_response_delay_frames=0
                    )
                active_decision = None
                predictor = Predictor()
                predictor.observe(observe(snapshot))
                pending = None
                pending_snapshot = None
                pending_index = next_index = cycle_frames = pending_request_frame = 0
                previous_response_delay_frames = 0
                terminated = truncated = False
                print("--- Restored checkpoint ---" if retry_checkpoint else "--- Restarted ---")


def run_episode(
    *,
    env_id: str,
    policy: Policy,
    frames_per_decision: int,
    max_decisions: int,
    seed: int,
    artifacts_dir: Path,
    display: str = "dashboard",
    screenshot_path: Path | None = None,
    resume_log: Path | None = None,
    resume_decision: int | None = None,
    resume_episode: str | None = None,
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    if (resume_log is None) != (resume_decision is None):
        raise ValueError("--resume-log and --resume-decision must be supplied together")
    if resume_episode is not None and resume_log is None:
        raise ValueError("--resume-episode requires --resume-log")
    render_mode = "human" if display == "game" else "rgb_array"
    env = create_mario_env(env_id, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    parser = MarioStateParser(decision_horizon_frames=frames_per_decision)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    run_config = {
        "env_id": env_id,
        "seed": seed,
        "display": display,
        "frames_per_decision": frames_per_decision,
        "control_mode": "fixed_frame_lookahead",
        "policy": type(policy).__name__,
        "prompt_version": 54,
        "state_version": 43,
        "prediction_backend": BACKEND,
        "prediction_version": VERSION,
        "landing_replan": False,
    }
    try:
        retry_checkpoint = None
        if resume_log is not None:
            retry_checkpoint = checkpoint_from_log(
                env,
                resume_log,
                decision=resume_decision,
                env_id=env_id,
                frames_per_decision=frames_per_decision,
                episode_id=resume_episode,
            )
            run_config["resume"] = retry_checkpoint.provenance
            run_config["seed"] = retry_checkpoint.provenance["source_seed"]
            frame, info = None, {}
            print(
                f"Checkpoint ready: before decision {resume_decision}, "
                f"x={retry_checkpoint.snapshot.x}, y={retry_checkpoint.snapshot.y}"
            )
        else:
            frame, info = env.reset(seed=seed)
        with log_path.open("w", encoding="utf-8") as log:
            _run_realtime_dashboard(
                env=env,
                dashboard=dashboard,
                policy=policy,
                parser=parser,
                frame=frame,
                info=info,
                log=log,
                frames_per_decision=frames_per_decision,
                max_decisions=max_decisions,
                screenshot_path=screenshot_path,
                run_config=run_config,
                retry_checkpoint=retry_checkpoint,
            )
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
