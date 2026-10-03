from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, Action, first_frame_action
from .checkpoint import Checkpoint, checkpoint_from_log
from .dashboard import DashboardCommand, LiveDashboard
from .observation import observe
from .policy import Decision, Policy
from .prediction import BACKEND, VERSION, Predictor
from .provenance import build_identity
from .questions import PROMPT_VERSION
from .requests import REQUEST_SCHEMA_VERSION, PreparedDecision
from .runlog import EpisodeLog  # re-export for existing checkpoint/log callers
from .state import MarioSnapshot, MarioStateParser
from .tactical import STATE_VERSION


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
    physics_parameters: dict | None = None,
) -> None:
    """Fixed emulator-frame cycles, with one future action being planned in parallel."""
    actions = tuple(Action)
    active_decision: Decision | None = None
    pending: Future[Decision] | None = None
    pending_snapshot: MarioSnapshot | None = None
    pending_prepared: PreparedDecision | None = None
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
    predictor = Predictor(physics_parameters=physics_parameters)
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
                except Exception as exc:
                    recorder.record_outcome(pending_index, "failed", error_type=type(exc).__name__)
                    recorder.end(snapshot, terminated, truncated, "policy_error")
                    raise
                recorder.record_outcome(
                    pending_index,
                    "received",
                    latency_ms=decision.latency_ms,
                    provider_metadata=dict(decision.provider_metadata),
                )
                try:
                    if (
                        pending_prepared is None
                        or pending_prepared.timing.apply_at_frame != snapshot.frame_index
                    ):
                        raise ValueError("Decision application frame does not match execution")
                    if decision.action not in pending_prepared.candidates:
                        raise ValueError(
                            "Decision action is not allowed by the prepared candidates"
                        )
                except ValueError:
                    recorder.record_outcome(
                        pending_index,
                        "discarded",
                        reason="decision_contract_error",
                        completion_state="returned",
                    )
                    recorder.end(snapshot, terminated, truncated, "decision_contract_error")
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
                    pending_prepared,
                )
                predictor.expect(
                    pending_snapshot.prediction or {},
                    decision.action,
                    apply_at_frame=snapshot.frame_index,
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
                try:
                    pending_prepared = policy.prepare(pending_snapshot, actions)
                except Exception:
                    recorder.end(snapshot, terminated, truncated, "policy_error")
                    raise
                recorder.record_request(pending_index, pending_prepared)
                pending = executor.submit(policy.resolve, pending_prepared)
                next_index += 1

            if not run_ended and active_decision is not None and not cycle_complete:
                try:
                    frame, reward, terminated, truncated, snapshot, actual = _advance_frame(
                        env,
                        parser,
                        snapshot,
                        active_decision.action,
                        new_decision=decision_updated,
                        delay_frames=previous_response_delay_frames,
                    )
                except Exception:
                    if pending is not None:
                        recorder.discard(pending_index, pending, "execution_error")
                        pending = None
                    recorder.end(snapshot, terminated, truncated, "execution_error")
                    raise
                recorder.step(actual, reward, snapshot)
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
                if pending is not None:
                    recorder.discard(
                        pending_index, pending, _end_reason(snapshot, terminated, truncated)
                    )
                    pending = None
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
                if pending is not None:
                    recorder.discard(pending_index, pending, "quit")
                    pending = None
                recorder.end(snapshot, terminated, truncated, "quit")
                break
            if command == DashboardCommand.RESTART:
                if pending is not None:
                    recorder.discard(pending_index, pending, "restart")
                    pending = None
                recorder.end(snapshot, terminated, truncated, "restart")
                recorder = EpisodeLog(log, run_config)
                if retry_checkpoint is not None:
                    frame, parser, snapshot = retry_checkpoint.restore(env)
                else:
                    frame, info = env.reset()
                    parser.reset()
                    snapshot = parser.parse(
                        info, _unwrap_ram(env), previous_response_delay_frames=0
                    )
                active_decision = None
                predictor = Predictor(physics_parameters=physics_parameters)
                predictor.observe(observe(snapshot))
                pending = None
                pending_snapshot = None
                pending_prepared = None
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
    physics_profile: Path | None = None,
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    if (resume_log is None) != (resume_decision is None):
        raise ValueError("--resume-log and --resume-decision must be supplied together")
    if resume_episode is not None and resume_log is None:
        raise ValueError("--resume-episode requires --resume-log")
    physics_parameters, physics_provenance = None, None
    if physics_profile is not None:
        from .calibration import load_profile

        physics_parameters, physics_provenance = load_profile(physics_profile, env_id=env_id)
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
        "prompt_version": PROMPT_VERSION,
        "state_version": STATE_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "prediction_backend": BACKEND,
        "prediction_version": VERSION,
        "landing_replan": False,
        "implementation": build_identity(),
        "predictor_initialization": "cold_start",
        "diagnostic_questions": bool(getattr(policy, "diagnostic_questions", False)),
        "provider_model": "sdk_default_unresolved"
        if type(policy).__name__ == "TypeSafePolicy"
        else None,
    }
    if physics_provenance is not None:
        run_config["physics_profile"] = physics_provenance
        run_config["predictor_initialization"] = "calibrated_physics"
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
                physics_parameters=physics_parameters,
            )
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
