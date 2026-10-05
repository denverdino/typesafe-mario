from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, JUMP_ACTIONS, JUMP_RELEASE_ACTION, Action
from .dashboard import DashboardCommand, LiveDashboard
from .policy import Decision, Policy
from .run_trace import RunTrace
from .state import CommittedControl, MarioSnapshot, MarioStateParser


def _next_frame_action(action: Action, snapshot: MarioSnapshot, *, cycle_start: bool) -> Action:
    if action in JUMP_ACTIONS:
        # A new press must follow a release. Rearm during observed descent so a
        # landing inside this cycle can jump without waiting for its boundary.
        descending = not snapshot.grounded and snapshot.jump_phase == "falling"
        held_on_ground = (
            cycle_start
            and snapshot.grounded
            and snapshot.previous_action in JUMP_ACTIONS
            # The last step may have submitted a fresh A before game logic
            # consumed it. Releasing now would cut short the pending jump.
            and not snapshot.jump_press_pending
        )
        if descending or held_on_ground:
            return JUMP_RELEASE_ACTION[action]
    return action


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


def _record_decision(
    log: Any,
    *,
    decision_index: int,
    snapshot: Any,
    decision: Decision,
    reward: float,
    terminated: bool,
    truncated: bool,
    execution: dict[str, int],
) -> None:
    record = {
        "decision": decision_index,
        "state": snapshot.to_state(),
        "debug_state": snapshot.to_debug_state(),
        "state_text": snapshot.to_text(),
        "action": decision.action.value,
        "confidence": decision.confidence,
        "probabilities": dict(decision.probabilities),
        "jump_needed_probability": decision.jump_needed_probability,
        "danger_score": decision.danger_score,
        "latency_ms": decision.latency_ms,
        "reward": reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "execution": execution,
    }
    log.write(json.dumps(record, separators=(",", ":")) + "\n")
    log.flush()


def _run_decision_cycles(
    *,
    env: Any,
    dashboard: LiveDashboard | None,
    policy: Policy,
    parser: MarioStateParser,
    frame: Any,
    info: dict[str, Any],
    log: Any,
    trace: RunTrace,
    seed: int,
    frames_per_decision: int,
    max_decisions: int,
    screenshot_path: Path | None,
) -> None:
    actions = tuple(Action)
    active_decision: Decision | None = None
    active_snapshot: Any = None
    active_index = 0
    pending: Future[Decision] | None = None
    pending_snapshot: Any = None
    pending_index = 0
    next_index = 0
    cycle_frames = 0
    episode_reward = 0.0
    reward_since_decision = 0.0
    screenshot_saved = False
    terminated = truncated = False
    # A request's action takes effect at the next cycle boundary. Wall-clock
    # waiting adds no simulation frames, even when inference takes much longer.
    snapshot = parser.parse(
        info, _unwrap_ram(env), previous_response_delay_frames=frames_per_decision
    )
    run_ended = max_decisions <= 0 or snapshot.dead or snapshot.clear
    trace.begin(info, _unwrap_ram(env), snapshot, seed=seed)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="typesafe-jev") as executor:
        while True:
            decision_updated = False
            if (
                not run_ended
                and pending is not None
                and cycle_frames >= frames_per_decision
                and pending.done()
            ):
                active_decision = pending.result()
                active_snapshot = pending_snapshot
                active_index = pending_index
                trace.apply(active_index, active_decision.action)
                print(
                    f"#{pending_index:04d} x={pending_snapshot.x:04d} "
                    f"action={active_decision.action.value:<15} "
                    f"confidence={active_decision.confidence:.2f} "
                    f"latency={active_decision.latency_ms:.0f}ms"
                )
                pending = None
                reward_since_decision = 0.0
                cycle_frames = 0
                decision_updated = True

            if not run_ended and pending is None and next_index < max_decisions:
                committed = active_decision.action if active_decision else Action.NOOP
                pending_snapshot = replace(
                    snapshot,
                    committed_control=CommittedControl(
                        action=committed.value,
                        first_frame_action=_next_frame_action(
                            committed, snapshot, cycle_start=decision_updated
                        ).value,
                        frames_before_selected_action=frames_per_decision - cycle_frames,
                        release_jump_while_falling=committed in JUMP_ACTIONS,
                        press_jump_on_landing=committed in JUMP_ACTIONS,
                    ),
                )
                pending_index = next_index
                trace.request(
                    pending_index,
                    pending_snapshot,
                    active_decision.action if active_decision else Action.NOOP,
                    frames_per_decision,
                )
                pending = executor.submit(policy.choose, pending_snapshot, actions)
                next_index += 1

            if not run_ended and cycle_frames < frames_per_decision:
                # The initial cycle uses NOOP while the first decision is made.
                action = _next_frame_action(
                    active_decision.action if active_decision else Action.NOOP,
                    snapshot,
                    cycle_start=decision_updated,
                )
                frame, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action])
                reward_since_decision += float(reward)
                episode_reward += float(reward)
                cycle_frames += 1
                # Parse exactly once per emulated frame, never on a paused UI tick.
                snapshot = parser.parse(
                    info,
                    _unwrap_ram(env),
                    previous_action=action.value,
                    previous_reward=float(reward),
                    previous_response_delay_frames=frames_per_decision,
                )
                trace.step(
                    action,
                    info,
                    _unwrap_ram(env),
                    snapshot,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                run_ended = bool(
                    snapshot.dead
                    or snapshot.clear
                    or terminated
                    or truncated
                    or (
                        next_index >= max_decisions
                        and pending is None
                        and cycle_frames >= frames_per_decision
                    )
                )
                if active_decision is not None and (
                    cycle_frames >= frames_per_decision or run_ended
                ):
                    _record_decision(
                        log,
                        decision_index=active_index,
                        snapshot=active_snapshot,
                        decision=active_decision,
                        reward=reward_since_decision,
                        terminated=terminated,
                        truncated=truncated,
                        execution=trace.execution(),
                    )

            if run_ended:
                trace.end(
                    "death"
                    if snapshot.dead
                    else "clear"
                    if snapshot.clear
                    else "terminated"
                    if terminated
                    else "truncated"
                    if truncated
                    else "max_decisions"
                )

            if dashboard is None:
                if run_ended:
                    break
                if pending is not None and cycle_frames >= frames_per_decision:
                    # No UI to pump: wait without advancing the emulator or spinning.
                    pending.result()
                continue

            command = dashboard.draw(
                frame,
                snapshot,
                active_decision,
                decision_index=max(0, next_index - 1),
                episode_reward=episode_reward,
                waiting=(
                    pending is not None
                    and cycle_frames >= frames_per_decision
                    and not pending.done()
                ),
                run_ended=run_ended,
            )
            if screenshot_path is not None and active_decision is not None and not screenshot_saved:
                dashboard.save(screenshot_path)
                screenshot_saved = True
            if command == DashboardCommand.QUIT:
                trace.end("quit")
                if pending is not None:
                    pending.cancel()
                break
            if command == DashboardCommand.RESTART:
                trace.end("restart")
                if pending is not None:
                    pending.cancel()
                frame, info = env.reset()
                parser.reset()
                active_decision = None
                active_snapshot = None
                active_index = 0
                pending = None
                pending_snapshot = None
                pending_index = 0
                next_index = 0
                cycle_frames = 0
                episode_reward = 0.0
                reward_since_decision = 0.0
                terminated = truncated = False
                snapshot = parser.parse(
                    info, _unwrap_ram(env), previous_response_delay_frames=frames_per_decision
                )
                run_ended = max_decisions <= 0 or snapshot.dead or snapshot.clear
                trace.begin(info, _unwrap_ram(env), snapshot, seed=None)
                print("--- Restarted ---")


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
) -> Path:
    if frames_per_decision < 1:
        raise ValueError("frames_per_decision must be at least 1")

    if display not in {"dashboard", "game", "none"}:
        raise ValueError("display must be dashboard, game, or none")
    render_mode = "human" if display == "game" else "rgb_array"
    env = create_mario_env(env_id, render_mode=render_mode)
    dashboard = LiveDashboard() if display == "dashboard" else None
    parser = MarioStateParser(decision_horizon_frames=frames_per_decision)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    log_path = artifacts_dir / f"run-{timestamp}.jsonl"

    try:
        with (
            log_path.open("x", encoding="utf-8") as log,
            log_path.with_suffix(".trace.jsonl").open("x", encoding="utf-8") as trace_log,
        ):
            meanings = getattr(env, "get_action_meanings", None)
            trace = RunTrace(
                trace_log,
                env_id=env_id,
                seed=seed,
                frames_per_decision=frames_per_decision,
                max_decisions=max_decisions,
                display=display,
                policy_type=type(policy).__name__,
                action_meanings=meanings() if meanings else None,
            )
            try:
                frame, info = env.reset(seed=seed)
                _run_decision_cycles(
                    env=env,
                    dashboard=dashboard,
                    policy=policy,
                    parser=parser,
                    frame=frame,
                    info=info,
                    log=log,
                    trace=trace,
                    seed=seed,
                    frames_per_decision=frames_per_decision,
                    max_decisions=max_decisions,
                    screenshot_path=screenshot_path,
                )
            except BaseException as exc:
                trace.write("error", error_type=type(exc).__name__)
                trace.end("error")
                raise
    finally:
        if dashboard:
            dashboard.close()
        env.close()
        close = getattr(policy, "close", None)
        if callable(close):
            close()

    return log_path
