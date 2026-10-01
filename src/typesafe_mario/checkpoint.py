"""In-process emulator checkpoints, reconstructed across launches from frame logs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import ACTION_TO_INDEX, Action
from .state import MarioSnapshot, MarioStateParser

# Native snapshots exclude Python reward bookkeeping and Gym's time limit.
_REWARD_FIELDS = (
    "_time_last",
    "_x_position_max",
    "_score_last",
    "_coins_last",
    "_status_last",
    "_completion_rewarded",
    "_last_reward_components",
    "_last_reward_unclipped",
    "_last_reward_clipped",
    "done",
)
_WRAPPER_FIELDS = ("_elapsed_steps", "_has_reset")


def _layers(env: Any) -> list[Any]:
    layers = [env]
    while "env" in vars(layers[-1]):
        layers.append(layers[-1].env)
    return layers


@dataclass
class Checkpoint:
    native: Any
    parser: MarioStateParser
    snapshot: MarioSnapshot
    layer_state: list[dict[str, Any]]
    provenance: dict[str, Any]

    @classmethod
    def capture(
        cls,
        env: Any,
        parser: MarioStateParser,
        snapshot: MarioSnapshot,
        provenance: dict[str, Any] | None = None,
    ) -> Checkpoint:
        if snapshot.dead or snapshot.clear or env.unwrapped.done:
            raise ValueError("Cannot save a finished episode as a retry checkpoint")
        layers = _layers(env)
        return cls(
            native=env.unwrapped.dump_state(),
            parser=copy.deepcopy(parser),
            snapshot=copy.deepcopy(snapshot),
            layer_state=[
                {
                    k: copy.deepcopy(vars(layer)[k])
                    for k in _REWARD_FIELDS + _WRAPPER_FIELDS
                    if k in vars(layer)
                }
                for layer in layers
            ],
            provenance=copy.deepcopy(provenance or {}),
        )

    def restore(self, env: Any) -> tuple[Any, MarioStateParser, MarioSnapshot]:
        layers = _layers(env)
        if len(layers) != len(self.layer_state):
            raise ValueError("Checkpoint wrapper stack does not match environment")
        env.unwrapped.load_state(self.native)
        for layer, state in zip(layers, self.layer_state, strict=True):
            for key, value in state.items():
                setattr(layer, key, copy.deepcopy(value))
        return (
            env.unwrapped.screen.copy(),
            copy.deepcopy(self.parser),
            copy.deepcopy(self.snapshot),
        )


def checkpoint_from_log(
    env: Any,
    path: Path,
    *,
    decision: int,
    env_id: str,
    frames_per_decision: int,
    episode_id: str | None = None,
    _visited: frozenset[tuple[str, str]] = frozenset(),
) -> Checkpoint:
    """Replay recorded *actual* button frames, stopping BEFORE the named decision.

    No model requests are made. The first resumed choice is a fresh immediate
    decision; stale/prefetched decisions from the old policy are intentionally dropped.
    """
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    episodes = list(dict.fromkeys(r["episode_id"] for r in records))
    if episode_id is None:
        if len(episodes) != 1:
            raise ValueError("Log contains multiple/no episodes; specify --resume-episode")
        episode_id = episodes[0]
    rows = [r for r in records if r["episode_id"] == episode_id and r["record_type"] == "decision"]
    if not rows:
        raise ValueError(f"No decisions found for episode {episode_id}")
    config = rows[0]["run_config"]
    if config.get("env_id") != env_id:
        raise ValueError("--env must match the resume log environment")
    if config.get("frames_per_decision") != frames_per_decision:
        raise ValueError("--frames-per-decision must match the resume log")
    if config.get("control_mode") != "fixed_frame_lookahead":
        raise ValueError("Resume requires a fixed-frame lookahead log")
    if decision < 0 or decision > len(rows):
        raise ValueError("--resume-decision is outside the recorded decision range")
    if any(r.get("schema_version") != 2 or r["decision"] != i for i, r in enumerate(rows)):
        raise ValueError("Resume requires a complete, sequential schema-2 episode log")
    key = (str(path.resolve()), episode_id)
    if key in _visited:
        raise ValueError("Cyclic checkpoint log ancestry")
    parent = config.get("resume")
    if parent:
        checkpoint = checkpoint_from_log(
            env,
            Path(parent["source_log"]),
            decision=parent["before_decision"],
            env_id=env_id,
            frames_per_decision=frames_per_decision,
            episode_id=parent["source_episode"],
            _visited=_visited | {key},
        )
        _, parser, snapshot = checkpoint.restore(env)
        frame_count = checkpoint.provenance["replayed_frames"]
    else:
        _, info = env.reset(seed=config["seed"])
        parser = MarioStateParser(decision_horizon_frames=frames_per_decision)
        snapshot = parser.parse(info, env.unwrapped.ram, previous_response_delay_frames=0)
        frame_count = 0
    for row in rows[:decision]:
        execution = row["execution"]
        actual_actions = execution["actions"]
        if len(actual_actions) != execution["frames"]:
            raise ValueError("Incomplete per-frame actions in resume log")
        for name in actual_actions:
            _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[Action(name)])
            snapshot = parser.parse(
                info,
                env.unwrapped.ram,
                previous_action=name,
                previous_reward=float(reward),
                previous_response_delay_frames=execution["response_delay_frames"],
            )
            frame_count += 1
            if terminated or truncated or snapshot.dead or snapshot.clear:
                raise ValueError(
                    "Resume point follows episode termination; choose an earlier decision"
                )
        expected = row["result_state"]["player"]
        if (snapshot.x, snapshot.y) != (expected["x"], expected["y"]):
            raise ValueError(f"Checkpoint replay diverged at decision {row['decision']}")
    return Checkpoint.capture(
        env,
        parser,
        snapshot,
        {
            "source_log": str(path.resolve()),
            "source_episode": episode_id,
            "before_decision": decision,
            "source_seed": config["seed"],
            "replayed_frames": frame_count,
            "bootstrap": "fresh_immediate_decision",
        },
    )
