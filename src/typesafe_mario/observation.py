"""Explicit observable-data boundary for prediction and provider requests.

Structured telemetry stands in for perception. Coordinates are world x rightward,
y upward with Mario's feet on the terrain support plane. No engine timers, hidden
actors, exact RAM velocities, or native forecasts cross this boundary.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from .timing import DecisionTiming

if TYPE_CHECKING:
    from .state import MarioSnapshot


@dataclass(frozen=True)
class Block:
    left: float
    right: float
    bottom: float
    top: float


@dataclass(frozen=True)
class Actor:
    identity: str
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class Observation:
    frame: int
    level: tuple[int, int, int]
    x: float
    y: float
    vx: float
    vy: float
    grounded: bool
    swimming: bool
    previous_action: str | None
    motion_reliable: bool
    known_x: tuple[float, float] | None
    blocks: tuple[Block, ...]
    actors: tuple[Actor, ...]
    height: int = 16
    terminal: bool = False
    score: int = 0


@dataclass
class VisibleActorTracker:
    """Assign identities using only successive visible observations, not RAM lifetimes."""

    epoch: int = 0
    generation: int = 0
    previous: Observation | None = None
    identities: dict[str, str] = field(default_factory=dict)

    def update(self, observation: Observation) -> dict[str, str]:
        previous = self.previous
        continuous = (
            previous is not None
            and previous.level == observation.level
            and previous.frame + 1 == observation.frame
            and previous.motion_reliable
            and observation.motion_reliable
            and not previous.terminal
        )
        old = {a.identity: a for a in previous.actors} if continuous else {}
        identities = {}
        for actor in observation.actors:
            before = old.get(actor.identity)
            if (
                before is not None
                and before.kind == actor.kind
                and abs(before.x - actor.x) <= 16
                and abs(before.y - actor.y) <= 16
            ):
                identity = self.identities[actor.identity]
            else:
                self.generation += 1
                identity = f"{self.epoch}:{actor.identity}:{self.generation}"
            identities[actor.identity] = identity
        self.previous, self.identities = observation, identities
        return dict(identities)


def observe(snapshot: "MarioSnapshot") -> Observation:
    """Project existing telemetry onto current visible content only."""
    s = snapshot
    blocks = []
    known = None
    viewport = None if s.camera_left_x is None else (s.camera_left_x, s.camera_left_x + 256)
    grid = s.viewport_grid or s.collision_grid
    if viewport is not None and grid:
        origin = (viewport[0] // 16) * 16 if s.viewport_grid else (s.x // 16 - 2) * 16
        left = max(viewport[0], origin)
        right = min(viewport[1], origin + min(map(len, grid)) * 16)
        if left < right:
            known = (left, right)
            for row, cells in enumerate(grid[:13]):
                column = 0
                while column < len(cells):
                    if cells[column] != "#":
                        column += 1
                        continue
                    start = column
                    while column < len(cells) and cells[column] == "#":
                        column += 1
                    lo, hi = max(left, origin + start * 16), min(right, origin + column * 16)
                    if lo < hi:
                        blocks.append(Block(lo, hi, 239 - row * 16, 255 - row * 16))
    actors = []
    if viewport is not None:
        # Use only observed positions, not track velocities/history that may have
        # been sampled while the actor was offscreen, or engine-phase fields.
        for track in s.enemy_tracks:
            m = track.measurement
            if (
                not track.observed
                or m.y_high != 1
                or not (viewport[0] < m.x + 16 and m.x < viewport[1] and 0 <= m.y < 240)
            ):
                continue
            y = 255 - m.y
            # Plants inside their pipes are occluded. Springs have their own
            # solid tiles, so generic solid overlap must not hide other actors.
            if m.kind == "piranha_plant":
                # Without pipe geometry we cannot establish plant visibility.
                if known is None or not (known[0] <= m.x and m.x + 16 <= known[1]):
                    continue
                if all(
                    any(b.left <= x < b.right and b.bottom <= z < b.top for b in blocks)
                    for x, z in ((m.x + 2, y + 2), (m.x + 14, y + 14))
                ):
                    continue
            key = f"{m.slot}:{m.kind}"
            actors.append(Actor(s.visible_actor_ids.get(key, key), m.kind, m.x, y))
    reliable = abs(s.dx) <= 16 and abs(s.dy) <= 16
    return Observation(
        s.frame_index,
        (s.world, s.stage, s.area),
        s.x,
        s.y,
        s.dx if reliable else 0,
        s.dy if reliable else 0,
        s.grounded,
        s.swimming,
        s.previous_action,
        reliable,
        known,
        tuple(blocks),
        tuple(actors),
        16 if s.status == "small" else 32,
        s.dead or s.clear,
        s.score,
    )


def model_state(snapshot: "MarioSnapshot") -> dict:
    """The provider gets this whitelist, never the raw debug snapshot."""
    o = observe(snapshot)
    timing = DecisionTiming(o.frame, snapshot.frames_until_action, snapshot.decision_horizon_frames)
    recovery = snapshot.recovery
    state = {
        "objective": snapshot.goal,
        "observed_at_frame": o.frame,
        "level": dict(zip(("world", "stage", "area"), o.level, strict=True)),
        "player": {
            "x": o.x,
            "y": o.y,
            "center_x": o.x + 8,
            "horizontal_speed_px_per_frame": o.vx,
            "vertical_speed_px_per_frame": o.vy,
            "grounded": o.grounded,
            "swimming": o.swimming,
            "height_pixels": o.height,
            "motion_reliable": o.motion_reliable,
        },
        "terrain": {"known_x": o.known_x, "blocks": [asdict(b) for b in o.blocks]},
        "visible_actors": [asdict(a) for a in o.actors],
        "recovery": {
            k: recovery[k]
            for k in (
                "active",
                "no_progress_frames",
                "anchor_x",
                "action_frames",
                "stationary_frames",
                "stationary_action_frames",
                "local_motion",
                "horizontal_motion",
            )
            if k in recovery
        },
        "reaction_timing": {
            **timing.to_state(),
            "total_reaction_horizon_frames": snapshot.frames_until_action
            + snapshot.decision_horizon_frames,
            "scheduled_action": snapshot.scheduled_action,
            "scheduled_first_frame_action": snapshot.scheduled_first_frame_action,
        },
        "recent_control": {"action": o.previous_action},
        "episode": {"dead": snapshot.dead, "stage_clear": snapshot.clear, "score": snapshot.score},
        "observation_limits": "Only currently visible tile geometry and actor positions. "
        "Unseen space, hidden enemies, spawn rules and internal timers are unknown.",
    }
    pipe = snapshot.side_exit_pipe
    if pipe and o.known_x and o.known_x[0] <= pipe["mouth_x"] < o.known_x[1]:
        state["terrain"]["side_exit_pipe"] = {k: pipe[k] for k in ("mouth_x", "entry_standing_y")}
    if snapshot.prediction and snapshot.prediction.get("backend") == "observation_dynamics":
        state["prediction"] = snapshot.prediction
    return state
