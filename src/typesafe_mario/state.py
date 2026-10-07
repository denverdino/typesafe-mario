from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import ceil
from typing import Any

from .scoring import ScoringObservation, ScoringTracker

ENEMY_NAMES: dict[int, str] = {
    0x00: "green_koopa",
    0x06: "goomba",
    0x05: "hammer_bro",
    0x07: "bloober",
    0x0A: "grey_cheep_cheep",
    0x0B: "red_cheep_cheep",
    0x0D: "piranha_plant",
    0x0E: "green_paratroopa_jump",
    0x12: "spiny",
    0x2D: "bowser",
    0x30: "flagpole_flag",
    0x31: "star_flag",
}


@dataclass(frozen=True)
class EnemyObservation:
    slot: int
    kind_id: int
    kind: str
    dx_pixels: int
    dy_pixels: int
    relative_velocity_x: int = 0
    collision_distance_pixels: int | None = None
    ram_relative_speed_x: float | None = None
    active_flag: int | None = None
    motion_state: int | None = None
    collision_box: tuple[int, int, int, int] | None = None
    vertical_overlap: bool | None = None

    @property
    def defeated(self) -> bool | None:
        if self.kind_id not in (0x00, 0x06) or self.motion_state is None:
            return None
        return bool(self.motion_state & 0x20 or (self.kind_id == 6 and self.motion_state == 4))

    @property
    def prediction_velocity_x(self) -> float:
        return (
            self.ram_relative_speed_x
            if self.ram_relative_speed_x is not None
            else self.relative_velocity_x
        )

    def to_state(self) -> dict[str, Any]:
        horizontal = "ahead" if self.dx_pixels >= 0 else "behind"
        return {
            "slot": self.slot,
            "kind": self.kind,
            "kind_id": self.kind_id,
            "relative_x_pixels": self.dx_pixels,
            "relative_y_pixels": self.dy_pixels,
            "relative_velocity_x": self.relative_velocity_x,
            "collision_distance_pixels": self.collision_distance_pixels,
            "ram_relative_speed_x": self.ram_relative_speed_x,
            "horizontal_relation": horizontal,
            "active_flag": self.active_flag,
            "motion_state": self.motion_state,
            "defeated": self.defeated,
            "collision_box_screen_xyxy": list(self.collision_box) if self.collision_box else None,
            "vertical_overlap": self.vertical_overlap,
        }


@dataclass(frozen=True)
class CommittedControl:
    action: str
    first_frame_action: str
    frames_before_selected_action: int
    release_jump_while_falling: bool
    press_jump_on_landing: bool


@dataclass(frozen=True)
class MarioSnapshot:
    goal: str
    world: int
    stage: int
    area: int
    x: int
    y: int
    dx: int
    dy: int
    direction: str
    vertical_motion: str
    airborne: bool
    status: str
    player_state: int
    lives: int
    coins: int
    score: int
    time_left: int
    progress: int
    best_progress: int
    stalled_steps: int
    enemies: tuple[EnemyObservation, ...] = field(default_factory=tuple)
    local_grid: tuple[str, ...] = field(default_factory=tuple)
    previous_action: str | None = None
    previous_reward: float = 0.0
    dead: bool = False
    clear: bool = False
    grounded: bool = True
    jump_phase: str = "grounded"
    action_frames: int = 0
    action_progress: int = 0
    airborne_frames: int = 0
    jump_distance_pixels: int = 0
    crossing_gap: bool = False
    gap_width_at_commit: int = 0
    decision_horizon_frames: int = 8
    last_response_delay_frames: int = 0
    last_grounded_gap_distance_tiles: int | None = None
    last_grounded_gap_width_tiles: int = 0
    last_grounded_gap_kind: str | None = None
    last_grounded_obstacle_distance_tiles: int | None = None
    last_grounded_obstacle_height_tiles: int = 0
    # Complete block-buffer columns aligned with local_grid, used only for terrain.
    terrain_support_y: tuple[tuple[int, ...], ...] = ()
    terrain_sample_y: int = 0
    committed_control: CommittedControl | None = None
    # Controller-only state: a submitted A press has not reached game logic yet.
    jump_press_pending: bool = False
    estimated_landing_frames: int | None = None
    observation_frame: int = 0
    last_grounded_sample_frame: int | None = None
    last_grounded_sample_x: int | None = None
    last_grounded_gap_right_edge_visible: bool = False
    player_collision_box: tuple[int, int, int, int] | None = None
    foot_y_screen: int | None = None
    scoring: ScoringObservation | None = None

    def navigation_features(self) -> dict[str, Any]:
        rows = self.local_grid
        if not rows:
            return {
                "geometry_available": False,
                "summary": "Local collision geometry is unavailable.",
            }

        mario_row = mario_col = -1
        for row_index, row in enumerate(rows):
            if "M" in row:
                mario_row = row_index
                mario_col = row.index("M")
                break
        if mario_row < 0:
            return {
                "geometry_available": False,
                "summary": "Mario was not located in the collision grid.",
            }

        ground_row = next(
            (row for row in range(mario_row + 1, len(rows)) if rows[row][mario_col] == "#"),
            None,
        )
        if ground_row is None:
            return {
                "geometry_available": False,
                "summary": "Ground support is outside the local grid; forward clearance is unknown.",
            }
        obstacle_distance: int | None = None
        obstacle_height = 0
        gap_distance: int | None = None
        gap_width = 0
        gap_closed = False
        clear_forward = 0
        drop_distance: int | None = None
        drop_height: int | None = None
        complete_support = len(self.terrain_support_y) == len(rows[0])
        ground_y = (self.terrain_sample_y + (ground_row - mario_row) * 16) // 16 * 16

        for column in range(mario_col + 1, len(rows[0])):
            distance = column - mario_col
            if ground_row is not None:
                height = 0
                row = ground_row - 1
                while row >= 0 and rows[row][column] == "#":
                    height += 1
                    row -= 1
                if height and obstacle_distance is None:
                    obstacle_distance = distance
                    obstacle_height = height
                supported = rows[ground_row][column] == "#"
                if complete_support:
                    support_y = next(
                        (y for y in self.terrain_support_y[column] if y >= ground_y), None
                    )
                    supported = support_y is not None
                    if support_y is not None and support_y > ground_y and drop_distance is None:
                        drop_distance = distance
                        drop_height = support_y - ground_y
                if not supported and gap_distance is None:
                    gap_distance = distance
                if gap_distance is not None:
                    if supported:
                        gap_closed = True
                    elif not gap_closed:
                        # Measure the first contiguous gap, not later separate gaps.
                        gap_width += 1
            if obstacle_distance is None and gap_distance is None and drop_distance is None:
                clear_forward = distance

        obstacle_ahead = obstacle_distance is not None and obstacle_distance <= 3
        gap_ahead = gap_distance is not None and gap_distance <= 3
        if obstacle_ahead:
            summary = (
                f"Blocking obstacle {obstacle_distance} tile(s) ahead, "
                f"{obstacle_height} tile(s) high. A forward jump is required."
            )
        elif gap_ahead:
            summary = f"Gap begins {gap_distance} tile(s) ahead. A forward jump is required."
        elif drop_distance is not None:
            summary = f"Lower landing {drop_distance} tile(s) ahead, {drop_height}px below."
        else:
            summary = f"Forward path is clear for at least {clear_forward} tile(s)."
        return {
            "geometry_available": True,
            "obstacle_ahead": obstacle_ahead,
            "obstacle_distance_tiles": obstacle_distance,
            "obstacle_height_tiles": obstacle_height,
            "gap_ahead": gap_ahead,
            "gap_distance_tiles": gap_distance,
            "gap_width_tiles_visible": gap_width,
            "gap_kind": ("pit" if complete_support else "unknown") if gap_distance else None,
            "gap_start_world_x": (self.x // 16 + gap_distance) * 16 if gap_distance else None,
            "gap_end_world_x": (
                (self.x // 16 + gap_distance + gap_width) * 16
                if gap_distance and gap_closed
                else None
            ),
            "gap_right_edge_visible": gap_closed,
            "drop_ahead": drop_distance is not None and drop_distance <= 3,
            "drop_distance_tiles": drop_distance,
            "drop_height_pixels": drop_height,
            "clear_forward_tiles": clear_forward,
            "summary": summary,
        }

    def threat_features(self) -> dict[str, Any]:
        landing_frames = self.estimated_landing_frames
        landing_source = "ram_descent_static_tiles" if landing_frames is not None else "unknown"
        # StarFlagObject shares the enemy RAM slots but is a level-end object.
        # Preserve it in observations while excluding it from enemy projections.
        enemies_ahead = sorted(
            (
                enemy
                for enemy in self.enemies
                if enemy.dx_pixels >= 0 and enemy.defeated is not True and enemy.kind_id != 0x31
            ),
            key=lambda enemy: enemy.dx_pixels,
        )
        if not enemies_ahead:
            return {
                "enemy_ahead": False,
                "upcoming_enemies": [],
                "spacing_to_second_enemy_pixels": None,
                "nearest_enemy_kind": None,
                "nearest_enemy_distance_pixels": None,
                "nearest_enemy_collision_distance_pixels": None,
                "relative_velocity_x": None,
                "closing_speed_pixels_per_frame": 0,
                "estimated_contact_frames": None,
                "contact_prediction_model": None,
                "velocity_source": None,
                "jump_clearance_frames_required": 8,
                "takeoff_deadline_frames": None,
                "jump_must_start_this_decision": False,
                "takeoff_window_already_missed": False,
                "projected_distance_after_reaction_pixels": None,
                "contact_within_reaction_horizon": False,
                "estimated_landing_frames": landing_frames,
                "landing_prediction_model": landing_source,
                "will_land_before_contact": None,
            }
        nearest = enemies_ahead[0]
        velocity = nearest.prediction_velocity_x
        closing_speed = max(0, -velocity)
        contact_distance = (
            nearest.collision_distance_pixels
            if nearest.collision_distance_pixels is not None
            else nearest.dx_pixels
        )
        contact_round = ceil if nearest.collision_distance_pixels is not None else round
        contact_frames = (
            contact_round(contact_distance / closing_speed) if closing_speed > 0 else None
        )
        reaction_horizon = self.decision_horizon_frames + self.last_response_delay_frames
        projected_distance = max(
            0,
            nearest.dx_pixels + velocity * reaction_horizon,
        )
        jump_clearance_frames = 8
        takeoff_deadline = (
            contact_frames - self.last_response_delay_frames - jump_clearance_frames
            if contact_frames is not None
            else None
        )
        jump_must_start_now = bool(
            (
                self.grounded
                or (
                    landing_frames is not None and landing_frames <= self.last_response_delay_frames
                )
            )
            and takeoff_deadline is not None
            # A missed clearance budget is still urgent, not permission to run.
            and takeoff_deadline <= self.decision_horizon_frames
        )
        upcoming_enemies = [
            {
                "kind": enemy.kind,
                "distance_pixels": enemy.dx_pixels,
                "collision_distance_pixels": enemy.collision_distance_pixels,
                "vertical_offset_pixels": enemy.dy_pixels,
                "relative_velocity_x": enemy.prediction_velocity_x,
                "projected_distance_after_reaction_pixels": max(
                    0,
                    enemy.dx_pixels + enemy.prediction_velocity_x * reaction_horizon,
                ),
            }
            for enemy in enemies_ahead[:3]
        ]
        return {
            "enemy_ahead": True,
            "upcoming_enemies": upcoming_enemies,
            "spacing_to_second_enemy_pixels": (
                enemies_ahead[1].dx_pixels - nearest.dx_pixels if len(enemies_ahead) > 1 else None
            ),
            "nearest_enemy_kind": nearest.kind,
            "nearest_enemy_distance_pixels": nearest.dx_pixels,
            "nearest_enemy_collision_distance_pixels": nearest.collision_distance_pixels,
            "relative_velocity_x": velocity,
            "closing_speed_pixels_per_frame": closing_speed,
            "estimated_contact_frames": contact_frames,
            "contact_prediction_model": (
                "constant_velocity_bbox"
                if nearest.collision_distance_pixels is not None
                else "constant_velocity_origin_fallback"
            ),
            "velocity_source": (
                "ram_q4_walker" if nearest.ram_relative_speed_x is not None else "frame_delta"
            ),
            "jump_clearance_frames_required": jump_clearance_frames,
            "takeoff_deadline_frames": takeoff_deadline,
            "jump_must_start_this_decision": jump_must_start_now,
            "takeoff_window_already_missed": (
                takeoff_deadline is not None and takeoff_deadline < 0
            ),
            "projected_distance_after_reaction_pixels": projected_distance,
            "contact_within_reaction_horizon": (
                contact_frames is not None and contact_frames <= reaction_horizon
            ),
            "estimated_landing_frames": landing_frames,
            "landing_prediction_model": landing_source,
            "will_land_before_contact": (
                landing_frames < contact_frames
                if landing_frames is not None and contact_frames is not None
                else None
            ),
        }

    def to_state(self) -> dict[str, Any]:
        observation = self.scoring
        reasons = []
        if observation is None or not observation.time_known:
            reasons.append("unknown_time")
        elif self.time_left <= 100:
            reasons.append("low_time")
        if observation is not None and observation.no_best_progress_frames >= 48:
            reasons.append("no_best_progress")
        scoring = {
            name: getattr(observation, name) if observation is not None else None
            for name in (
                "score",
                "coin_counter",
                "score_delta_since_previous_observation",
                "confirmed_coins_collected",
                "confirmed_stomps",
                "unattributed_score_gain",
                "observation_frame",
            )
        }
        scoring["pursuit"] = {
            "allow_extra_effort": not reasons,
            "no_best_progress_frames": (
                observation.no_best_progress_frames if observation is not None else None
            ),
            "reasons": reasons,
        }
        opportunities = (
            asdict(observation.opportunities)
            if observation is not None
            else {
                "availability": "unavailable",
                "observation_frame": self.observation_frame,
                "world_x_bounds": None,
                "items": (),
            }
        )
        opportunities["truncated"] = len(opportunities["items"]) > 8
        opportunities["items"] = list(opportunities["items"][:8])
        opportunities["relative_y_positive"] = "down"
        opportunities["world_x_bounds_convention"] = "[left, right)"
        terrain = self.navigation_features()
        terrain.pop("summary", None)
        terrain["observation_reliability"] = (
            ("high" if self.grounded else "low_airborne")
            if terrain["geometry_available"]
            else "unavailable"
        )
        terrain["last_grounded_preview"] = {
            "sample_frame": self.last_grounded_sample_frame,
            "sample_x": self.last_grounded_sample_x,
            "age_frames": (
                self.observation_frame - self.last_grounded_sample_frame
                if self.last_grounded_sample_frame is not None
                else None
            ),
            "displacement_since_sample_pixels": (
                self.x - self.last_grounded_sample_x
                if self.last_grounded_sample_x is not None
                else None
            ),
            "gap_distance_tiles": self.last_grounded_gap_distance_tiles,
            "gap_width_tiles_visible": self.last_grounded_gap_width_tiles,
            "gap_kind": self.last_grounded_gap_kind,
            "obstacle_distance_tiles": self.last_grounded_obstacle_distance_tiles,
            "obstacle_height_tiles": self.last_grounded_obstacle_height_tiles,
        }
        preview = terrain["last_grounded_preview"]
        gap_start = (
            (self.last_grounded_sample_x // 16 + self.last_grounded_gap_distance_tiles) * 16
            if self.last_grounded_sample_x is not None
            and self.last_grounded_gap_distance_tiles is not None
            else None
        )
        preview["gap_start_world_x"] = gap_start
        preview["gap_end_world_x"] = (
            gap_start + self.last_grounded_gap_width_tiles * 16
            if gap_start is not None and self.last_grounded_gap_right_edge_visible
            else None
        )
        return {
            "objective": self.goal,
            "scoring": scoring,
            "opportunities": opportunities,
            "level": {"world": self.world, "stage": self.stage, "area": self.area},
            "player": {
                "x": self.x,
                "y": self.y,
                "horizontal_speed_px_per_frame": self.dx,
                "vertical_speed_px_per_frame": self.dy,
                "grounded": self.grounded,
                "jump_phase": self.jump_phase,
                "powerup_status": self.status,
                "collision_box_screen_xyxy": self.player_collision_box,
            },
            "trajectory": {
                "airborne_frames": self.airborne_frames,
                "horizontal_distance_since_takeoff_pixels": self.jump_distance_pixels,
                "crossing_known_gap": self.crossing_gap,
                "gap_width_at_commit_tiles": self.gap_width_at_commit,
            },
            "hazard": self.threat_features(),
            "terrain": terrain,
            "reaction_timing": {
                "action_horizon_frames": self.decision_horizon_frames,
                "last_inference_delay_frames": self.last_response_delay_frames,
                "total_reaction_horizon_frames": (
                    self.decision_horizon_frames + self.last_response_delay_frames
                ),
            },
            "recent_control": {
                "action": self.previous_action,
                "frames_observed": self.action_frames,
                "progress_gained_pixels": self.action_progress,
                "outcome": self.control_outcome(),
            },
            "committed_control": asdict(self.committed_control) if self.committed_control else None,
            "episode": {
                "lives": self.lives,
                "time_left": self.time_left,
                "progress": self.progress,
                "best_progress": self.best_progress,
                "stalled_frames": self.stalled_steps,
                "dead": self.dead,
                "stage_clear": self.clear,
            },
        }

    def control_outcome(self) -> str:
        if self.dead:
            return "death"
        if self.clear:
            return "stage_clear"
        if self.previous_action is None or self.action_frames < 2:
            return "not_enough_evidence"
        if not self.grounded:
            return "jump_in_progress"
        if self.action_progress > 0:
            return "advanced"
        if self.stalled_steps >= 4:
            return "blocked"
        return "no_progress_yet"

    def to_debug_state(self) -> dict[str, Any]:
        return {
            "model_state": self.to_state(),
            "scoring_observations": asdict(self.scoring) if self.scoring is not None else None,
            "raw": {
                "player_state": self.player_state,
                "jump_press_pending": self.jump_press_pending,
                "coins": self.coins,
                "score": self.score,
                "previous_reward": self.previous_reward,
                "horizontal_motion": self.direction,
                "vertical_motion": self.vertical_motion,
            },
            "observed_geometry": {
                "observation_frame": self.observation_frame,
                "collision_box_screen_xyxy": (
                    list(self.player_collision_box) if self.player_collision_box else None
                ),
                "foot_probes": {
                    "x_world": [self.x + 3, self.x + 12],
                    "y_screen": self.foot_y_screen,
                },
                "static_columns": [
                    {
                        "x_start_world": (self.x // 16 + i - 2) * 16,
                        "x_end_world": (self.x // 16 + i - 1) * 16,
                        "surface_y_screen": [y for y in column if y - 16 not in column],
                    }
                    for i, column in enumerate(self.terrain_support_y)
                ],
            },
            "visible_enemies": [enemy.to_state() for enemy in self.enemies],
            "local_grid": {
                "legend": {".": "empty", "#": "solid", "E": "enemy", "M": "mario"},
                "orientation": "Mario is at M; columns run left-to-right and rows top-to-bottom.",
                "rows": list(self.local_grid),
            },
        }

    def to_text(self) -> str:
        enemy_text = (
            ", ".join(
                f"{enemy.kind} {abs(enemy.dx_pixels)}px "
                f"{'ahead' if enemy.dx_pixels >= 0 else 'behind'}"
                for enemy in self.enemies
            )
            or "none visible"
        )
        grid = "\n".join(self.local_grid) if self.local_grid else "(unavailable)"
        threat = self.threat_features()
        threat_text = (
            "none visible"
            if not threat["enemy_ahead"]
            else (
                f"{threat['nearest_enemy_kind']} {threat['nearest_enemy_distance_pixels']}px "
                f"ahead; contact estimate={threat['estimated_contact_frames']} frames"
            )
        )
        return (
            f"Goal: {self.goal}\n"
            f"Mario: x={self.x} y={self.y}, {self.direction}, "
            f"vertical={self.vertical_motion}, airborne={self.airborne}, status={self.status}\n"
            f"Progress: {self.progress} (best {self.best_progress}), "
            f"time={self.time_left}, lives={self.lives}, stalled={self.stalled_steps}\n"
            f"Navigation: {self.navigation_features()['summary']}\n"
            f"Threats: {threat_text}\n"
            f"Nearby enemies: {enemy_text}\n"
            "Local grid (# solid, . empty, E enemy, M Mario):\n"
            f"{grid}"
        )


class MarioStateParser:
    """Convert Gym/RAM observations into compact structured game state.

    The RAM grid is deliberately coarse. It communicates local collision geometry,
    not artwork. Addresses follow the original SMB NES memory layout used by the
    Gym wrapper and historical Mario AI tooling.
    """

    def __init__(
        self,
        goal: str = (
            "Reach the flag in World 1-1 without dying. Prioritize survival and forward "
            "progress, then safely obtain mushroom/fire-flower upgrades, collect visible coins, "
            "hit reachable reward blocks from below, and stomp suitable enemies "
            "for additional score when the observed geometry supports a low-risk opportunity."
        ),
        decision_horizon_frames: int = 8,
    ) -> None:
        self.goal = goal
        self._scoring_tracker = ScoringTracker()
        self.decision_horizon_frames = decision_horizon_frames
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._best_x = 0
        self._stalled_steps = 0
        self._last_enemy_dx: dict[int, int] = {}
        self._last_enemy_kind: dict[int, int] = {}
        self._tracked_action: str | None = None
        self._action_start_x = 0
        self._action_frames = 0
        self._last_jump_phase = "grounded"
        self._airborne_frames = 0
        self._takeoff_x = 0
        self._crossing_gap = False
        self._gap_width_at_commit = 0
        self._last_grounded_gap_distance: int | None = None
        self._last_grounded_gap_width = 0
        self._last_grounded_gap_kind: str | None = None
        self._last_grounded_obstacle_distance: int | None = None
        self._last_grounded_obstacle_height = 0
        self._observation_frame = -1
        self._last_grounded_sample_frame: int | None = None
        self._last_grounded_sample_x: int | None = None
        self._last_grounded_gap_right_edge_visible = False

    def reset(self) -> None:
        """Clear episode history while preserving the configured objective."""
        self.__init__(
            goal=self.goal,
            decision_horizon_frames=self.decision_horizon_frames,
        )

    @staticmethod
    def _integer(info: Mapping[str, Any], key: str, default: int = 0) -> int:
        value = info.get(key, default)
        return int(value) if value is not None else default

    @staticmethod
    def _ram_byte(ram: Sequence[int] | None, address: int, default: int = 0) -> int:
        if ram is None or address < 0 or address >= len(ram):
            return default
        return int(ram[address])

    def parse(
        self,
        info: Mapping[str, Any],
        ram: Sequence[int] | None = None,
        *,
        previous_action: str | None = None,
        previous_reward: float = 0.0,
        previous_latency_ms: float = 0.0,
        previous_response_delay_frames: int | None = None,
    ) -> MarioSnapshot:
        """Parse the initial observation, then once per actual emulator step.

        Velocities are pixels per emulator frame; counters measure emulator frames.
        Do not call again for UI refreshes while the emulator is paused.
        """
        self._observation_frame += 1
        x = self._integer(info, "x_pos", self._integer(info, "progress"))
        y = self._integer(info, "y_pos", self._integer(info, "y_pixel"))
        screen_y = self._integer(
            info,
            "y_pixel",
            self._ram_byte(ram, 0x03B8, max(0, 255 - y)),
        )
        dx = 0 if self._last_x is None else x - self._last_x
        dy = 0 if self._last_y is None else y - self._last_y

        if dx > 1:
            direction = "moving_right"
        elif dx < -1:
            direction = "moving_left"
        else:
            direction = "nearly_stationary"

        if dy > 1:
            vertical_motion = "rising"
        elif dy < -1:
            vertical_motion = "falling"
        else:
            vertical_motion = "level_or_grounded"

        self._best_x = max(self._best_x, self._integer(info, "progress_max", x), x)
        if self._last_x is None:
            self._stalled_steps = 0
        elif x <= self._last_x:
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0

        enemies = self._extract_enemies(info, ram, x, screen_y)
        grid = self._extract_local_grid(ram, x, screen_y, enemies)
        support_below = self._has_support_below(grid)
        grounded = abs(dy) <= 1 and support_below
        dead = bool(info.get("death", info.get("is_dead", False)))
        clear = bool(info.get("clear", info.get("flag_get", False)))
        if ram is not None and len(ram) > 0x001D:
            # SMB1 Player_State: 0=ground, 1=jump/swim, 2=fall, 3=climb.
            # $000E is the engine subroutine, not the player's movement state.
            grounded = self._ram_byte(ram, 0x000E) == 8 and self._ram_byte(ram, 0x001D) == 0
        grounded = grounded and not dead and not clear
        if grounded:
            jump_phase = "grounded"
        elif dy > 1:
            jump_phase = "rising"
        elif dy < -1:
            jump_phase = "falling"
        elif self._last_jump_phase == "rising":
            jump_phase = "apex"
        else:
            jump_phase = "airborne"

        if grounded:
            self._airborne_frames = 0
            self._takeoff_x = x
            self._crossing_gap = False
            self._gap_width_at_commit = 0
        else:
            if self._airborne_frames == 0:
                self._takeoff_x = self._last_x if self._last_x is not None else x
                if (
                    self._last_grounded_gap_kind == "pit"
                    and self._last_grounded_gap_distance is not None
                    and self._last_grounded_gap_distance <= 3
                ):
                    self._crossing_gap = True
                    self._gap_width_at_commit = self._last_grounded_gap_width
            self._airborne_frames += 1

        if previous_action != self._tracked_action:
            self._tracked_action = previous_action
            self._action_start_x = x
            self._action_frames = 1 if previous_action else 0
        elif previous_action is not None:
            self._action_frames += 1
        action_progress = x - self._action_start_x

        snapshot = MarioSnapshot(
            goal=self.goal,
            world=self._integer(info, "world", 1),
            stage=self._integer(info, "stage", 1),
            area=self._integer(info, "area", 1),
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            direction=direction,
            vertical_motion=vertical_motion,
            airborne=not grounded,
            status=str(info.get("status", "small")),
            player_state=self._integer(info, "player_state", 8),
            lives=self._integer(info, "life", self._integer(info, "lives", 0)),
            coins=self._integer(info, "coins"),
            score=self._integer(info, "score"),
            time_left=self._integer(info, "time"),
            progress=self._integer(info, "progress", x),
            best_progress=self._best_x,
            stalled_steps=self._stalled_steps,
            enemies=tuple(enemies),
            local_grid=tuple(grid),
            terrain_support_y=self._extract_support_columns(ram, x),
            terrain_sample_y=screen_y,
            previous_action=previous_action,
            jump_press_pending=bool(
                self._ram_byte(ram, 0x06FC) & 0x80  # SavedJoypadBits: submitted A
                and not self._ram_byte(ram, 0x000A) & 0x80  # A_B_Buttons: processed A
            ),
            estimated_landing_frames=self._estimate_landing(ram),
            observation_frame=self._observation_frame,
            last_grounded_sample_frame=self._last_grounded_sample_frame,
            last_grounded_sample_x=self._last_grounded_sample_x,
            last_grounded_gap_right_edge_visible=self._last_grounded_gap_right_edge_visible,
            player_collision_box=self._collision_box(ram, 0x04AC),
            foot_y_screen=(
                self._ram_byte(ram, 0xCE) + 32
                if ram is not None and len(ram) > 0xCE and ram[0xB5] == 1
                else None
            ),
            previous_reward=float(previous_reward),
            dead=dead,
            clear=clear,
            grounded=grounded,
            jump_phase=jump_phase,
            action_frames=self._action_frames,
            action_progress=action_progress,
            airborne_frames=self._airborne_frames,
            jump_distance_pixels=max(0, x - self._takeoff_x),
            crossing_gap=self._crossing_gap,
            gap_width_at_commit=self._gap_width_at_commit,
            decision_horizon_frames=self.decision_horizon_frames,
            last_response_delay_frames=(
                max(0, previous_response_delay_frames)
                if previous_response_delay_frames is not None
                else max(0, round(previous_latency_ms / (1000 / 60)))
            ),
            last_grounded_gap_distance_tiles=self._last_grounded_gap_distance,
            last_grounded_gap_width_tiles=self._last_grounded_gap_width,
            last_grounded_gap_kind=self._last_grounded_gap_kind,
            last_grounded_obstacle_distance_tiles=self._last_grounded_obstacle_distance,
            last_grounded_obstacle_height_tiles=self._last_grounded_obstacle_height,
        )
        navigation = snapshot.navigation_features()
        # Missing geometry says nothing about whether the previous gap remains.
        # Keep its original timestamp and position until a valid sample replaces it.
        if grounded and navigation["geometry_available"]:
            self._last_grounded_gap_distance = navigation.get("gap_distance_tiles")
            self._last_grounded_gap_width = int(navigation.get("gap_width_tiles_visible", 0))
            self._last_grounded_gap_kind = navigation.get("gap_kind")
            self._last_grounded_obstacle_distance = navigation.get("obstacle_distance_tiles")
            self._last_grounded_obstacle_height = int(navigation.get("obstacle_height_tiles", 0))
            self._last_grounded_sample_frame = self._observation_frame
            self._last_grounded_sample_x = x
            self._last_grounded_gap_right_edge_visible = navigation.get(
                "gap_right_edge_visible", False
            )
            snapshot = replace(
                snapshot,
                last_grounded_sample_frame=self._last_grounded_sample_frame,
                last_grounded_sample_x=x,
                last_grounded_gap_right_edge_visible=self._last_grounded_gap_right_edge_visible,
                last_grounded_gap_distance_tiles=self._last_grounded_gap_distance,
                last_grounded_gap_width_tiles=self._last_grounded_gap_width,
                last_grounded_gap_kind=self._last_grounded_gap_kind,
                last_grounded_obstacle_distance_tiles=self._last_grounded_obstacle_distance,
                last_grounded_obstacle_height_tiles=self._last_grounded_obstacle_height,
            )
        elif (
            navigation.get("gap_kind") == "pit"
            and navigation.get("gap_distance_tiles") is not None
            and navigation["gap_distance_tiles"] <= 3
        ):
            self._crossing_gap = True
            self._gap_width_at_commit = max(
                self._gap_width_at_commit,
                int(navigation.get("gap_width_tiles_visible", 0)),
            )
            snapshot = replace(
                snapshot,
                crossing_gap=True,
                gap_width_at_commit=self._gap_width_at_commit,
            )
        self._last_x = x
        self._last_y = y
        self._last_jump_phase = jump_phase
        return replace(snapshot, scoring=self._scoring_tracker.observe(info, ram, snapshot))

    @staticmethod
    def _has_support_below(grid: Sequence[str]) -> bool:
        for row_index, row in enumerate(grid):
            if "M" not in row:
                continue
            column = row.index("M")
            return any(
                grid[next_row][column] == "#"
                for next_row in range(row_index + 1, min(len(grid), row_index + 3))
            )
        return False

    def _extract_support_columns(
        self, ram: Sequence[int] | None, mario_x: int
    ) -> tuple[tuple[int, ...], ...]:
        if ram is None or len(ram) < 0x06A0:
            return ()
        columns = []
        for offset in range(-2, 9):
            sample_x = mario_x + offset * 16
            base = 0x0500 + (sample_x // 256 % 2) * 208 + sample_x % 256 // 16
            columns.append(
                tuple(
                    32 + row * 16
                    for row in range(13)
                    if self._supports_landing(self._ram_byte(ram, base + row * 16))
                )
            )
        return tuple(columns)

    @staticmethod
    def _supports_landing(tile: int) -> bool:
        # SMB1 DoFootCheck excludes coins, invisible blocks, the axe, and
        # climbable metatiles (ClimbMTileUpperExt indexed by the top two bits).
        return (
            tile not in (0, 0x5F, 0x60, 0xC2, 0xC3, 0xC5)
            and tile < (0x24, 0x6D, 0x8A, 0xC6)[tile >> 6]
        )

    def _estimate_landing(self, ram: Sequence[int] | None) -> int | None:
        """Project an SMB1 descent onto static tiles, at constant horizontal speed.

        Upward motion, swimming, springs and unknown support have no estimate.
        The 32-step limit bounds projection; it is not a jump-duration assumption.
        """
        if (
            ram is None
            or len(ram) <= 0x0747
            or ram[0x0E] != 8
            or ram[0x1D] not in (1, 2)
            or ram[0xB5] != 1
            or ram[0x9F] >= 128
            or ram[0x0704]  # SwimmingFlag
            or ram[0x070E]  # JumpspringAnimCtrl
            or ram[0x0747]  # TimerControl: motion frozen
        ):
            return None
        x = origin_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        y = origin_y = int(ram[0xCE])
        vx = (int(ram[0x57]) + 128) % 256 - 128  # signed Q4 pixels/frame
        vy = int(ram[0x9F])
        x_fraction = int(ram[0x0400])
        y_fraction = int(ram[0x0416])
        y_force = int(ram[0x0433])
        gravity = int(ram[0x070A])
        player_box = self._collision_box(ram, 0x04AC)
        enemy_boxes = []
        for slot in range(5):
            state = int(ram[0x1E + slot])
            if not ram[0x0F + slot] or state & 0x20 or (ram[0x16 + slot] == 6 and state == 4):
                continue
            box = self._collision_box(ram, 0x04B0 + 4 * slot)
            if box is not None:
                speed = (
                    ((int(ram[0x58 + slot]) + 128) % 256 - 128) / 16
                    if ram[0x16 + slot] in (0, 6) and state == 0
                    else 0
                )
                enemy_boxes.append((box, speed))
        for frames in range(1, 33):
            previous_foot_y = y + 32
            dx, x_fraction = divmod(x_fraction + vx * 16, 256)
            x += dx
            carry, y_fraction = divmod(y_fraction + y_force, 256)
            y += vy + carry
            carry, y_force = divmod(y_force + gravity, 256)
            vy += carry
            # SMB1 ImposeGravity: cap only after the fractional threshold.
            # The installed SMB1 ROM uses a maximum of 5, confirmed by trace.
            if vy >= 5 and y_force >= 0x80:
                vy, y_force = 5, 0
            if player_box is not None:
                left, top, right, bottom = player_box
                # Do not let constant-X projection pass through a side wall.
                # Only a surface approached from above can support a landing.
                relative_x = int(ram[0x03AD])
                for tile_x in range(
                    (x + left - relative_x) // 16, (x + right - relative_x) // 16 + 1
                ):
                    if not -2 <= tile_x - origin_x // 16 <= 8:
                        return None
                    for row in range(
                        max(0, (top + y - origin_y - 32) // 16),
                        min(13, (bottom + y - origin_y - 33) // 16 + 1),
                    ):
                        address = 0x0500 + (tile_x // 16 % 2) * 208 + row * 16 + tile_x % 16
                        if 32 + row * 16 < previous_foot_y and self._supports_landing(
                            int(ram[address])
                        ):
                            return None
                for (e_left, e_top, e_right, e_bottom), speed in enemy_boxes:
                    if (
                        left + x - origin_x <= e_right + speed * frames
                        and right + x - origin_x >= e_left + speed * frames
                        and top + y - origin_y <= e_bottom
                        and bottom + y - origin_y >= e_top
                    ):
                        return None  # Enemy contact/stomping can change this trajectory.
            if y >= 0xCF:  # DoFootCheck no longer checks below this height.
                return None
            if y % 16 >= 5:
                continue
            # SMB1's two foot probes: X+3/X+12, Y+32. Buffer starts at Y=32.
            tiles = []
            for foot_x in (x + 3, x + 12):
                if not -2 <= foot_x // 16 - origin_x // 16 <= 8:
                    return None  # Do not extrapolate outside observed columns.
                row = y // 16
                if 0 <= row < 13:
                    address = 0x0500 + (foot_x // 256 % 2) * 208 + row * 16 + foot_x % 256 // 16
                    tiles.append(int(ram[address]))
                else:
                    return None
            # A nonempty left tile wins, including invisible/climbable tiles.
            if self._supports_landing(tiles[0] or tiles[1]):
                selected_x = x + (3 if tiles[0] else 12)
                address = 0x0500 + (selected_x // 256 % 2) * 208 + row * 16 + selected_x % 256 // 16
                surface_y = 32 + row * 16
                if not previous_foot_y <= surface_y <= y + 32 or (
                    row > 0 and self._supports_landing(int(ram[address - 16]))
                ):
                    return None  # Side entry or a buried tile, not a top surface.
                return frames
        return None

    @staticmethod
    def _collision_box(ram: Sequence[int] | None, address: int) -> tuple[int, int, int, int] | None:
        if ram is None or len(ram) < address + 4:
            return None
        left, top, right, bottom = (int(v) for v in ram[address : address + 4])
        if not (0 <= left < right < 255 and 0 <= top < bottom < 255):
            return None  # Uninitialized, offscreen or wrapped boxes.
        return left, top, right, bottom

    def _collision_distance(self, ram: Sequence[int] | None, slot: int) -> int | None:
        player = self._collision_box(ram, 0x04AC)
        enemy = self._collision_box(ram, 0x04B0 + 4 * slot)
        if player is None or enemy is None:
            return None
        return max(0, enemy[0] - player[2])

    def _extract_enemies(
        self,
        info: Mapping[str, Any],
        ram: Sequence[int] | None,
        mario_x: int,
        mario_screen_y: int,
    ) -> list[EnemyObservation]:
        enemies: list[EnemyObservation] = []
        fallback_types = tuple(int(value) for value in info.get("enemy_types", ()))
        for slot in range(5):
            active = self._ram_byte(ram, 0x000F + slot, 0)
            kind_id = self._ram_byte(
                ram,
                0x0016 + slot,
                fallback_types[slot] if slot < len(fallback_types) else 0,
            )
            # Type zero is a green Koopa when RAM confirms an active object.
            # Without RAM, zero in enemy_types remains an ambiguous empty slot.
            if (ram is not None and active == 0) or (ram is None and kind_id == 0):
                self._last_enemy_dx.pop(slot, None)
                self._last_enemy_kind.pop(slot, None)
                continue
            if ram is None:
                enemy_x = mario_x
                enemy_y = mario_screen_y
            else:
                enemy_x = self._ram_byte(ram, 0x006E + slot) * 256 + self._ram_byte(
                    ram, 0x0087 + slot
                )
                enemy_y = self._ram_byte(ram, 0x00CF + slot)
            dx = enemy_x - mario_x
            dy = enemy_y - mario_screen_y
            if ram is not None and len(ram) > 0x00CF + slot:
                # Both Y coordinates include a page byte; low-byte subtraction
                # flips above/below when Mario or an enemy crosses a page edge.
                mario_y = self._ram_byte(ram, 0x00B5) * 256 + self._ram_byte(ram, 0x00CE)
                enemy_y += self._ram_byte(ram, 0x00B6 + slot) * 256
                dy = enemy_y - mario_y
            previous_dx = (
                self._last_enemy_dx.get(slot)
                if self._last_enemy_kind.get(slot) == kind_id
                else None
            )
            relative_velocity_x = 0 if previous_dx is None else dx - previous_dx
            self._last_enemy_dx[slot] = dx
            self._last_enemy_kind[slot] = kind_id
            collision_distance = self._collision_distance(ram, slot)
            ram_relative_speed = None
            # These ground walkers use MoveObjectHorizontally's signed Q4 speed.
            # Other enemy routines can interpret the speed bytes differently.
            if (
                collision_distance is not None
                and kind_id in (0x00, 0x06)
                and self._ram_byte(ram, 0x1E + slot) == 0
                and not active & 0x80
            ):
                player_speed = (self._ram_byte(ram, 0x57) + 128) % 256 - 128
                enemy_speed = (self._ram_byte(ram, 0x58 + slot) + 128) % 256 - 128
                ram_relative_speed = (enemy_speed - player_speed) / 16
            if -192 <= dx <= 320:
                player_box = self._collision_box(ram, 0x04AC)
                enemy_box = self._collision_box(ram, 0x04B0 + 4 * slot)
                enemies.append(
                    EnemyObservation(
                        slot=slot,
                        kind_id=kind_id,
                        kind=ENEMY_NAMES.get(kind_id, f"enemy_0x{kind_id:02x}"),
                        dx_pixels=dx,
                        dy_pixels=dy,
                        relative_velocity_x=relative_velocity_x,
                        collision_distance_pixels=collision_distance,
                        ram_relative_speed_x=ram_relative_speed,
                        active_flag=active if ram is not None else None,
                        motion_state=self._ram_byte(ram, 0x1E + slot) if ram is not None else None,
                        collision_box=enemy_box,
                        vertical_overlap=(
                            player_box[1] <= enemy_box[3] and enemy_box[1] <= player_box[3]
                            if player_box is not None and enemy_box is not None
                            else None
                        ),
                    )
                )
        return enemies

    def _extract_local_grid(
        self,
        ram: Sequence[int] | None,
        mario_x: int,
        mario_screen_y: int,
        enemies: Sequence[EnemyObservation],
    ) -> list[str]:
        if ram is None:
            return []

        width, height = 11, 9
        x_offsets = range(-2, 9)
        y_offsets = range(-4, 5)
        cells = [["." for _ in range(width)] for _ in range(height)]

        for row, dy_tiles in enumerate(y_offsets):
            for col, dx_tiles in enumerate(x_offsets):
                box_x = dx_tiles * 16
                box_y = dy_tiles * 16
                sample_x = mario_x + box_x
                sample_y = mario_screen_y + box_y
                page = (sample_x // 256) % 2
                sub_x = (sample_x % 256) // 16
                sub_y = (sample_y - 32) // 16
                if 0 <= sub_y < 13:
                    address = 0x0500 + page * 13 * 16 + sub_y * 16 + sub_x
                    if self._ram_byte(ram, address) != 0:
                        cells[row][col] = "#"

        mario_col = 2
        mario_row = 4
        cells[mario_row][mario_col] = "M"
        for enemy in enemies:
            col = mario_col + round(enemy.dx_pixels / 16)
            row = mario_row + round(enemy.dy_pixels / 16)
            if 0 <= row < height and 0 <= col < width and cells[row][col] != "M":
                cells[row][col] = "E"
        return ["".join(row) for row in cells]
