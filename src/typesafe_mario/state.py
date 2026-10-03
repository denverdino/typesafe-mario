from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import ceil
from typing import Any

from .actions import JUMP_ACTIONS
from .observation import VisibleActorTracker, observe
from .recovery import ProgressMemory
from .routing import staging_geometry
from .terrain import is_support_tile, jump_reach_reference, landing_projection, navigation
from .tracking import EnemyMeasurement, EnemyTrack, EnemyTracker

ENEMY_NAMES: dict[int, str] = {
    0x00: "green_koopa",
    0x03: "red_koopa",
    0x06: "goomba",
    0x05: "hammer_bro",
    0x07: "bloober",
    0x0A: "grey_cheep_cheep",
    0x0B: "red_cheep_cheep",
    0x0E: "green_paratroopa_jump",
    0x0F: "red_paratroopa",
    0x10: "green_paratroopa_fly",
    0x14: "flying_cheep_cheep",
    0x0D: "piranha_plant",
    0x12: "spiny",
    0x2D: "bowser",
    0x31: "flagpole",
    0x32: "springboard",
}


@dataclass(frozen=True)
class EnemyObservation:
    slot: int
    kind_id: int
    kind: str
    dx_pixels: int
    dy_pixels: int
    relative_velocity_x: int = 0
    vertical_velocity_y: int | None = None
    relative_velocity_y: int | None = None
    plant: dict[str, Any] | None = None

    def to_state(self) -> dict[str, Any]:
        horizontal = "ahead" if self.dx_pixels >= 0 else "behind"
        return {
            "slot": self.slot,
            "kind": self.kind,
            "kind_id": self.kind_id,
            "relative_x_pixels": self.dx_pixels,
            "relative_y_pixels": self.dy_pixels,
            "relative_velocity_x": self.relative_velocity_x,
            "vertical_velocity_y": self.vertical_velocity_y,
            "relative_velocity_y": self.relative_velocity_y,
            "horizontal_relation": horizontal,
            **({"plant": self.plant} if self.plant is not None else {}),
        }


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
    collision_grid: tuple[str, ...] = field(default_factory=tuple)
    viewport_grid: tuple[str, ...] = field(default_factory=tuple)
    screen_y: int = 0
    frame_index: int = 0
    preview_frame: int | None = None
    preview_gap_start_x: int | None = None
    preview_obstacle_start_x: int | None = None
    previous_action: str | None = None
    previous_reward: float = 0.0
    dead: bool = False
    clear: bool = False
    grounded: bool = True
    swimming: bool = False
    landed_this_frame: bool = False
    precision_landing_target: dict[str, Any] | None = None
    jump_phase: str = "grounded"
    action_frames: int = 0
    action_progress: int = 0
    airborne_frames: int = 0
    jump_distance_pixels: int = 0
    crossing_gap: bool = False
    gap_width_at_commit: int = 0
    decision_horizon_frames: int = 8
    frames_until_action: int = 0
    scheduled_action: str | None = None
    scheduled_first_frame_action: str | None = None
    last_response_delay_frames: int = 0
    last_grounded_gap_distance_tiles: int | None = None
    last_grounded_gap_width_tiles: int = 0
    last_grounded_obstacle_distance_tiles: int | None = None
    last_grounded_obstacle_height_tiles: int = 0
    horizontal_speed_exact: float | None = None
    side_exit_pipe: dict[str, Any] | None = None
    enemy_tracks: tuple[EnemyTrack, ...] = field(default_factory=tuple)
    visible_actor_ids: dict[str, str] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    prediction_traces: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    recovery: dict[str, Any] = field(default_factory=dict)
    camera_left_x: int | None = None

    def recovery_features(self) -> dict[str, Any]:
        waiting = bool(
            self.grounded
            and abs(self.dx) <= 1
            and self.previous_action in {"noop", "left"}
            and self.recovery.get("no_progress_frames", 0) <= 192
            and any(
                e.kind == "piranha_plant"
                and 0 <= e.dx_pixels <= 96
                and (e.plant or {}).get("phase") in {"extended", "rising", "retracting"}
                and self.y < (e.plant or {}).get("pipe_top_y", 0) + 16
                for e in self.enemies
            )
        )
        return {
            **self.recovery,
            "intentional_wait": waiting,
            "active": bool(self.recovery.get("active") and not waiting),
        }

    def interactable_features(self) -> list[dict[str, Any]]:
        objects = [
            {
                "id": t.id,
                "kind": "springboard",
                "target_center_x": t.measurement.x + 8,
                "screen_y": t.measurement.y,
                "interaction": "Land on top to bounce; time the jump input during compression "
                "for extra height. A short retreat may be needed to align above it.",
            }
            for t in self.enemy_tracks
            if t.observed and t.measurement.kind == "springboard"
        ]
        if self.side_exit_pipe:
            objects.append(
                {
                    "kind": "side_exit_pipe",
                    **self.side_exit_pipe,
                    "interaction": "Descend to the entry floor, then walk right inside.",
                }
            )
        return objects

    def navigation_features(self) -> dict[str, Any]:
        return navigation(
            self.collision_grid or self.local_grid,
            x=self.x,
            y=self.y,
            full_height=bool(self.collision_grid),
            screen_y=self.screen_y,
        )

    def landing_features(self) -> dict[str, Any]:
        if self.swimming:
            return {
                "confidence": "not_applicable",
                "assumption": "Swimming uses repeated A strokes, not a ballistic landing arc.",
                "frames": None,
                "x": None,
                "standing_y": None,
                "within_safe_surface": None,
            }
        return landing_projection(
            self.navigation_features()["landing_surfaces"],
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            grounded=self.grounded,
        )

    def landing_threat_features(self) -> dict[str, Any]:
        projection = self.landing_features()
        frames = projection["frames"]
        threats = []
        braking_supported = False
        if frames is not None:
            # Assume enemy vertical position stays fixed over this short descending
            # horizon. This is separate from current-frame vertical collision tests.
            for enemy in self.enemies:
                if enemy.kind in {"springboard", "flagpole"}:
                    continue
                relative_x = enemy.dx_pixels + enemy.relative_velocity_x * frames
                relative_y = enemy.dy_pixels + projection["standing_y"] - self.y
                if abs(relative_x) <= 16 and abs(relative_y) <= 16:
                    threats.append(
                        {
                            "slot": enemy.slot,
                            "kind": enemy.kind,
                            "projected_distance_pixels": relative_x,
                            "current_distance_pixels": enemy.dx_pixels,
                        }
                    )
            braking_supported = any(
                surface["safe_center_min_x"] <= self.x + 8 <= surface["safe_center_max_x"]
                and surface["standing_y"] == projection["standing_y"]
                for surface in self.navigation_features()["landing_surfaces"]
            )
        return {
            "landing_collision_predicted": bool(threats),
            "landing_threat_from_behind": any(t["current_distance_pixels"] <= 0 for t in threats),
            "rear_contact_imminent": any(
                -16 <= e.dx_pixels < 0
                and abs(e.dy_pixels) <= 16
                and e.kind not in {"springboard", "flagpole"}
                for e in self.enemies
            ),
            "landing_threats": threats,
            "landing_braking_supported": braking_supported,
            "landing_prediction_assumption": "Constant enemy velocity and vertical position; approximate.",
        }

    def overhead_threat_features(self) -> list[dict[str, Any]]:
        """Possible drops into our path, including enemies just behind Mario.

        Screen y and relative y increase downward. Edge detection anticipates a
        walking enemy's fall; extrapolation is a warning, not a collision solver.
        """
        horizon = self.frames_until_action + self.decision_horizon_frames
        surfaces = self.navigation_features()["landing_surfaces"]
        threats = []
        for enemy in self.enemies:
            if enemy.kind in {"springboard", "flagpole"}:
                continue
            if not -128 <= enemy.dy_pixels < -16:
                continue
            # A falling enemy already below us at application can be stomped;
            # do not cut that jump with a stale overhead-escape instruction.
            if (
                enemy.relative_velocity_y is not None
                and enemy.vertical_velocity_y is not None
                and enemy.vertical_velocity_y > 0
                and enemy.dy_pixels + enemy.relative_velocity_y * self.frames_until_action >= 16
            ):
                continue
            start_x = enemy.dx_pixels + enemy.relative_velocity_x * self.frames_until_action
            end_x = enemy.dx_pixels + enemy.relative_velocity_x * horizon
            if min(start_x, end_x) > 24 or max(start_x, end_x) < -24:
                continue
            center = self.x + enemy.dx_pixels + 8
            feet_y = self.y - enemy.dy_pixels + 8
            # Ordinary red koopas turn at ledges; only observed falling warns for them.
            near_edge = enemy.kind in {"goomba", "green_koopa"} and any(
                abs(surface["standing_y"] - feet_y) <= 8
                and surface["start_x"] - 8 <= center <= surface["end_x"] + 8
                and (
                    center <= surface["safe_center_min_x"] + 4
                    or center >= surface["safe_center_max_x"] - 4
                )
                for surface in surfaces
            )
            falling = enemy.vertical_velocity_y is not None and enemy.vertical_velocity_y > 0
            if falling or near_edge:
                threats.append(
                    {
                        **enemy.to_state(),
                        "falling": falling,
                        "near_platform_edge": near_edge,
                        "distance_at_application_pixels": start_x,
                        "distance_at_cycle_end_pixels": end_x,
                    }
                )
        return threats

    def threat_features(self) -> dict[str, Any]:
        landing_threats = self.landing_threat_features()
        landing_threats["overhead_threats"] = self.overhead_threat_features()
        plants = [
            {
                "slot": e.slot,
                "distance_pixels": e.dx_pixels,
                **(e.plant or {"phase": "unknown"}),
                "wait_before_pipe_jump": bool(
                    (e.plant or {}).get("phase") != "hidden"
                    and e.dx_pixels >= 0
                    and (
                        e.plant is None
                        or e.plant.get("pipe_top_y") is None
                        or self.y < e.plant["pipe_top_y"] + 16
                    )
                ),
            }
            for e in self.enemies
            if e.kind == "piranha_plant" and -24 <= e.dx_pixels <= 96
        ]
        if plants:
            landing_threats["pipe_plants"] = plants
        enemies_ahead = sorted(
            (
                enemy
                for enemy in self.enemies
                if enemy.dx_pixels >= 0 and enemy.kind not in {"springboard", "flagpole"}
            ),
            key=lambda enemy: enemy.dx_pixels,
        )
        if not enemies_ahead:
            return {
                **landing_threats,
                "enemy_ahead": False,
                "upcoming_enemies": [],
                "spacing_to_second_enemy_pixels": None,
                "nearest_enemy_kind": None,
                "nearest_enemy_distance_pixels": None,
                "relative_velocity_x": None,
                "closing_speed_pixels_per_frame": 0,
                "estimated_contact_frames": None,
                "jump_clearance_frames_required": 8,
                "takeoff_deadline_frames": None,
                "jump_must_start_this_decision": False,
                "takeoff_window_already_missed": False,
                "projected_distance_after_reaction_pixels": None,
                "contact_within_reaction_horizon": False,
                "estimated_landing_frames": None,
                "will_land_before_contact": None,
            }
        vertical_margin = 32 if self.status != "small" else 16
        immediate_candidates = [
            enemy for enemy in enemies_ahead if abs(enemy.dy_pixels) <= vertical_margin
        ]
        nearest = (immediate_candidates or enemies_ahead)[0]
        closing_speed = max(0, -nearest.relative_velocity_x)
        # Conservative horizontal box margin. Vertical overlap is only a current-frame
        # test; predictions remain estimates, particularly for moving/flying enemies.
        vertical_overlap = abs(nearest.dy_pixels) <= vertical_margin
        contact_frames = None
        if vertical_overlap:
            if nearest.dx_pixels <= 16:
                contact_frames = 0
            elif closing_speed > 0:
                contact_frames = max(0, (nearest.dx_pixels - 16) // closing_speed)
        reaction_horizon = self.decision_horizon_frames + self.frames_until_action
        projected_distance = max(
            0,
            nearest.dx_pixels + nearest.relative_velocity_x * reaction_horizon,
        )
        landing_frames = self.landing_features()["frames"]
        jump_clearance_frames = 8
        takeoff_deadline = (
            contact_frames - self.frames_until_action - jump_clearance_frames
            if contact_frames is not None
            else None
        )
        jump_must_start_now = bool(
            self.grounded
            and vertical_overlap
            and takeoff_deadline is not None
            and 0 <= takeoff_deadline <= self.decision_horizon_frames
        )
        upcoming_enemies = [
            {
                "kind": enemy.kind,
                "distance_pixels": enemy.dx_pixels,
                "vertical_offset_pixels": enemy.dy_pixels,
                "relative_velocity_x": enemy.relative_velocity_x,
                "projected_distance_after_reaction_pixels": max(
                    0,
                    enemy.dx_pixels + enemy.relative_velocity_x * reaction_horizon,
                ),
            }
            for enemy in enemies_ahead[:3]
        ]
        return {
            **landing_threats,
            "enemy_ahead": True,
            "prediction_confidence": "approximate",
            "vertical_overlap_now": vertical_overlap,
            "horizontal_collision_margin_pixels": 16,
            "upcoming_enemies": upcoming_enemies,
            "spacing_to_second_enemy_pixels": (
                enemies_ahead[1].dx_pixels - enemies_ahead[0].dx_pixels
                if len(enemies_ahead) > 1
                else None
            ),
            "nearest_enemy_kind": nearest.kind,
            "nearest_enemy_distance_pixels": nearest.dx_pixels,
            "relative_velocity_x": nearest.relative_velocity_x,
            "closing_speed_pixels_per_frame": closing_speed,
            "estimated_contact_frames": contact_frames,
            "jump_clearance_frames_required": jump_clearance_frames,
            "takeoff_deadline_frames": takeoff_deadline,
            "jump_must_start_this_decision": jump_must_start_now,
            "takeoff_window_already_missed": (
                takeoff_deadline is not None and takeoff_deadline < 0
            ),
            "projected_distance_after_reaction_pixels": projected_distance,
            "contact_within_reaction_horizon": (
                vertical_overlap
                and contact_frames is not None
                and contact_frames <= reaction_horizon
            ),
            "estimated_landing_frames": landing_frames,
            "will_land_before_contact": (
                landing_frames < contact_frames
                if landing_frames is not None and contact_frames is not None
                else None
            ),
        }

    def gap_takeoff_features(self) -> dict[str, Any] | None:
        """Last safe action cycle before walking off a visible supported ledge."""
        if not self.grounded or self.dx <= 0:
            return None
        if self.frames_until_action and self.scheduled_action not in {"right", "right_run"}:
            return None
        terrain = self.navigation_features()
        gap = terrain["gap_start_x"]
        if gap is None:
            return None
        support = next(
            (
                s
                for s in terrain["landing_surfaces"]
                if s["start_x"] <= self.x + 8 < s["end_x"]
                and s["end_x"] == gap
                and abs(s["standing_y"] - self.y) <= 1
            ),
            None,
        )
        if support is None:
            return None
        limit = support["safe_center_max_x"]
        at_start = self.x + 8 + self.dx * self.frames_until_action
        at_end = at_start + self.dx * self.decision_horizon_frames
        if at_end <= limit:
            return None
        # An immediate grounded re-press spends one frame releasing held A.
        release_frames = int(self.frames_until_action == 0 and self.previous_action in JUMP_ACTIONS)
        missed = at_start + self.dx * release_frames > limit
        return {
            "assumption": "Constant observed forward speed on this floor; conservative center "
            "margin. Scheduled forward movement only. Ignores acceleration, ceilings and enemies; "
            "this is a takeoff deadline, not proof that a jump can reach the far side.",
            "gap_start_x": gap,
            "gap_end_x": terrain["gap_end_x"],
            "safe_takeoff_center_max_x": limit,
            "predicted_application_center_x": at_start,
            "predicted_cycle_end_center_x": at_end,
            "must_jump_this_cycle": not missed,
            "safe_takeoff_window_missed": missed,
        }

    def jump_reach_features(self) -> dict[str, Any] | None:
        terrain = self.navigation_features()
        gap, end = terrain["gap_start_x"], terrain["gap_end_x"]
        speed = self.horizontal_speed_exact if self.horizontal_speed_exact is not None else self.dx
        if gap is None or not 0 < gap - self.x <= 128 or speed <= 0:
            return None
        if self.grounded:
            if self.frames_until_action and self.scheduled_action not in {"right", "right_run"}:
                return None
            takeoff_y = self.y
        else:
            landing = landing_projection(
                terrain["landing_surfaces"],
                x=self.x,
                y=self.y,
                dx=speed,
                dy=self.dy,
                grounded=False,
            )
            if landing["frames"] is None or landing["frames"] > self.frames_until_action:
                return None
            takeoff_y = landing["standing_y"]
        takeoff_x = self.x + speed * self.frames_until_action
        support = next(
            (
                s
                for s in terrain["landing_surfaces"]
                if s["start_x"] <= takeoff_x + 8 < s["end_x"] and s["standing_y"] == takeoff_y
            ),
            None,
        )
        if support is None:
            return None
        # This correction is for premature jumps from a raised step before the
        # actual gap. Flat approaches retain the independent takeoff deadline.
        if not any(
            s["standing_y"] < takeoff_y and s["start_x"] <= support["end_x"] < s["end_x"] <= gap
            for s in terrain["landing_surfaces"]
        ):
            return None
        reference = jump_reach_reference(
            terrain["landing_surfaces"],
            x=takeoff_x,
            y=takeoff_y,
            speed=speed,
            horizon=self.decision_horizon_frames,
            gap_start=gap,
            gap_end=end,
        )
        # Emit only the corrective context; ordinary reachable jumps keep their
        # established input and questions.
        return reference if reference and not reference["current_speed_jump"]["can_reach"] else None

    def stair_approach_features(self) -> dict[str, Any] | None:
        """Avoid a fresh slow jump too far before a rising staircase."""
        surfaces = self.navigation_features()["landing_surfaces"]
        speed = self.horizontal_speed_exact if self.horizontal_speed_exact is not None else self.dx
        if not 0 < speed <= 2:
            return None
        if self.grounded:
            if self.frames_until_action and self.scheduled_action not in {"right", "right_run"}:
                return None
            floor_y = self.y
        else:
            landing = landing_projection(
                surfaces, x=self.x, y=self.y, dx=speed, dy=self.dy, grounded=False
            )
            if landing["frames"] is None or landing["frames"] > self.frames_until_action:
                return None
            floor_y = landing["standing_y"]
        application_x = self.x + speed * self.frames_until_action
        floor = next(
            (
                s
                for s in surfaces
                if s["standing_y"] == floor_y
                and s["safe_center_min_x"] <= application_x + 8 <= s["safe_center_max_x"]
            ),
            None,
        )
        if floor is None or not 48 < floor["end_x"] - application_x <= 112:
            return None
        edge, height, steps = floor["end_x"], floor_y, []
        while True:
            step = next(
                (s for s in surfaces if s["start_x"] == edge and s["standing_y"] == height + 16),
                None,
            )
            if step is None:
                break
            steps.append(step)
            edge, height = step["end_x"], step["standing_y"]
        if len(steps) < 3:
            return None
        end_center = application_x + 8 + speed * self.decision_horizon_frames
        if end_center > floor["safe_center_max_x"]:
            return None
        return {
            "application_x_estimate": application_x,
            "application_standing_y": floor_y,
            "first_step_x": floor["end_x"],
            "distance_at_application_pixels": floor["end_x"] - application_x,
            "visible_steps": len(steps),
            "approach_supported": True,
            "assumption": "Constant current speed and predicted landing; ignores enemy motion. "
            "A supported approach is not proof of safety from enemies.",
        }

    def to_state(self) -> dict[str, Any]:
        terrain = self.navigation_features()
        staging = staging_geometry(self)
        if staging is not None:
            terrain["staging_route"] = staging
        stair_approach = self.stair_approach_features()
        if stair_approach is not None:
            terrain["stair_approach"] = stair_approach
        if self.player_state == 2:
            terrain["side_exit_pipe"] = {"entering": True}
        elif self.side_exit_pipe is not None:
            terrain["side_exit_pipe"] = {
                **self.side_exit_pipe,
                "entering": False,
                "distance_pixels": self.side_exit_pipe["mouth_x"] - self.x,
                "above_entry": self.y > self.side_exit_pipe["entry_standing_y"] + 4,
                "over_mouth": self.x + 16 > self.side_exit_pipe["mouth_x"],
            }
        jump_reach = self.jump_reach_features()
        if jump_reach is not None:
            terrain["jump_reach_reference"] = jump_reach
        gap_window = self.gap_takeoff_features()
        if gap_window is not None:
            terrain["gap_takeoff_window"] = gap_window
        terrain.pop("summary", None)
        terrain["observation_reliability"] = (
            "unknown"
            if not terrain["geometry_available"]
            else "high"
            if self.grounded
            else "low_airborne"
        )
        terrain["last_grounded_preview"] = {
            "gap_distance_tiles": (
                ceil((self.preview_gap_start_x - self.x) / 16)
                if self.preview_gap_start_x is not None
                else None
            ),
            "gap_start_x": self.preview_gap_start_x,
            "gap_distance_pixels": (
                self.preview_gap_start_x - self.x if self.preview_gap_start_x is not None else None
            ),
            "age_frames": (
                self.frame_index - self.preview_frame if self.preview_frame is not None else None
            ),
            "gap_width_tiles_visible": self.last_grounded_gap_width_tiles,
            "obstacle_distance_tiles": (
                ceil((self.preview_obstacle_start_x - self.x) / 16)
                if self.preview_obstacle_start_x is not None
                else None
            ),
            "obstacle_start_x": self.preview_obstacle_start_x,
            "obstacle_height_tiles": self.last_grounded_obstacle_height_tiles,
        }
        state = {
            "objective": self.goal,
            "level": {"world": self.world, "stage": self.stage, "area": self.area},
            "player": {
                "x": self.x,
                "center_x": self.x + 8,
                "y": self.y,
                "horizontal_speed_px_per_frame": self.dx,
                "vertical_speed_px_per_frame": self.dy,
                "grounded": self.grounded,
                "swimming": self.swimming,
                "landed_this_frame": self.landed_this_frame,
                "jump_button_held": self.previous_action in JUMP_ACTIONS,
                "can_start_jump": self.grounded or self.swimming,
                "jump_requires_release": (self.grounded or self.swimming)
                and self.previous_action in JUMP_ACTIONS,
                "jump_phase": self.jump_phase,
                "powerup_status": self.status,
            },
            "trajectory": {
                "airborne_frames": self.airborne_frames,
                "precision_landing_target": self.precision_landing_target,
                "precision_target_cleared": bool(
                    not self.grounded
                    and self.precision_landing_target
                    and self.y >= self.precision_landing_target["standing_y"] + 4
                    and self.x + 8 <= self.precision_landing_target["safe_center_max_x"]
                ),
                "precision_target_missed": bool(
                    not self.grounded
                    and self.precision_landing_target
                    and self.x + 8 > self.precision_landing_target["safe_center_max_x"]
                ),
                "horizontal_distance_since_takeoff_pixels": self.jump_distance_pixels,
                "crossing_known_gap": self.crossing_gap,
                "gap_width_at_commit_tiles": self.gap_width_at_commit,
                "landing_projection": self.landing_features(),
            },
            "hazard": self.threat_features(),
            "enemy_tracks": [track.to_state() for track in self.enemy_tracks],
            "enemy_coordinates": "world x rightward; screen y downward; pixels/frame",
            "interactables": self.interactable_features(),
            "recovery": self.recovery_features(),
            **({"prediction": self.prediction} if self.prediction is not None else {}),
            "terrain": terrain,
            "reaction_timing": {
                "action_horizon_frames": self.decision_horizon_frames,
                "frames_until_action": self.frames_until_action,
                "scheduled_action": self.scheduled_action,
                "scheduled_first_frame_action": self.scheduled_first_frame_action,
                "last_inference_delay_frames": self.last_response_delay_frames,
                "total_reaction_horizon_frames": (
                    self.decision_horizon_frames + self.frames_until_action
                ),
            },
            "recent_control": {
                "action": self.previous_action,
                "frames_observed": self.action_frames,
                "progress_gained_pixels": self.action_progress,
                "outcome": self.control_outcome(),
            },
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
        if self.swimming:
            # Land-only heuristics otherwise contradict the native swimming physics.
            state["trajectory"] = {
                "movement_mode": "swimming",
                "landing_projection": self.landing_features(),
                "stroke_control": "A strokes can start while swimming, including while sinking. "
                "Consecutive A actions release for one frame at each decision boundary, then "
                "press again. Releasing A for a whole cycle allows descent; it does not hover.",
            }
            state["hazard"] = {
                "enemy_ahead": any(e.dx_pixels >= 0 for e in self.enemies),
                "upcoming_enemies": [e.to_state() for e in self.enemies],
                "assumption": "Use native forecasts for water currents and enemy collisions. "
                "Do not assume swimming over an enemy permits stomping it.",
            }
            for key in (
                "stair_approach",
                "jump_reach_reference",
                "gap_takeoff_window",
                "last_grounded_preview",
            ):
                terrain.pop(key, None)
            terrain["gap_semantics"] = (
                "Gaps are seabed openings, not mandatory ballistic jumps. "
                "Keep enough height to swim across; currents can pull Mario down."
            )
        return state

    def control_outcome(self) -> str:
        if self.dead:
            return "death"
        if self.clear:
            return "stage_clear"
        if self.previous_action is None or self.action_frames < 2:
            return "not_enough_evidence"
        if not self.grounded:
            return "jump_in_progress"
        if self.stalled_steps >= 4:
            return "blocked"
        if self.action_progress > 0:
            return "advanced"
        return "no_progress_yet"

    def to_debug_state(self, *, include_model_prediction: bool = True) -> dict[str, Any]:
        from .observation import model_state

        observation = model_state(self)
        if not include_model_prediction:
            observation.pop("prediction", None)
        return {
            "model_state": observation,
            "diagnostic_contract": {
                "prediction_source": "prediction_details",
                "legacy_estimates": "Snapshot.to_state hazard/trajectory are diagnostic only",
                "actor_coordinates": "visible_enemies use screen y downward; model uses y upward",
            },
            "raw": {
                "player_state": self.player_state,
                "coins": self.coins,
                "score": self.score,
                "previous_reward": self.previous_reward,
                "horizontal_motion": self.direction,
                "vertical_motion": self.vertical_motion,
            },
            "visible_enemies": [enemy.to_state() for enemy in self.enemies],
            "prediction_traces": list(self.prediction_traces),
            "prediction_details": self.prediction,
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
        goal: str = "Complete the current stage without dying.",
        decision_horizon_frames: int = 8,
    ) -> None:
        self.goal = goal
        self.decision_horizon_frames = decision_horizon_frames
        self._precision_landing_target: dict[str, Any] | None = None
        self._frame_index = -1
        self._preview_frame: int | None = None
        self._preview_gap_start_x: int | None = None
        self._preview_obstacle_start_x: int | None = None
        self._last_x: int | None = None
        self._last_y: int | None = None
        self._best_x = 0
        self._stalled_steps = 0
        self._last_enemy_dx: dict[int, int] = {}
        self._last_enemy_vertical: dict[int, tuple[int, int, int]] = {}
        self._enemy_tracker = EnemyTracker()
        self._visible_actor_tracker = VisibleActorTracker()
        self._progress_memory = ProgressMemory()
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
        self._last_grounded_obstacle_distance: int | None = None
        self._last_grounded_obstacle_height = 0

    def reset(self) -> None:
        """Clear episode history while preserving the configured objective."""
        epoch = self._enemy_tracker.epoch + 1
        self.__init__(
            goal=self.goal,
            decision_horizon_frames=self.decision_horizon_frames,
        )
        self._enemy_tracker.epoch = epoch
        self._visible_actor_tracker.epoch = epoch

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
        include_geometry: bool = True,
    ) -> MarioSnapshot:
        # Call once at reset, then once per emulator frame in every runner mode.
        # Rollout-only parsers may omit derived terrain: native engine collisions
        # still run on every frame. Do not reuse those parsers for live control.
        self._frame_index += 1
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

        if dy > 0:
            vertical_motion = "rising"
        elif dy < 0:
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
        level = tuple(self._integer(info, key, 1) for key in ("world", "stage", "area"))
        measurements = None
        if ram is not None and len(ram) > 0xD3:
            measurements = [
                EnemyMeasurement(
                    slot,
                    int(ram[0x16 + slot]),
                    ENEMY_NAMES.get(int(ram[0x16 + slot]), f"enemy_0x{int(ram[0x16 + slot]):02x}"),
                    int(ram[0x6E + slot]) * 256 + int(ram[0x87 + slot]),
                    int(ram[0xCF + slot]),
                    int(ram[0x1E + slot]),
                    self._plant_state(
                        ram,
                        slot,
                        int(ram[0xCF + slot]),
                        int(ram[0x6E + slot]) * 256 + int(ram[0x87 + slot]) - x,
                    )
                    if int(ram[0x16 + slot]) == 0x0D
                    else None,
                    y_high=int(ram[0xB6 + slot]),
                    engine_direction_raw=int(ram[0x46 + slot]),
                    active_flag=int(ram[0x0F + slot]),
                )
                for slot in range(5)
                if ram[0xF + slot]
            ]
        enemy_tracks = self._enemy_tracker.update(self._frame_index, level, measurements)
        native_grounded = ram is not None and len(ram) > 0x001D
        grid = (
            self._extract_local_grid(ram, x, screen_y, enemies)
            if include_geometry or not native_grounded
            else []
        )
        support_below = self._has_support_below(grid)
        grounded = abs(dy) <= 1 and support_below
        if native_grounded:
            grounded = (
                self._ram_byte(ram, 0x001D) == 0
                and self._integer(info, "player_state", 8) == 8
                and not info.get("death", info.get("is_dead", False))
            )
        if grounded:
            jump_phase = "grounded"
        elif dy > 0:
            jump_phase = "rising"
        elif dy < 0:
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
                    self._last_grounded_gap_distance is not None
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
            # SMB SwimmingFlag, rather than world/stage or zero-filled AreaType.
            # This also switches off after the underwater exit pipe transition.
            swimming=bool(self._ram_byte(ram, 0x0704)),
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
            enemy_tracks=enemy_tracks,
            recovery=self._progress_memory.update(self._frame_index, level, x, y, previous_action),
            local_grid=tuple(grid),
            collision_grid=tuple(self._extract_collision_grid(ram, x)) if include_geometry else (),
            viewport_grid=(
                tuple(
                    self._extract_tile_columns(
                        ram,
                        (x - self._integer(info, "left_x_pos")) // 16,
                        (x - self._integer(info, "left_x_pos") + 255) // 16 + 1,
                    )
                )
                if include_geometry and info.get("left_x_pos") is not None
                else ()
            ),
            side_exit_pipe=self._extract_side_exit_pipe(ram, x) if include_geometry else None,
            screen_y=screen_y,
            frame_index=self._frame_index,
            camera_left_x=(
                x - self._integer(info, "left_x_pos")
                if info.get("left_x_pos") is not None
                else None
            ),
            preview_frame=self._preview_frame,
            preview_gap_start_x=self._preview_gap_start_x,
            preview_obstacle_start_x=self._preview_obstacle_start_x,
            previous_action=previous_action,
            previous_reward=float(previous_reward),
            dead=bool(info.get("death", info.get("is_dead", False))),
            clear=bool(info.get("clear", info.get("flag_get", False))),
            grounded=grounded,
            precision_landing_target=self._precision_landing_target,
            landed_this_frame=grounded and self._last_jump_phase != "grounded",
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
            last_grounded_obstacle_distance_tiles=self._last_grounded_obstacle_distance,
            last_grounded_obstacle_height_tiles=self._last_grounded_obstacle_height,
            horizontal_speed_exact=(
                (int(ram[0x57]) if int(ram[0x57]) < 128 else int(ram[0x57]) - 256) / 16
                if ram is not None and len(ram) > 0x57
                else None
            ),
        )
        if include_geometry and grounded:
            navigation = snapshot.navigation_features()
            gap_start = navigation.get("gap_start_x")
            candidates = [
                surface
                for surface in navigation["landing_surfaces"]
                if gap_start is not None
                and 0 < gap_start - x <= 128
                and x < surface["start_x"] < surface["end_x"] <= gap_start
                and y < surface["standing_y"] <= y + 64
            ]
            self._precision_landing_target = max(
                candidates, key=lambda s: (s["standing_y"], s["start_x"]), default=None
            )
            if self._precision_landing_target is not None:
                self._precision_landing_target = {
                    k: v
                    for k, v in self._precision_landing_target.items()
                    if k != "height_above_player_pixels"
                }
            snapshot = replace(snapshot, precision_landing_target=self._precision_landing_target)
            self._last_grounded_gap_distance = navigation.get("gap_distance_tiles")
            self._last_grounded_gap_width = int(navigation.get("gap_width_tiles_visible", 0))
            self._last_grounded_obstacle_distance = navigation.get("obstacle_distance_tiles")
            self._last_grounded_obstacle_height = int(navigation.get("obstacle_height_tiles", 0))
            self._preview_frame = self._frame_index
            self._preview_gap_start_x = navigation.get("gap_start_x")
            self._preview_obstacle_start_x = navigation.get("obstacle_start_x")
            snapshot = replace(
                snapshot,
                preview_frame=self._preview_frame,
                preview_gap_start_x=self._preview_gap_start_x,
                preview_obstacle_start_x=self._preview_obstacle_start_x,
                last_grounded_gap_width_tiles=self._last_grounded_gap_width,
                last_grounded_obstacle_height_tiles=self._last_grounded_obstacle_height,
            )
        self._last_x = x
        self._last_y = y
        self._last_jump_phase = jump_phase
        snapshot = replace(
            snapshot,
            visible_actor_ids=self._visible_actor_tracker.update(observe(snapshot)),
        )
        return snapshot

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
            if ram is not None and active == 0:
                self._last_enemy_dx.pop(slot, None)
                self._last_enemy_vertical.pop(slot, None)
                continue
            if ram is None and kind_id == 0:
                self._last_enemy_dx.pop(slot, None)
                self._last_enemy_vertical.pop(slot, None)
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
            previous_vertical = self._last_enemy_vertical.get(slot)
            vertical_velocity_y = relative_velocity_y = None
            if ram is not None and previous_vertical is not None:
                old_kind, old_y, old_dy = previous_vertical
                if old_kind == kind_id and abs(enemy_y - old_y) <= 32:
                    vertical_velocity_y = enemy_y - old_y
                    if abs(dy - old_dy) <= 32:
                        relative_velocity_y = dy - old_dy
            if ram is not None:
                self._last_enemy_vertical[slot] = (kind_id, enemy_y, dy)
            else:
                self._last_enemy_vertical.pop(slot, None)
            previous_dx = self._last_enemy_dx.get(slot)
            relative_velocity_x = 0 if previous_dx is None else dx - previous_dx
            self._last_enemy_dx[slot] = dx
            if -192 <= dx <= 320:
                enemies.append(
                    EnemyObservation(
                        slot=slot,
                        kind_id=kind_id,
                        kind=ENEMY_NAMES.get(kind_id, f"enemy_0x{kind_id:02x}"),
                        dx_pixels=dx,
                        dy_pixels=dy,
                        relative_velocity_x=relative_velocity_x,
                        vertical_velocity_y=vertical_velocity_y,
                        relative_velocity_y=relative_velocity_y,
                        plant=self._plant_state(ram, slot, enemy_y, dx)
                        if kind_id == 0x0D
                        else None,
                    )
                )
        return enemies

    @staticmethod
    def _plant_state(ram: Sequence[int] | None, slot: int, y: int, dx: int) -> dict[str, Any]:
        # SMB InitPiranhaPlant / MovePiranhaPlant: movement is one pixel every
        # other frame, with a 64-frame pause at each endpoint. The engine phase
        # avoids confusing an idle movement frame with an extended/hidden pause.
        if ram is None or len(ram) <= 0x78A + slot:
            return {"phase": "unknown"}
        up, down = int(ram[0x417 + slot]), int(ram[0x434 + slot])
        if down - up != 24 or not up <= y <= down:
            return {"phase": "unknown"}
        speed, moving, timer = int(ram[0x58 + slot]), int(ram[0xA0 + slot]), int(ram[0x78A + slot])
        if moving and speed == 255:
            phase, until_hidden = "rising", (y - up) * 2 + 64 + 48
        elif y == down:
            phase, until_hidden = "hidden", 0
        elif moving and speed == 1:
            phase, until_hidden = "retracting", (down - y) * 2
        elif y == up:
            phase, until_hidden = "extended", timer + 48
        else:
            phase, until_hidden = "unknown", None
        return {
            "phase": phase,
            "pipe_top_y": 255 - down + 32,
            "estimated_frames_until_hidden": until_hidden,
            "hidden_pause_frames_remaining": timer if phase == "hidden" else None,
            "emergence_suppressed_by_proximity": phase == "hidden" and not moving and abs(dx) < 33,
        }

    def _extract_side_exit_pipe(
        self, ram: Sequence[int] | None, mario_x: int
    ) -> dict[str, Any] | None:
        if ram is None or len(ram) < 0x6A0:
            return None

        def tile(tx: int, row: int) -> int:
            if tx < 0:
                return 0
            return self._ram_byte(ram, 0x500 + ((tx // 16) % 2) * 208 + row * 16 + tx % 16)

        # Preserve semantic metatiles before the collision grid reduces them to '#'.
        # SMB side collision accepts the lower mouth tile while grounded/facing right.
        for tx in range(max(0, mario_x // 16 - 4), mario_x // 16 + 7):
            for row in range(11):
                # Normal sideways pipe and the underwater exit's distinct pair.
                if (tile(tx, row), tile(tx, row + 1)) not in {(0x1C, 0x1F), (0x6B, 0x6C)}:
                    continue
                return {
                    "mouth_x": tx * 16,
                    "entry_standing_y": 255 - (row + 2) * 16,
                    "retreat_target_x": tx * 16 - 24,
                    "approach_supported": all(
                        is_support_tile(tile(col, row + 2)) for col in range(tx - 3, tx + 1)
                    ),
                }
        return None

    def _extract_collision_grid(self, ram: Sequence[int] | None, mario_x: int) -> list[str]:
        return self._extract_tile_columns(ram, mario_x // 16 - 2, mario_x // 16 + 9)

    def _extract_tile_columns(self, ram: Sequence[int] | None, start: int, end: int) -> list[str]:
        if ram is None:
            return []
        rows = []
        for row in range(13):
            cells = []
            for tile_x in range(start, end):
                address = 0x500 + ((tile_x // 16) % 2) * 208 + row * 16 + tile_x % 16
                cells.append(
                    "#" if tile_x >= 0 and is_support_tile(self._ram_byte(ram, address)) else "."
                )
            rows.append("".join(cells))
        return rows

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
                    if is_support_tile(self._ram_byte(ram, address)):
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
