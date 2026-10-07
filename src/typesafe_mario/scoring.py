"""Observed scoring facts; no action selection or simulated stomp predictions.

RAM rules and recorded contact evidence: tests/fixtures/scoring-observations.md.
Confirmed counts are lower bounds. Score attribution remains unknown unless its
source can be proved; temporal coincidence alone is not sufficient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import EnemyObservation, MarioSnapshot


@dataclass(frozen=True)
class BlockBumpGeometry:
    """Current SMB background head probe, not a future hit prediction."""

    head_world_x: int
    head_screen_y: int
    horizontal_interval_relative_to_head_pixels: tuple[int, int]
    rise_to_block_bottom_pixels: int
    block_bounce_active: bool


@dataclass(frozen=True)
class Opportunity:
    target_id: str
    kind: str
    observed_frame: int
    relative_x_pixels: int
    relative_y_pixels: int
    world_x: int
    collision_box_screen_xyxy: tuple[int, int, int, int] | None = None
    source: str = "ram"
    validity: str = "confirmed"
    enemy_kind: str | None = None
    motion_state: int | None = None
    stompable: bool | None = None
    relative_velocity_x: float | None = None
    block_kind: str | None = None
    required_interaction: str | None = None
    tile_bounds_screen_xyxy: tuple[int, int, int, int] | None = None
    head_bump_geometry: BlockBumpGeometry | None = None
    expected_powerup_if_hit_now: str | None = None
    powerup_kind: str | None = None
    emergence_phase: str | None = None


@dataclass(frozen=True)
class OpportunitySet:
    availability: str
    observation_frame: int
    world_x_bounds: tuple[int, int] | None
    items: tuple[Opportunity, ...] = ()


@dataclass(frozen=True)
class ScoringEvent:
    kind: str
    frame: int
    target_id: str | None
    amount: int | None
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ScoringObservation:
    score: int | None
    coin_counter: int | None
    score_delta_since_previous_observation: int | None
    confirmed_coins_collected: int
    confirmed_stomps: int
    unattributed_score_gain: int
    observation_frame: int
    opportunities: OpportunitySet
    events: tuple[ScoringEvent, ...]
    no_best_progress_frames: int
    time_known: bool
    coverage_gaps: tuple[str, ...]


@dataclass(frozen=True)
class _Frame:
    snapshot: MarioSnapshot
    ram: bytes | None
    score: int | None
    coins: int | None
    lives: int | None


def _integer(info: Mapping[str, Any], name: str) -> int | None:
    value = info.get(name)
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None


class ScoringTracker:
    def __init__(self) -> None:
        self._previous: _Frame | None = None
        self._coins = self._stomps = self._unattributed = 0
        self._best_x: int | None = None
        self._best_frame = 0
        self._generation = 0
        self._identities: dict[int, tuple[int, int, int, str]] = {}
        self._powerup_identity: tuple[int, int, int, int, str] | None = None

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _scope(snapshot: MarioSnapshot) -> tuple[int, int, int]:
        return snapshot.world, snapshot.stage, snapshot.area

    def observe(
        self, info: Mapping[str, Any], ram: Sequence[int] | None, snapshot: MarioSnapshot
    ) -> ScoringObservation:
        frame = snapshot.observation_frame
        data = bytes(ram[:0x800]) if ram is not None and len(ram) >= 0x800 else None
        score, coins = _integer(info, "score"), _integer(info, "coins")
        score = score if score is not None and score >= 0 else None
        coins = coins if coins is not None and 0 <= coins < 100 else None
        lives = _integer(info, "life")
        if lives is None:
            lives = _integer(info, "lives")
        previous = self._previous
        same_area = previous is not None and self._scope(previous.snapshot) == self._scope(snapshot)
        continuous = bool(
            same_area
            and frame == previous.snapshot.observation_frame + 1
            and not previous.snapshot.dead
            and not previous.snapshot.clear
        )
        gaps: list[str] = []
        if not continuous:
            self._identities.clear()
            self._powerup_identity = None
            if previous is not None and same_area:
                gaps.append("frame_gap")
        if not same_area or self._best_x is None or snapshot.x > self._best_x:
            self._best_x, self._best_frame = snapshot.x, frame
        stagnant = max(0, frame - self._best_frame)
        if data is None:
            gaps.append("ram_unavailable")
        if score is None:
            gaps.append("score_unavailable")
        if coins is None:
            gaps.append("coin_counter_unavailable")
        events: list[ScoringEvent] = []
        delta = None
        if continuous:
            assert previous is not None
            if score is not None and previous.score is not None:
                delta = score - previous.score
                if delta < 0:
                    gaps.append("score_discontinuity")
                    delta = None
                elif delta > 0:
                    self._unattributed += delta
                    events.append(ScoringEvent("score_change", frame, None, delta, ("info.score",)))
            else:
                gaps.append("score_pair_unavailable")
            if coins is not None and previous.coins is not None:
                gained = coins - previous.coins
                if gained < 0:
                    # Only the individually evidenced 99 -> 0 rollover is accepted.
                    if (
                        previous.coins == 99
                        and coins == 0
                        and lives is not None
                        and previous.lives is not None
                        and lives == previous.lives + 1
                        and not snapshot.dead
                    ):
                        gained = 1
                    else:
                        gaps.append("coin_discontinuity")
                        gained = 0
                if gained:
                    self._coins += gained
                    events.append(
                        ScoringEvent(
                            "coin_collected",
                            frame,
                            None,
                            gained,
                            ("consecutive info.coins", "same world/stage/area"),
                        )
                    )
            else:
                gaps.append("coin_pair_unavailable")
        old_ids = self._identities.copy()
        identities = self._track_enemies(snapshot, data)
        if continuous and previous is not None and previous.ram is not None and data is not None:
            for enemy in snapshot.enemies:
                identity = identities.get(enemy.slot)
                old_identity = old_ids.get(enemy.slot)
                if (
                    identity is not None
                    and old_identity is not None
                    and identity[3] == old_identity[3]
                    and self._is_stomp(previous, snapshot, data, enemy)
                ):
                    self._stomps += 1
                    events.append(
                        ScoringEvent(
                            "enemy_stomped",
                            frame,
                            identity[3],
                            1,
                            ("top crossing", "collision bit", "state 0/1 -> 4", "bounce -4"),
                        )
                    )
            if (
                previous.ram[0x756] in (1, 2)
                and data[0x756] == 0
                and previous.ram[0xE] == 8
                and data[0xE] == 10
                and data[0x79E] > 0
            ):
                events.append(ScoringEvent("player_hurt", frame, None, 1, ("ForceInjury",)))
        opportunities = self._opportunities(snapshot, data, old_ids)
        if opportunities.availability != "available":
            gaps.append("opportunity_coverage_incomplete")
        result = ScoringObservation(
            score,
            coins,
            delta,
            self._coins,
            self._stomps,
            self._unattributed,
            frame,
            opportunities,
            tuple(events),
            stagnant,
            _integer(info, "time") is not None,
            tuple(dict.fromkeys(gaps)),
        )
        self._previous = _Frame(snapshot, data, score, coins, lives)
        if snapshot.dead or snapshot.clear:
            self._identities.clear()
        return result

    def _track_enemies(
        self, snapshot: MarioSnapshot, ram: bytes | None
    ) -> dict[int, tuple[int, int, int, str]]:
        current: dict[int, tuple[int, int, int, str]] = {}
        if ram is not None:
            scope = ":".join(map(str, self._scope(snapshot)))
            for enemy in snapshot.enemies:
                slot = enemy.slot
                if not 0 <= slot < 5 or ram[0xF + slot] == 0 or ram[0xF + slot] & 0x80:
                    continue
                x = ram[0x6E + slot] * 256 + ram[0x87 + slot]
                y = ram[0xB6 + slot] * 256 + ram[0xCF + slot]
                old = self._identities.get(slot)
                # Identity continuity bound, not a forecast or enemy speed rule.
                if (
                    old is not None
                    and old[0] == enemy.kind_id
                    and abs(x - old[1]) <= 16
                    and abs(y - old[2]) <= 16
                ):
                    target = old[3]
                else:
                    self._generation += 1
                    target = f"enemy:{scope}:{slot}:{self._generation}"
                current[slot] = (enemy.kind_id, x, y, target)
        self._identities = current
        return current

    @staticmethod
    def _is_stomp(
        previous: _Frame, snapshot: MarioSnapshot, ram: bytes, enemy: EnemyObservation
    ) -> bool:
        old = previous.ram
        assert old is not None
        slot = enemy.slot
        before = next((item for item in previous.snapshot.enemies if item.slot == slot), None)
        player_box = snapshot.player_collision_box
        old_player_box = previous.snapshot.player_collision_box
        enemy_box = enemy.collision_box
        old_enemy_box = before.collision_box if before else None
        if not all((player_box, old_player_box, enemy_box, old_enemy_box)):
            return False
        assert player_box and old_player_box and enemy_box and old_enemy_box
        return bool(
            enemy.kind_id in (0, 6)
            and old[0x1E + slot] in (0, 1)
            and ram[0x1E + slot] == 4
            and old[0xE] == ram[0xE] == 8
            and old[0x74E] != 0
            and ram[0x74E] != 0
            and old[0x79F] == ram[0x79F] == 0
            and 0 < old[0x9F] < 128
            and ram[0x9F] == 0xFC
            and not old[0x491 + slot] & 1
            and ram[0x491 + slot] & 1
            and old_player_box[3] <= old_enemy_box[1]
            and player_box[1] < enemy_box[1] <= player_box[3]
            and player_box[0] <= enemy_box[2]
            and enemy_box[0] <= player_box[2]
        )

    def _powerup_opportunity(
        self, snapshot: MarioSnapshot, ram: bytes, left: int, right: int
    ) -> Opportunity | None:
        # SMB SetupPowerUp uses slot 5, separate from the five hostile slots.
        kind, state = ram[0x39], ram[0x23]
        x = ram[0x73] * 256 + ram[0x8C]
        y = ram[0xBB] * 256 + ram[0xD4]
        box = tuple(ram[0x4C4:0x4C8])
        if not (
            ram[0x14] == 1
            and ram[0x1B] == 0x2E
            and kind in (0, 1)
            and (1 <= state <= 17 or state == 0x80 or (kind == 0 and state == 0xC0))
            and ram[0xBB] == 1
            and left <= x < x + 16 <= right
            and x + 16 > snapshot.x
            and (
                state < 6
                or (ram[0x3DD] == 0 and 0 <= box[0] < box[2] < 255 and 0 <= box[1] < box[3] < 255)
            )
        ):
            self._powerup_identity = None
            return None
        old = self._powerup_identity
        if (
            old
            and old[0] == kind
            and abs(x - old[1]) <= 16
            and abs(y - old[2]) <= 16
            and not (state < 0x80 and state < old[3])
        ):
            target_id = old[4]
        else:
            self._generation += 1
            scope = ":".join(map(str, self._scope(snapshot)))
            target_id = f"powerup:{scope}:{self._generation}"
        self._powerup_identity = (kind, x, y, state, target_id)
        # Emerging objects and flowers are stationary; active mushrooms use
        # MoveNormalEnemy's signed Q4 speed. No trajectory/contact prediction.
        speed = ((ram[0x5D] + 128) % 256 - 128) / 16 if kind == 0 and state & 0x80 else 0
        return Opportunity(
            target_id,
            "powerup",
            snapshot.observation_frame,
            x - snapshot.x,
            y - (ram[0xB5] * 256 + ram[0xCE]),
            x,
            collision_box_screen_xyxy=box if state >= 6 else None,
            source="ram_powerup_slot",
            motion_state=state,
            relative_velocity_x=speed - snapshot.dx,
            required_interaction="touch",
            powerup_kind="mushroom" if kind == 0 else "fire_flower",
            emergence_phase="active" if state & 0x80 else "emerging",
        )

    def _opportunities(
        self,
        snapshot: MarioSnapshot,
        ram: bytes | None,
        previous_ids: Mapping[int, tuple[int, int, int, str]],
    ) -> OpportunitySet:
        frame = snapshot.observation_frame
        if ram is None or ram[0xE] != 8 or snapshot.dead or snapshot.clear:
            self._powerup_identity = None
            return OpportunitySet("unavailable", frame, None)
        left = ram[0x71A] * 256 + ram[0x71C]
        right = ram[0x71B] * 256 + ram[0x71D] + 1
        viewport_valid = (
            right - left == 256
            and left <= snapshot.x < right
            and snapshot.x == ram[0x6D] * 256 + ram[0x86]
        )
        items = []
        if viewport_valid and ram[0xB5] == 1:
            scope = ":".join(map(str, self._scope(snapshot)))
            # Include the column Mario currently overlaps: rounding up would
            # lose a block as he moves under it. Still require a fully visible tile.
            first_x = max(((left + 15) // 16) * 16, (snapshot.x // 16) * 16)
            for x in range(first_x, right - 15, 16):
                for row in range(13):
                    address = 0x500 + (x // 256 % 2) * 208 + row * 16 + x % 256 // 16
                    tile = ram[address]
                    y = 32 + row * 16
                    if tile in (0xC2, 0xC3) and x >= snapshot.x:
                        items.append(
                            Opportunity(
                                f"coin:{scope}:{x}:{y}",
                                "coin",
                                frame,
                                x - snapshot.x,
                                y - ram[0xCE],
                                x,
                                source="ram_coin_metatile",
                            )
                        )
                    elif tile in (0xC0, 0x58, 0x5D, 0xC1):
                        # BrickQBlockMetatiles -> BumpBlock dispatches these to
                        # CoinBlock; $c1 dispatches to MushFlowerBlock.
                        # Hidden $5f, empty $c4 and the
                        # temporary bouncing-block placeholder $23 are excluded.
                        # PlayerBGCollision / BlockBufferAdderData: head probe
                        # is x+8, y+18 for small/crouching, y+4 for standing big.
                        # Object collision-box top is a different coordinate.
                        geometry = None
                        if ram[0x754] in (0, 1) and not ram[0x704] and ram[0x74E] != 0:
                            head_x = snapshot.x + 8
                            head_y = ram[0xCE] + (18 if ram[0x754] or ram[0x714] else 4)
                            geometry = BlockBumpGeometry(
                                head_x,
                                head_y,
                                (x - head_x, x + 16 - head_x),
                                head_y - (y + 16),
                                bool(ram[0x784]),
                            )
                        target_kind = "powerup_block" if tile == 0xC1 else "coin_block"
                        items.append(
                            Opportunity(
                                f"{target_kind}:{scope}:{x}:{y}",
                                target_kind,
                                frame,
                                x - snapshot.x,
                                y - ram[0xCE],
                                x,
                                source=f"ram_{target_kind}_metatile",
                                block_kind="question" if tile in (0xC0, 0xC1) else "brick",
                                required_interaction="hit_from_below",
                                tile_bounds_screen_xyxy=(x - left, y, x - left + 16, y + 16),
                                head_bump_geometry=geometry,
                                expected_powerup_if_hit_now=(
                                    {0: "mushroom", 1: "fire_flower", 2: "fire_flower"}.get(
                                        ram[0x756]
                                    )
                                    if tile == 0xC1
                                    else None
                                ),
                            )
                        )
            powerup = self._powerup_opportunity(snapshot, ram, left, right)
            if powerup is not None:
                items.append(powerup)
        else:
            self._powerup_identity = None
        for enemy in snapshot.enemies:
            identity = self._identities.get(enemy.slot)
            if (
                identity is None
                or enemy.kind_id not in (0, 6)
                or enemy.motion_state not in (0, 1)
                or ram[0x74E] == 0
                or enemy.dx_pixels < 0
                or enemy.collision_box is None
                or ram[0x3D8 + enemy.slot] != 0
            ):
                continue
            items.append(
                Opportunity(
                    identity[3],
                    "stomp_enemy",
                    frame,
                    enemy.dx_pixels,
                    enemy.dy_pixels,
                    identity[1],
                    enemy.collision_box,
                    "ram_walker_state",
                    enemy_kind=enemy.kind,
                    motion_state=enemy.motion_state,
                    stompable=True,
                    relative_velocity_x=(
                        enemy.ram_relative_speed_x
                        if enemy.ram_relative_speed_x is not None
                        else enemy.relative_velocity_x
                        if previous_ids.get(enemy.slot, (None, None, None, None))[3] == identity[3]
                        else None
                    ),
                )
            )
        items.sort(key=lambda item: (item.relative_x_pixels, item.kind, item.target_id))
        return OpportunitySet(
            "available" if viewport_valid and ram[0xB5] == 1 else "partial",
            frame,
            (left, right) if viewport_valid else None,
            tuple(items),
        )


class ScoringSummary:
    """Diagnostic aggregation of immutable observations, using (start, end] intervals."""

    def begin(self, snapshot: MarioSnapshot) -> None:
        self._initial = self._last = snapshot
        self._score_complete = self._score(snapshot) is not None
        self._before_clear: int | None = None
        self._at_clear = self._score(snapshot) if snapshot.clear else None
        self._clear_seen = snapshot.clear
        self._hurts = 0
        self._score_pairs = self._coin_pairs = self._ram_frames = 0
        self._gaps: set[str] = set()
        self.mark_interval(snapshot.observation_frame)

    @staticmethod
    def _score(snapshot: MarioSnapshot) -> int | None:
        return snapshot.scoring.score if snapshot.scoring is not None else None

    @staticmethod
    def _count(snapshot: MarioSnapshot, name: str) -> int | None:
        return getattr(snapshot.scoring, name) if snapshot.scoring is not None else None

    def mark_interval(self, start_frame: int) -> None:
        if start_frame != self._last.observation_frame:
            raise ValueError("Scoring interval must start at the last observed frame")
        self._interval_start = self._last
        self._interval_complete = self._score(self._last) is not None
        self._interval_gaps: set[str] = set()

    def observe(self, snapshot: MarioSnapshot) -> None:
        old = self._last
        if snapshot.observation_frame == old.observation_frame:
            return
        if snapshot.observation_frame < old.observation_frame:
            raise ValueError("Scoring observations must be in frame order")
        observation = snapshot.scoring
        contiguous = snapshot.observation_frame == old.observation_frame + 1
        gaps = set(observation.coverage_gaps if observation else ("scoring_unavailable",))
        if not contiguous:
            gaps.add("frame_gap")
        pair_valid = bool(
            contiguous
            and observation
            and observation.score_delta_since_previous_observation is not None
        )
        self._score_complete &= pair_valid
        self._interval_complete &= pair_valid
        self._score_pairs += int(pair_valid)
        self._coin_pairs += int(
            bool(
                contiguous
                and observation
                and old.scoring
                and observation.coin_counter is not None
                and old.scoring.coin_counter is not None
                and ScoringTracker._scope(snapshot) == ScoringTracker._scope(old)
                and "coin_discontinuity" not in gaps
                and not old.dead
                and not old.clear
            )
        )
        self._ram_frames += int(observation is not None and "ram_unavailable" not in gaps)
        self._gaps.update(gaps)
        self._interval_gaps.update(gaps)
        if snapshot.clear and not self._clear_seen:
            self._before_clear = self._score(old) if contiguous else None
            self._at_clear = self._score(snapshot)
            self._clear_seen = True
        if observation is not None:
            self._hurts += sum(e.kind == "player_hurt" for e in observation.events)
        self._last = snapshot

    def interval_result(self) -> dict[str, Any]:
        start, end = self._interval_start, self._last
        first_score, last_score = self._score(start), self._score(end)
        result = {
            "start_frame": start.observation_frame,
            "end_frame": end.observation_frame,
            "score_gain": last_score - first_score
            if self._interval_complete and first_score is not None and last_score is not None
            else None,
            "coverage_gaps": sorted(self._interval_gaps),
        }
        for key, name in (
            ("confirmed_coins", "confirmed_coins_collected"),
            ("confirmed_stomps", "confirmed_stomps"),
        ):
            before, after = self._count(start, name), self._count(end, name)
            result[key] = after - before if before is not None and after is not None else None
        return result

    def episode_result(self) -> dict[str, Any]:
        first_score, last_score = self._score(self._initial), self._score(self._last)
        last = self._last
        return {
            "initial_score": first_score,
            "end_score": last_score,
            "net_score_gain": last_score - first_score
            if self._score_complete and first_score is not None and last_score is not None
            else None,
            "score_before_clear": self._before_clear,
            "score_at_clear": self._at_clear,
            "confirmed_coins_collected": self._count(last, "confirmed_coins_collected"),
            "confirmed_stomps": self._count(last, "confirmed_stomps"),
            "confirmed_hurts": self._hurts,
            "unattributed_score_gain": self._count(last, "unattributed_score_gain"),
            "stage_clear": last.clear,
            "dead": last.dead,
            "time_left": last.time_left if last.scoring and last.scoring.time_known else None,
            "best_progress": last.best_progress,
            "coverage": {
                "total_frames": last.observation_frame - self._initial.observation_frame,
                "valid_score_pairs": self._score_pairs,
                "valid_coin_pairs": self._coin_pairs,
                "ram_available_frames": self._ram_frames,
                "gaps": sorted(self._gaps),
                "confirmed_counts_are_lower_bounds": True,
            },
        }
