from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from typesafe_mario.scoring import ScoringTracker
from typesafe_mario.state import MarioStateParser


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ScoringTracker()
        self.parser = MarioStateParser()
        self.frame = -1
        self.ram = bytearray(2048)
        self.ram[0xE] = 8
        self.ram[0xB5] = 1
        self.ram[0x86] = 100
        self.ram[0xCE] = 150
        self.ram[0x71D] = 255
        self.ram[0x74E] = 1
        self.ram[0x4AC:0x4B0] = bytes([103, 170, 113, 182])
        self.info = {
            "world": 1,
            "stage": 1,
            "area": 1,
            "x_pos": 100,
            "y_pos": 105,
            "y_pixel": 150,
            "score": 0,
            "coins": 0,
            "time": 390,
            "life": 2,
            "status": "small",
        }

    def observe(self, *, ram=True, frame=None, **changes):
        self.info.update(changes)
        self.frame = self.frame + 1 if frame is None else frame
        source = self.ram if ram is True else ram if ram is not False else None
        snapshot = replace(self.parser.parse(self.info, source), observation_frame=self.frame)
        return self.tracker.observe(self.info, source, snapshot)

    def enemy(self, *, slot=0, kind=6, state=0, x=140, y=176, box=(143, 196, 153, 204)):
        self.ram[0xF + slot] = 1
        self.ram[0x16 + slot] = kind
        self.ram[0x1E + slot] = state
        self.ram[0x6E + slot], self.ram[0x87 + slot] = divmod(x, 256)
        self.ram[0xB6 + slot], self.ram[0xCF + slot] = divmod(256 + y, 256)
        self.ram[0x4B0 + 4 * slot : 0x4B4 + 4 * slot] = bytes(box)

    def fixture(self, name):
        data = json.loads(
            (Path(__file__).parent / "fixtures/scoring-observations.json").read_text()
        )
        return next(c for c in data["cases"] if c["id"] == name)

    def replay(self, case):
        observations = []
        for f in case["frames"]:
            ram = bytes.fromhex(f["ram_hex"])
            snapshot = replace(self.parser.parse(f["info"], ram), observation_frame=f["frame"])
            observations.append(self.tracker.observe(f["info"], ram, snapshot))
        return observations

    def test_missing_and_first_observation_preserve_unknown(self):
        self.info.pop("score")
        self.info.pop("coins")
        missing = self.observe()
        self.assertIsNone(missing.score)
        self.assertIsNone(missing.coin_counter)
        self.assertIsNone(missing.score_delta_since_previous_observation)
        first = self.observe(score=500, coins=3)
        self.assertEqual(first.confirmed_coins_collected, 0)
        self.assertEqual(first.unattributed_score_gain, 0)
        self.assertIsNone(first.score_delta_since_previous_observation)

    def test_coin_wrap_requires_continuous_verified_context(self):
        self.observe(coins=99)
        wrapped = self.observe(coins=0, life=3, score=200)
        self.assertEqual(wrapped.confirmed_coins_collected, 1)
        self.tracker.reset()
        self.observe(coins=99, life=2)
        ambiguous = self.observe(coins=0)
        self.assertEqual(ambiguous.confirmed_coins_collected, 0)
        self.assertIn("coin_discontinuity", ambiguous.coverage_gaps)

    def test_area_change_preserves_totals_but_invalidates_matching(self):
        self.observe()
        before = self.observe(coins=1, score=200)
        after = self.observe(area=2, coins=0, score=0)
        self.assertEqual(after.confirmed_coins_collected, before.confirmed_coins_collected)
        self.assertIsNone(after.score_delta_since_previous_observation)
        self.assertEqual(after.events, ())

    def test_reset_starts_new_episode(self):
        self.observe()
        self.observe(coins=1, score=200)
        self.tracker.reset()
        fresh = self.observe(coins=0, score=0)
        self.assertEqual(fresh.confirmed_coins_collected, 0)
        self.assertEqual(fresh.confirmed_stomps, 0)
        self.assertIsNone(fresh.score_delta_since_previous_observation)

    def test_visible_coins_use_verified_tiles_and_viewport(self):
        self.ram[0x500 + 8 * 16 + 8] = 0xC2
        self.ram[0x500 + 8 * 16 + 9] = 1
        self.ram[0x500 + 208 + 8 * 16 + 1] = 0xC2  # offscreen buffered column
        observation = self.observe()
        items = observation.opportunities.items
        self.assertEqual(len(items), 1)
        self.assertEqual(
            (items[0].world_x, items[0].relative_x_pixels, items[0].relative_y_pixels),
            (128, 28, 10),
        )
        self.assertIsNone(items[0].collision_box_screen_xyxy)  # tile bounds are not a bbox
        self.assertEqual(observation.opportunities.world_x_bounds, (0, 256))

    def test_unknown_and_short_ram_do_not_fabricate_targets(self):
        for ram in (False, bytearray(16), bytearray(2048)):
            with self.subTest(size=len(ram) if ram is not False else None):
                result = self.observe(ram=ram)
                self.assertEqual(result.opportunities.items, ())
                self.assertNotEqual(result.opportunities.availability, "available")

    def test_coin_blocks_are_distinct_from_touch_collectible_coins(self):
        for tile, block_kind in ((0xC0, "question"), (0x58, "brick"), (0x5D, "brick")):
            with self.subTest(tile=hex(tile)):
                self.ram[0x500 + 7 * 16 + 8] = tile
                items = self.observe().opportunities.items
                self.assertEqual(len(items), 1)
                block = items[0]
                self.assertEqual(block.kind, "coin_block")
                self.assertEqual(block.block_kind, block_kind)
                self.assertEqual(block.required_interaction, "hit_from_below")
                self.assertEqual(block.tile_bounds_screen_xyxy, (128, 144, 144, 160))
                self.assertEqual(block.relative_y_pixels, -6)
                self.assertIsNone(block.collision_box_screen_xyxy)

    def test_block_head_geometry_uses_background_probe_for_player_size(self):
        self.ram[0x500 + 7 * 16 + 8] = 0xC0  # bottom y=160, left x=128
        for size, crouching, head_y, rise in ((1, 0, 168, 8), (0, 0, 154, -6), (0, 1, 168, 8)):
            with self.subTest(size=size, crouching=crouching):
                self.ram[0x754], self.ram[0x714] = size, crouching
                snapshot = self.parser.parse(self.info, self.ram)
                block = snapshot.to_state()["opportunities"]["items"][0]
                self.assertEqual(
                    block["head_bump_geometry"],
                    {
                        "head_world_x": 108,
                        "head_screen_y": head_y,
                        "horizontal_interval_relative_to_head_pixels": (20, 36),
                        "rise_to_block_bottom_pixels": rise,
                        "block_bounce_active": False,
                    },
                )

    def test_block_head_geometry_reports_overlap_and_bounce_without_predicting_hit(self):
        self.ram[0x500 + 7 * 16 + 8] = 0x58
        self.ram[0x86] = 130
        self.ram[0x784] = 12
        snapshot = self.parser.parse(dict(self.info, x_pos=130), self.ram)
        geometry = snapshot.to_state()["opportunities"]["items"][0]["head_bump_geometry"]
        self.assertEqual(geometry["horizontal_interval_relative_to_head_pixels"], (-10, 6))
        self.assertTrue(geometry["block_bounce_active"])

    def test_block_head_geometry_is_unknown_for_unsupported_swimming(self):
        self.ram[0x500 + 7 * 16 + 8] = 0xC0
        self.ram[0x704] = 1
        block = self.parser.parse(self.info, self.ram).to_state()["opportunities"]["items"][0]
        self.assertIsNone(block["head_bump_geometry"])

    def test_empty_hidden_and_other_reward_blocks_are_not_coin_targets(self):
        for tile in (
            0xC4,
            0x5F,
            0x60,
            0x55,
            0x56,
            0x57,
            0x59,
            0x5A,
            0x5B,
            0x5C,
            0x5E,
            0x51,
            0x23,
        ):
            with self.subTest(tile=hex(tile)):
                self.ram[0x500 + 7 * 16 + 8] = tile
                self.assertEqual(self.observe().opportunities.items, ())

    def test_question_powerup_block_predicts_contents_from_current_ram_status(self):
        self.ram[0x500 + 7 * 16 + 8] = 0xC1
        for status, expected in (
            (0, "mushroom"),
            (1, "fire_flower"),
            (2, "fire_flower"),
            (3, None),
        ):
            with self.subTest(status=status):
                self.ram[0x756] = status
                blocks = self.observe().opportunities.items
                self.assertEqual(len(blocks), 1)
                block = blocks[0]
                self.assertEqual(block.kind, "powerup_block")
                self.assertEqual(block.expected_powerup_if_hit_now, expected)
                self.assertEqual(block.required_interaction, "hit_from_below")
                self.assertIsNotNone(block.head_bump_geometry)

    def powerup(self, *, kind=0, state=0x80, x=140, y=144):
        self.ram[0x14] = 1
        self.ram[0x1B] = 0x2E
        self.ram[0x23] = state
        self.ram[0x39] = kind
        self.ram[0x73], self.ram[0x8C] = divmod(x, 256)
        self.ram[0xBB], self.ram[0xD4] = divmod(256 + y, 256)
        self.ram[0x5D] = 16
        self.ram[0x4C4:0x4C8] = bytes((142, 153, 154, 165))

    def test_emerged_powerups_are_touch_targets_separate_from_enemies(self):
        for kind, name, velocity in ((0, "mushroom", 1.0), (1, "fire_flower", 0.0)):
            with self.subTest(kind=kind):
                self.powerup(kind=kind)
                snap = self.parser.parse(self.info, self.ram)
                items = snap.to_state()["opportunities"]["items"]
                self.assertEqual(len(items), 1)
                item = items[0]
                self.assertEqual(item["kind"], "powerup")
                self.assertEqual(item["powerup_kind"], name)
                self.assertEqual(item["required_interaction"], "touch")
                self.assertEqual(item["emergence_phase"], "active")
                self.assertEqual(item["relative_y_pixels"], -6)
                self.assertEqual(item["relative_velocity_x"], velocity)
                self.assertEqual(snap.enemies, ())
                self.assertEqual(
                    snap.to_state()["player"]["collision_box_screen_xyxy"], (103, 170, 113, 182)
                )

    def test_powerup_emergence_requires_current_valid_contact_geometry(self):
        self.powerup(state=1)
        early = self.observe().opportunities.items[0]
        self.assertEqual(early.emergence_phase, "emerging")
        self.assertIsNone(early.collision_box_screen_xyxy)  # stale until state >= 6
        self.ram[0x23] = 6
        item = self.observe().opportunities.items[0]
        self.assertEqual(item.emergence_phase, "emerging")
        self.assertEqual(item.relative_velocity_x, 0)
        self.ram[0x3DD] = 1
        self.assertEqual(self.observe().opportunities.items, ())
        self.ram[0x3DD] = 0
        self.ram[0x4C4:0x4C8] = bytes(4)
        self.assertEqual(self.observe().opportunities.items, ())

    def test_powerup_identity_and_exclusions(self):
        self.powerup()
        first = self.observe().opportunities.items[0].target_id
        self.assertEqual(self.observe().opportunities.items[0].target_id, first)
        self.ram[0x14] = 0
        self.assertEqual(self.observe().opportunities.items, ())
        self.powerup()
        self.assertNotEqual(self.observe().opportunities.items[0].target_id, first)
        for kind in (2, 3, 4):  # Stars and 1-ups are outside the approved scope.
            self.powerup(kind=kind)
            self.assertEqual(self.observe().opportunities.items, ())
        self.powerup(x=80)  # Behind Mario: do not chase backwards.
        self.assertEqual(self.observe().opportunities.items, ())
        self.powerup(x=260)  # Outside verified viewport.
        self.assertEqual(self.observe().opportunities.items, ())

    def test_mushroom_remains_a_target_when_falling_off_its_block(self):
        self.powerup()
        first = self.observe().opportunities.items[0].target_id
        self.ram[0x23] = 0xC0
        falling = self.observe().opportunities.items
        self.assertEqual(len(falling), 1)
        self.assertEqual(falling[0].target_id, first)
        self.assertEqual(falling[0].emergence_phase, "active")
        self.assertEqual(falling[0].relative_velocity_x, 1.0)

    def test_powerup_slot_replacement_restarts_identity_without_inactive_frame(self):
        self.powerup(kind=1, state=6)
        first = self.observe().opportunities.items[0].target_id
        self.powerup(kind=1, state=1, x=156)
        self.assertNotEqual(self.observe().opportunities.items[0].target_id, first)

    def test_recorded_mushroom_is_present_before_contact_geometry_is_ready(self):
        case = json.loads((Path(__file__).parent / "fixtures/powerup-emergence.json").read_text())
        snapshot = MarioStateParser().parse(case["info"], bytes.fromhex(case["ram_hex"]))
        items = [o for o in snapshot.to_state()["opportunities"]["items"] if o["kind"] == "powerup"]
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0]["world_x"], items[0]["powerup_kind"]), (1248, "mushroom"))
        self.assertEqual(items[0]["emergence_phase"], "emerging")
        self.assertIsNone(items[0]["collision_box_screen_xyxy"])

    def test_coin_block_disappears_while_bouncing_and_after_becoming_empty(self):
        address = 0x500 + 7 * 16 + 8
        self.ram[address] = 0x58
        first = self.observe().opportunities.items
        self.assertEqual(len(first), 1)
        self.ram[address] = 0x23
        self.assertEqual(self.observe().opportunities.items, ())
        self.ram[address] = 0x58
        self.assertEqual(self.observe().opportunities.items[0].target_id, first[0].target_id)
        self.ram[address] = 0xC4
        self.assertEqual(self.observe().opportunities.items, ())

    def test_coin_block_above_player_survives_column_alignment_and_page_wrap(self):
        self.ram[0x6D], self.ram[0x86] = 1, 4  # world x=260, inside block column
        self.ram[0x71A], self.ram[0x71C] = 0, 160
        self.ram[0x71B], self.ram[0x71D] = 1, 159
        self.ram[0x500 + 208 + 7 * 16] = 0xC0  # world x=256
        self.ram[0x500 + 7 * 16] = 0xC0  # buffered world x=512, offscreen
        items = self.observe(x_pos=260).opportunities.items
        self.assertEqual(len(items), 1)
        self.assertEqual((items[0].world_x, items[0].relative_x_pixels), (256, -4))
        self.assertEqual(items[0].tile_bounds_screen_xyxy, (96, 144, 112, 160))
        self.ram[0x86] = 16
        self.assertEqual(self.observe(x_pos=272).opportunities.items, ())

    def test_recorded_question_and_multicoin_blocks_reach_model_input(self):
        cases = json.loads(
            (Path(__file__).parent / "fixtures/coin-block-observations.json").read_text()
        )["cases"]
        expected = {
            "question_and_powerup": {(256, "question"), (352, "question"), (368, "question")},
            "multicoin_brick": {(1504, "question"), (1504, "brick")},
        }
        for case in cases:
            with self.subTest(case=case["id"]):
                snapshot = MarioStateParser().parse(case["info"], bytes.fromhex(case["ram_hex"]))
                blocks = [
                    o
                    for o in snapshot.to_state()["opportunities"]["items"]
                    if o["kind"] == "coin_block"
                ]
                self.assertEqual(
                    {(b["world_x"], b["block_kind"]) for b in blocks}, expected[case["id"]]
                )

    def test_vertical_page_boundary_preserves_relative_y(self):
        self.enemy(y=0)
        self.ram[0xB5] = 0
        self.ram[0xCE] = 252
        items = self.observe(y_pixel=252).opportunities.items
        enemy = next(item for item in items if item.kind == "stomp_enemy")
        self.assertEqual(enemy.relative_y_pixels, 4)

    def test_slot_reuse_changes_target_identity(self):
        self.enemy()
        old = self.observe().opportunities.items[0].target_id
        self.ram[0xF] = 0
        self.observe()
        self.enemy()
        new = self.observe().opportunities.items[0].target_id
        self.assertNotEqual(new, old)
        self.enemy(x=200)
        jumped = self.observe().opportunities.items[0].target_id
        self.assertNotEqual(new, jumped)

    def test_new_identity_does_not_inherit_observed_velocity(self):
        self.enemy(state=1, x=140)
        first = self.observe().opportunities.items[0]
        self.assertIsNone(first.relative_velocity_x)
        self.enemy(state=1, x=139)
        continuous = self.observe().opportunities.items[0]
        self.assertEqual(continuous.relative_velocity_x, -1)
        self.enemy(state=1, x=200, box=(203, 196, 213, 204))
        reused = self.observe().opportunities.items[0]
        self.assertNotEqual(reused.target_id, continuous.target_id)
        self.assertIsNone(reused.relative_velocity_x)

    def test_new_identity_preserves_independent_ram_velocity(self):
        self.enemy(state=0)
        self.ram[0x58] = 0xF0
        first = self.observe().opportunities.items[0]
        self.assertEqual(first.relative_velocity_x, -1)

    def test_unknown_defeated_flags_and_shells_are_not_targets(self):
        for kind, state in ((0x31, 0), (0x30, 0), (0x12, 0), (6, 4), (0, 4), (0, 0x80)):
            with self.subTest(kind=kind, state=state):
                self.enemy(kind=kind, state=state)
                self.assertEqual(self.observe().opportunities.items, ())

    def test_disappearance_is_not_collection_or_stomp(self):
        self.enemy()
        self.ram[0x500 + 8 * 16 + 8] = 0xC2
        self.observe()
        self.ram[0xF] = 0
        self.ram[0x500 + 8 * 16 + 8] = 0
        after = self.observe()
        self.assertEqual(after.confirmed_coins_collected, 0)
        self.assertEqual(after.confirmed_stomps, 0)
        self.assertEqual(after.events, ())

    def test_recorded_coin_increment_confirms_collection_without_target_guess(self):
        result = self.replay(self.fixture("coin"))[-1]
        self.assertEqual(result.confirmed_coins_collected, 1)
        event = next(e for e in result.events if e.kind == "coin_collected")
        self.assertEqual(event.frame, 998)
        self.assertIsNone(event.target_id)

    def test_stomp_requires_top_contact_and_state_transition(self):
        result = self.replay(self.fixture("stomp"))[-1]
        self.assertEqual(result.confirmed_stomps, 1)
        self.assertEqual(result.score_delta_since_previous_observation, 0)
        self.assertEqual([e.frame for e in result.events if e.kind == "enemy_stomped"], [122])
        for change in ("side", "star", "no_collision_bit", "no_bounce"):
            with self.subTest(change=change):
                self.setUp()
                case = self.fixture("stomp")
                for f in case["frames"]:
                    ram = bytearray.fromhex(f["ram_hex"])
                    if change == "side":
                        ram[0x4AD], ram[0x4AF] = 195, 210
                    elif change == "star":
                        ram[0x79F] = 1
                    elif change == "no_collision_bit":
                        ram[0x491] = 0
                    elif change == "no_bounce":
                        ram[0x9F] = 5
                    f["ram_hex"] = ram.hex()
                self.assertEqual(self.replay(case)[-1].confirmed_stomps, 0)

    def test_repeated_defeated_frames_count_once(self):
        case = self.fixture("stomp")
        last = case["frames"][-1]
        case["frames"].extend(dict(last, frame=f) for f in range(123, 127))
        self.assertEqual(self.replay(case)[-1].confirmed_stomps, 1)

    def test_simultaneous_events_keep_ambiguous_score_unattributed(self):
        case = self.fixture("stomp")
        case["frames"][-1]["info"].update(score=300, coins=1)
        result = self.replay(case)[-1]
        self.assertEqual(result.confirmed_coins_collected, 1)
        self.assertEqual(result.confirmed_stomps, 1)
        self.assertEqual(result.unattributed_score_gain, 300)

    def test_upgrade_is_not_player_hurt(self):
        self.observe()
        self.ram[0x756] = 1
        upgraded = self.observe(status="tall")
        self.assertNotIn("player_hurt", [e.kind for e in upgraded.events])
        self.ram[0x756] = 0
        self.ram[0x79E] = 8
        self.ram[0xE] = 10
        injured = self.observe(status="small")
        self.assertEqual([e.amount for e in injured.events if e.kind == "player_hurt"], [1])

    def test_frame_gap_and_missing_counter_do_not_infer_events(self):
        self.observe()
        gap = self.observe(frame=3, score=400, coins=2)
        self.assertIsNone(gap.score_delta_since_previous_observation)
        self.assertEqual(gap.confirmed_coins_collected, 0)
        self.assertIn("frame_gap", gap.coverage_gaps)
        self.info.pop("coins")
        self.observe()
        restored = self.observe(coins=3)
        self.assertEqual(restored.confirmed_coins_collected, 0)

    def test_observations_do_not_retain_mutable_ram(self):
        self.ram[0x500 + 8 * 16 + 8] = 0xC2
        old = self.observe()
        self.ram[0x500 + 8 * 16 + 8] = 0
        self.assertEqual(len(old.opportunities.items), 1)
        self.assertEqual(self.observe().opportunities.items, ())


class ScoringStateTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/scoring-observations.json").read_text()
        )
        row = next(c for c in fixture["cases"] if c["id"] == "visible_coin")["frames"][0]
        self.ram = bytearray.fromhex(row["ram_hex"])
        self.info = row["info"].copy()
        self.parser = MarioStateParser()

    def test_model_scoring_preserves_unknown(self):
        state = self.parser.parse({}).to_state()
        self.assertIsNone(state["scoring"]["score"])
        self.assertIsNone(state["scoring"]["coin_counter"])
        self.assertEqual(state["opportunities"]["availability"], "unavailable")
        self.assertFalse(state["scoring"]["pursuit"]["allow_extra_effort"])

    def test_scoring_serialization_is_read_only(self):
        old = self.parser.parse(self.info, self.ram)
        saved = old.to_state()
        self.info.update(coins=0, life=3, score=200)
        new = self.parser.parse(self.info, self.ram)
        self.assertEqual(saved, old.to_state())
        self.assertEqual(new.to_state(), new.to_state())
        self.assertEqual(new.to_state()["scoring"]["confirmed_coins_collected"], 1)
        self.assertEqual(new.to_state()["scoring"]["score_delta_since_previous_observation"], 200)

    def test_model_limits_opportunities_without_changing_hazard(self):
        for row in range(9):
            self.ram[0x500 + row * 16 + 9] = 0xC2
        snapshot = self.parser.parse(self.info, self.ram)
        hazard = snapshot.threat_features()
        state = snapshot.to_state()
        self.assertEqual(len(state["opportunities"]["items"]), 8)
        self.assertTrue(state["opportunities"]["truncated"])
        self.assertEqual(state["hazard"], hazard)
        self.assertEqual(
            len(snapshot.to_debug_state()["scoring_observations"]["opportunities"]["items"]), 10
        )
        self.assertEqual(state["opportunities"]["relative_y_positive"], "down")

    def test_no_best_progress_is_not_previous_frame_stall(self):
        for x in (100, 110, 100, 109):
            self.info["x_pos"] = x
            snapshot = self.parser.parse(self.info, self.ram)
        self.assertEqual(snapshot.stalled_steps, 0)
        self.assertEqual(snapshot.to_state()["scoring"]["pursuit"]["no_best_progress_frames"], 2)

    def test_pursuit_threshold_boundaries(self):
        for time_left, stagnant, allowed in (
            (101, 47, True),
            (100, 47, False),
            (101, 48, False),
            (None, 0, False),
        ):
            with self.subTest(time_left=time_left, stagnant=stagnant):
                parser = MarioStateParser()
                info = dict(self.info, time=time_left)
                for _ in range(stagnant + 1):
                    snapshot = parser.parse(info, self.ram)
                self.assertEqual(
                    snapshot.to_state()["scoring"]["pursuit"]["allow_extra_effort"], allowed
                )

    def test_reset_clears_scoring_history_and_preserves_goal(self):
        self.parser.parse(self.info, self.ram)
        self.parser.parse(dict(self.info, coins=0, life=3, score=200), self.ram)
        self.parser.reset()
        fresh = self.parser.parse(self.info, self.ram)
        self.assertEqual(fresh.to_state()["scoring"]["confirmed_coins_collected"], 0)
        self.assertIsNone(fresh.to_state()["scoring"]["score_delta_since_previous_observation"])
