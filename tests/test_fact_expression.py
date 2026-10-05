import json
import unittest
from dataclasses import replace
from pathlib import Path

from typesafe_mario.state import EnemyObservation, MarioStateParser


class FactExpressionTests(unittest.TestCase):
    def test_recorded_star_flag_stays_visible_without_becoming_a_threat(self):
        row = json.loads((Path(__file__).parent / "fixtures/star-flag.json").read_text())
        snapshot = MarioStateParser().parse(row["info"], bytes.fromhex(row["ram_hex"]))
        objects = snapshot.to_debug_state()["visible_enemies"]
        flag = next(obj for obj in objects if obj["kind_id"] == 0x31)
        self.assertEqual(flag["kind"], "star_flag")
        self.assertEqual(flag["relative_x_pixels"], 186)
        hazard = snapshot.to_state()["hazard"]
        self.assertFalse(hazard["enemy_ahead"])
        self.assertEqual(hazard["upcoming_enemies"], [])
        self.assertIsNone(hazard["nearest_enemy_kind"])
        self.assertIsNone(hazard["estimated_contact_frames"])
        self.assertFalse(hazard["jump_must_start_this_decision"])

    def test_star_flag_does_not_displace_live_enemies_from_threat_summary(self):
        snapshot = MarioStateParser().parse({})
        enemies = (
            EnemyObservation(0, 0x31, "star_flag", 5, 0),
            EnemyObservation(1, 0x06, "goomba", 20, 0, relative_velocity_x=-2),
            EnemyObservation(2, 0x00, "green_koopa", 50, 0),
            EnemyObservation(3, 0x05, "hammer_bro", 70, 0),
        )
        hazard = replace(snapshot, enemies=enemies).threat_features()
        self.assertEqual(hazard["nearest_enemy_kind"], "goomba")
        self.assertEqual(hazard["estimated_contact_frames"], 10)
        self.assertEqual(hazard["spacing_to_second_enemy_pixels"], 30)
        self.assertEqual(
            [enemy["kind"] for enemy in hazard["upcoming_enemies"]],
            ["goomba", "green_koopa", "hammer_bro"],
        )

    def fixture(self):
        return json.loads((Path(__file__).parent / "fixtures/fact-expression.json").read_text())

    def test_recorded_missing_geometry_preserves_last_valid_grounded_preview(self):
        parser = MarioStateParser()
        ram = bytearray()
        for row in self.fixture()["history"]:
            if "ram_hex" in row:
                ram = bytearray.fromhex(row["ram_hex"])
            else:
                for address, value in row["ram_changes"].items():
                    ram[int(address)] = value
            snapshot = parser.parse(row["info"], ram, previous_action=row["action"])
            if row["frame"] in (1304, 1305):
                with self.subTest(frame=row["frame"]):
                    terrain = snapshot.to_state()["terrain"]
                    self.assertFalse(terrain["geometry_available"])
                    preview = terrain["last_grounded_preview"]
                    self.assertEqual(preview["sample_x"], 2351)
                    self.assertEqual(preview["age_frames"], row["frame"] - 1265)
                    self.assertEqual(preview["gap_start_world_x"], 2448)
                    self.assertEqual(preview["gap_kind"], "pit")

    def test_missing_geometry_is_unavailable_on_ground_and_in_air(self):
        ram = bytearray(0x800)
        ram[0x0E] = 8
        for movement in (0, 2):
            with self.subTest(movement=movement):
                ram[0x1D] = movement
                snapshot = MarioStateParser().parse({"x_pos": 40, "y_pixel": 176}, ram)
                self.assertEqual(snapshot.grounded, movement == 0)
                terrain = snapshot.to_state()["terrain"]
                self.assertFalse(terrain["geometry_available"])
                self.assertEqual(terrain["observation_reliability"], "unavailable")
                self.assertIsNone(terrain["last_grounded_preview"]["sample_frame"])

    def test_valid_clear_geometry_replaces_old_gap_and_reset_clears_history(self):
        ram = bytearray(0x800)
        ram[0x0E] = 8
        for col in range(16):
            ram[0x500 + 11 * 16 + col] = 1 if col not in (5, 6) else 0
        info = {"x_pos": 40, "y_pixel": 176}
        parser = MarioStateParser()
        first = parser.parse(info, ram).to_state()["terrain"]
        self.assertEqual(first["last_grounded_preview"]["gap_start_world_x"], 80)
        self.assertEqual(first["observation_reliability"], "high")
        missing = parser.parse({**info, "y_pixel": 80}, ram).to_state()["terrain"]
        self.assertEqual(missing["last_grounded_preview"]["gap_start_world_x"], 80)
        ram[0x500 + 11 * 16 + 5] = ram[0x500 + 11 * 16 + 6] = 1
        clear = parser.parse(info, ram).to_state()["terrain"]
        self.assertTrue(clear["geometry_available"])
        self.assertEqual(clear["last_grounded_preview"]["sample_frame"], 2)
        self.assertIsNone(clear["last_grounded_preview"]["gap_start_world_x"])
        parser.reset()
        fresh = parser.parse({**info, "y_pixel": 80}, ram).to_state()["terrain"]
        self.assertIsNone(fresh["last_grounded_preview"]["sample_frame"])

    def test_recorded_defeated_enemy_is_observable_but_not_the_nearest_threat(self):
        for row, expected_distance in zip(self.fixture()["enemies"], (40, 21), strict=True):
            with self.subTest(frame=row["frame"]):
                snapshot = MarioStateParser().parse(row["info"], bytes.fromhex(row["ram_hex"]))
                observed = snapshot.to_debug_state()["visible_enemies"]
                self.assertEqual(len(observed), 2)
                self.assertTrue(observed[0]["defeated"])
                hazard = snapshot.threat_features()
                self.assertEqual(hazard["nearest_enemy_distance_pixels"], expected_distance)
                self.assertEqual(len(hazard["upcoming_enemies"]), 1)
                self.assertIsNone(hazard["spacing_to_second_enemy_pixels"])

    def test_only_confirmed_defeated_enemies_are_excluded(self):
        snapshot = MarioStateParser().parse({})
        for kind, motion, excluded in (
            (6, 0x22, True),
            (6, 4, True),
            (0, 0x20, True),
            (0, 4, False),  # A Koopa shell is not a defeated Goomba.
            (6, 0, False),
            (6, None, False),
            (5, 0x20, False),  # Unknown state semantics stay unknown.
        ):
            with self.subTest(kind=kind, motion=motion):
                enemy = EnemyObservation(0, kind, "test_enemy", 12, 0, motion_state=motion)
                state = replace(snapshot, enemies=(enemy,)).threat_features()
                self.assertEqual(state["enemy_ahead"], not excluded)
                self.assertEqual(len(state["upcoming_enemies"]), 0 if excluded else 1)
                if excluded:
                    self.assertIsNone(state["estimated_contact_frames"])
                    self.assertFalse(state["jump_must_start_this_decision"])
