import json
import unittest
from pathlib import Path

from typesafe_mario.state import MarioStateParser


class ObservationBoundaryTests(unittest.TestCase):
    def fixture(self):
        return json.loads(
            (Path(__file__).parent / "fixtures/observation-boundaries.json").read_text()
        )

    def snapshot(self, ep, frame):
        row = next(r for r in self.fixture()["cases"] if (r["episode"], r["frame"]) == (ep, frame))
        return MarioStateParser().parse(
            row["info"],
            bytes.fromhex(row["ram_hex"]),
            previous_action=row["action"],
            previous_response_delay_frames=8,
        )

    def test_high_enemy_ram_facts_remain_available_for_diagnosis(self):
        state = self.snapshot(3, 688).to_debug_state()
        enemy = next(e for e in state["visible_enemies"] if e["slot"] == 1)
        self.assertEqual(enemy["relative_y_pixels"], -120)
        self.assertFalse(enemy["vertical_overlap"])
        self.assertEqual(enemy["motion_state"], 0)
        self.assertEqual(enemy["collision_box_screen_xyxy"], [146, 68, 156, 76])

    def test_history_carries_reference_position_age_and_visible_gap_extent(self):
        parser = MarioStateParser()
        ram = bytearray()
        for row in self.fixture()["history"]:
            if "ram_hex" in row:
                ram = bytearray.fromhex(row["ram_hex"])
            else:
                for i, value in row["ram_changes"].items():
                    ram[int(i)] = value
            snapshot = parser.parse(row["info"], ram, previous_action=row["action"])
        state = snapshot.to_state()
        history = state["terrain"]["last_grounded_preview"]
        self.assertEqual(history["sample_x"], 1272)
        self.assertEqual(history["age_frames"], 40)
        self.assertEqual(history["displacement_since_sample_pixels"], 70)
        self.assertEqual(history["gap_start_world_x"], 1376)
        self.assertIsNone(history["gap_end_world_x"])  # Right edge was not yet visible.
        self.assertEqual(state["terrain"]["gap_end_world_x"], 1424)
        parser.reset()
        fresh = parser.parse({"x_pos": 40}).to_state()
        self.assertIsNone(fresh["terrain"]["last_grounded_preview"]["sample_x"])

    def test_full_height_columns_remain_available_when_local_grid_loses_ground(self):
        state = self.snapshot(0, 880).to_debug_state()
        self.assertFalse(state["model_state"]["terrain"]["geometry_available"])
        columns = state["observed_geometry"]["static_columns"]
        self.assertTrue(columns)
        self.assertTrue(any(208 in c["surface_y_screen"] for c in columns))
        self.assertEqual(state["observed_geometry"]["foot_probes"]["x_world"], [1743, 1752])

    def test_staircase_side_entry_is_not_a_landing(self):
        for frame in (1264, 1272, 1280):
            with self.subTest(frame=frame):
                snapshot = self.snapshot(1, frame)
                self.assertIsNone(snapshot.estimated_landing_frames)
