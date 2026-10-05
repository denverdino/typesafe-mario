from __future__ import annotations

import json
import unittest
from pathlib import Path

from typesafe_mario.state import MarioStateParser


class RamObservationTests(unittest.TestCase):
    def recorded_frames(self, name):
        fixture = json.loads((Path(__file__).parent / "fixtures/observation-ram.json").read_text())
        ram = bytearray()
        for entry in fixture[name]:
            if "ram_hex" in entry:
                ram = bytearray.fromhex(entry["ram_hex"])
            else:
                for address, value in entry["ram_changes"].items():
                    ram[int(address, 16)] = value
            yield entry["frame"], entry["info"], ram

    def test_recorded_airborne_frames_do_not_reset_near_platform(self):
        parser = MarioStateParser()
        snapshots = {}
        for frame, info, ram in self.recorded_frames("motion_frames"):
            snapshots[frame] = parser.parse(info, ram)
        # The game enters Player_State=1 at f923 (A was pressed at f922).
        for frame, airtime in ((931, 9), (932, 10), (933, 11), (934, 12), (936, 14)):
            with self.subTest(frame=frame):
                self.assertFalse(snapshots[frame].grounded)
                self.assertEqual(snapshots[frame].airborne_frames, airtime)
        self.assertTrue(snapshots[943].grounded)
        self.assertEqual(snapshots[943].airborne_frames, 0)

    def test_recorded_enemy_y_above_screen_keeps_correct_vertical_relation(self):
        expected = {1136: 188, 1144: 201, 1152: 204}
        for frame, info, ram in self.recorded_frames("enemy_y_frames"):
            with self.subTest(frame=frame):
                snapshot = MarioStateParser().parse(info, ram)
                self.assertEqual(
                    [enemy.dy_pixels for enemy in snapshot.enemies], [expected[frame]] * 3
                )
                self.assertTrue(
                    all(
                        enemy["vertical_offset_pixels"] == expected[frame]
                        for enemy in snapshot.threat_features()["upcoming_enemies"]
                    )
                )

    def test_ram_ground_contact_wins_over_vertical_delta_and_missing_grid(self):
        parser = MarioStateParser()
        ram = bytearray(0x800)
        ram[0x0E] = 8
        ram[0x1D] = 1
        parser.parse({"y_pos": 84}, ram)
        ram[0x1D] = 0
        landed = parser.parse({"y_pos": 79}, ram)
        self.assertTrue(landed.grounded)
        self.assertEqual(landed.jump_phase, "grounded")
        self.assertEqual(landed.airborne_frames, 0)

    def test_jump_fall_and_climb_are_not_grounded_even_with_support(self):
        ram = bytearray(0x800)
        ram[0x0E] = 8
        ram[0x500 + 5 * 16 + 6] = 1
        for motion in (1, 2, 3):
            with self.subTest(motion=motion):
                ram[0x1D] = motion
                self.assertFalse(
                    MarioStateParser().parse({"x_pos": 100, "y_pixel": 80}, ram).grounded
                )

    def test_terminal_or_nonplay_state_is_not_grounded(self):
        ram = bytearray(0x800)
        ram[0x500 + 5 * 16 + 6] = 1
        for engine, flags in ((8, {"death": True}), (8, {"clear": True}), (11, {}), (4, {})):
            with self.subTest(engine=engine, flags=flags):
                ram[0x0E] = engine
                self.assertFalse(
                    MarioStateParser().parse({"x_pos": 100, "y_pixel": 80, **flags}, ram).grounded
                )

    def test_enemy_y_uses_both_objects_ram_positions(self):
        for player_y, enemy_y, expected in ((432, 416, -16), (252, 440, 188), (512, 440, -72)):
            with self.subTest(player_y=player_y):
                ram = bytearray(0x800)
                ram[0xF] = 1
                ram[0x16] = 6
                ram[0xB5], ram[0xCE] = divmod(player_y, 256)
                ram[0xB6], ram[0xCF] = divmod(enemy_y, 256)
                snapshot = MarioStateParser().parse({"y_pixel": player_y % 256}, ram)
                self.assertEqual(snapshot.enemies[0].dy_pixels, expected)

    def test_missing_or_short_ram_keeps_fallback_behavior(self):
        for ram in (None, bytearray(0x1D)):
            with self.subTest(ram=ram):
                snapshot = MarioStateParser().parse({"y_pos": 80}, ram)
                self.assertFalse(snapshot.grounded)
        fallback = MarioStateParser().parse({"enemy_types": [6], "y_pixel": 80})
        self.assertEqual(fallback.enemies[0].dy_pixels, 0)


class MarioStateParserTests(unittest.TestCase):
    def base_info(self, **overrides: object) -> dict[str, object]:
        info: dict[str, object] = {
            "world": 1,
            "stage": 1,
            "area": 1,
            "x_pos": 100,
            "y_pos": 80,
            "y_pixel": 80,
            "progress": 100,
            "time": 390,
            "life": 2,
            "status": "small",
        }
        info.update(overrides)
        return info

    def test_first_snapshot_is_not_stalled(self) -> None:
        snapshot = MarioStateParser().parse(self.base_info())

        self.assertEqual(snapshot.stalled_steps, 0)
        self.assertEqual(snapshot.direction, "nearly_stationary")

    def test_motion_and_stall_are_derived_across_frames(self) -> None:
        parser = MarioStateParser()
        parser.parse(self.base_info())

        moving = parser.parse(self.base_info(x_pos=108, progress=108, y_pos=75))
        stalled = parser.parse(self.base_info(x_pos=108, progress=108, y_pos=75))

        self.assertEqual(moving.direction, "moving_right")
        self.assertEqual(moving.vertical_motion, "falling")
        self.assertEqual(moving.best_progress, 108)
        self.assertEqual(stalled.stalled_steps, 1)

    def test_ram_enemy_and_grid_become_structured_state(self) -> None:
        ram = bytearray(0x0800)
        ram[0x000E] = 8
        ram[0x00CE] = 80
        ram[0x000F] = 1
        ram[0x0016] = 0x06
        ram[0x006E] = 0
        ram[0x0087] = 142
        ram[0x00CF] = 80
        ram[0x0010] = 1
        ram[0x0017] = 0x06
        ram[0x006F] = 0
        ram[0x0088] = 180
        ram[0x00D0] = 80

        snapshot = MarioStateParser().parse(self.base_info(), ram)
        state = snapshot.to_state()
        debug = snapshot.to_debug_state()

        self.assertEqual(state["hazard"]["nearest_enemy_kind"], "goomba")
        self.assertEqual(state["hazard"]["nearest_enemy_distance_pixels"], 42)
        self.assertEqual(len(state["hazard"]["upcoming_enemies"]), 2)
        self.assertEqual(state["hazard"]["spacing_to_second_enemy_pixels"], 38)
        self.assertEqual(len(debug["local_grid"]["rows"]), 9)
        self.assertIn("M", "".join(debug["local_grid"]["rows"]))
        self.assertIn("goomba 42px ahead", snapshot.to_text())

        moving = MarioStateParser()
        moving.parse(self.base_info(), ram)
        next_state = moving.parse(
            self.base_info(x_pos=108, progress=108),
            ram,
            previous_latency_ms=100,
        ).to_state()
        self.assertEqual(next_state["hazard"]["relative_velocity_x"], -8)
        self.assertEqual(next_state["hazard"]["estimated_contact_frames"], 4)
        self.assertTrue(next_state["hazard"]["contact_within_reaction_horizon"])
        self.assertTrue(next_state["hazard"]["takeoff_window_already_missed"])
        self.assertTrue(next_state["hazard"]["jump_must_start_this_decision"])
        self.assertEqual(next_state["reaction_timing"]["last_inference_delay_frames"], 6)
        self.assertEqual(next_state["hazard"]["projected_distance_after_reaction_pixels"], 0)

        deadline_ram = bytearray(0x0800)
        deadline_ram[0x000E] = 8
        deadline_ram[0x00CE] = 80
        deadline_ram[0x000F] = 1
        deadline_ram[0x0016] = 0x06
        deadline_ram[0x006E] = 0
        deadline_ram[0x0087] = 236
        deadline_ram[0x00CF] = 80
        for column in range(16):
            deadline_ram[0x0500 + 5 * 16 + column] = 1
        deadline_parser = MarioStateParser()
        deadline_parser.parse(self.base_info(), deadline_ram)
        deadline_state = deadline_parser.parse(
            self.base_info(x_pos=108, progress=108),
            deadline_ram,
            previous_latency_ms=100,
            previous_response_delay_frames=8,
        ).to_state()
        self.assertEqual(deadline_state["hazard"]["estimated_contact_frames"], 16)
        self.assertEqual(deadline_state["hazard"]["takeoff_deadline_frames"], 0)
        self.assertEqual(deadline_state["reaction_timing"]["last_inference_delay_frames"], 8)
        self.assertTrue(deadline_state["hazard"]["jump_must_start_this_decision"])

    def test_pipe_geometry_is_exposed_as_navigation_state(self) -> None:
        ram = bytearray(0x0800)
        ram[0x000E] = 8
        for column in range(16):
            ram[0x0500 + 5 * 16 + column] = 1
        pipe_column = 7
        ram[0x0500 + 3 * 16 + pipe_column] = 1
        ram[0x0500 + 4 * 16 + pipe_column] = 1

        snapshot = MarioStateParser().parse(self.base_info(), ram)
        terrain = snapshot.to_state()["terrain"]

        self.assertTrue(terrain["obstacle_ahead"])
        self.assertEqual(terrain["obstacle_distance_tiles"], 1)
        self.assertEqual(terrain["obstacle_height_tiles"], 2)
        self.assertNotIn("summary", terrain)

    def test_recent_control_tracks_macro_outcome(self) -> None:
        parser = MarioStateParser()
        parser.parse(self.base_info(), previous_action="right_run")
        state = parser.parse(
            self.base_info(x_pos=106, progress=106),
            previous_action="right_run",
        ).to_state()

        self.assertEqual(state["recent_control"]["action"], "right_run")
        self.assertEqual(state["recent_control"]["frames_observed"], 2)
        self.assertEqual(state["recent_control"]["progress_gained_pixels"], 6)

    def test_airborne_state_exposes_trajectory_and_low_reliability_geometry(self) -> None:
        ram = bytearray(0x0800)
        ram[0x000E] = 8
        for column in range(16):
            ram[0x0500 + 5 * 16 + column] = 1
        parser = MarioStateParser()
        parser.parse(self.base_info(), ram)

        ram[0x001D] = 1
        state = parser.parse(self.base_info(x_pos=106, y_pos=90, progress=106), ram).to_state()

        self.assertEqual(state["trajectory"]["airborne_frames"], 1)
        self.assertEqual(state["trajectory"]["horizontal_distance_since_takeoff_pixels"], 6)
        self.assertEqual(state["terrain"]["observation_reliability"], "low_airborne")

    def test_reset_clears_episode_history(self) -> None:
        parser = MarioStateParser(goal="Finish safely")
        parser.parse(self.base_info(x_pos=300, progress=300), previous_action="right_run")

        parser.reset()
        snapshot = parser.parse(self.base_info(x_pos=40, progress=40))

        self.assertEqual(snapshot.goal, "Finish safely")
        self.assertEqual(snapshot.best_progress, 40)
        self.assertEqual(snapshot.stalled_steps, 0)
        self.assertIsNone(snapshot.previous_action)


class EnemyParsingRegressionTests(unittest.TestCase):
    def scene(self, kind=6, active=1):
        ram = bytearray(0x800)
        ram[0xF] = active
        ram[0x16] = kind
        ram[0x87] = 140
        ram[0xCF] = 80
        return ram

    def test_enemy_ids_are_reported_with_correct_roles(self):
        cases = (
            (0, "green_koopa"),
            (0x0A, "grey_cheep_cheep"),
            (0x0D, "piranha_plant"),
            (0x0E, "green_paratroopa_jump"),
            (0x12, "spiny"),
            (0x30, "flagpole_flag"),
            (0x31, "star_flag"),
        )
        for kind, name in cases:
            with self.subTest(kind=kind):
                snapshot = MarioStateParser().parse({"x_pos": 100}, self.scene(kind))
                self.assertEqual(len(snapshot.enemies), 1)
                self.assertEqual(snapshot.enemies[0].kind, name)
                self.assertEqual(snapshot.to_debug_state()["visible_enemies"][0]["kind"], name)

    def test_inactive_type_zero_and_missing_ram_are_not_phantom_koopas(self):
        parser = MarioStateParser()
        self.assertEqual(parser.parse({"x_pos": 100}, self.scene(0, active=0)).enemies, ())
        self.assertEqual(parser.parse({"x_pos": 100}).enemies, ())
        self.assertEqual(parser.parse({"x_pos": 100, "enemy_types": [0] * 5}).enemies, ())

    def test_inactive_slot_does_not_leak_velocity_into_reused_slot(self):
        parser = MarioStateParser()
        ram = self.scene()
        parser.parse({"x_pos": 100}, ram)
        ram[0xF] = 0
        parser.parse({"x_pos": 100}, ram)
        ram[0xF] = 1
        ram[0x87] = 250
        fresh = parser.parse({"x_pos": 100}, ram).enemies[0]
        self.assertEqual(fresh.relative_velocity_x, 0)
        ram[0x87] = 249
        moved = parser.parse({"x_pos": 103}, ram).enemies[0]
        self.assertEqual(moved.relative_velocity_x, -4)

    def test_type_change_resets_velocity_without_observed_empty_frame(self):
        parser = MarioStateParser()
        ram = self.scene()
        parser.parse({"x_pos": 100}, ram)
        ram[0x16] = 0x12
        ram[0x87] = 240
        self.assertEqual(parser.parse({"x_pos": 100}, ram).enemies[0].relative_velocity_x, 0)

    def test_missing_motion_data_does_not_invent_a_landing_time(self):
        from dataclasses import replace

        from typesafe_mario.state import EnemyObservation

        snapshot = replace(
            MarioStateParser().parse({"x_pos": 100}),
            enemies=(EnemyObservation(0, 6, "goomba", 34, 0, -2),),
            grounded=False,
            airborne_frames=22,
            last_response_delay_frames=8,
        )
        hazard = snapshot.threat_features()
        self.assertEqual(hazard["estimated_contact_frames"], 17)
        self.assertIsNone(hazard["estimated_landing_frames"])
        self.assertIsNone(hazard["will_land_before_contact"])
        self.assertEqual(hazard["takeoff_deadline_frames"], 1)
        self.assertEqual(hazard["projected_distance_after_reaction_pixels"], 2)


class PredictionTests(unittest.TestCase):
    def cases(self):
        return json.loads((Path(__file__).parent / "fixtures/prediction-ram.json").read_text())[
            "cases"
        ]

    def snapshot(self, case):
        parser = MarioStateParser()
        for frame in case["frames"]:
            snapshot = parser.parse(
                frame["info"],
                bytes.fromhex(frame["ram_hex"]),
                previous_action=frame["action"],
                previous_response_delay_frames=8,
            )
        return snapshot

    def test_recorded_descent_predicts_actual_static_landing(self):
        for case in self.cases():
            with self.subTest(episode=case["episode"], frame=case["frame"]):
                self.assertEqual(
                    self.snapshot(case).threat_features()["estimated_landing_frames"],
                    case["expected_landing_frames"],
                )

    def test_airborne_request_warns_when_landing_precedes_action_start(self):
        for case in self.cases():
            if (case["episode"], case["frame"]) not in ((0, 352), (1, 800)):
                continue
            with self.subTest(frame=case["frame"]):
                snapshot = self.snapshot(case)
                hazard = snapshot.threat_features()
                self.assertFalse(snapshot.grounded)
                self.assertTrue(hazard["will_land_before_contact"])
                self.assertTrue(hazard["jump_must_start_this_decision"])

    def test_contact_uses_box_edges_and_fractional_walking_speed(self):
        case = next(c for c in self.cases() if c["episode"] == 0 and c["frame"] == 352)
        hazard = self.snapshot(case).threat_features()
        self.assertEqual(hazard["nearest_enemy_distance_pixels"], 36)
        self.assertEqual(hazard["nearest_enemy_collision_distance_pixels"], 26)
        self.assertEqual(hazard["closing_speed_pixels_per_frame"], 2.375)
        self.assertEqual(hazard["estimated_contact_frames"], 11)

    def test_expired_clearance_window_still_warns_of_immediate_danger(self):
        case = next(c for c in self.cases() if c["episode"] == 1 and c["frame"] == 808)
        hazard = self.snapshot(case).threat_features()
        self.assertTrue(hazard["takeoff_window_already_missed"])
        self.assertTrue(hazard["jump_must_start_this_decision"])

    def test_missing_support_or_nonstandard_motion_returns_unknown(self):
        case = next(c for c in self.cases() if c["episode"] == 1 and c["frame"] == 800)
        frame = case["frames"][-1]
        for scenario in ("pit", "swimming", "climbing", "death", "short_ram"):
            with self.subTest(scenario=scenario):
                ram = bytearray.fromhex(frame["ram_hex"])
                if scenario == "pit":
                    ram[0x500:0x6A0] = bytes(416)
                elif scenario == "swimming":
                    ram[0x704] = 1
                elif scenario == "climbing":
                    ram[0x1D] = 3
                elif scenario == "death":
                    ram[0x0E] = 11
                else:
                    ram = ram[:0x100]
                snapshot = MarioStateParser().parse(frame["info"], ram)
                self.assertIsNone(snapshot.threat_features()["estimated_landing_frames"])

    def test_invisible_left_foot_tile_prevents_right_foot_landing(self):
        case = next(c for c in self.cases() if c["episode"] == 1 and c["frame"] == 800)
        frame = case["frames"][-1]
        ram = bytearray.fromhex(frame["ram_hex"])
        # At projected contact X=1492: left foot is on column 93, right on 94.
        # SMB checks the nonempty left tile even if the right tile is solid.
        for row in (11, 12):
            ram[0x500 + (1495 // 256 % 2) * 208 + row * 16 + 1495 % 256 // 16] = 0x5F
        snapshot = MarioStateParser().parse(frame["info"], ram)
        self.assertIsNone(snapshot.threat_features()["estimated_landing_frames"])

    def test_stomped_enemy_does_not_use_stale_ram_walking_speed(self):
        case = next(c for c in self.cases() if c["episode"] == 0 and c["frame"] == 123)
        snapshot = self.snapshot(case)
        enemy = snapshot.enemies[0]
        self.assertTrue(enemy.defeated)
        self.assertIsNone(enemy.ram_relative_speed_x)
        self.assertEqual(enemy.relative_velocity_x, -1)
        self.assertFalse(snapshot.threat_features()["enemy_ahead"])


class TerrainObservationTests(unittest.TestCase):
    def recorded_support_cases(self):
        return json.loads((Path(__file__).parent / "fixtures/terrain-support.json").read_text())[
            "cases"
        ]

    def test_recorded_pipe_edges_are_lower_landings_not_pits(self):
        for case in self.recorded_support_cases()[:-1]:
            with self.subTest(episode=case["episode"], frame=case["frame"]):
                parser = MarioStateParser()
                ram = bytes.fromhex(case["ram_hex"])
                snapshot = parser.parse(case["info"], ram)
                terrain = snapshot.to_state()["terrain"]
                self.assertFalse(terrain["gap_ahead"])
                self.assertIsNone(terrain["gap_distance_tiles"])
                self.assertTrue(terrain["drop_ahead"])
                self.assertGreater(terrain["drop_height_pixels"], 0)
                self.assertEqual(terrain["clear_forward_tiles"], terrain["drop_distance_tiles"] - 1)
                # A jump off the pipe must not inherit a fictitious pit crossing.
                airborne_ram = bytearray(ram)
                airborne_ram[0x1D] = 1
                airborne = parser.parse(case["info"], airborne_ram)
                self.assertFalse(airborne.crossing_gap)

    def test_recorded_true_pit_keeps_width_and_classification(self):
        case = self.recorded_support_cases()[-1]
        terrain = (
            MarioStateParser()
            .parse(case["info"], bytes.fromhex(case["ram_hex"]))
            .navigation_features()
        )
        self.assertEqual(terrain["gap_width_tiles_visible"], 2)
        self.assertEqual(terrain["gap_kind"], "pit")
        self.assertFalse(terrain["drop_ahead"])

    def test_lower_landing_does_not_hide_a_later_pit(self):
        case = self.recorded_support_cases()[-2]
        ram = bytearray.fromhex(case["ram_hex"])
        # x1008: remove both bottom ground tiles after the lower landing.
        ram[0x500 + 208 + 11 * 16 + 15] = 0
        ram[0x500 + 208 + 12 * 16 + 15] = 0
        terrain = MarioStateParser().parse(case["info"], ram).navigation_features()
        self.assertEqual(terrain["drop_distance_tiles"], 1)
        self.assertEqual(terrain["drop_height_pixels"], 64)
        self.assertEqual(terrain["gap_distance_tiles"], 5)
        self.assertEqual(terrain["gap_width_tiles_visible"], 1)

    def test_cropped_grid_without_full_ram_cannot_confirm_a_pit(self):
        terrain = self.snapshot_with_grid(("..M........", "###..######")).navigation_features()
        self.assertEqual(terrain["gap_kind"], "unknown")

    def test_unknown_gap_is_not_promoted_to_known_crossing_in_history(self):
        ram = bytearray(0x5E0)  # Second block-buffer page is missing.
        ram[0xE] = 8
        ram[0x500 + 11 * 16 + 6] = 0x54
        parser = MarioStateParser()
        parser.parse({"x_pos": 100, "y_pixel": 176}, ram)
        ram[0x1D] = 1
        snapshot = parser.parse({"x_pos": 102, "y_pixel": 172, "y_pos": 83}, ram)
        self.assertFalse(snapshot.crossing_gap)
        self.assertEqual(
            snapshot.to_state()["terrain"]["last_grounded_preview"]["gap_kind"], "unknown"
        )

    def test_coins_hidden_blocks_and_vines_do_not_count_as_lower_landings(self):
        case = self.recorded_support_cases()[-2]
        for tile in (0xC2, 0xC3, 0x5F, 0x60, 0x26, 0xC5):
            with self.subTest(tile=tile):
                ram = bytearray.fromhex(case["ram_hex"])
                ram[0x500 + 208 + 11 * 16 + 15] = tile
                ram[0x500 + 208 + 12 * 16 + 15] = 0
                terrain = MarioStateParser().parse(case["info"], ram).navigation_features()
                self.assertEqual(terrain["gap_distance_tiles"], 5)
                self.assertEqual(terrain["gap_kind"], "pit")

    def snapshot_with_grid(self, rows):
        from dataclasses import replace

        return replace(MarioStateParser().parse({}), local_grid=tuple(rows))

    def test_separate_gaps_are_not_combined(self):
        state = self.snapshot_with_grid(("..M........", "###..##..##")).to_state()
        terrain = state["terrain"]
        self.assertEqual(terrain["gap_distance_tiles"], 1)
        self.assertEqual(terrain["gap_width_tiles_visible"], 2)

    def test_obstacle_after_first_gap_is_still_observed(self):
        terrain = self.snapshot_with_grid(("..M...#....", "###..######")).navigation_features()
        self.assertEqual(terrain["gap_width_tiles_visible"], 2)
        self.assertEqual(terrain["obstacle_distance_tiles"], 4)
        self.assertEqual(terrain["obstacle_height_tiles"], 1)

    def test_gap_at_view_edge_counts_only_visible_cells(self):
        terrain = self.snapshot_with_grid(("..M........", "########...")).navigation_features()
        self.assertEqual(terrain["gap_distance_tiles"], 6)
        self.assertEqual(terrain["gap_width_tiles_visible"], 3)

    def test_no_ground_reference_does_not_claim_clear_path(self):
        terrain = self.snapshot_with_grid(("..M........", "...........")).to_state()["terrain"]
        self.assertFalse(terrain["geometry_available"])
        self.assertNotIn("clear_forward_tiles", terrain)
        self.assertNotIn("gap_ahead", terrain)

    def test_recorded_staircase_ram_reports_first_gap_as_two_tiles(self):
        import json
        from pathlib import Path

        fixture = json.loads((Path(__file__).parent / "fixtures/staircase-gap.json").read_text())
        ram = bytearray(0x800)
        ram[0x000E] = 8
        ram[0x500:0x6A0] = bytes.fromhex(fixture["block_buffer_hex"])
        snapshot = MarioStateParser().parse(fixture["info"], ram)
        self.assertEqual(snapshot.local_grid[6], "..###..##..")
        terrain = snapshot.to_state()["terrain"]
        self.assertEqual(terrain["gap_distance_tiles"], 3)
        self.assertEqual(terrain["gap_width_tiles_visible"], 2)
        self.assertEqual(terrain["obstacle_distance_tiles"], 1)
        self.assertEqual(terrain["obstacle_height_tiles"], 1)


if __name__ == "__main__":
    unittest.main()
