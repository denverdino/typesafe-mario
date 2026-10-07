from __future__ import annotations

import unittest
from dataclasses import replace

from typesafe_mario.actions import JUMP_RELEASE_ACTION, Action
from typesafe_mario.dashboard import DashboardCommand, clamp01, scoring_metrics
from typesafe_mario.state import MarioStateParser


class DashboardTests(unittest.TestCase):
    def test_scoring_metrics_preserve_unknown_and_lower_bounds(self):
        snapshot = MarioStateParser().parse({})
        metrics = dict(scoring_metrics(snapshot))
        self.assertEqual(metrics["Score"], "—")
        self.assertEqual(metrics["Coin counter"], "—")
        self.assertEqual(metrics["Collected (confirmed)"], "0")
        self.assertEqual(metrics["Stomps (confirmed)"], "0")
        snapshot = replace(
            snapshot,
            scoring=replace(
                snapshot.scoring, score=999990, coin_counter=0, confirmed_coins_collected=100
            ),
        )
        metrics = dict(scoring_metrics(snapshot))
        self.assertEqual(metrics["Score"], "999,990")
        self.assertEqual(metrics["Coin counter"], "0")
        self.assertEqual(metrics["Collected (confirmed)"], "100")
        self.assertTrue(
            all(value == "—" for _, value in scoring_metrics(replace(snapshot, scoring=None)))
        )

    def test_clamp01_handles_missing_and_out_of_range_values(self) -> None:
        self.assertEqual(clamp01(None), 0.0)
        self.assertEqual(clamp01(-0.2), 0.0)
        self.assertEqual(clamp01(0.42), 0.42)
        self.assertEqual(clamp01(1.8), 1.0)

    def test_jump_macros_have_non_jump_release_frames(self) -> None:
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_JUMP], Action.RIGHT)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.RIGHT_RUN_JUMP], Action.RIGHT_RUN)
        self.assertEqual(JUMP_RELEASE_ACTION[Action.JUMP], Action.NOOP)

    def test_dashboard_commands_are_stable_strings(self) -> None:
        self.assertEqual(DashboardCommand.RESTART, "restart")
        self.assertEqual(DashboardCommand.QUIT, "quit")


if __name__ == "__main__":
    unittest.main()
