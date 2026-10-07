from __future__ import annotations

import io
import json
import unittest
from dataclasses import replace

from typesafe_mario.actions import Action
from typesafe_mario.run_trace import RunTrace
from typesafe_mario.scoring import ScoringSummary
from typesafe_mario.state import MarioStateParser


class TraceScoringTests(unittest.TestCase):
    def setUp(self):
        self.parser = MarioStateParser()
        self.summary = ScoringSummary()

    def snapshot(self, score=0, **info):
        return self.parser.parse({"score": score, "coins": 0, "time": 390, **info})

    def test_clear_keeps_preclear_and_clear_scores_distinct(self):
        self.summary.begin(self.snapshot(0))
        self.summary.observe(self.snapshot(200))
        self.summary.observe(self.snapshot(400, clear=True))
        result = self.summary.episode_result()
        self.assertEqual(result["score_before_clear"], 200)
        self.assertEqual(result["score_at_clear"], 400)
        self.assertEqual(result["net_score_gain"], 400)
        self.assertTrue(result["stage_clear"])

    def test_missing_score_is_not_zero_gain(self):
        self.summary.begin(self.snapshot(0))
        self.summary.mark_interval(0)
        self.summary.observe(self.snapshot(None))
        self.summary.observe(self.snapshot(400, clear=True))
        self.assertIsNone(self.summary.interval_result()["score_gain"])
        self.assertIsNone(self.summary.episode_result()["net_score_gain"])
        self.assertIsNone(self.summary.episode_result()["score_before_clear"])
        self.assertEqual(self.summary.episode_result()["coverage"]["valid_score_pairs"], 0)

    def test_dead_or_error_episode_is_not_success(self):
        self.summary.begin(self.snapshot(0))
        self.summary.observe(self.snapshot(200, death=True))
        result = self.summary.episode_result()
        self.assertFalse(result["stage_clear"])
        self.assertTrue(result["dead"])
        self.assertIsNone(result["score_at_clear"])
        self.assertIsNone(result["score_before_clear"])

    def test_hurt_is_not_upgrade(self):
        ram = bytearray(2048)
        ram[0xE] = 8
        first = self.parser.parse({"status": "small"}, ram)
        self.summary.begin(first)
        ram[0x756] = 1
        self.summary.observe(self.parser.parse({"status": "tall"}, ram))
        self.assertEqual(self.summary.episode_result()["confirmed_hurts"], 0)
        ram[0x756] = 0
        ram[0xE] = 10
        ram[0x79E] = 8
        self.summary.observe(self.parser.parse({"status": "small"}, ram))
        self.assertEqual(self.summary.episode_result()["confirmed_hurts"], 1)

    def test_frame_gaps_preserve_incomplete_coverage(self):
        self.summary.begin(self.snapshot(0))
        frame = self.snapshot(200)
        self.summary.observe(replace(frame, observation_frame=3))
        self.assertEqual(self.summary.episode_result()["coverage"]["total_frames"], 3)
        self.assertIsNone(self.summary.interval_result()["score_gain"])
        self.assertIn("frame_gap", self.summary.interval_result()["coverage_gaps"])

    def test_end_is_idempotent(self):
        stream = io.StringIO()
        trace = RunTrace(stream)
        initial = self.snapshot(0)
        trace.begin({}, None, initial, seed=123)
        trace.apply(0, Action.RIGHT)
        final = self.snapshot(200, clear=True)
        trace.step(
            Action.RIGHT, {"score": 200}, None, final, reward=1, terminated=True, truncated=False
        )
        trace.end("clear")
        trace.end("quit")
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        ends = [event for event in events if event["event"] == "episode_end"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["reason"], "clear")
        self.assertEqual(ends[0]["summary"]["net_score_gain"], 200)
        self.assertEqual(events[0]["schema_version"], 2)
        self.assertEqual([e["frame"] for e in events if e["event"] == "score_change"], [1])

    def test_duplicate_observation_cannot_add_hurts_or_coverage(self):
        self.summary.begin(self.snapshot(0))
        frame = self.snapshot(200)
        self.summary.observe(frame)
        saved = self.summary.episode_result()
        self.summary.observe(frame)
        self.assertEqual(self.summary.episode_result(), saved)
