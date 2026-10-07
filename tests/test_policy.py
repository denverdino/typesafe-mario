from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from typesafe_sdk import Choice, Noul, Score

from typesafe_mario.actions import Action
from typesafe_mario.policy import TypeSafePolicy
from typesafe_mario.state import MarioStateParser


class CapturingClient:
    def __init__(self):
        self.call = None
        self.answers = {
            "next_action": SimpleNamespace(
                choice="right", confidence=0.8, probabilities={"right": 0.8, "jump": 0.2}
            ),
            "jump_needed": SimpleNamespace(noul=0.99),
            "danger": SimpleNamespace(score=2.0),
        }

    def system_one(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(choices={}, nouls={}, scores={}, answers=self.answers)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.client = CapturingClient()
        self.policy = TypeSafePolicy.__new__(TypeSafePolicy)
        self.policy._client = self.client
        self.policy._Choice = Choice
        self.policy._Noul = Noul
        self.policy._Score = Score
        self.snapshot = MarioStateParser().parse({"score": 400, "coins": 2, "time": 90})

    def choose(self, actions=tuple(Action)):
        return self.policy.choose(self.snapshot, actions)

    def test_scoring_goal_and_jump_question_are_consistent(self):
        self.choose()
        call = self.client.call
        self.assertEqual(set(call["questions"]), {"next_action", "jump_needed", "danger"})
        self.assertEqual(call["questions"]["next_action"].instructions["goal"], self.snapshot.goal)
        self.assertEqual(call["state"]["scoring"]["score"], 400)
        self.assertFalse(call["state"]["scoring"]["pursuit"]["allow_extra_effort"])
        # Request contract: scoring guidance is present in both action and jump questions.
        self.assertIn("scoring", call["questions"]["next_action"].instructions)
        self.assertIn("opportunities", str(call["questions"]["jump_needed"].instructions))
        self.assertIsInstance(call["questions"]["next_action"], Choice)
        self.assertIsInstance(call["questions"]["jump_needed"], Noul)

    def test_block_guidance_is_scoped_to_current_eligible_targets(self):
        case = json.loads(
            (Path(__file__).parent / "fixtures/coin-block-observations.json").read_text()
        )["cases"][0]
        ram = bytearray.fromhex(case["ram_hex"])
        self.snapshot = MarioStateParser().parse(case["info"], ram)
        self.choose()
        self.assertIn("reward_blocks", self.client.call["questions"]["next_action"].instructions)
        self.snapshot = MarioStateParser().parse(dict(case["info"], time=90), ram)
        self.choose()
        self.assertNotIn("reward_blocks", self.client.call["questions"]["next_action"].instructions)
        ram[0x500:0x6A0] = bytes(416)
        self.snapshot = MarioStateParser().parse(case["info"], ram)
        self.choose()
        self.assertNotIn("reward_blocks", self.client.call["questions"]["next_action"].instructions)

    def test_danger_stays_hazard_score(self):
        self.choose()
        danger = self.client.call["questions"]["danger"]
        self.assertIsInstance(danger, Score)
        self.assertEqual(
            danger.criteria,
            [
                "Safe open movement",
                "Potential obstacle or enemy soon",
                "Immediate collision, fall, or enemy threat",
            ],
        )

    def test_powerup_instructions_follow_block_and_emerging_item_then_stop(self):
        case = json.loads(
            (Path(__file__).parent / "fixtures/coin-block-observations.json").read_text()
        )["cases"][0]
        ram = bytearray.fromhex(case["ram_hex"])
        ram[0x500:0x6A0] = bytes(416)
        ram[0x500 + 208 + 7 * 16 + 5] = 0xC1  # x=336, y=144
        self.snapshot = MarioStateParser().parse(case["info"], ram)
        self.choose()
        instructions = self.client.call["questions"]["next_action"].instructions
        self.assertIn("reward_blocks", instructions)
        self.assertIn("powerups", instructions)
        ram[0x500:0x6A0] = bytes(416)
        ram[0x14], ram[0x1B], ram[0x23], ram[0x39] = 1, 0x2E, 1, 0
        ram[0x73], ram[0x8C], ram[0xBB], ram[0xD4] = 1, 80, 1, 136
        self.snapshot = MarioStateParser().parse(case["info"], ram)
        self.choose()
        instructions = self.client.call["questions"]["next_action"].instructions
        self.assertIn("powerups", instructions)
        self.assertNotIn("reward_blocks", instructions)
        self.snapshot = MarioStateParser().parse(dict(case["info"], time=90), ram)
        self.choose()
        self.assertNotIn("powerups", self.client.call["questions"]["next_action"].instructions)
        ram[0x14] = 0
        self.snapshot = MarioStateParser().parse(case["info"], ram)
        self.choose()
        self.assertNotIn("powerups", self.client.call["questions"]["next_action"].instructions)

    def test_scoring_does_not_override_selected_action(self):
        decision = self.choose()
        self.assertEqual(decision.action, Action.RIGHT)
        self.assertEqual(decision.jump_needed_probability, 0.99)
        self.assertEqual(decision.danger_score, 2.0)

    def test_choice_criteria_respect_allowed_actions(self):
        self.choose((Action.RIGHT, Action.JUMP))
        self.assertEqual(
            set(self.client.call["questions"]["next_action"].criteria), {"right", "jump"}
        )

    def test_answer_fallback_and_missing_answer(self):
        self.assertEqual(self.choose().action, Action.RIGHT)
        preferred = SimpleNamespace(choice="jump")
        response = SimpleNamespace(choices={"next_action": preferred}, answers=self.client.answers)
        self.assertIs(self.policy._answer(response, "next_action", "choices"), preferred)
        del self.client.answers["danger"]
        with self.assertRaises(KeyError):
            self.choose()
