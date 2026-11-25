import re
import unittest
from pathlib import Path

from feedback import RULES, example_takes, explain, review

ROOT = Path(__file__).resolve().parents[1]


def _messages():
    """Every ValueError text in the planner and every replay rejection reason."""
    retarget = (ROOT / "retarget.py").read_text(encoding="utf-8")
    run = (ROOT / "run.py").read_text(encoding="utf-8")
    planner = re.findall(r'raise ValueError\(f?"([^"]+)"', retarget)
    reasons = re.findall(r'reasons\.append\(f?"([^"]+)"', run)
    return planner + reasons


class FeedbackTest(unittest.TestCase):
    def test_every_planner_and_replay_message_has_an_instruction(self):
        messages = _messages()
        self.assertGreaterEqual(len(messages), 10)
        for message in messages:
            with self.subTest(message=message):
                self.assertNotEqual(explain(message)["problem"], "The recording could not be used.")

    def test_rules_are_all_reachable(self):
        joined = "\n".join(_messages())
        for pattern, _, _ in RULES:
            with self.subTest(pattern=pattern):
                self.assertRegex(joined, pattern)

    def test_examples_get_the_expected_advice(self):
        expected = {
            "good": None,
            "object_out_of_reach": "The object is out of the robot's reach.",
            "never_released": "No clear release was found.",
            "hand_left_the_object": "Your hand lost the object during the carry.",
            "object_pushed_before_grasp": "The object moved before you grasped it.",
        }
        for name, frames in example_takes().items():
            with self.subTest(take=name):
                result = review(frames)
                self.assertEqual(result["usable"], expected[name] is None)
                self.assertEqual(result["feedback"][0]["problem"], expected[name])

    def test_frame_number_and_unknown_messages(self):
        self.assertEqual(explain("frame 7 object is outside the gantry workspace")["frame"], 7)
        unknown = explain("something unexpected")
        self.assertEqual(unknown["instruction"], "Record the demonstration again.")
