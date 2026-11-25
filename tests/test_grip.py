import unittest

from grip import FORCE_TARGETS, grip_mode


class GripModeTest(unittest.TestCase):
    def test_instructions_select_distinct_force_targets(self):
        self.assertEqual(grip_mode("  Grip gently.  "), "gentle")
        self.assertEqual(grip_mode("firmly"), "firm")
        self.assertLess(FORCE_TARGETS["gentle"], FORCE_TARGETS["firm"])

    def test_ambiguous_instruction_is_rejected(self):
        for instruction in ("", "do not grip firmly", "gently or firmly"):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                grip_mode(instruction)
