import tempfile
import unittest
from pathlib import Path

from policy import TEST_LAYOUTS, TRAIN_LAYOUTS, _synthetic_actions, evaluate, fit_policy, load_demo, rollout


class PolicyTest(unittest.TestCase):
    def test_learned_policy_places_block_in_a_held_out_layout(self):
        demos = [_synthetic_actions(*layout) for layout in TRAIN_LAYOUTS]
        model = fit_policy(demos)
        layout = TEST_LAYOUTS[0]
        self.assertNotIn(layout, TRAIN_LAYOUTS)
        actions, _, metrics = evaluate(model, *layout, "gentle")
        self.assertTrue(metrics["accepted"], metrics)
        self.assertLess(metrics["goal_error_m"], 0.015)
        self.assertEqual(tuple(actions[0]["object_position"][:2]), layout[0])

        for demo in demos:
            demo[0]["position"][2] += 0.03
        raised = rollout(fit_policy(demos), *layout)
        self.assertAlmostEqual(raised[0]["position"][2] - actions[0]["position"][2], 0.03)

    def test_rejected_trial_cannot_train_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "report.json").write_text('{"accepted": false}')
            with self.assertRaisesRegex(ValueError, "training trial was rejected"):
                load_demo(directory)
