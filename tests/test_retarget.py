import unittest

from retarget import demo_frames, plan


class RetargetTests(unittest.TestCase):
    def test_preserves_grasp_carry_and_place(self):
        frames = demo_frames()
        actions = plan(frames)
        self.assertEqual([row["time"] for row in actions], [row["time"] for row in frames])
        self.assertFalse(actions[0]["closed"])
        self.assertTrue(actions[10]["closed"])
        self.assertEqual(actions[10]["position"], frames[10]["object_position"])
        self.assertEqual(actions[25]["position"], frames[25]["object_position"])
        self.assertGreater(actions[25]["position"][2], actions[10]["position"][2])
        self.assertFalse(actions[43]["closed"])
        self.assertEqual(actions[45]["position"], frames[43]["object_position"])
        self.assertGreater(actions[-1]["position"][2], actions[43]["position"][2])

    def test_requires_timed_finite_tracks_within_workspace(self):
        for field, value, error in (
            ("time", 0.0, "strictly increase"),
            ("time", True, "finite number"),
            ("thumb", [0, 0, float("nan")], "finite number"),
            ("object_position", [1.0, 0, 0.025], "outside the gantry workspace"),
        ):
            with self.subTest(field=field, value=value):
                frames = demo_frames()
                frames[1][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    plan(frames)

    def test_accepts_numpy_scalars_from_trackers(self):
        import numpy as np
        frames = [
            {key: np.float64(value) if key == "time" else [np.float64(v) for v in value]
             for key, value in frame.items()}
            for frame in demo_frames()
        ]
        self.assertEqual(len(plan(frames)), len(frames))

    def test_ignores_pinches_away_from_the_object(self):
        frames = demo_frames()
        frames[3]["thumb"] = [0.1, 0, 0.1]
        frames[3]["index"] = [0.1, 0.04, 0.1]
        actions = plan(frames)
        self.assertFalse(actions[3]["closed"])
        self.assertTrue(actions[10]["closed"])

    def test_rejects_release_in_the_air(self):
        frames = demo_frames()
        frames[25]["index"][1] = 0.1
        with self.assertRaisesRegex(ValueError, "near the table"):
            plan(frames)
