import unittest

from robot import replay


def pick_and_place():
    knots = [
        (0.0, -0.1, 0.13, False),
        (0.7, -0.1, 0.025, False),
        (0.9, -0.1, 0.025, True),
        (1.4, -0.1, 0.025, True),
        (2.2, -0.1, 0.13, True),
        (3.0, 0.1, 0.13, True),
        (3.8, 0.1, 0.025, True),
        (4.3, 0.1, 0.025, False),
        (4.8, 0.1, 0.025, False),
        (5.3, 0.1, 0.13, False),
    ]
    return [
        dict(time=time, position=[x, 0, z], closed=closed,
             object_position=[-0.1, 0, 0.025])
        for time, x, z, closed in knots
    ]


class ReplayTest(unittest.TestCase):
    def test_grip_force_and_placement_depend_on_contact(self):
        actions = pick_and_place()
        gentle = replay(actions, 2.0)
        firm = replay(actions, 5.0)
        for samples in (gentle, firm):
            self.assertEqual([row["time"] for row in samples], [row["time"] for row in actions])
            self.assertTrue(samples[3]["contact"])
            self.assertGreater(samples[4]["object_position"][2], 0.10)
            self.assertAlmostEqual(samples[-1]["object_position"][0], 0.1, delta=0.01)
            self.assertAlmostEqual(samples[-1]["object_position"][2], 0.025, delta=0.002)
            self.assertEqual(samples[-1]["grip_force"], 0)
        self.assertGreater(firm[3]["grip_force"], gentle[3]["grip_force"] + 1.5)

        for action in actions:
            action["closed"] = False
        no_grasp = replay(actions, 2.0)
        self.assertAlmostEqual(no_grasp[-1]["object_position"][0], -0.1, delta=0.01)

    def test_rejects_nonfinite_position(self):
        actions = pick_and_place()
        actions[3]["position"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "three finite coordinates"):
            replay(actions, 2.0)
