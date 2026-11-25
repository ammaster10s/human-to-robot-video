import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import config

MODEL = ET.parse(Path(__file__).resolve().parents[1] / "models" / "gantry.xml").getroot()


def _floats(text):
    return [float(value) for value in text.split()]


class ConfigMatchesModelTest(unittest.TestCase):
    def test_block_geometry_and_rest_height(self):
        block = MODEL.find(".//geom[@name='block']")
        self.assertEqual(_floats(block.get("size")), list(config.BLOCK_HALF_EXTENTS))
        body = MODEL.find(".//body[@name='object']")
        self.assertAlmostEqual(_floats(body.get("pos"))[2], config.TABLE_Z)

    def test_gantry_and_finger_joint_limits(self):
        for axis in "xy":
            low, high = _floats(MODEL.find(f".//joint[@name='{axis}']").get("range"))
            self.assertEqual((low, high), (-config.GANTRY_JOINT_XY, config.GANTRY_JOINT_XY))
        z_range = _floats(MODEL.find(".//joint[@name='z']").get("range"))
        self.assertEqual(tuple(z_range), config.GANTRY_JOINT_Z)
        finger = MODEL.find(".//default[@class='finger']/joint")
        self.assertEqual(_floats(finger.get("range"))[1], config.OPEN_JAW_POSITION)

    def test_workspaces_nest_inside_the_gantry(self):
        self.assertLess(config.OBJECT_WORKSPACE_XY, config.TOOL_WORKSPACE_XY)
        self.assertLessEqual(config.TOOL_WORKSPACE_XY, config.GANTRY_JOINT_XY)
        self.assertGreaterEqual(config.TOOL_MIN_Z, config.GANTRY_JOINT_Z[0])
        self.assertLessEqual(config.TOOL_MAX_Z, config.GANTRY_JOINT_Z[1])
        self.assertLess(config.OBJECT_MAX_Z, config.TOOL_MAX_Z)
