import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from perception import MARKERS, track
from retarget import demo_frames, plan

CAMERA = {
    "fx": 300, "fy": 300, "cx": 90, "cy": 50, "depth_scale": 0.001,
    "camera_to_robot": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.6], [0, 0, 0, 1]],
    "object_center_offset": [0, 0, -0.025],
}


def write_captures(rgb_dir, depth_dir, frames):
    rgb_dir.mkdir()
    depth_dir.mkdir()
    for i, frame in enumerate(frames):
        rgb = np.zeros((100, 180, 3), dtype=np.uint8)
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)
        for name, color in MARKERS.items():
            x, y, z = frame[name]
            if name == "object_position":
                z -= CAMERA["object_center_offset"][2]
            z = 0.6 - z
            u, v = round(90 + 300 * x / z), round(50 - 300 * y / z)
            rgb[v - 1:v + 2, u - 1:u + 2] = color
            depth[v - 1:v + 2, u - 1:u + 2] = round(1000 * z)
            depth[v - 1, u - 1] = 1000
        name = f"frame_{i:06d}.png"
        Image.fromarray(rgb).save(rgb_dir / name)
        Image.fromarray(depth).save(depth_dir / name)


class PerceptionTests(unittest.TestCase):
    def test_backprojection_follows_source_grasp_and_place(self):
        frames = demo_frames()
        with TemporaryDirectory() as temp:
            rgb_dir, depth_dir = Path(temp) / "rgb", Path(temp) / "depth"
            write_captures(rgb_dir, depth_dir, frames)
            observed = track(rgb_dir, depth_dir, CAMERA, 10)

        self.assertEqual([row["time"] for row in observed], [row["time"] for row in frames])
        for first, second in zip(frames, observed):
            for name in MARKERS:
                np.testing.assert_allclose(first[name], second[name], atol=0.002)
        actions = plan(observed)
        self.assertTrue(actions[10]["closed"])
        self.assertFalse(actions[43]["closed"])

    def test_rejects_misaligned_input_and_missing_depth(self):
        with TemporaryDirectory() as temp:
            rgb_dir, depth_dir = Path(temp) / "rgb", Path(temp) / "depth"
            write_captures(rgb_dir, depth_dir, demo_frames()[:1])
            path = depth_dir / "frame_000000.png"
            Image.fromarray(np.zeros((100, 180), dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "thumb needs a visible marker with valid depth"):
                track(rgb_dir, depth_dir, CAMERA, 10)
            path.rename(path.with_name("frame_000001.png"))
            with self.assertRaisesRegex(ValueError, "matching PNG sequences"):
                track(rgb_dir, depth_dir, CAMERA, 10)
