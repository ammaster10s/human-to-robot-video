import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import numpy as np

from video import _regions, render, verify


def samples():
    return [
        {"time": 0, "position": [-0.10, 0, 0.10], "closed": False,
         "object_position": [-0.10, 0, 0.025], "grip_force": 0},
        {"time": 0.1, "position": [0.08, 0, 0.08], "closed": True,
         "object_position": [0.08, 0, 0.08], "grip_force": 2},
    ]


class VideoTests(unittest.TestCase):
    def test_tracker_observations_allow_recolored_unmarked_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            render(samples()[:1], reference)
            name = "frame_000000.png"
            observations = candidate / "observations"
            (observations / "robot_masks").mkdir(parents=True)
            (observations / "object_masks").mkdir()
            conditions = reference / "conditions"
            with Image.open(reference / name) as image:
                rgb = np.array(image)
                _, robot, obj = _regions(image)
            rgb[robot | obj] = (132, 90, 180)
            Image.fromarray(rgb).save(candidate / name)
            for subdir, mask in (("robot_masks", robot), ("object_masks", obj)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(observations / subdir / name)
            shutil.copyfile(conditions / "keypoints.json", observations / "keypoints.json")
            self.assertTrue(verify(reference, candidate)["accepted"])

            path = observations / "keypoints.json"
            points = json.loads(path.read_text())
            points["frames"][0]["keypoints"]["left_jaw"][0] += 15
            path.write_text(json.dumps(points))
            result = verify(reference, candidate)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["max_tool_error_px"], 15)

    def test_conditions_and_independent_pixel_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render(samples()[:1], root / "reference")
            render(samples()[:1], root / "candidate")
            name = "frame_000000.png"
            conditions = root / "reference" / "conditions"
            camera = json.loads((conditions / "keypoints.json").read_text())
            wrist = tuple(camera["frames"][0]["keypoints"]["wrist"])
            with (Image.open(conditions / "masks" / name) as mask,
                  Image.open(conditions / "depth" / name) as depth):
                self.assertEqual(mask.size, depth.size)
                self.assertEqual(mask.getpixel(wrist), 255)
                self.assertGreater(depth.getpixel(wrist), 0)
                self.assertEqual(depth.getpixel((0, 0)), 0)

            candidate = root / "candidate" / name
            with Image.open(candidate) as image:
                edited = image.copy()
            ImageDraw.Draw(edited).rectangle((20, 20, 60, 50), fill="white")
            edited.save(candidate)
            report = verify(root / "reference", root / "candidate")
            self.assertFalse(report["accepted"])
            self.assertEqual(report["min_robot_iou"], 1)
            self.assertGreater(report["max_background_change"], 0)

    def test_pose_and_object_are_verified_from_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = samples()
            render(rows, root / "reference")
            render(rows, root / "match")
            self.assertTrue(verify(root / "reference", root / "match")["accepted"])
            with Image.open(root / "reference" / "preview.gif") as preview:
                self.assertEqual(preview.n_frames, 2)

            rows[1]["object_position"] = [0.11, 0.01, 0.08]
            render(rows, root / "drift")
            report = verify(root / "reference", root / "drift")
            self.assertFalse(report["accepted"])
            self.assertEqual(report["max_tool_error_px"], 0)
            self.assertGreater(report["max_object_error_px"], report["tolerance_px"])

            (root / "match" / "frame_000001.png").unlink()
            self.assertFalse(verify(root / "reference", root / "match")["accepted"])

    def test_mask_replaces_only_selected_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, masks = root / "frames", root / "masks"
            frames.mkdir()
            masks.mkdir()
            size = (320, 240)
            source_color, background_color = (110, 62, 18), (11, 44, 76)
            name = "frame_000000.png"
            Image.new("RGB", size, source_color).save(frames / name)
            Image.new("RGB", size, background_color).save(root / "background.png")
            mask = Image.new("L", size)
            ImageDraw.Draw(mask).rectangle((0, 0, 10, 10), fill=255)
            mask.save(masks / name)

            render(samples()[:1], root / "result", frames_dir=frames, masks_dir=masks,
                   background=root / "background.png")
            with Image.open(root / "result" / name) as image:
                self.assertEqual(image.getpixel((5, 5)), background_color)
                self.assertEqual(image.getpixel((20, 5)), source_color)

            Image.new("L", (10, 10)).save(masks / name)
            with self.assertRaisesRegex(ValueError, "sizes differ"):
                render(samples()[:1], root / "invalid", frames_dir=frames,
                       masks_dir=masks, background=root / "background.png")
            self.assertFalse((root / "invalid").exists())
