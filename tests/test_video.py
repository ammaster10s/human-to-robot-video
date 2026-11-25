import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from video import render, verify


def samples():
    return [
        {"time": 0, "position": [-0.10, 0, 0.10], "closed": False,
         "object_position": [-0.10, 0, 0.025], "grip_force": 0},
        {"time": 0.1, "position": [0.08, 0, 0.08], "closed": True,
         "object_position": [0.08, 0, 0.08], "grip_force": 2},
    ]


class VideoTests(unittest.TestCase):
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
