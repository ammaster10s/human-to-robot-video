import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wan_input import prepare, unpack


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is needed for video packaging")
class WanInputTest(unittest.TestCase):
    def test_round_trip_pads_for_vace_and_restores_source_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            masks = reference / "conditions" / "masks"
            masks.mkdir(parents=True)
            for index in range(6):
                name = f"frame_{index:06d}.png"
                Image.new("RGB", (96, 72), (index * 30, 60, 120)).save(reference / name)
                Image.new("L", (96, 72), 255).save(masks / name)
            bundle = root / "wan"
            manifest = prepare(reference, bundle)
            self.assertEqual(manifest["model_frames"], 9)
            self.assertTrue((bundle / "robot-mask.mp4").is_file())
            unpack(bundle / "control.mp4", bundle / "frames", manifest)
            paths = sorted((bundle / "frames").glob("*.png"))
            self.assertEqual(len(paths), 6)
            with Image.open(paths[-1]) as image:
                self.assertEqual(image.size, (96, 72))
                self.assertAlmostEqual(image.getpixel((20, 20))[0], 150, delta=5)
