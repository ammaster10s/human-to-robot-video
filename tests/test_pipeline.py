import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from retarget import demo_frames, plan
from robot import replay
from run import check_replay


class PipelineTest(unittest.TestCase):
    def test_same_demo_passes_both_grip_settings_and_exports_labels(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "paired"
            process = subprocess.run(
                [sys.executable, str(project / "run.py"), "--compare", "--out", str(output)],
                check=True, capture_output=True, text=True, timeout=45,
            )
            summary = json.loads(process.stdout)
            self.assertTrue(summary["accepted"])
            self.assertGreater(
                summary["mean_carry_force_n"]["firm"], summary["mean_carry_force_n"]["gentle"]
            )
            traces = []
            for mode in ("gentle", "firm"):
                trial = output / mode
                self.assertTrue((trial / "video" / "preview.gif").is_file())
                labels = json.loads((trial / "dataset.json").read_text())
                self.assertEqual(len(labels), 54)
                self.assertTrue(all((trial / label["image"]).is_file() for label in labels))
                self.assertTrue(any(label["closed"] for label in labels))
                traces.append([(row["time"], row["position"], row["closed"]) for row in labels])
            self.assertEqual(*traces)

    def test_object_drift_rejects_replay_even_when_the_tool_tracks(self):
        actions = plan(demo_frames())
        samples = replay(actions, 2.0)
        samples[25]["object_position"][1] += 0.04
        result = check_replay(actions, samples)
        self.assertFalse(result["accepted"])
        self.assertGreater(result["max_carry_error_m"], 0.025)
        self.assertIn("object strayed", " ".join(result["reasons"]))
