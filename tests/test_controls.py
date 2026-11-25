import tempfile
import unittest
from pathlib import Path

from verify_controls import run_controls


class VerificationControlsTest(unittest.TestCase):
    def test_faithful_candidate_passes_and_corrupted_candidates_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run_controls(Path(temp) / "controls")
        decisions = {name: result["decision"] for name, result in report["controls"].items()}
        self.assertEqual(decisions, {
            "faithful": "accept",
            "object_drift": "reject",
            "tool_offset": "reject",
            "background_patch": "reject",
            "missing_frame": "reject",
        })
        self.assertTrue(report["all_as_expected"])
