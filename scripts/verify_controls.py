"""Negative controls for stage 5: show that verification rejects bad candidates.

The default pipeline verifies a candidate that is rendered from the physical
replay, which passes almost by construction. This script renders one faithful
candidate plus several deliberately corrupted ones that imitate typical
generation failures, and reports whether each is accepted or rejected.
"""

import argparse
import copy
import json
from pathlib import Path

from PIL import Image, ImageDraw

from grip import FORCE_TARGETS
from retarget import demo_frames, plan
from robot import replay
from run import check_replay, write_json
from video import render, verify

OBJECT_DRIFT_M = 0.04
TOOL_OFFSET_M = 0.03
PATCH_PX = 40


def _carry_indices(actions):
    start = actions[0]["object_position"][2]
    return [
        i for i, action in enumerate(actions)
        if action["closed"] and action["object_position"][2] > start + 0.02
    ]


def _paint_patch(directory: Path) -> None:
    """Change a background corner in every frame, as a generator might."""
    for path in sorted(directory.glob("frame_*.png")):
        with Image.open(path) as image:
            frame = image.convert("RGB")
        ImageDraw.Draw(frame).rectangle((4, 4, 4 + PATCH_PX, 4 + PATCH_PX), fill="#c0392b")
        frame.save(path)


def run_controls(output: Path, mode: str = "gentle") -> dict:
    actions = plan(demo_frames())
    samples = replay(actions, FORCE_TARGETS[mode])
    output.mkdir(parents=True)
    reference = output / "reference"
    render([dict(action, grip_force=FORCE_TARGETS[mode] if action["closed"] else 0.0)
            for action in actions], reference)

    carry = _carry_indices(actions)
    drifted = copy.deepcopy(samples)
    for i in carry:
        drifted[i]["object_position"][1] += OBJECT_DRIFT_M
    shifted = copy.deepcopy(samples)
    for row in shifted:
        row["position"][0] += TOOL_OFFSET_M

    controls = {
        "faithful": (samples, None, True),
        "object_drift": (drifted, None, False),
        "tool_offset": (shifted, None, False),
        "background_patch": (samples, _paint_patch, False),
        "missing_frame": (samples, lambda d: sorted(d.glob("frame_*.png"))[-1].unlink(), False),
    }
    results = {}
    for name, (rows, corrupt, expect_accept) in controls.items():
        candidate = output / name
        render(rows, candidate)
        if corrupt:
            corrupt(candidate)
        visual = verify(reference, candidate)
        physical = check_replay(actions, rows)
        accepted = visual["accepted"] and physical["accepted"]
        results[name] = {
            "expected": "accept" if expect_accept else "reject",
            "decision": "accept" if accepted else "reject",
            "as_expected": accepted == expect_accept,
            "visual_issues": visual["issues"],
            "physical_reasons": physical["reasons"],
            "max_tool_error_px": visual["max_tool_error_px"],
            "max_object_error_px": visual["max_object_error_px"],
        }
    report = {"all_as_expected": all(r["as_expected"] for r in results.values()), "controls": results}
    write_json(output / "controls.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new results directory")
    args = parser.parse_args()
    try:
        report = run_controls(args.out)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["all_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
