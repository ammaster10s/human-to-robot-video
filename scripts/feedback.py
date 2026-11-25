"""Tell the demonstrator, in plain words, whether a recording is usable and how to fix it.

This is the fast path of the human-in-the-loop idea in the proposal: right after
a take, plan and replay the motion (no video generation) and turn every failure
into one instruction the person who recorded the demonstration can act on.
"""

import argparse
import json
import re
from pathlib import Path

from config import OBJECT_WORKSPACE_XY
from grip import FORCE_TARGETS, grip_mode
from retarget import demo_frames, plan
from robot import replay
from run import check_replay

# (pattern in the planner or replay message, short problem, instruction to the demonstrator)
RULES = [
    (r"at least three tracked frames",
     "The recording is too short.",
     "Record the whole movement: reach, pick up, carry, put down and move your hand away."),
    (r"frame times must start at zero",
     "The timestamps are out of order.",
     "Export the track again; this is a recording-tool problem, not your movement."),
    (r"must be an object|must be a finite number|must contain three coordinates",
     "Tracking was lost.",
     "Keep your hand and the object fully visible to the camera for the whole take."),
    (r"outside the gantry workspace",
     "The object is out of the robot's reach.",
     f"Keep the object within about {OBJECT_WORKSPACE_XY * 100:.0f} cm of the centre of the table."),
    (r"open approach followed by a pinch",
     "No clear grasp was found.",
     "Start with an open hand, then pinch the object between your thumb and index finger."),
    (r"release followed by a retreat",
     "No clear release was found.",
     "After putting the object down, open your fingers and move your hand away before stopping."),
    (r"near the table",
     "The object was picked up or released in the air.",
     "Pick the object up from the table and put it back on the table before letting go."),
    (r"moved before the grasp",
     "The object moved before you grasped it.",
     "Do not touch or push the object until you pinch it."),
    (r"moved after the release",
     "The object moved after you released it.",
     "Let go gently and keep your hand clear so the object stays still."),
    (r"separated from the object during the carry",
     "Your hand lost the object during the carry.",
     "Keep the object between your fingertips while you carry it."),
    (r"lift clear of the table",
     "The object was not lifted.",
     "Lift the object clearly off the table while you move it."),
    (r"contact was unstable",
     "The robot could not hold the object steadily.",
     "Carry the object more slowly and smoothly."),
    (r"not released within",
     "The robot did not place the object where you did.",
     "Put the object down gently and hold still for a moment before releasing it."),
    (r"tool strayed",
     "The robot could not follow your hand.",
     "Move more slowly so the robot can follow your path."),
    (r"object strayed",
     "The object slipped away from your path in the robot replay.",
     "Carry the object more slowly and avoid sudden turns."),
]


def explain(message: str) -> dict:
    """Map one planner or replay message to a problem and an instruction."""
    frame = re.search(r"frame (\d+)", message)
    for pattern, problem, instruction in RULES:
        if re.search(pattern, message):
            return {
                "problem": problem,
                "instruction": instruction,
                "frame": int(frame.group(1)) if frame else None,
                "detail": message,
            }
    return {"problem": "The recording could not be used.", "instruction": "Record the demonstration again.",
            "frame": int(frame.group(1)) if frame else None, "detail": message}


def review(frames, mode: str = "gentle") -> dict:
    """Plan and replay one recording and return usable/unusable plus feedback."""
    try:
        actions = plan(frames)
    except ValueError as error:
        return {"usable": False, "stage": "planning", "feedback": [explain(str(error))]}
    samples = replay(actions, FORCE_TARGETS[mode])
    physical = check_replay(actions, samples)
    if physical["accepted"]:
        return {
            "usable": True,
            "stage": "replay",
            "feedback": [{"problem": None, "instruction": "Good take: the robot can reproduce it.",
                          "frame": None, "detail": None}],
            "physical": physical,
        }
    return {
        "usable": False,
        "stage": "replay",
        "feedback": [explain(reason) for reason in physical["reasons"]],
        "physical": physical,
    }


def example_takes() -> dict:
    """A good take and typical mistakes, built from the synthetic demo."""
    takes = {"good": demo_frames()}

    far = demo_frames()
    for frame in far:
        frame["object_position"][1] += 0.20
        frame["thumb"][1] += 0.20
        frame["index"][1] += 0.20
    takes["object_out_of_reach"] = far

    no_release = demo_frames()
    for frame in no_release[40:]:
        frame["thumb"][1], frame["index"][1] = -0.021, 0.021
    takes["never_released"] = no_release

    lost = demo_frames()
    for frame in lost[20:30]:
        frame["thumb"][2] += 0.12
        frame["index"][2] += 0.12
    takes["hand_left_the_object"] = lost

    pushed = demo_frames()
    pushed[3]["object_position"][0] += 0.03
    takes["object_pushed_before_grasp"] = pushed
    return takes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracks", type=Path, help="tracked demonstration JSON (see README)")
    source.add_argument("--examples", action="store_true", help="review a good take and typical mistakes")
    parser.add_argument("--prompt", default="grip gently")
    args = parser.parse_args()
    try:
        mode = grip_mode(args.prompt)
        if args.examples:
            report = {name: review(frames, mode) for name, frames in example_takes().items()}
            usable = report["good"]["usable"] and not any(
                result["usable"] for name, result in report.items() if name != "good")
        else:
            report = review(json.loads(args.tracks.read_text(encoding="utf-8")), mode)
            usable = report["usable"]
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2))
    return 0 if usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
