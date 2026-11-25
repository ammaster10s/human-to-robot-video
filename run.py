"""Retarget, replay, and verify one fixed-camera pick-and-place episode."""

import argparse
import json
import shutil
from math import dist
from pathlib import Path
from statistics import mean

from grip import FORCE_TARGETS, grip_mode
from retarget import demo_frames, plan
from robot import replay
from video import render, verify

# Physical acceptance thresholds for a replayed episode (metres).
CARRY_HEIGHT_M = 0.02        # object counts as carried once lifted this far
MIN_LIFT_M = 0.01
MIN_CONTACT_RATE = 0.8
PLACEMENT_TOLERANCE_M = 0.015
TOOL_TOLERANCE_M = 0.025
CARRY_TOLERANCE_M = 0.025


def check_replay(actions: list[dict], samples: list[dict]) -> dict:
    start_height = actions[0]["object_position"][2]
    pairs = list(zip(actions, samples, strict=True))
    carry = [
        (action, sample) for action, sample in pairs
        if action["closed"] and action["object_position"][2] > start_height + CARRY_HEIGHT_M
    ]
    holding = [sample for _, sample in carry if sample["contact"]]
    lift = max((sample["object_position"][2] - start_height for sample in holding), default=0.0)
    contact_rate = len(holding) / len(carry) if carry else 0.0
    placement = dist(actions[-1]["object_position"], samples[-1]["object_position"])
    tool_error = max(dist(action["position"], sample["position"]) for action, sample in pairs)
    carry_error = max((
        dist(action["object_position"], sample["object_position"])
        for action, sample in carry
    ), default=0.0)
    reasons = []
    if lift < MIN_LIFT_M or contact_rate < MIN_CONTACT_RATE:
        reasons.append("Bilateral contact was unstable during the lift and carry.")
    if placement > PLACEMENT_TOLERANCE_M or samples[-1]["contact"]:
        reasons.append(f"The object was not released within {PLACEMENT_TOLERANCE_M * 1000:.0f} mm of the placement target.")
    if tool_error > TOOL_TOLERANCE_M:
        reasons.append(f"The tool strayed more than {TOOL_TOLERANCE_M * 1000:.0f} mm from the planned path.")
    if carry_error > CARRY_TOLERANCE_M:
        reasons.append(f"The object strayed more than {CARRY_TOLERANCE_M * 1000:.0f} mm from its tracked path during the carry.")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "lift_m": lift,
        "carry_contact_rate": contact_rate,
        "placement_error_m": placement,
        "max_tool_error_m": tool_error,
        "max_carry_error_m": carry_error,
        "mean_carry_force_n": mean(sample["grip_force"] for _, sample in carry) if carry else 0.0,
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_trial(actions: list[dict], output: Path, mode: str, args: argparse.Namespace) -> dict:
    target = FORCE_TARGETS[mode]
    samples = replay(actions, target)
    physical = check_replay(actions, samples)
    candidate = output / "video"
    reference = output / "reference"
    output.mkdir(exist_ok=True)
    write_json(output / "actions.json", actions)
    write_json(output / "replay.json", samples)
    footage = dict(frames_dir=args.frames_dir, masks_dir=args.masks_dir, background=args.background)
    render([
        dict(action, grip_force=target if action["closed"] else 0.0)
        for action in actions
    ], reference, **footage)
    external = args.candidate_dir
    sources = [external] if isinstance(external, Path) else list(external or ())
    if not sources:
        render(samples, candidate, **footage)
    attempts = []
    for source in sources or [candidate]:
        visual = verify(reference, source)
        attempts.append({"source": str(source), "video": visual})
        if visual["accepted"]:
            break
    accepted = physical["accepted"] and visual["accepted"]
    if accepted:
        if sources:
            candidate.mkdir()
            for sidecar in ("conditions", "observations"):
                if (source / sidecar).is_dir():
                    shutil.copytree(source / sidecar, candidate / sidecar)
        pairs = []
        for index, action in enumerate(actions):
            name = f"frame_{index:06d}.png"
            if sources:
                shutil.copyfile(source / name, candidate / name)
            pairs.append({
                "image": f"video/{name}",
                "time": action["time"],
                "position": action["position"],
                "closed": action["closed"],
                "grip_target_n": target if action["closed"] else 0.0,
            })
        write_json(output / "dataset.json", pairs)

    report = {
        "accepted": accepted,
        "grip_mode": mode,
        "grip_target_n": target,
        "physical": physical,
        "video": visual,
        "decision": "accept" if accepted else "drop",
        "attempts": attempts,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new results directory")
    parser.add_argument("--tracks", type=Path, help="tracked human demo JSON. Defaults to the synthetic demo.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prompt", default="grip gently")
    modes.add_argument("--compare", action="store_true", help="replay both grip settings on the same track")
    parser.add_argument("--frames-dir", type=Path, help="source PNG sequence at the tracked timestamps")
    parser.add_argument("--masks-dir", type=Path, help="corresponding human masks")
    parser.add_argument("--background", type=Path, help="clean fixed-camera background PNG")
    parser.add_argument("--candidate-dir", type=Path, action="append",
                        help="pre-edited PNGs to verify, repeat for retry attempts")
    args = parser.parse_args()

    footage = (args.frames_dir, args.masks_dir, args.background)
    if any(footage) and not all(footage):
        parser.error("--frames-dir, --masks-dir and --background must be supplied together")
    if args.candidate_dir and args.compare:
        parser.error("--candidate-dir requires a single grip setting")

    try:
        mode = grip_mode(args.prompt)
        frames = json.loads(args.tracks.read_text(encoding="utf-8")) if args.tracks else demo_frames()
        actions = plan(frames)
        args.out.mkdir(parents=True)
        write_json(args.out / "tracks.json", frames)
        if args.compare:
            trials = {
                setting: run_trial(actions, args.out / setting, setting, args)
                for setting in FORCE_TARGETS
            }
            means = {
                setting: trial["physical"]["mean_carry_force_n"]
                for setting, trial in trials.items()
            }
            report = {
                "accepted": all(trial["accepted"] for trial in trials.values())
                and means["firm"] - means["gentle"] >= 0.5,
                "mean_carry_force_n": means,
                "placement_error_m": {
                    setting: trial["physical"]["placement_error_m"]
                    for setting, trial in trials.items()
                },
            }
            write_json(args.out / "comparison.json", report)
        else:
            report = run_trial(actions, args.out, mode, args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
