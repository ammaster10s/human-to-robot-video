"""Fit and evaluate a goal-conditioned pick-and-place policy in MuJoCo."""

import argparse
import json
from math import dist
from pathlib import Path
from statistics import mean

import numpy as np

from grip import FORCE_TARGETS, grip_mode
from retarget import TABLE_Z, demo_frames, plan
from robot import replay
from run import check_replay, run_trial, write_json


TRAIN_LAYOUTS = (
    ((-0.10, 0.0), (0.10, 0.0)),
    ((-0.08, -0.04), (0.08, 0.04)),
    ((0.07, 0.02), (-0.07, -0.02)),
)
TEST_LAYOUTS = (
    ((-0.11, 0.05), (0.07, -0.03)),
    ((0.09, -0.04), (-0.09, 0.06)),
    ((-0.04, -0.08), (0.04, 0.08)),
)


def load_demo(directory: Path) -> list[dict]:
    report = json.loads((directory / "report.json").read_text())
    if not isinstance(report, dict) or report.get("accepted") is not True:
        raise ValueError(f"training trial was rejected: {directory}")
    labels = json.loads((directory / "dataset.json").read_text())
    actions = json.loads((directory / "actions.json").read_text())
    if not isinstance(labels, list) or not isinstance(actions, list):
        raise ValueError(f"training labels and actions must be arrays: {directory}")
    if not labels or len(labels) != len(actions):
        raise ValueError(f"training labels and actions have different lengths: {directory}")
    for index, (label, action) in enumerate(zip(labels, actions, strict=True)):
        image = f"video/frame_{index:06d}.png"
        if (
            not isinstance(label, dict) or not isinstance(action, dict)
            or label.get("image") != image or not (directory / image).is_file()
            or any(label[key] != action[key] for key in ("time", "position", "closed"))
        ):
            raise ValueError(f"training label {index} does not match its verified action: {directory}")
    return [
        dict(time=label["time"], position=label["position"], closed=label["closed"],
             object_position=action["object_position"])
        for label, action in zip(labels, actions, strict=True)
    ]


def fit_policy(demos: list[list[dict]]) -> dict:
    """Regress 3D trajectories on the start and goal of aligned demonstrations."""
    if not demos or not demos[0]:
        raise ValueError("at least one verified demonstration is required")
    times = [row["time"] for row in demos[0]]
    if any([row["time"] for row in demo] != times for demo in demos):
        raise ValueError("training demonstrations must use the same time grid")

    origins = np.asarray([demo[0]["object_position"] for demo in demos], dtype=float)
    endpoints = np.asarray([demo[-1]["object_position"] for demo in demos], dtype=float)
    delta = endpoints[:, :2] - origins[:, :2]
    squared_length = float(np.square(delta).sum())
    if not np.isfinite(squared_length) or squared_length < 1e-6:
        raise ValueError("training requires horizontal object displacement")

    policy = {
        "time": times,
        "closed": (np.mean([[row["closed"] for row in demo] for demo in demos], axis=0) > 0.5).tolist(),
    }
    # shortcut: scalar travel and shared heights assume straight tabletop paths and aligned phases.
    for key in ("position", "object_position"):
        points = np.asarray([[row[key] for row in demo] for demo in demos], dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("training points must be finite")
        offsets = points - origins[:, None, :]
        progress = (offsets[:, :, :2] * delta[:, None, :]).sum(axis=(0, 2)) / squared_length
        policy[key] = {"progress": progress.tolist(), "height_m": offsets[:, :, 2].mean(axis=0).tolist()}
    return policy


def rollout(policy: dict, start: tuple[float, float], goal: tuple[float, float]) -> list[dict]:
    """Predict open-loop commands from the fitted model and a specified scene."""
    delta = np.asarray(goal) - start
    rows = []
    for index, (time, closed) in enumerate(zip(policy["time"], policy["closed"], strict=True)):
        action = {"time": time, "closed": closed}
        for key in ("position", "object_position"):
            curve = policy[key]
            progress = curve["progress"][index]
            x, y = start + progress * delta
            action[key] = [float(x), float(y), TABLE_Z + curve["height_m"][index]]
        rows.append(action)
    return rows


def _synthetic_actions(start, goal):
    frames = demo_frames()
    x0, x1 = (frames[i]["object_position"][0] for i in (0, -1))
    for frame in frames:
        progress = (frame["object_position"][0] - x0) / (x1 - x0)
        x = start[0] + progress * (goal[0] - start[0])
        y = start[1] + progress * (goal[1] - start[1])
        for key in ("thumb", "index", "object_position"):
            frame[key][0] = x
            frame[key][1] += y
    return plan(frames)


def evaluate(policy: dict, start, goal, mode: str) -> tuple[list[dict], list[dict], dict]:
    commands = rollout(policy, start, goal)
    samples = replay(commands, FORCE_TARGETS[mode])
    result = check_replay(commands, samples)
    # Score against the requested goal even if the predicted object track ends elsewhere.
    goal_error = dist(samples[-1]["object_position"], [*goal, TABLE_Z])
    return commands, samples, {
        "start_xy_m": list(start),
        "goal_xy_m": list(goal),
        "goal_error_m": goal_error,
        "accepted": result["accepted"] and goal_error <= 0.015,
        "replay": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new training and evaluation directory")
    parser.add_argument("--demo-run", action="append", type=Path, default=[],
                        help="accepted pipeline run directory, repeatable")
    parser.add_argument("--prompt", default="grip gently", help="grip mode for evaluation")
    args = parser.parse_args()
    try:
        mode = grip_mode(args.prompt)
        demos = [load_demo(path) for path in args.demo_run]
        args.out.mkdir(parents=True)
        if not demos:
            render_args = argparse.Namespace(frames_dir=None, masks_dir=None, background=None, candidate_dir=None)
            (args.out / "training").mkdir()
            for index, layout in enumerate(TRAIN_LAYOUTS):
                directory = args.out / "training" / str(index)
                run_trial(_synthetic_actions(*layout), directory, mode, render_args)
                demos.append(load_demo(directory))

        policy = fit_policy(demos)
        write_json(args.out / "policy.json", policy)
        seen = sorted(set(
            (tuple(demo[0]["object_position"][:2]), tuple(demo[-1]["object_position"][:2]))
            for demo in demos
        ))
        held_out = [
            layout for layout in TEST_LAYOUTS
            if not any(np.allclose(layout, trained, rtol=0, atol=1e-5) for trained in seen)
        ]
        if not held_out:
            raise ValueError("no built-in evaluation layouts remain held out")

        report = {
            "grip_mode": mode,
            "training_runs": len(demos),
            "training_frames": sum(map(len, demos)),
        }
        for split, layouts in (("seen", seen), ("held_out", held_out)):
            results = []
            for index, (start, goal) in enumerate(layouts):
                commands, samples, metrics = evaluate(policy, start, goal, mode)
                trial = args.out / split / str(index)
                trial.mkdir(parents=True)
                write_json(trial / "actions.json", commands)
                write_json(trial / "replay.json", samples)
                results.append(metrics)
            report[split] = {
                "successful": sum(result["accepted"] for result in results),
                "trials": len(results),
                "mean_goal_error_m": mean(result["goal_error_m"] for result in results),
                "results": results,
            }
        write_json(args.out / "report.json", report)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["held_out"]["successful"] == report["held_out"]["trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
