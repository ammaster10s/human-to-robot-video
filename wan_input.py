"""Prepare masked Wan/VACE inputs from a rendered reference sequence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

WIDTH, HEIGHT, FPS = 832, 480, 16
FRAME_PATTERN = "frame_%06d.png"


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", *map(str, args)], check=True)


def prepare(reference: Path, output: Path) -> dict:
    masks = reference / "conditions" / "masks"
    frames = sorted(reference.glob("frame_*.png"))
    names = [FRAME_PATTERN % i for i in range(len(frames))]
    if [frame.name for frame in frames] != names or sorted(p.name for p in masks.glob("frame_*.png")) != names:
        raise ValueError("reference and robot-mask sequences need matching, consecutive frame names")
    if not 5 <= len(frames) <= 81:
        raise ValueError("VACE handoff expects a clip of 5 to 81 frames")
    with Image.open(frames[0]) as image:
        width, height = image.size
    for frame in frames:
        with Image.open(frame) as image, Image.open(masks / frame.name) as mask:
            if image.size != (width, height) or mask.size != image.size:
                raise ValueError(f"{frame.name}: frame and mask dimensions must match the sequence")

    scale = min(WIDTH / width, HEIGHT / height)
    fitted_w, fitted_h = 2 * int(width * scale / 2), 2 * int(height * scale / 2)
    if min(fitted_w, fitted_h) < 2:
        raise ValueError("source aspect ratio cannot fit the model canvas")
    left, top = 2 * ((WIDTH - fitted_w) // 4), 2 * ((HEIGHT - fitted_h) // 4)
    model_frames = len(frames) + (1 - len(frames)) % 4
    output.mkdir(parents=True)
    for source, filename, interpolation in (
        (reference, "control.mp4", "bicubic"), (masks, "robot-mask.mp4", "neighbor"),
    ):
        filters = (
            f"scale={fitted_w}:{fitted_h}:flags={interpolation},"
            f"pad={WIDTH}:{HEIGHT}:{left}:{top},setsar=1,"
            f"tpad=stop_mode=clone:stop={model_frames - len(frames)}"
        )
        ffmpeg("-framerate", FPS, "-start_number", 0, "-i", source / FRAME_PATTERN,
               "-vf", filters, "-frames:v", model_frames, "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", 8, output / filename)
    manifest = {
        "frames": len(frames), "model_frames": model_frames, "model_fps": FPS,
        "source_size": [width, height], "content_box": [fitted_w, fitted_h, left, top],
        "control": "control.mp4", "mask": "robot-mask.mp4",
    }
    (output / "input.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def unpack(video: Path, output: Path, manifest: dict) -> None:
    fitted_w, fitted_h, left, top = manifest["content_box"]
    width, height = manifest["source_size"]
    output.mkdir()
    ffmpeg("-i", video, "-vf", f"crop={fitted_w}:{fitted_h}:{left}:{top},scale={width}:{height}",
           "-fps_mode", "passthrough", "-frames:v", manifest["frames"],
           "-start_number", 0, output / FRAME_PATTERN)
    if len(list(output.glob("frame_*.png"))) != manifest["frames"]:
        raise ValueError(f"{video.name}: generated video ended before the reference sequence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference directory from run.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wan-repo", type=Path, help="local Wan2.1 checkout")
    parser.add_argument("--checkpoint", type=Path, help="local Wan2.1-VACE-1.3B checkpoint")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Wan environment Python")
    parser.add_argument("--prompt", help="describe robot appearance for masked editing")
    parser.add_argument("--seed", type=int, action="append", help="repeat to render multiple attempts")
    args = parser.parse_args()
    if any((args.wan_repo, args.checkpoint, args.prompt, args.seed)):
        if not all((args.wan_repo, args.checkpoint, args.prompt)):
            parser.error("inference requires --wan-repo, --checkpoint and --prompt")
        if any(seed < 0 for seed in args.seed or ()):
            parser.error("--seed must be nonnegative")
        if not (args.wan_repo / "generate.py").is_file() or not args.checkpoint.is_dir():
            parser.error("Wan generate.py or checkpoint directory is missing")
    try:
        output = args.out.resolve()
        manifest = prepare(args.reference, output)
        if args.wan_repo:
            repo = args.wan_repo.resolve()
            for seed in dict.fromkeys(args.seed or [42]):
                generated = output / f"seed-{seed}.mp4"
                subprocess.run([
                    str(args.python.absolute()), str(repo / "generate.py"), "--task", "vace-1.3B",
                    "--ckpt_dir", str(args.checkpoint.resolve()), "--size", f"{WIDTH}*{HEIGHT}",
                    "--src_video", str(output / manifest["control"]),
                    "--src_mask", str(output / manifest["mask"]), f"--prompt={args.prompt}",
                    "--frame_num", str(manifest["model_frames"]), "--base_seed", str(seed),
                    "--save_file", str(generated),
                ], cwd=repo, check=True)
                unpack(generated, output / f"seed-{seed}", manifest)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
