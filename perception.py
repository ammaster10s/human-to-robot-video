"""Track three colored landmarks in synchronized, calibrated RGB-D frames."""

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image

MARKERS = {
    "thumb": (255, 0, 0),
    "index": (0, 255, 0),
    "object_position": (0, 0, 255),
}


def _calibration(camera):
    try:
        fx, fy, cx, cy, scale = (float(camera[key]) for key in (
            "fx", "fy", "cx", "cy", "depth_scale"
        ))
        transform = np.asarray(camera["camera_to_robot"], dtype=float)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("expected fx, fy, cx, cy, depth_scale and a camera_to_robot matrix") from error
    if not all(isfinite(x) for x in (fx, fy, cx, cy, scale)) or min(fx, fy, scale) <= 0:
        raise ValueError("camera intrinsics and depth_scale must be finite, with positive fx, fy and depth_scale")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_robot must be a finite 4×4 matrix")
    rotation = transform[:3, :3]
    if not (
        np.allclose(transform[3], [0, 0, 0, 1])
        and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-4)
    ):
        raise ValueError("camera_to_robot must be a rigid transform")
    return fx, fy, cx, cy, scale, transform


def _landmark(rgb, depth, color, name, calibration):
    fx, fy, cx, cy, scale, transform = calibration
    # shortcut: controlled color markers, replace with a learned tracker for unmarked footage.
    rows, cols = np.where(np.max(np.abs(rgb - color), axis=2) <= 55)
    raw = depth[rows, cols]
    valid = raw > 0
    if valid.sum() < 3 or valid.sum() * 2 <= raw.size:
        raise ValueError(f"{name} needs a visible marker with valid depth")
    u, v = np.median(cols[valid]), np.median(rows[valid])
    z = float(np.median(raw[valid])) * scale
    camera_point = [(u - cx) * z / fx, (v - cy) * z / fy, z, 1]
    point = (transform @ camera_point)[:3]
    if not np.isfinite(point).all():
        raise ValueError(f"{name} projected outside finite coordinates")
    # Drop floating-point residue at exact workspace boundaries such as the tabletop.
    return point.round(9).tolist()


def track(rgb_dir: Path, depth_dir: Path, camera: dict, fps: float) -> list[dict]:
    """Return tracked fingertip and object positions in meters at fixed intervals."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    calibration = _calibration(camera)
    names = sorted(path.name for path in rgb_dir.glob("*.png"))
    expected = [f"frame_{i:06d}.png" for i in range(len(names))]
    if not names or names != expected or names != sorted(p.name for p in depth_dir.glob("*.png")):
        raise ValueError("RGB and depth need matching PNG sequences starting at frame_000000.png")

    tracks = []
    for i, name in enumerate(names):
        try:
            with Image.open(rgb_dir / name) as color, Image.open(depth_dir / name) as raw_depth:
                if color.format != "PNG" or raw_depth.format != "PNG":
                    raise ValueError("expected PNG input")
                if color.size != raw_depth.size:
                    raise ValueError("RGB and depth sizes differ")
                if raw_depth.mode not in {"I", "I;16", "I;16B", "I;16L"}:
                    raise ValueError("depth must be a single-channel 16-bit PNG")
                rgb = np.asarray(color.convert("RGB"), dtype=np.int16)
                depth = np.asarray(raw_depth)
            tracks.append({
                "time": i / fps,
                **{
                    name: _landmark(rgb, depth, color, name, calibration)
                    for name, color in MARKERS.items()
                },
            })
        except (OSError, ValueError) as error:
            raise ValueError(f"{name}: {error}") from error
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb_dir", type=Path)
    parser.add_argument("depth_dir", type=Path)
    parser.add_argument("camera", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()
    try:
        camera = json.loads(args.camera.read_text(encoding="utf-8"))
        tracks = track(args.rgb_dir, args.depth_dir, camera, args.fps)
        payload = json.dumps(tracks, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
