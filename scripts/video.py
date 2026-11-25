"""Fixed-camera previews and pixel checks for a simulated replay."""

import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageChops, ImageColor, ImageDraw

from config import BLOCK_HALF_EXTENTS


SIZE = (640, 420)
TOOL_MARKER = (255, 137, 50)
OBJECT_MARKER = (68, 232, 178)
MARKER_RADIUS = 4
TOLERANCE_PX = 8
MIN_ROBOT_IOU = 0.70
MAX_BACKGROUND_CHANGE = 0.001
MAX_TRACKED_BACKGROUND_CHANGE = 0.02
RAIL = "#7d97a5"
ARM = ("#63808f", "#4b626e", "#a9c0c9")
FINGERS = ("#afbec7", "#657c88", "#d6e3e6")
BLOCK = ("#146456", "#104741", "#328b76")


def _color_code(color):
    r, g, b = ImageColor.getrgb(color) if isinstance(color, str) else color
    return (r << 16) | (g << 8) | b


ROBOT_COLORS = [_color_code(color) for color in (RAIL, *ARM, *FINGERS, TOOL_MARKER)]
OBJECT_COLORS = [_color_code(color) for color in (*BLOCK, OBJECT_MARKER)]


def project(point, size):
    x, y, z = point
    width, height = size
    scale = min(size) * 1.45
    return (
        round(width * 0.5 + scale * (x - 0.36 * y)),
        round(height * 0.76 - scale * (0.46 * y + z)),
    )


def _table():
    canvas = Image.new("RGB", SIZE, "#17242d")
    draw = ImageDraw.Draw(canvas)
    edge = 0.23
    corners = [(-edge, -edge, 0), (edge, -edge, 0), (edge, edge, 0), (-edge, edge, 0)]
    draw.polygon([project(p, SIZE) for p in corners], fill="#263842", outline="#546b76")
    for i in range(-4, 5):
        offset = i * 0.05
        draw.line([project((offset, y, 0), SIZE) for y in (-edge, edge)], fill="#354b56")
        draw.line([project((x, offset, 0), SIZE) for x in (-edge, edge)], fill="#354b56")
    return canvas


def _depth_mm(point):
    x, y, z = point
    return round(1000 * (1 + (0.36 * x + y - 0.46 * z) / math.sqrt(1 + 0.36**2 + 0.46**2)))


def _box(draw, depth, size, center, half, colors):
    x, y, z = center
    dx, dy, dz = half
    footprint = [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)]
    top = [project((u, v, z + dz), size) for u, v in footprint]
    base = [project((u, v, z - dz), size) for u, v in footprint]
    faces = [
        ((base[0], base[1], top[1], top[0]), (x, y - dy, z)),
        ((base[1], base[2], top[2], top[1]), (x + dx, y, z)),
        (top, (x, y, z + dz)),
    ]
    for (polygon, face_center), color in zip(faces, colors, strict=True):
        draw.polygon(polygon, fill=color)
        depth.polygon(polygon, fill=_depth_mm(face_center) if colors != BLOCK else 0)


def _overlay(canvas, row):
    draw = ImageDraw.Draw(canvas)
    size = canvas.size
    depth_image = Image.new("I", size)
    depth = ImageDraw.Draw(depth_image)
    x, y, z = row["position"]
    ox, oy, oz = row["object_position"]
    gap = row.get("gripper_width", 0.04 if row["closed"] else 0.082)
    wrist = (x, y, z + 0.047)
    left = (x - gap / 2 - 0.007, y, z + 0.014)
    right = (x + gap / 2 + 0.007, y, z + 0.014)
    object_marker = (ox, oy - 0.02, oz - 0.012)

    rail = [project((end, y, 0.30), size) for end in (-0.25, 0.25)]
    rail_width = max(2, round(min(size) * 0.018))
    draw.line(rail, fill=RAIL, width=rail_width)
    depth.line(rail, fill=_depth_mm((0, y, 0.30)), width=rail_width)
    shaft_height = 0.30 - wrist[2]
    if shaft_height > 0:
        _box(draw, depth, size, (x, y, wrist[2] + shaft_height / 2),
             (0.011, 0.011, shaft_height / 2), ARM)
    _box(draw, depth, size, wrist, (gap / 2 + 0.016, 0.015, 0.008), ARM)
    _box(draw, depth, size, left, (0.007, 0.012, 0.029), FINGERS)
    _box(draw, depth, size, (ox, oy, oz), BLOCK_HALF_EXTENTS, BLOCK)
    _box(draw, depth, size, right, (0.007, 0.012, 0.029), FINGERS)

    for point, color in ((wrist, TOOL_MARKER), (object_marker, OBJECT_MARKER)):
        u, v = project(point, size)
        r = MARKER_RADIUS
        bounds = (u - r, v - r, u + r, v + r)
        draw.ellipse(bounds, fill=color)
        depth.ellipse(bounds, fill=_depth_mm(point) if color == TOOL_MARKER else 0)
    return depth_image, {
        key: project(point, size)
        for key, point in (("wrist", wrist), ("left_jaw", left),
                           ("right_jaw", right), ("object", object_marker))
    }


def _frame_names(directory):
    return sorted(path.name for path in directory.glob("frame_*.png"))


def _image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def render(rows, output: Path, *, frames_dir=None, masks_dir=None, background=None):
    """Write PNGs, a preview GIF and aligned schematic render conditions."""
    times = [row["time"] for row in rows]
    if not times or not all(math.isfinite(t) for t in times) or any(
        t >= next_t for t, next_t in zip(times, times[1:])
    ):
        raise ValueError("render needs at least one row with strictly increasing times")
    names = [f"frame_{i:06d}.png" for i in range(len(rows))]
    if any(path is not None for path in (frames_dir, masks_dir, background)):
        if frames_dir is None or masks_dir is None or background is None:
            raise ValueError("provide the source frames, masks and background together")
        if _frame_names(frames_dir) != names or _frame_names(masks_dir) != names:
            raise ValueError("source and mask sequences must match the replay frame names")

    if output.exists():
        raise FileExistsError(output)
    base = _table() if background is None else _image(background, "RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = []
    preview_times = []
    preview_step = math.ceil(len(rows) / 120)

    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "frames"
        staged.mkdir()
        conditions = staged / "conditions"
        (conditions / "masks").mkdir(parents=True)
        (conditions / "depth").mkdir()
        keypoint_rows = []
        for i, (name, row) in enumerate(zip(names, rows)):
            canvas = base.copy()
            if frames_dir is not None:
                source = _image(frames_dir / name, "RGB")
                mask = _image(masks_dir / name, "L")
                if source.size != base.size or mask.size != base.size:
                    raise ValueError(f"{name}: source, mask and background sizes differ")
                canvas = Image.composite(base, source, mask)
            depth, points = _overlay(canvas, row)
            canvas.save(staged / name)
            depth.convert("I;16").save(conditions / "depth" / name)
            Image.fromarray((np.asarray(depth) > 0).astype(np.uint8) * 255).save(
                conditions / "masks" / name
            )
            keypoint_rows.append({"file": name, "time": times[i], "keypoints": points})
            if i % preview_step == 0 or i == len(rows) - 1:
                canvas.thumbnail((480, 320))
                preview.append(canvas)
                preview_times.append(times[i])

        (conditions / "keypoints.json").write_text(json.dumps({
            "camera": {"image_size": base.size, "scale_px_per_m": min(base.size) * 1.45,
                       "depth_unit": "mm", "depth_origin_m": 1.0,
                       "depth_convention": "schematic optical-axis depth sampled at face centers"},
            "frames": keypoint_rows,
        }, indent=2) + "\n", encoding="utf-8")
        delays = [
            max(10, min(65_535 * 10, round((end - start) * 1000)))
            for start, end in zip(preview_times, preview_times[1:])
        ]
        delays.append(500)
        preview[0].save(staged / "preview.gif", save_all=True, append_images=preview[1:],
                        duration=delays, loop=0)
        _rename_with_retry(staged, output)


def _rename_with_retry(source, target, attempts=8, delay=0.25):
    # Windows sync clients (e.g. OneDrive) briefly lock freshly written folders.
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _center(image, color):
    # shortcut: exact palette markers, use a tracker when validating model-generated video.
    red, green, blue = ImageChops.difference(image, Image.new("RGB", image.size, color)).split()
    distance = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    bounds = distance.point([255] + [0] * 255).getbbox()
    if bounds:
        x, y, right, bottom = bounds
        diameter = 2 * MARKER_RADIUS + 1
        if right - x == diameter and bottom - y == diameter:
            return x + MARKER_RADIUS, y + MARKER_RADIUS
    raise ValueError("expected one complete palette marker")


def _regions(image):
    pixels = np.asarray(image, dtype=np.uint32)
    codes = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
    return codes, np.isin(codes, ROBOT_COLORS), np.isin(codes, OBJECT_COLORS)


def _geometry(expected, actual):
    expected_pixels, expected_robot, expected_object = _regions(expected)
    actual_pixels, actual_robot, actual_object = _regions(actual)
    robot = expected_robot | actual_robot
    background = ~(robot | expected_object | actual_object)
    iou = (expected_robot & actual_robot).sum() / max(1, robot.sum())
    changed = (background & (expected_pixels != actual_pixels)).sum() / max(1, background.sum())
    return float(iou), float(changed)


def _keypoints(manifest, names, size):
    rows = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
    if [row["file"] for row in rows] != names:
        raise ValueError("observation frame names or order differ")
    keys = ("wrist", "left_jaw", "right_jaw", "object")
    points = np.asarray([[row["keypoints"][key] for key in keys] for row in rows], dtype=float)
    if points.shape != (len(names), len(keys), 2) or not np.isfinite(points).all():
        raise ValueError("observations need finite 2D points for the wrist, jaws and object")
    if (points < 0).any() or (points >= size).any():
        raise ValueError("observation keypoints fall outside the image")
    return points


def _mask(path, size):
    mask = _image(path, "L")
    if mask.size != size:
        raise ValueError("observation mask dimensions differ from the frame")
    pixels = np.asarray(mask) > 127
    if not pixels.any():
        raise ValueError("observation mask is empty")
    return pixels


def _tracked_geometry(expected, actual, reference_mask, observations, name):
    _, _, expected_object = _regions(expected)
    expected_robot = _mask(reference_mask, expected.size)
    actual_robot = _mask(observations / "robot_masks" / name, actual.size)
    actual_object = _mask(observations / "object_masks" / name, actual.size)
    robot = expected_robot | actual_robot
    objects = expected_object | actual_object
    background = ~(robot | objects)
    color_change = np.abs(np.asarray(expected, dtype=np.int16) - np.asarray(actual, dtype=np.int16))
    changed = background & (color_change.max(axis=2) > 24)
    robot_iou = (expected_robot & actual_robot).sum() / robot.sum()
    object_iou = (expected_object & actual_object).sum() / objects.sum()
    return float(robot_iou), float(object_iou), float(changed.sum() / max(1, background.sum()))


def verify(reference: Path, candidate: Path):
    """Verify marked renders or a candidate with independent tracker observations."""
    names = _frame_names(reference)
    observations = candidate / "observations"
    tracked = observations.exists()
    geometry = all((path / "conditions" / "keypoints.json").is_file()
                   for path in (reference, candidate))
    checks = ["markers", "robot_iou", "background"] if geometry else ["markers"]
    if tracked:
        checks = ["tracked_keypoints", "robot_iou", "object_iou", "background"]
    report = {
        "accepted": False, "frames": 0, "tolerance_px": TOLERANCE_PX,
        "max_tool_error_px": None, "max_object_error_px": None, "issues": [],
        "min_robot_iou": None, "max_background_change": None,
        "checks": checks,
    }
    if not names or names != _frame_names(candidate):
        report["issues"].append("frame names or counts differ")
        return report

    if tracked:
        try:
            with Image.open(reference / names[0]) as image:
                size = image.size
            target_points = _keypoints(reference / "conditions" / "keypoints.json", names, size)
            seen_points = _keypoints(observations / "keypoints.json", names, size)
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["issues"].append(f"invalid tracker observations: {error}")
            return report
        errors = np.linalg.norm(target_points - seen_points, axis=2)
        worst = {"tool": float(errors[:, :3].max()), "object": float(errors[:, 3].max())}
    else:
        worst = {"tool": 0.0, "object": 0.0}
    minimum_iou, background_change = 1.0, 0.0
    object_iou = 1.0
    for name in names:
        try:
            expected = _image(reference / name, "RGB")
            actual = _image(candidate / name, "RGB")
            if expected.size != actual.size:
                raise ValueError("frame sizes differ")
            if tracked:
                iou, block_iou, changed = _tracked_geometry(
                    expected, actual, reference / "conditions" / "masks" / name, observations, name
                )
                minimum_iou = min(minimum_iou, iou)
                background_change = max(background_change, changed)
                object_iou = min(object_iou, block_iou)
            else:
                for marker, color in (("tool", TOOL_MARKER), ("object", OBJECT_MARKER)):
                    error = math.dist(_center(expected, color), _center(actual, color))
                    worst[marker] = max(worst[marker], error)
                if geometry:
                    iou, changed = _geometry(expected, actual)
                    minimum_iou = min(minimum_iou, iou)
                    background_change = max(background_change, changed)
        except (OSError, ValueError) as exc:
            report["issues"].append(f"{name}: {exc}")
            return report
        report["frames"] += 1
    for marker, error in worst.items():
        report[f"max_{marker}_error_px"] = round(error, 2)
        if error > TOLERANCE_PX:
            report["issues"].append(f"{marker} drift exceeds {TOLERANCE_PX} px")
    if tracked or geometry:
        report["min_robot_iou"] = round(minimum_iou, 4)
        report["max_background_change"] = round(background_change, 6)
        if minimum_iou < MIN_ROBOT_IOU:
            report["issues"].append(f"robot silhouette IoU below {MIN_ROBOT_IOU}")
        if background_change > (MAX_TRACKED_BACKGROUND_CHANGE if tracked else MAX_BACKGROUND_CHANGE):
            report["issues"].append("background changed outside the robot and object")
    if tracked:
        report["min_object_iou"] = round(object_iou, 4)
        if object_iou < 0.5:
            report["issues"].append("object silhouette IoU below 0.5")
    report["accepted"] = not report["issues"]
    return report
