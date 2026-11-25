"""Fixed-camera previews and pixel checks for a simulated replay."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageDraw


SIZE = (640, 420)
TOOL_MARKER = (255, 137, 50)
OBJECT_MARKER = (68, 232, 178)
MARKER_RADIUS = 4
TOLERANCE_PX = 8


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


def _box(draw, size, center, half, colors):
    x, y, z = center
    dx, dy, dz = half
    footprint = [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)]
    top = [project((u, v, z + dz), size) for u, v in footprint]
    base = [project((u, v, z - dz), size) for u, v in footprint]
    draw.polygon((base[0], base[1], top[1], top[0]), fill=colors[0])
    draw.polygon((base[1], base[2], top[2], top[1]), fill=colors[1])
    draw.polygon(top, fill=colors[2])


def _overlay(canvas, row):
    draw = ImageDraw.Draw(canvas)
    size = canvas.size
    x, y, z = row["position"]
    ox, oy, oz = row["object_position"]
    gap = row.get("gripper_width", 0.04 if row["closed"] else 0.082)
    wrist = (x, y, z + 0.047)
    arm = ("#63808f", "#4b626e", "#a9c0c9")
    fingers = ("#afbec7", "#657c88", "#d6e3e6")

    rail = [project((end, y, 0.30), size) for end in (-0.25, 0.25)]
    draw.line(rail, fill="#7d97a5", width=max(2, round(min(size) * 0.018)))
    shaft_height = 0.30 - wrist[2]
    if shaft_height > 0:
        _box(draw, size, (x, y, wrist[2] + shaft_height / 2), (0.011, 0.011, shaft_height / 2), arm)
    _box(draw, size, wrist, (gap / 2 + 0.016, 0.015, 0.008), arm)
    _box(draw, size, (x - gap / 2 - 0.007, y, z + 0.014), (0.007, 0.012, 0.029), fingers)
    _box(draw, size, (ox, oy, oz), (0.02, 0.02, 0.025), ("#146456", "#104741", "#328b76"))
    _box(draw, size, (x + gap / 2 + 0.007, y, z + 0.014), (0.007, 0.012, 0.029), fingers)

    for point, color in ((wrist, TOOL_MARKER), ((ox, oy - 0.02, oz - 0.012), OBJECT_MARKER)):
        u, v = project(point, size)
        r = MARKER_RADIUS
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color)


def _frame_names(directory):
    return sorted(path.name for path in directory.glob("frame_*.png"))


def _image(path, mode):
    with Image.open(path) as image:
        return image.convert(mode)


def render(rows, output: Path, *, frames_dir=None, masks_dir=None, background=None):
    """Write PNGs and a preview GIF to a new directory."""
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
        for i, (name, row) in enumerate(zip(names, rows)):
            canvas = base.copy()
            if frames_dir is not None:
                source = _image(frames_dir / name, "RGB")
                mask = _image(masks_dir / name, "L")
                if source.size != base.size or mask.size != base.size:
                    raise ValueError(f"{name}: source, mask and background sizes differ")
                canvas = Image.composite(base, source, mask)
            _overlay(canvas, row)
            canvas.save(staged / name)
            if i % preview_step == 0 or i == len(rows) - 1:
                canvas.thumbnail((480, 320))
                preview.append(canvas)
                preview_times.append(times[i])

        delays = [
            max(10, min(65_535 * 10, round((end - start) * 1000)))
            for start, end in zip(preview_times, preview_times[1:])
        ]
        delays.append(500)
        preview[0].save(staged / "preview.gif", save_all=True, append_images=preview[1:],
                        duration=delays, loop=0)
        staged.rename(output)


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


def verify(reference: Path, candidate: Path):
    """Compare projected tool and object positions in a pair of PNG sequences."""
    names = _frame_names(reference)
    report = {
        "accepted": False, "frames": 0, "tolerance_px": TOLERANCE_PX,
        "max_tool_error_px": None, "max_object_error_px": None, "issues": [],
    }
    if not names or names != _frame_names(candidate):
        report["issues"].append("frame names or counts differ")
        return report

    worst = {"tool": 0.0, "object": 0.0}
    for name in names:
        try:
            expected = _image(reference / name, "RGB")
            actual = _image(candidate / name, "RGB")
            if expected.size != actual.size:
                raise ValueError("frame sizes differ")
            for marker, color in (("tool", TOOL_MARKER), ("object", OBJECT_MARKER)):
                error = math.dist(_center(expected, color), _center(actual, color))
                worst[marker] = max(worst[marker], error)
        except (OSError, ValueError) as exc:
            report["issues"].append(f"{name}: {exc}")
            return report
        report["frames"] += 1
    for marker, error in worst.items():
        report[f"max_{marker}_error_px"] = round(error, 2)
        if error > TOLERANCE_PX:
            report["issues"].append(f"{marker} drift exceeds {TOLERANCE_PX} px")
    report["accepted"] = not report["issues"]
    return report
