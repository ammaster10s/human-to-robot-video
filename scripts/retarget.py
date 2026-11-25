"""Plan a fixed-orientation pick-and-place from tracked fingertips and an object."""

from math import dist, isfinite
from numbers import Real

from config import OBJECT_MAX_Z, OBJECT_WORKSPACE_XY, TABLE_Z

CLOSE_WIDTH = 0.055
OPEN_WIDTH = 0.075
GRASP_DISTANCE = 0.035
TRACK_DISTANCE = 0.065
REST_TOLERANCE = 0.012
CLEARANCE = 0.09


def _number(value, label):
    # numpy scalars from trackers are Real; bool is excluded on purpose.
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        if isfinite(value):
            return float(value)
    except OverflowError:
        pass
    raise ValueError(f"{label} must be a finite number")


def _point(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    return [_number(coordinate, label) for coordinate in value]


def plan(frames: list[dict]) -> list[dict]:
    """Map camera tracks in robot coordinates to timed gantry commands."""
    if not isinstance(frames, list) or len(frames) < 3:
        raise ValueError("expected a list of at least three tracked frames")

    samples = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {index} must be an object")
        time = _number(frame.get("time"), f"frame {index} time")
        if (index == 0 and time != 0) or (samples and time <= samples[-1][0]):
            raise ValueError("frame times must start at zero and strictly increase")

        thumb = _point(frame.get("thumb"), f"frame {index} thumb")
        finger = _point(frame.get("index"), f"frame {index} index")
        obj = _point(frame.get("object_position"), f"frame {index} object_position")
        x, y, z = obj
        if abs(x) > OBJECT_WORKSPACE_XY or abs(y) > OBJECT_WORKSPACE_XY or not TABLE_Z <= z <= OBJECT_MAX_Z:
            raise ValueError(f"frame {index} object is outside the gantry workspace")

        midpoint = [0.5 * a + 0.5 * b for a, b in zip(thumb, finger)]
        samples.append((time, obj, dist(thumb, finger), dist(midpoint, obj)))

    grasp = next(
        (i for i, (_, _, width, distance) in enumerate(samples)
         if width <= CLOSE_WIDTH and distance <= GRASP_DISTANCE),
        None,
    )
    if grasp is None or grasp == 0:
        raise ValueError("expected an open approach followed by a pinch near the object")

    release = next(
        (i for i in range(grasp + 1, len(samples)) if samples[i][2] >= OPEN_WIDTH),
        None,
    )
    if release is None or release == len(samples) - 1:
        raise ValueError("expected release followed by a retreat")

    pickup = samples[grasp][1]
    placement = samples[release][1]
    if pickup[2] > TABLE_Z + REST_TOLERANCE or placement[2] > TABLE_Z + REST_TOLERANCE:
        raise ValueError("pick up and release must be near the table")
    if any(dist(obj, pickup) > REST_TOLERANCE for _, obj, _, _ in samples[:grasp]):
        raise ValueError("object moved before the grasp")
    if any(dist(obj, placement) > REST_TOLERANCE for _, obj, _, _ in samples[release:]):
        raise ValueError("object moved after the release")

    carried = samples[grasp:release]
    if any(distance > TRACK_DISTANCE for _, _, _, distance in carried):
        raise ValueError("the tracked hand separated from the object during the carry")
    if max(obj[2] for _, obj, _, _ in carried) <= max(pickup[2], placement[2]) + REST_TOLERANCE:
        raise ValueError("expected the object to lift clear of the table")

    grasp_time = samples[grasp][0]
    release_time = samples[release][0]
    approach_duration = 0.8 * grasp_time
    actions = []
    # shortcut: vertical approach and retreat assume a clear tabletop, use collision planning for clutter.
    for i, (time, obj, _, _) in enumerate(samples):
        if i < grasp:
            x, y, z = pickup
            position = [x, y, z + CLEARANCE * max(0.0, 1.0 - time / approach_duration)]
        elif i < release:
            position = obj.copy()
        else:
            x, y, z = placement
            progress = min(1.0, max(0.0, (time - release_time - 0.3) / 0.6))
            position = [x, y, z + CLEARANCE * progress]
        actions.append({
            "time": time,
            "position": position,
            "closed": grasp <= i < release,
            "object_position": obj,
        })
    return actions


def demo_frames() -> list[dict]:
    frames = []
    for tick in range(54):
        time = tick / 10
        travel = min(1.0, max(0.0, (time - 1.5) / 2.1))
        x = -0.10 + 0.20 * travel**2 * (3 - 2 * travel)
        if time < 2.0:
            lift = min(1.0, max(0.0, (time - 1.4) / 0.6))
        else:
            lift = min(1.0, max(0.0, (3.8 - time) / 0.8))
        z = TABLE_Z + 0.115 * lift
        closed = 1.0 <= time < 4.3
        half_width = 0.021 if closed else 0.05
        hand_z = z + CLEARANCE * min(1.0, max(0.0, 1 - time, time - 4.6))
        frames.append({
            "time": time,
            "thumb": [x, -half_width, hand_z],
            "index": [x, half_width, hand_z],
            "object_position": [x, 0.0, z],
        })
    return frames
