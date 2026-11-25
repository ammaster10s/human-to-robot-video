"""Replay Cartesian pick-and-place actions on a simulated parallel gripper."""

from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).with_name("gantry.xml")
OPEN_JAW_POSITION = 0.035
MAX_JAW_EFFORT = 10.0
CONTACT_THRESHOLD = 1e-3


def replay(actions: list[dict], grip_force: float) -> list[dict]:
    """Sample actual poses and mean bilateral normal force in N per jaw.

    Each open/close command takes effect from its timestamp onward.
    """
    if not 1 <= grip_force <= MAX_JAW_EFFORT:
        raise ValueError("Simulated grip force must be between 1 and 10 N per jaw.")
    if not actions:
        raise ValueError("Replay requires at least one action.")

    times = np.asarray([action["time"] for action in actions], dtype=float)
    positions = np.asarray([action["position"] for action in actions], dtype=float)
    start_object = np.asarray(actions[0]["object_position"], dtype=float)
    if times.shape != (len(actions),) or not np.isfinite(times).all():
        raise ValueError("Action times must be finite numbers.")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Action times must start at zero and strictly increase.")
    if times[-1] > 60:
        raise ValueError("Replay accepts clips of at most 60 seconds.")
    if positions.shape != (len(actions), 3) or not np.isfinite(positions).all():
        raise ValueError("Each gripper position must have three finite coordinates.")
    if np.any(positions < (-0.18, -0.18, 0.025)) or np.any(positions > (0.18, 0.18, 0.25)):
        raise ValueError("Gripper motion exceeds the simulated gantry workspace.")
    if start_object.shape != (3,) or not np.isfinite(start_object).all():
        raise ValueError("The initial object position must have three finite coordinates.")
    if any(type(action["closed"]) is not bool for action in actions):
        raise ValueError("Gripper commands must be booleans.")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    axes = [model.actuator(f"move_{axis}").id for axis in "xyz"]
    jaws = [model.actuator(f"grip_{side}").id for side in ("left", "right")]
    finger_qpos = [model.joint(side).qposadr.item() for side in ("left", "right")]
    object_start = model.joint("object_pose").qposadr.item()
    pad_ids = (model.geom("left_pad").id, model.geom("right_pad").id)
    object_id = model.geom("block").id

    data.qpos[:3] = positions[0]
    data.qpos[finger_qpos] = OPEN_JAW_POSITION
    data.qpos[object_start : object_start + 3] = start_object
    data.ctrl[axes] = positions[0]
    data.ctrl[jaws] = OPEN_JAW_POSITION
    dt = model.opt.timestep
    effort = grip_force
    min_effort = 0.75 * grip_force
    max_effort = min(MAX_JAW_EFFORT, 1.25 * grip_force)

    samples = []
    for i, (action, time, position) in enumerate(zip(actions, times, positions, strict=True)):
        if i:
            duration = time - times[i - 1]
            steps = int(np.ceil(duration / dt))
            step_time = duration / steps
            model.opt.timestep = step_time
            was_closed = actions[i - 1]["closed"]
            data.ctrl[jaws] = 0.0 if was_closed else OPEN_JAW_POSITION
            if not was_closed:
                effort = grip_force
            start = positions[i - 1]
            for step in range(1, steps + 1):
                limit = effort if was_closed else MAX_JAW_EFFORT
                model.actuator_forcerange[jaws] = (-limit, limit)
                data.ctrl[axes] = start + (position - start) * (step / steps)
                mujoco.mj_step(model, data)
                if was_closed:
                    measured = _grip_force(model, data, pad_ids, object_id)
                    # Integrate slowly so contact impulses do not trigger a large correction.
                    if measured:
                        effort += step_time * (grip_force - measured)
                        effort = min(max_effort, max(min_effort, effort))
                    else:
                        effort = grip_force

        mujoco.mj_forward(model, data)
        measured = _grip_force(model, data, pad_ids, object_id)
        samples.append({
            "time": float(time),
            "position": data.body("gripper").xpos.tolist(),
            "object_position": data.body("object").xpos.tolist(),
            "closed": action["closed"],
            "gripper_width": float(0.012 + data.qpos[finger_qpos].sum()),
            "grip_force": measured,
            "contact": measured > 0.0,
        })
    return samples


def _grip_force(model, data, pad_ids, object_id):
    forces = dict.fromkeys(pad_ids, 0.0)
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact):
        if contact.geom1 == object_id:
            pad = contact.geom2
        elif contact.geom2 == object_id:
            pad = contact.geom1
        else:
            continue
        if pad in forces:
            mujoco.mj_contactForce(model, data, i, wrench)
            forces[pad] += float(wrench[0])
    left, right = forces.values()
    return (left + right) / 2 if min(left, right) > CONTACT_THRESHOLD else 0.0
