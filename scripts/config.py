"""Shared scene geometry and limits for the simulated tabletop gantry.

`models/gantry.xml` cannot import these values, so `tests/test_config.py` checks that
the MuJoCo model and this module stay in sync.
"""

# Test block: 4 cm wide, 5 cm tall, resting on the table plane at z = 0.
BLOCK_HALF_EXTENTS = (0.02, 0.02, 0.025)
TABLE_Z = BLOCK_HALF_EXTENTS[2]  # block centre height when it rests on the table

# Tracked object positions accepted by the planner (metres, robot frame).
OBJECT_WORKSPACE_XY = 0.17
OBJECT_MAX_Z = 0.24

# Commanded tool positions accepted by the replay.
TOOL_WORKSPACE_XY = 0.18
TOOL_MIN_Z = TABLE_Z
TOOL_MAX_Z = 0.25

# Gantry joint limits in gantry.xml; the limits above must fit inside them.
GANTRY_JOINT_XY = 0.20
GANTRY_JOINT_Z = (0.018, 0.28)

# Finger joint travel in gantry.xml, used as the fully open jaw command.
OPEN_JAW_POSITION = 0.035
