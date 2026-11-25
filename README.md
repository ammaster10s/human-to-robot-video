# Human-to-robot video

A pick-and-place prototype for the Group 6 research proposal *Human-to-Robot
Video Translation for Scalable Robot Learning* (I664E Human-AI Interaction II).
It retargets tracked human motion to a simulated gripper, checks contact during
replay, and pairs accepted robot frames with action labels.

## Pipeline at a glance

| Proposal stage | Module | What it does here |
|---|---|---|
| 1. Hand and scene perception | `perception.py` | Tracks thumb, index finger and object markers in calibrated RGB-D frames |
| 2. Action retargeting | `retarget.py` | Finds grasp and release from fingertip spacing and plans gantry actions |
| 3. Human removal and robot render | `video.py` (`render`) | Composites the robot and writes masks, depth and keypoints |
| 4. Conditional video generation | `wan_input.py` | Packages control and mask clips for Wan2.1-VACE (optional, GPU) |
| 5. Verification and filtering | `run.py`, `video.py` (`verify`) | Physical replay checks and visual drift, IoU and background checks |
| 6. Policy learning and evaluation | `policy.py` | Goal-conditioned baseline scored on seen and held-out layouts |

`robot.py` and `gantry.xml` hold the MuJoCo replay. `grip.py` maps the phrases
`grip gently` and `grip firmly` to grip-force targets.

## Run it

```sh
uv sync --python 3.12
uv run --locked python run.py --compare --out runs/comparison
uv run --locked python run.py --prompt "grip gently" --out runs/single
uv run --locked python -m unittest discover -s tests
```

The video-packaging test is skipped when FFmpeg is not installed.

On Windows inside a synced folder such as OneDrive, keep the virtual
environment outside the folder so it is not uploaded:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\human-to-robot-video"
uv sync --python 3.12
```

### Example results

On the built-in synthetic demo (`run.py --compare`), the measured mean carry
force was about 2.0 N for `grip gently` and 5.0 N for `grip firmly`, with a
placement error below 0.2 mm in both trials. `policy.py` succeeded on 3 of 3
seen and 3 of 3 held-out layouts. These numbers come from one simulated gantry
and rigid block. They show that the pipeline runs end to end, not that it
transfers to real footage or hardware.

Open `runs/comparison/gentle/video/preview.gif` to view the replay. The
comparison runs the same motion and object twice. It reports mean measured grip
force during the carry and the final placement error under each instruction.

Each trial saves the planned `actions.json`, measured `replay.json`, a reference
and candidate PNG sequence, and `report.json` with the two sets of checks.
Accepted trials also save `dataset.json` with one image and action label per
sample. Exit status 2 means a verification check rejected the trial. Choose a
new output directory for each invocation.

## Use a tracked recording

Pass `--tracks tracks.json` with a JSON array of timestamped samples. A sample
looks like this:

```json
{"time": 0.0, "thumb": [-0.1, -0.05, 0.115], "index": [-0.1, 0.05, 0.115], "object_position": [-0.1, 0, 0.025]}
```

Time is in seconds and starts at zero. Clips can run for up to 60 seconds.
Coordinates are metres in the robot frame. Use one fixed-orientation
pick-and-place, a clear tabletop and a block 4 cm wide and 5 cm tall.
The track should start and end at rest, with enough
time for the jaws to close before lifting and to open before retreating.
The planner finds the pinch and release from fingertip spacing and validates
the tracked hand and object relationship during the carry.

For fixed-camera footage, supply `--frames-dir`, `--masks-dir`, and
`--background` together. Number the PNGs `frame_000000.png` onward in track
order. Use a clean background with the same dimensions and projection as the
source. White areas in each mask are replaced with the background before the
robot is drawn.

### Track a calibrated RGB-D sequence

```sh
uv run --locked python perception.py rgb/ depth/ camera.json tracks.json --fps 10
uv run --locked python run.py --tracks tracks.json --out runs/recording
```

Use aligned RGB and 16-bit depth PNGs with matching frame names. The CPU
tracker expects red on the thumb, green on the index finger, and blue on the
object. Camera JSON contains `fx`, `fy`, `cx`, `cy`, `depth_scale` in metres
per raw depth unit, and a rigid 4×4 `camera_to_robot` transform. The camera
axes are right, down and forward. For a top marker on the upright test
block, set `"object_center_offset": [0, 0, -0.025]` in robot coordinates.

## Render conditions and appearance editing

The renderer writes aligned robot masks, 16-bit depth PNGs, and wrist, jaw,
and object keypoints under `reference/conditions/` and `video/conditions/`.
The fixed schematic camera reports approximate depth in millimetres on the
robot and zero elsewhere. For compositing a recording, align the camera to
the render projection before supplying the source frames and masks.

To prepare a masked Wan2.1-VACE clip with FFmpeg:

```sh
uv run --locked python wan_input.py runs/single/reference --out runs/wan-input
```

The bundle contains aligned control and robot-mask MP4s, padded to a VACE
frame count with a manifest that maps back to the source. For inference, add
`--wan-repo /path/to/Wan2.1 --checkpoint /path/to/Wan2.1-VACE-1.3B` plus
`--python /path/to/wan-env/bin/python`, `--prompt "Your appearance description"`
and one or more `--seed` values. The optional runner uses the upstream
[Wan CLI](https://github.com/Wan-Video/Wan2.1#run-vace) in a prepared GPU
environment and saves one source-aligned PNG directory per seed.

Use `run.py --candidate-dir PATH` to review pre-edited frames at the
reference size. Repeat the flag to check alternatives in order. The report
records each checked attempt and accepts the first passing candidate when the
physical replay also passes. Rejected runs retain their reports.

For an unmarked candidate, supply independently detected points in
`observations/keypoints.json` using the same `frames`, `file`, and
`keypoints` fields as the render manifest. The point names are `wrist`,
`left_jaw`, `right_jaw`, and `object`. Supply matching binary PNGs under
`observations/robot_masks/` and `observations/object_masks/`. The verifier checks
point drift, robot and object overlap, and background outside those regions.
For built-in renders it derives the silhouette from the rendered palette.

## Grip and verification

The phrases `grip gently` and `grip firmly` select 2 N and 5 N per jaw in
MuJoCo. During bilateral contact, the controller adjusts jaw effort using
the measured normal force. Replay logs the mean across the two jaws. The
targets are for the simulated block. The language-to-force idea is inspired
by [Tacinf](https://arxiv.org/abs/2611.29240).

The physical check uses lift, contact and object drift during the carry, tool
error and final placement. The default video check measures tool and object
marker displacement from the reference projection.

## Policy baseline

```sh
uv run --locked python policy.py --out runs/policy
```

The CPU baseline fits goal-conditioned motion to accepted action labels. It
scores seen and held-out layouts in MuJoCo, using the supplied start and goal
positions to predict an open-loop trajectory. Pass repeated `--demo-run` flags
to train on other accepted runs with the same time grid.

## Scope and limitations

The runnable evaluation uses a single simulated gantry and rigid block.
Learned perception on unmarked human footage, per-robot LoRA training, GPU
generation validation, Diffusion Policy, and hardware evaluation remain
research work.
