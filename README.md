# Human-to-robot video

A pick-and-place prototype based on the research proposal. It retargets tracked
human motion to a simulated gripper, checks contact during replay, and pairs
accepted robot frames with action labels.

## Run it

```sh
uv sync --python 3.12
uv run --locked python run.py --compare --out runs/comparison
uv run --locked python run.py --prompt "grip gently" --out runs/single
uv run --locked python -m unittest discover -s tests
```

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
pick-and-place, a clear tabletop and a block
4 cm wide and 5 cm tall. The track should start and end at rest, with enough
time for the jaws to close before lifting and to open before retreating.
The planner finds the pinch and release from fingertip spacing and validates
the tracked hand and object relationship during the carry.

For fixed-camera footage, supply `--frames-dir`, `--masks-dir`, and
`--background` together. Number the PNGs `frame_000000.png` onward in track
order. Use a clean background with the same dimensions and projection as the
source. White areas in each mask are replaced with the background before the
robot is drawn.

`--candidate-dir` checks a pre-edited 640×420 PNG sequence for a single grip
setting. It expects the reference frame names and the renderer's tool and
object markers. Use a pose and object tracker for unmarked model output.

## Grip and verification

The phrases `grip gently` and `grip firmly` select 2 N and 5 N per jaw in
MuJoCo. During bilateral contact, the controller adjusts jaw effort using
the measured normal force. Replay logs the mean across the two jaws. The
targets are for the simulated block. The language-to-force idea is inspired
by [Tacinf](https://arxiv.org/abs/2611.29240).

The physical check uses lift, contact and object drift during the carry, tool
error and final placement. The video check measures tool and object marker
displacement from the reference projection. The demo builds paired training
data and verifies it through replay. Pose extraction, generative video,
learned-policy training and hardware validation are outside this prototype.
