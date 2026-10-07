# Run Jumper in Google Colab

[Home](../README.md) · [中文](COLAB.zh.md) · [Notebook](../notebooks/jumper_colab.ipynb)

The notebook lets you watch a supplied dance policy, try a small training run, and save
the resulting video and policy files. You do not need to install Jumper on your computer.
The video shows the simulated robot; it does not demonstrate real-robot performance.

Open [the notebook in Colab](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb),
save a copy if you want to keep your changes, and select a GPU runtime. Run the setup
and checks before choosing the demonstration or training cells. A GPU allocation is
variable: use the device and memory reported by the checks, rather than assuming a
particular GPU is available.

This link currently opens the pre-merge `codex/colab-training-and-video` branch in
the `tianrking/jumper` fork. After the change is merged upstream, the entry can move
to `KingKongRobotics/jumper` on `main`; until then, use the branch link above.

## Watch the demonstration first

Run the notebook cells in order through the demonstration. They clone the repository
into the Colab machine, install it in its own `.venv`, check the available CUDA device
and MuJoCo Warp backend, and render about ten seconds of the supplied `jumper.dance`
checkpoint. The next cell displays the MP4 inside the notebook; a download cell lets
you save it to your computer.

The first Warp run can take longer because kernels have to compile. Installation,
compilation, simulation and video encoding are separate operations; a slow first run
does not tell you how fast the later training loop will be. Read the cell output if
a check fails, and fix that condition before running dependent cells.

The Python environment is local to this checkout. Notebook shell commands use
`/content/jumper/.venv/bin/python`, so the notebook kernel's preinstalled packages do
not silently replace the vendored training packages. Jumper requires Python
3.10 through 3.13 (`>=3.10,<3.14`). The setup should stop on an unsupported Python
version, an unavailable CUDA device or a GPU driver/backend mismatch; it should not
turn the requested GPU training into an unnoticed CPU run.

The supplied checkpoints are examples built with their recorded task and model
configuration. A newer source revision can change observations, actions or geometry.
If the loader refuses an example, retain that error and check its provenance rather
than changing tensor sizes or joint order to make it load. Replaying a packaged app's
prebuilt controller checks that artifact; it does not build the current controller
source or prove current physics compatibility.

## Try training

The first training run uses `jumper.tripod`, 256 parallel environments and five PPO
iterations. This is a smoke check of the GPU training path and checkpoint writing.
Five iterations are not enough to establish a good walking policy. The demonstration
video uses a supplied trained checkpoint and must not be mistaken for the outcome
of this short run.

Once the smoke run succeeds, change the notebook's training settings for a longer
run. Increase the environment count only within the allocated GPU's memory budget.
Environment count changes PPO's effective batch size, so it is a training choice
as well as a memory choice. Use the reward curves and replay of the resulting
checkpoint to judge progress; completing an iteration count is not an acceptance
criterion for a gait.

For reference, from the repository root the short run is:

```bash
.venv/bin/python scripts/train.py --task jumper.tripod \
  --backend warp --device cuda:0 --num_envs 256 --max-iterations 5 --headless
```

Runs are written to `logs/<model>/<task>/<timestamp>/`. Replay and export should use
the checkpoint selected by the notebook from that run, rather than an unrelated
newest checkpoint. See [Usage](USAGE.md) for the full training controls.

## Record a replay

The notebook uses the same video option as local replay. From the repository root:

```bash
.venv/bin/python scripts/play.py --task jumper.dance \
  --checkpoint tasks/jumper/dance/out/example/model_2300.pt \
  --backend warp --device cuda:0 --num_envs 1 --physics-hz 200 --headless --steps 500 \
  --video out/colab/dance.mp4 --video-fps 30 --video-width 640 --video-height 480
```

For this task, 500 control steps are ten simulated seconds. `--video-fps` selects
the recording frame rate; it does not change the policy's control rate. Other tasks
have different control periods, so the same step count need not produce the same
duration. `--physics-hz 200` keeps the dance replay's physics rate at the rate used
for training. Headless replay still needs working offscreen rendering. On the Colab
Linux GPU runtime the notebook requests EGL with `MUJOCO_GL=egl`.

| Option | Meaning |
|---|---|
| `--video PATH` | Write an MP4 to this path. |
| `--video-fps 30` | Frames per second in the recording. |
| `--video-width 640 --video-height 480` | Output image size in pixels. |
| `--video-env 0` | Environment to follow when replay has more than one. |
| `--video-distance 1.0` | Tracking camera distance in metres. |
| `--steps 500` | Number of control steps to replay before finishing. |

Invalid recording settings are refused rather than silently substituted. For a
failed recording, inspect the rendering or encoder error before trying a larger
image or a longer run. An MP4 that plays and an effective trained policy are
different results.

## Keep the checkpoint before the runtime ends

Treat the Colab machine's local files as temporary. Download what you need before
the runtime disconnects or is replaced. Saving the notebook itself does not save
the cloned repository, virtual environment or training checkpoints.

The optional Drive cells require you to choose and authorize Drive mounting. When
enabled, the training backup copies a snapshot after each completed checkpoint,
together with a manifest. Training, replay and export stay on the local runtime
disk. Checkpoint snapshots provide persistence without writing every training
update through the remote filesystem.

Keep the checkpoint together with the run's configuration and provenance: task,
repository commit, training settings, resolved backend/device and package versions.
The notebook's saved run metadata and snapshot manifest provide context for the
checkpoint. A policy file alone cannot establish which robot geometry and
observation layout it expects. Enable the optional checkpoint backup before a long
run if you want these snapshots; a lost runtime cannot copy files afterwards.

To continue in a later runtime, run setup and the GPU checks again, restore the
saved run to local storage, and check the recorded task, source revision and
configuration before selecting its checkpoint. Matching tensor shapes alone does
not prove matching geometry or joint semantics. The CLI restores optimizer and
iteration state as well as weights, and writes the continued run into a new
timestamped directory:

```bash
.venv/bin/python scripts/train.py --task jumper.tripod \
  --backend warp --device cuda:0 --num_envs 256 \
  --checkpoint /content/restored-run/model_4.pt --max-iterations 100 --headless
```

Replace the example checkpoint name with the saved file. `--checkpoint` implies
resume; `--max-iterations 100` means 100 additional iterations, not a total of 100.
If compatibility checks fail, use a matching revision/configuration or start a
new run. See [Usage](USAGE.md) and [agent setup](AGENT_SETUP.md) for diagnostics.

## Export and download

The export cell writes an ONNX actor and its `layout.json` contract, README and
checkpoint copy. Keep that directory together: the contract carries observation
and action ordering, scales, gains and other information a consumer needs.

```bash
.venv/bin/python scripts/export.py --task jumper.tripod \
  --checkpoint /content/restored-run/model_4.pt --backend warp --device cuda:0
```

An export that passes its checks establishes the tested host-side conversion. It
does not establish RKNN inference, motor-bus timing or safe physical behavior on a
robot. This notebook's first workflow ends at downloadable simulation and policy
artifacts. Board conversion and deployment follow the separate
[deployment procedure](../deploy/README.md).

## Validation boundary

Real Google Colab validation is in progress; the complete workflow has not yet
passed that acceptance run. Local source checks and replay of an existing bundle
are narrower evidence. Installation
on an allocated Colab GPU, Warp execution, offscreen MP4 rendering, a training run,
export and persistence/resume each need their own successful run before claiming
the full notebook works there.
