# Run Jumper in Google Colab

[Home](../README.md) · [中文](COLAB.zh.md) · [Notebook](../notebooks/jumper_colab.ipynb)
· [Quick start and training video](../notebooks/README.md)

The notebook trains Jumper on a Colab GPU, shows the training curves, and replays
your policy in MuJoCo with video and joint measurements. You can change the replay
settings without training again, export ONNX, and save a backup for later resume.
The 500-iteration training and backup/restore workflow has run on a real Colab T4.
Your computer only needs a browser; no local installation, WSL or local GPU is needed.
The [quick start](../notebooks/README.md) includes the video from this training run.
That video shows simulation, not physical-robot performance.

Open [the notebook in Colab](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb),
save a copy if you want to keep your changes, and select a GPU runtime. Run the setup
and checks before choosing the demonstration or training cells. A GPU allocation is
variable: use the device and memory reported by the checks, rather than assuming a
particular GPU is available.

The main link and notebook defaults use `KingKongRobotics/jumper` on `main`.
Before merge, open the [PR preview](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb) and set
`REPOSITORY_URL` to `https://github.com/tianrking/jumper.git` and
`SOURCE_REVISION` to `codex/colab-training-and-video` in the first form.

## Choose the settings

The first form selects the task and training length; its main run defaults to
256 environments and 500 iterations. A separate five-iteration smoke check runs
before it. The same form controls the later simulation and downloads:

| Setting | How to use it |
|---|---|
| `TASK`, `NUM_ENVS`, `ITERATIONS` | Select the motion, parallel environment count and additional training iterations. |
| `SIM_SECONDS`, `SIM_FPS` | Simulated duration and recording frame rate; both must be positive and finite. |
| `SIM_WIDTH`, `SIM_HEIGHT` | Image dimensions, each an even integer of at least two. |
| `SIM_SCENE` | Keep `task default` to replay the task's world; another scene changes the test conditions, such as terrain or friction. |
| `SIM_CAMERA_DISTANCE` | Zero uses automatic framing; a positive value sets the camera distance in metres. |
| `RECORD_JOINTS` | Save and display simulation joint telemetry alongside the video. |
| `DOWNLOAD_FILES` | Enable automatic browser downloads; disabling it keeps inline previews and files available for manual download. |
| `CHECKOUT_FOLDER` | One simple folder name under `/content`; the default is `jumper`, and paths follow the chosen name. |
| `CORE_VERSION_OVERRIDES` | For resume, paste the saved manifest's `config.runtime_versions` JSON. |

Version overrides accept only exact versions of `torch`, `mujoco`, `mujoco-warp`,
`warp-lang`, `numpy` and `tensordict`. URLs, ranges and arbitrary pip arguments are
not accepted. The supported Torch/torchvision pair is 2.9.1/0.24.1; a different
pair requires its own validation. Also restore the manifest's source commit using
`SOURCE_REVISION`. The repository must contain that exact commit: if a squash
merge does not retain it upstream, restore the tested `19e8f4d` backup from
`https://github.com/tianrking/jumper.git`. Use a fresh runtime or a new
`CHECKOUT_FOLDER` when switching repositories. A matching package list still
needs a compatible GPU driver.

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

The Python environment is local to this checkout. With the default checkout folder,
notebook shell commands use `/content/jumper/.venv/bin/python`, so the kernel's preinstalled packages do
not silently replace the vendored training packages. Jumper requires Python
3.10 through 3.13 (`>=3.10,<3.14`). The setup should stop on an unsupported Python
version, an unavailable CUDA device or a GPU driver/backend mismatch; it should not
turn the requested GPU training into an unnoticed CPU run.

In the tested Colab session, `.venv` occupied about 7.2 GiB and the project about
580 MiB excluding the virtual environment and training logs. These are measured
disk sizes, not download traffic; allow additional space for checkpoints, videos
and temporary installation files.

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

After the main run, the training-curve cell reads the TensorBoard events from that
exact run. It exports the complete recorded scalar history to `metrics.json` and
`metrics.csv`, and displays `training-curves.png` with up to six selected curves.
The PNG is a summary; the JSON and CSV retain all recorded scalar tags and samples.
Non-finite values remain explicit `NaN`, `Infinity` or `-Infinity` strings in JSON
and CSV, with warning counts; the plot leaves gaps rather than inventing values.
Check reward, episode length and losses before interpreting the policy's progress.

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
The setup locates the runtime's existing NVIDIA EGL libraries and writes a vendor
JSON under the task's own directory. It does not replace the system GPU driver.

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

The trained-policy replay uses the selected checkpoint and computes its control
step count from `SIM_SECONDS` and the task's actual control rate. Its physics rate
matches the task, observation noise stays enabled, and the result is shown inline.
Colab does not open the desktop keyboard/gamepad viewer.

With `RECORD_JOINTS` enabled, replay adds the existing `--measure` option. It records
one sample per control step to `measure.csv`: simulated time in seconds, absolute
joint position in radians, velocity in radians/second and applied actuator torque
in N·m, with joint and actuator names. Contact-enabled tasks may also record foot
force in newtons. The notebook displays `measure.png` and keeps both files with
the run's attachments. These are simulation measurements, not hardware telemetry.

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

The final portable backup ZIP includes the selected run's complete saved checkpoints,
configuration and logs, plus a separate artifact directory for that run. Available
attachments include the runtime and EGL reports, full scalar JSON/CSV and curve PNG,
simulation settings, and joint CSV/PNG. A successful MP4 or ONNX export is included
only when it belongs to the selected checkpoint; rerunning cells must not pull in
an old checkpoint's video or export. If plotting, recording or export fails, use
the backup cell to preserve the complete checkpoints and attachments already saved.

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
The notebook selects the checkpoint explicitly and writes a new folder for each
export, so repeating the cell does not overwrite the preceding export. Its
`--no-video` option avoids recording the task's full motion a second time. When
`DOWNLOAD_FILES` is enabled, download cells offer the videos, joint CSV and final
backup ZIP; otherwise download the printed paths manually before the runtime ends.

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

The core workflow has been exercised in a real Google Colab session on an NVIDIA
Tesla T4, with Python 3.13, PyTorch 2.9.1+cu126, MuJoCo 3.11 and Warp 1.18.
These results cover the tested branch's core code, rather than guaranteeing every
future Colab image or GPU allocation:

The current source (`19e8f4d`) completed the default 500-iteration T4 run, including
joint telemetry, scalar exports and per-run artifact packaging. The confirmed
workflow and regression results are recorded below.

| Check | Observed result |
|---|---|
| GPU training smoke | Five PPO iterations with 256 `jumper.tripod` environments completed. |
| Main training | 500 iterations with 256 `jumper.tripod` environments completed and produced `model_499.pt`. |
| Restore | The ZIP was restored and training continued for five iterations from `model_499.pt`, saving `model_503.pt` with the same source and core package versions. |
| ONNX export comparison | The 411-input, 20-output actor passed comparison with a maximum absolute error of `7.15e-7`. |
| Supplied dance recording | A 500-step MP4 was generated with NVIDIA EGL on the actual T4 GPU and verified by local decoding and cloud `ffprobe`. |
| Trained-policy recording | A 500-step MP4 of the newly trained policy was generated with NVIDIA EGL on the actual T4 GPU and passed the same video checks. |
| Joint telemetry | 500 CSV rows include position, velocity, applied actuator torque and foot force; the corresponding PNG was generated. |
| Training curves | 51 scalar tags and 25,500 samples were exported, with a six-panel curve PNG. |
| Portable backup | A 121,887,490-byte ZIP passed CRC checks and contains the run's checkpoints and generated artifacts. |
| Replay parameter variations | A two-second, 320 x 240, 20 fps studio replay at 1.5 m camera distance succeeded with joint recording off, producing 41 H.264 frames. Restoring the default ten-second settings succeeded and produced a fresh 500-row joint recording. |
| Focused checks | All 104 passed in 6.89 seconds. |
| All test files | All 62 `test_*.py` files ran in 16 sequential batches, at most four files per pytest process. JUnit totals: 1123 tests, 1096 passed, 25 skipped, 2 failed, 0 errors; summed run time 354.651 seconds. |
| Google Drive mounting | Disabled in this session and not tested. |

The graphics check used NVIDIA EGL rather than Mesa software rendering. The setup
selected the existing NVIDIA graphics libraries through task-local vendor JSON;
it did not change the system driver. The backup/restore result verifies the local
snapshot mechanism, not Google Drive authorization or remote persistence.
Both recordings contain 301 H.264 frames at 640 x 480 and 30 fps, with a duration
of 10.033333 seconds. This includes the initial frame as well as frames sampled
during the ten-second simulation. With joint recording disabled, the preceding
measurement files were preserved under `replay-history` and excluded from that
replay's attachments, rather than being presented as new measurements.

The single-process full pytest run was terminated by the system with `SIGKILL`
(`-9`) at about 57% completion. The completed coverage therefore comes from the
16 bounded batches above, not a completed single-process suite. The batch totals
include two failures: posture mirror geometry and TensorBoard port selection with
a lingering `TIME_WAIT` socket. Both were reproduced on the upstream baseline in
the same Colab environment. Skipped tests are not passes.

A separate baseline sequence, running the claw operator test before the resolver
default-thread test, also reproduced the native thread-count discrepancy of two
versus eight. The resolver passed in the bounded branch run, so that baseline
sequence result is not counted as a third failure in the 1123-test batch total.
Separate processes do not establish that all tests coexist in one shared process.

These runs establish that the tested training, checkpoint, restore and
export paths work. They do not establish policy convergence, a robust gait or
real-robot behavior. The two batch failures, the single-process interruption and
the baseline sequence dependence limit the regression evidence. Google Drive
mounting was not tested.
