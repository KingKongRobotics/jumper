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

The core workflow has been exercised in a real Google Colab session on an NVIDIA
Tesla T4, with Python 3.13, PyTorch 2.9.1+cu126, MuJoCo 3.11 and Warp 1.18.
These results cover the tested branch's core code, rather than guaranteeing every
future Colab image or GPU allocation:

| Check | Observed result |
|---|---|
| Focused tests | 76 passed: 57 Colab/video checks and 19 translation checks. |
| GPU training smoke | Five PPO iterations with 256 `jumper.tripod` environments completed. |
| Training and restore | A 20-iteration run completed; its checkpoint snapshot and manifest were backed up, restored into a fresh local destination, and used for five additional iterations. |
| ONNX export | The 411-input, 20-output actor passed comparison with a maximum absolute error of `3.58e-7`. |
| Supplied dance recording | A 500-step MP4 was generated with NVIDIA EGL on the actual T4 GPU and verified by local decoding and cloud `ffprobe`. |
| Trained-policy recording | A 500-step MP4 of the newly trained policy was generated with NVIDIA EGL on the actual T4 GPU and passed the same video checks. |
| Complete test suite | The sequential run finished with 1067 passed, 25 skipped, 3 failed and 140 warnings in 239.48 seconds. |
| Google Drive mounting | Disabled in this session and not tested. |

The graphics check used NVIDIA EGL rather than Mesa software rendering. The setup
selected the existing NVIDIA graphics libraries through task-local vendor JSON;
it did not change the system driver. The backup/restore result verifies the local
snapshot mechanism, not Google Drive authorization or remote persistence.
Both recordings contain 301 H.264 frames at 640 x 480 and 30 fps, with a duration
of 10.033333 seconds. This includes the initial frame as well as frames sampled
during the ten-second simulation.

The full suite is not clean. Its three failures are in existing tests outside the
Colab/video changes: posture mirror geometry, agreement on the native backend's
automatic thread count, and TensorBoard port selection with a lingering `TIME_WAIT`
socket. The relevant core code and tests were not changed. All three failures were
reproduced on the upstream baseline in the same Colab environment. The thread-count
failure was reproduced in order with
`test_five_foot.py::test_the_operator_actually_closes_the_claw` followed by
`test_resolve.py::test_backend_and_resolve_agree_on_the_default`: the resolver
reported two threads where the test expected eight. This records the sequence
dependence rather than diagnosing its underlying cause. The 25 skips are not passes.

Ruff 0.16.10 reported 346 findings across the repository, compared with 350 on the
upstream baseline; comparison found no added findings. On the changed Python files
it reported only the existing `EXE001` finding for `scripts/play.py`; that file's
shebang and executable mode were unchanged. This is not a claim that the
repository-wide lint check passes.

These short runs establish that the tested training, checkpoint, restore and
export paths work. They do not establish policy convergence, a robust gait or
real-robot behavior. The full-suite failures remain known limitations of this
validation, and Google Drive mounting was not tested.
