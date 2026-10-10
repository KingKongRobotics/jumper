# Run Jumper in Google Colab

[Home](../README.md) · [中文](COLAB.zh.md) · [Notebook](../notebooks/jumper_colab.ipynb)
· [Quick start and training video](../notebooks/README.md)

Train on a Colab GPU, inspect progress, replay in MuJoCo, export ONNX and keep a
portable backup. Your computer only needs a browser.

[Open in Colab](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb),
save a copy and select a GPU runtime. The link and the notebook's defaults use
this repository's `main`; the advanced source settings take another repository
or revision.

## New training or continue training

- **New training:** basic settings → setup and GPU checks → train → curves, replay, export and backup.
- **Continue training:** select backup → read saved settings → setup and GPU checks → restore and train more → download a new backup.

The basic form contains `WORKFLOW_MODE`, `TASK`, `NUM_ENVS`, `ITERATIONS`,
`USE_DRIVE` and `DOWNLOAD_FILES`. New training defaults to `jumper.tripod`,
256 environments and 500 iterations. Iterations always means updates to perform
in this run, including additional updates when continuing.

For continuation, choose `Upload ZIP` or `Drive checkpoint`. The manifest is read
before setup to match the task, model, environment count, source commit and core
package versions. A missing manifest or incompatible choice stops the run. The
restored checkpoint is kept separately from the forms: rerunning settings or
changing the camera does not clear it. To start over deliberately, choose
`New training` and set `START_NEW_RUN`.

Before training, the cell prints the mode, task, environment count, iteration
count and checkpoint. Resume restores the optimizer, learning rate, observation
normalization, iteration counter and saved curriculum. Physical environment and
random-number states are not saved; it resumes learning, not an identical trajectory.

The supplied dance (`RUN_DEMO`) and five-iteration smoke check (`RUN_SMOKE`) are
optional. The smoke check uses the selected task and at most 256 environments;
it is a separate run, never implicitly selected for the main training. Neither
the supplied dance nor a successful smoke check establishes your policy's quality.

## Source and runtime

Advanced source settings contain `REPOSITORY_URL`, `SOURCE_REVISION`,
`CHECKOUT_FOLDER` and `CORE_VERSION_OVERRIDES`. The checkout is a simple directory
under `/content`, defaulting to `jumper`. Existing checkouts are reused only when
origin and commit match; switching sources needs a new folder or runtime.

The backup's commit must exist in the selected repository: a backup records the
repository, commit and core package versions it was made with, and continuation
checks out that commit before restoring, so a backup made from a fork needs that
fork in `REPOSITORY_URL`. Core versions are read from the backup without manual
copying. Manual overrides accept exact versions of `torch`, `mujoco`, `mujoco-warp`,
`warp-lang`, `numpy` and `tensordict`, not URLs, ranges or pip arguments. The
supported Torch/torchvision pair is 2.9.1/0.24.1; other pairs need validation.

Setup installs into the checkout's isolated `.venv`, without changing kernel
packages. Jumper requires Python 3.10–3.13. Official Torch wheels use CUDA 12.6
for T4 and CUDA 12.8 for supported newer GPUs. Actual CUDA operations, a compiled
Warp kernel and NVIDIA EGL rendering must succeed. Missing GPU, unsupported
Python or driver mismatch stops setup. The graphics check selects existing NVIDIA
libraries with task-local vendor JSON; it does not replace the system driver.

Colab decides GPU allocation and session duration. Check the reported device and
memory before increasing `NUM_ENVS`; it also changes PPO batch size. The original
T4 environment occupied about 7.2 GiB, and the repository about 580 MiB excluding
that environment and logs. These are disk sizes, not download traffic; allow
extra space for checkpoints, installation files and videos.

## Inspect progress

Runs use `logs/<model>/<task>/<timestamp>/`. Curves read the selected run's events.
`metrics.json` and `metrics.csv` keep every recorded scalar tag and sample; the
inline PNG is a summary. Non-finite values remain explicit strings with warning
counts, and leave gaps in plots.

For tasks that report them, the summary includes command curriculum `level`,
tracking error `lin_err`, promotion threshold `lin_err_bar`, and linear/angular
command ranges. The original `model_499.pt` stayed at level 0: final linear
tracking error was about 0.195, above the 0.105 promotion threshold. The video
demonstrates the workflow, not a converged gait. Inspect reward, episode length,
losses, curriculum and replay together; 500 completed iterations alone is not
acceptance. Training samples noisy actions; replay uses the actor's deterministic
mean, while observations and resets can still be noisy or randomized.

## Replay and evaluate

Advanced replay settings select duration, FPS, even image dimensions, scene,
camera distance and joint recording. Run those settings, then replay again;
there is no need to rerun setup or training. Task and model must match the
checkpoint. `task default` retains the task's world; another scene changes
terrain or friction. Zero camera distance uses automatic framing. Recording FPS
does not change control speed; control and physics rates follow the task.

`SIM_COMMAND` selects sampled commands, forward, sideways, turn, stand or custom
body-frame velocity. Fixed commands must lie within the saved training ranges.
Notebook replay and evaluation match the saved curriculum command ranges by
default, using the checkpoint's level instead of a later one;
the replay records resolved ranges, source and simulation rates. Command controls
apply to velocity tasks; unsupported tasks are refused.

With `RUN_EVALUATION`, forward, sideways, turn and stand each run for
`EVAL_SECONDS`, producing a JSON tracking/reset report. Compare requested commands
with measured velocity; this is not an automatic robot acceptance verdict.
The shared CLI exposes `--checkpoint-command-ranges`, `--command VX VY YAW`,
`--evaluation-out JSON` and `--replay-info-out JSON`; ordinary replay is unchanged
without them.

The replay is `scripts/play.py --video`, the repository's one MP4 recorder, so
what the notebook shows is what `play` shows on a desktop; [Usage](USAGE.md) has
its camera switches. The notebook displays the MP4 inline. `RECORD_JOINTS` also writes `measure.csv` and
a PNG: one sample per control step, with time, absolute joint position (rad),
velocity (rad/s), applied actuator torque (N·m) and available foot force (N).
These are simulation measurements. Colab does not open a desktop keyboard/gamepad
viewer. The [quick start](../notebooks/README.md) includes the original video.

## Export, backup and interruption

Export checks task and model, then writes a new folder with ONNX actor,
`layout.json`, README and checkpoint copy. Keep it together: the contract carries
observation/action order, scales and gains. Board conversion and physical use
follow the separate [deployment procedure](../deploy/README.md).

Download the backup before the runtime disappears. The ZIP contains complete
checkpoints, configuration and logs, plus available curves, reports, measurements
and matching replay/export. Missing runtime/EGL reports or failed plots, replay
or export do not prevent checkpoint backup. Another checkpoint's old results are
excluded. After interruption, continue from the last complete save; subsequent
unsaved updates are lost. Saving the notebook does not save the model.

`USE_DRIVE` optionally mounts your Drive and copies each completed checkpoint
with its manifest, while training remains on local runtime disk. Drive mounting
requires your authorization and has not been tested. Enable snapshots before a
long run; a final ZIP cannot protect a deleted runtime. With Drive off, download
the ZIP yourself. `DOWNLOAD_FILES` controls automatic downloads; previews and
manual downloads remain available when it is off.

In a fresh session, choose `Continue training` and read the backup before setup.
Restore does not replace existing runs and rejects unsafe ZIP entries, incomplete
checkpoints and incompatible settings. Continuation writes a new timestamped run.
CLI controls and diagnostics are in [Usage](USAGE.md) and [setup](AGENT_SETUP.md).

## What was validated

The workflow was exercised in real Google Colab sessions on an NVIDIA Tesla T4
with Python 3.13, PyTorch 2.9.1+cu126, MuJoCo 3.11 and Warp 1.18, on the code
this notebook was merged from. That establishes that the paths work, not that
every future Colab image or GPU allocation behaves the same, and not that a
policy trained this way is any good: the 500-iteration `jumper.tripod` run whose
video the [quick start](../notebooks/README.md) shows stayed at curriculum level
0, with a linear tracking error of 0.195 against a promotion threshold of 0.105.

What ran, end to end: a smoke run and a 500-iteration run on 256 environments;
ZIP backup, restore and five further iterations with the same source and core
versions; a 64-environment run continued from an uploaded ZIP; a real `SIGINT`
partway through a run, backed up, restored and trained further; ONNX export with
a maximum absolute error below `1e-6` against the torch actor; MP4 replays through
NVIDIA EGL, with joint telemetry; the four fixed-command evaluations; training
curves with every scalar tag exported. The repository's test suite was run in the
same environment in batches of files; two failures reproduced on the baseline
without this work (a posture mirror geometry check, and TensorBoard's port
selection against a lingering `TIME_WAIT` socket).

Google Drive mounting has not been tested. The ZIP upload was exercised with real
bytes at the notebook's upload boundary, not through the browser's file picker.
