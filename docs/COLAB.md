# Run Jumper in Google Colab

[Home](../README.md) · [中文](COLAB.zh.md) · [Notebook](../notebooks/jumper_colab.ipynb)
· [Quick start and training video](../notebooks/README.md)

Train on a Colab GPU, inspect progress, replay in MuJoCo, export ONNX and keep a
portable backup. Your computer only needs a browser.

[Open in Colab](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb),
save a copy and select a GPU runtime. The main link and defaults use upstream
`main`. Before merge, use the [PR preview](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb).
For new training, select `https://github.com/tianrking/jumper.git` and
`codex/colab-training-and-video` in the advanced source settings.

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

The backup's commit must exist in the selected repository. For the original
`19e8f4d` backup, use `tianrking/jumper`: an upstream squash merge may not retain
that commit. Core versions are read from the backup, without manual copying.
Manual overrides accept exact versions of `torch`, `mujoco`, `mujoco-warp`,
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
Checkpoint command ranges use the saved curriculum level instead of a later one;
the replay records resolved ranges, source and simulation rates. Command controls
apply to velocity tasks; unsupported tasks are refused.

With `RUN_EVALUATION`, forward, sideways, turn and stand each run for
`EVAL_SECONDS`, producing a JSON tracking/reset report. Compare requested commands
with measured velocity; this is not an automatic robot acceptance verdict.
The shared CLI exposes `--checkpoint-command-ranges`, `--command VX VY YAW`,
`--evaluation-out JSON` and `--replay-info-out JSON`; ordinary replay is unchanged
without them.

The notebook displays MP4 inline. `RECORD_JOINTS` also writes `measure.csv` and
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

## Validation boundary

The core workflow has been exercised in a real Google Colab session on an NVIDIA
Tesla T4, with Python 3.13, PyTorch 2.9.1+cu126, MuJoCo 3.11 and Warp 1.18.
These results cover the tested branch's core code, rather than guaranteeing every
future Colab image or GPU allocation:

The tested source (`19e8f4d`) completed the default 500-iteration T4 run, including
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

## Later workflow changes

The basic/advanced forms, early backup settings, retained restore selection,
curriculum summary and fixed-command evaluation were added after the `19e8f4d`
GPU run above. Their new Colab end-to-end validation is pending and is not implied
by the historical record. Local checks and new GPU receipts will be recorded here
when completed. Drive mounting remains untested.
