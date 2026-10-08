# Jumper on Colab

[Open in Colab](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)
· [中文](README.zh.md) · [Full guide](../docs/COLAB.md)

Train with a Colab GPU, inspect the curves, replay the policy in MuJoCo, and
download the results. Your computer only needs a browser.

Before merge, use the [PR preview](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb).
For new training, set the advanced source settings to `https://github.com/tianrking/jumper.git`
and `codex/colab-training-and-video`. The main link and defaults use the upstream `main`.

## Start here

Save a notebook copy and select a GPU runtime. Choose one route:

- **New training:** choose task, environment count and iterations → setup and GPU checks → train → curves, replay, export and backup.
- **Continue training:** choose an uploaded ZIP or Drive checkpoint → read its saved settings → setup and GPU checks → restore and train more → download the new backup.

The default new run uses `jumper.tripod`, 256 environments and 500 iterations.
On resume, iterations means additional updates. The training cell prints the
mode, task, environment count and checkpoint before starting. Source and core
versions for resume come from the backup; changing replay settings does not clear
the restored checkpoint. The supplied dance and short smoke check are optional
and separate from your training run.

## What you can do

| Function | Result |
|---|---|
| Train or resume | Checkpoints, configuration and logs. |
| Inspect progress | Full scalar JSON/CSV, plots and available curriculum levels, tracking error and command ranges. |
| Replay | Inline MP4; change duration, FPS, size, scene and camera without retraining. Optional joint position, velocity, torque and foot-force CSV/PNG. |
| Evaluate a velocity policy | Fixed forward, sideways, turn and stand commands, with an optional checkpoint curriculum range and a JSON report. |
| Export | ONNX actor, `layout.json`, README and checkpoint copy in a new folder. |
| Keep a backup | ZIP of complete checkpoints and available outputs. A failed plot, replay or export does not prevent checkpoint backup. |

Set advanced replay options, then run the replay cell again. Keep the task and
model compatible with the checkpoint. Measurements and videos describe simulation;
ONNX export is followed by the separate [robot deployment procedure](../deploy/README.md).

Download the ZIP before the runtime ends. Saving the notebook does not save the
model. Optional Drive snapshots copy completed checkpoints during training;
Drive mounting has not been tested. `DOWNLOAD_FILES` turns automatic downloads on
or off; previews and manual downloads remain available.

## Video and test record

This ten-second replay is from `model_499.pt`, trained for 500 iterations on a
real Colab T4 at source `19e8f4d`. It is the newly trained walking policy, not the
supplied dance. The run remained at curriculum level 0; completion does not mean
the gait has converged.

https://github.com/user-attachments/assets/e0e3b700-d1f6-486f-be20-98d4b69639ea

That session completed training, video and joint recording, scalar export, ONNX,
ZIP backup, restore and five further iterations. Its ZIP was about 116 MiB. To
restore this older backup, its recorded `19e8f4d` commit remains in
`tianrking/jumper`; a squash merge may not retain it upstream.

The reorganized forms, early backup setup and fixed-command evaluation are later
changes; their Colab acceptance is recorded separately in the
[full guide](../docs/COLAB.md#validation-boundary).
