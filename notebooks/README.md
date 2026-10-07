# Jumper on Colab

[Open in Colab](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)
· [中文](README.zh.md) · [Full guide](../docs/COLAB.md)

Train Jumper on a Colab GPU, watch it in MuJoCo, and download the results.
Your computer only needs a browser; WSL and a local GPU are not needed.

The main link and notebook defaults use `KingKongRobotics/jumper` on `main`.
Before this PR is merged, open the [preview](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb) and set
`REPOSITORY_URL` to `https://github.com/tianrking/jumper.git` and
`SOURCE_REVISION` to `codex/colab-training-and-video` in the first form.

## Start here

1. Open the notebook, save a copy, and select a GPU runtime.
2. Choose the task and settings in the first form, then run the cells in order.
3. Watch the supplied dance, run the five-iteration smoke check, and start training.
4. Inspect the curves, replay your checkpoint, and download the export and backup.

Setup installs into the notebook's own environment and checks real CUDA and Warp
execution. The default main run trains `jumper.tripod` for 500 iterations with
256 parallel environments. The dance demonstration uses a supplied policy; your
training run produces a separate walking policy.

## This run's video

This is the walking replay from `model_499.pt`, produced by the 500-iteration
Colab run. It is ten seconds of real MuJoCo simulation at 640 x 480 and 30 fps,
not the supplied dance demonstration.

https://github.com/user-attachments/assets/e0e3b700-d1f6-486f-be20-98d4b69639ea

## Change the replay without training again

Set the simulation duration, FPS, even image dimensions, scene and camera distance
in the form. Run the settings cell, then the trained-policy replay cell; do not
rerun setup or training. Keep the task and model unchanged for this checkpoint.
Use `task default` to replay the task's own
world, or select another scene to try different terrain or friction. The notebook
shows the MP4 inline. These settings change the replay, not the saved policy.

With `RECORD_JOINTS` on, replay also saves a CSV and plot of joint position, velocity
and torque; tasks with contact sensors can include foot forces. These are measured
in the simulation, not on a physical robot.

## Files you can keep

| Output | Contents |
|---|---|
| Training curves | All recorded TensorBoard scalars in JSON and CSV, plus an inline six-panel PNG summary. |
| Replay | MP4, simulation settings, and optional joint CSV/PNG. |
| Policy export | ONNX actor, `layout.json`, README and checkpoint copy; each export uses a new folder. |
| Backup ZIP | Checkpoints, configuration, logs, curves, runtime/EGL reports and matching replay/export files. The tested 500-iteration backup was about 116 MiB. |

The files are kept per run and tied to the selected checkpoint, so an old video
or export is not silently included as a new result. Restore the ZIP in a later
session, use its recorded source and core package versions, and continue training.
Restoring the tested backup and running five more iterations succeeded. For that
backup, use `tianrking/jumper` at the recorded `19e8f4d` commit, not a newer `main`;
a squash merge may not retain the original commit. Use a fresh runtime or checkout
folder when switching repositories.

`DOWNLOAD_FILES` controls automatic browser downloads; turning it off leaves
previews and files available for manual download. Drive checkpoint snapshots are
optional. Drive mounting was disabled in the test and has not been verified.

## Tested on Colab

The workflow ran on a Tesla T4 using source `19e8f4d`: five smoke iterations,
500 main iterations, video replay, 500 rows of joint measurements, 51 scalar tags
with 25,500 samples, ONNX export, ZIP backup, restore and five further iterations.
Later notebook edits update imports, documentation, upstream defaults and the
export task/model check.

This demonstrates the workflow for the tested task. It does not establish that
every task works, that the walking policy has converged, or that it can run safely
on a physical robot. For settings and the test results, see the
[Colab guide](../docs/COLAB.md).
