---
name: train
description: Train a policy in this repository and report what was measured -- pick the backend and the environment count from the machine in front of you, run `scripts/train.py`, read the numbers that say whether it is learning, replay the checkpoint with `scripts/play.py --measure`, and hand over a checkpoint path with measured tracking rather than an assumed success. Use this whenever someone asks for training or for a result from it: "训练一个三足步态", "train tripod", "把 jumper.jump 跑到收敛", "how is the run doing", "is this checkpoint any good", "continue the run from last night", or the README's one-sentence prompt. Reach for it before typing a `--num_envs` copied from a tutorial, because the failures are quiet -- a GPU number on a CPU machine exhausts memory an hour in, a run that "finished" with a flat tracking reward looks like one that learnt, a replay with observation noise off flatters every policy, and a policy from the native backend is reported as if it were the warp one.
---

# Training a policy

`docs/USAGE.md` is the manual for the entry points, `docs/TUTORIAL.md` walks one policy from
training to a bundle, and `docs/DESIGN.md` §9.2 says why CPU and GPU are not the same run.
**This skill does not restate them.** It gives the order, the numbers that were measured
here, and the two places where a run goes wrong without saying so.

## The one thing to understand first

**A finished run is not a result; a measured replay is.** `train.py` stops when its
iteration count is up whether or not the policy learnt anything, and nothing in its last
screen says which. The result of this skill is three things stated together: the checkpoint
path, the backend and environment count it was trained with, and what `play --measure` read
off it. Leave one out and the next person has a file and a guess.

## Procedure

### 1. Measure the machine before choosing anything

```bash
python3 .claude/skills/setup-env/scripts/detect.py
```

It says which backend this machine has. **`warp:cuda` on an NVIDIA machine, `native:cpu`
everywhere else**, including all of macOS -- there is no third answer, and `--backend auto`
(the default) picks the same one. If the environment has never passed `gates.py`, that is
the `setup-env` skill, first.

### 2. Size the run for the backend, not from a page

On the native backend every environment holds its own `MjModel` and `MjData`, memory grows
with `--num_envs`, and the thread pool is capped at 8 (`--cpu_threads`). Measured on
`jumper.tripod`, 3 iterations, M3 Max (16 cores, 64 GB), 2026-10-08:

| `--num_envs` | env-steps / s | iteration | note |
|---|---|---|---|
| 16 | 454 | 0.84 s | enough to see it run, not to learn |
| 64 | 1777 | 0.86 s | the knee: four times the batch for the same wall time |
| 256 | 1850 | 3.3 s | the pool is saturated; more environments only lengthen each iteration |

So on a CPU machine start at 64, and raise it only when memory says there is room and the
step rate has not flattened. On a GPU, 4096 is the tutorial's number and memory is the only
ceiling. **A policy trained at 64 environments is not the policy the GPU recipe produces**:
the effective batch is two orders of magnitude smaller, and `rl_cfg.py`'s
`num_steps_per_env` / learning rate were chosen for the large batch (DESIGN §9.2). Say which
backend the checkpoint came from every time it is mentioned.

Resolve first, then run:

```bash
python scripts/train.py --task jumper.tripod --dry-run                 # the arguments, no build
python scripts/train.py --task jumper.tripod --backend native --device cpu --num_envs 64 --headless
```

`--headless` when nobody is watching; on macOS the live viewer needs `.venv/bin/mjpython`
and without it the run goes headless on its own and says so. TensorBoard starts with the
run unless `--no-tensorboard`; the log directory is printed as `[mjrl] log directory ...`
and is `logs/<model>/<task>/<timestamp>/`.

### 3. Read the run, not the clock

Every iteration prints a block. The lines that say whether it is learning:

- **`Mean reward`** rising, and for the locomotion tasks the **tracking** terms in
  `Episode_Reward/` -- `track_lin_vel_xy_exp` and `track_ang_vel_z_exp` -- climbing toward
  their weights. Flat tracking with a rising mean reward is a policy that found the
  stand-still or the regularisers and not the command.
- **`Mean action std`** falling from 1.0. A std that never moves is a learning rate or a
  batch that is too small for the signal.
- **`Metrics/`** -- the task's own: `slip_velocity_mean` down, `peak_height_mean` up for
  the gaits.
- **the curriculum level**, where the task has one (`Curriculum/`): a ladder that never
  climbs past level 0 in thousands of iterations is a command range the policy cannot reach.

`--resume` continues the newest run, `--checkpoint <path>` a particular one, and
`--max-iterations N` under either is **N more**, not N in total (USAGE "Resuming training").

### 4. Replay with the measurement on

```bash
python scripts/play.py --task jumper.tripod --headless --measure --speed 0
```

With no `--checkpoint` it takes the newest. `--measure` writes `measure.csv` and
`measure.png` into `measure/` beside the checkpoint: joint positions, velocities and
actuator torques per control step, which is the evidence a report quotes. Observation
noise stays **on** by default, as in training -- `--no-obs-noise` is the clean signal and
flatters the policy, so compare the two rather than reporting only the second. On a
macOS desktop the same command without `--headless` under `.venv/bin/mjpython` shows it.

A replay in a Design map (`--scene <file>.map`) is replay-only; `train.py` refuses maps,
and that is the documented boundary, not a bug to work around.

### 5. Report what was measured

The handover is one block, and every number in it came from a run:

```
checkpoint   logs/jumper/jumper.tripod/<timestamp>/model_<N>.pt
backend      native:cpu, 64 envs, <N> iterations, <wall time>   (or warp:cuda, 4096 envs)
training     mean reward <a> -> <b>; tracking lin <x>, ang <y> of weights <wx>, <wy>; level <L>
replay       <seconds> s, obs noise on: <what the robot did>; measure/ at <path>
```

"Training took time and compute; here is the checkpoint" is not a report. Neither is a
reward number with no backend beside it. When the replay was not run, say so.

### 6. From here

Exporting the policy and packaging it is the `deploy` skill, and it begins with
`scripts/export.py` on the checkpoint this skill ends with. A new task is the `new-task`
skill, before any of the above.

## Failures worth recognising

| symptom | likely cause |
|---|---|
| `MemoryError`, or the machine swaps, an hour in | a GPU `--num_envs` on the native backend; each environment is a model copy |
| warp reports `['cpu']` alone on an NVIDIA machine | the environment, not the run: `setup-env`, gate A |
| the reward rises and the tracking terms do not | the policy found the regularisers; read the `Episode_Reward/` lines, not the total |
| `Mean action std` sits at 1.00 for hundreds of iterations | the batch is too small for the signal: more environments, or the GPU recipe |
| the replay looks better than the training curve | observation noise off (`--no-obs-noise`); compare with it on |
| `--max-iterations 100` on a resume stops almost at once | it is 100 **more**; the run was already past it |
| `train.py` refuses a `--scene` | an imported `.map`; maps are replay-only by design |
| the run goes headless on macOS without being asked | `python` instead of `.venv/bin/mjpython`; the run is fine, only the window is missing |

## When something does not match

Do not change a reward weight or a command range to make a run look better. Those are the
task's decisions (`CLAUDE.md`, "Where a parameter lives"), and a number moved to fix one
run is a comparison broken for every run after it. Report the run as it was; a tuning
change is a separate, named change with a measurement beside it.
