# Manual

Setting up, the entry points, adding a task, adding a model, setting defaults — the
full version of each.

[`../README.md`](../README.md) is the introduction — what the repository is, how it is laid
out and which document covers what. The operating instructions are here, with the
reasoning, the criteria, and the places people get wrong. For design trade-offs see
[`DESIGN.md`](DESIGN.md).

Every code block here is taken from working code in the repository; the `jumper.stairs`
example below is `jumper.flat`'s files under another name, numbers included.

---

## Contents

- [Setting up](#setting-up)
- [The entry points](#the-entry-points)
- [Adding a task](#adding-a-task)
- [A task that needs material: `jumper.dance`](#a-task-that-needs-material-jumperdance)
- [A task with a second command: `jumper.posture`](#a-task-with-a-second-command-jumperposture)
- [Adding a model (asset)](#adding-a-model-asset)
- [Setting defaults](#setting-defaults)
- [Scenes](#scenes)
- [TensorBoard](#tensorboard)
- [Resuming training](#resuming-training)
- [Command cheat sheet](#command-cheat-sheet)

---

## Setting up

Every dependency is a PyPI wheel. **No system CUDA toolkit is needed** — the cu128 torch
wheels carry their own runtime. On a GPU machine the NVIDIA driver is the only system-level
prerequisite.

> For an AI agent, hand it [`AGENT_SETUP.md`](AGENT_SETUP.md) instead: same steps,
> written as hard verification gates. Two scripts do the mechanical parts on any platform and
> need nothing installed --
> `python3 .claude/skills/setup-env/scripts/detect.py` reports the machine and the commands that
> follow from it, and `gates.py` checks the result. Agents that load skills can use
> [`.claude/skills/setup-env/`](../.claude/skills/setup-env/) directly.

**1. Python 3.10 – 3.13.** Check with `python3 -V`; if it is older, see the table below.

If you have not cloned the repository yet:

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
```

**2. Create and activate a venv** at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
```

The last line must print the repository's `.venv`. If it prints the system Python or a conda
environment, stop — every later `pip` will install into the wrong place.

**3. Install torch.** With an NVIDIA GPU, confirm `nvidia-smi` lists the card, then:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The cu128 index must be explicit; Blackwell cards (RTX 50 series, `sm_120`) do not run on
cu126. Without a GPU, install the CPU build and omit `--index-url`.

**4. Install this repository:**

```bash
pip install -e .
```

> ⚠️ **Required, not optional.** The vendored `mjlab` / `rsl_rl` under [`rl/`](../rl) are
> importable only through the mapping this creates. Without it every entry point fails with
> `ModuleNotFoundError: No module named 'mjlab'`.

> 🛑 **Do not `pip install mjlab` or `rsl-rl-lib`** — they are vendored (see
> [`VENDOR.md`](VENDOR.md)) and a pip copy shadows them ambiguously. If one is
> already present: `pip uninstall -y mjlab rsl-rl-lib`.

**5. Verify:**

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "import mujoco; from mujoco import rollout; print(mujoco.__version__)"
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
python -m pytest tests/ -q
```

`cuda:0` should appear in the first on a GPU machine. The third must print paths inside this
repository's `rl/`; `site-packages` means a pip copy is still installed.

### If the system Python is too old

| Platform | What to do |
|---|---|
| Ubuntu 22.04 / 24.04 | Ships 3.10 / 3.12; nothing to do |
| Ubuntu 20.04 and older | `add-apt-repository ppa:deadsnakes/ppa`, then `apt install python3.11 python3.11-venv` |
| macOS | `brew install python@3.12` |
| Windows | Installer from [python.org](https://www.python.org/downloads/), tick "Add python.exe to PATH" |

On Debian / Ubuntu, `ensurepip is not available` means venv is a separate package:
`apt install python3-venv`. [uv](https://docs.astral.sh/uv/) also works:
`uv venv --python 3.11 .venv`.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install` landed elsewhere | venv not activated; re-check `sys.prefix`. On Windows, `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `torch.cuda.is_available()` is False | CPU build or wrong CUDA version. `print(torch.__version__)` must show `+cu128` on Blackwell |
| Warp reports MSVC errors on Windows | Install the "Desktop development with C++" workload of Visual Studio Build Tools (several GB; wait for the error first) |
| No live viewer on macOS | Launch through `.venv/bin/mjpython`; plain `python` falls back to headless and says so |
| Camera sensors report GL errors headless | `MUJOCO_GL=egl` (no `DISPLAY` needed) |

---

## The entry points

`train.py`, `play.py` and `export.py` share one parser (`scripts/_cli.py`), so
`--task`, `--model` and the backend flags are written identically everywhere.
**Adding a robot or task requires no edit under `scripts/`.**

| Entry point | Purpose |
|---|---|
| `scripts/train.py` | Training, with the live viewer and TensorBoard |
| `scripts/play.py` | Replay a trained policy, with the live viewer |
| `scripts/export.py` | Export a trained policy: actor.onnx + layout.json + README + the checkpoint |
| `scripts/deploy.py` | Build the controller and bundle it with the policies it drives; reads no environment, so it has a parser of its own |

### Shared arguments

| Argument | Default | Meaning |
|---|---|---|
| `--task <id>` | `MJRL_TASK` | Task id; choices from `--list` |
| `--model <name\|path>` | task's default asset | Registered asset name, or a path to an `.xml` file |
| `--list` | — | List tasks, their assets and the scenes, then exit |
| `--backend {auto,warp,native}` | `auto` | Never falls back when explicit |
| `--device <auto\|cuda:0\|cpu>` | `auto` | |
| `--num_envs <n>` | train: `MJRL_NUM_ENVS` (4096 in the shipped `.env`); empty: warp 4096 / native 64. play: `MJRL_PLAY_NUM_ENVS`; empty: 1 | Train: **changes PPO's effective batch size**. Play: how many robots are on screen |
| `--cpu_threads <n>` | `0` | Native threads; `0` = cores capped at 8, then by `num_envs`. **Not all cores** -- past ~8 the step is serial-bound and extra threads cost time |
| `--strip-visual {auto,on,off}` | `auto` | Strip visual meshes on native; `auto` = once per-env models exceed 2 GiB |
| `--dry-run` | — | Resolve and print; build nothing |

### `deploy.py`

Builds the app: `deploy/fsm` -- the shared observation, action decode and
state machine -- compiled for every host and packaged with the policies it
drives. One crate, three hosts: a browser loads the wasm, `play --app` imports
the extension, the robot runs `controller`. No `--task`: it reads no environment.

With no arguments it builds the bundle `deploy/manifests.json` defines, and
first makes what the robot needs: the board's controller is cross-built in
docker every time, and each policy without a current `.rknn` is converted in
docker. An app that still lacks the board's controller, a `.rknn` or the
reference vectors is refused. The Chinese manual is the one step left, to an
agent: the `bundle-manual` skill writes it and `--translate` adds it.

| Argument | Default | Meaning |
|---|---|---|
| `--bundle [<dir>]` | `out/bundle_<timestamp>` | Where to write the app: `<dir>/<name>/` and `<dir>/<name>.app`, one for every host |
| `--manifest [<name>]` | the only one | Which bundle of `deploy/manifests.json` to build: which exports, which controls |
| `--manifests <file>` | `deploy/manifests.json` | Read the manifests from another file |
| `--mode NAME=DIR` | — | One-off alternative to `--manifest`: a mode and its `export.py` directory. Repeat per mode |
| `--fsm <file>` | synthesised | With `--mode`: the FSM to ship. Required above one mode |
| `--allow-incomplete` | off | Skip the cross build and the conversions, take what is there, and say in the notes what is missing. For a browser, a bench or the tests |
| `--require-clean` | off | Refuse an uncommitted controller source -- the crate and every file its build reads, each task's hook among them. Off by default; an uncommitted build records `-dirty` in `bundle.json` |
| `--translate <dir>` | — | Add a translation of a bundle's `manual.en.json` (`--language zh --manual <file>`), checked against the English, and pack the `.app` again |
| `--check-reference <dir>` | — | Replay a bundle's `reference.json` through this machine's extension and report where this host differs |

```bash
python scripts/deploy.py
python scripts/deploy.py --check-reference out/bundle_<timestamp>
```

It used to publish the wasm alone into a consumer's asset directory (`--out`,
`--check`) and install the extension into site-packages (`--python`); the app
carries both, and those flags are gone.

`deploy/manifests.json` is the build's input, maintained by hand: which
exports go in and which control switches into each. Exports, not checkpoints --
a checkpoint is not something a robot can run, and the export records which one
it came from. It pairs with a `.controller.toml` holding the cascade and the safety
limits, and the two may not say each other's half -- the manifest owns the
modes and the bindings, the cascade file owns everything else. What was on the
command line before is now written down, so a bundle records its ancestry and
not only its contents.

One build writes one bundle, `<name>/` and `<name>.app` beside it, and each
host takes its own part: the board its `.rknn` models and
`runtime/board/controller`, a browser the ONNX and `runtime/web/controller.wasm`,
`play --app` the ONNX and the extension built for its platform. It was three
directories, one per host, that held the FSM config and the contracts byte for
byte and were hashed against each other to prove it; one directory cannot
disagree with itself, and the `jumper` bundle is one 10.9 MB `.app` against
22.8 MB for the three. [`../deploy/BUNDLE.md`](../deploy/BUNDLE.md) is the format.

Every bundle carries `reference.json` -- recorded frames of state in,
observation, action and joint targets out -- so the three hosts are held
against one set of numbers rather than against each other. `--check-reference`
is this host's side of that; the robot's is
`runtime/board/controller --bundle . --check-reference`. It also carries `manual.en.json`,
every key and pad button and what it does in each mode, generated from the
controller itself; `--translate` adds the Chinese one.

### `train.py`

Its own switches -- the viewer, the logger, resuming, TensorBoard -- are under
[Command-line-only switches](#command-line-only-switches); the viewer's and
`--scene` are `play.py`'s as well.

### `play.py` and `export.py`

Both take the same `--task` / `--model` / backend flags as `train.py`.

```bash
python scripts/play.py --task jumper.flat                       # newest checkpoint
python scripts/play.py --task jumper.tripod --checkpoint <path>/model_4999.pt
python scripts/play.py --task jumper.flat --agent zero --num_envs 4   # no checkpoint
python scripts/export.py --task jumper.tripod --checkpoint <path>/model_4999.pt
python scripts/play.py --app out/bundle_<timestamp>/<name>.app   # play an app, every mode of it
```

`--app <app>` plays an app — the `.app` the robot and a browser load — through the
**deployment** controller rather than a task's policy, and no checkpoint: every
mode, entered on the app's own buttons and keys; the viewer's keys and a gamepad
read through each mode's own controls; and the controller's own joint targets and
gains driving the servos, so its ramps and a task's deploy hook reach the joints.
The environment is the world only, the app's default mode's task unless `--task`
names another. It is how you find out what an app does
without uploading it anywhere. It replaced `--fsm`, which handed the policy's raw
action back to mjlab and so could show neither a hook nor a mode switch
(`rl/mjrl/app_play.py`).

`play.py` builds the environment in replay mode (no external disturbances, and
episodes that run on rather than timing out; the one-shot jumps keep theirs, so a
replay loops whole attempts). Observation noise stays at training levels by
default -- `--no-obs-noise` gives the clean signal. It takes the newest checkpoint under
`logs/<model>/<task>/` when `--checkpoint` is omitted — the same resolution
`train.py --resume` uses, so both commands mean the same file by "the newest one". `--agent zero|random` needs
no checkpoint and is for looking at the environment itself. A replay is one
environment unless `--num_envs` or `MJRL_PLAY_NUM_ENVS` says otherwise:
`MJRL_NUM_ENVS` is training's batch size, and play does not read it.

`--video <file.mp4>` records the replay, with or without a window, through the
repository's **one MP4 recorder** (`rl/mjrl/viewer/video.py`): the README clips
(`tools/readme_media.py`), the dance export and the [Colab notebook](COLAB.md) all
run this command rather than drawing for themselves, so every picture of a policy
is the same camera and the same geoms. It needs a finite `--steps`; `--video-fps`,
`--video-width` and `--video-height` size the file, and the camera is placed from
the robot's **first frame** -- `--video-azimuth` in degrees from the way it faces
(0 from behind, 180 at its front), `--video-elevation`, `--video-distance`, and
`--no-video-follow` with `--video-lookat-height` for a camera that stays where the
first frame put it. A Linux machine without a display needs `MUJOCO_GL=egl` set
before Python starts.

```bash
python scripts/play.py --task jumper.tripod --headless --steps 500 --video clip.mp4
```

Three more switches are for a replay that measures or records rather than one
somebody watches. `--stop-on-done` ends the replay at the first episode end, so a
recording of a one-shot motion holds one attempt rather than looping. `--command
VX VY YAW` holds a fixed body-frame velocity, checked against the range the
checkpoint was trained on, and `--evaluation-out <json>` writes the tracking error
it got; `--checkpoint-command-ranges` samples only the curriculum rung the
checkpoint saved, and `--replay-info-out <json>` records what was resolved.
`--command-script <file.py>[:name]` hands the command terms a function of time --
`name(t)` returns `{term: values}` -- written where the operator writes, so the
pad does not drive a scripted term; it is how the README clips walk and change
posture (`mjrl.replay.script_commands`).

`export.py` writes `actor.onnx`, `layout.json`, a README and a copy of the checkpoint
(plus `<name>.trajectory.json`, for a task that tracks a recording) to
`tasks/<task path>/out/<date-time>/` -- one directory per export, beside the task
it came from, named by date rather than by checkpoint so a second export of the
same checkpoint cannot silently replace the first. It runs ten validations first --
dimensional consistency, that every observation term is one the deployment can
actually build, that none of them is a quantity this robot cannot measure, the home
pose against `init_state`, the checkpoint against the current config, and the ONNX
against the torch policy in both shape and value.

To run that export on a Rockchip NPU, [`deploy/convert/`](../deploy/convert/README.md) converts its
`actor.onnx` to an `.rknn` and checks the converted model against the ONNX the same way
`export.py` checks the ONNX against torch. It carries its own virtualenv —
rknn-toolkit2 pins `numpy`, `torch` and `onnx` below what training needs.

---

## Adding a task

### The rule everything rests on

**A task id is its module path relative to `tasks/`**, with dots as directory separators:

```
tasks/jumper/tripod/   <->   id "jumper.tripod"
tasks/jumper/dance/    <->   id "jumper.dance"
```

The registry holds **no config values**. `load_env_cfg` turns the id straight into
`tasks.<id>.env_cfg` and imports it. So a wrong directory name means the config is not found —
and you get an error that states the convention and points at `tasks/registry.py`, not a bare
`ImportError`.

### Three files, and a fourth for a task a person drives

Taking `jumper.stairs` as the example, create `tasks/jumper/stairs/`. A task a person can
drive needs a fourth file, `controls.yaml`, written by the `controls` skill:
`velocity_env_cfg` raises without it unless the task passes `operator_command=False`. The
`new-task` skill writes all four.

**`__init__.py`** — registers the name and the available assets, nothing else. **Do not import
config modules here**: this file runs during `import tasks`, and `--list` has to stay usable
on a machine without the simulation dependencies.

```python
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stairs",                      # must match the directory path
    assets=JUMPER_ASSETS,                    # what --model may select; see the next section
    description="jumper hexapod on stairs",  # this line is what --list shows
    tags=("locomotion", "jumper", "stairs"), # free-form tags, for filtering
)
```

**`env_cfg.py`** — the environment config. The signature is fixed: `asset` and `play`, plus
any keywords the task's own `cli_args` adds (`jumper.swing`'s `swing_angle`, say).

```python
from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import velocity_env_cfg


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    Args:
        asset: model XML path, from `--model`. None uses the task's default asset.
        play:  replay mode -- observation noise and external disturbances off,
               longer episodes.
    """
    cfg = velocity_env_cfg(
        controls=Path(__file__).parent / "controls.yaml",
        asset=asset,
        play=play,
        # The skeleton raises on any of these left out: each is a decision this
        # task makes, not a default it inherits. See DESIGN.md §14.
        foot_target_height=0.025,
        action_rate_weight=-0.1,
        pose_weight=1.0,
        command_lin_ceiling=0.8,
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )
    cfg.scene.terrain.terrain_type = "generator"   # what makes this task different goes here
    return cfg
```

**`rl_cfg.py`** — the hyper-parameters **for this task only**. Editing it affects nothing else.

```python
from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.stairs")
    # To depart from the baseline, edit here, e.g.:
    # cfg.algorithm.entropy_coef = 0.005
    # cfg.max_iterations = 20_000
    return cfg


def runner_cls() -> type:
    """mjlab's velocity runner (it logs extra tracking metrics), with the curriculum
    levels added to its checkpoints so `--resume` carries on at the same rung."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner
```

A task whose curriculum climbs a ladder must return `CurriculumRunner`: mjlab's own runner
saves only the step counter, so `--resume` would start the ladder again at level 0.
`tests/test_curriculum_resume.py` fails on a task that gets this wrong.

**`mdp/` (optional)** — rewards / observations / terminations only this task uses. The test is
**how many tasks need it**: used by one, it stays with that task; used by more than one, it
moves into `tasks/<family>/common/mdp/`. That is exactly how the three gait rewards are split.

### Last step: trigger registration

Add a line to [`../tasks/__init__.py`](../tasks/__init__.py):

```python
from .jumper import stairs  # noqa: F401,E402
```

**Nothing under `scripts/` changes.**

### Check

```bash
python scripts/train.py --list          # the new task should appear
python scripts/train.py --task jumper.stairs --dry-run
pytest tests/                           # structural invariants check the directory conventions
```

`tests/test_registry.py` checks three things: that ids match directory paths, that each
directory has all three files, and that every task directory on disk is actually registered —
forget the import and the task quietly does not exist.

### Should hyper-parameters be shared or per-task

**Per-task.** Every hyper-parameter and both networks' hidden dims are keyword arguments to
`jumper_ppo_baseline`, and each task states its own in its `rl_cfg.py`. The four locomotion
tasks state the same values today, because nobody has had a measured reason to diverge — not
because they have to match; the others (`posture`, `five_foot`, `swing`, `dance`, `jump`,
`ref_free_jump`) depart where they had one.

This used to say the four tasks share one baseline "because they are each other's control
groups: with the algorithm side held fixed, differences in training attribute cleanly to the
gait prior". **That was not true.** The tasks differ in reward budget (6.00 / 7.50 / 7.00 /
7.00 of positive weight) and `jumper.tripod` also sees an observation the others do not, so
only two of the six pairings — `flat vs tetrapod` and `flat vs ripple` — differ by a single
variable. The hyper-parameters were never what made those two valid, and differing ones do
not break them. `tasks/jumper/common/ppo.py` carries the measured table.

`tests/test_ppo_cfg.py` requires the four locomotion tasks to state the values that matter
rather than inherit them, so none of them can silently pick up an edit made for another one.
A new task is covered once it is added to that test's `JUMPER_TASKS`.

---

## A task that needs material: `jumper.dance`

Every other task describes a robot and a goal. `jumper.dance` describes a robot and
**a specific recorded performance** — its behaviour comes from data rather than from
config, and that changes where things can go wrong, so it is worth knowing how it is
arranged before writing another one like it.

The demonstration choreography and its face animation are committed, 30.5 MB, so it
runs from a fresh clone like the rest. That is a deliberate trade: a task that cannot
start without a file obtained through some side channel is a task nobody checks still
works. The music it was made for is **not** committed — its source could not be
established — and training does not need it: only the performance video
`scripts/export.py` renders does.

### The material

It goes in [`../tasks/jumper/dance/media/`](../tasks/jumper/dance/media/), where the
`demo.npz` and `demo.mp4` are committed and anything else is git-ignored, and is found **by
extension** — the names are the user's, at most one of each kind:

| File | Purpose | |
|---|---|---|
| `<name>.npz` | the choreography — the only file training reads | required |
| `<name>.{mp3,wav,m4a,ogg,flac}` | the music the dance was made for; the exported video needs it | optional |
| `<name>.{mp4,mov,mkv,webm}` | the face-screen animation, shipped at export | optional |

The reference recording the choreography was made from is **not** part of this, and
the reason is worth borrowing. It was required for a while, and no code ever opened
it. What it cost was structural: the search is by extension, so a second video in a
different role made **both** ambiguous, and the face animation — a file that is
genuinely used — had to hide in an `eyes/` subdirectory to get clear of one that
was not. Requiring material for documentation's sake buys a collision and a
subdirectory; the reference lives with the rest of the source material instead.

What remains required is what the environment cannot be built without, which is the
choreography alone. The music and the face animation are optional because training
reads neither, so insisting on them would stop clones that have a complete
choreography.

Nothing has to be run by hand. The first environment build converts the clip into
the format mjlab's `MotionLoader` reads — resampled to the 50 Hz control rate, run
through forward kinematics for the per-body poses, velocities differenced — and
caches it in `media/.cache/`, keyed on **the contents** of the source. Replace the
`.npz` and the next run reconverts; there is no step to forget, and no stale cache
to train the previous dance from.

### What it refuses, and why that is the point

A motion clip is data, and data fails silently. Read in the wrong joint order,
indexed against the wrong body list, or resampled a frame off, it produces arrays
of exactly the right shape holding exactly the wrong numbers — and training
converges on all of them. So the conversion checks before it writes:

| Check | Catches |
|---|---|
| joint limits | a clip from another robot, and **any exchanged leg pair** — the left and right limits are reflections, so a swapped leg is asked to leave its own range |
| support-foot coplanarity | a base pose and joint angles from different takes, or a different ground height |
| single-frame continuity | two takes concatenated, or a dropped frame |
| music duration | the wrong track — grossly wrong fails, a trim difference is printed |

Each has a control group in
[`../tests/test_dance_motion.py`](../tests/test_dance_motion.py), because a check
of this kind is worth exactly nothing until something has been shown to fail it.

### Three things it does differently

**Symmetry augmentation is off.** `jumper_ppo_baseline` turns it on and the four
velocity tasks want it — the hexapod is mirror-symmetric. A choreography is not:
mirroring asserts that when the reference raises the left arm, raising the right is
equally correct. It would train, to the average of the dance and its mirror image.

**All 22 joints are actioned**, not the 20 of `GAIT_JOINTS`. The velocity tasks
hold the grippers closed; this dance sweeps them.

**The observations are named for the deployment**, not for mjlab. `scripts/export.py`
already reserves `clip_phase`, `ref_joint_pos`, `ref_joint_vel`, `ref_tilt_error`
and `ref_future`; mjlab's tracking task calls the same quantities `command` and
`motion_anchor_*`, which are not in that whitelist — adopting its observation group
gives a policy that trains perfectly and cannot be exported.

### Exporting a performance as well as a policy

A task may define `export_media.py`, one of the optional hooks `scripts/` calls
without knowing the task (`cli_args` and `play_status` are the others):
`scripts/export.py` calls it via `tasks.load_export_media` knowing only that it
might exist. Eight of the ten tasks have none, and for them the call resolves to
`None` and nothing changes; `jumper.posture`'s writes `POSTURE_COMMAND.md` beside
the contract.

`jumper.dance` uses it to render the trained policy dancing the whole clip, mux the
music back on, and copy the soundtrack and face animation alongside:

```
tasks/jumper/dance/out/<date-time>/
├── actor.onnx   layout.json   README.md      the exported policy
├── model_<n>.pt                              the weights it came from
├── demo.motion.trajectory.json               the clip it is scored against
└── media/       dance.mp4  music.mp3  eyes.mp4
```

The trajectory is not this seam's doing: any task that attaches a
`reference_contract` gets one, `jumper.jump` included. A policy whose observation
is a reference is not runnable without the reference, so it travels beside the
ONNX rather than being something the board is assumed to already have.

`media/` is a subdirectory because `out/<date-time>/` **is** the exported policy —
`deploy/` reads that directory to assemble the bundle a board loads, and a 40 MB
video has no business making the trip. `--no-video` skips the recording, which
takes minutes; the export itself still takes seconds.

The picture is `play.py --video`'s. The hook draws nothing: it runs
`scripts/play.py --video` — the one MP4 recorder, the same one the README clips
and the Colab notebook use — for the whole clip with a fixed camera, at the
task's own control rate so the video's time axis is the simulation's, and muxes
the music onto what comes back. Two decisions inside it are worth borrowing:

- **`--stop-on-done`, so failure terminations stay on.** If the policy falls
  partway through, the recording stops there and the export says so, rather than
  showing a robot teleported back to the reference and carrying on — which would
  misrepresent the policy exactly where it matters most.
- **The music offset is read from the clip.** This choreography opens with a 2.0 s
  silent lead-in (`audio_start_in_sim`, exact across all 496 beats). Muxing at zero
  puts the performance two and a half beats early — close enough to be believed.

### Reusing the imitation stack for another task

The machinery is mjlab's `mjlab.tasks.tracking`, a BeyondMimic re-implementation
that sits in the vendored tree and that nothing used before this task. It is
imported, never modified: `rl/` stays word-for-word with upstream, and everything
robot-specific lives in `tasks/jumper/common/dance/` — the clip loader, the
observation names, the torque-peak term, and `dance_env_cfg`, the environment itself.

Another dance on this robot is a task directory: its clip in `media/`, and an
`env_cfg.py` that hands `dance_env_cfg` every number, since the builder has no
defaults. `jumper.dance_brazilian`, `jumper.dance_dream_wings`, `jumper.dance_waist` and
`jumper.dance_maze` are exactly that — the other four dances rl-wbc-fsm plays,
imported by `tools/import_wbc_dances.py`, whose docstring says what was taken as the
state and how the base pose was solved. So are `jumper.gesture_hello`,
`jumper.gesture_bow`, `jumper.gesture_paw` and `jumper.gesture_salute`, the four
one-shot gestures rl-wbc-fsm puts on the d-pad, imported by
`tools/import_wbc_gestures.py`: timed as the board plays them, eased in from `HOME`
and back out to it, and stood on all six feet rather than four. Another robot copies
`common/dance/` rather than touching `rl/`.

---

## A task with a second command: `jumper.posture`

The four gait tasks have one command, mjlab's `twist` — where to go
(`jumper.five_foot` adds a `body_pose` of its own). `jumper.posture` adds a second one, `posture` — how to hold the body while going
there — and that is the pattern worth taking from it rather than the four
particular numbers.

```bash
python scripts/train.py --task jumper.posture --model jumper
python scripts/play.py  --task jumper.posture
```

### What it commands

| channel | meaning | range |
|---|---|---|
| `twist` | body yawed over its own feet | ±30° parked, ±15° walking |
| `pitch` | nose down positive | ±20° parked, ±15° walking |
| `roll` | left side up positive | ±15° |
| `height` | base above the ground | 0.07–0.15 m (it stands at 0.107) |

Each angle comes from the **standing band** while the velocity command is parked
(the norm of its `[vx, vy, wz]` under 0.06) and from the walking band otherwise.
The bands and the signs are `jumper.five_foot`'s, on purpose: the two tasks' pose
commands reach the robot on one set of channels and one stick layout, so a full
stick has to mean the same lean in both, and the controller holds a walking robot
to the walking band on every host. Pitch was nose-up positive until 2026-09-26;
a checkpoint trained before that reads it inverted.

Sampled independently every 3–8 s, held until the next resample, and active while
walking as well as standing. A fifth of the environments, moving and parked alike,
are pinned to the neutral posture exactly, because a uniform draw over four axes
essentially never produces "level, square and standing" and that is the posture the
tripod gait was learned at. (The parked fifth was zero until 2026-09-29, and a
policy trained on that holds its middle legs off HOME when told nothing; the task's
`env_cfg.py` has the measurement.)

The range in that table is the **top of a ladder**, not where a run starts.
`PostureRangeCurriculum` climbs ±7.5°, ±10°, ±15° on the walking band — the
standing band and the height climb the same fractions of their own ranges — and
promotes only when **every** axis
tracks below its bar, so the slowest axis sets the pace and the logs say which
one it is (`Curriculum/posture/<axis>_err` against `<axis>_err_bar`). `std`
scales with each rung, so a narrow level is a smaller ask rather than a looser
ruler; `tests/test_posture.py` pins that with the free score a
never-leans robot collects, which has to stay flat across the ladder.

The bars come from a measurement rather than a guess — 128 environments all
commanded neutral, random actions, which is the "what does the robot do to
itself" reference:

```
sigma   |twist|   |pitch|   |roll|    |height - 0.107|
 0.0     0.127°    0.269°   0.144°       3.57 mm
 0.8     1.137°    1.542°   1.136°       5.02 mm
```

The sigma-0 row is **bias, not floor** — a passive robot sags ~3.6 mm and a
policy pushes it back — so the floor is what the noise adds on top, ~1.3° and
1.4 mm.

That measurement is what set the first rung. At ±5° the bar was 2.0° against
2.5° of never-trying and 1.3° of noise, so most of what level 0 measured was
PPO's exploration — the same fault the velocity ladder found in its own 0.15
rung, where it dropped the rung rather than loosening the gate. At ±7.5° the bar
is 2.5° against 3.75°, and every rung on both axes now asks for 42–67% of what
ignoring the command gives.

It walks the tripod gait at a **cadence that follows the command**, with the
stride held at 72 mm: `f = clamp(v / (2 × 0.072), 2.0, 5.56) Hz`. At 200 Hz
control the ceiling is 36 control steps a cycle and the floor 100 — both whole and
both **even**, which `tasks/jumper/tripod/mdp/phase.py` requires or the two tripod
groups get unequal time. `tasks/jumper/posture/mdp/cadence.py` has why 72 mm and
not the 60 that would have been natural. Ask for a speed the cadence cannot
deliver and nothing raises; the tracking reward simply becomes uncollectable.

### Driving it in `play`

Both commands are the operator's, from a pad or from the keyboard. Either one
alone drives all seven channels, and the one touched last drives:

```
pad                                                 keyboard
left stick      walk: forward/back, left/right      W S A D, or the arrows
right stick     up/down: nose down/up               I K
                left/right: twist, then turn        H ; twist, J L turn
R3 held         left/right: roll                    U O
                up/down: height, as a speed         N M, held: the high / low stance
R3 tapped       height back to standing             (let N or M go)
B               hand both commands back             Esc
(hands off)     stand still, level and square,      (let go)
                at the height last set
```

**The right stick's left-right is two things in order.** The first half of its
travel twists the body over its planted feet; past half-travel the turn climbs
from zero and the twist unwinds, back to zero at three quarters, where the turn is
at half its top rate -- the body leads into the turn and is square again before
the turn is fast. **Held down, the right stick is its other layer**: with `R3`
held its left-right rolls the body and its up-down moves the height, with twist,
turn and pitch at zero; let go of `R3` and it twists, turns and pitches again at
once. **On the pad the height is moved, not placed** (asked for on 2026-09-29):
the stick is a speed -- full deflection crosses from standing to either end of
the range in one second, `integrate_s` -- and let go, the body stays at that
height. `R3` tapped, with the stick left alone, puts it back at standing
(`reset: R3`), and `B` does too. Earlier that day the height was on the triggers, and then on `R3`
and the stick as a position, back at standing the moment `R3` came up.

**The keyboard is a path of its own.** Each key is bound to one direction of one
axis -- `"+"` the axis's own positive direction, so the keyboard carries no sign
to get wrong -- and pushes it linearly to full after the file's `full_after_s`
(2.0 s), back at rest the moment the key comes up -- the height included:
`N` and `M` are the high and the low stance, and let go the body is standing
again. The layout is the robot's operator guide's, Control-agent 3.1, as revised
on 2026-09-29: `J` and `L` turn and `H` and `;` twist, one row under the right
hand, and Esc lets go of everything. Until that day the keyboard was a virtual
pad, each key a stick direction read through the pad's mapping, and `M` held was
`R3`; that could not say "`J` turns and `H` twists", and a keyboard laid out for
the hands needed to.

All of it is `tasks/jumper/posture/controls.yaml`: `command` is a list of the two
command terms, a pad binding may take part of a stick's travel -- and, with two
more numbers, give it back (`travel: [0.0, 0.5, 0.5, 0.75]`) -- or sit on the `R3`
layer, an axis may be moved rather than placed (`integrate_s`), and
`load_controls` refuses a keyboard that leaves a command axis out without saying
why. `tasks/jumper/common/mdp/operator.py` drives `play` from it and
`deploy/fsm/src/operator.rs` the robot and the browser; both command terms share
one operator, so `B` hands both back at once. The four locomotion tasks carry the
same layout for their one command -- `W S A D` or the arrows, and `J L` -- written
by the `controls` skill's script. [`CONTROLS.md`](CONTROLS.md) has the schema.

**Every one of those letters also toggles a viewer flag** (W wireframe, S shadow,
A auto-connect, D static body, I inertia, K skybox, J joint, L additive, U
actuator, O perturb object, N island, M centre of mass, B perturb force). MuJoCo
handles its shortcuts in C++ and calls the user callback *as well as*, not
instead of, its own, so there is nothing to intercept from Python. The robot obeys
the key and the picture flickers; pressing the letter again puts the flag back.

### Why a second command term rather than four more channels

`twist` is read positionally all over this repository — `moving_gate` takes
`cmd[:, :3]`, the gait rewards, the curriculum, the teleop and gamepad subclasses
and `scripts/export.py`'s contract all assume a three-vector. Widening it would
have been a silent change to every one of them. A second term costs one
observation and is inert for every task that does not mount it.

### What a new command has to displace

The part that took the most care was not adding terms but finding the ones that
already measured the same quantity against a fixed frame:

| term | why it had to go |
|---|---|
| `upright` | pays for pitch = roll = 0, which half the command range asks against. Replaced by `track_tilt`, the same function at a zero command. |
| `pose`'s `std_standing` | six times tighter than the walking width, and holding a commanded posture *is* leaving HOME. Widened to the walking value -- but only away from the neutral posture: at neutral, HOME is still the answer, and a parked robot held to the walking width there settled 0.37 rad off it. The parked width ramps from `jumper.tripod`'s standing width at neutral to the walking one a tenth of the way out. |

This is `jumper.swing`'s lesson on a second task: a term named for what you want,
measured against a frame the robot is no longer in. It produces a number every
step either way.

### On the robot and in a browser

It exports and bundles like any other task. `deploy/fsm` builds the three terms
this task added -- `posture_command`, the five frames taken four steps apart, and
the clock whose tempo follows the command -- and `play --fsm --fsm-diff` (since
replaced by `play --app`, which does not compare observations) against
a live mjlab matched every term to float precision, bar the training-only
encoder bias on `joint_pos`. The clock's contract also says *when* it advances,
`params.advance`, because the task changed that once: an export from before the
change runs in the order it was trained in, and one after it matched mjlab's
`gait_phase` to 5e-6 degrees through 26 tempo changes -- where the old order,
forced on the same export, was 6.3 degrees off (`play --fsm`'s own loop, one
environment, a walk redrawn every 0.5 s; `play`'s own commands mostly sit at the
tempo ceiling, where the two orders agree). `deploy/manifests.json`'s `jumper`
bundle runs it as the default walking mode.

A page that loads the bundle carries no copy of these controls. It hands the
wasm every key by its browser name, the pad (on-screen sticks included) and its
own Stop, and the controller inside decides what each one means -- the same
`operator.rs` the robot runs (`deploy/fsm/BUNDLE_README.md`, "Handing over a
person's input").

### The precision rung

Like the velocity ladder, the posture ladder ends by repeating its top range with
a tighter `std`: level 3 is the full range marked by half of level 2's, 3.75
degrees and 10 mm. It went in once it had been measured -- a policy holding level
2 fell short of every command by a fixed fraction, 16-21% of a twist and up to half
of a small roll, while the gait's own ripple walking was only 0.2-1.0 degrees.
`mdp/curriculum.py::POSTURE_STD_SCALES` has the table.

---

## Adding a model (asset)

### Three things, three places

| What | Where | Test |
|---|---|---|
| Model files, meshes | `assets/<name>/` | Pure data |
| The scripts that generate / calibrate it | `assets/<name>/tools/` | **Bound to this asset**, lives and dies with it |
| Python constants describing it | `tasks/<family>/common/` | Joint names, HOME pose, collision scheme, `EntityCfg` |

The last row is the one that gets misplaced: `constants.py` describes that XML, but it imports
mjlab and is referenced by tasks — it is code, not data.

### Declaring the asset

The asset table lives in a **lightweight module**, `assets.py`, which must **not import
mjlab**:

```python
# tasks/jumper/common/assets.py
from __future__ import annotations

from pathlib import Path

from ...paths import ASSETS_DIR
from ...registry import AssetSpec

JUMPER_XML: Path = ASSETS_DIR / "jumper" / "jumper.xml"

JUMPER_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        name="jumper",                              # the name used on --model
        path=JUMPER_XML,
        description="22-DoF heterogeneous hexapod",   # shown by --list; keep it accurate
    ),
    # Add further assets below; the first one is the default
    AssetSpec(name="jumper_v2", path=ASSETS_DIR / "jumper" / "jumper_v2.xml",
              description="variant with longer shanks"),
)
```

**Why it has to be lightweight**: a task's `__init__.py` needs the asset path during
`import tasks`. Taking it from `constants.py` would pull mjlab / torch / mujoco in with it and
make `--list` heavy. As it stands, after running `--list` there is **not one heavy dependency**
in `sys.modules` (no mjlab, torch, mujoco or warp) — that is the guarantee this buys.
`constants.py` in turn takes `JUMPER_XML` from here, so the path has a single source of truth.

### The three forms of `--model`

```bash
python scripts/train.py --task jumper.flat                          # default asset (first in the table)
python scripts/train.py --task jumper.flat --model jumper_v2          # by registered name
python scripts/train.py --task jumper.flat --model /path/to/a.xml   # a path
```

The path form is a **back door for experiments**: a task's joint names, HOME pose and foot geom
names are written against its default asset, so a structurally different model fails while the
environment is being built (joint not found, sensor matches no geom) — **rather than quietly
producing wrong results**.

A wrong name lists the available ones; a nonexistent path says the file does not exist. Two
different errors pointing in two different directions, so they are reported separately.

### Log isolation per asset

The log path is **`logs/<model>/<task>/<date-time>`**, where `<model>` is the model file's
stem. So training one task against different `--model` values naturally splits into separate
subtrees:

```
logs/jumper/jumper.flat/<timestamp>/
logs/jumper_v2/jumper.flat/<timestamp>/
```

### Convex hulls as collision geometry

`tools/hull_collision.py` gives a new model convex hulls as collision geometry. `-k` is a
count of directions to keep, and **`0` is a sentinel for "keep all of them"** rather than
the bottom of the scale: `-k 0` is the exact hull and changes contact by
nothing, and every other value decimates, the smaller the more. Measured in
`tools/hull_collision.py`: `0` leaves 28662 faces at 0.00 mm of inward error,
`256` leaves 6110 at 3.57 mm, `64` leaves 1772 at 7.81 mm.

```bash
python tools/hull_collision.py --model assets/<name>/<model>.xml --exclude <meshes that touch the ground>
```

It reports by default (including the maximum inward shrink); `--apply` writes the STLs and
edits the XML.

---

## Setting defaults

### Precedence

```
command line  >  shell environment  >  .env.local  >  .env  >  the argument's own default
```

One rule carries this: `load_dotenv` **does not overwrite** existing environment variables, and
argparse defaults then read from `os.environ`. So "the command line wins if given" needs no
special-casing.

If nothing supplies a value and the argument has no built-in default (`MJRL_TASK` is the only
one), the entry point **errors and says what is missing**.

### Two files

| File | Committed | Contents |
|---|---|---|
| [`../.env`](../.env) | yes | Project-level defaults, shared by the team |
| `.env.local` | no | Personal overrides and secrets; higher precedence |

### Every key

```bash
# -- which task --
MJRL_TASK=jumper.flat        # see --list. Empty and not given on the command line -> error

# which asset: a registered name, or a path to an .xml file
MJRL_MODEL=                # empty -> the task's default asset

# how the world looks: ground, sky and lights. See --list. Empty -> the task's own
MJRL_SCENE=

# -- a task's own settings --
# Both of these have a matching flag, `--swing-angle` and `--swing-length`, and
# **the task declares them, not the parser**: `scripts/_cli.py` is one parser for
# three entry points and a task's vocabulary in it is the thing
# `tests/test_log_layout.py` forbids. See `cli_args` in `tasks/jumper/swing/`.

# jumper.swing: the angle the swing is released from, in degrees. One value releases
# every episode from there; two are a range drawn from uniformly, sign randomised
# either way. Empty keeps each mode's default -- the training range 0,45 and a dead
# stop on replay.
MJRL_SWING_ANGLE=

# jumper.swing: the rope length in metres, beam to the plank. One value hangs every
# swing at it; two are a range drawn from per episode, per environment. Empty is
# the whole 0.6-1.8 m the frame can be rigged at, in replay as well as training --
# the period runs 1.50 to 2.65 s across it, which is what stops a policy learning
# one rhythm. Pin it to one value to ask whether a policy really generalises.
MJRL_SWING_LENGTH=

# -- a scene's own settings --
# Declared by the scene, like the pair above by the task. See cli_args in scenes/rough.py.

# --scene rough: pin the 10 x 7 grid to one row, 0 (flat) to 9 (hardest). Empty is
# the full ladder.
MJRL_TERRAIN_ROW=

# --scene rough: pin it to one column, by sub-terrain name (flat, pyramid_stairs, ...).
# Empty is the mix.
MJRL_TERRAIN_COL=

# -- backend and device --
MJRL_BACKEND=auto          # auto | warp | native
MJRL_DEVICE=auto           # auto | cuda:0 | cpu
MJRL_NUM_ENVS=             # training; empty -> per backend: warp 4096 / native 64
MJRL_PLAY_NUM_ENVS=1       # play.py; empty -> 1. Not a batch size, so a key of its own
MJRL_CPU_THREADS=0         # native thread count, 0 = cores capped at 8 (then by num_envs)
MJRL_STRIP_VISUAL=auto     # native: strip visual-only meshes. auto | on | off

# -- TensorBoard --
MJRL_TENSORBOARD=on        # on | off (true/false, yes/no, 1/0; any case)
MJRL_TB_PORT=6006          # preferred port; a busy one is skipped
```

#### Why the automatic thread count is not "every core"

About half of a native step is serial -- the gather and scatter between the batch
buffers and the per-environment `MjData` is Python-side -- so by Amdahl the
parallel part is spent by around eight workers, and every thread after that only
adds synchronisation and scheduling cost. Measured on an i9-14900KF (8
performance + 16 efficiency cores, 32 logical), ms per policy step on
`jumper.tetrapod`:

| envs | 2 | 4 | **8** | 16 | 24 | 32 (every core) |
|---|---|---|---|---|---|---|
| 16 | 10.63 | 8.10 | **7.17** | 9.70 | — | 9.85 |
| 64 | 24.14 | 16.31 | **15.27** | 21.18 | 22.39 | 22.36 |
| 256 | 92.84 | 69.79 | **59.09** | 62.84 | 67.39 | 68.82 |

Eight is the optimum at every environment count and "every core" costs 16-46%.
Pinning does not rescue it (eight threads on eight dedicated performance cores:
21.44 ms against 15.71 unpinned -- the main thread needs somewhere to run too),
and neither does the physical core count, 24 here, which measured no better than
32.

The thread count is **not a hyper-parameter**: whichever thread steps an
environment, it starts from its own warm start and its own control, so any thread
count gives the same result bit for bit (`tests/test_native_determinism.py`) and
this changes throughput and nothing else. A machine with fewer than 8 cores uses
what it has. Set `MJRL_CPU_THREADS` if yours measures otherwise.

`auto` detects: `warp:cuda` if CUDA is available, otherwise `native:cpu`.
**An explicit choice never falls back** — asking for a GPU on a machine without one is an
error, not a silent switch to CPU. That is the easiest trap for a framework like this to set.

### What counts as a value

Keys with a fixed set of values are read **case-insensitively**, and a value
outside the set **errors** rather than falling back to the default:

| Key | Accepts |
|---|---|
| `MJRL_TENSORBOARD` | `on`/`off`, and `true`/`false`, `yes`/`no`, `1`/`0` |
| `MJRL_BACKEND` | `auto`, `warp`, `native` |
| `MJRL_STRIP_VISUAL` | `auto`, `on`, `off` |
| `MJRL_NUM_ENVS`, `MJRL_PLAY_NUM_ENVS`, `MJRL_CPU_THREADS`, `MJRL_TB_PORT` | an integer |

```
error: MJRL_STRIP_VISUAL='of' is not one of: auto, on, off (check .env)
```

Falling back would be worse than stopping. The whole point of these keys is to
change what a run does, so one that is quietly ignored produces a run that looks
normal and is not the one asked for -- and nothing in the output mentions the
setting, so the natural conclusion is that `.env` is not read at all.

**argparse does not do this for you.** `choices=[...]` is checked only for values
typed on the command line, never for a default -- so an argument whose default
comes from `.env` validates nothing, and the bad value travels on to whatever
consumes it. `MJRL_STRIP_VISUAL=OFF` used to surface as a bare `KeyError: 'OFF'`
from a lookup table three call levels away, with `.env` nowhere in the traceback.
An empty value (`MJRL_MODEL=`) still means "use the built-in default"; it is only
a *wrong* value that stops the run.

### Three ways to set them

```bash
# 1) edit .env (project level, committed)
MJRL_BACKEND=native

# 2) shell environment (this session only, beats .env)
MJRL_BACKEND=native python scripts/train.py

# 3) command line (beats everything)
python scripts/train.py --backend native --device cpu --num_envs 512
```

### Common combinations

```bash
# Dev box, checking logic quickly: few environments + the live viewer
MJRL_BACKEND=native
MJRL_NUM_ENVS=16

# Training box: leave backend on auto (it picks warp:cuda) and take the backend default of 4096
MJRL_BACKEND=auto
MJRL_NUM_ENVS=
```

### Command-line-only switches

These are given on the command line. Most have no `.env` key at all; the two
TensorBoard ones do (`MJRL_TENSORBOARD`, `MJRL_TB_PORT`) and the switch overrides
it, as always:

| Switch | Effect |
|---|---|
| `--headless` | **Turn the live viewer off. It is on by default.** |
| `--viewer-env N` | Which environment the camera follows. Defaults to the one nearest the **scene centre** |
| `--viewer-env-num N` | **How many** environments to draw (including the followed one); 128 if not given. Not the same as the previous switch |
| `--viewer-fps F` | Frames a second the window shows: 30 when training, 60 when replaying. Drawing runs on the viewer's own thread, so the simulation pays one copy of the state per frame |
| `--viewer-ui` / `--no-viewer-ui` | MuJoCo's own panels -- rendering flags, visualisation options, the model tree. On when replaying, off when training; they edit the followed environment only |
| `--logger tensorboard\|wandb` | **Defaults to tensorboard.** mjlab defaults to wandb, and that path is broken on wandb 0.29.0 |
| `--max-iterations N` | Override the iteration count in the task's `rl_cfg.py`. It is how many iterations **this run** performs, so on a resumed run it is how many *more* |
| `--resume` | Continue training from the newest checkpoint under `logs/<model>/<task>/` |
| `--checkpoint PATH` | Which checkpoint to resume from: a `model_*.pt`, or a directory whose newest one is taken. Implies `--resume` |
| `--dry-run` | Resolve and print only; do not build an environment |
| `--tensorboard` / `--no-tensorboard` | Start TensorBoard, or do not. **It is on by default.** The pair exists so the command line can override `MJRL_TENSORBOARD` in **both** directions -- with only the negative form there is no way back on for one run once `.env` has turned it off |
| `--tb-port N` | Preferred port, default 6006. A busy one is skipped |
| `--tb-scope run\|task\|all` | How much to serve; `task` if not given |

> **At most 128 environments are drawn** (including the followed one); fewer than 128 and they
> are all drawn. Above that it prints:
>
> ```
> [mjrl] live viewer: 4096 environments, drawing 128 of them (cap 128)
> ```
>
> The cap is necessary, for two reasons:
>
> - **Geom capacity**: the viewer's `user_scn` has room for 100000 geoms; past that MuJoCo
>   prints one warning that scrolls away in the training log and then **silently drops** the
>   rest — the picture still shows "a field of robots" and there is no sign that 60% are
>   missing (measured with warp + 4096: 253890 geoms expected, 100000 arrived, 1613
>   environments actually drawn).
> - **Time budget**: drawing grows with the number drawn. Measured with native + 4096, drawing
>   everything costs **405 ms per frame** — which, while drawing hung off `sim.step()`, came
>   straight out of training. It runs on the viewer's own thread now; the frame is still
>   *taken* at the end of `sim.step()`, the point at which the state is consistent and cannot
>   tear, but that is one copy of the state. The same 405 ms would be a slideshow rather than a
>   slower simulation.
>
> 128 is far below both lines on either backend. The geom-capacity bound is still computed as a
> backstop: a sufficiently complex robot could fill 100000 geoms with 128 environments, and
> then the limit comes from capacity and says so.
>
> With no usable display (no `DISPLAY`, or macOS without `mjpython`) it **falls back to
> headless automatically**, so "viewer on by default" never stops a run from starting.
>
> **Watching training on warp is still not free**, and not because of the frames: MuJoCo's own
> render loop redraws the scene continuously on the GPU the simulation runs on. Measured with
> jumper.swing at 4096 environments on an RTX 5090 D: 1.69 s an iteration headless, 3.92 with
> the window, 4.09 with it at one frame a second, and 1.98 with `--viewer-env-num 1`.

`play.py` also **does not evaluate the reward terms**: nothing in a replay reads them, and they
were the largest single cost of a replay step (4.7 of 11.3 ms on jumper.posture, one environment
on warp). `tests/test_replay.py` checks, task by task, that skipping them changes nothing the
policy is shown; `rl/mjrl/replay.py` has why that can go wrong.

### TensorBoard

**Training starts TensorBoard itself** and prints the URL:

```
[mjrl] log directory logs/jumper/jumper.tetrapod/2026-09-04_14-30-51
[mjrl] tensorboard http://localhost:6006/ serving logs/jumper/jumper.tetrapod
```

It starts **before** the environment is built, so the page is already open
during the tens of seconds that scene assembly and model compilation take. The
event files come from `rsl_rl`'s writer either way -- `WandbLogWriter` subclasses
`SummaryWriter`, so `--logger wandb` writes them too.

**What it serves** is `--tb-scope`:

| Scope | Directory | Why |
|---|---|---|
| `run` | `logs/<model>/<task>/<timestamp>` | This run alone; fastest to load |
| `task` | `logs/<model>/<task>` | **Default.** TensorBoard labels each subdirectory as a run, so the timestamps become the series names and the previous run is right there to compare against |
| `all` | `logs/` | Every model and task; slow once the tree is large |

**The port** is 6006, and one already in use is skipped -- a second run lands on
6007 rather than failing, and says so. Set a different starting point with
`--tb-port` or `MJRL_TB_PORT`.

**It binds to localhost only.** To reach it from another machine, forward the
port rather than exposing the metrics to the network:

```bash
ssh -L 6006:localhost:6006 <training-host>
```

A browser is opened automatically when there is one to open -- not over ssh
(the page would open on the remote console) and not without a display.

**Turning it off.** For one run, `--no-tensorboard`; for good,
`MJRL_TENSORBOARD=off` in `.env` (or `.env.local`, or the shell environment).
`--tensorboard` turns it back on for a single run over an `off` in the file --
the command line beats `.env` in both directions, which is the whole reason the
switch is a pair rather than a lone `--no-tensorboard`.

The value is read as a switch, not as a string: `on/off`, `true/false`, `yes/no`
and `1/0` are all accepted in any case. Anything else **errors**:

```
error: MJRL_TENSORBOARD='nope' is not on or off (true/false, yes/no and 1/0 are
accepted too, in any case) (check .env)
```

That is deliberate, and the same rule `MJRL_NUM_ENVS` follows for integers: a
value quietly falling back to the default would mean `MJRL_TENSORBOARD=of`
silently meaning "on", with nothing to say why the setting had no effect.

**Failure is never fatal.** A server that will not start prints its reason and
the run goes on; TensorBoard's own output is kept out of the training log and
written to `tensorboard.log` inside the run directory.

The server is shut down when training ends, including on Ctrl-C and on
`kill <pid>`. A `kill -9` or an OOM kill leaves it behind -- `pgrep -f
tensorboard.main` finds it.

### Scenes

A **scene** is the ground, the sky and the lights -- the axis orthogonal to the
task. The same gait task is worth running under a neutral studio grey for a
screenshot and a sunset for a video, and the same look is worth reusing across
tasks; keeping the two apart is what makes that cheap.

```bash
python scripts/train.py --list                          # the scenes are listed too
python scripts/train.py --task jumper.flat --scene studio
MJRL_SCENE=beach python scripts/train.py         # or from .env
```

| Scene | `use` | What it is |
|---|---|---|
| `default` | both | mjlab's own look, written out so it can be named and diffed against |
| `studio` | watching | neutral grey, key plus fill, dark sky -- for screenshots and video |
| `daylight` | watching | blue sky, warm ground, one hard sun with shadows |
| `beach` | watching | low warm sun over damp sand, dusk sky, long shadows |
| `football` | watching | a pitch at one fifth scale: mown turf, markings, goals, afternoon sun, **and a ball the robot can kick** |
| `swing` | watching | **adds a swing**: an A-frame and a plank seat the robot can stand on. Not what `jumper.swing` trains against — that task builds its own, and `--scene swing` on top of it raises rather than rigging two |
| `ice` | both | **changes the ground**: wet ice, sliding friction 0.03 against the feet's 1.2 |
| `rubber` | both | **changes the ground**: a coarse anti-slip mat at 1.8 -- the other end of `ice` |
| `soft` | training | **changes the ground**: `default`'s look, but the feet sink into it |
| `plain` | training | no textures or materials -- for `geom_rgba` randomisation, and cheapest to draw |
| `rough` | training | **changes the ground**: generated slopes, stairs and rough surface |

`python scripts/train.py --list` is the live answer; this table is a copy and can
drift.

Passing no scene leaves the task's own configuration untouched, so the default
behaviour is unchanged rather than merely equivalent.

**`use` is not a tag.** It says what the scene is *for*, and it decides whether
the scene is exported for another simulator -- see below. `training` means the
scene is an instrument: it exists to produce a training effect or to hold one
variable still, and it is not somewhere anyone wants to watch a robot. `watching`
does not mean a scene cannot be trained in; `studio` is only lights. It means
training in it is beside the point, and for `football` actively misleading, since
one pitch is drawn at the world origin while environments sit on a spacing grid.
#### A scene may take arguments of its own

`rough` generates a 10 x 7 grid — ten difficulty rows, one column per
sub-terrain — and a run normally meets all of it through the terrain curriculum.
To test **one tile**:

```bash
# the hardest random roughness, nothing else
python scripts/play.py  --task jumper.posture --scene rough \
    --terrain-row 9 --terrain-col random_rough

# train on one difficulty of the stairs
python scripts/train.py --task jumper.posture --scene rough \
    --terrain-row 3 --terrain-col pyramid_stairs

```

or from `.env`, like every other default:

```bash
MJRL_TERRAIN_ROW=9
MJRL_TERRAIN_COL=random_rough
```

Either flag alone works: a row without a column pins the difficulty across all
seven sub-terrains, a column without a row keeps the full ladder of one of them.
A pinned tile is **the tile the full grid would have generated** at that row and
column — the same generator, narrowed — which is what makes it a test of the
training terrain rather than a lookalike. A name that matches nothing raises and
lists the alternatives; falling back to the full mix would produce a run that
looks correct and was not what was asked for.

The flags are **declared by the scene**, not by `scripts/_cli.py` — one parser
serves three entry points and is not allowed a scene's or a task's vocabulary.
`scenes.load_cli_args` is the mirror of `tasks.load_cli_args`, which is how
`jumper.swing` gets its `--swing-angle`. A scene that wants options defines
`cli_args(group)` beside its `scene()` factory, returning the `dest` names it
added, and takes them as keyword arguments to `scene()`. Scenes that declare none
are unaffected, and `export.py` never offers them: nothing a scene changes
reaches an ONNX file.

**Adding one** is a module and a line: write `scenes/<id>.py` with a `scene()`
factory returning a `Scene`, and add a `register` call in `scenes/__init__.py`.
Image files go in `scenes/assets/<id>/`, beside the code that describes them, so
a scene is one module plus one directory.

A scene that generates its own sky defines a `sky(dirs) -> rgb` beside its
`scene()` factory and builds it on [`scenes/skygen.py`](../scenes/skygen.py),
which holds the parts of MuJoCo's cube maps that were measured rather than
assumed: which world direction each face covers, how it samples the image within
a face, and noise that is continuous across the seams. All three fail silently --
the cube compiles and the sky looks like a sky. Render the faces with:

```bash
python tools/make_skybox.py <scene-id>     # or --all
```

The PNGs are **committed, not built on demand**, so re-run it after changing a
sky's colours or the scene goes on loading the faces written last time.

What to know before writing one, in the order it bites:

- **A material has to match the ground geom.** mjlab names it `terrain`, and
  `scenes.registry.GROUND` is the expression that matches it. A material that
  misses binds to nothing: it is created, the ground renders unchanged, and
  nothing reports a problem.
- **The ambient light is the headlight, and it dominates.** MuJoCo's
  camera-attached light follows the viewer and lights everything it can see, so it
  sets the floor on how dark a shadow can get. mjlab's scene template ships 0.3
  ambient / 0.6 diffuse -- three times MuJoCo's own default -- and at that strength
  adjusting a fill light changes almost nothing. Each scene declares its own
  through `Scene.headlight`; it has to go through `SceneCfg.spec_fn`, because a
  `<visual>` block in an attached entity is overridden by the parent spec without
  a word.
- **A scene may add a physical prop, and it must declare it.** `Scene.props`
  holds entities -- the pitch's ball -- and they have mass and they collide, so
  contacts, contact count and solver load all change. A policy trained with one is
  not comparable to one trained without, and the observation and reward are
  untouched so nothing else reports it: `apply` warns. Placement is wired
  automatically, because an entity with no reset event stays at (0, 0, 0) in
  *every* environment while the config reads correct.
- **Other geometry a scene adds must be visual.** `Scene.decorate` can put things in
  the world -- pitch markings, goalposts -- through the same `spec_fn`, and all of
  it needs `contype=0, conaffinity=0`. Something the robot can collide with
  changes the task it is training on, for every task the scene is selected on,
  with the reward and the observation unchanged and only the world quietly
  different.
- **Changing `terrain_type` is not cosmetic.** The tasks here remove the
  `terrain_scan` sensor when they set flat ground, so a scene turning generated
  terrain back on gives a policy that cannot perceive what it walks on -- it
  trains and converges regardless. **Nothing checks this**; restore the sensor in
  the task's `env_cfg` before reading anything into such a run.
- **A new scene has to say what it is for.** `register(..., use=...)` has no
  default, because both defaults are wrong in a way that looks right: exported
  ships a curriculum grid as a playground, withheld means a new scene quietly
  never arrives anywhere.

#### Exporting a scene

A scene can leave this repository as a package another simulator imports -- the
counterpart of the robot package, and deliberately its opposite half: that one
ships the robot and no world, this ships the world and no robot.

```bash
python scenes/tools/export_web_scene.py --list                 # what would go, and what would not
python scenes/tools/export_web_scene.py --scene football --out out/scenes
python scenes/tools/export_web_scene.py --all --out out/scenes
```

What travels is everything the scene is, provided it survives being written
down: the ground *and its contact parameters* -- so `ice` arrives slippery rather
than merely blue -- textures, materials, lights, the sky, the headlight,
decorative geometry, and props as separate models with their placement stated.

Two independent gates decide, and they catch different things:

- **`use == "training"` is refused**, by the scene's own declaration. The
  exporter keeps no list of scene names; the judgement lives in
  `scenes/__init__.py` beside the registration, and the reasoning with it.
- **A scene whose behaviour is Python is refused.** `soft` applies a force at
  each foot every step to fake granular resistance the contact model cannot
  express. No MJCF field means that, and a package that ignored it would compile,
  render correctly, and be wrong only underfoot.

`soft` fails both, which is worth keeping rather than a reason to drop one.

The exporter then compiles what it wrote and holds it to the spec it came from:
geom, light, texture and material counts; that every decorative geom came back
non-colliding; and that nothing in the package still refers to the machine that
made it. It runs on a machine that could not train -- a scene compiles onto an
empty world without a task or a robot.

---

## Resuming training

```bash
# Continue the run that just finished
python scripts/train.py --task jumper.flat --resume

# Continue from one particular checkpoint
python scripts/train.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-09-04_14-30-51/model_1999.pt

# Continue from a run directory: its newest checkpoint is taken
python scripts/train.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-09-04_14-30-51
```

`--checkpoint` implies `--resume`; `--resume` on its own means "the newest
checkpoint under `logs/<model>/<task>/`".

### What "the newest" means

**The most recently written file, not the highest number.** Resuming from an early
checkpoint starts a new run whose numbers begin below the highest already on disk,
so sorting by name would send the next `--resume` back to the abandoned run and
every one after it -- training would look like it was progressing while going
nowhere. Ties inside the same second (some filesystems keep mtime to a whole one)
are broken by the iteration number.

The resolution is `mjrl.checkpoint`, and `play.py` uses the same one: which file
"the newest checkpoint" means must not depend on which command is asking.

### What is restored

The **whole training state**, not the weights alone:

| Restored | Why it matters |
|---|---|
| Policy and value function | The obvious part |
| Optimiser state | The adaptive learning rate keeps its place instead of jumping back to `learning_rate` |
| Iteration counter | The new run logs and saves from where the old one stopped |
| `common_step_counter` | Curricula count environment steps; without it every step-driven curriculum rewinds |
| Curriculum levels | A ladder that earns its level (command, posture, payload) would restart at level 0 with the step counter already past every dwell. Saved by `CurriculumRunner`; an `MJRL_*_LEVEL` in the environment overrides it. Terrain rows are drawn afresh |

Loading the policy alone would be a warm restart rather than a resumption -- and it
trains, so nothing looks wrong. (`play.py` *does* load the policy alone, with
`load_cfg={"actor": True}`: replay needs nothing else.)

### Where it writes

A **new** timestamped run directory, so a resumed run never overwrites the
checkpoints it started from:

```
logs/jumper/jumper.flat/2026-09-04_14-30-51/   model_0.pt ... model_4999.pt      <- resumed from
logs/jumper/jumper.flat/2026-09-05_09-12-04/   model_5000.pt ... model_6998.pt   <- the resumed run
```

The curves join up regardless: the iteration counter continues, so under the
default `--tb-scope task` TensorBoard draws the second run as a series that begins
where the first ended.

### How many iterations it runs

`--max-iterations` (or the task's `rl_cfg.py`) is how many iterations **this run**
performs, added on top of the ones the checkpoint already has -- otherwise resuming
a finished run would have nothing left to do. The load prints the arithmetic:

```
[mjrl] resuming from logs/jumper/jumper.flat/2026-09-04_14-30-51/model_4999.pt
[mjrl] log directory logs/jumper/jumper.flat/2026-09-05_09-12-04
[mjrl] resumed at iteration 4999; running 2000 more (through 6998)
```

### When it refuses

A checkpoint from a different configuration -- changed observation terms, a
different network shape, another task -- stops the run:

```
[mjrl] cannot resume from logs/jumper/jumper.flat/2026-09-04_14-30-51/model_4999.pt:
  RuntimeError: Error(s) in loading state_dict for MLPModel: size mismatch for ...
The usual cause is a checkpoint from a different configuration: it has to come from
the same task, model and network as this run, observation and action dimensions
included. Train from scratch (drop --resume), or point --checkpoint at a matching run.
```

The checkpoint is resolved **before** the environment is built, so a typo in
`--checkpoint` costs a second rather than the minute of scene assembly and model
compilation that would otherwise precede it.

**There is no `.env` key.** Every other default is settable there; this one is
deliberately not: a sticky "always resume" default would silently continue an old
run on a launch meant to start a new one -- and that failure is invisible in the
output.

---

## Command cheat sheet

```bash
# List every task and the assets it accepts
python scripts/train.py --list

# Run with the defaults from .env
python scripts/train.py

# Resolve without building an environment, to check the arguments
python scripts/train.py --task jumper.tripod --model jumper --dry-run

# Watch while training (the default; no switch needed)
python scripts/train.py --task jumper.flat --num_envs 64

# Force real CPU training
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64

# Train without TensorBoard (it is started automatically otherwise)
python scripts/train.py --task jumper.flat --no-tensorboard

# Continue the run that just finished, for 2000 more iterations
python scripts/train.py --task jumper.flat --resume --max-iterations 2000

# Continue from a particular checkpoint
python scripts/train.py --task jumper.flat --checkpoint <path>/model_1999.pt
```

Native across a chosen number of threads, a smaller picture, and macOS:

```bash
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 4096 --cpu_threads 32
python scripts/train.py --task jumper.flat --viewer-env-num 16 --viewer-env 0
.venv/bin/mjpython scripts/train.py --task jumper.flat --num_envs 64   # macOS, for the viewer
```

Replay and export take the same arguments:

```bash
python scripts/play.py --task jumper.flat                      # newest checkpoint
python scripts/play.py --task jumper.flat --agent zero         # no checkpoint needed
python scripts/export.py --task jumper.flat --checkpoint <path>/model_4999.pt
```
