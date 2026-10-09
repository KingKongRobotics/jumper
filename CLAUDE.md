# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The reference documents are thorough and worth reading rather than rediscovering:
[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) for Train and Design routing from the shared root,
[`README.md`](README.md) for the tour, [`docs/DESIGN.md`](docs/DESIGN.md) for why the framework is
shaped this way, [`docs/USAGE.md`](docs/USAGE.md) for tasks / assets / defaults,
[`docs/TUTORIAL.md`](docs/TUTORIAL.md) for the whole path as one worked example,
[`docs/CONTROLS.md`](docs/CONTROLS.md) for the pad, the keyboard and the mode switches,
[`docs/VENDOR.md`](docs/VENDOR.md) for the vendored copies, [`docs/AGENT_SETUP.md`](docs/AGENT_SETUP.md)
for bringing a machine up, [`deploy/README.md`](deploy/README.md) for getting a policy onto the robot,
and [`deploy/BUNDLE.md`](deploy/BUNDLE.md) for the bundle a host loads. What follows is what those
documents assume you already know.

## Commands

Everything runs from the repository root. `pip install -e .` is mandatory, not a convenience —
see "Two package roots" below.

```bash
pip install -e ".[dev]"                    # dev extra adds pytest and ruff
python -m pytest tests/ -q                 # the whole suite, ~30 s
python -m pytest tests/test_seam.py -q     # one file
python -m pytest "tests/test_seam.py::test_seam_is_confined_to_two_files" -q   # one test
python -m ruff check .                     # line-length 100; vendored copies excluded
```

```bash
python scripts/train.py --list                          # every task and its assets
python scripts/train.py --task jumper.flat --dry-run      # resolve without building anything
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64
python scripts/play.py --task jumper.flat                 # newest checkpoint
python scripts/play.py --task jumper.flat --headless --steps 500 --video clip.mp4   # the one MP4 recorder
python scripts/export.py --task jumper.flat --checkpoint <path>/model_4999.pt
```

```bash
python scripts/deploy.py                                    # the app: cross-build, .rknn, bundle
python scripts/deploy.py --translate <bundle> --language zh --manual <file>   # bundle-manual skill
bash deploy/convert/docker-convert.sh --bundle <export>     # one ONNX -> .rknn by hand
bash deploy/fsm/docker-build.sh                             # cross-build by hand
python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> --bundle <export>   # read-only preflight
./runtime/board/controller --bundle . --check-reference         # on the board; touches no bus
```

An **export directory** is one policy; a **bundle** is what every host loads -- the FSM config,
one policy per mode, and each host's controller build (the board's cross-compiled binary, the
browser's wasm, `play --app`'s extension), in one directory and one `.app` from which each host
takes its own part. The crate is the controller on all three hosts, so `controller` is the robot's
whole program. Every bundle carries `manual.en.json`, the pad and the keyboard as the controller
reads them; after each build the `bundle-manual` skill adds the Chinese one.

Deployment failures are silent rather than loud -- a wrong joint order or an unmatched QoS
profile gives a robot that runs and is wrong. The `deploy` skill sequences the whole path;
`deploy/README.md` is the map.

On macOS the live viewer needs `.venv/bin/mjpython` rather than `python`; without it the run
goes headless and says so. `--headless` turns the viewer off explicitly.

## Architecture

### The seam is one swapped class

The framework runs the same manager, reward and rsl_rl code on two simulators. That is achieved
by **replacing mjlab's `Simulation` class**, not by an abstraction layer: `mjlab.sim` holds a
module-level `_SIMULATION_CLS`, `mjrl.backend.select.use_backend` sets it, and
`ManagerBasedRlEnv.__init__` constructs through `get_simulation_cls()`.

Two consequences that are easy to get wrong:

- **`use_backend(res)` must run before the env is built.** The construction point reads the
  registry once; setting it afterwards has no effect and nothing raises.
- `mjrl/backend/native_sim.py` is interface-compatible with mjlab's `Simulation` by hand. When
  mjlab's changes, this must follow.

### Two package roots

`pyproject.toml` declares `.` and `rl` as package roots. `rl/` itself has **no `__init__.py`**,
so `mjrl`, `mjlab` and `rsl_rl` are all top-level packages and upstream code needs no edits. The
price is that nothing under `rl/` is importable without the editable install — no `sys.path`
manipulation anywhere compensates for it, deliberately (see the docstring in `scripts/_cli.py`).

`import mjrl.backend.resolve as R` binds the **function** `resolve`, not the module: the
package's `__init__` re-exports it. Use `importlib.import_module` when you need the module.

### Tasks are directories, not registry entries

A task id **is** its module path under `tasks/`: `tasks/jumper/tripod/` ⇄ `"jumper.tripod"`. The
registry stores no config values; `load_env_cfg` turns the id straight into an import. A task is
three files (`__init__.py` registering metadata only, `env_cfg.py`, `rl_cfg.py`) plus one import
line in `tasks/__init__.py` — forget that line and the task silently does not exist, which
`tests/test_registry.py` checks for. A task a person can drive needs a fourth, `controls.yaml`:
`velocity_env_cfg` raises without it unless the task passes `operator_command=False`, and the
`controls` skill writes it from `controller/vocabulary.json` rather than by hand.

A task that needs something done **on the robot** that no contract field describes adds
`deploy/lib.rs` — a Rust `ModeHook` (`deploy/fsm/src/hook.rs`), called at fixed points of a
mode's loop. `deploy/fsm/build.rs` finds every such file and compiles it into all three hosts,
keyed by the task id its path spells, so the controller's own source names no task; a manifest
mode opts in with `hook`. `jumper.five_foot`'s is the one there is: its mirror carries the
claw on the right from a policy trained with it on the left, and it closes the claw on the
trigger at its side. A hook reads the pad only through the controls its task's
`controls.yaml` keeps for it (`task:`).

`tasks/*/__init__.py` must not import config modules: `--list` has to work on a machine without
the simulation dependencies, and after it runs there is not one heavy dependency in
`sys.modules`.

**Nothing under `scripts/` changes when adding a robot or a task.** `train.py`, `play.py` and
`export.py` share one parser in `scripts/_cli.py`; `deploy.py` builds the deployment side, reads
no environment, and has a plain parser of its own. `scripts/` holds exactly `_cli.py`,
`train.py`, `play.py`, `export.py`, `deploy.py` and `tests/test_log_layout.py` enforces that —
the rule it defends is **two answers to one question**, so a fifth entry is only a problem if it
duplicates one of the four.

### The native backend

`mjrl/backend/native_sim.py` presents mjwarp's batched interface over plain MuJoCo. Two facts
about it cause most of the bugs:

- **`mujoco.rollout` dispatches rolls from a work queue**, so `data` is per-thread scratch and
  `_datas[i]` is *not* environment `i`. State comes back from `state_out`, indexed by roll.
- **`mj_setConst` uses its `MjData` as scratch** and recomputes `stat`, so domain randomisation
  needs a dedicated scratch `MjData` and has to restore the visual statistics.

Logs land in `logs/<model>/<task>/<timestamp>` — `scripts/train.py` composes all three segments.

## Working rules

**Where a parameter lives.** Three places, and the split is enforced by
`tests/test_task_parity.py` rather than by good intentions:

| | holds | example |
|---|---|---|
| `tasks/jumper/common/constants.py` | facts about the **hardware** — one robot, so one value | `EFFORT_LIMIT`, `STAND_Z`, the IMU noise levels, `PUSH_VELOCITY_RANGE` |
| `tasks/jumper/common/` | **mechanism, and no tuning** — reward functions, curriculum classes, the ladder's *shape* | `CommandRangeCurriculum`, `ladder()`, `track_yaw_velocity` |
| `tasks/<task>/env_cfg.py` | **every parameter**, even where all tasks agree | weights, the foot-lift target, the speed ceiling |

A tuning number in `common/` is a number every task in the family inherits without
choosing — all ten of them sit on it — and that is not a hypothetical:
`jumper.tetrapod` ran for months on a command ladder no one had picked for it, and it
looked exactly like agreement. `velocity_env_cfg` therefore **raises rather than
defaulting** for the parameters a task must own. Writing the same value in every
task that calls it is the point, not the cost — agreement that is written down
survives one of them changing.

**The locomotion tasks are each other's controls.** `jumper.flat`, `jumper.tripod`,
`jumper.ripple` and `jumper.tetrapod` must differ in the gait and in the speed ceiling
the gait implies, and in **nothing else**. A reward or observation that differs
makes every comparison between them mean something other than the gait, silently.
`tests/test_task_parity.py` fails on any other difference and tells you to justify
it in `GAIT_EXEMPT` / `CEILING_EXEMPT` if it is deliberate. The curriculum ladder
is one shape scaled to each ceiling, so `level 1` means the same fraction of
capability everywhere — it did not, once, and three tasks were running three
different curricula under one name.

**Vendored code.** `rl/mjlab/` and `rl/rsl_rl/` are modifiable copies kept word-for-word with
upstream so they can be rebased. Every change carries an `[mjrl]` marker saying why and against
which upstream version (`# [mjrl] reason: … (against mjlab 1.6.0)`) — that marker is the only
way to find our changes again at upgrade time, so grep for `[mjrl]` rather than for the
`reason:` wording, which two of the five do not use. Exactly five files carry one, and
`tests/test_seam.py::test_seam_is_confined_to_two_files` fails when a sixth appears: either
justify it and update the list, or the change belongs somewhere else. Never
`pip install mjlab` / `rsl-rl-lib` — a PyPI copy shadows the vendored one ambiguously.

**English.** Everything that lands in the repository — comments, docstrings, assertion messages,
docs, commit messages — is written in English. Chat with the user is a separate matter.

The one exception is a **`*.zh.md` translation beside an English document**, for a reader who
will use it *instead of* the English rather than alongside it. That is also why each one carries
a `<!-- tracks: <source> @ sha256:… -->` line and `tests/test_translations.py` fails when the
source moves: a stale translation does not look stale, it looks authoritative, and the change
that invalidated it happened in the other file where nobody reviewing it would look. Restamp
with `tools/checks/stamp_translations.py` **after** bringing the translation along, never
instead. Do not translate `.claude/skills/` — an agent reads those, and the `description:` is
what decides whether a skill triggers at all.

**How tests are written here.** The suite is aimed at failures that produce no error: a depth
image that is uniformly scaled, environments reshuffled every step, a config parsed and printed
and never passed on, a viewer whose geoms are silently dropped past 100000. So a test states
*which* silent failure it pins and, where the assertion could pass vacuously, includes a control
group that proves it can fail. More than one test here was written the obvious way first and
passed against the code it was meant to catch; where that happened the test says so, in
`test_resolve.py` and in `test_viewer_limit.py`.

**Tests are for the framework, never for a task.** A test pins shared machinery: `rl/mjrl`, the
vendored copies, `scripts/`, `controller/`, `deploy/`, `tools/`, and the contracts every task is
walked through (`test_registry.py`, `test_controls.py`). Nothing under `tasks/` gets a test of
its own, `tasks/jumper/common/` included -- a task's rewards, parameters, commands and curricula
are checked by training and replaying it. When a task change needs checking, run a one-off probe
outside the repository and quote its numbers in the comment beside the decision; do not add it
to `tests/`. The files there that test one task's own behaviour (`test_posture.py`,
`test_five_foot.py` and the like) predate this rule and are not a pattern to follow.

**Measurements belong next to the decision.** Numbers in this repository are from real runs and
are quoted with the machine they came from, because the defaults they justify are otherwise
indistinguishable from guesses — `resolve.DEFAULT_MAX_THREADS` and `_MAX_DRAW_ENVS` are the
pattern to follow.
