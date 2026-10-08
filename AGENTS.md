# AGENTS.md

The working rules for this repository live in [`CLAUDE.md`](CLAUDE.md) — commands, the
architecture worth knowing before editing, and the conventions that are not visible from any one
file. They are not Claude-specific; read that file first whatever agent you are.

## Start here

Jumper is the shared entry point for Train and Design. Read
[`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) to select the route. Movement training uses the root
repository. Appearance and environment creation use the independent
[jumper-design repository](https://github.com/KingKongRobotics/jumper-design): read its current
[AGENTS.md](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md) and referenced
workflow/protocol first. Read remotely for guidance; when execution needs source or assets,
reuse a suitable existing checkout or clone it separately, then fetch its required LFS assets.
Record the Design commit used for generated outputs. Do not add it as a submodule or assume
its code or binary assets are present in this checkout.
Keep separate Train and Design virtual environments. Imported maps are replay-only in Train;
there is no Train `.skin` loader. Do not silently reconcile robot mechanical differences.

Use lightweight models for bounded reading/documentation, standard models for routine
implementation and checks, and advanced models for architecture or difficult debugging.
Delegate independent tasks and preserve required validation regardless of model choice.

| task | source of truth |
|---|---|
| Installation, documentation navigation and project status | [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) |
| Any code or documentation change | [`CLAUDE.md`](CLAUDE.md) — repository rules and architecture |
| Set up or repair an environment | [`docs/AGENT_SETUP.md`](docs/AGENT_SETUP.md), then the `setup-env` skill |
| Add a task | [`docs/USAGE.md`](docs/USAGE.md), then the `new-task` skill |
| Train a policy and report it | [`docs/TUTORIAL.md`](docs/TUTORIAL.md), then the `train` skill |
| Change controls | [`docs/CONTROLS.md`](docs/CONTROLS.md), then the `controls` skill |
| Export or deploy a policy | [`deploy/README.md`](deploy/README.md), then the `deploy` skill |
| Read or write a bundle consumer | [`deploy/BUNDLE.md`](deploy/BUNDLE.md) |
| Navigate the repository or run checks | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Skills

A skill is the playbook for one kind of task, kept at `.claude/skills/<name>/SKILL.md`. Claude
Code finds them there by itself; nothing else about them is Claude-specific. Whatever agent you
are, **before starting a task one of them covers, read its `SKILL.md` in full and follow it** —
each exists because the failures on its path are silent, and its steps are the checks that make
them loud. The `description:` at the top of each file decides whether it applies; the table only
points at it.

| skill | reach for it when |
|---|---|
| [`setup-env`](.claude/skills/setup-env/SKILL.md) | getting the Python environment and the controller's Rust toolchain running on a new machine, or repairing one that imports the wrong checkout, ignores the GPU or cannot build the controller |
| [`new-task`](.claude/skills/new-task/SKILL.md) | creating a training task — its directory, registration, configs and operator controls |
| [`train`](.claude/skills/train/SKILL.md) | training a policy and reporting it — sizing the run for the machine's backend, reading whether it learns, replaying it with the measurement on, and handing over a checkpoint with measured numbers rather than an assumed success |
| [`controls`](.claude/skills/controls/SKILL.md) | giving a stick, a button or a key a meaning — a task's `controls.yaml`, a mode switch in `deploy/manifests.json`, a recorded motion's `go` |
| [`deploy`](.claude/skills/deploy/SKILL.md) | taking a policy from a checkpoint to the robot — export, RKNN, the bundle, the cross-build, the board |
| [`bundle-manual`](.claude/skills/bundle-manual/SKILL.md) | translating a built bundle's `manual.en.json` into Chinese and packing it into the `.app` — after every `scripts/deploy.py` build |

A new skill gets a row here; `tests/test_layout.py` fails until it has one.

Their scripts need no agent framework, only Python and this repository:

- For bringing an environment up on a new machine, [`docs/AGENT_SETUP.md`](docs/AGENT_SETUP.md)
  is the procedure and `.claude/skills/setup-env/scripts/` holds two standard-library scripts
  that automate the mechanical parts.
- For putting a trained policy on the robot, [`deploy/README.md`](deploy/README.md) is the map
  and `.claude/skills/deploy/scripts/check_board.py` is a read-only preflight of the board.
- For binding a stick or a button to a meaning, [`docs/CONTROLS.md`](docs/CONTROLS.md) is the
  design and `.claude/skills/controls/scripts/write_controls.py` writes the file from
  `controller/vocabulary.json` and refuses a name that is not in it, because an invented
  control name parses, binds nothing and looks exactly like one that does not work. It
  arranges `sys.path` itself.

## The bundle

[`deploy/BUNDLE.md`](deploy/BUNDLE.md) is the format of what `scripts/deploy.py`
writes — one directory and its `.app`, from which every host takes its own part, file by
file — and what a host that loads one has to do. It is the document to work from when
writing or changing a consumer, in this repository or outside it.
