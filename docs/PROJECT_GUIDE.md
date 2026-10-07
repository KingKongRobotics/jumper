# Jumper project guide

[Home](../README.md) · [中文](PROJECT_GUIDE.zh.md)

## Start from a prompt

Follow [Train and Design workflows](WORKFLOWS.md) for setup, agent routing and package checks.
A prompt starts a workflow and may require tools, compute or a design choice. Skins are
currently display-only; imported maps work in replay but not training, and Train has no
`.skin` loader. See [integration boundaries](WORKFLOWS.md#current-integration-boundaries).

## Try it

Python 3.10–3.13 is required. On an NVIDIA machine, install the GPU build of PyTorch; without
one, install the CPU build and select the native backend.

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .

python scripts/train.py --list
python scripts/train.py --task jumper.tripod
python scripts/play.py --task jumper.tripod
```

Appearance and scene workflows use the independent
[jumper-design repository](https://github.com/KingKongRobotics/jumper-design).
An assistant can read its instructions remotely and prepare a separate checkout when needed;
see [Design setup](WORKFLOWS.md#design-an-appearance). Train does not require its LFS assets.

For CPU training:

```bash
python scripts/train.py --task jumper.tripod --backend native --device cpu --num_envs 64
```

The [setup guide](USAGE.md#setting-up) covers Linux, macOS and Windows, including the
checks that catch an incorrect Python, PyTorch or vendored-package installation. The
[tutorial](TUTORIAL.md) follows one policy all the way from training to a bundle.

## Where to go next

### You want to train a policy

| | |
|---|---|
| [Setup](USAGE.md#setting-up) | Bring a machine from a fresh clone to a passing test suite. |
| [Tutorial](TUTORIAL.md) | Train, replay, export and bundle a worked example. |
| [Colab notebook](COLAB.md) | Train on a hosted GPU, preserve checkpoints and watch headless MP4 replays. |
| [Manual](USAGE.md) | Commands, tasks, assets, scenes, defaults, TensorBoard and resuming. |
| [Controls](CONTROLS.md) | The gamepad, the keyboard and what each mode does with them. |

### You want to put it on a robot

| | |
|---|---|
| [Deployment](../deploy/README.md) | The path from a checkpoint to each host, and what is checked where. |
| [Bundle format](../deploy/BUNDLE.md) | Every file in an app and the contract a host implements. |
| [ONNX to RKNN](../deploy/convert/README.md) | Convert a policy for the board's NPU. |
| [Controller](../deploy/fsm/README.md) | The observation, action decode and state machine shared by every host. |

### You want to change the project

| | |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Setup, tests, repository layout and where new code belongs. |
| [Design](DESIGN.md) | Why the framework is shaped this way and what the measurements say. |
| [Vendored code](VENDOR.md) | The local mjlab and rsl_rl copies, their provenance and their changes. |

## Under the hood

The same task configurations, rewards and PPO code run over two physics backends. MuJoCo Warp
is the primary GPU trainer; native MuJoCo spreads environments across CPU threads. One small
seam swaps the simulator while the manager and learning layers stay unchanged.

The exported policy joins `deploy/fsm`, one Rust controller compiled for the robot board, a
browser and `play --app`. A bundle carries those runtimes, one policy per mode, the control
map and reference frames that let every host check the same inputs and outputs.

Jumper itself lives in `assets/jumper/jumper.xml`, with its servo curve, dToF sensor and
onboard camera. Scenes put it on a studio floor, rough ground, ice, stairs or a rope swing.

## Project status

Training, simulation, export and host-side bundle checks are implemented and covered by the
test suite. Real-hardware work is a separate boundary: RKNN inference has not yet been
verified on the board, and the 1 kHz controller loop has not yet driven the live motor bus.
The current hardware gaps are tracked in [Deployment — Open](../deploy/README.md#open); simulator
differences are recorded under [Known asymmetries](DESIGN.md#9-known-asymmetries).

## Related projects

- [mjlab](https://github.com/mujocolab/mjlab) — the MuJoCo training framework this builds on
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — PPO
- [MuJoCo](https://github.com/google-deepmind/mujoco) and
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) — the two physics backends

## License

Apache License 2.0 — see [`LICENSE`](../LICENSE). Third-party material and its licences are listed
in [`NOTICE`](../NOTICE).
