![Jumper](docs/media/jumper-hero-en.png)

# Jumper

**English** | [简体中文](README.zh.md)

**Jumper is a 22-DoF crab robot.** [View hardware →](docs/HARDWARE.md)

This repo is Jumper’s AI toolkit for appearance, motion, and scene creation. Open this repo in an AI coding assistant and try the prompts below.

> 🦀 **Get a free Jumper!** [Find out how →](https://beunlimited.me/zh/events/crab-robot-challenge-2026)

## One sentence to design an appearance

> Design a warm sand ranger appearance for Jumper with coordinated body and limb colors, then export a `.skin`.

| | | |
|:-:|:-:|:-:|
| ![Warm sand ranger appearance](docs/media/design-warm-sand.png) | ![Silver armor appearance](docs/media/design-silver-armor.png) | ![Raphael Turtle appearance](docs/media/design-raphael.png) |
| [**Warm sand ranger**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/warm-sand-ranger-integrated-v2.skin) | [**Silver armor**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/mecha-tripo-v3.skin) | [**Raphael Turtle**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/raphael-turtle-v1.skin) |

[Browse all skins](https://github.com/KingKongRobotics/jumper-design/tree/main/library/skins)

## One sentence to train a motion

> Train a stable tripod gait for Jumper, replay and evaluate the result, then package it as an `.app`.

| | | |
|:-:|:-:|:-:|
| ![Jumper walking](docs/media/walk.gif) | ![Jumper changing posture](docs/media/posture.gif) | ![Jumper waving](docs/media/gesture.gif) |
| **Walk** | **Posture** | **Gesture** |
| ![Dance simulation](docs/media/dance.gif)<br>![Dance website showcase](docs/media/official-dance.gif) | ![Jump simulation](docs/media/jump.gif)<br>![Jump website showcase](docs/media/official-jump.gif) | ![Grasp simulation](docs/media/claw.gif)<br>![Grasp website showcase](docs/media/official-grasp.gif) |
| **Dance** | **Jump** | **Grasp** |

## One sentence to create a scene

> Create a park pump-track scene for Jumper with rolling terrain, trees and benches, then export a `.map`.

| | | |
|:-:|:-:|:-:|
| ![Park pump track scene](docs/media/design-park.png) | ![Bedroom scene](docs/media/design-bedroom.png) | ![Soccer scene](docs/media/design-soccer.png) |
| [**Park pump track**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/park-pump-track.map) | [**Bedroom**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/bedroom.map) | [**Soccer**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/soccer.map) |

[Browse all maps](https://github.com/KingKongRobotics/jumper-design/tree/main/library/maps)

## Where to find things

| Resource | Purpose |
|---|---|
| [jumper-design](https://github.com/KingKongRobotics/jumper-design) | Appearance and scene creation; your assistant reads its [instructions](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md) and uses its tools as needed. |
| [Training tutorial](docs/TUTORIAL.md) | Motion training, replay and policy export in this repository. |
| [Colab training](docs/COLAB.md) | GPU training in a notebook, checkpoint backup, resume, ONNX export and MP4 simulation replay. |
| [Motion bundles](deploy/BUNDLE.md) | Package trained motions and their controller as an `.app`; see the [build guide](deploy/README.md) for prerequisites. |
| [Project guide](docs/PROJECT_GUIDE.md) | Setup, current capabilities and further documentation. |

Training builds on [mjlab](https://github.com/mujocolab/mjlab),
[rsl_rl](https://github.com/leggedrobotics/rsl_rl), [MuJoCo](https://github.com/google-deepmind/mujoco)
and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
Example appearance and scene images come from jumper-design; [image sources](docs/media/DESIGN_SOURCES.md)
and [third-party notices](NOTICE) record attribution.

## License

Copyright 2026 KingKong Robotics.

Maintainer-owned project materials are licensed under Apache-2.0. See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [licensing details](docs/PROJECT_GUIDE.md#license).
Third-party materials remain under their respective terms, and generated outputs do not
automatically inherit this repository's license.
