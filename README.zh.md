<!-- tracks: README.md @ sha256:b30ad6626549dfc2 -->

![跳跳](docs/media/jumper-hero-zh.png)

# 跳跳

[English](README.md) | **简体中文**

**跳跳是一台 22 自由度螃蟹机器人。** [查看硬件 →](docs/HARDWARE.zh.md)

本仓库是跳跳的外观、动作与场景创作 AI 工具集。在 AI 编程助手中打开本仓库，试试下面的提示词。

> 🦀 **免费获得跳跳！** [了解如何领取 →](https://beunlimited.me/zh/events/crab-robot-challenge-2026)

## 本分支：跳跳写地书

![跳跳用毛笔在地上写自己的名字“跳跳”和“你好”（俯视）](docs/media/calligraphy-wu.gif)

在中国的公园里，人们用蘸水的长杆毛笔在石板地上写字——这就是地书，字迹干了便消失。
本分支在仿真中教跳跳做同样的事：它用钳子夹住毛笔，按笔顺一笔一笔写出自己的名字 **跳跳**，
再写 **你好**——然后走到一旁，转身面向字，跳一段舞，再鞠一躬，最后石面上的水迹渐渐干去。

- **笔画**来自 [Make Me a Hanzi](https://github.com/skishore/makemeahanzi)：
  取每一笔的中线，平滑后按比例铺到地面上，并带有压力曲线——起笔下压，收笔渐提。
- **每一笔都由手臂一气呵成，机身保持不动**，就像人在纸上写字：机器人在笔画之间收起手臂行走，
  到位后展开手臂；毛笔夹在合拢的钳子里，黑色笔头与笔杆同宽、收成笔锋，下压时会没入石面，笔画随之变粗。
  手臂按机身的实际位置用逆运动学驱动。
- **手臂写字时，腿像雕塑一样不动。**行走用跳跳的五足步态（`jumper.five_foot`）；
  一停下，它给腿的最后一条指令就保持住，直到手臂再次收起，所以手臂伸出时脚不会在地上蹭动。
- **实测**（影片所用的那次运行）：“跳跳 你好”每字 12.0 厘米，共 39 笔；墨迹离笔画中线
  0.9 毫米（中位数，p95 为 2.5 毫米），有两笔分两段写成；钳子始终离地 5 毫米以上。
  同一版本的另一次运行中：笔头没入石面约为其长度的四分之一（中位数；最多一半），
  写一笔时机身移动 0.2 毫米，脚始终离墨迹 6 厘米以上——机器人总站在所写之字的右侧，
  而这一行从左往右写。有一次在靠近机身处收笔时，钳子贴到机身正面约半秒。

动手试试——无需训练，写字也不需要 GPU：

```bash
git clone -b calligraphy https://github.com/AgusBM/jumper && cd jumper
python -m venv .venv && source .venv/bin/activate && pip install -e .
python tasks/jumper/calligraphy/tools/write.py --text "跳跳 你好" --layout horizontal   # 4 核 CPU 约 40 分钟
python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u8df3-u8df3_u4f60-u597d/<run>
```

`render.py` 会在该次运行的目录里生成 `film.mp4` 和 `result.png`（Linux 上有 NVIDIA GPU 时，
设 `MUJOCO_GL=egl` 会快得多）。其他汉字、各项选项，以及规划器、控制器、
渲染和供后期描墨的笔画数据，都在
[`tasks/jumper/calligraphy/`](tasks/jumper/calligraphy/README.md)。

## 一句话，设计外观

> 为跳跳设计一个暖沙色游侠外观，统一身体和四肢配色，并导出 `.skin`。

| | | |
|:-:|:-:|:-:|
| ![暖沙色游侠外观](docs/media/design-warm-sand.png) | ![银色装甲外观](docs/media/design-silver-armor.png) | ![Raphael Turtle 外观](docs/media/design-raphael.png) |
| [**暖沙色游侠**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/warm-sand-ranger-integrated-v2.skin) | [**银色装甲**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/mecha-tripo-v3.skin) | [**Raphael Turtle**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/skins/raphael-turtle-v1.skin) |

[浏览全部外观](https://github.com/KingKongRobotics/jumper-design/tree/main/library/skins)

## 一句话，训练动作

> 为跳跳训练稳定的三足步态，回放并评估效果，然后打包生成 `.app` 动作包。

| | | |
|:-:|:-:|:-:|
| ![跳跳行走](docs/media/walk.gif) | ![跳跳改变姿态](docs/media/posture.gif) | ![跳跳挥手](docs/media/gesture.gif) |
| **行走** | **姿态** | **手势** |
| ![舞蹈仿真](docs/media/dance.gif)<br>![舞蹈官网展示](docs/media/official-dance.gif) | ![跳跃仿真](docs/media/jump.gif)<br>![跳跃官网展示](docs/media/official-jump.gif) | ![抓取仿真](docs/media/claw.gif)<br>![抓取官网展示](docs/media/official-grasp.gif) |
| **舞蹈** | **跳跃** | **抓取** |

## 一句话，生成场景

> 为跳跳生成一个有起伏地形、树木和长椅的公园泵道场景，并导出 `.map`。

| | | |
|:-:|:-:|:-:|
| ![公园泵道场景](docs/media/design-park.png) | ![卧室场景](docs/media/design-bedroom.png) | ![足球场景](docs/media/design-soccer.png) |
| [**公园泵道**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/park-pump-track.map) | [**卧室**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/bedroom.map) | [**足球**](https://github.com/KingKongRobotics/jumper-design/blob/be74e0f2e5e2433d24a3a7b1c1aa0dbeae480356/library/maps/soccer.map) |

[浏览全部场景](https://github.com/KingKongRobotics/jumper-design/tree/main/library/maps)

## 相关项目与指南

| 资源 | 用途 |
|---|---|
| [jumper-design](https://github.com/KingKongRobotics/jumper-design) | 外观与场景生成；AI 读取其[工作说明](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md)，按需使用工具。 |
| [训练教程](docs/TUTORIAL.zh.md) | 本仓库中的动作训练、回放与策略导出。 |
| [动作包格式](deploy/BUNDLE.md) | 将训练好的动作及控制器打包为 `.app`；环境要求见[构建指南](deploy/README.md)。 |
| [项目指南](docs/PROJECT_GUIDE.zh.md) | 环境安装、当前能力和更多文档。 |

训练基于 [mjlab](https://github.com/mujocolab/mjlab)、
[rsl_rl](https://github.com/leggedrobotics/rsl_rl)、[MuJoCo](https://github.com/google-deepmind/mujoco)
和 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)。
外观与场景示例图片来自 jumper-design，出处见[图片来源](docs/media/DESIGN_SOURCES.md)，
第三方材料归属见 [NOTICE](NOTICE)。

## 许可证

Copyright 2026 KingKong Robotics.

维护者拥有权利的项目内容采用 Apache-2.0。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE)
和[许可说明](docs/PROJECT_GUIDE.zh.md#许可证)。第三方内容保留其各自的许可条款；使用本工具生成的文件不会自动继承本仓库许可证。
