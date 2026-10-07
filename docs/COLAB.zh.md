<!-- tracks: COLAB.md @ sha256:99c28618f3a03269 -->

# 在 Google Colab 中运行 Jumper

[首页](../README.zh.md) · [English](COLAB.md) · [Notebook](../notebooks/jumper_colab.ipynb)

这个 Notebook 可以让你观看已有舞蹈策略、尝试一次小规模训练，并保存视频和策略
文件。你的电脑不需要安装 Jumper。视频展示的是仿真机器人，不代表实机表现。

[在 Colab 中打开 Notebook](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb)，
如果想保留自己的修改，先保存一份副本，再选择 GPU 运行时。在运行演示或训练单元
之前，先完成安装和检查。分配到的 GPU 会变化，应以检查单元报告的设备和显存为准，
不要假定一定能获得某一种 GPU。

这个入口目前指向 `tianrking/jumper` fork 中尚未合并的
`codex/colab-training-and-video` 分支。变更合并到上游后，可以把入口切换为
`KingKongRobotics/jumper` 的 `main`；合并前请使用上面的分支入口。

## 先观看演示

按顺序运行到演示单元。Notebook 会把仓库克隆到 Colab 机器，在独立 `.venv` 中
安装项目，检查 CUDA 设备和 MuJoCo Warp 后端，并用已有 `jumper.dance` 检查点
渲染约十秒视频。后面的单元会直接显示 MP4，也可以把视频下载到自己的电脑。

第一次使用 Warp 可能较慢，因为需要编译内核。安装、编译、仿真和视频编码是不同
过程，第一次运行慢并不能说明之后训练循环的速度。某项检查失败时，先阅读输出，
解决该问题，再运行依赖它的单元。

Python 环境属于这个独立仓库。Notebook 的 shell 命令使用
`/content/jumper/.venv/bin/python`，避免 Notebook 内核预装的包悄悄替代仓库内的
训练包。Jumper 要求 Python 3.10 至 3.13，即 `>=3.10,<3.14`。遇到不支持的
Python、CUDA 设备不可用或驱动与后端不兼容时，安装和检查应明确停止，不能把
用户要求的 GPU 训练悄悄变成 CPU 运行。

本次 Colab 实测中，`.venv` 占用约 7.2 GiB，项目本身约 580 MiB，不含虚拟
环境和训练日志。这是实测磁盘占用，不是下载流量；还需为检查点、视频和安装
临时文件预留空间。

已有检查点来自它们记录的任务和模型配置。较新的源码可能改变观测、动作或几何。
如果加载器拒绝示例，应保留错误并检查来源，不要为了加载成功随意改张量维度或
关节顺序。回放应用包里的预编译控制器，检查的是那份交付物；这不会构建当前
控制器源码，也不能证明它与当前物理模型兼容。

## 尝试训练

第一次训练使用 `jumper.tripod`、256 个并行环境和五次 PPO 迭代。这是 GPU
训练链路与检查点写入的冒烟检查，五次迭代不足以证明已经学到良好的行走策略。
演示视频使用已有训练检查点，不能把它当成这次短训练的成果。

冒烟训练成功后，可以修改 Notebook 的训练设置，运行更长时间。增加并行环境数
时要考虑当前 GPU 的显存。环境数也会改变 PPO 的有效批次大小，因此它既是显存
设置，也是训练选择。结合奖励曲线和新检查点的回放判断进展，完成一定迭代次数
本身不是步态验收标准。

在仓库根目录执行的短训练命令如下：

```bash
.venv/bin/python scripts/train.py --task jumper.tripod \
  --backend warp --device cuda:0 --num_envs 256 --max-iterations 5 --headless
```

训练记录写入 `logs/<model>/<task>/<timestamp>/`。回放和导出应使用 Notebook
从这次运行中选出的检查点，不要误用另一次运行里时间最新的文件。完整训练选项
见[使用指南](USAGE.zh.md)。

## 录制回放视频

Notebook 与本地回放使用相同的视频选项。在仓库根目录执行：

```bash
.venv/bin/python scripts/play.py --task jumper.dance \
  --checkpoint tasks/jumper/dance/out/example/model_2300.pt \
  --backend warp --device cuda:0 --num_envs 1 --physics-hz 200 --headless --steps 500 \
  --video out/colab/dance.mp4 --video-fps 30 --video-width 640 --video-height 480
```

这个任务的 500 个控制步相当于十秒仿真时间。`--video-fps` 设置录制帧率，不会
改变策略的控制频率。其他任务的控制周期可能不同，因此同样步数的视频时长未必
相同。`--physics-hz 200` 让舞蹈回放保持训练时的物理仿真频率。无窗口回放仍
需要可用的离屏渲染；Notebook 在 Colab Linux GPU 运行时中
通过 `MUJOCO_GL=egl` 请求 EGL。
安装流程会找到运行时已有的 NVIDIA EGL 图形库，并在任务自己的目录下写入
vendor JSON，不会替换系统 GPU 驱动。

| 选项 | 含义 |
|---|---|
| `--video PATH` | 把 MP4 写入指定路径。 |
| `--video-fps 30` | 视频录制帧率。 |
| `--video-width 640 --video-height 480` | 视频像素尺寸。 |
| `--video-env 0` | 多环境回放时，选择摄像机跟随的环境。 |
| `--video-distance 1.0` | 跟随摄像机距离，单位为米。 |
| `--steps 500` | 回放多少个控制步后结束。 |

无效的录制设置会被拒绝，而不是静默换成其他值。录制失败时，先检查渲染或编码
错误，再尝试提高分辨率或延长运行。MP4 能播放与策略已经训练有效，是两种不同
的结果。

## 在运行时结束前保存检查点

应把 Colab 机器上的本地文件视为临时数据。在运行时断开或被替换之前，下载需要
保留的内容。保存 Notebook 本身不会保存克隆的仓库、虚拟环境或训练检查点。

可选 Drive 单元需要你主动选择并授权挂载 Drive。启用后，训练备份会在每个
检查点写入完成后复制一份快照及其清单。训练、回放和导出仍在运行时的本地磁盘
上完成。检查点快照提供长期保存能力，无需把每次训练更新写入远程文件系统。

检查点应与运行配置及来源信息一起保存，包括任务、仓库提交、训练设置、实际
解析的后端和设备，以及包版本。Notebook 保存的运行元数据和快照清单用于提供
这些上下文。只有一个策略文件，无法确认它期待哪种机器人几何和观测布局。
如果需要这些快照，应在长训练开始前启用可选检查点备份；运行时丢失后就无法
再复制文件。

在后续运行时继续训练时，先重新安装并通过 GPU 检查，把已保存的运行文件恢复到
本地磁盘，再核对任务、源码版本和配置，选择对应检查点。张量形状相同并不能证明
几何或关节含义相同。命令行会同时恢复优化器和迭代状态，并把继续训练的结果写入
新的时间戳目录：

```bash
.venv/bin/python scripts/train.py --task jumper.tripod \
  --backend warp --device cuda:0 --num_envs 256 \
  --checkpoint /content/restored-run/model_4.pt --max-iterations 100 --headless
```

将示例检查点名称替换为实际保存的文件。`--checkpoint` 隐含恢复训练；
`--max-iterations 100` 表示再训练 100 次迭代，而不是总共训练到第 100 次。
兼容性检查失败时，应使用匹配的版本和配置，或者开始新训练。诊断信息见
[使用指南](USAGE.zh.md)和[环境安装说明](AGENT_SETUP.md)。

## 导出并下载

导出单元生成 ONNX actor、`layout.json` 合同、README 和检查点副本。应把整个
目录一起保存：合同中包含观测和动作顺序、缩放、增益及消费者所需的其他信息。

```bash
.venv/bin/python scripts/export.py --task jumper.tripod \
  --checkpoint /content/restored-run/model_4.pt --backend warp --device cuda:0
```

导出通过检查，说明本次主机侧转换经过验证；这不能证明实机 RKNN 推理、电机
总线时序或物理表现安全可靠。Notebook 的第一条工作流程到可下载的仿真与策略
文件为止。板端转换和部署应按照独立的[部署流程](../deploy/README.zh.md)进行。

## 验证边界

核心流程已经在真实 Google Colab 会话的 NVIDIA Tesla T4 上运行，环境为
Python 3.13、PyTorch 2.9.1+cu126、MuJoCo 3.11 和 Warp 1.18。以下结果对应
已测试分支的核心代码，不能保证未来每一种 Colab 镜像或分配到的 GPU 都相同：

| 检查 | 实际结果 |
|---|---|
| 针对性测试 | 53 项通过。 |
| GPU 训练冒烟 | `jumper.tripod` 的 256 个环境完成五次 PPO 迭代。 |
| 训练和恢复 | 完成 20 次迭代；检查点快照和清单备份后，恢复到新的本地目录，再完成五次继续训练迭代。 |
| ONNX 导出 | 411 维输入、20 维输出的 actor 通过对比，最大绝对误差为 `3.58e-7`。 |
| 已有舞蹈视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成 500 步 MP4，通过本地解码和云端 `ffprobe` 检查。 |
| 新训练策略视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成新训练策略的 500 步 MP4，并通过相同视频检查。 |
| 完整测试套件 | 新的一次顺序运行尚待结果，这里不宣称完整套件最终通过。 |
| Google Drive 挂载 | 本次关闭，未测试。 |

图形检查使用的是 NVIDIA EGL，而非 Mesa 软件渲染。安装流程通过任务目录中的
vendor JSON 选择已有 NVIDIA 图形库，没有更改系统驱动。备份和恢复结果验证的是
本地快照机制，不是 Google Drive 授权或远程持久化。
两段视频均为 301 帧、H.264、640 x 480、30 fps，时长 10.033333 秒；其中包含
初始画面，以及十秒仿真期间采样的画面。

这些短训练说明已测试的训练、检查点、恢复和导出链路可用，并不证明策略收敛、
步态稳健或实机行为。完整套件的结果仍明确待确认，Drive 挂载也需要单独验收。
