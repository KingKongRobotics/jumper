<!-- tracks: COLAB.md @ sha256:3ef1d8c25383f177 -->

# 在 Google Colab 中运行 Jumper

[首页](../README.zh.md) · [English](COLAB.md) · [Notebook](../notebooks/jumper_colab.ipynb)
· [快速开始和本次训练视频](../notebooks/README.zh.md)

这个 Notebook 在 Colab GPU 上训练 Jumper，显示训练曲线，并用 MuJoCo 回放
你的策略，生成视频和关节测量数据。可以不重新训练就调整回放参数，导出 ONNX，
保存备份并在以后继续训练。500 次迭代训练及备份恢复流程已经在真实 Colab T4
上运行。你的电脑只需要浏览器，不需要本地安装、WSL 或本地 GPU。
[快速开始](../notebooks/README.zh.md)中包含本次训练的视频；视频展示的是仿真，
不代表实体机器人表现。

[在 Colab 中打开 Notebook](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)，
如果想保留自己的修改，先保存一份副本，再选择 GPU 运行时。在运行演示或训练单元
之前，先完成安装和检查。分配到的 GPU 会变化，应以检查单元报告的设备和显存为准，
不要假定一定能获得某一种 GPU。

正式入口和 Notebook 默认值使用 `KingKongRobotics/jumper` 的 `main`。
合并前，请打开 [PR 预览](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb)，在第一个表单中将
`REPOSITORY_URL` 设为 `https://github.com/tianrking/jumper.git`，
`SOURCE_REVISION` 设为 `codex/colab-training-and-video`。

## 选择参数

第一个表单选择任务和训练长度；主训练默认使用 256 个环境、500 次迭代。在主
训练之前，另有五次迭代的冒烟检查。同一个表单也控制后续仿真和下载：

| 设置 | 使用方法 |
|---|---|
| `TASK`、`NUM_ENVS`、`ITERATIONS` | 选择动作、并行环境数和本次新增训练迭代次数。 |
| `SIM_SECONDS`、`SIM_FPS` | 仿真时长和视频帧率，两者都必须是有限正数。 |
| `SIM_WIDTH`、`SIM_HEIGHT` | 视频像素尺寸，均须为至少二的偶数整数。 |
| `SIM_SCENE` | `task default` 保留任务自己的场景；选择其他场景会改变地形、摩擦等测试条件。 |
| `SIM_CAMERA_DISTANCE` | 零表示自动取景，正数指定摄像机距离，单位为米。 |
| `RECORD_JOINTS` | 在视频之外保存并显示仿真关节测量数据。 |
| `DOWNLOAD_FILES` | 控制浏览器自动下载；关闭后仍保留内嵌预览和可手动下载的文件。 |
| `CHECKOUT_FOLDER` | `/content` 下的单个简单目录名，默认 `jumper`；其他路径随所选名称变化。 |
| `CORE_VERSION_OVERRIDES` | 恢复训练时，粘贴备份清单中的 `config.runtime_versions` JSON。 |

版本覆盖只接受 `torch`、`mujoco`、`mujoco-warp`、`warp-lang`、`numpy` 和
`tensordict` 的精确版本，不接受 URL、版本范围或任意 pip 参数。支持的
Torch/torchvision 配对为 2.9.1/0.24.1，其他配对需要单独验证。同时要通过
`SOURCE_REVISION` 恢复清单记录的源码提交。仓库必须包含这个确切提交：
如果压缩合并后上游不保留它，恢复本次 `19e8f4d` 备份时请使用
`https://github.com/tianrking/jumper.git`。切换仓库时，使用新的运行时或新的
`CHECKOUT_FOLDER`。包版本一致仍需要兼容的 GPU 驱动。

## 先观看演示

按顺序运行到演示单元。Notebook 会把仓库克隆到 Colab 机器，在独立 `.venv` 中
安装项目，检查 CUDA 设备和 MuJoCo Warp 后端，并用已有 `jumper.dance` 检查点
渲染约十秒视频。后面的单元会直接显示 MP4，也可以把视频下载到自己的电脑。

第一次使用 Warp 可能较慢，因为需要编译内核。安装、编译、仿真和视频编码是不同
过程，第一次运行慢并不能说明之后训练循环的速度。某项检查失败时，先阅读输出，
解决该问题，再运行依赖它的单元。

Python 环境属于这个独立仓库。使用默认仓库目录时，Notebook 的 shell 命令使用
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

主训练完成后，曲线单元读取这一次运行的 TensorBoard 事件，把完整标量历史
导出到 `metrics.json` 和 `metrics.csv`，并内嵌显示 `training-curves.png`，
展示最多六条选定曲线。PNG 是摘要，JSON 和 CSV 保留全部已记录的标量标签与
采样。非有限值在 JSON 和 CSV 中保留为明确的 `NaN`、`Infinity` 或
`-Infinity` 字符串，并显示警告计数；曲线在这些位置留空，不填造数值。
判断策略进展时，应结合奖励、episode 长度和 loss 曲线。

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

新训练策略回放使用选定检查点，根据 `SIM_SECONDS` 和任务实际控制频率计算
控制步数。物理仿真频率与任务一致，保留观测噪声，并内嵌显示结果。Colab 不会
打开桌面键盘或手柄查看窗口。

启用 `RECORD_JOINTS` 后，回放增加已有的 `--measure` 选项，每个控制步记录
一条采样到 `measure.csv`：仿真时间以秒为单位，关节绝对位置为弧度，速度为
弧度/秒，实际施加的执行器力矩为 N·m，并记录关节与执行器名称。启用接触传感
的任务还可能记录以牛顿为单位的足端力。Notebook 显示 `measure.png`，并将
两份文件存入该次运行的附件。这是仿真测量，不是实机数据。

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

最终可移植备份 ZIP 包含所选运行完整写入的检查点、配置和日志，以及该次运行
单独的 artifact 目录。已生成的附件包括运行环境与 EGL 报告、完整标量 JSON/CSV
和曲线 PNG、仿真设置、关节 CSV/PNG。只有与选定检查点一致且成功生成的 MP4
或 ONNX 导出才会加入，重复执行单元不能混入旧检查点的视频或导出。如果画图、
录制或导出失败，可以运行备份单元，保存已完整写入的检查点和已有附件。

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
Notebook 明确选择检查点，每次导出创建新目录，重复执行不会覆盖前一次导出。
`--no-video` 避免再次录制任务的完整动作。启用 `DOWNLOAD_FILES` 时，下载
单元提供视频、关节 CSV 和最终备份 ZIP；关闭时，应在运行时结束前按打印出的
路径手动下载。

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

当前源码（`19e8f4d`）已经完成默认 500 次迭代 T4 运行，包括关节测量、标量
导出和每次运行的附件打包。下表记录已确认的工作流程与回归测试结果。

| 检查 | 实际结果 |
|---|---|
| GPU 训练冒烟 | `jumper.tripod` 的 256 个环境完成五次 PPO 迭代。 |
| 主训练 | `jumper.tripod` 的 256 个环境完成 500 次迭代，生成 `model_499.pt`。 |
| 恢复 | 恢复 ZIP 后，以相同源码和核心包版本从 `model_499.pt` 继续训练五次，保存 `model_503.pt`。 |
| ONNX 导出对比 | 411 维输入、20 维输出 actor 通过对比，最大绝对误差为 `7.15e-7`。 |
| 已有舞蹈视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成 500 步 MP4，通过本地解码和云端 `ffprobe` 检查。 |
| 新训练策略视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成新训练策略的 500 步 MP4，并通过相同视频检查。 |
| 关节测量 | CSV 有 500 行，包含位置、速度、实际施加的执行器力矩和足端力，配套 PNG 已生成。 |
| 训练曲线 | 导出 51 个标量标签、25,500 个采样，并生成六面板曲线 PNG。 |
| 可移植备份 | 121,887,490 字节 ZIP 通过 CRC 检查，包含本次运行的检查点和已生成产物。 |
| 回放参数变化 | 两秒、320 x 240、20 fps、studio 场景、摄像机距离 1.5 米、关闭关节记录的回放成功，生成 41 帧 H.264；恢复默认十秒设置后成功生成新的 500 行关节记录。 |
| 针对性检查 | 104 项全部通过，用时 6.89 秒。 |
| 全部测试文件 | 全部 62 个 `test_*.py` 文件按每个 pytest 进程最多四个文件，分 16 批顺序完成；JUnit 合计 1123 项，其中 1096 项通过、25 项跳过、2 项失败、0 项错误，各批用时合计 354.651 秒。 |
| Google Drive 挂载 | 本次关闭，未测试。 |

图形检查使用的是 NVIDIA EGL，而非 Mesa 软件渲染。安装流程通过任务目录中的
vendor JSON 选择已有 NVIDIA 图形库，没有更改系统驱动。备份和恢复结果验证的是
本地快照机制，不是 Google Drive 授权或远程持久化。
两段视频均为 301 帧、H.264、640 x 480、30 fps，时长 10.033333 秒；其中包含
初始画面，以及十秒仿真期间采样的画面。关闭关节记录时，之前的测量文件保存在
`replay-history`，并从该次回放附件中排除，不会被当成新测量展示。

单进程完整 pytest 运行在约 57% 处被系统以 `SIGKILL`（`-9`）终止，因此完成
的测试覆盖来自上述 16 批，而不是一次完成的单进程套件。分批结果中的两项失败
分别涉及姿态镜像几何，以及 TensorBoard 对残留 `TIME_WAIT` socket 的端口
选择；两项都在同一个 Colab 环境的上游基线上复现。跳过的测试不等于通过。

另一次基线顺序检查先运行夹爪操作测试，再运行解析器默认线程测试，也复现了
native 线程数返回二、期待八的差异。解析器在分批分支运行中通过，因此该基线
顺序结果不计为 1123 项分批合计中的第三项失败。独立进程运行不能证明所有测试
在同一个共享进程中可以共存。

这些运行说明已测试的训练、检查点、恢复和导出链路可用，并不证明策略收敛、
步态稳健或实机行为。两项分批失败、单进程中断和基线顺序依赖限制了回归证据
的范围，Google Drive 挂载没有实测。
