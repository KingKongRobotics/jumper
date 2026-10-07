<!-- tracks: COLAB.md @ sha256:aba33a8f09a3820d -->

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
`SOURCE_REVISION` 恢复清单记录的源码提交。包版本一致仍需要兼容的 GPU 驱动。

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

下面的结果来自此前 20 次迭代训练加五次迭代恢复训练的验收。新增的回放参数、
关节数据、完整标量导出与每次运行的附件打包，尚未完成新的 500 次迭代 T4
验收，不能把此前结果当成所有新增 Notebook 选项已经通过验证。

| 检查 | 实际结果 |
|---|---|
| 针对性测试 | 76 项通过：57 项 Colab/视频检查和 19 项译文检查。 |
| GPU 训练冒烟 | `jumper.tripod` 的 256 个环境完成五次 PPO 迭代。 |
| 训练和恢复 | 完成 20 次迭代；检查点快照和清单备份后，恢复到新的本地目录，再完成五次继续训练迭代。 |
| ONNX 导出 | 411 维输入、20 维输出的 actor 通过对比，最大绝对误差为 `3.58e-7`。 |
| 已有舞蹈视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成 500 步 MP4，通过本地解码和云端 `ffprobe` 检查。 |
| 新训练策略视频 | 使用真实 T4 GPU 上的 NVIDIA EGL，成功生成新训练策略的 500 步 MP4，并通过相同视频检查。 |
| 完整测试套件 | 顺序运行已经完成：1067 项通过、25 项跳过、3 项失败、140 条警告，用时 239.48 秒。 |
| Google Drive 挂载 | 本次关闭，未测试。 |

图形检查使用的是 NVIDIA EGL，而非 Mesa 软件渲染。安装流程通过任务目录中的
vendor JSON 选择已有 NVIDIA 图形库，没有更改系统驱动。备份和恢复结果验证的是
本地快照机制，不是 Google Drive 授权或远程持久化。
两段视频均为 301 帧、H.264、640 x 480、30 fps，时长 10.033333 秒；其中包含
初始画面，以及十秒仿真期间采样的画面。

完整测试套件并未全部通过。三项失败来自 Colab/视频变更之外的既有测试，分别
涉及姿态镜像几何、native 后端自动线程数的一致性，以及 TensorBoard 对残留
`TIME_WAIT` socket 的端口选择。相关核心代码和测试没有修改。三项失败均在
同一个 Colab 环境的上游基线上复现。线程数失败的复现顺序为先运行
`test_five_foot.py::test_the_operator_actually_closes_the_claw`，再运行
`test_resolve.py::test_backend_and_resolve_agree_on_the_default`：解析器返回
两个线程，而测试期待八个。这里记录的是顺序依赖，并未诊断其底层原因。
25 项跳过也不等于通过。

Ruff 0.16.10 对整个仓库报告了 346 项问题，上游基线为 350 项，对比没有发现
新增问题。对于本次改动的 Python 文件，仅报告 `scripts/play.py` 原有的
`EXE001`，该文件的 shebang 和可执行权限没有变化。这不意味着整个仓库已经
通过 lint 检查。

这些短训练说明已测试的训练、检查点、恢复和导出链路可用，并不证明策略收敛、
步态稳健或实机行为。完整套件的失败项仍是本次验证的已知局限，Google Drive
挂载没有实测。
