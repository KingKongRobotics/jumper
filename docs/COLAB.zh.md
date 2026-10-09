<!-- tracks: COLAB.md @ sha256:ee37e579ae37dcc5 -->

# 在 Google Colab 中运行 Jumper

[首页](../README.zh.md) · [English](COLAB.md) · [Notebook](../notebooks/jumper_colab.ipynb)
· [快速开始和本次训练视频](../notebooks/README.zh.md)

用 Colab GPU 训练、查看进展、在 MuJoCo 中回放、导出 ONNX，再保存备份。
你的电脑只需要浏览器。

[在 Colab 中打开](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)，
保存副本并选择 GPU 运行时。链接和 Notebook 的默认值都使用本仓库的 `main`；高级源码设置
可以指定别的仓库或提交。

## 新训练或继续训练

- **新训练：** 基本设置 → 安装与 GPU 检查 → 训练 → 曲线、回放、导出和备份。
- **继续训练：** 选择备份 → 读取保存的设置 → 安装与 GPU 检查 → 恢复并继续训练 → 下载新备份。

基本表单包含 `WORKFLOW_MODE`、`TASK`、`NUM_ENVS`、`ITERATIONS`、`USE_DRIVE` 和
`DOWNLOAD_FILES`。新训练默认使用 `jumper.tripod`、256 个环境、500 次迭代。
迭代次数始终指本次要执行的更新次数，续训时也是新增次数。

续训时选择 `Upload ZIP` 或 `Drive checkpoint`。安装前读取清单，匹配任务、模型、
环境数、源码提交和核心包版本。清单缺失或配置不兼容时会停止。恢复的检查点与表单
分开保存，重跑设置或改摄像机不会清掉它。确实想重新开始时，选择 `New training`
并打开 `START_NEW_RUN`。

训练前会打印模式、任务、环境数、迭代次数和检查点。恢复带上优化器、学习率、观测
归一化、迭代计数和保存的课程状态。不保存物理环境和随机数状态，因此是继续学习，
不是逐步复现断线前的轨迹。

已有舞蹈（`RUN_DEMO`）和五轮短训练检查（`RUN_SMOKE`）都是可选项。短训练检查
使用所选任务、最多 256 个环境，单独保存，不会被自动当成主训练。已有舞蹈和短训练
成功都不代表自己的策略已经训练好。

## 源码和运行环境

高级源码设置包含 `REPOSITORY_URL`、`SOURCE_REVISION`、`CHECKOUT_FOLDER` 和
`CORE_VERSION_OVERRIDES`。克隆目录使用 `/content` 下的简单名称，默认 `jumper`。
只有远程地址和提交都一致时才复用已有克隆；换源码时使用新的目录或运行时。

备份记录的提交必须在所选仓库中存在：备份会记录它所用的仓库、提交和核心包版本，
续训会先检出那个提交再恢复，所以从 fork 做出的备份需要在 `REPOSITORY_URL` 里填那个
fork。核心包版本直接从备份读取，无需手动抄写。
手动覆盖只接受 `torch`、`mujoco`、`mujoco-warp`、`warp-lang`、
`numpy` 和 `tensordict` 的精确版本，不接受 URL、版本范围或 pip 参数。
支持的 Torch/torchvision 配对是 2.9.1/0.24.1，其他配对需另行验证。

安装使用克隆目录中独立的 `.venv`，不修改 Notebook 内核的包。Jumper 要求 Python
3.10–3.13。T4 使用官方 CUDA 12.6 Torch 包，支持的较新 GPU 使用 CUDA 12.8。
真实 CUDA 运算、编译后的 Warp 内核和 NVIDIA EGL 渲染都必须通过。没有 GPU、
Python 不支持或驱动不匹配时停止安装。图形检查通过任务目录里的 vendor JSON
选择已有 NVIDIA 图形库，不替换系统驱动。

GPU 和会话时长由 Colab 分配。提高 `NUM_ENVS` 前先看实际设备和显存；环境数也会
改变 PPO 批次大小。最初 T4 虚拟环境约 7.2 GiB，仓库约 580 MiB，不含虚拟环境和
日志。这是磁盘占用，不是下载流量，检查点、安装文件和视频还需要额外空间。

## 查看进展

运行目录为 `logs/<model>/<task>/<timestamp>/`，曲线读取所选运行的事件。
`metrics.json` 和 `metrics.csv` 保留全部标量标签与采样；内嵌 PNG 是摘要。
非有限值保留为明确字符串并显示警告计数，曲线在这些位置留空。

任务提供相应数据时，摘要包括命令课程 `level`、跟踪误差 `lin_err`、升级阈值
`lin_err_bar` 和线速度/角速度命令范围。最初的 `model_499.pt` 仍在等级 0，最终
线速度跟踪误差约 0.195，高于 0.105 的升级阈值。视频说明工作流程已经运行，
不代表步态收敛。结合奖励、episode 长度、loss、课程和回放判断进展；完成 500
次迭代本身不是验收。训练会采样带噪声的动作，回放使用 actor 的确定性均值，
仿真观测和重置条件仍可能有噪声或随机化。

## 回放和评估

高级回放设置包括时长、帧率、偶数像素尺寸、场景、摄像机距离和关节记录。
运行设置后重跑回放即可，无需重新安装或训练。任务和模型必须匹配检查点。
`task default` 保留任务自己的场景，换场景会改变地形或摩擦。摄像机距离为零时
自动取景。视频 FPS 不改变控制速度，控制和物理仿真频率跟随任务。

`SIM_COMMAND` 可选随机命令、前进、侧移、转向、站立或自定义机体坐标系速度。
固定命令必须在保存的训练范围内。Notebook 回放和评估默认匹配检查点命令范围，
使用已保存的课程等级，而不是
较高等级；回放记录实际范围、源码和仿真频率。这些命令选项适用于速度任务，
不支持的任务会被拒绝。

启用 `RUN_EVALUATION` 后，前进、侧移、转向和站立各运行 `EVAL_SECONDS` 秒，
生成跟踪和重置情况的 JSON。应对照目标命令和实际速度判断表现，它不会自动给
实体机器人判定合格。共享 CLI 提供 `--checkpoint-command-ranges`、
`--command VX VY YAW`、`--evaluation-out JSON` 和 `--replay-info-out JSON`；
不使用这些选项时，普通回放行为不变。

回放就是 `scripts/play.py --video`，仓库里唯一的 MP4 录像器，所以 Notebook 里看到的
和桌面上 `play` 显示的是同一个画面；相机开关见[使用指南](USAGE.zh.md)。
Notebook 内嵌显示 MP4。`RECORD_JOINTS` 还会保存 `measure.csv` 和 PNG，每控制步
一条记录，包含时间、关节绝对位置（rad）、速度（rad/s）、实际施加的执行器力矩
（N·m）和可用足端力（N）。这是仿真测量。Colab 不会打开桌面键盘/手柄窗口。
[快速开始](../notebooks/README.zh.md)保留了最初的视频。

## 导出、备份和中断

导出会检查任务和模型，再在新目录中保存 ONNX actor、`layout.json`、README 和
检查点副本。应把目录一起保存，合同包含观测/动作顺序、缩放和增益。板端转换和
实机使用仍需走独立的[部署流程](../deploy/README.zh.md)。

运行时消失前下载备份。ZIP 包含完整检查点、配置和日志，以及已有曲线、报告、
测量和匹配的回放/导出。缺少运行环境/EGL 报告，或曲线、回放、导出失败，不会阻止
检查点备份。其他检查点的旧产物不会混入。中断后从最近完整保存的检查点继续，
之后尚未保存的更新会丢失。只保存 Notebook 不会保存模型。

`USE_DRIVE` 可选挂载自己的 Drive，每次完整保存后复制检查点及清单；训练仍在
本地运行时磁盘进行。挂载需要你的授权，尚未实测。需要快照时，应在长训练前启用；
最终 ZIP 无法保护已经删除的运行时。关闭 Drive 时请自行下载 ZIP。
`DOWNLOAD_FILES` 控制自动下载，关闭后仍能预览和手动下载。

新会话中选择 `Continue training`，安装前读取备份。恢复不会替换已有运行，并会
拒绝不安全的 ZIP 条目、不完整检查点和不兼容的设置。续训写入新的时间戳目录。
CLI 参数和诊断见[使用指南](USAGE.zh.md)和[安装说明](AGENT_SETUP.md)。

## 实际验证过什么

这套流程在真实 Google Colab 会话的 NVIDIA Tesla T4 上跑过，环境为 Python 3.13、
PyTorch 2.9.1+cu126、MuJoCo 3.11 和 Warp 1.18，代码是这份 Notebook 合入前的版本。
这只说明链路能走通，不保证以后每一种 Colab 镜像或分配到的 GPU 都一样，更不说明
这样训出来的策略就是好的：[快速开始](../notebooks/README.zh.md)里那段视频对应的
500 次迭代 `jumper.tripod` 训练仍停在课程等级 0，线速度跟踪误差 0.195，高于 0.105
的升级阈值。

完整跑通的内容：256 个环境的短训练检查和 500 次迭代训练；ZIP 备份、恢复并以相同源码
和核心包版本再训练五次；从上传的 ZIP 续训的 64 环境运行；训练中途一次真实的 `SIGINT`
中断，随后备份、恢复并继续训练；ONNX 导出与 torch actor 的最大绝对误差低于 `1e-6`；
通过 NVIDIA EGL 录制的 MP4 回放和关节测量；四种固定命令评估；导出全部标量标签的
训练曲线。仓库的测试套件在同一环境里按文件分批跑过；两项失败在没有这份工作的基线上
同样复现（一项是姿态镜像几何检查，一项是 TensorBoard 对残留 `TIME_WAIT` socket 的
端口选择）。

Google Drive 挂载没有实测。ZIP 上传是在 Notebook 的上传边界用真实字节测试的，没有
经过浏览器的文件选择器。
