<!-- tracks: README.md @ sha256:7880c98fa5d5b090 -->

# 在 Colab 上运行 Jumper

[在 Colab 中打开](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)
· [English](README.md) · [完整指南](../docs/COLAB.zh.md)

用 Colab GPU 训练 Jumper，在 MuJoCo 中看回放，再把结果下载回来。
你的电脑只需要浏览器，不需要 WSL 或本地 GPU。

正式入口和 Notebook 默认值使用 `KingKongRobotics/jumper` 的 `main`。
PR 合并前，请打开[预览入口](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb)，在第一个表单中将
`REPOSITORY_URL` 设为 `https://github.com/tianrking/jumper.git`，
`SOURCE_REVISION` 设为 `codex/colab-training-and-video`。

## 开始使用

1. 打开 Notebook，保存副本，选择 GPU 运行时。
2. 在第一个表单中选择任务和参数，然后按顺序运行各单元。
3. 观看已有舞蹈，完成五轮短训练检查，再开始训练。
4. 查看曲线，回放自己的检查点，下载导出文件和备份。

安装在 Notebook 独立环境中进行，并检查真实 CUDA 和 Warp 执行。主训练默认
使用 `jumper.tripod`、256 个并行环境，训练 500 次迭代。舞蹈演示使用已有策略；
你自己的训练会产生另一份行走策略。

## 本次训练的视频

这是本次 Colab 训练 500 次迭代后，`model_499.pt` 的行走回放。视频是十秒真实
MuJoCo 仿真，640 x 480、30 fps，不是已有舞蹈演示。

https://github.com/user-attachments/assets/e0e3b700-d1f6-486f-be20-98d4b69639ea

## 调整回放，不用重新训练

在表单中设置仿真时长、帧率、偶数像素尺寸、场景和摄像机距离。先运行参数设置
单元，再运行训练后策略的回放单元，不需要重跑安装或训练。同一检查点的任务和
模型保持不变。
`task default` 使用任务自己的场景，也可以换场景尝试不同地形或摩擦条件。
Notebook 直接显示 MP4。这些设置改变回放，不修改已保存的策略。

打开 `RECORD_JOINTS` 后，回放还会保存关节位置、速度和力矩的 CSV 与图；有
接触传感器的任务也可以包含足端力。这些数据在仿真中测量，不是实机采集。

## 可以保存的文件

| 输出 | 内容 |
|---|---|
| 训练曲线 | 全部已记录 TensorBoard 标量的 JSON、CSV，以及内嵌显示的六面板 PNG 摘要。 |
| 回放 | MP4、仿真设置，以及可选关节 CSV/PNG。 |
| 策略导出 | ONNX actor、`layout.json`、README 和检查点副本，每次导出使用新目录。 |
| 备份 ZIP | 检查点、配置、日志、曲线、运行环境/EGL 报告及匹配的回放和导出。本次 500 次迭代备份约 116 MiB。 |

文件按每次运行保存，并与选定检查点关联，不会把旧视频或旧导出悄悄当成新结果
打包。下次会话可以恢复 ZIP，使用其中记录的源码和核心包版本继续训练。本次
备份已经成功恢复，并继续训练了五次迭代。恢复这份备份时，请使用
`tianrking/jumper` 中记录的 `19e8f4d` 提交，不要直接改用较新的 `main`；
压缩合并可能不保留原始提交。切换仓库时，使用新的运行时或克隆目录。

`DOWNLOAD_FILES` 控制浏览器自动下载。关闭后，预览和文件仍可手动查看、下载。
Drive 检查点快照为可选项，本次测试关闭了 Drive 挂载，尚未验证它。

## Colab 实测

流程已在 Tesla T4 上使用源码 `19e8f4d` 运行：五轮短训练检查、500 次主训练、
视频回放、500 行关节测量、51 个标量标签和 25,500 个采样、ONNX 导出、ZIP
备份、恢复，以及再训练五次迭代。此后 Notebook 更新了 imports、文档、上游默认入口和导出的任务/模型检查。

这说明已测试任务的工作流程可用，不代表所有任务都已验证、行走策略已经收敛，
或可以安全地在实体机器人上运行。参数说明和测试结果见
[Colab 指南](../docs/COLAB.zh.md)。
