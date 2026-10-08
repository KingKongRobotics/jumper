<!-- tracks: README.md @ sha256:0ae3f849ff213d6c -->

# 在 Colab 上运行 Jumper

[在 Colab 中打开](https://colab.research.google.com/github/KingKongRobotics/jumper/blob/main/notebooks/jumper_colab.ipynb)
· [English](README.md) · [完整指南](../docs/COLAB.zh.md)

用 Colab GPU 训练，查看曲线，在 MuJoCo 中回放策略，再下载结果。你的电脑只需要浏览器。

合并前使用 [PR 预览](https://colab.research.google.com/github/tianrking/jumper/blob/codex/colab-training-and-video/notebooks/jumper_colab.ipynb)。
开始新训练时，在高级源码设置中选择 `https://github.com/tianrking/jumper.git` 和
`codex/colab-training-and-video`。正式入口和默认值使用上游 `main`。

## 开始使用

保存 Notebook 副本，选择 GPU 运行时，然后选一条路径：

- **新训练：** 选择任务、环境数和迭代次数 → 安装与 GPU 检查 → 训练 → 曲线、回放、导出和备份。
- **继续训练：** 选择上传 ZIP 或 Drive 检查点 → 读取保存的设置 → 安装与 GPU 检查 → 恢复并继续训练 → 下载新备份。

新训练默认使用 `jumper.tripod`、256 个环境、500 次迭代。续训时，迭代次数指新增的更新次数。
训练单元会先打印模式、任务、环境数和检查点。续训的源码与核心包版本来自备份；
修改回放设置不会清掉已恢复的检查点。已有舞蹈演示和短训练检查都是可选项，与主训练分开。

## 可以做什么

| 功能 | 结果 |
|---|---|
| 训练或续训 | 检查点、配置和日志。 |
| 查看进展 | 完整标量 JSON/CSV、曲线，以及任务提供的课程等级、跟踪误差和命令范围。 |
| 回放 | 内嵌 MP4；无需重训即可改时长、帧率、尺寸、场景和摄像机。可选关节位置、速度、力矩、足端力 CSV/PNG。 |
| 评估速度策略 | 固定前进、侧移、转向和站立命令，可选检查点的课程范围，输出 JSON 报告。 |
| 导出 | ONNX actor、`layout.json`、README 和检查点副本，保存到新目录。 |
| 备份 | 完整检查点和已有产物的 ZIP。曲线、回放或导出失败时，仍能备份检查点。 |

修改高级回放设置后，重新运行回放单元即可。任务和模型必须与检查点兼容。
视频和测量数据来自仿真；ONNX 导出之后，实机使用仍需走独立的[部署流程](../deploy/README.zh.md)。

运行时结束前下载 ZIP。只保存 Notebook 不会保存模型。可选 Drive 快照会在训练期间复制完整
检查点；Drive 挂载尚未实测。`DOWNLOAD_FILES` 控制自动下载，关闭后仍能预览并手动下载。

## 视频与实测记录

这段十秒回放来自真实 Colab T4 上使用源码 `19e8f4d` 训练 500 次迭代的
`model_499.pt`，是本次训练出的行走策略，不是已有舞蹈。该次运行仍处于课程等级 0；
完成训练次数不代表步态已经收敛。

https://github.com/user-attachments/assets/e0e3b700-d1f6-486f-be20-98d4b69639ea

该会话完成了训练、视频与关节记录、标量导出、ONNX、ZIP 备份、恢复和再训练五次。
ZIP 约 116 MiB。恢复这份旧备份时，记录的 `19e8f4d` 提交仍在 `tianrking/jumper`；
上游压缩合并可能不保留它。

表单重排、提前读取备份设置和固定命令评估是后续改动；新版 Colab 验证另行记录在
[完整指南](../docs/COLAB.zh.md#验证边界)中。
