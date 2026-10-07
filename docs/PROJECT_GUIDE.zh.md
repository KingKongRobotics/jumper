<!-- tracks: PROJECT_GUIDE.md @ sha256:83337a5e4c8afaba -->

# Jumper 项目指南

[首页](../README.zh.md) · [English](PROJECT_GUIDE.md)

## 从一句话开始

环境安装、助手分流和产物检查见[Train 与 Design 工作流](WORKFLOWS.md)。
一句话启动工作流，过程中可能需要工具、算力或设计选择。外观包目前仅用于显示；导入地图支持
回放、不支持训练，Train 尚无 `.skin` 加载器。详见[当前集成边界](WORKFLOWS.md#current-integration-boundaries)。

## 试一试

需要 Python 3.10–3.13。使用 NVIDIA GPU 时安装 PyTorch 的 GPU 版本；没有 GPU 时安装 CPU
版本，并选择 native 后端。

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

外观和场景工作流使用独立的 [jumper-design 仓库](https://github.com/KingKongRobotics/jumper-design)。
AI 可以在线读取说明，实际执行时再准备独立的本地副本，详见
[Design 环境安装](WORKFLOWS.md#design-an-appearance)。动作训练不需要下载 Design 的 LFS 资产。

使用 CPU 训练：

```bash
python scripts/train.py --task jumper.tripod --backend native --device cpu --num_envs 64
```

[环境搭建指南](USAGE.zh.md#环境搭建)覆盖 Linux、macOS 和 Windows，也包含检查 Python、
PyTorch 或 vendored package 是否装错的方法。[教程](TUTORIAL.zh.md)会把一个策略从训练一路
带到 bundle。

## 接下来去哪里

### 你想训练一个策略

| | |
|---|---|
| [环境搭建](USAGE.zh.md#环境搭建) | 从一份全新克隆走到测试全部通过。 |
| [教程](TUTORIAL.zh.md) | 用一个完整示例走过训练、回放、导出和打包。 |
| [Colab Notebook](COLAB.zh.md) | 使用云端 GPU 训练，保存检查点并观看无窗口 MP4 回放。 |
| [手册](USAGE.zh.md) | 命令、任务、资产、场景、默认值、TensorBoard 和续训。 |
| [控制方式](CONTROLS.zh.md) | 手柄、键盘，以及每种模式如何使用它们。 |

### 你想把它放到机器人上

| | |
|---|---|
| [部署](../deploy/README.zh.md) | 从 checkpoint 到各个宿主，以及每一步在哪里检查。 |
| [Bundle 格式](../deploy/BUNDLE.md) | App 中的每个文件，以及宿主需要实现的契约。 |
| [ONNX 转 RKNN](../deploy/convert/README.md) | 把策略转换给板端 NPU。 |
| [Controller](../deploy/fsm/README.md) | 各宿主共用的 observation、动作解码和状态机。 |

### 你想修改这个项目

| | |
|---|---|
| [参与开发](../CONTRIBUTING.md) | 环境、测试、仓库结构，以及新代码应该放在哪里。 |
| [设计](DESIGN.md) | 框架为什么这样组织，以及实测结果说明了什么。 |
| [Vendored 代码](VENDOR.md) | 本地 mjlab、rsl_rl 副本的来源和改动。 |

## 内部原理

同一套任务配置、奖励和 PPO 代码运行在两个物理后端之上。MuJoCo Warp 是主要的 GPU 训练后端；
原生 MuJoCo 把多个环境分给不同 CPU 线程。框架只用一道很小的接缝替换仿真器，manager 和学习层
保持不变。

导出的策略会和 `deploy/fsm` 汇合。后者是同一个 Rust controller，分别为机器人板端、浏览器和
`play --app` 编译。一个 bundle 携带这些 runtime、每种模式的策略、控制映射和参考帧，让所有
宿主检查同一组输入与输出。

Jumper 本身位于 `assets/jumper/jumper.xml`，包含舵机曲线、dToF 传感器和机载相机。场景可以把
它放到摄影棚地面、崎岖地面、冰面、台阶或绳索秋千上。

## 项目状态

训练、仿真、导出和宿主侧 bundle 检查已经实现，并由测试套件覆盖。真实硬件是另一条边界：
RKNN 推理尚未在板端验证，1 kHz controller 循环也尚未真正驱动电机总线。当前硬件缺口记录在
[部署文档的 Open 部分](../deploy/README.md#open)，仿真器之间的差异记录在
[Known asymmetries](DESIGN.md#9-known-asymmetries)。

## 相关项目

- [mjlab](https://github.com/mujocolab/mjlab) —— 本项目所基于的 MuJoCo 训练框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) —— PPO
- [MuJoCo](https://github.com/google-deepmind/mujoco) 和
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) —— 两个物理后端

## 许可证

Apache License 2.0 —— 见 [`LICENSE`](../LICENSE)。第三方材料及其许可证列在 [`NOTICE`](../NOTICE)。
