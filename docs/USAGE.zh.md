<!-- tracks: USAGE.md @ sha256:1bfa25ca3ae20213 -->

# 手册

搭建环境、入口脚本、新增一个任务、新增一个模型、设置默认值 —— 每一项的完整版本。

[`../README.zh.md`](../README.zh.md) 是介绍 —— 这个仓库是什么、怎么组织、哪份文档讲哪件事。
操作说明都在本文，连同其中的理由、判断标准，以及大家实际会弄错的地方。设计上的取舍见
[`DESIGN.md`](DESIGN.md)。

本文里的每一个代码块都取自仓库里能跑的代码；下面的 `jumper.stairs` 例子就是换了个名字的
`jumper.flat` 的文件，数字也是它的。

---

## 目录

- [环境搭建](#环境搭建)
- [入口脚本](#入口脚本)
- [新增一个任务](#新增一个任务)
- [需要素材的任务：`jumper.dance`](#需要素材的任务jumperdance)
- [带第二个命令的任务：`jumper.posture`](#带第二个命令的任务jumperposture)
- [新增一个模型（asset）](#新增一个模型asset)
- [设置默认值](#设置默认值)
- [场景](#场景)
- [TensorBoard](#tensorboard)
- [继续训练](#继续训练)
- [命令速查表](#命令速查表)

---

## 环境搭建

每一个依赖都是 PyPI wheel。**不需要系统级的 CUDA toolkit** —— cu128 的 torch wheel 自带运行
时。在 GPU 机器上，NVIDIA 驱动是唯一的系统级前置条件。

> 如果是 AI agent，改把 [`AGENT_SETUP.md`](AGENT_SETUP.md) 交给它：同样的步骤，
> 写成了硬性的验证关卡。两个脚本在任何平台上都能做掉机械的那部分，而且不需要预先装任何东西 ——
> `python3 .claude/skills/setup-env/scripts/detect.py` 报告这台机器是什么、以及由此该跑哪些
> 命令，`gates.py` 检查结果。能加载 skill 的 agent 可以直接用
> [`.claude/skills/setup-env/`](../.claude/skills/setup-env/)。

**1. Python 3.10 – 3.13。** 用 `python3 -V` 查；如果更老，见下面那张表。

如果还没有克隆仓库：

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
```

**2. 在仓库根目录创建并激活一个 venv**：

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
```

最后一行必须打印出本仓库的 `.venv`。如果打印的是系统 Python 或某个 conda 环境，停下来 ——
后面每一次 `pip` 都会装到错的地方去。

**3. 安装 torch。** 有 NVIDIA GPU 的话，先确认 `nvidia-smi` 能列出显卡，然后：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

cu128 这个 index 必须显式写出来；Blackwell 显卡（RTX 50 系，`sm_120`）在 cu126 上跑不起来。
没有 GPU 就装 CPU 构建，并去掉 `--index-url`。

**4. 安装本仓库：**

```bash
pip install -e .
```

> ⚠️ **这是必需的，不是可选的。** [`rl/`](../rl) 下 vendored 的 `mjlab` / `rsl_rl` 只有通过这一
> 步建立的映射才能被 import。不做这一步，每一个入口脚本都会以
> `ModuleNotFoundError: No module named 'mjlab'` 失败。

> 🛑 **不要 `pip install mjlab` 或 `rsl-rl-lib`** —— 它们是 vendored 的（见
> [`VENDOR.md`](VENDOR.md)），pip 装的副本会以一种说不清的方式把它们遮掉。如果已经
> 装上了：`pip uninstall -y mjlab rsl-rl-lib`。

**5. 验证：**

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "import mujoco; from mujoco import rollout; print(mujoco.__version__)"
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
python -m pytest tests/ -q
```

在 GPU 机器上，第一条里应该出现 `cuda:0`。第三条必须打印出本仓库 `rl/` 里面的路径；打印出
`site-packages` 说明 pip 装的副本还在。

### 如果系统 Python 太老

| 平台 | 怎么做 |
|---|---|
| Ubuntu 22.04 / 24.04 | 自带 3.10 / 3.12；什么都不用做 |
| Ubuntu 20.04 及更老 | `add-apt-repository ppa:deadsnakes/ppa`，然后 `apt install python3.11 python3.11-venv` |
| macOS | `brew install python@3.12` |
| Windows | 用 [python.org](https://www.python.org/downloads/) 的安装器，勾上 "Add python.exe to PATH" |

在 Debian / Ubuntu 上，`ensurepip is not available` 意味着 venv 是一个单独的包：
`apt install python3-venv`。[uv](https://docs.astral.sh/uv/) 也可以：
`uv venv --python 3.11 .venv`。

### 故障排查

| 症状 | 原因与处理 |
|---|---|
| `pip install` 装到了别处 | venv 没激活；重新检查 `sys.prefix`。在 Windows 上，`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `torch.cuda.is_available()` 是 False | 装成了 CPU 构建，或者 CUDA 版本不对。在 Blackwell 上 `print(torch.__version__)` 必须显示 `+cu128` |
| Windows 上 Warp 报 MSVC 错误 | 安装 Visual Studio Build Tools 的 "Desktop development with C++" 工作负载（好几个 GB；先等报错出现再装） |
| macOS 上没有实时 viewer | 用 `.venv/bin/mjpython` 启动；直接用 `python` 会退回 headless，并且会说出来 |
| headless 下相机传感器报 GL 错误 | `MUJOCO_GL=egl`（不需要 `DISPLAY`） |

---

## 入口脚本

`train.py`、`play.py` 和 `export.py` 共用一个 parser（`scripts/_cli.py`），所以 `--task`、
`--model` 和后端那些 flag 在哪里的写法都完全一样。
**加一台机器人或一个任务，不需要动 `scripts/` 下的任何东西。**

| 入口脚本 | 用途 |
|---|---|
| `scripts/train.py` | 训练，带实时 viewer 和 TensorBoard |
| `scripts/play.py` | 回放训练好的策略，带实时 viewer |
| `scripts/export.py` | 导出训练好的策略：actor.onnx + layout.json + README + checkpoint |
| `scripts/deploy.py` | 构建 controller，并把它和它驱动的那些策略打成 bundle；它不读取环境，所以有自己的一个 parser |

### 共用参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--task <id>` | `MJRL_TASK` | 任务 id；可选值见 `--list` |
| `--model <name\|path>` | 任务的默认资产 | 已注册的资产名，或一个 `.xml` 文件的路径 |
| `--list` | — | 列出任务、它们的资产和场景，然后退出 |
| `--backend {auto,warp,native}` | `auto` | 显式指定时绝不回退 |
| `--device <auto\|cuda:0\|cpu>` | `auto` | torch 设备。Apple Silicon 装了 `metal` extra 之后，`cpu` 配 `--backend warp` 会把仿真放到 Apple GPU 上；物理和学习各在哪里由 `.env` 里的 `MJRL_SIM_DEVICE` / `MJRL_AGENT_DEVICE` 决定 |
| `--num_envs <n>` | train：`MJRL_NUM_ENVS`（随仓库发布的 `.env` 里是 4096）；为空时：warp 4096 / native 64。play：`MJRL_PLAY_NUM_ENVS`；为空时：1 | train：**会改变 PPO 的有效 batch size**。play：屏幕上有几台机器人 |
| `--cpu_threads <n>` | `0` | native 的线程数；`0` = 核心数，上限 8，再受 `num_envs` 限制。**不是全部核心** —— 超过约 8 之后 step 受串行部分制约，多出来的线程只会增加耗时 |
| `--strip-visual {auto,on,off}` | `auto` | 在 native 上剥离视觉网格；`auto` = 一旦每环境模型超过 2 GiB 就剥 |
| `--dry-run` | — | 解析并打印；什么都不构建 |

### `deploy.py`

构建 app：把 `deploy/fsm` —— 共用的 observation、动作解码和状态机 —— 为每种宿主各编译一份，
并和它驱动的那些策略打包在一起。一个 crate，三种宿主：浏览器加载 wasm，`play --app` import
那个扩展，机器人运行 `controller`。没有 `--task`：它不读取环境。

不带参数时，它构建 `deploy/manifests.json` 定义的那个 bundle，并先把机器人需要的东西做出来：
每次都在 docker 里交叉编译板端 controller，没有最新 `.rknn` 的策略在 docker 里转换。做完之后
仍然缺板端 controller、缺 `.rknn` 或缺 reference 向量的 app 会被拒绝。剩下唯一一步交给
agent，就是中文说明：由 `bundle-manual` skill 写出，再用 `--translate` 加进去。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--bundle [<dir>]` | `out/bundle_<timestamp>` | app 写到哪里：`<dir>/<name>/` 和 `<dir>/<name>.app`，所有宿主共用一个 |
| `--manifest [<name>]` | 唯一那个 | 构建 `deploy/manifests.json` 里的哪个 bundle：用哪些 export，用哪些控制 |
| `--manifests <file>` | `deploy/manifests.json` | 从另一个文件读 manifests |
| `--mode NAME=DIR` | — | `--manifest` 的一次性替代品：一个模式和它的 `export.py` 目录。每个模式写一次 |
| `--fsm <file>` | 自动合成 | 配合 `--mode`：要发布的那个 FSM。超过一个模式时必填 |
| `--allow-incomplete` | 关 | 跳过交叉编译和转换，有什么用什么，并在 notes 里写明缺了什么。用于浏览器、台架或测试 |
| `--require-clean` | 关 | 拒绝未提交的 controller 源码 —— crate 本身和它构建时读的每个文件，包括每个任务的钩子。默认关闭；未提交状态下的构建会在 `bundle.json` 里记下 `-dirty` |
| `--translate <dir>` | — | 给一个 bundle 的 `manual.en.json` 加一份译文（`--language zh --manual <file>`），先对照英文检查，再重新打包 `.app` |
| `--check-reference <dir>` | — | 用这台机器的扩展重放一个 bundle 的 `reference.json`，并报告这台宿主哪里不一样 |

```bash
python scripts/deploy.py
python scripts/deploy.py --check-reference out/bundle_<timestamp>
```

它以前还能把 wasm 单独发布到使用方的资产目录（`--out`、`--check`），以及把扩展装进
site-packages（`--python`）；app 里两样都带着，这几个参数已经删除。

`deploy/manifests.json` 是这次构建的输入，手工维护：放进哪些 export，以及哪个控制切换到
哪一个。是 export，不是 checkpoint —— checkpoint 不是机器人能跑的东西，而 export 会记下它
来自哪一个 checkpoint。它和一份 `.controller.toml` 配对，后者放优先级级联和安全限位，两者
不许说对方那一半 —— manifest 管模式和绑定，级联文件管其余的一切。以前写在命令行上的东西
现在写了下来，于是一个 bundle 记录下来的不只是它的内容，还有它的来历。

一次构建写出一个 bundle，即 `<name>/` 和旁边的 `<name>.app`，每种宿主从中取自己那一份：板子取
它的 `.rknn` 模型和 `runtime/board/controller`，浏览器取 ONNX 和
`runtime/web/controller.wasm`，`play --app` 取 ONNX 和为它所在平台构建的那个扩展。以前是三
个目录，每种宿主一个，里面逐字节相同地放着 FSM 配置和 contract，还要互相比对哈希来证明这一
点；一个目录不可能跟自己不一致，而 `jumper` bundle 现在是一个 10.9 MB 的 `.app`，以前三个
加起来 22.8 MB。格式见 [`../deploy/BUNDLE.md`](../deploy/BUNDLE.md)。

每个 bundle 都带一份 `reference.json` —— 录下来的一帧帧数据：进去的状态，出来的
observation、动作和关节目标 —— 这样三种宿主是对着同一组数字校验，而不是互相校验。
`--check-reference` 是这台宿主这一侧的做法；机器人那一侧是
`runtime/board/controller --bundle . --check-reference`。它还带一份 `manual.en.json`：每个按键和手柄
按钮在每个模式下做什么，由 controller 自己生成；`--translate` 再加上中文那一份。

### `train.py`

它自己的开关 —— viewer、logger、续训、TensorBoard —— 见
[只在命令行上的开关](#只在命令行上的开关)；其中 viewer 的那几个和 `--scene`
`play.py` 也有。

### `play.py` 和 `export.py`

两者都接受和 `train.py` 一样的 `--task` / `--model` / 后端 flag。

```bash
python scripts/play.py --task jumper.flat                       # 最新的 checkpoint
python scripts/play.py --task jumper.tripod --checkpoint <path>/model_4999.pt
python scripts/play.py --task jumper.flat --agent zero --num_envs 4   # 不需要 checkpoint
python scripts/export.py --task jumper.tripod --checkpoint <path>/model_4999.pt
python scripts/play.py --app out/bundle_<timestamp>/<name>.app   # 播放一个 app 的全部模式
```

`--app <app>` 播放一个 app —— 就是机器人和浏览器加载的那个 `.app` —— 走**部署**用的那个
controller，而不是任务的策略，也不用 checkpoint：所有模式都能用 app 自己的按键和按钮切进去；
viewer 的按键和手柄经由每个模式自己的控制定义读入；执行器跟踪的是 controller 自己的关节目标和
增益，所以它的切换斜坡和任务的部署钩子都会作用到关节上。环境只提供世界本身，默认用 app 默认
模式的任务（`--task` 可以另指）。这就是在不往任何地方上传的前提下、搞清楚一个 app
到底做什么的办法。它取代了 `--fsm`：那个参数把策略的原始动作交回 mjlab 解码，所以既看不到钩子，
也切不了模式（`rl/mjrl/app_play.py`）。

`play.py` 以回放模式构建环境（没有外部扰动，episode 不再按时长截断；一次性的跳跃保留它们的
episode 时长，所以回放会循环完整的一次尝试）。observation 噪声默认保持训练时的水平，
`--no-obs-noise` 给出干净的信号。省略 `--checkpoint` 时取 `logs/<model>/<task>/` 下最新的 checkpoint —— 和 `train.py --resume` 用的
是同一套解析，所以两条命令说“最新的那个”指的是同一个文件。`--agent zero|random` 不需要
checkpoint，用来看环境本身。回放默认只开一个环境，要更多就给 `--num_envs` 或设
`MJRL_PLAY_NUM_ENVS`：`MJRL_NUM_ENVS` 是训练的 batch size，play 不读它。

`export.py` 把 `actor.onnx`、`layout.json`、一份 README 和 checkpoint 的副本（跟踪录制的
任务还有 `<name>.trajectory.json`）写到 `tasks/<task path>/out/<date-time>/`
—— 每次导出一个目录，就在它出自的那个任务旁边，按日期命名而不是按 checkpoint 命名，这样同一
个 checkpoint 的第二次导出不会悄悄把第一次替换掉。它会先跑十项校验 —— 维度一致性、每一个
observation 项都是部署侧真的构建得出来的、其中没有哪一项是这台机器人测不到的量、home pose
对上 `init_state`、checkpoint 对上当前的配置、以及 ONNX 在形状和数值两方面都对上 torch 策略。

要在 Rockchip NPU 上跑那份导出，[`deploy/convert/`](../deploy/convert/README.md) 把它的
`actor.onnx` 转成 `.rknn`，并用 `export.py` 拿 ONNX 校验 torch 的同一种办法，拿转换后的模型
校验 ONNX。它自带一个 virtualenv —— rknn-toolkit2 把 `numpy`、`torch` 和 `onnx` 钉在了低于
训练所需的版本上。

---

## 新增一个任务

### 一切都依赖的那条规则

**一个 task id 就是它相对 `tasks/` 的模块路径**，用点号代替目录分隔符：

```
tasks/jumper/tripod/   <->   id "jumper.tripod"
tasks/jumper/dance/    <->   id "jumper.dance"
```

注册表里**不存任何配置值**。`load_env_cfg` 把 id 直接变成 `tasks.<id>.env_cfg` 并导入它。所以目录名
写错就意味着配置找不到 —— 而你拿到的是一条讲明这条约定、并指向 `tasks/registry.py` 的错误，
不是一个光秃秃的 `ImportError`。

### 三个文件，人来驾驶的任务再加第四个

以 `jumper.stairs` 为例，建 `tasks/jumper/stairs/`。一个人能驾驶的任务还需要第四个文件
`controls.yaml`，由 `controls` 技能来写：没有它 `velocity_env_cfg` 会报错，除非任务传了
`operator_command=False`。`new-task` 技能四个文件都会写。

**`__init__.py`** —— 只注册名字和可用的 asset，别的什么都不做。**不要在这里导入配置模块**：这个文件
在 `import tasks` 期间就会执行，而 `--list` 必须在一台没有仿真依赖的机器上照样可用。

```python
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stairs",                      # 必须和目录路径一致
    assets=JUMPER_ASSETS,                    # --model 可以选的东西；见下一节
    description="jumper hexapod on stairs",  # --list 显示的就是这一行
    tags=("locomotion", "jumper", "stairs"), # 自由格式的标签，用来过滤
)
```

**`env_cfg.py`** —— 环境配置。函数签名是固定的：`asset` 和 `play`，再加上任务自己的
`cli_args` 添加的关键字参数（比如 `jumper.swing` 的 `swing_angle`）。

```python
from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.velocity_env import velocity_env_cfg


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    Args:
        asset: 模型 XML 路径，来自 `--model`。None 表示用任务的默认资产。
        play:  回放模式 —— 关掉 observation 噪声和外部扰动，
               episode 更长。
    """
    cfg = velocity_env_cfg(
        controls=Path(__file__).parent / "controls.yaml",
        asset=asset,
        play=play,
        # 少写其中任何一个，骨架都会报错：每一个都是这个任务自己做的决定，
        # 不是继承来的默认值。见 DESIGN.md §14。
        foot_target_height=0.025,
        action_rate_weight=-0.1,
        pose_weight=1.0,
        command_lin_ceiling=0.8,
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )
    cfg.scene.terrain.terrain_type = "generator"   # 让这个任务与众不同的东西写在这里
    return cfg
```

**`rl_cfg.py`** —— **只属于这个任务**的超参数。改它不影响任何别的东西。

```python
from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.stairs")
    # 要偏离这条 baseline 就改这里，例如：
    # cfg.algorithm.entropy_coef = 0.005
    # cfg.max_iterations = 20_000
    return cfg


def runner_cls() -> type:
    """mjlab 的速度任务 runner（它会多记录一些跟踪指标），另外把课程等级存进 checkpoint，
    让 `--resume` 从同一级接着爬。"""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner
```

课程里有阶梯（按表现升级）的任务必须返回 `CurriculumRunner`：mjlab 自带的 runner 只存步数计数器，
`--resume` 会让阶梯从 0 级重新爬。任务写错了，`tests/test_curriculum_resume.py` 会报红。

**`mdp/`（可选）** —— 只有这个任务用到的 reward / observation / termination。判断标准是**有几个任务
需要它**：一个任务在用，就留在那个任务里；不止一个在用，就挪进 `tasks/<family>/common/mdp/`。三个步态
reward 正是照这条线拆开的。

### 最后一步：触发注册

往 [`../tasks/__init__.py`](../tasks/__init__.py) 里加一行：

```python
from .jumper import stairs  # noqa: F401,E402
```

**`scripts/` 下面什么都不用改**。

### 检查

```bash
python scripts/train.py --list          # 新任务应该出现在里面
python scripts/train.py --task jumper.stairs --dry-run
pytest tests/                           # 结构性不变量会检查这些目录约定
```

`tests/test_registry.py` 检查三件事：id 与目录路径对得上、每个目录三个文件齐全、磁盘上每一个任务目录
都真的被注册了 —— 忘了那行 import，任务就悄无声息地不存在。

### 超参数该共享还是每个任务各自一份

**各自一份**。每一个超参数、两个网络的 hidden dims，都是 `jumper_ppo_baseline` 的关键字参数，每个任务在
自己的 `rl_cfg.py` 里写出自己的那一份。四个运动任务今天写的值全都一样，是因为还没有人拿到过分开的实测
理由 —— 不是因为它们必须一致；其余的（`posture`、`five_foot`、`swing`、`dance`、`jump`、
`ref_free_jump`）在有了理由的地方各自偏离。

这里以前写的是四个任务共用一份 baseline，"因为它们互为对照组：把算法这一侧固定住，训练上的差异就能
干净地归因到步态先验"。**那不是真的**。这些任务的 reward 预算并不相同（正权重 6.00 / 7.50 / 7.00 /
7.00），而且 `jumper.tripod` 还多看到一项别的任务看不到的 observation，所以六种两两配对里只有两对 ——
`flat vs tetrapod` 和 `flat vs ripple` —— 才是只差一个变量。让这两对成立的从来不是超参数，超参数
不一样也不会让它们失效。实测的表在 `tasks/jumper/common/ppo.py` 里。

`tests/test_ppo_cfg.py` 要求四个运动任务把要紧的值写出来而不是继承下来，这样它们谁都不会悄悄捡到一处
为别的任务做的改动。新任务要加进这个测试的 `JUMPER_TASKS` 才受它约束。

---

## 需要素材的任务：`jumper.dance`

其他每一个任务描述的都是一台机器人和一个目标。`jumper.dance` 描述的是一台机器人和**一段具体的、录下来的
表演** —— 它的行为来自数据而不是来自配置，出错的位置也因此不同，所以在写下一个这样的任务之前，值得先
知道它是怎么摆放的。

仓库里提交了示例的编舞和脸部动画，共 30.5 MB，好让它和其他任务一样，从一份新 clone 就能跑起来。这是
一个有意的取舍：一个不先通过某条旁路拿到文件就启动不了的任务，就是一个没人会去检查它还能不能跑的任务。
这支舞所配的音乐**没有**提交 —— 来源无法确认 —— 训练也用不到它：只有 `scripts/export.py` 渲染表演
视频时才需要。

### 素材

素材放在 [`../tasks/jumper/dance/media/`](../tasks/jumper/dance/media/)，其中 `demo.npz` 和 `demo.mp4` 是提交进
仓库的，其余都被 git 忽略；查找**按扩展名**进行 —— 名字由使用者自己定，每种最多一个：

| 文件 | 用途 | |
|---|---|---|
| `<name>.npz` | 编舞数据 —— 训练唯一会读的文件 | 必需 |
| `<name>.{mp3,wav,m4a,ogg,flac}` | 这支舞所配的那段音乐；导出视频时需要 | 可选 |
| `<name>.{mp4,mov,mkv,webm}` | 面屏动画，导出时一并带上 | 可选 |

编舞所依据的那段参考录像**不**在其中，理由值得借用。它曾经是必需项，而从来没有代码打开过它。它的代价是
结构性的：查找按扩展名进行，于是第二个扮演另一种角色的视频让**两个**都变得有歧义，而面屏动画 —— 一个
真的会被用到的文件 —— 只好躲进 `eyes/` 子目录，才能跟一个没人用的文件分开。为了留档而要求一份素材，
换来的是一次冲突加一个子目录；那段参考录像于是和其余源素材放在一起。

仍然保留为必需的，是没有它就建不出环境的那些，也就是只有编舞数据。音乐和面屏动画都是可选的，因为训练
两者都不读，硬要求它们只会挡住那些编舞数据本身已经齐全的 clone。

不需要手工跑任何东西。第一次构建环境时会把这段片段转成 mjlab 的 `MotionLoader` 读得懂的格式 ——
重采样到 50 Hz 的控制频率、过一遍正向运动学得到每个 body 的位姿、速度用差分算出 —— 然后缓存到
`media/.cache/`，以源文件**的内容**为键。换掉 `.npz`，下一次运行就会重新转换；没有会被忘掉的步骤，
也不会有一份过期缓存让你训练上一支舞。

### 它会拒绝什么，以及为什么那才是重点

动作片段是数据，而数据的失败是无声的。关节顺序读错、按错误的 body 列表索引、或者重采样偏了一帧，它产出的
数组形状完全正确，里面的数字完全错误 —— 而训练在这些上面全都会收敛。所以转换在写出之前先检查：

| 检查 | 能抓到 |
|---|---|
| 关节限位 | 来自另一台机器人的片段，以及**任何一对被交换的腿** —— 左右限位互为镜像，所以被调换过的腿会被要求走出自己的范围 |
| 支撑脚共面性 | base 位姿与关节角来自不同的两次录制，或者地面高度不同 |
| 单帧连续性 | 两次录制被拼接在一起，或者掉了一帧 |
| 音乐时长 | 配错了曲子 —— 差得离谱就直接失败，只是裁剪长度不同则打印出来 |

每一项在 [`../tests/test_dance_motion.py`](../tests/test_dance_motion.py) 里都有一个对照组，因为
这类检查在没有东西被证明能让它失败之前，一文不值。

### 它有三处做法不同

**对称性增广是关的**。`jumper_ppo_baseline` 把它打开，四个速度任务也想要它 —— 六足是镜像对称的。一段
编舞不是：做镜像等于断言"参考动作抬左臂时，抬右臂同样正确"。它照样能训练，训练成这支舞和它镜像的平均。

**22 个关节全部参与 action**，而不是 `GAIT_JOINTS` 的 20 个。速度任务把夹爪一直闭着；这支舞会甩动它们。

**observation 是按部署侧命名的**，不是按 mjlab。`scripts/export.py` 已经预留了 `clip_phase`、
`ref_joint_pos`、`ref_joint_vel`、`ref_tilt_error` 和 `ref_future`；mjlab 的 tracking 任务把同样的
量叫作 `command` 和 `motion_anchor_*`，而这些不在那份白名单里 —— 直接采用它的 observation group，
得到的是一个训练得完美、却导不出去的 policy。

### 导出的不只是 policy，还有一段表演

任务可以定义 `export_media.py`，这是 `scripts/` 在不认识任务的前提下调用的几个可选钩子之一（另外两个
是 `cli_args` 和 `play_status`）：`scripts/export.py` 通过 `tasks.load_export_media` 调用它，而只知道
它可能存在。十个任务里有八个没有，对它们来说这次调用解析成 `None`，什么都不变；`jumper.posture` 的那个
在契约旁边写一份 `POSTURE_COMMAND.md`。

`jumper.dance` 用它把训练好的 policy 跳完整段片段渲染出来，把音乐混回去，并把音轨和面屏动画一起拷在旁边：

```
tasks/jumper/dance/out/<date-time>/
├── actor.onnx   layout.json   README.md      导出的策略
├── model_<n>.pt                              它出自的那份权重
├── demo.motion.trajectory.json               它被对着打分的那段片段
└── media/       dance.mp4  music.mp3  eyes.mp4
```

那条 trajectory 不是这道接缝的功劳：任何挂了 `reference_contract` 的任务都会得到一份，`jumper.jump`
也在内。一个 observation 里含参考轨迹的 policy，没有参考轨迹就跑不起来，所以它跟在 ONNX 旁边一起走，
而不是假定板子上本来就有。

`media/` 之所以是个子目录，是因为 `out/<date-time>/` **就是**那个导出的 policy —— `deploy/` 读这个
目录来组装板子加载的 bundle，一个 40 MB 的视频没有理由跟着跑这一趟。`--no-video` 跳过渲染，渲染要花
几分钟；导出本身仍然只要几秒。

它内部有三个决定值得借用：

- **rollout 记录的是 `qpos`；渲染在之后离线进行**。从一个活着的 env 里取像素，需要给 mjwarp 一条路径、
  给 native 另一条（在 native 里 `_datas[i]` 不是第 `i` 个环境）。通过 `Entity` API 记录，意味着
  渲染器永远看不到 backend。
- **在画出一帧之前先校验重建结果**。四元数约定或者关节顺序错了照样能渲染 —— 一台看着像那么回事的机器人，
  跳着 policy 从没跳过的舞，而下游没有任何东西会抱怨。所以 `qpos` 会在一份临时 `MjData` 上过一遍
  `mj_forward`，把 body 位置和活着的环境里的那一份对比；并且如果*没有一个* body 名字解析成功，这个
  检查就判失败，而不是因为什么都没比而通过。
- **音乐的偏移量是从片段里读出来的**。这段编舞开头有 2.0 s 的静默引子（`audio_start_in_sim`，在全部
  496 拍上都精确）。从零开始混音会让表演早两拍半 —— 早得刚好还能让人信以为真。

如果 policy 中途摔了，视频就停在那里，导出会把这件事说出来。

### 把这套模仿机制用到另一个任务上

这套机制是 mjlab 的 `mjlab.tasks.tracking`，一份 BeyondMimic 的重新实现，躺在 vendored 目录里，在
这个任务之前没有任何东西用过它。它只被导入，从不被修改：`rl/` 与上游保持逐字一致，所有与机器人有关的
东西都住在 `tasks/jumper/common/dance/` —— 片段加载器、观测项的名字、扭矩峰值项，以及环境本身
`dance_env_cfg`。

这台机器人的另一支舞就是一个任务目录：片段放在 `media/`，`env_cfg.py` 把每一个数值都交给
`dance_env_cfg`，因为这个构建函数没有默认值。`jumper.dance_brazilian`、`jumper.dance_dream_wings`、
`jumper.dance_waist` 和 `jumper.dance_maze` 正是这样 —— rl-wbc-fsm 播放的另外四支舞，由
`tools/import_wbc_dances.py` 导入，它的 docstring 写明了什么被当作状态、机身位姿是怎么解出来的。
`jumper.gesture_hello`、`jumper.gesture_bow`、`jumper.gesture_paw` 和 `jumper.gesture_salute`
也是这样 —— rl-wbc-fsm 放在十字键上的四个一次性动作，由 `tools/import_wbc_gestures.py` 导入：
按板上的播放速度计时，从 `HOME` 缓入、再缓出回到 `HOME`，并且用六只脚而不是四条腿来定地面。
换一台机器人则复制 `common/dance/`，而不是去动 `rl/`。

---

## 带第二个命令的任务：`jumper.posture`

四个步态任务都只有一个命令，mjlab 的 `twist` —— 往哪里走（`jumper.five_foot` 有它自己的一个
`body_pose`）。`jumper.posture` 加了第二个，
`posture` —— 走的时候身体怎么摆 —— 值得从它身上借走的是这个模式，而不是那四个具体的数。

```bash
python scripts/train.py --task jumper.posture --model jumper
python scripts/play.py  --task jumper.posture
```

### 它命令什么

| 通道 | 含义 | 范围 |
|---|---|---|
| `twist` | 机身在自己的脚上方偏航 | 站着 ±30°，走着 ±15° |
| `pitch` | 低头为正 | 站着 ±20°，走着 ±15° |
| `roll` | 左侧抬起为正 | ±15° |
| `height` | 机身离地高度 | 0.07–0.15 m（站立时是 0.107） |

速度命令停着（它的 `[vx, vy, wz]` 范数小于 0.06）时，每个角度从**站立档**里抽，否则从行走档里抽。
两档和符号都照 `jumper.five_foot` 的来，这是有意的：两个任务的姿态命令走的是机器人上同一组通道、
同一套摇杆布局，满偏必须在两边代表同样的倾斜；控制器在所有平台上都会把行走中的机器人收回到行走档。
俯仰在 2026-09-26 之前是抬头为正，那之前训出的 checkpoint 读到的俯仰是反的。

每 3–8 秒独立重采样一次，保持到下一次重采样，行走和站立时都生效。无论在走还是停着，都有五分之一的环境
被精确钉在中性姿态上，因为在四个轴上均匀抽样几乎永远抽不到"水平、端正、站着"，而三角步态恰恰是在这个
姿态下学出来的。（停着的那五分之一在 2026-09-29 之前是零，这样训出的策略在什么指令都没有时，中腿会偏离
HOME；任务的 `env_cfg.py` 里有实测。）

表里的范围是**阶梯的顶端**，不是一次运行的起点。`PostureRangeCurriculum` 在行走档上依次爬 ±7.5°、±10°、±15°
—— 站立档和高度按各自范围的相同比例跟着爬 —— 并且只在**每一个**轴都跟踪到门槛以下时才晋级，所以最慢的那个轴
决定节奏，日志也会说出是哪一个（`Curriculum/posture/<axis>_err` 对 `<axis>_err_bar`）。`std` 随每一级
缩放，所以窄的一级是更小的要求，而不是一把更松的尺子；`tests/test_posture.py` 用一台从不倾斜的机器人
白拿的分数把这一点钉住 —— 那个分数在整条阶梯上必须保持平坦。

门槛来自测量而不是猜测 —— 128 个环境全部命令为中性、随机动作，这就是"机器人自己会对自己做什么"的
参照：

```
sigma   |twist|   |pitch|   |roll|    |height - 0.107|
 0.0     0.127°    0.269°   0.144°       3.57 mm
 0.8     1.137°    1.542°   1.136°       5.02 mm
```

sigma 为 0 的那一行是**偏置，不是底噪** —— 一台被动的机器人会下沉约 3.6 mm，策略会把它推回去 ——
所以底噪是噪声在此之上额外加的部分，约 1.3° 和 1.4 mm。

第一级就是这个测量定下来的。在 ±5° 时门槛是 2.0°，对比"从不尝试"的 2.5° 和噪声的 1.3°，所以 level 0
测到的大部分其实是 PPO 的探索 —— 和速度阶梯在它自己的 0.15 那一级上发现的毛病一样，当时的做法是去掉
那一级而不是放松门槛。在 ±7.5° 时门槛是 2.5° 对 3.75°，现在两个轴上的每一级要求的都是"忽略命令"所得
的 42–67%。

它走的是三角步态，**步频跟着命令走**，步幅固定为 72 mm：`f = clamp(v / (2 × 0.072), 2.0, 5.56) Hz`。
在 200 Hz 控制下，上限是每周期 36 个控制步，下限是 100 —— 两者都是整数而且都是**偶数**，这是
`tasks/jumper/tripod/mdp/phase.py` 要求的，否则两组三角腿分到的时间不相等。为什么是 72 mm 而不是
看起来更自然的 60，见 `tasks/jumper/posture/mdp/cadence.py`。要求一个步频给不出的速度，什么都不会
报错；跟踪奖励只是变得拿不到。

### 在 `play` 里驾驶它

两个命令都归操作者，用手柄或者键盘都行。任何一个单独都能驾驶全部七个通道，最后碰过的那个说了算：

```
手柄                                                键盘
左摇杆          行走：前/后、左/右                  W S A D，或方向键
右摇杆          上/下：低头/抬头                    I K
                左/右：先扭身，再转向               H ; 扭身，J L 转向
R3（按住）      左/右：横滚                         U O
                上/下：高度，是速度                 N M，按住：高 / 低站姿
R3（点按）      高度回到站立高度                    （松开 N 或 M）
B               把两个命令都交还                    Esc
（松手）        站定、水平、端正，                  （松开按键）
                停在最后设定的高度
```

**右摇杆的左右是先后两件事。** 行程的前一半让身体在踩定的脚上扭转；过了半程，转向从零开始增加，扭转同时
往回退，到四分之三处回到零，这时转速正好是最高转速的一半 —— 身体先朝着将要转的方向领过去，在转快之前又
摆正。**按住右摇杆，它就是另一层**：按住 `R3` 时，左右让机身横滚，上下移动高度，扭转、转向和俯仰都是零；
松开 `R3`，立即恢复扭转、转向和俯仰。**在手柄上，高度是被移动的，而不是被摆到某处的**（2026-09-29
提出）：摇杆是速度 —— 推满时一秒从站立高度走到范围的任一端，即 `integrate_s` —— 松手后机身就停在那个
高度。不碰摇杆、
点按一下 `R3`，高度回到站立高度（`reset: R3`），`B` 也会。当天早些时候，高度先是在扳机上，后来在 `R3`
和摇杆上、作为一个位置，`R3` 一松开就回到站立高度。

**键盘是一条独立的路。** 每个键绑在一个轴的一个方向上 —— `"+"` 是这个轴自己的正方向，所以键盘没有可以
弄错的正负号 —— 按住时线性地推向满，按住文件里的 `full_after_s`（2.0 秒）后到满，键一抬起立刻回到
静止值 —— 高度也一样：`N` 和 `M` 是高、低两种站姿，松开后机身回到站立高度。布局来自机器人的操作指南
Control-agent 3.1（2026-09-29 修订版）：`J` 和 `L` 转向，`H` 和 `;` 扭身，都在右手下的一排；Esc
放掉一切。在那之前，键盘是一个虚拟手柄，每个键是一个摇杆方向、走手柄的映射，按住 `M` 就是 `R3`；那样
没法表达"`J` 转向、`H` 扭身"，而一个按手的习惯排布的键盘需要这样。

这些都在 `tasks/jumper/posture/controls.yaml` 里：`command` 是两个命令项的列表，一条手柄绑定可以只占
摇杆的一段行程 —— 再多写两个数还能把它交还出去（`travel: [0.0, 0.5, 0.5, 0.75]`）—— 或者放在 `R3`
那一层上，一个轴可以被移动而不是被摆到某处（`integrate_s`）；`load_controls` 会拒绝一个漏掉某个命令轴、
又不说明原因的键盘。`tasks/jumper/common/mdp/operator.py` 用它驱动 `play`，`deploy/fsm/src/operator.rs`
用它驱动机器人和浏览器；两个命令项共用一个操作者，所以 `B` 一次把两个都交还。四个行走任务的那一个命令
用的是同一套布局 —— `W S A D` 或方向键，以及 `J L` —— 由 `controls` skill 的脚本写出。格式见
[`CONTROLS.md`](CONTROLS.md)。

**上面每一个字母也会切换 viewer 的一个开关**（W 线框、S 阴影、A 自动连接、D 静态体、I 惯量、K 天空盒、
J 关节、L 叠加、U 执行器、O 扰动对象、N 岛、M 质心、B 扰动力）。MuJoCo 在 C++ 里处理自己的快捷键，
并且**额外**调用用户回调，而不是代替它，所以从 Python 这边没有什么可以拦截的。机器人照按键行动，画面
跟着闪；再按一次那个字母，开关就恢复。

### 为什么是第二个命令项，而不是再加四个通道

`twist` 在整个仓库里是按位置读取的 —— `moving_gate` 取 `cmd[:, :3]`，步态奖励、课程、teleop 和
gamepad 的子类、`scripts/export.py` 的契约都假设它是一个三维向量。把它加宽，会对其中每一处都造成一次
静默的改变。第二个命令项的代价是一个观测，对每一个不挂载它的任务都毫无影响。

### 一个新命令要挤掉什么

最费心的部分不是加项，而是找出那些已经在对一个固定参照系测量同一个量的项：

| 项 | 为什么必须拿掉 |
|---|---|
| `upright` | 为 pitch = roll = 0 付钱，而命令范围的一半都在反着要。换成 `track_tilt`，在零命令下就是同一个函数。 |
| `pose` 的 `std_standing` | 比行走时的宽度紧六倍，而保持一个被命令的姿态**本身**就是在离开 HOME。放宽到行走时的值——但只在离开中性姿态时：在中性姿态下 HOME 仍然是正确答案，而按行走宽度约束的静止机器人在那里停在了离 HOME 0.37 rad 的地方。静止时的宽度从中性姿态处 `jumper.tripod` 的站立宽度，渐变到离中性十分之一处的行走宽度。 |

这是 `jumper.swing` 在第二个任务上的教训：一个按你想要的东西命名的项，却在对一个机器人已经不在其中的
参照系做测量。无论如何它每一步都会产出一个数。

### 上板和进浏览器

它和其他任务一样能导出、能打包。这个任务新增的三样观测 —— `posture_command`、隔四步取的五帧历史、
步频跟着命令走的时钟 —— `deploy/fsm` 都会构建；`play --fsm --fsm-diff`（后来被不做观测比对的
`play --app` 取代）在 mjlab 里逐项比对，除了
`joint_pos` 上训练专用的编码器偏置，每一项都对到浮点精度。这个时钟的契约还写明了它**何时**推进，
即 `params.advance`，因为这个任务改过一次推进顺序：改动之前的导出仍按它训练时的顺序运行；改动之后的
导出在 26 次步频变化中与 mjlab 的 `gait_phase` 相差不超过 5e-6 度 —— 而把旧顺序强加到同一份导出上，
则差 6.3 度（用的是 `play --fsm` 自己的循环、一个环境、每 0.5 s 重采样一次的行走命令；`play` 自带的
命令大多停在步频上限，那里两种顺序是一致的）。`deploy/manifests.json` 的 `jumper` bundle 把它作为
默认的行走模式。

载入这个 bundle 的网页不保存这套操控的任何副本。它把每个按键（按浏览器的名字）、手柄（屏幕摇杆也算）
和它自己的「停止」原样交给 wasm，每一样是什么意思由里面的控制器决定 —— 就是机器人上跑的那个
`operator.rs`（见 `deploy/fsm/BUNDLE_README.md` 的 "Handing over a person's input"）。

### 精度级

和速度阶梯一样，姿态阶梯最后也用更紧的 `std` 重复一次顶端范围：第 3 级是满范围，`std` 取第 2 级的一半，
即 3.75 度和 10 mm。它是在有了测量之后才加的 —— 停在第 2 级的策略对每个指令都按固定比例欠跟踪，扭转差
16–21%，小横滚差到一半；而行走时步态本身带来的波动只有 0.2–1.0 度。表格见
`mdp/curriculum.py::POSTURE_STD_SCALES`。

---

## 新增一个模型（asset）

### 三样东西，三个地方

| 什么 | 放在哪 | 判据 |
|---|---|---|
| 模型文件、网格 | `assets/<name>/` | 纯数据 |
| 生成 / 标定它的脚本 | `assets/<name>/tools/` | **绑死在这个 asset 上**，与它共存亡 |
| 描述它的 Python 常量 | `tasks/<family>/common/` | 关节名、HOME 位姿、碰撞方案、`EntityCfg` |

最后一行是最容易放错的那一行：`constants.py` 描述的是那份 XML，但它 import 了 mjlab，又被任务引用 ——
它是代码，不是数据。

### 声明 asset

asset 表放在一个**轻量模块** `assets.py` 里，这个模块**不能 import mjlab**：

```python
# tasks/jumper/common/assets.py
from __future__ import annotations

from pathlib import Path

from ...paths import ASSETS_DIR
from ...registry import AssetSpec

JUMPER_XML: Path = ASSETS_DIR / "jumper" / "jumper.xml"

JUMPER_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        name="jumper",                              # --model 上用的就是这个名字
        path=JUMPER_XML,
        description="22-DoF heterogeneous hexapod",   # --list 会显示它；保持它准确
    ),
    # 后面继续加资产；第一个是默认值
    AssetSpec(name="jumper_v2", path=ASSETS_DIR / "jumper" / "jumper_v2.xml",
              description="variant with longer shanks"),
)
```

**为什么它必须轻**：任务的 `__init__.py` 在 `import tasks` 期间就需要 asset 路径。从 `constants.py`
去拿会把 mjlab / torch / mujoco 一起拉进来，让 `--list` 变重。照现在这样，跑完 `--list` 之后
`sys.modules` 里**一个重依赖都没有**（没有 mjlab、torch、mujoco 或 warp）—— 这就是它买到的保证。
`constants.py` 反过来从这里取 `JUMPER_XML`，所以路径只有一个真值来源。

### `--model` 的三种形式

```bash
python scripts/train.py --task jumper.flat                          # 默认资产（表里的第一个）
python scripts/train.py --task jumper.flat --model jumper_v2          # 按注册的名字
python scripts/train.py --task jumper.flat --model /path/to/a.xml   # 一个路径
```

路径这种形式是**留给实验的后门**：一个任务的关节名、HOME 位姿和足端 geom 名都是照着它的默认 asset 写
的，所以结构不同的模型会在环境构建过程中失败（找不到关节、传感器匹配不到 geom）—— **而不是悄悄给出
错误的结果**。

名字写错会把可用的名字列出来；路径不存在会说文件不存在。两种错误指向两个不同的方向，所以分开报。

### 按 asset 隔离日志

日志路径是 **`logs/<model>/<task>/<date-time>`**，其中 `<model>` 是模型文件的主干名。所以同一个任务
在不同 `--model` 上训练，天然就分进不同的子树：

```
logs/jumper/jumper.flat/<timestamp>/
logs/jumper_v2/jumper.flat/<timestamp>/
```

### 用凸包作碰撞几何

`tools/hull_collision.py` 给一个新模型配上凸包作为碰撞几何。`-k` 是要保留的方向数，而
**`0` 是"全部保留"的哨兵值**，
不是刻度的低端：`-k 0` 就是精确凸包、对接触没有任何改变，其余任何取值都会做抽取，越小抽得越
狠。`tools/hull_collision.py` 里的实测：`0` 留下 28662 个面、向内误差 0.00 mm，`256` 留下
6110 个、3.57 mm，`64` 留下 1772 个、7.81 mm。

```bash
python tools/hull_collision.py --model assets/<name>/<model>.xml --exclude <meshes that touch the ground>
```

它默认只做报告（包括最大的向内收缩量）；`--apply` 会写出 STL 并改 XML。

---

## 设置默认值

### 优先级

```
命令行  >  shell 环境  >  .env.local  >  .env  >  参数自身的默认值
```

撑起这条链的只有一条规则：`load_dotenv` **不覆盖**已经存在的环境变量，而 argparse 的默认值再从
`os.environ` 里读。所以"给了命令行就以命令行为准"不需要任何特判。

如果没有任何一层提供值，而这个参数自身又没有内置默认值（只有 `MJRL_TASK` 是这种情况），入口脚本会
**报错并说明缺了什么**。

### 两个文件

| 文件 | 是否提交 | 内容 |
|---|---|---|
| [`../.env`](../.env) | 是 | 项目级默认值，全组共用 |
| `.env.local` | 否 | 个人覆盖和密钥；优先级更高 |

### 全部的键

```bash
# -- 选哪个任务 --
MJRL_TASK=jumper.flat        # 见 --list。留空且命令行也没给 -> 报错

# 选哪个资产：一个已注册的名字，或者一个 .xml 文件的路径
MJRL_MODEL=                # 留空 -> 任务的默认资产

# 世界长什么样：地面、天空和灯光。见 --list。留空 -> 任务自带的那个
MJRL_SCENE=

# -- 任务自己的设置 --
# 这两个各有一个对应的 flag，`--swing-angle` 和 `--swing-length`，而且
# **是任务声明它们，不是 parser**：`scripts/_cli.py` 是三个入口脚本共用的
# 一个 parser，任务的词汇出现在它里面，正是
# `tests/test_log_layout.py` 禁止的事。见 `tasks/jumper/swing/` 里的 `cli_args`。

# jumper.swing：秋千从多大的角度放开，单位为度。给一个值，每一局都从那里放开；
# 给两个是一个范围，从中均匀采样，两种写法下正负号都会随机。留空则保持各模式
# 自己的默认值 —— 训练是 0,45 这个范围，回放是完全静止。
MJRL_SWING_ANGLE=

# jumper.swing：绳长，单位米，从横梁量到坐板。给一个值，每个秋千都挂在它上面；
# 给两个是一个范围，按 episode、按环境各自采样。留空就是这套架子能挂出来的
# 整个 0.6-1.8 m，回放和训练都一样 —— 周期在这个范围上从 1.50 走到 2.65 s，
# 正是它让策略学不成单一的节奏。把它钉在一个值上，可以问一个策略是不是真的泛化了。
MJRL_SWING_LENGTH=

# -- 场景自己的设置 --
# 由场景声明，就像上面那一对由任务声明。见 scenes/rough.py 里的 cli_args。

# --scene rough：把 10 x 7 的网格钉在某一行，0（平地）到 9（最难）。留空是整条阶梯。
MJRL_TERRAIN_ROW=

# --scene rough：钉在某一列，按子地形的名字（flat、pyramid_stairs……）。留空是混合。
MJRL_TERRAIN_COL=

# -- 后端与设备 --
MJRL_BACKEND=auto          # auto | warp | native
MJRL_DEVICE=auto           # auto | cuda:0 | cpu（torch 设备；torch 没有 metal）
MJRL_SIM_DEVICE=           # warp 配 cpu 时物理在哪算。留空 -> Warp 列出 metal:0 就用它，否则 cpu
MJRL_AGENT_DEVICE=         # 仿真在 metal:0 时策略在哪学。留空 -> torch 有 mps 就用 mps
MJRL_NUM_ENVS=             # 训练用；留空 -> 按后端定：warp 4096 / native 64
MJRL_PLAY_NUM_ENVS=1       # play.py 用；留空 -> 1。它不是 batch size，所以单独一个键
MJRL_CPU_THREADS=0         # native 的线程数，0 = 核心数，上限 8（再受 num_envs 限制）
MJRL_STRIP_VISUAL=auto     # native：剥掉纯视觉的网格。auto | on | off

# -- TensorBoard --
MJRL_TENSORBOARD=on        # on | off（true/false、yes/no、1/0；大小写不限）
MJRL_TB_PORT=6006          # 首选端口；被占用就跳过
```

#### 为什么自动线程数不是"所有核"

native 的一步里大约有一半是串行的 —— 批量缓冲区与每个环境的 `MjData` 之间的 gather 和 scatter 在
Python 这一侧 —— 所以按 Amdahl 定律，并行的那部分在八个 worker 左右就被吃干净了，之后每多一个线程只
增加同步和调度开销。在一台 i9-14900KF（8 个性能核 + 16 个能效核，32 逻辑核）上实测，`jumper.tetrapod`
每个 policy step 的毫秒数：

| envs | 2 | 4 | **8** | 16 | 24 | 32（所有核） |
|---|---|---|---|---|---|---|
| 16 | 10.63 | 8.10 | **7.17** | 9.70 | — | 9.85 |
| 64 | 24.14 | 16.31 | **15.27** | 21.18 | 22.39 | 22.36 |
| 256 | 92.84 | 69.79 | **59.09** | 62.84 | 67.39 | 68.82 |

八在每一个环境数下都是最优，而"所有核"要多付 16-46%。绑核救不回来（八个线程绑在八个专用性能核上：
21.44 ms，对不绑核的 15.71 —— 主线程也得有地方跑），物理核数也救不回来，这里是 24，实测并不比 32 好。

线程数**不是超参数**：不管哪个线程来推进一个环境，它都从自己的 warm start、自己的控制量出发，所以任何
线程数给出的结果都逐位相同（`tests/test_native_determinism.py`），改它只改吞吐，别的都不改。核数少于 8
的机器就用它有的。如果你的机器实测出来不是这样，就设 `MJRL_CPU_THREADS`。

`auto` 的探测：有 CUDA 就用 `warp:cuda`，否则 `native:cpu`。**显式给出的选择绝不回退** —— 在一台
没有 GPU 的机器上要 GPU 是错误，不是悄悄切到 CPU。那是这类框架最容易给自己埋的坑。

### 什么才算一个值

取值集合固定的键**不区分大小写**地读取，而集合之外的值会**报错**，不会回退到默认值：

| 键 | 接受 |
|---|---|
| `MJRL_TENSORBOARD` | `on`/`off`，以及 `true`/`false`、`yes`/`no`、`1`/`0` |
| `MJRL_BACKEND` | `auto`、`warp`、`native` |
| `MJRL_SIM_DEVICE` | 空、`cpu`、`metal:0`（最后一个只在 Warp 列出它时有效） |
| `MJRL_AGENT_DEVICE` | 空，或一个 torch 设备，例如 `mps`、`cpu` |
| `MJRL_STRIP_VISUAL` | `auto`、`on`、`off` |
| `MJRL_NUM_ENVS`、`MJRL_PLAY_NUM_ENVS`、`MJRL_CPU_THREADS`、`MJRL_TB_PORT` | 一个整数 |

```
error: MJRL_STRIP_VISUAL='of' is not one of: auto, on, off (check .env)
```

回退比停下来更糟。这些键存在的全部意义就是改变一次运行的行为，所以一个被悄悄忽略的键会产出一次看起来
正常、却不是你要的那一次运行 —— 而且输出里不会提到这个设置，于是最自然的结论是 `.env` 根本没被读。

**argparse 不会替你做这件事**。`choices=[...]` 只对命令行上敲进去的值做检查，对默认值从不检查 ——
所以一个默认值来自 `.env` 的参数什么都没校验，坏值会一路传到消费它的地方。`MJRL_STRIP_VISUAL=OFF`
过去表现为三层调用之外一张查找表抛出的光秃秃的 `KeyError: 'OFF'`，traceback 里没有一处提到 `.env`。
空值（`MJRL_MODEL=`）仍然表示"用内置默认值"；只有*错误的*值才会让运行停下来。

### 设置它们的三种方式

```bash
# 1) 改 .env（项目级，提交进仓库）
MJRL_BACKEND=native

# 2) shell 环境变量（只管这次会话，压过 .env）
MJRL_BACKEND=native python scripts/train.py

# 3) 命令行（压过一切）
python scripts/train.py --backend native --device cpu --num_envs 512
```

### 常见组合

```bash
# 开发机，快速验证逻辑：少量环境 + 实时 viewer
MJRL_BACKEND=native
MJRL_NUM_ENVS=16

# 训练机：backend 留在 auto（它会选 warp:cuda），环境数取后端默认的 4096
MJRL_BACKEND=auto
MJRL_NUM_ENVS=
```

### 只在命令行上的开关

这些是在命令行上给的。大多数根本没有对应的 `.env` 键；TensorBoard 那两个有
（`MJRL_TENSORBOARD`、`MJRL_TB_PORT`），而开关一如既往地覆盖它：

| 开关 | 作用 |
|---|---|
| `--headless` | **把实时 viewer 关掉。它默认是开的**。 |
| `--viewer-env N` | 相机跟随哪一个环境。默认跟随离**场景中心**最近的那个 |
| `--viewer-env-num N` | 画**多少个**环境（包含被跟随的那个）；不给就是 128。和上一个开关不是一回事 |
| `--viewer-fps F` | 窗口每秒显示多少帧：训练时 30，回放时 60。画图在 viewer 自己的线程上，仿真每帧只付出一次状态拷贝 |
| `--viewer-ui` / `--no-viewer-ui` | MuJoCo 自带的面板 —— 渲染开关、可视化选项、模型树。回放时开，训练时关；它们只改被跟随的那个环境 |
| `--logger tensorboard\|wandb` | **默认 tensorboard**。mjlab 默认 wandb，而那条路径在 wandb 0.29.0 上是坏的 |
| `--max-iterations N` | 覆盖任务 `rl_cfg.py` 里的迭代次数。它是**这一次运行**跑多少次迭代，所以在续训的运行上它是*再*跑多少次 |
| `--resume` | 从 `logs/<model>/<task>/` 下最新的 checkpoint 继续训练 |
| `--checkpoint PATH` | 从哪个 checkpoint 续训：一个 `model_*.pt`，或者一个目录（取其中最新的那个）。隐含 `--resume` |
| `--dry-run` | 只解析并打印；不构建环境 |
| `--tensorboard` / `--no-tensorboard` | 启动 TensorBoard，或者不启动。**它默认是开的。**这一对开关存在，是为了让命令行能在**两个**方向上覆盖 `MJRL_TENSORBOARD` —— 只有否定形式的话，一旦 `.env` 把它关掉，就没有办法为单次运行再把它打开 |
| `--tb-port N` | 首选端口，默认 6006。被占用就跳过 |
| `--tb-scope run\|task\|all` | 服务多大范围；不给就是 `task` |

> **最多画 128 个环境**（包含被跟随的那个）；不到 128 就全画。超过就打印：
>
> ```
> [mjrl] live viewer: 4096 environments, drawing 128 of them (cap 128)
> ```
>
> 这个上限是必要的，有两个理由：
>
> - **Geom 容量**：viewer 的 `user_scn` 只放得下 100000 个 geom；超过之后 MuJoCo 打印一条会在训练
>   日志里滚走的警告，然后**静默丢掉**其余的 —— 画面上仍然是"一片机器人"，没有任何迹象表明少了 60%
>   （用 warp + 4096 实测：期望 253890 个 geom，到达 100000 个，实际画出 1613 个环境）。
> - **时间预算**：画图的开销随画的数量增长。用 native + 4096 实测，全画一遍要 **405 ms 每帧** —— 当画图
>   还挂在 `sim.step()` 上的时候，这些时间**直接从训练里扣**。现在它在 viewer 自己的线程上；帧仍然是在
>   `sim.step()` 结束时*拍下*的，那是状态一致、拷贝不会撕裂的时刻，但那只是一次状态拷贝。同样的 405 ms
>   现在只会让画面变成幻灯片，而不是让仿真变慢。
>
> 128 在两个 backend 上都远低于这两条线。geom 容量这条界限仍然会被算出来作为兜底：一台足够复杂的
> 机器人，128 个环境就能填满 100000 个 geom，那时上限就来自容量，并且会这么说。
>
> 没有可用显示时（没有 `DISPLAY`，或者 macOS 上没用 `mjpython`），它会**自动退回 headless**，所以
> "viewer 默认开"永远不会挡住一次运行的启动。
>
> **在 warp 上边训练边看仍然不是免费的**，而且原因不在帧：MuJoCo 自己的渲染循环会在仿真所用的同一块 GPU
> 上不停地重画场景。用 jumper.swing、4096 个环境、RTX 5090 D 实测：headless 每轮 1.69 s，开窗口 3.92 s，
> 窗口每秒只推一帧 4.09 s，`--viewer-env-num 1` 时 1.98 s。

`play.py` 还**不计算奖励项**：回放里没有任何东西读奖励，而它们曾是回放一步里最大的单项开销（jumper.posture、
warp 上一个环境，11.3 ms 里占 4.7 ms）。`tests/test_replay.py` 逐个任务检查跳过它们不会改变策略看到的任何
东西；`rl/mjrl/replay.py` 写了这件事可能在哪里出错。

### TensorBoard

**训练自己会启动 TensorBoard**，并把 URL 打印出来：

```
[mjrl] log directory logs/jumper/jumper.tetrapod/2026-09-04_14-30-51
[mjrl] tensorboard http://localhost:6006/ serving logs/jumper/jumper.tetrapod
```

它在环境构建**之前**启动，所以在场景装配和模型编译花掉的那几十秒里，页面已经开着了。event 文件两种
情况下都由 `rsl_rl` 的 writer 产出 —— `WandbLogWriter` 继承自 `SummaryWriter`，所以 `--logger wandb`
也会写。

**它服务什么**由 `--tb-scope` 决定：

| Scope | 目录 | 为什么 |
|---|---|---|
| `run` | `logs/<model>/<task>/<timestamp>` | 只有这一次运行；加载最快 |
| `task` | `logs/<model>/<task>` | **默认**。TensorBoard 把每个子目录标成一次 run，于是时间戳成了曲线的名字，上一次运行就在旁边可以直接对比 |
| `all` | `logs/` | 所有模型和任务；目录树一大就慢 |

**端口**是 6006，已经被占用的会跳过 —— 第二次运行落到 6007 而不是失败，并且会说出来。用 `--tb-port`
或 `MJRL_TB_PORT` 换一个起点。

**它只绑定 localhost**。要从另一台机器访问，转发端口，而不是把指标暴露到网络上：

```bash
ssh -L 6006:localhost:6006 <training-host>
```

有浏览器可开的时候会自动开一个 —— 走 ssh 时不开（页面会开在远端的控制台上），没有显示时也不开。

**关掉它**。只关一次运行，用 `--no-tensorboard`；永久关掉，在 `.env`（或 `.env.local`，或 shell
环境）里写 `MJRL_TENSORBOARD=off`。文件里写着 `off` 时，`--tensorboard` 能为单次运行把它重新打开 ——
命令行在两个方向上都压过 `.env`，这正是这个开关是一对、而不是孤零零一个 `--no-tensorboard` 的全部理由。

这个值按开关读，不按字符串读：`on/off`、`true/false`、`yes/no` 和 `1/0` 都接受，大小写不限。其他任何
东西都**报错**：

```
error: MJRL_TENSORBOARD='nope' is not on or off (true/false, yes/no and 1/0 are
accepted too, in any case) (check .env)
```

这是有意的，`MJRL_NUM_ENVS` 对整数遵循同一条规则：一个悄悄回退到默认值的值，会让
`MJRL_TENSORBOARD=of` 无声地意味着"开"，而没有任何东西说明这个设置为什么没起作用。

**失败从不致命**。起不来的服务会把原因打印出来，运行照常继续；TensorBoard 自己的输出不会混进训练日志，
而是写进运行目录里的 `tensorboard.log`。

训练结束时服务会被关掉，包括 Ctrl-C 和 `kill <pid>` 的情况。`kill -9` 或者被 OOM 杀掉会把它留下 ——
`pgrep -f tensorboard.main` 能找到它。

### 场景

一个**场景**就是地面、天空和灯光 —— 与任务正交的那条轴。同一个步态任务，值得在中性的影棚灰下跑一遍
截图、在日落下跑一遍出视频，同一套外观也值得跨任务复用；把这两者分开，才让这件事变得便宜。

```bash
python scripts/train.py --list                          # 场景也会一起列出来
python scripts/train.py --task jumper.flat --scene studio
MJRL_SCENE=beach python scripts/train.py         # 或者写在 .env 里
```

| 场景 | `use` | 它是什么 |
|---|---|---|
| `default` | both | mjlab 自己的外观，写出来是为了它能被命名、能被拿来做 diff |
| `studio` | watching | 中性灰，主光加补光，暗色天空 —— 给截图和视频用 |
| `daylight` | watching | 蓝天，暖色地面，一盏带阴影的硬太阳 |
| `beach` | watching | 湿沙上的低角度暖阳，黄昏的天空，长长的影子 |
| `football` | watching | 五分之一尺度的球场：修剪过的草皮、线标、球门、午后的阳光，**以及一个机器人能踢的球** |
| `swing` | watching | **加了一个秋千**：一个 A 字架和一块机器人能站上去的板子座。这不是 `jumper.swing` 训练用的那个 —— 那个任务自己搭一个，在它之上再加 `--scene swing` 会直接报错，而不是搭出两个 |
| `ice` | both | **改的是地面**：湿冰，滑动摩擦 0.03，对上脚的 1.2 |
| `rubber` | both | **改的是地面**：粗糙的防滑垫，1.8 —— `ice` 的另一端 |
| `soft` | training | **改的是地面**：`default` 的外观，但脚会陷进去 |
| `plain` | training | 没有纹理也没有材质 —— 给 `geom_rgba` 随机化用，也是画起来最便宜的 |
| `rough` | training | **改的是地面**：生成的坡道、台阶和粗糙地形 |

`python scripts/train.py --list` 才是实时的答案；这张表是一份副本，会漂移。

不传场景就完全不动任务自己的配置，所以默认行为是原封不动，而不只是等价。

**`use` 不是标签**。它说明这个场景是*干什么用的*，并且决定这个场景会不会被导出给另一个仿真器 —— 见下文。
`training` 表示这个场景是一件仪器：它存在是为了产生某种训练效果，或者把某一个变量固定住，它不是谁会想
在里面看机器人的地方。`watching` 不表示这个场景不能拿来训练；`studio` 只有灯光而已。它表示在里面训练
没有意义，对 `football` 来说甚至是主动误导，因为球场只画一块、在世界原点上，而环境是排在一张间距网格上的。

#### 场景可以有自己的参数

`rough` 生成一张 10 × 7 的网格 —— 十个难度行，每种子地形一列 —— 一次运行通常通过地形课程遇到它的全部。
要测试**一块瓦片**：

```bash
# 最难的随机起伏，别的都不要
python scripts/play.py  --task jumper.posture --scene rough \
    --terrain-row 9 --terrain-col random_rough

# 在楼梯的某一个难度上训练
python scripts/train.py --task jumper.posture --scene rough \
    --terrain-row 3 --terrain-col pyramid_stairs

```

或者像其他默认值一样，写在 `.env` 里：

```bash
MJRL_TERRAIN_ROW=9
MJRL_TERRAIN_COL=random_rough
```

单独给哪一个都行：只给行不给列，是在全部七种子地形上钉住难度；只给列不给行，是保留其中一种的完整阶梯。
钉住的那块瓦片**就是完整网格在那一行那一列本来会生成的那块** —— 同一个生成器，只是收窄了 —— 这正是
它能用来测试训练地形、而不是一个长得像的替代品的原因。一个什么都匹配不上的名字会报错并列出可选项；
退回到完整的混合会产生一次看起来正确、却不是所要求的那次运行。

这些参数**由场景声明**,而不是由 `scripts/_cli.py` 声明 —— 一个解析器服务三个入口，不允许带上任何
场景或任务的词汇。`scenes.load_cli_args` 是 `tasks.load_cli_args` 的镜像，`jumper.swing` 的
`--swing-angle` 就是这么来的。想要参数的场景在它的 `scene()` 工厂旁边定义 `cli_args(group)`,返回它
加上的 `dest` 名字，并把它们作为关键字参数传给 `scene()`。不声明参数的场景不受影响，`export.py` 也从不
提供它们：场景改变的任何东西都到不了 ONNX 文件里。

**新增一个**就是一个模块加一行：写 `scenes/<id>.py`，里面一个返回 `Scene` 的 `scene()` 工厂函数，
再在 `scenes/__init__.py` 里加一次 `register` 调用。图片文件放进 `scenes/assets/<id>/`，就挨着描述
它们的代码，所以一个场景就是一个模块加一个目录。

自己生成天空的场景，在 `scene()` 工厂函数旁边定义一个 `sky(dirs) -> rgb`，并搭在
[`scenes/skygen.py`](../scenes/skygen.py) 上，那里放着 MuJoCo 立方体贴图中那些实测出来、而不是猜
出来的部分：每个面覆盖哪个世界方向、它在一个面内怎么采样图像，以及跨接缝连续的噪声。这三件事出错都是
静默的 —— 立方体照样编译，天空照样看着像天空。用这个渲染各个面：

```bash
python tools/make_skybox.py <scene-id>     # 或者 --all
```

这些 PNG 是**提交进仓库的，不是按需生成的**，所以改过某个天空的颜色之后要重新跑一遍，否则场景会继续
加载上一次写出来的那些面。

写一个之前要知道的，按它们咬人的先后：

- **材质必须匹配到地面 geom**。mjlab 把它命名为 `terrain`，`scenes.registry.GROUND` 就是匹配它的那个
  表达式。匹配不上的材质什么都绑不到：它被创建出来，地面渲染出来毫无变化，也没有任何东西报告有问题。
- **环境光就是 headlight，而且它是主导**。MuJoCo 挂在相机上的这盏灯跟着视点走，照亮它能看到的一切，
  所以它决定了阴影最深能有多深。mjlab 的场景模板给的是 0.3 ambient / 0.6 diffuse —— 是 MuJoCo 自己
  默认值的三倍 —— 在这个强度下调补光几乎什么都改变不了。每个场景通过 `Scene.headlight` 声明自己的
  那一份；它必须走 `SceneCfg.spec_fn`，因为附加实体里的 `<visual>` 块会被父 spec 一声不响地覆盖掉。
- **场景可以加一个物理道具，而且它必须声明**。`Scene.props` 里放的是实体 —— 球场上的那个球 —— 它们
  有质量、会碰撞，所以接触、接触数量和求解器负载都会变。带着它训练出来的 policy 和不带它训练出来的
  不可比，而 observation 和 reward 都没动，所以没有别的东西会报告这件事：`apply` 会给出警告。摆放是
  自动接好的，因为一个没有 reset 事件的实体会在*每一个*环境里都停在 (0, 0, 0)，而配置读起来却是对的。
- **场景加的其他几何体必须是纯视觉的**。`Scene.decorate` 可以通过同一个 `spec_fn` 往世界里放东西 ——
  球场线标、球门柱 —— 而这些全都需要 `contype=0, conaffinity=0`。机器人能撞上的东西会改变它正在训练的
  那个任务，对每一个选了这个场景的任务都是如此，而 reward 和 observation 毫无变化，只有世界悄悄地
  不一样了。
- **改 `terrain_type` 不是换个皮**。这里的任务在设成平地时会把 `terrain_scan` 传感器去掉，所以一个把
  生成地形重新打开的场景，会得到一个感知不到自己脚下是什么的 policy —— 而它照样训练、照样收敛。
  **没有任何东西会检查这一点**；在从这样一次运行里读出任何结论之前，先在任务的 `env_cfg` 里把传感器
  加回来。
- **新场景必须说明自己是干什么用的**。`register(..., use=...)` 没有默认值，因为两种默认都错得像是
  对的：默认导出，会把一张 curriculum 网格当成游乐场发出去；默认不发，意味着一个新场景悄无声息地
  哪儿也到不了。

#### 导出一个场景

一个场景可以作为另一个仿真器导入的包离开这个仓库 —— 它是机器人包的对应物，而且是有意为之的另一半：
那一半发的是机器人、没有世界，这一半发的是世界、没有机器人。

```bash
python scenes/tools/export_web_scene.py --list                 # 哪些会走、哪些不会
python scenes/tools/export_web_scene.py --scene football --out out/scenes
python scenes/tools/export_web_scene.py --all --out out/scenes
```

跟着走的是这个场景的全部，只要它能在被写下来之后活下来：地面*以及它的接触参数* —— 所以 `ice` 到那边
是滑的，而不只是蓝的 —— 纹理、材质、灯光、天空、headlight、装饰性几何体，以及作为独立模型、并写明了
摆放位置的道具。

有两道互相独立的闸门在做判断，它们抓的是不同的东西：

- **`use == "training"` 会被拒绝**，依据是场景自己的声明。导出器不保存任何场景名字的列表；这个判断
  连同它的理由，一起住在 `scenes/__init__.py` 里注册调用的旁边。
- **行为写在 Python 里的场景会被拒绝**。`soft` 每一步都在每只脚上施加一个力，来伪造接触模型表达不出来
  的颗粒阻力。没有哪个 MJCF 字段表示这件事，而一个忽略了它的包会编译通过、渲染正确，只在脚底下是错的。

`soft` 两道都过不了，这件事值得留着，而不是拿掉其中一道的理由。

导出器接着把自己写出来的东西编译一遍，拿它对着来源的 spec 核账：geom、灯光、纹理和材质的数量；每一个
装饰性 geom 回来时仍然不碰撞；以及包里没有任何东西还指向造出它的那台机器。它能在一台训练不了的机器上
跑 —— 一个场景在没有任务也没有机器人的空世界上就能编译。

---

## 继续训练

```bash
# 接着跑刚刚结束的那一次
python scripts/train.py --task jumper.flat --resume

# 从某一个特定的 checkpoint 继续
python scripts/train.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-09-04_14-30-51/model_1999.pt

# 从一个运行目录继续：取它里面最新的那个 checkpoint
python scripts/train.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-09-04_14-30-51
```

`--checkpoint` 隐含 `--resume`；单独的 `--resume` 表示"`logs/<model>/<task>/` 下最新的那个
checkpoint"。

### "最新"是什么意思

**最近被写入的那个文件，不是编号最大的那个**。从一个早期 checkpoint 续训，会开出一次新的运行，它的
编号起点低于磁盘上已有的最大编号，所以按名字排序会把下一次 `--resume`、以及之后的每一次，都送回那次
被放弃的运行 —— 训练看上去在推进，实际上哪儿也没去。同一秒内的并列（有些文件系统的 mtime 只到整秒）
由迭代号来打破。

解析逻辑是 `mjrl.checkpoint`，`play.py` 用的是同一份："最新的 checkpoint"指的是哪个文件，不能取决于
是哪条命令在问。

### 恢复了什么

**整个训练状态**，不只是权重：

| 恢复的 | 为什么要紧 |
|---|---|
| policy 和值函数 | 显而易见的那部分 |
| 优化器状态 | 自适应学习率保持在原来的位置，而不是跳回 `learning_rate` |
| 迭代计数器 | 新的运行从旧的停下的地方继续记录和保存 |
| `common_step_counter` | curriculum 数的是环境步数；没有它，每一条按步数推进的 curriculum 都会倒回去 |
| curriculum 级别 | 靠挣来升级的阶梯（命令、姿态、负载）会从第 0 级重新开始，而 step 计数器早已越过每一段停留。由 `CurriculumRunner` 保存；环境里设了 `MJRL_*_LEVEL` 则以它为准。地形行是重新抽的 |

只加载 policy 是热启动，不是续训 —— 而它照样能训练，所以看起来没有哪里不对。（`play.py` 确实*只*加载
policy，用 `load_cfg={"actor": True}`：回放不需要别的。）

### 它写在哪里

一个**新的**带时间戳的运行目录，所以续训的那次运行绝不会覆盖它从之出发的那些 checkpoint：

```
logs/jumper/jumper.flat/2026-09-04_14-30-51/   model_0.pt ... model_4999.pt      <- 从这里续训
logs/jumper/jumper.flat/2026-09-05_09-12-04/   model_5000.pt ... model_6998.pt   <- 续训的那一次
```

曲线照样接得上：迭代计数器是连续的，所以在默认的 `--tb-scope task` 下，TensorBoard 会把第二次运行画成
一条从第一次结束处开始的曲线。

### 它会跑多少次迭代

`--max-iterations`（或者任务的 `rl_cfg.py`）是**这一次运行**跑多少次迭代，叠加在 checkpoint 已经有的
那些之上 —— 否则续训一次已经跑完的运行就无事可做了。加载时会把这笔账打印出来：

```
[mjrl] resuming from logs/jumper/jumper.flat/2026-09-04_14-30-51/model_4999.pt
[mjrl] log directory logs/jumper/jumper.flat/2026-09-05_09-12-04
[mjrl] resumed at iteration 4999; running 2000 more (through 6998)
```

### 它什么时候会拒绝

来自另一套配置的 checkpoint —— 改过的 observation 项、不同的网络形状、另一个任务 —— 会让运行停下来：

```
[mjrl] cannot resume from logs/jumper/jumper.flat/2026-09-04_14-30-51/model_4999.pt:
  RuntimeError: Error(s) in loading state_dict for MLPModel: size mismatch for ...
The usual cause is a checkpoint from a different configuration: it has to come from
the same task, model and network as this run, observation and action dimensions
included. Train from scratch (drop --resume), or point --checkpoint at a matching run.
```

checkpoint 在环境构建**之前**就解析完，所以 `--checkpoint` 里打错一个字只花掉一秒，而不是本来会排在
它前面的那一分钟场景装配和模型编译。

**没有对应的 `.env` 键**。其他每一个默认值都能在那里设；这一个是有意不设的：一个黏住的"总是续训"
默认值，会在一次本该重新开始的启动里悄悄接着跑旧的那一次 —— 而这种失败在输出里是看不见的。

---

## 命令速查表

```bash
# 列出每一个任务和它接受的资产
python scripts/train.py --list

# 用 .env 里的默认值运行
python scripts/train.py

# 只解析不构建环境，用来检查参数
python scripts/train.py --task jumper.tripod --model jumper --dry-run

# 一边训练一边看（这是默认行为，不需要开关）
python scripts/train.py --task jumper.flat --num_envs 64

# 强制用 CPU 真训练
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64

# 不带 TensorBoard 训练（否则它会自动启动）
python scripts/train.py --task jumper.flat --no-tensorboard

# 接着跑刚刚结束的那一次，再跑 2000 轮
python scripts/train.py --task jumper.flat --resume --max-iterations 2000

# 从某一个特定的 checkpoint 继续
python scripts/train.py --task jumper.flat --checkpoint <path>/model_1999.pt
```

native 指定线程数、少画几个环境，以及 macOS：

```bash
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 4096 --cpu_threads 32
python scripts/train.py --task jumper.flat --viewer-env-num 16 --viewer-env 0
.venv/bin/mjpython scripts/train.py --task jumper.flat --num_envs 64   # macOS，为了 viewer
```

回放和导出接受同样的参数：

```bash
python scripts/play.py --task jumper.flat                      # 最新的 checkpoint
python scripts/play.py --task jumper.flat --agent zero         # 不需要 checkpoint
python scripts/export.py --task jumper.flat --checkpoint <path>/model_4999.pt
```
