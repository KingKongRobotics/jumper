<!-- tracks: AGENT_SETUP.md @ sha256:10f9f79be9be42ae -->

# 环境搭建说明（写给 AI agent）

本文档写给**在一台全新机器上把 mjrl-lab 跑起来的 AI agent**。
人类读者请从 [`../README.md`](../README.md) 开始。

## 机械的部分由两个脚本完成

两个都是纯标准库 Python，各自独立可运行 —— 不需要 agent 框架，不需要 skill 加载器，
不需要安装任何东西：

```bash
python3 .claude/skills/setup-env/scripts/detect.py    # 这台机器是什么，以及由此该做什么
.venv/bin/python .claude/skills/setup-env/scripts/gates.py   # 它能不能用，以及要修什么
```

`detect.py` 针对眼前这台机器解出第 0 节的决策表，并打印出对应的命令；`gates.py` 跑第 3 节的
六道 gate，并为每一种失败指出补救办法。它们是本文档的可执行形式，不是它的替代品 —— 一旦有
什么对不上，解释在这里。

对于会加载 skill 的 agent，[`.claude/skills/setup-env/`](../.claude/skills/setup-env/)
把同一套流程包装了一遍。

## 怎么用这份文档

- **按顺序执行**，不要跳步。每一节末尾的“验证”步骤是一道**硬 gate**：不通过就不要往下走。
- 每条命令都写明了它的**预期输出**。实际输出不一致时，去第 6 节按错误字符串查；不要猜，也
  不要反复重试同一条命令。
- 看到 **🛑 STOP** 标记，立刻停下并报告用户。不要绕过它。
- 所有命令都假定**当前工作目录是仓库根目录**（含 `pyproject.toml` 的那一层）。

---

## 0. 准备工作

先把事实收集齐，再决定走哪个分支。**不要跳过这一节；后面每一步都依赖它的结果**。

```bash
python -c "import platform,sys; print('OS      :', platform.system()); print('ARCH    :', platform.machine()); print('PYTHON  :', sys.version.split()[0])"
```

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

如果 `nvidia-smi` 不存在或者报错，这台机器没有可用的 NVIDIA GPU —— 走**无 GPU 分支**。

### 决策表

三个平台上的安装过程完全相同（第 1 节）；**只有 torch 那一步会按有没有 GPU 分叉**。五种组
合各需要哪些小节：

| 操作系统 | NVIDIA GPU | 需要的小节 | 可用后端 |
|---|---|---|---|
| Linux | 有 | 1.1 – 1.4 | `warp:cuda`（主力）、`native:cpu` |
| Linux | 无 | 1.1 – 1.3 | `native:cpu` |
| Windows | 有 | 1.1 – 1.5 | `warp:cuda`、`native:cpu` |
| Windows | 无 | 1.1 – 1.3、**1.5** | `native:cpu` |
| macOS | ——（没有 CUDA，没有例外） | 1.1 – 1.3 | `native:cpu` |

每一行还都要做 2.1 和 2.2，也就是控制器的工具链，它不按有没有 GPU 分叉。

### Python 版本

框架要求 **Python 3.10 – 3.13**（不含 3.14）。

如果第 0 节量到的版本不在这个区间，**先装一个合规的**；不要试图在更老的解释器上安装依赖。
见 1.1 的平台表。

---

## 1. 建一个隔离环境并安装 PyTorch

**环境工具只有一个：标准库的 `venv`**，环境放在仓库根目录的 `.venv/`。不要用 conda —— 在
这套流程里它只负责提供一个解释器，而下面每个平台都另有办法做到这件事。所有依赖都是 PyPI
wheel；不需要系统级 CUDA toolkit，只需要 NVIDIA 驱动。

### 1.1 弄到一个 3.10–3.13 的解释器

如果第 0 节得到的版本合规，直接用 `python3`。**如果不合规**，先装一个：

| 平台 | 怎么做 | 之后创建 venv 用的命令 |
|---|---|---|
| Ubuntu 22.04 / 24.04 | 自带 3.10 / 3.12，无需处理 | `python3` |
| Ubuntu 20.04 及更早 | `sudo add-apt-repository ppa:deadsnakes/ppa`<br>`sudo apt install -y python3.11 python3.11-venv` | `python3.11` |
| macOS | `brew install python@3.12` | `python3.12` |
| Windows | python.org 的安装器，勾上 "Add python.exe to PATH" | `py -3.11` |

> **一个 Debian / Ubuntu 的坑**：如果 `python3 -m venv` 报 `ensurepip is not available`，
> 说明这个发行版把 venv 拆成了单独的包。装上它：
> `sudo apt install -y python3-venv`（版本要和解释器对上，例如
> `python3.11-venv`）。

### 1.2 创建并激活

```bash
python3 -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

> 在 Windows 上，如果 `Activate.ps1` 报执行策略错误：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**验证（硬 gate）**：

```bash
python -c "import sys; print(sys.prefix); print(sys.version.split()[0])"
```

第一行必须是仓库内 `.venv` 的路径；第二行必须在 3.10 到 3.13 之间。**指向系统 Python 或任何
conda 环境都是失败** —— 不要继续，因为之后每一次 `pip` 都会装到错误的地方，而症状要到第 3
节才会暴露。

### 1.3 安装 PyTorch

**有 GPU 时**，按第 0 节量到的 `compute_cap` 选 CUDA 构建：

| compute_cap | 架构 | 需要的 torch 构建 |
|---|---|---|
| 12.0 | Blackwell（RTX 50 系列） | **cu128 或更新** |
| 8.9 / 8.6 | Ada / Ampere | cu126 或更新 |
| < 8.0 | 更老 | 查 PyTorch 官方的支持矩阵 |

> 🛑 **STOP**：在 Blackwell 上装 cu126 的 torch，会在 CUDA 初始化时失败。拿不准 compute
> capability 就选 cu128 —— 它向后兼容 Ada/Ampere。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**没有 GPU 时**（包括所有 macOS）装 CPU 构建，并且**不要**加 `--index-url`：

```bash
pip install torch torchvision
```

> **`warp:cuda` 在 macOS 上确定不可用**（Apple Silicon 没有 CUDA，没有例外）。那里的 CPU
> 训练走 `native:cpu` 后端，它完全不需要 CUDA。

### 1.4 GPU 机器的系统前置：NVIDIA 驱动

**没有 GPU 的机器跳过本小节**，去 1.5（Windows）或第 2 节。

如果 `nvidia-smi` 列不出显卡，**说明没有驱动**。这一层 pip 和 venv 都帮不上忙；必须在操作
系统层面安装：

- **Linux**：`ubuntu-drivers devices` 看推荐，然后
  `sudo apt install nvidia-driver-XXX`，并且**重启**
- **Windows**：NVIDIA 官网的 GeForce/Studio 驱动安装器
- **WSL2**：驱动装在 **Windows 宿主**上；不能在 WSL 里面装

> **不需要 CUDA toolkit**。`nvidia-smi` 右上角的 `CUDA Version` 是**驱动支持的最高运行时版
> 本**，不是“已安装的 toolkit”。cu128 的 torch wheel 自带 CUDA 运行时，warp 自带工具链。

---

### 1.5 Windows 的系统前置：一个 C++ 编译器

**有没有 GPU 都适用**。NVIDIA Warp 会在首次运行时 JIT 编译 kernel，CPU 设备也一样，所以没
有 GPU 的 Windows 机器同样会撞上这个。

如果第 3 节的 Gate A 报编译器 / MSVC 错误，安装
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 的
"Desktop development with C++" 工作负载，然后在新开的终端里重试。

> **先往下走，只有错误真的出现时才装** —— 不要预先安装；那个工作负载有好几 GB。例外是
> Rust（2.1），它每次构建都要用它来链接。

---

## 2. 安装本仓库

```bash
pip install -e .
```

这会装上 `pyproject.toml` 里列出的传递依赖（`mujoco`、`mujoco-warp`、`warp-lang`、
`tensordict`、`tyro`、`viser` 等等）。

> ⚠️ **这一步是强制的，不能跳过**。vendored 的 `mjlab` / `rsl_rl` 放在 `rl/` 下面，而
> `rl/` 不在 `sys.path` 上 —— 它们只能通过这次 editable 安装建立的映射被 import
>（`pyproject.toml` 把 `.` 和 `rl` 声明成了两个 package root）。
> 跳过它，每一个入口点都会以
> `ModuleNotFoundError: No module named 'mjlab'` 失败。

> 🛑 **绝不要执行 `pip install mjlab` 或 `pip install rsl-rl-lib`**。两者都以**可修改的副
> 本**的形式存在于 `rl/` 下（`rl/mjlab/`、`rl/rsl_rl/`；见 [`VENDOR.md`](VENDOR.md)）。
> 同一个环境里再有 pip 版本，就会造成**遮蔽（shadowing）的歧义**：`site-packages` 里同名
> 的包和这份副本相互竞争，行为不一样，而且非常难察觉。

如果这个环境里曾经装过，先卸载（这不会删掉它们的依赖）：

```bash
pip uninstall -y mjlab rsl-rl-lib
```

### 2.1 控制器的工具链：Rust

控制器（`deploy/fsm`）是 Rust 写的，一个 crate 跑在三种宿主上。`scripts/deploy.py` 在本机用
cargo 构建其中两种 —— 浏览器的 wasm 和 `play --app` 的扩展 —— 板端二进制和 Windows 扩展则在
Docker 里交叉编译，镜像自带工具链。**第 0 节决策表的每一行都需要本小节**；它不按有没有 GPU
分叉。

> ⚠️ **没有 cargo 时测试集不会失败，而是跳过。** 没有 cargo，`tests/test_fsm_extension.py` 和
> `tests/test_deploy_sources.py` 会 skip，于是在一台从没构建过控制器的机器上，
> `pytest tests/` 也是绿的。

一共四样。`detect.py` 只打印这台机器缺的那几样：

| 组件 | 为什么 | 安装 |
|---|---|---|
| 一个 C 链接器 | build script 和过程宏要为宿主机链接，`cargo check` 也一样 | Linux：`sudo apt install -y build-essential`<br>macOS：`xcode-select --install`<br>Windows：1.5 |
| rustup，stable，rustc ≥ 1.85 | `Cargo.lock` 锁定的包里最高的 `rust-version`（hashbrown 0.17.1 和 indexmap 2.14.2，经由 `toml` 引入；2026-09-29 读取） | 见下 |
| `wasm32-unknown-unknown` target | 浏览器的构建 | `rustup target add wasm32-unknown-unknown` |
| `wasm-bindgen` CLI，版本与 `Cargo.lock` **完全一致** | CLI 和 crate 共用一套 ABI，却各自发版本 | `cargo install wasm-bindgen-cli --version <Cargo.lock 里的版本> --locked` |

```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
```

```powershell
# Windows PowerShell，装完新开一个终端
winget install --id Rustlang.Rustup -e
```

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked    # Cargo.lock 里的版本，2026-09-29
```

> **wasm-bindgen 的版本取自 `Cargo.lock`，不是 `Cargo.toml`。** manifest 里写的是 `"0.2"`，
> 这是一个任何 0.2.x 都满足的范围，`cargo install --version` 还会直接拒绝它。不带
> `--version` 的 `cargo install` 装的是最新版，而最新版只在下一次发布之前恰好等于 lock 里的
> 版本。`detect.py`、gate E 和 `scripts/deploy.py` 都拿 lock 里的版本来卡 CLI。

> **发行版自带的 cargo 不够用。** `apt install cargo` 能回答 `cargo --version`，但没有 rustup
> 可以用来加 wasm target，而且通常比 lock 要求的版本旧。

> **Windows 上默认工具链用 MSVC 链接**，所以不管 warp 有没有要求过，Rust 都需要 1.5 的
> Build Tools。

`~/.cargo/bin` 在登录 shell 的 `PATH` 上，但常常不在编辑器或服务的 `PATH` 上。`deploy.py`、
测试和 `gates.py` 都会去那里找；shell 则需要重开，或者执行 `. "$HOME/.cargo/env"`。

### 2.2 设备构建：CycloneDDS 和 libclang

crate 的默认 feature `device` 是机器人的 DDS 总线，在 `deploy/fsm` 里直接跑 `cargo test`、以及
`tests/test_deploy_sources.py` 构建的都是它。`build.rs` 会运行 `idlc`，拿
`$CYCLONEDDS_HOME`（默认 `/usr/local`）编译它的输出，经由 libclang 生成 Rust 绑定，并链接
`libddsc`。`deploy.py` 的宿主构建一样都用不到，板端构建在 Docker 镜像里自带一份 —— 但一旦装
了 cargo，`test_deploy_sources` 就不再 skip，没有它就会失败。

Linux 上：

```bash
sudo apt install -y cmake git libclang-dev
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON
cmake --build /tmp/cyclonedds/build -j
sudo cmake --install /tmp/cyclonedds/build && sudo ldconfig
```

11.0.1 是开发机（i9-14900KF，Ubuntu 24.04）上跑的版本，它的 soname `libddsc.so.11` 与板子上
的一致。**`sudo ldconfig` 不能省**：`/usr/local/lib` 是通过加载器的缓存查找的，没有它，测试
二进制能链接却启动不了 —— 这正是 gate F 要把它跑起来、而不只是编出来的原因。

macOS 上是同样的源码构建，但装进你自己的前缀 —— 不碰 `/usr/local`，不用 `sudo`，也没有
`ldconfig`（macOS 根本没有这个东西）。crate 会在构建时把 `CYCLONEDDS_HOME` 下的 `lib/` 写进测试
二进制自己的 rpath，所以下面两行 `export` 必须在跑 `cargo` 和 `gates.py` 的那个 shell 里生效，
而不只是在安装时的那个：

```bash
brew install cmake
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/opt/cyclonedds-11.0.1 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON
cmake --build /tmp/cyclonedds/build -j
cmake --install /tmp/cyclonedds/build
export CYCLONEDDS_HOME=$HOME/.local/opt/cyclonedds-11.0.1
export PATH=$CYCLONEDDS_HOME/bin:$PATH
```

bindgen 从 Xcode 命令行工具里找 libclang。2026-10-08 在一台 M3 Max（Darwin 25.5，cargo
1.95.0）上实测：构建约两分钟，gate F 列出 195 个测试，`deploy/fsm` 里 `cargo test --lib` 全部
通过。在 `export CYCLONEDDS_HOME` 之前构建出来的测试二进制没有 rpath，会以
`Library not loaded: @rpath/libddsc.11.dylib` 中止；重新构建它（`cargo clean -p mjrl-fsm`），
不必重装 CycloneDDS。

Windows 上是同样的源码构建：把 `CYCLONEDDS_HOME` 设成安装前缀，把它的 `bin/` 放到 `PATH` 上
（`idlc` 从 `PATH` 找，头文件和库从 `CYCLONEDDS_HOME` 找），bindgen 需要装 LLVM
（`LIBCLANG_PATH`）。**没有实测过。**

**Rockchip 的 NPU 头文件是唯一一样按 checkout 拉取、而不是按机器安装的东西。** `build.rs` 还会针对
`deploy/fsm/vendor/rknpu2/include/rknn_api.h` 编译一个探针，而这个头文件属于 Rockchip，不提交进
仓库（原因见 `deploy/fsm/vendor/rknpu2/README.md`）。每个 checkout 执行一次：

```bash
bash deploy/fsm/vendor/rknpu2/fetch.sh
```

它拉取固定的版本并校验 sha256。缺了它，设备构建会在 `build.rs` 里停下，报错里会写出这个脚本。

---

## 3. 验证 gate

**六道 gate 必须全过**。任何一道失败就停下来，去第 6 节查。

### Gate A —— Warp 能枚举设备

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

**在有 GPU 的机器上**，stdout 必须是：

```
['cpu', 'cuda:0']
```

`wp.init()` 还会往 **stderr** 打一段 banner，里面有显卡名和 compute capability（例如
`sm_120`）、CUDA 工具链版本以及 kernel 缓存路径。那是信息性的 ——
**不要拿它当判据**。判据是 stdout 上那个列表里有没有 `cuda:0`。

只看到 `['cpu']` 是失败：驱动或者 CUDA 工具链有问题，`warp:cuda` 后端不可用。

**在没有 GPU 的机器上**：只有 `['cpu']` 就是**预期**结果，不是失败。

### Gate B —— 原生 MuJoCo 的多线程批量接口

`native:cpu` 后端依赖它。

```bash
python -c "import mujoco; from mujoco import rollout; print('mujoco', mujoco.__version__)"
```

预期：打印出一个版本号，没有异常。版本应当是 `3.11.x` 或 `3.12.x`。

### Gate C —— vendored 的副本没有被 pip 版本遮蔽

**这是最容易被跳过的一步，也是失败起来最隐蔽的一步**。

```bash
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
```

预期：**两个路径都在本仓库的 `rl/` 里**，形如：

```
/path/to/mjrl-lab/rl/mjlab
/path/to/mjrl-lab/rl/rsl_rl
```

只要有一个里面出现 `site-packages`，就回到第 2 节，执行
`pip uninstall -y mjlab rsl-rl-lib`，然后重新 `pip install -e .`。

这里出 `ModuleNotFoundError`，说明 `pip install -e .` 没跑或者没跑成功 —— 见第 2 节的
⚠️。

### Gate D —— 任务注册表能用

```bash
python scripts/train.py --list
```

预期：十个任务，全都是 `jumper.*` —— 四个步态变体（flat / tripod / tetrapod / ripple）加上
`dance`、`five_foot`、`jump`、`posture`、`ref_free_jump` 和 `swing`。

这条命令**不加载任何仿真依赖**，所以即便 GPU 是坏的它也应该通过：任务是通过惰性工厂注册
的，`import tasks` 不会把 mjlab / torch / mujoco 拖进来。这里失败是打包问题（最可能是没跑
`pip install -e .`），不是仿真问题。

### Gate E —— Rust 工具链能构建控制器的宿主目标

```bash
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --target wasm32-unknown-unknown --no-default-features --features web
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --no-default-features --features py
wasm-bindgen --version
```

预期：两次 check 都没有 `error` 地结束，并且版本号等于 `deploy/fsm/Cargo.lock` 里
`wasm-bindgen` 那一项。

这里是真的去构建，而不是读版本号：对每种宿主的 feature 做一次 check，能证明 target 已安装、
宿主链接器可用，而 `cargo --version` 两样都证明不了。在 i9-14900KF 上、crate 已经下载好的情况
下，两次冷 check 分别是 8 s 和 7 s；新机器第一次跑还要先下载它们。

### Gate F —— 设备构建能链接、能加载

```bash
cargo test --locked --lib --manifest-path deploy/fsm/Cargo.toml -- --list
```

预期：一串以 `: test` 结尾的行（2026-09-29 时是 179 行）。`--list` 会运行测试二进制但不执行
任何测试，所以它能证明加载时找得到 `libddsc` —— 这是 `cargo check` 永远证明不了的 —— 同时
crate 自己的测试失败也没法冒充成环境故障。同一台机器上冷构建 9 s。缺 Rockchip NPU 头文件的
checkout 会在这里、在 `build.rs` 中失败；解决办法是 `fetch.sh`（2.2 节）。

---

## 4. 冒烟测试

六道 gate 都过了之后，跑一次短训练，确认整条链路端到端是通的。

### 有 GPU

```bash
python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
```

预期：先打印 Actor/Critic 的网络结构，然后是若干行 `Iteration time: ...`，进程以退出码 0
结束。

### 没有 GPU

```bash
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64 --max-iterations 3 --headless
```

预期：banner 报告 `backend=native device=cpu`，并且迭代能跑完。

`--headless` 在这里只是为了让冒烟测试不依赖有没有显示器；想看着跑就把它去掉。

**实测吞吐**（i9-14900KF，32 线程，带混合碰撞的六足，`native:cpu`）：4096 个环境，每次迭代
26.0 s。同一台机器上的 RTX 5090 用 `warp:cuda` 跑同样的 4096 个环境，每次迭代 0.82 s ——
大约快 32×。没有 GPU 的机器也能真的训练，但一块 GPU 值大约三十倍。

---

## 5. 首次运行的正常表现

下面这些**不是错误**；不要去修：

- **Warp 会在首次运行时编译 kernel，可能要几分钟**，并把结果缓存到
  `~/.cache/warp/<version>/`（Windows 上是 `%LOCALAPPDATA%`）。第二次运行就快多了。
- 训练开始时会打印 `[INFO] <NullRecorderManager> (inactive)`。
- 头几次迭代的 `Mean reward` 是负的。

---

## 6. 已知故障与补救

按**错误字符串**索引。

### `ensurepip is not available` / `The virtual environment was not created successfully`

Debian / Ubuntu 把 venv 拆成了单独的包。**补救**：

```bash
sudo apt install -y python3-venv     # 或者 python3.11-venv，与你的解释器版本对应
```

然后把半成品的 `.venv/` 删掉重建：`rm -rf .venv && python3 -m venv .venv`。

### Windows：`Activate.ps1 cannot be loaded because running scripts is disabled on this system`

PowerShell 的执行策略。**补救**：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 包被装到了别的地方 / `pip list` 里看不到刚装的东西

虚拟环境没有激活。**补救**：回到 1.2 的验证；`sys.prefix` 必须指向仓库的 `.venv`。这是最常
见的一类失败，而它的症状通常要到 gate C 才暴露。

### `IndexError: list index out of range`（traceback 里有 `select_gpus`）

这是 mjlab 自己的 GPU 选择逻辑在一台没有 GPU 的机器上抛出来的：`CUDA_VISIBLE_DEVICES`
**未设置**时它会走 `torch.cuda.device_count()`（无卡机器上是 0），而 `gpu_ids` 默认是
`[0]`，于是下标越界。只有把这个变量设成**空字符串**，它才会进入 CPU 模式。

**补救**：用 `--backend native --device cpu`，它根本不走那条路径。`scripts/` 下的入口点自己
解析后端，永远到不了那里。

### `ModuleNotFoundError: No module named 'wandb'`

`import mjlab.tasks` 会**把每一个任务包都 import 一遍**，而
`tasks/manipulation/rl/runner.py` 里有一句顶层的 `import wandb`，绕不开。

**补救**：`pip install -e .` —— `wandb` 和 `PyYAML` 都列在 `pyproject.toml` 里。在更老的环
境里，`pip install wandb PyYAML`。

### `ValidationError: 1 validation error for Settings ... start_method ... extra_forbidden`

mjlab 和 wandb 0.29.0 不兼容。**补救**：`scripts/train.py` 已经默认用
`--logger tensorboard`，所以只有显式要求 wandb 时才会碰到这个。（降级 wandb 也行，但
tensorboard 省事。）

### `nefc overflow - please increase njmax to N`

`njmax` / `nconmax` 是每个 world 上约束行数和接触数的上限。mujoco_warp 从静止姿态的
`mjData` 推断默认值，而运动中的机器人接触数比静止时多得多。

**补救**：在任务配置里显式调高。jumper 设的是 `cfg.sim.njmax = 512` 和
`cfg.sim.nconmax = 128`（见 `tasks/jumper/common/velocity_env.py`）；换一个新机器人时，取错误
信息里的 N，设一个比它大的值。

### `"cuda" device requested but this build of Warp does not support CUDA`

装的是纯 CPU 构建的 Warp，或者驱动不可见。**补救**：跑 gate A 确认设备枚举，然后检查
`nvidia-smi` 能不能用。

### `torch.cuda.is_available()` 返回 False

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

版本号里没有 `+cuXXX`，说明装的是 CPU 构建；Blackwell 卡上出现 `cu126`，说明版本不匹配。
**补救**：按第 1 节重装正确的 CUDA 构建。

### Windows：Warp 报编译器 / MSVC 错误

安装 Visual Studio Build Tools 的 "Desktop development with C++" 工作负载，然后在新开的终
端里重试。

### `FileNotFoundError: .../assets/jumper/jumper.xml`

资产文件缺失。它正常情况下是纳入版本控制的，生成它所用的 URDF 包（`assets/jumper/urdf/jumper/`）
也是；如果确实不在，用绑定到该资产的生成脚本重新生成，它的 `--src` 和 `--out` 默认就是这两个
路径：

```bash
python assets/jumper/tools/build_jumper.py
```

### mjlab / rsl_rl 是从 site-packages import 的

见 gate C。

### `cargo: command not found`，或者 `tests/test_fsm_extension.py` 和 `tests/test_deploy_sources.py` 被 skip

没有 Rust 工具链，或者 `~/.cargo/bin` 不在当前 shell 的 `PATH` 上。**补救**：2.1；后一种情况
执行 `. "$HOME/.cargo/env"` 或新开一个 shell。

### `the wasm32-unknown-unknown target may not be installed` / ``can't find crate for `core` ``

**补救**：`rustup target add wasm32-unknown-unknown`。用的是发行版的 cargo 时没有 rustup 可以
执行它 —— 装 rustup（2.1）。

### ``linker `cc` not found``（Windows 上是 `link.exe`）

没有 C 链接器。**补救**：2.1 表格的第一行。

### `is not supported by the following packages` / `does not understand this lock file`

工具链比 `Cargo.lock` 要求的旧。**补救**：`rustup update stable`。

### `wasm-bindgen CLI is …, the crate builds … (deploy/fsm/Cargo.lock)`

`scripts/deploy.py` 要求 CLI 与 lock 里构建的版本完全一致（2.1）。**补救**：执行它打印的那
条命令，`cargo install wasm-bindgen-cli --version <Cargo.lock 里的版本> --locked`。

### `idlc not found (...); it ships with CycloneDDS`

**补救**：2.2。如果 CycloneDDS 装在 `/usr/local` 以外的地方，把它的 `bin/` 放到 `PATH` 上，并
把 `CYCLONEDDS_HOME` 设成它的前缀。

### `unable to find library -lddsc` / `Unable to find libclang`

前者：`libddsc` 不在 `$CYCLONEDDS_HOME/lib` 下。后者：bindgen 找不到 libclang —— Linux 上
`sudo apt install -y libclang-dev`。**补救**：2.2。

### `error while loading shared libraries: libddsc.so.11`

编译、链接都过了，但加载器找不到这个库。**补救**：装进 `/usr/local` 之后执行
`sudo ldconfig`，或者把 CycloneDDS 的 `lib/` 加进 `LD_LIBRARY_PATH`。

### macOS：`Library not loaded: @rpath/libddsc.11.dylib`

同一种失败换到 macOS 的加载器上，而它没有缓存可刷新：测试二进制的 rpath 取自构建时的
`CYCLONEDDS_HOME`，这个二进制要么是在 `export` 之前构建的，要么指向了另一个前缀。**补救**：按
2.2 `export CYCLONEDDS_HOME` 后重新构建（`cargo clean -p mjrl-fsm`）；把 `DYLD_LIBRARY_PATH`
指到前缀的 `lib/` 可以不重建地跑过一次。

---

## 7. 绝对不要做的事

1. **绝不 `pip install mjlab` / `rsl-rl-lib`** —— 它们是 vendored 的副本（第 2 节）。
2. **绝不用 conda 创建环境** —— 用仓库的 `.venv`（第 1 节）。已有的 conda 安装放着别动；本
   项目不碰它。
3. **绝不装进一个已有的 Isaac Lab 环境** —— mjlab 需要 `mujoco~=3.11`，而 Isaac Lab 用的是
   `mujoco 3.10`，两者会互相破坏。每次 `pip` 之前，用 1.2 的验证确认 `sys.prefix` 指向仓库
   的 `.venv`。
4. **绝不试图让 `warp:cuda` 在 macOS 上跑起来** —— 这个平台没有 CUDA，也没有绕过的办法。
5. **绝不用 `warp:cpu` 训练** —— 它把 kernel 编译成 CPU 代码并串行执行，官方定位就是调试用
   的。用 `native:cpu`，它大约快 30×。
6. **绝不在没有 marker 的情况下改 `rl/mjlab/` 或 `rl/rsl_rl/`** —— 见
   [`VENDOR.md`](VENDOR.md)：每一处改动都必须带 `# [mjrl] reason: …`，否则和上游同步时就找
   不回来了。
