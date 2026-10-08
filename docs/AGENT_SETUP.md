# Setup instructions (for an AI agent)

This document is written for **an AI agent bringing mjrl-lab up on a fresh machine**.
Human readers should start from [`../README.md`](../README.md).

## Two scripts do the mechanical parts

Both are standard-library Python and run on their own -- no agent framework, no skill loader,
nothing to install:

```bash
python3 .claude/skills/setup-env/scripts/detect.py    # what this machine is, and what follows
.venv/bin/python .claude/skills/setup-env/scripts/gates.py   # whether it works, and what to fix
```

`detect.py` resolves section 0's decision table for the machine in front of you and prints the
resulting commands; `gates.py` runs section 3's six gates and names the remedy for each failure.
They are the executable form of this document, not a replacement for it -- when something does
not match, the explanation is here.

For an agent that loads skills, [`.claude/skills/setup-env/`](../.claude/skills/setup-env/)
wraps the same procedure.

## How to use this document

- **Follow the steps in order**, do not skip. The "verify" step at the end of each section is a
  **hard gate**: do not proceed until it passes.
- Every command states its **expected output**. When actual output differs, go to section 6 and
  look up the error string; do not guess or retry the same command repeatedly.
- On a **🛑 STOP** marker, stop immediately and report to the user. Do not work around it.
- All commands assume **the current working directory is the repository root** (the level
  containing `pyproject.toml`).

---

## 0. Preliminaries

Gather facts first, then decide which branch to take. **Do not skip this; every later step
depends on the result.**

```bash
python -c "import platform,sys; print('OS      :', platform.system()); print('ARCH    :', platform.machine()); print('PYTHON  :', sys.version.split()[0])"
```

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

If `nvidia-smi` is missing or errors, this machine has no usable NVIDIA GPU — take the
**no-GPU branch**.

### Decision table

Installation is identical on all three platforms (section 1); **only the torch step branches on
whether there is a GPU.** Which subsections each of the five combinations needs:

| OS | NVIDIA GPU | Subsections | Usable backends |
|---|---|---|---|
| Linux | yes | 1.1 – 1.4 | `warp:cuda` (primary), `native:cpu` |
| Linux | no | 1.1 – 1.3 | `native:cpu` |
| Windows | yes | 1.1 – 1.5 | `warp:cuda`, `native:cpu` |
| Windows | no | 1.1 – 1.3, **1.5** | `native:cpu` |
| macOS | — (no CUDA, no exceptions) | 1.1 – 1.3 | `native:cpu` |

Every row also takes 2.1 and 2.2, the controller's toolchain, which does not branch on the GPU.

### Python version

The framework requires **Python 3.10 – 3.13** (3.14 excluded).

If the version measured in section 0 is outside that range, **install a compliant one first**;
do not try to install the dependencies on an older interpreter. See the platform table in 1.1.

---

## 1. Create an isolated environment and install PyTorch

**There is one environment tool: the standard library's `venv`**, and the environment goes in
`.venv/` at the repository root. Do not use conda — in this workflow it only supplies an
interpreter, and every platform below has a way to do that. All dependencies are PyPI wheels;
no system CUDA toolkit is needed, only the NVIDIA driver.

### 1.1 Get a 3.10–3.13 interpreter

If the version from section 0 is compliant, use `python3` directly. **If not**, install one
first:

| Platform | What to do | Command to create the venv afterwards |
|---|---|---|
| Ubuntu 22.04 / 24.04 | Ships 3.10 / 3.12; nothing to do | `python3` |
| Ubuntu 20.04 and older | `sudo add-apt-repository ppa:deadsnakes/ppa`<br>`sudo apt install -y python3.11 python3.11-venv` | `python3.11` |
| macOS | `brew install python@3.12` | `python3.12` |
| Windows | Installer from python.org, tick "Add python.exe to PATH" | `py -3.11` |

> **A Debian / Ubuntu trap**: if `python3 -m venv` reports `ensurepip is not available`, the
> distribution has split venv into its own package. Install it:
> `sudo apt install -y python3-venv` (match the version to the interpreter, e.g.
> `python3.11-venv`).

### 1.2 Create and activate

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

> On Windows, if `Activate.ps1` reports an execution-policy error:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Verify (hard gate)**:

```bash
python -c "import sys; print(sys.prefix); print(sys.version.split()[0])"
```

The first line must be the `.venv` path inside the repository; the second must be between 3.10
and 3.13. **Pointing at the system Python or any conda environment is a failure** — do not
continue, because every subsequent `pip` will install into the wrong place and the symptom will
not surface until section 3.

### 1.3 Install PyTorch

**With a GPU**, pick the CUDA build from the `compute_cap` measured in section 0:

| compute_cap | Architecture | Required torch build |
|---|---|---|
| 12.0 | Blackwell (RTX 50 series) | **cu128 or newer** |
| 8.9 / 8.6 | Ada / Ampere | cu126 or newer |
| < 8.0 | older | Consult PyTorch's official support matrix |

> 🛑 **STOP**: installing a cu126 torch on Blackwell fails when CUDA initialises. If unsure of
> the compute capability, choose cu128 — it is backward compatible with Ada/Ampere.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Without a GPU** (including all of macOS) install the CPU build, and do **not** pass
`--index-url`:

```bash
pip install torch torchvision
```

> **`warp:cuda` is definitively unavailable on macOS** (Apple Silicon has no CUDA, no
> exceptions). CPU training there goes through the `native:cpu` backend, which needs no CUDA at
> all.

### 1.4 System prerequisite on GPU machines: the NVIDIA driver

**Machines without a GPU skip this subsection** and go to 1.5 (Windows) or section 2.

If `nvidia-smi` lists no card, **there is no driver**. Neither pip nor venv can help at this
layer; it must be installed at OS level:

- **Linux**: `ubuntu-drivers devices` for the recommendation, then
  `sudo apt install nvidia-driver-XXX`, and **reboot**
- **Windows**: the GeForce/Studio driver installer from NVIDIA's site
- **WSL2**: the driver goes on the **Windows host**; it cannot be installed inside WSL

> **No CUDA toolkit is needed.** The `CUDA Version` in the top-right of `nvidia-smi` is **the
> highest runtime the driver supports**, not "the installed toolkit". The cu128 torch wheels
> bring their own CUDA runtime and warp brings its own toolchain.

---

### 1.5 System prerequisite on Windows: a C++ compiler

**Applies with or without a GPU.** NVIDIA Warp JIT-compiles kernels on first run, including for
the CPU device, so a Windows machine without a GPU can hit this too.

If Gate A in section 3 reports compiler / MSVC errors, install the "Desktop development
with C++" workload of
[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and
retry in a fresh terminal.

> **Continue first and install it only if the error appears** — do not install pre-emptively;
> that workload is several GB. The exception is Rust (2.1), which links with it on every build.

---

## 2. Install this repository

```bash
pip install -e .
```

This installs the transitive dependencies listed in `pyproject.toml` (`mujoco`, `mujoco-warp`,
`warp-lang`, `tensordict`, `tyro`, `viser` and so on).

> ⚠️ **This step is mandatory and cannot be skipped.** The vendored `mjlab` / `rsl_rl` live
> under `rl/`, which is not on `sys.path` — they are importable only through the mapping this
> editable install creates (`pyproject.toml` declares `.` and `rl` as two package roots).
> Skip it and every entry point fails with
> `ModuleNotFoundError: No module named 'mjlab'`.

> 🛑 **Never run `pip install mjlab` or `pip install rsl-rl-lib`.** Both are present as
> **modifiable copies** under `rl/` (`rl/mjlab/`, `rl/rsl_rl/`; see [`VENDOR.md`](VENDOR.md)).
> Having pip versions in the same environment creates **shadowing ambiguity**: the same-named
> package in `site-packages` competes with the copy, behaves differently, and is very hard to
> notice.

If this environment ever had them, uninstall first (this does not remove their dependencies):

```bash
pip uninstall -y mjlab rsl-rl-lib
```

### 2.1 The controller's toolchain: Rust

The controller (`deploy/fsm`) is Rust, one crate on three hosts. `scripts/deploy.py` builds two of
them on this machine with cargo — the browser's wasm and `play --app`'s extension — and
cross-builds the board's binary and the Windows extension in Docker, which brings its own
toolchain. **Every row of section 0's decision table needs this subsection**; it does not branch
on the GPU.

> ⚠️ **Without cargo the test suite does not fail, it skips.** `tests/test_fsm_extension.py` and
> `tests/test_deploy_sources.py` skip when there is no cargo, so `pytest tests/` is green on a
> machine that has never built the controller.

Four pieces. `detect.py` prints only the ones this machine lacks:

| Piece | Why | Install |
|---|---|---|
| A C linker | build scripts and proc macros are linked for the host, even by `cargo check` | Linux: `sudo apt install -y build-essential`<br>macOS: `xcode-select --install`<br>Windows: 1.5 |
| rustup, stable, rustc ≥ 1.85 | the highest `rust-version` among the packages `Cargo.lock` pins (hashbrown 0.17.1 and indexmap 2.14.2, through `toml`; read on 2026-09-29) | below |
| the `wasm32-unknown-unknown` target | the browser's build | `rustup target add wasm32-unknown-unknown` |
| the `wasm-bindgen` CLI, **exactly** `Cargo.lock`'s version | the CLI and the crate share an ABI and version separately | `cargo install wasm-bindgen-cli --version <Cargo.lock's> --locked` |

```bash
# Linux / macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
```

```powershell
# Windows PowerShell, then open a new terminal
winget install --id Rustlang.Rustup -e
```

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked    # Cargo.lock's, on 2026-09-29
```

> **The wasm-bindgen version comes from `Cargo.lock`, not `Cargo.toml`.** The manifest says
> `"0.2"`, a range every 0.2.x satisfies and one `cargo install --version` refuses.
> `cargo install` without `--version` takes the newest, which is the lock's only until the next
> release. `detect.py`, gate E and `scripts/deploy.py` all hold the CLI to the lock.

> **A distribution's cargo is not enough.** `apt install cargo` answers `cargo --version`, but
> there is no rustup to add the wasm target with, and it is usually older than the lock needs.

> **On Windows the default toolchain links with MSVC**, so 1.5's Build Tools are needed for Rust
> whether or not warp ever asks for them.

`~/.cargo/bin` is on a login shell's `PATH` and often not on an editor's or a service's.
`deploy.py`, the tests and `gates.py` look there anyway; a shell needs a restart or
`. "$HOME/.cargo/env"`.

### 2.2 The device build: CycloneDDS and libclang

The crate's default feature, `device`, is the robot's DDS bus, and it is what a plain
`cargo test` in `deploy/fsm` and `tests/test_deploy_sources.py` build. `build.rs` runs `idlc`,
compiles its output against `$CYCLONEDDS_HOME` (default `/usr/local`), generates Rust bindings
through libclang and links `libddsc`. `deploy.py`'s host builds need none of it, and the board's
build has its own inside the Docker image — but once cargo is installed, `test_deploy_sources`
stops skipping and fails without it.

On Linux:

```bash
sudo apt install -y cmake git libclang-dev
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON
cmake --build /tmp/cyclonedds/build -j
sudo cmake --install /tmp/cyclonedds/build && sudo ldconfig
```

11.0.1 is what the development machine runs (i9-14900KF, Ubuntu 24.04), and its soname,
`libddsc.so.11`, is the board's. **`sudo ldconfig` is not optional**: `/usr/local/lib` is
searched through the loader's cache, and without it the test binary links and then cannot start
— which is why gate F runs it rather than only building it.

On macOS it is the same source build into a prefix of your own -- nothing under `/usr/local`,
no `sudo`, and no `ldconfig`, which macOS does not have. The crate writes the prefix's `lib/`
into the test binary's rpath from `CYCLONEDDS_HOME` instead, so the two exports below belong in
the shell that runs `cargo` and `gates.py`, not only in the one that installed:

```bash
brew install cmake
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/opt/cyclonedds-11.0.1 -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON
cmake --build /tmp/cyclonedds/build -j
cmake --install /tmp/cyclonedds/build
export CYCLONEDDS_HOME=$HOME/.local/opt/cyclonedds-11.0.1
export PATH=$CYCLONEDDS_HOME/bin:$PATH
```

bindgen finds libclang in Xcode's command-line tools. Measured on an M3 Max (Darwin 25.5,
cargo 1.95.0) on 2026-10-08: the build takes about two minutes, gate F lists 195 tests, and
`cargo test --lib` in `deploy/fsm` passes all of them. A test binary built before
`CYCLONEDDS_HOME` was exported carries no rpath and aborts with
`Library not loaded: @rpath/libddsc.11.dylib`; rebuild it (`cargo clean -p mjrl-fsm`) rather
than reinstalling CycloneDDS.

On Windows it is the same source build, with `CYCLONEDDS_HOME` at the install prefix, its
`bin/` on `PATH` (`idlc` is taken from `PATH`, the headers and the library from
`CYCLONEDDS_HOME`) and an LLVM install for bindgen (`LIBCLANG_PATH`). **Not measured.**

**Rockchip's NPU header is the one part fetched per checkout rather than installed per
machine.** `build.rs` also compiles a probe against `deploy/fsm/vendor/rknpu2/include/rknn_api.h`,
which is Rockchip's and is not committed (`deploy/fsm/vendor/rknpu2/README.md` says why). Once
in each checkout:

```bash
bash deploy/fsm/vendor/rknpu2/fetch.sh
```

It takes the pinned release and checks its sha256. Without it the device build stops in
`build.rs`, with a message naming the script.

---

## 3. Verification gates

**All six gates must pass.** On any failure, stop and look up section 6.

### Gate A — Warp can enumerate devices

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

**On a GPU machine**, stdout must be:

```
['cpu', 'cuda:0']
```

`wp.init()` also prints a banner to **stderr** with the card name and compute capability (e.g.
`sm_120`), the CUDA toolchain version and the kernel cache path. That is informational —
**do not use it as the criterion**. The criterion is whether `cuda:0` appears in the list on
stdout.

Seeing only `['cpu']` is a failure: the driver or CUDA toolchain has a problem and the
`warp:cuda` backend is unavailable.

**On a machine without a GPU**: `['cpu']` alone is the **expected** result, not a failure.

### Gate B — Native MuJoCo's multi-threaded batch interface

The `native:cpu` backend depends on it.

```bash
python -c "import mujoco; from mujoco import rollout; print('mujoco', mujoco.__version__)"
```

Expected: a version number and no exception. The version should be `3.11.x` or `3.12.x`.

### Gate C — The vendored copies are not shadowed by pip versions

**This is the easiest step to skip and the one whose failure is most obscure.**

```bash
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
```

Expected: **both paths inside this repository's `rl/`**, of the form:

```
/path/to/mjrl-lab/rl/mjlab
/path/to/mjrl-lab/rl/rsl_rl
```

If either contains `site-packages`, go back to section 2, run
`pip uninstall -y mjlab rsl-rl-lib`, then `pip install -e .` again.

A `ModuleNotFoundError` here means `pip install -e .` was not run or did not succeed — see the
⚠️ in section 2.

### Gate D — The task registry works

```bash
python scripts/train.py --list
```

Expected: ten tasks, all of them `jumper.*` — the four gait variants (flat / tripod /
tetrapod / ripple) plus `dance`, `five_foot`, `jump`, `posture`, `ref_free_jump` and `swing`.

This command **loads no simulation dependencies**, so it should pass even if the GPU is broken:
tasks register through lazy factories, and `import tasks` does not drag in mjlab / torch /
mujoco. A failure here is a packaging problem (most likely `pip install -e .` was not run), not
a simulation problem.

### Gate E — The Rust toolchain builds the controller's host targets

```bash
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --target wasm32-unknown-unknown --no-default-features --features web
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --no-default-features --features py
wasm-bindgen --version
```

Expected: both checks finish without an `error`, and the version equals the `wasm-bindgen` entry
in `deploy/fsm/Cargo.lock`.

This builds rather than reading versions: checking each host's features proves the target is
installed and that the host linker works, and `cargo --version` proves neither. Cold, on the
i9-14900KF with the crates already downloaded, the two checks take 8 s and 7 s; the first run on
a new machine also downloads them.

### Gate F — The device build links and loads

```bash
cargo test --locked --lib --manifest-path deploy/fsm/Cargo.toml -- --list
```

Expected: a list of lines ending in `: test` (179 of them on 2026-09-29). `--list` runs the test
binary without running a test, so it proves `libddsc` is found at load time — which a
`cargo check` never would — and a failing crate test cannot pass itself off as an environment
fault. 9 s cold on the same machine. A checkout without Rockchip's NPU header fails here, in
`build.rs`; the remedy is `fetch.sh` (section 2.2).

---

## 4. Smoke test

With all six gates passing, run a short training to confirm the pipeline works end to end.

### With a GPU

```bash
python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
```

Expected: the Actor/Critic network structure is printed, then several `Iteration time: ...`
lines, and the process exits with code 0.

### Without a GPU

```bash
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64 --max-iterations 3 --headless
```

Expected: the banner reports `backend=native device=cpu`, and the iterations complete.

`--headless` is only there to keep the smoke test independent of whether a display exists;
drop it to watch the run.

**Measured throughput** (i9-14900KF, 32 threads, hexapod with hybrid collision, `native:cpu`):
4096 environments at 26.0 s per iteration. On the same machine's RTX 5090, `warp:cuda` does the
same 4096 environments in 0.82 s per iteration — roughly 32× faster. A machine without a GPU can
train for real, but a GPU is worth about a factor of thirty.

---

## 5. Normal first-run behaviour

The following are **not errors**; do not try to fix them:

- **Warp compiles kernels on first run, which can take minutes**, and caches them under
  `~/.cache/warp/<version>/` (`%LOCALAPPDATA%` on Windows). The second run is much faster.
- `[INFO] <NullRecorderManager> (inactive)` is printed at training start.
- `Mean reward` is negative for the first several iterations.

---

## 6. Known failures and remedies

Indexed by **error string**.

### `ensurepip is not available` / `The virtual environment was not created successfully`

Debian / Ubuntu split venv into its own package. **Remedy**:

```bash
sudo apt install -y python3-venv     # or python3.11-venv, matching your interpreter
```

Then delete the half-built `.venv/` and recreate it: `rm -rf .venv && python3 -m venv .venv`.

### Windows: `Activate.ps1 cannot be loaded because running scripts is disabled on this system`

PowerShell's execution policy. **Remedy**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Packages installed elsewhere / `pip list` does not show what you just installed

The virtual environment was not activated. **Remedy**: go back to the verification in 1.2;
`sys.prefix` must point at the repository's `.venv`. This is the most common class of failure
and its symptom usually does not surface until gate C.

### `IndexError: list index out of range` (with `select_gpus` in the traceback)

Raised by mjlab's own GPU selection on a machine without one: with
`CUDA_VISIBLE_DEVICES` **unset** it goes through `torch.cuda.device_count()` (zero on a
card-less machine) while `gpu_ids` defaults to `[0]`, so the index is out of range. It only
enters CPU mode when the variable is set to an **empty string**.

**Remedy**: use `--backend native --device cpu`, which does not go through that path at all.
The entry points under `scripts/` resolve the backend themselves and never reach it.

### `ModuleNotFoundError: No module named 'wandb'`

`import mjlab.tasks` **imports every task package**, and
`tasks/manipulation/rl/runner.py` has a top-level `import wandb` that cannot be avoided.

**Remedy**: `pip install -e .` — `wandb` and `PyYAML` are listed in `pyproject.toml`. In an
older environment, `pip install wandb PyYAML`.

### `ValidationError: 1 validation error for Settings ... start_method ... extra_forbidden`

mjlab is incompatible with wandb 0.29.0. **Remedy**: `scripts/train.py` already defaults to
`--logger tensorboard`, so this only appears if wandb was asked for explicitly. (Downgrading
wandb also works, but tensorboard is less trouble.)

### `nefc overflow - please increase njmax to N`

`njmax` / `nconmax` are the per-world limits on constraint rows and contacts. mujoco_warp
infers defaults from the `mjData` of a resting pose, and a moving robot has many more contacts
than a resting one.

**Remedy**: raise them explicitly in the task config. jumper sets `cfg.sim.njmax = 512` and
`cfg.sim.nconmax = 128` (see `tasks/jumper/common/velocity_env.py`); for a new robot, take the N
from the error message and go above it.

### `"cuda" device requested but this build of Warp does not support CUDA`

Warp was installed as a CPU-only build, or the driver is not visible. **Remedy**: run gate A to
confirm device enumeration, then check that `nvidia-smi` works.

### `torch.cuda.is_available()` returns False

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

No `+cuXXX` in the version means the CPU build was installed; `cu126` on a Blackwell card means
a version mismatch. **Remedy**: reinstall the right CUDA build per section 1.

### Windows: Warp reports compiler / MSVC errors

Install the "Desktop development with C++" workload of Visual Studio Build Tools and retry in a
fresh terminal.

### `FileNotFoundError: .../assets/jumper/jumper.xml`

The asset is missing. It is normally in version control, and so is the URDF package it is
generated from (`assets/jumper/urdf/jumper/`); if it really is absent, regenerate it with the
generator script bound to that asset, whose `--src` and `--out` default to those two paths:

```bash
python assets/jumper/tools/build_jumper.py
```

### mjlab / rsl_rl imported from site-packages

See gate C.

### `cargo: command not found`, or `tests/test_fsm_extension.py` and `tests/test_deploy_sources.py` skipped

There is no Rust toolchain, or `~/.cargo/bin` is not on this shell's `PATH`. **Remedy**: 2.1; for
the second, `. "$HOME/.cargo/env"` or a new shell.

### `the wasm32-unknown-unknown target may not be installed` / ``can't find crate for `core` ``

**Remedy**: `rustup target add wasm32-unknown-unknown`. With a distribution's cargo there is no
rustup to run it with — install rustup (2.1).

### ``linker `cc` not found`` (`link.exe` on Windows)

No C linker. **Remedy**: the first row of the table in 2.1.

### `is not supported by the following packages` / `does not understand this lock file`

The toolchain is older than `Cargo.lock` needs. **Remedy**: `rustup update stable`.

### `wasm-bindgen CLI is …, the crate builds … (deploy/fsm/Cargo.lock)`

`scripts/deploy.py` holds the CLI to the version the lock builds, exactly (2.1). **Remedy**: the
command it prints, `cargo install wasm-bindgen-cli --version <Cargo.lock's> --locked`.

### `idlc not found (...); it ships with CycloneDDS`

**Remedy**: 2.2. If CycloneDDS is installed somewhere other than `/usr/local`, put its `bin/` on
`PATH` and set `CYCLONEDDS_HOME` to its prefix.

### `unable to find library -lddsc` / `Unable to find libclang`

The first: `libddsc` is not under `$CYCLONEDDS_HOME/lib`. The second: bindgen has no libclang —
`sudo apt install -y libclang-dev` on Linux. **Remedy**: 2.2.

### `error while loading shared libraries: libddsc.so.11`

Built and linked, and the loader cannot find the library. **Remedy**: `sudo ldconfig` after
installing into `/usr/local`, or add CycloneDDS's `lib/` to `LD_LIBRARY_PATH`.

### macOS: `Library not loaded: @rpath/libddsc.11.dylib`

The same failure under the macOS loader, which has no cache to refresh: the test binary's
rpath comes from `CYCLONEDDS_HOME` at build time, and this one was built before it was
exported, or against another prefix. **Remedy**: export `CYCLONEDDS_HOME` as in 2.2 and
rebuild (`cargo clean -p mjrl-fsm`); `DYLD_LIBRARY_PATH` pointed at the prefix's `lib/`
gets one run through without a rebuild.

---

## 7. Things never to do

1. **Never `pip install mjlab` / `rsl-rl-lib`** — they are vendored copies (section 2).
2. **Never create the environment with conda** — use the repository's `.venv` (section 1).
   Leave any existing conda installation alone; this project does not touch it.
3. **Never install into an existing Isaac Lab environment** — mjlab needs `mujoco~=3.11` while
   Isaac Lab uses `mujoco 3.10`, and they break each other. Before every `pip`, use the
   verification in 1.2 to confirm `sys.prefix` points at the repository's `.venv`.
4. **Never try to make `warp:cuda` work on macOS** — the platform has no CUDA and there is no
   workaround.
5. **Never train with `warp:cpu`** — it compiles kernels to CPU code and runs them serially, and
   is officially a debugging device. Use `native:cpu`, which is about 30× faster.
6. **Never modify `rl/mjlab/` or `rl/rsl_rl/` without a marker** — see [`VENDOR.md`](VENDOR.md):
   every change must carry `# [mjrl] reason: …`, or it cannot be recovered when syncing with
   upstream.
