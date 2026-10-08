#!/usr/bin/env python3
"""Report what this machine is, and which install path follows from it.

Run with **whatever interpreter the machine already has** -- that is the point.
It answers "is this Python usable at all", so it cannot assume a modern one, and
it cannot import anything that is not in the standard library of Python 3.6.

Every later step depends on these four facts, and each of them changes what the
install has to do:

    OS + arch      -> whether a C++ compiler is a prerequisite (Windows)
    Python version -> whether an interpreter has to be installed first
    GPU present    -> which torch wheel index, and whether a driver is required
    compute cap    -> which CUDA build; a cu126 torch dies on Blackwell

and, for the controller (`deploy/fsm`, Rust), what of its toolchain is already
here -- that part is additive, so only the missing steps are printed.

Prints a plain report and, at the end, the exact commands for this machine.
"""

from __future__ import print_function

import glob
import os
import platform
import re
import shutil
import subprocess
import sys

MIN_PY = (3, 10)
MAX_PY = (3, 13)  # inclusive; 3.14 is excluded by pyproject

# The highest `rust-version` among the packages `deploy/fsm/Cargo.lock` pins,
# read from their manifests on 2026-09-29: hashbrown 0.17.1 and indexmap 2.14.2,
# both through `toml`. A floor for the report only -- gate E builds the crate,
# which is the criterion, and says so when the lock moves past this.
MIN_RUST = (1, 85)

# The CycloneDDS the device build is known to work with: what the development
# machine (i9-14900KF, Ubuntu 24.04) runs, `idlc` 11.0.1, and a release whose
# soname, libddsc.so.11, is the board's.
CYCLONEDDS_TAG = "11.0.1"


def sh(cmd):
    """Run a command, returning (ok, output). Never raises: a missing tool is an
    answer, not an error."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, out.decode("utf-8", "replace").strip()
    except Exception as e:  # noqa: BLE001 - any failure means "not available"
        return False, str(e)


def metal_eligible(system, machine, release):
    """Whether this machine can simulate on its Apple GPU: Apple Silicon on
    macOS 15 or newer, which is what the warp-metal overlay needs. `release` is
    the Darwin kernel version (`platform.release()`); Darwin 24 is macOS 15.

    Eligibility, not presence: the overlay is the `metal` extra, installed per
    environment, and gates.py is what checks it is actually loaded.
    """
    if system != "Darwin" or machine != "arm64":
        return False
    try:
        return int(release.split(".")[0]) >= 24
    except ValueError:
        return False


def detect_gpu():
    """(has_gpu, name, compute_cap). No nvidia-smi means no usable NVIDIA GPU.

    macOS is decided without asking: Apple Silicon has no CUDA and there is no
    workaround, so a stray nvidia-smi there would be misleading. Its own GPU is
    a separate question, `metal_eligible`, and a separate backend.
    """
    if sys.platform == "darwin":
        return False, None, None
    ok, out = sh(["nvidia-smi", "--query-gpu=name,compute_cap",
                  "--format=csv,noheader"])
    if not ok or not out:
        return False, None, None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else None
    cap = parts[1] if len(parts) > 1 else None
    return True, name, cap


def torch_plan(has_gpu, cap):
    """Which torch wheel to install, and why.

    cu128 is the recommendation whenever the capability is unknown or >= 8.0: it
    is required on Blackwell (12.0) and backward compatible with Ada and Ampere,
    so choosing it cannot be wrong in that range -- whereas guessing cu126 on a
    Blackwell card installs cleanly and then dies when CUDA initialises, which
    is a far more expensive mistake than an unnecessarily new wheel.
    """
    if not has_gpu:
        return ("pip install torch torchvision",
                "CPU build -- no --index-url. native:cpu is the training backend here.")
    try:
        capf = float(cap)
    except (TypeError, ValueError):
        capf = None
    if capf is not None and capf < 8.0:
        return ("# consult https://pytorch.org for a build matching compute capability %s" % cap,
                "compute capability %s is below 8.0; the right wheel needs looking up." % cap)
    why = "cu128: required on Blackwell (12.0), backward compatible with Ada/Ampere."
    if capf is not None:
        why = "compute capability %s -> %s" % (cap, why)
    return ("pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu128", why)


def python_verdict():
    v = sys.version_info[:3]
    ok = MIN_PY <= v[:2] <= MAX_PY
    return ok, ".".join(str(x) for x in v)


def interpreter_advice(system):
    """What to install when the interpreter is out of range."""
    return {
        "Linux": ("Ubuntu 22.04/24.04 already ship 3.10/3.12. On older releases:\n"
                  "    sudo add-apt-repository ppa:deadsnakes/ppa\n"
                  "    sudo apt install -y python3.11 python3.11-venv\n"
                  "  then create the venv with python3.11"),
        "Darwin": "brew install python@3.12, then create the venv with python3.12",
        "Windows": ("Installer from python.org with \"Add python.exe to PATH\" ticked, "
                    "then create the venv with py -3.11"),
    }.get(system, "Install a CPython between 3.10 and 3.13.")


def find_repo():
    """The repository root, if we are standing in it.

    Identified by pyproject naming this project rather than by the directory
    name, which is routinely different (kk-rl-mjlab, mjrl-lab, a worktree path).
    """
    here = os.path.abspath(os.getcwd())
    while True:
        candidate = os.path.join(here, "pyproject.toml")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "rb") as fh:
                    text = fh.read().decode("utf-8", "replace")
                if 'name = "mjrl-lab"' in text:
                    return here
            except OSError:
                pass
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def which(name):
    """A tool on PATH, then in `$CARGO_HOME/bin`, where rustup and `cargo
    install` put things -- on the PATH of a login shell, and often not on that
    of whatever launched this."""
    found = shutil.which(name)
    if found:
        return found
    home = os.environ.get("CARGO_HOME") or os.path.join(os.path.expanduser("~"), ".cargo")
    for exe in (name, name + ".exe"):
        candidate = os.path.join(home, "bin", exe)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def version_of(path, flag="--version"):
    """(text, (major, minor, patch)) from `<tool> --version`, or (None, None).

    `flag` because idlc has no `--version`; it answers `-v`.
    """
    if not path:
        return None, None
    ok, out = sh([path, flag])
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out) if ok else None
    if not m:
        return None, None
    return m.group(0), tuple(int(x) for x in m.groups())


def locked_version(repo, package):
    """The version of `package` that `deploy/fsm/Cargo.lock` pins."""
    if not repo:
        return None
    try:
        with open(os.path.join(repo, "deploy", "fsm", "Cargo.lock"), "rb") as fh:
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    m = re.search(r'^name = "%s"\nversion = "([^"]+)"' % re.escape(package), text, re.MULTILINE)
    return m.group(1) if m else None


def detect_rust(repo):
    """What of the controller's toolchain is already on this machine.

    `rustup` is asked for separately from `cargo` because a distribution's
    cargo (`apt install cargo`) answers `cargo --version` and cannot add the
    wasm target -- it has no rustup to ask -- and is usually older than this
    lock needs. The rest of the install goes through rustup.
    """
    cargo = which("cargo")
    rustup = which("rustup")
    rustc_text, rustc = version_of(which("rustc"))
    targets = []
    if rustup:
        ok, out = sh([rustup, "target", "list", "--installed"])
        targets = out.split() if ok else []
    bindgen_text, _ = version_of(which("wasm-bindgen"))
    home = os.environ.get("CYCLONEDDS_HOME", "/usr/local")
    return {
        "cargo": cargo,
        "on_path": shutil.which("cargo") is not None,
        "rustup": rustup,
        "rustc_text": rustc_text,
        "rustc": rustc,
        "wasm32": "wasm32-unknown-unknown" in targets,
        "wasm_bindgen": bindgen_text,
        "wasm_bindgen_wanted": locked_version(repo, "wasm-bindgen"),
        "cc": which("cc"),
        "idlc": which("idlc"),
        "idlc_text": version_of(which("idlc"), "-v")[0],
        "dds_home": home,
        "dds_headers": os.path.isfile(os.path.join(home, "include", "dds", "dds.h")),
        "libclang": find_libclang(),
        # Per checkout, unlike everything above: Rockchip's header is fetched into
        # the repository (deploy/fsm/vendor/rknpu2/README.md). None when there is
        # no repository to look in.
        "npu_header": (os.path.isfile(os.path.join(
            repo, "deploy", "fsm", "vendor", "rknpu2", "include", "rknn_api.h"))
            if repo else None),
    }


def find_libclang():
    """Where bindgen will find libclang, on Linux; None where it will not.

    The directories clang-sys searches after `LIBCLANG_PATH`, not all of them:
    enough to tell a machine that has it from one that needs
    `libclang-dev`. Elsewhere it ships with the compiler (Xcode's command-line
    tools, an LLVM install on Windows), and gate F is what finds out.
    """
    if not sys.platform.startswith("linux"):
        return "not checked on this OS"
    dirs = [os.environ.get("LIBCLANG_PATH", "")] + glob.glob("/usr/lib/llvm-*/lib") + [
        "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
        "/usr/local/lib"]
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(d, "libclang*.so*"))) if d else []
        hits = [h for h in hits if "libclang-cpp" not in h]
        if hits:
            return hits[0]
    return None


def rust_commands(system, rust):
    """The steps still missing for the controller, in order, for this OS.

    Only what is missing: unlike the Python environment this is not built
    fresh per checkout, and re-running an installer that is already satisfied
    is noise at best.
    """
    steps = []
    fresh = rust["rustup"] is None
    if system == "Windows":
        if fresh:
            steps.append("winget install --id Rustlang.Rustup -e    # then open a new terminal;"
                         " it links with MSVC (section 1.5)")
    else:
        if rust["cc"] is None:
            steps.append("sudo apt install -y build-essential curl" if system == "Linux"
                         else "xcode-select --install")
        if fresh:
            if rust["cargo"]:
                steps.append("# cargo here is not rustup's (a distribution package?): it cannot "
                             "add the wasm target")
            steps.append("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
                         " | sh -s -- -y --profile minimal")
            steps.append('. "$HOME/.cargo/env"')
    if not fresh and rust["rustc"] and rust["rustc"][:2] < MIN_RUST:
        steps.append("rustup update stable")
    if fresh or not rust["wasm32"]:
        steps.append("rustup target add wasm32-unknown-unknown")
    wanted = rust["wasm_bindgen_wanted"]
    if wanted and rust["wasm_bindgen"] != wanted:
        steps.append("cargo install wasm-bindgen-cli --version %s --locked" % wanted)
    return steps


def device_commands(system, rust):
    """The device build's prerequisites: CycloneDDS (idlc, headers, libddsc) and
    libclang for bindgen. The source build is the Dockerfile's host half, at the
    release this machine family runs. Rockchip's NPU header is the one step that
    belongs to the checkout rather than the machine, so a second checkout can be
    missing it on a machine that has everything else."""
    steps = []
    if rust["npu_header"] is False:
        steps.append("bash deploy/fsm/vendor/rknpu2/fetch.sh    # Rockchip's NPU header, once per checkout")
    if system == "Linux" and rust["libclang"] is None:
        steps.append("sudo apt install -y libclang-dev")
    if rust["idlc"] and not rust["idlc_text"] and system == "Linux":
        # Installed and not running: idlc links libddsc, so this is the loader,
        # and rebuilding CycloneDDS would not be the fix.
        steps.append("sudo ldconfig    # idlc is installed and does not run: libddsc is not "
                     "in the loader's cache")
        return steps
    if rust["idlc"] and rust["dds_headers"]:
        return steps
    src = "/tmp/cyclonedds"
    if system == "Darwin":
        # The same source build into a prefix of one's own: no sudo, nothing under
        # /usr/local, and no ldconfig, which macOS does not have -- the crate writes
        # the prefix's lib/ into the test binary's rpath from CYCLONEDDS_HOME
        # instead, so the exports belong in the shell that runs cargo. Measured on
        # an M3 Max on 2026-10-08 (section 2.2 of docs/AGENT_SETUP.md).
        prefix = "$HOME/.local/opt/cyclonedds-%s" % CYCLONEDDS_TAG
        return steps + [
            "brew install cmake",
            "git clone --depth 1 --branch %s https://github.com/eclipse-cyclonedds/cyclonedds.git %s"
            % (CYCLONEDDS_TAG, src),
            "cmake -S %s -B %s/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%s"
            " -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON" % (src, src, prefix),
            "cmake --build %s/build -j" % src,
            "cmake --install %s/build" % src,
            "export CYCLONEDDS_HOME=%s PATH=%s/bin:$PATH"
            "    # in the shell that builds: the crate takes its rpath from it" % (prefix, prefix),
        ]
    if system != "Linux":
        return steps + ["# CycloneDDS %s from source with cmake, CYCLONEDDS_HOME at its prefix,"
                " idlc on PATH" % CYCLONEDDS_TAG,
                "# -- section 2.2 of docs/AGENT_SETUP.md; measured on Linux and macOS only"]
    return steps + [
        "sudo apt install -y cmake git",
        "git clone --depth 1 --branch %s https://github.com/eclipse-cyclonedds/cyclonedds.git %s"
        % (CYCLONEDDS_TAG, src),
        "cmake -S %s -B %s/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local"
        " -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON" % (src, src),
        "cmake --build %s/build -j" % src,
        "sudo cmake --install %s/build && sudo ldconfig" % src,
    ]


def existing_env(repo):
    """An environment that already exists, and whether it is this repo's .venv.

    Reported rather than assumed away, because the common case after the first
    run is not a bare machine -- it is an environment that exists and has gone
    wrong, and blindly creating a second one hides the problem instead of
    fixing it.
    """
    rows = []
    if repo:
        for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
            p = os.path.join(repo, rel)
            if os.path.isfile(p):
                rows.append(("repo .venv", p))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv or os.environ.get("CONDA_PREFIX"):
        kind = "conda env" if os.environ.get("CONDA_PREFIX") else "active venv"
        rows.append((kind + " (currently active)", sys.executable))
    return rows


def main():
    system = platform.system()
    py_ok, py_ver = python_verdict()
    has_gpu, gpu_name, cap = detect_gpu()
    repo = find_repo()
    cmd, why = torch_plan(has_gpu, cap)

    print("== machine ==")
    print("  os            : %s (%s)" % (system, platform.machine()))
    print("  python        : %s  %s" % (py_ver, "OK" if py_ok else "OUT OF RANGE (need 3.10-3.13)"))
    print("  interpreter   : %s" % sys.executable)
    if has_gpu:
        print("  nvidia gpu    : %s (compute capability %s)" % (gpu_name, cap))
    elif system == "Darwin":
        print("  nvidia gpu    : none -- macOS has no CUDA, no exceptions")
    else:
        print("  nvidia gpu    : none detected (no nvidia-smi, or it failed)")
    print("  repository    : %s" % (repo or "NOT FOUND -- run this from inside the repo"))

    envs = existing_env(repo)
    print("\n== environments already present ==")
    if envs:
        for label, path in envs:
            print("  %-28s %s" % (label, path))
        print("  -> One exists. Verify it with gates.py before building another;")
        print("     a second environment hides the fault instead of fixing it.")
    else:
        print("  none")

    print("\n== what follows ==")
    if not py_ok:
        print("  ! Install a 3.10-3.13 interpreter FIRST. Do not install dependencies")
        print("    on this one -- the failures come much later and read as unrelated.")
        print("    %s" % interpreter_advice(system))
    metal = metal_eligible(system, platform.machine(), platform.release())
    print("  backends available here : %s" % (
        "warp:cuda (primary), native:cpu" if has_gpu
        else "warp on metal:0 (with the `metal` extra), native:cpu" if metal
        else "native:cpu"))
    if metal:
        print("  apple gpu               : Apple Silicon on macOS 15+ -- the `metal` extra puts the")
        print("                            simulation on it (section 1.6); measured 2.8x native:cpu")
        print("                            at 4096 envs on an M3 Max, ~48x slower than an RTX 5090")
    if has_gpu:
        print("  prerequisite            : NVIDIA driver (nvidia-smi already answers, so it is present)")
    if system == "Windows":
        print("  prerequisite            : a C++ compiler (MSVC Build Tools) -- warp compiles kernels")
    print("  torch                   : %s" % why)

    rust = detect_rust(repo)
    rustc_ok = rust["rustc"] is not None and rust["rustc"][:2] >= MIN_RUST
    wanted = rust["wasm_bindgen_wanted"]
    print("\n== the controller's toolchain (deploy/fsm, Rust) ==")
    print("  cargo         : %s" % (rust["cargo"] or "not found"))
    if rust["cargo"] and not rust["on_path"]:
        print("                  not on PATH -- a new shell, or . \"$HOME/.cargo/env\"")
    print("  rustup        : %s" % (rust["rustup"] or "not found"))
    print("  rustc         : %s  %s" % (rust["rustc_text"] or "-", "OK" if rustc_ok else
                                        "NEED >= %d.%d" % MIN_RUST))
    print("  wasm32 target : %s" % ("installed" if rust["wasm32"] else "missing"))
    print("  wasm-bindgen  : %s  %s" % (rust["wasm_bindgen"] or "not found",
                                         "OK" if rust["wasm_bindgen"] == wanted
                                         else "Cargo.lock pins %s" % wanted))
    if system != "Windows":
        print("  C linker (cc) : %s" % (rust["cc"] or "not found"))
    idlc = rust["idlc_text"] or ("at %s, and it does not run" % rust["idlc"] if rust["idlc"]
                                 else "not found")
    print("  CycloneDDS    : idlc %s, headers %s under %s" % (
        idlc, "present" if rust["dds_headers"] else "missing", rust["dds_home"]))
    print("  libclang      : %s" % (rust["libclang"] or "not found"))
    print("  NPU header    : %s" % {True: "fetched", False: "missing (once per checkout)",
                                   None: "-"}[rust["npu_header"]])

    print("\n== commands ==")
    venv_py = "python3" if system != "Windows" else "py -3.11"
    activate = (".venv\\Scripts\\Activate.ps1" if system == "Windows"
                else "source .venv/bin/activate")
    print("  %s -m venv .venv" % venv_py)
    print("  %s" % activate)
    print("  %s" % cmd)
    print("  pip install -e .")
    if metal:
        print("  pip install -e \".[metal]\"    # the Apple GPU: warp-metal + the patched mujoco_warp (section 1.6)")
    steps = rust_commands(system, rust)
    print("\n  # the controller's toolchain (section 2.1)%s" % ("" if steps else
                                                          ": nothing missing"))
    for step in steps:
        print("  %s" % step)
    steps = device_commands(system, rust)
    print("\n  # the device build: CycloneDDS, libclang, the NPU header (section 2.2)%s" % (
        "" if steps else ": nothing missing"))
    for step in steps:
        print("  %s" % step)
    print("\n  Then verify:  python <skill>/scripts/gates.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
