#!/usr/bin/env python3
"""Check that an mjrl-lab environment actually works, and say what to fix.

Run this **with the environment's own interpreter** (`.venv/bin/python gates.py`,
or whatever interpreter the environment uses) from anywhere; pass `--repo` if the
working directory is not inside the repository.

Each gate exists because its failure is quiet. An environment can import
everything, start training, and produce numbers, while the GPU is unused, or the
vendored code being executed belongs to a different checkout. So each gate prints
what it measured, not merely OK, and names the remedy -- an agent reading only
"FAIL" would have to go and rediscover it.

Exit code is the number of failed gates, so this drops into a script:

    python gates.py || echo "environment is not ready"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOC = "docs/AGENT_SETUP.md"
CRATE = Path("deploy") / "fsm"

#: What cargo says when the toolchain, not the source, is the problem, and what
#: to do about it. Matched against the tail of stderr; the tail is printed too,
#: so a message nobody listed here still reaches the reader.
RUST_HINTS = (
    ("may not be installed", "rustup target add wasm32-unknown-unknown"),
    ("linker `", ("no C linker: build-essential (Linux), xcode-select --install (macOS),\n"
                  f"MSVC Build Tools (Windows, section 1.5 of {DOC})")),
    ("is not supported by the following package", "rustup update stable"),
    ("does not understand this lock file", "rustup update stable"),
    ("feature `edition2024` is required", "rustup update stable"),
    ("no Python 3.x interpreter found",
     "pyo3's build script needs a python3 on PATH: activate the environment, or set PYO3_PYTHON"),
    ("lock file needs to be updated",
     ("Cargo.toml and Cargo.lock disagree in this checkout -- a source problem, "
      "not the environment")),
    ("idlc not found",
     f"CycloneDDS is not installed, or idlc is not on PATH: section 2.2 of {DOC}"),
    ("dds/dds.h", ("CycloneDDS headers are not under $CYCLONEDDS_HOME/include "
                   f"(default /usr/local): section 2.2 of {DOC}")),
    ("Unable to find libclang", "bindgen needs libclang: sudo apt install -y libclang-dev"),
    ("rknn_api.h is missing",
     ("Rockchip's NPU header is fetched, not committed, once per checkout: "
      "bash deploy/fsm/vendor/rknpu2/fetch.sh")),
    ("-lddsc", "libddsc is not under $CYCLONEDDS_HOME/lib (default /usr/local)"),
    ("error while loading shared libraries",
     ("built, and the loader cannot find libddsc: sudo ldconfig, or add its lib/ to "
      "LD_LIBRARY_PATH")),
    ("Library not loaded: @rpath/libddsc",
     ("built, and the macOS loader cannot find libddsc: the crate writes its rpath from "
      "$CYCLONEDDS_HOME at build time, so export it and rebuild (cargo clean -p mjrl-fsm), "
      "or add its lib/ to DYLD_LIBRARY_PATH for one run")),
)


def which(name: str) -> str | None:
    """A tool on PATH, then where `cargo install` and rustup put it.

    The same rule `scripts/deploy.py::which` follows, so this gate and the
    script agree on whether a tool is installed. `$CARGO_HOME/bin` is on the
    PATH of a login shell and often not on that of an editor or a service, and
    a tool found only there is installed, not missing.
    """
    found = shutil.which(name)
    if found:
        return found
    home = Path(os.environ.get("CARGO_HOME") or (Path.home() / ".cargo"))
    for candidate in (home / "bin" / name, home / "bin" / f"{name}.exe"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def locked_version(repo: Path, package: str) -> str | None:
    """The version of `package` that `deploy/fsm/Cargo.lock` pins."""
    lock = repo / CRATE / "Cargo.lock"
    if not lock.is_file():
        return None
    m = re.search(rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"',
                  lock.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else None


def find_repo(start: Path | None = None) -> Path | None:
    """The repository root, identified by pyproject rather than directory name.

    The directory is routinely named something else -- kk-rl-mjlab, a git
    worktree path -- so matching on the name would fail on the machines this is
    most needed on.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and 'name = "mjrl-lab"' in pyproject.read_text(
            encoding="utf-8", errors="replace"
        ):
            return candidate
    return None


class Gates:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.failed: list[str] = []

    def report(self, name: str, ok: bool, detail: str, remedy: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for line in detail.splitlines():
            print(f"       {line}")
        if not ok:
            self.failed.append(name)
            if remedy:
                for line in remedy.splitlines():
                    print(f"       -> {line}")
        print()

    # ── A ───────────────────────────────────────────────────────────────
    def gate_a_warp(self) -> None:
        """Whether warp can enumerate devices.

        The criterion is `cuda:0` in the device list on a GPU machine. wp.init()
        also prints a banner to stderr naming the card -- that banner appears
        even when CUDA is unusable, so reading it instead of the list is how a
        broken GPU gets mistaken for a working one.
        """
        try:
            import warp as wp

            wp.init()
            devices = [str(d) for d in wp.get_devices()]
        except Exception as e:  # noqa: BLE001
            self.report("A  warp device enumeration", False, f"{type(e).__name__}: {e}",
                        f"warp is missing or broken; see section 2 of {DOC}")
            return

        has_gpu = self._nvidia_present()
        detail = f"devices: {devices}"
        if has_gpu:
            self.report("A  warp device enumeration", "cuda:0" in devices,
                        detail + "  (GPU machine: cuda:0 required)",
                        "the driver or CUDA toolchain is wrong -- warp:cuda is unavailable.\n"
                        f"see 'torch.cuda.is_available() returns False' in section 6 of {DOC}")
        elif self._warp_metal_installed():
            # The `metal` extra is in this environment, so the overlay has to be
            # active: it loads only beside the exact warp-lang it was built for,
            # and otherwise Warp starts unmodified with a line on stderr -- which
            # is a working CPU-only Warp and a silently missing GPU.
            self.report("A  warp device enumeration", "metal:0" in devices,
                        detail + "  (warp-metal installed: metal:0 required)",
                        "warp-metal is installed and Warp lists no Metal device: the warp-lang "
                        "version does not match the overlay's (python -c 'import warp_metal; "
                        f"print(warp_metal.status())'), or macOS is older than 15. Section 1.6 of {DOC}")
        else:
            self.report("A  warp device enumeration", True,
                        detail + "  (no GPU: ['cpu'] alone is the expected result)")

    @staticmethod
    def _warp_metal_installed() -> bool:
        import importlib.util

        return importlib.util.find_spec("warp_metal") is not None

    @staticmethod
    def _nvidia_present() -> bool:
        if sys.platform == "darwin":
            return False
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── B ───────────────────────────────────────────────────────────────
    def gate_b_mujoco(self) -> None:
        """native:cpu is built on `mujoco.rollout`, so importing mujoco is not enough."""
        try:
            import mujoco
            from mujoco import rollout  # noqa: F401

            v = mujoco.__version__
            ok = v.startswith(("3.11", "3.12"))
            self.report("B  native MuJoCo batch interface", ok, f"mujoco {v}",
                        "expected 3.11.x or 3.12.x; pyproject pins mujoco~=3.11.0.\n"
                        "an Isaac Lab environment carries 3.10 and the two break each other")
        except Exception as e:  # noqa: BLE001
            self.report("B  native MuJoCo batch interface", False, f"{type(e).__name__}: {e}",
                        "pip install -e . was not run, or ran into the wrong environment")

    # ── C ───────────────────────────────────────────────────────────────
    def gate_c_vendored(self) -> None:
        """The vendored copies must come from **this** checkout's `rl/`.

        The published gate only rules out site-packages, which is not enough. An
        editable install left pointing at an older copy of the repository
        satisfies "not site-packages" while executing code from a different
        checkout -- measured on a training machine on 2026-09-04, where `tasks`
        resolved from the current repository and `mjlab` from a snapshot three
        days stale. Everything imported; the seam simply was not the one being
        edited. So compare against the repository root, not against a substring.
        """
        try:
            import mjlab
            import rsl_rl
        except Exception as e:  # noqa: BLE001
            self.report("C  vendored copies are this checkout's", False,
                        f"{type(e).__name__}: {e}",
                        "pip install -e . creates the mapping that makes rl/ importable;\n"
                        f"without it nothing under rl/ is on sys.path. See section 2 of {DOC}")
            return

        want = (self.repo / "rl").resolve()
        rows, ok = [], True
        for mod in (mjlab, rsl_rl):
            path = Path(mod.__file__).resolve().parent
            inside = path == want / path.name
            ok = ok and inside
            rows.append(f"{mod.__name__:7s} {path}{'' if inside else '   <-- not this repo'}")
        self.report("C  vendored copies are this checkout's", ok,
                    "\n".join(rows) + f"\nexpected under: {want}",
                    "pip uninstall -y mjlab rsl-rl-lib   # if they came from PyPI\n"
                    f"pip install -e .                    # from {self.repo}\n"
                    "then re-run this gate: an editable install can point at another checkout")

    # ── D ───────────────────────────────────────────────────────────────
    def gate_d_registry(self) -> None:
        """The registry, exercised the way a user meets it.

        This loads no simulation dependencies -- tasks register through lazy
        factories -- so it passes even where the GPU is broken. A failure here is
        a packaging problem, not a simulation one, and that distinction saves
        looking in the wrong place.
        """
        train = self.repo / "scripts" / "train.py"
        try:
            out = subprocess.run([sys.executable, str(train), "--list"],
                                 cwd=self.repo, capture_output=True, text=True, timeout=180)
        except Exception as e:  # noqa: BLE001
            self.report("D  task registry", False, f"{type(e).__name__}: {e}")
            return
        tasks = [ln.split()[0] for ln in out.stdout.splitlines()
                 if ln and not ln.startswith((" ", "-", "task id")) and "." in ln.split()[0]]
        ok = out.returncode == 0 and len(tasks) >= 5
        detail = (f"{len(tasks)} tasks: {', '.join(tasks)}" if tasks
                  else (out.stderr or out.stdout).strip()[:400])
        self.report("D  task registry", ok, detail,
                    "expected at least 5 (ten register today: four jumper gaits, "
                    "plus dance, five_foot, jump, posture, ref_free_jump and swing).\n"
                    "a packaging problem -- most likely pip install -e . was not run")

    # ── E ───────────────────────────────────────────────────────────────
    def cargo(self, cargo: str, *args: str,
              passthrough: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
        """Run cargo on the controller's crate.

        `--locked` always: a gate must not rewrite `Cargo.lock`, and the
        toolchain floor this setup states was measured against the locked
        versions, not against whatever resolves today. `passthrough` goes
        after `--`, to the test binary, so it has to follow the options that
        are cargo's.
        """
        return subprocess.run(
            [cargo, *args, "--locked", "--manifest-path", str(self.repo / CRATE / "Cargo.toml"),
             *(("--", *passthrough) if passthrough else ())],
            cwd=self.repo, capture_output=True, text=True, timeout=900, check=False)

    @staticmethod
    def errors(stderr: str) -> list[str]:
        """The lines of cargo's stderr worth reading: its errors, the last few.

        Not simply the tail. A link failure ends in the linker's command line,
        one line of several kilobytes that buries the one naming the library.
        """
        lines = stderr.strip().splitlines()
        shown = [ln for ln in lines if "error" in ln.lower()][-6:] or lines[-6:]
        return ["  " + (ln if len(ln) <= 160 else ln[:157] + "...") for ln in shown]

    @staticmethod
    def hints(stderr: str) -> list[str]:
        """The remedy for every message in `RUST_HINTS` that stderr contains."""
        return list(dict.fromkeys(fix for needle, fix in RUST_HINTS if needle in stderr))

    def gate_e_rust(self) -> None:
        """The controller's host builds: cargo, the wasm target, pyo3, wasm-bindgen.

        `deploy/fsm` is the controller on all three hosts, and `scripts/deploy.py`
        builds two of them here with cargo -- the browser's wasm and `play
        --app`'s extension (the board's binary and the Windows extension are
        cross-built in Docker, not by this toolchain). The failure this gate is
        for is quiet at the level of the test suite: without cargo,
        `tests/test_fsm_extension.py` and `tests/test_deploy_sources.py` skip,
        and `pytest tests/` is green on a machine that has never built the
        controller.

        So it builds rather than reading versions. `cargo check` of each host's
        features proves the target is installed and that build scripts and
        proc macros link on this machine, which `cargo --version` does not.

        The wasm-bindgen CLI is held to `Cargo.lock`, as `scripts/deploy.py`
        holds it: the CLI and the crate share an ABI and version separately,
        and `Cargo.toml`'s `"0.2"` is a range every 0.2.x satisfies. Checked
        here so a mismatch is found while setting up, not at the first bundle.
        """
        name = "E  Rust toolchain builds the controller's host targets"
        cargo = which("cargo")
        if cargo is None:
            self.report(name, False, "cargo: not on PATH and not in $CARGO_HOME/bin",
                        f"install rustup: section 2.1 of {DOC}")
            return

        rows, ok, remedies = [], True, []
        version = subprocess.run([cargo, "--version"], capture_output=True, text=True,
                                 check=False).stdout.strip()
        on_path = shutil.which("cargo") is not None
        rows.append(f"{version or 'cargo --version failed'}  ({cargo})")
        if not on_path:
            rows.append("  not on PATH: deploy.py and the tests look in $CARGO_HOME/bin too; "
                        "a shell will not until it sources ~/.cargo/env")

        for label, args in (
            ("web  (wasm32)", ["check", "--lib", "--target", "wasm32-unknown-unknown",
                               "--no-default-features", "--features", "web"]),
            ("py   (play --app)", ["check", "--lib", "--no-default-features",
                                   "--features", "py"]),
        ):
            try:
                out = self.cargo(cargo, *args)
            except Exception as e:  # noqa: BLE001
                ok = False
                rows.append(f"{label}: {type(e).__name__}: {e}")
                continue
            ok = ok and out.returncode == 0
            rows.append(f"{label}: {'builds' if out.returncode == 0 else 'FAILS'}")
            if out.returncode != 0:
                rows += self.errors(out.stderr)
                remedies += self.hints(out.stderr)

        wanted = locked_version(self.repo, "wasm-bindgen")
        cli = which("wasm-bindgen")
        have = None
        if cli:
            have = subprocess.run([cli, "--version"], capture_output=True, text=True,
                                  check=False).stdout.split()[-1:]
            have = have[0] if have else None
        match = cli is not None and have == wanted
        ok = ok and match
        rows.append(f"wasm-bindgen CLI: {have or 'not found'}, Cargo.lock pins {wanted}")
        if not match:
            remedies.append(f"cargo install wasm-bindgen-cli --version {wanted} --locked")

        self.report(name, ok, "\n".join(rows), "\n".join(dict.fromkeys(remedies)))

    # ── F ───────────────────────────────────────────────────────────────
    def gate_f_device(self) -> None:
        """The device build: CycloneDDS, libclang, and a libddsc the loader finds.

        The crate's default feature is `device`, the robot's DDS bus, and it is
        what a plain `cargo test` and `tests/test_deploy_sources.py` build.
        `build.rs` runs `idlc`, compiles its output against `$CYCLONEDDS_HOME`
        (default /usr/local), generates bindings through libclang, links libddsc
        and compiles a probe against Rockchip's NPU header, which each checkout
        fetches (deploy/fsm/vendor/rknpu2/fetch.sh). deploy.py's host builds need
        none of it, and the board's build
        has its own inside the Docker image -- but once cargo is installed,
        test_deploy_sources stops skipping and fails without it.

        Built and **run**: `--list` executes the test binary without running a
        test, so a libddsc that links and then cannot be loaded fails here, as
        `cargo check` never would. A failing crate test is not an environment
        fault, and listing keeps this gate from reporting one.
        """
        name = "F  device build (CycloneDDS, libclang)"
        cargo = which("cargo")
        if cargo is None:
            self.report(name, False, "cargo: not found (gate E)", f"section 2.1 of {DOC} first")
            return
        idlc = which("idlc")
        home = os.environ.get("CYCLONEDDS_HOME", "/usr/local")
        detail = [f"idlc: {idlc or 'not on PATH'}", f"CYCLONEDDS_HOME: {home}"]
        try:
            out = self.cargo(cargo, "test", "--lib", passthrough=("--list",))
        except Exception as e:  # noqa: BLE001
            self.report(name, False, "\n".join(detail + [f"{type(e).__name__}: {e}"]))
            return
        listed = sum(1 for ln in out.stdout.splitlines() if ln.endswith(": test"))
        ok = out.returncode == 0 and listed > 0
        if ok:
            detail.append(f"built, linked and loaded; {listed} tests listed")
        else:
            detail += ["cargo test --lib: FAILS", *self.errors(out.stderr)]
        self.report(name, ok, "\n".join(detail),
                    "\n".join([*self.hints(out.stderr), f"see section 2.2 of {DOC}"]))

    def run(self) -> int:
        print(f"repository: {self.repo}")
        print(f"interpreter: {sys.executable}\n")
        self.gate_a_warp()
        self.gate_b_mujoco()
        self.gate_c_vendored()
        self.gate_d_registry()
        self.gate_e_rust()
        self.gate_f_device()
        if self.failed:
            print(f"{len(self.failed)} gate(s) failed: {', '.join(self.failed)}")
            print(f"Remedies are above; the long forms are in section 6 of {DOC}.")
        else:
            print("All six gates passed. Next: the smoke test in section 4 of "
                  f"{DOC}.")
        return len(self.failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=None,
                    help="repository root; found from the working directory if omitted")
    args = ap.parse_args()
    repo = args.repo.resolve() if args.repo else find_repo()
    if repo is None or not (repo / "pyproject.toml").is_file():
        print("error: not inside an mjrl-lab checkout; pass --repo", file=sys.stderr)
        return 2
    return Gates(repo).run()


if __name__ == "__main__":
    sys.exit(main())
