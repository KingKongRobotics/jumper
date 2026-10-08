"""The environment-setup skill under `.claude/skills/setup-env/`.

The skill exists so that getting this repository running on a new machine is a
measured procedure rather than a remembered one. Its two scripts are the part
that can rot silently: they encode decisions (which torch wheel, which paths
count as "this checkout") that are right today and would keep printing confident
output after becoming wrong.

They are checked here rather than by running them, because what matters is the
decisions, and the decisions can be exercised on any machine -- including the
Blackwell case, which no CI runner is going to have.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".claude" / "skills" / "setup-env"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SKILL / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_skill_is_complete() -> None:
    """A skill missing a piece fails at the moment someone needs it most."""
    assert (SKILL / "SKILL.md").is_file()
    for script in ("detect.py", "gates.py"):
        assert (SKILL / "scripts" / script).is_file(), f"{script} is missing"


def test_frontmatter_declares_name_and_description() -> None:
    """Both are required, and the description is what decides whether the skill
    is ever consulted -- it is the only part most readers see."""
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md needs YAML frontmatter"
    front = text.split("---", 2)[1]
    assert "\nname: setup-env" in front
    assert "\ndescription:" in front
    body = text.split("---", 2)[2]
    assert "docs/AGENT_SETUP.md" in body, (
        "the skill must point at the reference doc rather than replace it"
    )


def test_detect_runs_on_an_interpreter_it_cannot_choose() -> None:
    """`detect.py` answers "is this Python usable", so it must run on one that
    is not.

    It is invoked with whatever the machine already has -- possibly a 3.8 that
    the repository does not support. Importing a dependency, or using syntax
    newer than it needs, would turn "your interpreter is too old" into a
    SyntaxError with no explanation attached.
    """
    src = (SKILL / "scripts" / "detect.py").read_text(encoding="utf-8")
    imports = {
        ln.split()[1].split(".")[0]
        for ln in src.splitlines()
        if ln.startswith("import ") or ln.startswith("from ")
    }
    # All of these predate 3.6; `shutil.which` is the newest, from 3.3.
    stdlib = {"__future__", "glob", "os", "platform", "re", "shutil", "subprocess", "sys"}
    assert imports <= stdlib, (
        f"detect.py may only use the standard library, found: {sorted(imports)}"
    )
    assert ":=" not in src, "the walrus operator needs 3.8; keep this runnable on older ones"

    # And it must actually run to completion, exit 0, on this interpreter.
    out = subprocess.run([sys.executable, str(SKILL / "scripts" / "detect.py")],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "== commands ==" in out.stdout


@pytest.mark.parametrize(
    ("has_gpu", "cap", "expect"),
    [
        (True, "12.0", "cu128"),   # Blackwell: cu126 installs cleanly and dies in CUDA init
        (True, "8.9", "cu128"),    # Ada: cu128 is backward compatible, so it is never wrong here
        (True, "8.6", "cu128"),    # Ampere: same
        (False, None, None),       # no GPU (all of macOS): CPU build, and no --index-url
    ],
)
def test_the_torch_wheel_follows_from_the_measured_capability(has_gpu, cap, expect) -> None:
    """Choosing the wheel is the only step that branches, and the expensive
    direction is asymmetric.

    Too new a CUDA build costs nothing on Ada/Ampere; too old a one on Blackwell
    installs without complaint and fails when CUDA initialises, which reads as a
    driver problem. So "unknown or >= 8.0 -> cu128" is deliberate, not vague.
    """
    cmd, _why = _load("detect").torch_plan(has_gpu, cap)
    if expect is None:
        assert "--index-url" not in cmd, "the CPU build must not be pointed at a CUDA index"
        assert "torch" in cmd
    else:
        assert expect in cmd, cmd


def test_an_ancient_card_is_referred_upward_rather_than_guessed() -> None:
    """Below 8.0 there is no single right answer, and inventing one would send
    someone down a long wrong path."""
    cmd, why = _load("detect").torch_plan(True, "7.5")
    assert "pytorch.org" in cmd or "pytorch.org" in why
    assert "cu128" not in cmd


def test_the_repo_is_found_by_content_not_by_name() -> None:
    """The checkout is routinely named something else.

    `kk-rl-mjlab` on the training machine, a hashed path inside `.claude/
    worktrees/` here. Matching on the directory name would fail on exactly the
    machines this is needed for.
    """
    gates = _load("gates")
    assert gates.find_repo(Path(__file__).parent) == REPO
    assert gates.find_repo(Path(REPO.anchor)) is None


def test_gate_c_compares_against_this_checkout_not_site_packages() -> None:
    """The published gate rules out `site-packages`, which is not enough.

    An editable install left pointing at an older copy of the repository passes
    that check and still runs another checkout's code -- measured on a training
    machine on 2026-09-04, where `tasks` resolved from the current repository
    and `mjlab` from a snapshot three days stale. Everything imported; the seam
    being edited was not the seam being run. So the comparison has to be against
    the repository root.
    """
    src = (SKILL / "scripts" / "gates.py").read_text(encoding="utf-8")
    body = src.split("def gate_c_vendored", 1)[1].split("\n    # ", 1)[0]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert "self.repo" in code, "gate C must compare against the repository root"
    assert '"site-packages"' not in code, (
        "a site-packages substring check is the weaker test this replaces"
    )


def test_wasm_bindgen_is_held_to_the_lock_not_the_manifest() -> None:
    """The CLI must equal the crate's wasm-bindgen exactly, and only
    `Cargo.lock` says what that is.

    `Cargo.toml` says `"0.2"`, a range every 0.2.x satisfies. Both scripts read
    the lock, as `scripts/deploy.py` does, and agree.
    The control is the manifest's own version: if it ever named the full
    version, the lock and the manifest would be indistinguishable here and
    this test would pass whichever file the scripts read.
    """
    import re

    detect, gates = _load("detect"), _load("gates")
    locked = detect.locked_version(str(REPO), "wasm-bindgen")
    assert locked == gates.locked_version(REPO, "wasm-bindgen")
    assert locked and re.fullmatch(r"\d+\.\d+\.\d+", locked), locked

    manifest = (REPO / "deploy" / "fsm" / "Cargo.toml").read_text(encoding="utf-8")
    declared = re.search(r'^wasm-bindgen\s*=\s*\{[^}]*version\s*=\s*"([^"]+)"', manifest, re.M)
    assert declared and declared.group(1) != locked, (
        "the manifest now pins the full version; this test no longer tells the files apart"
    )

    rust = _complete_rust(wasm_bindgen="0.2.0")
    assert f"cargo install wasm-bindgen-cli --version {locked} --locked" in (
        detect.rust_commands("Linux", dict(rust, wasm_bindgen_wanted=locked)))


def _complete_rust(**changes):
    """`detect_rust`'s answer for a machine that has everything."""
    rust = {
        "cargo": "/c/cargo", "on_path": True, "rustup": "/c/rustup",
        "rustc_text": "1.98.1", "rustc": (1, 98, 1), "wasm32": True,
        "wasm_bindgen": "0.2.128", "wasm_bindgen_wanted": "0.2.128", "cc": "/usr/bin/cc",
        "idlc": "/usr/local/bin/idlc", "idlc_text": "11.0.1", "dds_home": "/usr/local",
        "dds_headers": True, "libclang": "/usr/lib/libclang.so", "npu_header": True,
    }
    rust.update(changes)
    return rust


def test_only_the_missing_rust_steps_are_printed() -> None:
    """The toolchain is per machine, not per checkout, so a second checkout
    must be told "nothing missing" rather than handed an installer to re-run.

    And the step that is printed has to be the one that is missing: a
    distribution's cargo answers `cargo --version` and cannot add the wasm
    target, so its presence must not suppress the rustup install.
    """
    detect = _load("detect")
    assert detect.rust_commands("Linux", _complete_rust()) == []
    assert detect.device_commands("Linux", _complete_rust()) == []

    distro = detect.rust_commands("Linux", _complete_rust(rustup=None, wasm32=False))
    assert any("sh.rustup.rs" in step for step in distro), distro
    assert "rustup target add wasm32-unknown-unknown" in distro

    assert detect.rust_commands("Linux", _complete_rust(wasm32=False)) == [
        "rustup target add wasm32-unknown-unknown"]
    assert detect.rust_commands("Linux", _complete_rust(rustc=(1, 75, 0))) == [
        "rustup update stable"]

    # An idlc that is installed and does not run is the loader, not a missing
    # CycloneDDS: the answer is ldconfig, not a second source build.
    broken = detect.device_commands("Linux", _complete_rust(idlc_text=None))
    assert len(broken) == 1 and "ldconfig" in broken[0], broken

    # Rockchip's NPU header is the one per-checkout step: a checkout without it
    # is told to fetch it on a machine that has everything else. The complete
    # case above is the control.
    assert detect.device_commands("Linux", _complete_rust(npu_header=False)) == [
        "bash deploy/fsm/vendor/rknpu2/fetch.sh    # Rockchip's NPU header, once per checkout"]

    # macOS gets a measured procedure of its own rather than the "measured on
    # Linux only" comment it used to: a source build into a prefix under $HOME,
    # with no sudo and no ldconfig (the crate writes its rpath instead), and the
    # exports the build takes the rpath from. The complete case is the control
    # again: a machine with everything is told nothing on this platform too.
    assert detect.device_commands("Darwin", _complete_rust()) == []
    mac = detect.device_commands("Darwin", _complete_rust(idlc=None, idlc_text=None,
                                                          dds_headers=False))
    assert not any("sudo" in step or "ldconfig" in step for step in mac), mac
    assert any("CMAKE_INSTALL_PREFIX=$HOME/" in step for step in mac), mac
    assert any(step.startswith("export CYCLONEDDS_HOME=$HOME/") for step in mac), mac
