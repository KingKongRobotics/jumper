"""Colab must preserve the selected run, not silently choose one that happens to exist.

These tests use valid ZIP-shaped fake checkpoints and real short-lived subprocesses.
They need no torch, MuJoCo, CUDA or Colab kernel; GPU acceptance is a separate gate.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import stat
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("colab_helpers", REPO / "rl/mjrl/colab.py")
assert SPEC is not None and SPEC.loader is not None
colab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab)


def config() -> dict:
    return {
        "task": "robot.walk", "model": "robot", "num_envs": 256,
        "backend": "warp", "device": "cuda:0", "revision": "a" * 40,
        "source_sha256": "b" * 64, "runtime_versions": {"torch": "test"},
        "mjrl_environment": {},
    }


def checkpoint(path: Path, value: str = "weights") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", value)


def saved_run(path: Path) -> Path:
    model = path / "model_10.pt"
    checkpoint(model)
    colab.atomic_json(path / colab.MANIFEST, {
        "schema": 1, "config": config(), "status": "finished",
        "checkpoints": {model.name: colab.sha256(model)},
    })
    return model


def test_resume_checks_every_compatibility_field_and_the_selected_file(tmp_path):
    """A valid unrelated latest run is a control: it must never make a mismatch pass."""
    chosen = saved_run(tmp_path / "chosen")
    unrelated = saved_run(tmp_path / "unrelated")
    checkpoint(unrelated.parent / "model_999.pt")
    colab.validate_resume(chosen, config())
    for key in colab.COMPATIBILITY:
        changed = config()
        changed[key] = "different"
        with pytest.raises(ValueError, match=key):
            colab.validate_resume(chosen, changed)
    checkpoint(chosen, "replacement")
    with pytest.raises(ValueError, match="does not match"):
        colab.validate_resume(chosen, config())
    with pytest.raises(ValueError, match="explicit"):
        colab.validate_resume(chosen.parent, config())


def test_latest_checkpoint_is_numeric_and_confined_to_one_run(tmp_path):
    checkpoint(tmp_path / "chosen/model_9.pt")
    checkpoint(tmp_path / "chosen/model_100.pt")
    checkpoint(tmp_path / "unrelated/model_9999.pt")
    assert colab.latest_checkpoint(tmp_path / "chosen").name == "model_100.pt"
    (tmp_path / "chosen/model_101.pt").write_bytes(b"partially written")
    with pytest.raises(ValueError, match="incomplete"):
        colab.latest_checkpoint(tmp_path / "chosen")


def test_persistence_publishes_checkpoint_and_manifest_together(tmp_path, monkeypatch):
    model = saved_run(tmp_path / "run")
    manifest = json.loads((model.parent / colab.MANIFEST).read_text())
    destination = tmp_path / "drive"
    published = colab.persist_checkpoint(model, manifest, destination)
    assert (published / model.name).read_bytes() == model.read_bytes()
    colab.validate_resume(published / model.name, config())

    def interrupted_copy(source, target):
        Path(target).write_bytes(b"partial")
        raise OSError("drive disconnected")

    monkeypatch.setattr(colab.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="disconnected"):
        colab.persist_checkpoint(model, manifest, destination)
    assert list(destination.iterdir()) == [published]


def test_backup_round_trip_preserves_checkpoint_export_and_existing_destination(tmp_path):
    model = saved_run(tmp_path / "run")
    export = tmp_path / "export"
    export.mkdir()
    (export / "actor.onnx").write_bytes(b"onnx")
    video = tmp_path / "policy.mp4"
    video.write_bytes(b"mp4")
    archive = colab.create_backup(model.parent, tmp_path / "backup.zip",
                                  export=export, video=video)
    restored = colab.restore_backup(archive, tmp_path / "restored")
    assert restored.read_bytes() == model.read_bytes()
    assert (tmp_path / "restored/export/actor.onnx").read_bytes() == b"onnx"
    assert (tmp_path / "restored/video/policy.mp4").read_bytes() == b"mp4"
    colab.validate_resume(restored, config())
    with pytest.raises(FileExistsError):
        colab.restore_backup(archive, tmp_path / "restored")
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        colab.create_backup(model.parent, archive)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("name", [
    "../outside", "/absolute", "C:/Windows/file", "run\\..\\outside",
    "run/../outside", "run/./model.pt", "run//model.pt",
])
def test_restore_refuses_traversal_before_extracting_any_file(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("an-otherwise-safe-file", "safe")
        stream.writestr(name, "escape")
    with pytest.raises(ValueError, match="unsafe"):
        colab.restore_backup(archive, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()
    assert not list(tmp_path.glob(".restore-*"))
    assert not (tmp_path / "outside").exists()


def test_restore_refuses_duplicate_symlink_and_oversized_entries(tmp_path):
    for kind in ("duplicate", "symlink", "oversized"):
        archive = tmp_path / f"{kind}.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            if kind == "duplicate":
                stream.writestr("same", "first")
                with pytest.warns(UserWarning, match="Duplicate"):
                    stream.writestr("same", "second")
            elif kind == "symlink":
                entry = zipfile.ZipInfo("link")
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                stream.writestr(entry, "../../outside")
            else:
                stream.writestr("large", "x" * 100)
        with pytest.raises(ValueError):
            colab.restore_backup(archive, tmp_path / kind, max_bytes=20)
        assert not (tmp_path / kind).exists()


def test_source_identity_tracks_runtime_edits_but_not_a_later_document_edit(tmp_path):
    """Documentation edits must not invalidate the same runtime's saved checkpoint."""
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    (tmp_path / "rl").mkdir()
    (tmp_path / "rl/runtime.py").write_text("original\n")
    (tmp_path / "README.md").write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                    "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    original = colab.source_identity(tmp_path)
    (tmp_path / "README.md").write_text("documentation edit\n")
    assert colab.source_identity(tmp_path) == original
    (tmp_path / "rl/runtime.py").write_text("runtime edit\n")
    assert colab.source_identity(tmp_path)["source_sha256"] != original["source_sha256"]
    (tmp_path / "rl/runtime.py").write_text("original\n")
    (tmp_path / ".env.local").write_text("MJRL_COMMAND_LEVEL=1\n")
    assert colab.source_identity(tmp_path)["source_sha256"] != original["source_sha256"]


def test_checkpoint_is_persisted_before_training_subprocess_finishes(tmp_path):
    """A final-only backup would pass a simple round trip, but fails this timing check."""
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "fake_train.py"
    release = repo / "release"
    script.write_text(textwrap.dedent('''
        import pathlib, time, zipfile
        run = pathlib.Path("logs/robot/robot.walk/new-run")
        run.mkdir(parents=True)
        print("[mjrl] log directory", run, flush=True)
        with zipfile.ZipFile(run / "model_0.pt", "w") as archive:
            archive.writestr("model/data.pkl", "checkpoint")
        deadline = time.monotonic() + 10
        while not pathlib.Path("release").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("checkpoint was not persisted during training")
            time.sleep(0.02)
        print("training complete", flush=True)
    '''))
    observed = []

    def acknowledge_snapshot():
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            snapshots = list((tmp_path / "drive").glob("*/model_0-*/model_0.pt"))
            if snapshots:
                observed.append(snapshots[0])
                release.touch()
                return
            time.sleep(0.02)

    observer = threading.Thread(target=acknowledge_snapshot, daemon=True)
    observer.start()
    result = colab.run_training([sys.executable, str(script)], repo=repo,
                                manifest_path=tmp_path / "operation.json", config=config(),
                                drive_root=tmp_path / "drive", poll_interval=0.03)
    observer.join(timeout=1)
    assert observed and result["status"] == "finished"
    colab.validate_resume(observed[0], config())
    assert json.loads((tmp_path / "operation.json").read_text())["status"] == "finished"


def test_failed_subprocess_keeps_a_loud_failure_manifest(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        colab.run_training([sys.executable, "-c", "raise SystemExit(7)"], repo=tmp_path,
                           manifest_path=tmp_path / "operation.json", config=config())
    assert failure.value.returncode == 7
    manifest = json.loads((tmp_path / "operation.json").read_text())
    assert manifest["status"] == "failed"
    assert "CalledProcessError" in manifest["error"]


def test_notebook_has_plain_python_cells_and_runs_repo_scripts_in_isolated_processes():
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    for index, source in enumerate(code_cells):
        ast.parse(source, filename=f"cell_{index}")
        assert not any(line.startswith(("!pip", "%pip", "%run"))
                       for line in source.splitlines())
    source = "\n".join(code_cells)
    assert 'system_site_packages=False' in source
    assert 'PYTHON, "scripts/play.py"' in source
    assert 'PYTHON, "scripts/export.py"' in source
    assert 'PYTHON, "-m", "mjrl.colab", "train"' in source
    assert '"--iterations", "5"' in source and '"--num-envs", "256"' in source
    assert '"--backend", "warp", "--device", "cuda:0"' in source
    assert '"--physics-hz", "200"' in source
    assert '"--steps", "500"' in source
    assert 'git", "reset' not in source
    assert "import torch" not in source
    assert "files.download(str(BACKUP_ZIP))" in source
