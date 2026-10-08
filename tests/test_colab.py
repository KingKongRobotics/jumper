"""Colab must preserve the selected run, not silently choose one that happens to exist.

These tests use valid ZIP-shaped fake checkpoints and real short-lived subprocesses.
They need no torch, MuJoCo, CUDA or Colab kernel; GPU acceptance is a separate gate.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

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
    assert colab.latest_checkpoint(tmp_path / "chosen").name == "model_100.pt"


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


def test_interrupted_backup_skips_partial_unrecorded_and_replaced_highest_saves(tmp_path):
    """An interrupted last save must not make the preceding complete save undownloadable."""
    model = saved_run(tmp_path / "run")
    partial = model.parent / "model_11.pt"
    partial.write_bytes(b"interrupted torch.save")
    checkpoint(model.parent / "model_12.pt", "complete but never recorded")
    replaced = model.parent / "model_13.pt"
    checkpoint(replaced, "original")
    manifest = json.loads((model.parent / colab.MANIFEST).read_text())
    manifest["checkpoints"][replaced.name] = colab.sha256(replaced)
    checkpoint(replaced, "replaced")
    colab.atomic_json(model.parent / colab.MANIFEST, manifest)
    assert colab.latest_checkpoint(model.parent) == model
    archive = colab.create_backup(model.parent, tmp_path / "interrupted.zip")
    with zipfile.ZipFile(archive) as content:
        weights = [name for name in content.namelist() if name.endswith(".pt")]
        assert weights == ["run/model_10.pt"]
    restored = colab.restore_backup(archive, tmp_path / "restored-interrupted")
    assert restored.name == "model_10.pt"
    colab.validate_resume(restored, config())


@pytest.mark.parametrize("drive_failure", [False, True])
def test_interruption_records_complete_save_not_yet_polled_and_preserves_error(
    tmp_path, monkeypatch, drive_failure
):
    """Interrupt immediately after a save, before the next watcher poll can see it."""
    script = tmp_path / "fake_train.py"
    script.write_text(textwrap.dedent('''
        import pathlib, time, zipfile
        run = pathlib.Path("logs/robot/robot.walk/interrupted-run")
        run.mkdir(parents=True)
        print("[mjrl] log directory", run, flush=True)
        with zipfile.ZipFile(run / "model_1.pt", "w") as archive:
            archive.writestr("model/data.pkl", "complete")
        (run / "model_2.pt").write_bytes(b"partial")
        pathlib.Path("saved").touch()
        while True:
            time.sleep(0.1)
    '''))
    real_queue = colab.queue.Queue

    class InterruptAfterAnnouncement(real_queue):
        reads = 0

        def get(self, *args, **kwargs):
            if self.reads:
                deadline = time.monotonic() + 5
                while not (tmp_path / "saved").exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError("fake subprocess did not finish saving")
                    time.sleep(0.01)
                raise KeyboardInterrupt("intentional notebook interruption")
            line = super().get(*args, **kwargs)
            self.reads += 1
            return line

    monkeypatch.setattr(colab.queue, "Queue", InterruptAfterAnnouncement)
    if drive_failure:
        def disconnected(*args, **kwargs):
            raise OSError("Drive disconnected during cleanup")

        monkeypatch.setattr(colab, "persist_checkpoint", disconnected)
    with pytest.raises(KeyboardInterrupt, match="intentional"):
        colab.run_training([sys.executable, str(script)], repo=tmp_path,
                           manifest_path=tmp_path / "operation.json", config=config(),
                           drive_root=tmp_path / "drive", poll_interval=60)
    manifest = json.loads((tmp_path / "operation.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["error"].startswith("KeyboardInterrupt:")
    assert set(manifest["checkpoints"]) == {"model_1.pt"}
    run = Path(manifest["run_directory"])
    colab.validate_resume(run / "model_1.pt", config())
    archive = colab.create_backup(run, tmp_path / "after-interruption.zip")
    assert colab.restore_backup(archive, tmp_path / "restored").name == "model_1.pt"
    if drive_failure:
        assert any("Drive disconnected" in error for error in manifest["cleanup_errors"])
    else:
        snapshots = list((tmp_path / "drive").glob("*/model_1-*/model_1.pt"))
        assert len(snapshots) == 1
        colab.validate_resume(snapshots[0], config())


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
    assert '"--iterations", "5"' in source and 'str(min(NUM_ENVS, 256))' in source
    assert '"--backend", "warp", "--device", "cuda:0"' in source
    assert '"--physics-hz", "200"' in source
    assert '"--steps", "500"' in source
    assert 'git", "reset' not in source
    assert "import torch" not in source
    assert "files.download(str(BACKUP_ZIP))" in source
    backup_cell = next(cell for cell in code_cells if "files.download(str(BACKUP_ZIP))" in cell)
    assert "from google.colab import files" in backup_cell


def notebook_cell(fragment):
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and fragment in "".join(cell["source"]))


@pytest.mark.parametrize("overrides", [
    {"CHECKOUT_FOLDER": "../another"},
    {"CHECKOUT_FOLDER": "a/b"},
    {"SIM_SECONDS": 0},
    {"SIM_SECONDS": float("inf")},
    {"SIM_FPS": -1},
    {"SIM_WIDTH": 641},
    {"SIM_HEIGHT": 0},
    {"SIM_CAMERA_DISTANCE": -1},
    {"SIM_SCENE": "invented"},
    {"CORE_VERSION_OVERRIDES": '{"untrusted-package":"1.0"}'},
    {"CORE_VERSION_OVERRIDES": '{"numpy":"2.5.2 --extra-index-url attacker"}'},
    {"CORE_VERSION_OVERRIDES": '{"torch":"2.8.0"}'},
])
def test_notebook_form_refuses_unsafe_or_invalid_choices_before_setup(overrides):
    """A valid form is a control; reject edited parameters before any GPU subprocess."""
    # Forms are separate; invalid values still have to stop before setup or replay.
    sources = [notebook_cell("WORKFLOW_MODE ="),
               notebook_cell("CORE_VERSION_OVERRIDES ="), notebook_cell("SIM_SECONDS =")]
    source = "\n".join(sources)
    namespace = {}
    exec(compile(source, "valid_colab_form", "exec"), namespace)  # noqa: S102
    assert namespace["NUM_ENVS"] == 256 and namespace["SIM_SCENE"] == "task default"
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in overrides:
                node.value = ast.Constant(overrides[target.id])
    with pytest.raises(ValueError):
        exec(compile(ast.fix_missing_locations(tree), "invalid_colab_form", "exec"), {})  # noqa: S102


@pytest.mark.parametrize("scene,distance,measure", [
    ("task default", 0.0, False), ("ice", 1.4, True),
])
def test_notebook_simulation_uses_control_time_and_records_the_selected_checkpoint(
    tmp_path, monkeypatch, scene, distance, measure
):
    """Changing recording FPS must not change control steps or the policy's physics rate."""
    session = tmp_path / "session"
    session.mkdir()
    run_dir = tmp_path / "selected-run"
    run_dir.mkdir()
    model = run_dir / "model_20.pt"
    model.touch()
    artifacts = session / "artifacts"
    artifacts.mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "rates" in command:
            Path(command[command.index("--out") + 1]).write_text(
                json.dumps({"physics_hz": 200.0, "control_hz": 50.0,
                            "supports_velocity_command": True})
            )
        else:
            Path(command[command.index("--video") + 1]).write_bytes(b"new MP4")
            Path(command[command.index("--replay-info-out") + 1]).write_text(
                json.dumps({"command_ranges": {"status": "matched", "curriculum_level": 0}})
            )
            if "--measure" in command:
                directory = model.parent / "measure"
                directory.mkdir()
                (directory / "measure.csv").write_text("t,pos/joint\n0.02,0.1\n")
                (directory / "measure.png").write_bytes(b"plot")

    namespace = {
        "SESSION": session, "ARTIFACTS": artifacts, "REPO": tmp_path,
        "PYTHON": tmp_path / ".venv/python", "TASK": "robot.walk", "MODEL": "robot",
        "CHECKPOINT_PATH": model, "RUN_DIR": run_dir,
        "training": {"operation": "selected", "config": {"task": "robot.walk", "model": "robot"}},
        "SIM_SECONDS": 1.01, "SIM_FPS": 12.0, "SIM_WIDTH": 800, "SIM_HEIGHT": 600,
        "SIM_SCENE": scene, "SIM_CAMERA_DISTANCE": distance, "RECORD_JOINTS": measure,
        "SIM_COMMAND": "Sampled commands",
        "run": run, "json": json, "math": __import__("math"), "shutil": __import__("shutil"),
        "uuid": __import__("uuid"),
        "display": lambda value: None, "Video": lambda *args, **kwargs: None,
        "Image": lambda **kwargs: None,
    }
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "IPython.display", SimpleNamespace(
        display=namespace["display"], Video=namespace["Video"], Image=namespace["Image"],
    ))
    exec(compile(notebook_cell("SIM_STEPS ="), "record_simulation", "exec"), namespace)  # noqa: S102
    command = commands[-1]
    assert command[command.index("--steps") + 1] == "51"
    assert command[command.index("--physics-hz") + 1] == "200.0"
    assert command[command.index("--video-fps") + 1] == "12.0"
    assert command[command.index("--video-width") + 1] == "800"
    assert "--checkpoint-command-ranges" in command
    assert "--measure" in command if measure else "--measure" not in command
    assert "--scene" not in command if scene == "task default" else command[
        command.index("--scene") + 1
    ] == scene
    assert "--video-distance" not in command if distance == 0 else command[
        command.index("--video-distance") + 1
    ] == str(distance)
    saved = json.loads((artifacts / "simulation-settings.json").read_text())
    assert saved["checkpoint"] == str(model) and saved["control_steps"] == 51
    assert saved["simulated_seconds"] == 1.02 and saved["observation_noise"] is True
    assert namespace["SIMULATION_CHECKPOINT"] == str(model)
    if measure:
        assert namespace["MEASUREMENT_CHECKPOINT"] == str(model)
        assert (artifacts / "measure.csv").is_file()
        first_video = namespace["TRAIN_VIDEO"]
        namespace.update(RECORD_JOINTS=False, SIM_SCENE="rubber")
        exec(compile(notebook_cell("SIM_STEPS ="), "repeat_simulation", "exec"), namespace)  # noqa: S102
        # A new unmeasured scene must not inherit the earlier ice scene's measurements.
        assert namespace["TRAIN_VIDEO"] != first_video and first_video.is_file()
        assert not (artifacts / "measure.csv").exists()
        assert not (artifacts / "measure.png").exists()
        assert not (model.parent / "measure").exists()
        assert namespace["MEASUREMENT_CHECKPOINT"] == ""
        previous = namespace["REPLAY_HISTORY"]
        assert (previous / "checkpoint-measure/measure.csv").is_file()
        assert (previous / "measure.csv").is_file()
        assert json.loads((previous / "simulation-settings.json").read_text())["scene"] == "ice"
        current = json.loads((artifacts / "simulation-settings.json").read_text())
        assert current["scene"] == "rubber" and current["record_joints"] is False


@pytest.mark.parametrize("changed", [{"TASK": "robot.other-gait"}, {"MODEL": "other-robot"}])
def test_notebook_replay_rejects_task_or_model_changes_before_archiving_or_gpu(changed):
    """Matching tensor dimensions cannot establish a checkpoint's gait or robot semantics."""
    namespace = {"TASK": "robot.walk", "MODEL": "robot", "training": {
        "config": {"task": "robot.walk", "model": "robot"},
    }, **changed}
    with pytest.raises(ValueError, match="Change only SIM_"):
        # No Path, shutil or GPU runner is supplied: the task/model gate must come first.
        exec(compile(notebook_cell("SIM_STEPS ="), "refuse_wrong_task", "exec"), namespace)  # noqa: S102


@pytest.mark.parametrize("changed", [{}, {"TASK": "robot.other-gait"}, {"MODEL": "other-robot"}])
def test_notebook_export_keeps_the_recorded_task_and_model(tmp_path, changed):
    """Equal tensor dimensions must not let a checkpoint acquire another task's contract."""
    session = tmp_path / "session"
    session.mkdir()
    artifacts = session / "artifacts"
    artifacts.mkdir()
    run_dir = tmp_path / "selected-run"
    run_dir.mkdir()
    model = run_dir / "model_20.pt"
    model.touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--out") + 1])
        output.mkdir()
        (output / "actor.onnx").write_bytes(b"exported actor")
        (output / "layout.json").write_text("{}")

    namespace = {
        "SESSION": session, "ARTIFACTS": artifacts, "REPO": tmp_path,
        "PYTHON": tmp_path / ".venv/python", "TASK": "robot.walk", "MODEL": "robot",
        "CHECKPOINT_PATH": model, "RUN_DIR": run_dir,
        "training": {"config": {"task": "robot.walk", "model": "robot"}},
        "run": run, "json": json, "uuid": __import__("uuid"),
        "EXPORT_SUCCESS": True, "EXPORTED_CHECKPOINT": "earlier-checkpoint",
        **changed,
    }
    source = compile(notebook_cell("EXPORT_DIR = SESSION"), "export_selected_policy", "exec")
    if changed:
        with pytest.raises(ValueError, match="different task/model"):
            exec(source, namespace)  # noqa: S102
        assert commands == [] and "EXPORT_DIR" not in namespace
        assert not namespace["EXPORT_SUCCESS"] and namespace["EXPORTED_CHECKPOINT"] == ""
        assert list(artifacts.iterdir()) == []
    else:
        # The matching control must reach export and associate the contract with this checkpoint.
        exec(source, namespace)  # noqa: S102
        assert len(commands) == 1
        command = commands[0]
        assert command[command.index("--task") + 1] == "robot.walk"
        assert command[command.index("--model") + 1] == "robot"
        assert command[command.index("--checkpoint") + 1] == model
        assert namespace["EXPORT_SUCCESS"] and namespace["EXPORTED_CHECKPOINT"] == str(model)
        saved = json.loads((artifacts / "export-settings.json").read_text())
        assert saved["checkpoint"] == str(model) and saved["run"] == str(run_dir)


@pytest.mark.parametrize("matching", [False, True])
def test_notebook_backup_refuses_stale_video_and_export_without_blocking_checkpoints(
    tmp_path, monkeypatch, matching
):
    """Old preview/export globals must not contaminate a newer training run's portable ZIP."""
    session = tmp_path / "session"
    session.mkdir()
    for name in ("runtime.json", "graphics.json"):
        (session / name).write_text("{}")
    run_dir = tmp_path / "new-run"
    run_dir.mkdir()
    model = run_dir / "model_20.pt"
    model.touch()
    manifest = session / "training.json"
    manifest.write_text(json.dumps({
        "run_directory": str(run_dir), "operation": "new", "checkpoint": str(model),
        "checkpoints": {model.name: "recorded"},
    }))
    video = session / "old-video.mp4"
    video.write_bytes(b"MP4")
    commands, downloads = [], []
    files = SimpleNamespace(download=lambda name: downloads.append(name))
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(files=files))

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--out") + 1]).write_bytes(b"ZIP")

    chosen = str(model) if matching else str(tmp_path / "old-run/model_20.pt")
    namespace = {
        "SESSION": session, "REPO": tmp_path, "PYTHON": tmp_path / ".venv/python",
        "TRAIN_MANIFEST": manifest, "TASK": "robot.walk", "DOWNLOAD_FILES": False,
        "EXPORT_SUCCESS": True, "EXPORTED_CHECKPOINT": chosen,
        "SIMULATION_CHECKPOINT": chosen, "EXPORT_DIR": session / "old-export",
        "TRAIN_VIDEO": video, "json": json, "Path": Path, "uuid": __import__("uuid"),
        "run": run,
    }
    exec(compile(notebook_cell("BACKUP_CHECKPOINT ="), "package_selected_run", "exec"), namespace)  # noqa: S102
    command = commands[0]
    assert "--export" in command if matching else "--export" not in command
    assert "--video" in command if matching else "--video" not in command
    assert command[command.index("--run") + 1] == run_dir
    assert command[command.index("--extras") + 1] == session / "artifacts/new"
    assert downloads == []


@pytest.mark.parametrize("existing_interpreter", [False, True])
def test_notebook_bootstraps_only_target_venv_when_ensurepip_is_unavailable(
    tmp_path, existing_interpreter
):
    """Model Colab's missing ensurepip and an earlier half-created pip-less environment."""
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "PYTHON = REPO" in "".join(cell["source"]))
    bootstrap = source[source.index("PYTHON = REPO"):source.index("SESSION =")]
    python = tmp_path / ".venv/bin/python"
    if existing_interpreter:
        python.parent.mkdir(parents=True)
        python.touch()
    builds, commands = [], []

    class PiplessEnvBuilder:
        def __init__(self, *, with_pip, system_site_packages):
            # This is the failure the previous with_pip=True notebook encountered.
            if with_pip:
                raise RuntimeError("ensurepip is unavailable in this Colab runtime")
            if system_site_packages:
                raise RuntimeError("kernel site packages must remain isolated")

        def create(self, directory):
            builds.append(directory)
            python.parent.mkdir(parents=True)
            python.touch()

    namespace = {
        "REPO": tmp_path, "sys": sys,
        "venv": SimpleNamespace(EnvBuilder=PiplessEnvBuilder),
        "subprocess": SimpleNamespace(check_output=lambda *a, **k: "pip 25.2 from kernel"),
        "run": lambda command: commands.append(command),
    }
    exec(compile(bootstrap, "notebook_venv_bootstrap", "exec"), namespace)  # noqa: S102
    assert bool(builds) is not existing_interpreter
    assert commands[-1] == [
        sys.executable, "-m", "pip", "--python", python, "install", "--upgrade", "pip",
    ]
    assert all("--python" in command for command in commands if "install" in command)


def test_notebook_reemits_subprocess_logs_and_keeps_a_bounded_failure_tail(capsys):
    """Colab's RPC captures print outputs but misses inherited child file descriptors."""
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def run(command" in "".join(cell["source"]))
    function = source[source.index("def run(command"):source.index("# Fail before downloads")]
    namespace = {"subprocess": subprocess, "deque": deque, "signal": signal}
    exec(compile(function, "notebook_process_run", "exec"), namespace)  # noqa: S102
    result = namespace["run"]([sys.executable, "-c", "print('visible child output')"])
    assert isinstance(result, subprocess.CompletedProcess) and result.returncode == 0
    assert "visible child output" in capsys.readouterr().out
    with pytest.raises(subprocess.CalledProcessError) as failure:
        namespace["run"]([
            sys.executable, "-c",
            ("import sys; [print('line', i) for i in range(70)]; "
             "print('visible stderr', file=sys.stderr); raise SystemExit(7)"),
        ])
    output = capsys.readouterr().out
    assert "visible stderr" in output and "line 0" in output
    assert failure.value.returncode == 7
    assert "line 69" in failure.value.output
    assert "line 0\n" not in failure.value.output


@pytest.mark.parametrize("nvidia_available", [False, True])
def test_notebook_egl_override_is_session_local_and_never_selects_mesa(
    tmp_path, nvidia_available
):
    """A GPU driver can exist while GLVND's only registered vendor remains Mesa."""
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "def configure_nvidia_egl(" in "".join(cell["source"]))
    function = source[source.index("def configure_nvidia_egl("):source.index("NVIDIA_EGL =")]
    session = tmp_path / "session"
    session.mkdir()
    vendor_directory = tmp_path / "system/egl_vendor.d"
    vendor_directory.mkdir(parents=True)
    mesa_config = vendor_directory / "50_mesa.json"
    mesa_config.write_text('{"ICD":{"library_path":"libEGL_mesa.so.0"}}')
    mesa_library = tmp_path / "libEGL_mesa.so.0"
    mesa_library.write_text("existing Mesa library")
    nvidia_library = tmp_path / "driver/libEGL_nvidia.so.0"
    if nvidia_available:
        nvidia_library.parent.mkdir()
        nvidia_library.write_text("existing NVIDIA library")
    candidates = [tmp_path / "missing/libEGL_nvidia.so.0", nvidia_library, mesa_library]
    environment = {"LD_LIBRARY_PATH": "/existing/path",
                   "__EGL_VENDOR_LIBRARY_FILENAMES": str(mesa_config)}
    original_environment = dict(environment)
    original_mesa = mesa_config.read_bytes()
    namespace = {"Path": Path, "json": json, "os": SimpleNamespace(environ=environment)}
    exec(compile(function, "notebook_egl_configuration", "exec"), namespace)  # noqa: S102
    if not nvidia_available:
        with pytest.raises(RuntimeError, match="No preinstalled NVIDIA EGL"):
            namespace["configure_nvidia_egl"](session, candidates)
        assert environment == original_environment
        assert not list(session.iterdir())
    else:
        selected = namespace["configure_nvidia_egl"](session, candidates)
        assert selected == nvidia_library.resolve()
        override = session / "nvidia-egl.json"
        assert json.loads(override.read_text()) == {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": str(nvidia_library.resolve())},
        }
        assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(override.resolve())
        assert environment["LD_LIBRARY_PATH"] == f"{selected.parent}:/existing/path"
        assert environment["MUJOCO_GL"] == environment["PYOPENGL_PLATFORM"] == "egl"
        assert list(session.iterdir()) == [override]
    assert mesa_config.read_bytes() == original_mesa
    assert list(vendor_directory.iterdir()) == [mesa_config]


@pytest.mark.parametrize("actual_vendor", ["NVIDIA Corporation", "Mesa/X.org"])
def test_notebook_egl_probe_checks_actual_vendor_before_accepting_pixels(
    tmp_path, monkeypatch, actual_vendor
):
    """Nonempty pixels alone also pass with llvmpipe; the actual vendor is the gate."""
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "graphics_probe = " in "".join(cell["source"]))
    tree = ast.parse(source)
    probe = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "graphics_probe"
                         for target in node.targets))
    result = tmp_path / "graphics.json"

    class Pixels:
        shape = (64, 64, 3)

        def __ne__(self, other):
            return self

    class Renderer:
        closed = False

        def update_scene(self, data):
            pass

        def render(self):
            return Pixels()

        def close(self):
            self.closed = True

    renderer = Renderer()
    fake_mujoco = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_string=lambda xml: object()),
        MjData=lambda model: object(), mj_forward=lambda model, data: None,
        Renderer=lambda model, **kwargs: renderer,
    )
    fake_gl = SimpleNamespace(
        GL_VENDOR=1, GL_RENDERER=2,
        glGetString=lambda kind: (actual_vendor if kind == 1 else "example renderer").encode(),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        any=lambda pixels, **kwargs: pixels, count_nonzero=lambda pixels: 4096,
    ))
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=fake_gl))
    monkeypatch.setattr(sys, "argv", ["isolated_egl_probe", str(result)])
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", str(tmp_path / "nvidia-egl.json"))
    if actual_vendor.startswith("NVIDIA"):
        exec(compile(probe, "isolated_egl_probe", "exec"), {})  # noqa: S102
        graphics = json.loads(result.read_text())
        assert graphics["gl_vendor"] == actual_vendor
        assert graphics["nonzero_pixels"] == 4096 and graphics["shape"] == [64, 64, 3]
    else:
        with pytest.raises(RuntimeError, match="actual GL vendor=.*Mesa"):
            exec(compile(probe, "isolated_egl_probe", "exec"), {})  # noqa: S102
        assert not result.exists()
    assert renderer.closed
