"""Notebook reruns must preserve an explicit recovery selection without GPU execution.

These checks execute the published cells, rather than copied workflow logic. CPU-only
ZIP-shaped checkpoints exercise the real portable backup/restore implementation;
Colab UI and GPU subprocesses are replaced at their external boundaries.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import sys
import uuid
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebook_workflow_colab", REPO / "rl/mjrl/colab.py")
assert SPEC is not None and SPEC.loader is not None
colab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab)


def cell(fragment):
    notebook = json.loads((REPO / "notebooks/jumper_colab.ipynb").read_text(encoding="utf-8"))
    return next("".join(item["source"]) for item in notebook["cells"]
                if item["cell_type"] == "code" and fragment in "".join(item["source"]))


def execute(fragment, namespace, **parameters):
    tree = ast.parse(cell(fragment))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in parameters:
                node.value = ast.Constant(parameters[target.id])
    code = compile(ast.fix_missing_locations(tree), "published_colab_cell", "exec")
    exec(code, namespace)  # noqa: S102 -- execute the actual notebook under test


def settings():
    namespace = {}
    execute("# @title Run settings", namespace)
    execute("# @title Advanced source", namespace)
    execute("# @title Continue training: backup selection", namespace)
    return namespace


def recorded_config():
    return {
        "task": "jumper.ripple", "model": "jumper", "num_envs": 64,
        "revision": "a" * 40, "source_sha256": "b" * 64,
        "backend": "warp", "device": "cuda:0", "mjrl_environment": {},
        "repository_url": "https://github.com/example/jumper.git",
        "runtime_versions": {
            "torch": "2.9.1+cu126", "mujoco": "3.11.0", "mujoco-warp": "3.11.0",
            "warp-lang": "1.18.0", "numpy": "2.5.2", "tensordict": "0.14.2",
        },
    }


def saved_run(directory):
    directory.mkdir(parents=True)
    checkpoint = directory / "model_499.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("model/data.pkl", "saved optimizer and weights fixture")
    manifest = {
        "schema": 1, "config": recorded_config(), "operation": "recorded-operation",
        "status": "interrupted", "run_directory": str(directory),
        "checkpoint": str(checkpoint), "checkpoints": {checkpoint.name: colab.sha256(checkpoint)},
    }
    colab.atomic_json(directory / colab.MANIFEST, manifest)
    return checkpoint, manifest


@pytest.fixture
def external_ui(monkeypatch):
    downloads, displays = [], []
    google = ModuleType("google")
    google_colab = ModuleType("google.colab")
    google_colab.files = SimpleNamespace(
        download=downloads.append,
        upload=lambda: pytest.fail("A cached selection must not request another upload"),
    )
    google.colab = google_colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", google_colab)
    ipython = ModuleType("IPython")
    display_module = ModuleType("IPython.display")
    display_module.Image = lambda *args, **kwargs: ("image", args, kwargs)
    display_module.Video = lambda *args, **kwargs: ("video", args, kwargs)
    display_module.display = displays.append
    ipython.display = display_module
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    return SimpleNamespace(downloads=downloads, displays=displays)


def test_basic_and_advanced_reruns_preserve_restored_checkpoint_and_configuration(tmp_path):
    """The old settings assignment silently erased resume and started training from scratch."""
    namespace = settings()
    checkpoint, manifest = saved_run(tmp_path / "selected")
    state = namespace["RUN_STATE"]
    state.update(restored_checkpoint=str(checkpoint), backup_config=manifest["config"])
    execute("# @title Run settings", namespace)
    execute("# @title Advanced source", namespace)
    assert namespace["RUN_STATE"] is state
    assert state["restored_checkpoint"] == str(checkpoint)
    assert state["backup_config"] == manifest["config"]
    execute("# @title Run settings", namespace, WORKFLOW_MODE="Continue training", ITERATIONS=12)
    assert namespace["TASK"] == "jumper.ripple" and namespace["NUM_ENVS"] == 64
    assert namespace["ITERATIONS"] == 12 and state["restored_checkpoint"] == str(checkpoint)


def test_new_mode_with_a_recovered_run_fails_before_any_training_subprocess(tmp_path):
    namespace = settings()
    checkpoint, manifest = saved_run(tmp_path / "selected")
    namespace["RUN_STATE"].update(backup_config=manifest["config"],
                                  restored_checkpoint=str(checkpoint))
    namespace.update(SESSION=tmp_path, PYTHON=tmp_path / "python", REPO=tmp_path,
                     PERSIST_ROOT=None, run=lambda *args, **kwargs: pytest.fail("Training started"))
    with pytest.raises(RuntimeError, match="Recovered run is preserved"):
        execute('TRAIN_MANIFEST = SESSION / "training.json"', namespace)
    assert checkpoint.is_file()


def test_legitimate_new_training_selects_its_new_run_and_retains_gate_reports(tmp_path):
    """Control for the recovered-run gate: it must not prohibit a legitimate fresh run."""
    namespace = settings()
    session = tmp_path / "session"
    session.mkdir()
    for name in ("runtime.json", "graphics.json"):
        (session / name).write_text(json.dumps({"report": name}))
    checkpoint, manifest = saved_run(tmp_path / "newly-trained-run")
    manifest["status"] = "finished"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        colab.atomic_json(Path(command[command.index("--manifest") + 1]), manifest)

    namespace.update(SESSION=session, PYTHON=tmp_path / "python", REPO=tmp_path,
                     PERSIST_ROOT=None, TASK="jumper.ripple", NUM_ENVS=64, ITERATIONS=5, run=run)
    execute('TRAIN_MANIFEST = SESSION / "training.json"', namespace)
    command = commands[0]
    assert command[command.index("--task") + 1] == "jumper.ripple"
    assert command[command.index("--num-envs") + 1] == "64" and "--resume" not in command
    assert namespace["CHECKPOINT_PATH"] == checkpoint
    assert namespace["RUN_STATE"]["training_manifest"] == str(session / "training.json")
    for name in ("runtime.json", "graphics.json"):
        assert (namespace["ARTIFACTS"] / name).read_bytes() == (session / name).read_bytes()


def test_explicit_new_run_clears_selection_but_preserves_existing_files(tmp_path):
    namespace = settings()
    checkpoint, manifest = saved_run(tmp_path / "old-run")
    namespace["RUN_STATE"].update(backup_config=manifest["config"],
                                  restored_checkpoint=str(checkpoint), backup_archive="old.zip",
                                  previous_backup_selection={"backup_archive": "earlier.zip"})
    execute("# @title Advanced source", namespace, START_NEW_RUN=True)
    assert "backup_config" not in namespace["RUN_STATE"]
    assert "restored_checkpoint" not in namespace["RUN_STATE"]
    assert "previous_backup_selection" not in namespace["RUN_STATE"]
    assert checkpoint.is_file() and (checkpoint.parent / colab.MANIFEST).is_file()


def test_setup_reruns_reuse_the_saved_session_directory(tmp_path):
    namespace = settings()
    namespace["RUN_STATE"]["session"] = str(tmp_path / "selected-session")
    namespace["uuid"] = uuid
    tree = ast.parse(cell("SESSION = Path(RUN_STATE.setdefault"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "SESSION"
                              for target in node.targets))
    # Execute the published state assignment without the preceding GPU/bootstrap commands.
    program = compile(ast.Module(body=[assignment], type_ignores=[]), "session_state", "exec")
    exec(program, namespace)  # noqa: S102
    first = namespace["SESSION"]
    exec(program, namespace)  # noqa: S102
    assert namespace["SESSION"] == first == tmp_path / "selected-session"


@pytest.mark.parametrize("environments,expected", [(64, 64), (1024, 256)])
def test_smoke_checks_the_selected_task_and_bounded_environment_count(
    tmp_path, environments, expected
):
    namespace = settings()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--manifest") + 1]).write_text(json.dumps({
            "status": "finished", "run_directory": str(tmp_path / "smoke-run"),
        }))

    namespace.update(SESSION=tmp_path, REPO=tmp_path, PYTHON=tmp_path / "python",
                     TASK="jumper.ripple", NUM_ENVS=environments, run=run)
    execute('SMOKE_MANIFEST = SESSION / "smoke.json"', namespace)
    command = commands[0]
    assert command[command.index("--task") + 1] == "jumper.ripple"
    assert command[command.index("--num-envs") + 1] == str(expected)
    assert command[command.index("--iterations") + 1] == "5"
    assert "--resume" not in command


def test_resume_selection_uses_all_recorded_versions_and_exact_source_before_install(capsys):
    namespace = settings()
    namespace["RUN_STATE"]["backup_config"] = recorded_config()
    namespace["WORKFLOW_MODE"] = "Continue training"
    execute("# @title Continue training: backup selection", namespace)
    assert namespace["SOURCE_REVISION"] == "a" * 40
    assert namespace["CORE_VERSIONS"] == recorded_config()["runtime_versions"]
    assert namespace["REPOSITORY_URL"] == recorded_config()["repository_url"]
    assert namespace["NUM_ENVS"] == 64
    assert "Continue:" in capsys.readouterr().out


def test_older_backup_without_repository_keeps_operator_selection_and_explains_it(capsys):
    namespace = settings()
    cfg = recorded_config()
    del cfg["repository_url"]
    namespace["RUN_STATE"]["backup_config"] = cfg
    namespace.update(WORKFLOW_MODE="Continue training",
                     REPOSITORY_URL="https://github.com/operator/chosen-fork.git")
    execute("# @title Continue training: backup selection", namespace)
    assert namespace["REPOSITORY_URL"] == "https://github.com/operator/chosen-fork.git"
    assert namespace["SOURCE_REVISION"] == cfg["revision"]
    assert "Older backup: use Advanced source" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [
    {"revision": "main"}, {"task": "../tasks"}, {"num_envs": True},
    {"backend": "native"}, {"device": "cpu"},
    {"repository_url": "https://github.com/user/repo.git;echo unwanted"},
    {"runtime_versions": {"torch": "2.9.1+cu126"}},
    {"runtime_versions": {**recorded_config()["runtime_versions"], "numpy": "https://bad/wheel"}},
])
def test_backup_metadata_refuses_incomplete_or_executable_setup_values(bad):
    namespace = settings()
    valid = {"schema": 1, "config": recorded_config(), "checkpoints": {"model_499.pt": "hash"}}
    assert namespace["validate_backup_metadata"](valid) == valid["config"]
    invalid = {**valid, "config": {**valid["config"], **bad}}
    with pytest.raises(ValueError):
        namespace["validate_backup_metadata"](invalid)


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", "a\\b", "a/../b", "./a", "C:/a"])
def test_early_backup_reader_rejects_unsafe_paths_without_extracting(tmp_path, unsafe, monkeypatch):
    namespace = settings()
    if "\\" in unsafe:
        class WireNames(zipfile.ZipInfo):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Windows ZipInfo normalizes backslashes; Colab reads the raw Linux wire name.
                self.filename = self.orig_filename

        monkeypatch.setattr(zipfile, "ZipInfo", WireNames)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("run/colab-run.json", json.dumps({"schema": 1}))
        output.writestr(unsafe, "must never be extracted")
    with pytest.raises(ValueError, match="Unsafe"):
        namespace["read_backup_metadata"](archive)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["unsafe.zip"]


def test_early_backup_reader_refuses_symlink_and_oversized_manifest(tmp_path):
    namespace = settings()
    for name, link in (("symlink", True), ("large", False)):
        archive = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive, "w") as output:
            if link:
                member = zipfile.ZipInfo("run/link")
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(member, "outside")
                output.writestr("run/colab-run.json", "{}")
            else:
                output.writestr("run/colab-run.json", " " * (1024**2 + 1))
        with pytest.raises(ValueError):
            namespace["read_backup_metadata"](archive)


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None])
def test_early_manifest_requires_integer_schema_without_boolean_coercion(schema):
    namespace = settings()
    valid = {"schema": 1, "config": recorded_config(), "checkpoints": {"model_499.pt": "hash"}}
    assert namespace["validate_backup_metadata"](valid) == valid["config"]
    with pytest.raises(ValueError, match="Unsupported backup manifest"):
        namespace["validate_backup_metadata"]({**valid, "schema": schema})


def test_early_reader_rejects_a_nul_truncated_zip_name(tmp_path):
    namespace = settings()
    archive = tmp_path / "nul-name.zip"
    original = b"run/dataXsuffix"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("run/colab-run.json", json.dumps({"schema": 1}))
        output.writestr(original.decode(), "payload")
    raw = archive.read_bytes()
    assert raw.count(original) == 2  # Local header and central directory are both altered.
    archive.write_bytes(raw.replace(original, b"run/data\0suffix"))
    with pytest.raises(ValueError, match="Truncated backup member name"):
        namespace["read_backup_metadata"](archive)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["nul-name.zip"]


@pytest.mark.parametrize("environment", [
    {"HOME": "/not-allowed"}, {"MJRL_lowercase": "1"}, {"MJRL_": "1"},
    {"MJRL_TASK": 1}, {"MJRL_TASK": "x" * 4097},
    {"MJRL_TASK": "invalid\0environment"},
    {f"MJRL_KEY_{index}": "1" for index in range(101)}, [],
])
def test_manifest_environment_accepts_only_bounded_mjrl_strings(environment):
    namespace = settings()
    valid = {"schema": 1, "config": recorded_config(), "checkpoints": {"model_499.pt": "hash"}}
    valid["config"]["mjrl_environment"] = {"MJRL_COMMAND_LEVEL": "2", "MJRL_TASK": "jumper.ripple"}
    assert namespace["validate_backup_metadata"](valid) == valid["config"]
    invalid = {**valid, "config": {**valid["config"], "mjrl_environment": environment}}
    with pytest.raises(ValueError, match="Invalid recorded MJRL environment"):
        namespace["validate_backup_metadata"](invalid)


def test_continuation_bootstrap_restores_only_mjrl_environment(monkeypatch):
    namespace = settings()
    cfg = recorded_config()
    cfg["mjrl_environment"] = {"MJRL_COMMAND_LEVEL": "2", "MJRL_TASK": "jumper.ripple"}
    namespace["RUN_STATE"]["backup_config"] = cfg
    namespace["WORKFLOW_MODE"] = "Continue training"
    environment = {
        "MJRL_COMMAND_LEVEL": "9", "MJRL_OLD_OVERRIDE": "old", "HOME": "preserved-home",
        "CUDA_VISIBLE_DEVICES": "0", "__EGL_VENDOR_LIBRARY_FILENAMES": "preserved-egl",
    }
    monkeypatch.setattr(os, "environ", environment)
    first_statement = ast.parse(cell("SESSION = Path(RUN_STATE.setdefault")).body[0]
    # Execute the real continuation branch before its GPU and clone subprocesses.
    program = compile(ast.Module(body=[first_statement], type_ignores=[]), "restore_environment", "exec")
    exec(program, namespace)  # noqa: S102
    assert {key: value for key, value in environment.items() if key.startswith("MJRL_")} == cfg[
        "mjrl_environment"
    ]
    assert environment["HOME"] == "preserved-home"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == "preserved-egl"


@pytest.mark.parametrize("bad_selection", [
    "boolean-schema", "foreign-environment", "nul-environment", "truncated-name",
])
def test_invalid_replacement_upload_keeps_the_existing_verified_selection(
    tmp_path, external_ui, monkeypatch, bad_selection
):
    checkpoint, manifest = saved_run(tmp_path / "previously-selected")
    namespace = settings()
    state = namespace["RUN_STATE"]
    state.update(backup_config=manifest["config"], backup_manifest=manifest,
                 backup_archive="previous.zip", restored_checkpoint=str(checkpoint))
    preserved = dict(state)
    invalid = {**manifest, "config": dict(manifest["config"])}
    if bad_selection == "boolean-schema":
        invalid["schema"] = True
    elif bad_selection == "foreign-environment":
        invalid["config"]["mjrl_environment"] = {"HOME": "not-a-training-setting"}
    elif bad_selection == "nul-environment":
        invalid["config"]["mjrl_environment"] = {"MJRL_TASK": "invalid\0environment"}
    archive = tmp_path / "replacement.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("run/colab-run.json", json.dumps(invalid))
        if bad_selection == "truncated-name":
            output.writestr("run/dataXsuffix", "payload")
    payload = archive.read_bytes()
    if bad_selection == "truncated-name":
        payload = payload.replace(b"run/dataXsuffix", b"run/data\0suffix")
    monkeypatch.setattr(sys.modules["google.colab"].files, "upload", lambda: {"replacement.zip": payload})
    namespace["WORKFLOW_MODE"] = "Continue training"
    namespace["Path"] = lambda value: tmp_path / "upload-inbox" if str(value) == (
        "/content/jumper-backups"
    ) else Path(value)
    with pytest.raises(ValueError):
        execute("# @title Continue training: backup selection", namespace,
                REPLACE_BACKUP_SELECTION=True)
    assert state == preserved and checkpoint.is_file()


@pytest.mark.parametrize("pending", [False, True])
def test_drive_selection_requires_a_completed_snapshot_and_preserves_failed_replacement(
    tmp_path, external_ui, monkeypatch, pending
):
    checkpoint, manifest = saved_run(tmp_path / "previous")
    drive_root = tmp_path / "drive/MyDrive"
    folder = ".pending-upload" if pending else "model_499-published"
    candidate, _ = saved_run(drive_root / "operation" / folder)
    namespace = settings()
    state = namespace["RUN_STATE"]
    state.update(backup_config=manifest["config"], backup_manifest=manifest,
                 backup_archive="previous.zip", restored_checkpoint=str(checkpoint))
    preserved = dict(state)
    namespace["WORKFLOW_MODE"] = "Continue training"
    namespace["Path"] = lambda value: drive_root if str(value) == "/content/drive/MyDrive" else Path(value)
    mounted = []
    monkeypatch.setattr(sys.modules["google.colab"], "drive",
                        SimpleNamespace(mount=mounted.append), raising=False)
    parameters = {"BACKUP_SOURCE": "Drive checkpoint", "DRIVE_CHECKPOINT": str(candidate),
                  "REPLACE_BACKUP_SELECTION": True}
    if pending:
        with pytest.raises(ValueError, match="completed Drive snapshot"):
            execute("# @title Continue training: backup selection", namespace, **parameters)
        assert state == preserved
    else:
        execute("# @title Continue training: backup selection", namespace, **parameters)
        assert state["restored_checkpoint"] == str(candidate.resolve())
        assert state["backup_archive"] is None
    assert mounted == ["/content/drive"] and checkpoint.is_file()


def test_restore_cell_reuses_the_verified_selection_without_upload_or_reextract(
    tmp_path, external_ui
):
    checkpoint, manifest = saved_run(tmp_path / "original")
    archive = colab.create_backup(checkpoint.parent, tmp_path / "saved.zip")
    namespace = settings()
    namespace["RUN_STATE"].update(backup_config=manifest["config"], backup_archive=str(archive))
    namespace.update(WORKFLOW_MODE="Continue training", SESSION=tmp_path / "session",
                     PYTHON=tmp_path / "python", REPO=tmp_path)
    namespace["SESSION"].mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "restore" in command:
            restored = colab.restore_backup(
                Path(command[command.index("--archive") + 1]),
                Path(command[command.index("--destination") + 1]),
            )
            colab.atomic_json(Path(command[command.index("--out") + 1]),
                              {"checkpoint": str(restored)})

    namespace["run"] = run
    execute("# @title Verify the selected continuation checkpoint", namespace)
    first = namespace["RESTORED_CHECKPOINT"]
    execute("# @title Verify the selected continuation checkpoint", namespace)
    assert sum("restore" in command for command in commands) == 1
    assert namespace["RESTORED_CHECKPOINT"] == first
    assert Path(first).read_bytes() == checkpoint.read_bytes()
    assert namespace["RUN_STATE"]["restored_checkpoint"] == first


def replacement_selection(tmp_path, monkeypatch, corruption=None):
    """Select a real uploaded candidate while retaining a usable earlier backup."""
    old_checkpoint, old_manifest = saved_run(tmp_path / "old-run")
    old_archive = colab.create_backup(old_checkpoint.parent, tmp_path / "old.zip")
    candidate_checkpoint, candidate_manifest = saved_run(tmp_path / "candidate-run")
    candidate_manifest["config"]["revision"] = "c" * 40
    candidate_manifest["config"]["task"] = "jumper.tripod"
    colab.atomic_json(candidate_checkpoint.parent / colab.MANIFEST, candidate_manifest)
    candidate_archive = tmp_path / "candidate.zip"
    if corruption:
        checkpoint_bytes = candidate_checkpoint.read_bytes()
        if corruption == "checkpoint":
            # Outer ZIP CRC is valid; the recorded checkpoint's contents are invalid.
            checkpoint_bytes = b"incomplete checkpoint, with a valid outer member CRC"
        with zipfile.ZipFile(candidate_archive, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("backup.json", json.dumps({"schema": 1, "run": "run"}))
            archive.writestr("run/" + colab.MANIFEST, json.dumps(candidate_manifest))
            archive.writestr("run/" + candidate_checkpoint.name, checkpoint_bytes)
        if corruption == "crc":
            # Corrupt only the stored policy bytes; manifest reading still succeeds.
            payload = candidate_archive.read_bytes()
            marker = b"saved optimizer and weights fixture"
            assert payload.count(marker) == 1
            candidate_archive.write_bytes(payload.replace(marker, b"X" + marker[1:]))
    else:
        colab.create_backup(candidate_checkpoint.parent, candidate_archive)
    namespace = settings()
    previous = {
        "backup_config": old_manifest["config"], "backup_manifest": old_manifest,
        "backup_archive": str(old_archive), "restored_checkpoint": str(old_checkpoint),
    }
    namespace["RUN_STATE"].update(previous)
    namespace.update(WORKFLOW_MODE="Continue training", SESSION=tmp_path / "session",
                     PYTHON=tmp_path / "python", REPO=tmp_path)
    namespace["SESSION"].mkdir()
    namespace["Path"] = lambda value: (
        tmp_path / "upload-inbox" if str(value) == "/content/jumper-backups" else Path(value)
    )
    monkeypatch.setattr(sys.modules["google.colab"].files, "upload",
                        lambda: {"candidate.zip": candidate_archive.read_bytes()})
    execute("# @title Continue training: backup selection", namespace,
            REPLACE_BACKUP_SELECTION=True)
    assert namespace["RUN_STATE"]["previous_backup_selection"] == previous
    assert namespace["RUN_STATE"]["backup_config"] == candidate_manifest["config"]
    assert namespace["RUN_STATE"]["restored_checkpoint"] is None
    return namespace, previous, candidate_checkpoint, candidate_manifest


def restore_runner(commands, configuration):
    """Run the real CPU restore/validator at the notebook's subprocess boundary."""
    def run(command, **kwargs):
        commands.append(command)
        if "restore" in command:
            checkpoint = colab.restore_backup(
                Path(command[command.index("--archive") + 1]),
                Path(command[command.index("--destination") + 1]),
            )
            colab.atomic_json(Path(command[command.index("--out") + 1]),
                              {"checkpoint": str(checkpoint)})
        else:
            assert command[1] == "-c"
            # Nested rollback try/except must not indent the child Python program.
            ast.parse(command[2])
            colab.validate_resume(Path(command[-1]), configuration)
    return run


@pytest.mark.parametrize("corruption,error,match", [
    ("crc", zipfile.BadZipFile, "Bad CRC"),
    ("checkpoint", ValueError, "no complete matching checkpoint"),
])
def test_metadata_readable_replacement_rolls_back_when_real_restore_fails(
    tmp_path, monkeypatch, external_ui, corruption, error, match
):
    namespace, previous, _, candidate_manifest = replacement_selection(
        tmp_path, monkeypatch, corruption
    )
    commands = []
    namespace["run"] = restore_runner(commands, candidate_manifest["config"])
    with pytest.raises(error, match=match):
        execute("# @title Verify the selected continuation checkpoint", namespace)
    assert len(commands) == 1 and "restore" in commands[0]
    assert all(namespace["RUN_STATE"][key] == value for key, value in previous.items())
    assert "previous_backup_selection" not in namespace["RUN_STATE"]
    assert namespace["RESTORED_CHECKPOINT"] == previous["restored_checkpoint"]
    assert Path(previous["restored_checkpoint"]).is_file()
    colab.validate_resume(Path(previous["restored_checkpoint"]), previous["backup_config"])


def test_replacement_rolls_back_when_real_resume_validation_fails(
    tmp_path, monkeypatch, external_ui
):
    namespace, previous, _, manifest = replacement_selection(tmp_path, monkeypatch)
    commands = []
    wrong_environment = {**manifest["config"], "num_envs": manifest["config"]["num_envs"] + 1}
    namespace["run"] = restore_runner(commands, wrong_environment)
    with pytest.raises(ValueError, match="resume configuration differs: num_envs"):
        execute("# @title Verify the selected continuation checkpoint", namespace)
    assert len(commands) == 2 and commands[1][1] == "-c"
    assert Path(commands[1][-1]).is_file()  # The candidate was fully restored first.
    assert all(namespace["RUN_STATE"][key] == value for key, value in previous.items())
    assert "previous_backup_selection" not in namespace["RUN_STATE"]
    assert namespace["RESTORED_CHECKPOINT"] == previous["restored_checkpoint"]
    colab.validate_resume(Path(previous["restored_checkpoint"]), previous["backup_config"])


def test_valid_replacement_commits_selection_and_repeat_does_not_reextract(
    tmp_path, monkeypatch, external_ui
):
    namespace, previous, checkpoint, manifest = replacement_selection(tmp_path, monkeypatch)
    commands = []
    namespace["run"] = restore_runner(commands, manifest["config"])
    execute("# @title Verify the selected continuation checkpoint", namespace)
    selected = namespace["RESTORED_CHECKPOINT"]
    assert selected != previous["restored_checkpoint"]
    assert Path(selected).read_bytes() == checkpoint.read_bytes()
    assert namespace["RUN_STATE"]["restored_checkpoint"] == selected
    assert namespace["RUN_STATE"]["backup_config"] == manifest["config"]
    assert "previous_backup_selection" not in namespace["RUN_STATE"]
    monkeypatch.setattr(sys.modules["google.colab"].files, "upload",
                        lambda: pytest.fail("Verified replacement requested another upload"))
    execute("# @title Continue training: backup selection", namespace)
    execute("# @title Verify the selected continuation checkpoint", namespace)
    assert sum("restore" in command for command in commands) == 1
    assert sum(command[1] == "-c" for command in commands) == 2
    assert namespace["RESTORED_CHECKPOINT"] == selected
    assert Path(previous["restored_checkpoint"]).is_file()


def test_demo_disabled_does_not_block_downloads_of_current_policy(tmp_path, external_ui):
    namespace = settings()
    execute("if RUN_DEMO:", namespace)
    assert "DEMO_VIDEO" not in namespace
    video = tmp_path / "trained.mp4"
    video.write_bytes(b"current recording")
    namespace.update(TRAIN_VIDEO=video, CHECKPOINT_PATH=tmp_path / "model_499.pt",
                     MEASUREMENT_CHECKPOINT="", DOWNLOAD_FILES=True)
    execute("files.download(str(TRAIN_VIDEO))", namespace)
    assert external_ui.downloads == [str(video)]


def test_curve_display_imports_are_independent_of_the_skipped_demo(tmp_path, external_ui):
    namespace = settings()
    execute("if RUN_DEMO:", namespace)
    assert "Image" not in namespace and "display" not in namespace
    artifacts, run_directory = tmp_path / "artifacts", tmp_path / "run"
    artifacts.mkdir()
    run_directory.mkdir()

    def run(command, **kwargs):
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps({"run": str(run_directory), "progress": {}}))
        Path(command[command.index("--plot") + 1]).write_bytes(b"curve fixture")

    namespace.update(ARTIFACTS=artifacts, RUN_DIR=run_directory, REPO=tmp_path,
                     PYTHON=tmp_path / "python", run=run)
    execute("METRICS_JSON =", namespace)
    assert external_ui.displays[0][0] == "image"
    replay = ast.parse(cell("SIM_STEPS ="))
    imports = [node for node in replay.body if isinstance(node, ast.ImportFrom)
               and node.module == "IPython.display"]
    assert {alias.name for node in imports for alias in node.names} >= {"Image", "Video", "display"}


def test_failed_run_backup_works_without_fresh_runtime_or_egl_metadata(tmp_path, external_ui):
    checkpoint, manifest = saved_run(tmp_path / "interrupted-run")
    namespace = settings()
    session = tmp_path / "fresh-session"
    session.mkdir()
    operation_manifest = tmp_path / "last-training.json"
    colab.atomic_json(operation_manifest, manifest)
    namespace["RUN_STATE"]["training_manifest"] = str(operation_manifest)
    namespace.update(SESSION=session, REPO=tmp_path, PYTHON=tmp_path / "python",
                     DOWNLOAD_FILES=False, uuid=uuid)
    assert not (session / "runtime.json").exists() and not (session / "graphics.json").exists()

    def run(command, **kwargs):
        colab.create_backup(
            Path(command[command.index("--run") + 1]),
            Path(command[command.index("--out") + 1]),
            extras=Path(command[command.index("--extras") + 1]),
        )

    namespace["run"] = run
    execute("BACKUP_CHECKPOINT =", namespace)
    restored = colab.restore_backup(namespace["BACKUP_ZIP"], tmp_path / "verified-restore")
    assert restored.read_bytes() == checkpoint.read_bytes()
    assert external_ui.downloads == []


def test_backup_after_a_second_run_fails_selects_that_runs_saved_checkpoint(
    tmp_path, external_ui
):
    """A failed second run leaves old Python globals; its manifest must select the new save."""
    first_checkpoint, first_manifest = saved_run(tmp_path / "successful-A")
    second_checkpoint, second_manifest = saved_run(tmp_path / "interrupted-B")
    first_manifest.update(status="finished", operation="run-A")
    second_manifest.update(status="failed", operation="run-B")
    session = tmp_path / "session"
    session.mkdir()
    for name in ("runtime.json", "graphics.json"):
        (session / name).write_text("{}")
    namespace = settings()
    namespace.update(SESSION=session, REPO=tmp_path, PYTHON=tmp_path / "python",
                     TASK="jumper.ripple", NUM_ENVS=64, PERSIST_ROOT=None, DOWNLOAD_FILES=False)

    def first_run(command, **kwargs):
        colab.atomic_json(Path(command[command.index("--manifest") + 1]), first_manifest)

    namespace["run"] = first_run
    execute('TRAIN_MANIFEST = SESSION / "training.json"', namespace)
    assert namespace["CHECKPOINT_PATH"] == first_checkpoint

    def interrupted_run(command, **kwargs):
        colab.atomic_json(Path(command[command.index("--manifest") + 1]), second_manifest)
        raise RuntimeError("second run interrupted after saving a complete checkpoint")

    namespace["run"] = interrupted_run
    with pytest.raises(RuntimeError, match="second run interrupted"):
        execute('TRAIN_MANIFEST = SESSION / "training.json"', namespace)
    assert namespace["CHECKPOINT_PATH"] == first_checkpoint  # Prove the stale-global trigger.

    def backup(command, **kwargs):
        colab.create_backup(Path(command[command.index("--run") + 1]),
                            Path(command[command.index("--out") + 1]),
                            extras=Path(command[command.index("--extras") + 1]))

    namespace["run"] = backup
    execute("BACKUP_CHECKPOINT =", namespace)
    assert namespace["CHECKPOINT_PATH"] == second_checkpoint
    assert namespace["RUN_DIR"] == second_checkpoint.parent
    assert namespace["ARTIFACTS"] == session / "artifacts/run-B"
    restored = colab.restore_backup(namespace["BACKUP_ZIP"], tmp_path / "restore-B")
    recovered = json.loads((restored.parent / colab.MANIFEST).read_text())
    assert recovered["run_directory"] == str(second_checkpoint.parent)


@pytest.mark.parametrize("operation", ["replay", "evaluation"])
def test_stale_checkpoint_replay_and_evaluation_fail_before_subprocess(
    tmp_path, external_ui, operation
):
    old_checkpoint, _ = saved_run(tmp_path / "old")
    selected_checkpoint, manifest = saved_run(tmp_path / "selected")
    selected_manifest = tmp_path / "training.json"
    colab.atomic_json(selected_manifest, manifest)
    namespace = settings()
    namespace.update(TASK=manifest["config"]["task"], MODEL=manifest["config"]["model"],
                     CHECKPOINT_PATH=old_checkpoint, training=manifest, TRAIN_MANIFEST=selected_manifest,
                     RUN_EVALUATION=True, rates={"supports_velocity_command": True},
                     run=lambda *args, **kwargs: pytest.fail("Stale checkpoint reached subprocess"))
    with pytest.raises(ValueError, match="checkpoint must match the selected training manifest"):
        execute("SIM_STEPS =" if operation == "replay" else "# @title Fixed-command evaluation", namespace)
    assert selected_checkpoint.is_file() and old_checkpoint.is_file()


@pytest.mark.parametrize("changed", [{"TASK": "jumper.flat"}, {"MODEL": "different-robot"}])
def test_evaluation_rejects_other_task_or_model_before_subprocess(tmp_path, external_ui, changed):
    checkpoint, manifest = saved_run(tmp_path / "selected")
    selected_manifest = tmp_path / "training.json"
    colab.atomic_json(selected_manifest, manifest)
    namespace = settings()
    namespace.update(TASK=manifest["config"]["task"], MODEL=manifest["config"]["model"],
                     CHECKPOINT_PATH=checkpoint, TRAIN_MANIFEST=selected_manifest,
                     RUN_EVALUATION=True, rates={"supports_velocity_command": True},
                     run=lambda *args, **kwargs: pytest.fail("Wrong task/model reached subprocess"))
    namespace.update(changed)
    with pytest.raises(ValueError, match="Evaluation task/model must match"):
        execute("# @title Fixed-command evaluation", namespace)


def test_matching_evaluation_control_executes_all_four_cases_with_selected_checkpoint(
    tmp_path, external_ui
):
    """The evaluation gates must reject mismatches without blocking legitimate evaluation."""
    checkpoint, manifest = saved_run(tmp_path / "selected")
    selected_manifest = tmp_path / "training.json"
    colab.atomic_json(selected_manifest, manifest)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    namespace = settings()
    execute("# @title Simulation and evaluation settings", namespace)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--evaluation-out") + 1]).write_text(json.dumps({
            "tracking_error": 0.02, "checkpoint": str(checkpoint),
        }))

    namespace.update(TASK=manifest["config"]["task"], MODEL=manifest["config"]["model"],
                     CHECKPOINT_PATH=checkpoint, TRAIN_MANIFEST=selected_manifest,
                     rates={"supports_velocity_command": True, "control_hz": 50.0, "physics_hz": 200.0},
                     SIM_SCENE="studio", EVAL_SECONDS=1.01, ARTIFACTS=artifacts,
                     REPO=tmp_path, PYTHON=tmp_path / "python", run=run)
    execute("# @title Fixed-command evaluation", namespace)
    assert len(commands) == 4
    vectors = [[float(value) for value in command[command.index("--command") + 1:
                                                 command.index("--command") + 4]]
               for command in commands]
    assert vectors == [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]
    for command in commands:
        assert command[command.index("--checkpoint") + 1] == checkpoint
        assert command[command.index("--task") + 1] == "jumper.ripple"
        assert command[command.index("--steps") + 1] == "51"
        assert command[command.index("--scene") + 1] == "studio"
    summary = json.loads((namespace["EVALUATION_DIR"] / "summary.json").read_text())
    assert [item["case"] for item in summary] == ["forward", "sideways", "turn", "stand"]
    assert all(item["scene"] == "studio" for item in summary)
