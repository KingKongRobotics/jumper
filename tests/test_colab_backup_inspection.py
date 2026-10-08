"""Backup inspection and extraction must reject the same unsafe archive before publishing."""

from __future__ import annotations

import importlib.util
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "colab_backup_helpers", Path(__file__).resolve().parents[1] / "rl/mjrl/colab.py"
)
assert SPEC is not None and SPEC.loader is not None
colab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab)


def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "model_9.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("model/data.pkl", "fixture policy; never deserialized")
    config = {"task": "robot.walk", "model": "robot", "num_envs": 256,
              "backend": "warp", "device": "cuda:0", "revision": "a" * 40,
              "source_sha256": "b" * 64, "runtime_versions": {"torch": "2.9.1+cu126"},
              "mjrl_environment": {"MJRL_THREADS": "1"}}
    colab.atomic_json(run / colab.MANIFEST, {
        "schema": 1, "config": config, "checkpoints": {checkpoint.name: colab.sha256(checkpoint)}
    })
    return run, config


def test_inspection_returns_provenance_and_needs_no_graphics_attachment(tmp_path):
    run, config = saved_run(tmp_path)
    archive = colab.create_backup(run, tmp_path / "backup.zip")
    inspected = colab.inspect_backup(archive)
    assert inspected["checkpoint"] == "run/model_9.pt"
    assert inspected["config"] == config
    assert inspected["runtime"] == config["runtime_versions"]
    assert inspected["source"] == {key: config[key] for key in ("revision", "source_sha256")}
    assert inspected["env_count"] == 256
    assert inspected["expanded_bytes"] == sum(member["size"] for member in inspected["members"])
    assert all(not member["path"].startswith("artifacts/") for member in inspected["members"])
    assert colab.restore_backup(archive, tmp_path / "restored").name == "model_9.pt"


@pytest.mark.parametrize("bad_path", ["../outside", "/absolute", "C:/file", "run\\bad",
                                      "run/../outside", "run/./file", "run//file"])
def test_inspection_and_restore_share_path_traversal_gate(tmp_path, bad_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        # ZipInfo's constructor normalizes Windows separators; set the raw wire name.
        entry = zipfile.ZipInfo()
        entry.filename = entry.orig_filename = bad_path
        stream.writestr(entry, "unsafe")
    with pytest.raises(ValueError, match="unsafe"):
        colab.inspect_backup(archive)
    with pytest.raises(ValueError, match="unsafe"):
        colab.restore_backup(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restore-*"))


@pytest.mark.parametrize("kind, message", [
    ("duplicate", "duplicate"), ("symlink", "non-regular"),
    ("parent_file", "parent file"), ("oversized", "size/member"),
])
def test_inspection_and_restore_share_member_safety_gate(tmp_path, kind, message):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        if kind == "duplicate":
            stream.writestr("same", "first")
            with pytest.warns(UserWarning, match="Duplicate"):
                stream.writestr("same", "second")
        elif kind == "symlink":
            entry = zipfile.ZipInfo("link")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(entry, "../outside")
        elif kind == "parent_file":
            stream.writestr("run", "file")
            stream.writestr("run/child", "cannot extract")
        else:
            stream.writestr("large", "x" * 100)
    limit = 20 if kind == "oversized" else 1024
    for action in (lambda: colab.inspect_backup(archive, max_bytes=limit),
                   lambda: colab.restore_backup(archive, tmp_path / "restored", max_bytes=limit)):
        with pytest.raises(ValueError, match=message):
            action()
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("key, value", [
    ("num_envs", True), ("num_envs", 0), ("runtime_versions", []),
    ("mjrl_environment", {"MJRL_THREADS": 4}), ("revision", "not-a-commit"),
    ("source_sha256", "short"), ("task", None),
])
def test_invalid_config_refused_by_both_inspection_and_restore(tmp_path, key, value):
    run, _ = saved_run(tmp_path)
    manifest = json.loads((run / colab.MANIFEST).read_text())
    manifest["config"][key] = value
    colab.atomic_json(run / colab.MANIFEST, manifest)
    archive = colab.create_backup(run, tmp_path / "backup.zip")
    for action in (lambda: colab.inspect_backup(archive),
                   lambda: colab.restore_backup(archive, tmp_path / "restored")):
        with pytest.raises(ValueError, match=key):
            action()
    assert not (tmp_path / "restored").exists()


def test_metadata_is_bounded_before_json_read_and_duplicates_are_refused(tmp_path):
    run, _ = saved_run(tmp_path)
    archive = colab.create_backup(run, tmp_path / "backup.zip")
    with pytest.raises(ValueError, match="metadata exceeds"):
        colab.inspect_backup(archive, max_metadata_bytes=20)
    duplicate = tmp_path / "duplicate.json.zip"
    with zipfile.ZipFile(duplicate, "w") as stream:
        stream.writestr("backup.json", '{"schema":1,"schema":1,"run":"run"}')
    with pytest.raises(ValueError, match="duplicate metadata key"):
        colab.inspect_backup(duplicate)


def test_inspection_uses_numeric_complete_recorded_checkpoint_and_checks_hash(tmp_path):
    run, _ = saved_run(tmp_path)
    checkpoint = run / "model_100.pt"
    with zipfile.ZipFile(checkpoint, "w") as stream:
        stream.writestr("model/data.pkl", "newer")
    manifest = json.loads((run / colab.MANIFEST).read_text())
    manifest["checkpoints"][checkpoint.name] = colab.sha256(checkpoint)
    colab.atomic_json(run / colab.MANIFEST, manifest)
    archive = colab.create_backup(run, tmp_path / "good.zip")
    assert colab.inspect_backup(archive)["checkpoint"] == "run/model_100.pt"
    bad = tmp_path / "replaced.zip"
    # Replacing every checkpoint leaves valid outer ZIP CRCs but invalid manifest hashes.
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, "w") as output:
        for entry in source.infolist():
            output.writestr(entry, b"replacement" if entry.filename.endswith(".pt")
                            else source.read(entry))
    for action in (lambda: colab.inspect_backup(bad),
                   lambda: colab.restore_backup(bad, tmp_path / "restored")):
        with pytest.raises(ValueError, match="no complete matching checkpoint"):
            action()


@pytest.mark.parametrize("ranges, expected", [
    (None, False), (SimpleNamespace(lin_vel_x=(0, 1)), False),
    (SimpleNamespace(lin_vel_x=(0, 1), lin_vel_y=(-1, 1), ang_vel_z=(-2, 2)), True),
    (SimpleNamespace(lin_vel_x=(0, 1), lin_vel_y=(-1, 1), ang_vel_z=(2, -2)), False),
])
def test_training_rates_resolves_velocity_support_from_command_config(ranges, expected):
    cfg = SimpleNamespace(commands={"twist": SimpleNamespace(ranges=ranges)},
                          sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)), decimation=4)
    assert colab.training_rates(cfg) == {"physics_hz": 200.0, "control_hz": 50.0,
                                         "supports_velocity_command": expected}
    cfg.commands = {}
    assert not colab.training_rates(cfg)["supports_velocity_command"]
