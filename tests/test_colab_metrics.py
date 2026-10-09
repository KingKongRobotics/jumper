"""Portable scalar reports must retain resume history and belong to the selected run."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "colab_metrics_helpers", Path(__file__).resolve().parents[1] / "rl/mjrl/colab.py"
)
assert SPEC is not None and SPEC.loader is not None
colab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(colab)


def event_reader(monkeypatch, samples):
    """A bounded reader is the control: omitting size_guidance loses most samples."""
    class Reader:
        def __init__(self, directory, *, size_guidance=None, purge_orphaned_data=True):
            self.limit = (size_guidance or {}).get("scalars", 500)
            self.purge = purge_orphaned_data

        def Reload(self):
            return self

        def Tags(self):
            return {"scalars": list(samples)}

        def Scalars(self, tag):
            events = samples[tag]
            if self.purge:
                events = [event for event in events if event.wall_time == 2]
            return events[-self.limit:] if self.limit else events

    for name in ("tensorboard", "tensorboard.backend", "tensorboard.backend.event_processing",
                 "tensorboard.backend.event_processing.event_accumulator"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["tensorboard.backend.event_processing.event_accumulator"].EventAccumulator = Reader
    return Reader


def run_events(tmp_path):
    run = tmp_path / "chosen-run"
    run.mkdir()
    (run / "events.out.tfevents.test").touch()
    return run


def test_all_resume_samples_are_sorted_and_written_to_json_and_csv(tmp_path, monkeypatch):
    events = [SimpleNamespace(step=index % 7, wall_time=index + 1, value=index / 10)
              for index in range(1003)]
    reader = event_reader(monkeypatch, {"Train/mean_reward": list(reversed(events)),
                                      "Loss/value_function": events[:3], "Empty": []})
    # This control fails completeness even though tags and valid event files are present.
    assert len(reader("control").Scalars("Train/mean_reward")) < len(events)
    output = tmp_path / "artifacts/metrics.json"
    selected = run_events(tmp_path)
    result = colab.collect_metrics(selected, output)
    data = json.loads(output.read_text())
    assert data["run"] == str(selected.resolve())
    points = data["tags"]["Train/mean_reward"]
    assert len(points) == 1003
    assert [(p["step"], p["wall_time"]) for p in points] == sorted(
        (e.step, float(e.wall_time)) for e in events
    )
    assert list(data["tags"]) == ["Loss/value_function", "Train/mean_reward"]
    with output.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1006
    assert rows[0] == {"tag": "Loss/value_function", "step": "0", "wall_time": "1.0",
                       "value": "0.0"}
    assert result["samples"] == 1006


def test_nonfinite_scalars_are_preserved_as_explicit_json_and_csv_markers(
    tmp_path, monkeypatch, capsys
):
    event_reader(monkeypatch, {
        "Train/mean_reward": [SimpleNamespace(step=3, wall_time=2, value=7.0)],
        "Curriculum/command/lin_err": [
            SimpleNamespace(step=index, wall_time=index + 1, value=value)
            for index, value in enumerate([float("nan"), float("inf"), -float("inf"), 0.25])
        ],
    })
    output = tmp_path / "metrics.json"
    summary = colab.collect_metrics(run_events(tmp_path), output)

    def reject_nonstandard_constant(value):
        raise ValueError(f"invalid bare JSON constant: {value}")

    data = json.loads(output.read_text(), parse_constant=reject_nonstandard_constant)
    values = [point["value"] for point in data["tags"]["Curriculum/command/lin_err"]]
    assert values == ["NaN", "Infinity", "-Infinity", 0.25]
    assert data["tags"]["Train/mean_reward"][0]["value"] == 7.0
    assert summary["samples"] == 5
    assert summary["nonfinite_samples"] == {"Curriculum/command/lin_err": 3}
    assert data["nonfinite_samples"] == summary["nonfinite_samples"]
    with output.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["value"] for row in rows[:4]] == ["NaN", "Infinity", "-Infinity", "0.25"]
    warning = capsys.readouterr().out
    assert "[WARN] Curriculum/command/lin_err: 3" in warning
    assert "retained" in warning


def test_nonfinite_wall_time_still_fails_before_publishing(tmp_path, monkeypatch):
    event_reader(monkeypatch, {
        "Train/mean_reward": [SimpleNamespace(step=0, wall_time=float("nan"), value=1.0)]
    })
    output = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="non-finite wall time"):
        colab.collect_metrics(run_events(tmp_path), output)
    assert not output.exists()


def test_progress_reports_latest_finite_curriculum_sample_without_inventing_missing_tags(
    tmp_path, monkeypatch
):
    event_reader(monkeypatch, {
        "Curriculum/command/level": [SimpleNamespace(step=4, wall_time=2, value=2.0),
                                     SimpleNamespace(step=8, wall_time=3, value=3.0)],
        "Curriculum/command/lin_err": [SimpleNamespace(step=7, wall_time=2, value=0.2),
                                       SimpleNamespace(step=8, wall_time=3, value=float("nan"))],
        "Curriculum/command/lin_err_bar": [SimpleNamespace(step=8, wall_time=3, value=0.08)],
        "Curriculum/command/ang_range": [SimpleNamespace(step=8, wall_time=3, value=float("nan"))],
        "Train/mean_reward": [SimpleNamespace(step=8, wall_time=3, value=7.0)],
    })
    output = tmp_path / "metrics.json"
    result = colab.collect_metrics(run_events(tmp_path), output)
    progress = result["progress"]
    assert progress == json.loads(output.read_text())["progress"]
    assert progress["Curriculum/command/level"] == {"step": 8, "wall_time": 3.0, "value": 3.0}
    assert progress["Curriculum/command/lin_err"] == {"step": 7, "wall_time": 2.0, "value": 0.2}
    assert set(progress) == {"Curriculum/command/level", "Curriculum/command/lin_err",
                             "Curriculum/command/lin_err_bar"}


@pytest.mark.parametrize("has_events", [False, True])
def test_missing_events_or_empty_scalar_samples_fail_before_publishing(
    tmp_path, monkeypatch, has_events
):
    event_reader(monkeypatch, {"Empty": []})
    selected = run_events(tmp_path)
    if not has_events:
        (selected / "events.out.tfevents.test").unlink()
    output = tmp_path / "artifacts/metrics.json"
    with pytest.raises(ValueError, match="event files"):
        colab.collect_metrics(selected, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("tags, expected", [
    (["Episode/return"], ["Episode/return"]),
    (["Curriculum/command", "Train/mean_reward", "Train/mean_episode_length", "Loss/value",
      "Loss/surrogate", "Policy/mean_std", "Perf/total_fps"],
     ["Train/mean_reward", "Train/mean_episode_length", "Loss/value", "Loss/surrogate",
      "Policy/mean_std", "Perf/total_fps"]),
    (["Train/mean_reward", "Train/mean_episode_length", "Loss/value", "Loss/surrogate",
      "Policy/mean_std", "Perf/total_fps", "Curriculum/command/level",
      "Curriculum/command/lin_err", "Curriculum/command/lin_err_bar"],
     ["Train/mean_reward", "Train/mean_episode_length", "Loss/value", "Loss/surrogate",
      "Policy/mean_std", "Perf/total_fps", "Curriculum/command/level", "Curriculum/command/lin_err"]),
])
def test_plot_uses_real_recorded_tags_and_closes_figure(tmp_path, monkeypatch, tags, expected):
    event_reader(monkeypatch, {
        **{tag: [SimpleNamespace(step=3, wall_time=1, value=float("nan")),
                 SimpleNamespace(step=4, wall_time=2, value=7.0),
                 SimpleNamespace(step=5, wall_time=3, value=float("inf")),
                 SimpleNamespace(step=6, wall_time=4, value=-float("inf"))] for tag in tags},
        "Curriculum/unavailable": [SimpleNamespace(step=4, wall_time=2, value=float("nan"))],
    })
    axes = []

    class Axis:
        def plot(self, x, y, **kwargs):
            self.points = x, y
            if kwargs:
                self.labels_plotted = getattr(self, "labels_plotted", []) + [kwargs["label"]]

        def legend(self):
            self.legend_visible = True

        def set(self, **labels):
            self.labels = labels

        def grid(self, *args, **kwargs):
            pass

        def set_visible(self, visible):
            self.visible = visible

    class Figure:
        def tight_layout(self):
            pass

        def savefig(self, path, **kwargs):
            Path(path).write_bytes(b"PNG generated by renderer fixture")

    figure = Figure()
    closed = []
    pyplot = ModuleType("matplotlib.pyplot")

    def subplots(rows, columns, **kwargs):
        axes.extend(Axis() for _ in range(rows * columns))
        return figure, SimpleNamespace(flat=axes)

    pyplot.subplots = subplots
    pyplot.close = closed.append
    matplotlib = ModuleType("matplotlib")
    backends = []
    matplotlib.use = backends.append
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    output = tmp_path / "artifacts/metrics.json"
    image = tmp_path / "images/training-curves.png"
    result = colab.collect_metrics(run_events(tmp_path), output, plot=image)
    assert backends == ["Agg"]
    assert image.is_file()
    assert axes[0].points[0] == [3, 4, 5, 6]
    assert axes[0].points[1][1] == 7.0
    assert all(math.isnan(axes[0].points[1][index]) for index in [0, 2, 3])
    assert axes[0].labels["xlabel"] == "Training iteration"
    assert axes[0].labels["ylabel"]
    assert closed == [figure]
    assert result["plotted_tags"] == expected
    if "Curriculum/command/lin_err" in expected:
        error_axis = axes[expected.index("Curriculum/command/lin_err")]
        assert error_axis.labels_plotted == ["Measured error", "Promotion threshold"]
        assert error_axis.legend_visible


def saved_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "model_1.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("model/data.pkl", "fixture")
    colab.atomic_json(run / colab.MANIFEST, {
        "schema": 1, "checkpoints": {checkpoint.name: colab.sha256(checkpoint)},
        "config": {"task": "robot.walk", "model": "robot", "num_envs": 256,
                   "backend": "warp", "device": "cuda:0", "revision": "a" * 40,
                   "source_sha256": "b" * 64, "runtime_versions": {"torch": "test"},
                   "mjrl_environment": {}},
    })
    return run


def test_backup_extras_round_trip_keeps_old_schema_and_selected_run(tmp_path):
    run = saved_run(tmp_path)
    extras = tmp_path / "artifacts"
    extras.mkdir()
    (extras / "metrics.json").write_text('{"run": "selected"}')
    (extras / "runtime.json").write_text('{"gpu": "cloud"}')
    archive = colab.create_backup(run, tmp_path / "backup.zip", extras=extras)
    with zipfile.ZipFile(archive) as content:
        assert json.loads(content.read("backup.json")) == {"schema": 1, "run": "run"}
        assert "artifacts/metrics.json" in content.namelist()
    checkpoint = colab.restore_backup(archive, tmp_path / "restored")
    assert checkpoint.read_bytes() == (run / "model_1.pt").read_bytes()
    assert (tmp_path / "restored/artifacts/runtime.json").read_text() == '{"gpu": "cloud"}'


def test_backup_extras_rejects_non_directory_without_publishing(tmp_path):
    run = saved_run(tmp_path)
    extras = tmp_path / "metrics.json"
    extras.write_text("{}")
    output = tmp_path / "backup.zip"
    with pytest.raises(ValueError, match="directory without symlinks"):
        colab.create_backup(run, output, extras=extras)
    assert not output.exists()


def test_backup_extras_rejects_nested_symlink_without_publishing(tmp_path):
    run = saved_run(tmp_path)
    extras = tmp_path / "artifacts"
    extras.mkdir()
    secret = tmp_path / "outside.json"
    secret.write_text("not part of the selected results")
    try:
        (extras / "link.json").symlink_to(secret)
    except OSError:
        pytest.skip("symlink permission is unavailable")
    output = tmp_path / "backup.zip"
    with pytest.raises(ValueError, match="refuses symlinks"):
        colab.create_backup(run, output, extras=extras)
    assert not output.exists()
