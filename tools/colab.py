"""Colab process supervision and portable backups, using only the standard library.

Training still goes through scripts/train.py. This module owns provenance, explicit
resume checks and checkpoint persistence, never an environment or a policy runner.
Heavy imports occur only in the runtime/rate checks, inside the isolated interpreter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import queue
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "colab-run.json"
SCHEMA = 1
CHECKPOINT = re.compile(r"model_(\d+)\.pt\Z")
COMPATIBILITY = (
    "task", "model", "backend", "device", "num_envs", "revision", "source_sha256",
    "runtime_versions", "mjrl_environment",
)
RUNTIME_PACKAGES = ("torch", "mujoco", "mujoco-warp", "warp-lang", "numpy", "tensordict")


def atomic_json(path: Path, value: dict) -> None:
    """Readers see either the old complete JSON or the new complete JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(repo: Path) -> dict:
    """Commit plus content changes, including local dotenv overrides and new runtime code."""
    repo = Path(repo).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    roots = {"rl", "tasks", "scripts", "assets", "scenes", "controller"}
    digest = hashlib.sha256(subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *sorted(roots), "pyproject.toml"],
        cwd=repo,
    ))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo
    ).decode("utf-8").split("\0")
    for name in sorted(n for n in untracked if n and PurePosixPath(n).parts[0] in roots):
        path = repo / name
        if path.is_file() and not path.is_symlink():
            digest.update(name.encode("utf-8"))
            digest.update(bytes.fromhex(sha256(path)))
    for name in (".env", ".env.local"):
        path = repo / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    return {"revision": revision, "source_sha256": digest.hexdigest()}


def make_config(repo: Path, *, task: str, model: str, num_envs: int) -> dict:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return {
        "task": task, "model": model, "num_envs": num_envs,
        "backend": "warp", "device": "cuda:0", **source_identity(repo),
        "repository_url": subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo, text=True
        ).strip(),
        "runtime_versions": {p: importlib.metadata.version(p) for p in RUNTIME_PACKAGES},
        "mjrl_environment": {k: v for k, v in os.environ.items() if k.startswith("MJRL_")},
    }


def complete_checkpoint(path: Path) -> bool:
    """torch.save uses ZIP; CRC validation rejects a half-written saved checkpoint."""
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError):
        return False


def _valid_checkpoints(run: Path) -> list[tuple[int, Path]]:
    """Ignore interrupted saves and, when provenance exists, unrecorded/replaced files."""
    run = Path(run)
    manifest = run / MANIFEST
    recorded = None
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        recorded = saved.get("checkpoints", {})
        if saved.get("schema") != SCHEMA or not isinstance(recorded, dict):
            raise ValueError("unsupported checkpoint manifest")
    candidates = []
    for path in run.iterdir():
        match = CHECKPOINT.fullmatch(path.name)
        if (match is None or not path.is_file() or path.is_symlink()
                or not complete_checkpoint(path)):
            continue
        if recorded is not None and recorded.get(path.name) != sha256(path):
            continue
        candidates.append((int(match[1]), path))
    return candidates


def latest_checkpoint(run: Path) -> Path:
    """Choose the latest complete recorded save inside the explicitly selected run."""
    candidates = _valid_checkpoints(run)
    if not candidates:
        raise ValueError(f"no complete matching model_<iteration>.pt in this run: {run}")
    return max(candidates, key=lambda pair: pair[0])[1]


def validate_resume(checkpoint: Path, config: dict) -> dict:
    """Refuse a different task, simulator batch, source or replaced checkpoint."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not CHECKPOINT.fullmatch(checkpoint.name):
        raise ValueError("resume needs an explicit model_<iteration>.pt file")
    manifest = checkpoint.parent / MANIFEST
    if not manifest.is_file():
        raise ValueError(f"resume requires its saved {MANIFEST}: {manifest}")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("schema") != SCHEMA:
        raise ValueError("unsupported resume manifest schema")
    changes = [key for key in COMPATIBILITY if saved.get("config", {}).get(key) != config[key]]
    if changes:
        raise ValueError("resume configuration differs: " + ", ".join(changes))
    expected = saved.get("checkpoints", {}).get(checkpoint.name)
    if expected is None or sha256(checkpoint) != expected:
        raise ValueError("checkpoint does not match the saved run manifest")
    if not complete_checkpoint(checkpoint):
        raise ValueError("resume checkpoint is incomplete")
    return saved


def persist_checkpoint(checkpoint: Path, manifest: dict, destination: Path) -> Path:
    """Publish checkpoint and provenance together by renaming one complete directory.

    A new version is retained for every save, even when the last save reuses a name.
    Drive completion means the mounted filesystem accepted the writes; the notebook
    does not promise persistence after an abrupt VM termination during a copy.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination))
    try:
        copied = temporary / checkpoint.name
        shutil.copy2(checkpoint, copied)
        digest = sha256(copied)
        expected = manifest["checkpoints"][checkpoint.name]
        if digest != expected or not complete_checkpoint(copied):
            raise ValueError("checkpoint changed during persistence; nothing was published")
        atomic_json(temporary / MANIFEST, manifest)
        final = destination / f"{checkpoint.stem}-{digest[:12]}-{uuid.uuid4().hex[:8]}"
        os.rename(temporary, final)
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_training(command: list[str], *, repo: Path, manifest_path: Path, config: dict,
                 drive_root: Path | None = None, resume: Path | None = None,
                 poll_interval: float = 0.5) -> dict:
    """Stream the existing entry point and watch checkpoints while it is alive."""
    repo = Path(repo).resolve()
    if resume is not None:
        validate_resume(resume, config)
    operation = uuid.uuid4().hex
    manifest = {"schema": SCHEMA, "operation": operation, "config": config,
                "command": command, "status": "starting", "checkpoints": {},
                "resume_from": str(resume) if resume else None, "run_directory": None}
    atomic_json(manifest_path, manifest)
    existing = set((repo / "logs").glob("*/*/*"))
    run = None
    known: dict[Path, tuple[int, int]] = {}
    stable: dict[Path, tuple[int, int]] = {}
    lines: queue.Queue[str] = queue.Queue()
    process = None
    reader = None

    def record(force: bool = False) -> None:
        if run is None:
            return
        for path in sorted(run.glob("model_*.pt")):
            if not CHECKPOINT.fullmatch(path.name) or not path.is_file():
                continue
            info = path.stat()
            signature = (info.st_size, info.st_mtime_ns)
            if known.get(path) == signature:
                continue
            ready = force or stable.get(path) == signature
            stable[path] = signature
            if not ready or not complete_checkpoint(path):
                continue
            digest = sha256(path)
            # A concurrent final save may replace a file between reads.
            again = path.stat()
            if (again.st_size, again.st_mtime_ns) != signature:
                continue
            manifest["checkpoints"][path.name] = digest
            manifest["checkpoint"] = str(path)
            atomic_json(run / MANIFEST, manifest)
            atomic_json(manifest_path, manifest)
            if drive_root is not None:
                published = persist_checkpoint(path, manifest, drive_root / operation)
                print(f"[colab] checkpoint persisted: {published}", flush=True)
            known[path] = signature

    def consume(line: str) -> None:
        nonlocal run
        print(line, end="", flush=True)
        prefix = "[mjrl] log directory "
        if line.startswith(prefix):
            discovered = Path(line[len(prefix):].strip())
            discovered = (repo / discovered).resolve() if not discovered.is_absolute() else (
                discovered.resolve()
            )
            try:
                discovered.relative_to(repo / "logs")
            except ValueError as exc:
                raise ValueError("training announced a directory outside this checkout's logs") from exc
            if discovered in existing or run is not None:
                raise ValueError("training must create one new run, without reusing existing logs")
            run = discovered
            manifest.update(status="running", run_directory=str(run))
            atomic_json(run / MANIFEST, manifest)
            atomic_json(manifest_path, manifest)

    try:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        if sys.platform.startswith("linux"):
            environment.setdefault("MUJOCO_GL", "egl")
        process = subprocess.Popen(command, cwd=repo, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None

        def read_output() -> None:
            for line in process.stdout:
                lines.put(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_poll = 0.0
        while process.poll() is None or reader.is_alive() or not lines.empty():
            try:
                consume(lines.get(timeout=min(poll_interval, 0.2)))
            except queue.Empty:
                pass
            if time.monotonic() - last_poll >= poll_interval:
                record()
                last_poll = time.monotonic()
        record(force=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        if run is None or not manifest["checkpoints"]:
            raise RuntimeError("training exited without a new run and a complete checkpoint")
        manifest["checkpoint"] = str(latest_checkpoint(run))
        manifest["status"] = "finished"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # KeyboardInterrupt must remain the original error even if Drive disconnects
        # during the final copy. Complete saves not yet polled still deserve a backup.
        unwinding = sys.exc_info()[0] is not None
        cleanup_errors: list[BaseException] = []

        def cleanup(label, action) -> None:
            try:
                action()
            except BaseException as cleanup_error:  # noqa: BLE001 -- retain the original interruption
                cleanup_errors.append(cleanup_error)
                manifest["cleanup_errors"] = [
                    f"{type(e).__name__}: {e}" for e in cleanup_errors
                ]
                if not unwinding:
                    manifest.update(status="failed", error=manifest["cleanup_errors"][0])
                print(f"[colab] cleanup {label}: {cleanup_error}", file=sys.stderr, flush=True)

        def stop_process() -> None:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        cleanup("stop training", stop_process)
        if reader is not None:
            cleanup("join output reader", lambda: reader.join(timeout=1))
        if run is None:
            # An interruption can arrive after stdout was read but before its run
            # announcement was consumed. Discover it without replacing the end status.
            end_status = manifest["status"]
            while not lines.empty():
                cleanup("pending output", lambda: consume(lines.get_nowait()))
            manifest["status"] = end_status
        cleanup("final checkpoint scan", lambda: record(force=True))
        if cleanup_errors:
            manifest["cleanup_errors"] = [f"{type(e).__name__}: {e}" for e in cleanup_errors]
            if not unwinding:
                manifest.update(status="failed", error=manifest["cleanup_errors"][0])
        if process is not None and process.stdout is not None:
            cleanup("close output", process.stdout.close)
        cleanup("operation manifest", lambda: atomic_json(manifest_path, manifest))
        if run is not None:
            cleanup("run manifest", lambda: atomic_json(run / MANIFEST, manifest))
            if drive_root is not None:
                cleanup("Drive directory", lambda: drive_root.mkdir(parents=True, exist_ok=True))
                cleanup("Drive manifest", lambda: atomic_json(
                    drive_root / f"{operation}.json", manifest
                ))
        if cleanup_errors and not unwinding:
            raise cleanup_errors[0]
    return manifest


def collect_metrics(run: Path, output: Path, *, plot: Path | None = None) -> dict:
    """Export every recorded scalar; plotting never starts a TensorBoard server."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run, output = Path(run).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("metrics output must be a .json file")
    if plot is not None:
        plot = Path(plot).resolve()
        if plot.suffix.lower() != ".png":
            raise ValueError("metrics plot must be a .png file")
    if not run.is_dir() or not any(run.glob("events.out.tfevents.*")):
        raise ValueError(f"no TensorBoard event files in the selected run: {run}")
    # Keep all samples, including repeated iteration numbers across resumed runs.
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0},
                                   purge_orphaned_data=False)
    accumulator.Reload()
    tags = {}
    nonfinite_samples = {}
    for tag in sorted(accumulator.Tags().get("scalars", [])):
        points = [{"step": int(event.step), "wall_time": float(event.wall_time),
                   "value": float(event.value)} for event in accumulator.Scalars(tag)]
        if points:
            if any(not math.isfinite(point["wall_time"]) for point in points):
                raise ValueError(f"TensorBoard scalar contains a non-finite wall time: {tag}")
            invalid = 0
            for point in points:
                value = point["value"]
                if not math.isfinite(value):
                    point["value"] = ("NaN" if math.isnan(value) else
                                      "Infinity" if value > 0 else "-Infinity")
                    invalid += 1
            if invalid:
                nonfinite_samples[tag] = invalid
                print(f"[WARN] {tag}: {invalid} non-finite scalar samples retained as "
                      "NaN/Infinity/-Infinity strings; plots show gaps.", flush=True)
            tags[tag] = sorted(points, key=lambda point: (point["step"], point["wall_time"]))
    if not tags:
        raise ValueError(f"TensorBoard event files contain no scalar samples: {run}")
    progress = {}
    for name in ("level", "lin_err", "lin_err_bar", "lin_range", "ang_range"):
        tag = f"Curriculum/command/{name}"
        finite = [point for point in tags.get(tag, []) if isinstance(point["value"], float)]
        if finite:
            progress[tag] = dict(max(finite, key=lambda point: (point["step"], point["wall_time"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"schema": 1, "run": str(run), "tags": tags,
               "nonfinite_samples": nonfinite_samples, "progress": progress}
    atomic_json(output, metrics)
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=output.parent, prefix=".scalars-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["tag", "step", "wall_time", "value"])
            for tag, points in tags.items():
                for point in points:
                    writer.writerow([tag, point["step"], point["wall_time"], point["value"]])
        os.replace(temporary, output.with_suffix(".csv"))
    finally:
        temporary.unlink(missing_ok=True)
    selected = []
    if plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        preferred = ("Train/mean_reward", "Train/mean_episode_length",
                     "Loss/value", "Loss/surrogate", "Policy/mean_std",
                     "Perf/total_fps")
        finite_tags = [tag for tag, points in tags.items()
                       if any(isinstance(point["value"], float) for point in points)]
        curriculum = [tag for tag in ("Curriculum/command/level", "Curriculum/command/lin_err")
                      if tag in finite_tags]
        selected = [tag for tag in preferred if tag in finite_tags]
        selected += [tag for tag in finite_tags if tag not in selected and tag not in curriculum
                     and tag != "Curriculum/command/lin_err_bar"][:6 - len(selected)]
        selected += curriculum
        rows = max(1, (len(selected) + 1) // 2)
        figure, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows), squeeze=False)
        try:
            for axis, tag in zip(axes.flat, selected):
                points = tags[tag]
                values = [point["value"] if isinstance(point["value"], float) else float("nan")
                          for point in points]
                if tag == "Curriculum/command/lin_err":
                    axis.plot([point["step"] for point in points], values, label="Measured error")
                    threshold = tags.get("Curriculum/command/lin_err_bar", [])
                    if threshold:
                        axis.plot([point["step"] for point in threshold],
                                  [point["value"] if isinstance(point["value"], float)
                                   else float("nan") for point in threshold],
                                  label="Promotion threshold", linestyle="--")
                        axis.legend()
                else:
                    axis.plot([point["step"] for point in points], values)
                axis.set(title=tag, xlabel="Training iteration", ylabel="Recorded scalar value")
                axis.grid(True, alpha=0.3)
            for axis in list(axes.flat)[len(selected):]:
                axis.set_visible(False)
            if not selected:
                figure.suptitle("No finite scalar values to plot; inspect metrics.json and CSV")
            figure.tight_layout()
            plot = Path(plot)
            plot.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(plot, dpi=140)
        finally:
            plt.close(figure)
    return {"run": str(run), "output": str(output), "scalar_tags": len(tags),
            "samples": sum(len(points) for points in tags.values()), "plotted_tags": selected,
            "nonfinite_samples": nonfinite_samples, "progress": progress}


def create_backup(run: Path, output: Path, *, export: Path | None = None,
                  video: Path | None = None, extras: Path | None = None) -> Path:
    """Package a named run and optional results, without searching unrelated logs."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if not (run / MANIFEST).is_file():
        raise ValueError("backup needs a Colab run manifest")
    latest_checkpoint(run)
    valid = {path.name: sha256(path) for _, path in _valid_checkpoints(run)}
    if output.exists():
        raise FileExistsError(f"backup already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [("run", run)]
    if export is not None:
        sources.append(("export", Path(export).resolve()))
    if video is not None:
        sources.append(("video", Path(video).resolve()))
    if extras is not None:
        extras = Path(extras)
        if extras.is_symlink() or not extras.is_dir():
            raise ValueError("backup extras must be a directory without symlinks")
        sources.append(("artifacts", extras.resolve()))
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".backup-", delete=False) as stream:
        temporary = Path(stream.name)
    created = False
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("backup.json", json.dumps({"schema": SCHEMA, "run": "run"}))
            for prefix, source in sources:
                if not source.exists():
                    raise FileNotFoundError(source)
                paths = sorted(source.rglob("*")) if source.is_dir() else [source]
                for path in paths:
                    if path.is_symlink():
                        raise ValueError(f"backup refuses symlinks: {path}")
                    if path.is_file():
                        if path.suffix == ".pt" and (
                            path.name not in valid or not complete_checkpoint(path)
                            or sha256(path) != valid[path.name]
                        ):
                            continue
                        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
                        archive.write(path, (PurePosixPath(prefix) / relative.as_posix()).as_posix())
        # Exclusive publication also refuses another backup written concurrently.
        with output.open("xb") as destination, temporary.open("rb") as source:
            created = True
            shutil.copyfileobj(source, destination)
    except BaseException:
        # Only our newly created incomplete output can be removed.
        if created:
            output.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _inspect_archive(archive: zipfile.ZipFile, *, max_bytes: int, max_members: int,
                     max_metadata_bytes: int) -> dict:
    """Shared inspection for listing and restore; never deserialize policy objects."""
    entries = archive.infolist()
    expanded_bytes = sum(entry.file_size for entry in entries)
    if len(entries) > max_members or expanded_bytes > max_bytes:
        raise ValueError("backup exceeds restore size/member limits")
    seen = set()
    for entry in entries:
        name = entry.orig_filename
        path = PurePosixPath(name)
        if (not name or "\x00" in name or "\\" in name or ":" in name or path.is_absolute()
                or any(part in ("..", ".") for part in name.rstrip("/").split("/"))
                or "" in name.rstrip("/").split("/") or str(path) in seen):
            raise ValueError(f"unsafe or duplicate archive path: {name}")
        seen.add(str(path))
        mode = entry.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError(f"archive contains a non-regular entry: {name}")
        if entry.flag_bits & 1:
            raise ValueError("encrypted backups are unsupported")
    # A regular member cannot also be an ancestor of another member.
    files = {entry.filename.rstrip("/") for entry in entries if not entry.is_dir()}
    if any(str(parent) in files for entry in entries
           for parent in PurePosixPath(entry.filename).parents if str(parent) != "."):
        raise ValueError("unsafe archive path conflicts with a parent file")

    def strict_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate metadata key: {key}")
            value[key] = item
        return value

    def nonfinite(value):
        raise ValueError(f"non-finite metadata value: {value}")

    def metadata(name):
        try:
            info = archive.getinfo(name)
        except KeyError as exc:
            raise ValueError(f"backup is missing metadata: {name}") from exc
        if info.file_size > max_metadata_bytes:
            raise ValueError(f"backup metadata exceeds size limit: {name}")
        with archive.open(info) as stream:
            data = stream.read(max_metadata_bytes + 1)
        if len(data) > max_metadata_bytes:
            raise ValueError(f"backup metadata exceeds size limit: {name}")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=strict_object,
                           parse_constant=nonfinite)
        if not isinstance(value, dict):
            raise ValueError(f"backup metadata must be an object: {name}")  # noqa: TRY004
        return value

    backup_metadata = metadata("backup.json")
    if (backup_metadata != {"schema": SCHEMA, "run": "run"}
            or type(backup_metadata.get("schema")) is not int):
        raise ValueError("unsupported backup schema")
    saved = metadata(f"run/{MANIFEST}")
    if saved.get("schema") != SCHEMA or type(saved.get("schema")) is not int:
        raise ValueError("unsupported run manifest schema")
    config = saved.get("config")
    if not isinstance(config, dict) or any(key not in config for key in COMPATIBILITY):
        raise ValueError("backup run config is missing compatibility fields")
    for key in ("task", "model", "backend", "device"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"backup run config has invalid {key}")
    if type(config["num_envs"]) is not int or config["num_envs"] < 1:
        raise ValueError("backup run config has invalid num_envs")
    for key, pattern in (("revision", r"[0-9a-f]{40}(?:[0-9a-f]{24})?"),
                         ("source_sha256", r"[0-9a-f]{64}")):
        if not isinstance(config[key], str) or not re.fullmatch(pattern, config[key]):
            raise ValueError(f"backup run config has invalid {key}")
    for key in ("runtime_versions", "mjrl_environment"):
        if not isinstance(config[key], dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in config[key].items()
        ):
            raise ValueError(f"backup run config has invalid {key}")
    if "repository_url" in config and not isinstance(config["repository_url"], str):
        raise ValueError("backup run config has invalid repository_url")
    recorded = saved.get("checkpoints")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("backup checkpoint manifest is empty or invalid")
    if any(not isinstance(name, str) or not CHECKPOINT.fullmatch(name)
           or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
           for name, digest in recorded.items()):
        raise ValueError("backup checkpoint manifest has invalid names or hashes")
    candidates = []
    for name, expected in recorded.items():
        member = f"run/{name}"
        if member not in files:
            continue
        digest = hashlib.sha256()
        # Policy files are ZIPs too. Spill larger saves to disk instead of accumulating RAM.
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024**2) as checkpoint:
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checkpoint.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                continue
            checkpoint.seek(0)
            try:
                with zipfile.ZipFile(checkpoint) as contents:
                    members = contents.infolist()
                    if not members or len(members) > max_members:
                        continue
                    if sum(info.file_size for info in members) > max_bytes:
                        raise ValueError("checkpoint exceeds expanded size limit")
                    if contents.testzip() is not None:
                        continue
            except (zipfile.BadZipFile, EOFError):
                continue
        candidates.append((int(CHECKPOINT.fullmatch(name)[1]), member))
    if not candidates:
        raise ValueError("backup has no complete matching checkpoint")
    return {"schema": SCHEMA, "checkpoint": max(candidates)[1], "config": config,
            "source": {key: config[key] for key in ("revision", "source_sha256")},
            "runtime": config["runtime_versions"], "env_count": config["num_envs"],
            "members": [{"path": entry.filename, "size": entry.file_size} for entry in entries],
            "expanded_bytes": expanded_bytes}


def inspect_backup(archive_path: Path, *, max_bytes: int = 10 * 1024**3,
                   max_members: int = 10000, max_metadata_bytes: int = 1024**2) -> dict:
    """Inspect bounded metadata, all member paths and recorded checkpoint integrity."""
    with zipfile.ZipFile(archive_path) as archive:
        result = _inspect_archive(archive, max_bytes=max_bytes, max_members=max_members,
                                  max_metadata_bytes=max_metadata_bytes)
    if not result["config"].get("repository_url"):
        print("[WARN] This older backup has no repository_url. Set the advanced "
              "REPOSITORY_URL to the repository containing its recorded revision; "
              "do not assume the upstream repository contains a fork commit.", flush=True)
    return {"archive": str(Path(archive_path).resolve()), **result}


def restore_backup(archive_path: Path, destination: Path, *, max_bytes: int = 10 * 1024**3,
                   max_members: int = 10000, max_metadata_bytes: int = 1024**2) -> Path:
    """Validate all paths before extracting into a new directory; never overwrite a run."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            inspected = _inspect_archive(archive, max_bytes=max_bytes, max_members=max_members,
                                         max_metadata_bytes=max_metadata_bytes)
            entries = archive.infolist()
            total = 0
            for entry in entries:
                target = temporary.joinpath(*PurePosixPath(entry.filename).parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("expanded backup exceeds restore size limit")
                        output.write(chunk)
        run = temporary / "run"
        saved = json.loads((run / MANIFEST).read_text(encoding="utf-8"))
        checkpoint = temporary.joinpath(*PurePosixPath(inspected["checkpoint"]).parts)
        recorded = saved.get("checkpoints", {}).get(checkpoint.name)
        if not recorded or sha256(checkpoint) != recorded:
            raise ValueError("backup checkpoint does not match its manifest")
        # The file tree is new; historical paths stay in provenance, not in resume selection.
        os.rename(temporary, destination)
        return destination / "run" / checkpoint.name
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_runtime(repo: Path) -> dict:
    """Training gates only: CUDA execution, Warp kernel, native interface and import origins."""
    repo = Path(repo).resolve()
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("this repository needs Python 3.10 through 3.13")
    if Path(sys.prefix).resolve() != repo / ".venv" or sys.prefix == sys.base_prefix:
        raise RuntimeError("run with this checkout's isolated .venv interpreter")
    expected = {"tasks": repo / "tasks", "mjrl": repo / "rl/mjrl",
                "mjlab": repo / "rl/mjlab", "rsl_rl": repo / "rl/rsl_rl"}
    for name, directory in expected.items():
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve().parent
        if actual != directory.resolve():
            raise RuntimeError(f"{name} imports from {actual}, expected {directory}")
        print(f"[PASS] import {name}: {actual}", flush=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime and rerun setup")
    value = torch.tensor([2.0], device="cuda:0") * 3.0
    torch.cuda.synchronize()
    if value.item() != 6.0:
        raise RuntimeError("torch CUDA arithmetic failed")
    print(f"[PASS] torch CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    import mujoco
    from mujoco import rollout

    if not callable(rollout.rollout):
        raise TypeError("native MuJoCo rollout interface is unavailable")
    importlib.import_module("mjrl.backend.native_sim")
    print(f"[PASS] native MuJoCo {mujoco.__version__}: rollout and backend imported", flush=True)
    probe = '''import warp as wp
wp.init()
devices = [str(device) for device in wp.get_devices()]
if "cuda:0" not in devices:
    raise RuntimeError(f"Warp cannot enumerate cuda:0: {devices}")
@wp.kernel
def add_one(output: wp.array(dtype=wp.int32)):
    i = wp.tid()
    output[i] = i + 1
output = wp.zeros(4, dtype=wp.int32, device="cuda:0")
wp.launch(add_one, dim=4, outputs=[output], device="cuda:0")
wp.synchronize_device("cuda:0")
if output.numpy().tolist() != [1, 2, 3, 4]:
    raise RuntimeError("Warp CUDA kernel failed")
print("[PASS] Warp CUDA enumeration and executed kernel:", devices)
'''
    with tempfile.TemporaryDirectory(prefix="jumper-warp-") as directory:
        script = Path(directory) / "probe.py"
        script.write_text(probe, encoding="utf-8")
        subprocess.run([sys.executable, str(script)], check=True, cwd=repo)
    return {"python": sys.version, "interpreter": sys.executable,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "versions": {p: importlib.metadata.version(p) for p in RUNTIME_PACKAGES},
            **source_identity(repo)}


def training_rates(cfg) -> dict:
    """Describe resolved simulation timing and whether fixed velocity replay applies."""
    twist = cfg.commands.get("twist")
    ranges = getattr(twist, "ranges", None)
    supported = True
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        bounds = getattr(ranges, name, None)
        if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in bounds) or bounds[0] > bounds[1]):
            supported = False
            break
    return {"physics_hz": 1.0 / cfg.sim.mujoco.timestep,
            "control_hz": 1.0 / (cfg.sim.mujoco.timestep * cfg.decimation),
            "supports_velocity_command": supported}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    check = commands.add_parser("check")
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--repo", type=Path, required=True)
    train.add_argument("--task", required=True)
    train.add_argument("--model", default="jumper")
    train.add_argument("--num-envs", type=int, required=True)
    train.add_argument("--iterations", type=int, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--drive-root", type=Path)
    train.add_argument("--resume", type=Path)
    backup = commands.add_parser("backup")
    backup.add_argument("--run", type=Path, required=True)
    backup.add_argument("--out", type=Path, required=True)
    backup.add_argument("--export", type=Path)
    backup.add_argument("--video", type=Path)
    backup.add_argument("--extras", type=Path)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--run", type=Path, required=True)
    metrics.add_argument("--out", type=Path, required=True)
    metrics.add_argument("--plot", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--out", type=Path, required=True)
    rates = commands.add_parser("rates")
    rates.add_argument("--repo", type=Path, required=True)
    rates.add_argument("--task", required=True)
    rates.add_argument("--model", default="jumper")
    rates.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "check":
        atomic_json(args.out, verify_runtime(args.repo))
    elif args.mode == "train":
        if args.iterations < 1:
            parser.error("iterations must be positive")
        config = make_config(args.repo, task=args.task, model=args.model, num_envs=args.num_envs)
        command = [sys.executable, "scripts/train.py", "--task", args.task,
                   "--model", args.model, "--backend", "warp", "--device", "cuda:0",
                   "--num_envs", str(args.num_envs), "--max-iterations", str(args.iterations),
                   "--headless", "--no-tensorboard", "--logger", "tensorboard"]
        if args.resume is not None:
            command += ["--checkpoint", str(args.resume.resolve())]
        run_training(command, repo=args.repo, manifest_path=args.manifest, config=config,
                     drive_root=args.drive_root, resume=args.resume)
    elif args.mode == "backup":
        print(create_backup(args.run, args.out, export=args.export, video=args.video,
                            extras=args.extras))
    elif args.mode == "metrics":
        print(json.dumps(collect_metrics(args.run, args.out, plot=args.plot)))
    elif args.mode == "restore":
        restored = restore_backup(args.archive, args.destination)
        atomic_json(args.out, {"checkpoint": str(restored), "run": str(restored.parent)})
        print(f"[colab] restored checkpoint: {restored}")
    elif args.mode == "inspect":
        atomic_json(args.out, inspect_backup(args.archive))
    else:
        from mjrl.dotenv import load_dotenv

        import tasks

        load_dotenv(args.repo / ".env", args.repo / ".env.local")
        asset = tasks.get(args.task).resolve_asset(args.model)
        cfg = tasks.load_env_cfg(args.task, asset=asset)
        atomic_json(args.out, training_rates(cfg))


if __name__ == "__main__":
    main()
