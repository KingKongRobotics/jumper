"""The performance, as opposed to the policy: what `scripts/export.py` writes here
beyond the policy and its contract.

`actor.onnx` + `layout.json` + `README.md` are what the robot needs. They are also
completely unwatchable, and this task's whole point is a dance. So this module
records the trained policy dancing the full choreography, puts the music back on
it, and ships the face-screen animation alongside.

## Why the artifacts are in a subdirectory

`out/<date-time>/` **is** the exported policy: `deploy/` copies that
directory to the board. A 30 MB video and a 5 MB mp3 sitting in it would be copied
too, over the network, to a device with an SD card. They go in `out/<checkpoint>/
media/` instead, which the importer does not walk into.

## The picture is `play.py --video`'s, not this module's

This module draws nothing. It runs `scripts/play.py --video` -- the repository's
one MP4 recorder (`rl/mjrl/viewer/video.py`) -- with this clip's camera and
length, and muxes the music onto what comes back. It used to roll the policy out
itself, record `qpos` and render afterwards through a `mujoco.Renderer` of its
own, and `tools/readme_media.py` had a third copy of the same idea. Three
renderers meant three cameras and three lists of which geoms to hide, and a
difference between two pictures of one policy read as a difference between two
policies. The one that remains is the one `play` shows everyone.

`--stop-on-done` is what keeps the old behaviour that mattered: **failure
terminations stay on.** If the policy falls over 40 seconds in, the recording
stops at 40 seconds and this says so, rather than showing a robot teleporting
back to the reference and carrying on -- which would misrepresent the policy
exactly where it matters most. The length is one short of the clip:
`MotionCommand._update_command` resamples on reaching `time_step_total`, which
puts the robot back at the first frame, and a replay would otherwise loop.

## The audio offset is real and is read, not assumed

The choreography opens with a silent lead-in: `audio_start_in_sim` is **2.0 s**,
and in the crab clip that figure is exact rather than approximate -- across all 496
beats, `beat_times_sim - beat_times_audio` is 2.0 at the first beat and 2.0 at the
last. Muxing the music at t=0 would put the whole performance two and a half beats
early, which looks *almost* right and is the kind of error nobody catches by
watching once.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..common.dance.motion import AUDIO_SUFFIXES, find_material

__all__ = ["export_media"]

REPO = Path(__file__).resolve().parents[3]
PLAY = REPO / "scripts" / "play.py"

#: Render size. 16:9 at a size that survives being watched full-screen without
#: making the file large enough to be annoying to move around: the crab clip comes
#: out around 25 MB at 235 s.
WIDTH, HEIGHT = 960, 540


def _fps_from(env_cfg) -> float:
    """Frames per second of the output video.

    **Not a free parameter**: it is the control rate, one rendered frame per policy
    step, so the video's time axis is the simulation's with no resampling and no
    accumulating drift. 50 Hz for this task (5 ms physics, decimation 4); changing
    the env's decimation changes this with no edit here.
    """
    return 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)


#: Camera. A fixed shot (`--no-video-follow`), because `body_x` and `body_y` are
#: **identically zero** through the entire clip -- the crab dances in place, and a
#: tracking camera would add motion the performance does not have. Aimed at the
#: mid-height of the body's 105-130 mm range. The azimuth is from the way the
#: robot faces in its first frame, which for this clip is the world's x.
#:
#: The elevation is set by a measurement rather than by taste. **The scene has no
#: skybox**, so everything above the horizon renders pure black, and a shallow
#: angle frames a band of it: at -18 degrees the top 66 rows of a 540-row frame
#: were black, which reads as an accidental letterbox. -26 degrees puts the horizon
#: off the top edge entirely (measured on the reference pose at frame 3000: 66 rows
#: at -18, 0 at -26, 0 at -32), and the distance comes in to 0.72 m to keep the
#: robot -- 105 mm tall, against mjlab's default framing for a 1.3 m humanoid --
#: filling the frame at the steeper angle.
CAM_LOOKAT_HEIGHT = 0.115
CAM_DISTANCE = 0.72
CAM_AZIMUTH = 138.0
CAM_ELEVATION = -26.0


def _ffmpeg() -> str:
    """The ffmpeg binary imageio ships, so this does not depend on a system one."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _frames_total(env_cfg) -> int:
    """How many control steps the whole dance is: the clip's frames, less one.

    One short, because on reaching `time_step_total` the motion command
    resamples, which teleports the robot back to the reference; stopping before
    that keeps the last frame part of the dance. The clip is the `.npz` the task
    tracks, read directly rather than through a built environment -- the
    environment is `play.py`'s to build.
    """
    import numpy as np

    clip = np.load(env_cfg.commands["motion"].motion_file)
    return int(clip["joint_pos"].shape[0]) - 1


def _record(path: Path, *, task_id: str, asset: Path | None, checkpoint: Path, res):
    """Run `play.py --video` for the whole clip. Returns `(steps, total, fps)`.

    The environment is built in **play mode**, which this task already defines as
    "the whole dance from its first frame, unperturbed": `episode_length_s` is
    effectively unbounded, observation noise and pushes are off, and the motion
    command's `sampling_mode` is `"start"` rather than the adaptive
    reference-state initialisation training uses. `--no-obs-noise` keeps it so
    (play puts training's noise back by default), and the physics rate is the
    task's own, so what is recorded is the dynamics the policy learned in.
    """
    import tasks

    env_cfg = tasks.load_env_cfg(task_id, asset=asset, play=True)
    fps = _fps_from(env_cfg)
    total = _frames_total(env_cfg)
    info = path.with_suffix(".replay.json")
    command = [
        sys.executable, str(PLAY), "--task", task_id, "--checkpoint", str(checkpoint),
        "--backend", str(res.backend), "--device", str(res.device), "--num_envs", "1",
        "--headless", "--no-obs-noise",
        "--physics-hz", f"{1.0 / env_cfg.sim.mujoco.timestep:g}",
        "--steps", str(total), "--stop-on-done",
        "--video", str(path), "--video-fps", f"{fps:g}",
        "--video-width", str(WIDTH), "--video-height", str(HEIGHT),
        "--video-distance", f"{CAM_DISTANCE:g}", "--video-azimuth", f"{CAM_AZIMUTH:g}",
        "--video-elevation", f"{CAM_ELEVATION:g}",
        "--no-video-follow", "--video-lookat-height", f"{CAM_LOOKAT_HEIGHT:g}",
        "--replay-info-out", str(info),
    ]
    if asset is not None:
        command += ["--model", str(asset)]
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        # Rendering must not need a window, and export is routinely run over
        # ssh. The GL backend binds when MuJoCo is imported, so it is set for the
        # child rather than in this process, where MuJoCo is already loaded.
        environment.setdefault("MUJOCO_GL", "egl")
    subprocess.run(command, check=True, cwd=REPO, env=environment)
    steps = int(json.loads(info.read_text(encoding="utf-8"))["control_steps"])
    info.unlink()
    return steps, total, fps


def _mux(video: Path, audio: Path, offset_s: float, out: Path) -> None:
    """Put the music on the video, starting `offset_s` in.

    `-shortest` cuts at whichever runs out first. That is the video in practice:
    the soundtrack is a few seconds longer than the choreography, and its tail is
    past the last beat.
    """
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-i", str(video),
        "-itsoffset", f"{offset_s:.6f}", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def export_media(out: Path, *, task_id: str, checkpoint: Path, asset, res) -> None:
    """Record the performance and ship it beside the bundle.

    Called by `scripts/export.py` through `tasks.load_export_media`; see that
    function for why the seam is shaped this way.
    """
    from ..common.dance.motion import load_source
    from .env_cfg import MEDIA

    material = find_material(MEDIA)
    # Checked before the recording, which takes minutes: the music is optional for
    # training, so this is the first place its absence matters, and the policy and
    # its contract are already written by the time this runs.
    if material.audio is None:
        raise FileNotFoundError(
            f"no music in {material.motion.parent} (looked for "
            f"{', '.join(AUDIO_SUFFIXES)}). The policy and its contract are written; "
            f"the performance video is the dance with its music on it, so it needs "
            f"the track the choreography was made for. Put it there and export "
            f"again, or pass --no-video."
        )
    media = Path(out) / "media"
    media.mkdir(parents=True, exist_ok=True)

    print(f"\n[export] recording the performance -> {media}")
    raw = media / "dance.silent.mp4"
    steps, total, fps = _record(
        raw, task_id=task_id, asset=asset, checkpoint=Path(checkpoint), res=res
    )
    if steps < total:
        print(
            f"[export] WARNING: the policy terminated at frame {steps} of "
            f"{total} ({steps / fps:.1f} s of {total / fps:.1f} s). The video is the "
            f"performance up to that point, not the whole choreography."
        )

    # The offset belongs to the choreography, so it is read from the clip rather
    # than written down here. See the module docstring.
    clip = load_source(material.motion)
    final = media / "dance.mp4"
    _mux(raw, material.audio, clip.audio_start_s, final)
    raw.unlink()

    shutil.copy2(material.audio, media / f"music{material.audio.suffix.lower()}")
    if material.eyes is not None:
        # Named, because the face animation is identified by being the one video in
        # `media/` -- so a reference recording left there is shipped as the robot's
        # face without anything else noticing. The name in the log is what makes
        # that visible.
        print(f"[export] face animation: {material.eyes.name}")
        shutil.copy2(material.eyes, media / f"eyes{material.eyes.suffix.lower()}")
    else:
        print(
            f"[export] no face animation found; put one in "
            f"{material.motion.parent}/ to ship it with the performance."
        )

    for f in sorted(media.iterdir()):
        print(f"    media/{f.name:<12}{f.stat().st_size / 1024 / 1024:8.1f} MB")
