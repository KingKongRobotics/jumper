"""Draw the pictures in README.md: trained policies, in simulation, off-screen.

    python tools/readme_media.py                    # every clip, and the still
    python tools/readme_media.py --only walk jump   # some of them
    python tools/readme_media.py --list

Each clip is one task's committed export (`tasks/<task>/out/example/`) run in a
replay environment, so a picture in the README is of a policy somebody can load,
not of one that lived in a `logs/` directory and is gone. Re-run it when a model,
a scene or an export changes; it writes `docs/media/`.

## The frames are `play.py --video`'s

This file draws nothing. Each clip is one run of `scripts/play.py --video` -- the
repository's one MP4 recorder, `rl/mjrl/viewer/video.py` -- on the native backend
with one environment, headless, in the studio scene, at the task's own physics
rate; the GIF and the still are cut from that MP4 with ffmpeg afterwards. It
used to render for itself, as did the dance export, and three renderers meant
three cameras and three lists of which geoms to hide: a difference between two
pictures of one policy read as a difference between two policies.

## The command is scripted, the way the operator's is written

A replay hands the velocity and posture commands to the operator -- the pad and
the keyboard. Nobody is holding either here, so each clip's command is a function
of time, handed to `play.py --command-script` and written into the command term
after the term's own `compute`, which is where the operator writes its own. The
pad is never opened for a scripted term: one that is plugged in and nudged would
drive the clip instead. The scripts are the module-level functions below, named
by `Clip.script`; `play` loads this file by path to find them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "media"
PLAY = REPO / "scripts" / "play.py"

#: What is drawn, and what the README shows. Drawn at twice the size and scaled
#: down, which is the anti-aliasing: a 360-pixel render of a robot whose legs are
#: 20 mm across is mostly stair-steps.
DRAW = (720, 540)
SHOW = (360, 270)

#: Frames a second. A GIF counts a frame's delay in hundredths of a second, and
#: 12.5 is 8 of them exactly -- 15 or 25 would be rounded, and the clip would play
#: at a speed the policy did not. It also divides both control rates there are:
#: one step in 4 of a 50 Hz task, one in 16 of jumper.posture's and jumper.jump's
#: 200 Hz. Taking "one in 4" for granted drew those two at 50 frames a second,
#: four times the file for the same picture. The recorder samples at exactly this
#: rate, so the GIF's frames are control steps and not interpolations.
FPS = 12.5

#: The still at the top of the README, drawn at twice this for the same reason.
STILL = (1600, 640)

SCENE = "studio"


@dataclass(frozen=True)
class Camera:
    """`play.py --video`'s camera, placed from the robot's first frame.

    The recorder puts the azimuth relative to the way the robot faces and, with
    `follow`, tracks the body over the ground looking at its centre of mass; a
    fixed camera (`follow=False`) stays where the first frame put it, looking at
    `height`. See the docstring of `mjrl/viewer/video.py`.
    """
    distance: float = 0.72
    #: Degrees, **from the way the robot faces in the clip's first frame** and not
    #: from the world's x: a reset turns the robot to a heading of its own, and a
    #: camera placed in the world showed the front of one clip and the back of the
    #: next. 0 looks along the robot from behind it, 180 straight at its front.
    azimuth: float = 140.0
    elevation: float = -16.0
    #: Height of the point a fixed camera looks at, metres. The body rides at
    #: about 0.11. A following camera looks at the body and ignores this.
    height: float = 0.06
    #: Follow the body over the ground. Off for a motion that stays where it is,
    #: where a camera that moved would add motion the policy did not make.
    follow: bool = True


@dataclass(frozen=True)
class Clip:
    task: str
    #: Seconds of rollout before the first frame kept -- the settling after a
    #: reset, or the part of a recording before the part worth showing.
    skip: float
    #: How long the clip is. `None` is to the end of the recording the task
    #: tracks: a gesture is drawn whole, from HOME and back to it, so the loop
    #: closes on the pose it opened with.
    seconds: float | None
    camera: Camera = field(default_factory=Camera)
    #: The name of a function in this file: `(seconds since the clip began) ->
    #: {command term: values}`, handed to `play.py --command-script`. A term not
    #: named keeps what it samples. None scripts nothing.
    script: str | None = None


def _ramp(t: float, t0: float, t1: float) -> float:
    """0 before `t0`, 1 after `t1`, smooth in between: a stick is not a switch."""
    x = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return x * x * (3.0 - 2.0 * x)


#: A posture's height that is NaN is the term's own neutral height; `(height,
#: share)` is that share of the way there from it. `mjrl.replay.script_commands`
#: reads both.
_NEUTRAL = (0.0, 0.0, 0.0, float("nan"))


def walk(t: float) -> dict:
    """jumper.posture walking forward at 0.5 m/s, holding its neutral posture."""
    return {"twist": (0.5 * _ramp(t, 0.3, 1.3), 0.0, 0.0), "posture": _NEUTRAL}


def claw(t: float) -> dict:
    """jumper.five_foot walking forward at 0.35 m/s."""
    return {"twist": (0.35 * _ramp(t, 0.3, 1.3), 0.0, 0.0)}


#: jumper.posture's standing band is 30, 20 and 15 degrees and 0.07 to 0.15 m
#: (`tasks/jumper/posture/env_cfg.py`); the clip asks for most of each, not all.
_TWIST, _PITCH, _ROLL, _HIGH = 0.45, 0.30, 0.22, 0.14


def pose(t: float) -> dict:
    """jumper.posture standing: pitch, roll, twist, then height, each back to rest."""
    def bump(t0: float) -> float:
        return _ramp(t, t0, t0 + 0.4) - _ramp(t, t0 + 1.0, t0 + 1.4)

    lift = bump(4.4)
    return {
        "twist": (0.0, 0.0, 0.0),
        "posture": (_TWIST * bump(3.0), _PITCH * bump(0.2), _ROLL * bump(1.6), (_HIGH, lift)),
    }


CLIPS: dict[str, Clip] = {
    "walk": Clip("jumper.posture", skip=1.0, seconds=4.0, script="walk"),
    "posture": Clip("jumper.posture", skip=1.0, seconds=6.0,
                    camera=Camera(follow=False), script="pose"),
    "claw": Clip("jumper.five_foot", skip=1.0, seconds=4.0, script="claw"),
    # Followed over the ground and not in height, so the jump is the robot
    # leaving the floor and not the floor leaving the picture.
    "jump": Clip("jumper.jump", skip=0.0, seconds=1.76,
                 camera=Camera(distance=0.85, height=0.17)),
    "dance": Clip("jumper.dance_brazilian", skip=6.0, seconds=5.0,
                  camera=Camera(follow=False)),
    # From the side of the arm that waves; from the other, the body is in the way.
    "gesture": Clip("jumper.gesture_hello", skip=0.0, seconds=None,
                    camera=Camera(azimuth=205.0, follow=False)),
}

#: The clip the still is a frame of, and how far into it.
STILL_FROM, STILL_AT = "posture", 0.2
STILL_CAMERA = Camera(distance=0.62, azimuth=145.0, elevation=-12.0, height=0.07, follow=False)


def _checkpoint(task: str) -> Path:
    out = REPO / "tasks" / Path(*task.split(".")) / "out" / "example"
    found = sorted(out.glob("model_*.pt"))
    if len(found) != 1:
        raise SystemExit(
            f"error: {out.relative_to(REPO)} holds {len(found)} checkpoints, and a clip "
            f"is of exactly one. `scripts/export.py --task {task}` writes that directory."
        )
    return found[0]


def _rates(clip: Clip) -> tuple[float, float, int]:
    """`(physics Hz, control Hz, control steps)` of `clip`, from the task's config.

    Read from the config rather than from a built environment: the environment
    is `play.py`'s to build. The whole recording, for a clip with no length of
    its own, is one step short of the motion the task tracks -- on reaching the
    end the command resamples, which puts the robot back at the first frame.
    """
    import tasks

    env_cfg = tasks.load_env_cfg(clip.task, play=True)
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    if clip.seconds is None:
        import numpy as np

        motion = np.load(env_cfg.commands["motion"].motion_file)
        steps = int(motion["joint_pos"].shape[0]) - 1
    else:
        steps = round((clip.skip + clip.seconds) / step_dt)
    return 1.0 / env_cfg.sim.mujoco.timestep, 1.0 / step_dt, steps


def _record(clip: Clip, camera: Camera, size: tuple[int, int], fps: float, steps: int,
            physics_hz: float, path: Path) -> None:
    """One run of `play.py --video`, the clip's camera and script on the line."""
    command = [
        sys.executable, str(PLAY), "--task", clip.task,
        "--checkpoint", str(_checkpoint(clip.task)),
        "--backend", "native", "--device", "cpu", "--num_envs", "1",
        "--headless", "--no-obs-noise", "--scene", SCENE,
        "--physics-hz", f"{physics_hz:g}", "--steps", str(steps),
        "--video", str(path), "--video-fps", f"{fps:g}",
        "--video-width", str(size[0]), "--video-height", str(size[1]),
        "--video-distance", f"{camera.distance:g}",
        "--video-azimuth", f"{camera.azimuth:g}",
        "--video-elevation", f"{camera.elevation:g}",
    ]
    if not camera.follow:
        command += ["--no-video-follow", "--video-lookat-height", f"{camera.height:g}"]
    if clip.script is not None:
        command += ["--command-script", f"{Path(__file__).resolve()}:{clip.script}"]
    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        # No window: the GL backend binds when MuJoCo is imported, in the child.
        environment.setdefault("MUJOCO_GL", "egl")
    subprocess.run(command, check=True, cwd=REPO, env=environment)


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _gif(mp4: Path, skip: float, seconds: float | None, path: Path) -> None:
    # One palette for the whole clip, from the clip: a GIF has 256 colours, and
    # the generic ones spend most of them on hues a grey studio lacks. `fps`
    # first: without it the muxer fills the clip out to 50 frames a second with
    # copies, which plays the same and is four times the frames.
    graph = (
        f"fps={FPS:g},scale={SHOW[0]}:{SHOW[1]}:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    cut = ["-ss", f"{skip:g}"] + ([] if seconds is None else ["-t", f"{seconds:g}"])
    subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", *cut, "-i", str(mp4),
         "-filter_complex", graph, "-loop", "0", str(path)],
        check=True,
    )


def _png(mp4: Path, frame: int, path: Path) -> None:
    """Frame `frame` of `mp4`, scaled to `STILL`."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "frame.png"
        subprocess.run(
            [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(raw)],
            check=True,
        )
        Image.open(raw).resize(STILL, Image.Resampling.LANCZOS).save(path, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(CLIPS), metavar="CLIP",
                    help="draw these and leave the rest as they are")
    ap.add_argument("--list", action="store_true", help="name the clips and exit")
    args = ap.parse_args()

    if args.list:
        for name, clip in CLIPS.items():
            length = "the whole recording" if clip.seconds is None else f"{clip.seconds:g} s"
            print(f"{name:<8}  {clip.task:<24} {length}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.only or list(CLIPS):
            clip = CLIPS[name]
            physics_hz, control_hz, steps = _rates(clip)
            every = control_hz / FPS
            if abs(every - round(every)) > 1e-6:
                raise SystemExit(f"error: {clip.task} steps at {control_hz:g} Hz, which "
                                 f"{FPS:g} frames a second does not divide")
            mp4 = Path(tmp) / f"{name}.mp4"
            _record(clip, clip.camera, DRAW, FPS, steps, physics_hz, mp4)
            path = OUT / f"{name}.gif"
            _gif(mp4, clip.skip, clip.seconds, path)
            print(f"[media] {path.relative_to(REPO)}  {path.stat().st_size / 1e6:.2f} MB")
            if name == STILL_FROM:
                # Its own recording, at the still's size and camera, up to the
                # frame wanted: the recorder writes the first frame and one at
                # every 1/FPS after it, so the last frame is the one at `at`.
                at = clip.skip + STILL_AT
                frame = round(at * FPS)
                if abs(at * FPS - frame) > 1e-6:
                    raise SystemExit(f"error: the still at {at:g} s is not on a frame at "
                                     f"{FPS:g} a second")
                still_mp4 = Path(tmp) / "still.mp4"
                twice = (2 * STILL[0], 2 * STILL[1])
                _record(clip, STILL_CAMERA, twice, FPS, round(at * control_hz), physics_hz,
                        still_mp4)
                still = OUT / "jumper.png"
                _png(still_mp4, frame, still)
                print(f"[media] {still.relative_to(REPO)}  {still.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
