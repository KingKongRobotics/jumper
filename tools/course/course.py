"""One crab round the obstacle course: the runner.

`tools/race/race.py` plays the app against itself in a room; this is one
`AppPlayer` on a virtual pad in the lane `track.py` builds, with a referee
(`referee.py`) that keeps the clock and calls the finish and a readout
(`hud.py`) that shows what the crab sees, maps and plans. Each crab runs the
course alone -- Andrew cuts the two takes side by side -- so `--crab` picks
which: the colour of the shell, the lane and the finish banner, and the
victory piece at the end (`tools/battle/moves.py`'s MOVES).

    python tools/course/course.py --crab red --scene studio --headless --video out/course/red.mp4
    python tools/course/course.py --crab blue --truth           # the pilot reads the world: the control

A pilot file defines `make_pilot(env, player, side) -> callable`, called once per
control step with the simulation time, returning a `GamepadState` (or None for
hands off). `side` says the crab's colour, the course it was told (`track.layout()`),
the pad presses that enter the app's modes, and carries the referee. `SENSES = True`
on the module keeps the depth sensor in the world.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
import importlib.util as _ilu

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path


@dataclass
class Side:
    """What the pilot is told about its crab."""

    name: str            # "red" or "blue"
    me: str              # the scene's entity name
    referee: Any
    layout: dict         # the course, world frame (`track.layout()`)
    presses: dict = None
    truth: bool = False  # read the world instead of the senses (the control)


def _load_pilot(spec: str):
    file = Path(spec)
    loader = importlib.util.spec_from_file_location(f"_pilot_{file.stem}", file)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def main() -> None:
    from _cli import (
        add_scene_args,
        add_viewer_args,
        apply_scene,
        build_parser,
        maybe_viewer,
        parse_with_task_args,
        resolve_all,
    )
    from mjrl.app_play import AppUnavailable, open_app

    parser = build_parser("tools/course/course.py", __doc__ or "", replay=True)
    add_viewer_args(parser, ui_default=True, fps_default=60.0)
    add_scene_args(parser)
    default_pilot = str(REPO / "tools/course/pilots/runner.py")
    parser.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    parser.add_argument("--crab", default="red", choices=("red", "blue"), help="which crab runs")
    parser.add_argument("--pilot", default=default_pilot, help="the pilot file, or `keys`")
    parser.add_argument("--truth", action="store_true", help="the pilot reads the world, not its senses")
    parser.add_argument("--no-moves", action="store_true", help="no dance at the finish")
    parser.add_argument("--ready", type=float, default=2.0, help="seconds on the mark before the bell")
    parser.add_argument("--timeout", type=float, default=240.0, help="the run is called off after this")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-distance", type=float, default=None)
    parser.add_argument("--video-azimuth", type=float, default=None)
    parser.add_argument("--video-elevation", type=float, default=None)
    parser.add_argument("--cam", action="append", default=[], metavar="NAME:AZ:EL:DIST",
                        help="an extra camera on the same take, written beside --video as <stem>-NAME.mp4")
    parser.add_argument("--no-hud", action="store_true", help="no readout in the video (hud.py)")
    parser.add_argument("--video-lookat", default=os.environ.get("COURSE_CAM_LOOK", "0.30:0:0.18"), metavar="DX:DY:DZ",
                        help="where the main camera looks, from the crab's centre, in the world (m): behind and above it "
                             "puts the crab right of centre and lower, clear of the readout; '0:0:0' for the crab itself")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    pre.add_argument("--task", default=None)
    pre_args = pre.parse_known_args()[0]
    opened = open_app(pre_args.app)
    args = parse_with_task_args(parser, pre_args.task or opened.default_task)
    if not args.task:
        args.task = opened.default_task
    spec, res, asset = resolve_all(args)
    module = None if args.pilot == "keys" else _load_pilot(args.pilot)
    senses = module is not None and getattr(module, "SENSES", False) and not args.truth

    from mjrl.backend.select import use_backend

    use_backend(res)
    import track
    from arena import RED
    from duel import VirtualPad, _AllVideos, skip_the_managers
    from hud import Hud, HudVideo
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjrl.app_play import AppPlayer, leave_falls_to_the_app, step_the_world_at
    from mjrl.replay import skip_rewards
    from moves import pad_presses
    from play import _loop, _speed_text
    from referee import Referee, SegmentDone

    import tasks

    presses = pad_presses(opened)
    how = "the viewer's keys" if module is None else ("truth" if args.truth else "senses")
    print(f"[course] the {args.crab} crab, {how}")

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=args.task_args)
    try:
        step_the_world_at(env_cfg, opened.control_hz)
    except AppUnavailable as error:
        raise SystemExit(f"error: {error}") from None
    leave_falls_to_the_app(env_cfg, opened)
    apply_scene(env_cfg, args)
    layout = track.apply_course(env_cfg, crab=args.crab, senses=senses)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    if os.environ.get("COURSE_SEED"):
        # a seeded take runs the same way again: the only randomness is the environment's
        # (its reset and its sensors' noise, drawn from torch's generator), so with the
        # generators seeded and the rendering outside the physics a good take at 720p can
        # be run again at 4K, the same take frame for frame (Andrew: the upscale "doesn't
        # cut it"; a recording-and-replay of the state was the alternative)
        import random

        import numpy as np
        import torch
        seed = int(os.environ["COURSE_SEED"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = seed
        print(f"[course] seeded {seed}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    skip_rewards(env)
    skip_the_managers(env)
    print(f"[course] world steps at {opened.control_hz:g} Hz, step_dt {env.step_dt:.4f}")
    videos = []
    player = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        azimuth = args.video_azimuth if args.video_azimuth is not None else float(env_cfg.viewer.azimuth)
        referee = Referee(env, layout, crab=args.crab, ready_s=args.ready, moves=not args.no_moves,
                          camera_yaw=math.radians(azimuth + 180.0), timeout_s=args.timeout)
        pad = VirtualPad()
        pad._dt = env.step_dt
        player = AppPlayer(opened, wrapped, pad=pad, entity=RED, read_keys=(module is None), label=args.crab)
        pilot = None
        if module is not None:
            side = Side(args.crab, RED, referee, layout, presses=presses, truth=args.truth)
            pad.pilot = pilot = module.make_pilot(env, player, side)

        class Run:
            """The player stepped with the referee, as the duel steps its two."""

            def __init__(self):
                self._t = 0.0

            def __call__(self, obs):
                busy = bool(pilot.busy(self._t)) if pilot is not None and hasattr(pilot, "busy") else False
                referee.step(self._t, busy)
                self._t += env.step_dt
                return player(obs)

            def __getattr__(self, item):
                return getattr(player, item)

        run = Run()
        with maybe_viewer(env, args) as viewer:
            if args.video is not None:
                from mjrl.viewer.video import ReplayVideo

                hud = Hud(args.crab, pad, player, pilot, referee)
                record = (lambda path, **kw: ReplayVideo(env, path, **kw)) if args.no_hud else \
                    (lambda path, **kw: HudVideo(env, path, hud=hud, **kw))
                look = tuple(float(v) for v in args.video_lookat.split(":"))
                videos.append(record(args.video, fps=args.video_fps, width=args.video_width, height=args.video_height,
                                     env_index=0, camera_distance=args.video_distance, azimuth=args.video_azimuth,
                                     elevation=args.video_elevation, lookat_offset=look if any(look) else None))
                for spec_ in args.cam:
                    cname, az, el, dist = spec_.split(":")
                    videos.append(record(args.video.with_name(f"{args.video.stem}-{cname}{args.video.suffix}"),
                                         fps=args.video_fps, width=args.video_width, height=args.video_height,
                                         env_index=0, camera_distance=float(dist), azimuth=float(az), elevation=float(el)))
                for v in videos:
                    v.__enter__()
            speed = args.speed if args.speed is not None else (0.0 if viewer is None else 1.0)
            try:
                _loop(wrapped, run, viewer, args.steps, speed, _speed_text, None, _AllVideos(videos), None)
            except SegmentDone as e:
                print(f"\n[course] {e}")
    finally:
        for v in videos:
            v.__exit__(None, None, None)
        if player is not None:
            player.close()
        env.close()
        opened.close()


if __name__ == "__main__":
    main()
