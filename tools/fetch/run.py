"""Play an app with a scripted or autonomous operator instead of a human on the pad.

`scripts/play.py --app` reads a real gamepad through `controller.open()`. This runner
builds the same environment and the same `AppPlayer`, and hands it a *virtual* pad: an
object with the gamepad's interface whose `state()` is computed from the simulation
each control step. Everything the controller does with the pad -- mode switches, the
claw's trigger, the arm's d-pad presets -- works unchanged, because the controller
cannot tell the two apart.

    python tools/fetch/run.py --pilot tools/fetch/probe_arm.py --headless --steps 600

The pilot file defines `make_pilot(env, player) -> callable`; the callable is called
once per control step with the simulation time and returns a `GamepadState`.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# play's helpers from scripts/ and the pilots, props and probes beside this file, without a
# sys.path insert (tools/fetch/_beside.py says why)
import importlib.util as _ilu

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path


def _load_pilot(spec: str):
    file = Path(spec)
    loader = importlib.util.spec_from_file_location(f"_pilot_{file.stem}", file)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module.make_pilot


class _AllVideos:
    """Several recorders of one take, stepped as one."""

    def __init__(self, first, extras):
        self._all = [v for v in [first, *extras] if v is not None]

    def update(self):
        for v in self._all:
            v.update()


class VirtualPad:
    """The gamepad's interface, driven by a pilot callable."""

    connected = True

    def __init__(self):
        self.pilot = None
        self.player = None
        self.env = None
        self._t = 0.0
        self._dt = 0.0
        from controller import GamepadState

        self._idle = GamepadState()

    def name(self) -> str:
        return "virtual pad"

    def state(self):
        if self.pilot is None:
            return self._idle
        s = self.pilot(self._t)
        self._t += self._dt
        return s if s is not None else self._idle

    def close(self) -> None:
        pass


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

    parser = build_parser("tools/fetch/run.py", __doc__ or "", replay=True)
    add_viewer_args(parser, ui_default=True, fps_default=60.0)
    add_scene_args(parser)
    parser.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    parser.add_argument("--pilot", required=True, help="python file defining make_pilot(env, player)")
    parser.add_argument("--mug", action="store_true", help="add the mug prop (tools/fetch/mug.py)")
    parser.add_argument("--scenery", action="store_true", help="dunes, palms and rocks around the beach (tools/fetch/scenery.py)")
    parser.add_argument("--room", action="store_true", help="beat two's pillars and tray, props that collide (tools/fetch/room.py)")
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
                        help="an extra camera on the same take, written beside --video as <stem>-NAME.mp4; "
                             "repeatable (e.g. --cam low:150:-8:0.8 --cam claw:60:-20:0.55)")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    pre.add_argument("--task", default=None)
    pre_args = pre.parse_known_args()[0]
    opened = open_app(pre_args.app)
    args = parse_with_task_args(parser, pre_args.task or opened.default_task)
    if not args.task:
        args.task = opened.default_task
    spec, res, asset = resolve_all(args)
    make_pilot = _load_pilot(args.pilot)

    from mjrl.backend.select import use_backend

    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjrl.app_play import AppPlayer, leave_falls_to_the_app, step_the_world_at
    from mjrl.replay import skip_rewards
    from play import _loop, _speed_text

    import tasks

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=args.task_args)
    try:
        step_the_world_at(env_cfg, opened.control_hz)
    except AppUnavailable as error:
        raise SystemExit(f"error: {error}") from None
    for name in leave_falls_to_the_app(env_cfg, opened):
        print(f"[fetch] `{name}` left to the app")
    apply_scene(env_cfg, args)
    if args.mug:
        from mug import apply_mug

        apply_mug(env_cfg)
    if args.scenery:
        from scenery import apply_scenery

        apply_scenery(env_cfg)
    if args.room:
        from room import apply_room

        apply_room(env_cfg)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    skip_rewards(env)
    print(f"[fetch] world steps at {opened.control_hz:g} Hz, step_dt {env.step_dt:.4f}")
    video = None
    extras = []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        pad = VirtualPad()
        pad._dt = env.step_dt
        player = AppPlayer(opened, wrapped, pad=pad)
        pad.env, pad.player = env, player
        pad.pilot = make_pilot(env, player)
        with maybe_viewer(env, args) as viewer:
            if args.video is not None:
                from mjrl.viewer.video import ReplayVideo

                video = ReplayVideo(env, args.video, fps=args.video_fps, width=args.video_width,
                                    height=args.video_height, env_index=0,
                                    camera_distance=args.video_distance, azimuth=args.video_azimuth,
                                    elevation=args.video_elevation)
                video.__enter__()
                for spec_ in args.cam:
                    name, az, el, dist = spec_.split(":")
                    extra = ReplayVideo(env, args.video.with_name(f"{args.video.stem}-{name}{args.video.suffix}"),
                                        fps=args.video_fps, width=args.video_width, height=args.video_height,
                                        env_index=0, camera_distance=float(dist), azimuth=float(az),
                                        elevation=float(el))
                    extra.__enter__()
                    extras.append(extra)
            speed = args.speed if args.speed is not None else (0.0 if viewer is None else 1.0)
            try:
                _loop(wrapped, player, viewer, args.steps, speed, _speed_text, None, _AllVideos(video, extras), None)
            except Exception as e:                       # a pilot's SegmentDone (by name: the pilot is loaded by path)
                if type(e).__name__ != "SegmentDone":
                    raise
                print(f"[fetch] segment done: {e}")
    finally:
        for v in [video, *extras]:
            if v is not None:
                v.__exit__(None, None, None)
        player.close() if "player" in locals() else None
        env.close()
        opened.close()


if __name__ == "__main__":
    main()
