"""Two crabs, one room, a line down the middle: the cleaning race.

`tools/battle/duel.py` plays the app against itself on a plank; this is the same
two `AppPlayer`s and virtual pads in the room `room.py` builds, with a referee
(`referee.py`) that counts the objects on each half and a readout (`hud.py`)
that shows what each crab sees, maps and plans.

    python tools/race/race.py --headless --video out/race/take1.mp4
    python tools/race/race.py --red tools/race/pilots/sweeper.py --red-truth

A pilot file defines `make_pilot(env, player, side) -> callable`, called once per
control step with the simulation time, returning a `GamepadState` (or None for
hands off). `side` says which crab it is, which half is its (`sign`: +1 for the
red crab, whose half is x > 0), where it started, the objects' names, the pad
presses that enter the app's modes, and carries the referee. `SENSES = True` on
the module keeps the depth sensors in the world.
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
    """What a pilot is told about its crab."""

    name: str            # "red" or "blue"
    me: str              # the scene's entity name
    foe: str
    referee: Any
    sign: float          # +1: my half is x > 0 (red); -1: x < 0 (blue)
    start: tuple         # (x, y, yaw) where I woke up, world frame
    objects: list        # the layout: name, kind, x, y at the start
    seed: int = 0
    presses: dict = None
    truth: bool = False  # read the world instead of the senses (the probe, the control)


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

    parser = build_parser("tools/race/race.py", __doc__ or "", replay=True)
    add_viewer_args(parser, ui_default=True, fps_default=60.0)
    add_scene_args(parser)
    default_pilot = str(REPO / "tools/race/pilots/sweeper.py")
    parser.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    parser.add_argument("--red", default=default_pilot, help="the red crab's pilot file, or `keys`")
    parser.add_argument("--blue", default=default_pilot, help="the blue crab's pilot file, or `keys`")
    parser.add_argument("--red-truth", action="store_true", help="the red pilot reads the world, not its senses")
    parser.add_argument("--blue-truth", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="the layout and the pilots' rolls; printed")
    parser.add_argument("--objects", type=int, default=None, help="objects on each half (RACE_OBJECTS)")
    parser.add_argument("--no-moves", action="store_true", help="no victory dance")
    parser.add_argument("--ready", type=float, default=2.0, help="seconds on the marks before the start")
    parser.add_argument("--race-s", type=float, default=120.0, help="the clock")
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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    pre.add_argument("--task", default=None)
    pre_args = pre.parse_known_args()[0]
    opened = open_app(pre_args.app)
    args = parse_with_task_args(parser, pre_args.task or opened.default_task)
    if not args.task:
        args.task = opened.default_task
    if args.red == "keys" and args.blue == "keys":
        raise SystemExit("error: the viewer has one keyboard; at most one crab can be `keys`")
    spec, res, asset = resolve_all(args)
    modules = {side: (None if choice == "keys" else _load_pilot(choice))
               for side, choice in (("red", args.red), ("blue", args.blue))}
    pilots = {side: (m.make_pilot if m is not None else None) for side, m in modules.items()}
    truth = {"red": args.red_truth, "blue": args.blue_truth}
    senses = any(getattr(m, "SENSES", False) and not truth[side] for side, m in modules.items() if m is not None)

    from mjrl.backend.select import use_backend

    use_backend(res)
    import room
    from arena import BLUE, RED
    from duel import Duel, VirtualPad, _AllVideos, skip_the_managers, tilt_limit
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
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    n_objects = args.objects if args.objects is not None else room.N_PER_SIDE
    print(f"[race] seed {seed}; red {'truth' if truth['red'] else 'senses'}, blue {'truth' if truth['blue'] else 'senses'}")

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=args.task_args)
    try:
        step_the_world_at(env_cfg, opened.control_hz)
    except AppUnavailable as error:
        raise SystemExit(f"error: {error}") from None
    leave_falls_to_the_app(env_cfg, opened)
    apply_scene(env_cfg, args)
    objects = room.apply_race(env_cfg, seed=seed, senses=senses, n_objects=n_objects)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    skip_rewards(env)
    skip_the_managers(env)
    print(f"[race] world steps at {opened.control_hz:g} Hz, step_dt {env.step_dt:.4f}")
    videos, players = [], []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        azimuth = args.video_azimuth if args.video_azimuth is not None else float(env_cfg.viewer.azimuth)
        referee = Referee(env, objects, ready_s=args.ready, race_s=args.race_s, moves=not args.no_moves,
                          camera_yaw=math.radians(azimuth + 180.0), tilt_limit=tilt_limit(opened))
        keys_to = "red" if args.red == "keys" else ("blue" if args.blue == "keys" else "red")
        made, pads = {}, {}
        starts = {"red": (room.START_X, 0.0, math.pi), "blue": (-room.START_X, 0.0, 0.0)}
        for i, (name, me, foe, sign) in enumerate((("red", RED, BLUE, 1.0), ("blue", BLUE, RED, -1.0))):
            pad = pads[name] = VirtualPad()
            pad._dt = env.step_dt
            player = AppPlayer(opened, wrapped, pad=pad, entity=me, read_keys=(keys_to == name), label=name)
            players.append(player)
            if pilots[name] is not None:
                side = Side(name, me, foe, referee, sign, starts[name], objects, seed=seed + i, presses=presses,
                            truth=truth[name])
                pad.pilot = made[name] = pilots[name](env, player, side)
            else:
                print(f"[race] {name}: the viewer's keys")
        race = Duel(players, referee, env.step_dt, made)
        with maybe_viewer(env, args) as viewer:
            if args.video is not None:
                from mjrl.viewer.video import ReplayVideo

                hud = Hud([(name, pads[name], player, made.get(name)) for name, player in zip(("red", "blue"), players)],
                          referee)
                record = (lambda path, **kw: ReplayVideo(env, path, **kw)) if args.no_hud else \
                    (lambda path, **kw: HudVideo(env, path, hud=hud, **kw))
                videos.append(record(args.video, fps=args.video_fps, width=args.video_width, height=args.video_height,
                                     env_index=0, camera_distance=args.video_distance, azimuth=args.video_azimuth,
                                     elevation=args.video_elevation))
                for spec_ in args.cam:
                    cname, az, el, dist = spec_.split(":")
                    videos.append(record(args.video.with_name(f"{args.video.stem}-{cname}{args.video.suffix}"),
                                         fps=args.video_fps, width=args.video_width, height=args.video_height,
                                         env_index=0, camera_distance=float(dist), azimuth=float(az), elevation=float(el)))
                for v in videos:
                    v.__enter__()
            speed = args.speed if args.speed is not None else (0.0 if viewer is None else 1.0)
            try:
                _loop(wrapped, race, viewer, args.steps, speed, _speed_text, None, _AllVideos(videos), None)
            except SegmentDone as e:
                print(f"\n[race] {e}")
        tally = referee.score()
        print(f"[race] final: red {tally[RED]} : {tally[BLUE]} blue, {tally[None]} on the line; "
              f"{len(referee.crossings)} crossings")
    finally:
        for v in videos:
            v.__exit__(None, None, None)
        for p in players:
            p.close()
        env.close()
        opened.close()


if __name__ == "__main__":
    main()
