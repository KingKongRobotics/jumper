"""Two crabs, two controllers, one plank: play the app against itself.

`tools/fetch/run.py` plays the app with one scripted operator on a virtual pad.
This builds the same world with a second robot in it (`arena.py`), runs **two**
`AppPlayer`s from the same app -- one per crab, each with its own controller,
its own policies and its own pad -- and a referee (`referee.py`) that scores
the rounds and resets the world between them.

    python tools/battle/duel.py --scene beach --headless --video out/battle/take1.mp4
    python tools/battle/duel.py --red keys --blue tools/battle/pilots/charge.py

A pilot file defines `make_pilot(env, player, side) -> callable`; the callable
is called once per control step with the simulation time and returns a
`GamepadState` (or None for hands off). `side` says which crab it is (`me`,
`foe`, the scene's names), its aggression and its seed, the pad presses that
enter the app's modes, and carries the referee, whose `live` says the round is
on and whose `cue` asks for a set piece. A pilot with a `busy(t)` tells the
referee its crab is still performing. `keys` instead of a file hands that crab
to the viewer's keyboard -- one crab at most, since the viewer has one
keyboard.

    python tools/battle/duel.py --red-aggression 0.9 --blue-aggression 0.2 --seed 7
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

    name: str       # "red" or "blue"
    me: str         # the scene's entity name
    foe: str
    referee: Any
    aggression: float = 0.5
    seed: int = 0
    presses: dict = None
    style: str = "bull"      # a `brain.py` playbook: bull, matador, wrestler, boxer


def tilt_limit(app) -> float:
    """The app's own tilt limit, radians: past it the controller drops to `safe`."""
    import tomllib

    return float(tomllib.loads((app.root / "controller.toml").read_text("utf-8"))["fsm"]["tilt_limit"])


def _load_pilot(spec: str):
    """The pilot module. `make_pilot` is what it has to define; `SENSES = True`
    says it looks through the crab's depth sensor, which is then built."""
    file = Path(spec)
    loader = importlib.util.spec_from_file_location(f"_pilot_{file.stem}", file)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


class VirtualPad:
    """The gamepad's interface, driven by a pilot callable (or idle, for the keys)."""

    connected = True

    def __init__(self):
        self.pilot = None
        self.last = None
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
        self.last = s if s is not None else self._idle      # (what the readout shows, `hud.py`)
        return self.last

    def close(self) -> None:
        pass


class _AllVideos:
    def __init__(self, videos):
        self._all = [v for v in videos if v is not None]

    def update(self):
        for v in self._all:
            v.update()


def skip_the_managers(env) -> None:
    """Stop the world computing what nobody reads.

    `play --app` builds the environment for its physics; its observation and its
    commands are the task's and the app never sees them, as `skip_rewards` says of
    the rewards. Measured on the duel (cProfile, 800 steps at 200 Hz, warp, RTX
    3060, 2026-10-14): of 14.5 s in `env.step`, 3.9 s was the posture task's
    operator command term and 2.5 s the observation groups -- a third of every
    take, spent on tensors that were thrown away. The command manager computes
    nothing; the observation manager hands back the one dictionary it computed
    at the start, which keeps `env.step`'s return the shape the wrapper expects.
    """
    env.command_manager.compute = lambda *a, **k: None
    first = env.observation_manager.compute()
    env.observation_manager.compute = lambda *a, **k: first


class Duel:
    """`play`'s policy: both controllers, then the referee."""

    def __init__(self, players, referee, dt: float, pilots=None):
        self.players, self.referee, self.dt = players, referee, dt
        self.pilots = pilots or {}
        self.t = 0.0

    def __call__(self, obs):
        zeros = None
        for player in self.players:
            zeros = player(obs)
        busy = {name: bool(getattr(pilot, "busy", lambda t: False)(self.t)) for name, pilot in self.pilots.items()}
        self.referee.step(self.t, busy)
        self.t += self.dt
        return zeros


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

    parser = build_parser("tools/battle/duel.py", __doc__ or "", replay=True)
    add_viewer_args(parser, ui_default=True, fps_default=60.0)
    add_scene_args(parser)
    default_pilot = str(REPO / "tools/battle/pilots/brain.py")
    parser.add_argument("--app", type=Path, default=REPO / "out/bundle_example/jumper.app")
    parser.add_argument("--red", default=default_pilot, help="the red crab's pilot file, or `keys`")
    parser.add_argument("--blue", default=default_pilot, help="the blue crab's pilot file, or `keys`")
    parser.add_argument("--red-style", default="bull", help="the brain's playbook: bull, matador, wrestler or boxer")
    parser.add_argument("--blue-style", default="matador")
    parser.add_argument("--red-aggression", type=float, default=None, help="0 defends, 1 charges in (pilots that roll); "
                        "the style's unless given")
    parser.add_argument("--blue-aggression", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="the pilots' rolls; printed, and a match replays from it")
    parser.add_argument("--no-moves", action="store_true", help="no taunt, cheer or victory dance (plumbing takes)")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--linger", type=float, default=3.0, help="seconds a fall plays out (and the winner walks back to the middle) before the cheer")
    parser.add_argument("--ready", type=float, default=2.0, help="seconds on the marks before each round")
    parser.add_argument("--round-s", type=float, default=45.0, help="a round nobody loses in this long is a draw")
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
    parser.add_argument("--bare", action="store_true", help="the table and the crabs only: no dressing round the "
                        "arena, no dunes or palms (arena.dress, tools/fetch/scenery.py)")
    parser.add_argument("--no-hud", action="store_true", help="no control readout in the video's corners (hud.py)")
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
    senses = any(getattr(m, "SENSES", False) for m in modules.values() if m is not None)

    from mjrl.backend.select import use_backend

    use_backend(res)
    from arena import BLUE, RED, apply_arena
    from hud import Hud, HudVideo
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjrl.app_play import AppPlayer, leave_falls_to_the_app, step_the_world_at
    from mjrl.replay import skip_rewards
    from moves import check_moves, pad_presses
    from play import _loop, _speed_text
    from referee import Referee, SegmentDone

    import tasks

    presses = pad_presses(opened)
    if not args.no_moves:
        check_moves(presses)
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(f"[duel] seed {seed}; red {args.red_style}" + (f" at {args.red_aggression:g}" if args.red_aggression is not None else "")
          + f", blue {args.blue_style}" + (f" at {args.blue_aggression:g}" if args.blue_aggression is not None else ""))

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=args.task_args)
    try:
        step_the_world_at(env_cfg, opened.control_hz)
    except AppUnavailable as error:
        raise SystemExit(f"error: {error}") from None
    leave_falls_to_the_app(env_cfg, opened)
    apply_scene(env_cfg, args)
    apply_arena(env_cfg, senses=senses, dressed=not args.bare)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    skip_rewards(env)
    skip_the_managers(env)
    print(f"[duel] world steps at {opened.control_hz:g} Hz, step_dt {env.step_dt:.4f}")
    videos = []
    players = []
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        # The heading that faces the recorder: MuJoCo's free camera sits `azimuth` degrees round
        # from +x on the far side of what it looks at (azimuth 90, the default, is on -y looking
        # along +y), so a crab facing it has the azimuth's opposite heading.
        azimuth = args.video_azimuth if args.video_azimuth is not None else float(env_cfg.viewer.azimuth)
        referee = Referee(env, tilt_limit=tilt_limit(opened), best_of=args.best_of, linger_s=args.linger,
                          ready_s=args.ready, round_s=args.round_s, moves=not args.no_moves,
                          camera_yaw=math.radians(azimuth + 180.0))
        keys_to = "red" if args.red == "keys" else ("blue" if args.blue == "keys" else "red")
        made, pads = {}, {}
        for i, (name, me, foe) in enumerate((("red", RED, BLUE), ("blue", BLUE, RED))):
            pad = pads[name] = VirtualPad()
            pad._dt = env.step_dt
            player = AppPlayer(opened, wrapped, pad=pad, entity=me, read_keys=(keys_to == name), label=name)
            players.append(player)
            if pilots[name] is not None:
                side = Side(name, me, foe, referee, aggression=getattr(args, f"{name}_aggression"),
                            seed=seed + i, presses=presses, style=getattr(args, f"{name}_style"))
                pad.pilot = made[name] = pilots[name](env, player, side)
            else:
                print(f"[duel] {name}: the viewer's keys")
        duel = Duel(players, referee, env.step_dt, made)
        with maybe_viewer(env, args) as viewer:
            if args.video is not None:
                from mjrl.viewer.video import ReplayVideo

                hud = Hud([(name, getattr(args, f"{name}_style"), pads[name], player, made.get(name))
                           for name, player in zip(("red", "blue"), players)], referee)
                record = (lambda path, **kw: ReplayVideo(env, path, **kw)) if args.no_hud else \
                    (lambda path, **kw: HudVideo(env, path, hud=hud, **kw))
                videos.append(record(args.video, fps=args.video_fps, width=args.video_width,
                                     height=args.video_height, env_index=0,
                                     camera_distance=args.video_distance, azimuth=args.video_azimuth,
                                     elevation=args.video_elevation))
                for spec_ in args.cam:
                    name, az, el, dist = spec_.split(":")
                    videos.append(record(args.video.with_name(f"{args.video.stem}-{name}{args.video.suffix}"),
                                         fps=args.video_fps, width=args.video_width, height=args.video_height,
                                         env_index=0, camera_distance=float(dist), azimuth=float(az),
                                         elevation=float(el)))
                for v in videos:
                    v.__enter__()
            speed = args.speed if args.speed is not None else (0.0 if viewer is None else 1.0)
            try:
                _loop(wrapped, duel, viewer, args.steps, speed, _speed_text, None, _AllVideos(videos), None)
            except SegmentDone as e:
                print(f"\n[duel] {e}")
        print("[duel] rounds: " + "; ".join(f"{r}: {w} ({why})" for r, w, why in referee.history))
    finally:
        for v in videos:
            v.__exit__(None, None, None)
        for p in players:
            p.close()
        env.close()
        opened.close()


if __name__ == "__main__":
    main()
