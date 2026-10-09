#!/usr/bin/env python3
"""Unified play entry point: run a trained policy and watch it.

This script **imports no task's config classes**. It knows only the registry at
the top of tasks/, and configs are loaded by `tasks.load_env_cfg` /
`load_agent_cfg` following the conventions of a task directory. Adding a robot or
a task requires no change under scripts/.

Pick the task with `--task` and the model asset with `--model`; `--list` shows the
choices for both. The backend comes from `--backend` / `--device`, or is
auto-detected, exactly as in train.

The environment is built in **replay mode** (`play=True`): external disturbances
are off and episodes run on rather than timing out (the one-shot jumps keep
theirs, so a replay loops whole attempts), so what you see is the policy rather
than the randomisation. Observation noise is the exception: `--obs-noise`, on by
default, puts training's back, because a clean replay flatters the policy.

A replay is **one environment** unless `--num_envs` or `MJRL_PLAY_NUM_ENVS` says
otherwise. `MJRL_NUM_ENVS` is training's batch size, and play does not read it.

    python scripts/play.py --task jumper.flat --checkpoint <path>/model_4999.pt

Without `--checkpoint` it takes the newest checkpoint under
`logs/<model>/<task>/`, which is usually the run that just finished. `--checkpoint`
also accepts a directory and takes the newest one inside it -- the same resolution
`train.py --resume` uses, in `mjrl.checkpoint`.

`--agent zero|random` needs no checkpoint at all and is for looking at the
environment itself -- a robot that does nothing, or one driven by noise.

`--app` plays an app instead -- the `.app` `scripts/deploy.py` writes, the one
the robot and a browser load -- through the deployment controller:
every mode, switched on the app's own buttons and keys, and the controller's own
joint targets driving the servos. No checkpoint, and the task is the app's
default mode's unless `--task` names another. See `mjrl/app_play.py`.

    python scripts/play.py --app out/bundle_<timestamp>/<name>.app

`--video` records an MP4, with or without a window (a Linux GPU runtime
without a display needs ``MUJOCO_GL=egl`` set before Python starts):

    python scripts/play.py --checkpoint <path>/model_N.pt --headless \
        --steps 500 --video replay.mp4 --video-fps 30

**It is the one MP4 recorder** (`mjrl/viewer/video.py`): the README clips, the
dance export and the Colab notebook all run this command rather than drawing for
themselves. The camera is placed from the robot's first frame (`--video-azimuth`,
`--video-elevation`, `--no-video-follow`), `--stop-on-done` records one attempt
of a one-shot motion rather than looping it, and `--command-script` hands the
command terms a script of time where the README clips need a walk that starts and
a posture that changes -- written where the operator writes, so the pad does not
drive a scripted term.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from _cli import (
    add_scene_args,
    add_viewer_args,
    apply_scene,
    build_parser,
    maybe_viewer,
    parse_with_task_args,
    print_task_table,
    resolve_all,
    resolve_checkpoint,
)


def _set_physics_rate(env_cfg, hz: float) -> None:
    """Raise the physics rate for replay, and hold the control rate where it is.

    **`decimation` has to move with the timestep or this changes the wrong thing.**
    The control period is `timestep * decimation`, and it is the deployment
    contract: 50 Hz is what `layout.json` publishes, what the policy's action rate
    was trained at, and what the robot runs. Shrinking the timestep alone would
    quietly put the policy at 250 Hz -- it would still walk, in a way that means
    nothing about the robot, and only `control_hz` in an export would show it.

    So the control period is read from the task, the timestep is set from `hz`, and
    `decimation` is whatever keeps the product identical. The assertion at the end
    is the one that matters.

    **This is a fidelity trade, not a free improvement.** 1000 Hz is closer to the
    real robot and *further from training*: contact at 200 Hz is the dynamics the
    policy actually learned in, so a gait that only holds at 1000 Hz is a gait the
    trained policy does not have. Use the task's own rate to reproduce training,
    and a higher one to ask what the mechanism does.
    """
    control_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    if hz <= 0:
        raise SystemExit(f"[mjrl] --physics-hz must be positive, got {hz}")

    step = 1.0 / hz
    exact = control_dt / step
    decimation = round(exact)
    if decimation < 1 or abs(exact - decimation) > 1e-9:
        raise SystemExit(
            f"[mjrl] --physics-hz {hz:g} gives {exact:.4f} physics steps per "
            f"control step, which is not a whole number. The control rate is the "
            f"task's {1 / control_dt:g} Hz and must not move, so pick a physics "
            f"rate that is a multiple of it -- "
            f"{', '.join(f'{1 / control_dt * k:g}' for k in (1, 2, 4, 10, 20))}."
        )

    was_hz = 1.0 / env_cfg.sim.mujoco.timestep
    env_cfg.sim.mujoco.timestep = step
    env_cfg.decimation = decimation
    assert env_cfg.sim.mujoco.timestep * env_cfg.decimation == control_dt, (
        "the control period moved, which is the one thing this must not do"
    )
    if was_hz != hz:
        print(f"[mjrl] physics {was_hz:g} -> {hz:g} Hz (decimation {decimation}); "
              f"control stays {1 / control_dt:g} Hz")


def _match_training_noise(env_cfg, task_id, asset) -> None:
    """Give replay the observations training had, corruption and all.

    `play=True` builds a clean world: the task sets
    `observations["actor"].enable_corruption = False`, so **every noise term it
    defines for deployment is inert**. That is the right default for a scripted
    audit and the wrong one for watching a policy, because what it shows is a
    robot reading sensors it will not have. Nothing reports the difference.

    **The levels are not here.** They are the task's, and they are taken by
    building that same task with `play=False` and copying what it produced -- both
    the per-group corruption flags and each term's noise. So there is no second
    copy of a number to drift, `--obs-noise` covers whatever the task defines
    rather than a list kept up to date by hand, and the asymmetry survives: the
    critic group is uncorrupted during training too, and copying reproduces that
    instead of assuming it.

    Reading the flags rather than setting them True also means a task that
    deliberately corrupts something else, or nothing, gets what it asked for.
    """
    import tasks

    train = tasks.load_env_cfg(task_id, asset=asset)
    changed = []
    for name, group in env_cfg.observations.items():
        reference = train.observations.get(name)
        if reference is None:
            continue
        was = group.enable_corruption
        group.enable_corruption = reference.enable_corruption
        for term_name, term in group.terms.items():
            source = reference.terms.get(term_name)
            if source is not None:
                term.noise = source.noise
        if was != group.enable_corruption:
            changed.append(f"{name}: corruption {was} -> {group.enable_corruption}")

    noisy = {
        name: sorted(t for t, term in group.terms.items() if getattr(term, "noise", None))
        for name, group in env_cfg.observations.items()
        if group.enable_corruption
    }
    if not any(noisy.values()):
        print("[mjrl] obs-noise: the task defines no observation noise to apply")
        return
    print(f"[mjrl] obs-noise: training levels, from the task ({'; '.join(changed)})")
    for name, terms in noisy.items():
        if terms:
            print(f"[mjrl]   {name}: {', '.join(terms)}")


def main() -> None:
    # `replay`: `--num_envs` is how many robots to watch, from its own `.env`
    # key, rather than training's batch size. See `build_parser`.
    parser = build_parser("play.py", __doc__ or "", replay=True)
    # Panels on: play is for looking at a policy, which is when they get used. And
    # 60 frames a second, since that is when smoothness is being judged.
    add_viewer_args(parser, ui_default=True, fps_default=60.0)
    add_scene_args(parser)

    v = parser.add_argument_group("video recording")
    v.add_argument("--video", type=Path, default=None, metavar="MP4",
                   help="record an MP4, including with --headless; requires finite --steps")
    v.add_argument("--video-fps", type=float, default=30.0,
                   help="encoded frames per simulation second (default: 30)")
    v.add_argument("--video-width", type=int, default=640)
    v.add_argument("--video-height", type=int, default=480)
    v.add_argument("--video-env", type=int, default=0,
                   help="environment index to record (default: 0)")
    v.add_argument("--video-distance", type=float, default=None,
                   help="optional camera distance in metres")
    v.add_argument("--video-azimuth", type=float, default=None, metavar="DEG",
                   help="camera azimuth in degrees from the way the recorded robot faces in "
                   "the first frame: 0 looks along it from behind, 180 straight at its "
                   "front. Default: the task viewer's, in the world")
    v.add_argument("--video-elevation", type=float, default=None, metavar="DEG",
                   help="camera elevation in degrees, negative looking down "
                   "(default: the task viewer's)")
    v.add_argument("--video-follow", action=argparse.BooleanOptionalAction, default=True,
                   help="follow the robot over the ground (default); --no-video-follow leaves "
                   "the camera where the first frame put it, for a motion that stays put")
    v.add_argument("--video-lookat-height", type=float, default=None, metavar="M",
                   help="height of the point a fixed camera looks at; needs --no-video-follow")

    r = parser.add_argument_group("replay")
    r.add_argument(
        "--checkpoint", default=None,
        help="model_*.pt to load, or a directory whose newest one is taken. "
        "Defaults to the newest one under logs/<model>/<task>/",
    )
    r.add_argument(
        "--agent", choices=["trained", "zero", "random"], default="trained",
        help="'zero' and 'random' need no checkpoint and are for looking at the "
        "environment itself",
    )
    r.add_argument(
        "--steps", type=int, default=None,
        help="stop after this many control steps. Runs until the window is closed "
        "(or forever, when headless) if not given",
    )
    r.add_argument("--checkpoint-command-ranges", action="store_true",
                   help="sample only the checkpoint's saved velocity curriculum rung; "
                   "refuses missing or unsupported command provenance")
    r.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"),
                   help="hold a fixed body-frame velocity command, including after resets; "
                   "must lie inside the checkpoint's trained command ranges")
    r.add_argument("--evaluation-out", type=Path, metavar="JSON",
                   help="write fixed-command velocity errors and reset counts; "
                   "requires --command and finite positive --steps")
    r.add_argument("--replay-info-out", type=Path, metavar="JSON",
                   help="write resolved replay command ranges, source and simulation rates")
    r.add_argument("--command-script", metavar="FILE[:NAME]",
                   help="a Python file whose NAME(t) (default: script) returns {command term: "
                   "values} for t seconds into the replay, written into each term after its "
                   "own compute, where the operator writes; the pad and the keyboard do not "
                   "drive a scripted term. See mjrl.replay.script_commands")
    r.add_argument("--stop-on-done", action="store_true",
                   help="stop at the first episode end rather than replaying the next attempt, "
                   "so a recording of a one-shot motion holds one attempt")
    r.add_argument(
        "--physics-hz", type=float, default=1000.0, metavar="HZ",
        help="physics rate for replay, **independent of the control rate**. The "
        "tasks train at 200 Hz; 1000 resolves contact five times finer, which is "
        "what makes a low-friction scene or a landing worth looking at. The "
        "control rate is the task's and does not move (50 Hz for most tasks, 200 "
        "for some), so the policy and the picture keep it whatever this is. "
        "Must be a whole multiple of the control rate -- under --app, of the "
        "app's fastest mode's. Pass the task's own rate to replay exactly what "
        "training ran",
    )
    r.add_argument(
        "--obs-noise", action=argparse.BooleanOptionalAction, default=True,
        help="feed the policy the same corrupted observations training did. **On "
        "by default**: a clean replay flatters the policy, because every noise "
        "term the task defines for deployment is switched off. `--no-obs-noise` "
        "for the clean signal, which is what to compare against. No effect under "
        "--app, whose controller builds the observation its policies see",
    )
    r.add_argument(
        "--measure", action="store_true",
        help="watch the joints as well as the robot: position, velocity and "
        "actuator torque of environment 0 (not necessarily the one the camera "
        "follows), live. Opens a plot window when a GUI toolkit is installed and "
        "prints a table when it is not; either way it writes measure.csv and "
        "measure.png into measure/ beside the checkpoint when the replay ends -- "
        "or into ./measure/ when there is none (--agent, --app). See "
        "mjrl/viewer/monitor.py",
    )
    r.add_argument(
        "--speed", type=float, default=None, metavar="X",
        help="how fast the world runs, as a multiple of real time. **The default "
        "is 1.0, real time**, so what you watch is what the robot would do; 2 is "
        "twice as fast, 0.25 is slow motion, 0 is as fast as the machine manages. "
        "With --headless nobody is watching, so pacing is off unless this is given",
    )
    r.add_argument(
        "--app", type=Path, default=None, metavar="APP",
        help="play an app: the `.app` `scripts/deploy.py` writes, or the bundle "
        "directory beside it. What runs is the **deployment** controller "
        "-- every mode, entered on the app's own buttons and keys, the viewer's "
        "keys and a gamepad read through each mode's controls, and the "
        "controller's own joint targets and gains driving the servos, so a "
        "task's deploy hook reaches the joints. No checkpoint is loaded. The "
        "environment is the world only, the app's default mode's task unless "
        "--task names another",
    )
    # An app decides the world before anything is parsed: the default mode's
    # task, unless this command line names another. MJRL_TASK in .env is
    # somebody's everyday training choice, not a choice about this app; and the
    # task decides which options of its own the command line takes, so a task
    # swapped in after parsing is handed another task's (`env_cfg() got an
    # unexpected keyword argument 'objects'`, with MJRL_TASK=jumper.five_foot).
    # As a default, so `--task` still wins however argparse lets it be spelt.
    # One environment is `build_parser(replay=True)`'s default, app or not.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--app", type=Path, default=None)
    app_path = pre.parse_known_args()[0].app
    opened = None
    if app_path is not None:
        from mjrl.app_play import AppUnavailable, open_app

        try:
            opened = open_app(app_path)
        except AppUnavailable as error:
            raise SystemExit(f"error: {error}") from None
    try:
        # The selected task may take arguments of its own; see
        # `tasks.load_cli_args`. Nothing here knows which, or how many.
        args = parse_with_task_args(parser, opened.default_task if opened else None)
        _validate_video_args(parser, args)
        _validate_command_args(parser, args)
        if opened is not None and (
            args.checkpoint or args.agent != "trained"
        ):
            parser.error(
                "--app brings its own policies; --checkpoint and --agent would pick "
                "a second one that nothing runs"
            )
    except BaseException:
        if opened is not None:
            opened.close()
        raise
    args.opened_app = opened

    # `_run` closes the app once it has it; until then, every way out does.
    handed = False
    try:
        if args.list:
            print_task_table()
            return

        spec, res, asset = resolve_all(args)
        if args.video is not None and args.video_env >= res.num_envs:
            parser.error("--video-env must be less than --num_envs")

        if args.dry_run:
            print(f"[mjrl] resolved: task={spec.id} -- --dry-run, stopping here")
            return

        handed = True
        _run(spec, res, asset, args)
    finally:
        if not handed and opened is not None:
            opened.close()


def _validate_video_args(parser, args) -> None:
    """Reject unbounded or invalid recordings before allocating a simulation."""
    if args.video is None:
        return
    if args.steps is None or args.steps <= 0:
        parser.error("--video requires a positive, finite --steps")
    if args.video.suffix.lower() != ".mp4":
        parser.error("--video must name an .mp4 file")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--video-fps must be finite and positive")
    if any(n <= 0 or n % 2 for n in (args.video_width, args.video_height)):
        parser.error("video dimensions must be positive even integers")
    if args.video_env < 0:
        parser.error("--video-env must be nonnegative")
    if args.video_distance is not None and (
        not math.isfinite(args.video_distance) or args.video_distance <= 0
    ):
        parser.error("--video-distance must be finite and positive")
    for flag in ("video_azimuth", "video_elevation", "video_lookat_height"):
        value = getattr(args, flag)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{flag.replace('_', '-')} must be finite")
    if args.video_lookat_height is not None and args.video_follow:
        parser.error("--video-lookat-height needs --no-video-follow: a following camera "
                     "looks at the body")


def _validate_command_args(parser, args) -> None:
    requested = args.checkpoint_command_ranges or args.command is not None or args.evaluation_out
    if requested and (args.app is not None or args.agent != "trained"):
        parser.error("fixed commands and checkpoint command ranges require a trained checkpoint")
    if args.command_script is not None:
        if args.app is not None:
            parser.error("--command-script scripts a task's command terms; an app's modes "
                         "take their commands from the app")
        if args.command is not None:
            parser.error("--command and --command-script would both write the command")
        path = args.command_script.rpartition(":")[0] or args.command_script
        if not Path(path).is_file() and not Path(args.command_script).is_file():
            parser.error(f"--command-script: no such file: {args.command_script}")
    if args.command is not None and not all(math.isfinite(x) for x in args.command):
        parser.error("--command must contain three finite values")
    if args.evaluation_out is not None and (
        args.command is None or args.steps is None or args.steps <= 0
    ):
        parser.error("--evaluation-out requires --command and positive finite --steps")


def _run(spec, res, asset: Path | None, args) -> None:
    """Build the environment and run the policy. Heavy imports happen here."""
    import torch
    from mjrl.backend.select import use_backend

    import tasks

    # Order matters: the backend must be registered before the env is built.
    # See the note in scripts/train.py.
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    trained = args.agent == "trained" and args.app is None
    ckpt: Path | None = None
    if trained:
        # Resolution is `mjrl.checkpoint`, shared with `train.py --resume`: which
        # file "the newest checkpoint" means must not depend on which command is
        # asking.
        ckpt = resolve_checkpoint(args.checkpoint, spec.id, asset)
        print(f"[mjrl] checkpoint {ckpt}")

    env_cfg = tasks.load_env_cfg(
        spec.id, asset=asset, play=True, task_args=args.task_args
    )
    command_ranges = None
    if args.checkpoint_command_ranges or args.command is not None:
        from mjrl.replay import configure_velocity_replay

        training_cfg = tasks.load_env_cfg(spec.id, asset=asset, task_args=args.task_args)
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        try:
            command_ranges = configure_velocity_replay(env_cfg, training_cfg, saved, args.command)
        except ValueError as error:
            raise SystemExit(f"[mjrl] replay command: {error}") from None
        del saved
        print(f"[mjrl] replay commands: {command_ranges}")
    _set_physics_rate(env_cfg, args.physics_hz)
    if args.app is not None:
        from mjrl.app_play import AppUnavailable, step_the_world_at

        try:
            step_the_world_at(env_cfg, args.opened_app.control_hz)
        except AppUnavailable as error:
            raise SystemExit(f"error: {error}") from None
        print(f"[play] the world steps at {args.opened_app.control_hz:g} Hz, the app's "
              f"fastest mode (decimation {env_cfg.decimation}); each mode infers at its own")
        from mjrl.app_play import leave_falls_to_the_app

        for name in leave_falls_to_the_app(env_cfg, args.opened_app):
            print(f"[play] the world's `{name}` moved past the app's tilt limit: a fall "
                  "is the controller's to handle first")
    if args.obs_noise:
        _match_training_noise(env_cfg, spec.id, asset)
    apply_scene(env_cfg, args)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    # Nothing in a replay reads a reward, and evaluating them was the largest cost
    # of a replay step. `mjrl/replay.py` has the measurement and why skipping them
    # cannot move what the policy sees.
    from mjrl.replay import skip_rewards

    print(f"[mjrl] replay: {skip_rewards(env)} reward terms are not evaluated "
          f"(nothing in a replay reads them)")
    # **Bound before the `try`, because `finally` reads it.** Everything between
    # here and where the monitor is really built can raise -- loading a
    # checkpoint, opening an app's controller -- and a
    # teardown that depends on how far construction got turns a clear message
    # into `UnboundLocalError: monitor`, printed *instead of* the reason.
    monitor = None
    video = None
    policy = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if args.app is not None:
            # The app brings its own policies, observation and joint targets, so
            # the checkpoint is not loaded at all -- what runs is what would ship.
            from mjrl.app_play import AppPlayer

            pad = None
            try:
                import controller

                pad = controller.open()
            except Exception as error:  # noqa: BLE001 - a missing pad is the keyboard
                print(f"[play] no gamepad ({error}); the viewer's keys drive")
            policy = AppPlayer(args.opened_app, wrapped, pad=pad)
        elif trained:
            from dataclasses import asdict

            runner_cls = tasks.load_runner_cls(spec.id) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped, asdict(agent_cfg), device=res.device)
            runner.load(
                str(ckpt), load_cfg={"actor": True}, strict=True,
                map_location=res.device,
            )
            policy = runner.get_inference_policy(device=res.device)
        else:
            shape = env.action_space.shape
            zero = args.agent == "zero"

            def policy(obs):
                if zero:
                    return torch.zeros(shape, device=env.device)
                return 2 * torch.rand(shape, device=env.device) - 1

        if args.command_script is not None:
            from mjrl.replay import load_command_script, script_commands

            try:
                scripted = script_commands(env, load_command_script(args.command_script))
            except (TypeError, ValueError) as error:
                raise SystemExit(f"[mjrl] command script: {error}") from None
            print(f"[mjrl] scripted commands: {', '.join(scripted)} (the pad does not drive them)")

        # Built before the viewer, so a missing plotting backend is reported
        # while the terminal is still readable rather than behind a window.
        # `out` is the checkpoint's own directory: the measurement describes that
        # policy, and a default that wrote into the working directory would
        # scatter identically-named files wherever play happened to be run from.
        if args.measure:
            from mjrl.viewer.monitor import make_monitor

            out = (ckpt.parent if trained else Path.cwd()) / "measure"
            monitor = make_monitor(*_watched(wrapped), wrapped, out)
            if monitor is None:
                print("[measure] this scene has no entity with joints; nothing to measure")

        with maybe_viewer(env, args) as viewer:
            if args.video is not None:
                from mjrl.viewer.video import ReplayVideo

                video = ReplayVideo(
                    env, args.video, fps=args.video_fps,
                    width=args.video_width, height=args.video_height,
                    env_index=args.video_env, camera_distance=args.video_distance,
                    azimuth=args.video_azimuth, elevation=args.video_elevation,
                    lookat_height=args.video_lookat_height, follow=args.video_follow,
                )
                video.__enter__()
            # Real time when someone is watching. `--headless` is the audit path
            # -- a scripted run measuring something, where pacing would only make
            # it take twenty times longer for no one's benefit -- so it is
            # unpaced unless --speed says otherwise. Passing --speed explicitly
            # always wins, including under --headless.
            speed = args.speed
            if speed is None:
                speed = 0.0 if viewer is None else 1.0
            # Under --app the task's own line would print the world's command
            # beside the speed -- the random sampler's, which nothing drives the
            # robot with -- so only the speed, which is the robot's.
            status = None if args.app is not None else tasks.load_play_status(spec.id)
            from mjrl.replay import FixedCommandEvaluation

            evaluation = FixedCommandEvaluation(wrapped, args.command, command_ranges) if (
                args.evaluation_out is not None
            ) else None
            completed_steps = _loop(wrapped, policy, viewer, args.steps, speed,
                                    status or _speed_text, monitor, video, evaluation,
                                    stop_on_done=args.stop_on_done)
            replay_metadata = {
                "task": spec.id, "checkpoint": str(ckpt.resolve()) if ckpt else None,
                "physics_hz": 1.0 / env_cfg.sim.mujoco.timestep,
                "control_hz": 1.0 / env.step_dt, "control_dt": env.step_dt,
                "observation_noise": args.obs_noise, "control_steps": completed_steps,
                "scene": args.scene or "task default",
                "simulated_seconds": completed_steps * env.step_dt,
            }
            if evaluation is not None:
                evaluation.write(args.evaluation_out, **replay_metadata)
                print(f"[mjrl] fixed-command evaluation {args.evaluation_out.resolve()}")
            if args.replay_info_out is not None:
                import json

                from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

                term = env_cfg.commands.get("twist")
                supported = isinstance(term, UniformVelocityCommandCfg)
                info = command_ranges or {
                    "status": "not_requested", "source": "replay_task_config",
                    "curriculum_level": None,
                    "ranges": {axis: list(getattr(term.ranges, axis)) for axis in (
                        "lin_vel_x", "lin_vel_y", "ang_vel_z"
                    )} if supported else None,
                }
                args.replay_info_out.parent.mkdir(parents=True, exist_ok=True)
                args.replay_info_out.write_text(json.dumps({
                    "schema": "replay_info/1", **replay_metadata,
                    "supports_velocity_command": supported, "command_ranges": info,
                }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                print(f"[mjrl] replay settings {args.replay_info_out.resolve()}")
    finally:
        # Before `env.close()`: the monitor holds no simulation state, but it
        # prints where it wrote things, and a line printed after the environment
        # tears down lands under mjlab's own shutdown output.
        # A failed encoder must still release the simulation and app resources.
        try:
            if video is not None:
                video.close()
                print(f"[mjrl] video {video.path.resolve()} "
                      f"({video.frame_count} frames at {video.fps:g} fps)")
        finally:
            try:
                if monitor is not None:
                    monitor.close()
            finally:
                try:
                    if args.app is not None:
                        if policy is not None and hasattr(policy, "close"):
                            policy.close()
                        args.opened_app.close()
                finally:
                    env.close()


def _watched(env):
    """The entity and environment index whose speed is worth printing.

    The one the camera is following, so the number on the terminal describes the
    robot being looked at rather than an average over a field of them. Returns
    `(entity, index)` or `(None, 0)` when there is nothing recognisable -- the
    readout then falls back to the rates alone, which are always meaningful.
    """
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    if scene is None:
        return None, 0
    entities = getattr(scene, "entities", {}) or {}
    # A scene may carry props (a ball); the robot is the one with actuators.
    for entity in entities.values():
        if getattr(getattr(entity, "data", None), "root_link_lin_vel_b", None) is None:
            continue
        if getattr(entity, "num_actuators", 0) or len(entities) == 1:
            return entity, 0
    return None, 0


def _speed_text(entity, index: int, env) -> str:
    """"speed 0.31 m/s" -- the one quantity every robot has.

    **Deliberately generic.** This used to reach into
    `command_manager.get_command("twist")` to print the commanded speed beside the
    achieved one, which is a task's vocabulary in a file that is not allowed any:
    nothing under `scripts/` changes when a task is added, and
    `tests/test_log_layout.py` says so. It also broke on the first task that was
    not about velocity -- `get_command` returns None rather than raising, so the
    `except` written for exactly that case missed it and replay died on frame one.

    A task that wants more on this line supplies it: see `tasks.load_play_status`.
    """
    del env  # a task that needs the environment gets it through its own readout
    if entity is None:
        return ""
    import torch

    with torch.inference_mode():
        return f"speed {float(torch.norm(entity.data.root_link_lin_vel_b[index, :2])):.2f} m/s"


def _loop(env, policy, viewer, max_steps: int | None, speed: float = 1.0,
          readout=_speed_text, monitor=None, video=None, evaluation=None,
          stop_on_done: bool = False) -> int:
    """Step the environment under the policy until the window closes.

    Returns the number of control steps taken. With `stop_on_done` the loop ends
    at the first episode end -- before the recorder and the monitor see the
    post-reset state the wrapper hands back on that step, which is the next
    attempt and not this one.

    The loop lives here rather than inside the viewer so that headless replay --
    for recording metrics, or on a machine with no display -- runs the same code
    path. `viewer` is None when headless, and the loop then simply runs.

    The rate readout goes to the terminal rather than into the window:
    `mujoco.viewer.launch_passive` has no overlay for a caller to write into, and
    the alternative -- geometry shaped like text in the scene -- would move with
    the camera and be in the way. It prints on one rewritten line so a long replay
    does not scroll.
    """
    import torch
    from mjrl.viewer.stats import Pacer, RunStats

    unwrapped = getattr(env, "unwrapped", env)
    stats = RunStats(unwrapped.step_dt)
    pacer = Pacer(unwrapped.step_dt, speed)
    entity, index = _watched(env)
    if speed > 0:
        print(f"[mjrl] running at {speed:g}x real time (--speed 0 for uncapped)")

    obs = env.get_observations()
    step = 0
    # The readout is one rewritten line with no newline of its own, so anything
    # printed after it would land on the same line. Both exits close it first.
    while True:
        if viewer is not None and not viewer.is_running:
            print("\n[mjrl] the live viewer was closed; stopping")
            break
        if max_steps is not None and step >= max_steps:
            print(f"\n[mjrl] reached --steps {max_steps}; stopping")
            break
        line = stats.update(readout(entity, index, env))
        if line is not None:
            print(f"\r[play] {line}   ", end="", flush=True)
        # The deadline inside the pacer is absolute, so the step's own cost comes
        # out of its slot rather than being added to it -- which is why this can
        # sit on either side of the step without changing the rate, and why a step
        # that overruns does not push the next one late.
        pacer.wait()
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if stop_on_done and bool(dones.any()):
            step += 1
            print(f"\n[mjrl] the episode ended at step {step}; stopping (--stop-on-done)")
            break
        if evaluation is not None:
            evaluation.update(dones)
        # After the step, so the telemetry is the state the picture is showing
        # rather than the one it was showing. The monitor decimates its own
        # redraw; see `mjrl/viewer/monitor.py` for why that matters to the pacer.
        if monitor is not None:
            monitor.update()
        if video is not None:
            video.update()
        step += 1
    return step


if __name__ == "__main__":
    main()
