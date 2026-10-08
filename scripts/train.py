#!/usr/bin/env python3
"""Unified train entry point.

This script **imports no task's config classes**. It knows only the registry at
the top of tasks/, and configs are loaded by `tasks.load_env_cfg` /
`load_agent_cfg` following the conventions of a task directory. Adding a robot or
a task requires no change under scripts/.

Pick the task with `--task` and the model asset with `--model`; `--list` shows the
choices for both. The backend comes from `--backend` / `--device`, or is
auto-detected (see mjrl.backend.resolve).

The live viewer is **on by default**, works on both backends, and draws **at most
128 environments** (all of them below 128; `--viewer-env-num` changes it). The two
reasons for the cap are in `_resolve_draw_limit` in mjrl/viewer/live.py. Turn it
off with `--headless`; with no usable display it goes headless automatically.

**TensorBoard is started automatically** on this run's log directory (the whole
task by default, so the previous run is there to compare against) and the URL is
printed. Turn it off with `--no-tensorboard`.

**Training resumes with `--resume`**, from the newest checkpoint under
`logs/<model>/<task>/`, or from the one `--checkpoint` names. What is restored is
the whole training state -- policy, value function, optimiser (so the adaptive
learning-rate schedule keeps its place), the iteration counter and the
environment's step counter, which is what curricula count in -- rather than the
weights alone, so a resumed run continues rather than restarts from a warm
policy.
"""

from __future__ import annotations

from pathlib import Path

from _cli import (
    add_scene_args,
    add_tensorboard_args,
    apply_scene,
    add_viewer_args,
    build_parser,
    parse_with_task_args,
    maybe_tensorboard,
    maybe_viewer,
    print_task_table,
    resolve_all,
    resolve_checkpoint,
)


def main() -> None:
    parser = build_parser("train.py", __doc__ or "")
    add_viewer_args(parser)
    add_scene_args(parser)
    add_tensorboard_args(parser)

    r = parser.add_argument_group("training")
    r.add_argument(
        "--logger",
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="logging backend. **tensorboard rather than mjlab's wandb**: the "
        "start_method='thread' mjlab passes to wandb is rejected by wandb 0.29.0's "
        "config validation and training crashes right after the model is built "
        "(see section 6 of docs/AGENT_SETUP.md)",
    )
    r.add_argument(
        "--max-iterations", type=int, default=None,
        help="override the iteration count in the task's rl_cfg.py. **How many "
        "iterations this run performs**, so on a resumed run it is how many more "
        "to do, on top of the ones the checkpoint already has",
    )
    r.add_argument(
        "--resume",
        action="store_true",
        help="continue training from the newest checkpoint under "
        "logs/<model>/<task>/ -- normally the run that just finished. The whole "
        "training state is restored, not the weights alone. There is deliberately "
        "no .env key for this: a sticky default would silently continue an old "
        "run on a launch meant to start a new one",
    )
    r.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="which checkpoint to resume from: a model_*.pt, or a directory whose "
        "newest one is taken (a run directory, say). Implies --resume",
    )
    # The selected task may take arguments of its own; see
    # `tasks.load_cli_args`. Nothing here knows which, or how many.
    args = parse_with_task_args(parser)

    if args.list:
        print_task_table()
        return

    spec, res, asset = resolve_all(args, for_training=True)

    if args.dry_run:
        print(f"[mjrl] resolved: task={spec.id} -- --dry-run, stopping here")
        return

    _run(spec, res, asset, args)


def _run(spec, res, asset: Path | None, args) -> None:
    """Build the environment and train. Every heavy import happens here."""
    import tasks
    from mjrl.backend.select import use_backend

    # ── Order matters: the backend must be registered before the env is built ──
    # `ManagerBasedRlEnv.__init__` consults the registry once, so changing it
    # after the env exists has no effect.
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, task_args=args.task_args)
    apply_scene(env_cfg, args, for_training=True)
    env_cfg.scene.num_envs = res.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    runner_cls = tasks.load_runner_cls(spec.id) or MjlabOnPolicyRunner

    from tasks.paths import log_root_for

    # Resolved before the directory is created, and long before the scene is
    # built: a typo in --checkpoint should cost a second rather than the minute
    # it takes to compile the models, and it should not leave an empty run
    # directory behind in logs/.
    ckpt = (
        resolve_checkpoint(args.checkpoint, spec.id, asset)
        if args.resume or args.checkpoint
        else None
    )
    if ckpt is not None:
        print(f"[mjrl] resuming from {ckpt}")

    log_dir = Path(log_root_for(asset)) / spec.id / _timestamp()
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mjrl] log directory {log_dir}")

    # The runner takes a **dict**, not a dataclass (`if key in train_cfg` in
    # mjlab/rl/runner.py requires something iterable), and it **mutates** that
    # config in place -- so build a fresh one every time. Passing the dataclass
    # directly gives
    # `TypeError: argument of type 'RslRlOnPolicyRunnerCfg' is not iterable`.
    from dataclasses import asdict

    # mjlab defaults the logger to wandb, and that path is broken on wandb 0.29.0.
    # Override it explicitly rather than letting people hit it and go read docs.
    agent_cfg.logger = args.logger
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    max_iterations = agent_cfg.max_iterations
    runner_kwargs = asdict(agent_cfg)

    # TensorBoard starts here rather than after the environment is built: the
    # event files do not exist yet, but building the scene and compiling the
    # models takes tens of seconds, and that is exactly the window in which
    # someone wants the page already open and waiting.
    with maybe_tensorboard(log_dir, args):
        env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
        try:
            wrapped = RslRlVecEnvWrapper(env)
            # The policy may learn on another torch device than the environment
            # produces tensors on: `res.agent_device` is MPS on Apple Silicon when
            # the simulation is on Metal, and `res.device` everywhere else. The
            # runner copies observations across and hands actions back.
            runner = runner_cls(wrapped, runner_kwargs, str(log_dir), res.agent_device)
            runner.add_git_repo_to_log(__file__)
            if ckpt is not None:
                _resume(runner, ckpt, res.agent_device, max_iterations)
            with maybe_viewer(env, args):
                runner.learn(
                    num_learning_iterations=max_iterations,
                    init_at_random_ep_len=True,
                )
        finally:
            env.close()


def _resume(runner, ckpt: Path, device: str, more_iterations: int) -> None:
    """Load `ckpt` into `runner` and report where training is picking up.

    A **new** timestamped run directory is used rather than the checkpoint's own,
    so a resumed run never overwrites the checkpoints it started from. The curves
    still join up: the iteration counter continues from the checkpoint, so with
    the default `--tb-scope task` TensorBoard draws the new run as a series that
    begins where the old one ended.

    The failure this catches is the one people actually hit: resuming after the
    observation terms or the network shape changed. `load_state_dict` then raises
    a wall of missing and mismatched keys, which reads as a bug in the framework
    rather than as "this checkpoint belongs to a different configuration".
    """
    try:
        runner.load(str(ckpt), map_location=device)
    except (RuntimeError, KeyError) as e:
        raise SystemExit(
            f"[mjrl] cannot resume from {ckpt}:\n"
            f"  {type(e).__name__}: {e}\n"
            f"The usual cause is a checkpoint from a different configuration: it "
            f"has to come from the same task, model and network as this run, "
            f"observation and action dimensions included. Train from scratch (drop "
            f"--resume), or point --checkpoint at a matching run."
        ) from None

    start = runner.current_learning_iteration
    print(
        f"[mjrl] resumed at iteration {start}; running {more_iterations} more "
        f"(through {start + more_iterations - 1})"
    )


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    main()
