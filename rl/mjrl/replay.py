"""Work a replay leaves out that training needs.

`scripts/play.py` builds the task with `play=True`, which is the task's own idea of
a replay: no observation noise, no pushes, long episodes. This module is the
framework's half -- things the environment does on every step for the learner that
nothing in a replay reads.

## The reward terms

**Every step evaluates every reward term**, and a replay throws the result away:
`play.py`'s loop discards what `env.step` returns but the observation, and neither
the live viewer, `--measure` nor a task's status line reads a reward. On
jumper.posture with one environment on warp (RTX 5090 D, 2026-09-26) the reward
manager was **4.7 of 11.3 ms** a control step, the largest single cost of a replay
step, and skipping it took the headless rate from 92 to 121 steps a second. That
is the difference between a 200 Hz task watched at 0.46x and at 0.61x of real time.

**What makes skipping safe is not that nothing reads the rewards; it is that the
policy's observation does not depend on a reward term having run.** A term can
leave state behind -- `soft_touchdown` banks a foot velocity, jumper.jump caches
its reference on the step counter, jumper.posture's gait clock advances lazily for
whoever asks first. If a reward term is the first to advance something an
observation reads, skipping it can move the observation by a step, and nothing
would say so. For every task registered today it does not. The jump cache is
filled first by its termination, which runs before the rewards. The posture clock
*is* advanced first by a reward term -- the actor, the critic and the rewards read
one clock -- but it gives the same number whoever asks: it integrates the tempo the
observation recorded, and a reset brings it up to date before zeroing (see
`CadenceClock` in `tasks/jumper/posture/mdp/cadence.py`). `tests/test_replay.py`
checks this per task, observation by observation, so a task that changes it fails
there rather than in someone's replay; made to integrate the live command at
whoever asks first, the posture clock fails it at step 0.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from functools import partial
from pathlib import Path

__all__ = [
    "FixedCommandEvaluation",
    "configure_velocity_replay",
    "load_command_script",
    "script_commands",
    "skip_rewards",
]


def load_command_script(spec: str):
    """`FILE[:NAME]` -> the callable `NAME` (default `script`) defined in `FILE`.

    The file is loaded by path, not imported by name: a clip's script lives next
    to whatever draws the clip (`tools/readme_media.py`), which is not a package.
    """
    path, sep, name = spec.rpartition(":")
    if not sep or not name.isidentifier():
        path, name = spec, "script"
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"no command script at {file}")
    loader = importlib.util.spec_from_file_location(f"_mjrl_command_script_{file.stem}", file)
    if loader is None or loader.loader is None:
        raise ValueError(f"{file} is not a Python file")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    script = getattr(module, name, None)
    if not callable(script):
        raise TypeError(f"{file} defines no callable {name!r}")
    return script


def script_commands(env, script) -> list[str]:
    """Have command terms hold what `script` says, from each term's own `compute` on.

    `script(t)` returns `{term name: values}` for `t` seconds since this was
    installed; it is written into each named term **after** the term's own
    `compute`, which is where the operator writes -- so a pad that is plugged in
    and nudged does not drive a scripted term, and the term's own bookkeeping
    (metrics, resample timers, the posture term's walking band) still runs.

    Two kinds of term are understood, the two the operator drives: a velocity
    command (`vel_command_b`, three values) and jumper.posture's posture command
    (`posture_command`, four). A posture's height may be NaN, the term's own
    neutral height, or `(height, share)`, that share of the way there from it.
    """
    import torch

    step_dt = env.step_dt
    origin = env.common_step_counter
    first = script(0.0)
    if not isinstance(first, dict) or not first:
        raise TypeError("a command script returns {term name: values}, and this one returned "
                        f"{type(first).__name__}")
    names = list(first)
    for name in names:
        try:
            term = env.command_manager.get_term(name)
        except KeyError:
            raise ValueError(f"the task has no command term {name!r} to script") from None
        if not hasattr(term, "posture_command") and not hasattr(term, "vel_command_b"):
            raise ValueError(f"command term {name!r} is neither a velocity nor a posture "
                             "command; a script cannot drive it")
        _install(term, name, script, env, origin, step_dt, torch)
    return names


def _install(term, name, script, env, origin, step_dt, torch) -> None:
    compute = term.compute

    def scripted(dt, env_ids=None):
        compute(dt, env_ids)
        t = (env.common_step_counter - origin) * step_dt
        values = list(script(t)[name])
        if hasattr(term, "posture_command"):
            rest = term.cfg.neutral_height
            height = values[3]
            if isinstance(height, tuple):
                values[3] = rest + (height[0] - rest) * height[1]
            elif math.isnan(height):
                values[3] = rest
            term.posture_command[:] = torch.tensor(
                values, device=term.device, dtype=torch.float32
            )
            term.hold_to_band()
            return
        columns = term.command.shape[1]
        if len(values) != columns:
            raise ValueError(f"a script of {len(values)} values for {name!r}, a command of "
                             f"{columns}")
        term.vel_command_b[:] = torch.tensor(values, device=term.device, dtype=torch.float32)
        if hasattr(term, "vel_command_w"):
            term.vel_command_w[:] = term.vel_command_b

    term.compute = scripted


def configure_velocity_replay(env_cfg, training_cfg, checkpoint: dict,
                              command=None) -> dict:
    """Use the saved command rung, without running or restoring a curriculum.

    This is opt-in for ordinary replay. Fixed-command evaluation always uses it:
    the task's final command ceiling is not evidence of a checkpoint's training
    range. Old checkpoints missing a required rung are deliberately refused.
    """
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

    trained = training_cfg.commands.get("twist")
    if not isinstance(trained, UniformVelocityCommandCfg) or "twist" not in env_cfg.commands:
        reason = "the task has no supported uniform 'twist' velocity command"
        if command is not None:
            raise ValueError(reason + "; fixed-command evaluation is unsupported")
        return {"status": "not_applicable", "source": None, "curriculum_level": None,
                "ranges": None, "resolved_ranges": None, "reason": reason}
    ranges = copy.deepcopy(trained.ranges)
    ladder = training_cfg.curriculum.get("command")
    level = None
    source = "training_task_config"
    if ladder is not None:
        params = ladder.params or {}
        levels, angles = params.get("levels"), params.get("ang_levels")
        if not levels or not angles or len(levels) != len(angles):
            raise ValueError("unsupported command curriculum: explicit levels/ang_levels required")
        infos = checkpoint.get("infos") or {}
        curriculum = infos.get("curriculum") or {}
        saved = curriculum.get("command") or {}
        level = saved.get("level")
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < len(levels):
            raise ValueError("checkpoint has no valid saved command curriculum level")
        lin, ang = float(levels[level]), float(angles[level])
        if not math.isfinite(lin) or not math.isfinite(ang) or min(lin, ang) <= 0:
            raise ValueError("saved command curriculum ranges must be finite and positive")
        ranges.lin_vel_x = ranges.lin_vel_y = (-lin, lin)
        ranges.ang_vel_z = (-ang, ang)
        source = "checkpoint_curriculum"
    axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    bounds = {axis: list(getattr(ranges, axis)) for axis in axes}
    if any(len(pair) != 2 or not all(math.isfinite(float(x)) for x in pair)
           or pair[0] > pair[1] for pair in bounds.values()):
        raise ValueError("training velocity ranges must be finite ordered pairs")
    report = {"status": "matched", "source": source, "curriculum_level": level,
              "ranges": bounds, "resolved_ranges": copy.deepcopy(bounds)}
    if command is None:
        env_cfg.commands["twist"].ranges = ranges
        # A hard-coded forward-only floor can exceed an early curriculum rung.
        env_cfg.commands["twist"].rel_forward_envs = 0.0
        return report
    values = tuple(float(x) for x in command)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("fixed body command must contain three finite values")
    for axis, value in zip(axes, values, strict=True):
        low, high = bounds[axis]
        if value < low or value > high:
            raise ValueError(f"fixed {axis}={value:g} is outside trained range [{low:g}, {high:g}]")
    # Use a replay-only sampler, rather than an operator subclass that a pad or
    # viewer could override. Zero random fractions alone are insufficient: the
    # standard sampler tests random <= fraction, so an exact zero can trigger.
    fixed = copy.deepcopy(trained)
    fixed.ranges = ranges
    for axis, value in zip(axes, values, strict=True):
        setattr(fixed.ranges, axis, (value, value))
    fixed.heading_command = False
    fixed.ranges.heading = None
    fixed.rel_heading_envs = fixed.rel_world_envs = fixed.rel_forward_envs = 0.0
    fixed.rel_standing_envs = 0.0  # (0, 0, 0) itself still requests standing.
    fixed.init_velocity_prob = 0.0
    fixed.build = partial(_build_fixed_velocity_command, fixed, values)
    env_cfg.commands["twist"] = fixed
    report["fixed_body_command"] = list(values)
    report["resolved_ranges"] = {axis: list(getattr(fixed.ranges, axis)) for axis in axes}
    return report


def _build_fixed_velocity_command(cfg, values, env):
    """Keep standard command metrics/timers/reset, replacing only its producer."""
    from mjlab.managers.command_manager import CommandTerm
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    class FixedBodyVelocityCommand(UniformVelocityCommand):
        def _resample_command(self, env_ids):
            self._update_command(env_ids)

        def _update_command(self, env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            self.vel_command_b[ids] = self.vel_command_b.new_tensor(values)
            self.vel_command_w[ids] = self.vel_command_b[ids]
            for flag in (self.is_heading_env, self.is_standing_env,
                         self.is_world_env, self.is_forward_env):
                flag[ids] = False

        def compute(self, dt, env_ids=None):
            # UniformVelocityCommand.compute applies GUI sliders afterward.
            # Fixed evaluation uses the base lifecycle without that override.
            CommandTerm.compute(self, dt, env_ids)

    return FixedBodyVelocityCommand(cfg, env)


class FixedCommandEvaluation:
    """Streaming body-frame tracking measurements, excluding auto-reset states.

    The wrapper returns a post-reset pose on done steps. Counting that velocity
    as the terminal gait would bias the result, so those samples are excluded and
    counted explicitly. This reports tracking, not a policy acceptance score.
    """

    def __init__(self, env, command, command_ranges: dict):
        self.env = getattr(env, "unwrapped", env)
        self.command = tuple(float(x) for x in command)
        term = self.env.command_manager.get_term("twist")
        self.robot = term.robot
        self.ranges = command_ranges
        self.steps = self.samples = self.excluded = 0
        self.resets = [0] * self.env.num_envs
        self.actual_sum = [0.0] * 3
        self.absolute_sum = [0.0] * 3
        self.square_sum = [0.0] * 3
        self.xy_error_sum = 0.0

    def update(self, dones) -> None:
        lin = self.robot.data.root_link_lin_vel_b.detach().cpu().tolist()
        ang = self.robot.data.root_link_ang_vel_b.detach().cpu().tolist()
        reset = dones.detach().cpu().tolist()
        self.steps += 1
        for index, (linear, angular, done) in enumerate(zip(lin, ang, reset, strict=True)):
            if done:
                self.resets[index] += 1
                self.excluded += 1
                continue
            actual = (linear[0], linear[1], angular[2])
            if not all(math.isfinite(value) for value in actual):
                raise ValueError("non-finite body velocity during fixed-command evaluation")
            self.samples += 1
            errors = [value - target for value, target in zip(actual, self.command, strict=True)]
            for axis in range(3):
                self.actual_sum[axis] += actual[axis]
                self.absolute_sum[axis] += abs(errors[axis])
                self.square_sum[axis] += errors[axis] ** 2
            self.xy_error_sum += math.hypot(errors[0], errors[1])

    def report(self) -> dict:
        count = self.samples
        return {
            "schema": "fixed_body_command_evaluation/1", "frame": "body",
            "axes": ["vx", "vy", "yaw_rate"], "units": ["m/s", "m/s", "rad/s"],
            "command": list(self.command), "command_ranges": self.ranges,
            "control_steps": self.steps, "num_envs": self.env.num_envs,
            "simulated_seconds": self.steps * self.env.step_dt,
            "valid_samples": count, "excluded_reset_samples": self.excluded,
            "episode_resets": sum(self.resets), "episode_resets_per_env": self.resets,
            "sampling": "post-step body velocities; auto-reset states excluded; no warmup exclusion",
            "actual_velocity_mean": [x / count for x in self.actual_sum] if count else None,
            "mean_absolute_error": [x / count for x in self.absolute_sum] if count else None,
            "root_mean_square_error": [math.sqrt(x / count) for x in self.square_sum]
            if count else None,
            "mean_xy_tracking_error": self.xy_error_sum / count if count else None,
        }

    def write(self, path: Path, **metadata) -> dict:
        result = {**self.report(), **metadata}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result


def skip_rewards(env) -> int:
    """Stop `env` evaluating its reward terms. Returns how many it had.

    `compute` is replaced on this **instance** only -- the same way the live viewer
    wraps `sim.step` -- and hands back a buffer of zeros, so `env.step` still
    returns a reward of the right shape. The rest of the manager is untouched:
    `reset` still runs, and still resets the terms that carry state, which is what
    keeps them coherent if anything later evaluates them again.
    """
    import torch

    manager = env.reward_manager
    zeros = torch.zeros(env.num_envs, device=env.device)

    def compute(dt: float) -> torch.Tensor:
        del dt  # nothing to scale
        return zeros

    manager.compute = compute
    return len(manager.active_terms)
