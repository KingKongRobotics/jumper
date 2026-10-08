"""A replay does not evaluate the reward terms, and the policy must not be able to tell.

`mjrl/replay.py` takes the reward manager out of a replay step. Nothing in a replay
reads a reward, and evaluating them was the largest single cost of the step -- 4.7
of 11.3 ms on jumper.posture with one environment on warp.

**What could go wrong is silent.** A reward term that is the first thing in a step
to advance some state an observation reads -- a clock advanced lazily for whoever
asks first, a cache keyed on the step counter -- would, once skipped, leave that
advance to the observation. The observation runs after the reset and after the
command update, so the policy would be shown a phase one step off after every
reset, or a reference one step late, and every replay would look fine. Both kinds
of state exist in this repository today: jumper.posture's gait clock and
jumper.jump's reference cache.

So this runs every registered task twice from the same seed, with and without the
rewards, through time-outs and command resamples, and compares what the actor is
shown bit for bit. jumper.posture's actor and rewards do share one gait clock, and
a reward term is the first to advance it; it passes here because the clock
integrates the tempo the observation recorded and counts a reset as asking
(`CadenceClock` in `tasks/jumper/posture/mdp/cadence.py`). Made to integrate the
command live at whoever asks first, it fails at step 0; made to skip that catch-up
at a reset, at step 19, the first time-out.
"""

from __future__ import annotations

import pytest

import tasks

try:
    import torch
except ImportError:
    torch = None

#: Steps per rollout, an episode length and a command resampling interval, all in
#: control steps. The episode is short so that the run crosses time-out resets --
#: where a lazily advanced clock goes wrong -- and the resampling so that it
#: crosses command changes, where one integrating a command-dependent rate does.
STEPS = 48
EPISODE_STEPS = 20
RESAMPLE_STEPS = 7

#: **Two environments, so a replay is compared batched, the way training runs** --
#: rollout on two threads, every environment stepped in per-thread scratch and put
#: back into its own MjData -- not the one-environment special case of one thread
#: and one history. It ran one until native was deterministic across environments:
#: each one's warm start and control leaked through rollout's per-thread scratch,
#: two identical unskipped runs of jumper.posture diverged at step 2 and
#: jumper.swing at step 5, and this test failed for a reason that had nothing to do
#: with rewards. `tests/test_native_determinism.py` holds that now.
#:
#: Both environments still time out on the same step, in every task, so a reset
#: never lands on one alone.
NUM_ENVS = 2


class _Counted:
    """A reward term that counts its calls and is otherwise the term.

    Attribute access falls through, so a class-based term keeps its `reset` --
    which the manager calls through `term_cfg.func` on every reset.
    """

    def __init__(self, func) -> None:
        self._func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._func, name)


def _clone(obs):
    if isinstance(obs, dict):
        return {k: _clone(v) for k, v in obs.items()}
    return obs.clone()


def _equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def _rollout(task_id: str, skip: bool):
    """The actor's observation after every step, and how many reward terms ran.

    **The training config, not `play=True`.** Replay opens a connected gamepad, and
    a pad touched between the two rollouts would make them differ for a reason that
    has nothing to do with rewards. What is under test -- the order the managers
    run in and whatever state a reward term leaves behind -- is the same in both.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.replay import skip_rewards

    cfg = tasks.load_env_cfg(task_id)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = EPISODE_STEPS * step_dt
    for term in cfg.commands.values():
        if hasattr(term, "resampling_time_range"):
            term.resampling_time_range = (RESAMPLE_STEPS * step_dt,) * 2

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        counted = []
        for term_cfg in env.reward_manager._term_cfgs:
            term_cfg.func = _Counted(term_cfg.func)
            counted.append(term_cfg.func)
        if skip:
            skip_rewards(env)

        actions = torch.Generator().manual_seed(1)
        seen, resets = [], 0
        for _ in range(STEPS):
            action = torch.rand(env.action_space.shape, generator=actions) - 0.5
            obs, _, terminated, truncated, _ = env.step(action)
            assert "actor" in obs, f"{task_id} has no 'actor' observation group"
            seen.append(_clone(obs["actor"]))
            resets += int((terminated | truncated).sum())
        return seen, sum(c.calls for c in counted), resets
    finally:
        env.close()


@pytest.fixture
def native_backend():
    """Build on native cpu, and put back whatever backend was registered before."""
    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=NUM_ENVS))
    try:
        yield
    finally:
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


@pytest.mark.parametrize("task_id", tasks.list_ids())
@pytest.mark.skipif(torch is None, reason="simulation parity requires torch")
def test_skipping_rewards_does_not_change_what_the_policy_sees(
    task_id: str, native_backend
) -> None:
    trained, ran, resets = _rollout(task_id, skip=False)
    replayed, ran_skipped, resets_skipped = _rollout(task_id, skip=True)

    # The controls: the counting works, and the run crossed the resets it exists
    # to cross. Without the first, "no reward term ran" below would pass vacuously.
    assert ran > 0, f"{task_id}: no reward term was ever evaluated, even unskipped"
    assert resets > 0, f"{task_id}: the rollout never reset; shorten EPISODE_STEPS"

    assert ran_skipped == 0, f"{task_id}: {ran_skipped} reward term calls in a replay"
    assert resets_skipped == resets, (
        f"{task_id}: {resets} resets with rewards, {resets_skipped} without"
    )
    for step, (a, b) in enumerate(zip(trained, replayed, strict=True)):
        if _equal(a, b):
            continue
        # Only now is it worth a third rollout: to tell a reward term that matters
        # from a simulation that is not deterministic, which would make any
        # difference here meaningless -- see `NUM_ENVS`.
        again, _, _ = _rollout(task_id, skip=False)
        if not all(_equal(x, y) for x, y in zip(trained, again, strict=True)):
            pytest.fail(
                f"{task_id}: two identical rollouts with the rewards on already "
                f"differ, so the difference at step {step} says nothing about "
                f"skipping them. See NUM_ENVS."
            )
        pytest.fail(
            f"{task_id}: the actor's observation differs at step {step} once rewards "
            f"are skipped -- a reward term is the first to advance something the "
            f"observation reads. See mjrl/replay.py."
        )


@pytest.fixture
def velocity_replay_configs(monkeypatch):
    """Use the real command schema with torch, a schema-shaped double otherwise."""
    import copy
    import sys
    from types import SimpleNamespace

    if torch is None:
        class UniformVelocityCommandCfg(SimpleNamespace):
            pass

        UniformVelocityCommandCfg.Ranges = SimpleNamespace
        monkeypatch.setitem(sys.modules, "mjlab.tasks.velocity.mdp.velocity_command",
                            SimpleNamespace(UniformVelocityCommandCfg=UniformVelocityCommandCfg))
    else:
        from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

    command = UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(.02, .02), heading_command=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-.5, .5), lin_vel_y=(-.5, .5), ang_vel_z=(-.75, .75),
            heading=(-3.14, 3.14),
        ),
    )
    training = SimpleNamespace(commands={"twist": command}, curriculum={
        "command": SimpleNamespace(params={
            "levels": (.25, .35, .5, .5), "ang_levels": (.3, .5, .75, .75),
        }),
    })
    replay = SimpleNamespace(commands={"twist": copy.deepcopy(command)})
    checkpoint = {"infos": {"curriculum": {"command": {"level": 0}}}}
    return training, replay, checkpoint


def test_velocity_replay_matches_saved_rung_instead_of_final_task_ceiling(velocity_replay_configs):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    assert replay.commands["twist"].ranges.lin_vel_x == (-.5, .5)  # control
    info = configure_velocity_replay(replay, training, checkpoint)
    assert info["status"] == "matched" and info["curriculum_level"] == 0
    assert info["ranges"]["lin_vel_x"] == [-.25, .25]
    assert info["ranges"]["ang_vel_z"] == [-.3, .3]
    assert replay.commands["twist"].ranges.lin_vel_x == (-.25, .25)
    assert training.commands["twist"].ranges.lin_vel_x == (-.5, .5)


@pytest.mark.parametrize("level", [None, -1, 4, True, 0.5])
def test_velocity_replay_refuses_missing_or_invalid_curriculum_provenance(
    velocity_replay_configs, level,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    checkpoint["infos"]["curriculum"]["command"]["level"] = level
    with pytest.raises(ValueError, match="saved command curriculum level"):
        configure_velocity_replay(replay, training, checkpoint)


def test_velocity_replay_distinguishes_no_curriculum_from_unsupported_task(
    velocity_replay_configs,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    training.curriculum = {}
    assert configure_velocity_replay(replay, training, {})["source"] == "training_task_config"
    training.commands = {}
    info = configure_velocity_replay(replay, training, checkpoint)
    assert info["status"] == "not_applicable"
    with pytest.raises(ValueError, match="unsupported"):
        configure_velocity_replay(replay, training, checkpoint, (0, 0, 0))


@pytest.mark.parametrize("command", [(.3, 0, 0), (0, -.4, 0), (0, 0, .5)])
def test_fixed_velocity_replay_refuses_commands_above_saved_rung(
    velocity_replay_configs, command,
):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    with pytest.raises(ValueError, match="outside trained range"):
        configure_velocity_replay(replay, training, checkpoint, command)


@pytest.mark.parametrize("command", [(.2, 0, 0), (0, .2, 0), (0, 0, .2), (0, 0, 0)])
def test_fixed_velocity_replay_disables_sampling_overrides(velocity_replay_configs, command):
    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    info = configure_velocity_replay(replay, training, checkpoint, command)
    cfg = replay.commands["twist"]
    assert info["fixed_body_command"] == list(command)
    assert cfg.ranges.lin_vel_x == (command[0], command[0])
    assert cfg.ranges.lin_vel_y == (command[1], command[1])
    assert cfg.ranges.ang_vel_z == (command[2], command[2])
    assert not cfg.heading_command and cfg.ranges.heading is None
    assert cfg.rel_heading_envs == cfg.rel_world_envs == cfg.rel_standing_envs == 0
    assert cfg.rel_forward_envs == cfg.init_velocity_prob == 0


@pytest.mark.skipif(torch is None, reason="the real command sampler requires torch")
def test_fixed_velocity_replay_survives_every_real_resample_and_partial_reset(
    velocity_replay_configs,
):
    from types import SimpleNamespace

    from mjrl.replay import configure_velocity_replay

    training, replay, checkpoint = velocity_replay_configs
    robot = SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(2, 3), root_link_ang_vel_b=torch.zeros(2, 3),
        heading_w=torch.zeros(2),
    ))
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, scene={"robot": robot})
    ids = torch.arange(2)
    sampled = training.commands["twist"].build(env)
    sampled.reset(ids)
    control = sampled.command.clone()
    sampled.reset(ids)
    assert not torch.equal(control, sampled.command)  # Random commands really can change.
    configure_velocity_replay(replay, training, checkpoint, (.2, -.1, .2))
    fixed = replay.commands["twist"].build(env)
    expected = torch.tensor([[.2, -.1, .2]] * 2)
    fixed.reset(ids)
    for step in range(12):
        fixed.is_standing_env[:] = True  # State overrides must be cleared too.
        fixed.is_world_env[:] = True
        fixed._joystick_enabled = SimpleNamespace(value=True)
        fixed._joystick_sliders = [SimpleNamespace(value=99.0)] * 3
        fixed._joystick_get_env_idx = lambda: 0
        fixed.compute(.02)
        if step % 3 == 0:
            fixed.reset(torch.tensor([step % 2]))
            fixed.compute(0)
        assert torch.equal(fixed.command, expected)


def test_fixed_velocity_evaluation_excludes_auto_reset_poses_and_counts_resets(tmp_path):
    import json
    from types import SimpleNamespace

    from mjrl.replay import FixedCommandEvaluation

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    data = SimpleNamespace(root_link_lin_vel_b=Tensor([[.1, 0, 0], [99, 99, 0]]),
                           root_link_ang_vel_b=Tensor([[0, 0, .1], [0, 0, 99]]))
    env = SimpleNamespace(num_envs=2, step_dt=.02, command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(robot=SimpleNamespace(data=data)),
    ))
    measured = FixedCommandEvaluation(env, (.2, 0, .2), {"source": "checkpoint_curriculum"})
    measured.update(Tensor([0, 1]))
    result = measured.write(tmp_path / "evaluation.json")
    assert result["episode_resets"] == result["excluded_reset_samples"] == 1
    assert result["valid_samples"] == 1
    assert result["actual_velocity_mean"] == [.1, 0, .1]
    assert result["mean_absolute_error"] == pytest.approx([.1, 0, .1])
    assert result["root_mean_square_error"] == pytest.approx([.1, 0, .1])
    assert result["simulated_seconds"] == .02
    assert json.loads((tmp_path / "evaluation.json").read_text())["frame"] == "body"


@pytest.fixture
def velocity_play_entrypoint(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("_cli", scripts / "_cli.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_cli", cli)
    spec.loader.exec_module(cli)
    spec = importlib.util.spec_from_file_location("_velocity_play_tests", scripts / "play.py")
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    return play


@pytest.mark.parametrize("arguments,message", [
    (["--command", "nan", "0", "0"], "three finite values"),
    (["--command", "0", "inf", "0"], "three finite values"),
    (["--evaluation-out", "metrics.json"], "positive finite --steps"),
    (["--command", "0", "0", "0", "--evaluation-out", "metrics.json", "--steps", "0"],
     "positive finite --steps"),
    (["--command", "0", "0", "0", "--agent", "zero"], "trained checkpoint"),
    (["--checkpoint-command-ranges", "--agent", "random"], "trained checkpoint"),
])
def test_velocity_play_rejects_invalid_evaluations_before_allocating(
    velocity_play_entrypoint, monkeypatch, capsys, arguments, message,
):
    import sys

    def should_not_resolve(args):
        pytest.fail("Invalid evaluation reached environment resolution")

    monkeypatch.setattr(velocity_play_entrypoint, "resolve_all", should_not_resolve)
    monkeypatch.setattr(sys, "argv", ["play.py", *arguments])
    with pytest.raises(SystemExit) as error:
        velocity_play_entrypoint.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [
    [],
    ["--replay-info-out", "info.json"],
    ["--checkpoint-command-ranges", "--command", ".2", "0", "0", "--steps", "50",
     "--evaluation-out", "metrics.json"],
])
def test_velocity_play_preserves_default_and_accepts_opt_in_cli(
    velocity_play_entrypoint, monkeypatch, capsys, arguments,
):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(velocity_play_entrypoint, "resolve_all", lambda args: (
        SimpleNamespace(id="robot.motion"), SimpleNamespace(num_envs=1), None,
    ))
    monkeypatch.setattr(sys, "argv", ["play.py", "--dry-run", *arguments])
    velocity_play_entrypoint.main()
    assert "--dry-run, stopping here" in capsys.readouterr().out
