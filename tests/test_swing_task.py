"""jumper.swing -- the task that stands a robot on a swing and asks it to pump.

The failures pinned here are the ones that produce a training run rather than an
error:

- a reward term that measures the robot against **gravity** on a floor that tilts,
  so the policy is paid to lean out of the deck at exactly the moments the swing
  needs it most;
- a net reward rate that is negative, so falling off is the cheapest thing
  available and the policy learns to end the episode;
- an initial pose off the ropes' constraint manifold, which unloads through the
  solver and reads as damping;
- a prop that only compiles on one of the two backends.

Every one of them was hit while writing the task. Three of them ran.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest


def _cfg(**overrides):
    """The task's config, built once per test that needs it."""
    import tasks

    cfg = tasks.load_env_cfg("jumper.swing")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ── The two terms that would have been wrong rather than absent ─────────


def test_nothing_scores_the_robot_against_gravity() -> None:
    """The deck tilts with the arc, so gravity is the wrong reference.

    `mdp.upright` and `mdp.bad_orientation` are what the four velocity tasks use,
    they are already in the config this task starts from, and both are named for
    what you want. A swing at 20 degrees carries its deck at 20 degrees, so a robot
    standing perfectly reads as 20 degrees of tilt: `upright` pays it to lean out
    of the deck, and `fell_over` spends half of its 50 degree budget on the task
    succeeding. Neither raises, both produce a number every step.

    `upright` **was still in the config** after the module docstring saying it had
    been removed was written, which is how this class of bug travels.
    """
    cfg = _cfg()
    assert "upright" not in cfg.rewards, (
        "the gravity-relative upright reward is back; deck_upright replaces it"
    )
    assert "fell_over" not in cfg.terminations, (
        "bad_orientation measures against gravity; tipped_on_the_deck replaces it"
    )
    assert "deck_upright" in cfg.rewards
    assert "tipped_on_the_deck" in cfg.terminations

    # The control that these are not merely renamed: the replacements must not be
    # reading the robot's own projected gravity either.
    from tasks.jumper.swing.mdp import rewards, terminations

    for func in (rewards.deck_upright, terminations.tipped_on_the_deck):
        source = func.__doc__ or ""
        assert "deck" in source.lower(), f"{func.__name__} does not say what it measures"


def test_no_term_scores_the_robot_in_a_frame_it_has_left() -> None:
    """Three terms have had this fault, and they all failed the same way.

    `upright`, `fell_over` and `body_tilt_rate` are each named for something the
    robot does and each measured against the **world**: the first two against
    gravity, the third against the world's angular frame. On a deck that tilts and
    turns with the arc, all three read a robot doing exactly the right thing as a
    robot in trouble, and none of them raises.

    `body_tilt_rate` is the one worth spelling out because re-weighting it was
    tried first and is not a fix. Measured on a robot doing nothing at all, its
    world-frame `wx^2 + wy^2` runs 0.757 at rest and **2.111** on a 35 degree
    swing; the charge above its deadband goes 0.627 to 1.699, so the robot pays 2.7
    times more for succeeding than for sitting still. A smaller weight charges less
    for succeeding -- it does not stop charging. Relative to the deck the same
    quantity runs 0.465 to 0.604, and the deck-relative term charges 0.481 to
    0.583.
    """
    cfg = _cfg()
    for gone, instead in (
        ("upright", "deck_upright"),
        ("body_tilt_rate", "body_tilt_rate_on_the_deck"),
    ):
        assert gone not in cfg.rewards, (
            f"{gone!r} is back; it measures against the world and this deck moves. "
            f"{instead!r} is the deck-relative replacement"
        )
        assert instead in cfg.rewards, f"{instead!r} is missing"
    assert "fell_over" not in cfg.terminations
    assert "tipped_on_the_deck" in cfg.terminations

    # The control that these are not the same function renamed: each replacement
    # has to go through the one helper that removes the deck's own motion.
    import inspect

    from tasks.jumper.swing.mdp import rewards, state

    for func in (rewards.deck_upright, rewards.body_tilt_rate_on_the_deck):
        source = inspect.getsource(func)
        assert any(
            name in source for name in ("deck_up_in_body", "body_rate_on_deck")
        ), f"{func.__name__} does not take the deck's own motion out"
    assert hasattr(state, "body_rate_on_deck")


def test_no_term_still_reads_the_twist_command() -> None:
    """There is no command, so a term that gates on one would raise at the first
    step -- which is a loud failure, unlike the two above, but it is easy to
    reintroduce by copying a reward across from a velocity task."""
    cfg = _cfg()
    assert not cfg.commands, f"a command survived: {sorted(cfg.commands)}"
    assert not cfg.curriculum, "both curricula take command_name='twist'"
    for group in cfg.observations.values():
        assert "command" not in group.terms
    for name, term in cfg.rewards.items():
        assert "command_name" not in (term.params or {}), (
            f"reward {name!r} still gates on a command this task does not have"
        )


# ── The reward has to pay for staying on ────────────────────────────────


def test_standing_on_the_plank_is_worth_more_than_falling_off() -> None:
    """A negative reward rate pays the policy to end the episode.

    There is no alive bonus in this task and there does not need to be --
    `deck_upright` and `centred_on_the_plank` are both near 1 for a robot simply
    standing on the plank, so together they are one. But they have to outweigh the
    penalties, and at the first weights tried they did not: measured over 30
    iterations, mean episode length was **10 steps**, 0.2 s, ending 66%
    `off_the_plank`; `action_rate_l2` was -0.031 per episode against +0.011 for the
    objective. The policy was not failing to learn to swing, it was learning to
    stop.

    So this steps a real environment with zero actions -- a robot standing still on
    a plank, doing nothing at all, which is the worst case for the positive terms
    since the swing pays almost nothing -- and requires the rate to be positive.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    res = resolve(backend="native", device="cpu", num_envs=2)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = res.device
    # The swing at rest: nothing to collect from the objective, which is what
    # makes this the worst case rather than a comfortable one.
    cfg.events["reset_on_the_swing"].params["angle_range"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        env.reset()
        zero = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        total = 0.0
        steps = 100
        for _ in range(steps):
            _, reward, _, _, _ = env.step(zero)
            total += float(reward.mean())
        rate = total / steps
    finally:
        env.close()

    assert rate > 0.0, (
        f"a robot standing still on the plank earns {rate:+.4f} per step. Negative "
        f"means the shortest episode is the best one, and the policy will find that "
        f"before it finds the swing"
    )


# ── The reset has to land on the ropes ──────────────────────────────────


@pytest.mark.parametrize("degrees", [0.0, 20.0, 45.0])
@pytest.mark.parametrize("hang", [0.60, 1.35, 1.80])
def test_the_reset_pose_does_not_stretch_the_ropes(degrees: float, hang: float) -> None:
    """Releasing the seat is a rotation about the beam, at whatever length it is.

    Six one-sided rope constraints have a configuration manifold, and a state off
    it has every rope over-stretched. Measured while calibrating the swing:
    displacing the seat along `x` alone cost 21% of the amplitude in the first
    half-swing as the constraints unloaded, which reads as heavy damping rather
    than as a bad initial condition.

    **Once the length is drawn per episode there is a second way onto the wrong
    manifold, and it is quieter than the first**: compute the pose with the
    nominal length while the ropes have been re-rigged to another one. At the ends
    of `HANG_L_RANGE` that is 0.65 m of over-stretch, and what it looks like is a
    swing that was shoved. Both mistakes are the control groups below, and they
    are the whole test -- "the ropes are at their length" is something a rig at
    rest passes for free.
    """
    import mujoco

    from scenes import swing as scene
    from tasks.jumper.swing.mdp import state

    theta = math.radians(degrees)
    model = scene._swing_spec().compile()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    # Rig it the way `reset_on_the_swing` does: all four ropes, each the
    # hypotenuse of its own triangle rather than the hang itself.
    for rope in scene.ROPES:
        rope_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, rope)
        model.tendon_range[rope_id, 1] = scene.rope_for(hang)
    drop = scene.seat_drop(hang)

    def stretch(slide_x: float, slide_z: float, pitch: float) -> np.ndarray:
        data = mujoco.MjData(model)
        for name, value in (
            ("seat_slide_x", slide_x), ("seat_slide_z", slide_z), ("seat_pitch", pitch)
        ):
            data.qpos[model.jnt_qposadr[names.index(name)]] = value
        mujoco.mj_forward(model, data)
        return np.asarray(data.ten_length) - np.asarray(model.tendon_range[:, 1])

    # What `reset_on_the_swing` writes: a rigid rotation about the beam, at the
    # length this episode drew.
    rotated = stretch(-drop * math.sin(theta), state.REST_L - drop * math.cos(theta),
                      theta)
    assert np.abs(rotated).max() < 1e-6, (
        f"at {degrees} degrees on a {hang} m rope the reset pose is off the ropes "
        f"by {np.abs(rotated).max() * 1000:.2f} mm"
    )

    # Control one: the length is ignored and the nominal is used instead.
    if abs(hang - scene.HANG_L) > 1e-9:
        nominal = stretch(-state.REST_L * math.sin(theta),
                          state.REST_L * (1.0 - math.cos(theta)), theta)
        assert np.abs(nominal).max() > 1e-3, (
            f"placing a {hang} m swing with the nominal {scene.HANG_L} m left the "
            f"ropes within {np.abs(nominal).max() * 1000:.2f} mm of their length, "
            f"so this test cannot tell the two apart"
        )

    # Control two: the seat is slid sideways instead of rotated. Slid from the
    # bottom of *its own* arc, not from the nominal height -- a long rope hangs
    # below the nominal, so starting there leaves every rope slack and the mistake
    # this is meant to represent cannot show up at all.
    if degrees > 0.0:
        slid = stretch(-drop * math.sin(theta), state.REST_L - drop, 0.0)
        assert slid.max() > 1e-3, (
            f"the control group -- sliding the seat sideways without rotating it "
            f"-- only stretched the ropes by {slid.max() * 1000:.2f} mm, so this "
            f"test would pass on a reset that does not place the seat on its arc"
        )


# ── Every environment has to get its own swing ──────────────────────────


def test_every_environment_gets_a_different_rope() -> None:
    """The batch has to contain different swings, and each seat has to hang on its.

    **This is the failure the whole change can produce in silence.** `tendon_range`
    is a per-world model field only once some event declares it through
    `requires_model_fields`; without that declaration mjlab hands out a *broadcast
    view* with a leading dimension of one, and on the warp backend writing to it
    succeeds and sets every world to the last value written. Nothing raises. What
    comes out is a training run on one swing, with a config that says otherwise,
    a `--swing-length 0.7,1.8` that appears to work, and a policy that generalises
    to nothing.

    So this asserts three separate things, because each of them can be true while
    the others are not:

    - the lengths **differ** across the batch (the broadcast failure kills this);
    - each seat actually **hangs at** the length its own environment was given,
      which is what says the model write reached the physics rather than only the
      buffer;
    - `rope_length` reports that same number, since every reward and observation
      that normalises reads it rather than the seat.

    Run on native, which raises on an in-place write to an unexpanded field rather
    than broadcasting it. That makes native the safe backend for this and warp the
    one that needs the test -- the assertions are written to fail on either.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    from tasks.jumper.swing.mdp import state

    n = 24
    res = resolve(backend="native", device="cpu", num_envs=n)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = n
    cfg.sim.device = res.device
    # At rest, so the seat's distance from the beam is the rope's length and not a
    # point on an arc.
    cfg.events["reset_on_the_swing"].params["angle_range"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        env.reset()
        zero = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        for _ in range(20):  # let the ropes take up their load
            env.step(zero)
        reported = state.rope_length(env).clone()
        hung = -state.swing_offset(env)[:, 2].clone()
    finally:
        env.close()

    low, high = cfg.events["reset_on_the_swing"].params["length_range"]
    assert float(reported.max() - reported.min()) > 0.5 * (high - low), (
        f"{n} environments drew lengths spanning only "
        f"{float(reported.max() - reported.min()):.3f} m out of {high - low:.2f}. "
        f"One swing shared across the batch is exactly what an unexpanded "
        f"tendon_range looks like"
    )
    assert type(reported) is torch.Tensor, (
        f"rope_length returned {type(reported).__name__}, a backend view class. "
        f"Arithmetic keeps it and the first thing to fail is an f-string in a "
        f"play readout, nowhere near here"
    )
    # The ropes are soft limits, so they stretch a few millimetres under the
    # robot's weight -- always in the same direction, never by a centimetre.
    error = hung - reported
    assert float(error.min()) > -1e-3 and float(error.max()) < 0.02, (
        f"seats hang between {float(error.min()) * 1000:+.1f} and "
        f"{float(error.max()) * 1000:+.1f} mm from the length their environment "
        f"was rigged at, so the model write and the physics disagree"
    )


def test_the_swing_observations_are_normalised_per_environment() -> None:
    """A batch of different swings must look like one shape to the critic.

    `swing_offset` divides by the rope's length so that hanging still is
    `(0, 0, -1)` whatever the swing is. Divide by the **nominal** instead and the
    term reports -0.52 on a short rope and -1.33 on a long one: the length is then
    back in the observation, in the most learnable form available, and the
    normalisation is doing the opposite of its job.

    Checked as a spread rather than a value, so it fails on the nominal divisor
    and passes only when the divisor is the one that environment was rigged at.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    from tasks.jumper.swing.mdp import observations, state

    n = 16
    res = resolve(backend="native", device="cpu", num_envs=n)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = n
    cfg.sim.device = res.device
    cfg.events["reset_on_the_swing"].params["angle_range"] = (0.0, 0.0)
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        env.reset()
        zero = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        for _ in range(20):
            env.step(zero)
        normalised = observations.swing_offset(env)[:, 2].clone()
        length = state.rope_length(env).clone()
        raw = state.swing_offset(env)[:, 2].clone()
    finally:
        env.close()

    assert float(length.max() - length.min()) > 0.3, "the lengths did not vary"
    assert abs(float(normalised.mean()) + 1.0) < 0.03, (
        f"a seat hanging still reads {float(normalised.mean()):.3f} rather than -1"
    )
    spread = float(normalised.max() - normalised.min())
    assert spread < 0.03, (
        f"the normalised offset spans {spread:.3f} across the batch, so the length "
        f"is still in it"
    )
    # The control: the same quantity against the nominal divisor, which is the
    # mistake. It has to be visibly worse, or this test proves nothing.
    nominal = float((raw / state.REST_L).max() - (raw / state.REST_L).min())
    assert nominal > 10 * spread, (
        f"dividing by the nominal length spans only {nominal:.3f}, so this test "
        f"would pass on the wrong divisor too"
    )


def test_the_objective_can_tell_swinging_from_swaying() -> None:
    """A sideways sway must be worth nothing, and a swing must cost nothing.

    The two are the same shape -- the seat hangs from two points 0.5 m apart on a
    1.95 m beam, so along the beam is a pendulum of nearly the same length as along
    the swing, and therefore of nearly the same period. Nothing tells them apart
    except which plane they are in.

    Written against the full 3-vector, which is how `swing_amplitude` started, the
    measure scores them **identically**: the reward is then indifferent between the
    thing the task is named after and a mode that is easier to excite and useless,
    and a policy optimising an indifferent reward takes whichever is cheaper. This
    is the test that would have caught it, and it is built as two pure modes
    precisely because a mixed one passes either way.
    """
    import torch

    from tasks.jumper.swing.mdp.state import _plane_amplitude

    rel_still = torch.tensor([[0.0, 0.0, -1.36]])
    vel_still = torch.zeros((1, 3))

    # A pure fore-aft swing, at the bottom of its arc and moving along x.
    fore_aft = _plane_amplitude(rel_still, torch.tensor([[1.0, 0.0, 0.0]]), 0)
    fore_aft_sway = _plane_amplitude(rel_still, torch.tensor([[1.0, 0.0, 0.0]]), 1)
    # 1 m/s through the bottom of a 1.36 m pendulum is 0.275 rad, 15.7 degrees.
    assert float(fore_aft) > 0.2, "a swing along x does not register as swinging"
    assert float(fore_aft_sway) == pytest.approx(0.0, abs=1e-6), (
        f"a pure fore-aft swing reads as {float(fore_aft_sway):.4f} rad of sway, so "
        f"the penalty charges the robot for doing the task"
    )

    # A pure sideways sway, same speed, along the beam instead.
    sway = _plane_amplitude(rel_still, torch.tensor([[0.0, 1.0, 0.0]]), 1)
    sway_as_swing = _plane_amplitude(rel_still, torch.tensor([[0.0, 1.0, 0.0]]), 0)
    assert float(sway) == pytest.approx(float(fore_aft), abs=1e-6), (
        "the two modes are the same size; only the plane differs"
    )
    assert float(sway_as_swing) == pytest.approx(0.0, abs=1e-6), (
        f"a pure sideways sway is worth {float(sway_as_swing):.4f} rad of swing, so "
        f"the objective pays for the wrong mode"
    )

    # Displacement, not just velocity: the seat parked out to one side.
    out_sideways = torch.tensor([[0.0, 0.3, -1.33]])
    assert float(_plane_amplitude(out_sideways, vel_still, 0)) == pytest.approx(
        0.0, abs=1e-6
    ), "a seat parked out sideways still reads as a swing"
    assert float(_plane_amplitude(out_sideways, vel_still, 1)) > 0.2


def test_pump_power_pays_for_pumping_and_charges_for_damping() -> None:
    """Leaning the way the swing is going must score positive; against, negative.

    This is the only term in the task that scores the *act* rather than the state,
    and it exists because at a dead stop every other term is flat -- nothing tells a
    policy which way to move on the first step of the first episode. A shaping term
    that got the sign wrong would be worse than none: it would teach the robot to
    damp the swing, densely and from step one.

    Checked as arithmetic on the two quantities it multiplies, because the thing it
    is really about -- a hexapod shearing its stance fore and aft with its feet
    planted -- has no closed form in joint space, and the scripted riders written to
    drive it end-to-end could not produce a lean at all. That end of it is not
    verified here and training is what will show it.
    """
    import torch

    from tasks.jumper.swing.mdp import state

    def power(offset: float, rate: float) -> float:
        """The term is `(deck offset - standing offset) * rope rate`."""
        return (offset - state.DECK_OFFSET[0]) * rate

    stand = state.DECK_OFFSET[0]
    lean = stand + 0.05  # 50 mm forward of where the reset puts it

    assert power(lean, +1.0) > 0.0, "leaning with the swing must pay"
    assert power(lean, -1.0) < 0.0, "the same lean against the swing must cost"
    assert power(stand, 1.0) == pytest.approx(0.0), (
        "a robot that has not moved from where it was placed must score zero, "
        "however fast the swing is going"
    )
    # Symmetric: leaning back while swinging back is pumping too.
    assert power(stand - 0.05, -1.0) == pytest.approx(power(lean, 1.0))

    # The other half: `swing_rate` must be signed by which way the rope is going,
    # and must read the fore-aft plane only. Sideways motion is a different mode
    # and `sideways_sway` is what is against it -- if it leaked in here, a robot
    # could farm the shaping term by rocking the seat along the beam.
    rel = torch.tensor([[0.0, 0.0, -1.36]])
    denom = (rel[:, 0] ** 2 + rel[:, 2] ** 2).clamp(min=1e-6)
    def rate(v):
        return float((rel[:, 2] * v[:, 0] - rel[:, 0] * v[:, 2]) / denom)

    assert rate(torch.tensor([[1.0, 0.0, 0.0]])) < 0.0
    assert rate(torch.tensor([[-1.0, 0.0, 0.0]])) > 0.0
    assert rate(torch.tensor([[0.0, 1.0, 0.0]])) == pytest.approx(0.0), (
        "a purely sideways sway registers as a fore-aft rate, so the shaping term "
        "can be collected by rocking along the beam"
    )


def test_the_euler_singularity_is_on_an_axis_that_does_not_move() -> None:
    """Three hinges in series are Euler angles, and one of them is singular at 90.

    **Which one is a choice, and the obvious choice is the wrong one.** Written
    yaw-pitch-roll the singularity lands on pitch, the swing's own axis: measured
    with random actions on the finished task, `seat_pitch` reaches 65 degrees at an
    action sigma of 0.35 and 80 at 1.5, and a policy that has learnt to pump goes
    past it. What comes out is a NaN in `qpos`, and what it looks like from the
    outside is a training run dying with "the observation group 'actor' contains
    NaN values" -- an error that names neither the swing nor the seat. That
    happened, and this is what it cost to find.

    Roll is the middle hinge now. It reaches 12.7 degrees at a sigma of 3.0, so the
    singularity sits 77 degrees from anything the model does rather than 10.
    """
    import mujoco

    from scenes.swing import _swing_spec

    spec = _swing_spec()
    model = spec.compile()
    order = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    seat = [n for n in order if n and n.startswith("seat_")]
    assert seat == [
        "seat_slide_x", "seat_slide_y", "seat_slide_z",
        "seat_yaw", "seat_roll", "seat_pitch",
    ], (
        f"the seat's hinges are ordered {seat[3:]}. The **middle** one carries the "
        f"Euler singularity, and it must not be pitch -- that is the axis the swing "
        f"swings on, and it reaches 80 degrees under noise"
    )


def test_a_swing_that_goes_right_over_ends_the_episode() -> None:
    """Neither of the other two terminations can see it.

    `tipped_on_the_deck` and `off_the_plank` are both measured in the deck's own
    frame, so a robot riding squarely on a plank that has gone 75 degrees past
    level reads as perfectly upright and perfectly centred. The whole assembly
    leaves the world the task is about together, and nothing notices.
    """
    import math

    cfg = _cfg()
    assert "swing_over" in cfg.terminations, (
        "nothing terminates on the seat's own angle, so a swing can carry the robot "
        "right over and the episode runs on into a region the model is not built for"
    )
    limit = cfg.terminations["swing_over"].params["limit_angle"]
    target = cfg.rewards["swing_amplitude"].params["target_angle"]
    # Above anything a legitimate swing produces -- the deck overshoots the rope by
    # about 18% -- and below the Euler singularity at 90 degrees.
    assert target * 1.2 < limit < math.radians(90.0), (
        f"the limit is {math.degrees(limit):.0f} degrees against a target of "
        f"{math.degrees(target):.0f}; it has to clear the deck's overshoot and stay "
        f"short of the singularity"
    )


#: The four terms that read the swing. Named here so the asymmetry test below and
#: the config cannot drift apart.
_PRIVILEGED = ("swing_offset", "swing_velocity", "deck_up", "deck_offset")


def test_the_actor_runs_on_proprioception_alone() -> None:
    """The policy may not read the swing; only the critic may.

    A robot standing on a real swing cannot measure where the plank is under its
    feet, so a policy that has been trained on it is a policy that cannot be
    deployed -- and nothing about a training run says so. The obvious regression is
    somebody mounting a new swing term on both groups because that is what the loop
    above it used to do.

    It costs the actor nothing, and that is worth stating because it is the reason
    the split is safe: **the swing is already in the IMU.** The deck hangs
    perpendicular to the rope and the robot stands square on it, so
    `projected_gravity`'s x component is the sine of the swing angle -- measured,
    +0.342 at -20.0 degrees against sin(20) = 0.342 -- and the gyro carries the
    rate, -1.05 rad/s through the bottom of the arc and near zero at both ends.
    """
    cfg = _cfg()
    actor = set(cfg.observations["actor"].terms)
    critic = set(cfg.observations["critic"].terms)

    leaked = sorted(actor & set(_PRIVILEGED))
    assert not leaked, (
        f"the actor can see {leaked}, which a robot on a real swing cannot measure"
    )
    missing = sorted(set(_PRIVILEGED) - critic)
    assert not missing, f"the critic is missing {missing}, so nobody sees the swing"

    # The control that the actor still has the two terms the argument rests on.
    assert {"projected_gravity", "base_ang_vel"} <= actor, (
        "the swing angle and its rate reach the policy only through these two; "
        "without them the actor genuinely cannot see the swing at all"
    )


def test_replay_starts_the_swing_at_rest() -> None:
    """Handing an episode a swing it did not build is a training device and must
    not reach replay.

    Most of a training batch starts part way up the arc, so the critic learns what
    a large swing is worth before the policy can produce one. Watching a replay,
    that is the one thing you do not want: a robot dropped onto a 45 degree swing
    looks exactly like a robot that has learnt to pump, for the first ten seconds,
    and there is no way to tell from the picture which it is. From rest the
    amplitude has nowhere to come from except the policy.
    """
    import tasks

    training = tasks.load_env_cfg("jumper.swing")
    replay = tasks.load_env_cfg("jumper.swing", play=True)

    assert replay.events["reset_on_the_swing"].params["angle_range"] == (0.0, 0.0), (
        "replay is handed a swing it did not build, and the picture cannot say so"
    )
    assert max(training.events["reset_on_the_swing"].params["angle_range"]) > 0.0, (
        "training lost its release range, which is the control group for this one"
    )
    # And the episode: `velocity_env_cfg` sets 1e9 in play so a replay runs
    # continuously. Writing the training length over it -- which this task did --
    # gives a silent reset every thirty seconds, which looks like the robot
    # teleporting back onto the plank rather than like an episode boundary.
    assert replay.episode_length_s > training.episode_length_s * 1000


def test_the_actor_sees_more_than_one_frame_of_the_imu() -> None:
    """The shared history table was chosen on a walking task and is wrong here.

    `velocity_env.py::HISTORY_TERMS` withholds history from `projected_gravity`
    and `base_ang_vel` because on a gait task their history is worth +0.0020 and
    +0.0016 of R^2. On a swing those two terms **are** the swing -- the deck hangs
    perpendicular to the rope -- and the actor is given nothing else about it.

    Measured on the actor's own noisy observation, the swing's angular rate reads
    R^2 0.50 from one frame and 0.95 from twenty. The rate's *sign* is what a
    pumping drive switches on, and it switches at the bottom of the arc where the
    rate passes through zero and one frame of a +/-0.35 rad/s gyro is a coin toss.

    Nothing about that is visible from inside this task: inheriting the table
    produces a config that builds, trains, and reports a policy that cannot hear
    the swing it is standing on. So this pins the override, and pins that the
    shared table still says otherwise -- if `HISTORY_TERMS` ever grows these two,
    the override is redundant and the reasoning above should move rather than rot.
    """
    from tasks.jumper.common.velocity_env import HISTORY_TERMS
    from tasks.jumper.swing.env_cfg import IMU_HISTORY

    imu = ("projected_gravity", "base_ang_vel")
    assert not any(term in HISTORY_TERMS for term in imu), (
        "the shared table now gives the IMU history of its own, so this task's "
        "override is no longer the thing keeping the swing visible"
    )

    cfg = _cfg()
    for name, group in cfg.observations.items():
        for term in imu:
            if term not in group.terms:
                continue
            got = group.terms[term].history_length
            assert got == IMU_HISTORY, (
                f"{name}/{term} has {got} frames of history, not {IMU_HISTORY}. "
                f"One frame of the gyro cannot give the sign of the swing's rate "
                f"at the bottom of the arc, which is when the drive reverses"
            )

    # The control: the four velocity tasks must be untouched, or this was a change
    # to `common/` wearing a task's clothes.
    import tasks

    other = tasks.load_env_cfg("jumper.tripod")
    assert other.observations["actor"].terms["projected_gravity"].history_length == 0, (
        "jumper.tripod's IMU history changed too, so the override leaked into the "
        "shared skeleton and the four gait tasks are no longer each other's controls"
    )


def test_the_deck_rides_square_to_the_rope() -> None:
    """Four terms and one termination are calibrated against this, and it is a
    property of the **rigging** rather than of anything in this directory.

    Two ropes a side leaving one beam point hold the plank square to the rope:
    pitch it and one of the pair has to lengthen. Nothing in `mdp/` enforces that,
    and if the rigging changes -- back to a knot, or to two beam points fore and
    aft -- the plank stops tracking and every one of these silently means something
    else:

    - `swing_over` fires on the **deck's** tilt and is set at 70 degrees. With a
      deck that overshoots by 18%, as the knotted rigging did, that is an amplitude
      of 59 and `swing_amplitude`'s falloff is calibrated against the wrong number.
    - `body_tilt_rate_on_the_deck` subtracts the deck's rate to leave the robot's
      own. Measured, a passive robot now reads 0.001 against a 0.5 deadband; the
      knotted rig read 0.465, which is almost the whole allowance.
    - `deck_upright` and the actor's `projected_gravity` both assume the robot
      standing square on the deck is standing square to the rope.

    Two beam points would be worse than a knot and is the case worth naming: the
    plank would then ride *level* at every point of the arc, the robot would stay
    upright, and its IMU would report **nothing at all** about the swing -- with
    every reward here still producing plausible numbers.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    from tasks.jumper.swing.mdp import state

    n = 8
    res = resolve(backend="native", device="cpu", num_envs=n)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = n
    cfg.sim.device = res.device
    cfg.events["reset_on_the_swing"].params["angle_range"] = (math.radians(35.0),) * 2
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        env.reset()
        zero = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        swing = env.scene["swing"]
        pitch = swing.joint_names.index("seat_pitch")
        geometric, euler = [], []
        for _ in range(150):
            env.step(zero)
            geometric.append(state.swing_rate(env).clone())
            euler.append(swing.data.joint_vel[:, pitch].clone())
        a, b = torch.cat(geometric), torch.cat(euler)
    finally:
        env.close()

    # The rope's rate from the seat's position and velocity, against the deck's own
    # pitch rate. They are the same number on a rig that tracks.
    worst = float((a - b).abs().max())
    assert worst < 0.05, (
        f"the deck's pitch rate and the rope's angular rate differ by up to "
        f"{worst:.3f} rad/s, so the plank is no longer riding square to the rope"
    )
    spread = float(a.std())
    assert spread > 0.3, (
        f"the swing barely moved (rate std {spread:.3f} rad/s), so the agreement "
        f"above is two numbers that are both near zero"
    )


def test_the_objective_peaks_at_the_target_rather_than_plateauing() -> None:
    """Overshoot has to cost something before the termination arrives.

    Measured on the open-loop rig: a rider that keeps the phase drives **every**
    length in `HANG_L_RANGE` past 85 degrees on +/-20 mm of travel, a fifth of what
    this robot has. So a policy that learns to pump at all learns something that,
    left alone, ends its own episode on `swing_over`.

    A reward that is flat from the target up to the termination gives no gradient
    over exactly that band: nothing tells the policy to stop until the episode
    does, and what a run shows is a curve that climbs and then collapses with no
    term to blame it on. `swing_amplitude` therefore falls off above the target,
    and this pins that -- the term used to be `clamp(A / target, 0, 1)`, which
    passes every other test in this file.
    """
    import torch

    from tasks.jumper.swing.env_cfg import TARGET_ANGLE
    from tasks.jumper.swing.mdp import rewards

    cfg = _cfg()
    params = cfg.rewards["swing_amplitude"].params
    assert "over_std" in params, "the objective has no falloff above the target"

    class _Fake:
        """Just enough env to stand in for `state.swing_amplitude`."""

    angles = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]) * TARGET_ANGLE
    original = rewards.state.swing_amplitude
    try:
        rewards.state.swing_amplitude = lambda env, name="swing": angles
        value = rewards.swing_amplitude(_Fake(), **params)
    finally:
        rewards.state.swing_amplitude = original

    assert value[0] < value[1] < value[2], "the ramp up to the target is not monotone"
    assert abs(float(value[2]) - 1.0) < 1e-6, "the target is not worth full marks"
    assert value[3] < value[2] and value[4] < value[3], (
        f"above the target the objective reads {[round(float(v), 3) for v in value]} "
        f"-- flat, so nothing turns the policy back before `swing_over` does"
    )
    # And it has to be most of the way down by the time the termination fires, or
    # the falloff is decoration. `swing_over` is a *deck* tilt, and with four ropes
    # to the beam the deck rides square to the rope -- so it is the amplitude
    # directly. This divided by 1.18 while the knotted rigging made the deck
    # overshoot; that factor going away made the test stricter, not looser.
    limit = cfg.terminations["swing_over"].params["limit_angle"]
    try:
        rewards.state.swing_amplitude = lambda env, name="swing": torch.tensor([limit])
        at_limit = float(rewards.swing_amplitude(_Fake(), **params)[0])
    finally:
        rewards.state.swing_amplitude = original
    assert at_limit < 0.5, (
        f"at the amplitude that terminates the episode the objective is still "
        f"worth {at_limit:.2f} of full marks, so the reward is not what turns the "
        f"policy back"
    )


@pytest.mark.parametrize(
    "given,expected",
    [(None, None), ("1.8", (1.8, 1.8)), ("0.7,1.5", (0.7, 1.5))],
)
def test_the_rope_length_can_be_pinned(given, expected) -> None:
    """One length is how you ask whether a policy really generalises.

    A policy trained across the range and replayed across the range looks the same
    as one that learnt a single rhythm and got lucky. `--swing-length 1.8` is the
    question that separates them, so it has to reach the event that draws.
    """
    from tasks.jumper.swing.env_cfg import ROPE_LENGTH_RANGE, _length_range

    assert _length_range(given) == (expected or ROPE_LENGTH_RANGE)


@pytest.mark.parametrize("bad", ["0.3", "0.5", "2.5", "1.8,0.9", "long", ""])
def test_an_unriggable_length_raises_rather_than_clamping(bad: str) -> None:
    """Outside what the frame can be rigged at is an error, not a nearest value.

    A swing whose upper rope came out negative is not a shorter swing and a seat
    below the ground is not a longer one, so clamping would answer a question
    nobody asked. The empty string is the exception and is not an error: it is
    what an unset `MJRL_SWING_LENGTH` looks like, and it means "use the range".
    """
    from tasks.jumper.swing.env_cfg import ROPE_LENGTH_RANGE, _length_range

    if bad == "":
        assert _length_range(bad) == ROPE_LENGTH_RANGE
        return
    with pytest.raises(ValueError, match="swing-length"):
        _length_range(bad)


def test_the_task_declares_its_own_flag() -> None:
    """A task's knob is a flag the task adds, not one `scripts/` grows.

    `scripts/_cli.py` is one parser for train, play and export, so a
    `--swing-angle` written there would be a task's vocabulary in a file that is
    not allowed any -- the rule `tests/test_log_layout.py` pins, and the one
    `play.py` was quietly breaking to print a commanded speed. `load_cli_args`
    hands the parser to the selected task instead.

    The control that this is real and not decoration: **another task must not get
    the flag.** A hook that added its argument unconditionally would look
    identical from inside `jumper.swing`.
    """
    import argparse

    import tasks

    add = tasks.load_cli_args("jumper.swing")
    assert add is not None, "the task declares no arguments"

    parser = argparse.ArgumentParser()
    names = add(parser.add_argument_group("swing"))
    assert names == ("swing_angle", "swing_length"), (
        f"the hook returned {names}, which is what the entry point forwards to "
        f"load_env_cfg -- it has to name exactly what was added"
    )
    parsed = parser.parse_args(["--swing-angle", "25", "--swing-length", "1.8"])
    assert parsed.swing_angle == "25"
    assert parsed.swing_length == "1.8"

    # And every name it returns has to be one `env_cfg` actually takes, or the
    # flag parses, forwards, and raises a TypeError three frames away from the
    # thing that is wrong.
    import inspect

    from tasks.jumper.swing.env_cfg import env_cfg

    accepted = set(inspect.signature(env_cfg).parameters)
    assert set(names) <= accepted, f"{set(names) - accepted} is not an env_cfg argument"

    for other in ("jumper.flat", "jumper.tripod", "jumper.dance"):
        assert tasks.load_cli_args(other) is None, (
            f"{other} would also get --swing-angle, so the flag is not the task's"
        )


def test_the_release_angle_can_be_set_without_a_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.env` and the shell reach the same knob, with the documented precedence.

    `scripts/_cli.py` holds one parser for all three entry points, so a
    `--swing-angle` on it would be a task's vocabulary in a file that is not
    allowed any -- the rule `tests/test_log_layout.py` pins, and the one `play.py`
    was breaking to print a commanded speed. `mjrl.dotenv` has already filled
    `os.environ` from `.env` and `.env.local` before `env_cfg` runs, so reading it
    in the task gets the project default, the personal override and the shell in
    the documented order without the parser knowing anything.

    **Unparseable is an error, not a fallback.** A setting that is silently
    ignored reads as a feature that does not work, which is what `.env` says about
    every other key in it -- and it is worse here than usual, because the value it
    would silently discard is the one thing the person watching the replay asked
    for.
    """
    import tasks
    from tasks.jumper.swing.env_cfg import SWING_ANGLE_ENV

    def choices(play: bool) -> tuple[float, float]:
        cfg = tasks.load_env_cfg("jumper.swing", play=play)
        return tuple(
            round(math.degrees(v), 3)
            for v in cfg.events["reset_on_the_swing"].params["angle_range"]
        )

    monkeypatch.delenv(SWING_ANGLE_ENV, raising=False)
    assert choices(play=True) == (0.0, 0.0), "replay's default is a dead stop"
    assert max(choices(play=False)) > 0.0, "training's default is the release range"

    monkeypatch.setenv(SWING_ANGLE_ENV, "20")
    assert choices(play=True) == (20.0, 20.0), "one value pins every episode to it"
    assert choices(play=False) == (20.0, 20.0), "the override applies to training too"

    monkeypatch.setenv(SWING_ANGLE_ENV, "5, 35")
    assert choices(play=True) == (5.0, 35.0), "a pair is a range"

    for bad, why in (
        ("abc", "not a number"),
        ("120", "past the singularity"),
        ("35,15", "descending"),
        ("-10,10", "a signed range, when the sign is already randomised"),
        ("35,15,0", "the three-point table this no longer takes"),
    ):
        monkeypatch.setenv(SWING_ANGLE_ENV, bad)
        with pytest.raises(ValueError, match=SWING_ANGLE_ENV):
            choices(play=True)


def test_the_key_is_documented_where_people_look() -> None:
    """`.env` is the committed file of project defaults and the only place a
    reader finds out a knob exists. One that is not in it is a knob nobody knows
    about -- the same reason `tests/test_scenes.py` checks `MJRL_SCENE=` is there.
    """
    from tasks.jumper.swing.env_cfg import SWING_ANGLE_ENV

    env = (
        pathlib.Path(__file__).resolve().parents[1] / ".env"
    ).read_text(encoding="utf-8")
    assert f"{SWING_ANGLE_ENV}=" in env, f"{SWING_ANGLE_ENV} is undocumented"


def test_the_task_builds_its_own_swing() -> None:
    """`--scene` is optional, so a task that needed one would train on an empty
    field the moment somebody forgot the flag -- and nothing would say so: the
    robot would fall 0.6 m on the first step and the run would look like a very
    bad policy.

    The placement event matters as much as the entity. mjlab applies `env_origins`
    inside `reset_root_state_uniform` and nowhere else, so without it every swing
    in the batch sits at the world origin while the robots spread out.
    """
    cfg = _cfg()
    assert "swing" in cfg.scene.entities, "the task does not build its own swing"
    assert "reset_prop_swing" in cfg.events
    assert cfg.events["reset_prop_swing"].params["pose_range"] == {}, (
        "the entity's own init_state already carries the offset; passing it again "
        "puts the swing at twice the distance"
    )
    assert "reset_base" not in cfg.events, (
        "mjlab's reset_base puts the robot at a fixed offset from the environment "
        "origin with half a metre of scatter, which is mid-air beside a 0.44 m plank"
    )
    assert "reset_on_the_swing" in cfg.events
    # The range reaches rest at one end and a real swing at the other. Both ends
    # are load-bearing: with every environment handed a swing, nothing in the batch
    # has to earn its amplitude; with every environment starting from rest, nothing
    # shows the critic what a large swing is worth before the policy can build one.
    low, high = cfg.events["reset_on_the_swing"].params["angle_range"]
    assert low == 0.0, (
        f"the release range starts at {math.degrees(low):.0f} degrees, so no "
        f"environment starts at rest and nothing in the batch has to earn its "
        f"amplitude"
    )
    assert high > 0.0, "every environment starts at rest; nothing shows "\
        "the critic what a large swing is worth"


# ── The speed through the bottom ────────────────────────────────────────


def test_the_speed_term_is_about_the_swing_and_not_about_a_sway() -> None:
    """A sideways sway carries real speed and must not be paid for it.

    This is the trap `_plane_energy` exists to close, arriving on a second term.
    The obvious way to write "how fast is the deck moving" is `|v|` over the whole
    3-vector, and it passes every other test in this file -- both modes are
    pendulums of very nearly the same length on this rig (`sway_amplitude` says
    why), so the policy has two ways to make the number go up and one of them is
    easier to excite and useless.

    The control group is the same energy put fore and aft: if that scored low too,
    the assertion below would be passing on a term that reads zero for everything.
    """
    import torch

    from tasks.jumper.swing.env_cfg import TARGET_SPEED
    from tasks.jumper.swing.mdp import rewards, state

    class _Fake:
        pass

    # A seat 1.36 m below the beam, moving at 2 m/s -- once along the swing, once
    # along the beam. Same speed, same height, same energy.
    offset = torch.tensor([[0.0, 0.0, -1.36], [0.0, 0.0, -1.36]])
    velocity = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    original = (state.swing_offset, state.swing_velocity)
    try:
        rewards.state.swing_offset = lambda env, name="swing": offset
        rewards.state.swing_velocity = lambda env, name="swing": velocity
        value = rewards.deck_speed_through_the_bottom(_Fake(), TARGET_SPEED)
    finally:
        rewards.state.swing_offset, rewards.state.swing_velocity = original

    fore_aft, sideways = float(value[0]), float(value[1])
    assert sideways < 0.02, (
        f"a seat moving 2 m/s sideways collects {sideways:.3f} of the speed term, "
        f"so the reward pays for a sway as if it were a swing"
    )
    assert fore_aft > 0.9, (
        f"the same 2 m/s along the swing collects {fore_aft:.3f}, so the term reads "
        f"near zero for everything and the assertion above is vacuous"
    )


def test_every_rope_can_earn_the_speed_term_without_going_past_the_target() -> None:
    """`TARGET_SPEED` has to come from the **shortest** rope, and nothing else in
    the code says so.

    Speed and amplitude are the same measurement in two currencies, so the
    saturation constant decides, per rope, the angle at which more swing stops
    paying more speed. Anchor it anywhere but the short end and one of two silent
    things happens:

    - anchored long (the obvious "use the biggest number"), a 0.6 m swing tops out
      at 0.58 of the term however well it is ridden -- a permanent tax for having
      drawn a short rope, which reads in tensorboard as the policy failing to
      learn on short swings;
    - anchored at the nominal length, the 1.8 m rope saturates at 34 degrees, which
      is still short of the target, so the same tax applies to the short end at
      two thirds strength.

    Neither raises. Both change what the task is by rope length, on a task whose
    whole point (`ROPE_LENGTH_RANGE`) is that it is the same task at every length.
    """
    import math

    from tasks.jumper.swing.env_cfg import (
        ROPE_LENGTH_RANGE,
        TARGET_ANGLE,
        TARGET_SPEED,
    )
    from tasks.jumper.swing.mdp.state import bottom_speed_for

    lo, hi = min(ROPE_LENGTH_RANGE), max(ROPE_LENGTH_RANGE)

    # Every rope reaches full marks at or before the target, and none is capped.
    for hang in (lo, 0.9, 1.35, hi):
        at_target = bottom_speed_for(TARGET_ANGLE, hang) / TARGET_SPEED
        assert at_target >= 1.0 - 1e-9, (
            f"a {hang:.2f} m swing ridden to the target angle collects only "
            f"{at_target:.2f} of the speed term, so it is taxed for its length"
        )

    # And the short end is not saturating early either, or the term would be flat
    # across the whole range and worth nothing.
    short = bottom_speed_for(math.radians(math.degrees(TARGET_ANGLE) - 1.0), lo)
    assert short < TARGET_SPEED, (
        f"the shortest rope already reads {short / TARGET_SPEED:.3f} a degree below "
        f"the target, so the term stops paying before the objective does"
    )

    # The control: the two anchors that were not chosen both fail the loop above.
    for name, anchor in (("nominal", 1.35), ("longest", hi)):
        other = bottom_speed_for(TARGET_ANGLE, anchor)
        capped = bottom_speed_for(TARGET_ANGLE, lo) / other
        assert capped < 1.0, (
            f"anchoring TARGET_SPEED on the {name} rope would also let the short "
            f"end reach full marks, so the assertion above is not pinning the "
            f"choice it claims to"
        )


def test_the_speed_the_reward_reads_is_the_speed_the_deck_reaches() -> None:
    """`bottom_speed_for` builds the constant and `swing_bottom_speed` reads the
    term; they are two derivations of one piece of geometry and must agree.

    If they do not, the term saturates at the wrong angle on every rope and nothing
    says so -- the reward still produces a number between 0 and 1 every step, still
    rises with the swing, and still looks exactly like what was intended. The
    plausible slips are all small, and the tolerance below is sized against the
    smallest of them: the radius is the rope **plus half the plank**, which is 0.4%
    of a short rope and worth 0.78% of the speed (0.47% on a long one). The
    measured disagreement with the geometry right is **0.000%** -- a fresh episode
    is placed at exactly `seat_drop(hang)` with zero velocity and this reads the
    same radius back out of the state -- so 0.3% is three hundred times the noise
    and still under the quietest way to get it wrong. It was 3% first, and break-
    tested at 3% this test did not notice the missing half-plank at all.

    The second half is what makes the name honest. "Speed through the bottom" is
    computed from the energy at whatever point of the arc the seat is at, so it is
    a prediction; this steps the swing and checks the deck actually gets there.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    from scenes.swing import BOARD_T
    from tasks.jumper.swing.env_cfg import TARGET_ANGLE
    from tasks.jumper.swing.mdp import state

    n = 8
    res = resolve(backend="native", device="cpu", num_envs=n)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = n
    cfg.sim.device = res.device
    cfg.events["reset_on_the_swing"].params["angle_range"] = (TARGET_ANGLE,) * 2
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        env.reset()
        hang = state.rope_length(env) - BOARD_T / 2
        predicted = torch.tensor(
            [state.bottom_speed_for(TARGET_ANGLE, float(h)) for h in hang]
        )
        at_release = state.swing_bottom_speed(env).clone()

        # A quarter period at the longest rope is 0.66 s; 60 steps at 50 Hz is 1.2 s,
        # so every environment has passed the bottom at least once.
        zero = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=env.device
        )
        peak = torch.zeros(n)
        for _ in range(60):
            env.step(zero)
            speed = state.swing_velocity(env)[:, 0].abs()
            peak = torch.maximum(peak, speed.cpu())
    finally:
        env.close()

    # The constant's geometry against the term's.
    error = ((at_release - predicted) / predicted).abs()
    assert float(error.max()) < 0.003, (
        f"`bottom_speed_for` and `swing_bottom_speed` disagree by up to "
        f"{float(error.max()) * 100:.1f}% about a swing released at the target, so "
        f"the term saturates somewhere other than where TARGET_SPEED says"
    )
    assert float(hang.max() - hang.min()) > 0.5, (
        f"every environment drew nearly the same rope ({float(hang.min()):.2f} to "
        f"{float(hang.max()):.2f} m), so the agreement above was checked at one "
        f"length and says nothing about the range"
    )

    # And the deck really does reach it. Below rather than above: the robot is
    # riding along and its own centre of mass sits above the deck, so a little of
    # the seat's energy is lent out and comes back.
    ratio = peak / predicted
    assert float(ratio.min()) > 0.90, (
        f"the deck only reached {float(ratio.min()) * 100:.0f}% of the speed the "
        f"term predicted for it, so 'speed through the bottom' is not what the "
        f"number means"
    )
    assert float(ratio.max()) < 1.10, (
        f"the deck overshot the predicted speed by "
        f"{(float(ratio.max()) - 1) * 100:.0f}%, so the energy the term reads is "
        f"not the energy the swing has"
    )


def test_the_release_angle_is_drawn_across_the_whole_range() -> None:
    """The reset draws uniformly, and a pinned angle really is pinned.

    This replaced reference-state initialisation -- a three-point table, a third of
    the batch each -- and the point of the replacement is that the critic sees every
    amplitude between rest and the target as a *start*, not only three of them. A
    draw that collapsed back towards a few values would undo that silently: the
    episodes still run, the reward still rises with the swing, and the only symptom
    is a value function that is good at three amplitudes and interpolating the rest.

    The second half is the one that would actually get written by accident.
    `high * rand(n)` is correct for the default range, because its low end is zero,
    and it silently turns **every pinned angle into a range from rest**. Someone
    replaying with `--swing-angle 30` to watch a policy hold a swing would get a
    uniform draw from 0 to 30 and no indication that the flag had been ignored. So
    the pin is checked against the same environment, and it is the control group:
    if the draw were broken in the other direction -- every environment landing on
    one value -- the spread assertions above would catch it and this one would not.
    """
    import torch
    from mjrl.backend import select
    from mjrl.backend.resolve import resolve

    from tasks.jumper.swing.env_cfg import RESET_ANGLE_RANGE_DEG
    from tasks.jumper.swing.mdp import state

    n = 64
    res = resolve(backend="native", device="cpu", num_envs=n)
    select.use_backend(res)

    cfg = _cfg()
    cfg.scene.num_envs = n
    cfg.sim.device = res.device
    cfg.events.pop("push_robot", None)

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg, device=res.device)
    try:
        # The seat is released at its turning point with zero velocity, so the
        # amplitude immediately after a reset **is** the angle it was drawn at.
        drawn = []
        for _ in range(4):
            env.reset()
            drawn.append(state.swing_amplitude(env).clone())
        drawn = torch.rad2deg(torch.cat(drawn))

        pinned_at = 30.0
        term = env.event_manager.get_term_cfg("reset_on_the_swing")
        term.params["angle_range"] = (math.radians(pinned_at),) * 2
        env.reset()
        pinned = torch.rad2deg(state.swing_amplitude(env).clone())
    finally:
        env.close()

    low, high = RESET_ANGLE_RANGE_DEG
    assert float(drawn.min()) < low + 2.0 and float(drawn.max()) > high - 2.0, (
        f"the draw spans {float(drawn.min()):.1f} to {float(drawn.max()):.1f} "
        f"degrees against a range of {low}-{high}, so it does not reach the ends"
    )
    # Four equal quarters of the range, each expected to hold a quarter of the
    # draws. A three-point table would leave two of them empty; anything clustered
    # would starve at least one. 10% of 256 samples is 6 standard deviations below
    # the 25% a uniform draw gives, so this cannot fail on an unlucky seed.
    edges = torch.linspace(low, high, 5)
    shares = [
        float(((drawn >= edges[i]) & (drawn < edges[i + 1])).float().mean())
        for i in range(4)
    ]
    assert min(shares) > 0.10, (
        f"the quarters of the range hold {[round(s, 3) for s in shares]} of the "
        f"draws, so the release angle is clustered rather than uniform"
    )

    # The control, and the failure mode that would otherwise be invisible.
    assert float((pinned - pinned_at).abs().max()) < 0.5, (
        f"pinned at {pinned_at} degrees, the batch was released from "
        f"{float(pinned.min()):.1f} to {float(pinned.max()):.1f} -- the low end of "
        f"the range is being dropped, so --swing-angle is a range and not a pin"
    )


def test_the_middle_of_the_plank_is_where_the_robot_stands() -> None:
    """Two terms measure the robot's fore-aft offset and they have to agree about
    what zero means.

    The robot's footprint is not under its base: the front legs reach 0.190 m
    forward and the rear ones 0.154 m back, so `DECK_OFFSET` stands the base 18 mm
    behind the plank's centre to put the six feet in the middle of it
    (`scenes/swing.py::STANCE_CENTRE_X`). A term that measures from the plank's
    **geometric** centre therefore peaks 18 mm ahead of where the robot naturally
    stands -- and measured on a passive robot that is the front foot's margin going
    from 47.6 mm to 29.5 while the rear grows to 63.4. The maximum of a term whose
    job is "do not drift towards an edge" would be the least safe place on the
    plank.

    Nothing raises. A robot standing correctly collects 0.985 rather than 1.0 for
    ever, which is a small constant pull in one direction and invisible in a
    training curve. `pump_power` already subtracts the offset and says why;
    `centred_on_the_plank` did not, and the two disagreeing about where the middle
    is was the whole of the bug.

    Pinned as a property of the functions rather than of one configuration, so it
    survives the weights and the width moving.
    """
    import torch

    from tasks.jumper.swing.mdp import rewards, state

    assert state.DECK_OFFSET[0] != 0.0, (
        "the standing offset is zero, so this test cannot distinguish a term that "
        "corrects for it from one that does not"
    )

    class _Fake:
        pass

    # Three bases in the deck's frame: the plank's geometric centre, where the
    # reset actually stands the robot, and the mirror of that.
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [state.DECK_OFFSET[0], 0.0, 0.0], [-state.DECK_OFFSET[0], 0.0, 0.0]]
    )
    original = state.robot_on_deck
    try:
        rewards.state.robot_on_deck = lambda env, r="robot", n="swing": offsets
        value = rewards.centred_on_the_plank(_Fake(), std=0.15)
    finally:
        rewards.state.robot_on_deck = original

    at_centre, at_stance, at_mirror = (float(v) for v in value)
    assert abs(at_stance - 1.0) < 1e-9, (
        f"a robot standing exactly where the reset put it collects {at_stance:.4f} "
        f"of this term, so its maximum is somewhere else on the plank"
    )
    assert at_centre < at_stance and at_mirror < at_stance, (
        f"the term reads {at_centre:.4f} at the plank's geometric centre and "
        f"{at_stance:.4f} at the stance, so it is not peaked at the stance at all"
    )
    # The control: without the correction the plank's centre would be the maximum,
    # which is what the assertions above are distinguishing against.
    assert at_centre < 1.0, "the plank's geometric centre is still the maximum"

    # And `pump_power` measures its lever from the same origin, or the two terms
    # disagree about which way "forward" is from neutral. Checked by what it
    # returns, not by what its source says: `grep`ing for `DECK_OFFSET` passes on
    # the *comment* inside the function, which is how this assertion was written
    # first and why deleting the correction did not fail it.
    stance = torch.tensor([[state.DECK_OFFSET[0], 0.0, 0.0]])
    rate = torch.tensor([1.0])
    originals = (state.robot_on_deck, state.swing_rate)
    try:
        rewards.state.robot_on_deck = lambda env, r="robot", n="swing": stance
        rewards.state.swing_rate = lambda env, n="swing": rate
        lever = float(rewards.pump_power(_Fake())[0])
    finally:
        rewards.state.robot_on_deck, rewards.state.swing_rate = originals

    assert abs(lever) < 1e-9, (
        f"a robot standing still where the reset put it, on a swing moving at "
        f"1 rad/s, feeds {lever:+.4f} of pump power -- so the term pays (or "
        f"charges) for doing nothing, and does it asymmetrically on a decaying swing"
    )
