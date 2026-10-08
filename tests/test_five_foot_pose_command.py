"""The body-attitude command: the four ways it goes wrong without raising.

1. **The gravity convention drifting.** Pitch and roll are read off the base-frame
   gravity vector, and a flipped sign still runs, still produces a plausible
   number, and simply asks the robot to lean the wrong way. `nose_pitch` was
   written twice with the sign wrong before it was made symmetric; the command
   makes the sign matter again, because now something is being *asked* for.
2. **The band selection reading the wrong thing.** Standing gets the wide bands,
   walking the narrow ones. Both produce plausible commands, so getting the test
   backwards is invisible -- what shows up is a policy that trips over its own
   swing amplitude at the top of the range, thirty minutes into a run.
3. **The neutral fraction quietly undoing the operator.** `_update_command` zeroes
   the environments pinned to neutral, and it runs *after* the sampler on every
   step. A teleop that writes before it drives a robot that snaps back to level
   for a quarter of the field and holds for the rest -- the exact failure
   `common/mdp/operator.py` documents for the velocity command.
4. **Keys that also mean something else.** The walk and the claw are on the same
   keyboard. A pose key that collides with one of theirs moves two things per
   press, and only one of them is the one being watched.

Where an assertion could pass vacuously it carries a control group, per this
repository's convention.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the command term is torch")

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from tasks.jumper.five_foot.mdp.pose_command import (
    PITCH,
    POSE_DIM,
    ROLL,
    TWIST,
    BodyPoseCommandCfg,
    TeleopBodyPoseCommandCfg,
    base_pitch_roll,
)

_DEG = math.pi / 180.0


# ── The convention, against rotations rather than against itself ──────────


def _gravity_for(pitch_deg: float, roll_deg: float) -> torch.Tensor:
    """Base-frame gravity for a body at this pitch and roll.

    Built by rotating rather than by writing the formula down a second time: a
    test that restates the implementation's algebra passes when the algebra is
    wrong. The rotation is applied roll-first so that it composes as
    `Ry(pitch) Rx(roll)`, which is the convention `base_pitch_roll` inverts.
    """

    def quat(axis: tuple[float, float, float], deg: float) -> torch.Tensor:
        half = math.radians(deg) / 2.0
        s = math.sin(half)
        return torch.tensor([[math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]])

    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = a[0]
        w2, x2, y2, z2 = b[0]
        return torch.tensor([[
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]])

    q = mul(quat((0.0, 1.0, 0.0), pitch_deg), quat((1.0, 0.0, 0.0), roll_deg))
    return quat_apply_inverse(q, torch.tensor([[0.0, 0.0, -1.0]])), q


def test_pitch_and_roll_mean_what_the_reward_terms_think_they_mean() -> None:
    """Pins failure 1.

    The two reward terms and the command have to agree about which direction is
    nose-down and which side is down in roll. They are now one function, so what
    is left to pin is that the one function is right -- checked against a body
    actually rotated by a quaternion, and against where the nose ends up.
    """
    for pitch_deg, roll_deg in ((10.0, 0.0), (-10.0, 0.0), (0.0, 12.0),
                                (0.0, -12.0), (8.0, -5.0)):
        g, q = _gravity_for(pitch_deg, roll_deg)
        pitch, roll = base_pitch_roll(g)
        assert float(pitch) / _DEG == pytest.approx(pitch_deg, abs=1e-4)
        assert float(roll) / _DEG == pytest.approx(roll_deg, abs=1e-4)

    # **Positive pitch is nose-down**, which is the half of the convention a sign
    # error preserves in the arithmetic and inverts in the world. Checked by
    # asking where the body's own +x axis ends up in the world.
    g, q = _gravity_for(10.0, 0.0)
    nose_z = float(quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))[0, 2])
    assert nose_z < 0, "positive pitch should put the nose below the horizon"
    assert float(base_pitch_roll(g)[0]) > 0

    # Control group: level reads as zero on both axes, so the assertions above
    # are not passing because the function returns its input.
    level, _ = _gravity_for(0.0, 0.0)
    pitch, roll = base_pitch_roll(level)
    assert float(pitch) == pytest.approx(0.0, abs=1e-6)
    assert float(roll) == pytest.approx(0.0, abs=1e-6)


def test_the_reward_terms_read_the_same_angles() -> None:
    """Pins failure 1 where it actually bites: the two consumers.

    `nose_pitch` and `body_roll` charge an angle each. If either stops agreeing
    with `base_pitch_roll` -- by being rewritten inline, say -- the command asks
    for one thing and the penalty charges another, and the policy settles wherever
    they balance. Nothing raises and both numbers look reasonable.
    """
    from tasks.jumper.five_foot.mdp.rewards import body_roll, nose_pitch

    class _Env:
        def __init__(self, g):
            self.scene = {"robot": type("A", (), {"data": type("D", (), {
                "projected_gravity_b": g})()})()}

    g, _ = _gravity_for(9.0, -7.0)
    pitch, roll = base_pitch_roll(g)
    # Each term is the squared angle (no command in this stub, so the target is
    # zero), so the angle it read back is its square root.
    got_pitch = float(nose_pitch(_Env(g))[0]) ** 0.5
    got_roll = float(body_roll(_Env(g))[0]) ** 0.5
    assert got_pitch == pytest.approx(abs(float(pitch)), rel=1e-6)
    assert got_roll == pytest.approx(abs(float(roll)), rel=1e-6)
    # Control group: the two axes are not the same number, so a term reading the
    # wrong one would fail the check above rather than pass it by symmetry.
    assert got_pitch != pytest.approx(got_roll, rel=1e-3)


# ── The sampler, on a stub environment ────────────────────────────────────


class _StubEnv:
    """Enough environment for a command term: `CommandTerm` reads `num_envs`,
    `device` and `step_dt`, this term also reads the robot's gravity and the
    velocity command, and its replay subclass the step counter."""

    def __init__(self, num_envs: int, twist: torch.Tensor | None) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = 0.02
        # The operator reads its devices once per step, keyed on this.
        self.common_step_counter = 0
        gravity = torch.zeros(num_envs, 3)
        gravity[:, 2] = -1.0
        robot = type("A", (), {"data": type("D", (), {
            "projected_gravity_b": gravity})()})()
        self.scene = {"robot": robot}
        self.command_manager = _StubCommands(twist)


class _StubCommands:
    def __init__(self, twist: torch.Tensor | None) -> None:
        self._twist = twist

    def get_command(self, name: str) -> torch.Tensor:
        if self._twist is None:
            raise KeyError(name)
        return self._twist


def _all_envs(n: int) -> torch.Tensor:
    return torch.arange(n)


def test_a_standstill_gets_the_wide_bands_and_walking_gets_the_narrow_ones() -> None:
    """Pins failure 2.

    The bands exist to protect leg swing amplitude: a wide twist is free when
    nothing is swinging and expensive when something is. Selecting on the wrong
    environments samples inside a legal range either way, so the mistake only
    shows as a policy that cannot hold the top of its own command range.

    **The bands this test uses are its own, not the shipped defaults.** Those are
    +-15 in every axis standing and walking, which makes the selection invisible
    -- both branches produce the same number, and a selection that picked the
    wrong branch, or no branch, would pass. What is being pinned here is the
    mechanism; `test_the_shipped_bands_are_fifteen_degrees_everywhere` pins the
    numbers.
    """
    n = 400
    twist = torch.zeros(n, 3)
    twist[n // 2:, 0] = 0.5  # the second half is asked to walk
    env = _StubEnv(n, twist)
    cfg = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0),
        rel_neutral_envs=0.0,
        stand_twist=(-30.0 * _DEG, 30.0 * _DEG),
        move_twist=(-15.0 * _DEG, 15.0 * _DEG),
    )
    term = cfg.build(env)
    term._resample_command(_all_envs(n))

    # `pose_target_b`, not `command`: what the sampler writes is the target, and
    # the observed command walks to it at `max_rate`.
    standing = term.pose_target_b[: n // 2, TWIST].abs().max() / _DEG
    walking = term.pose_target_b[n // 2:, TWIST].abs().max() / _DEG
    assert standing > 20.0, f"standing twist tops out at {standing:.1f} deg"
    assert walking <= 15.0 + 1e-6, f"walking twist reached {walking:.1f} deg"

    # Control group: with everybody standing, both halves reach past the walking
    # band -- so the assertion above is about the selection and not about one
    # half of the buffer never being written.
    env = _StubEnv(n, torch.zeros(n, 3))
    term = cfg.build(env)
    term._resample_command(_all_envs(n))
    assert term.pose_target_b[n // 2:, TWIST].abs().max() / _DEG > 20.0

    # Every axis stays inside the standing band, which is the outer envelope.
    for index, band in ((PITCH, cfg.stand_pitch), (ROLL, cfg.stand_roll),
                        (TWIST, cfg.stand_twist)):
        assert float(term.pose_target_b[:, index].min()) >= band[0] - 1e-6
        assert float(term.pose_target_b[:, index].max()) <= band[1] + 1e-6


def test_the_command_is_sampled_before_the_velocity_command_exists() -> None:
    """Pins the build-order failure.

    Command terms are built in config order, and one built before `twist` sees no
    velocity command at all. Raising there would be loud and fine; what must not
    happen is a crash *or* a silent zero band, so the fallback is the wide one.
    """
    n = 64
    env = _StubEnv(n, None)  # no velocity command in the manager at all
    term = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=0.0
    ).build(env)
    term._resample_command(_all_envs(n))
    assert term.command.shape == (n, POSE_DIM)
    assert float(term.pose_target_b.abs().max()) > 0.0, "the fallback sampled nothing"


def test_the_neutral_fraction_is_level_and_square() -> None:
    """Pins failure 3 in the sampler.

    Without it every environment is always asked for some lean, and the policy
    never learns that neutral is also a command -- which is the pose the robot
    holds for most of its life.
    """
    n = 200
    env = _StubEnv(n, torch.zeros(n, 3))
    term = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=1.0
    ).build(env)
    term._resample_command(_all_envs(n))
    assert float(term.pose_target_b.abs().max()) > 0.0, "nothing was sampled to zero"
    term._update_command(None)
    assert float(term.pose_target_b.abs().max()) == 0.0, "the neutral envs are not neutral"
    assert float(term.command.abs().max()) == 0.0, "the ramp moved toward a zeroed target"

    # Control group: at zero the same code path leaves everything alone.
    term = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=0.0
    ).build(env)
    term._resample_command(_all_envs(n))
    before = term.pose_target_b.clone()
    term._update_command(None)
    assert torch.equal(before, term.pose_target_b)


# ── The operator ──────────────────────────────────────────────────────────


def _settle(term, steps: int = 200) -> None:
    """Run the ramp to convergence. A tenth of the stick is 2 degrees and the
    ramp is 30 deg/s, so a tenth resolves in 4 steps; a full band takes about 65."""
    for _ in range(steps):
        term._env.common_step_counter += 1
        term.compute(0.02)


FIVE_FOOT_CONTROLS = Path(__file__).parents[1] / "tasks/jumper/five_foot/controls.yaml"
POSTURE_CONTROLS = Path(__file__).parents[1] / "tasks/jumper/posture/controls.yaml"


def _teleop(n: int = 8, twist: torch.Tensor | None = None, **kw):
    """The replay term on a stub environment, driven by the task's own file.

    The operator is registered for the stub before the term is built, with a
    pad whose state a test sets and not listening to the viewer, so it leaves no
    handler behind; the walk is attached to it at this task's replay ranges,
    because the operator refuses to run while a command the file describes has
    no term. `twist` is what the velocity command reads -- standing unless a test
    says.
    """
    import tasks
    from tasks.jumper.common.mdp import operator as operator_module
    from tasks.jumper.common.mdp.controls import load_controls

    env = _StubEnv(n, torch.zeros(n, 3) if twist is None else twist)
    controls = load_controls(FIVE_FOOT_CONTROLS)
    op = operator_module.Operator(controls, _Pad(), listen=False, clock=_Clock())
    op._wants = (False, None)
    operator_module._OPERATORS[env] = op
    walk = tasks.load_env_cfg("jumper.five_foot", play=True).commands["twist"]
    op.attach("twist", operator_module.spans(controls.command("twist"), walk))
    cfg = TeleopBodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=1.0,
        controls=controls, term="body_pose", pad=False, **kw,
    )
    term = cfg.build(env)
    term._resample_command(_all_envs(n))
    return term


class _Clock:
    """Seconds, moved by the test: what a key's hold is measured by."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Pad:
    """A pad whose state a test sets directly."""

    name = "test pad"

    def __init__(self) -> None:
        from controller.xbox import GamepadState

        self.s = GamepadState()

    def state(self):
        return self.s


def _press(term, target: str, times: int = 1) -> None:
    """`times` tenths of `target`'s travel on the pad -- a stick direction,
    `Ry-` forward -- held there while `_settle` runs; or a button held, `R3` to
    shift and `B` to let go. Until 2026-09-29 these went through the keyboard,
    which was a copy of the pad; the pad is what these tests are about."""
    from dataclasses import replace

    pad = term._operator._pad
    if target in ("R3", "B"):
        pad.s = replace(pad.s, buttons=pad.s.buttons | {target})
    else:
        sign = -1.0 if target.endswith("-") else 1.0
        pad.s = replace(pad.s, **{target[:-1].lower(): sign * times / 10})


def _let_go(term) -> None:
    """A hand taken off the pad: every stick centred, every button up."""
    from controller.xbox import GamepadState

    term._operator._pad.s = GamepadState()


def test_the_operator_writes_last() -> None:
    """Pins failure 3 where it is hardest to see.

    `_update_command` zeroes the neutral environments on *every* step, so a teleop
    that writes before it is silently overruled. This term is built with the whole
    field neutral, which is the configuration that catches it: write first and the
    command reads zero no matter what was pressed.
    """
    term = _teleop()
    _press(term, "Ry-", 2)  # a fifth of the stick, forward
    _settle(term)
    got = float(term.command[0, PITCH]) / _DEG
    # Two tenths of a 20 degree standing band.
    assert got == pytest.approx(4.0, abs=1e-5), (
        f"pitch reads {got:.1f} deg after two tenths of the stick; the neutral fraction "
        "overwrote the operator"
    )
    # Every environment gets it, not just the one the camera follows.
    assert float(term.command[:, PITCH].std()) == pytest.approx(0.0)

    # Control group: before any key is pressed the term is the sampler, and the
    # neutral fraction does apply.
    fresh = _teleop()
    _settle(fresh)
    assert float(fresh.command.abs().max()) == 0.0


def test_the_stick_moves_the_axis_it_names_and_full_is_the_standing_edge() -> None:
    """Pins a swapped axis, a flipped sign, and an operator that cannot reach
    the top of the range the policy trained on.

    Every control moves *something*, so a swapped pair still looks alive in the
    viewer. And a full stick short of the standing edge is a readout that simply
    stops moving, which reads as the policy's limit.
    """
    term = _teleop()
    # **Signed the way the operator sees it.** Positive pitch is nose-down and
    # positive roll is right-side-down, so the stick forward is +pitch and the
    # stick right, shifted, is +roll; the stick left is a twist to the left, +.
    cases = (
        (("Ry-",), PITCH, 1), (("Ry+",), PITCH, -1),
        (("Rx-",), TWIST, 1), (("Rx+",), TWIST, -1),
        (("R3", "Rx+"), ROLL, 1), (("R3", "Rx-"), ROLL, -1),
    )
    for targets, index, sign in cases:
        for target in targets:
            _press(term, target)
        _settle(term)
        moved = term.command[0]
        assert float(moved[index]) * sign > 0, f"{targets} did not move its own axis"
        others = [i for i in (PITCH, ROLL, TWIST) if i != index]
        assert float(moved[others].abs().max()) == 0.0, f"{targets} moved another axis"
        _let_go(term)  # R3 included: let go, the layer is out
        _settle(term)

    # **Equal to the edge, not merely inside it.** Written the obvious way first
    # -- a band check -- a clamp pointed at the *moving* bands, which sit strictly
    # inside the standing ones, would pass it. Ten tenths is a full stick; five
    # are the whole twist, which owns the first half of the right stick's travel.
    for targets, times, index, edge in (
        (("Ry-",), 10, PITCH, term.cfg.stand_pitch[1]),
        (("Rx-",), 5, TWIST, term.cfg.stand_twist[1]),
        (("R3", "Rx-"), 10, ROLL, term.cfg.stand_roll[0]),
    ):
        for target in targets[:-1]:
            _press(term, target)
        _press(term, targets[-1], times)
        _settle(term)
        value = float(term.command[0, index])
        assert value == pytest.approx(edge, abs=1e-5), (
            f"{targets} x{times} stopped at {value / _DEG:+.1f} deg, not the "
            f"standing edge {edge / _DEG:+.1f}"
        )
        _let_go(term)
        _settle(term)

    # **Release hands it back, and that has to include what taking over changed.**
    # The operator drives the target and clears the neutral fraction; both outlive
    # the release until the next resample, so without an explicit one the sampler
    # appears not to have taken the term back for up to five seconds.
    _press(term, "Ry-", 10)
    _settle(term)
    _press(term, "B")
    _settle(term)
    assert float(term.command.abs().max()) == 0.0


def test_a_walking_robot_is_held_to_the_moving_bands() -> None:
    """Full deflection is the standing edge; walking, the command comes back in.

    The bands exist because a leg in mid-swing needs its amplitude back, and the
    policy was trained walking under 15 degrees. A stick that could hand a walking
    robot 30 degrees of twist is a command from off the distribution, and the
    robot answers it in some way nobody measured.

    The control group is the same stick on a standing robot, which has to reach
    the standing edge -- or the clamp could be a band that was always there.
    """
    walking = torch.tensor([[0.3, 0.0, 0.0]]).repeat(8, 1)
    for twist, pitch_edge, twist_edge in (
        (None, "stand_pitch", "stand_twist"),
        (walking, "move_pitch", "move_twist"),
    ):
        term = _teleop(twist=twist)
        _press(term, "Ry-", 10)
        _settle(term)
        got = float(term.command[0, PITCH])
        assert got == pytest.approx(getattr(term.cfg, pitch_edge)[1], abs=1e-5), (
            f"a full stick forward reads {got / _DEG:+.1f} deg with the robot "
            f"{'walking' if twist is not None else 'standing'}"
        )
        _let_go(term)
        _press(term, "Rx-", 5)
        _settle(term)
        got = float(term.command[0, TWIST])
        assert got == pytest.approx(getattr(term.cfg, twist_edge)[1], abs=1e-5), (
            f"a full twist reads {got / _DEG:+.1f} deg with the robot "
            f"{'walking' if twist is not None else 'standing'}"
        )


def test_the_claw_is_on_the_operator_s_keys() -> None:
    """Pins failure 4, which is gone by construction now, and stays gone here.

    Two handlers used to see every keycode -- the operator, which drives the walk
    and the attitude from `controls.yaml`, and the claw on the brackets and `G`
    -- so a code the two shared fired both. The claw has no keys of its own any
    more: it is a control the file declares for the task, `claw_left`, on `LT`
    and on Space, and a command binding either is refused on load.
    """
    import tasks

    # The operator's keys as this task's replay actually installs them, read off
    # its own file rather than restated here.
    controls = tasks.load_env_cfg("jumper.five_foot", play=True).commands["twist"].controls
    assert set(controls.task) == {"claw_left", "claw_right", "arm_thumb_up", "arm_thumb_down",
                                  "arm_web_up"}
    assert (controls.pad_task["claw_left"], controls.key_task["claw_left"]) == ("LT", ("key_space",))
    assert not {"LT", "RT"} & controls.sources, "a command reads the claw's trigger"

    # Control group: the keys are the ones its legend claims -- W S A D and the
    # arrows, I K, J L, H ;, U O, Esc, Space, Shift, Alt and Ctrl.
    assert len(controls.keystrokes) == 21, controls.keystrokes


def test_five_foot_drives_its_attitude_as_posture_does() -> None:
    """One stick, one meaning, across the two tasks that lean the body.

    Both tasks' pose commands reach the robot on `Command::base_pitch`,
    `base_roll` and `base_twist`, so a control bound differently in one of them
    is a stick that tips the nose forward in one mode and back in the next --
    with each file correct on its own and nothing to say they disagree. Compared
    parsed, binding by binding: every axis the two share, the shift and the
    release. The triggers are the one thing on the pad the two mean differently:
    this task's claw, nothing of posture's.

    The keyboards agree key for key on every axis they share, the twist
    included since it moved to H and ; (Control-agent 3.1, as revised on
    2026-09-29); until then it was Shift + J / L in posture and unbound here,
    where Shift holds the arm out.
    """
    from tasks.jumper.common.mdp.controls import load_controls

    five = load_controls(FIVE_FOOT_CONTROLS)
    posture = load_controls(POSTURE_CONTROLS)
    shared = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "twist", "pitch", "roll")
    for axis in shared:
        assert five.bindings[axis] == posture.bindings[axis], (
            f"{axis}: five_foot binds {five.bindings[axis]}, posture "
            f"{posture.bindings[axis]}"
        )
    assert set(posture.bindings) - set(five.bindings) == {"height"}
    assert (five.shift, five.release_button, five.full_after_s) == (
        posture.shift, posture.release_button, posture.full_after_s
    )
    for axis in shared:
        assert five.key_axes[axis] == posture.key_axes[axis], (
            f"{axis}: five_foot's keys {five.key_axes[axis]}, posture's {posture.key_axes[axis]}"
        )
    assert five.key_release == posture.key_release
    assert five.key_task["arm_thumb_up"] == ("shift",) and not posture.task

    # The same positive directions, word for word where the two word them alike,
    # and the one they word differently -- roll -- is one sense seen from two sides.
    pose = {a.name: a.positive for a in five.command("body_pose").axes}
    posture_pose = {a.name: a.positive for a in posture.command("posture").axes}
    assert pose["pitch"] == posture_pose["pitch"] == "nose down"
    assert pose["twist"] == posture_pose["twist"] == "counter-clockwise"
    assert (pose["roll"], posture_pose["roll"]) == ("right side down", "left side up")

    # Control group: the comparison can fail. Posture's own height binding is not
    # any binding of this task's, so a check that compared nothing would say so.
    assert posture.bindings["height"] not in five.bindings.values()


def test_the_shipped_bands_are_the_reference_implementation_s() -> None:
    """Pins the numbers, which the mechanism cannot show on its own.

    Standing is wider than walking on two axes and equal on the third, so a band
    that changed shows up at runtime only as a policy asked for something it was
    not trained on -- at the top of the range, which is where this mechanism loses
    feet: `nose_pitch` measures 4 of 5 planted by 14 degrees nose-down and 2 of 5
    by 17.

    These are kk-rl-lab's measured bands. +-15 everywhere was tried for one
    configuration and put back: it narrows what the machine can be asked for in
    the state where the ask is cheapest -- feet planted, nothing swinging, which
    is when a claw is aimed.
    """
    cfg = BodyPoseCommandCfg(resampling_time_range=(5.0, 5.0))
    def deg(band):
        return tuple(round(b / _DEG, 1) for b in band)

    assert deg(cfg.stand_pitch) == (-20.0, 20.0), deg(cfg.stand_pitch)
    assert deg(cfg.stand_roll) == (-15.0, 15.0), deg(cfg.stand_roll)
    assert deg(cfg.stand_twist) == (-30.0, 30.0), deg(cfg.stand_twist)
    for name in ("move_pitch", "move_roll", "move_twist"):
        assert deg(getattr(cfg, name)) == (-15.0, 15.0), f"{name} {deg(getattr(cfg, name))}"

    # The walking bands are the ones that must not quietly widen: they are what a
    # swinging leg has to live with.
    for axis in ("pitch", "twist"):
        move, stand = getattr(cfg, f"move_{axis}"), getattr(cfg, f"stand_{axis}")
        assert move[1] < stand[1], f"walking {axis} stopped narrowing"


# ── The deploy boundary ───────────────────────────────────────────────────


def _rust_base_pose_block() -> str:
    """The `"base_pose" => { ... }` arm of the on-robot observation builder."""
    import pathlib

    src = pathlib.Path("deploy/fsm/src/obs.rs").read_text()
    head = src.index('"base_pose" => {')
    return src[head : src.index("}", head)]


def test_the_robot_fills_the_three_channels_in_this_order() -> None:
    """Pins the silent failure at the deploy boundary.

    The observation is built twice in two languages that cannot check each other:
    here by `generated_commands` reading `pose_command_b`, and on the robot by
    `deploy/fsm/src/obs.rs`, which pushes three scalars under the name
    `base_pose`. **Both produce three floats whatever the order is.** A contract
    whose channels are pitch/roll/twist loaded by a controller that fills
    roll/pitch/twist has matching shapes, loads clean, runs, and leans the robot
    the wrong way -- the class of failure this repository's deploy path is built
    around.

    The Rust is read rather than restated, because a copy of the order here would
    agree with itself while disagreeing with the robot.
    """
    block = _rust_base_pose_block()
    order = [
        line.split("command.")[1].split(")")[0].strip()
        for line in block.splitlines()
        if "self.frame.push(command." in line
    ]
    assert order == ["base_pitch", "base_roll", "base_twist"], (
        f"the controller fills base_pose as {order}, which is not the "
        f"pitch/roll/twist this command is laid out in"
    )
    assert [PITCH, ROLL, TWIST] == [0, 1, 2], (
        "the command's own layout moved out from under the controller"
    )
    assert len(order) == POSE_DIM, (
        f"the controller pushes {len(order)} channels for a command of {POSE_DIM}"
    )


def test_the_observation_term_is_named_what_the_deployment_calls_it() -> None:
    """Pins the feature stopping one metre short.

    `scripts/export.py` builds the contract from the **actor** group and refuses
    any term name outside `_DEPLOY_TERMS`, translating only the two names it
    inherits from vendored mjlab. `base_pose` is already in that set and already
    built by the controller, so the only thing standing between this command and a
    robot that can be given it is the string the task registers it under.
    """
    import pathlib
    import re

    from tasks.registry import load_env_cfg

    src = pathlib.Path("scripts/export.py").read_text()
    deployable = set(
        re.findall(r'"([a-z_]+)"', src.split("_DEPLOY_TERMS = frozenset({")[1].split("})")[0])
    )
    rename = dict(
        re.findall(
            r'"([a-z_]+)": "([a-z_]+)"',
            src.split("_TERM_RENAME = {")[1].split("}")[0],
        )
    )
    actor = list(load_env_cfg("jumper.five_foot").observations["actor"].terms)
    assert "base_pose" in actor, (
        f"the attitude command is observed as {[n for n in actor if 'pose' in n]}, "
        "which the deployment's builder does not know"
    )
    undeployable = [n for n in actor if rename.get(n, n) not in deployable]
    assert not undeployable, (
        f"{undeployable} would make `scripts/export.py` refuse the contract"
    )


def test_the_stick_tilts_the_robot_the_way_it_is_pushed() -> None:
    """Pins the layout end to end, stick to body in the world.

    Three sign conventions sit between a stick and the nose moving -- evdev's Y
    is positive towards the operator, `controls.yaml` signs it, and `asin(g_x)` is
    positive nose-*down* -- and getting any one of them backwards produces a robot
    that tilts smoothly in the wrong direction. Nothing raises, the readout moves,
    and the operator corrects by pushing harder.

    So the check goes all the way through: push the stick, take the commanded
    angle, rotate a body to it, and ask where the nose ended up.
    """
    term = _teleop()
    for target, expect_nose_up in (("Ry+", True), ("Ry-", False)):
        _let_go(term)
        _press(term, target, 3)
        _settle(term)
        pitch_deg = float(term.command[0, PITCH]) / _DEG
        _, q = _gravity_for(pitch_deg, 0.0)
        nose_z = float(quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]]))[0, 2])
        assert (nose_z > 0) is expect_nose_up, (
            f"the stick {'back' if expect_nose_up else 'forward'} put the nose at "
            f"z={nose_z:+.3f}, which is the other way"
        )

    # Roll, the same way, with R3 held to bring the right stick's left-right
    # over to it. The stick left puts the robot's *left* side down -- the body
    # +y axis going below the horizon.
    for target, expect_left_down in (("Rx-", True), ("Rx+", False)):
        _let_go(term)
        _press(term, "R3")
        _press(term, target, 3)
        _settle(term)
        roll_deg = float(term.command[0, ROLL]) / _DEG
        _, q = _gravity_for(0.0, roll_deg)
        left_z = float(quat_apply(q, torch.tensor([[0.0, 1.0, 0.0]]))[0, 2])
        assert (left_z < 0) is expect_left_down, (
            f"the stick {'left' if expect_left_down else 'right'} put the left side "
            f"at z={left_z:+.3f}, which is the other way"
        )


def test_every_attitude_term_is_measured_against_the_command() -> None:
    """Pins the two config mistakes that make attitude quietly inaccurate.

    **A term without `command_name` charges the absolute angle**, and then it and
    the command pull in opposite directions -- the command asks for 15 degrees of
    pitch, the penalty charges 15 degrees of pitch, and the policy settles
    wherever they balance, which is neither. Nothing raises; the robot simply
    obeys attitude commands halfway.

    **The deadzone is the precision ceiling.** Inside `threshold` these terms are
    exactly zero, so nothing asks the policy to do better than the threshold. At
    the 0.03 rad this task carried before the accuracy pass, that ceiling was 1.72
    degrees. A change back would read as a tidy-up and would cap what the command
    can deliver, with `Metrics/body_pose/error_pitch` settling at the new floor
    and no test objecting.
    """
    from tasks.registry import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot")
    # **Whichever of the three this configuration registers.** The task ships more
    # than one attitude set -- the run after `2400.pt` registers `nose_pitch`
    # alone -- and the rule is about the terms that are switched on, not about a
    # fixed list of three.
    terms = {
        n: cfg.rewards[n]
        for n in ("nose_pitch", "body_roll", "body_twist")
        if n in cfg.rewards and cfg.rewards[n].weight != 0.0
    }
    assert terms, "no attitude term is registered at all; the command scores nothing"
    for name, term in terms.items():
        assert term.params.get("command_name") == "body_pose", (
            f"{name} scores the absolute angle, so it fights the attitude command "
            "instead of enforcing it"
        )
    # **The shape, not a deadzone.** These terms carried
    # `excess + 12 * excess^2` past a 0.57 to 1.72 degree band, and that priced
    # the same error **13 times higher** than kk-rl-lab's plain square: 8 degrees
    # at 3.81 per step against 0.29, 20 degrees at 23.1 against 1.83. Three runs
    # froze solid under it -- `2026-09-10_15-47-30`, `17-07-12` and `18-16-19`,
    # the last swept at iteration 2999 and still 0.000 m/s against every command
    # in six directions -- because a policy that cannot walk yet lives at exactly
    # the 5 to 12 degrees where that multiple bites hardest.
    #
    # So what is pinned is that the cost of an error is its square and nothing
    # else. A linear term riding along is invisible in any single number and is
    # precisely what came back last time; two points of the curve catch it,
    # because `a*x + b*x^2` cannot match `b*x^2` at two errors an octave apart.
    import math as _math

    from tasks.jumper.five_foot.mdp import rewards as _r

    for name, func in (("nose_pitch", _r.nose_pitch), ("body_roll", _r.body_roll)):
        if name not in terms:
            continue
        small = _math.radians(5.0)
        g_small = _gravity_for(5.0, 0.0)[0] if name == "nose_pitch" else _gravity_for(0.0, 5.0)[0]
        g_large = _gravity_for(10.0, 0.0)[0] if name == "nose_pitch" else _gravity_for(0.0, 10.0)[0]

        class _E:
            def __init__(self, g):
                self.scene = {"robot": type("A", (), {"data": type("D", (), {
                    "projected_gravity_b": g})()})()}

        c_small = float(func(_E(g_small))[0])
        c_large = float(func(_E(g_large))[0])
        assert c_small == pytest.approx(small**2, rel=1e-3), (
            f"{name} charges {c_small} at 5 degrees, not the {small**2:.5f} a plain "
            "square would; the deadzone-and-linear shape that froze three runs is back"
        )
        assert c_large / c_small == pytest.approx(4.0, rel=1e-2), (
            f"{name} grows {c_large / c_small:.2f}x between 5 and 10 degrees rather "
            "than the 4x of a square -- something linear is riding along"
        )

    # `body_twist` is the same axis treated the other way round: a Gaussian that
    # tops out at zero rather than a penalty, because a 3+2 gait rotates the trunk
    # against its own footholds by 4.9 degrees no matter what the policy does, and
    # a penalty charged that every step (-1.75) while walking earned at most 2.0.
    twist = cfg.rewards.get("body_twist")
    if twist is not None and twist.weight != 0.0:
        assert twist.weight > 0.0, (
            "body_twist is a penalty again; the gait's own 4.9 degrees of twist "
            "then costs on every step a walking robot takes"
        )
        assert twist.params["std"] >= _math.radians(10.0), (
            f"body_twist's std is {twist.params['std']} rad, tighter than the "
            "gait's own twist; walking is taxed for geometry it cannot avoid"
        )

    # And the positive term, when registered, has to track the same command.
    pose = cfg.rewards.get("track_body_pose")
    if pose is not None and pose.weight != 0.0:
        assert pose.params["command_name"] == "body_pose"
        assert pose.params["std"] <= 0.175, (
            f"the tracking kernel's std is {pose.params['std']}, wider than the "
            "0.175 this task started from; the near field got flatter, not sharper"
        )


def test_the_command_ramps_instead_of_stepping() -> None:
    """Pins the rate limit, and what it is for.

    The attitude terms are linear-plus-quadratic in the error, so the instant
    after a resample a stepped command charges for a target the trunk cannot
    reach: at 30 degrees, `nose_pitch` costs 5.1 per step at -1.5 against a whole
    velocity tracking budget of 4.0, for a robot that did nothing wrong.
    Measured on `2026-09-10_14-50-24`, which stepped: 3 degrees of steady pitch
    error and 0.433 of linear tracking, against 0.633 for the same reward set with
    no attitude command at all.

    Two things have to hold and neither shows up as an error: the command may
    never move faster than `max_rate`, and it must still *arrive* -- a ramp that
    undershoots leaves every command permanently unmet and every term permanently
    charging.
    """
    n = 4
    env = _StubEnv(n, torch.zeros(n, 3))
    cfg = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=0.0, max_rate=30.0 * _DEG
    )
    term = cfg.build(env)
    term.pose_target_b[:] = 0.0
    term.pose_target_b[:, PITCH] = 20.0 * _DEG   # a full standing-band step

    per_step = cfg.max_rate * env.step_dt
    seen = []
    for _ in range(200):
        before = term.command[0, PITCH].clone()
        term._update_command(None)
        seen.append(float(term.command[0, PITCH] - before))
    # 1e-6 rather than 0: the rate is a float64 product and the buffer is
    # float32, so the per-step delta lands a few 1e-9 either side of it.
    assert max(seen) <= per_step + 1e-6, (
        f"the command moved {max(seen) / _DEG:.2f} deg in one step, past the "
        f"{per_step / _DEG:.2f} the rate allows"
    )
    assert float(term.command[0, PITCH]) == pytest.approx(20.0 * _DEG), (
        "the ramp never arrived; every command would stay permanently unmet"
    )
    # It took about the time the rate implies -- 20 degrees at 30 deg/s is 0.67 s,
    # 34 steps -- rather than arriving instantly or crawling.
    moving = sum(1 for d in seen if d > 1e-9)
    assert 30 <= moving <= 40, f"the step took {moving} steps, not the ~34 implied"

    # **The shipped rate is 30 deg/s**, and the test has to say so: every
    # assertion above passes an explicit `max_rate`, so the default could be
    # anything -- ten times faster is most of the way back to a step, and slower
    # than about 8 deg/s cannot cross the standing band inside its own 5 second
    # window, which would leave the command permanently chasing itself.
    shipped = BodyPoseCommandCfg(resampling_time_range=(5.0, 5.0)).max_rate
    assert shipped == pytest.approx(30.0 * _DEG), (
        f"the shipped ramp is {shipped / _DEG:.0f} deg/s"
    )
    widest = 40.0 * _DEG  # -20 to +20, the standing pitch band
    assert widest / shipped < 5.0, "a full step cannot cross inside its own window"

    # **Control group: `max_rate=None` is the reference implementation**, which
    # steps. Without this branch the test above would pass against a ramp that is
    # simply always on, and the config option would be untested.
    stepping = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=0.0, max_rate=None
    ).build(_StubEnv(n, torch.zeros(n, 3)))
    stepping.pose_target_b[:, PITCH] = 20.0 * _DEG
    stepping._update_command(None)
    assert float(stepping.command[0, PITCH]) == pytest.approx(20.0 * _DEG)


def test_an_episode_starts_at_the_pose_the_robot_spawns_in() -> None:
    """Pins the transient the ramp removes coming back once per episode.

    The robot is spawned level and square. A command that begins at its freshly
    sampled value begins with up to 20 degrees of error the policy did nothing to
    earn -- once every episode, which at a 20 second episode is far more often
    than it is worth. Nothing raises: the term simply charges from step one.
    """
    n = 16
    env = _StubEnv(n, torch.zeros(n, 3))
    term = BodyPoseCommandCfg(
        resampling_time_range=(5.0, 5.0), rel_neutral_envs=0.0
    ).build(env)
    # Mid-episode: the command is somewhere out on its band. Without this the
    # buffer is already zero from construction and the test passes against a reset
    # that does nothing -- which is how it was written first.
    term.pose_command_b[:] = 0.3
    term.reset(_all_envs(n))
    assert float(term.command.abs().max()) == 0.0, (
        "the episode starts with the command already off level, which is an error "
        "charged before the first action"
    )
    # Control group: the target *was* sampled, so this is not passing because the
    # reset cleared everything.
    assert float(term.pose_target_b.abs().max()) > 0.0


def test_the_attitude_tracking_term_tops_out_at_zero() -> None:
    """Pins the one character that separates this term from the shape that parked
    this task twice.

    `track_body_pose` is kk-rl-lab's kernel and weight -- `exp(-err^2/std^2)` at
    2.5 -- **minus one**. Drop the `- 1.0` and it becomes a positive term a
    motionless robot collects in full: hold still, hold whatever attitude, take
    2.5 per step forever. Two cold starts in this task did exactly that, at 4.75
    collectable against 4.0 of velocity tracking, and the readout looked healthy
    the whole time because the attitude reward really was being earned.

    Nothing about that failure raises, and the gradient is identical either way --
    a constant differentiates away -- so no learning-curve shape gives it away
    early. Only the sign of the term at zero error does.
    """
    from tasks.registry import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot")
    term = cfg.rewards.get("track_body_pose")
    if term is None or term.weight == 0.0:
        pytest.skip("this configuration does not register the tracking term")

    def scored(err_deg: float) -> float:
        g, _ = _gravity_for(err_deg, 0.0)
        env = _StubEnv(1, torch.zeros(1, 3))
        env.scene["robot"].data.projected_gravity_b = g
        params = {k: v for k, v in term.params.items() if k != "asset_cfg"}
        return term.weight * float(term.func(env, **params)[0])

    assert scored(0.0) == pytest.approx(0.0, abs=1e-6), (
        f"holding the commanded attitude pays {scored(0.0):+.3f}; a robot that "
        "stands still and holds it collects that forever"
    )
    # Control group: the term is not simply zero everywhere -- it has to keep the
    # gradient that makes it worth having.
    assert scored(5.0) < -0.1, "the kernel is flat; there is no pull toward the command"
    assert scored(10.0) < scored(5.0), "the cost stops growing with the error"
