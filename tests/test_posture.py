"""The posture task's measurements, which are the part of it that can be wrong
quietly.

Every one of `jumper.posture`'s four measurements returns a number of the right
shape and plausible magnitude whatever it is fed. A twist estimate fitted against
a permuted footprint is an angle; one fitted against a stale reference is an
angle offset by a constant; a pitch with the wrong sign is a pitch. None of them
raises, none of them is visibly odd in a log, and all of them train a policy to
hold the wrong thing.

So each test here states which of those it pins, and the ones that could pass
vacuously carry a control group.
"""

from __future__ import annotations

import functools
import math

import pytest

torch = pytest.importorskip("torch", reason="the measurements are torch code")

from tasks.jumper.common.constants import (  # noqa: E402
    FOOT_SITE_Z,
    HOME,
    LEGS,
    NOMINAL_FOOT_XY,
    STAND_Z,
)
from tasks.jumper.posture.mdp.state import (  # noqa: E402
    reference_footprint,
    twist_from_footprint,
)


def _footprint(twist_rad: float, shift=(0.0, 0.0)) -> torch.Tensor:
    """The nominal footprint as seen from a body twisted by `twist_rad`.

    Twisting the body **left** by `t` moves every foot to `R(-t)` of where it
    was in the body frame, which is the relation the estimator has to invert.
    `shift` translates the whole footprint, as walking does.
    """
    n = torch.tensor(NOMINAL_FOOT_XY, dtype=torch.float32)
    c, s = math.cos(-twist_rad), math.sin(-twist_rad)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    return n @ rot.T + torch.tensor(shift, dtype=torch.float32)


@pytest.mark.parametrize("deg", [-15.0, -7.5, 0.0, 7.5, 15.0, 45.0])
def test_twist_reads_back_the_angle_the_footprint_was_built_from(deg: float) -> None:
    """The estimator inverts the rotation it is given, over the commanded range.

    This is the whole of the twist measurement: if it does not invert a known
    rotation, the reward is scoring something else. 45 degrees is past anything
    the task commands and is there to show the estimator is not merely linear
    near zero.
    """
    got = twist_from_footprint(_footprint(math.radians(deg)))
    assert got.item() == pytest.approx(math.radians(deg), abs=1e-5)


def test_twist_ignores_a_body_shifted_over_its_feet() -> None:
    """Walking is mostly translation, and translation must not read as twist.

    **Written expecting the centroid subtraction to be what makes this pass, and
    it is not.** The control below -- the same fit with `p`'s centroid left in --
    also returns zero, because the *reference* is stored centred and
    `sum_i n_i x (p_i + d) = sum_i n_i x p_i + (sum_i n_i) x d` with `sum_i n_i =
    0`. One centred side is enough, and `state.twist_from_footprint` centres the
    other one anyway so that the invariance does not depend on how the reference
    happens to be cached.

    So this test pins the property and not the mechanism, and the control is kept
    as the record of that: if `reference_footprint` ever stops centring, the two
    assertions come apart and the first one is what still has to hold.
    """
    shifted = _footprint(0.0, shift=(0.05, 0.02))
    assert twist_from_footprint(shifted).item() == pytest.approx(0.0, abs=1e-6)

    n = reference_footprint(shifted.device)
    assert n.mean(dim=0).abs().max().item() < 1e-6, (
        "the reference is no longer centred, so translation invariance now rests "
        "entirely on the subtraction inside twist_from_footprint"
    )
    cross = (n[:, 0] * shifted[:, 1] - n[:, 1] * shifted[:, 0]).sum()
    dot = (n[:, 0] * shifted[:, 0] + n[:, 1] * shifted[:, 1]).sum()
    assert -math.atan2(float(cross), float(dot)) == pytest.approx(0.0, abs=1e-6)


def test_a_permuted_footprint_is_not_silently_an_angle() -> None:
    """Why `state._check_foot_order` exists, demonstrated rather than asserted.

    Feeding the legs in the wrong order leaves a perfectly well-formed fit that
    returns a wrong angle instead of complaining. Nothing downstream could tell.

    **No swap of two legs shows it at all, and that is the more useful half of
    this test.** The estimator's numerator is `sum_i n_i x p_i`, and exchanging
    any two rows contributes `a x b + b x a`, which is exactly zero -- so every
    transposition reads as *no twist whatsoever*, whichever two legs it is and
    however far apart they sit. Written expecting a front-to-middle swap to show
    up where a left-right one would not; both are invisible, for that reason.

    It takes a cyclic permutation to move the answer, which is precisely why the
    order is checked by name rather than inferred from the answer looking
    sensible.
    """
    good = _footprint(0.0)
    assert twist_from_footprint(good).item() == pytest.approx(0.0, abs=1e-6)

    for a, b in ((0, 1), (0, 2), (2, 5)):  # LF<->RF, LF<->LM, LM<->RR
        swapped = good.clone()
        swapped[[a, b]] = swapped[[b, a]]
        assert twist_from_footprint(swapped).item() == pytest.approx(0.0, abs=1e-6)

    rotated = good[[1, 2, 3, 4, 5, 0]]
    assert abs(twist_from_footprint(rotated).item()) > math.radians(5.0), (
        "even a cyclic relabelling of the legs reads as no twist, which would "
        "leave the order check with nothing at all to catch"
    )


def test_the_nominal_footprint_still_describes_the_model() -> None:
    """`NOMINAL_FOOT_XY` and `FOOT_SITE_Z` are measurements off `jumper.xml`.

    A robot revision replaces that file in place. A stale footprint here does not
    raise; it redefines zero twist as a few degrees of twist and zero height
    error as a few millimetres, and the policy is then trained to hold the bias.
    """
    mujoco = pytest.importorskip("mujoco", reason="re-deriving needs the model")

    model = mujoco.MjModel.from_xml_path("assets/jumper/jumper.xml")
    data = mujoco.MjData(model)
    for joint, value in HOME.items():
        data.qpos[model.joint(joint).qposadr[0]] = value
    mujoco.mj_forward(model, data)

    base_id = model.body("base_link").id
    base_pos = data.xpos[base_id]
    base_mat = data.xmat[base_id].reshape(3, 3)

    depths = []
    for leg, expected in zip(LEGS, NOMINAL_FOOT_XY):
        local = base_mat.T @ (data.site_xpos[model.site(leg).id] - base_pos)
        assert float(local[0]) == pytest.approx(expected[0], abs=1e-3), f"{leg} x"
        assert float(local[1]) == pytest.approx(expected[1], abs=1e-3), f"{leg} y"
        depths.append(float(local[2]))

    mean_depth = sum(depths) / len(depths)
    assert STAND_Z + mean_depth == pytest.approx(FOOT_SITE_Z, abs=1e-4), (
        "FOOT_SITE_Z no longer makes a base-minus-feet height read STAND_Z at "
        "HOME, so the posture height command is quoted in the wrong units"
    )


def test_the_mirror_of_a_posture_is_the_posture_of_the_mirror() -> None:
    """`posture4` negates the twist and the roll and keeps pitch and height.

    Stated as the sign table because that is what the mirror is, and checked
    against the geometry it comes from: mirroring the *footprint* in y has to
    negate the twist the estimator reports. A wrong entry here trains the actor
    on samples that say "lean left" beside a body leaning right, and the shapes
    match either way.
    """
    from tasks.jumper.common.mdp.symmetry import _mirror_slice

    posture = torch.tensor([[0.2, -0.1, 0.3, 0.02]])
    got = _mirror_slice(posture, "posture4")
    assert got[0].tolist() == pytest.approx([-0.2, -0.1, -0.3, 0.02], abs=1e-6)

    # The geometric half: a footprint mirrored in y is a robot twisted the other
    # way. Note the leg order mirrors too -- LF<->RF, LM<->RM, LR<->RR -- which is
    # the permutation `symmetry.LEG_PERM` carries for per-foot quantities.
    from tasks.jumper.common.mdp.symmetry import LEG_PERM

    flip = torch.tensor([1.0, -1.0])
    twisted = _footprint(math.radians(10.0))
    mirrored = twisted[LEG_PERM] * flip
    # The robot's own footprint is not exactly mirror symmetric -- LF stands at
    # (0.1871, 0.0728) against RF's (0.1836, -0.0690), 3.5 mm apart fore-aft --
    # so the mirror of the *nominal* footprint already reads as a twist: 1.68e-3
    # rad, 0.096 degrees, off `NOMINAL_FOOT_XY` on 2026-10-08. That is the
    # model's asymmetry, not the estimator's error, and a tolerance tight enough
    # to object to it pins the URDF's rounding -- which this test did, at 1e-4,
    # written against an earlier model whose feet sat 0.2 mm apart, and it
    # failed the day the model moved while the estimator had not changed.
    #
    # So the control group is the mirrored nominal footprint itself: what the
    # estimator reads off it is the model's bias, and the mirror of a twist has
    # to land exactly minus the twist away from that bias (6e-8 rad, measured
    # from -45 to 45 degrees). The bias is still bounded, separately, so a model
    # that stops being nearly symmetric is a failure here and not a silent shift
    # of every mirrored training sample.
    bias = twist_from_footprint(_footprint(0.0)[LEG_PERM] * flip).item()
    assert abs(bias) < math.radians(0.5), (
        f"the mirrored nominal footprint reads {math.degrees(bias):.3f} degrees of "
        f"twist; the model's left-right asymmetry is no longer small enough to call "
        f"rounding, and every mirrored sample carries it"
    )
    assert twist_from_footprint(mirrored).item() - bias == pytest.approx(
        -math.radians(10.0), abs=1e-5
    )


# ──────────────────────────────────────────────────────────────────────
# The curriculum
# ──────────────────────────────────────────────────────────────────────


def test_every_axis_climbs_the_ladder_at_the_same_rate() -> None:
    """One level is one rung on all four axes, and the top rung is the full range.

    The rungs themselves (7.5, 10, 15 degrees) are not asserted here: they are a
    tuning decision with a measurement behind them in
    `POSTURE_LEVEL_FRACTIONS`, and a test that restates them only means the
    constant has not been edited. What has to hold whatever they are is that the
    height climbs with the angles and that the ladder ends where the command
    ends.

    The height's excursion is asymmetric -- 37 mm of squat against 13 mm of
    extension -- so "the same rate" has to mean the same *fraction* of each side
    rather than the same millimetres. A ladder that moved both sides by a fixed
    step would reach the full squat a rung before the full extension, and a level
    would then mean two different things on the two halves of one axis.
    """
    from tasks.jumper.posture.env_cfg import HEIGHT_RANGE, MOVE_LEAN
    from tasks.jumper.posture.mdp.curriculum import (
        POSTURE_LEVEL_FRACTIONS,
        posture_levels,
    )

    angles, heights = posture_levels(MOVE_LEAN, HEIGHT_RANGE, STAND_Z)

    assert POSTURE_LEVEL_FRACTIONS[-1] == 1.0
    assert angles[-1] == pytest.approx(MOVE_LEAN)
    assert heights[-1] == pytest.approx(HEIGHT_RANGE)
    assert list(angles) == sorted(angles), "the ladder has to widen, not wander"

    for level, fraction in enumerate(POSTURE_LEVEL_FRACTIONS):
        low, high = heights[level]
        assert angles[level] == pytest.approx(MOVE_LEAN * fraction)
        assert STAND_Z - low == pytest.approx(
            (STAND_Z - HEIGHT_RANGE[0]) * fraction, abs=1e-9
        )
        assert high - STAND_Z == pytest.approx(
            (HEIGHT_RANGE[1] - STAND_Z) * fraction, abs=1e-9
        )


def test_the_ruler_does_not_change_length_as_the_ladder_widens() -> None:
    """What a robot that never leans collects must not depend on the level.

    This is the property the whole scaled-std argument rests on, and it is the one
    that decides whether a range curriculum is legitimate at all: narrowing the
    range without narrowing `std` does not make the task easier to do, it makes it
    easier to *score*, and this robot's documented local optimum is collecting
    reward for doing nothing.

    Measured the way `common/mdp/curriculum.py` measures it -- score a robot
    holding the neutral posture against commands drawn uniformly from each level's
    own range.

    **The control group is the same sweep with `std` pinned at the top rung's
    value**, which is how this would look if the scaling were dropped. It has to
    fail for the assertion above it to mean anything.
    """
    from tasks.jumper.posture.env_cfg import HEIGHT_RANGE, MOVE_LEAN
    from tasks.jumper.posture.mdp.curriculum import (
        angle_std,
        height_std,
        posture_levels,
    )

    angles, heights = posture_levels(MOVE_LEAN, HEIGHT_RANGE, STAND_Z)
    generator = torch.Generator().manual_seed(0)

    def free_score(level: int, angle_std_: float, height_std_: float) -> float:
        """What holding the neutral posture collects, on all three terms."""
        n = 200_000
        half = angles[level]
        twist = torch.empty(n).uniform_(-half, half, generator=generator)
        pitch = torch.empty(n).uniform_(-half, half, generator=generator)
        roll = torch.empty(n).uniform_(-half, half, generator=generator)
        low, high = heights[level]
        height = torch.empty(n).uniform_(low, high, generator=generator) - STAND_Z
        # The error of a robot that holds neutral is the command itself.
        return float(
            torch.exp(-twist.square() / angle_std_**2).mean()
            + torch.exp(-(pitch.square() + roll.square()) / angle_std_**2).mean()
            + torch.exp(-height.square() / height_std_**2).mean()
        )

    scaled = [
        free_score(level, angle_std(level, angles), height_std(level, heights))
        for level in range(len(angles))
    ]
    spread = (max(scaled) - min(scaled)) / min(scaled)
    assert spread < 0.05, (
        f"the free score moves {spread:.1%} across the ladder, so a level changes "
        f"how strictly the posture is marked as well as how far it is asked to "
        f"lean: {['%.4f' % s for s in scaled]}"
    )

    pinned = [
        free_score(level, angle_std(-1, angles), height_std(-1, heights))
        for level in range(len(angles))
    ]
    assert (max(pinned) - min(pinned)) / min(pinned) > 0.2, (
        "with std pinned at the top rung the free score barely moves either, so "
        "the test above cannot tell a scaled ruler from a fixed one"
    )


def test_a_promotion_bar_sits_between_the_noise_and_doing_nothing() -> None:
    """Each bar has to be above the measured floor and below what neutral gives.

    Either side of that window is a bar that measures nothing. Below the floor and
    no policy can clear it -- the failure `common/mdp/curriculum.py` records
    costing a run all 37500 of its iterations at level 0. Above what holding the
    neutral posture scores, and the level promotes without anything having been
    learned.

    The floor numbers are the exploration-noise component measured in that
    module's docstring (sigma 0.8 minus sigma 0), not the raw sigma-0.8 row: the
    static part of that row is a sag the policy removes.
    """
    from tasks.jumper.posture.env_cfg import HEIGHT_RANGE, MOVE_LEAN
    from tasks.jumper.posture.mdp.curriculum import (
        GATE_FLOOR_ANGLE,
        GATE_FLOOR_HEIGHT,
        GATE_RATIO,
        posture_levels,
    )

    angles, heights = posture_levels(MOVE_LEAN, HEIGHT_RANGE, STAND_Z)
    # Measured, 128 envs at action sigma 0.8 minus the same at sigma 0.
    noise_angle = math.radians(1.3)  # the worst axis, pitch
    noise_height = 0.0015

    for level in range(len(angles) - 1):  # the top rung's bar never gates
        half = angles[level]
        low, high = heights[level]

        angle_bar = GATE_FLOOR_ANGLE + GATE_RATIO * half
        # A robot ignoring the command sits at the mean |command|, which for a
        # uniform draw over +/- half is half/2.
        assert noise_angle < angle_bar < half / 2.0, (
            f"level {level}: the angle bar {math.degrees(angle_bar):.2f} deg is "
            f"not between the noise {math.degrees(noise_angle):.2f} and what "
            f"holding neutral gives, {math.degrees(half / 2.0):.2f}"
        )

        height_bar = GATE_FLOOR_HEIGHT + GATE_RATIO * (high - low) / 2.0
        # Mean |h - neutral| for h uniform over an asymmetric range.
        neutral_err = ((STAND_Z - low) ** 2 + (high - STAND_Z) ** 2) / (
            2.0 * (high - low)
        )
        assert noise_height < height_bar < neutral_err, (
            f"level {level}: the height bar {height_bar * 1000:.2f} mm is not "
            f"between the noise {noise_height * 1000:.2f} and what holding "
            f"neutral gives, {neutral_err * 1000:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────
# Teleoperation
# ──────────────────────────────────────────────────────────────────────
#
# One person drives both commands, from a pad or from the keyboard, through
# `common/mdp/operator.py`. Everything below reads the task's own
# `controls.yaml` and the replay config's own ranges, so it checks what an
# operator actually gets rather than a copy of it.


@functools.lru_cache(maxsize=1)
def _replay_cfg():
    import tasks

    return tasks.load_env_cfg("jumper.posture", play=True)


def _operator(controls=None, pad=None):
    """The task's operator, both commands attached at their replay ranges.

    Built directly rather than through an environment, and not listening to the
    viewer, so a test feeds it keys itself and leaves no handler behind.
    """
    from tasks.jumper.common.mdp.controls import load_controls
    from tasks.jumper.common.mdp.operator import Operator, spans
    from tasks.jumper.posture.mdp.teleop import CONTROLS

    controls = controls or load_controls(CONTROLS)
    cfg = _replay_cfg()
    op = Operator(controls, pad, listen=False, clock=_Clock())
    for term in ("twist", "posture"):
        op.attach(term, spans(controls.command(term), cfg.commands[term]))
    return op


class _Clock:
    """Seconds, moved by the test: what a key's hold is measured by."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _keys_of(stroke: str) -> list[str]:
    """The dictionary keys a keystroke is held with: `shift+key_j` is a Shift
    and J, `ctrl` one Ctrl."""
    from controller import vocabulary

    modifier, key = vocabulary.keystroke(stroke)
    mods = vocabulary.modifiers()
    return [k if k not in mods else mods[k][0] for k in (modifier, key) if k is not None]


def _press(op, stroke: str, times: int = 1) -> None:
    """`times` tenths of `full_after_s` of `stroke` held, the test's clock
    stopped there with its keys still down. Called again for the same
    keystroke, the hold goes on from where it was."""
    for k in _keys_of(stroke):
        op.key(k, True)
    op._clock.t += times * op.controls.full_after_s / 10


def _let_up(op, stroke: str) -> None:
    """Let up the keys `stroke` is held with."""
    for k in _keys_of(stroke):
        op.key(k, False)


class _Pad:
    """A pad whose state a test sets directly."""

    name = "test pad"

    def __init__(self):
        from controller.xbox import GamepadState

        self.s = GamepadState()

    def state(self):
        return self.s


def _frame(op, pad, **state):
    """One step with the pad held as given: both commands, as the terms ask."""
    from controller.xbox import GamepadState

    pad.s = GamepadState(**state)
    return op.command("twist"), op.command("posture")


def _edges(op) -> dict[str, tuple[float, float, float]]:
    return {a: span for term in ("twist", "posture") for a, span in op._spans[term].items()}


def test_the_pad_alone_reaches_both_ends_of_every_channel() -> None:
    """Every command axis, both directions, from pad controls only.

    The failure this pins is a channel no control reaches -- a binding lost in
    an edit, or a stick claimed by another axis over its whole travel. Nothing
    raises, and it shows up as a robot that will not lean one way.

    Each binding is driven where its climb ends, `travel[1]`: the full
    deflection for one that holds there, and half the stick for the twist,
    which is back at zero by the time the stick is all the way over.
    """
    from tasks.jumper.common.mdp.controls import _BIPOLAR
    from tasks.jumper.common.mdp.operator import deflections

    op = _operator()
    reached: dict[str, set[float]] = {a: set() for a in op.controls.bindings}
    for parts in op.controls.bindings.values():
        for b in parts:
            full = b.travel[1]
            for d in (full, -full) if b.source in _BIPOLAR else (full,):
                values = dict.fromkeys(("Lx", "Ly", "Rx", "Ry", "LT", "RT"), 0.0)
                values[b.source] = d
                for axis, x in deflections(values, b.shifted, op.controls).items():
                    reached[axis].add(round(x, 9))
    for axis, seen in reached.items():
        assert {1.0, -1.0} <= seen, f"the pad cannot reach both ends of {axis}: {sorted(seen)}"


def test_the_keyboard_alone_reaches_both_ends_of_every_channel() -> None:
    """The same from the keys, read back as the command a term is given.

    Each end is driven from hands-off by holding each keystroke the file puts on
    that side of the axis -- its "+" towards the top of the range, its "-"
    towards the bottom -- and compared with the edge of the range the replay
    config clamps to. The height moves rather than is placed, so it is held on
    until it gets there.
    """
    op = _operator()
    term_of = {a.name: c.term for c in op.controls.commands for a in c.axes}
    index = {a.name: i for c in op.controls.commands for i, a in enumerate(c.axes)}
    for axis, keys in op.controls.key_axes.items():
        lo, hi, _rest = _edges(op)[axis]
        for strokes, want in ((keys.plus, hi), (keys.minus, lo)):
            for stroke in strokes:
                fresh = _operator()
                fresh.command("twist")  # the clock's first step: nothing moves yet
                _press(fresh, stroke, 10)
                if axis in fresh.controls.integrating:
                    fresh.command(term_of[axis])
                    _press(fresh, stroke, 10)
                got = fresh.command(term_of[axis])[index[axis]]
                assert got == pytest.approx(want, abs=1e-6), f"{stroke} drives {axis} to {got}, not {want}"


def test_the_keyboard_s_directions_are_the_file_s_words() -> None:
    """Swap the keys of `lin_vel_x` in the file, and W walks backwards.

    The control group for the reach test above: a keyboard with directions
    written into the operator would pass it by agreeing with the file by
    chance. Since 2026-09-29 the keyboard is not the pad's copy and has signs
    of its own -- "+" is the axis's positive as the file words it -- so this is
    the assertion that those words are what it reads.
    """
    from dataclasses import replace

    op = _operator()
    _press(op, "key_w", 10)
    forward = op.command("twist")[0]

    keys = dict(op.controls.key_axes)
    keys["lin_vel_x"] = replace(keys["lin_vel_x"], plus=keys["lin_vel_x"].minus,
                                minus=keys["lin_vel_x"].plus)
    flipped = _operator(replace(op.controls, key_axes=keys))
    _press(flipped, "key_w", 10)
    assert forward > 0.0, "W did not walk forward, so the swap below proves nothing"
    assert flipped.command("twist")[0] == pytest.approx(-forward)


@pytest.mark.parametrize(
    "rx, twist, turn",
    [
        (0.25, -0.5, 0.0),
        (0.5, -1.0, 0.0),     # the split: full twist, not yet turning
        (0.625, -0.5, -0.25),
        (0.75, 0.0, -0.5),    # the twist is gone at half the top turn rate
        (1.0, 0.0, -1.0),
        (-0.625, 0.5, 0.25),  # and the same to the left
    ],
)
def test_the_right_stick_twists_first_and_turns_second(rx, twist, turn) -> None:
    """The first half of the travel twists, the second half turns, and the twist
    unwinds while the turn climbs, to zero where the turn is at half its top
    rate. Right is clockwise, negative for both.

    What it guards against is the split drifting off half travel, the order
    reversing, or the twist holding on into a fast turn, each of which is a
    working stick that does something else.
    """
    from tasks.jumper.common.mdp.operator import deflections

    op = _operator()
    values = dict.fromkeys(("Lx", "Ly", "Rx", "Ry", "LT", "RT"), 0.0)
    values["Rx"] = rx
    x = deflections(values, False, op.controls)
    assert x["twist"] == pytest.approx(twist)
    assert x["ang_vel_z"] == pytest.approx(turn)
    assert x["roll"] == 0.0


def test_j_turns_and_h_twists() -> None:
    """On the keyboard the right stick's two halves are two keys: J turns and
    H twists, both to the left, as Control-agent 3.1 lays them out. Each is
    the other's control group; ; twists the other way."""
    op = _operator()
    wz_max = _edges(op)["ang_vel_z"][1]
    twist_lo, twist_max = _edges(op)["twist"][:2]
    _press(op, "key_j", 10)
    assert op.command("twist")[2] == pytest.approx(wz_max), "J turns left, counter-clockwise"
    assert op.command("posture")[0] == 0.0, "J alone twisted"
    _let_up(op, "key_j")
    _press(op, "key_h", 10)
    assert op.command("posture")[0] == pytest.approx(twist_max), "H twists left"
    assert op.command("twist")[2] == 0.0, "H turned as well"
    _let_up(op, "key_h")
    _press(op, "key_semicolon", 10)
    assert op.command("posture")[0] == pytest.approx(twist_lo), "; twists right"


def test_hands_off_is_the_neutral_posture_rather_than_zero() -> None:
    """A centred pad stands still, level, square and at standing height.

    Zero height is a body on the floor, and a range starting at 0.07 m would
    clamp it there -- so without `rest` every release of the pad would squat
    the robot, and the robot squatting looks like a policy failing.

    The control group is the same file with `rest` taken off the height: it is
    refused when the term's ranges are read, rather than commanding the clamp.
    """
    from dataclasses import replace

    from tasks.jumper.common.mdp.operator import spans

    pad = _Pad()
    op = _operator(pad=pad)
    _frame(op, pad, lx=0.5, rt=1.0)
    vel, post = _frame(op, pad)
    neutral = _replay_cfg().commands["posture"].neutral_height
    assert vel == [0.0, 0.0, 0.0]
    assert post == pytest.approx([0.0, 0.0, 0.0, neutral])

    command = op.controls.command("posture")
    bare = replace(command, axes=tuple(
        replace(a, rest=None) if a.name == "height" else a for a in command.axes
    ))
    with pytest.raises(ValueError, match="rests at 0"):
        spans(bare, _replay_cfg().commands["posture"])


def test_the_device_touched_last_drives_and_the_pad_lets_go_of_the_keys() -> None:
    """Both devices live; the last one touched drives.

    The part worth pinning is the hand-back: touching the pad drops the
    keyboard's holds, so letting go of the pad leaves the robot at the neutral
    command rather than reviving what the keys had set before -- which would be
    a robot walking off on its own while nobody touches anything.
    """
    pad = _Pad()
    op = _operator(pad=pad)
    vx_max = _edges(op)["lin_vel_x"][1]

    _press(op, "key_w", 5)
    assert op.command("twist")[0] == pytest.approx(0.5 * vx_max), "the keys drive an idle pad"
    vel, _ = _frame(op, pad, lx=-1.0)
    assert vel[0] == 0.0 and vel[1] > 0.0, "the pad in hand wins"
    vel, _ = _frame(op, pad)
    assert vel == [0.0, 0.0, 0.0], "letting go of the pad brought back the keys' command"
    _let_up(op, "key_w")
    _press(op, "key_w")
    assert op.command("twist")[0] == pytest.approx(0.1 * vx_max), "the keys start from rest"


def test_one_release_hands_both_commands_back() -> None:
    """`B` gives the walk and the posture back to the sampler together.

    Two releases would let an operator hand one command back and go on driving
    the other without noticing which.
    """
    op = _operator()
    _press(op, "key_w")
    _press(op, "key_i")
    assert op.command("twist") is not None and op.command("posture") is not None
    _press(op, op.controls.key_release[0])
    assert op.command("twist") is None and op.command("posture") is None


def test_a_command_with_no_term_stops_play_rather_than_driving_nothing() -> None:
    """A command the file describes and no term takes keeps its bindings in the
    file and in the banner and does nothing at all. The operator refuses on the
    first step instead."""
    from tasks.jumper.common.mdp.controls import load_controls
    from tasks.jumper.common.mdp.operator import Operator, spans
    from tasks.jumper.posture.mdp.teleop import CONTROLS

    controls = load_controls(CONTROLS)
    op = Operator(controls, listen=False)
    op.attach("twist", spans(controls.command("twist"), _replay_cfg().commands["twist"]))
    with pytest.raises(RuntimeError, match="posture"):
        op.command("twist")


def test_replay_hands_both_commands_to_one_operator() -> None:
    """Both terms are the operator's in `play`, read one file and want one pad.

    `gamepad=False` is what this replaced: a pad took the walk and left the
    posture on the keyboard, so the letters silently stopped driving half of
    it. The training config is the control group -- it must still sample.
    """
    import tasks
    from tasks.jumper.common.mdp.operator import OperatorVelocityCommandCfg
    from tasks.jumper.posture.env_cfg import POSTURE_COMMAND
    from tasks.jumper.posture.mdp.teleop import CONTROLS, TeleopPostureCommandCfg

    replay = _replay_cfg()
    twist, posture = replay.commands["twist"], replay.commands[POSTURE_COMMAND]
    assert isinstance(twist, OperatorVelocityCommandCfg)
    assert isinstance(posture, TeleopPostureCommandCfg)
    assert twist.controls.source == posture.controls.source == CONTROLS
    assert (twist.pad, twist.pad_name) == (posture.pad, posture.pad_name)
    assert twist.pad, "the pad is off for this task again"

    train = tasks.load_env_cfg("jumper.posture", play=False)
    assert not isinstance(train.commands["twist"], OperatorVelocityCommandCfg)
    assert not isinstance(train.commands[POSTURE_COMMAND], TeleopPostureCommandCfg)


def test_every_letter_this_task_binds_is_also_a_viewer_shortcut() -> None:
    """The cost of letter keys, asserted so that it cannot be forgotten.

    MuJoCo's viewer handles its own shortcuts in C++ and calls the user callback
    **in addition**, so every letter this task binds also toggles a render flag.
    That is a known and accepted cost -- see `posture/mdp/teleop.py` -- and this
    test exists so that the next person to read the bindings finds out from the
    suite rather than from a strobing viewport.

    The control group is `W`: a table read that found nothing would make the
    main assertion vacuous, and wireframe on `W` is the one everybody has seen.
    """
    mujoco = pytest.importorskip("mujoco", reason="the flag tables are mujoco's")

    from controller import vocabulary
    from tasks.jumper.common.mdp.controls import load_controls
    from tasks.jumper.posture.mdp.teleop import CONTROLS

    bound = {
        entry[2].upper(): entry[0]
        for table in (mujoco.mjVISSTRING, mujoco.mjRNDSTRING)
        for entry in table
        if entry[2].strip()
    }
    assert bound.get("W", "").lower() == "wireframe", "the viewer's shortcut table was not read"

    keys = {vocabulary.keystroke(s)[1] for s in load_controls(CONTROLS).keystrokes}
    letters = [k[len("key_"):].upper() for k in keys if len(k) == len("key_") + 1]
    assert letters, "the letter bindings have gone; this test is about them"
    assert all(letter in bound for letter in letters), (
        "a letter this task binds is no longer a viewer shortcut -- good news, "
        "and the docstring in posture/mdp/teleop.py needs to stop saying it is: "
        f"{sorted(set(letters) - set(bound))}"
    )


def _export_module():
    """`scripts/export.py`, imported the way `test_jump_reference.py` does it."""
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    cs = importlib.util.spec_from_file_location("_cli", scripts / "_cli.py")
    cli = importlib.util.module_from_spec(cs)
    sys.modules["_cli"] = cli
    cs.loader.exec_module(cli)
    spec = importlib.util.spec_from_file_location("_export_for_test", scripts / "export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_export_admits_posture_only_with_what_the_controller_needs() -> None:
    """Each of these terms is built on the robot from parameters its contract
    carries, and a missing one would be built from a default: a gait clock at
    the host's 0.32 s while this policy's tempo follows its speed, or a posture
    height 0.107 m out for ever. The export refuses both rather than packing
    them; the controller refuses them too, later and further from the cause.
    """
    mod = _export_module()
    law = {"stride": 0.072, "freq_min": 2.0, "freq_max": 5.56, "turn_radius": 0.2,
           "command_threshold": 0.05}

    def contract(gait=law, posture={"neutral_height": 0.107}):
        return {"obs_joint_order": ["a"], "observation": {"terms": [
            {"name": "gait_phase", "dim": 2, "params": dict(gait)},
            {"name": "posture_command", "dim": 4, "params": dict(posture)},
        ]}}

    mod._validate_deployable(contract())  # the control: the real shape passes
    mod._validate_deployable(contract(gait={"period": 0.32}))  # a fixed clock too
    with pytest.raises(SystemExit, match="neither `period` nor the cadence law"):
        mod._validate_deployable(contract(gait={"command_threshold": 0.05}))
    with pytest.raises(SystemExit, match="neutral_height"):
        mod._validate_deployable(contract(posture={}))


def test_the_export_opens_a_half_range_into_a_pair() -> None:
    """`jumper.posture`'s angles are one number each, symmetric by construction.

    Skipped -- which is what the export did with any range that was not a pair --
    a stick on those axes has nothing to scale to on the robot, and the
    controller refuses the contract for it. The height, already a pair, is the
    control group: it must come through as it is.
    """
    import types

    import tasks

    mod = _export_module()
    cfg = tasks.load_env_cfg("jumper.posture")
    env = types.SimpleNamespace(command_manager=types.SimpleNamespace(
        active_terms=["twist", "posture"]))
    ranges = mod._command_ranges(env, cfg)
    posture = cfg.commands["posture"].ranges
    for axis in ("twist", "pitch", "roll"):
        half = getattr(posture, axis)
        assert ranges["posture"][axis] == pytest.approx([-half, half]), axis
    assert ranges["posture"]["height"] == pytest.approx(list(posture.height))
    assert set(ranges["twist"]) >= {"lin_vel_x", "lin_vel_y", "ang_vel_z"}


def test_replay_commands_the_full_range_the_policy_was_trained_on() -> None:
    """No curriculum in `play`, so the operator is not clamped to rung zero.

    **This is a regression test for a shipped bug.** The posture curriculum is
    installed after `velocity_env_cfg` returns, and that function ends with
    `cfg.curriculum = {}` under `play` -- which drops the velocity and terrain
    ladders for a replay and did not drop this one, because it was not there yet.
    So replay ran the curriculum, its first call wrote level 0's ranges into the
    command term, and the teleop -- which clamps the operator to `cfg.ranges` --
    would not go past 7.5 degrees.

    Nothing raised. The robot simply stopped leaning, which reads as the policy's
    limit rather than as the command having been narrowed underneath it, and the
    only way to find it is to know what the number should have been.

    The control group is the training config: the curriculum has to be there, or
    this test would pass against a task that had simply lost its ladder.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import (
        HEIGHT_RANGE,
        MOVE_LEAN,
        POSTURE_COMMAND,
        STAND_PITCH,
        STAND_ROLL,
        STAND_TWIST,
    )

    train = tasks.load_env_cfg("jumper.posture", play=False)
    assert POSTURE_COMMAND in train.curriculum, (
        "the training config has no posture curriculum, so the assertion below "
        "about replay not having one proves nothing"
    )

    replay = tasks.load_env_cfg("jumper.posture", play=True)
    assert POSTURE_COMMAND not in replay.curriculum, (
        "a range curriculum in replay silently narrows what the operator can "
        "command, and the first thing it writes is the narrowest rung"
    )

    term = replay.commands[POSTURE_COMMAND]
    top = {"twist": STAND_TWIST, "pitch": STAND_PITCH, "roll": STAND_ROLL}
    for axis in ("twist", "pitch", "roll"):
        assert getattr(term.ranges, axis) == pytest.approx(top[axis]), (
            f"replay clamps {axis} to something other than the full standing band"
        )
        assert getattr(term.moving, axis) == pytest.approx(MOVE_LEAN), (
            f"replay holds a walking {axis} to something other than the full "
            "walking band"
        )
    assert term.ranges.height == pytest.approx(HEIGHT_RANGE)


def test_the_gait_clock_divides_the_control_rate_evenly() -> None:
    """A cycle has to be a whole, even number of control steps -- **when it is fixed**.

    `tripod/mdp/phase.py` derives why and measures the cost: a frequency that is
    not a whole number of steps splits the two tripod groups unevenly -- 12.2%
    apart at 3.571 Hz -- and a standing left-right bias is exactly what the mirror
    augmentation assumes is not there. Even, because the two halves of the cycle
    are the two groups.

    **Under `VARIABLE_CADENCE` the premise is gone**: the cadence follows the
    command, so there is no single steps-per-cycle to be whole or even, and the
    property that replaces this one is measured over a command sequence by
    `test_the_tripod_split_is_even_over_a_command_sequence`. What is checked here
    instead is the two ends of the clamp -- the cadences the law *can* hold
    steady at, which is where the parity still bites.

    The rest of it stays for the case that matters most: `VARIABLE_CADENCE` going
    back to False, which is one line and restores every word of the paragraph
    above. This task sets three of the four numbers involved and they arrive from
    different files, so nothing else would say the fixed clock had become illegal.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import (
        DECIMATION,
        GAIT_FREQ_HZ,
        GAIT_FREQ_MIN_HZ,
        SIM_DT,
        VARIABLE_CADENCE,
    )

    cfg = tasks.load_env_cfg("jumper.posture")
    assert cfg.sim.mujoco.timestep == SIM_DT
    assert cfg.decimation == DECIMATION

    step_dt = SIM_DT * DECIMATION

    def steps_per_cycle(freq: float) -> float:
        return 1.0 / (freq * step_dt)

    if not VARIABLE_CADENCE:
        n = steps_per_cycle(GAIT_FREQ_HZ)
        assert n == pytest.approx(round(n), abs=1e-9), (
            f"{GAIT_FREQ_HZ} Hz at {1 / step_dt:.0f} Hz control is {n} control "
            f"steps per cycle, which is not whole"
        )
        assert round(n) % 2 == 0, (
            f"{round(n)} steps per cycle is odd, so the two tripod groups cannot "
            f"get the same number of them"
        )

    # The floor is a cadence the law holds steady at -- every command below
    # `2 * stride * freq_min` is clamped to it -- so its parity is not academic.
    floor = steps_per_cycle(GAIT_FREQ_MIN_HZ)
    assert floor == pytest.approx(round(floor), abs=1e-9) and round(floor) % 2 == 0, (
        f"the cadence floor is {floor} control steps per cycle, and every slow "
        f"command sits exactly on it"
    )

    # **The ceiling is the other cadence the law holds steady at**, and it is
    # what `GAIT_STRIDE` is chosen against: at a fixed top speed the stride can
    # only take the values that make this even. 72 mm gives 18; the 60 mm that
    # was wanted gives 15, and `cadence.py` measures the 53.3% / 46.7% split that
    # costs for as long as a command is held at full stick.
    ceiling = steps_per_cycle(GAIT_FREQ_HZ)
    assert ceiling == pytest.approx(round(ceiling), abs=1e-9), (
        f"the cadence ceiling is {ceiling} control steps per cycle, not whole"
    )
    assert round(ceiling) % 2 == 0, (
        f"the cadence ceiling is {round(ceiling)} steps per cycle, odd -- a "
        f"command held at the ceiling then asks for one tripod group more often "
        f"than the other, for as long as it is held"
    )

    # The swing itself, which is what the resolution argument is about, at the
    # cadence that gives the fewest steps.
    swing = round(ceiling) / 2
    assert swing >= 8.0, (
        f"a swing at the top cadence is {swing} control steps; phase.py names 5 "
        f"as the practical floor and this is the number to raise the control rate "
        f"for, not the number to lower the cadence for"
    )


def test_the_self_collision_sensor_buffers_every_substep() -> None:
    """One entry per physics substep, and the skeleton sizes it before we retune.

    `velocity_env_cfg` builds the sensor with `history_length=cfg.decimation`
    **while constructing**, so a task that changes `decimation` afterwards --
    this one does, 4 to 5 -- leaves the buffer one short. The term then counts
    four of five substeps and reports a collision rate that is low by a fifth,
    which is a number that looks like a slightly better policy.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import DECIMATION

    cfg = tasks.load_env_cfg("jumper.posture")
    sensors = [s for s in (cfg.scene.sensors or ()) if s.name == "self_collision"]
    assert sensors, "the self-collision sensor has gone"
    assert sensors[0].history_length == DECIMATION


def test_the_cadence_follows_the_command_and_the_turn_is_in_it() -> None:
    """The law, and the axis it is easiest to leave out.

    `moving_gate` counts a pure turn as motion -- it reads `norm(cmd[:, :3])` --
    so the gait reward demands stepping for a command with no linear component at
    all. A cadence law reading only `|v_xy|` would hand that command the floor
    while the robot has to swing its outermost foot at `wz * r` to deliver it,
    and nothing would raise: the gait would simply be too slow to turn at, which
    reads as a policy that cannot turn.
    """
    import torch

    from tasks.jumper.posture.env_cfg import (
        COMMAND_ANG_CEILING,
        GAIT_FREQ_HZ,
        GAIT_FREQ_MIN_HZ,
        GAIT_STRIDE,
        GAIT_TURN_RADIUS,
    )
    from tasks.jumper.posture.mdp.cadence import gait_frequency

    class _Env:
        def __init__(self, cmd):
            self.command_manager = type(
                "M", (), {"get_command": staticmethod(lambda n, c=cmd: c)}
            )()

    def freq(vx, vy, wz):
        env = _Env(torch.tensor([[vx, vy, wz]]))
        return float(
            gait_frequency(
                env, "twist", GAIT_STRIDE, GAIT_FREQ_MIN_HZ, GAIT_FREQ_HZ,
                GAIT_TURN_RADIUS,
            )[0]
        )

    # The ceiling is what the top cadence was derived from, so it has to land on
    # it exactly rather than near it.
    assert freq(0.8, 0.0, 0.0) == pytest.approx(GAIT_FREQ_HZ)
    assert freq(0.0, 0.0, 0.0) == pytest.approx(GAIT_FREQ_MIN_HZ)
    assert freq(0.4, 0.0, 0.0) == pytest.approx(0.4 / (2 * GAIT_STRIDE))
    # Lateral counts the same as forward: the legs do not know the difference.
    assert freq(0.0, 0.4, 0.0) == pytest.approx(freq(0.4, 0.0, 0.0))

    # **The turn is in the law**, which is the axis easiest to leave out: it has
    # to raise the cadence once it is asking for more than the floor delivers.
    fast_spin = 3.0  # beyond the command range, to get clear of the clamp
    assert freq(0.0, 0.0, fast_spin) == pytest.approx(
        fast_spin * GAIT_TURN_RADIUS / (2 * GAIT_STRIDE)
    ), "the yaw command does not reach the cadence at all"
    assert freq(0.4, 0.0, fast_spin) > freq(0.4, 0.0, 0.0), (
        "turning while walking does not raise the cadence, so the feet are asked "
        "to cover the turn inside the stride the straight-line speed bought"
    )

    # **A spin at the angular ceiling has to clear the floor**, which is what
    # `COMMAND_ANG_CEILING` was raised to 2.0 for. Below it the yaw term is in the
    # law and inert: the clamp answers every turn with `freq_min` whatever it is
    # commanded, and nothing says so -- the gait simply runs slower than the turn
    # needs, which reads as a policy that cannot turn.
    #
    # This is a pairing, not a constant: it held at 1.25 with a 0.06 m stride and
    # stopped holding when the stride moved to 0.072. Either number can break it.
    spin = freq(0.0, 0.0, COMMAND_ANG_CEILING)
    assert spin > GAIT_FREQ_MIN_HZ, (
        f"a turn at the angular ceiling asks for "
        f"{COMMAND_ANG_CEILING * GAIT_TURN_RADIUS:.3f} m/s of foot speed and the "
        f"cadence floor already delivers "
        f"{2 * GAIT_FREQ_MIN_HZ * GAIT_STRIDE:.3f}, so it runs clamped and the "
        f"yaw term in the cadence law does nothing"
    )
    assert spin == pytest.approx(
        COMMAND_ANG_CEILING * GAIT_TURN_RADIUS / (2 * GAIT_STRIDE)
    )


@pytest.mark.parametrize("first", ["reward", "observation"])
def test_one_clock_answers_the_reward_and_the_observation(first: str) -> None:
    """Every reader of the gait phase reads one clock -- through a command
    resample and a reset, and whichever of them asks first in a step.

    The readers are the actor's and the critic's `gait_phase` observations,
    `tripod_gait` and `stance_load` through `tripod/mdp/phase.py::gait_phase`,
    `stance_ground_gap` through `common/mdp/phase.py::gait_phase`, and the swing
    gate on `foot_clearance`, which reads `env.gait_clock` itself.

    **The previous version of this test passed against two clocks.** mjlab builds
    a class-based observation term once per group, so the actor and the critic
    each had a `VariableGaitClock`, and `env.gait_clock` was whichever group was
    built last -- the critic's. The rewards advanced that one before the step's
    command update; the actor's was advanced by its own observation, after it.
    Inside one command and one episode the two agree exactly, and that is all the
    previous version stepped through. Measured on native cpu, 2 envs, zero
    actions, the walk command redrawn every 20 steps and a 39-step episode:
    identical until the first redraw that changed the tempo, 4.94 degrees apart
    from then on, and one whole control step apart after every reset -- 10.0
    degrees at the ceiling, 3.6 at the floor. `stance_ground_gap` was on neither:
    `common/mdp/phase.py` did not look for an installed clock, so it ran the
    fixed one at the ceiling and named the other stance group on 49.8% of
    walking steps at the first command rung (64 envs x 2000 steps, i9-14900KF).

    So this steps through what the old one did not -- resets that take some
    environments and not others, as the runner's staggered episode lengths make
    them, and redraws that change the tempo -- and reports every reader that
    leaves the clock, where it first did and by how much, rather than stopping at
    the first.

    `first="observation"` switches every reward off, which makes the observation
    the first to ask in each step and the reset the first on a reset step. The
    answer must not change: the clock integrates the tempo of the command the
    policy was **shown**, and a reset brings it up to date before zeroing. A clock
    that read the live command at whoever asked first would take the redrawn
    command here, and one that did not count the reset as a caller would start
    every new episode a step in.
    """
    import torch
    from mjlab.managers.reward_manager import RewardTermCfg

    import tasks
    from tasks.jumper.common.mdp.phase import gait_phase as common_gait_phase
    from tasks.jumper.common.mdp.rewards import moving_gate
    from tasks.jumper.posture.env_cfg import (
        GAIT_FREQ_HZ,
        GAIT_FREQ_MIN_HZ,
        GAIT_STRIDE,
        GAIT_TURN_RADIUS,
    )
    from tasks.jumper.posture.mdp.cadence import gait_frequency
    from tasks.jumper.tripod.mdp.phase import gait_phase

    cfg = tasks.load_env_cfg("jumper.posture")
    cfg.scene.num_envs = 4
    cfg.seed = 0
    # A 39-step episode, and a straight walk redrawn every 20 steps between 0.3
    # and 0.8 m/s: every redraw moves the tempo somewhere in 2.08-5.56 Hz, and no
    # command is slow enough for the standing gate to blank the observation.
    cfg.episode_length_s = 0.195
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (0.1, 0.1)
    twist.rel_standing_envs = 0.0
    twist.ranges.lin_vel_x = (0.3, 0.8)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.curriculum.pop("command", None)  # it would put its own ranges back

    # What the rewards read, from inside the step: the gait reward's reader and
    # `stance_ground_gap`'s, called exactly as those terms call them.
    read_by_reward: list[tuple[torch.Tensor, torch.Tensor]] = []

    def probe(env):
        read_by_reward.append(
            (gait_phase(env).clone(), common_gait_phase(env, GAIT_FREQ_HZ).clone())
        )
        return torch.zeros(env.num_envs, device=env.device)

    if first == "reward":
        cfg.rewards["phase_probe"] = RewardTermCfg(func=probe, weight=1.0)
    else:
        for term in cfg.rewards.values():
            if term is not None:
                term.weight = 0.0  # `RewardManager` skips a zero weight entirely

    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=4))

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        assert getattr(env, "gait_clock", None) is not None, (
            "no clock on the env, so `gait_phase` is still the fixed one and the "
            "gait reward is running at a cadence the observation does not show"
        )
        om = env.observation_manager

        def shown(obs, group: str) -> torch.Tensor:
            """The phase `group` is handed, decoded from its (sin, cos)."""
            off = 0
            for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
                if name == "gait_phase":
                    sc = obs[group][:, off:off + 2]
                    return torch.remainder(torch.atan2(sc[:, 0], sc[:, 1]) / math.tau, 1.0)
                off += math.prod(dims)
            raise AssertionError(f"the {group!r} group has no gait_phase term")

        def tempo() -> torch.Tensor:
            return gait_frequency(
                env, "twist", GAIT_STRIDE, GAIT_FREQ_MIN_HZ, GAIT_FREQ_HZ, GAIT_TURN_RADIUS
            )

        diverged: dict[str, tuple[int, float]] = {}

        def agree(what: str, k: int, got: torch.Tensor, want: torch.Tensor) -> None:
            # In degrees of cycle and circular, so 0.9999999 and 0.0 are adjacent.
            deg = float((torch.remainder(got - want + 0.5, 1.0) - 0.5).abs().max()) * 360
            if deg > 1e-3:
                k0, worst = diverged.get(what, (k, 0.0))
                diverged[what] = (k0, max(worst, deg))

        obs, _ = env.reset()
        # Staggered, as `OnPolicyRunner.learn` staggers them, so that the resets
        # below take one environment while the other three run on.
        env.episode_length_buf[:] = torch.tensor([15, 10, 5, 0])
        read_by_reward.clear()
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)

        # The clock this test expects, integrated here: each step advances by the
        # tempo of the command the policy was shown when it chose the action, and
        # an environment that resets starts its new episode at 0.
        expected = torch.zeros(env.num_envs)
        rate = tempo()
        partial_resets = redraws = 0
        for k in range(1, 61):
            obs, *_ = env.step(action)
            reset = env.episode_length_buf == 0
            before = torch.remainder(expected + rate * env.step_dt, 1.0)
            expected = torch.where(reset, torch.zeros_like(before), before)

            assert bool(moving_gate(env, "twist", 0.05).all()), (
                f"step {k}: an environment is commanded to stand, so its clock is "
                f"shown as (0, 0) and cannot be compared"
            )
            now = gait_phase(env)
            agree("the actor's observation against `gait_phase`", k, shown(obs, "actor"), now)
            agree("the critic's observation against `gait_phase`", k, shown(obs, "critic"), now)
            agree(
                "`common/mdp/phase.py::gait_phase` (stance_ground_gap) against "
                "`gait_phase`",
                k, common_gait_phase(env, GAIT_FREQ_HZ), now,
            )
            if first == "reward":
                assert len(read_by_reward) == k, "the probe did not run once a step"
                in_step, gap_in_step = read_by_reward[-1]
                agree("what the gait reward read in the step", k, in_step, before)
                agree("what stance_ground_gap read in the step", k, gap_in_step, before)
            agree(
                "`gait_phase` against one step of the tempo the policy was shown",
                k, now, expected,
            )

            partial_resets += int(bool(reset.any()) and not bool(reset.all()))
            new_rate = tempo()
            redraws += int(bool(((new_rate - rate).abs() > 1e-3)[~reset].any()))
            rate = new_rate

        # The controls: without them every comparison above could pass on a run
        # that never crossed what the first version of this test did not.
        assert partial_resets > 0, "no step reset some environments and not others"
        assert redraws > 0, (
            "no redraw changed the tempo of an environment that was not resetting"
        )
        assert not diverged, (
            "readers of the gait phase left the clock (first step, largest gap):\n"
            + "\n".join(
                f"  {what}: step {k0}, {deg:.2f} degrees"
                for what, (k0, deg) in diverged.items()
            )
            + "\nThe policy is being scored against a clock it was not shown."
        )

        # **And one object, which the numbers above cannot show on their own.**
        # Two clocks that each integrate the tempo they were shown and each
        # catch up on a reset stay equal to the last bit -- checked, with one
        # made per group -- and part only when something reaches one and not
        # the other. The first version's term was its own clock, hence the
        # fallback to the term itself.
        for group in ("actor", "critic"):
            term = om.get_term_cfg(group, "gait_phase").func
            assert getattr(term, "clock", term) is env.gait_clock, (
                f"the {group}'s gait_phase term shows a clock of its own rather "
                f"than `env.gait_clock`, so there are two integrators"
            )
    finally:
        env.close()
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def test_the_cadence_term_will_not_be_built_without_saying_when_it_advances() -> None:
    """A term without `advance` would export a contract without it, and every
    host reads that as the order policies were trained in before this clock --
    off it by `(f_now - f_at_entry) * step_dt` once the tempo moves, up to 6.4
    degrees, on a robot, with nothing to say so. So the term refuses any value but the one it implements, absence
    included, and the config the task trains with carries it in both groups.
    """
    from types import SimpleNamespace

    import tasks
    from tasks.jumper.posture.mdp.cadence import ADVANCE, VariableGaitClock

    law = {
        "command_name": "twist", "stride": 0.072, "freq_min": 2.0,
        "freq_max": 5.5556, "turn_radius": 0.2,
    }
    # The env is a bare namespace, so a term that let the value through would
    # fail on it with an AttributeError, not with this.
    for params in (law, {**law, "advance": "before_frame"}):
        with pytest.raises(ValueError, match="advance"):
            VariableGaitClock(SimpleNamespace(params=params), env=SimpleNamespace())

    cfg = tasks.load_env_cfg("jumper.posture")
    for group in ("actor", "critic"):
        got = cfg.observations[group].terms["gait_phase"].params.get("advance")
        assert got == ADVANCE, f"the {group}'s gait_phase says advance={got!r}"


def test_the_tripod_split_is_even_over_a_command_sequence() -> None:
    """The parity rule, restated for a cadence that moves.

    `phase.py` requires a whole, **even** number of control steps per cycle, and
    records a 12.2% left-right split from a frequency that was not. A fixed 0.06 m
    stride at this ceiling needs 20/3 Hz, which at 100 Hz control is 15 steps --
    odd -- so the rule has to be answered rather than inherited.

    **Integration does not answer it.** With 15 steps the phase takes the values
    0, 1/15, ... 14/15, of which eight are below 0.5 and seven are not, every
    cycle, in the same direction: a standing 6.7% bias that arrives however the
    phase is computed. That is asserted below, because a test that only checked
    the good case would let a future edit reintroduce a fixed odd cadence.

    What answers it is that the cadence is no longer fixed. The bias needs the
    frequency to sit at a rational value with an odd step count, and a command
    drawn from a range essentially never does.
    """
    import torch

    from tasks.jumper.posture.env_cfg import (
        DECIMATION,
        GAIT_FREQ_HZ,
        GAIT_FREQ_MIN_HZ,
        GAIT_STRIDE,
        SIM_DT,
    )

    dt = SIM_DT * DECIMATION
    steps = 60_000

    def share(freqs) -> float:
        """Fraction of steps assigning the swing to group A, by `phase < 0.5`."""
        phase, group_a = 0.0, 0
        for f in freqs:
            phase = (phase + f * dt) % 1.0
            group_a += phase < 0.5
        return group_a / len(freqs)

    # **Both ends of the clamp are even, and that is what `GAIT_STRIDE` buys.**
    # These are the two cadences the law can be *held* at -- the ceiling by a
    # command at full stick, the floor by every command below the changeover --
    # so they are where the parity still has to hold exactly.
    for name, freq in (("ceiling", GAIT_FREQ_HZ), ("floor", GAIT_FREQ_MIN_HZ)):
        n = 1.0 / (freq * dt)
        assert round(n) % 2 == 0, f"the {name} is {n} steps a cycle, odd"
        held = share([freq] * steps)
        assert held == pytest.approx(0.5, abs=1e-3), (
            f"held at the {name} the split is {held:.1%} / {1 - held:.1%}; an "
            f"even cycle has to be exactly even, so the phase is not landing "
            f"where the step count says"
        )

    # The control: an odd cycle is not evenly split however it is computed, which
    # is what the stride was chosen to avoid and what a 0.06 m stride would give
    # back. Without this the assertions above would pass on any cadence at all.
    odd = share([1.0 / (15 * dt)] * steps)
    assert abs(odd - 0.5) > 0.02, (
        "a 15-step cycle split evenly, so this test cannot tell an even cycle "
        "from an odd one and the stride was chosen against nothing"
    )

    # And the case that actually runs: commands drawn uniformly, resampled on the
    # command term's own 3-8 s clock.
    generator = torch.Generator().manual_seed(0)
    freqs: list[float] = []
    while len(freqs) < steps:
        v = float(torch.empty(1).uniform_(-0.8, 0.8, generator=generator).abs())
        f = min(max(v / (2 * GAIT_STRIDE), GAIT_FREQ_MIN_HZ), GAIT_FREQ_HZ)
        hold = float(torch.empty(1).uniform_(3.0, 8.0, generator=generator))
        freqs += [f] * int(hold / dt)
    sampled = share(freqs[:steps])
    assert abs(sampled - 0.5) < 0.005, (
        f"over a realistic command sequence group A is selected {sampled:.2%} of "
        f"the time, so the bias the fixed clock has survives a varying cadence "
        f"and the mirror augmentation is being fed a standing left-right bias"
    )


def test_the_angular_ceiling_is_what_the_gait_can_deliver() -> None:
    """The two ceilings and the cadence are three numbers with two degrees of freedom.

    A leg supports for half a cycle, so the gait moves a foot `2 * stride * freq`
    per support. The linear ceiling fixes the cadence from that; the cadence then
    fixes the angular ceiling, because a turn asks the outermost foot for
    `wz * turn_radius`. Any two of them pin the third, and writing all three
    independently is how they drift apart -- which is what happened: 1.25 rad/s
    cleared the cadence floor at a 0.06 m stride and stopped clearing it at 0.072,
    so a pure turn ran clamped and the yaw term in the cadence law did nothing,
    silently.
    """
    from tasks.jumper.posture.env_cfg import (
        COMMAND_ANG_CEILING,
        COMMAND_LIN_CEILING,
        GAIT_FREQ_HZ,
        GAIT_STRIDE,
        GAIT_TURN_RADIUS,
    )

    assert GAIT_FREQ_HZ == pytest.approx(COMMAND_LIN_CEILING / (2 * GAIT_STRIDE))
    assert COMMAND_ANG_CEILING == pytest.approx(
        2 * GAIT_STRIDE * GAIT_FREQ_HZ / GAIT_TURN_RADIUS
    ), "the angular ceiling is no longer the turn the gait can actually deliver"
    # Which is the same as saying a spin at the ceiling asks the foot for exactly
    # what the top speed does.
    assert COMMAND_ANG_CEILING * GAIT_TURN_RADIUS == pytest.approx(
        COMMAND_LIN_CEILING
    )


def test_the_command_ladder_starts_where_the_robot_has_been() -> None:
    """Five rungs, because the angular range tripled and three would not do.

    The shared ladder is `(0.4, 2/3, 1.0)` on the angular axis and was written
    against a ceiling of 0.75-1.25 rad/s. Against 4.0 its first rung is 1.6 --
    above this task's entire previous ceiling -- so a run would open by asking for
    a turn faster than any policy here has performed. The rungs are the pacing,
    and pacing written for one range is not pacing for a range three times wider.

    What is asserted is the property, not the numbers: the first rung has to be
    reachable-looking and the last has to be the ceiling, on both axes, with the
    two lists the same length because one level is one rung on both.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import (
        COMMAND_ANG_CEILING,
        COMMAND_LIN_CEILING,
        POSTURE_COMMAND,
    )

    del POSTURE_COMMAND
    params = tasks.load_env_cfg("jumper.posture").curriculum["command"].params
    lin, ang = params["levels"], params["ang_levels"]

    assert len(lin) == len(ang) == len(params["lin_std_scales"]), (
        "the two axes have different numbers of rungs, so a level means a "
        "different fraction of capability on each"
    )
    assert len(lin) >= 5, "the ladder is back to the shared three-rung shape"
    assert list(lin) == sorted(lin) and list(ang) == sorted(ang)
    assert lin[-1] == pytest.approx(COMMAND_LIN_CEILING)
    assert ang[-1] == pytest.approx(COMMAND_ANG_CEILING)
    # The last rung repeats the one below it: that is the precision rung, and it
    # is the only place the ruler may tighten.
    assert lin[-1] == pytest.approx(lin[-2]) and ang[-1] == pytest.approx(ang[-2])
    assert params["lin_std_scales"][-1] < 1.0
    assert set(params["lin_std_scales"][:-1]) == {1.0}

    # **The first rung is the one this exists for.** 1.25 rad/s is the fastest
    # this task has ever asked for, so a ladder whose opening rung is above it is
    # not a curriculum.
    assert ang[0] < 1.25, (
        f"the ladder opens at {ang[0]:.2f} rad/s, faster than the 1.25 that was "
        f"this task's entire ceiling until the gait fixed a new one"
    )

    # And it does not open above the shared ladder either, which is the other way
    # a wider range quietly becomes a harder start.
    assert ang[0] <= 0.4 * COMMAND_ANG_CEILING


def test_the_acceleration_penalty_is_weak_at_the_measured_magnitude() -> None:
    """A weight on a quantity that is not order-one has to be checked against it.

    `sum(joint_acc^2)` over the driven joints is 8.7e-03 while standing and
    1.5e+05 at 0.6 m/s -- seven orders of magnitude -- so "a small weight" means
    nothing until it is multiplied out. The failure this pins is not a crash: a
    weight one decade high makes this the second-largest term in the task and
    every gait decision follows it, and one decade low makes it a column of zeros
    that looks like a preference being expressed.

    The measured magnitudes are the fixture, so this fails when the weight moves
    without the measurement moving with it.
    """
    import tasks

    # Measured on model_9999, 64 envs x 400 control steps at a held speed; the
    # block beside the term in `env_cfg.py` has the full table.
    walking = 1.521e5
    standing = 8.669e-3

    cfg = tasks.load_env_cfg("jumper.posture")
    term = cfg.rewards["joint_acc"]
    assert term.weight < 0.0, "an acceleration reward pays for shaking"

    cost = walking * abs(term.weight)
    assert 0.005 < cost < 0.10, (
        f"at the measured 0.6 m/s magnitude this term is worth {cost:.4f} a step. "
        f"Below 0.005 it is inert; above 0.10 it is `foot_clearance`'s size and "
        f"is no longer the weak preference it is documented as"
    )

    # And it has to vanish while standing, or it is a tax on being still -- the
    # local optimum this task is documented as falling into.
    assert standing * abs(term.weight) < 1e-6

    # The joints are the driven ones. Including the two gripper joints would add
    # a constant: they are held by their PD and accelerate with the body.
    from tasks.jumper.common.constants import GAIT_JOINTS

    assert tuple(term.params["asset_cfg"].joint_names) == tuple(GAIT_JOINTS)


def test_clearance_charges_a_low_foot_and_not_a_high_one() -> None:
    """A floor, not a set point -- and the control group is mjlab's own term.

    `feet_clearance` upstream is `|height - target| * |foot speed|`, symmetric, so
    a foot carried 5 mm above the target costs exactly what one dragged 5 mm below
    it costs. On this task that put every term that touches the swing on the same
    side: after the target came down to 15 mm against a measured 16.7 mm peak,
    `foot_swing_height`, `soft_landing`, `joint_acc` and clearance's own velocity
    factor all pushed the lift down, and nothing pushed it up.

    What clearance exists to prevent is a foot catching the ground while it
    translates. Lifting higher than necessary is waste, and waste is priced by
    `joint_acc` and `energy` -- charging it again here made the term say something
    it is not named for.

    The symmetric form is evaluated alongside, because "the high foot is free"
    only means something if the same numbers would have cost something before.
    """
    import torch

    from tasks.jumper.posture.mdp.rewards import foot_clearance_shortfall

    target = 0.015

    class _Env:
        class scene:
            @staticmethod
            def __getitem__(name):
                if name == "height_scan":
                    return type("S", (), {"data": type("D", (), {
                        "heights": torch.tensor([[0.030, 0.015, 0.005]])
                    })()})()
                return type("A", (), {"data": type("D", (), {
                    # 1 m/s of horizontal speed on every foot, so the velocity
                    # weighting cannot be what makes a column zero.
                    "site_lin_vel_w": torch.tensor([[[1.0, 0.0, 0.0]] * 3])
                })()})()

        command_manager = None

    env = _Env()
    env.scene = _Env.scene()
    cost = float(
        foot_clearance_shortfall(env, target, "height_scan", command_name=None)
    )

    # Only the third foot is low, by 10 mm, at 1 m/s.
    assert cost == pytest.approx(0.010, abs=1e-9), (
        f"cost {cost} is not the 10 mm shortfall of the one low foot -- either "
        f"the high foot is being charged or the low one is not"
    )

    # The control group: the symmetric form on the same three feet charges the
    # high one 15 mm and the low one 10, so it is 2.5x this and mostly made of a
    # foot that has done nothing wrong.
    heights = torch.tensor([0.030, 0.015, 0.005])
    symmetric = float((heights - target).abs().sum())
    assert symmetric == pytest.approx(0.025, abs=1e-7)  # float32 heights
    assert symmetric > cost * 2, (
        "the symmetric form costs no more than the one-sided one on this "
        "example, so the example does not distinguish them"
    )


def test_the_unpaid_footprint_metrics_still_measure_what_they_name() -> None:
    """Nothing scores these, which is exactly where a wrong expression survives.

    `standing_foot_anchor` charged each foot's world displacement while parked and
    was removed on request. `foot_displacement` and `footprint_rotation` stayed as
    metrics, and they are now the **only** thing in a run that says whether a
    tracked twist was delivered by the body or by the feet -- `track_twist` scores
    the body against its own footprint and is structurally unable to tell "the
    body turned +theta" from "the footprint turned -theta", measured on
    model_38000 as 10.05 degrees scored against a base heading change of 0.00.

    An unpaid metric has no reward curve to look wrong, so what is pinned here is
    the arithmetic itself: the identity the footprint yaw rests on, that the
    displacement is measured from the anchor rather than from the nominal stance,
    and the two refresh rules that make "since you were parked" mean anything.
    """
    import torch

    import tasks

    cfg = tasks.load_env_cfg("jumper.posture", play=True)
    cfg.scene.num_envs = 4

    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=4))

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        posture = env.command_manager.get_term("posture")
        velocity = env.command_manager.get_term("twist")

        # Nothing pays for either of them, or this test is about the wrong thing.
        assert not any(
            "foot_anchor" in name or "footprint" in name
            for name in env.reward_manager.active_terms
        ), "a reward term is scoring the footprint again; this test is stale"

        # **The identity the footprint yaw rests on.** It is the body's heading
        # *minus* the twist, so that the three add up. Written as the heading
        # alone it would agree at a square stance and diverge exactly when the
        # footprint deforms, which is the case it exists for -- so it is checked
        # with the stance deformed.
        from mjlab.utils.lab_api.math import wrap_to_pi

        from tasks.jumper.posture.mdp import state

        torch.manual_seed(0)
        for _ in range(25):
            env.step(torch.randn(env.num_envs, env.action_manager.total_action_dim))
        twist_now = state.body_twist(env, posture.asset_cfg)
        assert float(twist_now.abs().max()) > 1e-3, (
            "the stance is still square, so the identity below cannot distinguish "
            "the footprint's yaw from the body's"
        )
        assert torch.allclose(
            wrap_to_pi(posture.footprint_yaw + twist_now),
            env.scene["robot"].data.heading_w,
            atol=1e-4,
        ), "footprint + twist is not the body's heading, so one of them is wrong"

        # Parked *after* the stepping: the steps resample commands, so a gate set
        # before them is not the gate in force after.
        velocity.is_standing_env[:] = True
        velocity.vel_command_b[:] = 0.0

        # **Displacement is from the anchor, not from the nominal stance.** The
        # second would report a constant offset for a robot standing perfectly
        # still in a stance that is merely not HOME, which reads as the drift this
        # is here to see.
        posture.foot_anchor[:] = posture.foot_xy
        assert float(posture.foot_displacement.max()) < 1e-6
        posture.foot_anchor[:] = posture.foot_xy - 0.03 / (2**0.5)
        assert float(posture.foot_displacement.mean()) == pytest.approx(
            0.03, rel=1e-4
        ), "the displacement is not the distance from the anchor"

        # The anchor tracks while free and freezes when parked, which is what
        # makes "since you were told to stand" mean anything.
        velocity.is_standing_env[:] = False
        velocity.vel_command_b[:, 0] = 0.5
        posture._update_command()
        assert torch.allclose(posture.foot_anchor, posture.foot_xy, atol=1e-6), (
            "the anchor did not follow a robot free to move"
        )
        velocity.is_standing_env[:] = True
        stale = posture.foot_xy - 0.3
        posture.foot_anchor[:] = stale
        posture._update_command()
        assert torch.allclose(posture.foot_anchor, stale), (
            "the anchor thawed under a parked robot, so a displacement can never "
            "accumulate against it"
        )

        # **And a resample does not thaw it either.** A resample is where a *new
        # twist command* arrives, so re-anchoring there would zero the metric at
        # exactly the moment the feet are most likely to move -- the reading would
        # stay near zero through the behaviour it exists to report.
        posture._resample_command(torch.arange(env.num_envs))
        assert torch.allclose(posture.foot_anchor, stale), (
            "a posture resample re-anchored a parked robot, so the metric resets "
            "every time a new twist is commanded"
        )

        # A fresh episode does get one, or an environment restarting straight into
        # standing is measured against the previous episode's stance, somewhere
        # else entirely. `_update_command` reads `episode_length_buf` for this.
        env.episode_length_buf[:] = 0
        posture._update_command()
        assert torch.allclose(posture.foot_anchor, posture.foot_xy, atol=1e-6), (
            "a restarted environment kept the old episode's anchor"
        )
    finally:
        env.close()
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def test_the_landing_peak_is_reduced_by_max_and_not_by_mean() -> None:
    """A peak metric averaged over time is a metric about cadence.

    `landing_force_max` is zero on every step where nothing landed, which is most
    of them -- one footfall per foot per gait cycle, 18 control steps at the top
    cadence. Under the default `reduce="mean"` those zeros would divide the number
    by the duty cycle, so a gait that landed half as often would report half the
    "peak" force without a single landing having changed.

    `reduce="max"` is what makes it the peak over the episode. The declaration is
    the whole mechanism, so the declaration is what is checked.
    """
    import torch

    import tasks
    from tasks.jumper.posture.mdp.metrics import landing_force_max

    term = tasks.load_env_cfg("jumper.posture").metrics["landing_force_max"]
    assert getattr(term, "reduce", "mean") == "max", (
        "the landing peak is being averaged over the episode, which turns it "
        "into a quantity about how often the robot lands rather than how hard"
    )

    # And the per-step value is the hardest of the feet that landed, not of the
    # feet that happen to be loaded: a planted foot carrying the robot's weight
    # is a larger force than most landings and must not be reported as one.
    class _Sensor:
        # All three feet are on the ground; only the last two *arrived* this step.
        # `found` is present so that reading it instead of `compute_first_contact`
        # produces a wrong number rather than an AttributeError -- a control group
        # that fails by crashing does not show the reading is wrong.
        data = type("D", (), {
            "force": torch.tensor([[[0.0, 0, 30.0],
                                    [0.0, 0, 5.0],
                                    [0.0, 0, 9.0]]]),
            "found": torch.tensor([[1, 1, 1]]),
        })()

        @staticmethod
        def compute_first_contact(dt):
            del dt
            return torch.tensor([[False, True, True]])

    class _Env:
        step_dt = 0.01
        scene = {"feet": _Sensor()}

    got = float(landing_force_max(_Env(), "feet")[0])
    assert got == pytest.approx(9.0), (
        f"reported {got} N: the 30 N foot was already standing on the ground and "
        f"did not land this step"
    )




def test_the_landing_speed_is_read_before_the_impact_not_at_it() -> None:
    """The velocity at first contact is post-impact, which is nearly zero.

    Contact resolves inside the five physics substeps of a control step, so by the
    time `compute_first_contact` reports a foot down, that foot has already been
    stopped by the ground. A term reading `vz` at that moment charges ~0.02 m/s
    for a landing that arrived at 0.54 -- alive in the logs, measuring nothing,
    and the harder the landing the less it looks like one, because a harder impact
    stops the foot more completely.

    So `soft_touchdown` keeps one step of history and charges the *previous*
    step's velocity. Pinned here with a control: the same fixture read the naive
    way gives the wrong answer by 27x, which is the margin this exists to keep.
    """
    import torch

    from tasks.jumper.posture.mdp.rewards import soft_touchdown

    APPROACH, POST_IMPACT = -0.54, -0.02

    class _Sensor:
        def __init__(self):
            self.data = type("D", (), {"found": torch.zeros(1, 6)})()
            self.landing = False

        def compute_first_contact(self, dt):
            del dt
            hit = torch.zeros(1, 6, dtype=torch.bool)
            hit[0, 0] = self.landing
            return hit

    class _Robot:
        def __init__(self):
            self.data = type("D", (), {"site_lin_vel_w": torch.zeros(1, 6, 3)})()

    sensor, robot = _Sensor(), _Robot()

    class _Env:
        num_envs, device, step_dt = 1, "cpu", 0.01
        scene = {"feet": sensor, "robot": robot}

        class command_manager:
            @staticmethod
            def get_command(_name):
                return torch.tensor([[0.8, 0.0, 0.0]])

    env = _Env()
    cfg = type("C", (), {"params": {"sensor_name": "feet"}})()
    term = soft_touchdown(cfg, env)
    params = {"sensor_name": "feet", "command_name": "twist"}

    # One step of approach, no contact yet: nothing owed, and the velocity is
    # banked.
    robot.data.site_lin_vel_w[0, :, 2] = APPROACH
    assert float(term(env, **params)) == pytest.approx(0.0)

    # The landing step. The foot has been stopped by the ground, so what is
    # readable *now* is the post-impact velocity -- and that is what must not be
    # charged.
    sensor.landing = True
    robot.data.site_lin_vel_w[0, :, 2] = POST_IMPACT
    charged = float(term(env, **params))
    assert charged == pytest.approx(-APPROACH, abs=1e-6), (
        f"the landing was charged {charged}, not the {-APPROACH} m/s it was "
        f"falling at a step earlier"
    )
    # The control: the naive spelling, on this same fixture.
    naive = -POST_IMPACT
    assert charged > 20 * naive, (
        f"reading vz at first contact gives {naive}, so a term written that way "
        f"would look alive while charging {naive / charged:.1%} of the landing"
    )

    # A foot moving *upward* is not a landing however the contact flags read, or
    # a foot scuffing on the way out is charged as a touchdown.
    robot.data.site_lin_vel_w[0, :, 2] = +0.3
    term(env, **params)
    assert float(term(env, **params)) == pytest.approx(0.0)

    # And a parked robot pays nothing: the gate is the command, not the contact.
    robot.data.site_lin_vel_w[0, :, 2] = APPROACH
    term(env, **params)
    _Env.command_manager.get_command = staticmethod(
        lambda _name: torch.zeros(1, 3)
    )
    assert float(term(env, **params)) == pytest.approx(0.0)


def test_the_clearance_gate_is_on_the_clock_and_not_on_the_descent() -> None:
    """Exempting a descending foot would pay for the exemption in falling speed.

    `foot_clearance_shortfall` charges the time a foot spends low while
    translating, so the cheapest way to pay it is to cross the low band fast --
    measured, it charged 0.0185 over the descent against 0.0079 over the rise,
    with its largest single sample at touchdown itself. Freeing the approach is
    the fix; the question is what "the approach" is keyed on.

    Keyed on the foot's own `vz < 0`, the policy buys the exemption by descending
    faster, which is the behaviour being corrected. Keyed on the gait clock it
    cannot: the clock is a function of the command.

    Pinned here: which feet the gate opens for at three phases, and that the gate
    does not move when the feet do.
    """
    import torch

    from tasks.jumper.posture.mdp.rewards import _swing_gate
    from tasks.jumper.tripod.mdp.rewards import TRIPOD_A, TRIPOD_B

    class _Clock:
        def __init__(self, p):
            self.p = torch.tensor([p])

        def phase(self, _env):
            return self.p

    class _Env:
        pass

    env = _Env()

    # Early in group A's half-cycle: A is lifting and tolled, B is planted.
    env.gait_clock = _Clock(0.1)
    g = _swing_gate(env, 0.6)
    assert [float(g[0, i]) for i in TRIPOD_A] == [1.0] * 3
    assert [float(g[0, i]) for i in TRIPOD_B] == [0.0] * 3

    # Late in the same half-cycle: A is on its way down and free.
    env.gait_clock = _Clock(0.4)
    g = _swing_gate(env, 0.6)
    assert float(g.sum()) == 0.0, (
        "the gate is still charging a foot 80% of the way through its swing, "
        "which is the approach it exists to free"
    )

    # The groups swap with the half-cycle, or half the robot is never tolled.
    env.gait_clock = _Clock(0.6)
    g = _swing_gate(env, 0.6)
    assert [float(g[0, i]) for i in TRIPOD_B] == [1.0] * 3
    assert [float(g[0, i]) for i in TRIPOD_A] == [0.0] * 3

    # The boundary is the gate, not a rounding of it: 0.59 of the way through is
    # charged and 0.61 is not.
    assert float(_swing_gate(env, 0.6)[0, TRIPOD_B[0]]) == 1.0
    env.gait_clock = _Clock(0.5 + 0.61 / 2)
    assert float(_swing_gate(env, 0.6)[0, TRIPOD_B[0]]) == 0.0
    env.gait_clock = _Clock(0.5 + 0.59 / 2)
    assert float(_swing_gate(env, 0.6)[0, TRIPOD_B[0]]) == 1.0

    # **And a gate with no clock refuses rather than falling back.** The fallback
    # that suggests itself is the foot's own velocity, which is the one spelling
    # that must not be reachable by accident.
    env.gait_clock = None
    with pytest.raises(RuntimeError, match="faster descent"):
        _swing_gate(env, 0.6)


def test_the_history_frames_are_a_stride_apart_and_oldest_first() -> None:
    """Same shape whether the stride works or not, which is the whole risk.

    `StridedHistory` exists because five consecutive frames span 25 ms at 200 Hz
    and the 5 was chosen at 50 Hz to span 100 ms. A wrapper that buffered
    correctly and then returned the five *newest* frames would produce a tensor of
    exactly the right width, pass every shape check, train without complaint, and
    give the policy the 25 ms window the class was written to avoid.

    So the frames are fed a ramp and read back by value. Oldest-first is pinned
    with them because `common/mdp/symmetry.py` mirrors against that layout and the
    reversed order is, again, the same shape.
    """
    import torch

    from tasks.jumper.posture.mdp.history import StridedHistory

    FRAMES, STRIDE = 5, 4

    class _Env:
        num_envs, device = 1, "cpu"
        common_step_counter = 0

    env = _Env()
    tick = {"n": 0}

    def _ramp(_env, **kw):
        del kw
        return torch.full((1, 2), float(tick["n"]))

    # The reserved prefix is the interface: everything without it is the wrapped
    # term's own params and is passed straight through.
    P = StridedHistory.PREFIX
    cfg = type("C", (), {"params": {
        P + "func": _ramp, P + "frames": FRAMES, P + "stride": STRIDE,
    }})()
    term = StridedHistory(cfg, env)

    # Long enough that the buffer has stopped backfilling: depth is
    # stride*(frames-1)+1 = 17.
    for i in range(40):
        tick["n"] = i
        env.common_step_counter = i
        out = term(env)

    got = out.reshape(FRAMES, 2)[:, 0].tolist()
    assert got == [23.0, 27.0, 31.0, 35.0, 39.0], (
        f"frames came back {got}; the newest is 39 and they should step back by "
        f"{STRIDE}, oldest first"
    )
    # Which is a 100 ms window at 200 Hz, and the reason for the class.
    assert (got[-1] - got[0]) * 0.005 == pytest.approx(0.080)

    # The control: consecutive frames would have been [35, 36, 37, 38, 39] --
    # same shape, same dtype, a quarter of the window.
    assert got != [35.0, 36.0, 37.0, 38.0, 39.0]

    # **Reading twice in one step does not advance it.** `ObservationManager`
    # carries its own `update_history` flag for this, and anything that inspects
    # the observation out of band would otherwise age the oldest frame by one
    # extra step per look.
    again = term(env)
    assert torch.equal(out, again)
    tick["n"] = 999
    assert torch.equal(term(env), out), "the history advanced without a new step"

    # A fresh episode is backfilled with its own first observation rather than
    # left at zero -- 17 steps of zeros occur nowhere else in training, and
    # `CircularBuffer` is used precisely to inherit that behaviour.
    term.reset(torch.tensor([0]))
    env.common_step_counter += 1
    tick["n"] = 7
    fresh = term(env).reshape(FRAMES, 2)[:, 0].tolist()
    assert fresh == [7.0] * FRAMES, f"a reset episode starts at {fresh}, not its own first frame"


def test_a_buffered_frame_keeps_the_noise_it_was_captured_with() -> None:
    """Re-noising the history every step makes the sim cleaner than the robot.

    mjlab's pipeline is `compute -> noise -> ... -> history`, so each frame
    carries the error it had when it was current. The obvious way to wrap it --
    buffer the raw value, let the manager add noise to the stacked result -- keeps
    every shape and resamples all five frames on every step. A policy can average
    that away. No robot can: a past reading is a number that was written down
    once. The sim would be quietly easier than the machine, in the direction that
    only shows up on deployment.

    Pinned by holding the underlying value constant and asking whether a frame
    that is carried forward is the same number.

    **Advance by `stride`, not by one.** With stride 2 a single step flips the
    parity of the sampled indices, so consecutive reads share no frame at all and
    a naive "the older frames did not move" assertion fails against a correct
    implementation. It did; the first draft of this test was wrong and the class
    was not.
    """
    import torch

    from tasks.jumper.posture.mdp.history import StridedHistory

    class _Noise:
        def apply(self, x):
            return x + torch.rand_like(x)

    class _Env:
        num_envs, device = 1, "cpu"
        common_step_counter = 0

    env = _Env()
    P = StridedHistory.PREFIX
    cfg = type("C", (), {"params": {
        P + "func": lambda _e, **kw: torch.zeros(1, 1),
        P + "frames": 3, P + "stride": 2, P + "noise": _Noise(),
    }})()
    term = StridedHistory(cfg, env)

    for i in range(20):
        env.common_step_counter = i
        out = term(env)
    older = out.reshape(3)[:2].clone()

    before = out.reshape(3).clone()
    for i in (20, 21):  # one full stride
        env.common_step_counter = i
        nxt = term(env).reshape(3)
    assert torch.allclose(nxt[:2], before[1:]), (
        f"after one stride the two newest frames {before[1:].tolist()} should be "
        f"the two oldest, but they read {nxt[:2].tolist()} -- the history is "
        f"being re-noised rather than carried"
    )

    # The control: the noise is live, so a test that passed by there being no
    # noise at all would be worthless.
    assert float(older.abs().sum()) > 0.0, "no noise was applied; the check above is vacuous"


def test_only_the_training_actor_reads_noise_into_its_history() -> None:
    """The history's noise is the training actor's, and nobody else's.

    `StridedHistory` applies the noise itself, so the manager's per-group switch
    -- it clears `term.noise` where `enable_corruption` is off -- no longer
    reaches it; what decides is what `env_cfg.py` hands the wrapper.

    mjlab's skeleton builds the critic as `{**actor_terms, ...}`, so `joint_vel`
    and `actions` are one object in both groups. Wrapped in place, the actor's
    pass put its noise into the shared params and the critic's pass found the
    term already wrapped and skipped it: the critic, uncorrupted by design, read
    the actor's +/-1.5 rad/s into its `joint_vel` history. Nothing raised and no
    width changed -- the manager deep-copies each group's term, so the buffers
    were never shared, only the noise riding in the params.

    Replay is checked beside training because the wrapper takes its noise when
    `env_cfg` runs: wrapping before `play` turns corruption off would carry
    training's noise into every replay, just as silently.

    The control is the training actor, which does read noise into every history
    it corrupts. A check that passed because no history carried any noise at all
    would pass against a wrapper that had lost it everywhere.
    """
    import tasks
    from tasks.jumper.posture.mdp.history import StridedHistory

    key = StridedHistory.PREFIX + "noise"
    # The actor's corrupted history terms: the joint encoders, the joint rates
    # and the current sensor. `actions` is stacked too and carries no noise.
    corrupted = {"joint_pos", "joint_vel", "actuator_force"}

    for play in (False, True):
        cfg = tasks.load_env_cfg("jumper.posture", play=play)
        noisy: dict[str, set[str]] = {}
        for group_name, group in cfg.observations.items():
            wrapped = {
                name: term for name, term in group.terms.items()
                if term is not None and term.func is StridedHistory
            }
            # The critic's history has to be the wrapper's for its check to say
            # anything about the wrapper.
            assert corrupted | {"actions"} <= wrapped.keys(), (
                f"{group_name} wraps only {sorted(wrapped)}"
            )
            noisy[group_name] = {
                name for name, term in wrapped.items() if term.params.get(key) is not None
            }

        mode = "replay" if play else "training"
        want = {"actor": set() if play else corrupted, "critic": set()}
        for group_name, names in want.items():
            assert noisy[group_name] == names, (
                f"in {mode} the {group_name} reads noise into the history of "
                f"{sorted(noisy[group_name]) or 'nothing'}, where it should be "
                f"{sorted(names) or 'nothing'}"
            )


def test_a_term_cannot_stack_frames_twice_without_saying_so() -> None:
    """Both stacks set is a five-times-wider observation that nothing reports.

    `StridedHistory` stacks frames itself, so the manager's `history_length` has
    to go to 0. Left at 5 the manager stacks five copies of an already-stacked
    term: the actor's input is 25 frames wide, the layout says 5, and
    `symmetry.py` mirrors against whichever number it happened to read.

    It cannot be caught by a shape assertion downstream because nothing downstream
    knows the intended width -- so it is caught where the two numbers are both
    visible, which is the frame-count lookup the mirror depends on.
    """
    from tasks.jumper.common.mdp import symmetry

    class _Cfg:
        flatten_history_dim = True
        history_length = 5

        class func:
            history_frames = 5

    class _OM:
        active_terms = {"actor": ["joint_pos"]}
        _group_obs_term_cfgs = {"actor": [_Cfg()]}

    class _Env:
        observation_manager = _OM()

    with pytest.raises(ValueError, match="stacks 5 frames itself"):
        symmetry._history_length(_Env(), "actor", "joint_pos")

    # And a plain term with neither still reads as one frame, so no other task's
    # mirror changes behaviour.
    _Cfg.history_length = 0
    _Cfg.func.history_frames = 0
    assert symmetry._history_length(_Env(), "actor", "joint_pos") == 1


def test_both_ends_of_the_cadence_clamp_are_whole_even_control_steps() -> None:
    """At 200 Hz too, or one tripod group gets more of the cycle than the other.

    `mdp/cadence.py` records the measurement: an odd number of control steps a
    cycle splits the two groups 8/7 whatever the arithmetic, because it is the
    parity of the step count and not the rounding. It is answered at the two ends
    of the clamp, where the cadence can be *held* -- in between the command moves
    and the split averages out.

    The rate change is exactly the edit that can break this without touching
    `cadence.py`, so it is asserted against the config rather than against a
    constant.
    """
    import tasks
    from tasks.jumper.posture import env_cfg as E

    cfg = tasks.load_env_cfg("jumper.posture")
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert dt == pytest.approx(E.SIM_DT * E.DECIMATION)

    for label, hz in (("ceiling", E.GAIT_FREQ_HZ), ("floor", E.GAIT_FREQ_MIN_HZ)):
        steps = 1.0 / (hz * dt)
        assert steps == pytest.approx(round(steps)), (
            f"the {label} cadence {hz:.4f} Hz is {steps:.3f} control steps a "
            f"cycle at {1/dt:.0f} Hz, not a whole number"
        )
        assert round(steps) % 2 == 0, (
            f"the {label} cadence is {round(steps)} control steps a cycle, which "
            f"is odd, so the two tripod groups split it {round(steps)//2+1}/"
            f"{round(steps)//2} for as long as the command is held there"
        )

    # And the swing resolution this rate change was made for.
    assert round(1.0 / (E.GAIT_FREQ_HZ * dt)) // 2 == 18


def test_a_wrapped_term_gets_its_scene_entities_resolved() -> None:
    """The manager resolves `params`, not `params["params"]`, and wrapping nests.

    `ManagerBase._resolve_common_term_cfg` walks a term's params and resolves each
    `SceneEntityCfg` it finds -- at the top level only. `StridedHistory` puts the
    wrapped term's params one dict down, out of its reach, and an unresolved
    `SceneEntityCfg` has no `joint_ids`: `joint_pos_rel` then returns **all 22
    joints rather than the 20 the policy drives**, gripper included.

    Nothing raises. The observation is 10% wider than the contract says and
    carries two joints nothing controls. It was caught by the symmetry mirror,
    whose joint permutation is 20 wide -- and symmetry is optional, so the catch
    was luck rather than coverage. This is the coverage.
    """
    import tasks

    cfg = tasks.load_env_cfg("jumper.posture", play=True)
    cfg.scene.num_envs = 2

    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=2))

    from mjlab.envs import ManagerBasedRlEnv
    from tasks.jumper.common.constants import GAIT_JOINTS

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        om = env.observation_manager
        driven = len(GAIT_JOINTS)
        widths = dict(
            zip(om.active_terms["actor"], (int(d[0]) for d in om._group_obs_term_dim["actor"]))
        )
        for name in ("joint_pos", "joint_vel", "actuator_force"):
            frames = om._group_obs_term_cfgs["actor"][
                om.active_terms["actor"].index(name)
            ].func.history_frames
            per = widths[name] // frames
            assert per == driven, (
                f"{name} is {per} wide a frame against {driven} driven joints, "
                f"so the wrapped term's asset_cfg was never resolved"
            )

        # The total is the number the rate-change note quotes, and the point of
        # striding rather than widening: the same input the 100 Hz run had.
        assert om.group_obs_dim["actor"] == (415,)
    finally:
        env.close()
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def test_the_curriculum_dwells_are_in_iterations_whatever_the_rollout_is() -> None:
    """A dwell counted in env steps changes pace when the rollout length changes.

    Both ladders here are calibrated in *iterations* -- `DWELL_STEPS`'s note
    records two promotions firing at exactly the dwell minimum and concludes the
    gate "never decided anything" -- and both are *expressed* in environment
    steps, which is what the curricula count. The conversion is
    `num_steps_per_env`, and it is not a constant of nature: it doubled to 48 when
    the control rate went to 200 Hz, so a literal `24 * 25` silently became 12.5
    iterations instead of 25.

    `DWELL_STEPS`'s own docstring warned about this before it happened, in so many
    words, and the warning was not enough. So the assertion is on the iteration
    count rather than the step count: either factor moving alone fails here.
    """
    import tasks
    from tasks.jumper.posture.mdp.curriculum import DWELL_STEPS
    from tasks.jumper.posture.rl_cfg import NUM_STEPS_PER_ENV, agent_cfg

    assert agent_cfg().num_steps_per_env == NUM_STEPS_PER_ENV, (
        "the runner's rollout and the constant the dwells convert through have "
        "come apart, so the arithmetic below describes neither"
    )

    assert DWELL_STEPS % NUM_STEPS_PER_ENV == 0
    assert DWELL_STEPS // NUM_STEPS_PER_ENV == 25, (
        f"the posture ladder dwells {DWELL_STEPS // NUM_STEPS_PER_ENV} iterations, "
        f"not the 25 its calibration was measured at"
    )

    # And the command ladder, which inherits a `24 * 100` literal from
    # `common/mdp/curriculum.py` that three other tasks are still right about.
    cfg = tasks.load_env_cfg("jumper.posture")
    dwell = cfg.curriculum["command"].params.get("dwell_steps")
    assert dwell is not None, (
        "the command curriculum is still on the shared default, which is "
        "24 * 100 environment steps and is 100 iterations only at a 24-step "
        "rollout -- this task's is not 24"
    )
    assert dwell // NUM_STEPS_PER_ENV == 100, (
        f"the command ladder dwells {dwell // NUM_STEPS_PER_ENV} iterations, not 100"
    )

    # The four locomotion tasks are untouched: they run a 24-step rollout and the
    # shared default is correct for them. Changing it there would have been the
    # fix that broke three tasks to mend one.
    from tasks.jumper.common.mdp.curriculum import CommandRangeCurriculum  # noqa: F401
    import inspect

    from tasks.jumper.common.mdp import curriculum as common_curriculum

    src = inspect.getsource(common_curriculum)
    assert "dwell_steps: int = 24 * 100" in src, (
        "the shared default moved; the four locomotion tasks pace their command "
        "ladder with it and none of them changed rollout length"
    )


def test_the_curriculum_takes_a_sample_when_envs_reset_one_at_a_time() -> None:
    """Which is how they do reset, and the gate used to require eight at once.

    `on_policy_runner.learn` randomises the initial episode lengths to decorrelate
    the environments, so they time out one at a time: the expected reset batch is
    `num_envs / episode_steps`, about 1 at any size worth running. A gate reading
    `valid.sum() >= min_samples` on a single batch therefore asks for eight
    simultaneous timeouts -- a Poisson coincidence, not a sample size.

    Measured before the fix, on `jumper.posture` at 200 Hz with its 4000-step
    episode: 1143 iterations, every batch of size 1, **not one sample taken**, all
    four `*_err` nan from the first iteration to the last and the ladder unable to
    leave level 0. At 100 Hz the coincidence is a hundred times likelier and one
    hit seeds an EMA that never returns to None -- which is why this survived a
    rate change to be found rather than being found when it was written.

    Pinned here by feeding batches of one and asking when the average appears: at
    the eighth, not the first and not never. The single-batch form passes the
    "not the first" half of that, so the "not never" half is what this is for.
    """
    import torch

    import tasks
    from tasks.jumper.posture.mdp.curriculum import PostureRangeCurriculum

    # **Not `play=True`**: a replay builds with `curriculum = {}` and gets a
    # `NullCurriculumManager`, so the thing under test would not exist.
    cfg = tasks.load_env_cfg("jumper.posture")
    cfg.scene.num_envs = 4

    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=4))

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        term_cfg = env.curriculum_manager.get_term_cfg("posture")
        term = term_cfg.func
        assert isinstance(term, PostureRangeCurriculum)
        params = dict(term_cfg.params)
        min_samples = params.get("min_samples", 8)
        min_steps = params.get("min_steps", 100)

        # A settled robot with a real episode behind it, so `valid` is true and
        # the error being pooled is a number rather than a zero accumulator.
        env.common_step_counter = 10_000
        env.episode_length_buf[:] = min_steps * 4
        posture = env.command_manager.get_term("posture")
        for axis in ("twist", "pitch", "roll", "height"):
            posture.metrics[f"error_{axis}"][:] = 0.01

        seen = []
        for i in range(min_samples + 2):
            out = term(env, torch.tensor([i % env.num_envs]), **params)
            seen.append(out["twist_err"])

        import math

        first = next(
            (i for i, v in enumerate(seen) if not math.isnan(v)), None
        )
        assert first is not None, (
            f"{len(seen)} resets of one environment each and the average is "
            f"still nan, so the ladder can never promote"
        )
        assert first == min_samples - 1, (
            f"the average appeared after {first + 1} single-env resets, not "
            f"{min_samples}; the pooling is not counting what min_samples names"
        )
    finally:
        env.close()
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def test_the_strided_history_reproduces_the_spacing_the_frame_count_was_chosen_at() -> None:
    """Two conventions for "how much history" live in this repository.

    `common/velocity_env.py` says five frames "spans 0.1 s", counting each frame
    as covering a control period. `mdp/history.py` and the bundle README quote
    *reach* -- how far back the oldest frame is, `(frames - 1) * stride / rate`.
    The same five frames are 100 ms under one and 80 ms under the other, and the
    first draft of the striding used both: the module docstring said the window
    was restored to 100 ms while the README it generates said 80.

    Neither number is wrong and the disagreement is, so what is pinned is the
    claim that does not depend on the convention: **four control steps at 200 Hz
    is 20 ms, which is one control step at 50 Hz.** The strided history is not a
    similar window, it is the same frames at the same spacing the 5 was chosen
    for.
    """
    import tasks
    from tasks.jumper.posture import env_cfg as E

    OBS_HISTORY = __import__(
        "tasks.jumper.common.velocity_env", fromlist=["OBS_HISTORY"]
    ).OBS_HISTORY

    dt = E.SIM_DT * E.DECIMATION
    spacing = E.OBS_HISTORY_STRIDE * dt
    assert spacing == pytest.approx(1.0 / 50.0), (
        f"frames are {spacing * 1000:.1f} ms apart, not the 20 ms they were at "
        f"the 50 Hz this task's frame count was calibrated on"
    )

    # And the depth the wrapper needs follows from the pair, so the buffer cannot
    # be sized from the frame count alone.
    cfg = tasks.load_env_cfg("jumper.posture")
    term = cfg.observations["actor"].terms["joint_pos"]
    p = "_history_"
    assert term.params[p + "frames"] == OBS_HISTORY
    assert term.params[p + "stride"] == E.OBS_HISTORY_STRIDE
    assert term.history_length == 0, (
        "the manager would stack these frames a second time"
    )


def test_a_stateful_reward_term_is_actually_in_the_reset_list() -> None:
    """`RewardManager` collects class terms by `hasattr(func, "reset")`.

    A term that carries state and does not define the method is therefore not
    reset, and nothing says so: it is simply absent from `_class_term_cfgs`.
    `soft_touchdown` banks one step of foot velocity, so without the method the
    first landing of a new episode is charged the speed a foot was falling at on
    the last step of the previous one -- a different robot pose, possibly mid-fall,
    and a reset teleports the base so the banked number can be anything.

    Pinned against the manager's own list rather than against the class, because
    `hasattr` is the actual rule and a method that got renamed would still exist.
    """
    import tasks

    cfg = tasks.load_env_cfg("jumper.posture")
    cfg.scene.num_envs = 2

    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=2))

    from mjlab.envs import ManagerBasedRlEnv

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        import torch

        rm = env.reward_manager
        registered = {c.func.__class__.__name__ for c in rm._class_term_cfgs}
        assert "soft_touchdown" in registered, (
            "soft_touchdown banks a step of velocity and is not in the reward "
            "manager's reset list, so that velocity survives the episode it "
            "was measured in"
        )

        # And the reset clears it, rather than existing only to get on the list.
        term = rm.get_term_cfg("soft_touchdown").func
        term.prev_vz[:] = -9.0
        term.reset(torch.tensor([0]))
        assert float(term.prev_vz[0].abs().max()) == 0.0
        assert float(term.prev_vz[1].abs().max()) == 9.0, (
            "the reset cleared an environment that did not reset"
        )
    finally:
        env.close()
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def test_action_rate_is_not_a_budget_that_can_be_lent_to_another_term() -> None:
    """Halving it doubled the motion and left the share where it was.

    `soft_touchdown` was 1.0% of the reward budget and had not moved in 6900
    iterations while `action_rate_l2` was 32.1%. The obvious move is to lighten
    the big one so the small one can bite, and there was a measurement that
    seemed to license it: the per-step action difference had not fallen when the
    control rate doubled (raw 9.61 at 100 Hz, 9.43 at 200), so the weight had been
    doubled for a correction that did not apply.

    **The measurement was taken under the weight it was used to justify changing.**
    Resumed at -0.2 for 1200 iterations:

                               at -0.4      at -0.2
        Episode_Reward          -3.79        -3.51      -7%
        raw quantity             9.43        17.54      +86%
        Policy/mean_std          0.58         0.79
        soft_touchdown          -0.118       -0.116     unchanged

    The policy expanded its action changes to fill the slack. The share went 32%
    to 30%, nothing reached `soft_touchdown`, and the robot began stepping in
    place under a zero command -- which the operator saw before any logged number
    did.

    So this pins -0.4, and the reason is not that -0.4 is calibrated. It is that
    **-0.2 was tried and measured**, and that the manoeuvre it belongs to does not
    work: a term that is not biting has to be raised on its own.
    """
    import tasks

    cfg = tasks.load_env_cfg("jumper.posture")
    weight = cfg.rewards["action_rate_l2"].weight
    assert weight == pytest.approx(-0.4), (
        f"action_rate_l2 is {weight}; -0.2 was run for 1200 iterations and cost "
        f"7% less while the motion grew 86%, so any change here needs its own "
        f"measurement rather than the rate argument or the budget argument"
    )

    # **Not** asserted: that this is the heaviest penalty by weight. It is the
    # heaviest by realised cost -- 9.43 x 0.4 = 3.77 a step against `joint_acc`'s
    # 0.60 -- and weights are not comparable across terms whose quantities have
    # different units. `foot_clearance` carries -4.0 on metres-times-speed and
    # costs 0.0194. The first draft of this test compared the weights and failed,
    # which is the confusion worth not encoding.
def test_no_rung_is_climbed_before_an_episode_has_been_run() -> None:
    """The first promotion waits for evidence, not just for the dwell.

    **This pins a promotion that was legal and wrong.** On the run of
    2026-09-17, the velocity ladder went from level 0 to level 1 at iteration 49
    on `lin_err` 0.131 against a 0.1515 bar -- while `lin_err` and `ang_err` were
    both still rising. An episode here is 4000 control steps, which is 83
    iterations of 48, so *no environment had finished one*. Every sample behind
    that number came from an environment that had terminated early, and an early
    termination contributes a small accumulated error that reads exactly like
    good tracking.

    The dwell does not protect against this and was never meant to: it is a gap
    between promotions, and the gap before the first one is measured from step 0.
    Worse, it is held in environment steps -- 2400 of them is iteration 100 at
    the velocity tasks' rollout of 24 and iteration 50 at this task's 48, so the
    same constant means half as much waiting here.

    The control group is the guard turned off, in the same stub: the first
    promotion then lands at the dwell, which is the behaviour this replaces.
    """
    import torch

    from tasks.jumper.posture.env_cfg import (
        POSTURE_ANGLES,
        POSTURE_HEIGHTS,
        POSTURE_STAND_ANGLES,
    )
    from tasks.jumper.posture.mdp.curriculum import (
        DWELL_STEPS,
        WARMUP_ITERATIONS,
        WARMUP_STEPS,
        PostureRangeCurriculum,
    )
    from tasks.jumper.posture.rl_cfg import NUM_STEPS_PER_ENV

    axes = ("twist", "pitch", "roll", "height")
    episode = 4000

    class _Cmd:
        class cfg:
            resampling_time_range = (3.0, 8.0)

            class ranges:
                twist = pitch = roll = 0.0
                height = (0.0, 0.0)

            class moving:
                twist = pitch = roll = 0.0

        # Far inside every bar on every axis, so the only thing that can refuse a
        # promotion is a clock.
        metrics = {
            f"error_{a}": torch.full((64,), 1e-4 * episode / (8.0 / 0.005))
            for a in axes
        }

    class _Rewards:
        def get_term_cfg(self, name):
            class T:
                params: dict = {}

            return T()

    class _Env:
        step_dt = 0.005
        common_step_counter = 0
        episode_length_buf = torch.full((64,), episode)
        command_manager = type("M", (), {"get_term": staticmethod(lambda n: _Cmd())})()
        reward_manager = _Rewards()

    class _Cfg:
        params = {
            "angles": POSTURE_ANGLES,
            "heights": POSTURE_HEIGHTS,
            "stand_angles": POSTURE_STAND_ANGLES,
        }

    def promotes_at(counter: int, warmup: int) -> bool:
        """A fresh ladder, one batch of sixteen resets, at that step counter."""
        env = _Env()
        env.common_step_counter = counter
        curriculum = PostureRangeCurriculum(_Cfg(), env)
        out = curriculum(
            env,
            torch.arange(16),
            command_name="posture",
            angles=POSTURE_ANGLES,
            heights=POSTURE_HEIGHTS,
            stand_angles=POSTURE_STAND_ANGLES,
            warmup_steps=warmup,
        )
        return bool(out["promoted"])

    def first_promotion(warmup: int) -> int:
        lo, hi = 1, 100_000
        assert promotes_at(hi, warmup), "nothing promotes at all; the stub is wrong"
        while lo < hi:
            mid = (lo + hi) // 2
            if promotes_at(mid, warmup):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # The control group: without the guard the dwell is the only clock, and it
    # fires at iteration 25 -- a third of the way into the first episode.
    assert first_promotion(0) == DWELL_STEPS, (
        "with the guard disabled the first promotion is not at the dwell, so "
        "the search below is measuring something other than the guard"
    )

    assert first_promotion(WARMUP_STEPS) == WARMUP_STEPS, (
        f"a rung was climbed before {WARMUP_ITERATIONS} iterations, on evidence "
        f"from environments that had only ever terminated early"
    )

    # And the floor is above one episode, which is the whole point of its value:
    # at least one environment must have run a full command sequence to the end.
    assert WARMUP_STEPS > episode, (
        f"{WARMUP_ITERATIONS} iterations of {NUM_STEPS_PER_ENV} is "
        f"{WARMUP_STEPS} steps, shorter than the {episode}-step episode it is "
        f"meant to outlast"
    )


def test_both_ladders_wait_the_same_two_hundred_iterations() -> None:
    """The guard is on the velocity ladder too, which is where it was earned.

    The iteration-49 promotion was `Curriculum/command`, not
    `Curriculum/posture`. A floor applied only to the ladder that did not misfire
    would leave the logged failure in place, and the two would then be paced by
    different clocks -- which is the thing the shared key names exist to avoid.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import POSTURE_COMMAND
    from tasks.jumper.posture.mdp.curriculum import WARMUP_ITERATIONS, WARMUP_STEPS
    from tasks.jumper.posture.rl_cfg import NUM_STEPS_PER_ENV

    assert WARMUP_STEPS == WARMUP_ITERATIONS * NUM_STEPS_PER_ENV, (
        "the warm-up is no longer this task's rollout times its iterations, so "
        "the number below is not the number of iterations it reads as"
    )

    train = tasks.load_env_cfg("jumper.posture", play=False)
    for term in (POSTURE_COMMAND, "command"):
        assert train.curriculum[term].params.get("warmup_steps") == WARMUP_STEPS, (
            f"the {term} ladder can promote before {WARMUP_ITERATIONS} "
            f"iterations, which is before its first episode has ended"
        )


def _gravity_after_pitch(theta: float) -> tuple[torch.Tensor, float]:
    """Base-frame gravity for a body turned `theta` about its own +y, and where its
    nose then points: the world z of the body's +x. Built from the rotation, so the
    nose is read off the geometry rather than off either task's sign convention."""
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)
    gravity = rot.T @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    nose_z = float((rot @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))[2])
    return gravity.float().unsqueeze(0), nose_z


def test_posture_and_five_foot_read_pitch_the_same_way() -> None:
    """Pins the sign both tasks' pose commands put on `Command::base_pitch`.

    The robot has one pitch channel and a mode switch between the two policies,
    so a sign that differs by task is a stick that tips the nose down in one mode
    and up in the next -- each task consistent with itself and nothing anywhere
    raising. It did differ until 2026-09-26: this task measured nose-up positive
    and `jumper.five_foot` nose-down.

    The nose comes from the rotation, not from either formula. The control group
    is the pre-flip formula, which reads the same bodies with the opposite sign,
    so this check can tell the two conventions apart.
    """
    from tasks.jumper.five_foot.mdp.pose_command import base_pitch_roll
    from tasks.jumper.posture.mdp.state import body_tilt

    class _Env:
        def __init__(self, g: torch.Tensor) -> None:
            data = type("D", (), {"projected_gravity_b": g})()
            self.scene = {"robot": type("R", (), {"data": data})()}

    asset = type("A", (), {"name": "robot"})()
    for theta in (math.radians(10.0), math.radians(-10.0)):
        g, nose_z = _gravity_after_pitch(theta)
        posture = float(body_tilt(_Env(g), asset)[0, 0])
        five = float(base_pitch_roll(g)[0][0])
        assert posture == pytest.approx(five, abs=1e-6), (
            f"a body at {math.degrees(theta):+.0f} deg about +y reads {posture:+.4f} "
            f"here and {five:+.4f} in five_foot"
        )
        assert (posture > 0) is (nose_z < 0), (
            f"the nose is {'down' if nose_z < 0 else 'up'} and pitch reads "
            f"{math.degrees(posture):+.1f} deg; nose down is +pitch"
        )
        before = math.atan2(-float(g[0, 0]), math.hypot(float(g[0, 1]), float(g[0, 2])))
        assert before == pytest.approx(-posture, abs=1e-6), "the control group moved"


def test_posture_leans_as_far_as_five_foot_does() -> None:
    """The two tasks' pose bands are one decision, written in both.

    A stick at full deflection is the standing edge in either mode, and the
    controller holds a walking robot to the moving band from the contract, so a
    band that differs by task is the same stick asking for two different leans --
    which the operator reads as one policy being worse than the other. Written out
    in each task on purpose, per this repository's rule; this is what notices when
    one of them moves.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import POSTURE_COMMAND

    five = tasks.load_env_cfg("jumper.five_foot").commands["body_pose"]
    posture = tasks.load_env_cfg("jumper.posture").commands[POSTURE_COMMAND]
    for axis in ("twist", "pitch", "roll"):
        stand, move = getattr(posture.ranges, axis), getattr(posture.moving, axis)
        assert (-stand, stand) == pytest.approx(getattr(five.ranges, axis)), axis
        assert (-move, move) == pytest.approx(getattr(five.moving, axis)), axis
    assert posture.stand_threshold == five.stand_threshold


def test_a_walking_posture_is_drawn_from_and_held_to_the_moving_band() -> None:
    """Parked, the angles come from the standing band; walking, from the moving one.

    Half the field is given a walking velocity command and half a parked one.
    The walking half must never exceed the moving band. The control group is the
    parked half, which has to reach past it on every axis whose standing band is
    wider -- otherwise "inside the moving band" could be a band that was simply
    never wider. Then the operator's case: a full stick on every environment is
    brought back to the moving band on the walking half only.
    """
    import tasks
    from tasks.jumper.posture.env_cfg import POSTURE_COMMAND
    from tasks.jumper.posture.mdp.commands import ANGLES, PostureCommand

    cfg = tasks.load_env_cfg("jumper.posture").commands[POSTURE_COMMAND]
    n = 4096
    twist = torch.zeros(n, 3)
    twist[n // 2:, 0] = 0.3

    class _Commands:
        def get_command(self, name: str) -> torch.Tensor:
            return twist

        def get_term(self, name: str):
            raise KeyError(name)

    term = PostureCommand.__new__(PostureCommand)
    term.cfg = cfg
    term._env = type("E", (), {"num_envs": n, "device": "cpu", "command_manager": _Commands()})()
    term.posture_command = torch.zeros(n, 4)
    term.is_neutral_env = torch.zeros(n, dtype=torch.bool)
    torch.manual_seed(0)
    term._resample_command(torch.arange(n))

    parked, walking = term.posture_command[: n // 2], term.posture_command[n // 2:]
    for i, axis in enumerate(ANGLES):
        stand, move = getattr(cfg.ranges, axis), getattr(cfg.moving, axis)
        assert float(walking[:, i].abs().max()) <= move + 1e-6, f"a walking {axis} past 15 deg"
        assert float(parked[:, i].abs().max()) <= stand + 1e-6, f"a parked {axis} past its band"
        if stand > move:
            assert float(parked[:, i].abs().max()) > move, (
                f"no parked {axis} reached past the moving band; the standing band is "
                "not being drawn from"
            )

    full = [getattr(cfg.ranges, axis) for axis in ANGLES]
    term.posture_command[:, :3] = torch.tensor(full)
    term.hold_to_band()
    for i, axis in enumerate(ANGLES):
        assert float(term.posture_command[0, i]) == pytest.approx(full[i]), axis
        assert float(term.posture_command[-1, i]) == pytest.approx(getattr(cfg.moving, axis))
