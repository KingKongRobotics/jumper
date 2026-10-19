"""`jumper.five_foot`'s deploy library against the robot it mirrors.

`tasks/jumper/five_foot/deploy/lib.rs` carries the claw on the right by showing
the policy the robot reflected in its x-z plane (see its docstring). Its two
tables -- which joint is whose twin, and which keep their sign -- are written in
Rust, and nothing on the robot can check them: a wrong sign is a joint driven
the wrong way by a policy that is otherwise working perfectly. So they are held
here against the two things that decide them, the table training mirrors with
and the model's own joint axes.

The mirror arithmetic itself is the library's own tests (`cargo test` in
`deploy/fsm`), which compile it in as the controller does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "tasks/jumper/five_foot/deploy/lib.rs"
JUMPER_XML = REPO / "assets/jumper/jumper.xml"


def _rust_strings(const: str) -> list[str]:
    """The string literals inside one `const NAME: ... = [...];` of the library."""
    text = LIB.read_text("utf-8")
    m = re.search(rf"const {const}: [^=]+= \[(.*?)\];", text, re.DOTALL)
    assert m, f"no `const {const}` in {LIB.name}; the test reads the wrong thing"
    return re.findall(r'"([^"]+)"', m.group(1))


def _rust_scalar(const: str) -> str:
    """The right-hand side of one scalar `const NAME: T = value;` of the library."""
    text = LIB.read_text("utf-8")
    m = re.search(rf"const {const}: [^=\[]+= ([^;]+);", text)
    assert m, f"no `const {const}` in {LIB.name}; the test reads the wrong thing"
    return m.group(1).strip()


def test_the_library_keeps_the_signs_training_keeps() -> None:
    """The same joints keep their sign on the robot as in training's mirror.

    `symmetry.py::_SIGN_KEEP` is what the six-legged tasks' augmentation trained
    under, and it is checked against the model by `test_symmetry.py`. Two tables
    that must agree and live in two languages agree only while something reads
    both.
    """
    from tasks.jumper.common.mdp.symmetry import _SIGN_KEEP

    keep = _rust_strings("SIGN_KEEP")
    assert keep, "the control group: the table parsed to something"
    assert set(keep) == set(_SIGN_KEEP)


def test_the_library_pairs_the_legs_training_pairs() -> None:
    from tasks.jumper.common.mdp.symmetry import _LEG_PAIR

    flat = _rust_strings("LEG_TWIN")
    pairs = dict(zip(flat[::2], flat[1::2], strict=True))
    assert pairs == _LEG_PAIR


def test_every_joint_on_the_wire_mirrors_by_its_axis() -> None:
    """Every joint's sign is the one its axis gives, and it maps its range onto
    its twin's.

    Wider than `test_symmetry.py`, which covers the twenty gait joints: the
    library mirrors all twenty-two, and the two it adds -- the claws' fingers --
    are exactly the ones five_foot observes and holds. Under the x-z mirror a
    joint rotation is a pseudovector, so a y axis keeps its sign and x or z
    negates; the range check is the physical half of the same claim, since a
    mirror whose sign maps a joint's range off its twin's is commanding a pose
    the other side cannot reach.
    """
    mujoco = pytest.importorskip("mujoco", reason="the axes are read off the model")

    model = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    keep = set(_rust_strings("SIGN_KEEP"))
    flat = _rust_strings("LEG_TWIN")
    twin_leg = dict(zip(flat[::2], flat[1::2], strict=True))
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert len(names) == 22 and "LF_J4_joint" in names and "RF_J4_joint" in names

    signs = set()
    for name in names:
        twin = twin_leg[name[:2]] + name[2:]
        assert twin in names, f"{name} has no twin {twin} in the model"
        jid, tid = model.joint(name).id, model.joint(twin).id
        axis = model.jnt_axis[jid]
        dominant = max(range(3), key=lambda k: abs(axis[k]))
        assert abs(axis[dominant]) > 1 - 1e-9, f"{name} axis {axis} is not cardinal"
        expected = 1.0 if dominant == 1 else -1.0
        library = 1.0 if name in keep else -1.0
        assert library == expected, (
            f"{name} turns about {'xyz'[dominant]}, so it mirrors with sign "
            f"{expected:+.0f}; the library says {library:+.0f}"
        )
        lo, hi = sorted(library * model.jnt_range[jid])
        tlo, thi = model.jnt_range[tid]
        assert abs(lo - tlo) < 1e-3 and abs(hi - thi) < 1e-3, (
            f"{name}'s range, mirrored, is [{lo:.3f}, {hi:.3f}] and {twin}'s is "
            f"[{tlo:.3f}, {thi:.3f}]"
        )
        signs.add(library)

    # The rule is not uniform -- J1 keeps its sign on the arms and negates on the
    # legs -- so a table of all -1.0 has to fail above rather than pass.
    assert signs == {1.0, -1.0}


def test_the_library_drives_the_claw_claw_py_measured() -> None:
    """The finger, its travel and its squeeze limit on the robot are replay's.

    `claw.py` measures all three -- the aperture sweep for the two ends, the
    turn under a squeezed can for the lead -- and `mdp/gripper.py` drives replay
    from them. The robot's `deploy/lib.rs` carries its own copy, in Rust, and a
    copy that drifted would squeeze the can harder, or shut short of shut, on
    the robot alone. Compared as the numbers the two languages write.
    """
    from tasks.jumper.five_foot.claw import (
        FINGER_JOINT,
        GRASP_BOX,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        LEAD_PER_SPEED,
        SQUEEZE_LEAD,
    )

    assert _rust_scalar("FINGER") == f'"{FINGER_JOINT}"'
    # GRIPPER_OPEN is no longer where the robot's finger opens to -- rl-wbc-fsm's
    # 60 and 90 degrees are -- but it is still the end of the travel the policy
    # is shown, and GRASP_BOX the box the arm has to be back in before the
    # policy is shown it again: both are training's, so both are held here.
    for name, value in [("GRIPPER_OPEN", GRIPPER_OPEN), ("GRIPPER_CLOSED", GRIPPER_CLOSED),
                        ("SQUEEZE_LEAD", SQUEEZE_LEAD), ("LEAD_PER_SPEED", LEAD_PER_SPEED),
                        ("GRASP_BOX", GRASP_BOX)]:
        assert float(_rust_scalar(name)) == value, (
            f"{name}: the library says {_rust_scalar(name)}, claw.py {value}"
        )


def test_the_arm_follows_the_stick_the_operator_holds() -> None:
    """rl-wbc-fsm's follow gains multiply the stick, `-1..1`; the library is
    handed the command, in rad, and divides it by the full stick's pitch. That
    has to be the standing band's edge this task trains and the operator scales
    to -- a library that divided by 15 degrees would carry a third more pitch
    onto the arm than the robot's operators tuned, and nothing would say so.
    Read off the task's own config, and both ends: the operator scales each
    end separately (`controls.yaml`).
    """
    import math

    import tasks
    from tasks.jumper.five_foot.env_cfg import POSE_COMMAND

    band = tasks.load_env_cfg("jumper.five_foot").commands[POSE_COMMAND].stand_pitch
    full = float(_rust_scalar("PITCH_FULL_DEG"))
    assert (math.degrees(-band[0]), math.degrees(band[1])) == pytest.approx((full, full)), (
        f"the library's full stick is {full} deg; the task's standing band is {band} rad"
    )


def test_each_claw_answers_the_control_its_task_declares() -> None:
    """The library answers `claw_left` for the claw on the left and `claw_right`
    on the right; both have to be controls the task's `controls.yaml` declares
    (`task:`), or the controller refuses the mode -- and the left one is the
    control replay's claw follows, since replay only ever carries it there. The
    rest of what the file declares is the arm's three presets, which both sides
    read (`ARM_CONTROLS` in the library).
    """
    from pathlib import Path

    from tasks.jumper.common.mdp.controls import load_controls
    from tasks.jumper.five_foot.mdp.gripper import CONTROL

    text = LIB.read_text("utf-8")
    sides = dict(re.findall(r'"(left|right)" => \((?:None|Some\([^)]*\)\?\)), "(\w+)"\)', text))
    assert sides == {"left": "claw_left", "right": "claw_right"}, sides
    kept = load_controls(Path(REPO / "tasks/jumper/five_foot/controls.yaml")).task
    arm = set(_rust_strings("ARM_CONTROLS"))
    assert arm == {"arm_thumb_up", "arm_thumb_down", "arm_web_up", "arm_to_face"}, arm
    assert set(sides.values()) | arm == set(kept)
    assert sides["left"] == CONTROL
