#!/usr/bin/env python3
"""The claw and the objects, in **plain MuJoCo**: no mjlab, no managers, no scene.

    python tasks/jumper/five_foot/tools/pick_place.py --bundle tasks/jumper/five_foot/out/model_6999

kk-rl-lab's `deploy/hexa_5d/backends/mujoco/` in one file: build the model, load
an exported policy, step MuJoCo, drive it from the keyboard. It exists because
that is the arrangement this robot is actually deployed under -- a plain MuJoCo
model, a policy that reads a flat observation vector and a controller that writes
torques -- and a second implementation of it here is the only thing that can
disagree with `scripts/play.py`. Where they disagree, one of them is wrong, and
finding that out in simulation is cheap.

**Nothing here goes through `scenes/`.** The objects are attached to the robot's
spec directly, which is the shape `pick_place.py` has there.

What it does **not** share with `scripts/play.py` is everything that matters:

    play.py                         this
    mjlab env + managers            MuJoCo model and data
    warp or the native backend      mujoco.mj_step
    observations from the manager   assembled here, from layout.json
    actions through an action term  torque written to qfrc_applied
    the servo actuator class        the same curve, reimplemented in numpy

So the two agree only if the contract is right, which is the point.
`tasks/jumper/five_foot/tools/mujoco_parity.py` is the check that says whether they do.

## Standing is right; walking is not, yet

Commanded nothing, this is the same robot: it settles at 106 mm against a 105 mm
nominal, |action| 0.33, 0.7 N*m of torque, joint velocities at 0.00 rad/s, and it
stays there. Commanded 0.3 m/s forward it travels 0.107 m in 10 s where the
manager-driven environment travels 2.87, thrashing in place with actions of the
same size.

Each half of the loop has been verified against that environment and each half
works:

  - the compiled models are identical field by field -- joint ranges, armature,
    damping, friction, solref, solimp, condim, priority, contype, masses,
    inertias and every solver option;
  - the observation is identical for the same state, term by term, to 1.2e-07;
  - the torque curve is identical to 2.2e-16 and the PD is the same law;
  - the ONNX and the training policy agree to 1.7e-06 on live observations;
  - **this simulation walks 2.52 m in 10 s when replayed with the environment's
    actions**, tracking its trajectory to 6 mm over the first 250 steps;
  - **the environment walks 1.87 m when driven by this file's observations.**

So both halves work when crossed and the loop does not when closed, which is the
signature of a feedback phase rather than a wrong quantity. That is what is being
looked for; until it is found, drive this to look at the scene and the claw, and
use `scripts/play.py` to judge a gait.

## The geometry is the repository's, the plumbing is not

The prop shapes come from `objects.py` -- a bar of soap, a notebook, a can and a
bin, each a primitive of one material. Copying them into this file would make two
rows that drift apart; reading them here is not the same thing as running the
task's `--objects` setup through mjlab, which is what this avoids.

## Keys

    arrows            forward / back / strafe
    A / D             turn            R  stop
    KP 8 / 2          nose down / up  KP 4 / 6  roll   KP 7 / 9  twist  KP 5  level
    [ / ]             open / close the claw
    X                 reset
    space             pause

Nothing is welded. The claw holds what it squeezes the way a real one does, by
friction: `]` until the jaws stop on an object and it comes up with the claw, `[`
and it drops. Before that the props are moved by contact and nothing else, so the
claw can be walked into one and push it. The friction that holds is the objects
scene's own solver settings (`tasks/jumper/five_foot/objects.py::PROP_CONE`), set here on the
model that carries the objects and on no other.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
JUMPER_XML = REPO / "assets" / "jumper" / "jumper.xml"

#: The physics settings the task trains with, from its `sim.mujoco` config.
#: Written out rather than imported because importing the task config pulls in
#: torch and mjlab, which is the dependency this file exists without; they are
#: **checked** against the live config by `tasks/jumper/five_foot/tools/mujoco_parity.py`.
SIM = {
    "timestep": 0.005,
    "iterations": 10,
    "ls_iterations": 20,
    "tolerance": 1e-8,
    "ls_tolerance": 0.01,
    "impratio": 1.0,
}

# ── The model ────────────────────────────────────────────────────────────


def build_model(with_objects: bool = True):
    """`jumper.xml`, a floor, and the row of objects under the objects' friction.

    `with_objects=False` is the task's physics exactly, which is what
    `tasks/jumper/five_foot/tools/mujoco_parity.py` compares against the trainer's. The friction
    settings the objects need belong to the scene rather than the task, so they
    come with them.
    """
    from tasks.jumper.common.constants import ARMATURE
    from tasks.jumper.five_foot.jaws import JAW_COLLISION, decompose_jaws
    from tasks.jumper.five_foot.objects import PROP_CONE, PROP_IMPRATIO, _row

    # The jaws as `jumper.five_foot` builds them: the shared asset has one hull
    # each, and a claw built from it cannot close on anything (`jaws.py`).
    spec = decompose_jaws(mujoco.MjSpec.from_file(str(JUMPER_XML)))

    # The solver the task trains with. `jumper.xml` carries MuJoCo's defaults --
    # 100 Newton iterations and 50 line-search steps at a 1 ms timestep -- and
    # the task runs 10 and 20 at 5 ms. A *more* accurate solver is still a
    # different one: `tasks/jumper/five_foot/tools/mujoco_parity.py` compares these against the
    # live environment's, so they cannot drift apart quietly.
    spec.option.timestep = SIM["timestep"]
    spec.option.iterations = SIM["iterations"]
    spec.option.ls_iterations = SIM["ls_iterations"]
    spec.option.tolerance = SIM["tolerance"]
    spec.option.ls_tolerance = SIM["ls_tolerance"]
    spec.option.impratio = SIM["impratio"]

    # **What the XML does not carry.** `jumper.xml` is the robot as it comes out of
    # the URDF; the contact structure and the rotor inertia are the task's, and
    # mjlab writes them into the model when it builds the entity. A standalone
    # build that skipped them would be a different robot with the same shape --
    # the feet at condim 1 instead of 3, no rotor inertia on any joint -- which
    # is precisely the kind of difference that shows up as "the policy stands
    # there", not as an error.
    #
    # Applied from the same config objects the training run used, so there is one
    # description of the contacts and this is a second consumer of it.
    JAW_COLLISION.edit_spec(spec)
    for joint in spec.joints:
        if joint.name.endswith("_joint"):
            joint.armature = ARMATURE

    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
        pos=[0, 0, 0], rgba=[0.55, 0.55, 0.57, 1.0],
        condim=3, friction=[1.0, 0.005, 0.0001],
    )
    spec.worldbody.add_light(
        name="key", pos=[-0.6, -0.8, 2.0], dir=[0.25, 0.35, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True,
        diffuse=[0.7, 0.68, 0.64],
    )

    props: dict[str, tuple] = {}
    if with_objects:
        for name, cfg in _row().items():
            frame = spec.worldbody.add_frame()
            spec.attach(cfg.spec_fn(), prefix=f"{name}/", frame=frame)
            props[name] = (tuple(cfg.init_state.pos), tuple(cfg.init_state.rot))
        # What `objects.py::apply_objects` does to the friction model, done by
        # hand: under the task's own settings a squeezed object creeps out.
        spec.option.cone = (
            mujoco.mjtCone.mjCONE_ELLIPTIC if PROP_CONE == "elliptic"
            else mujoco.mjtCone.mjCONE_PYRAMIDAL
        )
        spec.option.impratio = PROP_IMPRATIO
    return spec.compile(), props


# ── The servo ────────────────────────────────────────────────────────────


class Servo:
    """PD bounded by the measured torque-speed curve, in numpy.

    `jumper.xml` declares no actuators: mjlab adds them from the task config, and
    this robot's is `ServoCurveActuator`, which computes its PD in Python and
    hands MuJoCo a torque. So there is nothing in the compiled model to drive --
    the torque is written to `qfrc_applied`, which is what that class does too.

    The numbers are imported rather than copied: `STIFFNESS`, `DAMPING` and the
    curve come from `tasks/jumper/common/`, the same objects the training run used.
    A copy here would be the drift `scripts/export.py`'s docstring is about.
    """

    def __init__(self, model, joint_names):
        from tasks.jumper.common.actuator import CURVE
        from tasks.jumper.common.constants import DAMPING, EFFORT_LIMIT, STIFFNESS

        self.kp, self.kd, self.limit = STIFFNESS, DAMPING, EFFORT_LIMIT
        self.curve = CURVE
        self.names = list(joint_names)
        self.qadr = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.names
        ])
        self.dadr = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.names
        ])
        self.torque = np.zeros(len(self.names))

    def available(self, speed):
        """Torque magnitude available at `speed`, from the measured curve."""
        w = np.abs(speed)
        limit = self.curve.plateau_torque * np.exp(
            -np.clip(w - self.curve.corner_speed, 0.0, None) / self.curve.decay_speed
        )
        return np.where(w >= self.curve.cutoff_speed, 0.0, limit)

    def apply(self, data, target):
        q, qd = data.qpos[self.qadr], data.qvel[self.dadr]
        raw = self.kp * (target - q) - self.kd * qd
        cap = np.minimum(self.available(qd), self.limit)
        self.torque = np.clip(raw, -cap, cap)
        data.qfrc_applied[self.dadr] = self.torque


# ── The observation ──────────────────────────────────────────────────────


class Observer:
    """Build the policy's input vector from `layout.json`.

    Every term is built **by name**, and a name the contract carries but this does
    not know is a hard failure: a deployment that quietly skipped a term would
    feed the policy a vector that is the right length and the wrong thing.
    """

    def __init__(self, layout, model):
        self.layout = layout
        obs = layout["observation"]
        self.dim = obs["dim"]
        self.terms = obs["terms"]
        self.obs_joints = layout["obs_joint_order"]
        self.act_joints = layout["action_joint_order"]
        self.default = layout["default_joint_pos"]
        def jid(n):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

        self.obs_qadr = np.array([model.jnt_qposadr[jid(n)] for n in self.obs_joints])
        self.obs_dadr = np.array([model.jnt_dofadr[jid(n)] for n in self.obs_joints])
        self.obs_default = np.array([self.default[n] for n in self.obs_joints])
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base < 0:
            self.base = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link"
            )
        self.sensors = {}
        for term in self.terms:
            name = (term.get("params") or {}).get("sensor_name")
            if name:
                # The contract names it as the trainer's model does, with the
                # entity prefix; a model loaded straight from the XML has no
                # prefix. Both spellings are tried so one file serves both.
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid < 0:
                    sid = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_SENSOR, name.split("/")[-1]
                    )
                if sid < 0:
                    raise SystemExit(f"[pick_place] no sensor {name!r} in the model")
                self.sensors[term["name"]] = (
                    model.sensor_adr[sid], model.sensor_dim[sid]
                )
        self.history: dict[str, np.ndarray] = {}

    zero_term: str | None = None

    def _now(self, term, data, action, twist, pose, torque):
        name = term["name"]
        if name == self.zero_term:
            return np.zeros(term.get("base_dim", term["dim"]))
        if name in self.sensors:
            adr, dim = self.sensors[name]
            return data.sensordata[adr:adr + dim].copy()
        if name == "projected_gravity":
            rot = data.xmat[self.base].reshape(3, 3)
            return rot.T @ np.array([0.0, 0.0, -1.0])
        if name == "joint_pos":
            return data.qpos[self.obs_qadr] - self.obs_default
        if name == "joint_vel":
            return data.qvel[self.obs_dadr].copy()
        if name == "actions":
            return np.asarray(action, dtype=float)
        if name == "velocity_commands":
            return np.asarray(twist, dtype=float)
        if name == "base_pose":
            return np.asarray(pose, dtype=float)
        if name == "joint_torque":
            return np.asarray(torque, dtype=float)
        raise SystemExit(
            f"[pick_place] the contract carries an observation term this does "
            f"not know how to build: {name!r}"
        )

    def __call__(self, data, action, twist, pose, torque, reset=False):
        out = np.zeros(self.dim, dtype=np.float32)
        for term in self.terms:
            name, offset = term["name"], term["offset"]
            value = self._now(term, data, action, twist, pose, torque)
            length = term.get("history_length", 1)
            if length == 1:
                out[offset:offset + term["dim"]] = value
                continue
            buf = self.history.get(name)
            if buf is None or reset:
                buf = np.tile(value, (length, 1))
            else:
                buf = np.roll(buf, -1, axis=0)
                buf[-1] = value
            self.history[name] = buf
            # `history_order: oldest_first`, flattened -- the contract says so
            # and this is the one place the order can be got wrong silently.
            assert self.layout["observation"]["history_order"] == "oldest_first"
            out[offset:offset + term["dim"]] = buf.reshape(-1)
        return out


# ── The run ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="tasks/jumper/five_foot/out/model_6999",
                    help="an export bundle: actor.onnx + layout.json")
    ap.add_argument("--steps", type=int, default=None, help="stop after this many")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="x real time")
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("VX", "VY", "WZ"),
                    help="the velocity command to start with, for headless runs")
    ap.add_argument("--zero-term", default=None,
                    help="diagnostic: build this observation term as zeros. For "
                         "asking which part of the observation a behaviour "
                         "depends on -- see the closed-loop note in the module "
                         "docstring")
    ap.add_argument("--report", action="store_true",
                    help="print where the robot got to when it stops")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    layout = json.loads((bundle / "layout.json").read_text())
    import onnxruntime

    policy = onnxruntime.InferenceSession(str(bundle / "actor.onnx"),
                                          providers=["CPUExecutionProvider"])

    model, props = build_model(with_objects=not args.no_objects)
    data = mujoco.MjData(model)
    obs = Observer(layout, model)
    obs.zero_term = args.zero_term
    wire = layout["wire_joint_order"]
    servo = Servo(model, wire)

    default = np.array([layout["default_joint_pos"][n] for n in wire])
    act_cols = np.array([wire.index(n) for n in layout["action_joint_order"]])
    obs_cols = np.array([wire.index(n) for n in layout["obs_joint_order"]])
    scale = layout["action_scale"]
    finger = wire.index("LF_J4_joint")
    decimation = layout["control"]["decimation"]
    control_dt = layout["control"]["control_dt"]

    from tasks.jumper.five_foot.claw import GRIPPER_CLOSED, GRIPPER_OPEN, SQUEEZE_LEAD

    state = {"twist": list(args.cmd), "pose": [0.0, 0.0, 0.0],
             "grip": GRIPPER_OPEN, "reset": False, "paused": False, "quit": False}

    def home():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = layout["base_init_pos"]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[servo.qadr] = default
        for name, (pos, quat) in props.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/")
            adr = model.jnt_qposadr[jid] if jid >= 0 else None
            if adr is None:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/{name}")
                adr = model.jnt_qposadr[model.body_jntadr[body]]
            data.qpos[adr:adr + 3] = pos
            data.qpos[adr + 3:adr + 7] = quat
        state["grip"] = GRIPPER_OPEN
        state["twist"][:] = list(args.cmd)
        state["pose"][:] = [0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    def key(code: int) -> None:
        t, p = state["twist"], state["pose"]
        step_lin, step_ang, step_pose = 0.1, 0.1, math.radians(2.0)
        if code == 265:   t[0] = round(min(0.5, t[0] + step_lin), 2)
        elif code == 264: t[0] = round(max(-0.5, t[0] - step_lin), 2)
        elif code == 263: t[1] = round(min(0.5, t[1] + step_lin), 2)
        elif code == 262: t[1] = round(max(-0.5, t[1] - step_lin), 2)
        elif code == ord("A"): t[2] = round(min(0.75, t[2] + step_ang), 2)
        elif code == ord("D"): t[2] = round(max(-0.75, t[2] - step_ang), 2)
        elif code == ord("R"): t[:] = [0.0, 0.0, 0.0]
        elif code == 324: p[0] = min(0.26, p[0] + step_pose)      # KP 8 nose down
        elif code == 322: p[0] = max(-0.26, p[0] - step_pose)     # KP 2 nose up
        elif code == 326: p[1] = min(0.26, p[1] + step_pose)      # KP 4
        elif code == 328: p[1] = max(-0.26, p[1] - step_pose)     # KP 6
        elif code == 327: p[2] = min(0.35, p[2] + step_pose)      # KP 7
        elif code == 329: p[2] = max(-0.35, p[2] - step_pose)     # KP 9
        elif code == 325: p[:] = [0.0, 0.0, 0.0]                  # KP 5
        elif code == ord("["):
            state["grip"] = max(GRIPPER_OPEN, state["grip"] - 0.05)
        elif code == ord("]"):
            state["grip"] = min(GRIPPER_CLOSED, state["grip"] + 0.05)
        elif code == ord("X"): state["reset"] = True
        elif code == 32: state["paused"] = not state["paused"]
        else: return
        print(f"  cmd vx={t[0]:+.2f} vy={t[1]:+.2f} wz={t[2]:+.2f} | "
              f"pitch={math.degrees(p[0]):+.0f} roll={math.degrees(p[1]):+.0f} "
              f"twist={math.degrees(p[2]):+.0f} deg | grip {state['grip']:+.2f}", flush=True)

    home()
    action = np.zeros(len(act_cols), dtype=np.float32)
    vector = obs(data, action, state["twist"], state["pose"],
                 servo.torque[obs_cols], reset=True)
    name = policy.get_inputs()[0].name

    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data, key_callback=key)
    print(f"[pick_place] {model.nbody} bodies, "
          f"{1 / control_dt:.0f} Hz control, bundle {bundle}")

    step, wall = 0, time.time()
    try:
        while viewer is None or viewer.is_running():
            if args.steps is not None and step >= args.steps:
                break
            if state["reset"]:
                state["reset"] = False
                home()
                vector = obs(data, action, state["twist"], state["pose"],
                             servo.torque[obs_cols], reset=True)
                print("[pick_place] X -- scene reset")
            if not state["paused"]:
                action = policy.run(None, {name: vector[None, :]})[0][0]
                target = default.copy()
                target[act_cols] += action * scale
                # No more than `SQUEEZE_LEAD` past where the finger is, as in replay.
                target[finger] = min(state["grip"], float(data.qpos[servo.qadr[finger]]) + SQUEEZE_LEAD)
                for _ in range(decimation):
                    servo.apply(data, target)
                    mujoco.mj_step(model, data)
                vector = obs(data, action, state["twist"], state["pose"],
                             servo.torque[obs_cols])
                step += 1
                if args.report and step % 100 == 0:
                    speed = float(np.linalg.norm(data.qvel[0:2]))
                    print(f"  [{step:5d}] |action| {np.abs(action).max():.3f}  "
                          f"speed {speed:.3f} m/s  base {data.qpos[2]*1000:.0f} mm  "
                          f"torque {np.abs(servo.torque).max():.2f} N*m  "
                          f"|qvel| max {np.abs(data.qvel[obs.obs_dadr]).max():.1f} "
                          f"mean {np.abs(data.qvel[obs.obs_dadr]).mean():.2f} rad/s")
            if viewer is not None:
                viewer.sync()
                if args.speed > 0:
                    ahead = wall + step * control_dt / args.speed - time.time()
                    if ahead > 0:
                        time.sleep(ahead)
    finally:
        if viewer is not None:
            viewer.close()
    print(f"[pick_place] {step} control steps")
    if args.report:
        travel = float(np.linalg.norm(data.qpos[0:2]))
        print(f"[pick_place] travelled {travel:.3f} m in {step * control_dt:.1f} s, "
              f"base at {data.qpos[2] * 1000:.0f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
