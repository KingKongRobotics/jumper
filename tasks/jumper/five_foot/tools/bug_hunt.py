#!/usr/bin/env python3
"""Search for bugs, pick one up, and drop it in the tray. Then search again.

    python tasks/jumper/five_foot/tools/bug_hunt.py --probe
    python tasks/jumper/five_foot/tools/bug_hunt.py --checkpoint <run>/model_N.pt

A replay on ``jumper.five_foot``. The policy only tracks a velocity command.
This file decides when a bug has been seen, walks the body there, holds the
arm out, closes the claw, and carries what it kept to the tray. It is not a
training task and not a deploy app. The soap row (``--objects``) is untouched.

Bugs are still. Their poses are read only to test whether they lie in the
onboard camera's forward cone, out to 1.2 m. The tray is a fixture, so its
pose is known the whole time. Nothing here is a furnished room, and nothing
moves except the robot and whatever the claw pushes.

The arm pose and the lift below are from ``--probe`` (native CPU, one
environment, no policy, 2026-10-08). Each preset endpoint leaves the mouth
between 77 and 158 mm up. A 30 mm bug on the floor has its centre at 15 mm,
and the shoulder lift carried none of them: ratio 0.00 at every endpoint.

The straight path from the stow to thumb-down is the one that dips. At 0.40
of the way, with the shoulder another 0.26 rad in the direction that lowers
the mouth, the mouth is at 18 mm (``GRASP_POSE``, -54, -51, -6, -45 degrees).
The bug was seated on the floor at the horizontal position ``Claw.seat``
gives it, the finger closed, and the shoulder lifted 0.25 rad the other way.
That carried it on 3 of 3 trials: the bug rose 19.1 to 19.7 mm against the
anvil's 21.7 to 22.4, ratio 0.88. The shoulder that raises the anvil is the
negative direction.

Grasping is friction, under ``objects.py``'s elliptic cone. A squeeze on the
training solver lets the bug creep out. Picked up means the bug rose at least
``FOLLOWED`` of the way the anvil rose. Delivered means its origin is inside
the tray walls and below the rim, scaled to this bug rather than to the can
the tray's own threshold was sized for -- a bug still in the claw sits higher
than that can's line and would otherwise count as already dropped.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from tasks.jumper.five_foot.claw import (
    ARM_JOINTS,
    FINGER_JOINT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LF_GRASP,
)
from tasks.jumper.five_foot.mdp.grasp import Claw
from tasks.jumper.five_foot.mdp.pose_command import PITCH
from tasks.jumper.five_foot.mdp.gripper import squeeze_limited
from tasks.jumper.five_foot.objects import (
    BIN_INNER_HALF,
    BIN_WALL_H,
    BIN_WALL_T,
    OBJECTS_NCONMAX,
    OBJECTS_NJMAX,
    PROP_CONE,
    PROP_IMPRATIO,
    _bin_spec,
    _free_body,
    _place_props,
    _prop,
    _prop_geom,
    _prop_spec,
)
from tasks.jumper.five_foot.tools.grasp_objects import FOLLOWED

#: Sphere 30 mm across, a few grams. Inside the open mouth (about 73 mm) and
#: light enough that the five-foot stand does not have to relearn the carry.
BUG_RADIUS = 0.015
BUG_DENSITY = 500.0
N_BUGS = 3

#: Known fixture, to the robot's right, outside the forward cone at the spawn.
TRAY_XY = (0.0, -1.15)

#: Onboard camera: a bug counts as seen inside this range. The cone's
#: half-angle is half of ``cam_fovy``, read off the compiled model.
SPOT_RANGE = 1.2

#: ``deploy/lib.rs`` ``PRESETS``, degrees, thumb up / thumb down / thumb-web up.
PRESET_DEG = {
    "thumb_up": (-90.0, -180.0, 0.0, -1.0),
    "thumb_down": (-90.0, -30.0, 30.0, -1.0),
    "thumb_web_up": (-90.0, -90.0, 0.0, -1.0),
}
PRESETS = {name: np.deg2rad(angles) for name, angles in PRESET_DEG.items()}

#: How far along the stow-to-thumb-down path the mouth is lowest, and how much
#: further the shoulder goes to bring it down onto a bug. See the docstring.
GRASP_ALPHA = 0.40
GRASP_SHOULDER = 0.26

#: How far the shoulder moves during the lift check, and which way raises the
#: anvil. Re-measured at startup; this is what the probe found.
LIFT_RAD = 0.25
LIFT_SIGN = -1.0

#: The bug is in the jaws when its origin is this close, horizontally, to where
#: ``Claw.seat`` would put it. The anvil itself is one jaw, and closing on a
#: bug parked under the anvil misses.
SEAT_TOL = 0.012

#: Setpoint step toward the arm goal, per control step. ``deploy/lib.rs``
#: ``ARM_INTERP``.
ARM_INTERP = 0.30

#: Trained command edges (``env_cfg.py``). A command past these is a command
#: the policy was never shown.
LIN_LIM = 0.5
ANG_LIM = 2.0

SEARCH_VX = 0.18
#: Nose down, in radians. Positive pitch is nose down, and 15 degrees is the
#: edge of the band the policy tracks while it is walking. The onboard camera
#: looks forward, so a level search walks over a bug on the floor.
LOOK_DOWN = math.radians(15.0)
#: A bug this close is one to turn and look at, not to walk past.
LOOK_RANGE = 0.90
#: Control steps, at 50 Hz.
APPROACH_LIMIT = 750
CREEP_LIMIT = 400
CLOSE_STEPS = 50
VERIFY_STEPS = 40
CARRY_LIMIT = 1000
DROP_LIMIT = 100
EXTEND_LIMIT = 150

STOW = np.array([LF_GRASP[name] for name in ARM_JOINTS], dtype=np.float64)
GRASP_POSE = (1.0 - GRASP_ALPHA) * STOW + GRASP_ALPHA * PRESETS["thumb_down"]
GRASP_POSE = GRASP_POSE.copy()
GRASP_POSE[1] += GRASP_SHOULDER


def _bug_spec():
    spec = _prop_spec()
    body = _free_body(spec, "bug")
    _prop_geom(
        body, name="shell", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BUG_RADIUS, 0.0, 0.0], density=BUG_DENSITY,
        rgba=[0.55, 0.72, 0.18, 1.0],
    )
    return spec


def _delivered_z() -> float:
    """Rim test from ``Highlight``, using this bug's own half-height.

    ``HIGHLIGHT_DELIVERED_Z`` is half a wall above the tallest solid standing
    on the tray floor. A 15 mm bug in the claw is still under that line, so
    the same rule applied to the can would mark a carry as a drop.
    """
    return BIN_WALL_T + BUG_RADIUS + BIN_WALL_H / 2


def add_hunt_scene(cfg, seed: int, n: int = N_BUGS) -> list[str]:
    """Bugs scattered off the spawn's forward axis, and the tray.

    Does not call ``apply_objects``: that installs the soap row and forces a
    zero command. The friction cone is the part that has to come with any
    prop the claw is meant to hold.
    """
    rng = np.random.default_rng(seed)
    placed: list[np.ndarray] = []
    tray = np.array(TRAY_XY, dtype=np.float64)
    for _ in range(n * 40):
        if len(placed) == n:
            break
        radius = float(rng.uniform(0.9, 1.35))
        angle = float(rng.uniform(-math.pi, math.pi))
        # Wider than the camera's half-angle, so a forward-looking camera at
        # yaw 0 does not start the episode already looking at a bug.
        if abs(_wrap(angle)) < 1.4:
            continue
        xy = np.array([radius * math.cos(angle), radius * math.sin(angle)])
        if np.linalg.norm(xy - tray) < 0.45:
            continue
        if any(np.linalg.norm(xy - other) < 0.35 for other in placed):
            continue
        placed.append(xy)
    if len(placed) != n:
        raise RuntimeError(f"could only place {len(placed)} of {n} bugs")

    entities = cfg.scene.entities
    names = []
    props = {}
    for i, xy in enumerate(placed):
        name = f"bug_{i}"
        props[name] = _prop(_bug_spec, (float(xy[0]), float(xy[1]), BUG_RADIUS))
        entities[name] = props[name]
        names.append(name)
    props["tray"] = _prop(_bin_spec, (TRAY_XY[0], TRAY_XY[1], 0.0))
    entities["tray"] = props["tray"]
    # Without a reset event a free body stays at the origin in every environment,
    # which puts every bug inside the tray and ends the hunt on the first step.
    _place_props(cfg, props)

    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = PROP_CONE
    cfg.sim.mujoco.impratio = PROP_IMPRATIO
    return names


def pin_spawn(cfg) -> None:
    reset = cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("bug hunt pins the spawn, and this config has no reset_base")
    pose = dict(reset.params.get("pose_range", {}))
    pose["x"] = (0.0, 0.0)
    pose["y"] = (0.0, 0.0)
    pose["yaw"] = (0.0, 0.0)
    reset.params["pose_range"] = pose

    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.resampling_time_range = (1.0e6, 1.0e6)
    pose_cmd = cfg.commands["body_pose"]
    pose_cmd.rel_neutral_envs = 1.0
    pose_cmd.resampling_time_range = (1.0e6, 1.0e6)
    cfg.terminations.clear()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _into(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of ``quat`` (wxyz)."""
    inv = quat.copy()
    mujoco.mju_negQuat(inv, quat)
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(vec, dtype=np.float64), inv)
    return out


def _ids(value) -> list[int]:
    if isinstance(value, slice):
        raise RuntimeError("expected a list of ids, got a slice")
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(i) for i in value]


class Arm:
    """Position targets for the four arm joints and the finger.

    The action term does not drive these. Targets are written before each
    step, which is what the physics step then tracks.
    """

    def __init__(self, robot, device) -> None:
        self.robot = robot
        self.device = device
        ids, names = robot.find_joints(list(ARM_JOINTS), preserve_order=True)
        if tuple(names) != ARM_JOINTS:
            raise RuntimeError(f"arm joints came back as {names}, not {ARM_JOINTS}")
        self.arm_ids = torch.tensor(ids, device=device, dtype=torch.long)
        finger, _ = robot.find_joints([FINGER_JOINT], preserve_order=True)
        self.finger_id = torch.tensor(finger, device=device, dtype=torch.long)
        self.goal = STOW.copy()
        self.finger = GRIPPER_OPEN
        self.cmd = STOW.copy()

    def measured(self) -> np.ndarray:
        q = self.robot.data.joint_pos[0, self.arm_ids]
        return q.detach().cpu().numpy().astype(np.float64)

    def write_state(self, angles: np.ndarray, finger: float) -> None:
        pos = torch.tensor(angles, device=self.device, dtype=torch.float32).view(1, -1)
        vel = torch.zeros_like(pos)
        self.robot.write_joint_state_to_sim(pos, vel, joint_ids=self.arm_ids)
        fpos = torch.tensor([[finger]], device=self.device, dtype=torch.float32)
        self.robot.write_joint_state_to_sim(
            fpos, torch.zeros_like(fpos), joint_ids=self.finger_id
        )
        self.cmd = np.array(angles, dtype=np.float64)
        self.finger = finger
        self.goal = self.cmd.copy()

    def step_targets(self) -> None:
        self.cmd = self.cmd + ARM_INTERP * (self.goal - self.cmd)
        pos = torch.tensor(self.cmd, device=self.device, dtype=torch.float32).view(1, -1)
        self.robot.set_joint_position_target(pos, joint_ids=self.arm_ids)
        angle = self.robot.data.joint_pos[0, self.finger_id]
        speed = self.robot.data.joint_vel[0, self.finger_id]
        target = torch.tensor(self.finger, device=self.device, dtype=angle.dtype)
        limited = squeeze_limited(target, angle, speed=speed)
        self.robot.set_joint_position_target(
            limited.view(1, 1), joint_ids=self.finger_id
        )

    def at_goal(self, tol: float = 0.12) -> bool:
        return float(np.max(np.abs(self.measured() - self.goal))) < tol


class ShownArm:
    """While the arm is outside the trained box, show the policy the stow.

    ``joint_pos`` is relative to the default, and the default is ``LF_GRASP``,
    so zeros are the stow. ``deploy/lib.rs`` does this because a preset is
    about a radian outside every sample the policy trained on. The finger is
    left as measured: that travel is in the training distribution.
    """

    def __init__(self, env, arm: Arm) -> None:
        self.env = env
        self.arm = arm
        robot = env.scene["robot"]
        self.slices = []
        for term in ("joint_pos", "joint_vel", "actuator_force"):
            cols = self._columns(robot, term)
            if cols:
                start, width, n = self._span(term)
                flat = [f * n + c for f in range(width // n) for c in cols]
                self.slices.append((start, flat))

    def _span(self, term: str) -> tuple[int, int, int]:
        mgr = self.env.observation_manager
        start = 0
        for name, shape in zip(
            mgr._group_obs_term_names["actor"],
            mgr._group_obs_term_dim["actor"],
            strict=True,
        ):
            width = int(np.prod(shape))
            if name == term:
                cfg = mgr.get_term_cfg("actor", term)
                n = self._count(cfg)
                if width % n != 0:
                    raise RuntimeError(f"{term} width {width} is not a multiple of {n}")
                return start, width, n
            start += width
        raise KeyError(term)

    def _count(self, cfg) -> int:
        asset = cfg.params["asset_cfg"]
        if asset.joint_ids is not None and not isinstance(asset.joint_ids, slice):
            return len(_ids(asset.joint_ids))
        return len(_ids(asset.actuator_ids))

    def _columns(self, robot, term: str) -> list[int]:
        cfg = self.env.observation_manager.get_term_cfg("actor", term)
        asset = cfg.params["asset_cfg"]
        if term == "actuator_force":
            ids = _ids(asset.actuator_ids)
            ordered = [robot.actuator_names[i] for i in ids]
        else:
            ids = _ids(asset.joint_ids)
            ordered = [robot.joint_names[i] for i in ids]
        return [ordered.index(name) for name in ARM_JOINTS]

    def off_stow(self) -> bool:
        return float(np.max(np.abs(self.arm.measured() - STOW))) > 0.10

    def apply(self, obs) -> None:
        if not self.off_stow():
            return
        # ``wrapped.step`` is run under inference mode, so the tensor it hands
        # back refuses an in-place write. The policy has to see the stow, and
        # a clone is a normal tensor.
        actor = obs["actor"].detach().clone()
        for start, cols in self.slices:
            index = [start + c for c in cols]
            actor[:, index] = 0.0
        obs["actor"] = actor


def install_command(env) -> dict:
    """Write the hunt's velocity last, where the operator writes the stick.

    A standing environment's update zeros ``vel_command_b`` after anything
    written earlier in the step. Replacing ``_update_command`` and writing
    after the original is what keeps the command.
    """
    term = env.command_manager.get_term("twist")
    held = {"v": torch.zeros(3, device=env.device)}
    original = term._update_command

    def update(env_ids=None):
        original(env_ids)
        term.is_standing_env[:] = False
        term.vel_command_b[:] = held["v"]
        term.vel_command_w[:] = held["v"]

    term._update_command = update
    return held


def set_command(held: dict, vx: float, vy: float, wz: float) -> None:
    held["v"][0] = float(np.clip(vx, -LIN_LIM, LIN_LIM))
    held["v"][1] = float(np.clip(vy, -LIN_LIM, LIN_LIM))
    held["v"][2] = float(np.clip(wz, -ANG_LIM, ANG_LIM))


def install_pose(env, held: dict) -> None:
    """Pitch the nose where the hunt asks, through the same ramp the policy trained on.

    ``pin_spawn`` pins the attitude command at level. Searching level points the
    camera over a bug on the floor. The target is written after the operator's
    own update and the ramp is run once from there, so a key at rest does not
    put the nose back up.
    """
    term = env.command_manager.get_term("body_pose")
    held["pitch"] = 0.0
    original = term.compute

    def compute(dt, env_ids=None):
        saved = term.pose_command_b.clone()
        original(dt, env_ids)
        term.pose_command_b[:] = saved
        term.is_neutral_env[:] = False
        term.pose_target_b.zero_()
        term.pose_target_b[:, PITCH] = held["pitch"]
        term.hold_to_band()
        term._update_command(env_ids)

    term.compute = compute


def _root(entity) -> np.ndarray:
    return entity.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)


def _base(robot) -> tuple[np.ndarray, float, np.ndarray]:
    pos = robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
    quat = robot.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float64)
    return pos, _yaw(quat), quat


def camera_look(robot, model) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera origin, world look direction, and the cone half-angle in radians.

    The camera looks along its own -Z. Half of ``cam_fovy`` is the cone: that
    is the vertical half-angle, and the horizontal field is wider, so the cone
    sits inside the image.
    """
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot/onboard")
    if cam < 0:
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
    if cam < 0:
        raise RuntimeError("the model has no onboard camera")
    body = int(model.cam_bodyid[cam])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body).split("/")[-1]
    idx = robot.find_bodies([bname])[0][0]
    bpos = robot.data.body_link_pos_w[0, idx].detach().cpu().numpy().astype(np.float64)
    bquat = robot.data.body_link_quat_w[0, idx].detach().cpu().numpy().astype(np.float64)
    offset = np.zeros(3)
    mujoco.mju_rotVecQuat(offset, np.asarray(model.cam_pos[cam], dtype=np.float64), bquat)
    origin = bpos + offset
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, bquat, np.asarray(model.cam_quat[cam], dtype=np.float64))
    look = np.zeros(3)
    mujoco.mju_rotVecQuat(look, np.array([0.0, 0.0, -1.0]), quat)
    half = math.radians(float(model.cam_fovy[cam]) / 2.0)
    return origin, look, half


def in_view(origin, look, half: float, point: np.ndarray) -> bool:
    delta = point - origin
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4 or dist > SPOT_RANGE:
        return False
    return float(np.dot(delta / dist, look)) > math.cos(half)


def over_tray(point: np.ndarray, tray: np.ndarray) -> bool:
    """The point is inside the tray's footprint, height ignored.

    A bug still in the claw is above the rim line ``in_tray`` uses, so that
    test cannot be what decides to open. The footprint can.
    """
    return (
        abs(point[0] - tray[0]) < BIN_INNER_HALF
        and abs(point[1] - tray[1]) < BIN_INNER_HALF
    )


def in_tray(point: np.ndarray, tray: np.ndarray) -> bool:
    return over_tray(point, tray) and point[2] < _delivered_z()


def mouth_in_base(robot, claw: Claw) -> np.ndarray:
    pos, _, quat = _base(robot)
    mouth = claw.mouth()[0].detach().cpu().numpy().astype(np.float64)
    return _into(quat, mouth - pos)


def body_error(robot, target_xy: np.ndarray, point_xy: np.ndarray):
    """Velocity that puts a body-frame point on ``target_xy``.

    The body's heading is the one that aims ``point_xy`` at the target, which
    is not the heading that aims the trunk there when the point sits off the
    centre line. Aiming the trunk instead has no pose where the point is on
    the target and the heading matches, and the walk circles the spot. During
    the approach the point is where the grasp pose seats a bug; during the
    carry it is where the bug actually is.
    """
    pos, yaw, _ = _base(robot)
    bearing = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0])
    # The point's bearing in the body. The trunk yaws this far the other way
    # so the point, not the trunk, faces the target.
    desired = _wrap(bearing - math.atan2(point_xy[1], point_xy[0]))
    c, s = math.cos(desired), math.sin(desired)
    offset = np.array([
        c * point_xy[0] - s * point_xy[1],
        s * point_xy[0] + c * point_xy[1],
    ])
    err_w = (target_xy - offset) - pos[:2]
    err_b = np.array([
        math.cos(yaw) * err_w[0] + math.sin(yaw) * err_w[1],
        -math.sin(yaw) * err_w[0] + math.cos(yaw) * err_w[1],
    ])
    yaw_err = _wrap(desired - yaw)
    # Loose on purpose: the creep covers the last centimetres, and a walking
    # policy does not settle inside a few centimetres while it is still turning.
    arrived = float(np.linalg.norm(err_w)) < 0.10 and abs(yaw_err) < 0.40
    return (
        float(np.clip(1.0 * err_b[0], -0.35, 0.35)),
        float(np.clip(1.0 * err_b[1], -0.25, 0.25)),
        float(np.clip(1.2 * yaw_err, -1.2, 1.2)),
    ), arrived


def in_jaws(env, claw: Claw, name: str, bug: np.ndarray) -> bool:
    """The bug is where a close carried it: on the floor, at ``Claw.seat``'s horizontal position."""
    seat = claw.seat(env, name)[0].detach().cpu().numpy()
    return float(np.hypot(bug[0] - seat[0], bug[1] - seat[1])) < SEAT_TOL


@dataclass
class Hunt:
    bugs: list[str]
    state: str = "search"
    bug: str | None = None
    seen: np.ndarray | None = None
    age: int = 0
    search_s: float = 0.0
    delivered: set[str] = field(default_factory=set)
    hold_yaw: float = 0.0
    seat_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lift_sign: float = LIFT_SIGN
    verify_z: tuple[float, float] | None = None
    creep_best: float = 1.0
    creep_err: np.ndarray = field(default_factory=lambda: np.zeros(2))
    nudge: int = 0
    settle: int = 0
    carry_z: float = 0.0
    preset: np.ndarray = field(default_factory=lambda: GRASP_POSE.copy())

    def go(self, state: str) -> None:
        if state != self.state:
            print(f"[hunt] {self.state} -> {state}"
                  + (f" ({self.bug})" if self.bug else ""))
        self.state = state
        self.age = 0
        self.settle = 0


def _body_bearing(robot, point: np.ndarray) -> tuple[float, float]:
    """Distance and bearing of ``point`` in the base frame. Bearing 0 is ahead."""
    pos, _, quat = _base(robot)
    rel = _into(quat, point - pos)
    return float(np.hypot(rel[0], rel[1])), math.atan2(rel[1], rel[0])


def _near(env, hunt: Hunt, robot) -> str | None:
    """The closest bug on the floor nearby, whether or not the camera is on it."""
    best, best_d = None, LOOK_RANGE
    for name in hunt.bugs:
        if name in hunt.delivered:
            continue
        dist, _bearing = _body_bearing(robot, _root(env.scene[name]))
        if dist < best_d:
            best, best_d = name, dist
    return best


def _spot(env, hunt: Hunt, model) -> str | None:
    robot = env.scene["robot"]
    origin, look, half = camera_look(robot, model)
    tray = _root(env.scene["tray"])
    best, best_d = None, SPOT_RANGE
    for name in hunt.bugs:
        if name in hunt.delivered:
            continue
        point = _root(env.scene[name])
        if in_tray(point, tray):
            hunt.delivered.add(name)
            print(f"[hunt] {name} is already in the tray")
            continue
        if not in_view(origin, look, half, point):
            continue
        dist = float(np.linalg.norm(point - origin))
        if dist < best_d:
            best, best_d = name, dist
    return best


def tick(env, hunt: Hunt, arm: Arm, model) -> None:
    """One control step of the mission. Sets the arm goal, the finger, nothing else.

    The caller writes the command from ``hunt`` after this returns, via the
    fields stashed on the function. Kept as attributes so the command install
    and the arm stay the only writers.
    """
    robot = env.scene["robot"]
    dt = float(env.step_dt)
    hunt.age += 1
    arm.finger = GRIPPER_OPEN
    arm.goal = STOW.copy()
    cmd = (0.0, 0.0, 0.0)
    tick.pitch = 0.0

    if hunt.state == "search":
        if len(hunt.delivered) == len(hunt.bugs):
            hunt.go("done")
        else:
            tick.pitch = LOOK_DOWN
            seen = _spot(env, hunt, model)
            if seen is None:
                seen = _near(env, hunt, robot)
            if seen is not None:
                _dist, bearing = _body_bearing(robot, _root(env.scene[seen]))
                # Still off to the side or behind: stop and turn while bent
                # over. Walking on would step over it.
                if abs(bearing) > 0.35:
                    cmd = (0.0, 0.0, float(np.clip(1.2 * bearing, -1.0, 1.0)))
                else:
                    hunt.bug = seen
                    hunt.seen = _root(env.scene[seen]).copy()
                    hunt.go("approach")
            else:
                hunt.search_s += dt
                radius = 0.35 + 0.04 * hunt.search_s
                cmd = (SEARCH_VX, 0.0, SEARCH_VX / radius)
    elif hunt.state == "approach":
        tick.pitch = LOOK_DOWN
        point = _root(env.scene[hunt.bug])
        origin, look, half = camera_look(robot, model)
        if in_view(origin, look, half, point):
            hunt.seen = point.copy()
        vel, arrived = body_error(robot, hunt.seen[:2], hunt.seat_xy)
        cmd = vel
        if arrived:
            _, hunt.hold_yaw, _ = _base(robot)
            hunt.go("extend")
            cmd = (0.0, 0.0, 0.0)
        elif hunt.age > APPROACH_LIMIT:
            pos, yaw, _ = _base(robot)
            print(f"[hunt] lost the approach to {hunt.bug} "
                  f"from ({pos[0]:+.2f}, {pos[1]:+.2f}) yaw {yaw:+.2f}, "
                  f"last seen ({hunt.seen[0]:+.2f}, {hunt.seen[1]:+.2f})")
            hunt.bug = None
            hunt.go("search")
    elif hunt.state == "extend":
        arm.goal = hunt.preset.copy()
        if arm.at_goal() or hunt.age > EXTEND_LIMIT:
            hunt.go("creep")
    elif hunt.state == "creep":
        arm.goal = hunt.preset.copy()
        bug = _root(env.scene[hunt.bug])
        seat = env.claw.seat(env, hunt.bug)[0].detach().cpu().numpy()
        _, yaw, quat = _base(robot)
        flat = _into(quat, bug - seat)[:2]
        gap = float(np.hypot(flat[0], flat[1]))
        if hunt.age == 1:
            hunt.creep_best = gap
        if gap <= hunt.creep_best:
            hunt.creep_best = gap
            hunt.creep_err = flat.copy()
        # A command under about 0.12 m/s never starts the gait, so the last
        # centimetres have to be walked at a speed the policy actually steps
        # at, and stopped once the seat is on the bug.
        if gap < SEAT_TOL:
            hunt.go("close")
            cmd = (0.0, 0.0, 0.0)
        elif hunt.age > CREEP_LIMIT:
            err = hunt.creep_err
            print(f"[hunt] {hunt.bug} never entered the mouth "
                  f"(gap {gap * 1000:.0f} mm, closest {hunt.creep_best * 1000:.0f} mm "
                  f"at ({err[0] * 1000:+.0f}, {err[1] * 1000:+.0f}) mm in the base)")
            hunt.bug = None
            hunt.go("stow")
        elif gap < 0.08:
            # A steady walk steps past, and a pulse shorter than a step only
            # leans. One step is about 0.4 s at the speed the gait actually
            # uses, then a pause so it can settle on the bug.
            hunt.nudge += 1
            yaw_err = _wrap(hunt.hold_yaw - yaw)
            if hunt.nudge % 60 < 20:
                direction = flat / max(gap, 1.0e-6)
                cmd = (
                    float(np.clip(direction[0] * 0.18, -0.25, 0.25)),
                    float(np.clip(direction[1] * 0.18, -0.20, 0.20)),
                    0.0 if abs(yaw_err) < 0.15 else float(np.clip(yaw_err, -0.5, 0.5)),
                )
            else:
                cmd = (0.0, 0.0, 0.0)
        else:
            hunt.nudge = 0
            direction = flat / max(gap, 1.0e-6)
            yaw_err = _wrap(hunt.hold_yaw - yaw)
            cmd = (
                float(np.clip(direction[0] * 0.18, -0.25, 0.25)),
                float(np.clip(direction[1] * 0.18, -0.20, 0.20)),
                0.0 if abs(yaw_err) < 0.15 else float(np.clip(yaw_err, -0.6, 0.6)),
            )
    elif hunt.state == "close":
        arm.goal = hunt.preset.copy()
        arm.finger = GRIPPER_CLOSED
        if hunt.age >= CLOSE_STEPS:
            bug_z = float(_root(env.scene[hunt.bug])[2])
            anvil_z = float(env.claw.mouth()[0, 2])
            hunt.verify_z = (bug_z, anvil_z)
            hunt.go("verify")
    elif hunt.state == "verify":
        arm.goal = hunt.preset.copy()
        # ARM_INTERP on the whole lift is a snap, and a bug in the jaws gets
        # flicked out. Raise the shoulder over half a second instead.
        frac = min(1.0, hunt.age / 25.0)
        arm.goal[1] = hunt.preset[1] + frac * hunt.lift_sign * LIFT_RAD
        arm.finger = GRIPPER_CLOSED
        if hunt.age >= VERIFY_STEPS:
            bug_z = float(_root(env.scene[hunt.bug])[2])
            anvil_z = float(env.claw.mouth()[0, 2])
            z0, a0 = hunt.verify_z
            rise_a = anvil_z - a0
            rise_b = bug_z - z0
            ok = rise_a > 0.005 and rise_b >= FOLLOWED * rise_a
            print(f"[hunt] verify {hunt.bug}: bug {rise_b * 1000:.1f} mm, "
                  f"anvil {rise_a * 1000:.1f} mm, "
                  f"{'carried' if ok else 'left behind'}")
            if ok:
                hunt.go("carry")
            else:
                hunt.bug = None
                arm.finger = GRIPPER_OPEN
                hunt.go("stow")
    elif hunt.state == "carry":
        arm.goal = hunt.preset.copy()
        arm.goal[1] = hunt.preset[1] + hunt.lift_sign * LIFT_RAD
        arm.finger = GRIPPER_CLOSED
        bug = _root(env.scene[hunt.bug])
        tray = _root(env.scene["tray"])
        mouth_z = float(env.claw.mouth()[0, 2])
        if hunt.age == 1:
            # Height above the anvil, not the floor. The gait crouches as it
            # starts, and the bug's world height falls with the body while it
            # is still in the jaws.
            hunt.carry_z = float(bug[2]) - mouth_z
        # A backward command walks the body onto the can while the claw, which
        # is in front, is still short of it, and the gait shakes the bug loose.
        # Face the can and walk forward, so the bug arrives first. If it has
        # already fallen, stop: steering at a bug on the floor walks the body
        # straight through the can.
        slipped = hunt.age > 1 and float(bug[2]) - mouth_z < hunt.carry_z - 0.025
        if slipped:
            print(f"[hunt] {hunt.bug} slipped on the way to the tray")
            hunt.bug = None
            arm.finger = GRIPPER_OPEN
            hunt.go("stow")
        elif over_tray(bug, tray):
            hunt.settle += 1
            if hunt.settle >= 15:
                hunt.go("drop")
        else:
            hunt.settle = 0
            pos, yaw, _ = _base(robot)
            yaw_err = _wrap(math.atan2(tray[1] - pos[1], tray[0] - pos[0]) - yaw)
            if abs(yaw_err) > 0.35:
                cmd = (0.0, 0.0, float(np.clip(1.2 * yaw_err, -1.0, 1.0)))
            else:
                cmd = (
                    0.22,
                    0.0,
                    0.0 if abs(yaw_err) < 0.15 else float(np.clip(yaw_err, -0.5, 0.5)),
                )
    elif hunt.state == "drop":
        arm.goal = hunt.preset.copy()
        arm.goal[1] = hunt.preset[1] + hunt.lift_sign * LIFT_RAD
        arm.finger = GRIPPER_OPEN
        bug = _root(env.scene[hunt.bug])
        if in_tray(bug, _root(env.scene["tray"])):
            hunt.delivered.add(hunt.bug)
            print(f"[hunt] {hunt.bug} is in the tray "
                  f"({len(hunt.delivered)}/{len(hunt.bugs)})")
            hunt.bug = None
            hunt.go("stow")
        elif hunt.age > DROP_LIMIT:
            print(f"[hunt] {hunt.bug} missed the tray")
            hunt.bug = None
            hunt.go("stow")
    elif hunt.state == "stow":
        arm.goal = STOW.copy()
        arm.finger = GRIPPER_OPEN
        if arm.at_goal(0.10) or hunt.age > EXTEND_LIMIT:
            hunt.go("search")
    elif hunt.state == "done":
        arm.goal = STOW.copy()

    if hunt.state == "done" and hunt.age == 1:
        print("[hunt] every bug is in the tray")
    # The gripper event runs during the step and, once a key has been touched,
    # writes the trigger. A trigger at rest is open, which is a drop wherever
    # the robot happens to be. Pin the finger to what this step asked for.
    _finger_hold(env, arm.finger)
    tick.command = cmd


tick.command = (0.0, 0.0, 0.0)
tick.pitch = 0.0


def _finger_hold(env, target: float) -> None:
    cfg = env.event_manager.get_term_cfg("gripper_teleop")
    hold = getattr(cfg.func, "hold_at", None)
    if hold is not None:
        hold(target)


def _place(entity, pos: torch.Tensor) -> None:
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos.device)
    entity.write_root_link_pose_to_sim(torch.cat([pos.view(1, 3), quat], dim=1))
    entity.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=pos.device))


def _zero_action(env) -> torch.Tensor:
    dim = env.action_manager.total_action_dim
    return torch.zeros(env.num_envs, dim, device=env.device)


def _hold_step(env, arm: Arm, angles: np.ndarray, finger: float) -> None:
    arm.goal = np.array(angles, dtype=np.float64)
    arm.cmd = arm.goal.copy()
    arm.finger = finger
    arm.step_targets()
    env.step(_zero_action(env))


def measure_preset(env, arm: Arm, claw: Claw, name: str) -> dict:
    """Floor bug under this preset's mouth: does the lift carry it?"""
    angles = PRESETS[name]
    env.reset()
    arm.write_state(angles, GRIPPER_OPEN)
    env.sim.forward()
    for _ in range(30):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    env.sim.forward()
    robot = env.scene["robot"]
    sign = _anvil_sign(env, arm, claw, angles)
    local = mouth_in_base(robot, claw)
    mouth = claw.mouth()[0]
    origin_z = float(env.scene.env_origins[0, 2])
    pos = mouth.detach().clone()
    pos[0] = mouth[0]
    pos[1] = mouth[1]
    pos[2] = origin_z + BUG_RADIUS
    bug = env.scene["bug_0"]
    _place(bug, pos)
    for _ in range(20):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    gap = float((bug.data.root_link_pos_w[0] - claw.mouth()[0]).norm())
    for _ in range(80):
        _hold_step(env, arm, angles, GRIPPER_CLOSED)
    z0 = float(bug.data.root_link_pos_w[0, 2])
    a0 = float(claw.mouth()[0, 2])
    for k in range(40):
        mid = angles.copy()
        mid[1] = angles[1] + sign * LIFT_RAD * (k + 1) / 40.0
        _hold_step(env, arm, mid, GRIPPER_CLOSED)
    rise_b = float(bug.data.root_link_pos_w[0, 2]) - z0
    rise_a = float(claw.mouth()[0, 2]) - a0
    ratio = rise_b / rise_a if rise_a > 1.0e-3 else 0.0
    return {
        "name": name,
        "mouth_mm": (local * 1000.0).round(1),
        "mouth_z_mm": float(mouth[2] - origin_z) * 1000.0,
        "gap_mm": gap * 1000.0,
        "rise_bug_mm": rise_b * 1000.0,
        "rise_anvil_mm": rise_a * 1000.0,
        "ratio": ratio,
        "sign": sign,
        "carried": rise_a > 0.005 and rise_b >= FOLLOWED * rise_a,
    }


def measure_grasp(env, arm: Arm, claw: Claw) -> dict:
    """Bug on the floor at ``Claw.seat``'s horizontal position, then the lift.

    This is the pose the hunt uses. The preset endpoints are measured by
    ``measure_preset``, which parks the bug under the anvil; that is where a
    floor bug is not.
    """
    angles = GRASP_POSE
    env.reset()
    for _ in range(25):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    origin_z = float(env.scene.env_origins[0, 2])
    seat = claw.seat(env, "bug_0")[0].detach().clone()
    pos = seat.clone()
    pos[2] = origin_z + BUG_RADIUS
    bug = env.scene["bug_0"]
    _place(bug, pos)
    for _ in range(12):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    mouth = claw.mouth()[0]
    local = mouth_in_base(env.scene["robot"], claw)
    for _ in range(60):
        _hold_step(env, arm, angles, GRIPPER_CLOSED)
    z0 = float(bug.data.root_link_pos_w[0, 2])
    a0 = float(claw.mouth()[0, 2])
    lifted = angles.copy()
    lifted[1] = angles[1] + LIFT_SIGN * LIFT_RAD
    for _ in range(30):
        _hold_step(env, arm, lifted, GRIPPER_CLOSED)
    rise_b = float(bug.data.root_link_pos_w[0, 2]) - z0
    rise_a = float(claw.mouth()[0, 2]) - a0
    ratio = rise_b / rise_a if rise_a > 1.0e-3 else 0.0
    return {
        "mouth_mm": (local * 1000.0).round(1),
        "mouth_z_mm": float(mouth[2] - origin_z) * 1000.0,
        "rise_bug_mm": rise_b * 1000.0,
        "rise_anvil_mm": rise_a * 1000.0,
        "ratio": ratio,
        "carried": rise_a > 0.005 and rise_b >= FOLLOWED * rise_a,
    }


def probe(env, arm: Arm) -> None:
    claw = Claw(env)
    env.claw = claw
    degrees = np.rad2deg(GRASP_POSE)
    print(f"probe  cone={PROP_CONE}  bug diameter {BUG_RADIUS * 2000:.0f} mm  "
          f"lift {LIFT_SIGN:+.0f} * {LIFT_RAD:.2f} rad")
    print(f"grasp pose  {GRASP_ALPHA:.2f} of the way stow -> thumb_down, "
          f"shoulder {GRASP_SHOULDER:+.2f} rad"
          f"  ({degrees[0]:.0f}, {degrees[1]:.0f}, {degrees[2]:.0f}, {degrees[3]:.0f}) deg\n")
    rows = [measure_preset(env, arm, claw, name) for name in PRESETS]
    for row in rows:
        flag = "carried" if row["carried"] else "left behind"
        mouth = row["mouth_mm"]
        print(f"{row['name']:<14} mouth ({mouth[0]:+.1f}, {mouth[1]:+.1f}, "
              f"{row['mouth_z_mm']:+.1f}) mm   gap {row['gap_mm']:.0f} mm   "
              f"bug {row['rise_bug_mm']:+.1f} / anvil {row['rise_anvil_mm']:+.1f} mm   "
              f"ratio {row['ratio']:.2f}  lift {row['sign']:+.0f}   {flag}")
    print()
    carried = 0
    for trial in range(3):
        row = measure_grasp(env, arm, claw)
        carried += int(row["carried"])
        flag = "carried" if row["carried"] else "left behind"
        mouth = row["mouth_mm"]
        print(f"grasp {trial}        mouth ({mouth[0]:+.1f}, {mouth[1]:+.1f}, "
              f"{row['mouth_z_mm']:+.1f}) mm   "
              f"bug {row['rise_bug_mm']:+.1f} / anvil {row['rise_anvil_mm']:+.1f} mm   "
              f"ratio {row['ratio']:.2f}   {flag}")
    print(f"\ngrasp pose carried the floor bug on {carried} of 3 trials")


def _anvil_sign(env, arm: Arm, claw: Claw, preset: np.ndarray) -> float:
    """+1 if increasing the shoulder angle raises the anvil, else -1.

    Leaves the arm back at ``preset``. The nudge is a teleport, so it has to
    happen before a bug is sitting in the mouth.
    """
    arm.write_state(preset, GRIPPER_OPEN)
    env.sim.forward()
    z0 = float(claw.mouth()[0, 2])
    nudged = preset.copy()
    nudged[1] = preset[1] + 0.05
    arm.write_state(nudged, GRIPPER_OPEN)
    env.sim.forward()
    sign = 1.0 if float(claw.mouth()[0, 2]) > z0 else -1.0
    arm.write_state(preset, GRIPPER_OPEN)
    env.sim.forward()
    return sign


def _calibrate(env, arm: Arm, claw: Claw, preset: np.ndarray):
    """Where the grasp pose seats a bug, and which way the shoulder lifts.

    Both are read after the arm has been held at ``preset`` while the robot
    stands, then the arm goes back to the stow. A teleport of the joints does
    not leave the mouth where the PD holds it.
    """
    for _ in range(25):
        _hold_step(env, arm, preset, GRIPPER_OPEN)
    robot = env.scene["robot"]
    seat = claw.seat(env, "bug_0")[0].detach().cpu().numpy()
    pos, _, quat = _base(robot)
    seat_b = _into(quat, seat - pos)
    z0 = float(claw.mouth()[0, 2])
    nudged = preset.copy()
    nudged[1] = preset[1] + 0.08
    for _ in range(12):
        _hold_step(env, arm, nudged, GRIPPER_OPEN)
    sign = 1.0 if float(claw.mouth()[0, 2]) > z0 + 0.004 else -1.0
    for _ in range(20):
        _hold_step(env, arm, STOW, GRIPPER_OPEN)
    return seat_b[:2].copy(), sign


def run_hunt(wrapped, env, policy, viewer, hunt: Hunt, arm: Arm, shown: ShownArm,
             held: dict, max_steps: int | None, speed: float) -> None:
    from mjrl.viewer.stats import Pacer

    pacer = Pacer(env.step_dt, speed)
    wrapped.reset()
    set_command(held, 0.0, 0.0, 0.0)
    hunt.seat_xy, hunt.lift_sign = _calibrate(env, arm, env.claw, hunt.preset)
    obs, _ = wrapped.reset()
    set_command(held, 0.0, 0.0, 0.0)
    twist = env.command_manager.get_term("twist")
    twist.vel_command_b[:] = 0.0
    twist.vel_command_w[:] = 0.0
    env.observation_manager._obs_buffer = None
    obs = wrapped.get_observations()
    print(f"[hunt] grasp seat offset "
          f"({hunt.seat_xy[0] * 1000:.0f}, {hunt.seat_xy[1] * 1000:.0f}) mm, "
          f"lift sign {hunt.lift_sign:+.0f}")
    print("[hunt] bugs: " + ", ".join(
        f"{name} ({_root(env.scene[name])[0]:+.2f}, {_root(env.scene[name])[1]:+.2f})"
        for name in hunt.bugs
    ))
    step = 0
    while hunt.state != "done" or viewer is not None:
        if viewer is not None and not viewer.is_running:
            print("\n[hunt] the live viewer was closed; stopping")
            break
        if max_steps is not None and step >= max_steps:
            print(f"\n[hunt] reached --steps {max_steps} in state {hunt.state}")
            break
        if viewer is None and hunt.state == "done":
            break
        pacer.wait()
        shown.apply(obs)
        with torch.inference_mode():
            action = policy(obs)
        tick(env, hunt, arm, env.sim.mj_model)
        set_command(held, *tick.command)
        held["pitch"] = tick.pitch
        arm.step_targets()
        with torch.inference_mode():
            obs, *_ = wrapped.step(action)
        step += 1


def build(args, bug_names_out: list):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend=args.backend, device=args.device, num_envs=1)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True, task_args={"objects": False})
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    names = add_hunt_scene(cfg, args.seed, 1 if args.probe else N_BUGS)
    pin_spawn(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    bug_names_out.extend(names)
    return env, res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true",
                        help="try the three arm presets on one floor bug and exit")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backend", choices=("warp", "native"), default="native")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()
    if not args.probe and args.checkpoint is None:
        parser.error("--checkpoint is required to hunt; --probe measures the arm without one")

    names: list[str] = []
    env, res = build(args, names)
    arm = Arm(env.scene["robot"], env.device)
    held = install_command(env)
    if not args.probe:
        install_pose(env, held)
    try:
        if args.probe:
            probe(env, arm)
            return 0
        from dataclasses import asdict

        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.rl.runner import MjlabOnPolicyRunner

        import tasks

        agent = tasks.load_agent_cfg("jumper.five_foot")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (tasks.load_runner_cls("jumper.five_foot") or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), device=res.device
        )
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)
        env.claw = Claw(env)
        shown = ShownArm(env, arm)
        hunt = Hunt(bugs=names, preset=GRASP_POSE.copy())

        from tasks.paths import REPO_ROOT

        cli_path = Path(REPO_ROOT) / "scripts" / "_cli.py"
        spec = importlib.util.spec_from_file_location("jumper_play_cli", cli_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load the play viewer from {cli_path}")
        play_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play_cli)
        maybe_viewer = play_cli.maybe_viewer

        viewer_args = argparse.Namespace(
            headless=args.headless, viewer_env=0, viewer_fps=60.0,
            viewer_env_num=1, viewer_ui=False,
        )
        with maybe_viewer(env, viewer_args) as viewer:
            speed = args.speed
            if speed is None:
                speed = 0.0 if viewer is None else 1.0
            run_hunt(wrapped, env, policy, viewer, hunt, arm, shown, held,
                     args.steps, speed)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
