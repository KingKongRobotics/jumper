"""The hunt, one function per state.

``tick`` resets the arm to the stow, asks the current state for a command,
then bends that command off the walls and the can. A state changes the hunt
by calling ``Hunt.go``. It does not steer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from tasks.jumper.common.constants import HOME
from tasks.jumper.five_foot.claw import (
    ARM_JOINTS,
    FINGER_JOINT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LF_GRASP,
)
from tasks.jumper.five_foot.mdp.gripper import squeeze_limited
from tasks.jumper.five_foot.mdp.pose_command import PITCH
from tasks.jumper.five_foot.objects import BIN_INNER_HALF
from tasks.jumper.five_foot.tools.grasp_objects import FOLLOWED, NOT_FOLLOWED
from tasks.jumper.five_foot.tools.hunt_motion import (
    LOOK_RANGE,
    SPOT_RANGE,
    SKIRT_BAND,
    WALL_BODY,
    WALL_CLAW,
    WALL_FREE,
    Cmd,
    _base,
    _body_bearing,
    _choose_side,
    _into,
    _off_wall,
    _root,
    _skirt,
    _steer,
    _turn_off_wall,
    _wall_gap,
    _wrap,
    camera_look,
    in_corner,
    in_tray,
    in_view,
    looks_at_wall,
    over_tray,
)
from tasks.jumper.five_foot.tools.hunt_scene import CAN_WALL_H

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
#: A creep that has stopped getting closer, and whose best gap is already
#: inside this, closes. 14 mm is a 15 mm bug in the mouth; waiting for 12 mm
#: burned the creep and started the approach over.
STALL_GAP = 0.020
#: Control steps, at 50 Hz, with no improvement before that close. About 0.4 s.
STALL_STEPS = 20
#: Right-front foot, tucked up. ``J2`` is the roll that plants that foot:
#: the leg sticks out in -Y, and a more negative roll lifts the toe. ``J1``
#: pitches it in so the raised claw is not left against the wall. Held only
#: while a corner grasp is still short of the seat.
RF_TUCK = {
    "RF_J1_joint": HOME["RF_J1_joint"] + 0.6,
    "RF_J2_joint": HOME["RF_J2_joint"] - 0.9,
}

#: Setpoint step toward the arm goal, per control step. ``deploy/lib.rs``
#: ``ARM_INTERP``.
ARM_INTERP = 0.30

#: Trained command edges (``env_cfg.py``). A command past these is a command
#: the policy was never shown.
LIN_LIM = 0.5
ANG_LIM = 2.0

SEARCH_VX = 0.18
#: How close a remembered bug has to be before a search that still cannot
#: see it gives up on that spot.
RECALL_REACH = 0.25
#: Extra shoulder travel, past the carry lift, once the bug is at the can.
#: The carry lift leaves the mouth about level with the rim, and the jaws
#: meet the wood instead of clearing it.
DROP_RAISE = 0.45
#: Nose down, in radians. Positive pitch is nose down, and 15 degrees is the
#: edge of the band the policy tracks while it is walking. The onboard camera
#: looks forward, so a level search walks over a bug on the floor.
LOOK_DOWN = math.radians(15.0)
#: After a full turn finds nothing, how far to reverse before looking at the
#: ground he was standing on. The claw stays up through that turn.
RETRACE_BACK = 0.40

#: Control steps, at 50 Hz.
APPROACH_LIMIT = 750
CREEP_LIMIT = 400
CLOSE_STEPS = 50
VERIFY_STEPS = 40
DROP_LIMIT = 100
EXTEND_LIMIT = 150

STOW = np.array([LF_GRASP[name] for name in ARM_JOINTS], dtype=np.float64)
GRASP_POSE = (1.0 - GRASP_ALPHA) * STOW + GRASP_ALPHA * PRESETS["thumb_down"]
GRASP_POSE = GRASP_POSE.copy()
GRASP_POSE[1] += GRASP_SHOULDER

STOP: Cmd = (0.0, 0.0, 0.0)
CHASING = ("approach", "extend", "creep", "scan", "retrace")


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


@dataclass
class Hunt:
    bugs: list[str]
    state: str = "search"
    bug: str | None = None
    seen: np.ndarray | None = None
    lost_xy: np.ndarray | None = None
    lost_yaw: float = 0.0
    scan_yaw: float = 0.0
    scan_done: float = 0.0
    scan_dir: float = 1.0
    looked: bool = False
    age: int = 0
    search_s: float = 0.0
    delivered: set[str] = field(default_factory=set)
    #: Bug name to the last xy the camera saw, for a bug he is not already chasing.
    noted: dict[str, np.ndarray] = field(default_factory=dict)
    #: +1 or -1, which way he is walking along the current wall. Kept until
    #: he is clear, so the skirt does not reverse him back into the face.
    wall_side: float | None = None
    hold_yaw: float = 0.0
    seat_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lift_sign: float = LIFT_SIGN
    verify_z: tuple[float, float] | None = None
    creep_best: float = 1.0
    creep_still: int = 0
    creep_err: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: Empty lifts on the bug in hand. One returns to creep. Two stows.
    empty_lifts: int = 0
    #: The right-front claw is held up. The gait action owns that leg, so
    #: ``tuck_off_claw`` rewrites its slice before the step.
    tuck_rf: bool = False
    nudge: int = 0
    rake: int = 0
    settle: int = 0
    carry_z: float = 0.0
    #: Shoulder raise past the carry, ramped once the bug is at the rim.
    #: A step straight to ``DROP_RAISE`` flicks the bug out of the jaws.
    raise_extra: float = 0.0
    preset: np.ndarray = field(default_factory=lambda: GRASP_POSE.copy())

    def go(self, state: str) -> None:
        """Enter ``state`` and zero the counters that belong to a visit.

        ``age``, ``settle`` and ``rake`` start over. A caller that stored a
        pose for the next state has to write it after this returns.
        """
        if state != self.state:
            print(f"[hunt] {self.state} -> {state}"
                  + (f" ({self.bug})" if self.bug else ""))
        self.state = state
        self.age = 0
        self.settle = 0
        self.rake = 0


def _in_can(env, hunt: Hunt, name: str) -> bool:
    """True when ``name`` is done, including a bug sitting in the can.

    Search starts beside the can, and the bug just dropped is the closest
    one. ``in_tray`` also demands a height under the rim line, which a bug
    in the can -- stacked, or still settling from the claw -- is above, so
    that test left it eligible and the hunt turned straight back to it.
    """
    if name in hunt.delivered:
        return True
    if over_tray(_root(env.scene[name]), _root(env.scene["tray"])):
        hunt.delivered.add(name)
        print(f"[hunt] {name} is already in the tray "
              f"({len(hunt.delivered)}/{len(hunt.bugs)})")
        return True
    return False


def _near(env, hunt: Hunt, robot) -> str | None:
    """The closest bug nearby that the camera is not on, so the body can turn to look.

    It does not start a chase. A bug behind the camera used to be acquired
    here and then followed backwards.
    """
    best, best_d = None, LOOK_RANGE
    for name in hunt.bugs:
        if _in_can(env, hunt, name):
            continue
        dist, _bearing = _body_bearing(robot, _root(env.scene[name]))
        if dist < best_d:
            best, best_d = name, dist
    return best


def _spot(env, hunt: Hunt, model) -> str | None:
    robot = env.scene["robot"]
    origin, look, half = camera_look(robot, model)
    best, best_d = None, SPOT_RANGE
    for name in hunt.bugs:
        if _in_can(env, hunt, name):
            continue
        point = _root(env.scene[name])
        if not in_view(origin, look, half, point):
            continue
        dist = float(np.linalg.norm(point - origin))
        if dist < best_d:
            best, best_d = name, dist
    return best


def _lose(hunt: Hunt, robot, bearing: float) -> None:
    """Start a look-around from the spot where the bug left the camera.

    The pose is what a failed scan backs up to. Turning toward the bug while
    the claw is down is what was knocking it away, so the scan is level and
    the arm stays stowed.
    """
    pos, yaw, _ = _base(robot)
    hunt.lost_xy = pos[:2].copy()
    hunt.lost_yaw = yaw
    hunt.go("scan")
    hunt.scan_yaw = yaw
    hunt.scan_done = 0.0
    hunt.scan_dir = 1.0 if bearing >= 0.0 else -1.0


def _face_or_chase(hunt: Hunt, name: str, point: np.ndarray, bearing: float) -> Cmd | None:
    """Turn onto a bug the camera can see. A chase starts only once it is ahead.

    Returns the turn, or ``None`` once the approach has been started. The
    claw is not lowered here: an angled reach was hitting the bug.
    """
    if abs(bearing) > 0.35:
        return (0.0, 0.0, float(np.clip(1.2 * bearing, -1.0, 1.0)))
    hunt.bug = name
    hunt.seen = point.copy()
    hunt.looked = False
    hunt.noted.pop(name, None)
    hunt.empty_lifts = 0
    hunt.go("approach")
    return None


def _turn_to(bearing: float) -> Cmd:
    return (0.0, 0.0, float(np.clip(1.2 * bearing, -1.0, 1.0)))


def _sight(env, hunt: Hunt, robot, model) -> tuple[Cmd, float] | None:
    """A command when the camera has a bug, or ``None`` when it does not.

    Off to the side the turn is level and the claw stays stowed. Looking
    down while yawing swings the claw through the bug.
    """
    seen = _spot(env, hunt, model)
    if seen is None:
        return None
    point = _root(env.scene[seen])
    _dist, bearing = _body_bearing(robot, point)
    faced = _face_or_chase(hunt, seen, point, bearing)
    return (STOP if faced is None else faced), 0.0


def _note_others(env, hunt: Hunt, model) -> None:
    """Store any bug the camera can see that is not the one already in hand.

    The chase does not switch. After the current bug is delivered, search
    walks back to the stored spot.
    """
    robot = env.scene["robot"]
    origin, look, half = camera_look(robot, model)
    # During a search the closest bug in frame is the one he will chase.
    # Remember the others, not that one.
    busy = hunt.bug if hunt.bug is not None else _spot(env, hunt, model)
    for name in hunt.bugs:
        if name == busy or _in_can(env, hunt, name):
            hunt.noted.pop(name, None)
            continue
        point = _root(env.scene[name])
        if not in_view(origin, look, half, point):
            continue
        if name not in hunt.noted:
            print(f"[hunt] remembered {name} at ({point[0]:+.2f}, {point[1]:+.2f})")
        hunt.noted[name] = point[:2].copy()


def _nearest_note(env, hunt: Hunt, robot) -> str | None:
    pos, _, _ = _base(robot)
    best, best_d = None, math.inf
    for name in list(hunt.noted):
        if _in_can(env, hunt, name):
            hunt.noted.pop(name, None)
            continue
        xy = hunt.noted[name]
        dist = float(np.hypot(xy[0] - pos[0], xy[1] - pos[1]))
        if dist < best_d:
            best, best_d = name, dist
    return best


def _recall(hunt: Hunt, robot, name: str) -> tuple[Cmd, float] | None:
    """Walk face-front to a spot remembered during another grab.

    ``None`` means he reached it and the camera still does not have the bug,
    so the note is dropped and the search carries on.
    """
    xy = hunt.noted[name]
    pos, yaw, _ = _base(robot)
    dist = float(np.hypot(xy[0] - pos[0], xy[1] - pos[1]))
    if dist < RECALL_REACH:
        print(f"[hunt] {name} was not at the remembered spot")
        hunt.noted.pop(name, None)
        return None
    face = math.atan2(xy[1] - pos[1], xy[0] - pos[0])
    turn = _wrap(face - yaw)
    if abs(turn) > 0.40:
        return (0.0, 0.0, float(np.clip(1.2 * turn, -1.0, 1.0))), LOOK_DOWN
    return (0.16, 0.0, float(np.clip(turn, -0.5, 0.5))), LOOK_DOWN


def _search_move(hunt: Hunt, xy: np.ndarray, yaw: float, inward: np.ndarray, gap: float) -> Cmd:
    """Spiral in the open, and bend along the wall as it gets close.

    The side along the wall is chosen once and kept, so leaving one wall
    does not turn him straight back into it.
    """
    radius = 0.35 + 0.04 * hunt.search_s
    spiral_wz = SEARCH_VX / radius
    if gap >= SKIRT_BAND:
        hunt.wall_side = None
        return (SEARCH_VX, 0.0, spiral_wz)
    hunt.wall_side = _choose_side(yaw, inward, hunt.wall_side, xy)
    skirt = _skirt(yaw, inward, hunt.wall_side)
    if gap < WALL_BODY:
        return skirt
    blend = (SKIRT_BAND - gap) / (SKIRT_BAND - WALL_BODY)
    return (
        (1.0 - blend) * SEARCH_VX + blend * skirt[0],
        0.0,
        (1.0 - blend) * spiral_wz + blend * skirt[2],
    )


def _search(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del arm
    if len(hunt.delivered) == len(hunt.bugs):
        hunt.go("done")
        return STOP, 0.0
    robot = env.scene["robot"]
    seen = _sight(env, hunt, robot, model)
    if seen is not None:
        return seen
    hinted = _near(env, hunt, robot)
    if hinted is not None and not hunt.looked:
        _dist, bearing = _body_bearing(robot, _root(env.scene[hinted]))
        _lose(hunt, robot, bearing)
        return STOP, 0.0
    if hinted is None:
        hunt.looked = False
    noted = _nearest_note(env, hunt, robot)
    if noted is not None:
        recalled = _recall(hunt, robot, noted)
        if recalled is not None:
            return recalled
    pos, yaw, _ = _base(robot)
    gap, inward = _wall_gap(pos[:2])
    origin, look, _half = camera_look(robot, model)
    if looks_at_wall(origin, look):
        # The camera is on the wall and no bug is in view. Turn along the
        # wall until the floor is in frame again.
        hunt.wall_side = _choose_side(yaw, inward, hunt.wall_side, pos[:2])
        return _turn_off_wall(yaw, inward, hunt.wall_side), LOOK_DOWN
    hunt.search_s += float(env.step_dt)
    return _search_move(hunt, pos[:2], yaw, inward, gap), LOOK_DOWN


def _scan(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    """A full turn, level, claw up.

    The camera is the only thing that reacquires. Aiming at a bug it cannot
    see is the angled hit.
    """
    del arm
    robot = env.scene["robot"]
    seen = _sight(env, hunt, robot, model)
    if seen is not None:
        return seen
    _, yaw, _ = _base(robot)
    hunt.scan_done += abs(_wrap(yaw - hunt.scan_yaw))
    hunt.scan_yaw = yaw
    if hunt.scan_done >= 2.0 * math.pi or hunt.age > 400:
        hunt.go("retrace")
        return STOP, 0.0
    return (0.0, 0.0, float(np.clip(hunt.scan_dir, -1.0, 1.0))), 0.0


def _retrace(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del arm
    robot = env.scene["robot"]
    seen = _sight(env, hunt, robot, model)
    if seen is not None:
        return seen
    if hunt.lost_xy is None or hunt.age > 400:
        hunt.bug = None
        hunt.looked = True
        hunt.go("search")
        return STOP, 0.0
    pos, yaw, _ = _base(robot)
    heading_err = _wrap(hunt.lost_yaw - yaw)
    forward = np.array([math.cos(hunt.lost_yaw), math.sin(hunt.lost_yaw)])
    along = float(np.dot(pos[:2] - hunt.lost_xy, forward))
    if abs(heading_err) > 0.35:
        return (0.0, 0.0, float(np.clip(1.2 * heading_err, -1.0, 1.0))), 0.0
    if along > -RETRACE_BACK:
        return (-0.16, 0.0, float(np.clip(heading_err, -0.4, 0.4))), 0.0
    # Far enough back. Turn onto the patch level, and only then drop the
    # nose. Pitching down mid-turn was the claw sweep.
    face = math.atan2(hunt.lost_xy[1] - pos[1], hunt.lost_xy[0] - pos[0])
    turn = _wrap(face - yaw)
    if abs(turn) > 0.25:
        return _turn_to(turn), 0.0
    hunt.settle += 1
    if hunt.settle > 50:
        hunt.bug = None
        hunt.looked = True
        hunt.go("search")
    return STOP, LOOK_DOWN


def _approach(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del arm
    robot = env.scene["robot"]
    point = _root(env.scene[hunt.bug])
    origin, look, half = camera_look(robot, model)
    _dist, bearing = _body_bearing(robot, point)
    arrived = False
    pitch = 0.0
    if not in_view(origin, look, half, point):
        _lose(hunt, robot, bearing)
        cmd = STOP
    elif abs(bearing) > 0.28:
        # The face, not the claw. The seat sits off the centre line, and
        # aiming it yaws the body until the jaw leads and knocks the bug away.
        cmd = _turn_to(bearing)
    else:
        pitch = LOOK_DOWN
        hunt.seen = point.copy()
        # The claw stays stowed for the walk. It comes down in extend, once
        # he has stopped. Lowering it on the way in stalls the gait: the
        # policy was trained with the claw carried, and an arm in motion
        # is not a walk it tracks.
        reach = max(float(hunt.seat_xy[0]), 0.18)
        if _dist > reach + 0.05:
            cmd = _off_wall(robot, (
                0.28, 0.0, float(np.clip(bearing, -0.35, 0.35)),
            ), WALL_BODY)
        elif abs(bearing) > 0.25:
            cmd = _turn_to(bearing)
        else:
            arrived = True
    if arrived:
        _, hunt.hold_yaw, _ = _base(robot)
        hunt.go("extend")
        cmd = STOP
    elif hunt.state == "approach" and hunt.age > APPROACH_LIMIT:
        pos, yaw, _ = _base(robot)
        where = "unknown" if hunt.seen is None else (
            f"({hunt.seen[0]:+.2f}, {hunt.seen[1]:+.2f})"
        )
        print(f"[hunt] lost the approach to {hunt.bug} "
              f"from ({pos[0]:+.2f}, {pos[1]:+.2f}) yaw {yaw:+.2f}, "
              f"last seen {where}")
        _lose(hunt, robot, bearing)
    return cmd, pitch


def _extend(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    robot = env.scene["robot"]
    point = _root(env.scene[hunt.bug])
    origin, look, half = camera_look(robot, model)
    if not in_view(origin, look, half, point):
        # Raise the claw before any turn. Extending while yawing is a
        # sweep through the bug.
        if arm.at_goal(0.20):
            _dist, bearing = _body_bearing(robot, point)
            _lose(hunt, robot, bearing)
        return STOP, 0.0
    arm.goal = hunt.preset.copy()
    if arm.at_goal() or hunt.age > EXTEND_LIMIT:
        hunt.go("creep")
    return STOP, 0.0


def _drag_off_wall(hunt: Hunt, arm: Arm, bearing: float) -> Cmd:
    """Close on a bug pinned to the wall and step back into the room.

    The jaw cannot pass the bug. Facing the wall, a backward step pulls it
    off the face.
    """
    arm.finger = GRIPPER_CLOSED
    hunt.rake += 1
    if hunt.rake > 45:
        print(f"[hunt] {hunt.bug} stayed on the wall")
        hunt.bug = None
        arm.finger = GRIPPER_OPEN
        hunt.go("stow")
        return STOP
    if abs(bearing) > 0.40:
        return (0.0, 0.0, float(np.clip(1.2 * bearing, -0.8, 0.8)))
    return (-0.16, 0.0, float(np.clip(bearing, -0.4, 0.4)))


def _seat_axes(flat: np.ndarray, wall_gap: float) -> tuple[float, float]:
    """Forward and sideways together, each a full step if that axis is off.

    Sharing one 0.18 across both axes left each one too small to start the
    gait, which is why the mouth stopped beside the bug. An axis already
    inside the seat tolerance contributes nothing. Backing up is open floor
    only: against the wall it follows the bug into the face.
    """
    vx = 0.0
    if float(flat[0]) > SEAT_TOL:
        vx = 0.18
    elif float(flat[0]) < -SEAT_TOL and wall_gap >= 0.20:
        vx = -0.16
    vy = 0.0
    if abs(float(flat[1])) > SEAT_TOL:
        vy = 0.18 if float(flat[1]) > 0.0 else -0.18
    return vx, vy


def _step_to_seat(robot, flat: np.ndarray, wall_gap: float, yaw_err: float, reach: float,
                  yaw_clip: float) -> Cmd:
    """One gait step toward the seat. Slower than this never starts the gait."""
    vx, vy = _seat_axes(flat, wall_gap)
    wz = 0.0 if abs(yaw_err) < 0.15 else float(np.clip(yaw_err, -yaw_clip, yaw_clip))
    return _off_wall(robot, (vx, vy, wz), max(0.10, reach - 0.02))


def _creep_closed(env, hunt: Hunt, arm: Arm, robot, bug: np.ndarray) -> Cmd:
    arm.goal = hunt.preset.copy()
    seat = env.claw.seat(env, hunt.bug)[0].detach().cpu().numpy()
    _, yaw, quat = _base(robot)
    flat = _into(quat, bug - seat)[:2]
    gap = float(np.hypot(flat[0], flat[1]))
    if hunt.age == 1:
        hunt.creep_best = gap
        hunt.creep_err = flat.copy()
        hunt.creep_still = 0
    if gap < hunt.creep_best - 0.001:
        hunt.creep_best = gap
        hunt.creep_err = flat.copy()
        hunt.creep_still = 0
    else:
        hunt.creep_still += 1
    wall_gap, _inward = _wall_gap(bug[:2])
    corner = in_corner(bug[:2])
    # The other front claw meets the wall in a corner. Hold it up until the
    # mouth is on the bug. One wall is still a drag: the jaw cannot pass it.
    if corner and gap >= SEAT_TOL:
        if not hunt.tuck_rf:
            print(f"[hunt] tucking the off claw, {hunt.bug} is in a corner")
        hunt.tuck_rf = True
    else:
        hunt.tuck_rf = False
    reach = float(np.hypot(hunt.seat_xy[0], hunt.seat_xy[1]))
    _dist, bearing = _body_bearing(robot, bug)
    stalled = hunt.creep_best < STALL_GAP and hunt.creep_still >= STALL_STEPS
    if wall_gap < WALL_FREE and not corner and gap < 0.05:
        return _drag_off_wall(hunt, arm, bearing)
    if (corner or wall_gap >= WALL_FREE) and (
        gap < SEAT_TOL or stalled or (hunt.rake > 0 and gap < 0.05)
    ):
        # After a drag, close while the bug is still in the mouth. Walking
        # forward again would put it back on the wall. A stall inside 20 mm
        # is the same close: the mouth has stopped on the bug.
        hunt.go("close")
        return STOP
    if hunt.age > CREEP_LIMIT:
        err = hunt.creep_err
        print(f"[hunt] {hunt.bug} never entered the mouth "
              f"(gap {gap * 1000:.0f} mm, closest {hunt.creep_best * 1000:.0f} mm "
              f"at ({err[0] * 1000:+.0f}, {err[1] * 1000:+.0f}) mm in the base)")
        hunt.bug = None
        hunt.go("stow")
        return STOP
    yaw_err = _wrap(bearing) if wall_gap < 0.20 else _wrap(hunt.hold_yaw - yaw)
    if gap < 0.04:
        # Both axes, but in pulses. A steady walk this close steps past the
        # bug. One step is about 0.4 s, then a short pause to settle.
        hunt.nudge += 1
        if hunt.nudge % 30 >= 20:
            return STOP
        vx, vy = _seat_axes(flat, wall_gap)
        return _off_wall(robot, (vx, vy, 0.0), max(0.10, reach - 0.02))
    hunt.nudge = 0
    return _step_to_seat(robot, flat, wall_gap, yaw_err, reach, 0.6)


def _creep(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    robot = env.scene["robot"]
    bug = _root(env.scene[hunt.bug])
    origin, look, half = camera_look(robot, model)
    if not in_view(origin, look, half, bug):
        if arm.at_goal(0.20):
            _dist, bearing = _body_bearing(robot, bug)
            _lose(hunt, robot, bearing)
        return STOP, 0.0
    return _creep_closed(env, hunt, arm, robot, bug), 0.0


def _close(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del model
    arm.goal = hunt.preset.copy()
    arm.finger = GRIPPER_CLOSED
    if hunt.age >= CLOSE_STEPS:
        bug_z = float(_root(env.scene[hunt.bug])[2])
        anvil_z = float(env.claw.mouth()[0, 2])
        hunt.verify_z = (bug_z, anvil_z)
        hunt.go("verify")
    return STOP, 0.0


def _verify(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del model
    arm.goal = hunt.preset.copy()
    # ARM_INTERP on the whole lift is a snap, and a bug in the jaws gets
    # flicked out. Raise the shoulder over half a second instead.
    frac = min(1.0, hunt.age / 25.0)
    arm.goal[1] = hunt.preset[1] + frac * hunt.lift_sign * LIFT_RAD
    arm.finger = GRIPPER_CLOSED
    bug_z = float(_root(env.scene[hunt.bug])[2])
    anvil_z = float(env.claw.mouth()[0, 2])
    z0, a0 = hunt.verify_z
    rise_a = anvil_z - a0
    rise_b = bug_z - z0
    # Once the anvil is clearly up, a bug that has not come with it is not
    # in the jaws. Finish the raise and it gets flicked out of a second try.
    held = rise_a > 0.005 and rise_b >= FOLLOWED * rise_a
    # A bug that has not moved while the anvil has is not in the jaws.
    # Cut the raise there. One that is merely slow still gets the full check.
    if hunt.age >= 15 and rise_a > 0.008 and rise_b < NOT_FOLLOWED * rise_a:
        return _empty_lift(hunt, arm, rise_b, rise_a)
    if hunt.age >= VERIFY_STEPS:
        if held:
            print(f"[hunt] verify {hunt.bug}: bug {rise_b * 1000:.1f} mm, "
                  f"anvil {rise_a * 1000:.1f} mm, carried")
            hunt.empty_lifts = 0
            hunt.go("carry")
        else:
            return _empty_lift(hunt, arm, rise_b, rise_a)
    return STOP, 0.0


def _empty_lift(hunt: Hunt, arm: Arm, rise_b: float, rise_a: float) -> tuple[Cmd, float]:
    """The anvil rose and the bug did not. Open, and try the seat once more."""
    hunt.empty_lifts += 1
    arm.goal = hunt.preset.copy()
    arm.finger = GRIPPER_OPEN
    print(f"[hunt] verify {hunt.bug}: bug {rise_b * 1000:.1f} mm, "
          f"anvil {rise_a * 1000:.1f} mm, left behind")
    if hunt.empty_lifts >= 2:
        hunt.bug = None
        hunt.go("stow")
    else:
        hunt.go("creep")
    return STOP, 0.0


def tuck_off_claw(env, action, hunt: Hunt):
    """Replace the right-front hip and knee in ``action`` with the tuck.

    Those joints are in the gait action, so a target written before
    ``step`` is overwritten. The other joints stay the policy's. No tuck
    leaves ``action`` untouched.
    """
    if not hunt.tuck_rf:
        return action
    term = env.action_manager.get_term("joint_pos")
    names = list(term.target_names)
    scale = term.scale
    offset = term.offset
    out = action.clone()
    flat = out.reshape(-1)
    for joint, target in RF_TUCK.items():
        idx = names.index(joint)
        sc = scale if not torch.is_tensor(scale) else scale.reshape(-1)[idx]
        off = offset if not torch.is_tensor(offset) else offset.reshape(-1)[idx]
        flat[idx] = (float(target) - float(off)) / float(sc)
    return out


def _hold_bug(arm: Arm, hunt: Hunt, extra: float) -> None:
    """Grasp pose, lifted, finger shut. ``extra`` is the raise over the carry."""
    arm.goal = hunt.preset.copy()
    arm.goal[1] = hunt.preset[1] + hunt.lift_sign * (LIFT_RAD + extra)
    arm.finger = GRIPPER_CLOSED


def _carry(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del model
    robot = env.scene["robot"]
    bug = _root(env.scene[hunt.bug])
    tray = _root(env.scene["tray"])
    # Raise before the jaws meet the rim. The carry height is about the
    # rim, so walking the last stretch at that height puts the claw on
    # the wood and the bug never gets over it.
    bug_r = float(np.hypot(bug[0] - tray[0], bug[1] - tray[1]))
    # The raise starts at the rim, not a body-length out. Lifting early
    # is what shook the bug loose and opened the claw short of the can.
    at_can = over_tray(bug, tray) or bug_r < BIN_INNER_HALF + 0.08
    if hunt.age == 1:
        hunt.raise_extra = 0.0
    if at_can:
        hunt.raise_extra = min(DROP_RAISE, hunt.raise_extra + 0.015)
    else:
        hunt.raise_extra = max(0.0, hunt.raise_extra - 0.03)
    _hold_bug(arm, hunt, hunt.raise_extra)
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
    # straight through the can. The mouth has to be at its pose first: while
    # the raise is still moving, the bug lags the anvil and looks dropped.
    slipped = (
        hunt.age > 1
        and arm.at_goal(0.20)
        and float(bug[2]) - mouth_z < hunt.carry_z - 0.025
    )
    if slipped:
        print(f"[hunt] {hunt.bug} slipped on the way to the tray")
        hunt.bug = None
        arm.finger = GRIPPER_OPEN
        hunt.go("stow")
        return STOP, 0.0
    # Stand and lift once the bug is at the can. Walking on while the jaws
    # are still at the rim is what puts the claw into the wall.
    if at_can and mouth_z < CAN_WALL_H + 0.02:
        return STOP, 0.0
    if over_tray(bug, tray):
        hunt.settle += 1
        if hunt.settle >= 15:
            # Retire it before the claw opens. The tray is shallower than
            # the bug, so the sphere goes in and rolls straight back out,
            # and the check at the end of the drop was then chasing it.
            if hunt.bug not in hunt.delivered:
                hunt.delivered.add(hunt.bug)
                print(f"[hunt] {hunt.bug} is in the tray "
                      f"({len(hunt.delivered)}/{len(hunt.bugs)})")
            hunt.go("drop")
        return STOP, 0.0
    hunt.settle = 0
    pos, yaw, _ = _base(robot)
    # Face the can and keep walking. A standoff a foot clear of the rim,
    # plus a 10 cm arrival band, parked the trunk a mouth-length short of
    # the opening and the bug never crossed it. The walk ends above, once
    # the bug is over the can, and ``_steer`` turns the body out if the
    # trunk reaches the wood.
    face = math.atan2(tray[1] - pos[1], tray[0] - pos[0])
    turn = _wrap(face - yaw)
    if abs(turn) > 0.40:
        return (0.0, 0.0, float(np.clip(1.2 * turn, -1.0, 1.0))), 0.0
    return (0.16, 0.0, float(np.clip(turn, -0.5, 0.5))), 0.0


def _drop(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del model
    _hold_bug(arm, hunt, DROP_RAISE)
    arm.finger = GRIPPER_OPEN
    bug = _root(env.scene[hunt.bug])
    tray = _root(env.scene["tray"])
    # The footprint retires the bug. The height line only says the open
    # claw has waited long enough to stow without snatching it back out.
    inside = over_tray(bug, tray)
    settled = in_tray(bug, tray) or (inside and hunt.age >= 25)
    if settled or hunt.age > DROP_LIMIT:
        if hunt.bug not in hunt.delivered and inside:
            hunt.delivered.add(hunt.bug)
            print(f"[hunt] {hunt.bug} is in the tray "
                  f"({len(hunt.delivered)}/{len(hunt.bugs)})")
        elif hunt.bug not in hunt.delivered:
            print(f"[hunt] {hunt.bug} missed the tray")
        hunt.bug = None
        hunt.seen = None
        hunt.go("stow")
    return STOP, 0.0


def _stow(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del env, model
    arm.goal = STOW.copy()
    arm.finger = GRIPPER_OPEN
    if arm.at_goal(0.10) or hunt.age > EXTEND_LIMIT:
        hunt.go("search")
    return STOP, 0.0


def _done(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    del env, hunt, model
    arm.goal = STOW.copy()
    return STOP, 0.0


STATES = {
    "search": _search,
    "scan": _scan,
    "retrace": _retrace,
    "approach": _approach,
    "extend": _extend,
    "creep": _creep,
    "close": _close,
    "verify": _verify,
    "carry": _carry,
    "drop": _drop,
    "stow": _stow,
    "done": _done,
}


def _finger_hold(env, target: float) -> None:
    cfg = env.event_manager.get_term_cfg("gripper_teleop")
    hold = getattr(cfg.func, "hold_at", None)
    if hold is not None:
        hold(target)


def tick(env, hunt: Hunt, arm: Arm, model) -> tuple[Cmd, float]:
    """One control step. The arm starts stowed and open; the state may change that.

    Returns the velocity command and the nose pitch. Steering runs after the
    state, so a carry that has just committed still gets the deliver keep-out
    and a search that has just spotted a bug still gets the foot keep-out.
    """
    robot = env.scene["robot"]
    hunt.age += 1
    arm.finger = GRIPPER_OPEN
    arm.goal = STOW.copy()
    if hunt.bug is not None and hunt.state in CHASING and _in_can(env, hunt, hunt.bug):
        # Knocked into the can, or already retired. Do not keep walking at it.
        hunt.bug = None
        hunt.seen = None
        hunt.go("search")
        cmd, pitch = STOP, 0.0
    else:
        cmd, pitch = STATES[hunt.state](env, hunt, arm, model)
    if hunt.state != "creep":
        hunt.tuck_rf = False
    if hunt.state == "done" and hunt.age == 1:
        print("[hunt] every bug is in the tray")
    # Search, approach and the carry all aim through the can. One steer at
    # the end is what keeps the trunk off it, whichever state asked to walk.
    margin = WALL_CLAW if hunt.state == "creep" else WALL_BODY
    cmd, side = _steer(
        robot, cmd, _root(env.scene["tray"])[:2], margin,
        deliver=hunt.state in ("carry", "drop"),
        wall_side=hunt.wall_side,
    )
    if side is not None:
        hunt.wall_side = side
    # Other bugs in frame are stored, not chased. The one in hand stays the target.
    _note_others(env, hunt, model)
    # The gripper event runs during the step and, once a key has been touched,
    # writes the trigger. A trigger at rest is open, which is a drop wherever
    # the robot happens to be. Pin the finger to what this step asked for.
    _finger_hold(env, arm.finger)
    return cmd, pitch
