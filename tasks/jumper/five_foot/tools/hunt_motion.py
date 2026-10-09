"""Where things are, and how a commanded velocity is bent off the walls and the can.

Nothing here decides which bug to chase. Callers pass a velocity in and get
one back that will not walk the trunk through a wall or, except while
delivering, into the can.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from tasks.jumper.five_foot.objects import BIN_INNER_HALF, BIN_WALL_T
from tasks.jumper.five_foot.tools.hunt_scene import ROOM_HALF, _delivered_z

#: Onboard camera: a bug counts as seen inside this range. The cone's
#: half-angle is half of ``cam_fovy``, read off the compiled model.
SPOT_RANGE = 1.2
#: A bug this close is one to turn and look at, not to walk past. Looking is
#: all it earns: a chase starts only once the camera has it.
LOOK_RANGE = 0.90

#: Bug-centre distance to the inner wall face below which the mouth cannot
#: sit on it -- a sphere against the face leaves the jaw in the wall. The
#: crab drags it inward before the grasp. Touching is ``BUG_RADIUS``.
WALL_FREE = 0.06
#: How far the trunk stays from a wall while the arm is still stowed, and
#: once the claw is out. The second leaves the mouth the last centimetres.
WALL_BODY = 0.28
WALL_CLAW = 0.16
#: Furthest a foot sits from the trunk. ``NOMINAL_FOOT_XY`` reaches about
#: 0.21 m and the pad is past the site. A keep-out measured from the trunk
#: lets those feet meet the rim while the body is still counted as clear,
#: and the gait then stalls against the can.
FOOT_REACH = 0.24
#: How early a search or a chase turns to walk around the can, past the
#: feet. The turn is done in place, so this is the room left to stop.
CAN_CLEAR = 0.20

Cmd = tuple[float, float, float]


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
    """The point is inside the can's circular footprint, height ignored.

    A bug still in the claw is above the rim line ``in_tray`` uses, so that
    test cannot be what decides to open. The footprint can.
    """
    return float(np.hypot(point[0] - tray[0], point[1] - tray[1])) < BIN_INNER_HALF


def in_tray(point: np.ndarray, tray: np.ndarray) -> bool:
    return over_tray(point, tray) and point[2] < _delivered_z()


def mouth_in_base(robot, claw) -> np.ndarray:
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


def _body_bearing(robot, point: np.ndarray) -> tuple[float, float]:
    """Distance and bearing of ``point`` in the base frame. Bearing 0 is ahead."""
    pos, _, quat = _base(robot)
    rel = _into(quat, point - pos)
    return float(np.hypot(rel[0], rel[1])), math.atan2(rel[1], rel[0])


def _wall_gap(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Clearance of ``xy`` inside the room, and the direction away from the nearest wall."""
    options = (
        (ROOM_HALF - float(xy[0]), np.array([-1.0, 0.0])),
        (ROOM_HALF + float(xy[0]), np.array([1.0, 0.0])),
        (ROOM_HALF - float(xy[1]), np.array([0.0, -1.0])),
        (ROOM_HALF + float(xy[1]), np.array([0.0, 1.0])),
    )
    gap, inward = min(options, key=lambda item: item[0])
    return gap, inward


def _can_half() -> float:
    return BIN_INNER_HALF + BIN_WALL_T


def _round_gap(xy: np.ndarray, center: np.ndarray, radius: float) -> tuple[float, np.ndarray]:
    """Signed clearance outside a circle, and the outward normal.

    Negative clearance means ``xy`` is inside the circle.
    """
    d = np.asarray(xy, dtype=np.float64)[:2] - np.asarray(center, dtype=np.float64)[:2]
    dist = float(np.hypot(d[0], d[1]))
    if dist < 1e-8:
        return -radius, np.array([1.0, 0.0])
    return dist - radius, d / dist


def _body_cmd(yaw: float, world: np.ndarray, wz: float) -> Cmd:
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        float(c * world[0] + s * world[1]),
        float(-s * world[0] + c * world[1]),
        wz,
    )


def _walk_out(yaw: float, outward: np.ndarray) -> Cmd:
    """Turn to face ``outward`` and walk. Used once the feet are already in the can."""
    face = math.atan2(float(outward[1]), float(outward[0]))
    yaw_err = _wrap(face - yaw)
    if abs(yaw_err) > 0.45:
        return (0.0, 0.0, float(np.clip(1.2 * yaw_err, -1.0, 1.0)))
    return (0.18, 0.0, float(np.clip(yaw_err, -0.5, 0.5)))


def _off_wall(robot, cmd: Cmd, margin: float) -> Cmd:
    """Drop the part of ``cmd`` that walks the trunk through a wall.

    A bug against the wall used to be chased by backing into the face: the
    seat is ahead of the trunk, the error comes out negative, and the body
    follows the bug backwards until the wall stops it.
    """
    vx, vy, wz = cmd
    pos, yaw, _ = _base(robot)
    gap, inward = _wall_gap(pos[:2])
    if gap >= margin:
        return cmd
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.array([c * vx - s * vy, s * vx + c * vy])
    into = float(np.dot(world, -inward))
    if into <= 0.0:
        return cmd
    world = world + inward * into
    return (
        float(c * world[0] + s * world[1]),
        float(-s * world[0] + c * world[1]),
        wz,
    )


def _steer(robot, cmd: Cmd, tray: np.ndarray, margin: float, *, deliver: bool = False) -> Cmd:
    """Bend a velocity off the can and the walls.

    The keep-out is the feet, not the trunk. A command aimed through the can
    used to stay untouched until the trunk itself was close, by which time a
    foot was already on the rim and the gait was pushing on it. Near the can
    the into-part is dropped and the rest becomes a walk along the face.
    A carry is the exception. It keeps walking until the bug is over the
    opening, and this only turns it around once the trunk itself is at the
    wood. Stopping at the foot circle left the bug short of the rim.
    """
    cmd = _off_wall(robot, cmd, margin)
    vx, vy, wz = cmd
    pos, yaw, _ = _base(robot)
    gap, outward = _round_gap(pos[:2], tray, _can_half())
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.array([c * vx - s * vy, s * vx + c * vy])
    into = float(np.dot(world, -outward))
    if deliver:
        if gap < 0.04:
            return _walk_out(yaw, outward)
        return cmd
    foot_gap = gap - FOOT_REACH
    if foot_gap < 0.0:
        return _walk_out(yaw, outward)
    if foot_gap >= CAN_CLEAR or into <= 0.0:
        return cmd
    world = world + outward * into
    tangent = np.array([-outward[1], outward[0]])
    if float(np.dot(world, tangent)) < 0.0:
        tangent = -tangent
    if float(np.linalg.norm(world)) < 0.12:
        world = tangent * 0.18
    else:
        world = tangent * min(float(np.linalg.norm(world)), 0.22)
    face = math.atan2(float(world[1]), float(world[0]))
    yaw_err = _wrap(face - yaw)
    if abs(yaw_err) > 0.55:
        return (0.0, 0.0, float(np.clip(1.2 * yaw_err, -1.0, 1.0)))
    return _body_cmd(yaw, world, float(np.clip(yaw_err, -0.8, 0.8)))
