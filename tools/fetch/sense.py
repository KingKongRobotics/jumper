"""The crab's own senses, as the autopilot is allowed to use them.

Nothing here reads a prop's pose or the robot's place in the world. What it reads
is what the robot carries: the dToF range image (`env.scene["tof"]`, 42 x 54 zones,
radial metres from the sensor, -1 where there is no return), the IMU's gyro and
up-vector, and the joint encoders (`joint_pos_biased`, with the encoder bias the
startup event put on them). From those it derives, in the **base frame**:

* the claw's mouth and the jaw axes, by forward kinematics of the robot's own
  model from the encoders alone;
* which zones of the range image are the robot itself, by casting the same 2268
  rays against that posed model (the arm and claw are in the sensor's view);
* the floor, as the plane under the lowest feet, by the same kinematics and the
  IMU's up; and
* the mug: the zones that are neither robot nor floor, clustered, the biggest
  cluster ahead; its cup from the bulk of the points and its handle bar from the
  points standing nearest the robot. When no handle protrudes from the cluster the
  bar is **assumed** at the cup's near face plus the handle's reach, and the
  estimate says so (`seen_handle`).

The sensor's geometry is the one `tasks/jumper/common/tof.py` simulates: an ideal
pinhole, 55 x 42 degrees, zone-centre rays, range measured from the link origin,
rays cast from an eye 4 mm ahead of it. `ray_dirs` reproduces its pixel -> ray map.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import mujoco
import numpy as np
from scipy import ndimage

from tasks.jumper.five_foot.mdp.grasp import (
    _closing_in_palm,
    _inward_in_palm,
    _mouth_in_palm,
    _named,
)

ROWS, COLS = 42, 54
HALF_X, HALF_Y = math.tan(math.radians(27.5)), math.tan(math.radians(21.0))
TOF_ORIGIN = np.array([0.090678, 0.0, 0.0168455])   # tof_sensor_link in base_link (jumper.xml)
TOF_EYE = TOF_ORIGIN + np.array([0.004, 0.0, 0.0])    # the rays start here (camera_config.yaml sim_eye_offset_m)
FOOT_R = 0.012            # the toe's radius under the lowest body origin, for the floor plane
FLOOR_CLEAR = 0.03        # a point this far above the floor plane is an object, not floor
MAX_RANGE = 1.5           # the search for the mug stops here
HANDLE_REACH = 0.050      # cup wall to the bar's centre, when the handle is not seen (mug.py: HANDLE_OUT + BAR)
CUP_R = 0.040
NEAR_FACE_BIAS = 0.012    # the visible wall's median x sits this far behind its nearest point (a 40 mm cylinder)
HISTORY_S = 0.12          # the mug estimate is the median of the fresh exposures this recent (6 at 50 Hz):
                          # longer lags the mug while walking (0.3 s overshot the depth by 4 cm, auto1)


def ray_dirs() -> np.ndarray:
    """[42*54, 3] unit directions in the base frame, row-major, row 0 at the top,
    column 0 at the robot's left (+y)."""
    u = ((np.arange(COLS) + 0.5) / COLS * 2 - 1) * HALF_X
    v = (1 - (np.arange(ROWS) + 0.5) / ROWS * 2) * HALF_Y
    uu, vv = np.meshgrid(u, v, indexing="xy")
    d = np.stack([np.ones_like(uu), -uu, vv], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class Senses:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.tof = env.scene["tof"]
        self.gyro = env.scene["robot/imu_ang_vel"]
        self.upvec = env.scene["robot/imu_upvector"]
        self.dt = float(env.step_dt)
        # The robot's own model, for kinematics and self-masking: the scene's compiled
        # model (it carries the task's jaws, which `assets/jumper/jumper.xml` does not --
        # cast against that file, the jaws were read as part of the mug at pinch depth,
        # auto5) with every geom that is not the robot's moved to a group the rays skip.
        self.m = copy.copy(env.sim.mj_model)
        self.d = mujoco.MjData(self.m)
        base = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        root = int(self.m.body_rootid[base])
        self.robot_bodies = np.flatnonzero(self.m.body_rootid == root)
        own = np.isin(self.m.geom_bodyid, self.robot_bodies)
        self.m.geom_group[~own] = 5
        self.free_adr = int(self.m.jnt_qposadr[self.m.body_jntadr[base]])
        self.qadr = np.array([self.m.jnt_qposadr[_named(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in self.robot.joint_names])
        self.finger = self.robot.joint_names.index("LF_J4_joint")
        self.palm = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
        self.tof_body = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "tof_sensor_link")
        self.mouth_in_palm = _mouth_in_palm(self.m)
        self.closing_in_palm = _closing_in_palm(self.m)
        self.inward_in_palm = _inward_in_palm(self.m)
        self.dirs = ray_dirs()
        self._geomid = np.zeros(ROWS * COLS, dtype=np.int32)
        self._dist = np.zeros(ROWS * COLS, dtype=np.float64)
        self._group = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)   # what the sensor sees of the robot
        self.yaw = 0.0                 # gyro-integrated heading since start, rad (drift unmodelled)
        self.history: deque = deque()  # (t, estimate) of fresh exposures
        self.frames = 0
        self.last = None               # the last processed frame's masks, for pictures and debugging
        self.max_range = MAX_RANGE     # the pilot widens this while wandering

    # ---- the robot's own state -------------------------------------------------
    def _pose_self(self) -> None:
        q = self.robot.data.joint_pos_biased[0].cpu().numpy()
        self.d.qpos[:] = 0.0
        self.d.qpos[self.free_adr + 3] = 1.0       # the robot's free joint at the origin, level
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def body(self):
        """Mouth, closing axis, inward axis (base frame) and the finger angle, from the
        encoders alone. Call after `update` or on its own."""
        self._pose_self()
        rot = self.d.xmat[self.palm].reshape(3, 3)
        mouth = self.d.xpos[self.palm] + rot @ self.mouth_in_palm
        finger = float(self.robot.data.joint_pos_biased[0, self.finger])
        return mouth, rot @ self.closing_in_palm, rot @ self.inward_in_palm, finger

    def up(self) -> np.ndarray:
        u = self.upvec.data[0].cpu().numpy().astype(np.float64)
        return u / max(1e-6, np.linalg.norm(u))

    def floor_level(self, up: np.ndarray) -> float:
        """`up . p` of the floor plane: the lowest body origin of the posed model, less a toe."""
        return float((self.d.xpos[self.robot_bodies] @ up).min()) - FOOT_R

    # ---- the range image ------------------------------------------------------------
    def self_range(self) -> np.ndarray:
        """What the sensor would read of the robot alone, per zone (-1 where it sees past it)."""
        mujoco.mj_multiRay(self.m, self.d, TOF_EYE, self.dirs.ravel(), self._group, 1,
                           self.tof_body, self._geomid, self._dist, None, ROWS * COLS, mujoco.mjMAXVAL)
        hit = self._geomid >= 0
        p = TOF_EYE + self._dist[:, None] * self.dirs
        r = np.linalg.norm(p - TOF_ORIGIN, axis=1)
        return np.where(hit, r, -1.0)

    def update(self, t: float) -> None:
        """Once per control step: integrate the gyro; on a fresh exposure, read the image."""
        # The heading is the gyro's rate about the WORLD's up, not the body's z: integrating
        # the body's z through the dance, the look-about's rolls and the orbit's leaning turns
        # read -60 deg where the body stood at -1 (w44, 2026-10-13), and the carry's heading
        # hold then turned the crab 40 deg off its line. The IMU's up vector gives the axis.
        g = self.gyro.data[0].cpu().numpy().astype(np.float64)
        self.yaw += float(np.dot(g[:3], self.up())) * self.dt
        if not bool(self.tof.data.fresh[0]):
            return
        r = self.tof.data.range[0].cpu().numpy().astype(np.float64)
        self._pose_self()
        up = self.up()
        floor = self.floor_level(up)
        self_r = self.self_range()
        self_mask = self_r >= 0
        self_mask = ndimage.binary_dilation(self_mask.reshape(ROWS, COLS), iterations=1).ravel()
        valid = r > 0
        pts = TOF_ORIGIN + r[:, None] * self.dirs
        h = pts @ up - floor
        obj = valid & ~self_mask & (h > FLOOR_CLEAR) & (r < self.max_range) & (pts[:, 0] > 0.10)
        est = self._mug(obj.reshape(ROWS, COLS), pts, h)
        self.frames += 1
        self.last = {"t": t, "range": r.reshape(ROWS, COLS), "self": self_mask.reshape(ROWS, COLS),
                     "obj": obj.reshape(ROWS, COLS), "floor": floor, "est": est, "pts": pts, "h": h}
        if est is not None:
            self.history.append((t, est))
        while self.history and self.history[0][0] < t - HISTORY_S:
            self.history.popleft()

    @staticmethod
    def _label(obj: np.ndarray, pts: np.ndarray, step: float = 0.10):
        """Connected components of the object mask that are also connected in depth:
        two neighbouring pixels join only when their x differs by less than `step`.
        Plain 8-connectivity merged the mug with a pillar a metre behind it into one
        blob 0.42 m tall and a metre wide, and the search never saw a mug (beat2a).
        Returns (labels [ROWS, COLS], n), labels 1..n, 0 outside."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        idx = np.arange(ROWS * COLS).reshape(ROWS, COLS)
        x = pts[:, 0].reshape(ROWS, COLS)
        rows, cols = [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idx[max(0, -dr):ROWS - max(0, dr), max(0, -dc):COLS - max(0, dc)]
            b = idx[max(0, dr):ROWS - max(0, -dr), max(0, dc):COLS - max(0, -dc)]
            ok = obj.ravel()[a] & obj.ravel()[b] & (np.abs(x.ravel()[a] - x.ravel()[b]) < step)
            rows.append(a[ok]); cols.append(b[ok])
        r_, c_ = np.concatenate(rows), np.concatenate(cols)
        g = coo_matrix((np.ones(len(r_), dtype=np.int8), (r_, c_)), shape=(ROWS * COLS, ROWS * COLS))
        n_all, comp = connected_components(g, directed=False)
        labels = np.where(obj.ravel(), comp + 1, 0)
        # renumber the object components 1..n
        uniq = np.unique(labels[labels > 0])
        remap = np.zeros(labels.max() + 1, dtype=np.int64)
        remap[uniq] = np.arange(1, len(uniq) + 1)
        return remap[labels].reshape(ROWS, COLS), len(uniq)

    def _mug(self, obj: np.ndarray, pts: np.ndarray, h: np.ndarray):
        labels, n = self._label(obj, pts)
        if n == 0:
            return None
        sizes = ndimage.sum(obj, labels, index=np.arange(1, n + 1))
        # the biggest blob that could be a mug: shorter than 0.2 m and narrower than 0.3
        # (a pillar is 0.4 m tall, the tray 0.3 m wide, and either outgrows the mug in pixels)
        k = 0
        for cand in np.argsort(-sizes) + 1:
            if sizes[cand - 1] < 4:
                break
            sel_c = (labels == cand).ravel()
            if float(h[sel_c].max()) < 0.20 and float(np.ptp(pts[sel_c][:, 1])) < 0.30:
                k = int(cand)
                break
        if k == 0:
            return None
        sel = (labels == k).ravel()
        p = pts[sel]
        x_bulk = float(np.median(p[:, 0]))
        wall = p[p[:, 0] >= x_bulk - 0.02]
        cup_y = float(np.median(wall[:, 1]))      # the median: seen side-on, the loop sits at the wall's depth
        handle = p[p[:, 0] < x_bulk - 0.03]
        seen = len(handle) >= 2
        # The handle's axis, from every pixel that is not the cup's body: nearer than the
        # wall (the loop face-on, from the handle side) or beside it (the loop side-on,
        # from the mug's flank -- orbA's pictures: a ring left of the cup, no pixel nearer).
        # Which way the handle points, for walking round to it; biased toward the line of
        # sight when seen from the flank, so the walk round re-measures as it goes.
        beside = np.abs(p[:, 1] - cup_y) > CUP_R + 0.012
        axis_px = p[(p[:, 0] < x_bulk - 0.03) | (beside & (p[:, 0] < x_bulk + 0.06))]
        if len(axis_px) >= 2:
            hx, hy = float(np.mean(axis_px[:, 0])), float(np.mean(axis_px[:, 1]))
            axis = float(np.arctan2(cup_y - hy, (x_bulk - NEAR_FACE_BIAS + CUP_R) - hx))
        else:
            axis = None
        if seen:
            bar = np.array([float(np.mean(handle[:, 0])), float(np.mean(handle[:, 1])), float(np.mean(handle[:, 2]))])
        else:
            bar = np.array([x_bulk - NEAR_FACE_BIAS - HANDLE_REACH, cup_y, float(np.mean(wall[:, 2]))])
        cup = np.array([x_bulk - NEAR_FACE_BIAS + CUP_R, cup_y, float(np.mean(wall[:, 2]))])
        top = float(np.max(h[sel]))
        width = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])))
        rows_, cols_ = np.nonzero(labels == k)
        edge = bool(cols_.min() == 0 or cols_.max() == COLS - 1)     # cut off by the picture's side: not all of it
        return {"bar": bar, "cup": cup, "n": int(sizes[k - 1]), "seen_handle": seen, "n_handle": len(handle),
                "axis": axis, "height": top, "width": width, "rows": int(np.ptp(rows_) + 1), "edge": edge}

    def scan(self, near: float = 0.40, far: float = 1.6, min_h: float = 0.0):
        """The room as the last frame shows it, for walking through it: `cols` is each
        column's nearest object point between `near` and `far` metres ahead (x, base
        frame; inf where the column is free), and `clusters` every object blob in that
        band with its bearing, range, height above the floor, width and size. `near`
        keeps the carried mug (0.25-0.35 m ahead, under the claw) out of the picture."""
        if self.last is None:
            return None
        pts, h, obj = self.last["pts"], self.last["h"], self.last["obj"].ravel()
        band = obj & (pts[:, 0] > near) & (pts[:, 0] < far)
        # the columns count only objects taller than min_h: the film's 3 cm tray, 40 cm
        # wide, read as a wall to the room walk and turned the crab away from it (film22)
        x = np.where(band & (h > min_h), pts[:, 0], np.inf).reshape(ROWS, COLS)
        cols = x.min(axis=0)
        labels, n = self._label(band.reshape(ROWS, COLS), pts)
        clusters = []
        for k in range(1, n + 1):
            sel = (labels == k).ravel()
            if sel.sum() < 3:
                continue
            p = pts[sel]
            cx, cy = float(np.median(p[:, 0])), float(np.median(p[:, 1]))
            cols_k = np.nonzero(sel.reshape(ROWS, COLS).any(axis=0))[0]
            rows_k = np.nonzero(sel.reshape(ROWS, COLS).any(axis=1))[0]
            clusters.append({"bearing": float(np.arctan2(cy, cx)), "range": float(np.hypot(cx, cy)),
                             "edge": bool(cols_k.min() == 0 or cols_k.max() == COLS - 1),
                             # cut by the picture's top: its height is the picture's, not its own
                             "top": bool(rows_k.min() == 0),
                             "near": float(p[:, 0].min()), "x": cx, "y": cy, "height": float(h[sel].max()),
                             "width": float(np.ptp(p[:, 1])), "depth": float(np.ptp(p[:, 0])), "n": int(sel.sum())})
        return {"cols": cols, "clusters": clusters}

    def mug(self):
        """The recent estimates' median (bar, cup), or None when the mug is not in view."""
        if not self.history:
            return None
        bars = np.array([e["bar"] for _, e in self.history])
        cups = np.array([e["cup"] for _, e in self.history])
        seen = sum(1 for _, e in self.history if e["seen_handle"])
        axes = [e["axis"] for _, e in self.history if e["axis"] is not None]
        last = self.history[-1][1]
        return {"bar": np.median(bars, axis=0), "cup": np.median(cups, axis=0), "n": len(self.history),
                "px": int(last["n"]),                      # the last frame's cluster size ("n" here is the history's length)
                "seen_handle": seen > len(self.history) / 2, "age": self.history[-1][0],
                "axis": float(np.median(axes)) if len(axes) > len(self.history) / 2 else None,
                "height": last["height"], "width": last["width"], "rows": last["rows"], "edge": last["edge"]}
