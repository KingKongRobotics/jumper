"""One crab's view of the other, from its own senses.

The fetch's `tools/fetch/sense.py` reads the dToF range image, the IMU and the
encoders and finds the mug in it; this is the same reading for a crab that is
looking for a crab, for either crab in the duel (`entity` and its sensor, as
`arena.TOF` names them), and nothing of the mug: no claw, no handle. What it
keeps, and why, is in that module's docstring -- the pixel-to-ray map, the
self-mask (the crab's own legs are in the picture, so the posed model is cast
against with the same rays and whatever it hits is not an object), the floor
as the plane under the lowest feet along the IMU's up (here the plank's top:
the sand 0.25 m further down reads as *below* the floor and is never an
object), and depth-connected clustering (`_label`, copied: two neighbouring
pixels join only when their depth agrees, so the crab and the dune behind it
stay apart).

A crab is 0.3-0.5 m across and 0.10-0.20 m tall at rest, 0.07-0.25 as it
walks, crouches or rears; `foe()` is the biggest cluster inside those gates,
with its bearing and range in the base frame, held for `HOLD_S` after the last
sighting so a blink of the exposure does not turn a charge into a search.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
from arena import TOF
from scipy import ndimage

ROWS, COLS = 42, 54
HALF_X, HALF_Y = math.tan(math.radians(27.5)), math.tan(math.radians(21.0))
TOF_ORIGIN = np.array([0.090678, 0.0, 0.0168455])   # tof_sensor_link in base_link (jumper.xml)
TOF_EYE = TOF_ORIGIN + np.array([0.004, 0.0, 0.0])    # the rays start here (camera_config.yaml sim_eye_offset_m)
FOOT_R = 0.012
FLOOR_CLEAR = 0.03
MAX_RANGE = 2.5
CRAB_HEIGHT = (0.05, 0.30)      # m above the floor, the cluster's tallest point
CRAB_WIDTH = (0.08, 0.70)       # m across, the cluster's span in y
HOLD_S = 0.5


def ray_dirs() -> np.ndarray:
    u = ((np.arange(COLS) + 0.5) / COLS * 2 - 1) * HALF_X
    v = (1 - (np.arange(ROWS) + 0.5) / ROWS * 2) * HALF_Y
    uu, vv = np.meshgrid(u, v, indexing="xy")
    d = np.stack([np.ones_like(uu), -uu, vv], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _named(model, objtype, name: str, entity: str) -> int:
    i = mujoco.mj_name2id(model, objtype, f"{entity}/{name}")
    if i < 0:
        raise KeyError(f"{entity}/{name} is not in the model")
    return i


class Eyes:
    def __init__(self, env, entity: str):
        self.env = env
        self.entity = entity
        self.robot = env.scene[entity]
        self.tof = env.scene[TOF[entity]]
        self.gyro = env.scene[f"{entity}/imu_ang_vel"]
        self.upvec = env.scene[f"{entity}/imu_upvector"]
        self.dt = float(env.step_dt)
        self.m = copy.copy(env.sim.mj_model)
        self.d = mujoco.MjData(self.m)
        base = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link", entity)
        root = int(self.m.body_rootid[base])
        self.own_bodies = np.flatnonzero(self.m.body_rootid == root)
        own = np.isin(self.m.geom_bodyid, self.own_bodies)
        self.m.geom_group[~own] = 5            # the rays skip everything that is not this crab
        self.free_adr = int(self.m.jnt_qposadr[self.m.body_jntadr[base]])
        self.qadr = np.array([self.m.jnt_qposadr[_named(self.m, mujoco.mjtObj.mjOBJ_JOINT, n, entity)]
                              for n in self.robot.joint_names])
        self.tof_body = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "tof_sensor_link", entity)
        self.dirs = ray_dirs()
        self._geomid = np.zeros(ROWS * COLS, dtype=np.int32)
        self._dist = np.zeros(ROWS * COLS, dtype=np.float64)
        self._group = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.yaw = 0.0
        self.frames = 0
        self.last = None
        self._foe = None
        self._foe_t = -1e9

    def _pose_self(self) -> None:
        q = self.robot.data.joint_pos_biased[0].cpu().numpy()
        self.d.qpos[:] = 0.0
        for j in range(self.m.njnt):            # every free joint level, so no quaternion is zero
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.d.qpos[self.m.jnt_qposadr[j] + 3] = 1.0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def up(self) -> np.ndarray:
        u = self.upvec.data[0].cpu().numpy().astype(np.float64)
        return u / max(1e-6, np.linalg.norm(u))

    def _self_mask(self) -> np.ndarray:
        mujoco.mj_multiRay(self.m, self.d, TOF_EYE, self.dirs.ravel(), self._group, 1,
                           self.tof_body, self._geomid, self._dist, None, ROWS * COLS, mujoco.mjMAXVAL)
        hit = (self._geomid >= 0).reshape(ROWS, COLS)
        return ndimage.binary_dilation(hit, iterations=1).ravel()

    @staticmethod
    def _label(obj: np.ndarray, pts: np.ndarray, step: float = 0.10):
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
        _, comp = connected_components(g, directed=False)
        labels = np.where(obj.ravel(), comp + 1, 0)
        uniq = np.unique(labels[labels > 0])
        remap = np.zeros(labels.max() + 1, dtype=np.int64)
        remap[uniq] = np.arange(1, len(uniq) + 1)
        return remap[labels].reshape(ROWS, COLS), len(uniq)

    def update(self, t: float) -> None:
        """Once per control step: integrate the gyro; on a fresh exposure, read the image."""
        g = self.gyro.data[0].cpu().numpy().astype(np.float64)
        self.yaw += float(np.dot(g[:3], self.up())) * self.dt
        if not bool(self.tof.data.fresh[0]):
            return
        r = self.tof.data.range[0].cpu().numpy().astype(np.float64)
        self._pose_self()
        up = self.up()
        floor = float((self.d.xpos[self.own_bodies] @ up).min()) - FOOT_R
        valid = r > 0
        pts = TOF_ORIGIN + r[:, None] * self.dirs
        h = pts @ up - floor
        obj = valid & ~self._self_mask() & (h > FLOOR_CLEAR) & (r < MAX_RANGE) & (pts[:, 0] > 0.10)
        self.frames += 1
        found = self._crab(obj.reshape(ROWS, COLS), pts, h)
        self.last = {"t": t, "range": r.reshape(ROWS, COLS), "obj": obj.reshape(ROWS, COLS), "floor": floor,
                     "foe": found}
        if found is not None:
            self._foe, self._foe_t = found, t

    def _crab(self, obj: np.ndarray, pts: np.ndarray, h: np.ndarray):
        labels, n = self._label(obj, pts)
        if n == 0:
            return None
        sizes = ndimage.sum(obj, labels, index=np.arange(1, n + 1))
        for cand in np.argsort(-sizes) + 1:
            if sizes[cand - 1] < 4:
                return None
            sel = (labels == cand).ravel()
            p = pts[sel]
            top, width = float(h[sel].max()), float(np.ptp(p[:, 1]))
            # Nearer than 0.35 m the crab fills the picture and its span is the picture's
            # width, so the width gate is only asked of what is further off.
            near = float(p[:, 0].min())
            if CRAB_HEIGHT[0] <= top <= CRAB_HEIGHT[1] and (near < 0.35 or CRAB_WIDTH[0] <= width <= CRAB_WIDTH[1]):
                cx, cy = float(np.median(p[:, 0])), float(np.median(p[:, 1]))
                cols_ = np.nonzero(sel.reshape(ROWS, COLS).any(axis=0))[0]
                return {"bearing": float(math.atan2(cy, cx)), "range": float(math.hypot(cx, cy)), "near": near,
                        "height": top, "width": width, "n": int(sizes[cand - 1]),
                        "edge": bool(cols_.min() == 0 or cols_.max() == COLS - 1)}
        return None

    def foe(self, t: float):
        """The other crab as last seen within `HOLD_S`, with `age` since that sighting, or None."""
        if self._foe is None or t - self._foe_t > HOLD_S:
            return None
        return {**self._foe, "age": t - self._foe_t}
