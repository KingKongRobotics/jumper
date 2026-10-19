"""One crab's view of the room, from its own senses: the objects on the floor.

`tools/fetch/sense.py` reads the dToF range image, the IMU and the encoders and
finds the mug in them; `tools/battle/eyes.py` is the same reading for a crab
looking for a crab. This is the reading for a crab looking for things on the
floor: every cluster of returns that is not the crab itself and not the floor,
gated to the sizes of the race's objects (`room.KINDS`: 5-12 cm across, 5-11 cm
tall), so the other crab (0.3-0.5 m across) and the kerb (wall to wall) are not
objects, with each one's position in the base frame. Everything else -- the
pixel-to-ray map, the self-mask, the floor plane along the IMU's up, the
depth-connected clustering -- is copied from those two, for the reasons their
docstrings give. The frame it keeps (`last`) is what the readout draws and what
`mapper.Mapper.step` folds into the map.
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
TOF_ORIGIN = np.array([0.090678, 0.0, 0.0168455])
TOF_EYE = TOF_ORIGIN + np.array([0.004, 0.0, 0.0])
FOOT_R = 0.012
FLOOR_CLEAR = 0.03                # a zone this far above the floor plane is not floor...
FLOOR_SLOPE = 0.025               # ...plus this per metre of range: the plane is the IMU's up through the lowest
                                  # foot, and a 1.4 degree tilt of it puts the floor a metre out 2.5 cm up, which
                                  # read as a low thing at the kerb's foot, seen twice while the crab turned on the
                                  # spot and picked (ghosts at 0.1-0.3 m from the wall, probe_orbitwall2). A 7 cm
                                  # puck keeps its top above the line to 1.6 m; beyond that it is 2-3 zones anyway
MAX_RANGE = 3.0
OBJECT_HEIGHT = (0.03, 0.14)      # m above the floor, the cluster's tallest point: the room's things are 7-12 cm
                                  # (room.KINDS); the picture's upper edge is 0.137 + 0.384 r m up at range r, so a
                                  # thing is never cut by the picture's top, and a cluster that IS cut, or is taller,
                                  # is the kerb or the other crab at any range (Andrew, 2026-10-15: "if something
                                  # does obscure the full view it must be a wall"; before, a 17 cm ball at 10 cm read
                                  # as wall and the wall fix pulled the pose 2 m onto it, day17, and a kerb from close
                                  # up read as a thing until the map was asked)
OBJECT_WIDTH = (0.07, 0.26)       # m across (a crab is 0.3-0.5, the kerb is the picture's width). The floor of
                                  # 0.07 is for the other crab's legs when its body is lost in the kerb behind it:
                                  # at 1.5 m they are four clusters of 4-6 zones, 0.03-0.05 wide and 0.08 tall,
                                  # and they were remembered as a cluster of four things and picked first; a thing
                                  # at 1.9 m is 5 zones and 0.10 wide (probe_pile4)
PILE_W = 0.17                     # m: one thing's share of a pile's width (pucks 19 and 16, balls 12)
PILE_SPLIT = 0.24                 # m across or deep, a low cluster is a pile and is cut: one thing spans 0.23 at
                                  # most (a 16 cm square on its diagonal); two touching read 0.21-0.34 wide by the
                                  # angle, and under OBJECT_WIDTH's 0.26 they were one sighting, which pulled their
                                  # two memories together until they merged (probe_pile3)
CRAB_SKIRT = 0.45                 # m round a crab-sized cluster: the small clusters there are its legs
CRAB_HEIGHT = (0.14, 0.27)        # a crab's shell is 0.15-0.20 up; the kerb starts where a crab ends
CRAB_DEPTH = 0.12                 # m front to back in the picture: a crab's legs spread, a thing's face does not
KERB_MIN = 0.27                   # the kerb is 0.35 m tall: nothing else in the room reaches this
CRAB_CUT_M = 0.17                 # nearer than this the other crab's shell (0.20 up) runs out of the picture's top
                                  # too; a cut cluster farther off is the kerb, nearer the map decides (`at_wall`)
JAM_M = 0.24                      # the kerb at the nose: at least JAM_FRAC of the picture's zones (the crab's own
JAM_FRAC, JAM_ZONES = 0.6, 300     # legs left out) nearer than this, JAM_ZONES of them at the least. Jammed, the base
                                  # stops 0.18 m from the face (probe_walljam) and the face is the whole picture
                                  # (2268 zones at 0.15-0.19 m, probe_jamband2); at a 45 degree slant every ray
                                  # meets it within 0.19 m too. A thing under the nose fills a third of the picture
                                  # at most (a square puck at 0.22-0.24 m: 700 zones), the floor the rest. Rules
                                  # tried before it: a kerb CLUSTER at the nose (a side wall alongside at 0.22 m is
                                  # one cluster whose nearest point is beside the shell: a jam on every shove along
                                  # it, day26); a band of kerb zones within 12 cm of the axis (the puck under the
                                  # nose, labelled kerb with the wall behind it, 58 zones at 8-9 cm up, day27);
                                  # that band above 0.10 then 0.13 m (the puck's zones read 9-18 cm up when the
                                  # nose pitched down on it, the floor's estimate being the lowest own body along
                                  # the body's up, probe_fix8_chamfer). The kerb 0.47 m off behind a thing is NOT a
                                  # jam: 0.47 m with a thing ahead gave up every shove along a side wall, 49 in a
                                  # race (day23); the wall fix handles that one
NEAR_M = 0.45                     # nearer than this a thing fills the picture's width and touches its edges: the
                                  # width and edge gates are not asked of it (at 27 cm a 15 cm cube was 1895 zones,
                                  # edge-cut, and not a thing: the crab went blind on contact, probe_contact)
PROUD_M = 0.04                    # a low zone this much nearer than the wall in its column is a thing against the
                                  # wall: the depth-connected labelling joins an object to the kerb behind it
                                  # through its edges, where the depth runs into the wall's (Andrew, 2026-10-15:
                                  # "when objects are up against a wall the crabs can't see them")


def ray_dirs() -> np.ndarray:
    u = ((np.arange(COLS) + 0.5) / COLS * 2 - 1) * HALF_X
    v = (1 - (np.arange(ROWS) + 0.5) / ROWS * 2) * HALF_Y
    uu, vv = np.meshgrid(u, v, indexing="xy")
    d = np.stack([np.ones_like(uu), -uu, vv], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


FEET = ("LF_finger_tip_link", "RF_finger_tip_link", "LM_foot_tip_link", "RM_foot_tip_link", "LR_foot_tip_link", "RR_foot_tip_link")
STANCE_M = 0.03                   # a foot within this of the lowest foot (along the IMU's up) MAY stand on the floor
LEG_SIDE_SCALE = 0.79             # the feet read a strafe 27% long (1.47 m for 1.16, probe_legodom, twice) where a walk
                                  # reads within 5%: the lateral gait's feet move through the body faster than the body
                                  # moves, and this is the measured ratio rather than an explanation


def leg_velocity(feet_prev, feet_now, stance_prev, stance_now, dt: float, yaw_rate: float, hint=(0.0, 0.0)):
    """The body's velocity in its own frame (m/s forward, left) from the feet that
    stand on the floor: a standing foot does not move in the world, so its motion
    through the body frame is the body's motion, reversed, less the turn
    (`yaw_rate` about up carries a point r round at up x r). Legged odometry from
    the encoders and the gyro alone.

    Which feet stand: the gait lifts a foot 2.3 cm at most (probe_feet, 2026-10-15),
    so height cannot tell stance from swing (a 1.5 cm band kept every middle foot
    and put the median 32% short). A standing foot moves through the body at the
    body's full speed and a swinging one forward, so against the direction the
    sticks ask for (`hint`, forward and left) the standing feet are the ones moving
    fastest: the median of those within 75% of the fastest (the fastest alone read
    0.569 m/s against a truth of 0.566; a mean over a 60% band put a strafe 23%
    long, probe_legodom). With no hint, the median of the low feet.
    None when nothing stood through the step."""
    both = stance_prev & stance_now
    if not both.any() or dt <= 0:
        return None
    d = (feet_now[both] - feet_prev[both]) / dt
    r = feet_now[both]
    vx = -d[:, 0] + yaw_rate * r[:, 1]
    vy = -d[:, 1] - yaw_rate * r[:, 0]
    hx, hy = hint
    n = math.hypot(hx, hy)
    if n > 1e-6:
        proj = (vx * hx + vy * hy) / n
        top = float(proj.max())
        if top < 0.02:
            return 0.0, 0.0
        keep = proj >= 0.75 * top
        return float(np.median(vx[keep])), float(np.median(vy[keep])) * LEG_SIDE_SCALE
    return float(np.median(vx)), float(np.median(vy))


def is_crab(cluster: dict) -> bool:
    """A cluster the size of the other crab: between the objects and the kerb in height,
    wider than any object, and deep -- its legs spread 0.3 m front to back, where a
    thing shows only its near face (a row of things against the kerb is wide and low
    but a few centimetres deep, and read as a crab until this, day12)."""
    return (CRAB_HEIGHT[0] < cluster["height"] < CRAB_HEIGHT[1] and cluster["width"] > OBJECT_WIDTH[1]
            and cluster.get("depth", 1.0) > CRAB_DEPTH)


def _named(model, objtype, name: str, entity: str) -> int:
    i = mujoco.mj_name2id(model, objtype, f"{entity}/{name}")
    if i < 0:
        raise KeyError(f"{entity}/{name} is not in the model")
    return i


class Sight:
    def __init__(self, env, entity: str, yaw0: float = 0.0):
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
        self.m.geom_group[~own] = 5
        self.qadr = np.array([self.m.jnt_qposadr[_named(self.m, mujoco.mjtObj.mjOBJ_JOINT, n, entity)]
                              for n in self.robot.joint_names])
        self.tof_body = _named(self.m, mujoco.mjtObj.mjOBJ_BODY, "tof_sensor_link", entity)
        self.feet_ids = np.array([_named(self.m, mujoco.mjtObj.mjOBJ_BODY, n, entity) for n in FEET])
        self.feet = None                   # the six feet in the base frame, this step
        self.stance = None                 # which of them stand on the floor
        self.yaw_rate = 0.0
        self.dirs = ray_dirs()
        self._geomid = np.zeros(ROWS * COLS, dtype=np.int32)
        self._dist = np.zeros(ROWS * COLS, dtype=np.float64)
        self._group = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
        self.yaw = yaw0                    # gyro-integrated from the heading the crab woke up at
        self.frames = 0
        self.last = None
        self.at_wall = None                # (x, y in the base frame) -> bool: the map's say on a close, cut cluster
        self.expected = None               # (x, y in the base frame) -> bool: a remembered thing is about here, so a
                                           # cluster that fails this frame's gates is still that thing (not the kerb)
        self._masks = []                   # the last frames' (thing, kerb, crab) masks, for the overlay's majority

    def _pose_self(self) -> None:
        q = self.robot.data.joint_pos_biased[0].cpu().numpy()
        self.d.qpos[:] = 0.0
        for j in range(self.m.njnt):
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
    def _split_proud(labels: np.ndarray, n: int, pts: np.ndarray, h: np.ndarray):
        """A thing against the kerb, cut out of the kerb's cluster: in each column of a
        cluster that reaches kerb height or runs out of the picture's top, the wall's
        depth is the median of its zones above thing height; the low zones nearer than
        that by `PROUD_M` are not wall. They become clusters of their own (connected
        among themselves). Thing height, not kerb height: with a puck under the nose
        the wall 0.4 m behind it shows only up to 0.30 m, one row above the kerb's
        0.27, and the puck stayed in the kerb's cluster -- a jam on every shove along
        a side wall (probe_jamband, day27)."""
        x = pts[:, 0].reshape(ROWS, COLS)
        hh = h.reshape(ROWS, COLS)
        proud = np.zeros((ROWS, COLS), dtype=bool)
        for k in range(1, n + 1):
            sel = labels == k
            if sel.sum() < 15 or (hh[sel].max() < KERB_MIN and not sel[0].any()):
                continue
            for c in range(COLS):
                col = sel[:, c]
                tall = col & (hh[:, c] > OBJECT_HEIGHT[1])
                if tall.sum() < 2:
                    continue
                wall_x = float(np.median(x[tall, c]))
                low = col & (hh[:, c] <= OBJECT_HEIGHT[1]) & (x[:, c] < wall_x - PROUD_M)
                proud[low, c] = True
        if not proud.any():
            return labels, n
        # connected in depth, not merely adjacent: two cubes side by side against the kerb
        # came out as one 0.35 m wide low cluster, which read as a crab (day12)
        sub, m = Sight._label(proud, pts, step=0.05)
        out = labels.copy()
        out[proud] = n + sub[proud]
        return out, n + m

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

    def update(self, t: float) -> bool:
        """Once per control step: integrate the gyro; on a fresh exposure, read the
        image. Returns whether a frame was read."""
        g = self.gyro.data[0].cpu().numpy().astype(np.float64)
        up = self.up()
        self.yaw_rate = float(np.dot(g[:3], up))
        self.yaw += self.yaw_rate * self.dt
        # the feet every step, for the odometry: forward kinematics from the encoders
        self._pose_self()
        self.feet = self.d.xpos[self.feet_ids].copy()
        h = self.feet @ up
        self.stance = h < h.min() + STANCE_M
        if not bool(self.tof.data.fresh[0]):
            return False
        r = self.tof.data.range[0].cpu().numpy().astype(np.float64)
        floor = float((self.d.xpos[self.own_bodies] @ up).min()) - FOOT_R
        self_mask = self._self_mask()
        valid = r > 0
        pts = TOF_ORIGIN + r[:, None] * self.dirs
        h = pts @ up - floor
        obj = valid & ~self_mask & (h > FLOOR_CLEAR + FLOOR_SLOPE * r) & (r < MAX_RANGE) & (pts[:, 0] > 0.10)
        self.frames += 1
        things, big, kerb, masks, jam = self._things(obj.reshape(ROWS, COLS), pts, h, int((valid & ~self_mask).sum()))
        # the overlay's colours by a majority of the last three frames: a zone flickering
        # between thing and wall frame to frame reads as noise, and mostly is
        self._masks = (self._masks + [tuple(m.reshape(ROWS, COLS) for m in masks)])[-3:]
        vote = [sum(fr[i].astype(np.int8) for fr in self._masks) * 2 > len(self._masks) for i in range(3)]
        self.last = {"t": t, "range": r.reshape(ROWS, COLS), "self": self_mask.reshape(ROWS, COLS),
                     "obj": obj.reshape(ROWS, COLS), "floor": floor, "pts": pts, "h": h,
                     "things": things, "big": big, "kerb": kerb, "jam": jam,
                     # for the readout: which zones are things, the kerb, the other crab
                     "thing_px": vote[0], "kerb_px": vote[1], "crab_px": vote[2]}
        return True

    @staticmethod
    def _pile(sel: np.ndarray, pts: np.ndarray, width: float) -> list[np.ndarray]:
        """A wide low cluster cut into round(width / PILE_W) pieces: k-means on the zones'
        floor positions, started as even slices across the cluster."""
        k = max(2, round(width / PILE_W))
        idx = np.flatnonzero(sel)
        p = pts[idx, :2]
        order = np.argsort(p[:, 1])
        centres = np.stack([p[order[len(order) * i // k:len(order) * (i + 1) // k]].mean(axis=0) for i in range(k)])
        for _ in range(8):
            lab = np.argmin(((p[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
            for i in range(k):
                if (lab == i).any():
                    centres[i] = p[lab == i].mean(axis=0)
        out = []
        for i in range(k):
            if (lab == i).sum() >= 3:
                piece = np.zeros(len(sel), dtype=bool)
                piece[idx[lab == i]] = True
                out.append(piece)
        return out

    def _things(self, obj: np.ndarray, pts: np.ndarray, h: np.ndarray, seen: int = ROWS * COLS):
        """The clusters that are object-sized, each with its position in the base
        frame (x ahead, y left: the nearest face's median), and the ones that are not
        (the other crab, the kerb), which the map still wants as obstacles."""
        labels, n = self._label(obj, pts)
        labels, n = self._split_proud(labels, n, pts, h)
        things, big, kerb = [], [], []
        thing_px, kerb_px, crab_px = (np.zeros(ROWS * COLS, dtype=bool) for _ in range(3))
        def cluster(sel):
            p = pts[sel]
            top, width = float(h[sel].max()), float(np.ptp(p[:, 1]))
            cols_ = np.nonzero(sel.reshape(ROWS, COLS).any(axis=0))[0]
            rows_ = np.nonzero(sel.reshape(ROWS, COLS).any(axis=1))[0]
            edge = bool(cols_.min() == 0 or cols_.max() == COLS - 1)
            top_cut = bool(rows_.min() == 0)                 # its top is above the picture: taller than it looks
            near = float(p[:, 0].min())
            cx, cy = float(np.median(p[:, 0])), float(np.median(p[:, 1]))
            d = {"x": cx, "y": cy, "near": near, "height": top, "width": width, "n": int(sel.sum()), "edge": edge,
                 "depth": float(np.ptp(p[:, 0])), "bearing": float(math.atan2(cy, cx)), "range": float(math.hypot(cx, cy))}
            return d, top_cut

        for k in range(1, n + 1):
            sel = (labels == k).ravel()
            if sel.sum() < 3:
                continue
            d, top_cut = cluster(sel)
            near, cx, cy = d["near"], d["x"], d["y"]
            close = near < NEAR_M
            tall = top_cut or d["height"] > OBJECT_HEIGHT[1]
            if tall:
                # not a thing, whatever is remembered there: the kerb (from close up it shows
                # only the band inside the picture, 8-19 cm up at 15 cm, but runs out of the
                # picture's top, which no thing does) or the other crab
                big.append(d)
                if (d["height"] >= KERB_MIN and sel.sum() >= 15) or \
                        (top_cut and (near >= CRAB_CUT_M or (self.at_wall is not None and self.at_wall(cx, cy)))):
                    kerb.append(np.flatnonzero(sel))
                    kerb_px |= sel
                    d["kerb"] = True
                elif is_crab(d):
                    crab_px |= sel
                continue
            pieces = [(d, sel)]
            if not close and max(d["width"], d["depth"]) > PILE_SPLIT:
                # a pile: things touching come out as one low cluster as wide as all of them,
                # and as wide as that it is no thing (three touching at 1 m read as one of
                # width 0.34 and were remembered as ONE thing, probe_pile; a pile of four at
                # the line was left for a lone thing on the far side, day30). Cut into as
                # many pieces as things of PILE_W fit across it.
                pieces = [cluster(s)[:1] + (s,) for s in self._pile(sel, pts, max(d["width"], d["depth"]))]
            for d, sel in pieces:
                if self.expected is not None and self.expected(d["x"], d["y"]):
                    # tracked already: this frame's gates do not get to drop it (a thing flickered
                    # between thing and wall frame to frame, Andrew: "make the tracking of the
                    # objects more reliable and less noisy")
                    things.append(d)
                    d["sel"] = sel
                    continue
                if OBJECT_HEIGHT[0] <= d["height"] and \
                        (close or (OBJECT_WIDTH[0] <= d["width"] <= OBJECT_WIDTH[1] and not d["edge"])):
                    things.append(d)
                    d["sel"] = sel
                else:
                    big.append(d)
                    if is_crab(d):
                        crab_px |= sel
        # The other crab's legs come out as small clusters beside its big one (probe_sight,
        # 2026-10-15: a crab at 0.26 m is one cluster of 1371 zones, 0.20 m tall, with
        # leg pieces of 3-5 zones round it), so a thing within `CRAB_SKIRT` of a crab-sized
        # cluster is part of the crab.
        crabs = [b for b in big if is_crab(b) and not b.get("kerb")]   # the kerb close up is crab-sized
        near = int((obj.ravel() & (pts[:, 0] < JAM_M)).sum())
        jam = near >= JAM_ZONES and near >= JAM_FRAC * seen
        kept = []
        for d in things:
            sel = d.pop("sel")
            if all(math.hypot(d["x"] - c["x"], d["y"] - c["y"]) > CRAB_SKIRT for c in crabs):
                kept.append(d)
                thing_px |= sel
            else:
                crab_px |= sel                                    # a leg
        return kept, big, (np.concatenate(kerb) if kerb else np.zeros(0, dtype=int)), (thing_px, kerb_px, crab_px), jam
