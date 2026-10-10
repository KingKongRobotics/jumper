"""The carried arm as a pen: inverse kinematics for the brush tip, and where it can write.

Works on the **environment's own compiled model** (`env.sim.mj_model`, names
prefixed `robot/`), with a private `MjData`: the kinematics are then the ones the
simulation steps, jaws, payload body and brush included, and nothing can drift
between a planning model and the simulated one.

## The solver

Damped least squares on the four arm joints (`LF_J0`..`LF_J3`) for the tip's
position, with the trunk where it actually is: the full base pose and the other
legs are copied from the simulation before each solve, so a trunk that has drifted
or tilted moves the targets with it instead of moving the ink. Three position
goals on four joints leave one free; a small pull towards the seed spends it on
staying close to where the arm already is, which keeps consecutive solutions on
the same branch.

## Where it can write

`reach_map` walks a grid of floor cells in the trunk's frame, standing at
`STAND_Z`, and keeps a cell when the tip can be put on the floor there, `HOVER`
above it **and** `DEPTH_MAX` below it -- the hair sinks into the floor to make a
wide stroke, see `brush.py` -- with

  * the palm tip (`LF` site) at least `PALM_X_MIN` ahead of the trunk's centre --
    the stable sector measured with the shipped policy (step 0, 2026-10-08,
    `native:cpu`, 16 poses x 8 s, no pushes, tip 15 mm above the floor): palm tip
    >= 0.15 m forward drifted 1.4-2.8 cm and turned 0.6-3.8 deg; nearer the trunk
    up to 36 cm and 67 deg;
  * the base of the hair, and every colliding part of the arm, at least
    `FLOOR_CLEARANCE` above the floor, and none of the arm in
    contact with the rest of the robot.

The kept cells form a band around the shoulder, not a square: 359 cm^2 with the
largest square in it only 6.5 cm (measured on a joint sweep, same model). So the
planner (`stations.py`) fits each stretch of stroke into the band rather than into
a box.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from ..five_foot.claw import LF_GRASP
from . import brush

PREFIX = "robot/"
#: Bumped whenever `reach_map` changes what it keeps, so the cache rebuilds.
REACH_VERSION = 5
ARM_JOINTS = ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint")
FINGER_JOINT = "LF_J4_joint"
#: The arm folded against the trunk, where five_foot walks with it.
STOW = np.array([LF_GRASP[j] for j in ARM_JOINTS])

#: Palm tip at least this far ahead of the trunk's centre while writing (see above).
PALM_X_MIN = 0.13
#: The brush is lifted this high between strokes, metres.
HOVER = 0.015
#: The apex goes this far below the floor at the deepest press, metres. The band
#: is where the tip reaches all three heights: HOVER, the floor and -DEPTH_MAX.
DEPTH_MAX = 0.012
#: Nothing of the arm but the brush tip nearer the floor than this, metres.
FLOOR_CLEARANCE = 0.003
#: `fold_path` searches via poses when no joint ordering keeps this much clearance.
FOLD_VIA_BELOW = 0.03
VIA_SAMPLES = 300
#: How near the claw and the brush may come to the rest of the robot while the arm
#: folds or unfolds. The model gives the carried arm no contacts with the trunk, the
#: legs or itself, so nothing stops it passing through them: the first unfold of
#: every run put the palm through LM's thigh and the arm's own shoulder, and in the
#: render the claw swept across the top of the trunk (hold20, 2026-10-09).
SELF_CLEARANCE = 0.005
#: A fold path must keep the brush and the palm this far off the floor to be taken.
FOLD_FLOOR_MIN = 0.005


@dataclass
class ReachMap:
    """Cells of `cell` metres in the trunk's frame; `ok[i, j]` covers
    x = x0 + i * cell, y = y0 + j * cell. `seed` holds a joint solution per cell."""

    x0: float
    y0: float
    cell: float
    ok: np.ndarray
    seed: np.ndarray
    #: The cache key it was stored under; the stability map is stored beside it.
    key: str = ""

    def index(self, xy: np.ndarray) -> np.ndarray:
        return np.round((np.asarray(xy) - (self.x0, self.y0)) / self.cell).astype(int)

    def margin(self) -> np.ndarray:
        """Distance of each cell to the edge of the band, metres (0 outside)."""
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(self.ok) * self.cell


class Arm:
    def __init__(self, model: mujoco.MjModel, palm_x_min: float = PALM_X_MIN):
        self.m = model
        #: See PALM_X_MIN. Lowered only to build the training poses
        #: (`tools/writing_poses.py`), where the point is to go where the shipped
        #: policy does not yet hold still.
        self.palm_x_min = palm_x_min
        self.d = mujoco.MjData(model)
        jid = [model.joint(PREFIX + j).id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.lo = model.jnt_range[jid, 0]
        self.hi = model.jnt_range[jid, 1]
        self.tip = model.site(PREFIX + brush.TIP_SITE).id
        self.base = model.site(PREFIX + brush.BASE_SITE).id
        self.palm_tip = model.site(PREFIX + "LF").id
        self.finger_tip = model.body(PREFIX + "LF_finger_tip_link").id
        self.lf_bodies = {
            b for b in range(model.nbody) if model.body(b).name.startswith(PREFIX + "LF_")
        }
        self._jacp = np.zeros((3, model.nv))
        # The claw and the brush, against every other collision geom of the robot
        # but the forearm the palm is mounted on (they overlap by construction).
        name = lambda g: model.body(model.geom_bodyid[g]).name[len(PREFIX):]
        robot = [g for g in range(model.ngeom)
                 if model.geom_bodyid[g] != 0 and (model.geom(g).name or "").endswith(
                     ("meshcol", "meshcol1", "meshcol2", "meshcol3", "meshcol4", "meshcol5"))]
        claw = [g for g in robot if name(g).startswith(("LF_palm", "LF_finger"))]
        claw += [model.geom(PREFIX + n).id for n in brush.GEOMS]
        forearm = [g for g in robot if name(g) == "LF_forearm_link"]
        rest = [g for g in robot if g not in claw and g not in forearm]
        own_arm = ("LF_shoulder", "LF_upper_arm")
        # The forearm against the trunk and the other legs too: lifting the brush
        # high over a stroke near the trunk raised the elbow and put the forearm
        # 6.7 mm into it, by the camera (grip3, 2026-10-09). Not against the upper
        # arm it is jointed to.
        self._pairs = [(a, b) for a in claw for b in rest]
        self._pairs += [(a, b) for a in forearm for b in rest
                        if not name(b).startswith(own_arm)]
        # The arm's own shoulder and upper arm, apart from the trunk and the other
        # legs: their collision hulls are convex hulls of a concave claw and arm,
        # so near the elbow they read as touching where the meshes do not.
        self._own = np.array([name(b).startswith(own_arm) for _, b in self._pairs])
        self._rbound = model.geom_rbound
        self._fromto = np.zeros(6)

    def set_state(self, qpos: np.ndarray) -> None:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)

    def tip_pos(self, q: np.ndarray | None = None) -> np.ndarray:
        if q is not None:
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.tip].copy()

    def ik(self, target: np.ndarray, seed: np.ndarray, iters: int = 40, tol: float = 5e-4,
           damping: float = 0.01, pull: float = 0.1) -> tuple[np.ndarray, float]:
        """Joint angles putting the tip on `target` (world), from `seed`. Returns
        (q, residual in metres). The base and the other joints are whatever the
        last `set_state` left."""
        q = np.clip(np.asarray(seed, dtype=float).copy(), self.lo, self.hi)
        err = np.inf
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            e = target - self.d.site_xpos[self.tip]
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self._jacp, None, self.tip)
            J = self._jacp[:, self.vadr]
            # Damped least squares, plus a pull towards the seed projected into J's
            # null space. Added inside the damped solve instead, the pull is divided
            # by damping^2 along the free direction -- 125x here -- and the arm
            # swings back and forth along it without converging.
            Jp = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
            dq = Jp @ e + (np.eye(4) - Jp @ J) @ (pull * (seed - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, err

    def clearance(self, q: np.ndarray) -> float:
        """Height above the floor of the lowest of the brush's apex and base, the
        palm tip and the finger tip, with the trunk as the last `set_state` left it."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        # The apex too: folding is not writing, and an apex dipped on the way
        # leaves ink where no stroke is (11 steps in run 10, when only the base counted).
        return float(min(self.d.site_xpos[self.tip][2], self.d.site_xpos[self.base][2],
                         self.d.site_xpos[self.palm_tip][2],
                         self.d.xpos[self.finger_tip][2]))

    def self_contact(self, q: np.ndarray, clearance: float = SELF_CLEARANCE
                     ) -> tuple[bool, bool]:
        """Whether the claw or the brush is within `clearance` of (the trunk or
        another leg, the arm's own shoulder or upper arm), with the trunk as the
        last `set_state` left it."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        x, r = self.d.geom_xpos, self._rbound
        trunk = own = False
        for i, (a, b) in enumerate(self._pairs):
            if trunk and own:
                break
            if (own if self._own[i] else trunk):
                continue
            if np.linalg.norm(x[a] - x[b]) - r[a] - r[b] > clearance:
                continue
            # Always asked out to 5 cm: with a short `distmax` two meshes that
            # overlap come back as `distmax`, not as touching (palm and upper arm,
            # 0.0 at 0.05 and 6.0 mm at 0.006).
            if mujoco.mj_geomDistance(self.m, self.d, a, b, 0.05, self._fromto) \
                    < clearance:
                if self._own[i]:
                    own = True
                else:
                    trunk = True
        return trunk, own

    def ik_clear(self, target: np.ndarray, q0: np.ndarray, max_step: float
                 ) -> np.ndarray | None:
        """A solution for `target` near `q0` that keeps the claw, the brush and the
        forearm out of the trunk and the other legs, or None.

        Four joints put a point in space, so one direction of the arm is free: the
        elbow turning about the line from the shoulder to the tip. Solving from
        seeds moved along it finds the same tip with the elbow elsewhere; the
        nearest such solution within `max_step` (rad, any joint) is taken. With the
        trunk as the last `set_state` left it."""
        self.d.qpos[self.qadr] = q0
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jacp, None, self.tip)
        free = np.linalg.svd(self._jacp[:, self.vadr])[2][-1]
        for k in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8):
            best = None
            for sign in (1.0, -1.0):
                q, err = self.ik(target, q0 + sign * k * free)
                step = float(np.abs(q - q0).max())
                if err > 1e-3 or step > max_step or self.self_contact(q, 0.0)[0]:
                    continue
                if best is None or step < best[1]:
                    best = (q, step)
            if best is not None:
                return best[0]
        return None

    def _path_cost(self, q_from, legs, samples):
        """(steps near the trunk or another leg, steps near the arm's own upper arm,
        -lowest floor clearance) along `legs`: smaller is better. The floor first,
        cheaply; a path below FOLD_FLOOR_MIN is not checked against the robot."""
        low, a = np.inf, q_from
        for b in legs:
            for t in np.linspace(0.0, 1.0, samples):
                low = min(low, self.clearance(a + t * (b - a)))
            a = b
        if low < FOLD_FLOOR_MIN:
            return (np.inf, np.inf, -low)
        trunk = own = 0
        a = q_from
        for b in legs:
            for t in np.linspace(0.0, 1.0, samples):
                hit = self.self_contact(a + t * (b - a))
                trunk += hit[0]
                own += hit[1]
            a = b
        return (trunk, own, -low)

    def fold_path(self, q_from: np.ndarray, q_to: np.ndarray,
                  samples: int = 30) -> tuple[list[np.ndarray], float]:
        """Waypoints from `q_from` to `q_to` that keep the arm off the floor and the
        claw and the brush clear of the rest of the robot.

        Interpolating all four joints at once swings the brush through the floor
        on the way between the stow and a writing pose: the arm turns about the
        shoulder while the palm is pointing down. So each subset of joints is
        tried moving first, the rest after. A path must keep FOLD_FLOOR_MIN off the
        floor; of those, the one whose claw comes near the trunk and the other legs
        least wins, then near the arm's own upper arm, then the highest. Picked by
        the floor alone, the first unfold of every run swept the claw across the
        trunk and through RF's claw (hold20). If no ordering stays clear of the
        trunk, via poses are tried. Returns (waypoints after q_from, the lowest
        floor clearance).
        """
        import itertools

        best = None
        for r in range(4):
            for first in itertools.combinations(range(4), r):
                mid = q_from.copy()
                mid[list(first)] = q_to[list(first)]
                legs = [q_to] if r == 0 else [mid, q_to]
                cost = self._path_cost(q_from, legs, samples)
                if best is None or cost < best[1]:
                    best = (legs, cost)
        if best[1][0] > 0 or -best[1][2] < FOLD_VIA_BELOW:
            # Sampled deterministically from the joint ranges, keeping the best.
            rng = np.random.default_rng(0)
            for via in rng.uniform(self.lo, self.hi, size=(VIA_SAMPLES, 4)):
                cost = self._path_cost(q_from, [via, q_to], samples)
                if cost < best[1]:
                    best = ([via, q_to], cost)
                if best[1][0] == 0 and best[1][1] == 0 and -best[1][2] >= FOLD_VIA_BELOW:
                    break
        return best[0], -best[1][2]

    def writable(self, q: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """The constraints above, on a model whose arm geoms carry a contact margin."""
        data.qpos[self.qadr] = q
        mujoco.mj_forward(model, data)
        if data.site_xpos[self.palm_tip][0] < self.palm_x_min:
            return False
        # Only the hair goes into the floor: its base, and so the handle, stays out.
        if data.site_xpos[self.base][2] < FLOOR_CLEARANCE:
            return False
        for c in data.contact[: data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 in self.lf_bodies) == (b2 in self.lf_bodies):
                continue
            other = b2 if b1 in self.lf_bodies else b1
            if other == 0:
                if c.dist < FLOOR_CLEARANCE:
                    return False
            elif c.dist < 0:
                return False
        # The arm against the trunk and the other legs, which the model's contacts
        # do not see: writing a stroke near the robot put the palm and the forearm
        # 7-18 mm into the trunk by the camera (跳跳, tt1, 2026-10-10). Into it, not
        # near it: the collision hulls are larger than the meshes, and keeping
        # SELF_CLEARANCE off them too took the band from 205 to 163 cm^2 and 跳跳
        # from 12.2 cm to 9.2.
        return not self.self_contact(q, 0.0)[0]


def reach_map(arm: Arm, standing_qpos: np.ndarray, cell: float = 0.005,
              sweep: int = 14) -> ReachMap:
    """The writable floor cells in the trunk's frame, with a joint seed for each.

    `standing_qpos` is the robot standing at the origin, level, facing +x.
    A coarse joint sweep finds where the tip goes at all; each cell near those
    points is then solved exactly, on the floor and at HOVER, and kept when both
    solve to under a millimetre and both satisfy `Arm.writable`.
    """
    import itertools

    m = copy.deepcopy(arm.m)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in arm.lf_bodies and m.geom_contype[g]:
            m.geom_margin[g] = max(m.geom_margin[g], 2 * FLOOR_CLEARANCE)
    d = mujoco.MjData(m)
    arm.set_state(standing_qpos)
    d.qpos[:] = standing_qpos

    grids = [np.linspace(lo, hi, sweep) for lo, hi in zip(arm.lo, arm.hi)]
    pts, qs = [], []
    for combo in itertools.product(*grids):
        p = arm.tip_pos(np.array(combo))
        if -DEPTH_MAX - 0.01 < p[2] < HOVER + 0.03:
            pts.append(p[:2])
            qs.append(combo)
    pts, qs = np.array(pts), np.array(qs)

    x0, y0 = np.floor(pts.min(0) / cell) * cell
    shape = tuple(np.ceil((pts.max(0) - (x0, y0)) / cell).astype(int) + 1)
    ok = np.zeros(shape, bool)
    seed = np.zeros(shape + (4,))
    near = np.zeros(shape, bool)
    ij = np.round((pts - (x0, y0)) / cell).astype(int)
    near[ij[:, 0], ij[:, 1]] = True
    from scipy.ndimage import binary_dilation

    near = binary_dilation(near, iterations=2)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    for i, j in zip(*np.nonzero(near)):
        xy = np.array([x0 + i * cell, y0 + j * cell])
        _, k = tree.query(xy, k=6)
        best, best_d = None, np.inf
        for kk in np.atleast_1d(k):
            q_floor, e0 = arm.ik(np.array([*xy, 0.0]), qs[kk])
            if e0 > 1e-3 or not arm.writable(q_floor, m, d):
                continue
            q_hover, e1 = arm.ik(np.array([*xy, HOVER]), q_floor)
            if e1 > 1e-3 or not arm.writable(q_hover, m, d):
                continue
            q_deep, e2 = arm.ik(np.array([*xy, -DEPTH_MAX]), q_floor)
            if e2 > 1e-3 or not arm.writable(q_deep, m, d):
                continue
            # The solution nearest the stow, of those that work. LF_J1 turns
            # through 6.15 rad, more than a revolution, so one cell has solutions
            # on branches a turn apart; a seed on the far one is a valid pose the
            # arm cannot unfold to without sweeping through the floor (-22 mm on
            # the best joint ordering, measured on 无's last stretch).
            dq = float(np.abs(q_floor - STOW).sum())
            if dq < best_d:
                best, best_d = q_floor, dq
        if best is not None:
            ok[i, j] = True
            seed[i, j] = best
    return ReachMap(float(x0), float(y0), cell, ok, seed)


def cached_reach_map(arm: Arm, standing_qpos: np.ndarray, cache_dir) -> ReachMap:
    """`reach_map`, kept on disk: it takes ~100 s on a 4-core container CPU.

    Keyed by everything it reads -- the model's kinematic and contact arrays, the
    standing pose and this module's constants -- so a changed brush, joint range
    or clearance rebuilds it instead of serving a stale band.
    """
    import hashlib
    from pathlib import Path

    m = arm.m
    h = hashlib.sha1()
    for a in (m.body_pos, m.body_quat, m.jnt_range, m.site_pos, m.geom_pos, m.geom_size,
              m.geom_quat, m.geom_contype, m.geom_conaffinity, np.asarray(standing_qpos),
              np.array([arm.palm_x_min, HOVER, FLOOR_CLEARANCE, DEPTH_MAX, REACH_VERSION])):
        h.update(np.ascontiguousarray(a).tobytes())
    key = h.hexdigest()[:12]
    path = Path(cache_dir) / f"reach_{key}.npz"
    if path.exists():
        z = np.load(path)
        return ReachMap(float(z["x0"]), float(z["y0"]), float(z["cell"]), z["ok"], z["seed"],
                        key)
    rm = reach_map(arm, standing_qpos)
    rm.key = key
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x0=rm.x0, y0=rm.y0, cell=rm.cell, ok=rm.ok, seed=rm.seed)
    return rm
