"""Author the sleep clip: the body settles to its lowest stance, nose down, on all six
feet, and breathes -- a slow bob of a few millimetres -- then rises again. Writes the
reference clip for `jumper.gesture_sleep` the way `make_zoidberg.py` writes the
Zoidberg's (same channel order, lead-in and lead-out from HOME, the base pose solved
from the planted feet), and renders a kinematic preview of it.

    .venv/bin/python tools/fetch/make_sleep.py            # writes the clip + preview
    .venv/bin/python tools/fetch/make_sleep.py --probe    # how low the legs can go

Why a trained clip rather than the posture mode's height stick: the posture policy
keeps balancing at 0.073 m and the crab jiggles (routine2, seg2i); a gesture
policy tracks a still reference and is still. The six feet stay planted throughout
(the sit's four-foot support with the claws tucked tipped the pose solver), each
leg's joints solved by least squares against the lowered, pitched base.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import importlib.util as _ilu

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
import import_wbc_dances as dances
import import_wbc_gestures as G

from tasks.jumper.common.constants import HOME

NAME = "sleep"
DT = 0.02                       # 50 Hz, the control rate
SETTLE_S, SLEEP_S, RISE_S = 3.0, 12.0, 2.5
DROP = 0.036                    # the base this much lower than at HOME while asleep: with the breath on top, 40 mm,
                                # the most the legs reach with the feet planted (--probe: 45 mm leaves LR 1.2 mm short)
NOSE = np.radians(5.0)          # nose-down pitch while asleep
BREATH = 0.004                  # the breathing bob, metres, down on the out-breath
BREATH_S = 4.0                  # one breath
CLAW_REST = 0.35                # the claws open this much (rad off HOME) as the body settles: slack, not pinched
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def pose_at(t: float) -> dict:
    """Joint targets by entity name at time t of the body of the clip (no leads)."""
    q = dict(HOME)
    if t < SETTLE_S:
        f, breath = smoothstep(t / SETTLE_S), 0.0
    elif t < SETTLE_S + SLEEP_S:
        f = 1.0
        u = (t - SETTLE_S) / BREATH_S
        # the breathing fades in over the first breath and out over the last: no step
        env = min(1.0, (t - SETTLE_S) / BREATH_S, (SETTLE_S + SLEEP_S - t) / BREATH_S)
        breath = env * (0.5 - 0.5 * np.cos(2 * np.pi * u))
    else:
        f, breath = 1.0 - smoothstep((t - SETTLE_S - SLEEP_S) / RISE_S), 0.0
    drop = DROP * f + BREATH * breath
    pitch = NOSE * f
    q.update(_legs_for(drop, pitch))
    for side, sign in (("LF", -1.0), ("RF", +1.0)):
        q[f"{side}_J4_joint"] = HOME[f"{side}_J4_joint"] + sign * CLAW_REST * f
    return q


_IK = {}


def _legs_for(drop: float, pitch: float) -> dict:
    """Joint angles of the six legs that keep their HOME foot positions when the base
    is lowered by `drop` and pitched nose-down by `pitch`: each foot's HOME position
    in the base frame, raised by `drop` and rotated into the pitched frame, solved per
    leg by least squares on its three joints (the claw joints stay)."""
    key = (round(drop, 4), round(pitch, 4))
    if key in _IK:
        return _IK[key]
    from scipy.optimize import least_squares
    robot = _robot()
    m, d, mj = robot.model, robot.data, robot.mujoco
    c, s_ = np.cos(pitch), np.sin(pitch)
    R = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])      # base pitched nose-down by +pitch sees the world point at R @ p
    out = {}
    for leg in LEGS:
        names = [f"{leg}_J{j}_joint" for j in range(3)]
        adr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in names]
        site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, leg)
        def fk(qq):
            d.qpos[:] = 0.0; d.qpos[3] = 1.0
            for n, v in HOME.items():
                d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]] = v
            d.qpos[adr] = qq
            mj.mj_kinematics(m, d)
            return d.site_xpos[site].copy()
        target = R @ (fk([HOME[n] for n in names]) + np.array([0.0, 0.0, drop]))
        lo = m.jnt_range[[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in names]]
        res = least_squares(lambda qq: fk(qq) - target, [HOME[n] for n in names], bounds=(lo[:, 0], lo[:, 1]))
        if np.linalg.norm(res.fun) > 1e-3:
            raise ValueError(f"{leg}: IK left {np.linalg.norm(res.fun) * 1000:.1f} mm at drop {drop * 1000:.0f} mm, "
                             f"pitch {np.degrees(pitch):.1f} deg")
        out.update(dict(zip(names, res.x.tolist())))
    _IK[key] = out
    return out


_ROBOT = None


def _robot():
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = G._Robot()
    return _ROBOT


def probe() -> None:
    """How far the legs can lower the base, and the trunk's clearance there."""
    robot = _robot()
    m, d, mj = robot.model, robot.data, robot.mujoco
    base = 1                                             # the root body: the trunk
    for drop in (0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07):
        for pitch in (0.0, NOSE):
            try:
                q = dict(HOME); q.update(_legs_for(drop, pitch))
            except ValueError as e:
                print(f"drop {drop * 1000:.0f} mm pitch {np.degrees(pitch):.0f}: {e}")
                continue
            qq = np.array([[q[nm] for nm in robot.names]])
            pose, height = G.solve_resting_pose(robot, qq)
            d.qpos[:] = 0.0
            d.qpos[0:3] = pose[0, :3]
            quat = np.zeros(4); mj.mju_euler2Quat(quat, pose[0, 3:6], "xyz"); d.qpos[3:7] = quat
            d.qpos[robot.q_adr] = qq[0]
            mj.mj_kinematics(m, d)
            trunk = [g for g in range(m.ngeom) if m.geom_bodyid[g] == base]
            low = min(float(d.geom_xpos[g, 2] - m.geom_rbound[g]) for g in trunk)
            print(f"drop {drop * 1000:.0f} mm pitch {np.degrees(pitch):.0f} deg: base z {pose[0, 2] * 1000:.0f} mm, "
                  f"pitch solved {np.degrees(pose[0, 4]):+.1f} deg, trunk geoms' bounding spheres reach {low * 1000:.0f} mm, "
                  f"feet down {int((height[0] < G.DOWN_TOL).sum())}/6")


def main() -> None:
    robot = _robot()
    keys = dances._keys()
    n_body = round((SETTLE_S + SLEEP_S + RISE_S) / DT)
    body = np.array([[pose_at(i * DT)[nm] for nm in robot.names] for i in range(n_body)])
    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - body.min(axis=0), body.max(axis=0) - hi)
    for i in np.nonzero(over > 0)[0]:
        print(f"clamped {robot.names[i]} by {over[i] * 1000:.1f} mrad")
    body = np.clip(body, lo, hi)
    home = np.array([HOME[n] for n in robot.names])
    lead_in, t_in = G._lead(home, body[0], DT)
    lead_out, t_out = G._lead(body[-1], home, DT)
    hold = np.repeat(home[None], round(G.HOLD_S / DT) + 1, axis=0)
    q = np.concatenate([lead_in, body, lead_out, hold])
    pose, height = G.solve_resting_pose(robot, q)
    n = len(q)
    f32 = np.float32
    out = {"time": (np.arange(n) * DT).astype(f32), "dt": np.array(DT), "phase": np.ones(n, dtype=f32)}
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(keys):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array("tools/fetch/make_sleep.py (keyframed, 2026-10-13)")
    out["source_sha256"] = np.array("")
    out["playback_speed"] = np.array(1.0)
    out["lead_in_s"], out["lead_out_s"], out["hold_s"] = np.array(t_in), np.array(t_out), np.array(G.HOLD_S)
    dest = REPO / "tasks" / "jumper" / f"gesture_{NAME}" / "media" / f"{NAME}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    lifted = {f: float((height[:, i] >= G.DOWN_TOL).mean()) for i, f in enumerate(G.FEET)}
    print(f"wrote {dest.relative_to(REPO)}: {n} rows at {1 / DT:.0f} Hz = {n * DT:.2f} s "
          f"(body {n_body * DT:.1f} s, in {t_in:.2f}, out {t_out:.2f}, hold {G.HOLD_S})")
    print(f"  base z {pose[:, 2].min() * 1000:.0f}..{pose[:, 2].max() * 1000:.0f} mm, roll up to "
          f"{np.degrees(np.abs(pose[:, 3]).max()):.1f} deg, pitch up to {np.degrees(np.abs(pose[:, 4]).max()):.1f} deg")
    print("  off the floor: " + ", ".join(f"{f} {v * 100:.0f}%" for f, v in lifted.items()))
    speed = np.abs(np.diff(q, axis=0)).max() / DT
    print(f"  fastest joint step {speed:.1f} rad/s")
    if os.environ.get("PREVIEW", "1") == "1":
        preview(robot, q, pose, REPO / "out" / "fetch" / f"{NAME}_preview.mp4")


def preview(robot, q, pose, path: Path) -> None:
    """A kinematic render: the robot posed per frame on the solved base pose."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import subprocess

    import mujoco
    mj, m, d = robot.mujoco, robot.model, robot.data
    r = mujoco.Renderer(m, 540, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.05, 0.0, 0.08]; cam.distance = 0.8; cam.azimuth = 150; cam.elevation = -12
    path.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540",
                           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    step = max(1, round(1 / (25 * DT)))
    for k in range(0, len(q), step):
        d.qpos[:] = 0.0
        d.qpos[0:3] = pose[k, :3]
        quat = np.zeros(4); mj.mju_euler2Quat(quat, pose[k, 3:6], "xyz")
        d.qpos[3:7] = quat
        d.qpos[robot.q_adr] = q[k]
        mj.mj_kinematics(m, d)
        r.update_scene(d, cam)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close(); ff.wait()
    print(f"  preview {path.relative_to(REPO)}")


if __name__ == "__main__":
    probe() if "--probe" in sys.argv else main()
