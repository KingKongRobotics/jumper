"""What the battle's keyframed clips share: the standing legs solved by IK for a
rolled, reared, lowered or shifted body, the clip written the way
`tools/import_wbc_gestures.py` writes the shipped gestures, and a kinematic
preview.

`tools/fetch/make_zoidberg.py` is where this came from, with its three traps:
the body is carried back over the four standing legs before the arms leave the
floor (at HOME the centre of mass is 17 mm ahead of the middle feet, inside the
six-foot support and outside the four-foot one), an extra elbow lift keeps the
claw tips off the floor mid-raise, and the standing feet are solved against the
moved base so they stay in one plane (nudging knees put them 126 mm out of plane
and the pose solver tipped the robot onto its claws). Each maker keeps its own
timeline and poses; this is the arithmetic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import importlib.util as _ilu

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
import import_wbc_dances as dances
import import_wbc_gestures as G

from tasks.jumper.common.constants import HOME

DT = 0.02                       # 50 Hz, the control rate
STANDING = ("LM", "RM", "LR", "RR")
#: The Zoidberg's arm pose: straight up over the shoulder (tip 143 mm above the base).
UP = {"LF": {"J0": -1.57, "J1": 0.0, "J2": 1.60, "J3": -0.06},
      "RF": {"J0": 1.57, "J1": 0.0, "J2": -1.60, "J3": 0.06}}
#: The threat's: spread wide and high (tip 19 cm out, 12 cm up; `make_threat.py`).
SPREAD = {"LF": {"J0": -0.59, "J1": -0.14, "J2": 1.17, "J3": -0.23},
          "RF": {"J0": 0.59, "J1": -0.14, "J2": -1.17, "J3": 0.23}}
CLAW = {"LF": (-1.6, 0.1), "RF": (1.6, -0.1)}     # (open, shut)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


_ROBOT = None


def robot():
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = G._Robot()
    return _ROBOT


def arm(q: dict, side: str, pose: dict, f: float, lift: float = 0.0) -> None:
    """`side`'s four arm joints `f` of the way from HOME to `pose`, the elbow
    lifted `lift` rad extra (largest mid-raise keeps the tip off the floor)."""
    sign = 1 if side == "LF" else -1
    for j, v in pose.items():
        key = f"{side}_{j}_joint"
        q[key] = HOME[key] + (v - HOME[key]) * f
    q[f"{side}_J2_joint"] += lift * sign


def claw(q: dict, side: str, f: float, shut: float) -> None:
    """`side`'s claw open `f` of the way (as the arm rises), then `shut` of the way closed."""
    open_, closed = CLAW[side]
    rest = HOME[f"{side}_J4_joint"] + (open_ - HOME[f"{side}_J4_joint"]) * f
    q[f"{side}_J4_joint"] = rest + (closed - open_) * shut


_IK: dict = {}


def legs_for(roll: float = 0.0, pitch: float = 0.0, back: float = 0.0, drop: float = 0.0) -> dict:
    """Joint angles of the four standing legs that keep their HOME foot positions
    when the base is rolled by `roll` (left side down positive), reared nose-up
    by `pitch`, carried `back` metres rearward and lowered `drop` metres."""
    key = (round(roll, 4), round(pitch, 4), round(back, 4), round(drop, 4))
    if key in _IK:
        return _IK[key]
    from scipy.optimize import least_squares
    r = robot()
    m, d, mj = r.model, r.data, r.mujoco
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])        # nose up by `pitch` is R_y(-pitch)
    R = (Rx @ Ry).T                                              # the moved base sees a world point p at R^T p

    def fk(adr, site, qq):
        d.qpos[:] = 0.0; d.qpos[3] = 1.0
        for n, v in HOME.items():
            d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]] = v
        d.qpos[adr] = qq
        mj.mj_kinematics(m, d)
        return d.site_xpos[site].copy()

    out = {}
    for leg in STANDING:
        names = [f"{leg}_J{j}_joint" for j in range(3)]
        adr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in names]
        site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, leg)
        target = R @ (fk(adr, site, [HOME[n] for n in names]) + np.array([back, 0.0, drop]))
        lo = m.jnt_range[[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in names]]
        res = least_squares(lambda qq, a=adr, s=site, tg=target: fk(a, s, qq) - tg, [HOME[n] for n in names],
                            bounds=(lo[:, 0], lo[:, 1]))
        if np.linalg.norm(res.fun) > 1e-3:
            raise ValueError(f"{leg}: IK left {np.linalg.norm(res.fun) * 1000:.1f} mm at roll {np.degrees(roll):.1f}, "
                             f"pitch {np.degrees(pitch):.1f} deg, back {back * 1000:.0f}, drop {drop * 1000:.0f} mm")
        out.update(dict(zip(names, res.x.tolist())))
    _IK[key] = out
    return out


def write(name: str, pose_at, body_s: float, source: str) -> Path:
    """The clip `tasks/jumper/gesture_<name>/media/<name>.npz` from `pose_at(t)`
    over `body_s` seconds, with the leads, the solved base pose and the numbers a
    task config needs printed; and a preview under out/battle/."""
    r = robot()
    keys = dances._keys()
    n_body = round(body_s / DT)
    body = np.array([[pose_at(i * DT)[nm] for nm in r.names] for i in range(n_body)])
    lo, hi = r.range[:, 0], r.range[:, 1]
    over = np.maximum(lo - body.min(axis=0), body.max(axis=0) - hi)
    for i in np.nonzero(over > 0)[0]:
        print(f"clamped {r.names[i]} by {over[i] * 1000:.1f} mrad")
    body = np.clip(body, lo, hi)
    home = np.array([HOME[n] for n in r.names])
    lead_in, t_in = G._lead(home, body[0], DT)
    lead_out, t_out = G._lead(body[-1], home, DT)
    hold = np.repeat(home[None], round(G.HOLD_S / DT) + 1, axis=0)
    q = np.concatenate([lead_in, body, lead_out, hold])
    pose, height = G.solve_resting_pose(r, q)
    n = len(q)
    f32 = np.float32
    out = {"time": (np.arange(n) * DT).astype(f32), "dt": np.array(DT), "phase": np.ones(n, dtype=f32)}
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(keys):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array(source)
    out["source_sha256"] = np.array("")
    out["playback_speed"] = np.array(1.0)
    out["lead_in_s"], out["lead_out_s"], out["hold_s"] = np.array(t_in), np.array(t_out), np.array(G.HOLD_S)
    dest = REPO / "tasks" / "jumper" / f"gesture_{name}" / "media" / f"{name}.npz"
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
    rms = float(np.sqrt(np.mean((body - home) ** 2)))
    vel = float(np.sqrt(np.mean((np.diff(body, axis=0) / DT) ** 2)))
    print(f"  clip RMS amplitude {rms:.3f} rad (x0.44 = joint_pos_std {0.44 * rms:.2f}), velocity RMS {vel:.3f} rad/s "
          f"(x2/3 = joint_vel_std {2 * vel / 3:.2f})")
    if os.environ.get("PREVIEW", "1") == "1":
        preview(q, pose, REPO / "out" / "battle" / f"{name}_preview.mp4")
    return dest


def preview(q, pose, path: Path) -> None:
    """A kinematic render: the robot posed per frame on the solved base pose,
    from the front left (`PREVIEW_AZ`, degrees, to look from elsewhere)."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import subprocess

    import mujoco
    r = robot()
    mj, m, d = r.mujoco, r.model, r.data
    ren = mujoco.Renderer(m, 540, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.05, 0.0, 0.12]; cam.distance = 0.9
    cam.azimuth = float(os.environ.get("PREVIEW_AZ", "40")); cam.elevation = -15
    path.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540",
                           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    step = max(1, round(1 / (25 * DT)))
    for k in range(0, len(q), step):
        d.qpos[:] = 0.0
        d.qpos[0:3] = pose[k, :3]
        quat = np.zeros(4); mj.mju_euler2Quat(quat, pose[k, 3:6], "xyz")
        d.qpos[3:7] = quat
        d.qpos[r.q_adr] = q[k]
        mj.mj_kinematics(m, d)
        ren.update_scene(d, cam)
        ff.stdin.write(ren.render().tobytes())
    ff.stdin.close(); ff.wait()
    print(f"  preview {path.relative_to(REPO)}")
