"""Author the Zoidberg clip: both claws straight up, the body swaying side to side,
the claws snapping open and shut in alternation. Writes the reference clip for
`jumper.gesture_zoidberg` the way `tools/import_wbc_gestures.py` writes the shipped
gestures' (same channel order, lead-in and lead-out from HOME, the base pose solved
from the planted feet), and renders a kinematic preview of it.

    .venv/bin/python tools/fetch/make_zoidberg.py            # writes the clip + preview

The arm poses were found by forward kinematics (2026-10-13): with the shoulder
yawed forward and no roll, LF J2 1.6 / J3 -0.23 puts the left claw 166 mm above the
base, straight over the shoulder; the right arm's joint ranges are the left's
mirrored, so its pose is the left's with J0, J2 and J3 negated (found by the foot
site's height, not a left-hand mouth offset, which misled the first pass). The sway is the
support legs' knees flexed in antiphase, which the pose solver turns into a roll of
the body; the claws open and shut on alternate half-beats.
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

NAME = "zoidberg"
DT = 0.02                       # 50 Hz, the control rate
RAISE_S, DANCE_S, LOWER_S = 1.6, 6.0, 1.6   # the dance a whole number of beats: ending mid-sway stepped the roll 7 deg
SHIFT_S = 0.6                   # of the raise: the body moves back over the legs first; same at the end
BEAT_S = 1.2                    # one sway left and right (0.8 trained to arms-up with the claws held open: the
                                # snap was too fast to be worth its action-rate cost)
SWAY = np.radians(7.0)          # the body's roll amplitude in the sway
BODY_BACK = 0.045               # the body carried this far back over the legs while the arms are up, see pose_at
LIFT_BUMP = 1.5                # rad of extra elbow lift mid-raise, see pose_at
ARM_WAVE = 0.25                 # rad of shoulder yaw the raised arms wave with, in phase with the sway
UP = {"LF": {"J0": -1.57, "J1": 0.0, "J2": 1.60, "J3": -0.06},
      "RF": {"J0": 1.57, "J1": 0.0, "J2": -1.60, "J3": 0.06}}    # the right arm is the left's mirror in J0/J2/J3
CLAW = {"LF": (-1.6, 0.1), "RF": (1.6, -0.1)}     # (open, shut)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def pose_at(t: float) -> dict:
    """Joint targets by entity name at time t of the body of the clip (no leads).

    Order matters at the start and the end: the body is carried back over the four
    standing legs BEFORE the arms leave the floor, and the arms are down again
    before it comes forward -- at HOME the centre of mass sits 17 mm ahead of the
    middle feet, inside the six-foot support but outside the four-foot one, and a
    clip that lifts the arms first tips onto its claws from the first lifted frame
    (the pose solver found only a 16 deg tipped plane at t = 0.26 s and kept it).
    """
    q = dict(HOME)
    if t < SHIFT_S:                                  # the body back, arms down
        f_back, f, phase, snap = smoothstep(t / SHIFT_S), 0.0, 0.0, 0.0
    elif t < RAISE_S:                                # the arms up
        f_back, f, phase, snap = 1.0, smoothstep((t - SHIFT_S) / (RAISE_S - SHIFT_S)), 0.0, 0.0
    elif t < RAISE_S + DANCE_S:
        f_back, f = 1.0, 1.0
        phase = 2 * np.pi * (t - RAISE_S) / BEAT_S
        # the snapping fades in over the first beat and out over the last: no step
        snap = min(1.0, (t - RAISE_S) / BEAT_S, (RAISE_S + DANCE_S - t) / BEAT_S)
    elif t < RAISE_S + DANCE_S + LOWER_S - SHIFT_S:  # the arms down
        u = (t - RAISE_S - DANCE_S) / (LOWER_S - SHIFT_S)
        f_back, f, phase, snap = 1.0, 1.0 - smoothstep(u), 0.0, 0.0
    else:                                            # the body forward again
        u = (t - RAISE_S - DANCE_S - LOWER_S + SHIFT_S) / SHIFT_S
        f_back, f, phase, snap = 1.0 - smoothstep(u), 0.0, 0.0, 0.0
    sway = np.sin(phase)
    # Straight interpolation from HOME to UP dips the claw tips 6 mm below the
    # standing feet on the way (the elbow swings through), and the pose solver put
    # the floor under the claws and tipped the robot 66 deg. An extra elbow lift,
    # largest mid-way, keeps the tips clear.
    lift = LIFT_BUMP * np.sin(np.pi * f)
    for side in ("LF", "RF"):
        for j, v in UP[side].items():
            key = f"{side}_{j}_joint"
            q[key] = HOME[key] + (v - HOME[key]) * f
        q[f"{side}_J2_joint"] += lift * (1 if side == "LF" else -1)
        q[f"{side}_J0_joint"] += f * ARM_WAVE * sway * (1 if side == "LF" else 1)   # both arms wave the same way
        open_, shut = CLAW[side]
        # The claws open as the arms rise, then alternate: the left shuts on the first
        # half-beat, the right on the second.
        beat = np.sin(phase + (0.0 if side == "LF" else np.pi))
        rest = HOME[f"{side}_J4_joint"] + (open_ - HOME[f"{side}_J4_joint"]) * f
        q[f"{side}_J4_joint"] = rest + (shut - open_) * (0.5 + 0.5 * beat) * snap
    # The sway: the body rolled by SWAY * sway, the four standing feet kept where
    # they stand -- their joints solved by IK against the rolled frame, so the feet
    # stay in one plane (nudging knees put them 126 mm out of plane and the pose
    # solver tipped the robot onto its claws).
    roll = SWAY * sway * f
    # With both arms overhead the centre of mass sits 15 mm ahead of the middle
    # feet, outside the four standing feet's support: the robot would tip onto its
    # claws (the pose solver did, 66 deg). The body is carried back over the legs
    # as the arms rise, the feet staying where they are.
    back = BODY_BACK * f_back
    if abs(roll) > 1e-6 or back > 1e-6:
        q.update(_legs_for(roll, back))
    return q


_IK = {}


def _legs_for(roll: float, back: float) -> dict:
    """Joint angles of the four standing legs that keep their HOME foot positions
    when the base is rolled by `roll` about x and carried `back` metres rearward:
    each foot's HOME position, moved +back along x and rotated by -roll, in the base
    frame, solved per leg by least squares on its three joints."""
    key = (round(roll, 4), round(back, 4))
    if key in _IK:
        return _IK[key]
    from scipy.optimize import least_squares
    robot = _robot()
    m, d, mj = robot.model, robot.data, robot.mujoco
    c, s_ = np.cos(roll), np.sin(roll)
    R = np.array([[1, 0, 0], [0, c, s_], [0, -s_, c]])      # base frame rolled by +roll sees the world point at R @ p
    out = {}
    for leg in ("LM", "RM", "LR", "RR"):
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
        target = R @ (fk([HOME[n] for n in names]) + np.array([back, 0.0, 0.0]))
        lo = m.jnt_range[[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in names]]
        res = least_squares(lambda qq: fk(qq) - target, [HOME[n] for n in names], bounds=(lo[:, 0], lo[:, 1]))
        if np.linalg.norm(res.fun) > 1e-3:
            raise ValueError(f"{leg}: IK left {np.linalg.norm(res.fun) * 1000:.1f} mm at roll {np.degrees(roll):.1f} deg")
        out.update(dict(zip(names, res.x.tolist())))
    _IK[key] = out
    return out


_ROBOT = None


def _robot():
    global _ROBOT
    if _ROBOT is None:
        _ROBOT = G._Robot()
    return _ROBOT


def main() -> None:
    robot = _robot()
    keys = dances._keys()
    n_body = round((RAISE_S + DANCE_S + LOWER_S) / DT)
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
    out["source"] = np.array("tools/fetch/make_zoidberg.py (keyframed, 2026-10-13)")
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
    cam.lookat[:] = [0.05, 0.0, 0.12]; cam.distance = 0.9; cam.azimuth = 150; cam.elevation = -12
    path.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540",
                           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    step = max(1, round(1 / (25 * DT)))
    for k in range(0, len(q), step):
        d.qpos[:] = 0.0
        d.qpos[0:3] = pose[k, :3]
        rpy = pose[k, 3:6]
        quat = np.zeros(4); mj.mju_euler2Quat(quat, rpy, "xyz")
        d.qpos[3:7] = quat
        d.qpos[robot.q_adr] = q[k]
        mj.mj_kinematics(m, d)
        r.update_scene(d, cam)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close(); ff.wait()
    print(f"  preview {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
