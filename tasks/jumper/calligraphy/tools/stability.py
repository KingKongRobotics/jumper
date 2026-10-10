#!/usr/bin/env python3
"""Where in the reach band does the trunk stand still with the arm out there?

    python tasks/jumper/calligraphy/tools/stability.py            # ~10 min on 4 CPU cores
    python tasks/jumper/calligraphy/tools/stability.py --checkpoint <run>/model_<n>.pt --palm-x-min 0.10

The reach band (`arm.reach_map`) is kinematics: where the brush can go. Whether
`jumper.five_foot`'s policy -- trained with the arm folded -- holds the trunk still
while the arm is there is a separate question, and the answer is not "everywhere":
with the arm across to the robot's right, writing 无's last stroke, the trunk rose
30 mm and crept 43 mm while the arm wrote (run 13, 2026-10-08).

So this puts one robot per grid point of the band, unfolds the arm to hover there
the way `write.py` does (3 s), holds it HOLD_S, and measures what the trunk did over
the hold. A cell is stable when the trunk moved less than DRIFT_MAX, turned less
than TURN_MAX and stayed within HEIGHT_MAX of its height before unfolding. The
result is stored beside the reach band it was measured on
(`logs/calligraphy/cache/stable_<band>_<checkpoint>.npz`), every band cell taking the verdict of
its nearest grid point, and `write.py` intersects the band with it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "logs" / "calligraphy" / "cache"

UNFOLD_S = 3.0
SETTLE_S = 1.0
HOLD_S = 6.0
DRIFT_MAX = 0.015    # m
TURN_MAX = 4.0       # deg
HEIGHT_MAX = 0.010   # m


def stable_name(reach_key: str, checkpoint: Path) -> str:
    """The map belongs to a band **and** a policy: a fine-tuned policy holds still
    where the shipped one does not, so the checkpoint is part of the name."""
    import hashlib

    tag = hashlib.sha1(str(Path(checkpoint).resolve()).encode()).hexdigest()[:8]
    return f"stable_{reach_key}_{Path(checkpoint).stem}_{tag}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=float, default=0.02, help="spacing of the probes, m")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="the policy to measure; default: five_foot's shipped one")
    ap.add_argument("--palm-x-min", type=float, default=None,
                    help="the band's palm limit (arm.PALM_X_MIN by default; 0.10 is the "
                         "training band)")
    args = ap.parse_args()

    import torch

    from tasks.jumper.calligraphy import arm as armmod
    from tasks.jumper.calligraphy import brush, sim
    from tasks.jumper.five_foot.claw import ARM_JOINTS, LF_GRASP

    # The band first, on a one-robot model -- the same model and cache write.py uses.
    ckpt = args.checkpoint or sim.CHECKPOINT
    probe = sim.build(1, ckpt)
    arm = armmod.Arm(probe.env.sim.mj_model,
                     palm_x_min=args.palm_x_min or armmod.PALM_X_MIN)
    standing = sim.standing_qpos(probe.env.sim.mj_model)
    rm = armmod.cached_reach_map(arm, standing, CACHE)
    probe.env.close()

    cells = np.argwhere(rm.ok)
    xy = cells * rm.cell + (rm.x0, rm.y0)
    step = max(1, round(args.grid / rm.cell))
    pick = cells[(cells[:, 0] % step == 0) & (cells[:, 1] % step == 0)]
    n = len(pick)
    print(f"[stability] band {rm.ok.sum()} cells; probing {n} at {args.grid * 100:.1f} cm")

    # Hover joint angles for each probe, solved for the robot standing at the origin.
    arm.set_state(standing)
    q_hover = np.zeros((n, 4))
    for i, (a, b) in enumerate(pick):
        p = np.array([rm.x0 + a * rm.cell, rm.y0 + b * rm.cell, armmod.HOVER])
        q_hover[i], _ = arm.ik(p, rm.seed[a, b])

    sm = sim.build(n, ckpt)
    stow = np.array([LF_GRASP[j] for j in ARM_JOINTS])
    obs = sm.wrapped.get_observations()
    robot = sm.robot
    fell = np.zeros(n, bool)

    def run(seconds, targets):
        nonlocal obs
        for k in range(int(seconds / sm.dt)):
            q = targets(k) if callable(targets) else targets
            hold = np.column_stack([q, np.full(n, brush.FINGER_HOLD)])
            robot.set_joint_position_target(torch.tensor(hold, dtype=torch.float32),
                                            joint_ids=sm.joint_ids)
            with torch.inference_mode():
                obs, _, dones, _ = sm.wrapped.step(sm.policy(obs))
            fell[dones.numpy().astype(bool)] = True

    def pose():
        pos = robot.data.root_link_pos_w.numpy().copy()
        q = robot.data.root_link_quat_w.numpy()
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                         1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return pos, yaw

    t0 = time.time()
    run(1.0, np.tile(stow, (n, 1)))
    z_stand = pose()[0][:, 2]
    m = int(UNFOLD_S / sm.dt)

    def unfold(k):
        f = min(1.0, (k + 1) / m)
        f = f * f * (3 - 2 * f)
        return stow + f * (q_hover - stow)

    run(UNFOLD_S, unfold)
    run(SETTLE_S, q_hover)
    p0, y0 = pose()
    run(HOLD_S, q_hover)
    p1, y1 = pose()
    drift = np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1)
    turn = np.degrees(np.abs(np.angle(np.exp(1j * (y1 - y0)))))
    rise = np.abs(p1[:, 2] - z_stand)
    ok = (~fell) & (drift < DRIFT_MAX) & (turn < TURN_MAX) & (rise < HEIGHT_MAX)
    print(f"[stability] {time.time() - t0:.0f} s; stable at {ok.sum()} of {n} probes "
          f"(fell {fell.sum()}, drift>{DRIFT_MAX * 1000:.0f} mm {(drift >= DRIFT_MAX).sum()}, "
          f"turn>{TURN_MAX:.0f} deg {(turn >= TURN_MAX).sum()}, "
          f"height>{HEIGHT_MAX * 1000:.0f} mm {(rise >= HEIGHT_MAX).sum()})")

    # Every band cell takes the verdict of its nearest probe.
    from scipy.spatial import cKDTree

    probe_xy = pick * rm.cell + (rm.x0, rm.y0)
    _, nearest = cKDTree(probe_xy).query(xy)
    stable = np.zeros_like(rm.ok)
    stable[cells[:, 0], cells[:, 1]] = ok[nearest]
    path = CACHE / stable_name(rm.key, ckpt)
    np.savez(path, ok=stable, probes=probe_xy, probe_ok=ok, drift=drift, turn=turn, rise=rise,
             fell=fell)
    print(f"[stability] band {rm.ok.sum() * rm.cell**2 * 1e4:.0f} cm^2 -> stable "
          f"{stable.sum() * rm.cell**2 * 1e4:.0f} cm^2; wrote {path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(-xy[:, 1], xy[:, 0], s=4, c="#ccc", label="reach band")
    sc = ax.scatter(-probe_xy[:, 1], probe_xy[:, 0], s=40, c=np.minimum(drift * 1000, 40),
                    cmap="viridis_r", edgecolors=np.where(ok, "k", "r"), linewidths=1.2)
    fig.colorbar(sc, label="trunk drift over the hold, mm (red edge: unstable)")
    ax.plot(0, 0, "r^")
    ax.set_aspect("equal")
    ax.set_xlabel("-y (trunk frame, m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Where five_foot holds still with the arm out")
    fig.savefig(path.with_suffix(".png"), dpi=90, bbox_inches="tight")
    sm.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
