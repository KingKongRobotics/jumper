"""Scripted probe: enter the claw mode, hold the arm out, squeeze, walk. Logs geometry."""
from __future__ import annotations

import math

import numpy as np


def _quat_rot(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def make_pilot(env, player):
    from controller import GamepadState
    from tasks.jumper.five_foot.mdp.grasp import _mouth_in_palm

    robot = env.scene["robot"]
    palm = robot.body_names.index("LF_palm_link")
    finger = robot.joint_names.index("LF_J4_joint")
    mouth_in_palm = _mouth_in_palm(env.sim.mj_model)
    print(f"[probe] mouth in palm frame {np.round(mouth_in_palm, 4)}")
    last_log = [-1.0]

    def geometry():
        d = robot.data
        base_p = d.root_link_pos_w[0].cpu().numpy()
        base_q = d.root_link_quat_w[0].cpu().numpy()
        palm_p = d.body_link_pos_w[0, palm].cpu().numpy()
        palm_q = d.body_link_quat_w[0, palm].cpu().numpy()
        mouth_w = palm_p + _quat_rot(palm_q, mouth_in_palm)
        # mouth in the base frame: rotate the world offset by the inverse base quat
        w, x, y, z = base_q
        mouth_b = _quat_rot(np.array([w, -x, -y, -z]), mouth_w - base_p)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return base_p, yaw, mouth_w, mouth_b, float(d.joint_pos[0, finger])

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if 2.0 <= t < 14.0:
            s["hat_y"] = -1            # kernel: down positive; the player negates, so -1 is up
        if 6.0 <= t < 14.0:
            s["lt"] = 1.0
        if 9.0 <= t < 13.0:
            s["ly"] = -0.6             # forward
        if t - last_log[0] >= 0.5:
            last_log[0] = t
            base_p, yaw, mouth_w, mouth_b, fin = geometry()
            print(f"\n[probe] t={t:5.2f} mode={player.fsm.mode:<10} run={player.fsm.running_policy} "
                  f"base=({base_p[0]:+.3f},{base_p[1]:+.3f},{base_p[2]:.3f}) yaw={math.degrees(yaw):+.1f} "
                  f"mouth_b=({mouth_b[0]:+.3f},{mouth_b[1]:+.3f},{mouth_b[2]:+.3f}) finger={fin:+.3f}")
        return GamepadState(**s)

    return pilot
