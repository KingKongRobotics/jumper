"""Scripted probe: cycle the three arm presets and log the mouth and the closing axis."""
from __future__ import annotations

import numpy as np
from fetch import _quat_rot


def make_pilot(env, player):
    from controller import GamepadState
    from tasks.jumper.five_foot.mdp.grasp import _closing_in_palm, _inward_in_palm, _mouth_in_palm

    robot = env.scene["robot"]
    palm = robot.body_names.index("LF_palm_link")
    finger = robot.joint_names.index("LF_J4_joint")
    m = env.sim.mj_model
    mouth_in_palm, closing_in_palm, inward_in_palm = _mouth_in_palm(m), _closing_in_palm(m), _inward_in_palm(m)
    last = [-1.0]
    presets = [("stow", {}), ("thumb_up", {"hat_y": -1}), ("thumb_down", {"hat_y": +1}), ("web_up", {"hat_x": -1})]

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        k = int((t - 3.0) // 3.0) if t >= 3.0 else -1
        name, held = presets[min(max(k, 0), 3)] if k >= 0 else ("none", {})
        s.update(held)
        if (t - 3.0) % 3.0 > 2.0 and k >= 0:
            s["lt"] = 1.0
        if t - last[0] >= 0.5:
            last[0] = t
            d = robot.data
            bp = d.root_link_pos_w[0].cpu().numpy(); bq = d.root_link_quat_w[0].cpu().numpy()
            pp = d.body_link_pos_w[0, palm].cpu().numpy(); pq = d.body_link_quat_w[0, palm].cpu().numpy()
            mouth = pp + _quat_rot(pq, mouth_in_palm)
            w, x, y, z = bq
            inv = np.array([w, -x, -y, -z])
            mouth_b = _quat_rot(inv, mouth - bp)
            closing_b = _quat_rot(inv, _quat_rot(pq, closing_in_palm))
            inward_b = _quat_rot(inv, _quat_rot(pq, inward_in_palm))
            print(f"\n[preset] t={t:5.2f} {name:<10} mode={player.fsm.mode:<10} base_z={bp[2]:.3f} "
                  f"mouth_b=({mouth_b[0]:+.3f},{mouth_b[1]:+.3f},{mouth_b[2]:+.3f}) mouth_z_w={mouth[2]:.3f} "
                  f"closing_b={np.round(closing_b, 2)} inward_b={np.round(inward_b, 2)} finger={float(d.joint_pos[0, finger]):+.3f}")
        return GamepadState(**s)

    return pilot
