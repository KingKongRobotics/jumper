"""Scripted probe: the right stick's sign, standing, arm web-up, nose down: twist then turn."""
from __future__ import annotations

import math

import numpy as np
from fetch import _quat_rot, _yaw

SEGMENTS = [("twist", +0.3), ("rest", 0.0), ("twist", -0.3), ("rest", 0.0),
            ("turn", +0.75), ("rest", 0.0), ("turn", -0.75), ("rest", 0.0)]
HOLD = 3.0


def make_pilot(env, player):
    from controller import GamepadState
    from tasks.jumper.five_foot.mdp.grasp import _mouth_in_palm

    robot = env.scene["robot"]
    palm = robot.body_names.index("LF_palm_link")
    mouth_in_palm = _mouth_in_palm(env.sim.mj_model)
    mark = {"k": -1}

    def read():
        d = robot.data
        yaw = _yaw(d.root_link_quat_w[0].cpu().numpy())
        pp = d.body_link_pos_w[0, palm].cpu().numpy(); pq = d.body_link_quat_w[0, palm].cpu().numpy()
        mouth = pp + _quat_rot(pq, mouth_in_palm)
        return yaw, mouth, d.root_link_pos_w[0].cpu().numpy()

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=-0.6, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 2.0:
            s["hat_x"] = -1
        k = int((t - 4.0) // HOLD) if t >= 4.0 else -1
        if 0 <= k < len(SEGMENTS):
            kind, value = SEGMENTS[k]
            s["rx"] = value
            if k != mark["k"]:
                yaw, mouth, base = read()
                if mark["k"] >= 0:
                    kind0, v0 = SEGMENTS[mark["k"]]
                    print(f"\n[twist] {kind0:<5} rx={v0:+.2f} for {HOLD:.0f}s: yaw {math.degrees(mark['yaw0']):+6.1f} -> {math.degrees(yaw):+6.1f} deg, "
                          f"mouth y {mark['my0']:+.3f} -> {mouth[1]:+.3f}, base moved {np.linalg.norm(base[:2] - mark['b0']):.3f} m")
                mark.update(k=k, yaw0=yaw, my0=mouth[1], b0=base[:2].copy())
        return GamepadState(**s)

    return pilot
