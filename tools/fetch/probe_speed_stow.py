"""Scripted probe: in the claw mode with the arm web-up, hold each stick value and measure speed."""
from __future__ import annotations

import numpy as np

SEGMENTS = [("ly", -0.5), ("ly", -0.5), ("lx", -0.5), ("ly", +0.5)]
HOLD = 4.0


def make_pilot(env, player):
    from controller import GamepadState

    robot = env.scene["robot"]
    mark = {"k": -1, "p0": None, "t0": 0.0, "yaw0": 0.0}

    def yaw():
        w, x, y, z = robot.data.root_link_quat_w[0].cpu().numpy()
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 2.0:
            pass  # arm stowed
        k = int((t - 4.0) // HOLD) if t >= 4.0 else -1
        if 0 <= k < len(SEGMENTS):
            axis, value = SEGMENTS[k]
            s[axis] = value
            if k != mark["k"]:
                if mark["k"] >= 0:
                    _report(t)
                mark.update(k=k, p0=robot.data.root_link_pos_w[0, :2].cpu().numpy().copy(), t0=t, yaw0=yaw())
        elif k >= len(SEGMENTS) and mark["k"] >= 0:
            _report(t)
            mark["k"] = -2
        return GamepadState(**s)

    def _report(t):
        axis, value = SEGMENTS[mark["k"]]
        p = robot.data.root_link_pos_w[0, :2].cpu().numpy()
        d = p - mark["p0"]
        dt = t - mark["t0"]
        dyaw = (yaw() - mark["yaw0"] + np.pi) % (2 * np.pi) - np.pi
        print(f"\n[speed] {axis}={value:+.2f} for {dt:.1f}s: moved dx={d[0]:+.3f} dy={d[1]:+.3f} m "
              f"-> {np.linalg.norm(d) / dt:.3f} m/s, yaw {np.degrees(dyaw):+.1f} deg ({np.degrees(dyaw) / dt:+.1f} deg/s)")

    return pilot
