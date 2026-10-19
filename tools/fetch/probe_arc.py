"""Does a twist held while walking bend the walk in the claw mode, and how fast? Arm
out, nose at ARC_NOSE, walk at ARC_LY with the right stick at ARC_RX for 8 s; logs the
true yaw each second. No mug."""
from __future__ import annotations

import math
import os

from fetch import _yaw

LY, RX, NOSE = float(os.environ.get("ARC_LY", "-0.5")), float(os.environ.get("ARC_RX", "0.4")), float(os.environ.get("ARC_NOSE", "0.5"))


def make_pilot(env, player):
    from controller import GamepadState

    robot = env.scene["robot"]
    last = [-1.0]

    def pilot(t):
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 3.0:
            s["hat_x"] = -1
            s["ry"] = NOSE
        if 5.0 <= t < 13.0:
            s["ly"], s["rx"] = LY, RX
        elif 13.0 <= t < 21.0:
            s["ly"], s["rx"] = LY, -RX                       # and the other way
        if os.environ.get("ARC_LT"):
            s["lt"] = 1.0
        if t - last[0] >= 1.0:
            last[0] = t
            bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
            print(f"\n[arc] t={t:5.1f} pos=({bp[0]:+.2f},{bp[1]:+.2f}) yaw={math.degrees(_yaw(bq)):+6.1f} mode={player.fsm.mode}")
        return GamepadState(**s)

    return pilot
