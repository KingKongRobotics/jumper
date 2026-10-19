"""The jaw beam's pitch against the nose stick, standing in claw mode with the arm out:
the setting at which the beam is level is where a pinched bar has no gravity along it.
Logs the inward axis (along the beam, base frame) every 0.5 s while the nose steps."""
from __future__ import annotations

import math

from sense import Senses

STEPS = [-0.6, -0.3, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def make_pilot(env, player):
    from controller import GamepadState

    senses = Senses(env)
    last = [-1.0]

    def pilot(t):
        senses.update(t)
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 3.0:
            s["hat_x"] = -1
            s["ry"] = STEPS[min(len(STEPS) - 1, int((t - 3.0) / 2.0))]
        if t - last[0] >= 0.5:
            last[0] = t
            mouth, closing, inward, fin = senses.body()
            up = senses.up()
            print(f"\n[pitch] t={t:5.1f} ry={s['ry']:+.2f} inward=({inward[0]:+.3f},{inward[1]:+.3f},{inward[2]:+.3f}) "
                  f"beam pitch {math.degrees(math.asin(max(-1.0, min(1.0, -float(inward[2]))))):+.1f} deg (base frame); body up z {up[2]:.3f}")
        return GamepadState(**s)

    return pilot
