"""Survey the bundle's shipped moves, one per run: MODE names a dance / gesture chord,
or `sit` (R3 held, the height stick down), `rise`, or `turn` (right stick full)."""
from __future__ import annotations

import math
import os

MODE = os.environ.get("MODE", "dance_crab")
CHORDS = {"dance_crab": ({"menu"}, 0, -1), "dance_brazilian": ({"menu"}, 0, +1), "dance_maze": ({"menu"}, -1, 0),
          "dance_dream_wings": ({"menu"}, +1, 0), "gesture_hello": (set(), 0, -1), "gesture_bow": (set(), 0, +1),
          "gesture_paw": (set(), -1, 0), "gesture_salute": (set(), +1, 0), "jump": ({"A"}, 0, 0), "gesture_zoidberg": ({"X"}, 0, 0),
          "gesture_sleep": ({"Y"}, 0, 0)}


def make_pilot(env, player):
    from fetch import _yaw

    from controller import GamepadState

    robot = env.scene["robot"]
    last = [-1.0]

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if MODE in CHORDS:
            btn, hx, hy = CHORDS[MODE]
            if 1.0 <= t < 1.3:
                s["buttons"], s["hat_x"], s["hat_y"] = frozenset(btn), hx, hy
        elif MODE in ("sit", "rise"):
            if 1.0 <= t < 4.0:
                s["buttons"] = frozenset({"R3"})
                s["ry"] = +1.0 if MODE == "sit" else -1.0      # R3 + stick down = height down (measured: -1 rose to 0.153)
        elif MODE == "turn":
            if 1.0 <= t < 9.0:
                s["rx"] = 1.0
        if t - last[0] >= (0.1 if os.environ.get("JOINTS") else 1.0):
            last[0] = t
            bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
            extra = ""
            if os.environ.get("JOINTS"):
                jp = robot.data.joint_pos[0].cpu().numpy()
                i4, j4 = robot.joint_names.index("LF_J4_joint"), robot.joint_names.index("RF_J4_joint")
                i2 = robot.joint_names.index("LF_J2_joint")
                w, x, y, z = bq
                roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                extra = f" LF_J2={jp[i2]:+.2f} LF_J4={jp[i4]:+.2f} RF_J4={jp[j4]:+.2f} roll={roll:+.1f}"
            print(f"\n[modes] t={t:5.1f} mode={player.fsm.mode:<18} base=({bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:.3f}) yaw={math.degrees(_yaw(bq)):+6.1f}{extra}")
        return GamepadState(**s)

    return pilot
