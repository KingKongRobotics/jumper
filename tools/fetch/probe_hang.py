"""How high the claw's mouth hangs over the ground against the nose stick and the
body height, in the claw mode with the arm out (web-up): the numbers `place` needs to
set the mug on the tray. Logs the mouth's world height (the anvil site via the
senses' body frame + the true base pose) every half second while the nose sweeps.
No mug needed."""
from __future__ import annotations

from fetch import _quat_rot
from sense import Senses

SWEEP = [(-0.6, 0.0), (-0.3, 0.0), (0.0, 0.0), (0.4, 0.0), (0.8, 0.0), (1.0, 0.0), (1.0, -1.0), (0.6, -1.0), (0.0, -1.0)]


def make_pilot(env, player):
    from controller import GamepadState

    senses = Senses(env)
    robot = env.scene["robot"]
    last = [-1.0]

    def pilot(t):
        senses.update(t)
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 3.0:
            s["hat_x"] = -1
            k = min(len(SWEEP) - 1, int((t - 3.0) / 3.0))
            nose, height = SWEEP[k]
            s["ry"] = -nose                                 # ry -0.6 is the grasp's nose-down
            if height != 0.0 and (t - 3.0) % 3.0 < 2.0:
                s["buttons"] = frozenset({"R3"}); s["ry"] = height   # R3 + stick: the body height, a speed
        if t - last[0] >= 0.5:
            last[0] = t
            mouth, closing, inward, fin = senses.body()
            bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
            mw = bp + _quat_rot(bq, mouth)
            print(f"\n[hang] t={t:5.1f} ry={s['ry']:+.2f} R3={'R3' in s['buttons']} base_z={bp[2]:.3f} mouth_b=({mouth[0]:+.3f},{mouth[1]:+.3f},{mouth[2]:+.3f}) mouth_world_z={mw[2]:.3f}")
        return GamepadState(**s)

    return pilot
