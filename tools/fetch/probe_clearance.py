"""Where in the mouth a standing can clears both jaws, computed from the claw's own
collision geometry at the real pose (arm web-up, nose down), open and then shut."""
from __future__ import annotations

import mujoco
import numpy as np

R, H = 0.029, 0.142          # the can


def make_pilot(env, player):
    from controller import GamepadState
    from tasks.jumper.five_foot.mdp.grasp import (
        _closing_in_palm,
        _inward_in_palm,
        _jaw_clouds,
        _mouth_in_palm,
        _named,
    )

    robot = env.scene["robot"]
    m = env.sim.mj_model
    palm = _named(m, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    finger_adr = m.jnt_qposadr[_named(m, mujoco.mjtObj.mjOBJ_JOINT, "LF_J4_joint")]
    mouth_p, closing_p, inward_p = _mouth_in_palm(m), _closing_in_palm(m), _inward_in_palm(m)
    done = [False]

    def clearance(cloud, centre):
        pts = cloud[(cloud[:, 2] >= 0.0) & (cloud[:, 2] <= H)]
        if len(pts) == 0:
            return np.inf
        return float(np.min(np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])) - R)

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=-0.6, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 2.0:
            s["hat_x"] = -1
        if t >= 5.5 and not done[0]:
            done[0] = True
            d = mujoco.MjData(m)
            d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(m, d)
            rot = d.xmat[palm].reshape(3, 3)
            anvil = d.xpos[palm] + rot @ mouth_p
            closing, inward = rot @ closing_p, rot @ inward_p
            closing[2] = 0; closing /= np.linalg.norm(closing)
            inward[2] = 0; inward /= np.linalg.norm(inward)
            print(f"\n[clear] finger {d.qpos[finger_adr]:+.3f}; anvil z {anvil[2]:.3f}; closing {np.round(closing, 2)}; inward {np.round(inward, 2)}")
            for finger, label in ((d.qpos[finger_adr], "open as held"), (-0.30, "shut on a 40 mm gap")):
                d.qpos[finger_adr] = finger
                mujoco.mj_forward(m, d)
                moving, fixed = _jaw_clouds(m, d)
                print(f"\n[clear] {label}: rows = depth along inward (cm), cols = gap off the anvil along closing (cm); "
                      f"fixed/moving clearance in mm (negative = inside the jaw)")
                gaps = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040]
                print("[clear]      " + "".join(f"{g * 100:>12.1f}" for g in gaps))
                for depth in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
                    row = []
                    for g in gaps:
                        c = anvil + inward * depth + closing * (R + g)
                        row.append(f"{clearance(fixed, c) * 1000:+5.0f}/{clearance(moving, c) * 1000:+5.0f}")
                    print(f"[clear] {depth * 100:4.0f}  " + "".join(f"{x:>12}" for x in row))
        return GamepadState(**s)

    return pilot
