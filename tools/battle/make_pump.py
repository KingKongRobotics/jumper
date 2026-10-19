"""Author the claw pump: both claws straight up, the body bouncing on its legs
twice a second, the claws snapping shut at the bottom of every bounce and the
arms thrusting at the top. The red crab's victory dance (`moves.py`).

    .venv/bin/python tools/battle/make_pump.py            # writes the clip + preview

The arms are the Zoidberg's (`clips.UP`), raised the Zoidberg's way (the body
back over the legs first, the elbow lifted mid-raise). The bounce is the four
standing legs solved for a body `DIP` lower at the bottom of each beat
(`clips.legs_for`); the feet never move.
"""
from __future__ import annotations

import importlib.util as _ilu

import numpy as np

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
from clips import UP, arm, claw, legs_for, smoothstep, write

from tasks.jumper.common.constants import HOME

NAME = "pump"
SHIFT_S = 0.6
RAISE_S, PUMP_S, LOWER_S = 1.6, 6.0, 1.6
BEAT_S = 0.6                    # one bounce; 2 Hz, below the Zoidberg's 1.2 s sway
BODY_BACK = 0.045
DIP = 0.015                     # m the body drops at the bottom of a bounce: 18 left the rear-right leg 1.2 mm short with the body back 45
LIFT_BUMP = 1.5                 # the Zoidberg's elbow lift mid-raise
THRUST = 0.25                   # rad of elbow the raised arms thrust with at the top of the bounce


def pose_at(t: float) -> dict:
    q = dict(HOME)
    end_pump = RAISE_S + PUMP_S
    if t < SHIFT_S:
        f_back, f, phase, beat = smoothstep(t / SHIFT_S), 0.0, 0.0, 0.0
    elif t < RAISE_S:
        f_back, f, phase, beat = 1.0, smoothstep((t - SHIFT_S) / (RAISE_S - SHIFT_S)), 0.0, 0.0
    elif t < end_pump:
        f_back, f = 1.0, 1.0
        phase = 2 * np.pi * (t - RAISE_S) / BEAT_S
        beat = min(1.0, (t - RAISE_S) / (2 * BEAT_S), (end_pump - t) / (2 * BEAT_S))   # fades in and out over two beats
    elif t < end_pump + LOWER_S - SHIFT_S:
        u = (t - end_pump) / (LOWER_S - SHIFT_S)
        f_back, f, phase, beat = 1.0, 1.0 - smoothstep(u), 0.0, 0.0
    else:
        u = (t - end_pump - LOWER_S + SHIFT_S) / SHIFT_S
        f_back, f, phase, beat = 1.0 - smoothstep(u), 0.0, 0.0, 0.0
    down = 0.5 - 0.5 * np.cos(phase)                 # 1 at the bottom of the bounce
    lift = LIFT_BUMP * np.sin(np.pi * f)
    for side in ("LF", "RF"):
        sign = 1 if side == "LF" else -1
        arm(q, side, UP[side], f, lift)
        q[f"{side}_J2_joint"] += sign * THRUST * (1.0 - down) * beat       # thrust at the top
        claw(q, side, f, down * beat)                                       # shut at the bottom
    back = BODY_BACK * f_back
    drop = DIP * down * beat
    if back > 1e-6 or drop > 1e-6:
        q.update(legs_for(back=back, drop=drop))
    return q


if __name__ == "__main__":
    write(NAME, pose_at, RAISE_S + PUMP_S + LOWER_S, "tools/battle/make_pump.py (keyframed, 2026-10-14)")
