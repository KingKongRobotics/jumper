"""Author the wave: both claws spread, the body rolling side to side and the arm
on the high side rising with each roll, the claws snapping as they come up --
raising the roof, one claw at a time. The blue crab's victory dance (`moves.py`).

    .venv/bin/python tools/battle/make_wave.py            # writes the clip + preview

The arms start from the threat's spread (`clips.SPREAD`); the roll is the four
standing legs solved against the rolled base (`clips.legs_for`), the Zoidberg's
sway at the Zoidberg's amplitude.
"""
from __future__ import annotations

import importlib.util as _ilu

import numpy as np

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
from clips import SPREAD, arm, claw, legs_for, smoothstep, write

from tasks.jumper.common.constants import HOME

NAME = "wave"
SHIFT_S = 0.6
RAISE_S, WAVE_S, LOWER_S = 1.8, 6.0, 1.8
BEAT_S = 1.2                    # one roll left and right; the Zoidberg's
BODY_BACK = 0.045
SWAY = np.radians(7.0)          # the Zoidberg's roll amplitude
LIFT_BUMP = 1.2
RAISE = 0.5                     # rad of shoulder pitch (J1) the high-side arm rises by
DIP = 0.3                       # rad the low-side arm dips by


def pose_at(t: float) -> dict:
    q = dict(HOME)
    end_wave = RAISE_S + WAVE_S
    if t < SHIFT_S:
        f_back, f, phase, beat = smoothstep(t / SHIFT_S), 0.0, 0.0, 0.0
    elif t < RAISE_S:
        f_back, f, phase, beat = 1.0, smoothstep((t - SHIFT_S) / (RAISE_S - SHIFT_S)), 0.0, 0.0
    elif t < end_wave:
        f_back, f = 1.0, 1.0
        phase = 2 * np.pi * (t - RAISE_S) / BEAT_S
        beat = min(1.0, (t - RAISE_S) / BEAT_S, (end_wave - t) / BEAT_S)
    elif t < end_wave + LOWER_S - SHIFT_S:
        u = (t - end_wave) / (LOWER_S - SHIFT_S)
        f_back, f, phase, beat = 1.0, 1.0 - smoothstep(u), 0.0, 0.0
    else:
        u = (t - end_wave - LOWER_S + SHIFT_S) / SHIFT_S
        f_back, f, phase, beat = 1.0 - smoothstep(u), 0.0, 0.0, 0.0
    sway = np.sin(phase) * beat                      # positive: the body rolls left side down
    lift = LIFT_BUMP * np.sin(np.pi * f)
    for side in ("LF", "RF"):
        up = 0.5 + 0.5 * (-sway if side == "LF" else sway)     # the right arm rises as the left side dips
        arm(q, side, SPREAD[side], f, lift)
        q[f"{side}_J1_joint"] += (RAISE * up - DIP * (1.0 - up)) * beat * f
        claw(q, side, f, up * beat)                             # snaps as it comes up
    roll = SWAY * sway * f
    back = BODY_BACK * f_back
    if abs(roll) > 1e-6 or back > 1e-6:
        q.update(legs_for(roll=roll, back=back))
    return q


if __name__ == "__main__":
    write(NAME, pose_at, RAISE_S + WAVE_S + LOWER_S, "tools/battle/make_wave.py (keyframed, 2026-10-14)")
