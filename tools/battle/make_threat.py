"""Author the threat display: the body reared nose-up, both claws spread wide and
high, sweeping in and snapping shut on every beat. The blue crab's taunt at the
bell (`moves.py`); the red one's is the Zoidberg, arms straight up.

    .venv/bin/python tools/battle/make_threat.py            # writes the clip + preview

The spread pose was found by a forward-kinematics sweep of the left arm
(2026-10-14): J0 -0.59 (the shoulder's HOME yaw), J1 -0.14 (pitched up from
-1.07), J2 1.17, J3 -0.23 puts the claw tip at (0.097, 0.190, 0.120) m in the
base frame, 19 cm out to the side and 12 cm up, against the Zoidberg's
(0.097, 0.075, 0.143) straight over the shoulder. The right arm is the left's
mirror: J0, J2 and J3 negated, J1 the same. The clip's arithmetic -- the
standing legs solved for the reared base, the writer, the preview -- is
`clips.py`'s, with `tools/fetch/make_zoidberg.py`'s three traps handled there.
"""
from __future__ import annotations

import importlib.util as _ilu

import numpy as np

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
from clips import SPREAD, arm, claw, legs_for, smoothstep, write

from tasks.jumper.common.constants import HOME

NAME = "threat"
SHIFT_S = 0.6                   # the body back over the legs first, arms still down (and last, at the end)
RAISE_S, MENACE_S, LOWER_S = 1.8, 4.8, 1.8
BEAT_S = 1.2                    # one sweep in and out; the Zoidberg's beat, which trained
BODY_BACK = 0.045               # as the Zoidberg: the raised arms move the centre of mass forward
REAR = np.radians(10.0)         # nose up while the arms are spread
BOB = np.radians(3.0)           # the rear bobs with the beat
LIFT_BUMP = 1.2                 # rad of extra elbow lift mid-raise
SWEEP = 0.35                    # rad of shoulder yaw the spread arms sweep in and out with


def pose_at(t: float) -> dict:
    """Joint targets by entity name at time t of the body of the clip (no leads)."""
    q = dict(HOME)
    end_menace = RAISE_S + MENACE_S
    if t < SHIFT_S:                                  # the body back, arms down
        f_back, f, phase, snap = smoothstep(t / SHIFT_S), 0.0, 0.0, 0.0
    elif t < RAISE_S:                                # the arms up and out, the nose up
        f_back, f, phase, snap = 1.0, smoothstep((t - SHIFT_S) / (RAISE_S - SHIFT_S)), 0.0, 0.0
    elif t < end_menace:
        f_back, f = 1.0, 1.0
        phase = 2 * np.pi * (t - RAISE_S) / BEAT_S
        snap = min(1.0, (t - RAISE_S) / BEAT_S, (end_menace - t) / BEAT_S)   # fades in and out: no step
    elif t < end_menace + LOWER_S - SHIFT_S:         # the arms down
        u = (t - end_menace) / (LOWER_S - SHIFT_S)
        f_back, f, phase, snap = 1.0, 1.0 - smoothstep(u), 0.0, 0.0
    else:                                            # the body forward again
        u = (t - end_menace - LOWER_S + SHIFT_S) / SHIFT_S
        f_back, f, phase, snap = 1.0 - smoothstep(u), 0.0, 0.0, 0.0
    # The sweep: both arms yaw inward together at the top of the beat (cos = 1)
    # and the claws shut there; open again as they sweep out.
    inward = 0.5 + 0.5 * np.cos(phase)
    lift = LIFT_BUMP * np.sin(np.pi * f)
    for side in ("LF", "RF"):
        sign = 1 if side == "LF" else -1
        arm(q, side, SPREAD[side], f, lift)
        # LF J0 more negative = yawed forward, so inward (toward the nose) is -SWEEP for LF, +SWEEP for RF
        q[f"{side}_J0_joint"] += -sign * SWEEP * inward * snap
        claw(q, side, f, inward * snap)
    pitch = (REAR + BOB * np.sin(phase) * snap) * f
    back = BODY_BACK * f_back
    if abs(pitch) > 1e-6 or back > 1e-6:
        q.update(legs_for(pitch=pitch, back=back))
    return q


if __name__ == "__main__":
    write(NAME, pose_at, RAISE_S + MENACE_S + LOWER_S, "tools/battle/make_threat.py (keyframed, 2026-10-14)")
