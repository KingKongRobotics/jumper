"""The parked crab: stands on its mark and does nothing. The target of a probe
(`lift.py`, `grab.py`), not an opponent: the shove, the lift and the claw are
measured against a crab that neither braces nor walks, as the fetch's props
were. With `PARK_MODE=claw_left` (or `claw_right`) it enters that mode at the
bell and holds its arm straight out (d-pad left, the web-up preset) for the
claw-lock probe, the arm a bar for the other crab's claw to take."""
from __future__ import annotations

import os

MODE = os.environ.get("PARK_MODE", "")
PRESS_S = 0.3


def make_pilot(env, player, side):
    from controller import GamepadState

    referee = side.referee
    st = {"pressed": None}

    def pilot(t):
        if not MODE or not referee.live:
            st["pressed"] = None
            return None
        fsm = player.fsm
        if fsm.mode == MODE and fsm.running_policy:
            return GamepadState(hat_x=-1)                       # the arm out, held
        if st["pressed"] is None:
            if fsm.mode == "locomotion" and fsm.running_policy:
                st["pressed"] = t
                return GamepadState(buttons=side.presses[MODE].buttons)
            return None
        if t - st["pressed"] < PRESS_S:
            return GamepadState(buttons=side.presses[MODE].buttons)
        return None

    return pilot
