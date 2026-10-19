"""The brawler: find the other crab in the depth picture, face it, walk into it.

`charge.py` with the cheat taken out: the foe is a crab-sized cluster in this
crab's own dToF image (`eyes.py`), its bearing and range in the base frame, and
nothing is read from the world. Not seen for `eyes.HOLD_S`, the brawler turns
in place toward where the foe was last, the way the fetch's wander looks about;
the sensor sees 55 degrees, so a foe lost to the side comes back into the
picture within a quarter turn. Nearer than `SHOVE_M` the picture is a wall of
crab and the bearing means little: the brawler just keeps walking.
"""
from __future__ import annotations

import math
import os

SENSES = True

TURN_DEG = float(os.environ.get("BRAWL_TURN_DEG", "12"))
TURN_STICK = float(os.environ.get("BRAWL_TURN_STICK", "0.62"))
WALK = float(os.environ.get("BRAWL_WALK", "1.0"))
SHOVE_M = float(os.environ.get("BRAWL_SHOVE_M", "0.30"))
CROUCH_S = float(os.environ.get("BRAWL_CROUCH_S", "0.0"))


def make_pilot(env, player, side):
    from eyes import Eyes

    from controller import GamepadState

    eyes = Eyes(env, side.me)
    referee = side.referee
    state = {"turning": False, "last_sign": 1.0, "seen": 0, "lost": 0}

    def pilot(t):
        eyes.update(t)
        if not referee.live:
            state["turning"] = False
            return None
        if t - referee.fight_since < CROUCH_S:
            return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
        foe = eyes.foe(t)
        if foe is None:
            state["lost"] += 1
            return GamepadState(rx=-state["last_sign"] * TURN_STICK)        # look about toward the last sighting
        state["seen"] += 1
        bearing = foe["bearing"]
        state["last_sign"] = math.copysign(1.0, bearing) if abs(bearing) > 0.02 else state["last_sign"]
        if foe["near"] < SHOVE_M:
            return GamepadState(ly=-WALK)
        limit = math.radians(TURN_DEG / 3 if state["turning"] else TURN_DEG)
        state["turning"] = abs(bearing) > limit
        if state["turning"]:
            return GamepadState(rx=-math.copysign(TURN_STICK, bearing))
        return GamepadState(ly=-WALK)

    pilot.eyes = eyes
    pilot.state = state
    return pilot
