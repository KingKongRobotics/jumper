"""The charger: face the other crab and walk into it.

The first opponent, and a cheat: it reads both crabs' poses from the world
rather than from its own senses, so the duel's plumbing can be tried before a
pilot that looks for the other crab in the dToF picture (the way
`tools/fetch/sense.py` finds the mug) replaces it. The stick numbers are the
posture mode's as measured for the fetch: the turn band is the right stick
past 0.5 (0.54 turns slowly, 0.62 about 60 deg/s), a twist under 0.5 while
walking bends the walk by a few degrees a second at best, so the charger stops
to turn and walks straight. `CHARGE_CROUCH_S` seconds of R3 with the stick
pulled lowers the stance at the bell (0.7 s reaches the lowest, 0.073 m): a
lower crab is harder to tip, and the height stays where the stick left it.
"""
from __future__ import annotations

import math
import os

TURN_DEG = float(os.environ.get("CHARGE_TURN_DEG", "12"))     # stop and turn when the foe is this far off the nose
TURN_STICK = float(os.environ.get("CHARGE_TURN_STICK", "0.62"))
WALK = float(os.environ.get("CHARGE_WALK", "1.0"))             # left stick forward, 1.0 = the policy's top speed
CROUCH_S = float(os.environ.get("CHARGE_CROUCH_S", "0.0"))
MODE = os.environ.get("CHARGE_MODE", "")                     # `claw_left`: charge in the claw mode, arm out, snapping
SNAP_M = float(os.environ.get("CHARGE_SNAP_M", "0.45"))        # squeeze the trigger when the foe is this near
JUMP_M = float(os.environ.get("CHARGE_JUMP_M", "0"))           # > 0: press A (the jump) when the foe is this near, once
                                                              # per round -- does the in-place jump, pressed on the
                                                              # walk, carry the crab onto the foe? (Andrew: "the crab
                                                              # can jump too... a dash movement")
STRIKE_S = float(os.environ.get("CHARGE_STRIKE_S", "0"))       # > 0 with MODE: within SNAP_M swing the arm between the
                                                              # out and the folded presets every this many seconds
PRESS_S = 0.3


def _nose_up_deg(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.degrees(math.asin(max(-1.0, min(1.0, -2.0 * (x * z - w * y)))))


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def make_pilot(env, player, side):
    from controller import GamepadState

    me, foe = env.scene[side.me], env.scene[side.foe]
    referee = side.referee
    turning = False
    pressed = None
    probe = {"jumped": None, "jump_from": None, "peak": 0.0, "printed": 0.0}
    log = lambda t, text: print(f"\n[{getattr(side, 'name', side.me)}] {t:6.2f}s  {text}")

    def pilot(t):
        nonlocal turning, pressed
        if not referee.live:
            if probe["jumped"] is not None or probe["peak"]:
                log(t, f"probe: round over; the foe's peak tilt {probe['peak']:.1f} deg"
                       + (f"; the jump carried me {probe['travel']:.2f} m" if probe.get("travel") is not None else ""))
            turning, pressed = False, None
            probe.update(jumped=None, jump_from=None, peak=0.0, travel=None)
            return None
        fq = foe.data.root_link_quat_w[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (float(fq[1]) ** 2 + float(fq[2]) ** 2)))))
        probe["peak"] = max(probe["peak"], tilt)
        if probe["jumped"] is not None and t - probe["jumped"] > 2.5 and probe.get("travel") is None:
            q = me.data.root_link_pos_w[0]
            probe["travel"] = math.hypot(float(q[0]) - probe["jump_from"][0], float(q[1]) - probe["jump_from"][1])
            log(t, f"probe: the jump carried me {probe['travel']:.2f} m; the foe's tilt now {tilt:.1f} deg (peak {probe['peak']:.1f})")
        fsm = player.fsm
        held = {}
        if MODE:
            # In the claw mode the arm is held out (d-pad left, the web-up preset) and the
            # trigger squeezed within SNAP_M; walking there stands still under 0.4 of stick.
            if not (fsm.mode == MODE and fsm.running_policy):
                if pressed is None and fsm.mode == "locomotion" and fsm.running_policy:
                    pressed = t
                if pressed is not None and t - pressed < PRESS_S:
                    return GamepadState(buttons=side.presses[MODE].buttons)
                return None
            held = {"hat_x": -1}
        elif t - referee.fight_since < CROUCH_S:
            return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
        p = me.data.root_link_pos_w[0]
        f = foe.data.root_link_pos_w[0]
        dx, dy = float(f[0] - p[0]), float(f[1] - p[1])
        if MODE and math.hypot(dx, dy) < SNAP_M:
            held["lt"] = 1.0
            if STRIKE_S > 0:
                held["hat_x"] = -1 if int(t / STRIKE_S) % 2 == 0 else +1      # the swing: out, folded, out...
        if JUMP_M > 0 and probe["jumped"] is None and math.hypot(dx, dy) < JUMP_M and fsm.mode == "locomotion" and fsm.running_policy:
            probe["jumped"], probe["jump_from"] = t, (float(p[0]), float(p[1]))
            log(t, f"probe: A at range {math.hypot(dx, dy):.2f} m")
            return GamepadState(ly=-WALK, buttons=frozenset({"A"}))
        if probe["jumped"] is not None and t - probe["jumped"] < 2.5:
            return GamepadState(ly=-WALK)                                   # the walk held through the jump
        bearing = math.atan2(dy, dx) - _yaw(me.data.root_link_quat_w[0])
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        limit = math.radians(TURN_DEG / 3 if turning else TURN_DEG)
        turning = abs(bearing) > limit
        if turning:
            # Right on the stick is clockwise; a foe to the left (positive bearing) needs counter-clockwise.
            return GamepadState(rx=-math.copysign(TURN_STICK, bearing), **held)
        return GamepadState(ly=-WALK, **held)

    return pilot
