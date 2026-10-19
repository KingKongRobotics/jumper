"""The lift: a scripted probe of the body lever, against a parked crab.

The crabs' pinch is 0.6 N*m by design and the carried arm holds its preset
with 1.75 N*m at the shoulder, 6 N at a claw 0.3 m out -- a quarter of the
foe's front (2.54 kg, 25 N). A claw cannot flip a crab. The body can: the
trunk's collision mesh sits 0.096 m over the ground at the standing height
(base z 0.106, the mesh from -0.010 to +0.070 in the base frame, 21 cm long,
18 cm wide), the posture mode's height runs 0.07 to 0.15 m and its pitch 20
degrees, so a crab crouched to 0.07 with its nose down has its visor at about
0.10 m, level with the foe's underside, and the same crab tall with its nose
up carries that visor to 0.25 m -- 14 cm under the foe's front edge, which
puts the foe at 25-30 degrees before its feet decide anything.

The script, after the bell (`LIFT_*` to change it): crouch (R3 with the stick
pulled, `CROUCH_S`), face the foe and walk at it nose down (`NOSE_DOWN` on the
right stick) until the range is under `UNDER_M` or stops shrinking for
`STALL_S`; then raise the stance (R3, stick pushed, `RAISE_S`), then nose up
and push (`PUSH_S`); then look again. Every quarter second of the lift it
prints the foe's pitch and height, and the most it reached when the round
ends. The referee calls the flip. Run it with `park.py` on the other side:

    duel.py --scene beach --headless --no-moves --best-of 1 \\
        --red tools/battle/pilots/lift.py --blue tools/battle/pilots/park.py --video out/battle/lift1.mp4
"""
from __future__ import annotations

import math
import os

TURN_DEG = float(os.environ.get("LIFT_TURN_DEG", "12"))
TURN_STICK = float(os.environ.get("LIFT_TURN_STICK", "0.62"))
WALK = float(os.environ.get("LIFT_WALK", "1.0"))
CROUCH_S = float(os.environ.get("LIFT_CROUCH_S", "0.7"))
NOSE_DOWN = float(os.environ.get("LIFT_NOSE_DOWN", "-1.0"))   # right stick up is nose down (posture's controls.yaml)
NOSE_UP = float(os.environ.get("LIFT_NOSE_UP", "1.0"))
UNDER_M = float(os.environ.get("LIFT_UNDER_M", "0.16"))       # base to base with the visor under the foe's front
STALL_S = float(os.environ.get("LIFT_STALL_S", "0.8"))       # counted once the walk has had SETTLE_S to start
SETTLE_S = float(os.environ.get("LIFT_SETTLE_S", "1.0"))
RAISE_S = float(os.environ.get("LIFT_RAISE_S", "1.5"))
PUSH_S = float(os.environ.get("LIFT_PUSH_S", "2.0"))
PUSH_WALK = float(os.environ.get("LIFT_PUSH_WALK", "1.0"))
FLANK_M = float(os.environ.get("LIFT_FLANK_M", "0"))          # > 0: go round to the foe's side first, this far out,
                                                              # and wedge in at the gap between its front and middle
                                                              # legs (head on, the front feet meet at 0.33-0.38 m and
                                                              # the nose never reaches the underside); Andrew: leverage
FLANK_AHEAD = float(os.environ.get("LIFT_FLANK_AHEAD", "0.06"))   # the wedge aims this far ahead of the foe's base


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _nose_up_deg(q) -> float:
    """The body's x axis against the horizon, positive nose up: the x axis's z is
    2(xz - wy). (The sign was flipped until probe_flank4, which logged the push's
    nose-up as -12 deg; the earlier probes' nose readings are reversed.)"""
    w, x, y, z = (float(v) for v in q)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y)))))


def make_pilot(env, player, side):
    from controller import GamepadState

    me, foe = env.scene[side.me], env.scene[side.foe]
    referee = side.referee
    log = lambda t, text: print(f"\n[{side.name}] {t:6.2f}s  {text}")
    st = {"phase": "", "since": 0.0, "turning": False, "closest": None, "closest_at": 0.0,
          "printed": 0.0, "peak": (0.0, 0.0), "round": -1}

    def go(t, phase):
        st.update(phase=phase, since=t, turning=False, closest=None, closest_at=t)
        log(t, f"lift: {phase}")

    def foe_state():
        f = foe.data.root_link_pos_w[0]
        return _nose_up_deg(foe.data.root_link_quat_w[0]), float(f[2])

    def foe_point(ahead, side):
        """A point in the foe's frame (ahead along its nose, side to its left), in the world."""
        f = foe.data.root_link_pos_w[0]
        yaw = _yaw(foe.data.root_link_quat_w[0])
        return (float(f[0]) + math.cos(yaw) * ahead - math.sin(yaw) * side,
                float(f[1]) + math.sin(yaw) * ahead + math.cos(yaw) * side)

    def pilot(t):
        if not referee.live:
            if st["phase"]:
                log(t, f"lift: round over; the foe reached {st['peak'][0]:+.1f} deg nose, base {st['peak'][1]:.3f} m")
                st.update(phase="", peak=(0.0, 0.0))
            return None
        if not st["phase"]:
            go(t, "crouch")
        p, f = me.data.root_link_pos_w[0], foe.data.root_link_pos_w[0]
        if FLANK_M > 0:
            # the flank: the foe's left side (the one away from its claw arm)
            tx, ty = foe_point(FLANK_AHEAD, FLANK_M) if st["phase"] == "flank" else foe_point(FLANK_AHEAD, 0.0)
            dx, dy = tx - float(p[0]), ty - float(p[1])
        else:
            dx, dy = float(f[0] - p[0]), float(f[1] - p[1])
        rng = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - _yaw(me.data.root_link_quat_w[0])
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        since = t - st["since"]
        nose, z = foe_state()
        if abs(nose) > abs(st["peak"][0]):
            st["peak"] = (nose, z)
        if st["phase"] in ("raise", "push") and t - st["printed"] >= 0.25:
            st["printed"] = t
            log(t, f"  foe nose {nose:+.1f} deg, base {z:.3f} m; me nose {_nose_up_deg(me.data.root_link_quat_w[0]):+.1f} deg, "
                   f"base {float(p[2]):.3f} m, range {rng:.3f}")
        if st["phase"] == "crouch":
            if since < CROUCH_S:
                return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
            go(t, "flank" if FLANK_M > 0 else "approach")
        if st["phase"] == "flank":
            # walk to the point out on the foe's side, then turn in on the wedge
            limit = math.radians(TURN_DEG / 3 if st["turning"] else TURN_DEG)
            st["turning"] = abs(bearing) > limit
            if rng < 0.12:
                log(t, f"on the flank at {rng:.2f} m from the mark")
                go(t, "approach")
                return GamepadState()                # (the range is re-read against the wedge's point next step)
            elif st["turning"]:
                return GamepadState(rx=-math.copysign(TURN_STICK, bearing))
            else:
                return GamepadState(ly=-WALK)
        if st["phase"] == "approach":
            limit = math.radians(TURN_DEG / 3 if st["turning"] else TURN_DEG)
            st["turning"] = abs(bearing) > limit
            if st["turning"]:
                return GamepadState(rx=-math.copysign(TURN_STICK, bearing))
            if st["closest"] is None or rng < st["closest"] - 0.01:
                st["closest"], st["closest_at"] = rng, t
            stalled = since > SETTLE_S and t - st["closest_at"] > STALL_S
            if rng < UNDER_M or stalled:
                log(t, f"under at range {rng:.3f} m" if rng < UNDER_M else f"stalled at range {rng:.3f} m")
                go(t, "raise")
                return GamepadState(ry=-1.0, buttons=frozenset({"R3"}))
            return GamepadState(ly=-WALK, ry=NOSE_DOWN)
        if st["phase"] == "raise":
            if since < RAISE_S:
                return GamepadState(ry=-1.0, buttons=frozenset({"R3"}))
            go(t, "push")
            return GamepadState(ly=-PUSH_WALK, ry=NOSE_UP)
        if st["phase"] == "push":
            if since < PUSH_S:
                return GamepadState(ly=-PUSH_WALK, ry=NOSE_UP)
            go(t, "crouch")
            return GamepadState(ry=1.0, buttons=frozenset({"R3"}))
        return None

    return pilot
