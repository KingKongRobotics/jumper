"""The grab: a scripted probe of the claw against a parked crab.

The body lever failed its probe (`lift.py`, 2026-10-14): head on, the two
crabs' front legs meet at 0.33-0.35 m base to base and the crouched nose never
reaches the foe's underside; the tall nose-up push only shoves. The carried
arm is the other reach: in a claw mode the web-up preset holds the claw 0.3 m
ahead of the base at ground level (`tools/fetch`, 2026-10-12), which is under
the foe's shell and among its legs before the trunks are near. This script
asks whether the claw can take a leg and what pulling on it does.

After the bell (`GRAB_*` to change it): LB into `claw_left`; the arm out
(d-pad left held) with the nose down (`NOSE` on the right stick); face the foe
and creep in (`WALK`, over the claw mode's 0.4 dead band) until the range is
under `REACH_M` or stops shrinking; squeeze (LT) for `GRIP_S`; then walk
backward (`DRAG_S`, stopping `EDGE_M` from the plank's end) with the claw
shut, and let go. It prints the finger's angle (shut on nothing is one number,
on a leg another), the foe's displacement and tilt, every quarter second of
the drag. The referee calls a fall or a flip. With `park.py` on the other side:

    BATTLE_CAM_X=0.3 BATTLE_CAM_DIST=1.5 duel.py --scene beach --headless --no-moves --best-of 1 \\
        --red tools/battle/pilots/grab.py --blue tools/battle/pilots/park.py --video out/battle/grab1.mp4
"""
from __future__ import annotations

import math
import os

TURN_DEG = float(os.environ.get("GRAB_TURN_DEG", "12"))
TURN_STICK = float(os.environ.get("GRAB_TURN_STICK", "0.62"))
WALK = float(os.environ.get("GRAB_WALK", "0.6"))
NOSE = float(os.environ.get("GRAB_NOSE", "-1.0"))             # right stick forward is nose down: claw to the ground
ARM_S = float(os.environ.get("GRAB_ARM_S", "1.5"))            # the arm swinging out to its preset
REACH_M = float(os.environ.get("GRAB_REACH_M", "0.025"))      # the mouth this close to the aim point
# The aim: a point on the foe's right-front strut (its claw, palm to finger
# tip, standing on the plank: palm (0.120, -0.133) 0.074 m up, tip (0.178,
# -0.020) 0.012 m up in its frame, measured parked 2026-10-14) at 0.04 m over
# the plank, where the mouth is with the nose down. Reaching it with the mouth
# puts this crab 4 cm to its own left of the foe's line, where its right-front
# foot passes the foe's left-front one instead of meeting it at 0.35 m.
AIM = (float(os.environ.get("GRAB_AIM_X", "0.152")), float(os.environ.get("GRAB_AIM_Y", "-0.071")))
MOUTH = (0.18, 0.005)                                          # the mouth's centre in this crab's frame, web-up
SIDESTEP_M = float(os.environ.get("GRAB_SIDESTEP_M", "0.4"))  # within this of the aim, a lateral miss over 3 cm is strafed out
SIDESTEP = float(os.environ.get("GRAB_SIDESTEP", "0.6"))       # the claw mode's strafe stands still under 0.4 of stick
STALL_S = float(os.environ.get("GRAB_STALL_S", "0.8"))
SETTLE_S = float(os.environ.get("GRAB_SETTLE_S", "1.0"))
GRIP_S = float(os.environ.get("GRAB_GRIP_S", "0.6"))
DRAG_S = float(os.environ.get("GRAB_DRAG_S", "3.0"))
DRAG_WALK = float(os.environ.get("GRAB_DRAG_WALK", "0.7"))
EDGE_M = float(os.environ.get("GRAB_EDGE_M", "0.15"))
PRESS_S = 0.3
MODE = os.environ.get("GRAB_MODE", "claw_left")
FINGER = "LF_J4_joint" if MODE == "claw_left" else "RF_J4_joint"


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _tilt_deg(q) -> float:
    _w, x, y, _z = (float(v) for v in q)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def make_pilot(env, player, side):
    from arena import PLANK_LEN

    from controller import GamepadState

    me, foe = env.scene[side.me], env.scene[side.foe]
    referee = side.referee
    buttons = side.presses[MODE].buttons                      # LB, from the app's own FSM table
    try:
        finger = list(me.joint_names).index(FINGER)
    except (AttributeError, ValueError):
        finger = None
    log = lambda t, text: print(f"\n[{side.name}] {t:6.2f}s  {text}")
    st = {"phase": "", "since": 0.0, "turning": False, "closest": None, "closest_at": 0.0, "printed": 0.0,
          "pressed": None, "foe0": None, "peak": 0.0}

    def go(t, phase):
        st.update(phase=phase, since=t, turning=False, closest=None, closest_at=t)
        log(t, f"grab: {phase} ({player.fsm.mode})")

    def in_frame(ent, world):
        """A world point in `ent`'s base frame: x forward, y left, z over the base."""
        import numpy as np
        q = ent.data.root_link_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        d = np.asarray([float(v) for v in world]) - np.asarray([float(v) for v in ent.data.root_link_pos_w[0]])
        return R.T @ d

    def where(t):
        """Where the claw is, in its own base frame, and where the foe's front
        legs are in the foe's: the numbers the reach is set by."""
        side_ = "LF" if MODE == "claw_left" else "RF"
        mine = {n: in_frame(me, me.data.body_link_pos_w[0, list(me.body_names).index(f"{side_}_{n}_link")])
                for n in ("palm", "finger_tip")}
        log(t, "  my claw in my frame: " + ", ".join(f"{n} ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})" for n, v in mine.items())
            + f"; my base {float(me.data.root_link_pos_w[0][2]):.3f} m up")
        for leg in ("LF", "RF"):
            parts = {n: in_frame(foe, foe.data.body_link_pos_w[0, list(foe.body_names).index(f"{leg}_{n}_link")])
                     for n in ("shoulder", "upper_arm", "forearm", "palm", "finger_tip")}
            foot = in_frame(foe, foe.data.site_pos_w[0, list(foe.site_names).index(leg)])
            log(t, f"  foe's {leg} in its frame: " + ", ".join(f"{n} ({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})" for n, v in parts.items())
                + f", foot ({foot[0]:+.3f}, {foot[1]:+.3f}, {foot[2]:+.3f})")
        claw_w = me.data.body_link_pos_w[0, list(me.body_names).index(f"{side_}_palm_link")]
        c = in_frame(foe, claw_w)
        log(t, f"  my palm in the foe's frame: ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})")

    def arm(**kw):
        return GamepadState(hat_x=-1, ry=NOSE, **kw)       # d-pad left: the web-up preset, held

    def pilot(t):
        fsm = player.fsm
        if not referee.live:
            if st["phase"]:
                log(t, f"grab: round over; the foe tilted {st['peak']:.1f} deg at most")
                st.update(phase="", pressed=None, peak=0.0)
            return None
        if not st["phase"]:
            go(t, "enter")
        p, f = me.data.root_link_pos_w[0], foe.data.root_link_pos_w[0]
        dx, dy = float(f[0] - p[0]), float(f[1] - p[1])
        rng = math.hypot(dx, dy)
        # The mouth to the aim point, in this crab's frame (yaw only)
        my_yaw, foe_yaw = _yaw(me.data.root_link_quat_w[0]), _yaw(foe.data.root_link_quat_w[0])
        tx = float(f[0]) + math.cos(foe_yaw) * AIM[0] - math.sin(foe_yaw) * AIM[1]
        ty = float(f[1]) + math.sin(foe_yaw) * AIM[0] + math.cos(foe_yaw) * AIM[1]
        mx = float(p[0]) + math.cos(my_yaw) * MOUTH[0] - math.sin(my_yaw) * MOUTH[1]
        my = float(p[1]) + math.sin(my_yaw) * MOUTH[0] + math.cos(my_yaw) * MOUTH[1]
        ex, ey = tx - mx, ty - my
        err_x = math.cos(my_yaw) * ex + math.sin(my_yaw) * ey
        err_y = -math.sin(my_yaw) * ex + math.cos(my_yaw) * ey
        dist = math.hypot(err_x, err_y)
        bearing = math.atan2(err_y, err_x)
        since = t - st["since"]
        tilt = _tilt_deg(foe.data.root_link_quat_w[0])
        st["peak"] = max(st["peak"], tilt)
        if st["phase"] in ("grip", "drag") and t - st["printed"] >= 0.25:
            st["printed"] = t
            moved = math.hypot(float(f[0] - st["foe0"][0]), float(f[1] - st["foe0"][1])) if st["foe0"] is not None else 0.0
            fin = f"{float(me.data.joint_pos[0, finger]):+.2f} rad" if finger is not None else "?"
            log(t, f"  finger {fin}; foe moved {moved:.3f} m, tilt {tilt:.1f} deg, base {float(f[2]):.3f}; range {rng:.3f}")
        if st["phase"] == "enter":
            if fsm.mode == MODE and fsm.running_policy:
                go(t, "arm")
            elif st["pressed"] is None:
                if fsm.mode == "locomotion" and fsm.running_policy:
                    st["pressed"] = t
                    return GamepadState(buttons=buttons)
                return None
            elif t - st["pressed"] < PRESS_S:
                return GamepadState(buttons=buttons)
            else:
                return None
        if st["phase"] == "arm":
            if since < ARM_S:
                return arm()
            go(t, "approach")
        if st["phase"] == "approach":
            limit = math.radians(TURN_DEG / 3 if st["turning"] else TURN_DEG)
            if dist < 0.12:
                limit = math.radians(30.0)                       # close in, a small miss is a wide bearing
            st["turning"] = abs(bearing) > limit and err_x > -0.02
            if st["turning"]:
                return arm(rx=-math.copysign(TURN_STICK, bearing))
            if dist < SIDESTEP_M and abs(err_y) > 0.03:
                return arm(lx=-math.copysign(SIDESTEP, err_y))  # a strafe pulse: left stick right is right
            if st["closest"] is None or dist < st["closest"] - 0.01:
                st["closest"], st["closest_at"] = dist, t
            stalled = since > SETTLE_S and t - st["closest_at"] > STALL_S
            if dist < REACH_M or stalled:
                log(t, f"{'in reach' if dist < REACH_M else 'stalled'}: the mouth {dist:.3f} m from the aim "
                       f"({err_x:+.3f} ahead, {err_y:+.3f} left), range {rng:.3f} m")
                st["foe0"] = (float(f[0]), float(f[1]))
                go(t, "grip")
                where(t)
                return arm(lt=1.0)
            return arm(ly=-WALK)
        if st["phase"] == "grip":
            if since < GRIP_S:
                return arm(lt=1.0)
            go(t, "drag")
        if st["phase"] == "drag":
            yaw = _yaw(me.data.root_link_quat_w[0])
            behind = PLANK_LEN / 2 - abs(float(p[0]) - math.cos(yaw) * 0.15)
            if since < DRAG_S and behind > EDGE_M:
                return arm(lt=1.0, ly=+DRAG_WALK)
            log(t, "lets go" + ("" if since >= DRAG_S else f" ({behind:.2f} m of plank behind)"))
            go(t, "approach")
            return arm()
        return None

    return pilot
