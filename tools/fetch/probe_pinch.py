"""Where the pinch lands against the mouth's height over the bar.

The handle loop's window is 36 mm tall inside (LOOP_HALF_Z 0.022 less a bar) and the
jaw beam 35 mm: the beam only enters the loop when the mouth sits within a millimetre
of the outer bar's mid-height, and otherwise the finger shuts on the bar held off the
beam -- a wide "mouth" reading (12-19 mm on an 8 mm bar) that slips out on the walk.
This stands the mug exactly where the creep leaves it (bar 31 mm behind the mouth,
29 mm to its right, the logs' means), closes at a row of nose settings, lifts, walks
5 s and logs the mouth reading, the hang tilt and whether the mug is still there.

    MUJOCO_GL=glfw .venv/bin/python tools/fetch/run.py --pilot tools/fetch/probe_pinch.py \
        --task jumper.five_foot --mug --scene beach --headless --steps 21000
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from fetch import _quat_rot, _yaw
from mug import BODY_HALF_H, FEATURE
from sense import Senses

from tasks.jumper.five_foot.claw import aperture_mm

NOSES = [float(v) for v in os.environ.get("PINCH_NOSES", "-0.40,-0.50,-0.55,-0.60,-0.65,-0.70,-0.80").split(",")]
DEPTH = float(os.environ.get("PINCH_DEPTH", "0.031"))     # the bar this far behind the mouth point at the close
SIDE = float(os.environ.get("PINCH_SIDE", "0.029"))       # and this far to its right (the finger's side)
YAWS = [math.radians(float(v)) for v in os.environ.get("PINCH_YAWS", "0").split(",")]   # the mug turned on its spot
SEQ = os.environ.get("PINCH_SEQ", "short")
WALK_RY = float(os.environ.get("PINCH_WALK_RY", "0.0"))    # the nose during the walks (0.0 is the film's level carry)
WALK_LY = float(os.environ.get("PINCH_WALK_LY", "-0.5"))   # the walk stick
STRAFE_RY = float(os.environ.get("PINCH_STRAFE_RY", "0.2"))  # the nose during the sidestep (NAV_STRAFE_NOSE)
TURN_S = float(os.environ.get("PINCH_TURN_S", "0.0"))      # a heading pulse (rx NAV_TURN 0.75) this long before the strafe                 # short: lift to 0, walk 5 s. film: the carry as auto.py does it
T0 = 3.0
# The film's carry: the lift to LIFT_TO 0.5 with its 1.5 s settle, the nose to level over
# 1.5 s, the look's twists, the room walk, the sidestep at nose 0.2, the walk on.
FILM = [("close", 2.0, dict(ry="nose", lt=1.0)), ("lift", 1.5, dict(ry="nose->0.5", lt=1.0)), ("settle", 1.5, dict(ry=0.5, lt=1.0)),
        ("level", 1.5, dict(ry="0.5->0", lt=1.0)), ("twist-", 1.0, dict(rx=-0.25, lt=1.0)), ("twist+", 1.0, dict(rx=0.25, lt=1.0)),
        ("walk", 8.0, dict(ly=WALK_LY, ry=WALK_RY, lt=1.0)), ("turn", TURN_S, dict(rx=0.75, lt=1.0)),
        ("strafe", 3.0, dict(lx=0.5, ry=STRAFE_RY, lt=1.0)),
        ("walk2", 8.0, dict(ly=WALK_LY, ry=WALK_RY, lt=1.0)),
        ("open", 1.5, dict(lt=0.0))]
SHORT = [("close", 2.0, dict(ry="nose", lt=1.0)), ("lift", 1.5, dict(ry="nose->0", lt=1.0)), ("hold", 1.5, dict(lt=1.0)),
         ("walk", 5.0, dict(ly=-0.5, lt=1.0)), ("open", 1.5, dict(lt=0.0))]
# The set-down and the way out of the handle (PINCH_RETREAT): the mug lowered to the
# tray's set height, released, then the retreat variant; the tilt 3 s later says whether
# the open jaw knocked it (the "down" retreat tipped the placed mug in w63/w68/film43).
RETREAT = os.environ.get("PINCH_RETREAT", "down")
_OUT = {"down": [("nose_down", 0.8, dict(ry="-0.2->-0.75", lt=0.0)), ("back_down", 1.5, dict(ry=-0.75, ly=0.5, lt=0.0)),
                 ("nose_up", 1.5, dict(ry="-0.75->0.3", ly=0.5, lt=0.0))],
        "up": [("nose_up", 1.5, dict(ry="-0.2->0.25", lt=0.0)), ("back_up", 2.0, dict(ry=0.25, ly=0.5, lt=0.0))],
        "level": [("back_level", 2.5, dict(ry=-0.2, ly=0.5, lt=0.0))],
        "stow": [("stow", 1.0, dict(ry=-0.2, lt=0.0, buttons="LB")), ("back_stowed", 2.0, dict(ly=0.5, lt=0.0, hat_x=0))]}
WALK_S = float(os.environ.get("PINCH_WALK_S", "3.0"))      # the carry before the set-down (0 with PINCH_MAT: stay over the mat)
MAT = float(os.environ.get("PINCH_MAT", "0"))               # the mug stands on a mat this tall (run.py --room puts one at ROOM_TRAY)
RETREAT_SEQ = [("close", 2.0, dict(ry="nose", lt=1.0)), ("lift", 1.5, dict(ry="nose->0", lt=1.0)), ("hold", 1.0, dict(lt=1.0)),
               ("walk", WALK_S, dict(ly=-0.5, lt=1.0)), ("settle", 1.5, dict(lt=1.0)),
               ("lower", 2.0, dict(ry="0->-0.35", lt=1.0)), ("relift", 0.5, dict(ry=-0.2, lt=1.0)),
               ("release", 1.0, dict(ry=-0.2, lt=0.0))] + _OUT[RETREAT] + [("after", 3.0, dict(lt=0.0))]
# The drink and a remedy for the hang it leaves (PINCH_DIP=1: a nose-down dip after it, so
# the mug can swing back and re-seat on the jaw before the carry).
DIP = os.environ.get("PINCH_DIP", "0") == "1"
SIP_SEQ = [("close", 2.0, dict(ry="nose", lt=1.0)), ("lift", 1.5, dict(ry="nose->0.5", lt=1.0)), ("settle", 1.5, dict(ry=0.5, lt=1.0)),
           ("fold", 2.5, dict(ry=0.55, lt=1.0)), ("head", 2.2, dict(ry=0.6, lt=1.0)), ("tip", 1.6, dict(ry=1.0, lt=1.0)),
           ("back", 2.0, dict(ry=0.6, lt=1.0)), ("unfold", 2.5, dict(ry="0.55->0", lt=1.0))] + \
          ([("dip", 1.5, dict(ry=-0.3, lt=1.0)), ("level", 1.5, dict(ry=0.0, lt=1.0))] if DIP else [("level", 3.0, dict(ry=0.0, lt=1.0))]) + \
          [("walk", 6.0, dict(ly=-0.5, lt=1.0)), ("open", 1.0, dict(lt=0.0))]
PHASES = {"film": FILM, "retreat": RETREAT_SEQ, "sip": SIP_SEQ}.get(SEQ, SHORT)
TRIAL_S = 1.5 + sum(d for _, d, _ in PHASES)
TRIALS = [(n, y) for n in NOSES for y in YAWS]


def make_pilot(env, player):
    from controller import GamepadState

    senses = Senses(env)
    robot = env.scene["robot"]
    mug = env.scene["mug"]
    offset = np.array(FEATURE["offset"])
    done = set()

    def base():
        return robot.data.root_link_pos_w[0].cpu().numpy(), robot.data.root_link_quat_w[0].cpu().numpy()

    def mug_truth():
        bp, bq = base()
        mp = mug.data.root_link_pos_w[0].cpu().numpy(); mq = mug.data.root_link_quat_w[0].cpu().numpy()
        bar_w = mp + _quat_rot(mq, np.array([offset[0], offset[1], 0.0]))
        w, x, y, z = bq
        bar_b = _quat_rot(np.array([w, -x, -y, -z]), bar_w - bp)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(_quat_rot(mq, np.array([0.0, 0.0, 1.0]))[2])))))
        return bar_b, tilt, float(mp[2])

    def place(turn):
        mouth, closing, inward, fin = senses.body()
        bp, bq = base()
        yaw = _yaw(bq) + turn
        bar_b = mouth + np.array([-DEPTH, -SIDE, 0.0])
        bar_w = bp + _quat_rot(bq, bar_b)
        c, s_ = math.cos(yaw), math.sin(yaw)
        centre = bar_w - np.array([c * offset[0] - s_ * offset[1], s_ * offset[0] + c * offset[1], 0.0])
        pose = torch.tensor([[centre[0], centre[1], BODY_HALF_H + MAT, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                            dtype=torch.float32, device=env.device)
        mug.write_root_link_pose_to_sim(pose)
        mug.write_root_link_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))
        mw = bp + _quat_rot(bq, mouth)
        return mw[2] - BODY_HALF_H

    def pilot(t):
        senses.update(t)
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t < T0:
            return GamepadState(**s)
        s["hat_x"] = -1
        k = int((t - T0) / TRIAL_S)
        if k >= len(TRIALS):
            s["ry"] = 0.0
            return GamepadState(**s)
        nose, turn = TRIALS[k]
        u = (t - T0) - k * TRIAL_S
        mouth, closing, inward, fin = senses.body()
        if u < 1.5:
            s["ry"], s["lt"] = nose, 0.0
            if u >= 1.4 and (k, "place") not in done:
                done.add((k, "place"))
                dz = place(turn)
                print(f"\n[pinch] trial {k} nose {nose:+.2f} mug yaw {math.degrees(turn):+.0f}: mouth {dz * 1000:+.1f} mm above the bar")
            return GamepadState(**s)
        u -= 1.5
        for name, dur, cmd in PHASES:
            if u < dur:
                break
            u -= dur
        else:
            return GamepadState(**s)
        if dur <= 0.0:
            return GamepadState(**s)
        for key, val in cmd.items():
            if key == "buttons":
                if u < 0.15:
                    s["buttons"] = frozenset({val})
                continue
            if isinstance(val, str):
                a, _, b = val.partition("->")
                a = nose if a == "nose" else float(a)
                val = a if not b else a + (float(b) - a) * min(1.0, u / dur)
            s[key] = val
        if name.startswith("walk") or name == "strafe":
            pass
        if name in ("walk", "walk2", "strafe", "turn") and int(u / 2.0) != int((u - senses.dt) / 2.0):
            bar_b, tilt, mz = mug_truth()
            print(f"\n[pinch]   {name} {u:4.1f} s: bar {(mouth - bar_b)[0] * 1000:+3.0f} behind, tilt {tilt:3.0f}, finger {fin:+.3f}, "
                  f"beam pitch {math.degrees(math.asin(max(-1.0, min(1.0, -float(inward[2]))))):+.1f} deg")
        if u >= dur - 0.1 and (k, name) not in done:
            done.add((k, name))
            bar_b, tilt, mz = mug_truth()
            d = mouth - bar_b
            held = d[0] > 0.0 and tilt < 60 and abs(d[2]) < 0.06      # (not the mug's z: the sand undulates)
            if SEQ == "retreat" and name == "after":
                print(f"\n[pinch] trial {k} RETREAT {RETREAT}: the mug {'STANDING' if tilt < 10 else 'TIPPED'} "
                      f"(tilt {tilt:.0f}) {bar_b[0]:.2f} m ahead after the retreat")
            print(f"\n[pinch] trial {k} after {name:7s}: finger {fin:+.3f} mouth {aperture_mm(fin):4.1f} mm; bar {d[0] * 1000:+3.0f} behind "
                  f"{-d[1] * 1000:+3.0f} right {-d[2] * 1000:+3.0f} under; tilt {tilt:3.0f} mug_z {mz:.3f}"
                  + ("" if name in ("close", "open") else f" -> {'held' if held else 'LOST'}"))
        return GamepadState(**s)

    return pilot
