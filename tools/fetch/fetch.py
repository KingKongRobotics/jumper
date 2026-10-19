"""Fetch: walk to a prop with the claw out, grasp it, carry it home, celebrate.

A state machine on top of the shipped controller, driving it through the same pad
the robot's operator would hold. The simulation's ground truth (prop poses, the
robot's pose) stands in for perception in this version.

Numbers below marked "measured" come from the probes beside this file, run on
2026-10-12 against `out/bundle_example/jumper.app` with `--task jumper.five_foot
--objects` on the warp backend (RTX 3060).
"""
from __future__ import annotations

import math
import os

import numpy as np

PROP = os.environ.get("FETCH_PROP", "can")      # which object of the row to fetch
HOME = (0.0, 0.0)          # where to carry it back to

# The claw at the web-up preset (d-pad left): the mouth sits here in the base frame
# and closes side to side, which is the only preset a standing object can be walked
# into (measured, probe_presets: thumb-up closes vertically, thumb-down points down).
MOUTH_AHEAD = 0.295
MOUTH_LEFT = 0.056
APERTURE = 0.073           # the open mouth, anvil to moving jaw (claw.py::APERTURE_MM at GRIPPER_OPEN)
GAP = 0.003                # the prop's near surface this far off the anvil across the mouth: the finger
                           # cannot shove the prop the rest of the way (14th run: 1.5 cm short, mouth 73 mm on a 58 mm can)
HOLD_DEPTH = 0.035         # the prop's centre this far inside the anvil. probe_clearance (2026-10-12): the
                           # open finger clears a standing can to 5 cm of depth and the shut one overlaps it
                           # most at 3-4 cm; at 0-1 cm the can sits between the jaw tips and the lift tips it out
STANDOFF = 0.05            # the far approach stops this short of the creep's start

# The policy's own stick response (measured, probe_speed): still below 0.35 of travel,
# 0.14 m/s at 0.4, 0.21 at 0.5, 0.26 at 0.6; the turn needs the stick past 0.5.
CREEP = 0.5                # 0.42 moved it in some runs and not others (11th run stalled nose-down)
WALK_MIN = 0.45
TURN = 0.75
TWIST_SIGN = -1.0          # the right stick right (+) twists and turns clockwise (probe_twist): to yaw +, push left
TWIST_GAIN = 8.0           # stick per metre of mouth: 0.3 of stick moved the mouth 7.4 cm (probe_twist), doubled for bite
TWIST_I = 6.0              # integral, per metre per second
HEADING_BIAS = -0.10       # rad: aim this far clockwise before the final creep, which yaws ccw by about that (mug runs 1-2)
NOSE_DOWN = float(os.environ.get("FETCH_NOSE", "0.5"))
SIP = os.environ.get("FETCH_SIP", "0") == "1"      # a drink from the mug after the lift, before the carry
SIP_TIP = int(os.environ.get("FETCH_SIP_TIP", "1"))  # which preset to blend in for the tip: +1 thumb down, -1 thumb up
LIFT = os.environ.get("FETCH_LIFT", "0") == "1"   # lift before the carry (drops the can: 13th-17th runs), or drag it           # right stick forward while approaching: pitch toward the floor
LIFT_S = 2.5               # seconds the nose takes to come up with the prop
LIFT_TO = 0.5              # how far up the nose comes for the carry: the jaw rotates with it
PRE = 0.06                 # the first creep stops this short of the prop's near face (anvil): room to turn
VEER = 0.3                 # the creep veers right by this fraction of its length, arm out (2.0/9.6, 4.3/9, 2.9/8 cm
                           # in beach, beach3, beach8): the standstill before it aims that far left of the line


def _quat_rot(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def _yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_pilot(env, player, prop: str = PROP, log=print):
    from controller import GamepadState
    from tasks.jumper.five_foot.claw import aperture_mm
    from tasks.jumper.five_foot.mdp.grasp import _closing_in_palm, _mouth_in_palm
    from tasks.jumper.five_foot.objects import SOLIDS

    robot = env.scene["robot"]
    palm = robot.body_names.index("LF_palm_link")
    tip = robot.body_names.index("LF_finger_tip_link")
    finger = robot.joint_names.index("LF_J4_joint")
    mouth_in_palm = _mouth_in_palm(env.sim.mj_model)
    closing_in_palm = _closing_in_palm(env.sim.mj_model)
    target_prop = env.scene[prop]
    if prop == "mug":
        from mug import FEATURE
        offset, half_x, half_y, hold = np.array(FEATURE["offset"]), FEATURE["half_x"], FEATURE["half_y"], FEATURE["hold"]
        gap = FEATURE.get("gap", GAP)
    else:
        solid = next(x for x in SOLIDS if x.name == prop)
        offset, hold, gap = np.zeros(2), HOLD_DEPTH, GAP
        if solid.kind == "cylinder":             # (radius, half-height): round across both
            half_x = half_y = solid.size[0]
        else:
            half_x, half_y = solid.size[0], solid.size[1]

    # How far each jaw reaches ahead of the anvil and across the mouth, from the
    # claw's own collision geometry at the open pose: what has to clear the prop.
    import mujoco

    from tasks.jumper.five_foot.mdp.grasp import _inward_in_palm, _jaw_clouds, _named, _posed
    _m = env.sim.mj_model
    _d = _posed(_m, -0.65)
    _moving, _fixed = _jaw_clouds(_m, _d)
    _palm_id = _named(_m, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    _rot = _d.xmat[_palm_id].reshape(3, 3)
    _anvil = _d.xpos[_palm_id] + _rot @ mouth_in_palm
    _fwd, _across = -(_rot @ _inward_in_palm(_m)), _rot @ closing_in_palm
    for _name, _cloud in (("fixed", _fixed), ("moving", _moving)):
        _rel = _cloud - _anvil
        log(f"[fetch] {_name} jaw: ahead of the anvil {np.dot(_rel, _fwd).max():+.3f} m, "
            f"across the mouth {np.dot(_rel, _across).min():+.3f}..{np.dot(_rel, _across).max():+.3f} m")

    st = {"name": "settle", "since": 0.0, "attempts": 0, "turn": 0, "turn_t0": 0.0, "turn_e0": 0.0,
          "target": None, "twist": 0.0, "twist_i": 0.0, "after_fine": "creep2",
          "stall_p": None, "stall_t": 0.0, "bump": 0.0, "recreeps": 0}

    def goto(name, t):
        log(f"\n[fetch] {t:6.2f}s  {st['name']} -> {name}")
        st["name"], st["since"], st["turn"] = name, t, 0

    def world():
        d = robot.data
        base_p = d.root_link_pos_w[0].cpu().numpy()
        base_q = d.root_link_quat_w[0].cpu().numpy()
        palm_p = d.body_link_pos_w[0, palm].cpu().numpy()
        palm_q = d.body_link_quat_w[0, palm].cpu().numpy()
        mouth = palm_p + _quat_rot(palm_q, mouth_in_palm)
        closing = _quat_rot(palm_q, closing_in_palm)
        return base_p, _yaw(base_q), mouth, float(d.joint_pos[0, finger]), closing

    def prop_pos():
        """The grasped feature's position: the prop's origin plus its offset, turned
        with the prop's own yaw."""
        p = target_prop.data.root_link_pos_w[0].cpu().numpy().copy()
        yaw = _yaw(target_prop.data.root_link_quat_w[0].cpu().numpy())
        c, sn = math.cos(yaw), math.sin(yaw)
        p[0] += c * offset[0] - sn * offset[1]
        p[1] += sn * offset[0] + c * offset[1]
        return p

    def prop_tilt_deg():
        q = target_prop.data.root_link_quat_w[0].cpu().numpy()
        up = _quat_rot(q, np.array([0.0, 0.0, 1.0]))
        return math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))

    def to_body(err_w, yaw):
        c, sn = math.cos(yaw), math.sin(yaw)
        return c * err_w[0] + sn * err_w[1], -sn * err_w[0] + c * err_w[1]

    def stick_mag(base_p, t, base=0.5):
        """A walking stick that grows while the base is not moving: the policy's dead
        band sits near 0.4 of travel and wanders with the posture (a 0.5 strafe stood
        still for 30 s in the 18th run)."""
        if st["stall_p"] is None or np.linalg.norm(base_p[:2] - st["stall_p"]) > 0.015:
            st["stall_p"], st["stall_t"] = base_p[:2].copy(), t
        elif t - st["stall_t"] > 1.0:
            st["bump"] = min(0.3, st["bump"] + 0.05)
            st["stall_t"] = t
            log(f"\n[fetch] stalled; stick now {base + st['bump']:.2f}")
        return base + st["bump"]

    def walk_to(err_w, yaw, s, tol=0.03, mag=0.5):
        """Stick deflections that walk the base by `err_w` (world xy) at heading `yaw`.

        One axis at a time: the policy stands still under a command below ~0.35 of
        travel on *each* axis (0.26 forward with 0.37 sideways did not move it, 8th
        run), so a proportional law stalls and a diagonal one stalls sooner. Strafe
        onto the line first, then walk along it with a small lateral trim.
        """
        ex, ey = to_body(err_w, yaw)
        if abs(ey) > 0.025:
            s["lx"] = -mag * (1 if ey > 0 else -1)         # Lx positive is right
            return False
        if abs(ex) > tol:
            s["ly"] = -mag * (1 if ex > 0 else -1)         # Ly positive is back
            s["lx"] = -max(-0.2, min(0.2, 8.0 * ey))
            return False
        return True

    def hold_heading(yaw, s, t, want=0.0, start=0.10, stop=0.03):
        """Pulse the turn whenever the heading error exceeds `start`, until under `stop`.

        The right stick's first half twists the body and only its second half turns,
        so there is no small continuous trim: 0.75 is where the twist has unwound and
        the turn runs at half rate. Measured standing (probe_twist, 2026-10-12): +0.75
        turns clockwise at ~30 deg/s, -0.75 counter-clockwise at ~70 deg/s.
        """
        e = _wrap(want - yaw)
        if st["turn"] == 0 and abs(e) > start:
            st["turn"] = 1 if e > 0 else -1
        elif st["turn"] != 0 and (abs(e) < stop or (e > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"] != 0:
            s["rx"] = TURN * st["turn"] * TWIST_SIGN
        return e

    def grip_target(closing):
        """Where the anvil goes: `DEPTH` past the prop's near face along x, and `GAP`
        off the prop's face on the anvil's side across y. The prop sits from the anvil
        along `closing`, so the anvil is on the far side of it from that direction."""
        p = prop_pos()
        side = -1.0 if closing[1] > 0 else 1.0
        return np.array([p[0] + hold, p[1] + side * (half_y + gap)])

    last_log = [-1.0]

    def creep_stick(base_p, t):
        """The creep's forward stick, raised while the base is not moving: the policy's
        dead band sits near 0.4 of travel and wanders with the posture."""
        if st["stall_p"] is None or np.linalg.norm(base_p[:2] - st["stall_p"]) > 0.015:
            st["stall_p"], st["stall_t"] = base_p[:2].copy(), t
        elif t - st["stall_t"] > 1.0:
            st["bump"] = min(0.25, st["bump"] + 0.05)
            st["stall_t"] = t
            log(f"\n[fetch] creep stalled; stick now {CREEP + st['bump']:.2f}")
        return -(CREEP + st["bump"])

    def pilot(t):
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        base_p, yaw, mouth, fin, closing = world()
        name, since = st["name"], t - st["since"]
        fsm = player.fsm

        if name == "settle":
            if since > 1.0:
                goto("enter_claw", t)
        elif name == "enter_claw":
            if since < 0.1:
                s["buttons"] = frozenset({"LB"})
            elif fsm.mode == "claw_left" and fsm.running_policy and since > 2.0:
                goto("approach", t)
        elif name == "approach":
            # Arm stowed: walk the base to where the creep starts, at the row's heading.
            p = prop_pos()
            standoff = np.array([p[0] - half_x - PRE - MOUTH_AHEAD - STANDOFF, p[1] - MOUTH_LEFT])
            there = walk_to(standoff - base_p[:2], yaw, s, mag=stick_mag(base_p, t))
            hold_heading(yaw, s, t)
            if there:
                st["bump"], st["stall_p"] = 0.0, None
                goto("arm_out", t)
        else:
            s["hat_x"] = -1                     # the arm out, web up, from here on
        if name in ("align", "creep", "fine", "creep2", "fine2", "grasp", "backoff"):
            # Level. Nose down (0.4-0.6) put the pinch at 4-5 cm, under the can's centre
            # of mass at 7.1: off the floor it swung about the pinch line and slid out of
            # the open-fronted wedge (13th-15th runs). Pinched above it, it hangs.
            s["ry"] = -NOSE_DOWN

        if name == "arm_out":
            if since > 1.5:
                goto("align", t)
        elif name == "align":
            e = hold_heading(yaw, s, t, start=0.05, stop=0.02)
            ey = grip_target(closing)[1] - mouth[1]
            if st["turn"] == 0 and abs(e) < 0.05:
                if abs(ey) > 0.008:
                    s["lx"] = -WALK_MIN * (1 if ey > 0 else -1)   # a strafe pulse toward the line
                elif since > 1.0:
                    # Frozen here: a target that follows the prop follows it when the
                    # claw pushes it, and the creep never ends.
                    st["target"] = grip_target(closing)
                    log(f"\n[fetch] closing axis in the world {np.round(closing, 3)}; "
                        f"grip target {np.round(st['target'], 3)}; mouth {np.round(mouth, 3)}")
                    goto("creep", t)
        elif name == "creep":
            err = st["target"] - mouth[:2]
            ex, ey = to_body(err, yaw)
            hold_heading(yaw, s, t, start=0.08, stop=0.03)
            if ex > hold + half_x + PRE + 0.01:   # the anvil PRE short of the near face
                # The gait veers right while walking with the arm out (-8 cm over 30 cm,
                # 3rd fetch run, 4 cm over 9 in beach3), and a trim under the dead band
                # does nothing about it: strafe back onto the line, one axis at a time.
                if abs(ey) > 0.02:
                    s["lx"] = -WALK_MIN * (1 if ey > 0 else -1)
                else:
                    s["ly"] = creep_stick(base_p, t)
                    s["lx"] = -max(-0.2, min(0.2, 8.0 * ey))
            else:
                st["twist"], st["twist_i"], st["bump"], st["stall_p"] = 0.0, 0.0, 0.0, None
                st["after_fine"], st["recreeps"] = "creep2", 0
                goto("fine", t)
        elif name == "creep2":
            # The last few centimetres, with the twist frozen where the standstill left
            # it: a twist command *while walking* makes the policy turn to put its feet
            # back under the body, so a twist loop closed on the mouth during a creep
            # yawed the robot 13 degrees and saturated (mug run 1).
            ex, ey = to_body(st["target"] - mouth[:2], yaw)
            s["rx"] = st["twist"]
            if ex > 0.03:
                hold_heading(yaw, s, t, start=0.08, stop=0.03)
            if ex > -0.005:
                # Past the line, not short of it: a can seated at the anvil's own depth
                # is squeezed straight out of the wedge (12th run; grasp.py::HOLD_DEPTH).
                s["ly"] = creep_stick(base_p, t)
            else:
                # Once more at depth, standing: the short creep still wanders a centimetre
                # or two, and the finger pushes the prop over by that much on the squeeze.
                st["after_fine"] = "grasp"
                st["twist"], st["twist_i"] = 0.0, 0.0     # from the pose the walk left, not the one before it
                goto("fine2", t)
        elif name in ("fine", "fine2"):
            # Standing still, swing the mouth onto its line with the body twist: the
            # first half of the right stick, a continuous command the gait's steps are
            # not. The mouth is 0.3 m out, so 1 rad of twist is ~0.3 m of mouth; a full
            # 0.5 of stick is 30 deg standing, hence ~3.2 stick per metre of error.
            if name == "fine2":
                st["target"] = grip_target(closing)      # live: the prop may have been nudged
            ex, ey = to_body(st["target"] - mouth[:2], yaw)
            if name == "fine":
                ey += VEER * max(0.0, ex)           # the creep to come drifts right; start left of the line by as much
                # Heading first, while the jaw is still clear of the prop: the final creep
                # yaws the body 5-15 degrees ccw, and a twist cannot hide that much.
                e = hold_heading(yaw, s, t, want=HEADING_BIAS, start=0.04, stop=0.015)
                if st["turn"] != 0 or abs(e) > 0.04 or since < 0.3:
                    return GamepadState(**s)
            if name == "fine" and abs(ey) > 0.04:
                # Beyond what a twist can reach (0.45 of stick is ~11 cm of mouth, and it
                # saturated 9 cm off the line in beach3): a strafe pulse, with no twist,
                # since a twist while walking turns the body instead.
                st["twist"], st["twist_i"] = 0.0, 0.0
                s["lx"] = -WALK_MIN * (1 if ey > 0 else -1)
                return GamepadState(**s)
            st["twist_i"] += TWIST_I * ey * (1.0 / 200.0)
            st["twist"] = max(-0.45, min(0.45, TWIST_SIGN * (TWIST_GAIN * ey + st["twist_i"])))
            s["rx"] = st["twist"]
            if name == "fine2" and since > 0.8 and ex > 0.02 and st["recreeps"] < 2:
                log(f"\n[fetch] settled {ex * 100:.1f} cm short of the depth; creeping again")
                st["bump"], st["stall_p"] = 0.0, None
                st["recreeps"] += 1
                goto("creep2", t)
            elif since > 0.8 and abs(ey) < 0.004 and (name == "fine" or ex < 0.012):
                goto(st["after_fine"], t)
            elif since > 6.0 or (name == "fine2" and since > 0.8 and ex >= 0.012 and st["recreeps"] >= 2):
                if name == "fine2" and (abs(ey) > 0.015 or ex >= 0.012) and st["attempts"] < 3:
                    # Off the bar by more than the finger's sweep forgives: a grasp here
                    # closes beside it (beach3, twice). Back off and come in again.
                    log(f"\n[fetch] not on the bar (ey={ey:+.3f}, ex={ex:+.3f}); backing off, attempt {st['attempts'] + 1}")
                    st["attempts"] += 1
                    st["recreeps"] = 0
                    goto("backoff", t)
                else:
                    log(f"\n[fetch] fine alignment gave up at ey={ey:+.3f}, ex={ex:+.3f}; going on anyway")
                    goto(st["after_fine"], t)
        elif name == "grasp":
            s["rx"] = st["twist"]
            s["lt"] = min(1.0, since / 0.6)
            if since > 1.8:
                # The finger stops where the prop stops it: the mouth should be open by
                # about the prop's thickness, not shut on a corner.
                # A finger stopped well wider than the prop is resting on a corner or a
                # rim, and such a hold lets go two centimetres into the carry (20th run).
                mouth_mm, want_mm = aperture_mm(fin), 2000.0 * half_y
                if fin < 0.09 and 0.5 * want_mm < mouth_mm <= want_mm + 12.0:
                    log(f"\n[fetch] grasped {prop}: finger at {fin:+.3f}, mouth {mouth_mm:.0f} mm for a {want_mm:.0f} mm prop")
                    goto("withdraw", t)
                elif st["attempts"] >= 3:
                    log(f"\n[fetch] third miss (finger {fin:+.3f}, mouth {mouth_mm:.0f} mm of {want_mm:.0f}); carrying on with what it has")
                    goto("withdraw", t)
                else:
                    st["attempts"] += 1
                    log(f"\n[fetch] missed (finger {fin:+.3f}, mouth {mouth_mm:.0f} mm of {want_mm:.0f}); "
                        f"backing off, attempt {st['attempts']}")
                    goto("backoff", t)
        elif name == "backoff":
            s["ly"] = +0.5
            if since > 1.0:
                goto("align", t)
        elif name == "withdraw":
            s["lt"] = 1.0
            s["rx"] = st["twist"]
            st["bump"], st["stall_p"] = 0.0, None
            if LIFT:
                s["ry"] = -NOSE_DOWN + (LIFT_TO + NOSE_DOWN) * min(1.0, since / LIFT_S)   # nose up, slowly: the item rises
            else:
                s["ry"] = -NOSE_DOWN                                                     # keep it on the floor: dragged, not lifted
            if since > (LIFT_S + 0.5 if LIFT else 0.5):
                goto("sip" if SIP and LIFT else "carry", t)
        elif name == "sip":
            # The drink: a look left and right (nobody watching?), the nose up to meet
            # the mug, a second preset blended in (web-up + thumb-down = the hand rolled
            # 45 degrees, rl-wbc-fsm V3.1's blend) so the mug tips toward the face,
            # held, and back. The pinch is friction: whether the mug stays is measured
            # by the log's tilt and finger, not assumed.
            s["lt"] = 1.0
            s["ry"] = LIFT_TO
            if since < 0.8:
                s["rx"] = -0.35                     # look left
            elif since < 1.6:
                s["rx"] = +0.35                     # look right
            elif since < 2.2:
                s["ry"] = LIFT_TO + (0.9 - LIFT_TO) * (since - 1.6) / 0.6
            elif since < 4.7:
                s["ry"] = 0.9
                s["hat_y"] = SIP_TIP                # tip: held with the web-up preset, the two blend
            elif since < 5.7:
                s["ry"] = 0.9 - (0.9 - LIFT_TO) * (since - 4.7) / 1.0
            else:
                goto("carry", t)
        elif name == "carry":
            s["lt"] = 1.0
            s["ry"] = LIFT_TO if LIFT else -NOSE_DOWN
            st["twist"] *= 0.995                     # let the twist go over the first second
            s["rx"] = st["twist"]
            if LIFT:
                there = walk_to(np.array(HOME) - base_p[:2], yaw, s, tol=0.05, mag=stick_mag(base_p, t))
                hold_heading(yaw, s, t)
            else:
                # Dragged along the floor, the prop stays in the open-fronted mouth only
                # while the pull is straight back: a strafe or a turn pulse at the start of
                # the carry slid it out sideways (19th run). So: straight back, no steering.
                s["ly"] = stick_mag(base_p, t)
                there = base_p[0] < HOME[0] + 0.05
            if there:
                goto("victory", t)
        elif name == "victory":
            s["lt"] = 1.0
            phase = since * 2 * math.pi / 1.6
            if since < 5.0:
                # The wiggle, with the prize in the jaws: R3 held, the right stick rolls.
                s["ry"] = LIFT_TO if LIFT else -NOSE_DOWN
                s["buttons"] = frozenset({"R3"})
                s["rx"] = 0.8 * math.sin(phase)
            elif since < 9.0:
                # The finale: the nose comes up and the prize with it, as far as the pinch allows.
                s["ry"] = min(1.0, (LIFT_TO if LIFT else -NOSE_DOWN) + (since - 5.0) / 3.0)
            else:
                goto("done", t)
        elif name == "done":
            s["lt"], s["ry"] = 1.0, 1.0          # hold the toast: letting go drops the prize on camera

        if t - last_log[0] >= 0.5:
            last_log[0] = t
            p = prop_pos()
            log(f"\n[fetch] t={t:6.2f} {name:<10} base=({base_p[0]:+.3f},{base_p[1]:+.3f},{base_p[2]:.3f}) "
                f"yaw={math.degrees(yaw):+5.1f} mouth=({mouth[0]:+.3f},{mouth[1]:+.3f},{mouth[2]:.3f}) "
                f"{prop}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f},tilt {prop_tilt_deg():.0f}) finger={fin:+.3f} tip_z={float(robot.data.body_link_pos_w[0, tip, 2]):.3f} "
                f"ly={s['ly']:+.2f} lx={s['lx']:+.2f} rx={s['rx']:+.2f} ry={s['ry']:+.2f}")
        return GamepadState(**s)

    return pilot
