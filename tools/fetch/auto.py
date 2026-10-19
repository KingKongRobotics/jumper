"""The autonomous fetch: the same pad-driven machine as fetch.py, fed by the robot's
own senses (sense.py) and nothing else.

Everything is in the base frame. The mug is where the dToF says it is; the claw's
mouth is where the encoders say it is; heading is the gyro integrated from the
moment the handle was sighted; distance walked is not measured at all -- the walk
is closed on what the sensor sees, and the carry home is a timed walk back. The
numbers that make the grasp work (hold, gap, the veer, the twist gains) are the
ones the ground-truth runs measured; see fetch.py for how each was learnt.

Ground truth appears in one place: the log, when AUTO_TRUTH=1, prints the true bar
beside the sensed one so a run can be judged. No decision reads it.

    FETCH_LIFT=1 FETCH_NOSE=0.6 AUTO_SIP=1 python tools/fetch/run.py --pilot tools/fetch/auto.py \\
        --task jumper.five_foot --mug --scene beach --headless --steps 8000 --video ...
"""
from __future__ import annotations

import math
import os

import numpy as np
from fetch import (
    CREEP,
    _yaw,
    HEADING_BIAS,
    LIFT,
    LIFT_S,
    LIFT_TO,
    MOUTH_AHEAD,
    MOUTH_LEFT,
    NOSE_DOWN,
    PRE,
    STANDOFF,
    TURN,
    TWIST_GAIN,
    TWIST_I,
    TWIST_SIGN,
    WALK_MIN,
    _quat_rot,
    _wrap,
    _yaw,
)
from mug import FEATURE
from sense import HALF_X, Senses

from tasks.jumper.five_foot.claw import aperture_mm

SIP = os.environ.get("AUTO_SIP", "0") == "1"
WANDER = os.environ.get("AUTO_WANDER", "0") == "1"    # start lost: a serpentine walk with look-outs until the mug is sighted
STOP_AT = os.environ.get("AUTO_STOP", "")             # end the take on entering this state (a segment's last frame)
START = os.environ.get("AUTO_START", "")              # "carry": the mug set into the jaw by hand at the start and the
                                                      # delivery straight away -- the room walk, the tray and the placing
                                                      # tested on their own (Andrew, 2026-10-14: "test just navigating and
                                                      # placing... not once has it arrived successfully")
MAP_PRIOR = os.environ.get("AUTO_MAP_PRIOR", "1") == "1"   # the staged scene's known points (the tray's flag, the mug) on the
                                                      # map before the first step, the pillars and the scenery the unknowns;
                                                      # a sighting of the flag then corrects the ODOMETRY (mapper.fix), as
                                                      # a dock does. "0": the flag is found, not known (film36-72)
LOCO_SCALE = float(os.environ.get("AUTO_LOCO_SCALE", "1.4"))   # the locomotion mode's speed per stick unit against the
                                                      # claw mode's (the mapper's SPEED_FWD was measured in the claw mode):
                                                      # at 1.0 the wander's reckoning fell 34-89 cm behind the truth over
                                                      # 2-3 m (film73/74, truly 3.11 m for 2.53 reckoned); the mug, sighted,
                                                      # then put the odometry right to 1 cm before the grasp
WANDER_RANGE = 2.0                                    # the mug counts as sighted inside this range (sense.py's 1.5 m is for the grasp)
WANDER_SWING = math.radians(22)                     # the serpentine swings this far off the line each way, on the gyro
WANDER_TURN = 0.54                                  # the turn stick on a leg (0.5-1.0 is the turn band): 0.62 spun 60 deg/s (wander1)
MUG_HEIGHT = (0.05, 0.18)                           # what counts as mug-sized when sighted: a palm trunk is metres tall and a
MUG_WIDTH = 0.25                                    # rock is wider; wander1 jumped at a palm
LOOK_EVERY = 3                                      # a look-out (the salute: claw to the brow) after this many swings
SIGHT_HOLD_S = 0.2                                  # a sighting must persist this long (20 exposures) before the jump
RUN_STICK = 0.6                                     # the run-in once the mug is sighted (0.8 overshot it once, wander5)
RUN_STOP = 0.9                                      # (0.7 for the film knocked the mug over on the run-in's stop, film19/23)                                     # the run-in ends with the bar this near, where the fetch begins
DANCE_RANGE = 0.55                                  # beat 3: creep from the run-in's stop to here before the dance (the
                                                    # Zoidberg sways in place; the run-in itself cannot stop nearer than
                                                    # 0.9 without bowling the mug over, film19/23)
RUN_STEER = 0.62                                    # a gentle turn while running: the turn band is 0.5-1.0 of the stick
FACE = os.environ.get("AUTO_FACE", "1") == "1"      # the sip with the arm-to-face preset (d-pad right, the hook's fourth
                                                    # preset, 2026-10-12), else the older hand roll
TRUTH = os.environ.get("AUTO_TRUTH", "1") == "1"    # log the true bar beside the sensed one (never used to decide ...)
TRUTH_CLOSE = os.environ.get("AUTO_TRUTH_CLOSE", "0") == "1"   # ... except, with this, for the last centimetres of the
                                                    # pinch (creep2/fine2): there the jaw hides the handle from the sensor
                                                    # and the assumed bar sits 3+ cm off in 54 of 60 frames; the sensor
                                                    # keeps the search, approach, carry, room and tray. A measured
                                                    # alternative to more alignment rules (2026-10-13 23:50)
MAP_DIR = os.environ.get("AUTO_MAP", "")             # the occupancy map (tools/fetch/mapper.py): "1" builds it, a directory
                                                    # also writes its picture there once a second
PLAN_HOLD_S = 3.0                                   # a plan is followed this long before it is re-made (w75 flip-flopped)
MAP_DT = 0.2                                        # the map's picture every 0.2 s (compose.py lays it at 5 fps)
PIP_DIR = os.environ.get("AUTO_PIP", "")             # write the sensor's view (range / self / mug) here at 30 Hz, for a picture-in-picture
PIP_FPS = 30.0
SEARCH_TURN = 0.75                                  # turning in place while nothing is in view
AXIS_MAX = math.radians(20)                         # the handle's axis this far off the nose is approached head-on (the
                                                    # heading hold + strafe-first cover it); further, circle to the handle side
HIDDEN_S = 1.0                                      # the mug in view this long with no handle protruding: it is side-on, circle
HANDLE_RANGE = 0.6                                  # the 8 mm bar shows (2 px nearer than the cup) only inside this range: at
                                                    # 0.85 m it read "assumed" from dead ahead (orbA) and the search circled for ever
ORBIT_RANGE = (0.42, HANDLE_RANGE)                  # keep this far from the cup while circling it: near enough to see the bar
ORBIT_S = 25.0                                      # give up circling after this long and approach as is
CARRY_S = 1.6                                       # the timed walk back with the prize
WIGGLE = 0.4                                        # roll stick amplitude of the victory wiggle (6 deg each way)
TOAST_NOSE = 0.8                                    # the toast's nose-up; 1.0 left some mugs hanging 50 deg into the set-down (auto40)
SIP_FOLD_NOSE = 0.55                                # the nose while the arm folds and unfolds: folded, the mouth sits 4 cm under the
                                                    # base, and level the mug's bottom dragged on the sand and the unfold pulled it
                                                    # out of the pinch (auto30/32/35); at 7 deg the sagging cup still touched
                                                    # (rec8); 11 deg lifts the bar to ~0.10 m, a centimetre clear at a 30 deg sag
SIP_NOSE = 0.6                                      # the head back for the drink, folded: 10 deg. More turns the mug about its
                                                    # handle (two thirds of a pitch, folded) and it swings sideways (auto6, auto20-22)
SIP_HOLD_S = 2.2                                    # the drink itself
SIP_MAX_EX = 0.006                                  # the drink needs the bar >= 20 mm behind the mouth at the close (ex <= hold - 0.020)
SIP_MAX_MM = 15.0                                   # a pinch wider than this on an 8 mm bar is shallow, and the drink is skipped
                                                    # (18 until 2026-10-13 23:30: of 45 drinks, pinches reading 16-18 mm
                                                    # came out hanging 30-62 deg in 9 of 15, and film39's at 17 mm went
                                                    # to 75 and down on its side on the tray; at <= 15 mm the median
                                                    # hang after the drink was 10 deg)
DELIVER = os.environ.get("AUTO_DELIVER", "1") == "1"  # after the toast: set the mug down, let go, step back, bow
DANCE = os.environ.get("AUTO_DANCE", "gesture_zoidberg")    # the excitement on sighting the mug; "" for none
DANCE_S = 11.0                                      # how long the dance runs before Menu ends it (the Zoidberg clip is 10.9 s
                                                    # and ends by itself; Menu then does nothing)
BACK_OFF_S = 1.5                                    # the step back before the dance, ~0.3 m
DANCES = {"dance_crab": ({"menu"}, 0, -1), "dance_brazilian": ({"menu"}, 0, +1), "dance_maze": ({"menu"}, -1, 0),
          "dance_dream_wings": ({"menu"}, +1, 0), "gesture_zoidberg": ({"X"}, 0, 0)}   # (buttons, hat_x, hat_y)
ENCORE = os.environ.get("AUTO_ENCORE", "1") == "1"  # after the bow: beckon with each arm, a circle on the gyro, sit down
BEAT = int(os.environ.get("AUTO_BEAT", "1"))       # 2: wake from the sleep, fetch the mug again, carry it through the room
                                                    # (run.py --room) and set it on the tray; 3: the single shot -- asleep,
                                                    # wake and look about, wander, sight, jump, run, dance, grasp, sip, look,
                                                    # carry to the tray, place, step back, spin, bow, sleep
SIGHT_MIN_PX = 10                                   # beat 3: the mug counts as seen when it fills this many pixels (a clear
                                                    # shape, not a speck: ~1.6 m for the 8 x 10 cm mug)
SIGHT_RANGE = 1.8
LOOK_ROLL = 0.95                                    # the wake's look about: the body rolled side to side (R3 + right stick)
LOOK_TWIST = 0.25                                   # the look after the sip: a twist each way with the mug level
ALERT_S = 12.0                                      # seconds into the sleep gesture the alert comes (asleep from ~4 to ~16 s)
NAV_STICK = 0.5                                     # the walk through the room with the mug
NAV_NEAR = 0.40                                     # the carried mug hangs 0.25-0.35 m ahead: the scan starts past it
NAV_FREE = 1.0                                      # a column with nothing nearer than this is free
NAV_GAP_COLS = 14                                   # free columns side by side that make a gap: ~14 deg, the body's 0.3 m at 1.2 m
NAV_CLEAR = 0.28                                    # metres of clearance to a gap's edge: half the body plus the hanging mug
                                                    # (a 6 deg margin put the crab's side into a pillar at 0.9 m, beat2e)
NAV_NOSE = 0.0 
NAV_DODGE_S = 2.0                                   # the sidestep's least length: 1 s wedged the crab on the pillar (w34)
NAV_DODGE_MAX = 5.0                                 # and its most: past this the walk and the heading hold take over
NAV_PATH_HALF = 0.36                                # half the width "in the path" is judged over: half the body plus the
                                                    # pillar's half plus clearance (w34 caught it at 0.36 with a 1 s dodge;
                                                    # at 0.45 the pillar being PASSED at 0.4-0.55 m read as in the path
                                                    # again and again, each sidestep turning the body ~15 deg, w58)
NAV_DODGE_FAR = 0.6                                 # a new sidestep is for something still this far out, or dead ahead
NAV_STRAFE_NOSE = 0.2                               # the nose up a touch for a sidestep: level, the mug's base is a centimetre
                                                    # off the sand and a sidestep dragged it out of the pinch (w1/w2); at 0.2
                                                    # the strafe test kept it for 8 s                                     # the carry through the room with the nose LEVEL: turning in place at 0.75
                                                    # for 8 s (500 deg) left the mug hanging 25-33 deg off and in the pinch;
                                                    # with the nose at LIFT_TO 0.5 it fell at the first pulse (turn test,
                                                    # 2026-10-13). Level, the mug settles at ~25 deg and STAYS (the handle's
                                                    # lower bar rests on the finger); at 0.25 it never settled and ratcheted
                                                    # to 40-90 deg in 17 s of plain walking (t_carry, film7-29). Its base then
                                                    # skims the sand, so the nose comes up to TRAY_LIFT_NOSE for the last
                                                    # TRAY_LIFT_AT metres only
NAV_ARC = 0.45                                      # the twist held while walking bends the walk (fine stages avoid it; here it
                                                    # is the steering): probe_arc.py 2026-10-13, walking at 0.5 with the nose up:
                                                    # 0.3 -> 3 deg/s, 0.4 -> 4 deg/s, and 0.6 is the turn band, 20 deg/s the
                                                    # other way. Turning in place at 0.75 swung the hanging mug out of the pinch
                                                    # in three seconds (beat2l), the 0.6 arc did the same (beat2o): gentle only.
                                                    # Which way a twist bends the walk depends on the posture (the probe and
                                                    # beat2o disagreed), so the sign is learned from the gyro as it goes.
NAV_TURN = 0.75                                     # the turn stick: 0.62 did not turn the crab at all with the mug (beat2d, 90 s); the search's 0.75 does
FLAG_WIDTH, FLAG_HEIGHT = 0.06, 0.28                # the flag pole: thinner and taller than this (room.py: 0.024 x 0.6;
                                                    # the pillars are 0.15 wide). The tray is found by its flag, not its slab
FLAG_BACK = 0.22                                    # the pole stands this far BEHIND the mug's spot (room.py MUG_SPOT_BACK)...
FLAG_SIDE = -0.12                                   # ...12 cm to the right of the mug's spot (room.py MUG_SPOT_SIDE: the
                                                    # pole stands in the middle of the 60 cm mat, the mug lands left of it)
FLAG_RANGE = 2.5                                    # the scan reaches this far for the flag (1.6 for the pillars)
FLAG_LINE_HALF = 1.2                                # the flag stands within this of the room's line (the map's x axis):
                                                    # the real one marked at y -0.03 (film68), the palm trunks taken for
                                                    # it at +2.0 and +3.6 (film70); the scenery keeps 2.2 m off the line
FLAG_GATE_M = 0.6                                   # with the flag KNOWN (MAP_PRIOR) a candidate counts only this close to
                                                    # where the map expects it: the odometry's error budget. 1.5 let the
                                                    # pillar 2 m ahead in for the pole 3.1 m ahead and the fix pulled the
                                                    # pose 0.6 m wrong (carry1); the real pole, seen next, put it right
FLAG_HOLD_M = 0.35                                  # a candidate whose top is out of the picture counts only this close
                                                    # to where the flag is already marked: a pillar at 0.47 m (film71)
                                                    # and a palm trunk at 0.45 m (film70) are thin strips 0.3 m tall
                                                    # too, and each got the mug set down at its foot
TRAY_NOSE = 0.6                                     # the nose up for the final approach: the mouth 0.158 m up, the mug's
                                                    # base ~0.10 (probe_hang 2026-10-13: ry +0.6 -> 0.158, 0.0 -> 0.107,
                                                    # -0.4 -> 0.072, -1.0 -> 0.028; R3 + stick did not change the height)
TRAY_LIFT_NOSE, TRAY_LIFT_AT = 0.0, -1.0            # the mug lifted 2 cm clear of the 3 cm tray for the last stretch (the
                                                    # strafe test held through 8 s at 0.2; 0.6 flung it, beat2w/x)
TRAY_SET_NOSE = -0.43                               # the mouth down to ~0.078 m: the mug, riding 3-4 cm below the mouth, meets
                                                    # (-0.35 while the mat collided; drawn only, the mug meets the sand 8 mm
                                                    # lower, and released from a hang of 8-15 deg a centimetre up it landed
                                                    # on its base's edge and the retreat's beam tipped it onto its handle,
                                                    # 25 deg, film67; 0.1 of nose is ~8 mm of mouth height, probe_hang)
                                                    # the 8 mm mat (probe_hang: ry -0.4 -> 0.072, 0.0 -> 0.107); no lift before
                                                    # it (TRAY_LIFT_AT -1): the level carry goes straight to the set-down
TRAY_STAND_BACK = 0.2                               # the room walk's goal stands this far before the placing base position, on
                                                    # the line (0.5 put it 0.37 m from the film layout's last pillar, inside
                                                    # the inflation: the planner moved the goal aside and the crab came in
                                                    # diagonally at +40 deg, film74)
TRAY_STOP_EARLY = 0.02                              # the place triggers this far short: the walk-in coasts about as far (w63)
                                                    # (0.08 until film67/68: with the spot 22 cm before the pole the mug landed
                                                    # 6-8 cm short of it, on the mat's near edge, twice -- the coast the 0.08
                                                    # allowed for is not there at the 0.45 walk-in)
TRAY_REACH = 0.22                                   # the MUG's centre this far ahead of the base: the pinch holds the handle at the
                                                    # mouth (0.295) and the cup hangs 0.086 behind it, toward the crab. At 0.29 the
                                                    # mug landed straddling the tray's near edge, 22 deg over, twice (film12/13)
TURN_MODE = os.environ.get("AUTO_TURN_MODE", "turn")
TURN_TEST = (float(os.environ["AUTO_TURN_TEST"]) if os.environ.get("AUTO_TURN_TEST") else None)   # see `carry`
RETREAT = os.environ.get("AUTO_RETREAT", "level")   # see `tray_retreat` ("down" until 2026-10-14: it tipped the placed mug
                                                    # in w63/film43; probe_pinch PINCH_SEQ=retreat, 12 trials on the mat and 12
                                                    # on the sand, every variant left it standing -- "level", the plainest,
                                                    # backs straight off at the set height with no sweep of the jaw near it)
HUD_DIR = os.environ.get("AUTO_HUD", "")             # write the sensor overlay's frames here at PIP_FPS (tools/fetch/hud.py)
SLEEP = os.environ.get("AUTO_SLEEP", "1") == "1"    # the ending: the trained sleep gesture (Y) rather than the posture mode's
                                                    # lowest stance, which keeps balancing and jiggles (routine2, seg2i)
SLEEP_END = 17.0                                    # beat 3 calls it done here, mid-breath (the clip rises at 18.5 s measured)
SLEEP_S = 19.0                                      # the clip runs this long and returns to locomotion by itself
CIRCLE_STICK = 0.65                                 # the turn band is 0.5-1.0 of the stick; 1.0 spun a full turn in 1.5 s (choreo1)
SIP_TIP = 1.0                                       # the drink's tip-back: the nose all the way up, briefly (SIP_TIP_S)
SIP_TIP_S = 1.6                                     # (at 0.8/0.8 s it did not read on camera; the lighter mug carries the full tip)
HOLD_OUT_NOSE = 0.7                                 # the cup held out after the drink
SET_DOWN_S = 2.0                                    # the nose comes down over this long; the mug meets the sand on the way
DRAG_S = 0.0                                        # the backward drag as it lands levelled a heavy mug hanging 30-45 deg forward;
                                                    # the 75 g mug hangs near level and the drag tipped it back onto its handle (light2/3)
RELEASE_S = 1.0                                     # seconds the trigger takes to come off, with the mug just clear of the sand
RELIFT = 0.10                                       # the nose back up this much after the touch-down: ~1.5 cm of hang
RELEASE_TO = 0.0                                    # the trigger eased all the way off before the step back: part-open (0.3) left the
                                                    # finger under the loop's lower bar and the step back dragged the mug over onto
                                                    # its handle (light6/8); let go at once it whips (deliver1), so: slow, and fully
SET_DOWN_NOSE = 0.9                                 # the nose goes this far down for the set-down, past the grasp's 0.6: a mug that
                                                    # hangs tilted (29-45 deg, deliver2) lands on its edge, and pushing the handle on
                                                    # down levels it on the sand before the finger lets go
STALE_S = 0.6                                       # a mug estimate older than this is dead-reckoned, not trusted
SEEN_HOLD_S = 1.5                                   # the final creep keeps the last SEEN bar this long over an assumed one
# The grasp stages run as they did through w68 (2026-10-14 ~06:00). The "error budget"
# changes that followed (fine at 0.008, no pulses at depth, a 1.5 s give-up into a
# recentre cycle, the creep stopped 1.8 cm short, closes refused past the depth) MEASURED
# WORSE: first-grasp success w33-w68 22 of 30 (73%), w69-w88 10 of 20 (50%), films 36-48
# 8 of 12 against 52-64 5 of 8. Each was reasoned from one failure and cost more than it
# saved; the constants keep their new names with the old values, and RECENTRE turns the
# cycle off (the state stays, for the fine stage's timeout).
RECENTRE = False
BODY_REAR = 0.20                                    # the legs reach this far behind the base (the posed model's geoms,
                                                    # carry2: 0.20 behind, 0.26 ahead, 0.26 each side)
RETREAT_MAX_M = 0.55                                # the retreat backs no further than this: a metre back (film67/68) put the
                                                    # ending at the pillar, out of the tray's frame
RECENTRE_S = 0.7                                    # the recentre backs this long (~5 cm) before aiming and creeping again
FINE_MAX_S = 8.0                                    # the fine stage gives up strafing after this (w80: 100 s in the dead band)
FINE_EY = 0.02                                     # the side error `fine` hands to the final creep (was 0.02: the hole in the budget)
CLOSE_HOLD_S = 0.3                                  # the close window must hold this long (w51: one frame in it closed on air)
FINE2_EY = (-0.006, 0.020)                          # the side error the close accepts at depth, where no twist is applied (w34):
                                                    # ey below -0.006 puts the bar > 35 mm right of the mouth and the finger's
                                                    # sweep then SPINS the mug instead of sliding it onto the beam (36-68 deg
                                                    # in w16/w38/w31, 37-40 mm; <= 35 mm pinched in 30 of 30 walk tests);
                                                    # toward the beam it is 14 mm at +0.015 and w14 pinched at 7
CREEP_STOP = -0.005                                  # the final creep lets go this far short; the coast (1.6-2.9 cm) does the rest
DEPTH_OVER = 1.0                                  # the close refuses the bar this far past its depth (w81: air at 1 and -11 mm)
DEPTH_TOL = 0.012                                   # (0.012 until 2026-10-14: film61 closed on a bar 9 mm behind the mouth
                                                    # -- the estimate 9 mm long -- and the lift to 0.5 shook it out in 2 s;
                                                    # the probes' lifts held every bar seated >= 16 mm; with CREEP_STOP the
                                                    # coast lands the creep at ex ~ -0.002, inside this)
_DEPTH_TOL_OLD = 0.012                                   # the anvil this much short of its depth still has the bar in the finger's
                                                    # sweep (it shuts 26 mm behind the anvil); at 0.018 the finger closed on
                                                    # nothing three times running (auto33/36)
RECREEPS = 4                                        # short of the depth at fine2 this many times before backing off (2 sent auto31 back)
CREEP_SPEED = 0.2                                   # m/s at the creep stick, for the dead reckoning (0.5 -> 0.21 measured)


class SegmentDone(Exception):
    """Raised by the pilot at AUTO_STOP: run.py ends the take there, videos closed."""


def make_pilot(env, player, log=print):
    from controller import GamepadState

    senses = Senses(env)
    mapper = None
    if MAP_DIR:
        from mapper import Mapper
        mapper = Mapper()
    offset, half_x, half_y, hold, gap = (np.array(FEATURE["offset"]), FEATURE["half_x"], FEATURE["half_y"],
                                         FEATURE["hold"], FEATURE["gap"])
    robot = env.scene["robot"]
    mug_entity = env.scene["mug"] if TRUTH else None
    if mapper is not None and MAP_PRIOR:
        # The wake frame is the spawn: the robot stands at the world's origin facing +x
        # (constants.py init_state), so the layout's world points are the map's.
        import mug as _mug
        prior = {"mug": (_mug.MUG_X, _mug.MUG_Y)}
        try:
            import room as _room
            prior["flag"] = (_room.TRAY_XY[0] + _room.FLAG_BACK, _room.TRAY_XY[1] + _room.FLAG_SIDE)
        except ImportError:
            pass
        mapper.prior(prior)
        mapper.soft["mug"] = 0.25                 # the mug is reached, not gone round (until it is in the claw)
        log("\n[auto] the map's prior: " + ", ".join(f"{k} at ({x:+.2f},{y:+.2f})" for k, (x, y) in prior.items()))

    def truth_pose():
        """The base in the world (= the wake frame), for the odometry's log only."""
        bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
        return float(bp[0]), float(bp[1]), float(_yaw(bq))

    def seed_mug():
        """The mug set into the open jaw as probe_pinch.py seats it: the outer bar 31 mm
        behind the mouth point and 29 mm to its right (the creep's means), squared to the
        body, standing on the ground. Returns the mouth's height over the bar."""
        from mug import BODY_HALF_H
        import torch
        ent = env.scene["mug"]
        mouth, closing, inward, fin = senses.body()
        bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
        yaw = _yaw(bq)
        bar_w = bp + _quat_rot(bq, mouth + np.array([-0.031, -0.029, 0.0]))
        c, s_ = math.cos(yaw), math.sin(yaw)
        centre = bar_w - np.array([c * offset[0] - s_ * offset[1], s_ * offset[0] + c * offset[1], 0.0])
        pose = torch.tensor([[centre[0], centre[1], BODY_HALF_H, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                            dtype=torch.float32, device=env.device)
        ent.write_root_link_pose_to_sim(pose)
        ent.write_root_link_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))
        mw = bp + _quat_rot(bq, mouth)
        return float(mw[2] - BODY_HALF_H)

    def footprint():
        """The body's extent in the base frame from its own posed model (every geom's
        bound): how far the legs reach behind, ahead and to the sides."""
        senses._pose_self()
        m, d = senses.m, senses.d
        rb = m.geom_rbound
        xs, ys = d.geom_xpos[:, 0], d.geom_xpos[:, 1]
        keep = (m.geom_bodyid > 0) & (rb < 0.5)
        return (float((xs - rb)[keep].min()), float((xs + rb)[keep].max()), float((ys - rb)[keep].min()),
                float((ys + rb)[keep].max()))

    st = {"name": "settle", "since": 0.0, "attempts": 0, "turn": 0, "twist": 0.0, "twist_i": 0.0,
          "after_fine": "creep2", "bump": 0.0, "stall_bar": None, "stall_t": 0.0, "recreeps": 0,
          "heading_ref": None, "axis_samples": [], "bar": None, "bar_t": -1.0, "cmd_v": 0.0,
          "pulse_until": -1.0, "pulse_dir": 0.0, "pulses": 0}
    last_log = [-1.0]
    pip = {"n": 0}
    if PIP_DIR:
        os.makedirs(PIP_DIR, exist_ok=True)
    if HUD_DIR:
        os.makedirs(HUD_DIR, exist_ok=True)

    def goto(name, t):
        log(f"\n[auto] {t:6.2f}s  {st['name']} -> {name}")
        st["name"], st["since"], st["turn"] = name, t, 0

    def truth_bar():
        bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
        mp = mug_entity.data.root_link_pos_w[0].cpu().numpy(); mq = mug_entity.data.root_link_quat_w[0].cpu().numpy()
        bar_w = mp + _quat_rot(mq, np.array([offset[0], offset[1], 0.0]))
        w, x, y, z = bq
        return _quat_rot(np.array([w, -x, -y, -z]), bar_w - bp)

    def mug_now(t):
        """The bar in the base frame: the senses' recent median, or, while it is out of
        view, the last sighting dead-reckoned by the forward speed commanded since."""
        est = senses.mug()
        if est is not None and t - est["age"] < STALE_S:
            st["bar"], st["bar_t"] = est["bar"].copy(), t
            return st["bar"], est
        if st["bar"] is not None:
            st["bar"][0] -= CREEP_SPEED * st["cmd_v"] / CREEP * senses.dt
            return st["bar"], None
        return None, None

    def stick_mag(bar, t, base=0.5):
        """A walking stick that grows while the mug does not move in the image: the
        policy's dead band wanders with the posture."""
        # Judged over a one-second window: a frame-to-frame test never saw the stall
        # when the heading pulses jiggled the mug 2 cm in the picture (beat2i: a 0.5
        # strafe stood still for 110 s and was never bumped).
        if st["stall_bar"] is None or bar is None:
            st["stall_bar"], st["stall_t"] = (None if bar is None else bar[:2].copy()), t
        elif t - st["stall_t"] > 1.0:
            if np.linalg.norm(bar[:2] - st["stall_bar"]) < 0.03:
                st["bump"] = min(0.3, st["bump"] + 0.05)
                log(f"\n[auto] stalled; stick now {base + st['bump']:.2f}")
            st["stall_bar"], st["stall_t"] = bar[:2].copy(), t
        return base + st["bump"]

    def walk_to(err, s, mag, tol=0.03):
        """One axis at a time (the policy stands still under a diagonal below ~0.35)."""
        ex, ey = float(err[0]), float(err[1])
        if abs(ey) > 0.025 and st["bump"] < 0.15:
            # Strafe first -- unless the strafe has stalled three times: a 0.7 stick
            # sideways moved nothing for 40 s once (auto41), where walking forward
            # changes the posture and the later stages strafe again from there.
            s["lx"] = -mag * (1 if ey > 0 else -1)
            return False
        if abs(ex) > tol:
            s["ly"] = -mag * (1 if ex > 0 else -1)
            s["lx"] = -max(-0.2, min(0.2, 8.0 * ey))
            return False
        return True

    def hold_heading(s, t, want=0.0, start=0.10, stop=0.03):
        """Turn pulses on the gyro heading, relative to the handle's axis as first sighted."""
        yaw = senses.yaw - (st["heading_ref"] or 0.0)
        e = _wrap(want - yaw)
        if st["turn"] == 0 and abs(e) > start:
            st["turn"] = 1 if e > 0 else -1
        elif st["turn"] != 0 and (abs(e) < stop or (e > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"] != 0:
            turn_in_place(s, st["turn"], t)        # (a turn that is not turning gets a step: w86 stood 90 s in the creep)
        else:
            st["tip_t"] = None
        return e

    def mug_sized(est):
        """A cluster the size of the mug. The search once took the robot's own arm, swinging
        out and ahead of the self-mask (0.22 m, 0.5 m wide, orbA), and a 1 cm sliver at the
        picture's edge, for the mug."""
        return (est is not None and est["width"] < 0.2 and MUG_HEIGHT[0] < est["height"] < MUG_HEIGHT[1]
                and est["width"] > 0.03)

    def is_flag(c, anchor=None):
        """The flag pole: thin, taller than a pillar (0.6 vs 0.4 m: a pillar half hidden
        behind the carried mug is a thin strip 0.34 m tall, beat2m), and whole -- a pillar
        cut by the picture's side is thin too, and one at -25 deg took turns with the
        real flag at +20 (beat2j). Its top is in view down to 0.5 m (34 deg up)."""
        # With the nose level the picture's top is 21 deg up: at 0.7 m only 0.40 m of the
        # pole shows, at 1 m 0.51 (beat2q walked past the tray with the flag "too short").
        need = min(FLAG_HEIGHT, 0.10 + 0.30 * c["range"])
        # (the pole is 0.42 m, the pillars 0.6: anything taller than 0.5 is a pillar's edge)
        if not (c["width"] < FLAG_WIDTH and need < c["height"] < 0.50 and not c["edge"]):
            return False
        span = st.get("mug_span")
        if span is not None and span[0] < c["bearing"] < span[1]:
            # Behind the carried mug: a pillar 1.9 m ahead showed beside the mug's silhouette
            # as a one-column strip 37 cm tall, passed for the pole 2.9 m ahead, and the
            # odometry was "corrected" a metre forward (carry11). Nothing seen through the
            # mug's edge is the flag; the real pole behind the mug is simply hidden.
            return False
        if c["top"]:
            # Its top above the picture: the height read is the picture's, and inside
            # 0.5 m a pillar (film71) or a palm trunk (film70) is a thin strip 0.3 m tall
            # too -- the near cut at NAV_NEAR slices the strip off their faces, so the
            # width and side-edge tests let them through. With the nose level the pole's
            # own top (0.42 m, the sensor at 0.14) is in the picture beyond 0.73 m, so a
            # cut candidate is the flag only where the flag is already marked.
            return anchor is not None and np.hypot(c["x"] - anchor[0], c["y"] - anchor[1]) < FLAG_HOLD_M
        if mapper is not None and "flag" in mapper.fixed:
            # The flag known: a candidate is it only within the odometry's error budget of
            # where the map puts it (the line band below, relative to the mark, included).
            return anchor is not None and np.hypot(c["x"] - anchor[0], c["y"] - anchor[1]) < FLAG_GATE_M
        if mapper is not None:
            # The tray stands on the room's line; the beach's palms and rocks do not. The
            # first sighting of film70 was a leaning trunk 2.5 m off to the left, and the
            # crab carried the mug 4 m up the beach to set it down at its foot.
            _, wy = mapper.to_world(np.array([[c["x"], c["y"]]]))
            if abs(float(wy[0])) > FLAG_LINE_HALF:
                return False
        return True

    def flag_truth(f=None):
        """For the log: the real pole in the body frame (and the candidate's shape)."""
        if not TRUTH or mapper is None or "flag" not in mapper.fixed:
            return ""
        tx, ty, tyaw = truth_pose()
        fx, fy = mapper.marks["flag"]
        c, s_ = math.cos(tyaw), math.sin(tyaw)
        bx, by = c * (fx - tx) + s_ * (fy - ty), -s_ * (fx - tx) + c * (fy - ty)
        out = f" (truly {math.hypot(bx, by):.2f} m at {math.degrees(math.atan2(by, bx)):+.0f})"
        if f is not None:
            out += f" [w {f['width'] * 100:.0f} h {f['height'] * 100:.0f} cm{' top-cut' if f.get('top') else ''}{' edge' if f.get('edge') else ''}]"
        return out

    def stop_truth(mouth):
        """For the log: at the stop, the mouth against the mug's spot and the base against
        the pole, in the world."""
        if not TRUTH:
            return ""
        try:
            import room as _room
        except ImportError:
            return ""
        bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
        mw = bp + _quat_rot(bq, mouth)
        px, py = _room.TRAY_XY[0] + _room.FLAG_BACK, _room.TRAY_XY[1] + _room.FLAG_SIDE
        sx, sy = px - _room.MUG_SPOT_BACK, py + _room.MUG_SPOT_SIDE
        return (f"; truly the pole {px - bp[0]:+.2f} ahead, {py - bp[1]:+.2f} left of the base; "
                f"the mouth {(mw[0] - sx) * 100:+.0f} cm along, {(mw[1] - sy) * 100:+.0f} cm left of the spot")

    def flag_seen(f, t):
        """A sighting on the map: with the flag known it corrects the odometry (the pose
        and the grid move, the mark stays); otherwise it is the mark."""
        if mapper is None:
            return
        # The pole's x is the cluster's NEAREST point (plus its radius), not the median:
        # the pennant stands 9-18 cm behind the pole and at 2 m the median read 17 cm long
        # (carry2: 2.43 m sensed, 2.26 true; 2.14 vs 1.97; exact inside 1.6 m)
        if "near" in f:
            f = dict(f, x=float(f["near"]) + 0.012)
            f["range"] = float(np.hypot(f["x"], f["y"]))
        if "flag" in mapper.fixed:
            if f.get("top"):
                return                                       # (a cut strip: held by the mark, not trusted to move it)
            d = mapper.fix("flag", f["x"], f["y"], max_m=FLAG_GATE_M)
            if d is not None and t - st.get("fix_log_t", -9.0) >= 2.0:
                st["fix_log_t"] = t
                log(f"\n[auto] the flag seen {f['range']:.2f} m at {math.degrees(f['bearing']):+.0f}{flag_truth(f)}: "
                    f"the odometry was {d * 100:.0f} cm off, corrected half-way")
        else:
            mapper.mark("flag", f["x"], f["y"])

    def flag_anchor(t):
        """Where the flag is expected now, in the body frame: the map's mark (turned and
        moved with the pose), else the last sighting while it is fresh, else None."""
        if mapper is not None and "flag" in mapper.marks:
            return mapper.mark_body("flag")
        prev = st.get("flag")
        if prev is not None and t - st.get("flag_t", -9.0) < 3.0:
            return (prev["x"], prev["y"])
        return None

    def pick_flag(cands, t, anchor=None):
        """The candidate nearest where the flag is expected (the map's mark, else where it
        was last seen); none while the one seen lately is simply hidden (the carried mug
        hid it for a second at a time and a pillar 0.8 m further took its place, beat2k);
        the nearest when there is no recent one. The body-frame memory alone slid back as
        the crab walked and never turned: after a turn pulse a trunk 0.35 m from the
        slid point took the flag's place and the mark walked 1.6 m up the beach (film70)."""
        prev = st.get("flag")
        if anchor is None and prev is not None:
            anchor = (prev["x"], prev["y"])
        if anchor is not None:
            near = [c for c in cands if np.hypot(c["x"] - anchor[0], c["y"] - anchor[1]) < 0.35]
            if near:
                return min(near, key=lambda c: np.hypot(c["x"] - anchor[0], c["y"] - anchor[1]))
            if t - st.get("flag_t", -9.0) < 3.0:
                return None
        return min(cands, key=lambda c: c["range"]) if cands else None

    def arc(s, want, t):
        """Steer by a held twist while walking, the sign learned from the gyro: every
        0.8 s the heading's drift is compared with the way wanted, and a twist that
        bent the walk the wrong way is flipped (at most once in 3 s)."""
        sign = st.get("arc_sign", 1.0)
        s["rx"] = NAV_ARC * (1 if want > 0 else -1) * sign
        if st.get("arc_t") is None or (st.get("arc_want", 0.0) > 0) != (want > 0):
            st["arc_t"], st["arc_yaw"], st["arc_want"] = t, senses.yaw, want
        elif t - st["arc_t"] > 0.8:
            moved = _wrap(senses.yaw - st["arc_yaw"])
            if (moved > 0) != (want > 0) and abs(moved) > math.radians(1.5) and t - st.get("arc_flip_t", -9.0) > 3.0:
                st["arc_sign"], st["arc_flip_t"] = -sign, t
                log(f"\n[auto] the twist bends the walk the other way: arc sign now {-sign:+.0f}")
            st["arc_t"], st["arc_yaw"], st["arc_want"] = t, senses.yaw, want

    def turn_in_place(s, direction, t):
        """The turn stick, watched on the gyro: a turn that has not moved the heading 2 deg
        in 1.5 s gets a short step forward to break the posture, then resumes (the final
        turn at the tray stood at rx -0.75 for 25 s with the heading fixed, beat2s/t,
        while the same stick turned the crab 50 deg/s in the probes)."""
        if st.get("tip_t") is None or st.get("tip_dir") != direction:
            st["tip_t"], st["tip_yaw"], st["tip_dir"], st["tip_step_until"] = t, senses.yaw, direction, -1.0
        if t < st["tip_step_until"]:
            s["ly"] = -0.5
            return
        if t - st["tip_t"] > 1.5:
            if abs(_wrap(senses.yaw - st["tip_yaw"])) < math.radians(2.0):
                st["tip_step_until"] = t + 0.4
                log("\n[auto] the turn is not turning: a step forward to free the gait")
            st["tip_t"], st["tip_yaw"] = t, senses.yaw
        s["rx"] = NAV_TURN * direction * TWIST_SIGN

    def tray_centre(c):
        """The tray's centre from its pole, in the base frame: the pole stands FLAG_BACK
        behind and FLAG_SIDE beside the centre along the ROOM's axes (the gyro's zero), so
        the offset is turned by the heading first -- added in the base frame, a 30 deg
        heading at the tray put the mug 14 cm off the centre, beside the tray (film5/8)."""
        cy, sy = math.cos(senses.yaw), math.sin(senses.yaw)
        ox, oy = -FLAG_BACK, -FLAG_SIDE
        return np.array([c["x"] + cy * ox + sy * oy, c["y"] - sy * ox + cy * oy])

    def face(s, ref, start=0.12, stop=0.04):
        """Turn pulses that keep the cup at the nose. Returns its bearing."""
        bearing = math.atan2(ref[1], ref[0])
        if st["turn"] == 0 and abs(bearing) > start:
            st["turn"] = 1 if bearing > 0 else -1
        elif st["turn"] != 0 and (abs(bearing) < stop or (bearing > 0) != (st["turn"] > 0)):
            st["turn"] = 0
        if st["turn"] != 0:
            s["rx"] = TURN * st["turn"] * TWIST_SIGN
        return bearing

    def grip_target(bar, closing):
        side = -1.0 if closing[1] > 0 else 1.0
        return np.array([bar[0] + hold, bar[1] + side * (half_y + gap)])

    def pilot(t):
        senses.update(t)
        s = dict(lx=0.0, ly=0.0, rx=0.0, ry=0.0, lt=0.0, rt=0.0, hat_x=0, hat_y=0, buttons=frozenset())
        mouth, closing, inward, fin = senses.body()
        name, since = st["name"], t - st["since"]
        fsm = player.fsm
        bar, est = mug_now(t)
        if est is not None and t - est["age"] < STALE_S and name in ("look", "nav", "tray_approach"):
            cx, cy, hw = float(est["cup"][0]), float(est["cup"][1]), float(est["width"]) / 2 + 0.03
            st["mug_span"] = (math.atan2(cy - hw, cx) - math.radians(3), math.atan2(cy + hw, cx) + math.radians(3))
        else:
            st["mug_span"] = None
        if (mapper is not None and "mug" in mapper.fixed and est is not None and t - est["age"] < STALE_S
                and mug_sized(est) and name in ("wander", "spot", "run", "search", "orbit", "align", "creep", "fine")
                and t - st.get("mug_fix_t", -9.0) >= 1.0):
            # The mug is the first landmark (the flag the second): the wander's and the
            # jump's odometry put the crab 0.9 m short by the grasp (film68), past the
            # flag's gate. The cup's centre, seen, against the known spot.
            st["mug_fix_t"] = t
            d = mapper.fix("mug", float(est["cup"][0]), float(est["cup"][1]), gain=0.5, max_m=2.5)
            if d is not None and t - st.get("mug_fix_log_t", -9.0) >= 5.0:
                st["mug_fix_log_t"] = t
                tp = truth_pose() if TRUTH else None
                log(f"\n[auto] the mug seen: the odometry was {d * 100:.0f} cm off, corrected half-way"
                    + (f" (map {mapper.x:+.2f},{mapper.y:+.2f} truly {tp[0]:+.2f},{tp[1]:+.2f})" if tp else ""))
        # The final creep steers on the handle as SEEN, never as assumed: at depth the jaw
        # hides the handle for a frame or two and the assumed bar (the cup's face plus the
        # handle's reach) sits 3+ cm off the real one in 54 of 60 such frames (18% when
        # seen), so the strafe pulses chased the flicker and the jaw knocked the mug round
        # or over (w50-w53). The last seen bar is kept for up to SEEN_HOLD_S instead.
        if est is not None and est["seen_handle"]:
            st["seen_bar"], st["seen_bar_t"] = est["bar"].copy(), t
        elif (name == "fine2" and est is not None and st.get("seen_bar") is not None
              and t - st.get("seen_bar_t", -9.0) < SEEN_HOLD_S):
            # (standing only: a body-frame point held through the creep's 6 cm and 8 deg
            # of veer was 2-3 cm wrong at depth, and three closes shut on air, w69/w71)
            bar = st["seen_bar"]
        if TRUTH_CLOSE and TRUTH and name in ("creep2", "fine2") and bar is not None:
            bar = truth_bar()

        def finish(s):
            """Every step ends here, early returns included: the HUD and depth frames
            and the truth line were skipped for as long as a state returned early
            (the `fine` stage: w80 shows no frames for its last 100 s)."""
            st["cmd_v"] = -s["ly"]

            if (PIP_DIR or HUD_DIR) and senses.last is not None and t >= pip["n"] / PIP_FPS:
                if PIP_DIR:
                    from probe_sense import _picture
                    _picture(senses.last, os.path.join(PIP_DIR, f"pip_{pip['n']:05d}.png"))
                if HUD_DIR:
                    import hud
                    hud.draw(os.path.join(HUD_DIR, f"hud_{pip['n']:05d}.png"), {
                        "t": t, "state": name, "mode": fsm.mode, "yaw": senses.yaw, "up": senses.up(),
                        "q": robot.data.joint_pos_biased[0].cpu().numpy().tolist(), "names": list(robot.joint_names),
                        "aperture_mm": aperture_mm(fin), "pad": s, "note": st.get("hud_note", "")})
                pip["n"] += 1

            if t - last_log[0] >= 0.5:
                last_log[0] = t
                b = "none" if bar is None else f"({bar[0]:+.3f},{bar[1]:+.3f}{'' if est else ' dr'})"
                if est is not None:
                    lf = senses.last["est"]
                    b += (f" px={lf['n'] if lf else 0} handle={'seen' if est['seen_handle'] else 'ASSUMED'}"
                          f" h={est['height']:.2f} w={est['width']:.2f} rows={est['rows']}")
                tr = ""
                if TRUTH:
                    tb = truth_bar()
                    mq = mug_entity.data.root_link_quat_w[0].cpu().numpy()
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(_quat_rot(mq, np.array([0.0, 0.0, 1.0]))[2])))))
                    mz = float(mug_entity.data.root_link_pos_w[0, 2])
                    up_w = _quat_rot(mq, np.array([0.0, 0.0, 1.0]))
                    bq = robot.data.root_link_quat_w[0].cpu().numpy()
                    up_b = _quat_rot(np.array([bq[0], -bq[1], -bq[2], -bq[3]]), up_w)   # the mug's up in the body frame
                    mq_ = mug_entity.data.root_link_quat_w[0].cpu().numpy(); bq_ = robot.data.root_link_quat_w[0].cpu().numpy()
                    myaw = math.degrees(_wrap(_yaw(mq_) - _yaw(bq_)))
                    tr = (f" truth=({tb[0]:+.3f},{tb[1]:+.3f}) tilt={tilt:.0f} myaw={myaw:+.0f} up_b=({up_b[0]:+.2f},{up_b[1]:+.2f}) "
                          f"mug_z={mz:.3f} mode={fsm.mode}")
                log(f"\n[auto] t={t:6.2f} {name:<10} bar_b={b}{tr} mouth_b=({mouth[0]:+.3f},{mouth[1]:+.3f},{mouth[2]:+.3f}) "
                    f"yaw={math.degrees(senses.yaw - (st['heading_ref'] or 0.0)):+5.1f} finger={fin:+.3f} "
                    f"ly={s['ly']:+.2f} lx={s['lx']:+.2f} rx={s['rx']:+.2f} ry={s['ry']:+.2f}")
            return GamepadState(**s)


        if STOP_AT and name == STOP_AT and not st.get("stopped"):
            st["stopped"] = True
            log(f"\n[auto] segment ends at {name}")
            raise SegmentDone(name)
        senses.max_range = WANDER_RANGE if name in ("wander", "spot", "run") else 1.5

        if name == "settle":
            if since > 1.0 and START == "carry":
                goto("seed", t)
            elif since > 1.0:
                if BEAT >= 2:
                    st["danced"] = BEAT == 2                # beat two remembers; beat three dances when it sees the mug
                    goto("asleep", t)
                else:
                    goto("wander" if WANDER else "enter_claw", t)
        elif name == "seed":
            # The delivery alone (AUTO_START=carry): into the claw mode, the arm out, the
            # nose down as at a grasp, the mug set into the open jaw by hand, then on to
            # the close, the lift and the carry as after a grasp.
            # As probe_pinch.py times it: LB, the arm out and the nose down 2 s later, the mug
            # placed 1.4 s after that (the arm held out from the start reached only 0.22 m
            # with the mouth 28 mm under the bar, carry1/2, and the jaw shut on air).
            if since < 0.1:
                s["buttons"] = frozenset({"LB"})
            if since > 2.0:
                s["hat_x"] = -1
                st.setdefault("seed_ry", -NOSE_DOWN)
                s["ry"] = st["seed_ry"]
            if fsm.mode == "claw_left" and fsm.running_policy and since > 3.4:
                # The stance varies run to run (the pitch -0.16 to -0.22 at this moment, the
                # mouth +3 to -27 mm off the bar, carry5-7): past 10 mm the nose is servoed
                # on the height first, slowly, and the mug placed inside 5 mm (or at 12 s).
                from mug import BODY_HALF_H
                bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
                dz = float((bp + _quat_rot(bq, mouth))[2] - BODY_HALF_H)
                if abs(dz) > 0.010 and since < 12.0:
                    st["seed_ry"] = max(-0.95, min(0.2, st["seed_ry"] - 2.0 * dz * senses.dt))
                    if t - st.get("seed_log_t", -9.0) >= 1.0:
                        st["seed_log_t"] = t
                        log(f"\n[auto] seed: nose {st['seed_ry']:+.2f}, the mouth {dz * 1000:+.0f} mm over the bar")
                    pass
                elif abs(dz) < 0.005 or since >= 12.0:
                    st["seed_place"] = True
            if st.get("seed_place"):
                if True:
                    dz = seed_mug()
                    bx0, bx1, by0, by1 = footprint()
                    log(f"\n[auto] seeded at nose {st['seed_ry']:+.2f}: the mouth {dz * 1000:+.1f} mm over the bar; the body reaches "
                        f"{-bx0:.2f} m behind, {bx1:.2f} ahead, {-by0:.2f} right, {by1:.2f} left of the base")
                    st["lift_from"] = st["seed_ry"]
                    goto("seed_close", t)
        elif name == "seed_close":
            # The trigger ramped over 0.6 s as the grasp does it, the arm-out hat held HERE
            # (the common rule did not reach this state: the arm drew back 56 mm as the
            # finger shut, carry5-7, and the jaw shut on air). A close that reads shut
            # (finger > 0.05) opens, seats the mug again at the mouth and tries again.
            s["hat_x"], s["ry"], s["lt"] = -1, st["seed_ry"], min(1.0, since / 0.6)
            if since > 2.0:
                mouth_mm = aperture_mm(fin)
                if fin > 0.05 and st.get("seed_tries", 0) < 3:
                    st["seed_tries"] = st.get("seed_tries", 0) + 1
                    log(f"\n[auto] the seeded close shut on air (finger {fin:+.3f}): opening and seating it again, try {st['seed_tries']}")
                    goto("seed_again", t)
                else:
                    log(f"\n[auto] grasped: finger at {fin:+.3f}, mouth {mouth_mm:.0f} mm for a 8 mm bar (seeded)")
                    st["grasp_yaw"], st["sip_ok"] = senses.yaw, True
                    goto("withdraw", t)
        elif name == "seed_again":
            s["hat_x"], s["ry"], s["lt"] = -1, st["seed_ry"], 0.0
            if since > 1.4:
                dz = seed_mug()
                log(f"\n[auto] seeded again at nose {st['seed_ry']:+.2f}: the mouth {dz * 1000:+.1f} mm over the bar")
                goto("seed_close", t)
        elif name == "asleep":
            # Beat two opens on the sleep (Y), the alert comes ALERT_S in: Menu leaves the
            # gesture, the walk stands it up, and A -- the jump -- is the startle.
            if since < 0.2:
                s["buttons"] = frozenset({"Y"})
            elif since > ALERT_S:
                log("\n[auto] the alert")
                goto("wake", t)
        elif name == "wake" and BEAT == 3:
            # Up from the sleep and a look about: the body rolled side to side (R3 held,
            # the right stick is roll), then off into the desert.
            if since < 0.2:
                s["buttons"] = frozenset({"menu"})
            elif since > 1.5 and fsm.mode == "locomotion":
                u = since - 1.5
                if u < 4.0:
                    s["buttons"] = frozenset({"R3"})
                    s["rx"] = LOOK_ROLL * (1 if int(u / 1.0) % 2 == 0 else -1) * (math.sin(math.pi * (u % 1.0)))
                elif u > 4.5:
                    log("\n[auto] awake: off into the desert")
                    goto("wander", t)
            elif since > 10.0:
                goto("wander", t)
        elif name == "wake":
            if since < 0.2:
                s["buttons"] = frozenset({"menu"})
            elif 1.2 < since < 1.4 and fsm.mode == "locomotion" and not st.get("jumped2"):
                st["jumped2"] = True
                s["buttons"] = frozenset({"A"})
            elif st.get("jumped2") and since > 3.5 and fsm.mode == "locomotion":
                log("\n[auto] awake: the mug")
                goto("enter_claw", t)
            elif since > 8.0:
                goto("enter_claw", t)
        elif name == "wander":
            # Lost in the desert: a serpentine walk in locomotion, a look-out every few
            # legs, until the depth image shows something mug-sized within range.
            # A sighting counts only in locomotion (mid-gesture the leg is in the picture)
            # and only from a fresh exposure, never the dead-reckoned guess.
            muglike = (fsm.mode == "locomotion" and est is not None and est["n"] >= 4 and est["rows"] >= 2
                       and MUG_HEIGHT[0] < est["height"] < MUG_HEIGHT[1] and est["width"] < MUG_WIDTH)
            # (a one-row sliver of dune slope was a "mug" in seg1b, hence rows >= 2; rejecting
            # clusters at the picture's side as well lost the mug entirely, since it comes
            # into view from the side as the body swings -- seg1c walked straight past it)
            if BEAT == 3:
                muglike = muglike and est["px"] >= SIGHT_MIN_PX   # (est["n"] is the history's length, never 10: film2)
            if muglike and np.hypot(est["bar"][0], est["bar"][1]) < (SIGHT_RANGE if BEAT == 3 else WANDER_RANGE):
                if st.get("seen_since") is None:
                    st["seen_since"] = t
            else:
                st["seen_since"] = None
            if st.get("seen_since") is not None and t - st["seen_since"] >= SIGHT_HOLD_S:
                # Held for a while: a mug stays put, a band of floor at the range limit
                # or a leg mid-stride flickers (seg1b jumped at a 6 cm phantom).
                log(f"\n[auto] something mug-sized at {np.hypot(est['bar'][0], est['bar'][1]):.2f} m, "
                    f"{math.degrees(math.atan2(est['bar'][1], est['bar'][0])):+.0f} deg, {est['height'] * 100:.0f} cm tall: a jump")
                goto("spot", t)
            elif st.get("look_until", -1.0) > 0:
                # The look-out: stop for a second, hold d-pad right for 0.2 s (one 5 ms
                # step was never seen), let the salute run (~5 s, back to locomotion by
                # itself), then on with the next swing. The reset used to come before
                # the press and nothing ever saluted (seg1d).
                if t < st["look_until"]:
                    pass                                     # standing, before the press
                elif not st.get("look_pressed"):
                    st["look_pressed"], st["look_press_until"] = True, t + 0.2
                if st.get("look_pressed") and t < st["look_press_until"]:
                    s["hat_x"] = +1                          # d-pad right: gesture_salute, from locomotion
                elif st.get("look_pressed") and t > st["look_press_until"] + 1.0 and fsm.mode == "locomotion":
                    st["look_until"], st["legs"], st["look_pressed"] = -1.0, 0, False   # back from the look-out
            else:
                # The serpentine, closed on the gyro: turn one way until the heading is
                # WANDER_SWING off the line, then the other, so it still goes forward.
                if "wander_ref" not in st:
                    # Beat 3 wanders along the line it woke on (the gyro's zero): the look-about
                    # had turned it 20 deg and a serpentine on that heading passed the mug
                    # 1.5 m to the side (film1).
                    st["wander_ref"], st["wander_dir"], st["legs"] = (0.0 if BEAT == 3 else senses.yaw), 1, 0
                if mapper is not None and "mug" in mapper.fixed:
                    # The mug marked on the map: the serpentine swings about the bearing to
                    # it, not the room's line (Andrew: "the map position is just to get to
                    # the object"); the sighting and the grasp are the depth sensor's alone.
                    st["wander_ref"] = senses.yaw + math.atan2(*mapper.mark_body("mug")[::-1])
                yaw = _wrap(senses.yaw - st["wander_ref"])
                if st["wander_dir"] * yaw < -WANDER_SWING:   # the heading has swung far enough
                    st["wander_dir"] *= -1
                    st["legs"] += 1
                    if st["legs"] >= LOOK_EVERY:
                        st["look_until"] = t + 1.0           # stop, then the look-out
                        st["look_pressed"] = False
                s["ly"] = -0.5
                s["rx"] = WANDER_TURN * st["wander_dir"]
                # A stall watchdog on the gyro: the serpentine stood jammed for 90 s at one
                # heading with the turn held (film2). No heading change in 2 s -> a second
                # of the full turn stick, then on.
                if st.get("wander_watch_t") is None or abs(_wrap(senses.yaw - st["wander_watch_yaw"])) > math.radians(2.0):
                    st["wander_watch_t"], st["wander_watch_yaw"] = t, senses.yaw
                elif t - st["wander_watch_t"] > 2.0:
                    st["wander_kick_until"], st["wander_watch_t"] = t + 1.0, t
                    log("\n[auto] the serpentine is stuck: a full turn pulse")
                if t < st.get("wander_kick_until", -1.0):
                    s["rx"] = TURN * st["wander_dir"]
                    s["ly"] = 0.0
        elif name == "spot":
            # Seen! A jump (A, from locomotion), then the run-in.
            if not st.get("jumped") and fsm.mode == "locomotion":
                st["jumped"], st["jump_t"] = True, t
                s["buttons"] = frozenset({"A"})
            elif st.get("jumped") and fsm.mode == "locomotion" and t - st["jump_t"] > 2.5 or since > 8.0:
                goto("run", t)
        elif name == "run":
            # Straight at it, fast, steering on the bearing from the depth image -- only
            # on frames that still look like a mug: the bounce of the run put a band of
            # floor or dune in the cluster (w=1.4 m) and the steering chased it (wander10).
            good = (est is not None and est["width"] < MUG_WIDTH and MUG_HEIGHT[0] < est["height"] < MUG_HEIGHT[1])
            if good:
                st["run_bar"], st["run_t"] = est["bar"].copy(), t
            elif st.get("run_bar") is not None:
                st["run_bar"][0] -= CREEP_SPEED * st["cmd_v"] / CREEP * senses.dt
            if st.get("run_bar") is None or t - st.get("run_t", -9.0) > 1.0:
                # Lost for a second: stand and look again. Running on at a dead-reckoned
                # fix and steering at it spun the crab 220 deg in 4 s (w40: the jump put
                # the mug at the picture's edge) and the search then turned in place
                # for 110 s with the mug 2 m off, past its sighting range.
                s["ly"] = -RUN_STICK if since < 1.0 else 0.0
                if since > 1.0:
                    goto("enter_claw", t)          # (the search turns in the claw mode; w41 searched in locomotion)
            else:
                rb = st["run_bar"]
                dist, bearing = np.hypot(rb[0], rb[1]), math.atan2(rb[1], rb[0])
                # The run stops at RUN_STOP, and also as soon as the mug drops out of the
                # picture inside 1.3 m: it slid to the picture's edge as the crab closed in,
                # the run kept its last fix and its steering, and turned into the mug at
                # full stick (w27). No steering inside a metre either.
                if dist < RUN_STOP or (not good and t - st.get("run_t", -9.0) > 0.4 and dist < 1.3):
                    goto("close" if BEAT == 3 else "enter_claw", t)
                else:
                    s["ly"] = -RUN_STICK
                    if good and abs(bearing) > 0.12 and dist > 1.0:        # (steer on a fresh sighting only, w40)
                        s["rx"] = RUN_STEER * (1 if bearing > 0 else -1) * TWIST_SIGN
        elif name == "close":
            # The last stretch to the cup at a walk, facing it, until it is DANCE_RANGE off.
            if est is not None and mug_sized(est):
                st["close_bar"], st["close_t"] = est["bar"].copy(), t
            elif st.get("close_bar") is not None:
                st["close_bar"][0] -= CREEP_SPEED * st["cmd_v"] / CREEP * senses.dt
            cb = st.get("close_bar")
            if cb is None:
                if since > 2.0:
                    goto("enter_claw", t)
            else:
                dist = float(np.hypot(cb[0], cb[1]))
                face(s, cb, start=0.10, stop=0.04)
                if dist > DANCE_RANGE and st["turn"] == 0:
                    s["ly"] = -0.45
                elif dist <= DANCE_RANGE:
                    log(f"\n[auto] at the cup ({dist:.2f} m): the dance")
                    goto("enter_claw", t)
            if since > 12.0:
                goto("enter_claw", t)
        elif name == "enter_claw":
            if since < 0.1:
                s["buttons"] = frozenset({"LB"})
            elif fsm.mode == "claw_left" and fsm.running_policy and since > 2.0:
                goto("search", t)
        elif name == "search":
            # Nothing but the sensor: turn until a mug is in view, then take the handle's
            # axis (cup to bar) as the heading to keep, averaged over a second of frames.
            if not mug_sized(est):
                s["rx"] = SEARCH_TURN * TWIST_SIGN * -1          # counter-clockwise, slowly
                st["axis_samples"], st["hidden_since"], st["turn"] = [], None, 0
                if since < 0.1:
                    st["search_yaw0"] = senses.yaw
                elif BEAT == 3 and abs(senses.yaw - st.get("search_yaw0", senses.yaw)) > 2.5 * 2 * math.pi:
                    # Two and a half turns and nothing mug-sized in range: it is not here.
                    # Back to the wander, which walks the line and looks (w40: 110 s spinning).
                    log("\n[auto] nothing mug-sized in two turns: wandering again")
                    st["bar"], st["run_bar"], st["close_bar"] = None, None, None
                    goto("leave_claw", t)
            elif abs(face(s, est["cup"])) > 0.25:
                # A sliver at the edge of the view (orbB: 8 px at 26 deg, in and out for 57 s,
                # the turn stopping each time it showed): centre it first.
                st["axis_samples"], st["hidden_since"] = [], None
            elif np.hypot(est["bar"][0], est["bar"][1]) > HANDLE_RANGE and est["axis"] is None:
                s["ly"] = -stick_mag(est["bar"], t)              # nearer, where the bar shows (0.5 flat stood 36 s, orbB)
                st["axis_samples"], st["hidden_since"] = [], None
            elif est["axis"] is None:
                # No handle protrudes from the cluster: the mug is side-on (seg2h/j came back
                # from a back-off 26 deg off the axis, took the guessed axis as "straight at
                # us" and walked at the mug from the side, three misses). Circle it instead.
                st["hidden_since"] = st.get("hidden_since") or t
                if t - st["hidden_since"] > HIDDEN_S:
                    if st["heading_ref"] is not None and abs(_wrap(st["heading_ref"] - senses.yaw)) < 0.2:
                        log("\n[auto] no handle where the axis said (the mug turned?): circling left until it shows")
                        st["heading_ref"] = None
                    else:
                        log("\n[auto] mug in view but no handle showing: circling to its handle side")
                    goto("orbit", t)
            else:
                st["hidden_since"] = None
                st["axis_samples"].append(est["axis"])
                if len(st["axis_samples"]) >= 20:
                    a = float(np.median(st["axis_samples"]))
                    # The heading at which the handle points straight at us: the axis (bar to
                    # cup) is a degrees off the nose, so turn BY a. (Was yaw - a until 2026-10-13:
                    # the sign was wrong, the strafes hid a 2a error near the axis, and from the
                    # flank the orbit went to the wrong side every time -- orbA/B.)
                    st["heading_ref"] = senses.yaw + a
                    log(f"\n[auto] handle axis {math.degrees(a):+.1f} deg off the nose over {len(st['axis_samples'])} frames; "
                        f"bar at {np.round(est['bar'], 3)}")
                    if DANCE in DANCES and not st.get("danced"):
                        st["danced"] = True
                        goto("excited", t)
                    elif abs(a) > AXIS_MAX:
                        log(f"\n[auto] {math.degrees(abs(a)):.0f} deg off the handle's axis: circling round to it")
                        goto("orbit", t)
                    else:
                        goto("approach", t)
        elif name == "leave_claw":
            # Out of the claw mode (LB) for the wander, which walks and sights in locomotion.
            if since < 0.1:
                s["buttons"] = frozenset({"LB"})
            elif since > 1.5 and fsm.mode == "locomotion" or since > 6.0:
                goto("wander", t)
        elif name == "orbit":
            # Round the mug to its handle side: face the cup, strafe along the circle, keep
            # the range. With the axis known (heading_ref) the way round is the short one;
            # without, go left until the handle shows and let the search measure it.
            if bar is None or (est is not None and not mug_sized(est)):
                if since > 1.5 and st.get("orbit_seen_t", -9.0) < t - 1.5:
                    goto("search", t)
            else:
                if since < senses.dt * 1.5:
                    st["bump"], st["stall_bar"] = 0.0, None
                st["orbit_seen_t"] = t
                ref = est["cup"] if est is not None else bar
                face(s, ref)
                rng = float(np.hypot(ref[0], ref[1]))
                e = _wrap(st["heading_ref"] - senses.yaw) if st["heading_ref"] is not None else None
                if e is not None and abs(e) < 0.12:
                    log(f"\n[auto] on the handle's axis ({math.degrees(e):+.0f} deg): measuring it again")
                    st["axis_samples"], st["hidden_since"] = [], None
                    goto("search", t)
                elif e is None and est is not None and est["axis"] is not None and since > 1.0:
                    log("\n[auto] the handle shows: measuring its axis")
                    st["axis_samples"], st["hidden_since"] = [], None
                    goto("search", t)
                elif since > ORBIT_S:
                    st["orbits"] = st.get("orbits", 0) + 1
                    if st["orbits"] < 3:
                        log("\n[auto] circled long enough without the handle showing: looking again")
                        st["heading_ref"], st["axis_samples"], st["hidden_since"] = None, [], None
                        goto("search", t)
                    else:
                        log("\n[auto] circled long enough; approaching as is")
                        goto("approach", t)
                elif rng < ORBIT_RANGE[0]:
                    s["ly"] = +stick_mag(ref, t, base=0.5)     # (a fixed 0.5 stood 75 s too close, w50)
                elif rng > ORBIT_RANGE[1]:
                    s["ly"] = -stick_mag(ref, t, base=0.5)
                elif st["bump"] >= 0.15 and since - st.get("nudge_t", -9.0) > 2.0:
                    # (as in align and creep: a strafe the gait will not take at any stick
                    # is freed by a step; w46 stood 75 s in three orbits with the mug 28 deg
                    # off its axis, the 0.5 strafe in the dead band the whole time)
                    st["nudge_t"], st["bump"], st["stall_bar"] = since, 0.0, None
                    log("\n[auto] the orbit's strafe stalled three times; a step forward to free the gait")
                elif since - st.get("nudge_t", -9.0) < 0.5:
                    s["ly"] = -0.5
                else:
                    # Facing the cup, a strafe left moves clockwise round it (seen from above),
                    # which turns the facing yaw down: strafe right when the axis heading is
                    # above the present one. lx < 0 is a strafe left (walk_to's convention).
                    # The stick grows while the cup stands still in the picture (w46).
                    mag = stick_mag(ref, t, base=0.5)
                    s["lx"] = +mag if (e is not None and e > 0) else -mag
        elif name == "excited":
            # A mug! The dance chord (Menu + d-pad) works from the claw mode; Menu alone
            # ends it, into locomotion, and LB brings the claw back for the fetch.
            btn, hx, hy = DANCES[DANCE]
            D0 = 0.0 if BEAT == 3 else BACK_OFF_S        # a step back first: the dance flails, and at
                                                         # (not in the film: the Zoidberg sways in place, and the cup
                                                         # has to be in the shot with the dancer -- Andrew)
            if since < D0:                               # half a metre it shoved the mug 15 cm (choreo2)
                s["ly"] = +0.5
            elif since < D0 + 0.3:
                s["buttons"], s["hat_x"], s["hat_y"] = frozenset(btn), hx, hy
            elif since < D0 + DANCE_S:
                pass
            elif since < D0 + DANCE_S + 0.3:
                s["buttons"] = frozenset({"menu"})
            elif fsm.mode == "locomotion" and since < D0 + DANCE_S + 2.0:
                pass                                     # let it settle
            elif fsm.mode == "locomotion":
                if since < D0 + DANCE_S + 2.2:
                    s["buttons"] = frozenset({"LB"})
            elif fsm.mode == "claw_left" and fsm.running_policy and since > D0 + DANCE_S + 3.5:
                st["heading_ref"] = None                 # the dance turned it; re-sight the handle
                st["axis_samples"] = []
                goto("search", t)
            if since > D0 + DANCE_S + 12.0:
                log(f"\n[auto] stuck in {fsm.mode} after the dance; carrying on")
                goto("search", t)
        elif name == "approach":
            if bar is None or t - st["bar_t"] > 4.0:
                # (the guess was walked at and strafed at for 90 s with the mug 1.8 m
                # behind, film6: four seconds unseen and it is a search again)
                st["bar"] = None
                goto("search", t)
            elif st.get("unstick_t", -9.0) > t - 3.2:
                # The gait would not walk at any stick (seg2: 20 s at 0.8 after the dance).
                # Out of the claw mode and back in, with the height reset between, which
                # starts the policy afresh.
                u = t - st["unstick_t"]
                if u < 0.15 or 1.6 < u < 1.75:
                    s["buttons"] = frozenset({"LB"})
                elif 0.8 < u < 0.95:
                    s["buttons"] = frozenset({"R3"})
                s["hat_x"] = 0
            else:
                standoff = np.array([bar[0] - half_x - PRE - MOUTH_AHEAD - STANDOFF, bar[1] - MOUTH_LEFT])
                there = walk_to(standoff, s, stick_mag(bar, t))
                hold_heading(s, t)
                # (stall_t is reset at every bump, so "2 s since the last stall" never came:
                # seg2f stood 50 s at full stick. Count the bumps at the cap instead.)
                st["capped"] = st.get("capped", 0) + (1 if st["bump"] >= 0.3 and t - st["stall_t"] < senses.dt * 1.5 else 0)
                if st.get("capped", 0) >= 3 and st.get("unsticks", 0) < 2:
                    st["unstick_t"], st["unsticks"], st["bump"], st["stall_bar"], st["capped"] = t, st.get("unsticks", 0) + 1, 0.0, None, 0
                    log("\n[auto] stuck at full stick; leaving and re-entering the claw mode")
                if there:
                    st["bump"], st["stall_bar"] = 0.0, None
                    goto("arm_out", t)
        elif name not in ("stow", "bow", "beckon", "circle", "sit", "sleep", "done", "excited", "wander", "spot", "run",
                          "orbit", "asleep", "wake", "leave_claw", "unstick", "settle", "seed"):   # (look keeps it out: the mug is in it)
            s["hat_x"] = -1                     # the arm out, web up, from here on

        if name in ("align", "creep", "fine", "creep2", "recentre") and st["bump"] >= 0.3:
            # The gait frozen at the stick's cap (w70: the creep stood 110 s at 0.75; w73 the
            # fine's strafe likewise): the approach's cure, out of the claw mode and back in,
            # applied from any of the walking stages, which then start over from the approach.
            st["capped_t"] = st.get("capped_t") or t
            if t - st["capped_t"] > 3.0 and st.get("unsticks", 0) < 3:
                st["unsticks"], st["bump"], st["stall_bar"], st["capped_t"] = st.get("unsticks", 0) + 1, 0.0, None, None
                log(f"\n[auto] {name} stuck at full stick for 3 s; leaving and re-entering the claw mode")
                goto("unstick", t)
        elif name in ("align", "creep", "fine", "creep2", "recentre"):
            st["capped_t"] = None
        if name == "unstick":
            if since < 0.15 or 1.6 < since < 1.75:
                s["buttons"] = frozenset({"LB"})
            elif 0.8 < since < 0.95:
                s["buttons"] = frozenset({"R3"})
            s["hat_x"] = 0
            if since > 3.2:
                goto("approach", t)
            return finish(s)
        if name in ("align", "creep", "fine", "creep2", "fine2", "grasp", "backoff", "recentre"):
            s["ry"] = -NOSE_DOWN

        if name == "arm_out":
            if since > 1.5:
                st["recentres"], st["veer_adj"] = 0, 0.0      # (a fresh attempt starts its count and its aim afresh)
                goto("align", t)
        elif name == "align":
            e = hold_heading(s, t, start=0.05, stop=0.02)
            if bar is None or t - st["bar_t"] > 4.0:
                st["bar"] = None
                goto("search", t)
            elif st["turn"] == 0 and abs(e) < 0.05:
                ey = grip_target(bar, closing)[1] - mouth[1]
                if abs(ey) > 0.008 and st["bump"] >= 0.15 and since - st.get("nudge_t", -9.0) > 2.0:
                    # Three stalled strafe pulses: the gait is stuck in a posture the
                    # strafe does not move (25 s after the dance, choreo2). A step
                    # forward breaks it; the strafe resumes from there.
                    st["nudge_t"], st["bump"], st["stall_bar"] = since, 0.0, None
                    log("\n[auto] strafe stalled three times; a step forward to free the gait")
                if abs(ey) > 0.008 and since - st.get("nudge_t", -9.0) < 0.5:
                    s["ly"] = -0.5
                elif abs(ey) > 0.008:
                    s["lx"] = -stick_mag(bar, t, base=WALK_MIN) * (1 if ey > 0 else -1)   # a 0.45 strafe stood still 25 s (auto5)
                elif since > 1.0:
                    log(f"\n[auto] closing axis {np.round(closing, 3)}; grip target {np.round(grip_target(bar, closing), 3)}; "
                        f"mouth {np.round(mouth, 3)}; bar {np.round(bar, 3)}")
                    goto("creep", t)
        elif name == "creep":
            if bar is None:
                goto("search", t)
            else:
                ex, ey = grip_target(bar, closing) - mouth[:2]
                hold_heading(s, t, start=0.08, stop=0.03)
                if st["turn"] != 0:
                    # Turn first, walk after: a heading pulse held together with the walk
                    # stick moved the crab on neither for 110 s, the stall watchdog meanwhile
                    # raising the walk to its cap (w17). The bump is not judged while turning.
                    st["stall_bar"] = None
                elif ex > hold + half_x + PRE + 0.01:
                    if abs(ey) > 0.02 and st["bump"] >= 0.15 and since - st.get("nudge_t", -9.0) > 2.0:
                        # (as in align: a stalled strafe is freed by a step forward)
                        st["nudge_t"], st["bump"], st["stall_bar"] = since, 0.0, None
                        log("\n[auto] creep's strafe stalled three times; a step forward to free the gait")
                    if abs(ey) > 0.02 and since - st.get("nudge_t", -9.0) < 0.5:
                        s["ly"] = -0.5
                    elif abs(ey) > 0.02:
                        # the stick grows while nothing moves: a fixed 0.45 stood 110 s (w37)
                        s["lx"] = -stick_mag(bar, t, base=WALK_MIN) * (1 if ey > 0 else -1)
                    else:
                        s["ly"] = -stick_mag(bar, t, base=CREEP)    # raised while the mug stands still in the image
                        s["lx"] = -max(-0.2, min(0.2, 8.0 * ey))
                else:
                    st["twist"], st["twist_i"], st["bump"], st["stall_bar"] = 0.0, 0.0, 0.0, None
                    st["after_fine"], st["recreeps"], st["pulses"] = "creep2", 0, 0
                    goto("fine", t)
        elif name == "creep2":
            if bar is None:
                goto("backoff", t)
            else:
                ex, ey = grip_target(bar, closing) - mouth[:2]
                # No twist while walking (it turns the body), and no frozen twist to
                # carry either: the line is kept by strafe pulses, the jaw being still
                # ahead of the bar, and the twist at depth then has only the last
                # 2-3 cm to do. A 3 cm pre-bias against the veer gambled on a veer
                # that comes some runs and not others (auto5: three misses to the left).
                if ex > 0.03:
                    hold_heading(s, t, start=0.08, stop=0.03)
                # The creep stops CREEP_STOP short of the depth: the gait coasts 1.6-2.9 cm
                # after the stick lets go (w83/w84, every creep), and stopped at -0.005 the
                # bar sat 2-3 cm past the mouth, the finger shut behind it, and the recentre
                # ran round and round (overshoot, back, overshoot).
                if ex > CREEP_STOP:
                    if abs(ey) > 0.025 and ex > 0.02:
                        s["lx"] = -stick_mag(bar, t, base=WALK_MIN) * (1 if ey > 0 else -1)
                    else:
                        s["ly"] = -stick_mag(bar, t, base=CREEP)    # a 0.5 stick stood still for 30 s (auto3)
                else:
                    st["after_fine"] = "grasp"
                    st["twist"], st["twist_i"], st["close_ok_since"] = 0.0, 0.0, None
                    goto("fine2", t)
        elif name in ("fine", "fine2"):
            if bar is None:
                goto("backoff" if name == "fine2" else "search", t)
                return finish(s)
            ex, ey = grip_target(bar, closing) - mouth[:2]
            if name == "fine":
                # Standing 6 cm short of the bar: the heading first, then strafe pulses
                # onto the line. No twist here -- it would be carried into the creep.
                if since < 2.5:
                    e = hold_heading(s, t, want=HEADING_BIAS + st.get("veer_adj", 0.0), start=0.04, stop=0.015)
                    if st["turn"] != 0 or abs(e) > 0.04 or since < 0.3:
                        return finish(s)
                # The precision lives HERE, 6 cm short with the jaw clear of the mug, not at
                # depth: the final creep veers +9 deg/s (32 of 33 runs, sd 5; HEADING_BIAS
                # cancels its mean) and takes no sideways correction, and at depth a pulse
                # shoves the mug (w68: 4 cm). Handing over at 2 cm against a 2.6 cm window at
                # depth was the error budget's hole; FINE_EY is what the window allows.
                if since > FINE_MAX_S:
                    # (no end otherwise: w80 strafed in the dead band for 100 s. Close enough
                    # goes on; wide of it steps back and starts the stage over, like recentre)
                    if abs(ey) < 0.02:
                        log(f"\n[auto] fine's strafe out of time at ey={ey * 100:+.1f} cm: creeping anyway")
                        st["twist"], st["twist_i"], st["bump"], st["stall_bar"] = 0.0, 0.0, 0.0, None
                        goto(st["after_fine"], t)
                    elif abs(ey) > 0.10 or st.get("recentres", 0) >= 3 or not RECENTRE:
                        # (the bar half a metre off the jaw is not the mug under it: look again)
                        log(f"\n[auto] fine's strafe out of time at ey={ey * 100:+.1f} cm: looking for the mug again")
                        st["bar"], st["recentres"], st["resight"] = None, 0, True
                        goto("backoff", t)
                    else:
                        log(f"\n[auto] fine's strafe out of time at ey={ey * 100:+.1f} cm: a step back and again")
                        st["bump"], st["stall_bar"], st["recentres"] = 0.0, None, st.get("recentres", 0) + 1
                        goto("recentre", t)
                elif abs(ey) > FINE_EY:
                    s["lx"] = -stick_mag(bar, t, base=WALK_MIN) * (1 if ey > 0 else -1)
                    st["fine_ok_since"] = None
                elif since > 0.8:
                    st["fine_ok_since"] = st.get("fine_ok_since") or t
                    if t - st["fine_ok_since"] >= CLOSE_HOLD_S:
                        st["twist"], st["twist_i"], st["bump"], st["stall_bar"] = 0.0, 0.0, 0.0, None
                        goto(st["after_fine"], t)
                return finish(s)
            # At depth the beam is inside the handle's loop and a twist turns the MUG with
            # the body (w34: the body 9 deg in 1.5 s, the mug 42, the bar closed on at the
            # jaw tip and lost on the lift; the same spin in 7 of 28 walk tests). So the
            # twist works only while the bar is still ahead of its depth; at depth the body
            # stands still and the finger's sweep takes what is left -- it moves the mug
            # the 29 mm from where the jaw lands anyway, and the 92 logged closes landed
            # +-12 mm off that and pinched.
            at_depth = name == "fine2" and ex < DEPTH_TOL + 0.01
            overshot = name == "fine2" and ex < -DEPTH_OVER
            if at_depth:
                st["twist"], st["twist_i"] = 0.0, 0.0
            else:
                st["twist_i"] += TWIST_I * ey * senses.dt
                st["twist"] = max(-0.45, min(0.45, TWIST_SIGN * (TWIST_GAIN * ey + st["twist_i"])))
            s["rx"] = st["twist"]
            if name == "fine2":
                # The twist's reach is ~13 deg of body, and when the final creep lands the
                # mouth beyond that (3 cm left of the line in sipface1) the loop sits
                # saturated. A strafe pulse moves the base the rest; the jaw is beside
                # the bar, not past it, so 2-3 cm sideways is safe either way.
                if t < st["pulse_until"]:
                    s["lx"] = -WALK_MIN * st["pulse_dir"]
                    s["rx"] = 0.0                       # a strafe under a held twist turns instead (auto5)
                elif ((abs(st["twist"]) >= 0.44 and abs(ey) > 0.01) or (at_depth and not FINE2_EY[0] < ey < FINE2_EY[1])) \
                        and since > 0.6 and st["pulses"] < (8 if at_depth else 3):
                    # (at depth the pulse is the only sideways move, the twist being off there,
                    # and a 0.3 s one gained ~5 mm: three of them left w36 3 cm off and backing
                    # away from a bar it was beside; so more of them, and a little longer)
                    # The pulse's length follows the error: a fixed 0.45 s moved ~5 cm, twice
                    # the 2.6 cm window, and eight pulses chased the error from side to side
                    # (+2.3, -2.5, +2.0, -3.0 ... w60) until the jaw knocked the mug over.
                    pulse_s = max(0.12, min(0.45, abs(ey) / 0.05 * 0.45)) if at_depth else 0.3
                    st["pulse_until"], st["pulse_dir"], st["pulses"] = (t + pulse_s, (1 if ey > 0 else -1), st["pulses"] + 1)
                    st["twist_i"] *= 0.5
                    log(f"\n[auto] twist saturated at ey={ey:+.3f}: strafe pulse {st['pulses']}")
            if name == "fine2" and est is not None and est["seen_handle"]:
                st["unseen_since"] = None
            elif name == "fine2" and est is not None and st.get("unseen_since") is None:
                st["unseen_since"] = t
            if (name == "fine2" and since > 1.0 and est is not None and not est["seen_handle"] and st["attempts"] < 3
                    and st.get("unseen_since") is not None and t - st["unseen_since"] > 0.6):
                # (gone for 0.6 s, not one blink: the twist at depth hides it for a frame or
                # two, and a single frame sent light25/28 back from a good position)
                # The bar is a guess (no handle protrudes from the cluster): the mug may
                # have turned, or something else is in the picture. Closing here caught
                # air three times in choreo4. Back off and look again.
                log("\n[auto] handle not seen at depth; backing off to look again")
                st["attempts"] += 1
                st["recreeps"] = 0
                st["resight"] = True                     # the mug may have turned: measure its handle's axis afresh
                goto("backoff", t)
            elif name == "fine2" and since > 0.8 and ex > DEPTH_TOL + 0.004 and st["recreeps"] < RECREEPS:
                # (creep again from just outside the close's window: between the window and
                # the old 0.02 nothing crept and nothing closed, and film63 stood 6 s at 1.5 cm
                # short, twice, then backed off and lost the mug)
                log(f"\n[auto] settled {ex * 100:.1f} cm short of the depth; creeping again")
                st["recreeps"] += 1
                goto("creep2", t)
            elif (since > 0.8 and (FINE2_EY[0] < ey < FINE2_EY[1] if at_depth else abs(ey) < 0.006)
                    and (name == "fine" or ex < DEPTH_TOL) and not overshot):
                # Held for CLOSE_HOLD_S, not one frame: the estimate jumps +-2 cm between
                # exposures and a single frame inside the window closed on a bar 45-65 mm
                # right of the mouth three times in w51 (the mug spun 27-37 deg each time).
                st["close_ok_since"] = st.get("close_ok_since") or t
                if name == "fine" or t - st["close_ok_since"] >= CLOSE_HOLD_S:
                    if name == "fine2":
                        st["close_ex"] = ex
                        log(f"\n[auto] closing: ex={ex:+.3f} ey={ey:+.3f} ({'at depth' if at_depth else 'short'})")
                    goto(st["after_fine"], t)
            elif name == "fine2" and st.get("close_ok_since") is not None and since > 0.8:
                st["close_ok_since"] = None
            elif RECENTRE and overshot and since > 0.4 and st.get("recentres", 0) < 3:
                # Past the depth: the creep coasts, and the finger then shuts BEHIND the bar
                # (w81: closes at 1 and -11 mm of depth shut on air; film40 three at -49).
                st["recentres"] = st.get("recentres", 0) + 1
                log(f"\n[auto] {-ex * 100:.1f} cm past the depth: recentring")
                st["recreeps"], st["pulses"], st["after_fine"] = 0, 0, "creep2"
                goto("recentre", t)
            elif RECENTRE and at_depth and since > 1.5 and st.get("recentres", 0) < 3:
                # Off the line at depth: the final creep's veer differs run to run (+2 to +15
                # deg/s) and the fixed bias cannot cancel it -- w71 landed 2.6 cm left three
                # times running. So back 5 cm, aim the next creep by the residual (the mouth
                # is 0.295 m out: a yaw of ey/0.295 moves it ey) and creep again; the full
                # back-off, align and search cost 8 s and learnt nothing.
                st["recentres"] = st.get("recentres", 0) + 1
                # (capped at +-7 deg: accumulated to -17 the jaw came at the handle aslant
                # and the finger closed on the cup's body, 42 mm, w76)
                st["veer_adj"] = max(-0.12, min(0.12, st.get("veer_adj", 0.0) + ey / 0.295))
                log(f"\n[auto] at depth {ey * 100:+.1f} cm off the line: recentring, aim {math.degrees(HEADING_BIAS + st['veer_adj']):+.1f} deg")
                st["recreeps"], st["pulses"], st["after_fine"] = 0, 0, "creep2"
                goto("recentre", t)
            elif (since > 6.0 or (RECENTRE and at_depth and since > 1.5)
                    or (name == "fine2" and since > 0.8 and ex >= DEPTH_TOL and st["recreeps"] >= RECREEPS)):
                # (at depth and out of the window there is nothing to do but back off: no
                # pulse there, w68; 1.5 s rather than 6)
                if name == "fine2" and (abs(ey) > 0.015 or ex >= DEPTH_TOL) and st["attempts"] < 3:
                    log(f"\n[auto] not on the bar (ey={ey:+.3f}, ex={ex:+.3f}); backing off, attempt {st['attempts'] + 1}")
                    st["attempts"] += 1
                    st["recreeps"] = 0
                    goto("backoff", t)
                else:
                    log(f"\n[auto] fine alignment gave up at ey={ey:+.3f}, ex={ex:+.3f}; going on anyway")
                    goto(st["after_fine"], t)
        elif name == "grasp":
            s["rx"] = st["twist"]
            if since < 0.05 and (est is None or not est["seen_handle"]) and st["attempts"] < 3:
                # Never close on a guessed bar: of 52 misses across every run, the ones that
                # shut on air had the handle "assumed" and the guess 6-10 cm from the real bar
                # (the mug had turned, its handle beside the cup); the 62 grasps had it seen in
                # 52. Back off, measure the handle's side afresh and come round to it.
                log("\n[auto] the bar is a guess here: backing off to find the handle")
                st["attempts"] += 1
                st["recreeps"], st["resight"] = 0, True
                goto("backoff", t)
            s["lt"] = min(1.0, since / 0.6)
            if since > 1.8:
                mouth_mm, want_mm = aperture_mm(fin), 2000.0 * half_y
                # Any pinch on the bar is kept: rejecting a 16 mm one and backing off spun the
                # mug 30 deg (the finger had the bar; opening and reversing dragged it, w5:
                # myaw -24 -> -54) and the next two attempts shut on air. One shallow carry
                # lost (beat3j) is the better trade.
                deep_enough = want_mm + 12.0
                if fin < 0.09 and 0.75 * want_mm <= mouth_mm <= deep_enough:   # 5 mm on an 8 mm bar was a corner (auto29)
                    log(f"\n[auto] grasped: finger at {fin:+.3f}, mouth {mouth_mm:.0f} mm for a {want_mm:.0f} mm bar")
                    st["grasp_yaw"] = senses.yaw             # the handle's line: the room walk's heading (beat 3)
                    # The drink is gated on the bar's DEPTH at the close, not the finger's
                    # reading: in probe_pinch's drink runs (12 trials, 2026-10-14) every bar
                    # seated >= 16 mm behind the mouth came through the fold, tip and unfold
                    # (tilt <= 24) and both at 9 and 15 mm swung up and out at the fold, while
                    # the finger read 15.6-19.1 mm in both groups. `close_ex` is the estimate's
                    # ex at the close (depth = hold - ex); the finger still rules out a close
                    # on the cup's body (> 19 mm).
                    # ... and then Andrew: "none of them sip any more" -- a take without the
                    # drink is worth nothing to the film, a drink that drops the mug fails the
                    # take either way, so the drink is never skipped (2026-10-14 ~04:00).
                    st["sip_ok"] = True
                    if not st["sip_ok"]:
                        log(f"\n[auto] a shallow pinch ({mouth_mm:.0f} mm): no drink this time")
                    goto("withdraw", t)
                elif st["attempts"] >= 3:
                    log(f"\n[auto] third miss (finger {fin:+.3f}, mouth {mouth_mm:.0f} mm); carrying on with what it has")
                    goto("withdraw", t)
                else:
                    st["attempts"] += 1
                    log(f"\n[auto] missed (finger {fin:+.3f}, mouth {mouth_mm:.0f} mm of {want_mm:.0f}); backing off, attempt {st['attempts']}")
                    goto("backoff", t)
        elif name == "recentre":
            s["ly"] = +0.5
            if since > RECENTRE_S:
                st["twist"], st["twist_i"], st["bump"], st["stall_bar"] = 0.0, 0.0, 0.0, None
                # (back in through the creep: straight to `fine` the strafe timed out, backed
                # again, and the loop walked the crab 2.5 m away from the mug, film44)
                goto("creep", t)
        elif name == "backoff":
            s["ly"] = +0.5
            if since > (2.0 if st.get("resight") else 1.0):
                if st.get("resight"):
                    # (heading_ref is kept: the search replaces it once the handle shows, and
                    # the orbit needs the old one to know the short way round)
                    st["resight"], st["axis_samples"], st["hidden_since"] = False, [], None
                    goto("search", t)
                else:
                    goto("align", t)
        elif name == "withdraw":
            s["lt"] = 1.0
            if mapper is not None:
                mapper.soft.pop("mug", None)          # (in the claw now: the standing mug's cells are gone with it)
            s["rx"] = st["twist"]
            st["bump"], st["stall_bar"] = 0.0, None
            # Beat two lifts only to level: the nose up at LIFT_TO rotates the jaw and a
            # shallow pinch (17 mm) let go of the mug right there (beat2s, turn test).
            lift_to = NAV_NOSE if BEAT == 2 else LIFT_TO   # (beat 3 lifts for the sip)
            nose0 = st.get("lift_from", -NOSE_DOWN)             # (the seeded start lifts from its own nose)
            s["ry"] = nose0 + (lift_to - nose0) * min(1.0, since / LIFT_S) if LIFT else nose0
            # The drink waits a further 1.5 s with the mug up: of 30 drinks, the three that lost
            # the mug all began with it hanging 22 deg or more; under 20 it held every time but one.
            if since > (LIFT_S + 0.5 + (1.5 if SIP and BEAT != 2 else 0.0) if LIFT else 0.5):
                if BEAT == 2:
                    st["nav_ref"] = 0.0                    # the room's line: the heading it woke up with (the gyro's zero)
                    st["nav_bump"] = 0.0
                    goto("nav", t)
                elif BEAT == 3:
                    # A shallow pinch skips the drink but not the delivery (w16 went down beat
                    # one's chain and set the mug on the sand)
                    goto("sip" if SIP and LIFT and st.get("sip_ok", True) else "look", t)
                else:
                    goto("sip" if SIP and LIFT and st.get("sip_ok", True) else "carry", t)
        elif name == "nav":
            # Through the room with the mug, following the gaps: each column of the depth
            # image gives the nearest object past the carried mug; a run of columns free
            # beyond NAV_FREE wide enough for the body is a gap, and the crab steers for
            # the gap nearest the room's line (the gyro heading it woke up with), walking
            # when that bearing is at the nose and turning in place when it is not. A
            # blocked-nose rule alone turned away, called the side "clear" and turned
            # straight back into the pillar (beat2c). The tray ends it: a low, wide
            # object inside 1.2 m.
            s["lt"] = 1.0
            senses.max_range = FLAG_RANGE
            scan = senses.scan(near=NAV_NEAR, min_h=0.12)
            if scan is not None:
                cols = scan["cols"]
                n_cols = len(cols)
                u = ((np.arange(n_cols) + 0.5) / n_cols * 2 - 1) * HALF_X
                bearings = np.arctan2(-u, 1.0)                       # column 0 is the robot's left
                e = _wrap(st["nav_ref"] - senses.yaw)                # the line, off the nose
                anchor = flag_anchor(t)
                flags = [c for c in senses.scan(near=NAV_NEAR, far=FLAG_RANGE)["clusters"] if is_flag(c, anchor)]
                f = pick_flag(flags, t, anchor) if flags else None
                if f is not None:
                    # The flag seen: it is the line from here on (the gyro line only points
                    # into the room), and close enough it is the tray.
                    st["flag"], st["flag_t"] = f, t
                    if not st.get("flag_seen"):
                        st["flag_seen"] = True
                        log(f"\n[auto] the flag: {f['range']:.2f} m at {math.degrees(f['bearing']):+.0f} deg{flag_truth()}")
                elif st.get("flag") is not None and t - st.get("flag_t", -9.0) < 3.0:
                    st["flag"]["x"] -= CREEP_SPEED * st["cmd_v"] / CREEP * senses.dt    # dead-reckoned while hidden
                    st["flag"]["range"] = float(np.hypot(st["flag"]["x"], st["flag"]["y"]))
                    st["flag"]["bearing"] = float(math.atan2(st["flag"]["y"], st["flag"]["x"]))
                if st.get("flag") is not None and t - st.get("flag_t", -9.0) < 3.0:
                    e = st["flag"]["bearing"]
                tray = [st["flag"]] if (st.get("flag") is not None and t - st.get("flag_t", -9.0) < 3.0
                                        and st["flag"]["range"] < 1.1) else []
                if not tray and mapper is not None and "flag" in mapper.fixed:
                    # The flag known: inside 1.1 m of where the map puts it the approach
                    # begins whether or not the pole is in the picture (the carried mug
                    # hides a pole dead ahead, beat2k)
                    kb = mapper.mark_body("flag")
                    if math.hypot(*kb) < 1.1:
                        tray = [dict(x=float(kb[0]), y=float(kb[1]), range=float(math.hypot(*kb)),
                                     bearing=float(math.atan2(kb[1], kb[0])))]
                free = cols > NAV_FREE
                # The runs of free columns, each shrunk by the clearance the body needs at
                # the range of the blocked column beside it (an edge at the picture's side
                # costs nothing): what is left is where the body fits.
                gaps, i = [], 0
                while i < n_cols:
                    if free[i]:
                        j = i
                        while j < n_cols and free[j]:
                            j += 1
                        if j - i >= NAV_GAP_COLS:
                            r_left = float(cols[i - 1]) if i > 0 else None          # the blocked column on the left
                            r_right = float(cols[j]) if j < n_cols else None
                            hi = float(bearings[i]) - (math.atan2(NAV_CLEAR, max(0.3, r_left)) if r_left is not None else 0.0)
                            lo = float(bearings[j - 1]) + (math.atan2(NAV_CLEAR, max(0.3, r_right)) if r_right is not None else 0.0)
                            if hi >= lo:
                                gaps.append((i, j, lo, hi))
                        i = j
                    else:
                        i += 1
                nearest = float(np.min(cols)) if np.isfinite(cols).any() else 9.0
                # Blind = the floor rows read nothing: with the nose up most of the picture
                # is sky anyway (beat2f backed off from the open desert for 90 s on a
                # whole-picture count); pressed to a pillar every row is invalid (beat2e).
                blind = float((senses.last["range"][-10:] > 0).mean()) < 0.3
                flag_known = st.get("flag") is not None and t - st.get("flag_t", -9.0) < 3.0
                # "In the path" is judged over the width the body needs at the obstacle's
                # range (half the body plus a pillar's half plus clearance, 0.36 m), not a fixed
                # 12 deg: a 1 s sidestep left the pillar 0.5 m off the nose and the line pull then
                # strafed straight back into it (w2).
                # ... and judged about the LINE's bearing, not the nose: a strafe turns the
                # body (~5 deg/s at 0.5), so judged at the nose the pillar stayed "in the
                # path" as the crab orbited it, the sidestep ran 9 s and the mug ratcheted to
                # 90 deg at nose 0.2 (w36). The heading hold brings the nose back to the line
                # before the walk, so the line is what the body will go along.
                c_line = int(round((-math.tan(max(-1.0, min(1.0, e))) / HALF_X + 1.0) / 2.0 * n_cols))
                c_line = max(6, min(n_cols - 6, c_line))
                r_ahead = float(np.min(cols[c_line - 6: c_line + 6]))
                half = math.atan2(NAV_PATH_HALF, max(0.4, min(r_ahead, 0.9)))
                wide = int(round(half / (2 * HALF_X / n_cols) * 1.0))
                nose = cols[max(0, c_line - wide): min(n_cols, c_line + wide)]
                in_path = float(np.min(nose)) < 0.9
                # (a NEW dodge only for what is still NAV_DODGE_FAR out, or dead ahead within
                # the body's own width: the pillar beside the front legs at 0.4 m is being
                # passed, and sidestepping from it walked w58 into the dunes)
                narrow = max(1, int(round(math.atan2(0.18, max(0.3, r_ahead)) / (2 * HALF_X / n_cols))))
                a0 = max(0, min(n_cols - 1, c_line - narrow))
                dead_ahead = float(np.min(cols[a0: max(a0 + 1, min(n_cols, c_line + narrow))])) < 0.9
                new_dodge = in_path and (float(np.min(nose)) >= NAV_DODGE_FAR or dead_ahead)
                side_cols = (cols[:n_cols // 2 - 4], cols[n_cols // 2 + 4:])        # left half, right half (col 0 is left)
                if tray:
                    log(f"\n[auto] the tray: its flag {tray[0]['range']:.2f} m at {math.degrees(tray[0]['bearing']):+.0f} deg")
                    goto("tray_approach", t)
                elif BEAT == 3 and mapper is not None:
                    # The carry on the MAP (tools/fetch/mapper.py): the goal is the flag where
                    # it was last seen (a world point, so losing sight of it changes nothing)
                    # or, before any sighting, three metres on along the room's line; A* over
                    # the inflated occupancy grid gives the way round whatever stands in the
                    # path, and the walk follows it with heading pulses -- no sidesteps, no
                    # "in the path" rule, no re-dodging a pillar already passed (w58).
                    if f is not None:
                        flag_seen(f, t)
                    goal = mapper.marks.get("flag") or (mapper.x + 3.0, mapper.y)
                    if "flag" in mapper.marks:
                        # Not the pole itself but the STAND: TRAY_STAND_BACK before where the
                        # base will be when the mouth is over the mug's spot, on the room's
                        # line, so the last stretch is walked facing the flag. Aimed at the
                        # pole, the path came in from the side and the crab placed the mug
                        # turned 74 deg to the room with its face 0.2 m from the pole (film73).
                        fx, fy = mapper.marks["flag"]
                        goal = (fx - FLAG_BACK - TRAY_REACH - TRAY_STAND_BACK, fy - FLAG_SIDE - MOUTH_LEFT)
                    # The plan is kept for PLAN_HOLD_S once made (re-made sooner only when its
                    # point is reached or lost): re-planned twice a second from a jittering
                    # pose, the path flipped sides of the pillar ahead and the crab turned
                    # left and right for 80 s without a step (w75).
                    pt = st.get("plan_pt")
                    reached = pt is not None and math.hypot(*mapper.to_body(*pt)) < 0.25
                    if pt is None or reached or t - st.get("plan_t", -9.0) >= PLAN_HOLD_S:
                        st["plan_t"], st["plan_pt"] = t, mapper.next_point(goal, lookahead=1.0)
                    b = None if st.get("plan_pt") is None else mapper.bearing_to(*st["plan_pt"])
                    if b is None:
                        b = _wrap(st["nav_ref"] - senses.yaw)
                        st["hud_note"] = "no path on the map: the line"
                    if abs(nearest - st.get("nav_near_ref", -9.0)) > 0.02 or nearest > 5.0:
                        st["nav_near_ref"], st["nav_near_t"] = nearest, t
                    elif t - st["nav_near_t"] > 1.0:
                        st["nav_bump"], st["nav_near_t"] = min(0.25, st.get("nav_bump", 0.0) + 0.05), t
                        log(f"\n[auto] nav stalled; stick now {NAV_STICK + st['nav_bump']:.2f}")
                    # (turn only past 17 deg: with the mug the turn pulses barely turn, and a
                    # path bending round the pillar kept the bearing at 20-50 deg and the crab
                    # turning in place for 90 s, film42; the inflation leaves 15 cm for a walk
                    # that is a little off the path)
                    if st["turn"] == 0 and abs(b) > 0.30:
                        st["turn"] = 1 if b > 0 else -1
                    elif st["turn"] != 0 and (abs(b) < 0.08 or (b > 0) != (st["turn"] > 0)):
                        st["turn"] = 0
                    if st["turn"] != 0:
                        turn_in_place(s, st["turn"], t)
                        st["hud_note"] = f"the path bears {math.degrees(b):+.0f} deg: a turn pulse"
                    else:
                        s["ly"] = -(NAV_STICK + st.get("nav_bump", 0.0))
                        st["hud_note"] = "along the planned path" + ("" if "flag" in mapper.marks else ", looking for the flag")
                    if t - st.get("nav_log_t", -9.0) >= 1.0:
                        st["nav_log_t"] = t
                        tp = truth_pose() if TRUTH else None
                        log(f"\n[auto] nav {st['hud_note']}; ly={s['ly']:+.2f} rx={s['rx']:+.2f} nearest {nearest:.2f} "
                            f"goal {goal[0]:+.2f},{goal[1]:+.2f} at {mapper.x:+.2f},{mapper.y:+.2f}"
                            + (f" (truly {tp[0]:+.2f},{tp[1]:+.2f}, yaw {math.degrees(tp[2]):+.0f} vs {math.degrees(mapper.yaw):+.0f})"
                               if tp else ""))
                elif BEAT == 3:
                    # The film's carry: a gyro heading hold toward the flag (the walk itself
                    # drifts the heading 10-17 deg/s with the arm out, and without the hold the
                    # crab circled 120 deg, w13), the one sidestep round what stands in the
                    # path, always to the right, then straight on. No other strafes: a strafe
                    # turns the crab as a side effect and loses the flag (w9-w11).
                    if f is not None:
                        # The heading at the flag, from THIS frame's sighting only: taken from
                        # the stored bearing, which is stale while the flag is merely "known",
                        # the target moved with the crab and the heading pulse spun it 90-200
                        # deg chasing a constant -20 deg error (w29/w30).
                        st["nav_ref"] = senses.yaw + f["bearing"]
                    e_line = _wrap(st["nav_ref"] - senses.yaw)
                    if abs(nearest - st.get("nav_near_ref", -9.0)) > 0.02 or nearest > 5.0:
                        st["nav_near_ref"], st["nav_near_t"] = nearest, t
                    elif t - st["nav_near_t"] > 1.0:
                        st["nav_bump"], st["nav_near_t"] = min(0.25, st.get("nav_bump", 0.0) + 0.05), t   # (0.6 stood 60 s, w19)
                        log(f"\n[auto] nav stalled; stick now {NAV_STICK + st['nav_bump']:.2f}")
                    walk = -(NAV_STICK + st.get("nav_bump", 0.0))
                    # The sidestep runs at least NAV_DODGE_S and while the thing stays in the
                    # path: a 1 s one (0.15-0.2 m at 0.5) cleared the pillar in w33 and wedged
                    # the crab against it in w34 -- the strafe turns the body a little, the
                    # pillar reads "off the path" early, and the walk goes straight into it.
                    if new_dodge and st.get("dodge_until", -1.0) < t - 3.0:
                        st["dodge_until"], st["dodge_t"] = t + NAV_DODGE_S, t
                        log(f"\n[auto] something in the path at {float(np.min(nose)):.2f} m: sidestepping right")
                    if in_path and t - st.get("dodge_t", -9.0) < NAV_DODGE_MAX:
                        st["dodge_until"] = max(st.get("dodge_until", -1.0), t + 1.0)
                    if t < st.get("dodge_until", -1.0):
                        s["lx"] = +0.5
                        st["hud_note"] = "sidestepping right round it"
                    else:
                        # heading pulses (on past 6 deg, off inside 2) and the walk otherwise
                        if st["turn"] == 0 and abs(e_line) > 0.10:
                            st["turn"] = 1 if e_line > 0 else -1
                        elif st["turn"] != 0 and (abs(e_line) < 0.03 or (e_line > 0) != (st["turn"] > 0)):
                            st["turn"] = 0
                        if st["turn"] != 0:
                            turn_in_place(s, st["turn"], t)
                            st["hud_note"] = f"heading {math.degrees(e_line):+.0f} deg off the flag: a turn pulse"
                        else:
                            st["tip_t"] = None
                            s["ly"] = walk
                            st["hud_note"] = ("to the flag" if flag_known else "walking on, looking for the flag")
                elif BEAT == 3:
                    pass
                elif blind or float(np.min(cols[n_cols // 2 - 6: n_cols // 2 + 6])) < 0.45:
                    # The sensor sees nothing (pressed against a pillar, beat2e: black frames for
                    # 80 s) or something is under the nose: a step back, turning away.
                    s["ly"] = +0.5
                    left, right = cols[:n_cols // 2], cols[n_cols // 2:]
                    turn_left = float(np.mean(np.minimum(left, 2.0))) > float(np.mean(np.minimum(right, 2.0)))
                    s["rx"] = NAV_TURN * (1 if turn_left else -1) * TWIST_SIGN
                    st["hud_note"] = f"{'blind' if blind else 'too close'}: backing off, turning {'left' if turn_left else 'right'}"
                    # (too close = under the nose only: a pillar passing 0.2 m off the side read as
                    # "nearest 0.42" at the picture's edge and backed the crab off six times, beat2l)
                elif gaps:
                    st["nogap_t"] = None
                    # the gap nearest the line; inside it, the bearing nearest the line
                    best = min(gaps, key=lambda g: abs(_wrap(min(max(e, g[2]), g[3]))))
                    lo, hi = best[2], best[3]
                    want = min(max(e, lo), hi)
                    # The line beyond the picture's edge on an open side: the clamp hid a
                    # 96 deg error behind a 27 deg want and the walk never turned (beat2n).
                    if abs(e) > 0.5 and ((e > 0 and best[0] == 0) or (e < 0 and best[1] == n_cols)):
                        want = e
                    st["nav_want"] = want
                    # Turn pulses with hysteresis (on past 0.14 rad, off inside 0.05), standing;
                    # walk when the want is at the nose.
                    if st["turn"] == 0 and abs(want) > 0.14:
                        st["turn"] = 1 if want > 0 else -1
                    elif st["turn"] != 0 and (abs(want) < 0.05 or (want > 0) != (st["turn"] > 0)):
                        st["turn"] = 0
                    if st["turn"] != 0:
                        turn_in_place(s, st["turn"], t)
                    else:
                        st["tip_t"] = None
                        if abs(nearest - st.get("nav_near_ref", -9.0)) > 0.02 or nearest > 5.0:
                            st["nav_near_ref"], st["nav_near_t"] = nearest, t      # (nothing in view: no stall to judge)
                        elif t - st["nav_near_t"] > 1.0:
                            st["nav_bump"], st["nav_near_t"] = min(0.1, st.get("nav_bump", 0.0) + 0.05), t   # 0.6 at most: harder walks shake the mug round
                            log(f"\n[auto] nav stalled; stick now {NAV_STICK + st['nav_bump']:.2f}")
                        s["ly"] = -(NAV_STICK + st.get("nav_bump", 0.0))
                    st["hud_note"] = (f"gap {math.degrees(float(bearings[best[0]])):+.0f}..{math.degrees(float(bearings[best[1] - 1])):+.0f}"
                                      f" -> {math.degrees(want):+.0f} deg, line {math.degrees(e):+.0f}")
                else:
                    # No gap in view: turn in place toward the side with the farther columns,
                    # and keep that way for 4 s once chosen (flipping every second as the view
                    # changed stood the crab between two pillars for two minutes, film3).
                    left, right = cols[:n_cols // 2], cols[n_cols // 2:]
                    turn_left = float(np.mean(np.minimum(left, 2.0))) > float(np.mean(np.minimum(right, 2.0)))
                    if st.get("nogap_t") is None or t - st["nogap_t"] > 4.0:
                        st["nogap_t"], st["nogap_left"] = t, turn_left
                    turn_left = st["nogap_left"]
                    s["rx"] = NAV_TURN * (1 if turn_left else -1) * TWIST_SIGN
                    if nearest < 0.5:
                        s["ly"] = +0.45                                   # too close: a step back while turning
                    st["hud_note"] = f"no gap: turning {'left' if turn_left else 'right'}, nearest {nearest:.2f} m"
                if t - st.get("nav_log_t", -9.0) >= 1.0:
                    st["nav_log_t"] = t
                    log(f"\n[auto] nav {st.get('hud_note', '')}; ly={s['ly']:+.2f} rx={s['rx']:+.2f} nearest {nearest:.2f} "
                        f"clusters={[(round(c['range'], 2), round(c['height'], 2), round(c['width'], 2)) for c in scan['clusters']][:4]}")
            want_nose = NAV_STRAFE_NOSE if s["lx"] != 0.0 else NAV_NOSE
            st["nav_nose"] = st.get("nav_nose", NAV_NOSE) + max(-0.002, min(0.002, want_nose - st.get("nav_nose", NAV_NOSE)))
            s["ry"] = st["nav_nose"]
            if since > 90.0:
                log("\n[auto] no tray found in 90 s; setting the mug down here")
                goto("deliver", t)
        elif name == "tray_approach":
            # To the tray by strafing, not turning: the turn stick stood the crab still
            # for 25 s twice at this point (beat2s/t/u) and, with the nose up, swung the
            # mug round; a strafe at 0.5 kept it (strafe test, nose 0.2). The tray's
            # centre is found from the pole, the mouth (left of the nose) is walked over
            # it, and the nose goes up only for the last few centimetres.
            s["lt"] = 1.0
            if st.get("tray_lift_t") is not None:
                # slowly: a 1 s lift to 0.25 shook out a mug hanging 33 deg (w20); 15 deg held
                s["ry"] = NAV_NOSE + (TRAY_LIFT_NOSE - NAV_NOSE) * min(1.0, (t - st["tray_lift_t"]) / 2.5)
            else:
                s["ry"] = st.get("nav_nose", NAV_NOSE)        # (set below: up a touch while strafing, w4)
            senses.max_range = FLAG_RANGE
            scan = senses.scan(near=NAV_NEAR, far=1.8)
            anchor = flag_anchor(t)
            flags = [c for c in (scan["clusters"] if scan else []) if is_flag(c, anchor)]
            f = pick_flag(flags, t, anchor) if flags else None
            if f is not None:
                st["flag"], st["flag_t"] = f, t
                flag_seen(f, t)
            if f is not None and mapper is not None and "flag" in mapper.fixed:
                bx, by = mapper.mark_body("flag")             # (the mark is the truth; the sighting moved the pose)
                st["flag"] = dict(f, x=float(bx), y=float(by))
            elif mapper is not None and "flag" in mapper.marks and f is None:
                # Hidden (the pole leaves the picture inside NAV_NEAR, and the mug hides it):
                # the MAP's world point, turned and moved with the pose. The body-frame
                # dead reckoning below only slid it back as the crab walked and never turned
                # it; after the 35 deg turn to face the spot the pole it walked at was 0.6 m
                # from the real one, and the crab pushed its face into the flag (film52).
                bx, by = mapper.mark_body("flag")
                st["flag"] = dict(st.get("flag") or {}, x=float(bx), y=float(by))
            elif f is None and st.get("flag") is not None:
                st["flag"]["x"] -= CREEP_SPEED * st["cmd_v"] / CREEP * senses.dt    # dead-reckoned while hidden
                st["flag"]["y"] -= CREEP_SPEED * st.get("cmd_vy", 0.0) / CREEP * senses.dt
            c = st.get("flag")
            if c is None:
                goto("nav", t)
            else:
                target = tray_centre(c)
                ex, ey = target[0] - TRAY_REACH, target[1] - MOUTH_LEFT
                # The heading held on the gyro through the approach as through the room walk:
                # without it the walk-in drifted 35 deg over the last 80 cm and the mug went
                # down 33 cm beside the mat (w24). The reference, while the flag is seen, is
                # the bearing of the MUG'S SPOT (the mouth over the tray's centre), not the
                # pole's: aimed at the pole, the lateral 24 cm was left to the strafe, which
                # stands in the dead band, and the mug went down 12-20 cm right of its spot
                # in every placing -- 4-12 cm from the pole, leaning on it or off the mat's
                # corner (w47/w48/w54, film37).
                if f is not None:
                    st["nav_ref"] = senses.yaw + math.atan2(target[1] - MOUTH_LEFT, max(0.05, target[0]))
                e_line = _wrap(st.get("nav_ref", senses.yaw) - senses.yaw)
                if st["turn"] == 0 and abs(e_line) > 0.10:
                    st["turn"] = 1 if e_line > 0 else -1
                elif st["turn"] != 0 and (abs(e_line) < 0.03 or (e_line > 0) != (st["turn"] > 0)):
                    st["turn"] = 0
                if ex < TRAY_LIFT_AT and st.get("tray_lift_t") is None:
                    st["tray_lift_t"] = t
                    log("\n[auto] the mug up a touch for the tray's edge")
                if st["turn"] != 0:
                    turn_in_place(s, st["turn"], t)                            # turn first, walk after
                    st["cmd_vy"] = 0.0
                elif st.get("tray_lift_t") is not None and t - st["tray_lift_t"] < 2.5:
                    st["cmd_vy"] = 0.0                                         # standing while it rises
                elif ex < -0.12:
                    s["ly"] = +0.45                                            # well past the centre: back up first (film4)
                    st["cmd_vy"] = 0.0
                elif ex <= TRAY_STOP_EARLY:
                    # (the gait coasts ~10 cm after the stick lets go: triggered at 0 the base
                    # stopped 13 cm from the pole instead of 26 (w63), the beam's tip at the cup
                    # wall, and the retreat's nose-down pushed the mug over; so a little early)
                    # (at or a little past the centre: place. Backing up from "a little past"
                    # and walking back in oscillated for 30 s and the mug ratcheted round
                    # meanwhile, film25; the 40 cm tray takes a few centimetres either way)
                    log(f"\n[auto] over the tray (the pole {c['x']:.2f} m ahead, {ey * 100:+.0f} cm off){stop_truth(mouth)}")
                    goto("place", t)
                elif abs(ey) > 0.12 and ex > 0.15 and st.get("tray_strafe_s", 0.0) < 5.0:
                    # A strafe toward the centre line, only for what the 40 cm tray cannot
                    # absorb, from 0.5 and never past 0.65: a bumped 0.8 strafe turned the
                    # crab 65 deg and made the error worse as it went (w11); a fixed 0.5 stood
                    # still 5 s in the dead band and the mug went down 17 cm off (film37).
                    s["lx"] = -min(0.65, stick_mag(np.array([c["x"], c["y"]]), t, base=0.5)) * (1 if ey > 0 else -1)
                    st["tray_strafe_s"] = st.get("tray_strafe_s", 0.0) + senses.dt
                    st["cmd_vy"] = -s["lx"]
                elif ex > 0.0:
                    # (the gait carries on ~10 cm after the stick lets go, which puts the mug
                    # past the centre; the 40 cm tray takes the spread either way. The walk-in
                    # is capped at 0.6: a bumped 0.75 overshot to the far corner, film18)
                    s["ly"] = -min(0.6, stick_mag(np.array([c["x"], c["y"]]), t, base=0.45))
                    st["cmd_vy"] = 0.0
                else:
                    log(f"\n[auto] over the tray (the pole {c['x']:.2f} m ahead, {ey * 100:+.0f} cm off){stop_truth(mouth)}")
                    goto("place", t)
                st["hud_note"] = f"flag {c['x']:.2f} m; tray {ex * 100:+.0f} cm ahead, {ey * 100:+.0f} cm left"
                if st.get("tray_lift_t") is None:
                    want_nose = NAV_STRAFE_NOSE if s["lx"] != 0.0 else NAV_NOSE
                    st["nav_nose"] = st.get("nav_nose", NAV_NOSE) + max(-0.002, min(0.002, want_nose - st.get("nav_nose", NAV_NOSE)))
                    s["ry"] = st["nav_nose"]
                if t - st.get("nav_log_t", -9.0) >= 1.0:
                    st["nav_log_t"] = t
                    log(f"\n[auto] tray_approach {st['hud_note']}; ly={s['ly']:+.2f} lx={s['lx']:+.2f} "
                        f"{'seen' if f is not None else 'reckoned'}")
            if since > 30.0:
                goto("place", t)
        elif name == "place":
            # The mug onto the tray: the nose comes down to the set height over 2 s, the
            # trigger eases off with the mug's base just on the tray, then back away.
            # As beat one's delivery: down from the level carry over 2 s, a touch back up
            # so the mug hangs a centimetre clear, the trigger eased off in the air.
            # A still second first: the mug arrives swinging from the walk-in (beat3a/b hung
            # 30-45 deg off at the tray and were knocked over on release).
            T_SETTLE, T_DOWN, T_LIFT, T_OPEN = 1.5, 3.5, 4.1, 4.7
            if since < T_SETTLE:
                s["lt"] = 1.0
                s["ry"] = TRAY_LIFT_NOSE
            elif since < T_DOWN:
                s["lt"] = 1.0
                s["ry"] = TRAY_LIFT_NOSE - (TRAY_LIFT_NOSE - TRAY_SET_NOSE) * (since - T_SETTLE) / (T_DOWN - T_SETTLE)
            elif since < T_LIFT:
                s["lt"] = 1.0
                s["ry"] = TRAY_SET_NOSE + RELIFT * (since - T_DOWN) / (T_LIFT - T_DOWN)
            elif since < T_OPEN + RELEASE_S:
                s["lt"] = max(0.0, 1.0 - (since - T_OPEN) / RELEASE_S) if since >= T_OPEN else 1.0
                s["ry"] = TRAY_SET_NOSE + RELIFT
            elif since < T_OPEN + RELEASE_S + 1.5:
                s["ry"] = TRAY_SET_NOSE + RELIFT                # (still for 1.5 s: a mug set down from a crooked hang rocks)
            else:
                goto("tray_retreat", t)
        elif name == "tray_retreat":
            # The jaw is still inside the handle's loop at the release; backing away with it
            # level dragged the loop and lifted the mug off the tray (w11, film9). Beat one
            # backs away with the nose steeply down, the beam under the loop's lower bar:
            # the same here -- nose down over 0.8 s, standing, then back, then up.
            # The film backs off further: the spin that follows swept the mug at 0.2 m (film7).
            # Three ways out of the handle's loop, measured (AUTO_RETREAT, 2026-10-13):
            #   down: the nose to -0.75 standing, then back (beat one's; lifted the mug 25 deg
            #         on the mat, w28, where the loop sits 8 mm higher than on the sand)
            #   up:   the nose to +0.25 standing (the open claw rises beside the loop), then back
            #   stow: LB first (the hook folds the arm in), then back in the walk
            if RETREAT == "level":
                s["ry"] = TRAY_SET_NOSE + RELIFT
            elif RETREAT == "up":
                s["ry"] = TRAY_SET_NOSE + RELIFT + (0.25 - TRAY_SET_NOSE - RELIFT) * min(1.0, since / 1.5)
            elif RETREAT == "stow":
                s["ry"] = TRAY_SET_NOSE + RELIFT
                if 0.3 < since < 0.45:
                    s["buttons"] = frozenset({"LB"})
                s["hat_x"] = 0
            elif since < 0.8:
                s["ry"] = TRAY_SET_NOSE + RELIFT - (TRAY_SET_NOSE + RELIFT + 0.75) * since / 0.8
                s["ly"] = 0.0
            elif since < 2.0:
                s["ry"] = -0.75
            else:
                s["ry"] = min(0.5, -0.75 + 1.2 * (since - 2.0))
            s["lt"] = 0.0
            if since >= {"down": 0.8, "level": 0.3}.get(RETREAT, 2.0):
                # (the map says how far back is clear: film36 backed into a pillar)
                # The stow that follows folds the arm, and folded 0.26 m from the mug it knocked
                # it over (film55: the map said 0.45 m behind, the crab did not back at all).
                # So: with less than 0.7 m behind on the map, a sidestep right first (away
                # from the mug, which hangs left of the nose), then back as far as the map
                # allows, 0.30 m short of what is there.
                # The body's own size, measured from its posed model (carry2): the legs reach
                # 0.20 m behind the base, 0.26 ahead and 0.26 to each side. So "room" (the
                # nearest occupied cell behind, from the base) less BODY_REAR less a hand's
                # margin is what can be backed, and a sidestep comes first only when that is
                # under 0.25 m (0.7 of room before: a pillar 0.48 m behind, 28 cm past the
                # feet, stopped the retreat, Andrew on film70)
                room = mapper.clearance(math.pi, 1.2) if mapper is not None else 1.2
                if since < 0.1:
                    st["retreat_side"] = room - BODY_REAR - 0.10 < 0.25
                    if st["retreat_side"]:
                        log(f"\n[auto] something {room:.2f} m behind on the map: a sidestep right before backing")
                st["retreat_m"] = st.get("retreat_m", 0.0) + (CREEP_SPEED * 0.5 / CREEP * senses.dt if s["ly"] > 0 else 0.0)
                if st.get("retreat_side") and since < 2.0:
                    s["lx"] = +0.5
                elif st["retreat_m"] < min(room - BODY_REAR - 0.10, RETREAT_MAX_M):
                    s["ly"] = +0.5
                elif since - st.get("retreat_log_t", -9.0) > 2.0:
                    st["retreat_log_t"] = since
                    log(f"\n[auto] something {room:.2f} m behind on the map: backed {st['retreat_m']:.2f} m, no further")
            if since > 3.0 and TRUTH and not st.get("placed_logged"):
                st["placed_logged"] = True
                try:
                    import room as _room
                    mp = mug_entity.data.root_link_pos_w[0].cpu().numpy(); mq = mug_entity.data.root_link_quat_w[0].cpu().numpy()
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(_quat_rot(mq, np.array([0.0, 0.0, 1.0]))[2])))))
                    sx, sy = _room.TRAY_XY[0] + _room.FLAG_BACK - _room.MUG_SPOT_BACK, _room.TRAY_XY[1] + _room.FLAG_SIDE + _room.MUG_SPOT_SIDE
                    on = abs(mp[0] - _room.TRAY_XY[0]) < _room.TRAY_HALF[0] and abs(mp[1] - _room.TRAY_XY[1]) < _room.TRAY_HALF[1]
                    log(f"\n[auto] placed: the mug {(mp[0] - sx) * 100:+.0f} cm along, {(mp[1] - sy) * 100:+.0f} cm left of its spot, "
                        f"tilt {tilt:.0f}, {'ON' if on else 'OFF'} the mat")
                except ImportError:
                    pass
            if since > (4.0 if BEAT == 3 else 2.4) + (1.2 if RETREAT not in ("down", "level") else 0.0) + (2.0 if st.get("retreat_side") else 0.0):
                goto("stow", t)
        elif name == "sip":
            # The drink. The arm folds in with the body LEVEL: folded, the hand points
            # 41 deg inward, and a body pitch then turns the mug partly about its own
            # handle, which the thin bar cannot resist -- the cup swung sideways under
            # the claw and came down on its side (auto6). The mug hangs from a friction
            # pinch, so its attitude is gravity's: the cup sags along the hand's "ahead"
            # by 20-30 deg whatever the body does, and trying to tip it back by rotating
            # the body about the hinge only raised the cup's side and it slid round to
            # 48 deg and fell on the way out (auto27). So: fold level, a small nose-up
            # for the head, hold, level, unfold. Whether the mug stays is read off the
            # finger, not assumed.
            s["lt"] = 1.0
            T0, T1, T2, T3, T4, T5, T6 = 1.0, 2.4, 3.2, 3.2 + SIP_HOLD_S, 4.0 + SIP_HOLD_S, 5.0 + SIP_HOLD_S, 6.0 + SIP_HOLD_S
            if since < T0:                                   # nose down to the fold pitch
                s["ry"] = LIFT_TO + (SIP_FOLD_NOSE - LIFT_TO) * since / T0
            elif since < T1:                                 # fold in
                s["ry"] = SIP_FOLD_NOSE
                s["hat_x"] = +1
            elif since < T4:                                 # the head back (ramp, hold with a full tip, ramp)
                s["hat_x"] = +1
                if since < T2:
                    f = (since - T1) / (T2 - T1)
                    nose = SIP_FOLD_NOSE + (SIP_NOSE - SIP_FOLD_NOSE) * f
                elif since < T3:
                    # The drink itself: the nose all the way up for a moment, so the tip
                    # reads on camera (at 0.6 it did not), then back to the hold pitch.
                    u = (since - T2) / SIP_TIP_S
                    nose = SIP_NOSE + (SIP_TIP - SIP_NOSE) * math.sin(math.pi * min(1.0, u)) if u < 1.0 else SIP_NOSE
                else:
                    f = 1.0 - (since - T3) / (T4 - T3)
                    nose = SIP_FOLD_NOSE + (SIP_NOSE - SIP_FOLD_NOSE) * f
                s["ry"] = nose
            elif since < T5:                                 # unfold
                s["ry"] = SIP_FOLD_NOSE
            elif since < T6:                                 # nose back up for the carry
                s["ry"] = SIP_FOLD_NOSE + (LIFT_TO - SIP_FOLD_NOSE) * (since - T5) / (T6 - T5)
            elif since < T6 + 2.5:                           # holds the cup out: the nose up, arm straight
                s["ry"] = LIFT_TO + (HOLD_OUT_NOSE - LIFT_TO) * math.sin(math.pi * min(1.0, (since - T6) / 2.5))
            else:
                goto("look" if BEAT == 3 else "carry", t)
        elif name == "look":
            # After the drink: the nose down to the level carry, then a look left and a look
            # right (a twist each way; |rx| < 0.5 twists the posture, it does not turn).
            s["lt"] = 1.0
            if since < 1.5:
                s["ry"] = LIFT_TO + (NAV_NOSE - LIFT_TO) * since / 1.5     # (from the lift's nose either way)
            else:
                s["ry"] = NAV_NOSE
                u = since - 1.5
                if u < 1.4:
                    s["rx"] = -LOOK_TWIST * math.sin(math.pi * u / 1.4)
                elif u < 2.8:
                    s["rx"] = +LOOK_TWIST * math.sin(math.pi * (u - 1.4) / 1.4)
                elif u > 3.3:
                    # The room walk's line is the heading it has now, not the room's axis: the
                    # grasp leaves it up to 20 deg off, and turning that out with the mug in
                    # the pinch is what drops the mug (film26/27). The flag pulls it in.
                    # The heading at the grasp, not the heading now: the drink's fold and
                    # tip-back and the look turned the body 60-70 deg where it stood (w15:
                    # -9 deg at the grasp, -79 at the carry's start) and the carry walked off
                    # into the desert. The heading hold turns it back first.
                    # In the single shot the mug and the tray both stand on the line the crab
                    # woke up facing (the gyro's zero), and the heading at the grasp is where
                    # two attempts and an orbit left it: +17 deg in w38, into the scenery.
                    st["nav_ref"], st["nav_bump"] = (0.0 if BEAT == 3 else st.get("grasp_yaw", senses.yaw)), 0.0
                    log(f"\n[auto] to the tray ({math.degrees(_wrap(st['nav_ref'] - senses.yaw)):+.0f} deg back to the handle's line)")
                    goto("nav", t)
        elif name == "carry" and TURN_TEST is not None:
            # A measurement, not a move: turn in place with the mug at the nose TURN_TEST
            # for 8 s (`turn_test` logs the tilt), to find a carry that survives a turn.
            s["lt"] = 1.0
            s["ry"] = TURN_TEST
            if 1.0 < since < 9.0:
                if TURN_MODE == "strafe":
                    s["lx"] = -0.5 if since < 5.0 else +0.5          # left, then right
                else:
                    s["rx"] = TURN * TWIST_SIGN
            if since > 10.0:
                goto("done", t)
        elif name == "carry":
            s["lt"] = 1.0
            s["rx"] = st["twist"]
            s["ry"] = LIFT_TO if LIFT else -NOSE_DOWN
            s["ly"] = +0.5                          # straight back, timed: no odometry, and steering slid the prize out
            if since > CARRY_S:
                # After a drink the mug already hangs 15-30 deg off and nine seconds of
                # wiggle and toast took it to 60 (auto46): deliver, and celebrate
                # hands-free with the bow. Without the drink, the toast is the gag.
                goto("deliver" if (SIP and st.get("sip_ok", True) and DELIVER) else "victory", t)
        elif name == "victory":
            s["lt"] = 1.0
            phase = since * 2 * math.pi / 1.6
            if since < 5.0:
                # The wiggle, gentler than fetch.py's 0.8: after the drink the mug already
                # hangs 20-30 deg off, and a 12 deg roll each way shook it out (auto37).
                s["ry"] = LIFT_TO if LIFT else -NOSE_DOWN
                s["buttons"] = frozenset({"R3"})
                s["rx"] = WIGGLE * math.sin(phase)
            elif since < 9.0:
                # The finale: the nose comes up and the prize with it, as far as the pinch allows.
                s["ry"] = min(TOAST_NOSE, (LIFT_TO if LIFT else -NOSE_DOWN) + (since - 5.0) / 3.0)
            else:
                goto("deliver" if DELIVER else "done", t)
        elif name == "deliver":
            # The delivery: the nose comes down until the mug stands on the sand, then
            # comes back up a touch so the mug hangs a centimetre clear, and the trigger
            # comes off with the mug in the air: its lower bar rests on the finger, and
            # opened with the mug standing the finger's arc lifted that bar and tipped
            # the light mug over backward onto its handle (light9/11/12). From a
            # centimetre a mug hanging within 10 deg of level lands on its base.
            T_DOWN, T_LIFT, T_OPEN = SET_DOWN_S, SET_DOWN_S + 0.6, SET_DOWN_S + 1.2
            if since < T_DOWN:
                s["lt"] = 1.0
                s["ry"] = 1.0 - (1.0 + SET_DOWN_NOSE) * since / T_DOWN
            elif since < T_LIFT:
                s["lt"] = 1.0
                s["ry"] = -SET_DOWN_NOSE + RELIFT * (since - T_DOWN) / (T_LIFT - T_DOWN)
            elif since < T_OPEN + RELEASE_S:
                s["lt"] = max(0.0, 1.0 - (since - T_OPEN) / RELEASE_S) if since >= T_OPEN else 1.0
                s["ry"] = -SET_DOWN_NOSE + RELIFT
            elif since < T_OPEN + RELEASE_S + 0.5:
                s["ry"] = -SET_DOWN_NOSE + RELIFT
            else:
                goto("retreat", t)
        elif name == "retreat":
            s["ry"] = -SET_DOWN_NOSE + RELIFT
            s["lt"] = 0.0
            s["ly"] = +0.5                           # straight back, off the mug
            if since > 1.4:
                goto("stow", t)
        elif name == "stow":
            # Leave the claw mode: LB toggles back to locomotion and the hook stows the arm.
            if since < 0.15:
                s["buttons"] = frozenset({"LB"})
            s["hat_x"] = 0
            if fsm.mode == "locomotion" and since > 1.5:
                goto("circle" if BEAT == 3 else "bow", t)
            elif since > 4.0:
                log(f"\n[auto] still in {fsm.mode} after LB; bowing anyway")
                goto("circle" if BEAT == 3 else "bow", t)
        elif name == "bow":
            s["hat_x"] = 0
            if since < 0.15:
                s["hat_y"] = +1                      # d-pad down: gesture_bow, from locomotion
            if since > 6.0 and fsm.mode == "locomotion":
                goto(("sleep" if SLEEP else "sit") if BEAT == 3 else ("circle" if ENCORE else "done"), t)   # (the beckon was cut: too slow)
            elif since > 14.0:
                goto(("sleep" if SLEEP else "sit") if BEAT == 3 else ("circle" if ENCORE else "done"), t)
        elif name == "beckon":
            # To the crowd: the left arm out and up (claw_left, thumb up), stowed, then
            # the right (claw_right carries the mirror), stowed, and back to locomotion.
            s["hat_x"] = 0
            B = [(0.0, "LB"), (1.8, "up"), (4.3, "stow"), (5.3, "RB"), (7.1, "up"), (9.6, "stow"), (10.6, "RB"), (12.4, "end")]
            for i, (t0, what) in enumerate(B):
                t1 = B[i + 1][0] if i + 1 < len(B) else t0 + 1.0
                if t0 <= since < t1:
                    if what in ("LB", "RB") and since < t0 + 0.15:
                        s["buttons"] = frozenset({what})
                    elif what == "up":
                        s["hat_y"] = -1                  # thumb up: the arm straight out and up
                        s["ry"] = 0.5
                    break
            if since >= B[-1][0] and fsm.mode == "locomotion":
                goto("circle", t)
            elif since > B[-1][0] + 4.0:
                log(f"\n[auto] still in {fsm.mode} after the beckon; circling anyway")
                goto("circle", t)
        elif name == "circle":
            # Once round on the spot, closed on the gyro.
            if since < 0.2:
                st["circle_ref"] = senses.yaw
            if abs(senses.yaw - st.get("circle_ref", senses.yaw)) < 2 * math.pi - 0.15 and since < 20.0:
                s["rx"] = CIRCLE_STICK
            else:
                goto("bow" if BEAT == 3 else ("sleep" if SLEEP else "sit"), t)
        elif name == "sleep":
            # Y: gesture_sleep, from the walk; it settles, breathes and rises by itself.
            if since < 0.2:
                s["buttons"] = frozenset({"Y"})
            elif BEAT == 3 and since > SLEEP_END:
                goto("done", t)                      # (before the clip rises: a take that stops at `done` ends asleep)
            elif since > 2.0 and fsm.mode == "locomotion" or since > SLEEP_S + 3.0:
                goto("done", t)
        elif name == "sit":
            # Down to the lowest stance (R3 + stick down is the height, a speed) and sleep.
            if 0.3 < since < 3.3:
                s["buttons"] = frozenset({"R3"})
                s["ry"] = +1.0
            if since > 4.0:
                goto("done", t)
        elif name == "done":
            s["hat_x"] = 0
            if ENCORE and 0.0 <= since < 0.5:
                pass                                 # asleep
        return finish(s)

    if mapper is not None:
        inner_pilot = pilot
        mp = {"n": 0, "odom_t": -9.0}

        def pilot(t):
            g = inner_pilot(t)
            mapper.step(senses, t, g.ly, g.lx, LOCO_SCALE if player.fsm.mode == "locomotion" else 1.0)
            if TRUTH and t - mp["odom_t"] >= 3.0:
                mp["odom_t"] = t
                tp = truth_pose()
                log(f"\n[auto] odom {st['name']} {player.fsm.mode}: map {mapper.x:+.2f},{mapper.y:+.2f} yaw {math.degrees(mapper.yaw):+.0f} "
                    f"truly {tp[0]:+.2f},{tp[1]:+.2f} yaw {math.degrees(tp[2]):+.0f} (off {math.hypot(mapper.x - tp[0], mapper.y - tp[1]) * 100:.0f} cm)"
                    + (" STALLED" if mapper.stalled else ""))
            if MAP_DIR != "1" and t >= mp["n"] * MAP_DT:
                os.makedirs(MAP_DIR, exist_ok=True)
                mapper.picture(os.path.join(MAP_DIR, f"map_{mp['n']:05d}.png"))
                mp["n"] += 1
            return g

    return pilot
