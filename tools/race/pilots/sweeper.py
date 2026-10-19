"""The sweeper: find the objects on the wrong half, get behind each and shove it
over the line, nearest first.

Two ways of knowing where things are, one machine. With `side.truth` the pilot
reads the world (every object's pose and its own) -- the probe that measured
what a walking crab does to a cube, a ball, a can and a puck, and the control
for the sensed pilot. Without it, the pilot knows only what the crab carries:
the dToF image, the IMU and the encoders, read by `sense.py`'s class for its
own crab, and the room map `mapper.py` builds from them with the pose
dead-reckoned from the sticks and matched to the map (the fetch film's
machinery, one per crab). The line is where it was when the crab woke up, which
a crab is allowed to know -- it started on its own mark, facing along the line.

The shove: walk to a point `BEHIND_M` behind the object (behind as seen from my
half), turn to face my half, and walk into it with the lateral stick keeping the
object centred under the nose, until it is `OVER_M` past the line; then back
off `BACK_S` so the legs do not drag it back, and pick again. A ball rolls
ahead of the shove and is left as soon as it is over; a cube slides; a can tips
and rolls on its side. The measurements are in the README.
"""
from __future__ import annotations

import math
import os

import numpy as np
from arena import BLUE, RED
from mapper import Mapper
from moves import MOVES, WALKING, Performer
from room import CHAMFER, ROOM_X, ROOM_Y, free, side_of
from sight import Sight, is_crab, leg_velocity

from controller import GamepadState

SENSES = True

WALK = float(os.environ.get("SWEEP_WALK", "0.75"))         # the walk stick: 0.56 m/s, measured, where 0.9 was 0.7 and
PUSH = float(os.environ.get("SWEEP_PUSH", "0.75"))         # the stick into the object -- the reckoning fits better there
                                                           # (Andrew: "slow down the speed a little so that the odom is
                                                           # more accurate")
TURN = 0.62                                                # the turn stick (the battle's TURN_STICK): 57 deg/s
TURN_BAND = 0.50                                           # the turn stick moves nothing below this (probe_turn, 2026-10-16:
                                                           # -1 deg/s at 0.2-0.5, 17 at 0.52, 23, 32, 40, 49, 57 at 0.62)...
TURN_PER = 470.0                                           # ...and deg/s per unit past it, a straight line 0.52-0.62. A stick
                                                           # proportional to the error (2.0 per radian) sat in the band for
                                                           # every error under 14 degrees: a shove along the end wall held 9
                                                           # degrees into the wall, uncorrected, and the legs on the wall
                                                           # slipped (probe_fix_corner)
TURN_RATE = 3.0                                            # deg/s of turn asked per degree of heading error
BEHIND_M = float(os.environ.get("SWEEP_BEHIND_M", "0.40")) # the approach point, behind the object: the orbit's radius.
                                                           # The shell's front is 0.18 m ahead of the base, so this is
                                                           # a 0.12 m gap to a puck's edge (0.45 before: the crabs "give
                                                           # a wide berth to objects when there is plenty of room", Andrew)
ARRIVE_M = 0.15                                            # close enough to the approach point
FACE_DEG = 30.0                                            # lined up when the heading is within this (the shove's
                                                           # lateral stick takes up the rest)
TURN_FULL = 1.0                                            # the turn stick when far off the heading (0.62 is 33 deg/s)
ORBIT_IN_M = 0.7                                           # nearer the object than this, the walk becomes the orbit (1.0
                                                           # before, a circle begun a metre out)...
ORBIT_SIDE_M = 0.05                                        # the orbit strafes on while the point this far to its side is
                                                           # KERB_BODY_M from every wall: the next step, at the body's own
                                                           # clearance. (0.35 at KERB_CLEAR blocked the way round a thing
                                                           # 0.55 m from a wall, which is every thing at a wall; 0.2 at
                                                           # KERB_CLEAR stacked to 0.4 and a standing point 0.2 m from the
                                                           # end wall, the wall route's, was never reached: the shove began
                                                           # "boxed", 49-66 degrees off, probe_fix2_corner, probe_fix3_chamfer)
ORBIT_OUT_M = 0.3                                          # ...and stays the orbit until this much farther (the two flipped
                                                           # three times in a second on a thing at the radius, day24)
ORBIT_DEG = 8.0                                            # the orbit strafes until within this of the standing angle, at
                                                           # a stick proportional to the angle left (ORBIT_GAIN) rather than
                                                           # full until a 30 degree band: a shove begun 45 degrees off the
                                                           # line turned the nose off a puck 0.4 m away, took it for under
                                                           # the nose and walked into the side wall (probe_jam9); at 25 and
                                                           # full stick the crabs circled for the whole race, day10)...
BOXED_DEG = 45.0                                           # blocked by the kerb short of the standing angle by less than
                                                           # this, the shove starts from here (a right angle before: a slant
                                                           # of 49 degrees into the end wall lost the puck, probe_fix2_corner)
ORBIT_GAIN = 1.5                                           # strafe stick per radian of standing angle left (full at 28 deg)
GO_DEG = 15.0                                              # ...and the shove starts from within this of it, facing the thing,
                                                           # where the thing stays in the 27.5 degree picture while the shove's
                                                           # heading is taken up (45 before: the orbit ran on to 30 degrees and
                                                           # a 25 cm band on the radius, and a crab circled a thing and
                                                           # reversed several times before shoving, Andrew: "it should just
                                                           # go for it"; the proportional strafe closes 45 to 15 in 0.3 s)
ORBIT_S = 10.0                                             # circling this long, shove from wherever I am
PUSH_TURN_DEG = 50.0                                       # a BLIND shove this far off its heading turns on the spot first
NOSE_SEEN_M, NOSE_SIDE_M = 0.45, 0.20                      # a thing last seen nearer and more central than this went out
                                                           # of the picture under the nose; one lost farther out or to the
                                                           # side left the picture by its edge, and the orbit resumes
THING_R = 0.08                                             # the least a thing's centre stands from a wall (the square's half)
NOSE_STOP_M = 0.25                                         # a shove ends when the point this far ahead of the base is
                                                           # past the kerb's face (the nose is 0.18 ahead; the jammed
                                                           # base measures 0.18 from the face): the thing under the nose
                                                           # is then on the wall, and the next leg starts from where it
                                                           # lies
KERB_STOPS = 3                                             # ...and a thing stopped at the kerb this often without moving
KERB_STOP_MOVED_M = 0.15                                   # this far between stops is given up
END_LEG_MIN_M = 0.30                                       # near the chamfer the end wall's leg needs the thing this far
                                                           # from the end wall (see wall_route)
ROUTE_SLANT_DEG = 15.0                                     # the end wall's leg heads this much toward the room
ROUTE_CLAMP_M = 0.15                                       # a wall route's standing point may be clamped into the room
                                                           # this far (the thing is then that far off the nose's axis)
ROUTE_HYST_M = 0.10                                        # a wall route's zone is kept this much past its edge: a thing
                                                           # shoved along the chamfer rides the zone's edge
STRAFE = 0.7                                               # the orbit's stick along the circle (0.49 m/s, measured)
OVER_M = float(os.environ.get("SWEEP_OVER_M", "0.30"))     # the object is done this far past the line
BACK_S = 0.7                                               # seconds of reverse after a shove, and only after a shove or
                                                           # a shove that stuck (Andrew: the crabs "orient towards an
                                                           # object and then back up first instead of just going for it")
STALL_S = 2.5                                              # a shove that moves nothing this long is abandoned
TRY_S = 30.0                                               # a job that takes this long is abandoned
KERB_CLEAR = 0.20                                          # the body stands this far from the kerb at least: jammed on it,
                                                           # nose first, the base is 0.18 m from the face (probe_walljam)
KERB_BODY_M = 0.17                                         # the map pose is kept this far inside the kerb: the body cannot
                                                           # be nearer (0.18 measured)
JAM_INTO_M_S = 0.05                                        # at the kerb, a leg velocity into it faster than this is slip
#: (A slip test for a shove with the thing under the nose -- the legs say 10 cm of walk in
#: half a second and the kerb in view ahead comes no nearer -- was tried and dropped, day24:
#: a side wall seen at a slant keeps its median range as the crab walks beside it, and the
#: nearest kerb cluster ahead is a different one from frame to frame, so it fired on nearly
#: every shove, 49 in a race, and none landed. With the things low the kerb above a shoved
#: thing stays in view, and the wall fix, 3 cm a frame at 50 frames a second, out-runs the
#: 0.56 m/s the slipping feet report.)
CORNER_M = 0.5                                             # a kerb point within this of a corner is left out of the wall fix
                                                           # (the chamfer's points belong to neither wall)
#: (DEEP_WEIGHT, 2 m of walk saved per metre from the line, was the pick's whole order
#: before the hierarchy: it sent red from a pile of four at the line to a lone thing on
#: the far side, day30.)
STAND_MOVED_M = 0.3                                        # a standing point, once chosen, is kept for the job unless it
                                                           # stops fitting or the thing moves this far: re-choosing it
                                                           # (every 1.5 s before) flipped the slant's side and sent the
                                                           # orbit back the way it came
CLAW = os.environ.get("SWEEP_CLAW", "0") == "1"            # shove in the claw mode with the arm out (Andrew: "a walking gait
                                                           # with claws extended"): measured (probe_armpush, 2026-10-15), a
                                                           # cube slides 41 cm in 1.5 s at a 0.8 stick without tipping, a ball
                                                           # and a can roll ahead, the thing stays in the picture as a thing
                                                           # with the kerb beside it. OFF by default: in a race the switch
                                                           # into the mode is a ramp the controller ends only at the home
                                                           # pose, which lifts the arm's leg, and against a thing it never
                                                           # gets there -- the crabs spent the race "ramping" (day15/16: 4
                                                           # and 10 crossings against 62 with the shell)
CLAW_FWD_PER, CLAW_SIDE_PER = 0.42, 0.35                   # the claw gait's m/s per unit of stick (tools/fetch/mapper.py)
CLAW_WAIT_S = 2.5                                          # the longest wait for the mode to come up before shoving anyway
FACE_S = 6.0                                               # the longest turn to the camera before the dance
LOOKOUT_X = 0.9                                            # the search: this far into the other half, on the line's axis...
SCAN_S = 12.0                                              # ...then a turn on the spot this long (33 deg/s: a full circle)
APPROACH_DEG = (0.0, 30.0, 55.0, 75.0)                   # where to stand round an object: straight behind, else slanting,
                                                           # else nearly beside it, each slant tried toward the room's middle
                                                           # first; every shove keeps a component toward the line (Andrew:
                                                           # "take a little more time to ensure they are pushing objects in
                                                           # the correct direction"). A slant that drives the thing at a
                                                           # side wall within WALL_SLANT_M costs a metre more.
WALL_SLANT_M = 0.6
WALL_ROUTE_M = 0.30                                        # a thing within this of a wall is shoved ALONG it: the end wall
                                                           # to the nearer corner, the chamfer onto the side wall, the side
                                                           # wall to the line. The peel -- a slant off the wall -- had no
                                                           # standing room the body fits in, so its slants drove the thing
                                                           # into the wall and the crab jammed on it (30 give-ups in a race,
                                                           # day25); and nobody fits between a thing and the kerb
JAM_COOL_S = 4.0                                           # a thing given up jammed is picked again after this (20 s for the
                                                           # other give-ups): the next route from where it now lies is a
                                                           # different one
#: The map's dead reckoning, measured in this room (probe_speed, 2026-10-15, seed 5, an empty
#: rink, the 2 s past each phase's first second): forward 0.56 m/s at a 0.75 stick and 0.7 at
#: 0.9 (the loop's readout), 0.13 at 0.5; strafe 0.36, 0.40, 0.59 and 0.70 m/s at 0.5, 0.6,
#: 0.75 and 0.9 -- a straight line from 0.08, no dead band to speak of, where the walk has one
#: of 0.4 (a model with the walk's band on the strafe put 1.1 m/s on a 0.9 stick and ran the
#: orbiting crab's map pose three metres off, day7); turn 33 deg/s at 0.62, ~50 at 0.8 and 1.0.
STICK_BAND = 0.40
FWD_PER = float(os.environ.get("SWEEP_FWD_PER", "1.4"))      # m/s per unit of walk stick past the band
SIDE_BAND = 0.08
SIDE_PER = float(os.environ.get("SWEEP_SIDE_PER", "0.85"))   # m/s per unit of strafe stick past its band
MATCH_WIN_M = 0.0                                          # the map's scan match is off: against moving things and the
                                                           # other crab it pulled the pose 0.4-1.0 m off (sense4); the
                                                           # known kerb fixes the pose instead (`Senses.wall_fix`)
WALL_GAIN = 0.3                                            # of each frame's kerb discrepancy applied...
WALL_STEP_M = 0.03                                         # ...and no more than this a frame: a wrong fix at 0.5 a frame ran
                                                           # the pose 2 m off in 5 s (sense6)
WALL_NEAR_M = 0.6                                          # a kerb point is the wall it is nearest to, if within this (0.45
                                                           # let a pose 0.45 m off drift on unfixed, sense5)...
WALL_GROW_M_S, WALL_NEAR_MAX = 0.15, 1.2                   # ...growing by this a second while the pose is outside it, to
                                                           # this, so a pose that ran off while the kerb was hidden is
                                                           # pulled back
WALL_MIN_PTS = 10                                          # per wall, for a fix...
CORNER_OFF_M, CORNER_MIN_PTS = 0.08, 8                     # a kerb point this far off a seen wall's line into the room, short
                                                           # of the chamfer's depth, is on the chamfer; this many of them
                                                           # agreeing on the corner within...
CORNER_TOL_M, CORNER_RUN_M = 0.06, 0.12                    # ...this, over a run of at least this along the chamfer, fix the
                                                           # pose along the wall
CORNER_MAX_M = 0.8                                         # a corner residual beyond this is the other corner's (the end
                                                           # wall's two are 1.7 m apart)
YAW_ALONG_M, YAW_RANGE_M = 0.8, 0.5                        # a wall seen over this much of its length, from this far, fixes
                                                           # the heading from its slant...
YAW_GAIN, YAW_STEP, YAW_MAX = 0.1, math.radians(0.05), math.radians(8)   # ...this much of it, no more than this a frame
                                                           # (2.5 deg/s: the gyro drifts well under 0.1 deg/s), and only a
                                                           # slant within this: a larger one is something else
WALL_ALONG_M, WALL_ACROSS_M = 0.30, 0.06                   # ...spread at least this far along the wall and no more than
                                                           # this (median absolute deviation) across it: a wall is a line
ON_WALL_M = 0.05                                           # a sighting nearer a kerb line than this is not a thing (a ball
                                                           # is 6 cm to its centre, a puck 8-9.5)
FOE_CLEAR_M = 0.6                                          # a thing this near the other crab is left to it
FOE_BLOCK_M = 0.45                                         # a crab-sized cluster nearer than this ahead: I am pushing the
                                                           # other crab, the sticks move nothing, and the reckoning stops
                                                           # (red's pose ran 1.2 m off in 5 s doing exactly that, sense7)
STALE_S = 15.0                                             # a remembered thing not seen this long is forgotten wherever it is
LEG_ODOM = os.environ.get("SWEEP_LEG_ODOM", "1") == "1"    # reckon from the feet (sight.leg_velocity), not the sticks.
                                                           # Jammed on the kerb the feet slip and say 1.9 m of walk for 0.27
                                                           # moved in 5 s nose-on, 2.5 for 0.6 at a 45 degree slant, 1.8 for
                                                           # 0.6 strafing (probe_walljam, 2026-10-15, no fix): so a kerb
                                                           # cluster at the nose (sight.JAM_M) with the legs saying forward
                                                           # is a slip and reckons nothing, and a velocity into a wall the
                                                           # pose is already at loses that component (Andrew: "whenever the
                                                           # crab walks into the wall the odom loses track, because even
                                                           # though it is moving its legs they slide")
MAP_DECAY_S = 6.0                                          # the map's log-odds half-life: a thing shoved away leaves no
                                                           # obstacle behind for the planner to route round (the inset was
                                                           # a smear of where things had been, Andrew: "the costmap basically
                                                           # isn't working at all")


def stick_model(ly: float, lx: float) -> tuple[float, float]:
    v = -math.copysign(FWD_PER * max(0.0, abs(ly) - STICK_BAND), ly) if ly else 0.0
    vy = -math.copysign(SIDE_PER * max(0.0, abs(lx) - SIDE_BAND), lx) if lx else 0.0
    return v, vy


def claw_model(ly: float, lx: float) -> tuple[float, float]:
    return -CLAW_FWD_PER * ly, -CLAW_SIDE_PER * lx
MATCH_M = 0.15                                             # a sighting within this of a remembered thing is that thing (0.30
                                                           # collapsed neighbours into one memory: the things are 16-19 cm
                                                           # wide, so two side by side are 20-25 cm apart, and a pile of
                                                           # three touching was remembered as ONE thing of 131 sightings,
                                                           # probe_pile; a pile of four at the line was "2 on the wrong
                                                           # half", day30)
EXPECT_M = 0.30                                            # a cluster within this of a remembered thing stays a thing whatever
                                                           # the frame's gates say (the tracking's anti-flicker)
PAIR_W = 0.22                                              # a sighting wider than this is two things in one cluster
WALKS_M, WALKS_S = 0.30, 1.0                               # a memory that moves this far in this long walks: the other crab
                                                           # (a thing moves only under a nose)
CLUSTER_M = 0.45                                           # things this near each other are a cluster: picked first, the
                                                           # most things for one walk (Andrew: "clusters of multiple objects
                                                           # first since they are highest value")
BACK_M = 1.2                                               # a thing this far from the line is the back row: picked next,
                                                           # the dearest to fetch and the other crab's for free if left
                                                           # (Andrew: "2-3 objects scattered on the backend that never get
                                                           # serviced")
FORGET_S = 1.0                                             # a remembered thing in plain view for this long with no sighting is gone
                                                           # (2 s had the crabs walking at ghosts of things long shoved,
                                                           # Andrew: "target a ghost object that is no longer there")
STALE_WEIGHT = 0.08                                        # m of walk a thing's memory costs per second unseen, when picking
REFIND_M = 0.45                                            # a lost target is the nearest remembered thing within this
PICK_SEEN_N, PICK_SEEN_S = 2, 0.2                          # a thing is picked once seen this many times over this long: a
                                                           # sighting of one frame (a leg of the other crab, a sliver of
                                                           # floor) was picked, walked at and "not there" a second later
                                                           # (7 in day28, 6 in day27)
VIEW_DEG, VIEW_NEAR, VIEW_FAR = 18.0, 0.35, 1.5            # "plain view": inside this cone of the sensor (its half
                                                           # width is 27.5 deg; a thing 2 m off is 2-3 zones and not
                                                           # always a cluster, so the far edge stays short)
OCCLUDED_M = 0.7                                           # a thing this close to the other crab may be behind it
TRUTH_LOG_S = 5.0                                          # the sensed pilot prints its pose against the truth this often
INFLATE_M = 0.25                                           # the planner's clearance round an obstacle: the map's 0.40 was
                                                           # for pillars in a 20 m wander; a 2.6 m room of 12 things is
                                                           # all inflation at that
PLAN_DT = 0.25                                             # the plan is redrawn this often (~20 ms each)
LOOKAHEAD = 0.45                                           # the point along the plan the walk steers at
BLIND_S = 3.0                                              # shoving, the thing under the nose may leave the picture (the
                                                           # sensor's lower edge meets the floor 0.31 m out: a 5 cm puck
                                                           # against the shell is under it); it is taken to be where it
                                                           # was, moving with me, for this long
NOSE_M = 0.28                                              # where a shoved thing sits ahead of the base


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def turn_stick(err: float, cap: float = TURN) -> float:
    """The rx stick that turns toward a heading `err` radians to the left: the rate
    TURN_RATE per degree, put past the stick's dead band, no more than `cap`. (The
    pad's right stick turns right when positive.)"""
    rate = TURN_RATE * abs(math.degrees(err))
    if rate < 1.0:
        return 0.0
    return -math.copysign(min(cap, TURN_BAND + rate / TURN_PER), err)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Truth:
    """The world as it is: the probe's eyes."""

    def __init__(self, env, side):
        self.env, self.side = env, side
        self.me = env.scene[side.me]
        self.names = [o["name"] for o in side.objects]
        self.kinds = {o["name"]: o["kind"] for o in side.objects}

    def update(self, t: float, frame) -> None:
        pass

    def pose(self):
        p = self.me.data.root_link_pos_w[0]
        return float(p[0]), float(p[1]), _yaw(self.me.data.root_link_quat_w[0])

    def objects(self) -> list[dict]:
        out = []
        for name in self.names:
            p = self.env.scene[name].data.root_link_pos_w[0]
            out.append({"name": name, "kind": self.kinds[name], "x": float(p[0]), "y": float(p[1])})
        return out


class Senses:
    """What the crab carries: the dToF, the IMU and the encoders (`sight.Sight`),
    and the room map built from them (`mapper.Mapper`), anchored at the mark the
    crab woke up on so the line is where it was. Things seen are remembered in
    the map's frame and matched sighting to sighting; one in plain view and not
    seen for `FORGET_S` is dropped (it was shoved, or it was never a thing)."""

    def __init__(self, env, side):
        self.env, self.side = env, side
        x0, y0, yaw0 = side.start
        self.sight = Sight(env, side.me, yaw0)
        self.mapper = Mapper()
        self.mapper.x, self.mapper.y, self.mapper.yaw = x0, y0, yaw0
        self.mapper.inflate_m = INFLATE_M
        self.mapper.model = stick_model
        self.mapper.match_win = MATCH_WIN_M
        self.mapper.stall_test = not LEG_ODOM
        self.once = 0                                      # things remembered from one sighting and forgotten
        self.hold = None                                   # the pilot's target: the one memory allowed to move
        self.walked = []                                   # (x, y, t) where a memory walked off: the other crab
        self._feet = None                                  # (feet, stance) last step, for the leg odometry
        self.leg_v = (0.0, 0.0)
        hx, hy, c = ROOM_X / 2, ROOM_Y / 2, CHAMFER       # the kerb: the room the crab was put in, known
        self.mapper.known([((-hx + c, -hy), (hx - c, -hy)), ((-hx + c, hy), (hx - c, hy)),
                           ((-hx, -hy + c), (-hx, hy - c)), ((hx, -hy + c), (hx, hy - c))]
                          + [((sx * (hx - c), sy * hy), (sx * hx, sy * (hy - c))) for sx in (-1, 1) for sy in (-1, 1)])
        self.things: dict[str, dict] = {}
        self._n = 0
        self.truth = env.scene[side.me]
        self.last_fix = 0.0
        self.mode_fn = None                                # () -> (the controller's mode, running), set by the pilot
        self.sight.at_wall = self.at_wall
        self.sight.expected = self.expected
        self.mapper.decay_s = MAP_DECAY_S
        self.blocked = False                               # the other crab right ahead this frame
        self.jammed = False                                # the kerb at the nose
        self.fixes = [0, 0, 0, 0, 0, 0]                    # fixes landed per wall (+x, -x, +y, -y), on a corner, on the
                                                           # heading, for the log
        self.kerb_frames = 0                               # frames with kerb points at all
        self.foe_w = None                                  # where the other crab was last seen, world

    def wall_fix(self) -> None:
        """The kerb is where it is, and the heading is the gyro's: each kerb point in
        the frame, put in the world by the current pose, lands some way off the wall
        it belongs to, and the median of that per wall is how far the pose is off
        along that wall's normal. Applied at `WALL_GAIN` a frame."""
        idx = self.sight.last["kerb"]
        if len(idx) < WALL_MIN_PTS:
            return
        self.kerb_frames += 1
        t = self.sight.last["t"]
        pts = self.sight.last["pts"][idx]
        # (what is under the nose was left out, nearer than 0.25 m, while a thing cut by the
        # picture's top could be labelled kerb; the things are low now and the kerb at the
        # nose is the fix a crab jammed on it needs)
        wx, wy = self.mapper.to_world(pts)
        hx, hy = ROOM_X / 2, ROOM_Y / 2
        dists = np.stack([np.abs(wx - hx), np.abs(wx + hx), np.abs(wy - hy), np.abs(wy + hy)], axis=1)
        wall = dists.argmin(axis=1)
        near_m = min(WALL_NEAR_MAX, WALL_NEAR_M + WALL_GROW_M_S * (t - self.last_fix))
        in_corner = (np.abs(wx) > hx - CORNER_M) & (np.abs(wy) > hy - CORNER_M)
        near = (dists[np.arange(len(wall)), wall] < near_m) & ~in_corner
        dx_terms, dy_terms, yaw_terms = [], [], []
        for w, (coord, along, target) in enumerate(((wx, wy, hx), (wx, wy, -hx), (wy, wx, hy), (wy, wx, -hy))):
            sel = near & (wall == w)
            if sel.sum() < WALL_MIN_PTS:
                continue
            c = coord[sel]
            med = float(np.median(c))
            # a wall at the nose fills the picture and is as wide as it: 1.04 m per m of range
            along_m = min(WALL_ALONG_M, 0.8 * 1.04 * float(np.median(np.hypot(pts[sel, 0], pts[sel, 1]))))
            if float(np.ptp(along[sel])) < along_m or float(np.median(np.abs(c - med))) > WALL_ACROSS_M:
                continue
            (dx_terms if w < 2 else dy_terms).append((target - med, int(sel.sum())))
            self.fixes[w] += 1
            # The heading from the wall's slant: the gyro's yaw walked 6-8 degrees off in
            # 100 s (day31b, red), which puts a corner 2 m off by 0.25 m along the wall
            if float(np.ptp(along[sel])) >= YAW_ALONG_M and float(np.median(np.hypot(pts[sel, 0], pts[sel, 1]))) >= YAW_RANGE_M:
                b = float(np.polyfit(along[sel], c, 1)[0])
                e = math.atan(b) if w >= 2 else -math.atan(b)
                if abs(e) <= YAW_MAX:
                    # (a slant beyond YAW_MAX is not the heading: the kerb at a slant under the
                    # nose, or a chamfer taken for a wall, said 45 degrees, and at 0.5 degrees a
                    # frame blue's heading ran away to -104 degrees in 40 s, day32b)
                    yaw_terms.append((e, int(sel.sum())))
            # The wall's END: the chamfer's points lie off the wall's line, into the room,
            # at 45 degrees, so each one says where the corner is along the wall (along
            # minus across, or plus, by the end), and that corner is at a known place --
            # the fix along the wall that a crab working beside one side wall otherwise
            # has nothing for (0.9-1.0 m off in x, three times in six runs, day29-day31).
            # Geometry in the frame of the wall seen, not the pose's corner box: taking the
            # chamfers as walls in that box jumped the pose 1.2 m when a stretch of side
            # wall near a corner passed the chamfer's line test (day21).
            across_all = (target - coord) if target > 0 else (coord - target)
            on_chamfer = (across_all > CORNER_OFF_M) & (across_all < CHAMFER - 0.05)
            if on_chamfer.sum() >= CORNER_MIN_PTS:
                best = None
                for end in (1.0, -1.0):
                    a_end = along[on_chamfer] - end * across_all[on_chamfer]
                    med_e = float(np.median(a_end))
                    inl = np.abs(a_end - med_e) < CORNER_TOL_M
                    if inl.sum() >= CORNER_MIN_PTS and float(np.ptp(across_all[on_chamfer][inl])) >= CORNER_RUN_M \
                            and (best is None or inl.sum() > best[2]):
                        best = (end, med_e, int(inl.sum()))
                if best is not None:
                    end, med_e, n_e = best
                    known = end * ((hy if w < 2 else hx) - CHAMFER)
                    if abs(known - med_e) < CORNER_MAX_M:
                        (dy_terms if w < 2 else dx_terms).append((known - med_e, n_e))
                        self.fixes[4] += 1
        dx = sum(d * n for d, n in dx_terms) / sum(n for _, n in dx_terms) if dx_terms else 0.0
        dy = sum(d * n for d, n in dy_terms) / sum(n for _, n in dy_terms) if dy_terms else 0.0
        clip = lambda v: max(-WALL_STEP_M, min(WALL_STEP_M, WALL_GAIN * v))
        if dx_terms or dy_terms:
            # The radius resets only once the pose is back inside it: a landed fix that
            # reset it at once left a 0.9 m error one 3 cm step every two seconds
            # (probe_wallfix), which is the metre blue's pose sat at for 50 s in day4.
            if max(abs(dx), abs(dy)) < WALL_NEAR_M * 0.5:
                self.last_fix = t
            self.mapper.shift(clip(dx), clip(dy))
        if yaw_terms:
            e = sum(d * n for d, n in yaw_terms) / sum(n for _, n in yaw_terms)
            self.sight.yaw -= max(-YAW_STEP, min(YAW_STEP, YAW_GAIN * e))
            self.fixes[5] += 1

    def expected(self, bx: float, by: float) -> bool:
        """Whether a remembered thing (seen twice, lately) is within EXPECT_M of a point
        in the base frame: the sensing keeps a cluster there a thing."""
        wx, wy = self.mapper.to_world(np.array([[bx, by]]))
        wx, wy = float(wx[0]), float(wy[0])
        t = self.sight.last["t"] if self.sight.last is not None else 0.0
        return any(th["n"] >= 2 and t - th["seen"] < 2.0 and math.hypot(th["x"] - wx, th["y"] - wy) < EXPECT_M
                   for th in self.things.values())

    def at_wall(self, bx: float, by: float, within: float = 0.12) -> bool:
        """Whether a point in the base frame lies on a known wall, by the map's pose."""
        wx, wy = self.mapper.to_world(np.array([[bx, by]]))
        wx, wy = float(wx[0]), float(wy[0])
        hx, hy = ROOM_X / 2, ROOM_Y / 2
        if min(abs(abs(wx) - hx), abs(abs(wy) - hy)) < within:
            return True
        return abs((hx - abs(wx)) + (hy - abs(wy)) - CHAMFER) < within * math.sqrt(2)

    def _off_the_kerb(self, v):
        """The leg velocity with its component into a wall the pose already stands at
        removed: the body cannot go there, and the feet that say it does are slipping."""
        vx, vy = v
        c, s_ = math.cos(self.mapper.yaw), math.sin(self.mapper.yaw)
        wx, wy = c * vx - s_ * vy, s_ * vx + c * vy
        hx, hy = ROOM_X / 2 - KERB_CLEAR, ROOM_Y / 2 - KERB_CLEAR
        if (self.mapper.x >= hx and wx > JAM_INTO_M_S) or (self.mapper.x <= -hx and wx < -JAM_INTO_M_S):
            wx = 0.0
        elif (self.mapper.y >= hy and wy > JAM_INTO_M_S) or (self.mapper.y <= -hy and wy < -JAM_INTO_M_S):
            wy = 0.0
        else:
            return v
        return (c * wx + s_ * wy, -s_ * wx + c * wy)

    def foe_ahead(self):
        """The other crab's cluster, if one is right ahead (the blocked shove)."""
        if self.sight.last is None:
            return None
        for c in self.sight.last["big"]:
            # not the kerb: from 0.3 m at a slant it is crab-sized inside the picture (top 0.26,
            # 0.37 wide, 0.29 deep) and read as the other crab on every corner shove, probe_jam9
            if is_crab(c) and not c.get("kerb") and c["near"] < FOE_BLOCK_M and abs(c["y"]) < 0.3:
                return c
        return None

    def update(self, t: float, frame) -> None:
        fresh = self.sight.update(t)
        ly = float(frame.ly) if frame is not None else 0.0
        lx = float(frame.lx) if frame is not None else 0.0
        mode, running = self.mode_fn() if self.mode_fn is not None else ("locomotion", True)
        if not running:
            ly = lx = 0.0                                      # ramping between modes, the sticks move nothing
        self.mapper.model = claw_model if mode.startswith("claw") else stick_model
        if LEG_ODOM and self.sight.feet is not None:
            if self._feet is not None:
                v = leg_velocity(self._feet[0], self.sight.feet, self._feet[1], self.sight.stance, self.sight.dt,
                                 self.sight.yaw_rate, hint=(-ly, -lx))
                if v is not None:
                    self.leg_v = v
            self._feet = (self.sight.feet.copy(), self.sight.stance.copy())
            v_now = self.leg_v
            if (self.jammed or self.blocked) and v_now[0] > 0.0:
                v_now = (0.0, 0.0)                         # feet slipping on the spot: the lateral slip is as false as
                                                           # the forward, 0.54 m said for 0.08 moved. Against the other
                                                           # crab as against the kerb: the sticks were stopped there and
                                                           # the legs were not, 0.33 m of pose in half a second
                                                           # (probe_fix14_chamfer)
            v_now = self._off_the_kerb(v_now)
            self.mapper.model = lambda ly_, lx_, v=v_now: v
        if fresh:
            self.blocked = self.foe_ahead() is not None
            self.jammed = bool(self.sight.last["jam"])
            crabs = [c for c in self.sight.last["big"] if is_crab(c) and not c.get("kerb")]
            if crabs:
                c = max(crabs, key=lambda c: c["n"])
                wx, wy = self.mapper.to_world(np.array([[c["x"], c["y"]]]))
                self.foe_w = (float(wx[0]), float(wy[0]), t)
        if self.blocked and ly < 0:
            # pushing into the other crab the walk stick moves nothing; the strafe and a
            # reverse still do (zeroing those too left an orbit round the other crab
            # unreckoned, and red's pose 0.7 m off in 5 s, day8)
            ly = 0.0
        self.mapper.step(self.sight, t, ly, lx)
        # the body is inside the kerb, whatever the sticks said
        hx, hy = ROOM_X / 2 - KERB_BODY_M, ROOM_Y / 2 - KERB_BODY_M
        self.mapper.x, self.mapper.y = max(-hx, min(hx, self.mapper.x)), max(-hy, min(hy, self.mapper.y))
        if not fresh:
            return
        self.wall_fix()
        seen = set()
        sightings = []
        for d in self.sight.last["things"]:
            wx, wy = self.mapper.to_world(np.array([[d["x"], d["y"]]]))
            wx, wy = float(wx[0]), float(wy[0])
            if self.at_wall(d["x"], d["y"], within=ON_WALL_M) or abs(wx) > ROOM_X / 2 - ON_WALL_M or abs(wy) > ROOM_Y / 2 - ON_WALL_M:
                continue                                   # a thing's centre is its radius off the kerb at the least: a
                                                           # sighting on the kerb's line, or past it, is a piece of the kerb
                                                           # (things at x -2.00, the deepest of all, were picked and "not
                                                           # there"; one remembered at y -1.44, beyond the wall, was
                                                           # walked at, probe_fix13_chamfer)
            sightings.append((wx, wy, d["width"]))
        # one memory per sighting and one sighting per memory, nearest pair first: two
        # sightings at a pile both landed on the one memory nearest them (probe_pile)
        pairs = sorted((math.hypot(th["x"] - wx, th["y"] - wy), i, name)
                       for i, (wx, wy, _) in enumerate(sightings) for name, th in self.things.items())
        matched = {}
        for dist, i, name in pairs:
            if dist >= MATCH_M:
                break
            if i not in matched and name not in seen:
                matched[i] = name
                seen.add(name)
        fresh_w = []
        for i, (wx, wy, width) in enumerate(sightings):
            name = matched.get(i)
            if name is None:
                if any(math.hypot(fx - wx, fy - wy) < MATCH_M for fx, fy in fresh_w):
                    continue                               # one thing in two clusters this frame (its face and its top)
                self._n += 1
                name = f"thing{self._n}"
                self.things[name] = {"name": name, "kind": "thing", "x": wx, "y": wy, "trail": [(wx, wy, t)],
                                     "seen": t, "first": t, "n": 0, "viewed": None}
                fresh_w.append((wx, wy))
                seen.add(name)
            th = self.things[name]
            if width <= PAIR_W or th["n"] < PICK_SEEN_N:
                # (a sighting wider than one thing is two of them in one cluster: it says the
                # thing is there, not where, or it drags its memory toward the neighbour's)
                th["x"], th["y"] = 0.5 * th["x"] + 0.5 * wx, 0.5 * th["y"] + 0.5 * wy
            th["seen"], th["n"] = t, th["n"] + 1
            if t - th["trail"][-1][2] >= WALKS_S / 2:
                th["trail"] = [q for q in th["trail"] if t - q[2] <= WALKS_S] + [(th["x"], th["y"], t)]
            x1, y1, t1 = th["trail"][0]
            if name != self.hold and t - t1 >= WALKS_S * 0.9 and math.hypot(th["x"] - x1, th["y"] - y1) > WALKS_M:
                # it walks: the other crab, which at range is a low cluster the size of a thing
                # (blue walking to its corner left memories along its way and red picked them
                # as a cluster, probe_pile3). Not my target, which moves under my nose; and a
                # pile's memories wander as its clusters merge and split, 0.25 m over 6 s
                # (probe_pile4), so it is the speed that tells
                self.walked.append((th["x"], th["y"], t))
                del self.things[name]
                seen.discard(name)
        crabs = [c for c in self.sight.last["big"] if is_crab(c) and not c.get("kerb")]
        for name in list(self.things):
            th = self.things[name]
            if name in seen or any(math.hypot(th["x"] - wx, th["y"] - wy) < EXPECT_M for wx, wy, _ in sightings):
                # seen, or a sighting is near enough to be it and went to a neighbour: two
                # things touching come out as one cluster some frames (0.21 wide at 2.5 m,
                # probe_pile2), and with one sighting a frame the other memory starved and
                # was forgotten in the second
                th["viewed"] = None
                continue
            bx, by = self.mapper.to_body(th["x"], th["y"])
            r, b = math.hypot(bx, by), math.atan2(by, bx)
            hidden = any(math.hypot(c["x"] - bx, c["y"] - by) < OCCLUDED_M for c in crabs)
            if VIEW_NEAR < r < VIEW_FAR and abs(b) < math.radians(VIEW_DEG) and not hidden:
                # in plain view: forgotten after FORGET_S of it with no sighting. The clock is
                # the time IN VIEW, not the time since the sighting: a memory 3-8 s old, picked
                # and turned to, was dropped the frame it entered the cone, at 17-18 degrees
                # (every "not there" in day29 but one, 7-13 a race), before the crab had looked
                th["viewed"] = th["viewed"] if th.get("viewed") is not None else t
            else:
                th["viewed"] = None
            if t - th["seen"] > STALE_S or (th["viewed"] is not None and t - th["viewed"] >= FORGET_S):
                if th["n"] < PICK_SEEN_N:
                    self.once += 1                         # a sighting that never came again: not a thing
                del self.things[name]
        # two memories of one thing (the map shifted between sightings): keep the fresher
        names = sorted(self.things, key=lambda n: -self.things[n]["seen"])
        for i, a in enumerate(names):
            if a not in self.things:
                continue
            for b in names[i + 1:]:
                if b in self.things and math.hypot(self.things[a]["x"] - self.things[b]["x"],
                                                   self.things[a]["y"] - self.things[b]["y"]) < MATCH_M * 0.8:
                    del self.things[b]

    def pose(self):
        return self.mapper.x, self.mapper.y, self.mapper.yaw

    def objects(self) -> list[dict]:
        return [dict(th) for th in self.things.values() if th["n"] >= 2]

    def sighted(self, name: str, t: float, within: float = 0.5):
        """Seconds since the remembered thing `name` was last in a frame (None if never)."""
        th = self.things.get(name)
        return None if th is None else t - th["seen"]

    def truth_error(self):
        p = self.truth.data.root_link_pos_w[0]
        yaw = _yaw(self.truth.data.root_link_quat_w[0])
        return float(p[0]) - self.mapper.x, float(p[1]) - self.mapper.y, _wrap(yaw - self.mapper.yaw)


def make_pilot(env, player, side):
    label = side.name
    mine = RED if side.sign > 0 else BLUE
    sign = side.sign
    push_yaw = 0.0 if sign > 0 else math.pi                 # the heading that shoves toward my half
    referee = side.referee
    performer = Performer(player, side.presses, label=label)
    moves = MOVES[side.name]
    eyes = Truth(env, side) if side.truth else Senses(env, side)
    truth = eyes if side.truth else Truth(env, side)       # the log's, to say whether a thing given up was there
    claw_press = side.presses.get("claw_left") if CLAW else None
    if not side.truth:
        eyes.mode_fn = lambda: (player.fsm.mode, player.fsm.running_policy)
    st = {"intent": None, "stage": "", "target": None, "target_w": None, "since": 0.0, "cue_id": referee.cue_id,
          "face": None, "line_w": ((0.0, -ROOM_Y / 2), (0.0, ROOM_Y / 2)), "last_xy": None, "still_since": None,
          "job_since": 0.0, "shoved": [], "abandoned": {}, "kerb_stops": {}, "push_yaw": push_yaw, "frame": None, "truth_log": 0.0,
          "plan_t": -1e9, "waypoint": None, "claw_since": None, "claw_pressed": None}

    def log(t, msg):
        print(f"\n[{label}] {t:6.2f}s  {msg}")

    def walking() -> bool:
        fsm = player.fsm
        return fsm.mode in WALKING and fsm.running_policy

    def claw_toggle(t):
        """The chord that toggles the claw mode, held PRESS_S; None once it has been sent."""
        if st["claw_pressed"] is None:
            st["claw_pressed"] = t
        if t - st["claw_pressed"] < 0.3:
            return GamepadState(buttons=claw_press.buttons, hat_x=claw_press.hat_x, hat_y=claw_press.hat_y)
        return None

    def arm(frame):
        """The shove's frame with the arm held out (web up) in the claw mode, once it runs."""
        if claw_press is None or player.fsm.mode != "claw_left" or not player.fsm.running_policy:
            return frame
        return GamepadState(lx=frame.lx, ly=frame.ly, rx=frame.rx, ry=frame.ry, lt=frame.lt, rt=frame.rt,
                            buttons=frame.buttons, hat_x=-1, hat_y=0)

    def head_to(x, y, yaw, gx, gy, stick=WALK):
        """The sticks that walk toward (gx, gy): turn to face it, walk once facing."""
        err = _wrap(math.atan2(gy - y, gx - x) - yaw)
        rx = turn_stick(err, TURN_FULL)
        ly = -stick if abs(err) < math.radians(40) else 0.0
        return GamepadState(rx=rx, ly=ly)

    def orbit(x, y, yaw, o, theta_goal):
        """The matador's circling, on an object: the nose stays on it while the body
        strafes round it on a circle of `BEHIND_M`, toward the angle `theta_goal`
        (where I should stand, measured round the object), with the forward stick
        holding the radius. The way round is chosen once, the short way unless it
        runs into the kerb, and kept: a way re-chosen each frame reversed whenever the
        goal angle moved under it. Returns (sticks, in position, facing it)."""
        dx, dy = o["x"] - x, o["y"] - y
        bearing = _wrap(math.atan2(dy, dx) - yaw)
        r = math.hypot(dx, dy)
        theta = math.atan2(-dy, -dx)                           # where I stand round it, now
        e = _wrap(theta_goal - theta)
        # facing it, a larger theta is to my right (a right strafe runs me anticlockwise round it)
        if st.get("orbit_way") is None:
            st["orbit_way"] = 1.0 if e >= 0 else -1.0
        lx = math.copysign(min(STRAFE, ORBIT_GAIN * abs(e)), st["orbit_way"]) if abs(e) > math.radians(ORBIT_DEG) else 0.0
        rx_w = (math.sin(yaw), -math.cos(yaw))                 # my right, in the world
        boxed = False
        if lx:
            # the point the body goes to next, at the body's own clearance (at KERB_CLEAR the two
            # margins stacked to 0.4 m, and a thing on the end wall, whose standing point is
            # 0.2 m from it, was "boxed" from 49 degrees off and shoved slantwise into the wall,
            # probe_fix2_corner)
            ahead = lambda way: free(x + math.copysign(ORBIT_SIDE_M, way) * rx_w[0], y + math.copysign(ORBIT_SIDE_M, way) * rx_w[1], KERB_BODY_M)
            if not ahead(lx):
                # that way round runs into the kerb. Short of the goal by less than BOXED_DEG:
                # stop here and shove from here, facing it (the goal is in the corner's clearance;
                # going round the long way met the other wall, and the orbit swung between the two
                # for its whole 10 s, probe_orbitwall). Farther: the long way, and KEEP going that
                # way (flipping each frame between two blocked ways is a crab jiggling on the spot,
                # day14)...
                if abs(e) < math.radians(BOXED_DEG) or not ahead(-lx):
                    lx, boxed = 0.0, True                      # ...and boxed in both ways, shove from here
                else:
                    st["orbit_way"] = -st["orbit_way"]
                    lx = -lx
        ly = -max(-0.8, min(0.8, 2.5 * (r - BEHIND_M)))        # ahead to close, back to open
        rx = turn_stick(bearing, TURN_FULL)
        facing = abs(bearing) < math.radians(FACE_DEG)
        # the nose on it within GO_DEG as well: facing within 30 and standing within 15 began
        # a shove 42 degrees off its heading, and the nose stop read the side wall ahead
        there = facing and abs(bearing) < math.radians(GO_DEG) and r < ORBIT_IN_M and (abs(e) < math.radians(GO_DEG) or boxed)
        if os.environ.get("SWEEP_TRACE") and st.get("_orbit_log", -1e9) < st["t_now"] - 0.5:
            st["_orbit_log"] = st["t_now"]
            log(st["t_now"], f"orbit: r {r:.2f} bearing {math.degrees(bearing):+.0f} theta {math.degrees(theta):+.0f} "
                             f"goal {math.degrees(theta_goal):+.0f} e {math.degrees(e):+.0f} way {st['orbit_way']:+.0f} "
                             f"lx {lx:+.2f} ly {ly:+.2f} boxed {boxed}")
        return GamepadState(lx=lx, ly=ly, rx=rx), there, facing

    def slide_to(x, y, yaw, gx, gy, face, stick=WALK):
        """The sticks that move the body toward (gx, gy) in whatever direction that is --
        a hexapod walks sideways and backwards as well as forwards -- while the nose turns
        to `face`. The crab arrives at the approach point already facing the shove, instead
        of walking there nose-first and then pirouetting on the spot for up to five seconds."""
        dx, dy = gx - x, gy - y
        bx = math.cos(yaw) * dx + math.sin(yaw) * dy
        by = -math.sin(yaw) * dx + math.cos(yaw) * dy
        d = math.hypot(bx, by)
        if d < 1e-6:
            return GamepadState()
        ly = -stick * bx / d                                 # ahead is ly < 0
        lx = -stick * by / d                                 # left (+by) is lx < 0
        err = _wrap(face - yaw)
        rx = turn_stick(err, TURN_FULL)
        return GamepadState(ly=ly, lx=lx, rx=rx)

    def kerb_ahead(x, y, yaw):
        """Whether the point NOSE_STOP_M ahead is past a wall's face, a wall the heading
        runs INTO, more than 30 degrees to it: a shove along a side wall holds a few
        degrees into it with the thing off-centre on the wall's side, and is not at
        the kerb. The face itself, no clearance: the thing's far edge is 0.37 m ahead
        of the base, so with the thing still under the nose the base cannot be nearer a
        wall than that, and the nose reaching 0.25 m of it means the thing has slid
        aside (a 0.17 m clearance on a point 0.30 m ahead ended a shove into the
        chamfer's face the moment it began, four times, probe_fix8_chamfer)."""
        c, s_ = math.cos(yaw), math.sin(yaw)
        px, py = x + c * NOSE_STOP_M, y + s_ * NOSE_STOP_M
        if abs(px) > ROOM_X / 2 and abs(c) > 0.5:
            return True
        if abs(py) > ROOM_Y / 2 and abs(s_) > 0.5:
            return True
        sx, sy = math.copysign(1.0, px), math.copysign(1.0, py)
        d_end, d_side = ROOM_X / 2 - abs(px), ROOM_Y / 2 - abs(py)
        into_chamfer = (c * sx + s_ * sy) / math.sqrt(2)
        # the chamfer's face runs only where both walls are within CHAMFER (a point 0.16 m
        # from the end wall and 0.53 from the side wall passed the sum alone, probe_fix5_corner)
        return d_end < CHAMFER and d_side < CHAMFER and d_end + d_side < CHAMFER and into_chamfer > 0.5

    def inside(gx, gy):
        """The nearest point the body fits at: KERB_CLEAR inside the walls, the chamfers too
        (a standing point in a chamfer's clearance was walked at and jammed on)."""
        gx = max(-ROOM_X / 2 + KERB_CLEAR, min(ROOM_X / 2 - KERB_CLEAR, gx))
        gy = max(-ROOM_Y / 2 + KERB_CLEAR, min(ROOM_Y / 2 - KERB_CLEAR, gy))
        short = CHAMFER + KERB_CLEAR * math.sqrt(2) + 0.001 - ((ROOM_X / 2 - abs(gx)) + (ROOM_Y / 2 - abs(gy)))
        if short > 0:
            gx, gy = gx - math.copysign(short / 2, gx), gy - math.copysign(short / 2, gy)
        return gx, gy

    def wrong_side(o) -> bool:
        return side_of(o["x"]) != mine

    def pick(t, x, y):
        todo = [o for o in eyes.objects() if wrong_side(o) and st["abandoned"].get(o["name"], -1e9) < t - 20.0
                and o.get("n", PICK_SEEN_N) >= PICK_SEEN_N and t - o.get("first", -1e9) >= PICK_SEEN_S]
        foe_w = getattr(eyes, "foe_w", None)
        if foe_w is not None and t - foe_w[2] < 3.0:
            # not the thing the other crab is at: that is its job, and the two of us shoved each
            # other over it fifteen times in a race (day3)
            clear = [o for o in todo if math.hypot(o["x"] - foe_w[0], o["y"] - foe_w[1]) > FOE_CLEAR_M]
            todo = clear or todo
        if not todo:
            st["target"], st["target_w"] = None, None
            if side.truth or st.get("searched", -1e9) > t - 10.0:
                st["intent"], st["stage"] = "done", "nothing left on the wrong half"
            else:
                # nothing remembered over there is not nothing there: the sensor sees a 55 degree
                # cone to about 2 m. Go and look from the middle of the other half.
                st["intent"], st["stage"], st["since"] = "search", "going to look", t
                log(t, "nothing remembered on the wrong half: goes to look")
            return
        # The hierarchy (Andrew, 2026-10-16): a cluster first, the biggest, since it is the most
        # things for one walk; then the back row, the dearest to fetch and the other crab's for
        # free if left (on nearest-first 2-3 things at the back were never serviced); then the
        # nearest. Within a tier the shortest walk to the standing point (behind the thing,
        # on the far side of it from my half), a thing seen lately over one remembered.
        def walk(o):
            return math.hypot(o["x"] - sign * BEHIND_M - x, o["y"] - y) + STALE_WEIGHT * (t - o.get("seen", t))

        def mass(o):
            return sum(1 for q in todo if q is not o and math.hypot(q["x"] - o["x"], q["y"] - o["y"]) < CLUSTER_M)
        hot = {o["name"]: mass(o) for o in todo}
        top = max(hot.values())
        if top >= 1:
            best = min((o for o in todo if hot[o["name"]] == top), key=walk)
            tier = f"a cluster of {top + 1}"
        else:
            back = [o for o in todo if abs(o["x"]) > BACK_M]
            best = min(back or todo, key=walk)
            tier = "the back row" if back else "the nearest"
        st["target"], st["target_w"] = best, (best["x"], best["y"])
        st["intent"], st["stage"], st["since"], st["job_since"] = "approach", f"to the {best['kind']}", t, t
        st["still_since"] = None
        log(t, f"after the {best['kind']} at ({best['x']:+.2f}, {best['y']:+.2f}), {len(todo)} on the wrong half: {tier}")

    def wall_route(o, held=None):
        """For a thing within WALL_ROUTE_M of the kerb: the heading that shoves it along
        the wall toward the line -- along the end wall to the corner, where the chamfer's
        face slides it onto the side wall, then along the side wall to the line -- and
        the angle round the thing to stand at. None for a thing in the open. Near the
        chamfer the shove runs along the nearer of the two walls: a shove ALONG the
        chamfer needs the crab behind the thing on the chamfer's line, which for a thing
        within 0.4 m of the end wall is inside the end wall (the standing point clamped
        to the room lay 64 degrees off the shove's heading, the nose turned off the
        thing, and the crab walked into the side wall, probe_fix4_chamfer). `held` is
        the heading a shove under way is on: it is kept ROUTE_HYST_M past its edge, so
        the thing does not flicker between two legs on the line between them."""
        hx, hy = ROOM_X / 2, ROOM_Y / 2
        sx, sy = math.copysign(1.0, o["x"]), math.copysign(1.0, o["y"])
        d_end, d_side = hx - abs(o["x"]), hy - abs(o["y"])
        # the end wall's leg slants ROUTE_SLANT_DEG toward the room: the thing is pushed on
        # its wall-side flank and goes round the corner rather than into it (pushed square
        # on, with the crab's axis a few centimetres on the room's side of it, a puck at the
        # chamfer's face went into the corner and through the kerb, probe_fix10_chamfer)
        end_h = math.atan2(sy * math.cos(math.radians(ROUTE_SLANT_DEG)), -sx * math.sin(math.radians(ROUTE_SLANT_DEG)))
        side_h = push_yaw
        more = lambda h: ROUTE_HYST_M if held is not None and abs(_wrap(h - held)) < 1e-3 else 0.0
        if d_end + d_side < CHAMFER + WALL_ROUTE_M:
            # near the chamfer: the nearer wall's leg first, the other's if the body does not
            # fit behind the thing for it (0.30 from the end wall and 0.29 from the side wall,
            # the side wall's leg stood inside the end wall and the thing was given up
            # though the end wall's leg fitted, probe_fix11_chamfer)
            legs = [end_h, side_h] if d_end - more(end_h) <= d_side - more(side_h) else [side_h, end_h]
            if d_end < END_LEG_MIN_M:
                # wedged: the end wall's leg needs the crab's axis on the thing's wall side,
                # 0.1 m nearer the end wall than it, and the body stops 0.17 m from the wall
                # (a puck on the chamfer's face 0.21-0.27 m from the end wall was shoved
                # deeper into the corner on every try, probe_fix13_chamfer)
                legs = [h for h in legs if h != end_h]
        elif d_end < WALL_ROUTE_M + more(end_h):
            legs = [end_h]                                 # along the end wall, to the corner
        elif d_side < WALL_ROUTE_M + more(side_h):
            legs = [side_h]                                # along the side wall, to the line
        else:
            return None
        return [(_wrap(h + math.pi), h) for h in legs]

    def route_stand(o, held=None):
        """The first leg of the thing's wall route the body fits behind it for: the angle to
        stand at, the point, the heading. None in the open, or where no leg fits. The
        angle to stand at is the clamped point's: the ideal one lies in the kerb's
        clearance for a thing on the wall's line, and the orbit can never get there.
        Clamped farther than ROUTE_CLAMP_M it is no standing point for that leg at all
        (a thing in the chamfer's corner: the point behind it on the side wall's line
        is inside the end wall, and clamped out of the chamfer it landed on the thing,
        a goal angle 60 degrees off the shove, probe_fix7_corner): the next leg, or the
        open costing, finds the slant the body fits in, and the chamfer's face does the
        rest."""
        for a, heading in wall_route(o, held=held) or ():
            gx0, gy0 = o["x"] + BEHIND_M * math.cos(a), o["y"] + BEHIND_M * math.sin(a)
            gx, gy = inside(gx0, gy0)
            if math.hypot(gx - gx0, gy - gy0) < ROUTE_CLAMP_M:
                return math.atan2(gy - o["y"], gx - o["x"]), gx, gy, heading
        return None

    def approach_point(o, x, y, t):
        """Where to stand to shove `o` toward my half -- the angle round it, the point,
        and the heading of the shove. Every candidate the body fits at (straight
        behind, slanting 30, 55, 75 degrees either way) is costed by the way round
        the thing from where I am plus how far the shove slants, and the cheapest
        taken; it is then kept for the job unless it stops fitting or the thing
        moves STAND_MOVED_M, so the choice does not dither beside a wall. The last
        resort is the nearest standing room to straight behind."""
        held = st.get("stand")
        if held is not None and held[1] == o["name"] and math.hypot(o["x"] - held[3][0], o["y"] - held[3][1]) < STAND_MOVED_M:
            a, heading = held[2], held[4]
            gx, gy = o["x"] + BEHIND_M * math.cos(a), o["y"] + BEHIND_M * math.sin(a)
            if heading is not None:                        # the wall route: the clamped point, as chosen
                return a, *inside(gx, gy), heading
            if free(gx, gy, KERB_CLEAR):
                return a, gx, gy, _wrap(a + math.pi)
        leg = route_stand(o)
        if leg is not None:
            a, gx, gy, heading = leg
            st["stand"] = (t, o["name"], a, (o["x"], o["y"]), heading)
            return a, gx, gy, heading
        theta = math.atan2(y - o["y"], x - o["x"])
        best = None
        for deg in APPROACH_DEG:
            for sg in ((1.0,) if deg == 0.0 else (-1.0, 1.0)):
                a = push_yaw + math.pi + sg * math.radians(deg)
                gx, gy = o["x"] + BEHIND_M * math.cos(a), o["y"] + BEHIND_M * math.sin(a)
                if not free(gx, gy, KERB_CLEAR):
                    continue
                # the slant toward the room's middle first (the sign was read off `sg` alone
                # before, which is the middle for red and the wall for blue: blue drove a puck
                # into the camera-side wall, day25)
                toward_wall = math.sin(_wrap(a + math.pi)) * o["y"] > 0
                cost = abs(_wrap(a - theta)) * BEHIND_M + math.radians(deg) * 0.6 + (0.1 if toward_wall else 0.0) \
                    + (1.0 if toward_wall and ROOM_Y / 2 - abs(o["y"]) < WALL_SLANT_M else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, a, gx, gy)
        if best is not None:
            st["stand"] = (t, o["name"], best[1], (o["x"], o["y"]), None)
            return best[1], best[2], best[3], _wrap(best[1] + math.pi)
        return None                                            # nowhere the body fits to shove it from

    def abandon(t, why, stuck=False, cool=20.0):
        """Drop the target; reverse only when `stuck` (against the thing, or the other crab) --
        a reverse after every give-up read as backing away from the objects."""
        o = st["target"]
        if o is not None:
            st["abandoned"][o["name"]] = t - 20.0 + cool
        log(t, f"gives up the {o['kind'] if o else '?'}: {why}")
        if stuck or (claw_press is not None and player.fsm.mode == "claw_left"):
            st["intent"], st["stage"], st["since"], st["claw_pressed"] = "back_off", why, t, None
        else:
            st["intent"], st["stage"] = None, ""

    def act(t):
        st["t_now"] = t
        if not side.truth:
            eyes.hold = st["target"]["name"] if st["target"] is not None else None
        x, y, yaw = eyes.pose()
        intent = st["intent"]
        if intent is None or intent == "done":
            pick(t, x, y)
            intent = st["intent"]
            if intent == "done":
                return None
        o = st["target"]
        if intent in ("approach", "push"):
            # the target as it is now (the probe sees it move; the sensed pilot re-sights it)
            now = next((q for q in eyes.objects() if q["name"] == o["name"]), None)
            if now is None and not side.truth:
                # forgotten (it moved, or the map shifted under it): the nearest thing where it was
                near_ = [q for q in eyes.objects() if math.hypot(q["x"] - o["x"], q["y"] - o["y"]) < REFIND_M]
                now = min(near_, key=lambda q: math.hypot(q["x"] - o["x"], q["y"] - o["y"])) if near_ else None
                if now is None:
                    bx, by = eyes.mapper.to_body(o["x"], o["y"])
                    near_t = min(math.hypot(q["x"] - o["x"], q["y"] - o["y"]) for q in truth.objects())
                    abandon(t, f"it is not there any more (seen {o.get('n', 0)} times, last {t - o.get('seen', t):.1f} s ago, "
                               f"{math.hypot(bx, by):.2f} m off at {math.degrees(math.atan2(by, bx)):+.0f} deg; "
                               f"truly the nearest thing is {near_t:.2f} m from there)")
                    return None
            o = now or o
            st["target"], st["target_w"] = o, (o["x"], o["y"])
            if sign * o["x"] > OVER_M:
                st["shoved"].append(o["name"])
                log(t, f"the {o['kind']} is over ({o['x']:+.2f}): {len(st['shoved'])} shoved")
                st["intent"], st["stage"], st["since"] = "back_off", "it is over", t
                return GamepadState(ly=0.6)
            if t - st["job_since"] > TRY_S:
                abandon(t, f"{TRY_S:g} s on it", stuck=True)
                return GamepadState(ly=0.6)
        if intent == "approach":
            foe_w = getattr(eyes, "foe_w", None)
            if foe_w is not None and t - foe_w[2] < 1.0 and math.hypot(o["x"] - foe_w[0], o["y"] - foe_w[1]) < FOE_CLEAR_M:
                abandon(t, "that is the other crab")           # (or a thing under it: not mine to circle)
                return None
            stand = approach_point(o, x, y, t)
            if stand is None:
                # a thing deep in a corner: no leg of a route and no slant the body fits in
                # (a puck on the chamfer's face 0.23 m from the end wall, probe_fix10_chamfer)
                abandon(t, "no room to stand behind it")
                return None
            theta_goal, gx, gy, st["push_yaw"] = stand
            mapper = getattr(eyes, "mapper", None)
            orbiting = st["stage"] == f"round the {o['kind']}"
            if math.hypot(o["x"] - x, o["y"] - y) > ORBIT_IN_M + (ORBIT_OUT_M if orbiting else 0.0):
                # far off: walk at it, the planner's way round whatever is in between
                st["stage"] = f"to the {o['kind']}"
                wx_, wy_ = o["x"], o["y"]
                if mapper is not None:
                    if t - st["plan_t"] >= PLAN_DT:
                        st["plan_t"] = t
                        st["waypoint"] = mapper.next_point((gx, gy), LOOKAHEAD)
                    if st["waypoint"] is not None:
                        wx_, wy_ = st["waypoint"]
                return head_to(x, y, yaw, wx_, wy_)
            # near: circle it to the far side, nose on it, and shove the moment I am there
            if st["stage"] != f"round the {o['kind']}":
                st["stage"], st["orbit_since"], st["orbit_way"] = f"round the {o['kind']}", t, None
            if mapper is not None:
                mapper.path = []
            frame, there, facing = orbit(x, y, yaw, o, theta_goal)
            # in the picture now, not merely remembered within the facing tolerance: a shove
            # begun on a memory 30 degrees off, just outside the picture, found nothing under
            # the nose and bounced between the orbit and the shove at 50 Hz (probe_fix3_corner)
            fresh = side.truth or (eyes.sighted(o["name"], t) or 0.0) <= 0.2
            if facing and t - st["orbit_since"] > ORBIT_S and not fresh:
                abandon(t, f"circled it {ORBIT_S:g} s and lost it")
                return None
            if fresh and (there or (facing and t - st["orbit_since"] > ORBIT_S)):
                st["intent"], st["stage"], st["since"] = "push", f"shoving the {o['kind']}", t
                st["last_xy"], st["still_since"], st["claw_pressed"], st["seen_at"] = (o["x"], o["y"]), None, None, None
                return None
            return frame
        if intent == "push":
            fsm = player.fsm
            if claw_press is not None and fsm.mode != "claw_left" and t - st["since"] < CLAW_WAIT_S:
                # into the claw mode first, the arm out: the shove is the claw's
                chord = claw_toggle(t)
                return chord if chord is not None else None
            if claw_press is not None and fsm.mode == "claw_left" and not fsm.running_policy:
                return None                                  # hands off through the ramp: a preset held then keeps the
                                                             # home pose from being reached, and the ramp never ends (day15)
            # the object in the body frame: ahead (bx) and across (by, +left)
            dx, dy = o["x"] - x, o["y"] - y
            bx = math.cos(yaw) * dx + math.sin(yaw) * dy
            by = -math.sin(yaw) * dx + math.cos(yaw) * dy
            err = _wrap(st["push_yaw"] - yaw)
            held = st.get("stand")
            age = None if side.truth else eyes.sighted(o["name"], t)
            if held is not None and held[1] == o["name"] and held[4] is not None and (age is None or age <= 0.2):
                # on a wall route: the thing reaching the next wall is the next leg of the
                # route, not a jam -- back off and take it again from its new side (each
                # corner cost a jam, 4 s and a re-pick, 21 a race, day27). Judged on the
                # thing in the picture, never on the blind estimate under the nose: pushed
                # into the chamfer the estimate crossed the legs' line half a second into
                # every shove and the puck moved a centimetre a time (probe_fix9_chamfer);
                # blind, the nose stop below ends the leg
                leg = route_stand(o, held=held[4])
                if leg is not None and abs(_wrap(leg[3] - held[4])) > math.radians(30):
                    log(t, f"the {o['kind']} is round the corner: again from the other side")
                    st["stand"] = None
                    st["kerb_stops"].pop(o["name"], None)
                    st["intent"], st["stage"], st["since"] = "back_off", "round the corner", t
                    return GamepadState(ly=0.6)
            if abs(err) < math.radians(30) and kerb_ahead(x, y, yaw):
                # the nose at the kerb's clearance, by the map: this leg of the shove is done
                # before the feet slip on the wall -- back off and take the thing again from
                # where it now lies (a thing in the chamfer's corner is shoved into the face,
                # which slides it along; the crab behind it meets the face first)
                n, lx_, ly_ = st["kerb_stops"].get(o["name"], (0, o["x"], o["y"]))
                n = 1 if math.hypot(o["x"] - lx_, o["y"] - ly_) >= KERB_STOP_MOVED_M else n + 1
                st["kerb_stops"][o["name"]] = (n, o["x"], o["y"])
                if n > KERB_STOPS:
                    abandon(t, f"at the kerb {n} times", stuck=True, cool=JAM_COOL_S)
                    return GamepadState(ly=0.6)
                log(t, f"the nose is at the kerb: the {o['kind']} again from where it lies ({n})")
                st["stand"] = None
                st["intent"], st["stage"], st["since"] = "back_off", "at the kerb", t
                return GamepadState(ly=0.6)
            seen = True
            if not side.truth:
                if age is None or age <= 0.2:
                    st["seen_at"] = (bx, by)
                elif age > 0.2:
                    seen = False
                    sx_, sy_ = st.get("seen_at") or (bx, by)
                    if not (sx_ < NOSE_SEEN_M and abs(sy_) < NOSE_SIDE_M):
                        # it left the picture by an edge, not under the nose: the orbit again
                        log(t, f"the {o['kind']} slipped out of view (ahead {sx_:+.2f}, across {sy_:+.2f}): round it again")
                        st["intent"], st["stage"] = "approach", ""
                        return None
                    # out of the picture under the nose: it is where the nose is, so the memory
                    # of it moves with me and is not left behind by the dead reckoning
                    if age < BLIND_S:
                        bx, by = max(bx, NOSE_M), by * 0.5
                        th = eyes.things.get(o["name"])
                        if th is not None:
                            th["x"], th["y"] = x + math.cos(yaw) * bx - math.sin(yaw) * by, y + math.sin(yaw) * bx + math.cos(yaw) * by
                            # a thing's centre is its radius inside the kerb at the least (a blind
                            # estimate on a pose a little off was put in the wall, probe_fix5_corner)
                            th["x"] = max(-ROOM_X / 2 + THING_R, min(ROOM_X / 2 - THING_R, th["x"]))
                            th["y"] = max(-ROOM_Y / 2 + THING_R, min(ROOM_Y / 2 - THING_R, th["y"]))
                            o = dict(th)
                            st["target"], st["target_w"] = o, (o["x"], o["y"])
                            if sign * o["x"] > OVER_M:
                                st["shoved"].append(o["name"])
                                log(t, f"the {o['kind']} is over, blind ({o['x']:+.2f}): {len(st['shoved'])} shoved")
                                st["intent"], st["stage"], st["since"] = "back_off", "it is over", t
                                return GamepadState(ly=0.6)
                    else:
                        abandon(t, f"not seen for {age:.1f} s", stuck=True)
                        return GamepadState(ly=0.6)
            if bx < -0.05 or abs(by) > 0.45:
                abandon(t, f"lost it (ahead {bx:+.2f}, across {by:+.2f})")
                return None
            if not side.truth and eyes.blocked:
                abandon(t, "that is the other crab", stuck=True)
                return GamepadState(ly=0.6)
            if not side.truth and eyes.jammed:
                abandon(t, "jammed on the kerb", stuck=True, cool=JAM_COOL_S)
                return GamepadState(ly=0.6)
            if bx > 0.9:
                # a ball that rolled ahead: walk after it rather than shove air
                return head_to(x, y, yaw, o["x"], o["y"])
            moved = math.hypot(o["x"] - st["last_xy"][0], o["y"] - st["last_xy"][1])
            if moved < 0.01:
                st["still_since"] = st["still_since"] or t
                if t - st["still_since"] > STALL_S and bx < 0.35:
                    abandon(t, "it does not move", stuck=True)
                    return GamepadState(ly=0.6)
            else:
                st["still_since"] = None
            st["last_xy"] = (o["x"], o["y"])
            # the shove's heading held, the lateral stick keeping the thing under the nose
            # (steering the nose onto the thing instead, with the body strafing round to the
            # line, turned a crab beside a side wall 37 degrees into the wall, probe_fix_side)
            rx = turn_stick(err)
            lx = -max(-0.6, min(0.6, 2.5 * by))
            if abs(err) > math.radians(PUSH_TURN_DEG):
                # far off the shove's heading: turn first; walking meanwhile walked into the
                # corner (day26)
                return arm(GamepadState(rx=-math.copysign(TURN_FULL, err)))
            return arm(GamepadState(ly=-PUSH, lx=lx, rx=rx))
        if intent == "back_off":
            if t - st["since"] < BACK_S:
                return arm(GamepadState(ly=0.6))
            if claw_press is not None and player.fsm.mode == "claw_left":
                if st["claw_pressed"] is None or t - st["claw_pressed"] > 0.3:
                    st["claw_pressed"] = None
                chord = claw_toggle(t)
                if chord is not None:
                    return chord
                if t - st["since"] < BACK_S + CLAW_WAIT_S:
                    return None
            st["intent"], st["claw_pressed"] = None, None
            return None
        if intent == "search":
            todo = [o for o in eyes.objects() if wrong_side(o) and st["abandoned"].get(o["name"], -1e9) < t - 20.0
                and o.get("n", PICK_SEEN_N) >= PICK_SEEN_N and t - o.get("first", -1e9) >= PICK_SEEN_S]
            if todo:
                st["intent"] = None                          # something came into view: pick it
                return None
            gx, gy = -sign * LOOKOUT_X, 0.0
            if st["stage"] == "going to look":
                if math.hypot(gx - x, gy - y) < 0.25 or t - st["since"] > 20.0:
                    st["stage"], st["since"] = "looking round", t
                    eyes.mapper.path = []
                    return None
                mapper = getattr(eyes, "mapper", None)
                if mapper is not None and t - st["plan_t"] >= PLAN_DT:
                    st["plan_t"] = t
                    st["waypoint"] = mapper.next_point((gx, gy), LOOKAHEAD)
                wx_, wy_ = st["waypoint"] if (mapper is not None and st["waypoint"] is not None) else (gx, gy)
                return slide_to(x, y, yaw, wx_, wy_, math.atan2(wy_ - y, wx_ - x))
            if t - st["since"] < SCAN_S:
                return GamepadState(rx=TURN)
            st["searched"] = t
            st["intent"], st["stage"] = "done", "nothing left on the wrong half"
            log(t, "looked round: nothing on the wrong half")
            return None
        return None

    def pilot(t):
        eyes.update(t, st["frame"])
        if os.environ.get("SWEEP_TRACE") and (st["intent"], st["stage"]) != st.get("_last_stage"):
            st["_last_stage"] = (st["intent"], st["stage"])
            x, y, yaw = eyes.pose()
            o = st["target"]
            err = eyes.truth_error() if not side.truth else (0.0, 0.0, 0.0)
            log(t, f"-> {st['intent']}: {st['stage']} at ({x:+.2f}, {y:+.2f}) yaw {math.degrees(yaw):+.0f}"
                   f" (truly {x + err[0]:+.2f}, {y + err[1]:+.2f}, {math.degrees(yaw + err[2]):+.0f})"
                   + (f", target {o['name']} at ({o['x']:+.2f}, {o['y']:+.2f})" if o else ""))
        if not side.truth and t >= st["truth_log"]:
            st["truth_log"] = t + TRUTH_LOG_S
            ex, ey, eyaw = eyes.truth_error()
            fx = eyes.fixes
            print(f"\n[{label}] {t:6.2f}s  map pose off by ({ex:+.2f}, {ey:+.2f}) m, {math.degrees(eyaw):+.0f} deg; "
                  f"{len(eyes.things)} things remembered; kerb in {eyes.kerb_frames} frames, fixes +x {fx[0]} -x {fx[1]} "
                  f"+y {fx[2]} -y {fx[3]} corner {fx[4]} heading {fx[5]}; {eyes.once} seen once and forgotten"
                  + (", stalled" if eyes.mapper.stalled else "") + (", blocked" if eyes.blocked else "")
                  + (", jammed" if eyes.jammed else ""))
            eyes.fixes = [0, 0, 0, 0, 0, 0]
            eyes.kerb_frames = 0
        frame = _pilot(t)
        st["frame"] = frame
        if not side.truth:
            st["things_w"] = [(th["x"], th["y"]) for th in eyes.things.values() if th["n"] >= 2]
            fw = getattr(eyes, "foe_w", None)
            st["foe_w"] = (fw[0], fw[1]) if fw is not None and t - fw[2] < 2.0 else None
        return frame

    def _pilot(t):
        if referee.cue_id != st["cue_id"]:
            st["cue_id"] = referee.cue_id
            piece = referee.cue.get(side.name)
            if piece == "victory" and referee.camera_yaw is not None:
                st["face"] = (piece, t)
            elif piece:
                performer.start(moves[piece], t)
        if st["face"] is not None:
            piece, t0 = st["face"]
            _, _, yaw = eyes.pose()
            err = _wrap(referee.camera_yaw - yaw)
            if abs(err) > math.radians(10) and t - t0 < FACE_S:
                if walking():
                    return GamepadState(rx=-math.copysign(1.0 if abs(err) > math.radians(40) else TURN, err))
                return None
            log(t, f"faces the camera ({math.degrees(err):+.0f} deg off after {t - t0:.1f} s)")
            st["face"] = None
            performer.start(moves[piece], t)
        if performer.busy(t):
            return performer.state(t)
        if not referee.live:
            st["intent"], st["stage"] = None, ""
            return None
        if not walking():
            return None
        return act(t)

    pilot.busy = lambda t: st["face"] is not None or performer.busy(t)
    pilot.state = st
    pilot.senses = getattr(eyes, "sight", None)
    pilot.mapper = getattr(eyes, "mapper", None)
    pilot.eyes = eyes
    return pilot
