"""The course pilot: along the lane from the mark to the finish, one obstacle at
a time, on the dToF, the IMU and the encoders.

What the crab is told: the course (`track.layout()`: where the hill, the forest,
the ball, the goal line and the finish are along the lane), that the lane has a
kerb down each side, and that it woke up on the mark facing along it. What it
reads: its depth image (`tools/race/sight.py`), its gyro and its encoders, from
which it reckons its pose (leg odometry) and corrects it against the kerbs
(`wall_fix`: the lateral offset from each kerb's distance, the heading from its
slant, as the race's sweeper does); the map (`tools/fetch/mapper.py`) is where
the forest's pillars go and where the way through them is planned.

The stages, by the reckoned x along the lane:

* `hill` -- walk straight, the heading held on the gyro, standing: the first
  flight's 2.5 cm risers climb standing and the raised body (`COURSE_RAISE=1`:
  R3 + stick up before the flight, a tap of R3 at the first ramp) jams more
  often than it helps (track.py's measurements). The
  feet slip on the risers and the slopes and the leg odometry runs ahead --
  two to one over the first flight, 1.0 m by the first platform and 2.8 m by
  the forest (sense1, 2026-10-22) -- so the hill is also where the pose is
  pinned: a sustained pitch of the body is a ramp, the only place on the course
  with one, nose down the first and nose up the second, and the pitch's onset
  and end put the base at each ramp's known start and end (`ramp_fix`). After
  the hill the ball, seen where it was told to be, and the lane's end wall are
  the fixes along x; the kerbs fix y and the heading throughout.
* `forest` -- A* on the map to a point past the pillars, replanned four times a
  second, steering at the point 0.45 m along the path.
* `pit` -- the box: to a point behind it, then the shove straight along the
  lane, the lateral stick holding the lane's centre so the shell stays on the
  box's middle (no guides: pushed off-centre a box turns), a pulsed nudge at
  the edge until the box is gone from the nose (flat in the trench, flush with
  the deck), then straight over it. (The loop takes any number of boxes, each
  in its own lane; the course has one.)
* `leap` -- along the second platform to LEAP_BACK_M before its end, the dash
  (a hop of ~0.3 m: walked off, the body goes over), then walking on until the
  reckoning says the crab is down past the edge: the rear legs come down on
  the edge and the walk scrambles them off. A fall here is the controller's
  safe mode and the end of the take -- the leap of faith.
* `ball` -- to a point behind the ball on the centreline, then the shove: walk
  into it with the lateral stick keeping it under the nose until the reckoning
  says it is over the finish line, as the race's sweeper shoves.
* `finish` -- on through the gate for another stretch, turn to the camera, dance
  the colour's victory piece (`tools/battle/moves.py`).

`--truth` reads the world instead (the control): the same stages on the true
pose and the true ball.
"""
from __future__ import annotations

import itertools
import math
import os
import time

import mapper as _mapper
import numpy as np
import track
from mapper import Mapper
from moves import MOVES, WALKING, Performer
from sight import FOOT_R, KERB_MIN, NEAR_M, TOF_ORIGIN, Sight, leg_velocity

# the mapper's grid spans SIZE metres about the wake pose (20 for the fetch's wander); the lane
# runs 21 m, so the grid is widened here -- the module's N is read at every cell lookup -- and
# the map frame's x is the world's less MAP_OFF, so the whole lane is on it
_mapper.SIZE, _mapper.N = 26.0, round(26.0 / _mapper.RES)
# ...and it maps what is close: the fetch's map dropped returns nearer than 0.45 m (the carried
# mug hangs there) and then cleared the cells along each column's ray to the next thing behind,
# so a pillar vanished from the map as the crab closed on it and the planner routed straight
# through -- "every single time they just ram right into the most obvious pillar in the middle"
# (Andrew, 2026-10-22, on sense4/blue1/blue2). The crab's own legs are masked out by the sight.
_mapper.MAP_NEAR = 0.12

SENSES = True
MAP_OFF = 10.0                      # the map frame's x is the world's less this (see above)
# the race's measurements (tools/race/pilots/sweeper.py, 2026-10-15/16)
WALK = float(os.environ.get("COURSE_WALK", "0.6"))        # ~0.45 m/s: was 0.75, a little slower for the leg odometry's sake
                                                         # (Andrew: "its tracking is usually messed up when it arrives at pillars")
HILL_STICK = 1.0                                         # the probes climbed at full stick
SIDEWAYS_DOWN = os.environ.get("COURSE_SIDEWAYS", "blue")  # the crab that takes the last flight sideways ("" for none)
SIDE_HEADING_DEG, SIDE_TURN_DEG = -90.0, 8.0            # ...facing the camera's side (-y), the heading held within this
DOWN_STICK = 0.7                                         # ...and walked down 4 cm flights at 0.6 and at 1.0 (8 of 8); the
                                                         # course's first descent went over at 1.0 (truth1), so slower
HOLD_TURN_DEG = 5.0                                      # walking straight, the heading is left alone inside this...
STAIR_TURN_DEG = 15.0                                    # ...and on a flight inside this: the risers push the heading 5-6
                                                         # degrees, and at the 5 degree line the crab turned on the spot on
                                                         # the stairs, where a turn moves nothing, for 45 s (red3); a flight
                                                         # is 2 m, and 10 degrees over it is 0.35 m across a 2.2 m lane
PUSH = 0.55                                              # the shove's stick: was 0.75 (Andrew: "take a moment to line itself up")
PUSH_BALL = 0.65                                         # the ball's: the crab follows the ball over the line (0.5 took 45-70 s
                                                         # for three balls, Andrew 2026-10-23: "push the crabs to go a little
                                                         # bit faster, especially with pushing the balls")
BALL_WALK = 0.75                                         # the walk round to the point behind a ball (WALK elsewhere)
TURN, TURN_BAND, TURN_PER, TURN_RATE = 0.62, 0.50, 470.0, 3.0
STICK_BAND, FWD_PER, SIDE_BAND, SIDE_PER = 0.40, 1.4, 0.08, 0.85
FACE_DEG = 25.0                                          # walk once the heading is within this of the way
RAISE = os.environ.get("COURSE_RAISE", "0") == "1"       # raise the body for the first flight (off: see the docstring)
RAISE_S = 1.0                                            # R3 + stick up held this long: the stance to 0.153 m
RAISE_BEFORE_M = 0.6                                     # ...from this far before the first flight
LOOKAHEAD = 0.35                                         # the point along the plan the walk steers at (0.45 cut the
                                                         # corner of a pillar, blue1)
PLAN_DT = 0.25
INFLATE_M = 0.33                                         # the planner's hard clearance round a pillar: the stance is ~0.25 half
                                                         # and the legs reach 0.31 ahead, ~0.32 to a corner. At 0.27 the path
                                                         # passed a pillar's corner closer than that and the crab caught it
                                                         # (red212); 0.28 had clipped pillars before (blue4). The no-way at
                                                         # 0.33 that once looked like the clearance (blue84, red63) was the
                                                         # known pillars drifting with every pose fix (Mapper.known_absolute)
PLAN_MARGIN_M, PLAN_MARGIN_COST = 0.60, 3.0              # ...and the soft clearance: a cell within 0.55 m of an obstacle costs
                                                         # up to 3 steps more, so the path keeps to a gap's middle
PLAN_SWING_M, PLAN_COMMIT_S = 0.30, 2.0                   # a new waypoint this far from the last is a swing, kept off this long
PLAN_SPECK_CELLS = 3                                     # a sensed blob smaller than this (15 cm) is not planned round
PILLAR_CLEAR_M = 0.65                                    # ...and this near a pillar: the sensed pillars came out 0.25-0.6 m short
                                                         # of the known ones with the pose right to 1 cm (blue211: blobs 0.5-0.6 m
                                                         # before the first two rows, at their y), and walled the gaps; the forest
                                                         # holds nothing else, so there the planner plans on the known pillars and
                                                         # the sensor's part is to place the crab (cause of the shortfall not found)
KNOWN_CLEAR_M = 0.35                                     # no sensed obstacle is kept this near a known one (the kerbs, the pillars)
PLAN_ESCAPE_M = 0.32                                     # no way at INFLATE_M: the crab is inside an inflation, and the cells this
                                                         # near it are freed for one more plan, rather than walk straight at the goal
BEHIND_M = 0.5                                           # the shove starts this far behind the ball
BALL_BACK_S = 0.5                                        # after a ball is over, this long in reverse so the legs leave it
                                                         # (the race's BACK_S), then the next pick
NOSE_M = 0.28                                            # a shoved ball sits about here ahead of the base
BALL_H = (0.15, 0.27)                                    # a 24 cm ball, over the race's things and under the kerb
BALL_W = (0.10, 0.34)
BALL_NEAR_H, BALL_NEAR_W = 0.12, 0.04                     # ...inside the sight's NEAR_M, where only a sliver of it is seen
FACE_S = 8.0                                             # the turn to the camera gives up after this
# the wall fix (the sweeper's, two straight kerbs)
WALL_NEAR_M, WALL_MIN_PTS, WALL_ALONG_M = 0.25, 12, 0.4
WALL_NEAR_PLAT_M = 0.5                                   # ...the window on the platforms, where nothing tall but the kerbs
                                                         # stands (the pillars, 0.45 m tall, read as kerb and sit 0.6 m
                                                         # from one): a sidestep the leg odometry read backwards put the
                                                         # pose 0.31 m out in 3 s, past the 0.25 m window, and the kerb
                                                         # points found could not be matched to their kerb (red102 B)
WALL_GAIN, WALL_STEP_M = 0.3, 0.03
YAW_GAIN, YAW_STEP, YAW_MAX, YAW_ALONG_M = 0.1, math.radians(0.05), math.radians(8), 0.6
END_NEAR_M, END_MIN_PTS = 0.8, 12                        # the end wall: kerb points this near its known x, across the lane
# the ramp fix: the body pitched past RAMP_PITCH for RAMP_HOLD_S is on a ramp (the flights wobble it
# +-7 deg for a step at a time, the slopes hold it at ~12), and the base is RAMP_LEAD_M past the
# ramp's start when the pitch sets in, and past its end when the pitch comes off
RAMP_PITCH, RAMP_HOLD_S, RAMP_LEAD_M, RAMP_MAX_M = math.radians(8.0), 0.5, 0.25, 1.5
BALL_MEM_M = 0.35                                         # a ball-sized sighting within this of a remembered ball is that ball
BALL_SEEN_N = 2                                           # a ball is picked once seen this often (the race's "seen twice")
BALL_STALE_S = 20.0                                       # a remembered ball unseen this long is forgotten
BALL_FACE_M = 0.30                                        # a sighting this near the second platform's end is its 12 cm face, whose
                                                          # corners came out ball-sized (red103: "ball-sized at x 19.41")
BALL_SCAN_TURN = 2 * math.pi                              # the search: a turn on the spot this far, every remembered ball over...
BALL_SCAN_STICK = 0.62                                    # ...at the race's turn stick (57 deg/s)
BALL_FIX_M, BALL_FIX_GAIN = 1.8, 0.5                    # the ball seen within this of where it was told: a landmark (the
                                                         # one thing its size on the course; 1.0 refused a 1.4 m error, blue2)
CLIMB = os.environ.get("COURSE_CLIMB", "pitch")          # the first flight: pitch (nose up CLIMB_RY: the claws clear the
                                                         # risers), dash (a hop a step, 2.5 s each), plain (the walk as it is)
CLIMB_RY = 0.5                                           # nose up ~10 deg (probe p05: climbed; 0.8 = 15 deg, climbed too)
DASH_EVERY_S = 2.5
BLOCK_H, BLOCK_W = (0.06, 0.18), (0.25, 2.4)             # the box as the sight sees it: low and wide (the race's things are
                                                         # 7-26 cm; the box's face is 0.8)
BLOCK_BEHIND_M = 0.55                                    # the shove starts this far behind a block's near face
NOSE_GONE_M = 0.42                                       # a thing at the nose beyond this is not the box being shoved
BLOCK_NOSE_M = 0.22                                      # a shoved block's near face sits about here ahead of the base
SHOVE_TURN_DEG, LINE_UP_S = 2.0, 0.5                     # squared to this before the shove, then this long still: a push off
                                                         # square pushes the box sideways and the errors compound (Andrew)
SHOVE_BIAS_Y = -0.10                                     # the shove's line, right of the box's centre by this: pushing, the
                                                         # crab walks LEFT 0.1-0.2 m over the 0.6 m (red64: +0.07 to +0.19,
                                                         # the no-hold takes to +0.18) and the box, pushed on its left side,
                                                         # turned 45-65 deg; started right of centre it crosses the centre
                                                         # on the way and the push stays near the middle
SHOVE_HOLD_M = 0.15                                      # the shove's lateral hold: no stick within this of the line = none
                                                         # while pushing (the hold's strafe against the face dragged the box
                                                         # 0.2-0.5 m aside; the drift is met by SHOVE_BIAS_Y instead)
SHOVE_CENTRE_M, SHOVE_SIDE_MIN = 0.02, 0.25                # the shove keeps the crab within this of the box's centre line,
                                                         # with at least this much strafe stick when outside it
LANE_WAIT_S = 1.5                                        # standing before the crossing for the bridge's line to be read
CROSS_STICK = float(os.environ.get("COURSE_CROSS", "1.0"))    # the walk over the box, at full stick: at 0.75 the feet caught in
                                                         # the 6 cm slot the dropped box leaves and the crab stalled 22 times
                                                         # (truth20); at 1.0 the stride carries them over (truth21, one stall)
EDGE_DROP_M, EDGE_FAR_M = 0.09, 1.6                      # the drop off the second platform, as the sight sees it: returns this
                                                         # far below the platform's plane (the notch's floor is 7 cm down and
                                                         # must not count; the ground is 12, and read against a plane fitted
                                                         # to the feet it came up a shade short of 10: blue19 "looking for the
                                                         # edge" walked off at half stick), within this reach
NOSE_TALL_M = 0.35                                       # a cluster at the nose up to this tall is the box (tipping, its rear
                                                         # rises); the kerb is taller and flagged
BLOCK_GONE_S = 0.3                                       # the block unseen at the nose this long: it has dropped in (flat
                                                         # it lies flush with the deck; tilted, its rear end is in view)
HUNG_DEG = 18.0                                          # nose down past this after the hop: hung on the edge (blue75 hung at 26)
PAST_LINE_S = 4.0                                        # on past the finish line this long (~1.8 m at the walk) before the dance
WALK_ON_S = 3.0                                          # the walk on after the hop, at most: then the ball's approach
LOOK_S = 0.6                                             # the beat at the brink before the hop: standing level, then the dash
                                                         # (1.0 s gave blue75 time to tip over the edge where it stood)
BRINK_READ_M, BRINK_READ_OFF_M = 0.35, 0.05              # the creep goes by the sight until the edge reads this: nose down
                                                         # the reading is the truth plus 3-5 cm from 0.9 m in to 0.34, and
                                                         # then sits at 0.30 whatever the truth (the lowest ray's own
                                                         # intercept: blue44 read 0.30-0.32 from 0.25 in to 0.15), so 0.34
                                                         # is the one reading that places the brink, 0.30 ahead (red27,
                                                         # blue44, blue43: 14 pairs within 2 cm)
BRINK_ON_S = 1.0                                         # ...then this long more at the creep, which makes 18-19 cm/s nose
                                                         # down (blue53, blue54, red42: 0.14-0.17 m in 0.78-0.94 s), to
                                                         # ~0.10 from the edge: the hops that came down clean were from
                                                         # 0.08-0.14 (red26, blue44, blue45, blue54), the rear legs caught
                                                         # the edge from 0.14-0.17 (blue49, red39-41; Andrew: "a tiny bit
                                                         # closer"), from 0.24 the base landed on the edge and hung (blue40,
                                                         # blue41), from 0.25-0.32 it went over short (red43, red27, blue42).
                                                         # By the leg odometry the stretch was 0.14-0.17 for 0.20-0.24
                                                         # counted, and once 0.24 in a quarter second (red43: a slip).
                                                         # 1.1 s came to 0.07-0.09 and two hops from 0.07-0.08 dived
                                                         # (blue60, blue61) where 0.08-0.14 had come down clean six
                                                         # times: 0.95 s, to ~0.11. Then the last stretch went level (see
                                                         # below) and level the creep is quicker: 0.95 s came to 0.07
                                                         # (red52) and four seeded blues in six went over at the brink,
                                                         # so 0.7 s. At the creep's 0.25 (Andrew: "it just needs to go slow")
                                                         # the stretch runs ~0.16 m/s: 1.0 s came to 0.14-0.16 (red56,
                                                         # blue72), so 1.15 s, to ~0.12; at 0.11 blue75 tipped nose down 37 deg
                                                         # standing its beat and hung, and from 0.10-0.11 two more went
                                                         # over or hung (blue77, red59) while every slow-creep hop from
                                                         # 0.12-0.16 came down clean (red56, blue72, blue74, red57, red58,
                                                         # blue78), so 1.0 s, to 0.14-0.16 -- where blue45 was
LEAP_NEAR_M, CREEP = 0.55, 0.25                          # the walk to the edge gives way to the nose-down creep when the edge
                                                         # reads this near (the level reading is true to ~0.3), at this stick
SHOVE_PAST_M = 0.40                                      # the shove's limit: the reckoned base this far past the notch's
                                                         # edge with the box still at the nose. Generous: the legs slip
                                                         # against the box and the leg odometry ran 0.6-1.4 m ahead over the
                                                         # shove (blue31, blue33), and a limit 17 cm short of the edge
                                                         # stopped the shove with the box 0.4 m from the notch
RESHOVES = 3                                             # a block still at the edge when the crossing is due is shoved again
BACK_GIVE_UP_S = 15.0                                    # backing off the edge for the next block this long without getting
                                                         # there: the crossing goes on the block that is in (blue10: 33 s
                                                         # stuck at the edge)
ESCAPE_DEG, ESCAPE_M = 30.0, 0.9                         # shoving, the ball seen this far off the nose and this near has got
                                                         # away sideways: round behind it again (blue2 walked on without it)
PILLAR_FIX_X, PILLAR_FIX_Y, PILLAR_RANGE = 1.2, 0.3, (0.5, 1.4)   # the forest's first row seen from the hill's foot: a tall
                                                         # thin cluster this near one of its pillars (across the lane the
                                                         # pose is good; along it the flight down slips -0.5 m, sense4 --
                                                         # seen from 2 m it fixed before the flight, hence the short range)
FOREST_WALK = 0.5                                        # the walk through the pillars (0.4 before 2026-10-23: 22-25 s for the forest)
FOREST_STRAFE = 0.45                                     # the strafe that takes up what the heading does not
FOREST_HEAD_M = 0.6                                      # the planner's path is followed facing along it: the heading is the way
FOREST_FACE_DEG = 20.0                                   # to the point this far along the plan, within this of the lane, turned
FOREST_SPOT_DEG = 15.0                                   # while walking and on the spot only past FOREST_SPOT_DEG. With the
                                                         # heading held along the lane and the way taken by the strafe alone the
                                                         # rear legs swept sideways into the pillars just passed (Andrew,
                                                         # 2026-10-23: "turn a little and walk in the direction of the path...
                                                         # not just strafe side to side"); turned to face every waypoint, with
                                                         # no cap, the crab had stood 30-42 degrees across the lane, but that
                                                         # was with the map's pillars drifting (red102 K, red104 K). Seeds
                                                         # 211-213 without video: the reds through in 18 s, no stall (22-25 s
                                                         # held along the lane), the blues in 18-21 s
THRESHOLD_S = 2.0                                        # standing at the forest's door for the first row to place the crab, at most
PILLAR_FIX_GAIN = 0.5
LM_GAIN, LM_STEP_M, LM_MATCH_M = 0.5, 0.08, 0.45         # every pillar and, until it is shoved, every block is a landmark:
                                                         # a sighting within LM_MATCH_M of where the course puts one moves
                                                         # the pose by LM_GAIN of the difference, no more than LM_STEP_M a
                                                         # frame (the pose drifted 1.3 m through the forest with nothing to
                                                         # fix it after the first row, blue4)
NOSE_CLEAR_M = 0.30                                      # the map shows something this near ahead: the feet are slipping on it
STALL_NEAR = 0.45                                        # a thing this near, anywhere in the picture, unchanged: a jam
STALL_S, STALL_M, STALL_FAR, STALL_COLS = 1.5, 0.08, 0.2, 20   # the middle of the picture beyond STALL_FAR unchanged (median zone within
                                                         # ...STALL_FAR was 0.5, and a pillar the crab was jammed on at 0.3 m
                                                         # was excluded: 35 s on it, no stall called, 2.8 m reckoned (blue8)
                                                         # STALL_M) this long with the walk stick held: the crab is stuck
                                                         # (a riser's edge, a pillar). Walking, everything ahead comes
                                                         # 0.7 m nearer in that time; jammed on a riser the body's sway
                                                         # moved the near zones 2-3 cm and a 1.5 cm test never fired
                                                         # (red3, 45 s on the first flight). The truth reads the base
                                                         # instead (< 5 cm in STALL_S)
UNSTICK_BACK_S, UNSTICK_SIDE_S = 1.0, 0.6                # the recovery: back off, then a half step sideways, then on
TRUTH_LOG_S = 3.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def turn_stick(err: float, cap: float = TURN) -> float:
    """The rx stick that turns toward a heading `err` radians to the left."""
    rate = TURN_RATE * abs(math.degrees(err))
    if rate < 1.0:
        return 0.0
    return -math.copysign(min(cap, TURN_BAND + rate / TURN_PER), err)


def stick_model(ly: float, lx: float) -> tuple[float, float]:
    v = -math.copysign(FWD_PER * max(0.0, abs(ly) - STICK_BAND), ly) if ly else 0.0
    vy = -math.copysign(SIDE_PER * max(0.0, abs(lx) - SIDE_BAND), lx) if lx else 0.0
    return v, vy


class Truth:
    """The world read directly: the control."""

    def __init__(self, env, side):
        self.env, self.side = env, side
        self.me = env.scene[side.me]
        self.mapper = Mapper()
        self.mapper.inflate_m = INFLATE_M
        self.sight = None
        self.balls = []
        _known_lane(self.mapper, side.layout)
        for x, y in side.layout["pillars"]:
            h = track.PILLAR_HALF[0]
            self.mapper.known([((x - MAP_OFF - h, y - h), (x - MAP_OFF + h, y - h)), ((x - MAP_OFF - h, y + h), (x - MAP_OFF + h, y + h)),
                               ((x - MAP_OFF - h, y - h), (x - MAP_OFF - h, y + h)), ((x - MAP_OFF + h, y - h), (x - MAP_OFF + h, y + h))])

        self.stalled = False
        self._trail = []

    def update(self, t: float, frame) -> None:
        x, y, yaw = self.pose()
        self.mapper.x, self.mapper.y, self.mapper.yaw = x - MAP_OFF, y, yaw
        self.balls = []
        for i in range(len(self.side.layout["balls"])):
            bp = self.env.scene["ball" if i == 0 else f"ball{i}"].data.root_link_pos_w[0]
            self.balls.append({"id": i, "x": float(bp[0]), "y": float(bp[1]), "t": t, "n": BALL_SEEN_N, "r": 9.0, "b": 0.0})
        walking = frame is not None and abs(frame.ly) > 0.3
        self._trail = [(t0, x0, y0) for t0, x0, y0 in self._trail if t - t0 <= STALL_S] + [(t, x, y)]
        self.stalled = walking and t - self._trail[0][0] >= STALL_S * 0.95 and \
            math.hypot(x - self._trail[0][1], y - self._trail[0][2]) < 0.05

    def pose(self):
        p = self.me.data.root_link_pos_w[0]
        return float(p[0]), float(p[1]), _yaw(self.me.data.root_link_quat_w[0])

    def block(self, i: int):
        p = self.env.scene[f"block{i}"].data.root_link_pos_w[0]
        return (float(p[0]), float(p[1]), 0.0)

    def edge_ahead(self):
        x, _, _ = self.pose()
        d = self.side.layout["plat2"][1] - x
        return d if 0.0 < d < EDGE_FAR_M and x > self.side.layout["pit"][1] - 0.1 else None

    def bridge_centre(self, dn: float):
        _, y, _ = self.pose()
        return float(self.env.scene["block0"].data.root_link_pos_w[0][1]) - y, None

    def block_left(self, px0: float):
        """A block not in the pit (its top above the deck), nearest the edge, or None."""
        for i in range(len(self.side.layout["blocks"])):
            p = self.env.scene[f"block{i}"].data.root_link_pos_w[0]
            # ...and still short of the notch: a box over it is in or going in, and one lying flat
            # in it read as "still up" when the crab's weight on its rear lifted it a few
            # centimetres, which sent the crab 0.9 m back down the lane to shove it again (red62)
            if float(p[2]) > track.PLAT_H - 0.02 and px0 - 1.0 < float(p[0]) < px0:
                return i
        return None

    def block_at_nose(self):
        x, y, yaw = self.pose()
        best = None
        for i in range(len(self.side.layout["blocks"])):
            p = self.env.scene[f"block{i}"].data.root_link_pos_w[0]
            if float(p[2]) < track.PLAT_H - 0.02:
                continue                                  # dropped: not at the nose any more
            dx, dy = float(p[0]) - x, float(p[1]) - y
            rng, bearing = math.hypot(dx, dy) - track.BLOCK[0], _wrap(math.atan2(dy, dx) - yaw)
            if rng < 0.9 and abs(bearing) < 0.7 and (best is None or rng < best[0]):
                best = (rng, bearing, dy)
        return best


def _known_lane(mapper: Mapper, layout=None) -> None:
    hy = track.LANE_W / 2
    x0, x1 = -0.5 - MAP_OFF, track.LANE_LEN - 0.5 - MAP_OFF
    segments = [((x0, -hy), (x1, -hy)), ((x0, hy), (x1, hy)), ((x0, -hy), (x0, hy)), ((x1, -hy), (x1, hy))]
    if layout is not None:
        # the pillars too, where the course puts them: the planner then threads the real pillars,
        # whatever a drifted pose drew on the map from the sightings -- coming off the stairs the
        # pose was 0.3 m out, the map's pillars with it, and the path it drew between them walked
        # the crab into the one it could see (Andrew: "a disconnect between what the robot sees
        # and where the pathfinder thinks the robot is"). The sightings still place the crab
        h = track.PILLAR_HALF[0]
        for px, py in layout["pillars"]:
            cx = px - MAP_OFF
            corners = [(cx - h, py - h), (cx + h, py - h), (cx + h, py + h), (cx - h, py + h)]
            segments += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    mapper.known(segments)


class Senses:
    """The dToF, the IMU and the encoders, and the map built from them."""

    def __init__(self, env, side):
        self.env, self.side = env, side
        self.sight = Sight(env, side.me, 0.0, slopes=True, round_tall=True)   # the course's two rules (see Sight)
        self.mapper = Mapper()
        self.mapper.x, self.mapper.y, self.mapper.yaw = -MAP_OFF, 0.0, 0.0
        self.x_shift = 0.0                  # every fix's shift along the lane, summed: a mark placed in the
                                            # reckoning's frame (the drop's edge) moves with the frame
        self.mapper.inflate_m = INFLATE_M
        self.mapper.model = stick_model
        self.mapper.stall_test = False
        self.mapper.match_win = 0.0         # no scan matching: against the rails and the blocks it slid the pose 7 cm
                                            # a frame -- 7.5 m in 3 s with no landmark firing (blue4); the landmarks
                                            # below are the fixes
        self.mapper.decay_s = 6.0
        self.mapper.known_absolute = True   # the kerbs and pillars stay put when a fix shifts the map
        _known_lane(self.mapper, side.layout)
        self._feet = None
        self._near_known = None             # cells within KNOWN_CLEAR_M of a known obstacle (see update)
        self._near_known_of = None
        self.leg_v = (0.0, 0.0)
        self.mode_fn = None
        self.balls = []                     # the balls remembered: {id, x, y (world), t (last seen), n (sightings),
                                            # r, b (range and bearing then)} -- the race's memory of things
        self._ball_id = 0
        self.ball_px = None                 # the balls' zones in the last picture, for the readout
        self.fixes = [0, 0, 0, 0, 0, 0]     # per kerb (-y, +y), the heading, the end wall, the ramps, the ball
        self.truth = env.scene[side.me]
        self.pitch = 0.0                    # nose up positive, filtered
        self._pitch_since = None            # when the pitch went past RAMP_PITCH (or came back under it)
        self._on_ramp = False
        self.ramp_fixes = 0                 # of the six, in order: the hill's down on, down off, up on, up off, then
                                            # the pit's ramp on and off (its 0.5 m at 12 deg slipped the feet and ran the
                                            # reckoning 0.6 m ahead in 3 s, blue103: the shove set off 0.7 m early on a
                                            # stale sighting and the box-gone test fired with the box untouched)
        self.ball_fixed = False
        self.on_platform = False            # set by the pilot on the platforms: the kerb is PLAT_H shorter there
        self.ball_stage = False             # set by the pilot from the ball stage on: a ball at the nose is not a jam
        self.in_forest = False              # set by the pilot in the forest: a strafe there is a walk to the stall test
        self.pillar_fixed = False
        self.seen_row_t = None            # when the first row last fixed the pose
        self.blocks_w: dict[int, tuple] = {}   # block index -> (x, y, t) world, the last sighting
        self.block_shoved: dict[int, bool] = {}   # set by the pilot once a block's shove begins: no landmark then
        self.stalled = False
        self._frames = []                   # (t, range image) of the last STALL_S, for the stall test
        self._walk_since = None
        self._turn_since = None
        self._yaw_trail = []                # (t, heading) of the last STALL_S, for the turn's stall test
        self.log = None

    def update(self, t: float, frame) -> None:
        # on the platforms the kerb stands PLAT_H lower above the plane the crab walks on, and under
        # the kerb's height it was a thing: not one kerb fix fired through the pit stage, a
        # sidestep the leg odometry read 0.06 m for 0.33 m went uncorrected, and the shove set off
        # 0.2 m right of the box's centre and spun it 60-80 degrees (five takes of six, 2026-10-23)
        self.sight.kerb_min = KERB_MIN - track.PLAT_H + 0.02 if self.on_platform else KERB_MIN
        fresh = self.sight.update(t)
        ly = float(frame.ly) if frame is not None else 0.0
        lx = float(frame.lx) if frame is not None else 0.0
        _mode, running = self.mode_fn() if self.mode_fn is not None else ("locomotion", True)
        if not running:
            ly = lx = 0.0
        if self.sight.feet is not None:
            if self._feet is not None:
                v = leg_velocity(self._feet[0], self.sight.feet, self._feet[1], self.sight.stance, self.sight.dt,
                                 self.sight.yaw_rate, hint=(-ly, -lx))
                if v is not None:
                    self.leg_v = v
            self._feet = (self.sight.feet.copy(), self.sight.stance.copy())
            v_now = self.leg_v
            # no faster than the sticks could have asked: shoving a block the feet slipped and the
            # legs read 1.6 m/s, 6.5 m of pose in 4 s (blue3); the walk's ceiling is ~0.8 m/s
            # (the floor is 0.45 in the ball stage: the shove's half stick walks 0.33 m/s with the ball, the
            # stick model says 0.14, and capped at 0.3 the reckoning fell 16-18 percent behind over every
            # push, so the ball crossed the line 4 s before the pilot saw it there, red102-104 G)
            vmax = max(0.45 if self.ball_stage else 0.3, abs(stick_model(ly, lx)[0]) * 1.3)
            v_now = (max(-vmax, min(vmax, v_now[0])), max(-0.6, min(0.6, v_now[1])))
            jam = self.sight.last is not None and self.sight.last["jam"] and not self.ball_stage
            if jam or self.stalled:
                # (not the ball at the nose: pushed with rolling friction it stays there, filled the
                # jam band, and the reckoning stood at 20.5 for 6 s while the crab walked to 23.0
                # and over the line without knowing, blue101 D)
                v_now = (0.0, v_now[1])     # the feet slipping against something: no reckoning along the lane, either
                                            # way (zeroed forward only, a crab stuck reversing at the pit's edge reckoned
                                            # itself 2 m back to the shove point and "shoved" from the edge, blue10). (A guard on
                                            # anything tall within 0.32 m of the nose did this too, and in the forest
                                            # every pillar passed close put the pose 1.4 m behind, blue2)
            self.mapper.model = lambda ly_, lx_, v=v_now: v
        self.mapper.step(self.sight, t, ly, lx)
        if self.mapper._known is not None and (self._near_known is None or self._near_known_of is not self.mapper._known):
            self._near_known_of = self.mapper._known      # (computed once: the known cells stay put, known_absolute)
            from scipy import ndimage

            def grown(mask, r):
                k = math.ceil(r / _mapper.RES)
                yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
                return ndimage.binary_dilation(mask, structure=(xx * xx + yy * yy) <= k * k)
            pillars = np.zeros_like(self.mapper._known)
            h = track.PILLAR_HALF[0]
            for px, py in self.side.layout["pillars"]:
                r0, c0 = self.mapper.cell(px - MAP_OFF - h, py - h)
                r1, c1 = self.mapper.cell(px - MAP_OFF + h, py + h)
                pillars[r0:r1 + 1, c0:c1 + 1] = True
            self._near_known = (grown(self.mapper._known, KNOWN_CLEAR_M) | grown(pillars, PILLAR_CLEAR_M)) & ~self.mapper._known
        if self._near_known is not None:
            # where the course knows the obstacle, the known map wins: the sensed returns round a
            # pillar's foot came out up to 0.25 m short of its face (the mapper's floor test has no
            # allowance for range, the sight's has 2.5 cm a metre), the pairs closed the gaps and
            # the planner found no way for 8 s with the pose right to 1 cm (blue101 L, traced)
            g = self.mapper.grid
            g[self._near_known] = np.minimum(g[self._near_known], 0.0)
        hy = track.LANE_W / 2 - 0.17
        self.mapper.y = max(-hy, min(hy, self.mapper.y))
        self.ramp_fix(t)
        # (the strafe is a walk too: the forest's path is followed with strafe-only steps between walking
        # ones, each reset the stall clock, and pinned on a pillar the reckoning ran 0.85 m ahead in 15 s
        # with no stall called, red212)
        walking = frame is not None and (abs(frame.ly) > 0.3 or (self.in_forest and abs(frame.lx) > 0.3)) and running
        if not walking:
            self._walk_since = None
        elif self._walk_since is None:
            self._walk_since = t
        # ...and a turn on the spot that makes no heading: a claw against a pillar's corner held
        # the crab turning back and forth for 20-40 s, and the stall test watched only the walk
        # (blue47, and the long forests of blue27, red29)
        turning = frame is not None and abs(frame.rx) > 0.3 and abs(frame.ly) <= 0.3 and running
        self._yaw_trail = [(t0, y0) for t0, y0 in self._yaw_trail if t - t0 <= STALL_S] + [(t, self.sight.yaw)]
        if not turning:
            self._turn_since = None
        elif self._turn_since is None:
            self._turn_since = t
        turn_stalled = self._turn_since is not None and t - self._turn_since >= STALL_S and \
            t - self._yaw_trail[0][0] >= STALL_S * 0.9 and abs(_wrap(self.sight.yaw - self._yaw_trail[0][1])) < math.radians(5.0)
        if not fresh:
            self.stalled = self.stalled or turn_stalled
            return
        f = self.sight.last
        r = f["range"]
        obj = f["obj"].reshape(r.shape)
        self._frames = [(t0, r0, o0) for t0, r0, o0 in self._frames if t - t0 <= STALL_S] + [(t, r, obj)]
        self.stalled = False
        if self._walk_since is not None and t - self._walk_since >= STALL_S and t - self._frames[0][0] >= STALL_S * 0.9:
            r0, obj0 = self._frames[0][1:]
            # the middle columns only, and only zones on something above the floor in both
            # frames: walking along the lane the kerbs beside the picture keep their range,
            # and the FLOOR keeps its range in every zone however far the crab walks (a ray at
            # a fixed angle meets a plane at one range), so a median over all zones said
            # "stuck" on the flat before the stairs and all through the forest while the
            # crab walked (blue3, blue7). Things ahead come nearer; a pillar the legs are
            # caught on does not.
            mid = np.zeros(r.shape, dtype=bool)
            mid[:, r.shape[1] // 2 - STALL_COLS // 2: r.shape[1] // 2 + STALL_COLS // 2] = True
            both = (r > STALL_FAR) & (r0 > STALL_FAR) & obj & obj0 & mid
            self.stalled = both.sum() > 40 and float(np.median(np.abs(r[both] - r0[both]))) < STALL_M
            # ...and a thing within reach ANYWHERE in the picture that has not moved: a pillar the
            # claws were caught on at the picture's side, the middle columns looking past it down
            # the open lane, held blue27 25 s with no stall called and the reckoning 1.9 m ahead.
            # Not the kerbs, which keep their range beside the crab however far it walks
            near = obj & obj0 & (r < STALL_NEAR) & (r0 < STALL_NEAR) & ~f["kerb_px"].reshape(r.shape)
            if self.ball_px is not None:
                near &= ~self.ball_px       # nor the ball being pushed, which keeps its range at the nose by design
                                            # (the approach to it was called a stall and backed off, blue103 D)
            self.stalled = self.stalled or near.sum() > 15 and float(np.median(np.abs(r[near] - r0[near]))) < STALL_M
        self.stalled = self.stalled or turn_stalled
        self.wall_fix()
        self._see_balls(t)
        self._see_blocks(t)
        self._see_pillars(t)
        self._landmarks(t)

    def _landmarks(self, t: float) -> None:
        """Every pillar, and each block until it is shoved: a sighting matched to the
        nearest told position fixes the pose along both axes (LM_GAIN, LM_STEP_M)."""
        if self.ramp_fixes < 4:
            return
        L = self.side.layout
        x, _y, _yaw = self.pose()
        if x > L["pit"][0] - 0.3:
            return
        marks = [(px, py, "pillar") for px, py in L["pillars"]]
        for i, (bx, by) in enumerate(L["blocks"]):
            if not self.block_shoved.get(i):
                marks.append((bx, by, "block"))
        dx_terms, dy_terms = [], []
        for c in self.sight.last["big"] + self.sight.last["things"]:
            if c["edge"] or not (0.4 <= c["range"] <= 2.5):
                continue
            tall = c["height"] >= 0.27 and c["width"] <= 0.35
            low = BLOCK_H[0] <= c["height"] <= BLOCK_H[1] and BLOCK_W[0] <= c["width"] <= BLOCK_W[1]
            if not (tall or low):
                continue
            kind = "pillar" if tall else "block"
            half = track.PILLAR_HALF[0] if tall else track.BLOCK[0]
            wx, wy = self.mapper.to_world(np.array([[c["near"] + half, c["y"]]]))
            wx, wy = float(wx[0]) + MAP_OFF, float(wy[0])
            near = [(math.hypot(wx - mx, wy - my), mx, my) for mx, my, k in marks if k == kind]
            if not near:
                continue
            d, mx, my = min(near)
            if d > LM_MATCH_M:
                continue
            dx_terms.append(mx - wx)
            dy_terms.append(my - wy)
        if dx_terms:
            dx = max(-LM_STEP_M, min(LM_STEP_M, LM_GAIN * float(np.median(dx_terms))))
            dy = max(-LM_STEP_M, min(LM_STEP_M, LM_GAIN * float(np.median(dy_terms))))
            self.shift(dx, dy)
            self.fixes[4] += 1

    def _see_pillars(self, t: float) -> None:
        """The forest's first row, once, from before it: see PILLAR_FIX_X."""
        if self.pillar_fixed or self.ramp_fixes < 4:
            return
        L = self.side.layout
        row_x = L["pillars"][0][0]
        row_ys = [y for x, y in L["pillars"] if abs(x - row_x) < 0.05]
        x, _y, _yaw = self.pose()
        if x > row_x - 0.5:
            self.pillar_fixed = True
            return
        if x < row_x - PILLAR_FIX_X - 1.0:
            return
        for c in self.sight.last["big"]:
            tall = c["height"] >= 0.27 or c.get("top")
            if not tall or c["edge"] or c["width"] > 0.35 or not (PILLAR_RANGE[0] <= c["range"] <= PILLAR_RANGE[1]):
                continue
            wx, wy = self.mapper.to_world(np.array([[c["near"] + track.PILLAR_HALF[0], c["y"]]]))
            wx, wy = float(wx[0]) + MAP_OFF, float(wy[0])
            if min(abs(wy - ry) for ry in row_ys) > PILLAR_FIX_Y or abs(wx - row_x) > PILLAR_FIX_X:
                continue
            ry = min(row_ys, key=lambda ry_: abs(wy - ry_))
            dx, dy = (row_x - wx) * PILLAR_FIX_GAIN, (ry - wy) * PILLAR_FIX_GAIN
            self.shift(dx, dy)                            # along the lane AND across it: the row's pillars
            self.fixes[4] += 1                            # stand at known y, and the pose across was what
            self.seen_row_t = t                           # walked the crab into them (y as well, 2026-10-23)
            if self.log is not None and (abs(dx) > 0.03 or abs(dy) > 0.03):
                self.log(t, f"pillar fix: the first row seen at {c['range']:.2f} m -> x {dx:+.2f} m, y {dy:+.2f} m")
            return

    def ramp_fix(self, t: float) -> None:
        """The hill as a landmark: see the module docstring."""
        up = self.sight.up()
        pitch = math.atan2(up[0], up[2])                  # the world's up leans back in the base frame when the nose is up
        self.pitch += 0.1 * (pitch - self.pitch)
        if self.ramp_fixes >= 6:
            return
        H = self.side.layout["hill"]
        on = abs(self.pitch) > RAMP_PITCH
        if on != self._on_ramp:
            if self._pitch_since is None:
                self._pitch_since = t
            if t - self._pitch_since < RAMP_HOLD_S:
                return
            self._pitch_since = None
            k = self.ramp_fixes
            want_up = k >= 2
            if on:
                # the wrong slope for this point in the course is not a ramp (the first
                # flight pitched the raised body up 10 deg for 6 s, sense3), and its end
                # is not a ramp's end either. Not latched as "on" either: latched, the
                # climb's nose-up (ry +0.5, 12 deg, now held over the top platform) hid the
                # first ramp's own onset, and the first steep moment after -- the flight
                # down, 3.4 m on -- was taken for it and the pose moved 3.4 m back (red46)
                self._ramp_valid = (self.pitch > 0) == want_up
                if not self._ramp_valid:
                    return
                self._on_ramp = True
            else:
                self._on_ramp = False
                if not getattr(self, "_ramp_valid", False):
                    return
            ramp = H["ramp1"] if k < 2 else H["ramp2"] if k < 4 else self.side.layout["ramp"]
            x_known = (ramp[0] if on else ramp[1]) + RAMP_LEAD_M
            dx = x_known - (self.mapper.x + MAP_OFF)
            if abs(dx) > RAMP_MAX_M:
                return
            self.shift(dx, 0.0)
            self.ramp_fixes += 1
            self.fixes[4] += 1
            if self.log is not None:
                self.log(t, f"ramp fix {k + 1}: pitch {math.degrees(self.pitch):+.0f} deg {'on' if on else 'off'} -> x {x_known:.2f} ({dx:+.2f} m)")
        else:
            self._pitch_since = None

    def wall_fix(self) -> None:
        """Each kerb point, put in the map by the pose, lands some way off the kerb it
        belongs to; the median per kerb is the pose's error across the lane, and the
        slant of a kerb seen over a stretch is the heading's."""
        idx = self.sight.last["kerb"]
        if len(idx) < WALL_MIN_PTS:
            return
        pts = self.sight.last["pts"][idx]
        wx, wy = self.mapper.to_world(pts)
        hy = track.LANE_W / 2
        # the end wall: kerb points across the lane near its known x fix the pose along it
        x_end = track.LANE_LEN - 0.5 - MAP_OFF
        sel = (np.abs(wx - x_end) < END_NEAR_M) & (np.abs(wy) < hy - 0.3)
        if sel.sum() >= END_MIN_PTS and float(np.ptp(wy[sel])) > 0.5:
            dx = x_end - float(np.median(wx[sel]))
            self.shift(max(-WALL_STEP_M, min(WALL_STEP_M, WALL_GAIN * dx)), 0.0)
            self.fixes[3] += 1
        dy_terms, yaw_terms = [], []
        near = WALL_NEAR_PLAT_M if self.on_platform else WALL_NEAR_M
        for w, target in ((0, -hy), (1, hy)):
            sel = (np.abs(wy - target) < near) & (wx > -0.5 - MAP_OFF + 0.2) & (wx < track.LANE_LEN - 0.5 - MAP_OFF - 0.2)
            if sel.sum() < WALL_MIN_PTS:
                continue
            along = wx[sel]
            if float(np.ptp(along)) < WALL_ALONG_M:
                continue
            med = float(np.median(wy[sel]))
            dy_terms.append((target - med, int(sel.sum())))
            self.fixes[w] += 1
            if float(np.ptp(along)) >= YAW_ALONG_M:
                b = float(np.polyfit(along, wy[sel], 1)[0])
                e = math.atan(b)
                if abs(e) <= YAW_MAX:
                    yaw_terms.append((e, int(sel.sum())))
        if dy_terms:
            dy = sum(d * n for d, n in dy_terms) / sum(n for _, n in dy_terms)
            self.shift(0.0, max(-WALL_STEP_M, min(WALL_STEP_M, WALL_GAIN * dy)))
        if yaw_terms:
            e = sum(d * n for d, n in yaw_terms) / sum(n for _, n in yaw_terms)
            self.sight.yaw -= max(-YAW_STEP, min(YAW_STEP, YAW_GAIN * e))
            self.mapper.yaw = self.sight.yaw
            self.fixes[2] += 1

    def _see_blocks(self, t: float) -> None:
        """The blocks: low, wide clusters ahead of the forest. Remembered by the
        nearest to each told spot, world frame."""
        L = self.side.layout
        for c in self.sight.last["things"] + self.sight.last["big"]:
            if c.get("kerb") or not (BLOCK_H[0] <= c["height"] <= BLOCK_H[1] and BLOCK_W[0] <= c["width"] <= BLOCK_W[1]):
                continue
            wx, wy = self.mapper.to_world(np.array([[c["near"] + track.BLOCK[0], c["y"]]]))
            wx, wy = float(wx[0]) + MAP_OFF, float(wy[0])
            k = min(range(len(L["blocks"])), key=lambda i: math.hypot(wx - self.blocks_w.get(i, L["blocks"][i])[0],
                                                                     wy - self.blocks_w.get(i, L["blocks"][i])[1]))
            kx, ky = self.blocks_w.get(k, L["blocks"][k])[:2]
            if math.hypot(wx - kx, wy - ky) < 0.5:
                self.blocks_w[k] = (wx, wy, t)

    def shift(self, dx: float, dy: float) -> None:
        self.mapper.shift(dx, dy)
        self.x_shift += dx

    def _see_balls(self, t: float) -> None:
        """Every ball in sight, remembered as the race remembers its things: a cluster taller
        than the race's things and shorter than the kerb, a ball's width, is matched to the
        remembered ball within BALL_MEM_M or remembered as a new one. Nothing here knows where
        the balls were put; the pilot picks from this memory. The one use of the layout is the
        pose fix: the first ball seen within BALL_FIX_M of a ball's starting place, before any
        shove, fixes the reckoning along the lane (the platforms give it nothing to hold)."""
        L = self.side.layout
        px = np.zeros(self.sight.last["range"].shape, dtype=bool)
        turned_down = []
        for c in self.sight.last["big"] + self.sight.last["things"]:   # under 0.6 m the ball fell out of "big" and
            close = c["range"] < NEAR_M                               # the shove went blind (red28, blue45)
            if c.get("kerb") or (c["edge"] and not close):
                # (at the nose the ball fills the picture's width and touches its edges, as any thing
                # does inside NEAR_M; pushed with rolling friction it stays there, and barred for its
                # edges it was "out of sight" 1.5 s into every shove, red102 E)
                continue
            if close:
                # under the nose the sensor sees a sliver of it: 0.16 m off, 15 cm tall and 8 cm wide
                # (red102 F, traced), under the far gates' 10 cm
                sized = BALL_NEAR_H <= c["height"] <= BALL_H[1] and c["width"] >= BALL_NEAR_W
            else:
                sized = BALL_H[0] <= c["height"] <= BALL_H[1] and BALL_W[0] <= c["width"] <= BALL_W[1] and c["range"] < 3.0
            if not sized:
                if c["range"] < 1.0:
                    turned_down.append(f"r {c['range']:.2f} h {c['height']:.2f} w {c['width']:.2f}")
                continue
            # the cluster is the ball's near face; its centre is a radius further
            r = c["range"] + track.BALL_R
            wx, wy = self.mapper.to_world(np.array([[r * math.cos(c["bearing"]), r * math.sin(c["bearing"])]]))
            wx, wy = float(wx[0]) + MAP_OFF, float(wy[0])
            if wx < L["plat2"][1] + BALL_FACE_M or abs(wy) > track.LANE_W / 2 - 0.05:
                turned_down.append(f"ball-sized at x {wx:.2f} y {wy:+.2f}: the platform's face or the kerb")
                continue
            px |= c["px"].reshape(px.shape)
            near = min(self.balls, key=lambda e: math.hypot(e["x"] - wx, e["y"] - wy), default=None)
            if near is not None and math.hypot(near["x"] - wx, near["y"] - wy) < BALL_MEM_M:
                near.update(x=wx, y=wy, t=t, n=near["n"] + 1, r=c["range"], b=c["bearing"])
                e = near
            else:
                e = {"id": self._ball_id, "x": wx, "y": wy, "t": t, "n": 1, "r": c["range"], "b": c["bearing"]}
                self._ball_id += 1
                self.balls.append(e)
            if not self.ball_fixed and c["range"] < 2.0:
                tx, ty = min(L["balls"], key=lambda b: abs(b[1] - wy))
                if abs(wy - ty) < 0.3 and abs(wx - tx) < BALL_FIX_M:
                    dx = (tx - wx) * BALL_FIX_GAIN
                    self.shift(dx, 0.0)
                    for m in self.balls:
                        m["x"] += dx
                    self.fixes[5] += 1
                    if c["range"] < 1.2:
                        self.ball_fixed = True             # under the nose it moves: no landmark then
                    if self.log is not None and (abs(dx) > 0.03 or self.fixes[5] == 1):
                        self.log(t, f"ball fix: seen at {c['range']:.2f} m -> x {dx:+.2f} m")
        self.balls = [e for e in self.balls if t - e["t"] < BALL_STALE_S]
        self.ball_px = px if px.any() else None
        self.sight.last["ball_px"] = self.ball_px
        if not px.any() and turned_down and os.environ.get("COURSE_TRACE") and self.ball_stage:
            now = time.monotonic()
            if now - getattr(self, "_ball_trace_t", 0.0) > 0.4:
                self._ball_trace_t = now
                print(f"[ball trace] no ball; near clusters turned down: {'; '.join(turned_down)}", flush=True)

    def pose(self):
        return self.mapper.x + MAP_OFF, self.mapper.y, self.mapper.yaw

    def block(self, i: int):
        return self.blocks_w.get(i)

    def block_left(self, px0: float):
        """A block still up before the edge: a low cluster seen within the last second
        in the band before the pit (a dropped one lies flush and is not seen); its
        index by the nearer told lane. None when none."""
        if self.sight.last is None:
            return None
        for c in self.sight.last["things"] + self.sight.last["big"]:
            if c.get("kerb") or not (BLOCK_H[0] <= c["height"] <= BLOCK_H[1]) or c["range"] > 1.2:
                continue
            wx, wy = self.mapper.to_world(np.array([[c["near"] + track.BLOCK[0], c["y"]]]))
            wx, wy = float(wx[0]) + MAP_OFF, float(wy[0])
            if px0 - 0.9 < wx < px0 + 0.25:
                return min(range(len(self.side.layout["blocks"])), key=lambda i: abs(self.side.layout["blocks"][i][1] - wy))
        return None

    def edge_ahead(self):
        """How far ahead the platform ends: the nearest return in the middle columns that
        lies more than EDGE_DROP_M below the floor plane the crab stands on (the ground
        beyond the drop), less the ray's own reach past the edge; None when none within
        EDGE_FAR_M. On the platforms the pilot has no landmark (the pillars are behind,
        the box is shoved, the ball on the ground is below the platform's plane and read
        as floor), and the reckoning ran ahead and began the leap at the notch (blue18)."""
        f = self.sight.last
        if f is None:
            return None
        pts = f["pts"]
        # heights against the four walking feet, not the sight's lowest body point: standing at the
        # hop mark the claws hang out over the drop and droop below the platform, the floor plane
        # went down with them, and the ground 12 cm below read 6 cm below -- the edge, seen at
        # 0.5 m, was lost at 0.25 m, where it mattered (blue26)
        up = self.sight.up()
        floor_ref = float(np.median(self.sight.feet[2:] @ up)) - FOOT_R
        h = pts @ up - floor_ref                          # along the IMU's up (the sight's own plane may be a slope's)
        mid = (f["range"].ravel() > 0) & ~f["self"].ravel() & (np.abs(pts[:, 1]) < 0.15) & (pts[:, 0] > 0.1) & (pts[:, 0] < EDGE_FAR_M)
        sel = mid & (h < -EDGE_DROP_M)
        if os.environ.get("COURSE_TRACE") and mid.sum():
            now = time.monotonic()
            self._edge_traced = now - getattr(self, "_edge_trace_t", 0.0) > 0.4
            if self._edge_traced:
                self._edge_trace_t = now
                hs = h[mid]
                print(f"[edge trace] mid {mid.sum()} below {sel.sum()}  h min {hs.min():+.3f} p10 {np.percentile(hs, 10):+.3f} "
                      f"median {np.median(hs):+.3f}  x of the lowest {pts[mid][np.argmin(hs), 0]:.2f}  far {pts[mid, 0].max():.2f}", flush=True)
        if sel.sum() < 6:
            return None
        # ...and not the notch: the drop's ground runs on, the notch's floor is 0.52 m long with
        # the second platform flush again beyond it. A box pushed 0.55 m aside left the open notch
        # in front of the crab, which read it as the drop and hopped into it (blue58)
        x_first = float(np.percentile(pts[sel, 0], 10))
        beyond = mid & (pts[:, 0] > x_first + 0.35) & (np.abs(h) < 0.04)
        if beyond.sum() >= 6:
            return None
        # each return projected back along its ray to where the ray crossed the platform's plane:
        # the ray grazing the edge lands on ground 12 cm lower well beyond it (from 0.25 m, 0.45 m
        # out with the eye 15 cm up), and taken as the edge itself it put the crab's base at the
        # edge, where it tipped over before the hop (blue26, blue29, red13; a flat 3 cm correction)
        h_eye = float(TOF_ORIGIN @ up) - floor_ref
        if h_eye < 0.05:
            return None                                   # the eye at or below the feet's plane (legs up on a step, a
                                                          # pitch): no geometry to project with -- it read -0.71 once
                                                          # and declared the brink 1.4 m from the edge (blue86)
        x_edge = TOF_ORIGIN[0] + (pts[sel, 0] - TOF_ORIGIN[0]) * h_eye / np.maximum(1e-3, h_eye - h[sel])
        x_edge = x_edge[x_edge > 0.05]
        if len(x_edge) < 6:
            return None
        if os.environ.get("COURSE_TRACE") and getattr(self, "_edge_traced", False):
            tx, _, _ = self.truth_pose()
            print(f"[edge trace]   read: min {np.min(x_edge):.3f} p5 {np.percentile(x_edge, 5):.3f} p10 {np.percentile(x_edge, 10):.3f}"
                  f"  truth {self.side.layout['plat2'][1] - tx:.3f}  eye {h_eye:.3f}", flush=True)
        return float(np.percentile(x_edge, 5))           # near the nearest ray's: the 10th percentile sat rows up the
                                                         # picture, 10-15 cm beyond the edge with 1000 zones below
                                                         # (blue38); the minimum dipped 4 cm on single zones and once
                                                         # placed the brink 12 cm early (blue46). Calibrated: the truth
                                                         # plus 4-5 cm from 0.9 m in to 0.35 (red27, blue44, blue43)

    def bridge_centre(self, dn: float):
        """Where the dropped box lies across the lane, as an offset from the crab's own
        line: the open notch (10 cm down) is seen either side of it and the box's top is
        flush, so the box is the longest flush run across the picture between open bins
        at the notch's range (dn: the notch's near edge ahead, by the reckoning). A box
        half the lane's width walks off sideways under the shove, and a crossing on the
        line it was told crossed into the open notch beside it (red15, red16); the
        lane-wide box needed none of this."""
        f = self.sight.last
        if f is None:
            return None, None
        pts, h = f["pts"], f["h"]
        # the notch found by its own open floor (10 cm down), wherever the reckoning puts it: the
        # leg odometry runs ahead over the shove and a band at the reckoned distance lay short of
        # the notch, all flush, every time (blue54)
        ahead = (f["range"].ravel() > 0) & ~f["self"].ravel() & (pts[:, 0] > 0.2) & (pts[:, 0] < 1.2) & (np.abs(pts[:, 1]) < 1.0)
        open0 = ahead & (h < -0.06)
        if open0.sum() < 15:
            return None, None
        xb = float(np.median(pts[open0, 0]))
        x_far = float(np.percentile(pts[open0, 0], 90))  # the open floor's far end: the far wall's foot
        flush_beyond = ahead & (pts[:, 0] > x_far + 0.05) & (np.abs(h) < 0.035)
        if flush_beyond.sum() < 10:
            # open floor with nothing flush beyond it is the drop past the second platform, not the
            # notch: taken for the notch it kept the edge test off, and two crabs walked off the end
            # looking for an edge (blue88, red65)
            return None, None
        sel = ahead & (pts[:, 0] > xb - 0.25) & (pts[:, 0] < x_far - 0.03)   # (not beyond it: the second platform
                                                                              # is flush too and widened the box's
                                                                              # run by 0.1-0.2, blue73)
        below, flush = sel & (h < -0.06), sel & (np.abs(h) < 0.035)
        if os.environ.get("COURSE_TRACE"):
            now = time.monotonic()
            if now - getattr(self, "_bridge_trace_t", 0.0) > 0.4:
                self._bridge_trace_t = self._bridge_traced_at = now
                print(f"[bridge trace] dn {dn:.2f}, open at {xb:.2f}, far {x_far:.2f}: {int(sel.sum())} in the band, {int(below.sum())} open, {int(flush.sum())} flush", flush=True)
            else:
                self._bridge_traced_at = -1.0
        if below.sum() < 15 or flush.sum() < 15:
            return None, x_far
        bins = np.linspace(-1.0, 1.0, 41)
        open_b = np.histogram(pts[below, 1], bins)[0] > 2
        flush_b = np.histogram(pts[flush, 1], bins)[0] > 2
        best, i = None, 0
        while i < 40:
            if flush_b[i] and not open_b[i]:
                j = i
                while j < 40 and flush_b[j] and not open_b[j]:
                    j += 1
                if best is None or j - i > best[1] - best[0]:
                    best = (i, j)
                i = j
            else:
                i += 1
        if best is None:
            return None, x_far
        lo, hi = float(bins[best[0]]), float(bins[best[1]])
        # a run reaching the picture's own edge at that range (the sight sees ~45 deg either way,
        # 0.5 m aside at 0.5 m) is one-sided: the box's far side is out of the picture and its
        # width is known. Taken as two-sided, a box 0.4 m aside read as centred, and the crab
        # crossed into the open notch beside it (blue52)
        y_seen = pts[sel, 1]
        at_left, at_right = lo <= float(y_seen.min()) + 0.06, hi >= float(y_seen.max()) - 0.06
        if at_left and at_right:
            c = None                                      # the box fills the picture: nothing to go by
        elif at_left:
            c = hi - track.BLOCK[1] if hi - lo > 0.3 else None
        elif at_right:
            c = lo + track.BLOCK[1] if hi - lo > 0.3 else None
        elif hi - lo < 0.6 or hi - lo > 2 * track.BLOCK[1] + 0.35:
            c = None                                      # not the box: a sliver, or the platform short of the notch
        else:
            c = (lo + hi) / 2
        if os.environ.get("COURSE_TRACE") and getattr(self, "_bridge_trace_t", 0.0) == getattr(self, "_bridge_traced_at", -1.0):
            print(f"[bridge trace]   flush run {lo:+.2f}..{hi:+.2f} (seen {y_seen.min():+.2f}..{y_seen.max():+.2f}),"
                  f" left {at_left} right {at_right} -> centre {c}", flush=True)
        return c, x_far

    def block_at_nose(self):
        """The nearest low wide cluster right ahead: (range, bearing), or None. What the
        shove steers on, and what tells it the block has dropped: a block in the pit
        lies flush with the deck and is not seen at all."""
        best = None
        if self.sight.last is None:
            return None
        for c in self.sight.last["things"] + self.sight.last["big"]:
            # up to NOSE_TALL_M: a box tipping about the notch's edge raises its rear to 15-25 cm, and
            # gated at the box's own height it read as "gone" and the nudge stopped with the box
            # hanging on the edge, three times (red8). Flat in the notch it is flush and not seen.
            if c.get("kerb") or c["height"] > NOSE_TALL_M or c["height"] < 0.04 or c["range"] > 0.9 or abs(c["bearing"]) > 0.7:
                continue
            if abs(c["y"]) > track.BLOCK[1] - 0.05:
                continue                                  # beside the box's lane: the notch's far wall either side of a
                                                          # dropped box, 10 cm tall at 0.4 m, kept the box "not gone"
                                                          # and sent the crab back down the lane for it (blue30)
            if best is None or c["range"] < best[0]:
                best = (c["range"], c["bearing"], c["y"])
        return best

    def truth_pose(self):
        p = self.truth.data.root_link_pos_w[0]
        return float(p[0]), float(p[1]), _yaw(self.truth.data.root_link_quat_w[0])


def make_pilot(env, player, side):
    from controller import GamepadState

    label = side.name
    referee = side.referee
    L = side.layout
    H = L["hill"]
    performer = Performer(player, side.presses, label=label)
    moves = MOVES[side.name]
    eyes = Truth(env, side) if side.truth else Senses(env, side)
    if not side.truth:
        eyes.mode_fn = lambda: (player.fsm.mode, player.fsm.running_policy)
        eyes.log = lambda t, msg: print(f"\n[{label}] {t:6.2f}s  {msg}")
    st = {"intent": None, "stage": "", "target": None, "ball": None, "balls": (), "tgt": None, "over_ids": set(), "over_t": {}, "final_look": False, "ball_back": None, "scan": None, "off": MAP_OFF, "marks": (L["pit"][0], L["pit"][1], L["finish"]),
          "cue_id": referee.cue_id, "face": None, "raised": None, "reset": None, "plan_t": -1e9, "waypoint": None,
          "truth_log": 0.0, "frame": None, "push_since": None, "done": False, "arrived": None, "unstick": None, "unsticks": 0, "block": 0, "push_y": 0.0, "back_until": None, "dash_t": -1e9, "block_gone": None, "nudge_until": None, "reshoves": 0, "backing": False, "back_since": 0.0, "leap_t": None, "creep_t": None, "brink_x": None, "brink_t": None, "edge_seen": [], "lined_t": None, "bridge_ys": [], "bridge_logged": False, "cross_t": None, "notch_ahead": None, "at_behind": False, "centring": False, "threshold_t": None, "cue_t": None, "sidestepping": False, "side_turning": False, "crossed": None}

    def log(t, msg):
        print(f"\n[{label}] {t:6.2f}s  {msg}")

    def walking() -> bool:
        fsm = player.fsm
        return fsm.mode in WALKING and fsm.running_policy

    STAGES = ("hill", "forest", "pit", "leap", "ball", "finish")

    def stage(t, name, intent):
        if st["stage"] != name or st["intent"] != intent:
            st["stage"], st["intent"] = name, intent
            log(t, f"{name}: {intent}")

    def reached(name) -> bool:
        """A stage once entered is not left for an earlier one (the reckoning hovering
        at a boundary flapped hill/forest twenty times a second, sense2)."""
        return STAGES.index(st["stage"]) >= STAGES.index(name) if st["stage"] in STAGES else False

    def hold(x, y, yaw, stick, y_goal=0.0, strafe=True):
        """Straight along the lane: the heading held at 0, the strafe stick easing back
        to `y_goal` (not on the stairs: a strafe there is a foot on a riser's edge).
        The turn and the walk are never sent together: past HOLD_TURN_DEG the crab
        turns on the spot, inside it walks with the turn stick at rest (the walk and a
        turn stick together froze the posture policy -- the fetch's creep deadlock,
        w17, 2026-10-13 -- and truth2 stood on the first platform with the walk stick
        held for 25 s)."""
        err = _wrap(0.0 - yaw)
        if abs(err) > math.radians(HOLD_TURN_DEG if strafe else STAIR_TURN_DEG):
            return GamepadState(rx=turn_stick(err))
        lx = 0.0
        if strafe and abs(y - y_goal) > 0.12:
            lx = max(-0.5, min(0.5, -1.2 * (y_goal - y)))      # lx > 0 strafes right (-y)
        return GamepadState(ly=-stick, lx=lx)

    def head_to(x, y, yaw, gx, gy, stick=WALK):
        """The sticks that walk toward (gx, gy): turn to face it, walk once facing."""
        err = _wrap(math.atan2(gy - y, gx - x) - yaw)
        if abs(err) > math.radians(FACE_DEG):
            return GamepadState(rx=turn_stick(err, cap=1.0 if abs(err) > math.radians(60) else TURN))
        return GamepadState(ly=-stick, rx=turn_stick(err))

    def along(path, d_want):
        """The point about `d_want` metres along a path (its end if shorter), None for no path."""
        if not path:
            return None
        d, pt = 0.0, path[-1]
        for (x0, y0), (x1, y1) in itertools.pairwise(path):
            d += math.hypot(x1 - x0, y1 - y0)
            if d >= d_want:
                return (x1, y1)
        return pt

    def planning_grid():
        """The map's occupied cells, inflated by INFLATE_M, without sensed specks: a sensed
        blob under PLAN_SPECK_CELLS cells is a stray return, not an obstacle. One cell in the
        middle of a 0.46 m gap, grown by the clearance, closed the lane at the third row
        (red212). The known cells always count."""
        from scipy import ndimage
        m = eyes.mapper
        occ = m.grid > _mapper.OCC
        known = m._known if m._known is not None else np.zeros_like(occ)
        sensed = occ & ~known
        lab, n = ndimage.label(sensed, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(sensed, lab, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:] = sizes >= PLAN_SPECK_CELLS
            sensed = keep[lab]
        k = math.ceil(m.inflate_m / _mapper.RES)
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        disk = (xx * xx + yy * yy) * _mapper.RES ** 2 <= m.inflate_m ** 2
        return ndimage.binary_dilation((occ & known) | sensed, structure=disk)

    def plan_to(t, x, y, yaw, gx, gy, stick=WALK, crab_walk=False):
        """Walk toward the map's way to (gx, gy) (world), replanned every PLAN_DT: A* on the
        map (the kerbs, the pillars where the course has them, and whatever the sensor has
        seen), keeping to the middle of a gap (PLAN_MARGIN_M); with no way at INFLATE_M, again
        tighter; with no way at all, stand rather than walk blind."""
        if t - st["plan_t"] >= PLAN_DT or st["waypoint"] is None:
            st["plan_t"] = t
            eyes.mapper.inflate_m = INFLATE_M
            blocked0 = planning_grid()
            path = eyes.mapper.plan((gx - MAP_OFF, gy), inflate=blocked0.copy(), margin=PLAN_MARGIN_M, margin_cost=PLAN_MARGIN_COST)
            pt = along(path, LOOKAHEAD)
            if pt is None:
                # no way: the crab itself inside an obstacle's inflation (beside a pillar, at the kerb).
                # Its own neighbourhood is cleared and the plan drawn again; everything else keeps its
                # clearance (shrinking every inflation instead opened the 0.42 m gaps at the kerbs)
                blocked = blocked0
                r0, c0 = eyes.mapper.cell(eyes.mapper.x, eyes.mapper.y)
                k = math.ceil(PLAN_ESCAPE_M / _mapper.RES)
                rr, cc = np.ogrid[-k:k + 1, -k:k + 1]
                disk = rr * rr + cc * cc <= k * k
                rs, cs = slice(max(0, r0 - k), r0 + k + 1), slice(max(0, c0 - k), c0 + k + 1)
                sub = blocked[rs, cs]
                sub[disk[: sub.shape[0], : sub.shape[1]]] = False
                path = eyes.mapper.plan((gx - MAP_OFF, gy), inflate=blocked, margin=PLAN_MARGIN_M, margin_cost=PLAN_MARGIN_COST)
                pt = along(path, LOOKAHEAD)
            old = st["waypoint"]
            if pt is not None and old is not None and st.get("wp_goal") == (gx, gy):
                new_w = (pt[0] + MAP_OFF, pt[1])
                held = t - st.get("wp_t", -1e9)
                reached_old = math.hypot(old[0] - x, old[1] - y) < 0.15
                if math.hypot(new_w[0] - old[0], new_w[1] - old[1]) > PLAN_SWING_M and held < PLAN_COMMIT_S and not reached_old:
                    # a swing to the other side of an obstacle is not taken until the old way is reached
                    # or has been kept PLAN_COMMIT_S: square in front of the centre pillar, its two gaps
                    # cost the same and the path flipped between them every replan, the crab sidestepping
                    # left and right for 11 s (red102 M)
                    pt = (old[0] - MAP_OFF, old[1])
                else:
                    st["wp_t"] = t
            else:
                st["wp_t"] = t
            st["wp_goal"] = (gx, gy)
            if pt is None and t - st.get("noway_log", -1e9) > 2.0:
                st["noway_log"] = t
                log(t, f"the planner finds no way: stands and plans again {getattr(eyes.mapper, 'last_plan', '')}")
                if os.environ.get("COURSE_TRACE_DIR"):
                    # the map as the planner had it, for the reason (the base a red arrow, the goal ringed)
                    eyes.mapper.marks["goal"] = (gx - MAP_OFF, gy)
                    path = os.path.join(os.environ["COURSE_TRACE_DIR"], f"noway_{label}_{t:06.2f}.png")
                    eyes.mapper.picture(path, span=3.0)
                    np.savez_compressed(path[:-4] + ".npz", grid=eyes.mapper.grid, known=eyes.mapper._known,
                                        x=eyes.mapper.x, y=eyes.mapper.y, inflate=eyes.mapper.inflate_m)
                    log(t, f"the map then: {path}")
            st["waypoint"] = (pt[0] + MAP_OFF, pt[1]) if pt is not None else None
            far = along(path, FOREST_HEAD_M) if pt is not None else None
            st["wp_far"] = (far[0] + MAP_OFF, far[1]) if far is not None else st["waypoint"]
            st["target"] = (gx - MAP_OFF, gy)
        if st["waypoint"] is None:
            return GamepadState()
        wx, wy = st["waypoint"]
        if crab_walk:
            # facing along the path (capped across the lane), turned while walking; the waypoint's
            # bearing beyond what the heading takes up is the strafe's
            fx, fy = st.get("wp_far") or (wx, wy)
            cap = math.radians(FOREST_FACE_DEG)
            want = max(-cap, min(cap, math.atan2(fy - y, fx - x)))
            err = _wrap(want - yaw)
            if abs(err) > math.radians(FOREST_SPOT_DEG):
                return GamepadState(rx=turn_stick(err))
            b = _wrap(math.atan2(wy - y, wx - x) - yaw)
            if math.cos(b) < 0.0:
                # the waypoint behind (a swing round a pillar just passed): turn toward it on the spot
                return GamepadState(rx=turn_stick(b))
            side = -math.copysign(1.0, math.sin(b))                    # the waypoint to the left: lx negative
            lat = abs(math.sin(b))
            lx = side * min(FOREST_STRAFE, 0.2 + 1.0 * lat) if lat > 0.15 else 0.0
            return GamepadState(ly=-stick, lx=lx, rx=turn_stick(err))
        return head_to(x, y, yaw, wx, wy, stick=stick)

    def unstick(t):
        """Stuck: back off, a half step to the side (alternating), and on again.
        None when nothing is to be done."""
        if st["unstick"] is None:
            if eyes.stalled and st["push_since"] is None and st["stage"] != "leap":
                st["unstick"] = t
                st["unsticks"] += 1
                log(t, f"stuck ({st['stage']}): backs off and tries again ({st['unsticks']})")
            else:
                return None
        since = t - st["unstick"]
        if st["backing"]:
            # stuck while backing off the pit's edge: a step toward the lane's middle, away from
            # the rail or the jammed block beside, and nothing forward or back (the recovery's
            # own reverse kept the crab against it, blue9: 95 s at the edge)
            if since < UNSTICK_SIDE_S + 0.4:
                _, y_now, _ = eyes.pose()
                return GamepadState(lx=0.5 if y_now > 0 else -0.5)
            st["unstick"] = None
            return None
        if since < UNSTICK_BACK_S:
            return GamepadState(ly=0.6)
        if since < UNSTICK_BACK_S + UNSTICK_SIDE_S:
            return GamepadState(lx=0.5 if st["unsticks"] % 2 else -0.5)
        st["unstick"] = None
        st["plan_t"] = -1e9
        return None

    def act(t):
        x, y, yaw = eyes.pose()
        eyes.on_platform = reached("pit") and st["leap_t"] is None          # up the ramp until the leap's dash
        eyes.ball_stage = reached("ball")
        eyes.in_forest = st["stage"] == "forest"
        mem = [e for e in eyes.balls if e["id"] not in st["over_ids"]]
        tgt = next((e for e in mem if e["id"] == st["tgt"]), None)
        ball = (tgt["x"], tgt["y"], tgt["t"]) if tgt is not None else None
        st["ball"] = (ball[0] - MAP_OFF, ball[1]) if ball is not None else None
        st["balls"] = [(e["x"] - MAP_OFF, e["y"]) for e in mem if e is not tgt and e["n"] >= BALL_SEEN_N]
        frame = unstick(t)
        if frame is not None:
            return frame
        # -- the hill ---------------------------------------------------------------
        if (x < H["end"] + 0.2 or abs(getattr(eyes, "pitch", 0.0)) > math.radians(5.0) and x < H["end"] + 0.6) and not reached("forest"):
            # ...until the crab is off the last step and level: the forest's door stand began on the
            # last riser, nose down 10 deg, and the slow walk off it took 15 s (blue86)
            if SIDEWAYS_DOWN == side.name and H["flight2"][0] - 0.45 < x < H["flight2"][1] + 0.35:
                # the last flight sideways, facing the camera (Andrew: "make the blue robot attempt to
                # descend side-ways"): the turn on the spot first, then the strafe alone, the heading
                # corrected by turns alone (the walk and the turn never together); probed on the
                # course's flight (4 cm x 3, 30 cm treads) facing -90 at lx -0.7: down and on, level
                err = _wrap(math.radians(SIDE_HEADING_DEG) - yaw)
                if abs(err) > math.radians(SIDE_TURN_DEG) or st["side_turning"] and abs(err) > math.radians(3.0):
                    st["side_turning"] = True             # (with hysteresis: at one threshold the turn and the
                    stage(t, "hill", "turns to take the last flight sideways")   # strafe toggled every step, blue52)
                    return GamepadState(rx=turn_stick(err))
                st["side_turning"] = False
                stage(t, "hill", "down the last flight sideways")
                return GamepadState(lx=-DOWN_STICK)       # facing -y, +x is the crab's left: lx negative
            stage(t, "hill", "stairs up, ramp down, ramp up, stairs down")
            on_stairs = H["flight1"][0] - 0.3 < x < H["plat1"][0] + 0.2 or H["flight2"][0] - 0.3 < x < H["flight2"][1] + 0.3
            if RAISE and st["raised"] is None and x > H["flight1"][0] - RAISE_BEFORE_M:
                st["raised"] = t
                log(t, "raises the body for the stairs")
            if st["raised"] is not None and t - st["raised"] < RAISE_S:
                return GamepadState(ly=-HILL_STICK, ry=-1.0, buttons=frozenset({"R3"}))
            flight_over = (x > H["plat1"][0] + 0.25) if side.truth else (eyes.ramp_fixes >= 1)
            if st["reset"] is None and st["raised"] is not None and flight_over:
                st["reset"] = t
                log(t, "stands back down")            # (keyed on the reckoned x, sense1/2 lowered the body on the
                                                      # second step -- the reckoning runs 2:1 on the risers -- and jammed)
            if st["reset"] is not None and t - st["reset"] < 0.3:
                return GamepadState(ly=-HILL_STICK, buttons=frozenset({"R3"}))
            going_down = H["flight2"][0] - 0.5 < x < H["flight2"][1] + 0.3
            # the nose stays up over the platform at the top as well: the reckoning runs ahead on the
            # risers (2:1 at worst, 0.65 m by the top in red25), and keyed on the flight's own end it
            # levelled the crab with its claws still at the last riser, where it stood for the take
            climbing = H["flight1"][0] - 0.4 < x < H["plat1"][1] - 0.2
            frame = hold(x, y, yaw, DOWN_STICK if going_down else HILL_STICK, strafe=not on_stairs)
            if climbing and CLIMB == "pitch" and frame.ly:
                frame = GamepadState(ly=frame.ly, lx=frame.lx, ry=CLIMB_RY)
            elif climbing and CLIMB == "dash":
                if t - st["dash_t"] > DASH_EVERY_S and walking():
                    st["dash_t"] = t
                    log(t, "hops a step")
                if t - st["dash_t"] < 0.3:
                    press = side.presses["dash"]
                    return GamepadState(buttons=press.buttons, hat_x=press.hat_x, hat_y=press.hat_y)
            return frame
        # -- the forest -------------------------------------------------------------
        fx1 = L["forest"][1]
        if x < fx1 + 0.3 and not reached("pit"):
            if st["threshold_t"] is None:
                st["threshold_t"] = t
            seen_row = getattr(eyes, "seen_row_t", None)
            if t - st["threshold_t"] < THRESHOLD_S and not (seen_row is not None and t - seen_row > 0.6 and t - st["threshold_t"] > 0.8):
                # a moment standing at the forest's door for the first row to place the crab (x and y)
                # before the way through is planned: planned off the stairs, with the pose 0.3 m out,
                # the path threaded pillars that were not where the map had them
                err = _wrap(0.0 - yaw)
                if abs(err) > math.radians(HOLD_TURN_DEG):
                    return GamepadState(rx=turn_stick(err))
                stage(t, "forest", "at the forest's door: the first row places the crab")
                return GamepadState()
            # the way through is the planner's: A* on the map, replanned four times a second, the walk
            # steering at the point LOOKAHEAD along it (Andrew: "the weaving through the pillars needs
            # to be with our path planner")
            stage(t, "forest", "through the pillars on the planner's path")
            return plan_to(t, x, y, yaw, fx1 + 0.8, 0.0, stick=FOREST_WALK, crab_walk=True)
        # -- the pit ----------------------------------------------------------------
        px0, px1 = L["pit"]
        # (and only from a level body before the creep: crossing a box lying 2 cm proud the crab was
        # pitched, the plane through its feet with it, and the second platform read as the drop
        # 0.9 m early, red33 -- the creep's own reading is taken nose down and calibrated so)
        level = st["creep_t"] is not None or abs(getattr(eyes, "pitch", 0.0)) < math.radians(2.5)
        na = st["notch_ahead"]
        notch_behind = na is None or t - na[1] > 1.0 or na[0] < -0.4    # the open notch not in sight ahead for a second
        edge_raw = eyes.edge_ahead() if st["crossed"] and level and notch_behind else None   # the notch's floor is never the
                                                                      # drop: read as it with the crab still on the first
                                                                      # platform, it ended the crossing and began the creep
                                                                      # 1.4 m from the drop (blue86)
        if edge_raw is not None and 0.45 < edge_raw < 1.3:
            # the edge is placed while it is seen from 0.45 to 1.3 m, in a frame that follows the
            # reckoning's fixes, and the last half metre walked to it on the reckoning: nearer, the
            # sight's lowest ray meets the platform 0.3 m out, so the first return past the edge is
            # that ray's, not the edge's, and the edge read 0.24 m off with the base 0.12 m from it
            # (blue34, blue35, red18); and a ball fix in the approach moved the frame 0.28 m under
            # a mark placed in it (red19)
            st["edge_seen"] = (st["edge_seen"] + [x + edge_raw - getattr(eyes, "x_shift", 0.0)])[-5:]
        edge = float(np.median(st["edge_seen"])) + getattr(eyes, "x_shift", 0.0) - x if st["edge_seen"] else edge_raw
        # the crossing runs until the drop beyond the second platform is in sight (1.3 m), not to
        # a reckoned x alone: the reckoning lagged 0.3 m over the bridge and the crossing ran at
        # full stick to 0.3 m from the drop, which the approach could not take back (red19, red20)
        if not reached("ball") and (st["block"] < len(L["blocks"]) or x < px1 + 0.4) and not (st["crossed"] and notch_behind and edge is not None and edge < 1.3):
            k = st["block"]
            if st["back_until"] is not None and t < st["back_until"]:
                return GamepadState(ly=0.6)
            if k < len(L["blocks"]):
                # where the course puts the block, not where a sighting says: side by side the two
                # blocks read as one wide low cluster the sight cuts into pieces, and matched to the
                # nearer told block the pieces pulled both estimates to the middle -- blue lined up
                # on block 2's edge for "block 1", yawed it and jammed it at the edge (blue9). The
                # pose is good here (the pillars and the blocks themselves are landmarks), and a
                # block does not move until it is shoved.
                kx, ky = L["blocks"][k]
                if st["push_since"] is None:
                    behind = (kx - track.BLOCK[0] - BLOCK_BEHIND_M, ky)
                    if x > behind[0] + 0.15 or (st["backing"] and x > behind[0] - 0.1):
                        if not st["backing"]:
                            st["backing"], st["back_since"] = True, t
                        if t - st["back_since"] > BACK_GIVE_UP_S:
                            # not getting back: cross on what is in the pit (the bridge is a block wide)
                            log(t, f"cannot get back for block {k + 1}: leaves it")
                            st["block"], st["backing"], st["plan_t"] = len(L["blocks"]), False, -1e9
                            return GamepadState()
                        # ahead of the point (after the first block, at the pit's edge): back down the
                        # lane first, nose to the pit. Turned round and walked, the straight way back
                        # ran through the other block and the crab shoved it backwards for 50 s (truth9)
                        stage(t, "pit", f"backs down the lane for block {k + 1}")
                        if int(t) != int(t - 0.005) and int(t) % 2 == 0:
                            tx, ty, tyaw = eyes.truth_pose() if hasattr(eyes, "truth_pose") else eyes.pose()
                            print(f"\n[{label}] {t:6.2f}s  backing: truth x {tx:.2f} y {ty:+.2f} yaw {math.degrees(tyaw):+.0f}, "
                                  f"mode {player.fsm.mode}{'' if player.fsm.running_policy else ' (ramping)'}")
                        err = _wrap(0.0 - yaw)
                        if abs(err) > math.radians(HOLD_TURN_DEG):
                            return GamepadState(rx=turn_stick(err))
                        # straight back along the lane it is in, no strafe: toward the other block's
                        # lane from on top of the first block put the crab in the empty half of the
                        # pit (truth10), and strafing over while still beside the other block backed
                        # the crab into it and shoved it 0.3 m the wrong way (blue10)
                        return GamepadState(ly=0.7)
                    st["backing"] = False
                    # at the point behind, latched: the test on the reckoned x alone toggled with the
                    # reckoning's jitter, the walk and the sidestep took turns every step for 50 s and
                    # the crab crept to the kerb with its lateral reckoning a metre out (red67)
                    if abs(behind[0] - x) < 0.35:
                        st["at_behind"] = True
                    e_c = ky + SHOVE_BIAS_Y - y
                    if st["at_behind"] and (abs(e_c) > 0.06 and abs(e_c) <= 0.25 or st["centring"] and abs(e_c) > SHOVE_CENTRE_M):
                        # centred to the centimetre before the shove, with hysteresis (0.06 in, 0.02 out):
                        # pushed 8 cm off its middle an unguided box turned and slid 0.55 m aside (blue12)
                        st["centring"] = True
                        stage(t, "pit", f"centres on block {k + 1}")
                        return GamepadState(lx=-math.copysign(SHOVE_SIDE_MIN, e_c))
                    st["centring"] = False
                    if st["at_behind"] and abs(_wrap(0.0 - yaw)) > math.radians(SHOVE_TURN_DEG):
                        return GamepadState(rx=turn_stick(_wrap(0.0 - yaw)))
                    if st["at_behind"] and st["lined_t"] is None:
                        st["lined_t"] = t                 # centred and square: a beat standing still before the push
                    if st["at_behind"] and t - st["lined_t"] < LINE_UP_S:
                        stage(t, "pit", f"lined up on block {k + 1}")
                        return GamepadState()
                    if abs(behind[0] - x) < 0.35:
                        # level with the point: across to the box's lane by the strafe, then the shove
                        # (a plan to the point from here flapped against the strafe at the lane's
                        # edge twenty times a second, red7)
                        if abs(ky - y) > 0.06:
                            stage(t, "pit", f"across to block {k + 1}'s lane")
                            err = _wrap(0.0 - yaw)
                            if abs(err) > math.radians(HOLD_TURN_DEG):
                                return GamepadState(rx=turn_stick(err))
                            return GamepadState(lx=-math.copysign(0.45, ky - y))     # to the left: strafe left
                    elif math.hypot(behind[0] - x, behind[1] - y) > 0.18:
                        stage(t, "pit", f"to the point behind block {k + 1}")
                        # straight up the ramp and along the platform, not a plan: the ramp and the
                        # platform's face read as a wall across the lane in the depth image, the map
                        # had no way through, and blue stood at the ramp's foot for 30 s (blue16)
                        return hold(x, y, yaw, WALK, y_goal=behind[1])
                    st["push_since"], st["push_y"], st["block_gone"] = t, ky + SHOVE_BIAS_Y, None
                    if hasattr(eyes, "block_shoved"):
                        eyes.block_shoved[k] = True
                stage(t, "pit", f"shoves block {k + 1} into the pit")
                st["target"] = (px0 - MAP_OFF, st["push_y"])
                nose = eyes.block_at_nose()
                if int(t * 2) != int((t - 0.005) * 2) and os.environ.get("COURSE_TRACE"):            # twice a second, the shove's geometry against the truth
                    bp = env.scene[f"block{k}"].data.root_link_pos_w[0]
                    tx, ty, _ = eyes.truth_pose() if hasattr(eyes, "truth_pose") else eyes.pose()
                    print(f"\n[{label}] {t:6.2f}s  shove: base x {tx:.2f} y {ty:+.2f}, block x {float(bp[0]):.2f} y {float(bp[1]):+.2f} "
                          f"(face {float(bp[0]) - track.BLOCK[0] - tx:.2f} ahead), nose {None if nose is None else round(nose[0], 2)}")
                since = t - st["push_since"]
                # the block drops into the pit and is seen no more (flush with the deck): gone
                # from the nose for BLOCK_GONE_S after a shove of at least 1.5 s, it is in
                if (nose is None or nose[0] > NOSE_GONE_M) and since > 1.5:
                    # ...a thing farther than the box's face can be is the notch's far wall seen
                    # beside a box that has walked aside, not the box (blue55 shoved 25 s on it)
                    st["block_gone"] = st["block_gone"] or t
                else:
                    st["block_gone"] = None
                dropped = st["block_gone"] is not None and t - st["block_gone"] > BLOCK_GONE_S and x > px0 - 1.0 \
                    and x > L["blocks"][k][0] - track.BLOCK[0] - 0.45      # ...and the crab within reach of where the box
                                                                           # stood: 1 m short of it, "gone" was never there (blue71)
                # the shove runs on steadily until the box is gone or the nose is over the edge: the
                # two drops that worked came during the shove (truth24, blue19), while a shove that
                # stopped at the edge with the box tipped but not over, and nudged in pulses, pushed
                # a rear end the tilt had raised out of the shell's reach and the claws lifted it
                # instead -- the box hung on the edge (red8, blue20). The pulses were for wedging
                # in a trench 2 cm longer than the box; the 0.58 m notch has the room.
                at_edge = x > px0 + SHOVE_PAST_M
                if at_edge and st["nudge_until"] is None and not dropped:
                    st["nudge_until"] = t
                    log(t, f"block {k + 1}: the nose is over the edge and the box is not gone")
                done = dropped or st["nudge_until"] is not None
                if done or since > 25.0:
                    bp = env.scene[f"block{k}"].data.root_link_pos_w[0]
                    tx, ty, _ = eyes.truth_pose() if hasattr(eyes, "truth_pose") else eyes.pose()
                    log(t, (f"block {k + 1} is in the pit" if dropped else (f"block {k + 1} still up at the edge"
                                                                             if done else f"gives up on block {k + 1}"))
                        + f"  [truth: base x {tx:.2f}, block x {float(bp[0]):.2f} z {float(bp[2]):.3f} "
                        + f"yaw {math.degrees(_yaw(env.scene[f'block{k}'].data.root_link_quat_w[0])):+.0f}, edge {px0:.2f}]")
                    st["block"], st["push_since"], st["back_until"], st["plan_t"], st["nudge_until"] = k + 1, None, t + 1.2, -1e9, None
                    return GamepadState(ly=0.6)
                err = _wrap(0.0 - yaw)
                if abs(err) > math.radians(HOLD_TURN_DEG):
                    return GamepadState(rx=turn_stick(err))
                # the lateral stick holds the box's centre line, past the strafe's dead band as soon as
                # the crab is SHOVE_CENTRE_M off it (at 1.5 per metre an 8 cm error was a 0.12 stick,
                # inside the band, and the box turned away, blue12)
                # ...but loosely once pushing, SHOVE_HOLD_M either way with no stick inside it: the
                # crab's gait drifts a few centimetres left over the shove, the hold's strafe to the
                # right leans the claws on the box's face, and the box is dragged right -- to -y in
                # fifteen takes of fifteen, 0.14-0.55 m, square (an off-centre push would have turned
                # it: the line set 0.12 m right to "lean against" the drift spun it 50-70 degrees,
                # blue59, red47, red48). Past the band the strafe is the dead band's minimum
                # (no squaring on the spot once pushing: turning with the claws on the box's face
                # dragged its near side round, 50-56 degrees in blue82 and blue83; the line-up before
                # the push is where the squaring happens, and the heading is held by the gait)
                e_y = st["push_y"] - y
                lx = -math.copysign(SHOVE_SIDE_MIN, e_y) if abs(e_y) > SHOVE_HOLD_M else 0.0
                return GamepadState(ly=-PUSH, lx=lx)
            left = eyes.block_left(px0)                       # a block still up at the edge: not a bridge yet
            if left is not None and st["reshoves"] < RESHOVES and x < px0 - 0.1:
                st["reshoves"] += 1
                st["block"], st["plan_t"] = left, -1e9
                log(t, f"block {left + 1} is still at the edge: shoves it again ({st['reshoves']})")
                return plan_to(t, x, y, yaw, px0 - 0.9, L["blocks"][left][1])
            st["crossed"] = st["crossed"] or t                        # (from now the drop ahead is the platform's end)
            c, x_far = eyes.bridge_centre(max(0.15, px0 - x))    # whatever the reckoning says: 0.6 m ahead after a
            if c is not None:                                     # shove it never asked (blue58); the reading finds
                st["bridge_ys"] = (st["bridge_ys"] + [y + c])[-7:]   # the notch by its own open floor or returns None
            if x_far is not None:
                st["notch_ahead"] = (x_far - 0.55, t)                  # the near edge, by the sight: the far wall's foot
                                                                      # less the notch's length (the reckoning runs
                                                                      # up to a metre ahead over the shove)
            lane_known = len(st["bridge_ys"]) >= 3
            lane_y = float(np.median(st["bridge_ys"])) if lane_known else L["blocks"][0][1]
            if lane_known and not st["bridge_logged"]:
                st["bridge_logged"] = True
                log(t, f"the bridge lies at y {lane_y:+.2f}: crosses there")
            if st["cross_t"] is None:
                st["cross_t"] = t
            if not lane_known and t - st["cross_t"] < LANE_WAIT_S:
                return GamepadState()                                 # a moment standing for the lane to be read
            na = st["notch_ahead"]
            near_edge = na is not None and t - na[1] < 1.0 and na[0] < 0.35
            off_line = abs(y - lane_y)
            if lane_known and not near_edge and (off_line > 0.1 or st["sidestepping"] and off_line > 0.04):
                # onto the box's line by a sidestep on the spot before the edge (the walk's own strafe
                # reached the edge first and put the left legs in the open notch, blue73); with
                # hysteresis, 0.1 in and 0.04 out, and the heading left alone inside 8 degrees: at
                # one threshold the sidestep, the turn and the walk took turns every step (blue75)
                st["sidestepping"] = True
                err = _wrap(0.0 - yaw)
                if abs(err) > math.radians(8.0):
                    return GamepadState(rx=turn_stick(err))
                stage(t, "pit", "sidesteps onto the bridge's line")
                return GamepadState(lx=-math.copysign(0.45, lane_y - y))
            st["sidestepping"] = False
            stage(t, "pit", "over the bridge")
            st["target"] = (px1 + 0.4 - MAP_OFF, lane_y)
            if abs(y - lane_y) > 0.1 and x < px0 - 0.5:
                return plan_to(t, x, y, yaw, px0 - 0.45, lane_y)
            return hold(x, y, yaw, CROSS_STICK, y_goal=lane_y, strafe=x < px0 - 0.4)
        # -- the leap of faith --------------------------------------------------------
        p2_end = L["plat2"][1]
        if st["leap_t"] is None and reached("pit") and st["stage"] != "leap" and (edge is None or edge > 1.2) \
                and x < p2_end + 0.5 and not (st["crossed"] and x >= px1 + 0.4):
            pass                                                      # still on the bridge: the pit stage's crossing goes on
            # (to the same x the crossing above runs to: with 0.5 here there was a 10 cm band past
            # the bridge where neither branch held and the ball stage began, and the leap, which
            # the ball stage outranks, was never run -- the crab walked off the drop, blue31/red15)
        elif not reached("ball") and (x < p2_end + 0.5 or st["leap_t"] is not None and t - st["leap_t"] < LOOK_S + 0.3 + WALK_ON_S):
            if st["leap_t"] is None:
                if st["creep_t"] is None:
                    if edge is not None and edge > LEAP_NEAR_M:
                        stage(t, "leap", "to the edge of the second platform")
                        return hold(x, y, yaw, 0.6, strafe=False)
                    if edge is None and x < p2_end - 0.35:
                        # however far the reckoning says, within reason: after a shove with the box
                        # propped on the claws the leg odometry was 0.6 m ahead, the pit stage ended
                        # with the crab on the box, and the creep began on the reckoning's word (red33,
                        # red34, red35); but 0.35 m short of the reckoned end with nothing seen the
                        # crab stops looking and hops, rather than walk off the end (blue88, red65)
                        stage(t, "leap", "looking for the edge")
                        return hold(x, y, yaw, 0.5, strafe=False)
                    if edge is None:
                        log(t, "no edge in sight near the reckoned end: hops on the reckoning")
                    err = _wrap(0.0 - yaw)
                    if abs(err) > math.radians(HOLD_TURN_DEG):
                        return GamepadState(rx=turn_stick(err))
                    st["creep_t"] = t
                    stage(t, "leap", "nose down over the drop, creeping to the brink")
                # the last stretch nose down (Andrew: "it needs to put its front claws over the edge
                # first"), creeping, the edge read as it comes: level, the sight's lowest ray meets
                # the platform ~0.3 m out and the edge reads 0.3 whatever is nearer, so a hold on
                # that reading never stopped (blue34, blue35, red18) and a mark placed at 0.5-1.3 m
                # and walked to on the odometry missed by the odometry's 0.3-0.5 m (blue37, red21);
                # nose down 10 deg the ray meets it at 0.2 and the edge reads true to the brink
                since_c = t - st["creep_t"]
                d = edge_raw if edge_raw is not None else (edge if edge is not None else p2_end - x)
                if since_c < 0.4:
                    return GamepadState(ry=-0.5)                      # the look settles before it is read
                if st["brink_x"] is None:
                    # the brink is declared from a reading, never from the reckoning's fallback: that
                    # declared it 0.6 m early with nothing in sight (red34, red35)
                    if (edge_raw is None or edge_raw > BRINK_READ_M) and since_c < 8.0:
                        return GamepadState(ry=-0.5, ly=-CREEP)
                    st["brink_x"], st["brink_t"] = x, t               # the reading has met the sight's floor
                    log(t, f"the brink read {BRINK_READ_M - BRINK_READ_OFF_M:.2f} m ahead: {BRINK_ON_S:.1f} s more at the creep"
                        + ("" if edge_raw is not None else " (nothing seen in 8 s: on the reckoning)"))
                if t - st["brink_t"] < BRINK_ON_S:
                    # the last stretch already level: levelling from the nose-down creep standing
                    # at the brink was the transient that put three crabs over the edge before
                    # any dash (blue60, blue61 from 0.07-0.08, red51 from 0.10), where the hop
                    # itself from 0.08-0.14 had come down clean seven times
                    return GamepadState(ry=0.0, ly=-CREEP)
                st["leap_t"] = t
                td = f", truth {p2_end - eyes.truth_pose()[0]:.2f}" if hasattr(eyes, "truth_pose") else ""
                log(t, f"the brink: the edge read {d:.2f} m ahead{td}")  # (the numbers kept out of the readout's stage line)
                stage(t, "leap", "at the brink, claws out over the drop")
            since = t - st["leap_t"]
            # the beat: the crab arrives level, its claw tips, 0.31 m ahead, hanging out over the
            # drop; it stands, and then the dash. Nothing changes in its posture here: a look down
            # standing 0.09 from the edge put red45 over it, and levelling here put three more over
            if since < LOOK_S:
                return GamepadState(ry=0.0)
            if since < LOOK_S + 0.3:
                stage(t, "leap", "the leap of faith")
                press = side.presses["dash"]
                return GamepadState(buttons=press.buttons, hat_x=press.hat_x, hat_y=press.hat_y)
            if since > LOOK_S + 1.8 and getattr(eyes, "pitch", 0.0) < -math.radians(HUNG_DEG):
                # hung on the edge, nose down, the rear feet hooked on it (a hop that carried the base
                # just to the edge, blue40, blue41): the walk on alone did not unhook them in 8 s;
                # a sidestep each half second and the nose up work the rear feet off (16 s, blue50),
                # and after 2.5 s of that a second hop throws the rear end after the front
                if LOOK_S + 4.3 < since < LOOK_S + 4.6:
                    stage(t, "leap", "hung on the edge: hops again")
                    press = side.presses["dash"]
                    return GamepadState(buttons=press.buttons, hat_x=press.hat_x, hat_y=press.hat_y)
                stage(t, "leap", "hung on the edge: works the rear feet off")
                return GamepadState(ly=-0.8, lx=0.5 if int(since * 2) % 2 else -0.5, ry=0.3)
            stage(t, "leap", "down onto its feet")
            return GamepadState(ly=-0.8)                      # walking on scrambles the rear legs off the edge
        # -- the balls --------------------------------------------------------------
        # the race's routine (tools/race/pilots/sweeper.py) on the course's line: pick the nearest
        # remembered ball not yet over, get behind it, shove it over the line, back off, pick again;
        # nothing remembered, turn on the spot and look. The pilot is not told where the balls
        # are (Andrew: "push a ball over the line, rotate around until it sees another ball,
        # target that one and push it")
        goal = L["goal"]
        n_balls = len(L["balls"])                       # how many there are is the task; where, it is not
        def mark_over(e):
            st["over_ids"].add(e["id"])
            st["over_t"][e["id"]] = t
            log(t, f"a ball is over ({len(st['over_ids'])} of {n_balls})")
        for e in mem:
            if e["x"] > goal + 0.1 and t - e["t"] < 0.5 and e["id"] not in st["over_ids"]:
                mark_over(e)
        pushing = st["push_since"] is not None
        if tgt is not None and pushing and x > goal + 0.4 and tgt["id"] not in st["over_ids"]:
            # following its ball the crab crosses after it: well past the line, the ball it was
            # shoving is over (seen or not; under the nose it is often a sliver)
            mark_over(tgt)
        for e in eyes.balls:
            # ...but a ball counted over and SEEN short of the line since is not: the shove lost it
            # sideways and the crab walked on over the line alone, or two memories of one ball were
            # both counted ("4 of 3"). Counted over and never looked at again, two and three balls
            # in a take were left short while the crab went on to the finish (blue223, blue224,
            # red221, at the 0.65 shove)
            if e["id"] in st["over_ids"] and e["x"] < goal - 0.05 and t - e["t"] < 0.3 and e["t"] - st["over_t"].get(e["id"], t) > 1.0:
                st["over_ids"].discard(e["id"])
                st["final_look"] = False
                log(t, f"a ball counted over is short of the line at x {e['x']:.2f} y {e['y']:+.2f}: back to it")
        if tgt is not None and tgt["id"] in st["over_ids"] or (pushing and t - st["push_since"] > 30.0):
            if pushing and tgt is not None and tgt["id"] not in st["over_ids"]:
                log(t, "thirty seconds of shoving and not over: picks again")
            st["tgt"], st["push_since"], st["plan_t"], st["ball_back"] = None, None, -1e9, t
            tgt = ball = None
        if len(st["over_ids"]) < n_balls or not st["final_look"]:
            if st["ball_back"] is not None and t - st["ball_back"] < BALL_BACK_S:
                stage(t, "ball", "backs off the ball")
                return GamepadState(ly=0.6)
            if tgt is None:
                # pick: the nearest remembered ball, seen twice, not over
                cands = [e for e in mem if e["n"] >= BALL_SEEN_N and e["x"] < goal]
                if cands:
                    tgt = min(cands, key=lambda e: math.hypot(e["x"] - x, e["y"] - y))
                    st["tgt"], st["scan"] = tgt["id"], None
                    ball = (tgt["x"], tgt["y"], tgt["t"])
                    log(t, f"picks the ball at x {tgt['x']:.2f} y {tgt['y']:+.2f}")
                else:
                    # the search: a turn on the spot, a whole turn, looking (the race's SCAN)
                    if st["scan"] is None:
                        st["scan"] = [yaw, 0.0]
                        log(t, "no ball in mind: turns on the spot to look" if len(st["over_ids"]) < n_balls else
                            "every ball counted over: one turn on the spot to look for any left")
                    st["scan"][1] += abs(_wrap(yaw - st["scan"][0]))
                    st["scan"][0] = yaw
                    if st["scan"][1] < BALL_SCAN_TURN:
                        stage(t, "ball", "turns to look for a ball")
                        return GamepadState(rx=-BALL_SCAN_STICK)
                    if len(st["over_ids"]) >= n_balls:
                        # the last look: a whole turn and no ball short of the line, on to the finish
                        st["final_look"], st["scan"] = True, None
                        log(t, "no ball left short of the line")
                        return GamepadState()
                    # a whole turn and nothing: back toward the middle of the floor, and look again
                    gx = L["plat2"][1] + 1.0
                    if math.hypot(x - gx, y) > 0.3:
                        stage(t, "ball", "nothing in sight: back to the middle to look again")
                        return plan_to(t, x, y, yaw, gx, 0.0)
                    st["scan"] = None
                    return GamepadState()
            bx, by = ball[0], ball[1]
            if os.environ.get("COURSE_TRACE") and int(t * 2) != int((t - 0.005) * 2):
                tx, ty, _ = eyes.truth_pose() if hasattr(eyes, "truth_pose") else (x, y, yaw)
                print(f"\n[{label}] {t:6.2f}s  ball: reckoned x {x:.2f} y {y:+.2f} (truth {tx:.2f} {ty:+.2f}), "
                      f"target x {bx:.2f} y {by:+.2f} seen {t - ball[2]:.1f} s ago", flush=True)
            if not pushing:
                # behind it: the standing point BEHIND_M short of it along the lane, facing the line.
                # From beside or past it the planner takes the crab round it (the map has it)
                sx = bx - BEHIND_M
                # with hysteresis: at one threshold the round and the line-up took turns every frame as
                # the ball's place jittered under the nose (blue103 L, red104 L)
                lining = st.get("ball_lining") == st["tgt"]
                if (x > bx - (0.10 if lining else 0.25)) or abs(y - by) > (0.7 if lining else 0.5):
                    st["ball_lining"] = None
                    stage(t, "ball", "round to the point behind the ball")
                    return plan_to(t, x, y, yaw, sx, by, stick=BALL_WALK)
                if x < sx - 0.1:
                    stage(t, "ball", "to the point behind the ball")
                    st["target"] = (sx - MAP_OFF, by)
                    return hold(x, y, yaw, BALL_WALK, y_goal=by)
                st["ball_lining"] = st["tgt"]
                err = _wrap(0.0 - yaw)
                if abs(err) > math.radians(12.0):
                    stage(t, "ball", "lines up on the ball")
                    return GamepadState(rx=turn_stick(err))
                bearing = _wrap(math.atan2(by - y, bx - x) - yaw)
                if abs(bearing) > math.radians(10.0):
                    stage(t, "ball", "lines up on the ball")
                    return GamepadState(lx=-math.copysign(0.45, bearing))     # the ball to the left: strafe left
                st["push_since"] = t
                pushing = True
            # the shove: the lateral stick keeps the ball under the nose
            if t - ball[2] < 0.3:
                bearing = _wrap(math.atan2(by - y, bx - x) - yaw)
                if abs(bearing) > math.radians(ESCAPE_DEG) and math.hypot(bx - x, by - y) < ESCAPE_M:
                    log(t, f"the ball got away ({math.degrees(bearing):+.0f} deg): round behind it again")
                    st["push_since"], st["plan_t"] = None, -1e9
                    return GamepadState()
            else:
                bearing = 0.0
            under_nose = tgt is not None and tgt["r"] < 0.6 and abs(tgt["b"]) < math.radians(25.0)
            lost_s = 5.0 if under_nose else 1.5          # last seen under the nose: pushed blind, give it the shove's length
            if t - st["push_since"] > lost_s and t - ball[2] > lost_s:
                log(t, "the ball is out of sight: picks again")
                st["tgt"], st["push_since"], st["plan_t"] = None, None, -1e9
                return GamepadState()
            stage(t, "ball", "the shove: across the finish line")
            st["target"] = (goal - MAP_OFF, by)
            lx = max(-0.6, min(0.6, -2.0 * bearing))
            return GamepadState(ly=-PUSH_BALL, lx=lx)
        # -- the finish -------------------------------------------------------------
        if x < L["finish"] + track.DANCE_PAST_M:
            stage(t, "finish", "on past the line")
            st["target"] = (L["finish"] + track.DANCE_PAST_M - MAP_OFF, 0.0)
            return hold(x, y, yaw, WALK)
        if st["arrived"] is None:
            st["arrived"] = t
            log(t, "past the finish")
        return None

    def pilot(t):
        frame = _pilot(t)
        eyes.update(t, frame)
        st["frame"] = frame
        if not side.truth and t - st["truth_log"] >= TRUTH_LOG_S:
            st["truth_log"] = t
            x, y, yaw = eyes.pose()
            tx, ty, tyaw = eyes.truth_pose()
            print(f"\n[{label}] {t:6.1f}s  pose x {x:+.2f} y {y:+.2f} yaw {math.degrees(yaw):+.0f}  truth x {tx:+.2f} y {ty:+.2f} "
                  f"yaw {math.degrees(tyaw):+.0f}  (off {tx - x:+.2f}, {ty - y:+.2f} m, {math.degrees(_wrap(tyaw - yaw)):+.1f} deg); "
                  f"pitch {math.degrees(eyes.pitch):+.0f}; fixes {eyes.fixes}")
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
            if st["cue_t"] is None:
                st["cue_t"] = t0
            if t - st["cue_t"] < PAST_LINE_S and walking():
                st["face"] = (piece, t)                       # the cue came at the line; the dance is 1.8 m on (Andrew:
                return act(t)                                 # "continue another meter or two and then celebrate"; red4
                                                              # turned and danced at the gate's post) -- by TIME, not by
                                                              # the reckoning, which the ball's shove leaves 2-3 m behind
                                                              # (blue79, red60 walked on to the end wall and never danced)
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
            return None
        if not walking():
            return None
        if referee.state != "run":
            return None
        return act(t)

    pilot.busy = lambda t: st["face"] is not None or performer.busy(t)
    pilot.state = st
    pilot.senses = getattr(eyes, "sight", None)
    pilot.mapper = getattr(eyes, "mapper", None)
    pilot.eyes = eyes
    return pilot
