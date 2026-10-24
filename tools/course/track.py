"""The obstacle course: one lane, five obstacles, one crab at a time.

Andrew (2026-10-22): "a series of obstacles like a dog course that the crabs race
on ... we have each crab race it separately and I will edit the video and play
them back side by side ... a stair case, a ramp, a forest of pillars they have to
navigate, a heavy ball they have to push and the crabs need to do a dance when
they arrive at the end", with the depth camera in the picture.

The lane runs along +x from the start mark, a kerb down each side the crab can
localise against (`tools/race/room.py`'s, straight), and the obstacles in order:

* **the hill** -- a staircase up to a platform and a ramp down it, then a ramp up
  to a second platform and a staircase down. Measured 2026-10-22 on jumper.posture's
  locomotion at full stick (`$S/probe_step.py`, native backend): a single riser of
  2 or 3 cm is walked over, 4 cm and up stops the crab at the face, body raised or
  not; a flight of 3 cm risers on 30 cm treads jams it on the first step at the
  standing height and climbs with the body raised (R3 + stick up, 0.153 m) once
  in three on the course (truth1, sense1 climbed; sense2, sense3, truth2 jammed on
  the second step); 2.5 cm risers on 40 cm treads climb standing 4 of 4 (raised,
  1 of 2) and 2 cm on 35 or 40 cm 6 of 6; going down,
  flights of 4 cm (four) and 6 cm (three) are walked, 8 cm hangs the crab on the
  first edge; ramps of 8-25 degrees are climbed and 15-20 descended at walking
  pace (0.7 m/s on the flat, ~0.5 on a slope); walked off a platform's edge the
  crab goes over. So the first flight is 2 cm risers on 40 cm treads, climbed
  standing (2.5 cm jammed one course take in three, red2), the last 4 cm risers
  down on 30 cm, the ramps 12 degrees, and nothing here has an edge that is not
  a step or a ramp.
* **the forest** -- pillars 0.15 m square and 0.45 m tall, staggered so no straight
  line runs through, with gaps a crab fits (its stance is ~0.45 m across). The
  ones on the camera's side are glass, as the kerb there is: solid, they hid the
  crab for most of the forest (sense4); the sensor sees glass as it sees stone.
* **the platforms** -- a ramp up to a raised platform, a notch between it and a
  second platform, and off the far end of the second a drop to the ground
  (Andrew: "a ramp going up and then two raised platforms with a trench between
  them. Then instead of a ramp down we have a 'leap of faith' where the crab has
  to jump off the second box and not end up on its back"). A trench cut into
  the floor was tried and the world's grey ground showed through it; here the
  notch has its own floor, 7 cm below the platforms' tops, and the one box
  standing on the first platform is shoved into it, no guides, and fills it
  flush: the crab walks over the box to the second platform. The box tips
  about the notch's near edge and drops once its rear end is over it, and its
  rear comes down about 6 cm past the edge whatever the notch's length (pusher
  probes), so the notch is longer than the box (shorter, the box catches the
  far wall before its rear clears and sits proud, truth17), and the box's
  bottom-rear corner is cut away so that its top closes the slot its bottom
  leaves (REBATE), its mass spread evenly (REAR_SHARE: at the rear it swung
  under the claws); the notch and the box are 10 cm, the deepest the shell
  still pushes a box into, and the box half the lane's width, so the crossing
  reads where it lies (the pilot's bridge_centre). At the far
  end of the second platform the crab hops off with the dash: walking off, the
  front legs drop and the body goes over at 10 cm and above; the dash from
  0.2 m before the edge carries the base past it, the rear legs come down on
  the edge, and walking on the crab scrambles down onto its feet -- two times
  in three at 15 cm, which is the drop; at 20 it goes over or hangs.
* **the ball** -- heavier than the race's (`tools/race/room.py`, 0.08 kg): half
  the crab's own mass, 24 cm across, to be shoved across the finish line itself.
* **the finish** -- a gate of two posts and a banner; the ball goes over its line
  and the crab after it, on for another stretch, then turns to the camera and
  dances its colour's victory piece (`tools/battle/moves.py`).

Numbers are environment variables (`COURSE_*`) so a take can be framed without
editing this file, as `tools/race/room.py` does.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import os
from typing import Any

from arena import RED, SHELL_BLUE, SHELL_RED, _pin, _recoloured

LANE_W = float(os.environ.get("COURSE_LANE_W", "2.2"))        # m across (y)
START_X = 0.0                                                 # the crab's mark; it faces +x
KERB_H, KERB_T = 0.35, 0.05                                   # the race's kerb: taller than a crab, never read as a thing
GLASS = 0.22                                                  # the camera-side kerb's alpha (the camera is on -y)

# -- the hill: stairs up, ramp down; ramp up, stairs down ------------------------------
STAIR_H = float(os.environ.get("COURSE_STAIR_H", "0.02"))     # the first flight's riser, m (up)
STAIR_N = int(os.environ.get("COURSE_STAIR_N", "5"))
STAIR_T = float(os.environ.get("COURSE_STAIR_T", "0.40"))     # its tread, m
DOWN_H = float(os.environ.get("COURSE_DOWN_H", "0.04"))       # the last flight's riser, m (down)
DOWN_N = int(os.environ.get("COURSE_DOWN_N", "3"))
DOWN_T = float(os.environ.get("COURSE_DOWN_T", "0.30"))       # its tread, m
RAMP_DEG = float(os.environ.get("COURSE_RAMP_DEG", "12"))
PLATFORM_L = 1.0                                              # m along each platform's top
HILL_X = float(os.environ.get("COURSE_HILL_X", "1.6"))        # where the first flight starts
HILL_GAP = 1.2                                                # flat between the two hills
# -- the forest -------------------------------------------------------------------
PILLAR_HALF = (0.075, 0.075, 0.225)
PILLAR_GAP = 1.0                                              # m after the hill
PILLAR_ROWS = ((-0.6, 0.55), (0.0,), (-0.5, 0.65), (0.1,), (-0.6, 0.55), (0.0,))   # y per row: a slalom. No pillar
                                                              # within 0.3 m of a kerb, where the pilot's wall fix reads kerb
                                                              # points, and no gap between faces under 0.85 m: three to a row
                                                              # 0.7 apart left the planner one cell of corridor at its 0.25 m
                                                              # clearance and blue's body clipped a corner for 6 s (blue1)
PILLAR_PITCH = 0.75                                           # m between rows
# -- the platforms, the notch and the box ------------------------------------------
PLAT_GAP = 1.4                                                # m after the forest to the ramp's foot
PLAT_H = float(os.environ.get("COURSE_PLAT_H", "0.12"))       # the platforms' height: the leap of faith is this drop. The
                                                              # dash hop pressed LEAP_BACK_M before the edge lands the base
                                                              # past the edge with the rear legs on it, and walking on the
                                                              # crab scrambles down onto its feet: at 12 cm 8 times in 8
                                                              # (pressed 20 and 25 cm before), at 15 cm 3 in 4 at 20 and 1
                                                              # in 4 at 25, at 20 cm 0 in 3; pressed 30 cm before, the hop
                                                              # lands short at any height (`$S/probe_step.py platform`)
PLAT_RAMP_DEG = 12.0                                          # the ramp up (the hill's ramps are 12 too)
PLAT1_LEN = 2.2                                               # the first platform: the box stands on it BLOCK_RUN before the notch
                                                              # (2.2, was 1.9: the point behind the box is 0.5 m behind its
                                                              # face, and with the box 0.6 m before the notch the crab lined
                                                              # up there with its rear legs on the ramp's crest, paused to
                                                              # square itself on the slope and slid back, blue71)
PLAT2_LEN = 1.0                                               # the second platform, to the drop
BLOCK = (0.25, LANE_W / 4, 0.05)                              # the box's half sizes: 0.5 m along, half the lane's width, 10 cm
                                                              # tall. It was the lane's width, so that the kerbs kept it square
                                                              # (Andrew: "not interesting ... make it half the width"):
                                                              # pushed from behind a box turns away from the side it is pushed
                                                              # on, and the crab pushes more with one claw than the other --
                                                              # a 0.8 m box walked off to -y, 0.3-0.7 m and up to 14 deg over
                                                              # the 1 m shove, in nine takes of twelve (red11, red12, blue28,
                                                              # red14, blue30, red15, blue32, red16, red17), and the crab then
                                                              # crossed into the open notch beside it; squaring it by strafing
                                                              # dragged it sideways instead (red16, red17). Between the kerbs
                                                              # it can yaw 6 deg and shift 2.5 cm, no more; the pusher probe
                                                              # lands the cut box flat to 6 deg. The notch is 10 cm deep --
                                                              # the deepest notch the shell still pushes a box into (12 cm
                                                              # tipped was a ramp the crab climbed, truth8; 16 cm tipped out
                                                              # of the shell's reach, truth14); Andrew: "the trench needs to
                                                              # be deeper" (it was 7)
REBATE = (0.07, 0.01)                                         # the box's bottom-rear corner is cut away square: this far in from
                                                              # the rear and this far up -- under the claws' reach (a tip is 1.2 cm
                                                              # round): 3 cm up, one claw went into the cut and one pushed the
                                                              # face above it, and the box walked off sideways, 0.58 m and 30
                                                              # degrees by the edge (red11); no pusher probe yaws a box pushed
                                                              # within 6 cm of its centre line. Its top stays 0.50 long and its bottom
                                                              # is 0.43, so when the box lies against the notch's far wall its
                                                              # top's rear is 2 cm from the near edge, a gap under a toe's
                                                              # width, where a plain box left 6 cm the feet caught in (truth20,
                                                              # truth23). A square cut, not a 45 deg bevel: the bevel was a
                                                              # ramp for the crab's claws, which push the box's rear low down,
                                                              # and twice they went under it and lifted the box's rear as it
                                                              # tipped, the crab ending nose-down in the notch under a propped
                                                              # box (blue25, red10); the step leaves them a vertical face. And
                                                              # no plate: a slanted slab standing proud of a 7 cm box did the
                                                              # same job and "looks ugly" (Andrew)
REAR_SHARE = 0.20                                             # the share of the box's mass in its rear 7 cm: uniform, near enough.
                                                              # With 0.80 there (the box tipping only at the edge, a trapdoor)
                                                              # the box rode on its two loaded rear corners and swung under each
                                                              # claw's push, walking off to the left 0.4-0.6 m over the 1 m
                                                              # shove three times out of four (red11, blue28, red12); a pusher
                                                              # probe offset 10 cm drifts it twice as far as a uniform box
BLOCK_MU = 0.6                                                # the box's friction on the deck. 0.3 slid under the shove and under
                                                              # the claws' sideways drag alike (the box walked 0.14-0.55 m aside
                                                              # over the 1 m shove, blue52-red49); doubled, the push the crab
                                                              # has to spare still moves it, the drag moves it half as far,
                                                              # and the pusher probe lands it flat to 6 deg of yaw (probe_push8)
BLOCK_MASS = float(os.environ.get("COURSE_BLOCK_KG", "0.25"))  # light: a 0.6 kg box tipping about the edge swung its rear
                                                              # end up under the crab's nose and put the crab over (truth11)
BLOCK_Y = (0.0,)                                              # one box, on the lane's centre line
BLOCK_RUN = 0.6                                               # ...this far before the notch's near edge: the shove's length.
                                                              # It was 1.0; the box walks aside under the shove (the hold's
                                                              # strafe drags it) by so much a metre, and over 0.6 m there is
                                                              # less of it, with a push still to be seen
PIT_LEN = float(os.environ.get("COURSE_PIT_LEN", "0.52"))     # the notch along the lane: the box's top (0.50) plus the 2 cm gap
                                                              # it leaves lying against the far wall -- where the crab's feet
                                                              # push it once it has dropped (a 0.58 notch, the length the
                                                              # pusher probes wanted, left an 8 cm slot at the near edge
                                                              # that way, blue22). Probed against the far wall: 0.52 flat at
                                                              # every yaw to 6 deg with a 6 or 7 cm cut, 0.50 not flat
PIT_DEPTH = 2 * BLOCK[2]                                      # as deep as the box is tall: shoved in, it lies flush
LEAP_BACK_M = 0.22                                            # the hop is pressed this far before the second platform's end:
                                                              # from 5-15 cm it fails (12 of 12 went over: the front legs
                                                              # over the void at take-off), from 20-25 the base lands past
                                                              # the edge, from 30 it lands short (see PLAT_H)
# -- the ball -----------------------------------------------------------------------
BALL_R = 0.12
BALL_MASS = float(os.environ.get("COURSE_BALL_KG", "1.2"))    # the crab is 2.54 kg
BALL_GAP = 1.5                                                # m after the pit's far edge, the ball on the centreline
BALL_RUN = 1.8                                                # the finish line this far past the balls
BALL_ROLL_MU = 0.008                                          # rolling friction (condim 6, or MuJoCo ignores it): with none
                                                              # the ball knocked at the walk's 0.6 m/s rolled 18 m, and over the
                                                              # line on its own 0.8 s into the shove (blue101, blue102); with
                                                              # this it rolls 0.66 m and stops (0.004: 2.2 m; probe_roll.py),
                                                              # so the crab has to keep pushing it to the line
BALL_Y = (0.0, 0.55, -0.55)                                   # three balls abreast on the ball line: the middle, one to the
                                                              # crab's left (+y) and one to its right (Andrew, 2026-10-23:
                                                              # "the robots will need to push all three across the finish
                                                              # line before they can complete the course"); the first is
                                                              # the layout's "ball" for the pilots' landmarks
# -- the finish ---------------------------------------------------------------------
GATE_H = 0.9
FINISH_RUN = 3.0                                              # lane beyond the finish line, where the dance is
DANCE_PAST_M = 1.8                                            # the dance this far past the line (Andrew: "it should
                                                              # continue another meter or two and then celebrate")

COLOURS = {"red": SHELL_RED, "blue": SHELL_BLUE}
FLOOR = (0.90, 0.90, 0.88)
LANE = {"red": (0.95, 0.42, 0.36), "blue": (0.40, 0.56, 0.95)}
LINE = (1.0, 1.0, 1.0)
KERB = (0.96, 0.96, 0.94, 1.0)
STONE = (0.62, 0.56, 0.48, 1.0)
WOOD = (0.70, 0.52, 0.30, 1.0)
PILLAR = (0.30, 0.32, 0.36, 1.0)
BALL = (0.12, 0.12, 0.12, 1.0)
DECK = (0.78, 0.76, 0.72, 1.0)
PIT = (0.62, 0.58, 0.54, 1.0)                                 # the trench's floor: a shade below the deck, not a black hole
                                                              # (Andrew: "I do not want a dark black pit")
BLOCK_C = ((0.95, 0.80, 0.20, 1.0),)
SET_FLOOR = (0.20, 0.20, 0.22, 1.0)
WALL = (0.80, 0.78, 0.74, 1.0)
POST = (0.95, 0.95, 0.95, 1.0)
SUN_DIR = (0.12, 0.08, -1.0)
WALL_H, WALL_OUT = 1.2, 1.0


def hill() -> dict:
    """Where everything on the hill is, in x: the two flights, the two ramps and
    the two platforms, each as (x0, x1); and the platforms' height."""
    top1, top2 = STAIR_H * STAIR_N, DOWN_H * DOWN_N
    x = HILL_X
    flight1 = (x, x + STAIR_T * STAIR_N); x = flight1[1]
    plat1 = (x, x + PLATFORM_L); x = plat1[1]
    ramp1 = (x, x + top1 / math.tan(math.radians(RAMP_DEG))); x = ramp1[1]
    x += HILL_GAP
    ramp2 = (x, x + top2 / math.tan(math.radians(RAMP_DEG))); x = ramp2[1]
    plat2 = (x, x + PLATFORM_L); x = plat2[1]
    flight2 = (x, x + DOWN_T * DOWN_N); x = flight2[1]
    return {"flight1": flight1, "plat1": plat1, "ramp1": ramp1, "ramp2": ramp2, "plat2": plat2, "flight2": flight2,
            "top1": top1, "top2": top2, "end": x}


def layout() -> dict:
    """Every landmark's x along the lane: the course the pilot is told."""
    h = hill()
    pillars = []
    x = h["end"] + PILLAR_GAP
    for row in PILLAR_ROWS:
        pillars += [(x, y) for y in row]
        x += PILLAR_PITCH
    forest = (h["end"] + PILLAR_GAP - 0.3, x - PILLAR_PITCH + 0.3)
    ramp_x0 = forest[1] + PLAT_GAP
    ramp = (ramp_x0, ramp_x0 + PLAT_H / math.tan(math.radians(PLAT_RAMP_DEG)))
    plat1 = (ramp[1], ramp[1] + PLAT1_LEN)
    pit = (plat1[1], plat1[1] + PIT_LEN)
    plat2 = (pit[1], pit[1] + PLAT2_LEN)
    blocks_x = pit[0] - BLOCK_RUN - BLOCK[0]
    ball_x = plat2[1] + BALL_GAP
    finish_x = ball_x + BALL_RUN
    return {"hill": h, "pillars": pillars, "forest": forest, "ramp": ramp, "plat1": plat1, "pit": pit, "plat2": plat2,
            "blocks": [(blocks_x, y) for y in BLOCK_Y], "ball": (ball_x, BALL_Y[0]), "balls": [(ball_x, y) for y in BALL_Y],
            "goal": finish_x, "finish": finish_x,
            "end": finish_x + FINISH_RUN}


LANE_LEN = layout()["end"] + 0.8                              # m along (x): the lane runs 0.5 m behind the mark to here


def _floor_texture(spec: Any, crab: str) -> None:
    """The lane painted onto a 1 mm sheet: the crab's colour, a white start and
    finish line, the goal line, a metre tick down each side."""
    import mujoco
    import numpy as np

    L = layout()
    n = 2048
    ny = int(n * LANE_W / LANE_LEN)
    xs = (np.arange(n) + 0.5) / n * LANE_LEN - 0.5
    ys = (np.arange(ny) + 0.5) / ny * LANE_W - LANE_W / 2
    X, Y = np.meshgrid(xs, ys)
    img = np.zeros((ny, n, 3))
    img[:] = 0.35 * np.array(FLOOR) + 0.65 * np.array(LANE[crab])
    rng = np.random.default_rng(7)
    img *= (1.0 + 0.03 * rng.standard_normal((ny, n, 1)))
    for x0 in (START_X, L["finish"]):
        img[np.abs(X - x0) < 0.03] = LINE
    # the start and finish lines chequered, the goal line plain
    for x0 in (START_X, L["finish"]):
        sq = ((np.floor(Y / 0.11) + np.floor((X - x0 + 0.06) / 0.06)) % 2 == 0) & (np.abs(X - x0) < 0.06)
        img[sq] = (0.1, 0.1, 0.1)
    img[(np.abs(X - L["ball"][0]) < 0.004) | (np.abs(Y) < 0.004) & (np.abs(X - L["ball"][0]) < 0.25)] = LINE
    for bx, by in L["blocks"]:
        rim = (np.abs(X - bx) < BLOCK[0] + 0.03) & (np.abs(Y - by) < BLOCK[1] + 0.03) & \
              ~((np.abs(X - bx) < BLOCK[0] + 0.01) & (np.abs(Y - by) < BLOCK[1] + 0.01))
        img[rim] = LINE
    # the raised section: the painted lane stops at the ramp's foot and starts again past the drop
    img[(X > L["ramp"][0]) & (X < L["plat2"][1])] = STONE[:3]
    img[np.abs(X - L["plat2"][1] - 0.02) < 0.02] = LINE
    for bx, by in L["balls"]:
        ring = np.abs(np.hypot(X - bx, Y - by) - 0.18) < 0.012
        img[ring] = LINE
    for x0 in range(1, int(LANE_LEN)):
        tick = (np.abs(X - x0) < 0.012) & (np.abs(Y) > LANE_W / 2 - 0.12)
        img[tick] = LINE
    edge = np.minimum(LANE_W / 2 - np.abs(Y), np.minimum(X + 0.5, LANE_LEN - 0.5 - X))
    img[edge < 0.03] = LINE
    tex = spec.add_texture(name="lane", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=ny, nchannel=3)
    tex.data = (np.clip(img, 0.0, 1.0)[::-1] * 255).astype(np.uint8).tobytes()
    mat = spec.add_material(name="lane", roughness=0.9, metallic=0.0, specular=0.1)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "lane"
    spec.worldbody.add_geom(name="lane", type=mujoco.mjtGeom.mjGEOM_BOX, size=[LANE_LEN / 2, LANE_W / 2, 0.0005],
                            pos=[LANE_LEN / 2 - 0.5, 0.0, 0.0005], material="lane", contype=0, conaffinity=0, group=2, mass=0.0)


def _box(spec, name, size, pos, rgba, quat=None, collide=True):
    """A static box."""
    import mujoco
    kw = {} if collide else {"contype": 0, "conaffinity": 0}
    if quat is not None:
        kw["quat"] = list(quat)
    pos = [pos[0], pos[1], pos[2]]
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=pos, rgba=list(rgba),
                            group=2, mass=0.0, **kw)


def _ramp(spec, name, x0, x1, top, rising: bool):
    """A slab from ground level at one end to `top` at the other: rising with +x or
    falling. Thin, and sunk so its upper face meets the ground and the platform."""
    t = 0.01
    L = math.hypot(x1 - x0, top)
    th = math.atan2(top, x1 - x0)
    s = -th if rising else th
    q = [math.cos(s / 2), 0.0, math.sin(s / 2), 0.0]
    _box(spec, name, (L / 2, LANE_W / 2, t), ((x0 + x1) / 2, 0.0, top / 2 - t / math.cos(th)), WOOD, quat=q)


def _hill(spec) -> None:
    h = hill()
    x0 = h["flight1"][0]
    for i in range(STAIR_N):
        z = STAIR_H * (i + 1)
        _box(spec, f"flight1_{i}", (STAIR_T / 2, LANE_W / 2, z / 2), (x0 + STAIR_T * (i + 0.5), 0.0, z / 2), STONE)
    x0 = h["flight2"][0]
    for i in range(DOWN_N):
        z = DOWN_H * (DOWN_N - i)
        _box(spec, f"flight2_{i}", (DOWN_T / 2, LANE_W / 2, z / 2), (x0 + DOWN_T * (i + 0.5), 0.0, z / 2), STONE)
    for k, top in (("plat1", h["top1"]), ("plat2", h["top2"])):
        x0, x1 = h[k]
        _box(spec, k, ((x1 - x0) / 2, LANE_W / 2, top / 2), ((x0 + x1) / 2, 0.0, top / 2), STONE)
    _ramp(spec, "ramp1", *h["ramp1"], h["top1"], rising=False)
    _ramp(spec, "ramp2", *h["ramp2"], h["top2"], rising=True)


def _forest(spec, L) -> None:
    for i, (x, y) in enumerate(L["pillars"]):
        rgba = list(PILLAR[:3]) + [GLASS + 0.15 if y < -0.3 else 1.0]
        _box(spec, f"pillar{i}", PILLAR_HALF, (x, y, PILLAR_HALF[2]), rgba)


def _finish(spec, L, crab) -> None:
    import mujoco
    fx = L["finish"]
    for i, sy in enumerate((-1.0, 1.0)):
        rgba = list(POST[:3]) + [GLASS + 0.2 if sy < 0 else 1.0]
        spec.worldbody.add_geom(name=f"gate{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, GATE_H / 2, 0.0],
                                pos=[fx, sy * (LANE_W / 2 + KERB_T + 0.08), GATE_H / 2], rgba=rgba, group=2, mass=0.0)
    _box(spec, "gate_banner", (0.012, LANE_W / 2 + KERB_T + 0.11, 0.10), (fx, 0.0, GATE_H - 0.10), COLOURS[crab], collide=False)
    _box(spec, "gate_stripe", (0.013, LANE_W / 2 + KERB_T + 0.11, 0.018), (fx, 0.0, GATE_H - 0.10), (1, 1, 1, 1), collide=False)


def _set(spec, L, crab) -> None:
    """Round the lane, drawn only: a dark floor, a back wall (+y) with a banner, the
    far end wall, posts with pennants at the lane's ends."""
    import mujoco
    off = {"collide": False}
    ox0, ox1 = -0.5 - WALL_OUT, LANE_LEN - 0.5 + WALL_OUT
    oy = LANE_W / 2 + KERB_T + WALL_OUT
    _box(spec, "set_floor", ((ox1 - ox0) / 2 + 3.0, oy + 3.0, 0.0004), ((ox0 + ox1) / 2, 0.0, 0.0004), SET_FLOOR, **off)
    t = 0.05
    _box(spec, "wall_back", ((ox1 - ox0) / 2 + t, t, WALL_H / 2), ((ox0 + ox1) / 2, oy + t, WALL_H / 2), WALL, **off)
    _box(spec, "wall_end", (t, oy, WALL_H * 0.35), (ox1 + t, 0.0, WALL_H * 0.35), WALL, **off)
    _box(spec, "wall_start", (t, oy, WALL_H * 0.35), (ox0 - t, 0.0, WALL_H * 0.35), WALL, **off)
    c = COLOURS[crab]
    _box(spec, "banner", ((ox1 - ox0) / 2 * 0.9, 0.012, 0.16), ((ox0 + ox1) / 2, oy + t - 0.06, 0.62), c, **off)
    _box(spec, "banner_stripe", ((ox1 - ox0) / 2 * 0.9, 0.013, 0.022), ((ox0 + ox1) / 2, oy + t - 0.06, 0.62), (1, 1, 1, 1), **off)
    _box(spec, "wall_back_band", ((ox1 - ox0) / 2, 0.012, 0.08), ((ox0 + ox1) / 2, oy + t - 0.06, 0.08), SET_FLOOR, **off)
    for i, (px, py) in enumerate(((ox0, -oy), (ox1, -oy), (ox1, oy), (ox0, oy))):
        spec.worldbody.add_geom(name=f"post{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.45, 0.0],
                                pos=[px, py, 0.45], rgba=list(POST), contype=0, conaffinity=0, group=2, mass=0.0)


def _platforms(spec, L) -> None:
    """The ramp, the two platforms and the notch's floor between them."""
    _ramp(spec, "plat_ramp", *L["ramp"], PLAT_H, rising=True)
    for name in ("plat1", "plat2"):
        x0, x1 = L[name]
        _box(spec, f"raised_{name}", ((x1 - x0) / 2, LANE_W / 2, PLAT_H / 2), ((x0 + x1) / 2, 0.0, PLAT_H / 2), STONE)
    px0, px1 = L["pit"]
    floor_h = PLAT_H - PIT_DEPTH
    _box(spec, "notch_floor", ((px1 - px0) / 2, LANE_W / 2, floor_h / 2), ((px0 + px1) / 2, 0.0, floor_h / 2), PIT)


def _block_spec(i: int) -> Any:
    import mujoco

    from tasks.jumper.five_foot import objects as O

    spec = O._prop_spec()
    body = O._free_body(spec, f"block{i}")
    L, W, H = BLOCK
    ch, sh = REBATE
    fr = {"friction": [BLOCK_MU, 0.005, 0.0001], "priority": 1}   # its own friction against the deck's (MuJoCo takes the higher
                                                                  # of a pair unless one has priority): it slides under the shove
    # the box in three pieces: the bottom part, short of the rear by the rebate; the top part's
    # front, above it; and the top part's rear, the overhang, which carries most of the mass
    O._prop_geom(body, name=f"block{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L - ch / 2, W, sh / 2],
                 pos=[ch / 2, 0.0, -H + sh / 2], rgba=list(BLOCK_C[i]), mass=BLOCK_MASS * (1 - REAR_SHARE) * 0.4, **fr)
    O._prop_geom(body, name=f"block{i}_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L - ch / 2, W, (2 * H - sh) / 2],
                 pos=[ch / 2, 0.0, sh / 2], rgba=list(BLOCK_C[i]), mass=BLOCK_MASS * (1 - REAR_SHARE) * 0.6, **fr)
    O._prop_geom(body, name=f"block{i}_rear", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ch / 2, W, (2 * H - sh) / 2],
                 pos=[-L + ch / 2, 0.0, sh / 2], rgba=list(BLOCK_C[i]), mass=BLOCK_MASS * REAR_SHARE, **fr)
    return spec


def decorate(spec: Any, crab: str = "red") -> None:
    """The scene's last edit: the deck and the pit, the lane, its kerbs, the obstacles and the set."""
    L = layout()
    _platforms(spec, L)
    _floor_texture(spec, crab)
    hz = KERB_H / 2
    for i, sy in enumerate((-1.0, 1.0)):
        glass = sy < 0
        rgba = list(KERB[:3]) + [GLASS if glass else 1.0]
        _box(spec, f"kerb{i}", (LANE_LEN / 2 + KERB_T, KERB_T / 2, hz), (LANE_LEN / 2 - 0.5, sy * (LANE_W / 2 + KERB_T / 2), hz), rgba)
    for i, x in enumerate((-0.5, LANE_LEN - 0.5)):
        _box(spec, f"kerb_end{i}", (KERB_T / 2, LANE_W / 2 + KERB_T, hz), (x + (-KERB_T / 2 if i == 0 else KERB_T / 2), 0.0, hz), KERB)
    _hill(spec)
    _forest(spec, L)
    _finish(spec, L, crab)
    _set(spec, L, crab)
    for light in spec.lights:
        if light.castshadow:
            light.dir = list(SUN_DIR)


def _ball_spec(name: str = "ball") -> Any:
    import mujoco

    from tasks.jumper.five_foot import objects as O

    spec = O._prop_spec()
    body = O._free_body(spec, name)
    O._prop_geom(body, name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0.0, 0.0], pos=[0.0, 0.0, 0.0],
                 rgba=list(BALL), mass=BALL_MASS, condim=6, friction=[0.5, 0.005, BALL_ROLL_MU])
    return spec


def apply_course(env_cfg: Any, *, crab: str = "red", senses: bool = True) -> dict:
    """One crab in the crab's colour on the start mark, the lane and its obstacles,
    the ball, and a camera that follows. Returns `layout()`."""
    from mjlab.entity import EntityCfg
    from mjlab.viewer.viewer_config import ViewerConfig

    from scenes.registry import _install_spec_edits
    from tasks.jumper.common.constants import STAND_Z
    from tasks.jumper.five_foot import objects as O

    entities = env_cfg.scene.entities
    robot = entities[RED]
    if crab == "blue":
        robot.spec_fn = _recoloured(robot.spec_fn, SHELL_RED, SHELL_BLUE)
    robot.init_state = dataclasses.replace(robot.init_state, pos=(START_X, 0.0, STAND_Z), rot=(1.0, 0.0, 0.0, 0.0))
    if "reset_base" not in env_cfg.events:
        raise ValueError("the course pins the spawn, and this config has no 'reset_base'")
    _pin(env_cfg.events["reset_base"])
    # a stumble plays out on camera; the controller's own tilt fallback is what a robot has
    for term in env_cfg.terminations.values():
        if "limit_angle" in (term.params or {}):
            term.params = {**term.params, "limit_angle": math.pi}
    if not senses:
        sensors = list(env_cfg.scene.sensors or ())
        env_cfg.scene.sensors = tuple(s for s in sensors if not (type(s).__name__ == "ToFSensorCfg" and s.name == "tof"))

    L = layout()
    props = {}
    for i, (bx, by) in enumerate(L["balls"]):
        name = "ball" if i == 0 else f"ball{i}"
        props[name] = EntityCfg(spec_fn=functools.partial(_ball_spec, name),
                                init_state=EntityCfg.InitialStateCfg(pos=(bx, by, BALL_R + 0.002)))
    for i, (kx, ky) in enumerate(L["blocks"]):
        props[f"block{i}"] = EntityCfg(spec_fn=functools.partial(_block_spec, i),
                                       init_state=EntityCfg.InitialStateCfg(pos=(kx, ky, PLAT_H + BLOCK[2] + 0.002)))
    entities.update(props)
    O._place_props(env_cfg, props)
    env_cfg.sim.mujoco.cone = O.PROP_CONE
    env_cfg.sim.mujoco.impratio = O.PROP_IMPRATIO
    _install_spec_edits(env_cfg.scene, None, functools.partial(decorate, crab=crab))
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 1024)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)

    # the camera follows the crab from the lane's open side (-y: azimuth 90), a little above
    viewer = env_cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    viewer.entity_name = RED
    viewer.distance = float(os.environ.get("COURSE_CAM_DIST", "1.6"))    # 2.4, then 1.9 before (Andrew: "zoom in a tiny bit more")
    viewer.azimuth = float(os.environ.get("COURSE_CAM_AZ", "90"))
    viewer.elevation = float(os.environ.get("COURSE_CAM_EL", "-18"))
    print(f"[course] lane {LANE_LEN:g} x {LANE_W:g} m; hill {L['hill']['flight1'][0]:.1f}-{L['hill']['end']:.1f} "
          f"(tops {L['hill']['top1'] * 100:.0f}/{L['hill']['top2'] * 100:.0f} cm), forest {L['forest'][0]:.1f}-{L['forest'][1]:.1f} ({len(L['pillars'])} pillars), "
          f"ramp {L['ramp'][0]:.1f}, platforms {L['plat1'][0]:.1f}-{L['plat2'][1]:.1f} ({PLAT_H * 100:.0f} cm) with the notch "
          f"{L['pit'][0]:.1f}-{L['pit'][1]:.1f} and the box at {L['blocks'][0][0]:.1f}, ball at {bx:.1f}, "
          f"finish {L['finish']:.1f}; the {crab} crab on the mark")
    return L
