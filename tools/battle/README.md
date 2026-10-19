# The crab battle: two Jumpers on a table

A red crab and a blue crab on a plank over the beach, each trying to push the
other off, best of three. The toolkit plays one robot per world; this directory
is the second robot and everything that follows from it. Nothing under
`scripts/` changes, and no task is added: the duel is `play --app` with two
controllers in one physics world.

```bash
# a match: both crabs rolling their moves, taunts at the bell, a dance for the winner
MUJOCO_GL=glfw .venv/bin/python tools/battle/duel.py --scene beach --headless --video out/battle/take1.mp4 \
    --red-aggression 0.9 --blue-aggression 0.2 --seed 7

# the plumbing: two chargers, no set pieces
.venv/bin/python tools/battle/duel.py --scene beach --headless --no-moves --red tools/battle/pilots/charge.py --blue tools/battle/pilots/charge.py

# the red crab looking for the blue one in its own depth picture
.venv/bin/python tools/battle/duel.py --scene beach --red tools/battle/pilots/brawl.py

# you on the keyboard against the charger, in the live viewer
.venv/bin/python tools/battle/duel.py --scene beach --red keys
```

| file | what it is |
|---|---|
| `arena.py` | the world: the second robot entity (`robot2`, the task's own robot copied and recoloured), the round table on its pedestal (or the first takes' plank), the two start marks, the cloned reset events, the cross-crab collision bits, a camera on the table, and the dressing round it (`dress`: nothing in it collides) |
| `referee.py` | who lost (flipped onto its head or back, off the plank and on the sand, or down and out on its side), the score, the set pieces it cues around each round, the linger before the reset |
| `moves.py` | each colour's taunt, cheer and victory dance by the bundle's mode names, the pad press that enters each read from the app's own FSM table, and the `Performer` that plays one |
| `duel.py` | the driver: builds the world, runs two `AppPlayer`s from one app, hands each a virtual pad with its pilot, loops with the referee, records |
| `hud.py` | the control readout painted into the take as it is recorded: each crab's pad in its corner (red bottom left, blue bottom right -- sticks, triggers, hat, buttons), its mode and intent, the score across the top; `--no-hud` leaves it off |
| `eyes.py` | one crab's senses -- the dToF image self-masked and clustered, the IMU, the encoders -- and the other crab found in it |
| `pilots/brain.py` | the default: rolls an intent (charge, brace, dodge, back off, circle, recentre, and its style's own) from weights set by its aggression, seeded per crab; keeps its crab on the table before anything else; plays the referee's cues, facing the camera |
| `register_gestures.py` | adds the three gestures to `deploy/manifests.json` (Menu + X / Y / A) and `deploy/jumper.controller.toml`, once trained and exported; then `scripts/deploy.py --bundle out/bundle_example --allow-incomplete` rebuilds the app |
| `clips.py`, `make_threat.py`, `make_pump.py`, `make_wave.py` | the battle's own gestures, keyframed the way `tools/fetch/make_zoidberg.py` keyframes the Zoidberg: the threat display (blue's taunt), the claw pump (red's victory) and the wave (blue's victory); each writes `tasks/jumper/gesture_<name>/media/<name>.npz` and a preview under `out/battle/` |
| `pilots/charge.py` | face the foe and walk into it, reading both poses from the world (a cheat, for the plumbing); `CHARGE_MODE=claw_left` charges in the claw mode, arm out, snapping |
| `pilots/brawl.py` | the same from the depth picture alone (`SENSES = True` builds the sensors) |
| `pilots/park.py`, `pilots/lift.py`, `pilots/grab.py` | the probes: a crab that stands still (in a claw mode with its arm out under `PARK_MODE`), the body lever scripted against it, and the claw aimed at one of its legs or at its arm; see "What a crab can do to another" |

## What the second robot needed

Three things that are invisible in the config and would have produced a duel
that looked right and was not; `arena.py`'s docstring has the detail.

* **Two copies of the robot cannot touch.** The family's collision masks give
  each leg a bit and the base none, so two crabs' bases and same-numbered legs
  pass through each other. The arena ORs one bit per crab into every collision
  geom, each crab's `conaffinity` carrying the *other's* bit.
  `tests/test_battle.py` runs MuJoCo's pair test on the compiled two-robot
  model, with the undecorated model as the control.
* **The world's fall reset hides the fall.** `fell_over` would teleport the
  loser upright mid-air. The arena moves it out of reach and the referee takes
  the fallen back after `--linger` seconds.
* **An entity with no reset event stays where it was.** Every startup and reset
  event the task gives `robot` is cloned for `robot2`; `reset_base` and its
  clone are pinned to the marks.

## The match

The referee's phases: `ready` (the controllers settle on the marks), `taunt`
(both crabs play their threat display, the red one the Zoidberg, the blue one
the paw swipe; cut short with Menu after `BRAIN_TAUNT_CUT_S`), `fight`,
`linger` (the fall plays out), `cheer` (the round's winner, hello or salute,
while the loser lies where it fell) and, after the match, `victory` (the
Brazilian for red, the maze for blue). Each piece is a mode of the shipped
bundle; `BATTLE_RED_TAUNT=gesture_hello` and the like swap one by name, which is
how the battle's own gestures go in once trained and bundled
(`gesture_threat`, `gesture_pump`, `gesture_wave`). `--no-moves` skips them all.

Two things the first takes taught about set pieces: a chord pressed while the
controller is in `safe` or still ramping is dropped, so `Performer` waits for
the crab to be walking with its policy running before it presses; and the
shove carries the shover -- in two takes of three the winner was on the sand
beside the loser when its cheer was cued -- so a winner not standing on the
plank is stood back up in its middle by the referee first, the loser left where
it fell. A piece that ends in the tilt fallback used to keep its toggle
latched and replay itself when the crab was next upright -- the controller now
ends a recording the cascade leaves early and releases its latch (see "The
reset" below); the brain still presses Menu before the next round for any mode
it finds itself in.

A round is lost three ways: **flipped** (tilt past 90 degrees for half a
second), **off the table and on the sand** (base under 0.16 m, which a crab
reaches a tenth of a second before it lands), or **down and out** (on its
side past the controller's own 50-degree limit for 3 s: the controller has
dropped to `safe` and the crab is not getting up).

The brain's roll: at the bell, when an intent ends, and when the foe comes
within `BRAIN_NEAR_M`, it draws one of its intents from weights set by
`--red-aggression` / `--blue-aggression` (0 to 1). At 1 the crab charges
nine times in ten; at 0 it braces, dodges and backs off and charges only when
the foe is already on it. Each crab has its own generator seeded from
`--seed` (red `seed`, blue `seed + 1`), the seed is printed, and every decision
is logged with the roll and the weights, so a take replays move for move and
a dull one is re-rolled rather than re-cut. Every strafe and the back-off stop
at the plank's edge: on the 0.7 m plank a circle at full stick from the mark
walked straight off the side in 0.7 s.

## Staying on the table

Andrew, on the first thirteen matches (2026-10-14): "in every fight both
crabs fall off... sometimes the crab seems to throw itself off unnecessarily
after it's already won. We want the crabs to try to beat the opponent, but
they also need to prioritize not falling off." The logs agree: in all 30
rounds of those matches the referee had to stand the winner back up for its
cheer, because it was on the sand beside the loser or out on the rim. A shove
is a walk at full stick into a foe, the foe stops resisting the moment it
tips, and the referee calls the round only when the loser's base is near the
sand, about a second later -- a second the winner spends walking off after it.

The brain now puts not falling first, over every intent (`brain.py`):

* **the guard** -- every frame's walk is checked against the table `BRAIN_STEP_M`
  (0.12 m) on in the direction the sticks ask for; where that leaves less
  than `BRAIN_KEEP_M` (0.22 m, the front feet stand 0.18 m ahead of the base)
  the walk is zeroed and the turn kept. A shover held there has its foe's base
  0.33-0.38 m ahead of its own, which is the rim. A charge, a brace or a wedge
  the guard has held for `BRAIN_HOLD_S` (0.4 s) is over, and the roll that
  follows is weighted to a dodge round the foe along the arc or a recentre,
  since the push cannot go on without a step off the table -- without that,
  the bull stood held at the rim charging at nothing until the matador shoved
  it over from the side, 0-4 in rounds on seeds 1 and 3.
* **the win is called by the crab, not the referee** -- the foe's base past
  the rim or under the table's top, and the crab stops shoving that instant
  and walks back to the middle. With all three rules in, the winner stood on
  the table at the end of every one of the 11 rounds on seeds 1, 2, 3 and 5
  (`out/battle/safe1b.mp4`, `safe2`, `safe3b`, `safe5`): the referee's
  restore never fired, against 30 of 30 before. The rounds are longer (5 to
  30 s, where the old ones ended in 3 to 8) and two of the eleven were 45 s
  draws, with a crab held at the rim and walking back to the middle three or
  four times in a round. The bull took 6 decided rounds to the matador's 3
  and the matches went bull 2-0, 2-1, 2-0 and matador 2-0.
* **`recentre`** -- an intent that ends further than `BRAIN_RIM_M` (0.5 m)
  from the centre rolls a walk back toward the middle, nose kept on the foe,
  at weight 2 against the charge's 0.9; a brace or a back-off on the rim is
  never rolled. The dash checks its landing (0.35 m on) the same way before
  it is pressed.

A cheer and the dance are played to the camera: the duel works out the
heading that faces the recorder from its azimuth (MuJoCo's free camera sits
`azimuth` degrees round from +x on the far side of what it looks at, so the
default 90 is on -y and a crab facing it heads -90), the referee stands a
winner back up facing it, and the brain turns its crab to within
`BRAIN_FACE_DEG` (10) of it before pressing the piece, full stick while more
than 40 degrees off.

## The readout and the buck

Andrew (2026-10-15): "I would like to see the control output in the bottom
left and bottom right corner for each robot." `hud.py` paints each crab's pad
-- the frame its pilot put on the virtual pad that step, so a chord shows for
exactly the ticks it was held -- into the recorder's frames as they are
written (`HudVideo` over `mjrl.viewer.video.ReplayVideo`), with the mode the
controller is in, the intent the brain is on (and its stage, or the lock), and
the score and the round across the top. One file, one pass; the fetch film's
overlay was PNGs composed afterwards.

And: "I don't see the robots using all their moves... the red crab was a
little over the top of the blue crab, it would be interesting to see what
would happen if the blue crab tried to dash or jump from this position." Two
things. Every style but the boxer (claw mode, where the hop cannot be pressed)
now rolls the `dash` -- the bull at 1.6, the wrestler 0.4, the matador 0.3.
And the **buck**: a crab whose foe's base is `BRAIN_BUCK_UP_M` (2.5 cm) or more
above its own, within `BRAIN_BUCK_M` (0.30 m) and ahead, presses the jump
(`BRAIN_BUCK=dash` for the hop) whatever it was doing, lets it play out, and
logs how far it rose and what it did to the foe's tilt and height; not again
within `BRAIN_BUCK_COOL_S` (4 s).

The buck is slow: the controller ramps for `mode_switch_ramp_s` (1.0 s) into
the jump's home pose, and the recording leaves the ground 0.70 s in (1.47 s of
clip at 250 Hz, apex at 0.87 s), so the crab lifts off about 1.7 s after the
press and is back walking by 3 s. That is a buck against a foe that stays on
it, not a reflex. The first flip in `out/battle/hud3.mp4` (seed 3, round 1,
"flipped (163 deg for 0.5 s)" at 2 s) is therefore **not** the buck's: the
matador hopped onto the bull, its base 12 cm above the bull's, and went over
backwards off the bull's shell on its own landing, past vertical at about
11.8 s, while the bull's jump pressed at 11.7 s was still ramping and left the
ground at about 13.4 s (Andrew saw it: "blue flops onto its back by itself
more so than red bucking it"). What the buck does do: in `hud1.mp4` (seed 1)
round 3 has four bucks, resolved 3 s after the press with the base 9-11 cm up,
so the jumps did launch; the three with a foe close over the back tilted it
17, 42 and 76 degrees, and after the last the bull stayed down (down and out).
In `hud2.mp4` (wrestler against bull) both bucked at once and tilted each
other 21 and 41 degrees. A buck under a foe that is merely near, not on top,
does 11 degrees. A jump pressed under a foe can end in the
tilt fallback, and until the controller fix below that left its toggle
latched, re-entering the jump every time the crab was upright again (18 s of
it in hud3's round 3, hands off meanwhile); the brain's press-it-off after
`BRAIN_BUCK_WAIT_S` + 1 s in the mode stays as a belt and braces, for the jump
and the dash alike.

## The reset

Andrew (2026-10-15): "in the second match the reset is not good, the blue
robot dashes immediately before the threat display as if it had the move
queued up from the previous match, so instead of starting in the right
position blue starts in the middle of the arena." It had: in `hud3.mp4` blue's
dash (a toggle, latched at 10.86 s) tipped it over at 11.69 s and the cascade
dropped to `safe` with the latch still on; `_next_round` resets the world but
not the controllers, and the moment blue stood level on its mark (24.40 s) the
cascade went `safe -> locomotion -> dash` in two ticks and played the dash
from the start, off the mark and into the middle, before the stare-down. The
brain's Menu press at 24.90 s came half a second late. The same latch is what
re-entered the jump for 18 s in round 3.

The fix is the controller's, not the brain's, because the robot would do the
same: fall mid-dash, be stood back up, dash again. `Controller::tick`
(`deploy/fsm/src/control.rs`) now reports a recorded motion the cascade leaves
before it plays out -- the tilt fallback, stale feedback -- as completed, the
same report a recording that ran to its end makes, and every host releases
the latch on it; a gait left the same way keeps its latch (`carry` is a held
LB). Pinned by `a_motion_the_tilt_fallback_cuts_short_is_over_and_a_gait_is_not`
in the crate. Replayed with the rebuilt controller (`hud3c.mp4`, seed 3):
blue's round-2 cascade reads `safe -> locomotion` and stays there until the
bell, the crab on its mark through the stare-down.

## The dressing

Andrew: "dress it up to make the space and platform look more interesting."
`arena.dress` (on unless `duel.py --bare`): the table's top decked in planks
(a 512 x 512 texture of nine planks, each its own shade with grain and seams,
on a 1 mm disc mesh with planar texture coordinates -- on the cylinder itself
MuJoCo maps a texture radially and the planks were a pinwheel), a painted ring
at the dodges' radius with the centre spot and a mark the colour of each crab
on its start, a brass lip over a dark skirt round the rim, a stone plinth under
the pedestal, four tiki torches 1.3 m out flanking the shot each with a point
light of its own (a third of it reaches the table, a tenth 3 m off), a line of
red and blue pennants strung behind the table, shells and two starfish on the
sand, and the fetch film's dunes, palms, bushes and rocks
(`tools/fetch/scenery.py`). Everything is `contype` 0, so the fight on the
dressed table is the fight on the bare one; the takes above are the first
with it.

## What a crab can do to another

Measured against a parked crab (`pilots/park.py`) on 2026-10-14, because the
fight was meant to be claws and flips and the first takes were shoves:

* **Nothing on the crab flips a standing crab.** Head on, the two crabs' front
  feet meet at 0.33-0.38 m base to base (the front feet stand 0.18 m ahead of
  the base, `NOMINAL_FOOT_XY`), so a crouched nose-down charge (`lift.py`) never
  reaches the underside, which is 0.096 m over the plank; the tall nose-up push
  only shoves. The shove moved the parked crab 0.35 m per push and tipped it
  3 degrees until it went off the plank's end backwards, where it reads as a
  flip (88 degrees nose up as its rear legs went over). Tipping a 2.54 kg crab
  about its rear feet needs about 20 N above its centre of mass (0.11 m); the
  carried arm holds its preset with 1.75 N*m, 12 N at the claw and 0.09 m up.
* **The claw does not take a leg in a live approach.** The web-up arm carries
  the mouth 0.18 m ahead of the base, 0.09 m over the plank, closing side to
  side (palm at (0.155, 0.07), finger tip at (0.20, -0.05) in the base frame).
  The foe's front strut (its claw, standing on the plank) tops out at 0.075 m,
  under the mouth, and its forearm sits at 0.085 m, 0.27 m from the grabber's
  base with a 12 cm sidestep, which the claw mode's dead-banded walk (`grab.py`,
  strafe pulses and all) brought to within 4-10 cm before stalling, five times
  in five. The finger shut fully every time (+0.10 rad; open is -1.47).
* **Two claws out clash.** With both crabs in a claw mode the arms are bars at
  one height with no front feet under them, and the mouths meet at 0.28 m with
  a 7 cm offset (`grab.py` with `PARK_MODE=claw_left`, aim (0.10, 0.066)): the
  claws touched tip to tip twice, the grip closed on nothing. A claw-mode charge
  (`CHARGE_MODE=claw_left`) shoves the parked crab along at about 4 cm/s, a
  third of the posture scrum's pace, and the charger toppled over the foe at the
  plank's end before the foe fell.

So the plank decides the rounds by shoving, the "flips" are falls off its
end, and the claws can be in the picture (out, snapping, clashing) without
deciding anything. Takes: `out/battle/lift1-3.mp4`, `grab1-5.mp4`,
`clawcharge1.mp4`.

That picture is `BRAIN_MODE=claw_left`: both brains enter the claw mode at
the bell (after the taunts, which play from the walk), carry the arm out and
snap at a foe within `BRAIN_SNAP_M`; the set pieces press from and hand back
to a claw mode as they do from the walk (`moves.WALKING`). `out/battle/claws1.mp4`
(seed 2, 0.9 against 0.2) is the same match as `brain4` fought with claws out:
red 2-0, rounds of 6.9 and 6.8 s, the claws meeting in the middle before the
shove, the hello cheered from the claw mode, 54.6 s of film.

## Styles

A crab fights one of four playbooks, `duel.py --red-style bull --blue-style
matador` (the defaults; `--red-aggression` and the like still override the
style's own weight):

| style | aggression | the extra intent | what it looks like |
|---|---|---|---|
| `bull` | 0.9 | `dash` (only where the bundle carries the hop) | charges, and inside `BRAIN_COMMIT_M` (0.32 m) stops re-aiming for `BRAIN_COMMIT_S` so a sidestep is not followed |
| `matador` | 0.2 | `counter` | braces until the foe is `BRAIN_COUNTER_M` (0.36 m) off, steps aside along the arc, turns and shoves the passing flank |
| `wrestler` | 0.6 | `wedge` | walks to a point `BRAIN_FLANK_M` out on the foe's side, crouches, drives in to `BRAIN_WEDGE_M` base to base and heaves |
| `boxer` | 0.5, claw out | `strike` | circles with the arm swinging every `BRAIN_STRIKE_S` |

Every sidestep, counter, circle and strike moves along the table's tangent,
`arc(side, stick)`: the stick is the tangent at the crab's bearing from the
centre plus an inward pull of `BRAIN_ARC_PULL` per metre outside `BRAIN_ARC_R`
(0.45 m on the 0.8 m disc), so a dodge is a step round the rim rather than
across it. On the plank the first dodges walked straight off the side.

Two bulls walking into each other stall nose to nose. A charge or a brace with
the foe within `BRAIN_NEAR_M` that gains under `BRAIN_LOCK_MOVE_M` (4 cm) in
`BRAIN_LOCK_S` (1.2 s) is a **lock**: the crab presses LB into the claw mode,
swings and snaps the arm for `BRAIN_LOCK_SWING_S` while still pushing, presses
LB back and rolls again. In `out/battle/lock2.mp4` (a bull against a bracing
bull) five locks broke that way.

The **dash** is `jumper.dash`, the high jump's own recording carried 0.47 m
forward (`make_dash.py`, `DASH_M`): the base's x ramps through the push, holds
0.72 m/s over the 0.42 s flight and stops through the absorb, the airborne
feet move with the body and the planted feet stay, the joint angles are the
jump's. It trains by resuming the jump's policy, and the jump's reward weights
were not enough: with them the hop gained 0.10 m a time in the duel, because
the joint and foot terms score the jump's shape standing still as well as
moving, and only the velocity term asks for the travel. `jumper.dash` owns
three numbers of its own (the velocity term at weight 6 and sigma_sq 0.08,
joint tracking at 1.5; `tasks/jumper/dash/env_cfg.py`), and retrained with them
for 2000 iterations (2048 environments, 1.1 s an iteration on the RTX 3060) its
training `travel` metric went from 0.20 to 0.39 m and the hop in the duel
(`BRAIN_RED_ONLY=dash` against `pilots/park.py`, `out/battle/dash4.mp4`) gains
0.21-0.26 m a press with the base 12-13 cm up at its apex, the jump's own 14:
from 0.55 m it lands at 0.38, which is where the front feet meet. The bull's
`dash` intent faces the foe and presses L3 when the foe is `BRAIN_DASH_MIN_M`
to `BRAIN_DASH_MAX_M` off, then waits `BRAIN_DASH_WAIT_S` for the clip to hand
back, and logs the range before and after and the apex. `jumper.sprint`
is the posture walk retrained with a 1.2 m/s ceiling and a longer stride, the
pad's B from `locomotion`; `BRAIN_RED_MODE=sprint` fights the whole round in
it.

The styles so far, on the disc (seeds 1, 2, 3, 5): the bull takes 6 rounds in
11 against the matador, and the matador's counter won `styles7.mp4` 2-1 on a
seed the bull had taken before. Wrestler against boxer went 0-2 and 2-0 with
one 45 s draw: the wedge gets under the foe's trunk (8 mm of lift, the posture
policy yielding under the load) but nothing trained flips a crab, so the
heave is a shove from the side.

## Measured

Framerates on the RTX 3060, warp backend, one environment, 200 Hz world,
headless, 2026-10-14:

| | steps / s | of real time |
|---|---|---|
| two crabs, no depth sensors | 48 | 0.24x |
| two crabs, both depth sensors | 37 | 0.19x |

So the sensors are built only when a pilot says `SENSES = True`. A profile of
the rest put a third of each step in the environment's own command and
observation managers, which `play --app` never reads; `duel.py` skips both
(`skip_the_managers`), as `mjrl.replay.skip_rewards` already skips the rewards.

The first charger-versus-charger match: the crabs meet in the middle within
2 s of the bell and one is on its side within 3-4 s; red lost both rounds, with
symmetric pilots, so the start marks are not yet known to be fair.

The brain takes (`out/battle/brain1..4.mp4`, red at aggression 0.9 against
blue at 0.2, the 0.7 m plank): rounds last 0.8 to 12.8 s and end nine times
in ten with a crab shoved off the plank's end; seed 1 went blue 2-1, red 2-1
and blue 2-1 as the brain was fixed between takes, seed 2 red 2-0. With both
crabs rolling, the same seed and the same brain replay the same match. A full
match with its taunts, cheers and dance is 55 to 98 s of film at 48 steps/s
headless, about four minutes to record.

## Knobs

Environment variables, read by `arena.py` and the pilots: `BATTLE_ARENA`
(`disc`, a round table of radius `BATTLE_DISC_R` 0.8 m on a pedestal, so a crab
can dodge to the side and the edge is the same distance in every direction;
`plank` for the first takes' board), `BATTLE_PLANK_LEN`
(2.0 m), `BATTLE_PLANK_WIDTH` (0.7; at 0.5 the first shove settled every
round), `BATTLE_PLANK_TOP` (0.25), `BATTLE_START_GAP` (0.9), `BATTLE_CAM_DIST` /
`BATTLE_CAM_AZ` / `BATTLE_CAM_EL` / `BATTLE_CAM_X` (2.3 / 90 / -12 / 0; 90 is the side view, the
look-at slides along the plank), `BATTLE_<COLOUR>_<PIECE>` for
the set pieces, `BRAIN_*` for the brain's distances and timings (`BRAIN_MODE`, or `BRAIN_RED_MODE` /
`BRAIN_BLUE_MODE` for one crab, `BRAIN_SNAP_M`, `BRAIN_SNAP_S` for the claw fight; the
styles' knobs are listed above; `BRAIN_KEEP_M`, `BRAIN_STEP_M`, `BRAIN_HOLD_S`, `BRAIN_RIM_M`,
`BRAIN_RECENTRE_S`, `BRAIN_FACE_DEG`, `BRAIN_FACE_S` for staying on the table and facing the
camera), `CHARGE_*` and
`BRAWL_*` for the other pilots' sticks (`TURN_DEG`, `TURN_STICK`, `WALK`,
`CROUCH_S`).
