# The obstacle course

Andrew (2026-10-22): "a series of obstacles like a dog course that the crabs race
on ... we will need to make one obstacle course and then we have each crab race
it separately and I will edit the video and play them back side by side ... we
want to see the depth camera and the goal is to have a few obstacles like a stair
case, a ramp, a forest of pillars they have to navigate, a heavy ball they have
to push and the crabs need to do a dance when they arrive at the end." And,
after the first takes: "use the dash movements or lift the claws so the climb up
the stairs is more fluid ... at the pillars every single time they just ram
right into the most obvious pillar in the middle ... for the ball section what I
actually want is a pit and the crabs will need to push two blocks in the pit,
one after the other to form a safe bridge across to the finish line. For the
ball they should have to push the ball across the finish line and also the crab
shouldn't stop on the finish line it should continue another meter or two and
then celebrate."

    python tools/course/course.py --crab red --scene studio --headless --video out/course/red.mp4
    python tools/course/course.py --crab blue --scene studio --headless --video out/course/blue.mp4
    python tools/course/course.py --crab red --truth            # the pilot reads the world: the control
    python tools/course/course.py --pilot keys                  # drive the course yourself

| file | what |
|---|---|
| `track.py` | the lane: a kerb down each side, the hill (stairs up, ramp down, ramp up, stairs down), the forest of pillars, the ramp, the two platforms with the notch and the box, the drop, the ball, the finish gate, the set round it; one crab in its colour on the mark |
| `referee.py` | the clock, the splits as the base passes each section, the ball over its line, the finish, the dance cue, the end of the run |
| `pilots/runner.py` | the pilot: the hill walked straight on the gyro (nose up for the flight), the forest planned on the map, the box shoved into the notch and walked over, the leap off the second platform, the ball shoved across the finish line, on past it, the turn to the camera and the dance |
| `hud.py` | the readout: the depth image shaded by what the sensing made of it, the map round the base with the plan, the sticks, the clock and the splits |
| `course.py` | the runner: one `AppPlayer` on a virtual pad, as `tools/race/race.py` runs two |

Everything reused is the race's and the battle's: the dToF reading
(`tools/race/sight.py`), the occupancy map and A* (`tools/fetch/mapper.py`), the
recoloured crab and the set pieces (`tools/battle/arena.py`, `moves.py`), the
virtual pad (`tools/battle/duel.py`). The pilot drives the shipped bundle through
the pad, so the whole thing is a `play --app` the robot could run.

## What the crab can climb

Measured 2026-10-22 on jumper.posture's locomotion at full stick, on the native
backend, with a probe outside the repository (`probe_step.py`: a step, a flight or
a ramp in front of the crab, the heading held on the true yaw). The gait is not
deterministic, so anything that mattered was run two or three times.

| obstacle | standing | body raised (R3 + stick up, 0.153 m) |
|---|---|---|
| one riser of 2 or 3 cm | walked over | walked over |
| one riser of 4 cm and up | stops at the face | stops at the face |
| 3 cm risers x 4 on 30 cm treads | jams on the first step | climbs (probe), but 1 of 3 on the course |
| 2.5 cm risers x 4 on 40 cm treads | climbs, 4 of 4 | 1 of 2 |
| 2 cm risers x 5 on 35 or 40 cm | climbs, 6 of 6 | -- |
| 4 cm risers x 3 or 4 down, 30 cm treads | walks down, 8 of 8 (stick 0.6 and 1.0) | -- |
| 6 cm x 3 down | walks down, pitching 15 deg | -- |
| 8 cm x 3 down | hangs on the first edge | -- |
| ramps of 8-25 degrees up, 15-20 down | climbs and descends at ~0.5 m/s | -- |
| a platform's edge | goes over, lands on its side, and the controller stays in `safe`: a crab on its side is never level again | |

So the hill is 2 cm risers on 40 cm treads up to a 10 cm platform (2.5 cm
jammed one course take in three, red2, where the probe had climbed 4 of 4), a 12
degree ramp down, a 12 degree ramp up to a 12 cm platform, and 4 cm risers on 30
cm treads down -- the staircase and the ramp Andrew asked for, in the sizes this
policy climbs, with nothing on the course that is an edge rather than a step.
The pillars are 0.15 m square and 0.45 m tall, a slalom with no gap between
faces under 0.85 m (three to a row 0.7 m apart left the planner one cell of
corridor at its clearance, and blue clipped a corner for 6 s, blue1). The raised
section is a 12 degree ramp up to a 12 cm platform, a notch 0.52 m long with
its own floor 10 cm below the platform tops, a second platform, and off its far
end a 12 cm drop to the ground: the leap of faith (Andrew: "a ramp going up and
then two raised platforms with a trench between them. Then instead of a ramp
down we have a 'leap of faith' where the crab has to jump off the second box
and not end up on its back"). On the first platform stands one box, 0.5 m
long, 10 cm tall, half the lane's width, and 0.25 kg, with no guides, its
bottom-rear corner cut away 7 cm in and 1 cm up so that its top is that much
longer than its bottom: shoved in, it lies flush in the notch and the crab
walks over it to the second platform, wherever across the lane it came to lie. The section replaced a
trench cut into the lane's floor, which showed the world's grey ground plane
through it however the lane was raised and painted.

Every number was set by a failure. A box whose centre is past the notch's edge
tips, then rests on its front corner and the edge and stays; it drops only
once its rear end is over the edge too, and its rear comes down about 6 cm
past the edge whatever the notch's length (pusher probes at 0.52 to 0.70 m) --
so a notch 2 cm longer than the box caught the box's front on the far wall
before its rear cleared and left it proud (truth17), a notch 6 cm longer drops
it every time but leaves a 6 cm slot at the near edge, which stalled the
crossing 22 and 26 times (truth20, truth23) even at full stick. The slot is
closed from above by the box's cut-away bottom-rear corner and a notch cut to
the box's top: the notch is 0.52 m, the box's top 0.50, so lying against the
far wall the top's rear is 2 cm from the near edge, a gap under a toe's width,
and the bottom's rear, 7 cm shorter, cleared the edge on the way down. Against
the far wall is where the box ends up: the crab's front feet walk on its top
while the shell pushes its rear, and they push it on after it drops, so a
notch cut for the drop alone (0.58, which the pusher probes wanted) left the
8 cm slot again (blue22). The cut is square, not a 45 degree bevel: the crab's
claws push the box's rear low down, and a bevel was a ramp for them -- twice
they went under it and lifted the box's rear as it tipped, the crab ending
nose-down in the notch under a propped box (blue25, red10); a cut 3 cm up
took a claw in (one claw in the cut, one on the face above, the box walked
off sideways, red11), 1 cm is under a claw tip's radius. The box is as wide
as the lane because a box pushed from behind turns away from the side it is
pushed on, and the crab pushes more with one claw than the other: a 0.8 m box
walked off to one side, 0.3 to 0.7 m and up to 14 degrees over the 1 m shove,
in nine takes of twelve, and the crab then crossed into the open notch beside
it; squaring it by strafing dragged it sideways instead (red16, red17); and a
box with its mass at the rear, which tips only at the edge, swung on its two
loaded rear corners under each claw's push and was worse (red11, blue28,
red12). Between the kerbs the box can yaw 6 degrees and shift 2.5 cm, no
more, and the pusher probe `probe_push7.py` (a face from 3 to 15 cm, the
claws and the shell, going on past the edge as the crab does) lands the cut
box flat and square to 6 degrees of yaw; lane-wide it dropped flat and square
in every take since (blue34 on). Then Andrew: "it is not interesting if the
bridge block is the width of the corridor. make it half the width" -- so it is
half the lane, 1.1 m, and it walks aside again under the shove, 0.16 to 0.42 m
in the first six takes, square still (its yaw within a degree: the wider box
has the friction to hold its heading, not its line). The crossing therefore
reads where the box lies instead of trusting the line it was told: once the
box is in, the open notch is 10 cm down either side of it and the box's top is
flush, so the longest flush run across the picture at the open floor's range
is the box, its centre the crossing's line; a run reaching the picture's own
edge is one-sided, and the box's known width gives its centre from the other
side (`bridge_centre`); the run is read between the open floor's start and
the far wall's foot, since the second platform beyond is flush too and widened
the run by 0.1 to 0.2 (blue73). The crossing then stands up to 1.5 s for the
line to be read, sidesteps onto it on the spot while the notch is still 0.35 m
ahead by the sight (with hysteresis, 0.1 in and 0.04 out: at one threshold
the sidestep, the turn and the walk took turns every step, blue75), and only
then walks over; the walk's own strafe reached the edge first and put the
left legs in the open notch (blue73). Found at the reckoned distance the notch was missed
every time, the reckoning being ahead after the shove (blue54); taken as
two-sided, a box 0.4 m aside read as centred and the crab crossed into the
open notch beside it (blue52). The "box gone" test, which ends the shove,
takes only a thing within 0.42 m of the nose for the box: the notch's far
wall, seen beside a box that has walked aside, kept one shove going 25 s
(blue55). The box walked off to -y in every take (fifteen of fifteen, 0.14 to
0.55 m) and square, which an off-centre push does not do (it turns a box; the
shove's line set 0.12 m right of centre to "lean against" the drift spun the
box 50 to 70 degrees, blue59, red47, red48). It was the shove's own lateral
hold: the crab's gait drifts a few centimetres left over the metre, the hold
strafed right against the box's face past a 2 cm band, and the claws dragged
the box right. A wider band did not help (6 cm: the gait's drift is 7 cm,
red49), and without the hold the crab drifted 0.18 m off the line and turned
the box 55 to 65 degrees (blue62, red50, blue63), so the hold stays, at 2 cm,
and the drag is fought on the other side: the box's friction on the deck is
doubled (0.3 to 0.6: the push the crab has to spare still moves it, the drag
moves it half as far, and the pusher probe `probe_push8.py` lands it flat to
6 degrees of yaw at 0.6), and the shove is 0.6 m instead of 1.0, so the drift
has less distance to grow in. The notch and the box are 10 cm, the deepest
the shell still pushes a box into (Andrew: "the trench needs to be deeper"; 7
before), and the cut is cut from the box, not a slab stood against it: the
slanted slab that did the same job on a 7 cm box stood proud of it (Andrew:
"it looks ugly"). Taller boxes tipped out of the shell's reach (16 cm, truth14)
or became a ramp the crab walked up and pinned (12 cm, truth8); a lid spanning
the notch hung on the near edge by its rear overhang (truth18); a 0.6 kg box
tipping about the edge levered the crab over by its nose (truth11; 0.25 kg);
rails kept a box straight but were guides (gone); pushed 8 cm off its centre
an unguided box turned and slid half a metre aside (blue12), so the shove is
centred to 2 cm before it starts and held there. The shove is one steady walk
until the box is gone from the nose (flush in the notch, the sight sees
nothing there) or the reckoned base is 0.4 m past the edge: a shove that
stopped at the edge and nudged the box on in pulses left it hanging on the
edge twice (red8, blue20), where the steady shoves had dropped it (truth24,
blue19), and a limit 17 cm short of the edge stopped the shove with the box
0.4 m from the notch when the leg odometry, its legs slipping against the
box, had run 0.6 to 1.4 m ahead (blue31, blue33).

Blue takes the last flight of the hill sideways, facing the camera (Andrew:
"make the blue robot attempt to descend side-ways"): on the top platform it
turns to -90 degrees on the spot, strafes down the three 4 cm risers at the
down-stick with the heading held by turns alone (past 8 degrees it stops and
turns, back within 3 it strafes on: at one threshold the two toggled every
step, blue52), and turns back to the lane past the foot. Probed first with
`probe_step2.py` (`MODE=side:-90:-0.7` over the course's flight): down and
walking on, level, three times in three; facing -y, the crab's left is +x,
so the strafe stick is negative.

The hill stage runs until the crab is off the last step and level: the
forest's door stand began on the last riser once, nose down 10 degrees, and
the slow walk off it took 15 s (blue86). And the leap's edge is not read, nor
the crossing ended, while the open notch is still in sight ahead: read as the
drop from the first platform, the notch ended the crossing and began the creep
1.4 m from the real edge (blue86). Open floor counts as the notch only with
flush platform beyond it: the ground past the second platform's drop is open
floor too, and taken for the notch it kept the edge test off, so two crabs
walked off the end still looking for an edge (blue88, which landed by luck,
and red65, which hung nose down). A crab 0.35 m short of the reckoned end with
no edge in sight stops looking and hops rather than walk off.

The leap: walking off a platform's edge the front legs drop and the body goes
over at 10 cm and above, which is why the drop is a hop. The dash hop from a
standstill travels ~0.3 m, the body is 0.5 m long: pressed 5 to 15 cm before
the edge it went over 12 times in 12 (the front legs over the void at
take-off); pressed 20 or 25 cm before, the base lands past the edge with the
rear legs on it, and walking on from there the crab scrambles down onto its
feet: 8 times in 8 off 12 cm, 3 in 4 off 15 cm (1 in 4 from 25 cm before), 0
in 3 off 20 cm, where it hangs or goes over; pressed 30 cm before, the hop
lands short at any height. The in-place jump pressed at the edge goes over at
every height. So the platforms are 12 cm. On the course, with the beat before
it, the hop's window moved: every hop from 8 to 16 cm before the edge came
down on its feet (red26, blue44, blue43, blue45, red28), from 24 cm the base
landed on the edge itself and the crab hung there nose down, rear feet hooked
(blue40, blue41), and from 27 and 32 cm it went over short (red27, blue42). So
the hop is pressed about 14 cm before the edge, its claws, which reach 0.31 m
ahead of the base, hanging 17 cm out over the drop for a beat with the nose
down (Andrew: "it needs to put its front claws over the edge first"), then
level, then the dash, then the walk straight on. The edge itself is seen, not
reckoned: on the platforms the pilot has no landmark (the pillars are behind
it, the box is shoved, and the ball on the ground lies below the platform's
floor plane, where the sight files it as floor), and the reckoning ran ahead
and began the leap at the notch (blue18), then 0.3-0.5 m either way over the
bridge (blue37, red21). The returns more than 9 cm below the plane through the
four walking feet (not the sight's lowest body point: at the brink the claws
droop below the platform and took the plane down with them, blue26) are the
ground beyond the drop, each projected back along its ray to where the ray
crossed the platform's plane (the first ray past the edge lands on ground 12
cm lower well beyond it). That reading is the truth plus 3 to 5 cm from 0.9 m
in to a reading of 0.34, and then sits at 0.30 whatever the truth -- the
sight's lowest ray meets the platform 0.3 m out, so nothing nearer is ever
seen (14 reading/truth pairs in red27, blue44, blue43, within 2 cm) -- so the
crossing ends when the drop is in sight at 1.3 m, the crab walks to a reading
of 0.55, creeps nose down (the sight pitched with it) until the reading meets
0.35, where the brink is 0.30 ahead, and creeps 1.1 s more: the creep makes
18 to 19 cm/s, so that is ~0.09 from the edge, the claw tips 22 cm out over
the drop; it levels, stands a beat, and dashes. Walked to a mark on the level
reading it never stopped (blue34, blue35, red18); walked to one placed at
0.5-1.3 m and reached on the odometry it missed by the odometry's drift
(blue37, red21); the last stretch by the leg odometry came out 0.14 to 0.17 m
for 0.20 to 0.24 counted, and once 0.24 in a quarter second, the feet slipping
in the nose-down creep (red43, which hopped from 0.25 and went over short), so
it goes by time. The brink was first set about 0.14 from the edge and every
hop from there caught the rear legs on the edge (blue49, red39 to red41;
Andrew: "a tiny bit closer ... blue45 is flawless"), and blue45's was from
0.14 too, so the window is narrow: from 0.08 to 0.14 the hop came down clean
seven times (red26, blue44, blue45, red44, blue56, blue57, blue64), from 0.07
to 0.08 it dived twice (blue60, blue61), so the last stretch is 0.95 s, to
about 0.11. Nothing changes in the crab's posture at the brink: the nose
pressed down standing there put red45 over the edge, and levelling there from
the nose-down creep put three more over before any dash (blue60, blue61,
red51), so the crab levels at the brink reading, 0.30 out, creeps the last
stretch level, and stands a beat before the dash. At the creep's slower 0.25
the stretch runs ~0.16 m/s: 1.0 s came to 0.14-0.16, 1.15 s to 0.11-0.12,
where blue75 tipped nose down 37 degrees standing a 1 s beat and hung and
blue77 and red59 went over or hung from 0.10-0.11, while every slow-creep hop
from 0.12 to 0.16 came down clean (red56, blue72, blue74, red57, red58,
blue78), so it is 1.0 s, to 0.14-0.16, where blue45 was, and the beat 0.6 s. A crab hung on the edge after the
hop, nose down past 18 degrees (blue75 hung at 26, under the 30 first set),
works its rear feet off with sidesteps and the nose up and hops again. And the drop is told from the
notch: with a box pushed 0.55 m aside the open notch lay in front of the crab,
read as the drop, and the crab hopped into it (blue58); the notch's floor is
0.52 m long with the second platform flush again beyond, where the real
drop's ground runs on, so flush returns 0.35 m beyond the first return below
the plane mean "notch", not "edge". A fall there is the controller's safe mode and the end of
the take. Measured with `probe_step.py` (`platform`, `leap:dash`, `LEAP_ON`),
`probe_tip.py`, `probe_push.py` and `probe_push2.py` to `probe_push7.py` in the session's
scratchpad, then on the course.
The ball stage is the race's routine (`tools/race/pilots/sweeper.py`) on the
course's line, and the pilot is not told where the balls are: the senses
remember every ball-sized cluster they see (matched within 0.35 m, picked once
seen twice, forgotten after 20 s unseen; sightings within 0.3 m of the second
platform's end are its face's corners), the pilot picks the nearest
remembered ball not yet over, plans round to the point 0.5 m behind it, lines
up, shoves it over with the lateral stick keeping it under the nose, backs
off 0.7 s, and picks again; with nothing remembered it turns on the spot a
whole turn to look, and after a turn with nothing, walks back to the middle of
the floor and looks again (Andrew: "it should push a ball over the line,
rotate around until it sees another ball, target that one and push it"). The
balls' starting places are used once, as a landmark, to fix the pose along
the lane at the first sighting. A first version walked to fixed points
behind each ball's starting place and was scripting, not a routine; it also
walked into the platform's face.

Faster from 2026-10-23 (Andrew: "push the crabs to go a little bit faster,
especially with pushing the balls"): the shove at 0.65 stick (was 0.5), the
walk round behind a ball at 0.75 (was 0.6), the back-off 0.5 s (was 0.7). At
those speeds a ball lost sideways under the shove was counted over when the
crab, walking on, crossed the line alone, and two memories of one ball were
both counted ("4 of 3"): in batch P (seeds 221-224) three takes went to the
finish with one or two balls short of the line. A ball counted over and seen
short of the line since is now not over, and the pilot goes back for it; and
with every ball counted the crab turns once on the spot to look for any left
before it walks to the finish.

There are three balls, 24 cm and 1.2 kg each (half the crab's own
mass; the race's balls are 0.08 kg), abreast on the ball line: one in the
middle and one 0.55 m to each side of it (Andrew, 2026-10-23: "push all three
across the finish line before they can complete the course themselves").
The referee calls the finish only once every ball is over; a ball not over after
thirty seconds of shoving is left and the crab goes on to the next. The balls roll with
friction (`BALL_ROLL_MU`, condim 6): without it a ball knocked at the walk's
speed rolled 18 m, and over the line on its own 0.8 s into the shove (blue101,
blue102) -- the crab then went back for a ball that was not there; with it the
ball rolls 0.66 m from 0.6 m/s and stops, so the crab has to keep pushing. A
ball that stays at the nose is what the senses call a jam and a stall, though:
the jam rule zeroed the forward reckoning for the whole shove (the pose stood
at 20.5 for six seconds while the crab walked to 23.0 and over the line without
knowing it, blue101) and the stall test called the approach a stall and backed
off (blue103), so from the ball stage on a jam is not a jam, and the ball's
own zones are left out of the stall test. And inside 0.45 m (the sight's
`NEAR_M`) the ball fills the picture's width and touches its edges, as any thing
does that close; the matcher dropped edge clusters, so in all six takes the
ball went "out of sight" 1.5 s into the first shove and the crab walked back
for it again and again. A close cluster is now taken with its edges and its
width, and a ball last seen under the nose has five seconds of blind shove
before it counts as lost (1.5 s for one last seen to a side). Under the nose
the ball is a sliver to the sensor, 0.16 m off, 15 cm tall and 8 cm wide
(traced), so a close cluster passes on 12 cm and 4 cm. The crab well past the
line counts as "over" only for the ball being shoved: standing there it was
past the line for every ball, and the next two were over the moment they were
aimed at (red102). And the odometry's speed cap, there to stop slipping feet
reading 1.6 m/s, has a floor of 0.45 m/s in the ball stage instead of 0.3: the
stick model puts the shove's half stick at 0.14 m/s where the crab walks 0.33
with the ball, and capped at 0.3 the reckoning fell 16 to 18 percent behind,
so the ball crossed four seconds before the pilot saw it there. The ball the
pilot is after is tinted green in the readout's depth image (its cluster's zones)
and on the map, the ones still to push as green rings where they were told. The
finish is a gate with a banner in the crab's colour, and the crab goes on 1.8 m
past it before it turns and dances. The camera follows the crab from the lane's open side, so the kerb, the
pillars and the gate post on that side are drawn as glass -- solid, the pillars
hid the crab for most of the forest.

## What the crab knows

The dToF (54 x 42 zones), the IMU and the encoders, and nothing else; it is
told the course (`track.layout()`: where along the lane each thing is, and that
the lane has a kerb down each side) and that it woke up on the mark facing along
it. The pose is reckoned from the feet (`sight.leg_velocity`) and the gyro; the
two kerbs fix it across the lane and fix the heading from their slant, as the
race's sweeper does (`wall_fix`), and they hold both to a few centimetres and a
degree all the way (sense4: y within 0.03 m everywhere a kerb was in view, the
heading within 1 degree).

Along the lane the feet slip on the risers and the slopes and the reckoning runs
ahead -- two to one over the first flight, 1.1 m by the first platform (sense4),
2.8 m by the forest with nothing to correct it (sense1), which walked the crab
into the pillars believing it was past them. The hill is its own landmark: a
sustained pitch of the body is a ramp, the only place on the course with one,
nose down the first and nose up the second, and the pitch setting in and coming
off puts the base at each ramp's known start and end (`ramp_fix`: |pitch| > 8
degrees for 0.5 s, the base 0.25 m past the ramp's start or end; the flights
wobble the pitch +-7 degrees a step at a time and do not trigger it). A
sustained pitch of the wrong sign is not a ramp (the climb's own nose-up is
12 degrees for the whole first flight and the top platform), and it must not
latch the detector "on" either: latched, the first ramp's real onset was no
transition, the first steep moment after it, on the flight down 3.4 m on, was
taken for that onset, and the pose moved 3.4 m back (red46 spent its take
"on the hill"); a fix is also capped at 1.5 m. sense4:
the first ramp's onset took 1.55 m of error down to 0.15 m, the fourth fix left
0.12 m, and the pose was within 0.01 m at the forest's edge. The flight down
slips again (-0.5 m), which the forest's first row of pillars fixes from close
up; the ball, seen where it was told to be, fixes the approach to it; and the
lane's end wall fixes the last metres.

Three things the fetch and the race learned that this pilot leans on:

* **The walk and the turn stick are never sent together** on the straight
  sections. The posture policy can freeze with both held (the fetch's creep
  deadlock, w17); the first control take stood on the first platform with the
  walk stick held for 25 s (truth2). Past 5 degrees of heading error the crab
  turns on the spot, inside it walks with the turn stick at rest.
* **The body is not raised for the stairs.** The 3 cm flight needed it and
  jammed on the course two times in three, with the reckoning lowering the body
  on the second step; the 2.5 cm flight climbs standing.
* **A stage once entered is not left.** The reckoning hovering at a boundary
  flapped hill/forest twenty times a second (sense2).
* **The map keeps what is close.** The fetch's map drops returns nearer than
  0.45 m (the carried mug hangs there) and clears the cells along each column's
  ray to the next thing behind, so a pillar vanished from the map as the crab
  closed on it and the planner routed straight through it, every take, at the
  most obvious pillar in the lane's middle. The course's map keeps returns from
  0.12 m; the crab's own legs are masked out by the sight.
* **The first flight is climbed nose up** (right stick, pitch ~10 degrees): the
  claws trail over each riser's edge walking level, and raised they clear it.
  Probed on the 2 cm flight at 10 and 15 degrees and on the 3 cm flight at 10,
  all climbed; the claw mode (one claw raised, the five-foot gait) climbs both
  too but a third slower, and a dash hop gains one step per hop at 2.5 s a hop
  (`COURSE_CLIMB=dash` for the hopping version, `plain` for the level walk).
* **A stall is read off the picture.** The leg odometry cannot tell a stall from
  a walk (slipping feet read as a walk), but a depth image that does not change
  for 1.5 s with the walk stick held is a crab that is not moving (the race's
  mapper had this test and dropped it because a shoved thing fills the picture
  and never changes; here the shove is the one stage it is off in). Judged on
  the picture's middle columns and only on zones that are on something above
  the floor in both frames: the kerbs beside the picture keep their range as
  the crab walks along them, and the floor keeps its range in every zone
  however far the crab walks (a ray at a fixed angle meets a plane at one
  range), and a median over all zones called "stuck" on the flat before the
  stairs and all through the forest while the crab walked (blue3, blue7).
  Stuck, the crab backs off a second, takes a half step to the side, and goes
  again.

The sight (the race's, `tools/race/sight.py`) had two course-shaped faults
Andrew read off the readout. A cluster whose top runs out of the picture was
the kerb if it was farther than 0.17 m, so the ball, 0.24 m round and cut from
0.3 m, was "wall" (yellow) just where the shove needed it as the ball; a cut
cluster is the kerb only if it is wide as well now (0.45 m, or at the picture's
side). And the floor plane was horizontal through the lowest foot, so on the
ramps the slope ahead and the platform at its top were "things" (red); the
plane is through the standing feet when three or more stand well spread, and
a point's height is the lower of its heights above that plane and above the
horizontal one through the lowest body point, so a slope the crab is on is
floor on the way up (it lies in the feet's plane) and the ground past a ramp's
foot is floor on the way down (it lies under the horizontal one, where the
slope's plane alone ran under it and left it red); a thing stands above both.
The course's own edge test keeps its heights along the IMU's up regardless. Both rules are the course's: `Sight(..., slopes=True, round_tall=True)`,
off by default, so the race's sweeper runs the sight it was measured with
(Andrew: "are the changes you're making going to impact performance on these
other simulations?").

After those takes Andrew also asked for a steadier crab rather than more
takes: "the crab should take a moment to line itself up" for the shove, "it
just needs to go slow" for the jump, "slow down a little so that its odom is
more stable" at the pillars, and the ball "should be following the ball and
trying to push it across the line". So the walk is 0.6 (0.75 before); the
shove waits until the crab is square within 2 degrees and then stands half a
second before it pushes and pushes at 0.55 (0.75); it does not square itself
on the spot as it goes, since turning with the claws on the box's face dragged
the box's near side round, 50 to 56 degrees (blue82, blue83). Pushing, the
crab walks left by 0.1 to 0.2 m over the 0.6 m (red64: +0.07 to +0.19; the
takes without a lateral hold to +0.18), and a box pushed on its left side
turns 45 to 65 degrees, while the hold's strafe against the face dragged the
box aside instead, so the shove has no lateral hold and starts 0.10 m right
of the box's centre, crossing the centre on the way; the creep is 0.25
(0.35). The arrival at the point behind the box is latched, with hysteresis
on the centring sidestep (6 cm in, 2 cm out): tested on the reckoned x alone
every step, the walk and the sidestep took turns at 50 Hz for 50 s and the
crab crept to the kerb with its lateral reckoning a metre out (red67); and the ball's shove
is at 0.5 with the shove over when the ball is seen past the line or the crab
is 0.4 m past it, not when the crab's nose reaches it. After the line the crab
walks on 4 s (about 1.8 m) by time before it turns to the camera and dances:
measured by the reckoning, which the ball's shove leaves 2 to 3 m behind, the
walk to the dance point never ended and two finished crabs walked on to the
end wall instead of dancing (blue79, red60).

The spin is a matter of where the push lands, and nothing else: a carriage
pushing the box with two claws 0.2 m apart (`probe_v.py`, the carriage free
to slide sideways) turns it 7 degrees over 0.9 m from 0.10 m off centre and
50 to 60 degrees from 0.20 m, whatever the face's friction (0.6 or 0.05), and
a rear face cut to a V to centre the push (6 or 10 cm deep) makes it worse --
the box, a tenth of the crab's mass, is what moves aside. A lateral hold on
the carriage brings 0.10 m back to half a degree but cannot save 0.20 m. So
the push has to land within a decimetre of the centre, and in the six takes
after the three balls went in (2026-10-23) five landed 0.2 m right and spun
the box 60 to 80 degrees: the reckoning said 0.05 to 0.10 m right while the
crab stood 0.18 to 0.26 m right. The kerb is 0.35 m tall and the sight's kerb
rule asks 0.27 m, but standing on a 0.12 m platform the crab sees a 0.23 m
kerb, a "thing", so not one kerb fix fired through the whole pit stage (the
fix counts freeze on the platform in every log, the finals included), and a
sidestep the leg odometry read as 0.06 m for 0.33 m stayed in the pose. The
sight's kerb height is therefore the pilot's to set (`Sight.kerb_min`) and
the course lowers it by the platform's height from the pit stage to the
brink; off the platforms it is the race's. The kerb fix's window is 0.5 m there
too (0.25 elsewhere, for the pillars, which read as kerb 0.6 m from one): the
sidestep put the pose 0.31 m out in three seconds, past the window, and the
kerb points found could not be matched to their kerb (red102). And the pit's
own ramp, 0.5 m at 12 degrees, is the fifth and sixth ramp fix: the feet slip
on it and the leg odometry ran 0.6 m ahead in three seconds (blue103), so the
shove set off 0.7 m early on a stale sighting and the box-gone test, gated on
the reckoned distance to the box, fired with the box untouched. The leg odometry itself is not the platform's
problem: a crab strafing on the spot on a 12 cm deck and on the flat
(`probe_strafe.py`) reads the same, the right way and 18 percent short with the
race's `LEG_SIDE_SCALE` of 0.79 (0.78 m read as 0.64), so a 0.3 m centring
sidestep is 6 cm short and the kerb fix has that to take up.

The forest is where the pose has to be right: the pillar faces are 0.85 m
apart and the crab 0.5 m wide. Coming off the stairs the pose is 0.3 m out
and the map's pillars with it, so the path the planner drew between them, safe
on the map, walked the crab into the pillar it could see in the readout
(Andrew: "a disconnect between what the robot sees and where the pathfinder
thinks the robot is"). Three things, 2026-10-23: the pillars are known to the
map at their true places (`_known_lane`, like the kerbs), so the planner
threads the real pillars and the sightings only place the crab; the first
row's fix places the crab across the lane as well as along it, each pillar at
its known y; and the crab stands at the forest's door, up to 2 s, until that
fix has had the row in view for 0.6 s, before it goes on. The running
landmark fix is at gain 0.5 and 8 cm a frame (0.3 and 5 before). With the pose
right to 5 cm the crab still walked into a centre pillar twice a take (blue84,
red63): the planner returned no path and the fallback walked straight at the
goal. For a while the forest was not planned at all but walked on a route of
known gaps, and that was wrong (Andrew: "the weaving through the pillars needs
to be with our path planner; we can't just script a path through, that
violates the whole spirit of this test"). It is the planner again: A* on the
map, replanned four times a second, the walk steering at the point 0.35 m
along the path at 0.4 stick. The hard clearance is 0.27 m, and a soft one
makes cells within 0.55 m of an obstacle cost up to three steps more, so the
path keeps to a gap's middle: from the forest's door it stays 0.33 to 0.34 m
from every pillar face (`probe_forest_plan.py`, the map as the course knows
it). The empty paths were not the known map's doing -- on it A* finds the
forest at 0.33 as well -- but the live map's, the crab standing inside an
inflation or sensed returns added; so with no way at 0.27 the planner tries
0.20 and 0.12, and with none at all the crab stands and plans again instead
of walking blind.

Then the real cause, found from the planner's own map saved at a "no way"
(`COURSE_TRACE_DIR`): the known kerbs and pillars were not where the course
has them. Every pose fix shifts the map's grid so that what was sensed from
the drifted pose moves with the body, and the mapper shifted the known cells
too, though they are facts about the world, not sightings. Over the hill's
ramp fixes and the kerb fixes the first row of pillars had walked 0.86 m along
the lane and the kerbs 0.2 m across it, and the drifted pillars walled the
lane off at the second row (red211: the kerbs at -1.3 and +0.9). This is very
likely what put blue84 and red63 into a pillar on the planner before the
scripted route. The mapper now leaves known cells in place on a shift when
asked (`known_absolute`, which the course sets; the race's mapper shifts its
known walls the same old way and is left as it was). Where the course knows an
obstacle, sensed returns within 0.35 m of it are not kept either: the mapper's
floor test has no allowance for range, so the returns round a pillar's foot
came out up to 0.25 m short of its face and doubled it (blue101). The path is
followed with the heading held along the lane, strafing while the next point
is more than 30 degrees off the nose and walking at 0.4 with a proportional
strafe otherwise: turned to face each point the crab stood 30 to 42 degrees
across the lane, 0.5 m wide, and caught the staggered pillars. A swing of the
next point to the other side of an obstacle is ignored for 2 s or until the old
point is reached: square in front of the centre pillar, its two gaps cost the
same and the path flipped between them every replan (red102).

Since 2026-10-23 the path is followed facing along it instead (Andrew: "the
crab should turn a little and walk in the direction of the path ... not just
strafe side to side"; strafing, the rear legs swept into pillars already
passed). The heading is the way to the point 0.6 m along the plan, held within
20 degrees of the lane and turned while walking at 0.5; what the heading does
not take up is strafed. With the same seeds the reds crossed in 18 s with no
stall (22 to 25 s before), the blues in 18 to 21 s. A cap of 35 degrees and a
heading held near pillars were tried as well, but only while the floor fit
below was still deciding what is a thing, so their stalls say little.

The readout's floor colour: the sight's floor plane is the IMU's up through the
lowest foot, and a zone counted as floor up to 3 cm plus 2.5 cm per metre above
it, an allowance for that plane's tilt. It left the lower half of a 24 cm ball
and the feet of the pillars grey (Andrew: "it only highlights half a ball",
after the leap, where a hung landing tilts the plane further). On the course
the floor is now found in the picture itself every frame: a plane fitted to the
zones within 1.6 m that the IMU's plane calls floor, refitted without the ones
more than 1.5 cm off, and a zone above it by 3 cm plus 0.5 cm per metre is not
floor. A fit tilted past 5 degrees or shifted past 5 cm is another level and
the old band applies. The fit is weighted by range and its band grows 0.8 cm
per metre: in the forest the pillars hide the far floor, and a plane fitted to
the floor within 1 m was 1 degree off and put the floor 5 cm up at 2.5 m.

The fitted floor draws what was found, it does not find it. Clusters are found
on the old band, and each one's zones (and the readout's colours) then grow
into the zones above the fitted plane that continue it in depth. Finding them on
the fitted floor, at 1.6 m and at 3 m of fit, gave the forest's stall test two
to five calls in each red take where it had none with the same seeds and the
old band, and red212's pose ran 0.86 m behind with the pillar landmarks matched
differently; the pilot was tuned on the old band and stays on it.

## Takes

| take | crab | pilot | result |
|---|---|---|---|
| truth1 | red | truth | climbed the first flight raised, went over at the bottom of the descending flight (one of one; the flight walks down 8 of 8 in the probe) |
| sense1 | red | senses | climbed, pose 2.8 m ahead by the forest, walked into the pillars |
| sense2, sense3, truth2 | red | | jammed on the first flight (3 cm risers, raised) or stood on the platform (walk + turn) |
| truth3 | red | truth | the whole course: 36.6 s, splits stairs 1.8 / ramp 9.0 / forest 15.7 / ball 29.9 / finish 34.6 |
| sense4 | red | senses | the whole course: 33.0 s, splits 1.8 / 9.7 / 17.3 / 25.7 / 30.9; pose within 0.05 m at the finish; the dance faced the camera |
| blue1 | blue | senses | the whole course: 43.6 s (6 s against a pillar in the forest); no dance -- the cue named red |
| red2 | red | senses | jammed on the first riser (2.5 cm) for the whole take |
| truth4-13, blue3-7 | | | the pit, one failure at a time: see "the pit" above |
| truthH0.10, truth15 | red | truth | the whole course with the pit: 88.0 s and 70.7 s, both blocks flat, the bridge built at 54 s |
| blue8 | blue | senses | jammed 35 s on a pillar at 0.3 m, below the stall test's floor; reckoned 2.8 m of walking there and "shoved" blocks 2.6 m away |
| **red4** | red | senses | **the whole course, 1080p with the wide camera: 61.5 s** -- splits stairs 1.8 / ramp 9.9 / forest 17.7 / pit 29.0 / ball 56.1 / finish 61.5; both blocks flat, the bridge built at 51.9 s, the ball over the finish line at 60.3 s; the dance at the gate's post (the cue comes at the line, and the pilot answered it there) |
| red5, red6 | red | senses | the whole course, 71.0 and 65.5 s, the dance 1.8 m past the line; the ball missed both times: the crab reached it turned 60 degrees and walked past (the shove now lines up first) |
| blue9, blue10 | blue | senses | block 1 in (blue10, the first sensed drop); block 2: lined up on the neighbour's edge by a sighting (blue9), backed into it and shoved it the wrong way (blue10), reckoned itself 2 m back while stuck reversing at the edge |
| truth14-23, blue11-18 | | | the trench redesigned as the raised section, one box, the leap: see above |
| red8 | red | senses | 1080p: the box tipped at the edge and hung; the nose test took its raised rear (15-25 cm) for something other than the box, called it "in the pit" and stopped the nudge, three times; the reckoning then ran ahead to the finish while the crab stood |
| **blue19** | blue | senses | **the whole new course on its senses: 67.3 s** -- splits stairs 1.8 / ramp 12.6 / forest 20.6 / pit 46.1 / leap 54.7 / ball 62.6 / finish 67.3; the box flat in the notch at 50.9 s, the bridge crossed with no stall, the ball over the line at 66.6 s, the wave; at the drop it did not see the edge (the threshold sat at the ground's depth) and walked off at half stick, landing on its feet by luck |
| **truth24** | red | truth | **the whole new course: 60.4 s** -- splits stairs 3.8 / ramp 11.4 / forest 21.5 / pit 33.4 / leap 44.2 / ball 55.3 / finish 60.4; the box flat in the notch at 38.3 s, the bridge crossed with one stall, the hop 25 cm from the seen edge at 46.7 s, down on its feet at 54.8 s, the ball over the line at 59.5 s |
| truth25-27, blue20-46, red9-31 | | | the bridge block and the leap rebuilt, 2026-10-22 evening: the box 10 cm, its rear chamfered (claws under the bevel: blue25, red10), then rebated and rear-heavy (walked off sideways: red11-red17), then lane-wide (flat and square every take since blue34); the leap on the seen edge from the lowest body point (lost at the brink: blue26), projected along the ray (saturates at 0.3: blue34-red18), from a mark placed at mid range (odometry drift: blue37, red21), then the nose-down creep to the 0.34 reading and 0.16 m on (red28, blue45: down on their feet from 14 cm). **red26 55.9 s, blue43 60.3 s, blue44, red28 56.2 s, blue45 62.8 s, red30 57.7 s, red31 67.0 s: the whole course on their senses.** The ball shoved blind under 0.6 m (out of "big"): red23, red24, red28 passed it |
| red32, blue47, blue48 | | senses | 1080p: a 2 s time floor on the creep's last stretch, set for a 6 cm/s creep that makes 15, walked two off the edge; blue47 turned against the last pillar 40 s (the stall test watched only the walk: a turn stall now) |
| **blue49** | blue | senses | **1080p with the wide camera, the whole course: 54.3 s** -- splits stairs 1.7 / ramp 11.7 / forest 19.8 / pit 32.4 / leap 41.9 / ball 49.3 / finish 54.3, the hop from 14 cm down on its feet in 2.9 s, the ball over the line, the wave. `out/course/blue49.mp4` and `blue49-wide.mp4` |
| blue50, red33-35 | | senses | 1080p: blue50 finished in 74.5 s after 16 s hung on the edge (a second hop now follows the wiggle); the reds' shoves ended with the box propped 2.5 cm on the claws and the leg odometry 0.6 m ahead, the pit stage ended on the box, and the brink was declared from the reckoning's fallback 0.8 m early -- red34 (61.1 s) and red35 (67.8 s) hopped on the platform, hung at the real edge, got down and finished; red33 fell. The brink now comes from a reading only, and the crab looks for the edge whatever the reckoning says |
| red36, red37, red38 | red | senses | the whole course, traced, after the brink fix: 58.1, **52.4** and 52.2 s, the brink at 15, 14 and 14 cm from the edge, every hop down on its feet, the ball over the line, the pump |
| **red39**, red40, red41 | red | senses | 1080p with the wide camera, the brink from a reading only: **red39 the whole course in 60.0 s** -- splits stairs 1.7 / ramp 12.4 / forest 20.7 / pit 39.1 / leap 48.1 / ball 55.1 / finish 60.0, the brink at 17 cm, down on its feet in 2.8 s, the ball over the line, the pump (`out/course/red39.mp4`, `red39-wide.mp4`); red41 the same in 62.2 s with two more stalls; red40 finished in 56.5 s but passed the ball |
| seeded takes, 2026-10-23 | | senses | `COURSE_SEED` makes a take reproducible; after Andrew's review ("fix the issues before brute forcing videos") the sight's ball-as-wall and slope-as-thing faults, the slower walk, the shove's line-up, the slow creep to a 0.14-0.16 brink, the ball followed over the line, the forest's door stand and route through the known gaps, and the walk past the line by time. **blue90 (seed 28): 69.8 s** and **red66 (seed 31): 70.9 s**, the forest with no stall, the box square, the hop down in 2.5 s, the dance: rendered at 3840x2160 with the wide camera as `out/course/final/blue-seed28-final.mp4` and `red-seed31-final.mp4` (and `-wide`) |

## The readout's size

The readout (`hud.py`) was laid out in pixels for a 720p frame, and at
3840x2160 it was a quarter the size it should be (Andrew: "the timer and the
depth and map displays are super tiny ... it is important people see the depth
camera"). It now scales with the frame's height, 720p being one, and is laid out
afresh rather than magnified (Andrew: "the map and depth camera are not
aligned vertically and the L and R controls are just out on their own ...
remove the legend and the map and plan text ... are there additional sensor
readouts or the joint positions that we can add"): two lines of text, then one
row with the depth camera and the map at one height and no captions, and a
column of instruments at its right -- the IMU's horizon with pitch and roll,
the two stick dials, and the legs' joints as bars about the standing pose,
half a radian filling a bar. The clock, in 44 px type with the section and the
splits under it, sits in its own box at the top right, the panel's mirror.
Text is drawn at the scaled size; the depth image and the map are enlarged by
whole pixels, their zones being blocky by nature. `Hud(..., scale=)` overrides
the scale. A seeded
take re-rendered with the bigger readout is the same take: the readout is
painted on the frame after the fact and touches nothing in the physics. The
camera's distance is not so innocent: it feeds the renderer's extent, which the
depth sensor's rendering shares, and the same seeds ran about a second apart
at 1.6 m and at 2.4 m, both clean; a seed reproduces a take with the same
camera settings.

The main camera follows the crab from 1.6 m (2.4 before: Andrew, "zoom the
camera in a little bit more", then "a tiny bit more") and looks at a point
0.30 m ahead of and 0.18 m above the crab's centre (`--video-lookat DX:DY:DZ`,
`COURSE_CAM_LOOK`), which puts the crab left of the frame's centre and low,
under the readout's bottom edge and clear of it, with the way ahead in view
(Andrew: "it is good to see a little bit more in front of the crab"). MuJoCo's tracking camera looks at the body's centre of mass and
nowhere else, so the recorder's following camera is made free and pointed at
the body plus the offset every frame (`rl/mjrl/viewer/video.py`,
`lookat_offset`); the wide camera, which has no offset, is unchanged.

## Seeded takes

`COURSE_SEED=<n>` seeds Python, NumPy and torch and the environment before it is
built, and two takes with one seed then run identically, step for step (seed 7,
22 s, two runs: the pose logs identical to the last digit). The only randomness
in a take is the environment's own, its reset and its sensors' noise, drawn from
torch's generator; the pilots draw nothing. The match holds across the video's
size and cameras (seed 7 at 720p against 1920x1080 with the wide camera, both
with the readout: identical) but not between a take with video and one without
(`--no-hud`, no `--video`: the same seed diverged from the third second), so a
take is found at 720p with the readout on (`scan.sh <crab> <seeds>` in the
session's scratchpad runs seeds three at a time and tabulates finish, ball,
brink, hang, stalls), and the good one is run again at the size and with the
cameras the film wants, the same take frame for frame; 3840x2160 renders at
about a twentieth of real time. An upscale of a 720p take "doesn't cut it" for
publishing (Andrew), and the runner records no state to replay. A seed is also
only good for one world: adding the two side balls changed every take from the
third second (seeds 28 and 31, the one-ball finals, against the same seeds with
three balls -- bodies in contact anywhere are in the same constraint system,
and the solver's answer for the crab moves by a rounding error that the gait
then amplifies), so a change to the track means a new scan, never a replay of
the old seed's first half.

