# Calligraphy — Jumper writes a Chinese character on the floor

Jumper holds a brush in its carried claw and writes a character on the floor, stroke
by stroke, the way 地书 (water calligraphy on paving stones) is written: walking to
where the next stroke can be reached, putting the brush down, writing, lifting it.

```bash
python tasks/jumper/calligraphy/tools/strokes.py                # step 1: 无 -> plan.json + plan.svg
python tasks/jumper/calligraphy/tools/write.py --plan-only      # the cut into stretches, drawn
python tasks/jumper/calligraphy/tools/write.py                  # step 2: write it, ~5 min on CPU
MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u65e0/<run>
                                                                # step 4: two shots, gif, ink export
```

Outputs go to `logs/calligraphy/u65e0/<run>/` (the character's code point):

| file | step | what it is |
|---|---|---|
| `plan.json`, `stretches.json` | 1, 2 | the strokes on the floor, and how they were cut and written |
| `log.npz` | 3 | one row per 20 ms control step -- phase, stroke, target and measured tip, the ink point and width, trunk pose, command -- and the full `qpos` |
| `model.mjb` | 3 | the compiled model that `qpos` belongs to |
| `topview.png` | 2 | the plan beside where the brush touched the floor |
| `top.mp4`, `low.mp4` | 4 | overhead and low shots, real time, 960x720 at 30 fps, ink drawn as it is laid (`--no-ink`: clean plates) |
| `wu.gif` | 4 | the overhead shot at 6x, 480x360, for the README |
| `ink.json`, `ink.svg` | 3, 4 | the ink, for the compositor: see below |

> **Status (2026-10-09): 无 at 12.2 cm, every stroke whole, the feet still.** The
> brush is a thick handle the claw closes onto and a black cone of hair that
> sinks into the floor. five_foot's shipped policy walks; while the arm is out its
> last action is held, so the legs stand like a statue -- see
> [Where it stands](#where-it-stands).

**Two things live here.** The tools write a character with a policy; the task
`jumper.calligraphy` trains that policy -- see [Training the fix](#training-the-fix).
With the shipped policy nothing is trained: `jumper.five_foot`'s shipped policy
(`tasks/jumper/five_foot/out/example/model_86600.pt`) walks and stands on five legs,
and everything here sits on top of it -- the arm that policy leaves out of its action
is driven by inverse kinematics, and its velocity command by a steering loop. There
is no `env_cfg.py` or `rl_cfg.py`, so the registry does not see the directory.

## A longer film: 跳跳

Jumper's name in Chinese, 跳跳 (*tiàotiào*), top to bottom as 地书 is usually
written, then the outro: the robot walks to the text's side, turns to it, dances
(the body twisting over its feet, a roll, a bow) and stands while the camera pulls
back to the whole text.

```bash
python tasks/jumper/calligraphy/tools/write.py --text 跳跳 --palm-x-min 0.10 --out logs/calligraphy/u8df3-u8df3/film
MUJOCO_GL=egl python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u8df3-u8df3/film
```

- `--text` takes any characters with stroke data in `data/` (`tools/strokes.py
  --import`); `--layout horizontal` writes them left to right.
- `--scene daylight` (the default) puts the repository's daylight sky behind the
  low shots; `--no-outro` stops at the last stroke.
- `film.mp4` is the shot for a video: real time, following the character being
  written, then the reveal and five seconds on the result. `result.png` is the
  result from above at 1920 x 1080.

Measured (2026-10-10, native:cpu, five_foot's shipped policy, the legs held): 跳跳
at 12.0 cm a character, 26 strokes, one seam, the ink 1.0 mm from the stroke
(median, p95 4.5), 232 s simulated with the outro; 无 in the same build at 12.0
cm, no seam, p95 2.9. Writing a text this long found four ways for the arm to go
through the trunk that 无 never had, each fixed where it arose:

- a writing pose in the trunk (the band now rejects one, `Arm.writable`);
- the lift before a fold raising the elbow into it (`clear_high`);
- the lift at a stroke's end aiming the brush's axis back towards it (it now goes
  straight up);
- a solve that still leaves the arm in it (`Arm.ik_clear` turns the elbow out).

What is left is the upper arm's hull grazing the trunk's by up to 4 mm for
0.2-0.4 s, where the meshes do not touch. And after a seam the arm once jumped to
another pose, swinging 145 deg in 0.1 s into the floor: hover moves now fold
instead when the line to the next start needs another pose, and no arm command
moves more than 4 deg a step.

## What is where

| file | what it does |
|---|---|
| `hanzi.py` | a character's medians (Make Me a Hanzi) -> stroke trajectories on the floor, with a press profile |
| `data/` | one verbatim `graphics.txt` entry per character, and the Arphic Public License they are under |
| `brush.py` | the brush: a 13 mm handle the claw closes onto, a black cone of hair that may sink into the floor; the ink's width is the cone's section there |
| `sim.py` | the environment, policy and the two commands taken over from the operator, shared by the tools |
| `tools/stability.py` | where in the reach band the policy holds the trunk still with the arm out |
| `env_cfg.py`, `rl_cfg.py` | the task `jumper.calligraphy`: five_foot's config with the brush, the writing command and `hold_position` |
| `mdp/writing.py` | the command that takes the arm out to write during training, and the reward for holding still |
| `tools/writing_poses.py`, `data/writing_poses.npz` | the arm poses the training writes with (1028 cells, 257 cm^2, palm >= 0.10 m) |
| `arm.py` | IK for the brush tip against the trunk's measured pose; the band of floor it can write on; floor-safe fold paths |
| `stations.py` | cuts each stroke into stretches the arm can write from one place, and where to stand for each |
| `tools/strokes.py` | step 1 |
| `tools/write.py` | step 2: the controller, the log and a top view |
| `ink.py` | step 3: which logged steps are ink, grouped into marks, with a width |
| `cameras.py` | the two shots, one definition for the renderer and the ink's pixels |
| `tools/render.py` | step 4: replays the log in plain MuJoCo and renders it |

## Frame

The character's up is world +x and its right is world -y: the robot faces +x and
writes the way a person writes on a sheet in front of them. Seen from above with +x
at the top it reads the right way round. Make Me a Hanzi's medians are in a 1024-unit
em square with **y up** -- the SVG on its site flips them, and copying that flip
writes every character upside down. See `hanzi.py`.

## What was measured, and what it decided

All on a 4-core Linux container, `native:cpu`, MuJoCo 3.11.0, 2026-10-08.

**The policy stands with the arm out only in front.** 16 arm poses held 8 s each,
no pushes, the tip 15 mm above the floor: with the palm tip >= 0.15 m ahead of the
trunk's centre the trunk drifted 1.4-2.8 cm and turned 0.6-3.8 deg (stowed: 0.4 cm,
2 deg); nearer the trunk, up to 36 cm and 67 deg. No pose fell. So the arm writes
only with the palm tip >= 0.13 m forward (`arm.PALM_X_MIN`).

**The writable floor is a band, not a square.** With that limit and the brush, the
tip can be put on the floor and 15 mm above it over 536 cm², shaped as an arc about
the shoulder; the largest square inside it is 6.5 cm. So `stations.py` fits each
stretch of stroke into the band, at one of five headings, rather than into a box. 无
at 0.30 m comes out as 5 stretches -- strokes 1, 2 and 3 whole.

**It does not walk with the arm out.** Commanded 0.05/0.08/0.12 m/s forward with the
arm held at a writing pose it moved 0.001-0.002 m/s; stowed, 0.027/0.059/0.109. So
the arm folds between stretches, along a path searched to keep it off the floor
(interpolating the four joints at once swept the brush through the floor on a
third of the steps).

**Pressing by depth pushes the robot around.** A fixed 3 mm press gave 0-6 N, and on
the long middle stroke turned the trunk 48 deg while the brush bounced. The press
is held at `press * 1.0 N` by an integrator on the tip's height instead.

**Result, 无 at 0.30 m** (`tools/write.py`, defaults). The robot starts each run
from a randomised pose, so runs differ; two of them:

| | simulated | stretches | tip to stroke, median / p95 / max | brush down while writing | force, median |
|---|---|---|---|---|---|
| run 6 | 86 s | 7 | 1.1 / 2.9 / 5.8 mm | 96% | 1.13 N |
| run 7 | 103 s | 8 | 1.2 / 6.4 / 17.4 mm | 90% | 1.20 N |

In both the brush touched the floor nowhere outside the strokes. Each extra
stretch is a replan: the trunk stopped too far from its station (run 7: three
times) or drifted out of reach while writing (once).

## The ink, for post-production

`ink.json` holds every **mark** -- one continuous trace of the brush -- with, per
point: the time it was laid, its floor xy, the depth, the planned press and
a suggested width; and its **pixel position in each shot**, with each shot's
intrinsics and pose. So a compositor can lay the ink down frame by frame on the
clean plates (`render.py --no-ink`) without knowing anything about the simulation.

- The hair counts as ink only while the arm is lowering, writing or lifting, and
  only where the cone is below the floor; the ink is centred where its axis meets
  the floor and is as wide as its section there (`brush.ink_point`,
  `brush.section_width`). Gaps of up to 0.1 s inside a stretch are bridged.
- The depth and the planned press are there for any other width rule.
- The pixels are checked against the renderer (`render.py --check`: markers at
  known floor points): <= 0.5 px overhead, 1.4-2.1 px in the low shot.

The renderer is plain MuJoCo on the run's `model.mjb` and `qpos` -- no mjlab, no
torch -- because OSMesa, the CPU renderer this needs without a GPU, crashes the
process when torch is loaded beside it. It repaints the training floor as paving
stones, draws the ink as flat wet patches and lights the scene with a low sun
(the scene's spot light leaves its shadow map's edge on the far floor). It is
slow on a CPU: run 7 (103 s) took 78 min on this 4-core container for both shots
and the gif -- 6614 frames, ~0.7 s each, OSMesa on all four cores.

`docs/media/calligraphy-wu.gif` in the repository's README is `wu.gif` of a run
with the legs held, the claw closed onto the brush and the arm folding around
the robot (2026-10-10, rendered on an RTX 3090 with EGL).

## Not done yet

- The policy was never trained with the arm out or with a load on the floor.
  Training `five_foot` with the arm sampled over the writing band would widen the
  band and let it walk while writing (option B).

## Where it stands

**The legs are held while the arm writes** (`write.py --hold-legs`, the default).
Every policy -- five_foot's and three rounds of `jumper.calligraphy` -- reacts to
the arm unfolding by dragging its feet along the floor, 10-124 mm per foot with
no lift-off; held, the walking policy's last action moves them 0.0-2.3 mm. Three
runs of 无 from three random starts, five_foot's shipped policy only (2026-10-09,
native:cpu, 4-core container):

| | size | seams | ink centre to stroke, median / p95 / max | trunk while writing | feet: unfold / write / fold |
|---|---|---|---|---|---|
| hold11 | 13.6 cm | 0 | 0.7 / 4.2 / 6.8 mm | <= 0.1 mm | <= 1.8 / 0.2 / 0.3 mm |
| hold12 | 13.6 cm | 0 | 0.7 / 4.6 / 6.8 mm | <= 0.1 mm | <= 2.1 / 0.1 / 0.2 mm |
| hold13 | 13.6 cm | 0 | 0.7 / 4.2 / 7.7 mm | <= 0.1 mm | <= 2.3 / 0.2 / 1.0 mm |

```bash
python tasks/jumper/calligraphy/tools/write.py --palm-x-min 0.10
```

- Held, the stance's height is the walking policy's, 101-109 mm, so the band is
  where the arm reaches at both 100 and 110 mm (`HELD_Z`); built at one height,
  the other left the ends of strokes 3 and 4 out of reach.
- The brush goes in to 1/1.7 of the width it is to leave (`WIDTH_PER_DEPTH`,
  measured), and the head of a stroke is pressed no harder than its body
  (`hanzi.HEAD` 1.0): the arm's lag already makes it 1.5x as wide. The heads were
  21-24 mm round blots before; now 12.6-13.1 mm, the hair at most 7.5 mm deep.
- The arm unfolds and folds around the robot, not through it. The model gives the
  carried arm no contacts with the trunk, the other legs or itself, and the fold
  path was chosen for floor clearance alone: the first unfold of every run swept
  the claw across the top of the trunk and through RF's claw. `arm.fold_path` now
  takes, of the paths 5 mm off the floor, the one whose claw and brush come within
  5 mm of the trunk and the other legs least, then of the arm's own upper arm
  (whose convex hull overlaps the claw's near the elbow where the meshes do not).
  Two runs: neither touches the trunk or another leg (self2-self3). The forearm
  is checked too, and so is the point the brush is lifted to over a stroke's start:
  near the trunk, lifting it 45 mm raised the elbow into it (grip3).
- The finger closes onto the handle, not through it (`brush.FINGER_HOLD`, -0.025;
  shut, its tip went 6.4 mm into the handle). Opened, the finger hangs lower and
  meets the floor in more writing poses: 无 is 12.2 cm, from 13.6. Two runs
  (grip5-grip6): 0 seams, ink p95 2.8 mm, nothing of the arm or brush in the
  trunk or another leg.
- With the brush down, the IK never takes a solution on the arm's other branch:
  with LF_J3 at its limit it once did, and swung the hair 83 mm up and 100 mm off
  the stroke mid-stroke.

What follows is how it got here: the fine-tuned policies, which held the trunk
but dragged the feet.

**Two policies, one each for what they do well.** With the fine-tuned checkpoint
(`out/model_89599.pt`, 3000 iterations from five_foot's on an RTX 3090) writing and
the shipped five_foot one walking (`write.py --walk-checkpoint`, the default), three
runs of 无 from three random starts (2026-10-09, native:cpu, 4-core container):

| | strokes / stretches | seams | ink centre to stroke, median / p95 / max | trunk while writing | hair in the floor elsewhere |
|---|---|---|---|---|---|
| ft2 | 4 / 4 | 0 | 0.7 / 3.6 / 6.0 mm | 0.3-2.0 mm, <= 0.2 deg | 4 steps |
| ft3 | 4 / 4 | 0 | 0.7 / 3.8 / 5.9 mm | 0.8-1.4 mm, <= 0.5 deg | 3 steps |
| ft4 | 4 / 4 | 0 | 0.7 / 3.7 / 5.4 mm | 0.3-1.3 mm, <= 0.3 deg | 0 steps |

```bash
python tasks/jumper/calligraphy/tools/stability.py --checkpoint tasks/jumper/calligraphy/out/model_89599.pt --palm-x-min 0.10
python tasks/jumper/calligraphy/tools/write.py --checkpoint tasks/jumper/calligraphy/out/model_89599.pt --palm-x-min 0.10
```

- **The fine-tuned policy holds still.** It holds the trunk still at 61 of 62 points
  of the training band (shipped: 44 of 57 of a smaller one), moves it 3.6-5.7 mm
  as the arm unfolds (shipped: ~20 mm, and 10 mm up) and 0-2 mm while the arm writes.
- **It does not walk slowly.** It stands for any command under ~0.15 m/s: a push of
  0.07 m/s for 0.12-1.5 s moved it 0.3-2.1 mm (shipped: 4-100 mm), and write.py's
  closed-loop walks at 0.08 m/s did not arrive. With it walking, the trunk never
  got where a stroke was in reach and strokes were cut into one-sample pieces.
  `hold_position` pays standing still and nothing in this task pays a slow walk,
  so that is what it learned. Hence the second policy: both are the same stateless
  MLP, so write.py switches between them from one step to the next.
- **The unfold is aimed off.** Switching to the writing policy as the arm unfolds
  moves the trunk the same way every time -- +7 to +22 mm forward, 0 to 19 mm
  right, 0 to 4 deg clockwise, and 2 cm up (18 unfolds in ft2-ft4). Relearned per
  stroke, it cost each stroke a fold and four walks (无's last: 31 s for 3 s of
  writing, 90-100 s in all). Carried from stroke to stroke as a running estimate
  that every walk aims short by, 无 takes 67-73 s and only the first stroke is ever
  refolded (ft5-ft7: 0 seams, ink p95 3.6-3.8 mm, trunk 0.2-1.5 mm while writing).
- **Walks are approaches.** The shipped policy stops 2.2-6.6 mm from a goal 4-8 cm
  away and 5.8-17.4 mm from one 1 cm away (16 walks, V_MIN 0.15), so a short
  correction backs off to 4 cm first and comes in again.

Measured before the fine-tune, on the shipped policy alone (2026-10-08, native:cpu,
MuJoCo 3.11.0):

- **The trunk holds still while the arm writes** -- 2.5-7.4 mm and under 1.3 deg
  through strokes 1 and 2 (run 15) -- once the brush no longer pushes on the floor
  and the trunk is not twisted. Twisting it (the body-pose command) to bring a
  stroke into reach made it stumble mid-stroke: 73 mm, 13 deg (run 14).
- **The policy does not hold still everywhere in the band.** `tools/stability.py`
  holds the arm at 57 points of the band for 6 s each: 44 stable, 13 not (drift
  over 15 mm, a turn over 4 deg or the trunk off its height by 10 mm), mostly at the
  band's two ends. The stable band is 177 of 226 cm^2.
- **Placing the trunk to a few millimetres is what fails.** A stroke written whole
  must lie inside the band from where the trunk stands, and the trunk ends up 5-10
  mm from where it was sent: the walk stops within 5 mm, and the last step and the
  unfolding arm move it again. With a 10 mm margin the character is 10.6 cm, and
  strokes 3 and 4 still come out in two pieces about half the time (runs 13-16);
  at that size the 28 mm cone is too large for the strokes.

The policy was never trained to stand still with the arm out, nor to step a few
millimetres on command. Fine-tuning `jumper.five_foot` for exactly that -- the arm
swept over the writing band while standing, precise low-speed positioning -- is
the change that would let the character grow back to 20-30 cm with every stroke
whole.

## Training the fix

`jumper.calligraphy` is `jumper.five_foot` fine-tuned from its shipped checkpoint
so that it stands still while the arm writes. What it adds, and why each number
is what it is, is in `env_cfg.py`'s docstring; in short: about half of every
episode the arm is out -- unfolded over 2-4 s, moved between floor poses 2-8 cm
apart at 2-6 cm/s, hovering, touching or sunk, then folded -- with the robot told
to stand level, and `hold_position` rewards staying within ~10 mm and ~3 deg of
where the trunk stood when it became still.

On a machine with an NVIDIA GPU (warp is chosen by default):

```bash
python scripts/train.py --task jumper.calligraphy --headless \
    --checkpoint tasks/jumper/five_foot/out/example/model_86600.pt
```

It resumes at five_foot's iteration 86600 with its curriculum levels, and runs
3000 iterations (`--max-iterations` to change). What to watch in TensorBoard:

| curve | should |
|---|---|
| `Episode_Reward/hold_position` | rise -- the trunk holding still with the arm out |
| `Metrics/writing/hold_drift` | fall towards a few millimetres |
| `Episode_Reward/track_linear_velocity`, `track_angular_velocity` | stay where five_foot had them: walking must not be forgotten |
| `Episode_Termination/fell_over`, `too_low` | stay near 0 |

Smoke-tested on this container's CPU (native, 64 robots, 20 iterations,
2026-10-08): the checkpoint loads strictly, the arm is out for half the robots,
no robot fell, `hold_drift` 8-12 mm at the start.

Then, with the trained checkpoint (`logs/<model>/jumper.calligraphy/<run>/model_<n>.pt`):

```bash
# where the new policy holds still, over the wider training band (~3 min)
python tasks/jumper/calligraphy/tools/stability.py --checkpoint <ckpt> --palm-x-min 0.10
# write 无 with it: the size is fitted to the band it now holds still over
python tasks/jumper/calligraphy/tools/write.py --checkpoint <ckpt> --palm-x-min 0.10
# render (on a GPU machine MUJOCO_GL=egl is much faster than osmesa)
MUJOCO_GL=egl python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u65e0/<run>
```

`write.py` prints the fitted size, the trunk's drift through every stroke and any
seam; a seam or a drift over ~5 mm is what the training has not fixed yet.

### Feet that do not slide

In the low shot the feet slide along the floor as the arm unfolds -- every policy
so far, 14-124 mm per foot, never lifting one. The second round (`model_92598`)
raised `feet_still` and did not change it, and `env_cfg.py` ("The second round")
says why: that term charges a step as much as a drag, so dragging stays the cheap
way at any weight. The third round charges the drag alone (`feet_slide`); train it
from the second:

```bash
python scripts/train.py --task jumper.calligraphy --headless \
    --checkpoint tasks/jumper/calligraphy/out/model_92598.pt
```

Watch `Episode_Reward/feet_slide` rise towards 0 while `hold_position` and
`Metrics/writing/hold_drift` stay where they were. The second round's other change
(`feet_planted` at 0.05, to make it walk slowly) did nothing and is undone; the
policy still stands for commands under ~0.15 m/s, so `write.py` still walks with
five_foot's.
