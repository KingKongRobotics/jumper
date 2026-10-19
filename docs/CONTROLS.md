# Controls — the pad, the keyboard, and who decides what they mean

How a person drives this robot: what the controller reads from a gamepad and a keyboard, how a
task says what its sticks and keys do, and how a bundle says which control switches between
modes. This is the design and the reasoning behind it, with the files that implement each
part. For the procedure — writing a `controls.yaml`, adding a mode switch — the
[`controls` skill](../.claude/skills/controls/SKILL.md) is the checklist; for the bundle's
file format, [`deploy/BUNDLE.md`](../deploy/BUNDLE.md).

---

## 1. The design in five rules

1. **The words are fixed; the meanings are not.** One file,
   [`controller/vocabulary.json`](../controller/vocabulary.json), lists everything a pad or a
   keyboard can say — `A` is down, the left stick is at (0.4, −0.9), `key_w` is down — and
   nothing about what any of it means. The meanings live with whoever consumes the input, in
   exactly two places (rule 2). A name that is not in the dictionary is refused everywhere, at
   load, because an invented name parses, binds nothing, and looks exactly like a control that
   does not work.

2. **Two files, and their sum is the whole.** What a stick or a key does *while a task is
   driving* is that task's [`controls.yaml`](#4-a-tasks-controls-controlsyaml). Which control
   *switches into* a mode is the bundle's entry in
   [`deploy/manifests.json`](#5-switching-modes-in-a-bundle). Neither may say the other's half: a
   task does not name a mode, a manifest does not bind a stick to a command axis.

3. **The pad and the keyboard are two paths.** Each binds its own inputs straight to what they
   do — one direction of a command axis, a control the task keeps, a mode switch, the release —
   and neither names a control of the other, so the two can disagree on purpose: Space is the
   jump from walking and the claw in a claw mode. They agree where they have to because a task's
   file names the same axes for both, and a keyboard that leaves an axis off has to say why.
   Until 2026-09-29 the keyboard was a *virtual pad* — a key held pushed a stick of a pad nobody
   held, read through the real pad's mapping — which made its layout a copy of the pad's. That
   is not the layout a hand on a keyboard wants: Control-agent 3.1, the robot's own operator
   guide, turns on J and L and twists on H and ;, where the pad does both on one stick, holds
   the body high and low on N and M as stances, where the pad moves the height and leaves it,
   and holds the claw's arm out on Shift, Alt and Ctrl.

4. **Each mode reads the operator through its own task's controls.** A bundle runs several
   policies, and different policies take different controls: in the `jumper` bundle the right
   stick, pressed down and pushed up, raises the body while walking and tips the nose down in a
   claw mode, the right trigger closes the claw in a claw mode and does nothing while walking,
   and a full stick is 0.8 m/s in one mode and 0.5 m/s in another. The controller gives each
   mode its own operator, built from that mode's contract, and never forces two tasks to agree.

5. **One control, one meaning at a time.** A control that switches modes and is also the
   current mode's control would do two things on one press. Every such overlap is refused, and
   a switch says with `from` which modes it may be pressed in — that is how Space is the jump
   in one mode and the claw in another. The one sharing allowed is a pad chord that leaves the
   mode on the press ([§5.7](#57-a-d-pad-direction-shared-by-a-task-and-a-switch)). And while a
   modifier is held, a control chorded with it answers only the chord: Menu + up is a dance,
   never also what up alone does.

Behind all five is the failure this repository is organised around: **input bugs are silent.**
A sign flipped the wrong way walks the robot backwards while every screen looks correct; a
key bound to nothing looks like a key that did not fire. So every value carries its source,
nothing defaults a sign, and anything ambiguous is refused at load rather than guessed at.

---

## 2. The dictionary

```bash
python -m controller --vocabulary          # on a board with no Python: controller --vocabulary
```

`controller/vocabulary.json` (`kk-control-vocabulary/2`) is read by three programs and written by
none of them:

| reader | how |
|---|---|
| `controller/vocabulary.py` | `python -m controller --vocabulary` prints it |
| `deploy/fsm/src/vocabulary.rs` | compiled in with `include_str!`, so a board carries no file |
| `tasks/jumper/common/mdp/controls.py` | validates every name in a task's `controls.yaml` |

What it lists:

| section | names | notes |
|---|---|---|
| `buttons` | `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3` | the robot's pad service publishes these ten. `view` is listed under `absent`: the pad has it and the service does not publish it. `X` and `Y` are marked *unresolved* — the service and `controller/xbox.py` disagree on which evdev code is which, and nobody has pressed the button to settle it |
| `dpad` | `dpad_up` `dpad_down` `dpad_left` `dpad_right` | the wire carries `dpad_x`/`dpad_y` as −1/0/+1, which no gesture can read, so they are offered as four booleans |
| `axes` | `Lx` `Ly` `Rx` `Ry` `LT` `RT` | sticks in [−1, 1] in evdev's convention — **`Ly` and `Ry` are positive down**, `Lx` and `Rx` positive right — and triggers in [0, 1]. The pad service applies a 0.15 deadband **and rescales**, so full deflection still reaches ±1 |
| `keys` | the numeric keypad; the main row's `key_1` to `key_4`; `key_w s a d i j k l m n o u q b g h`; `key_space`; the four arrows; both Ctrls, both Shifts and both Alts | each carries its GLFW code (a MuJoCo window) and its `KeyboardEvent.code` (a browser) |
| `modifiers` | `ctrl` `shift` `alt` | each is either of its two keys: `ctrl` is down while the left Ctrl or the right one is |
| `gestures` | `rise` `fall` `hold` `toggle` `single` `double` `triple` `quadruple` `quintuple` | [below](#gestures) |

**A keystroke** is how a keyboard binding names what is pressed: a key (`key_j`), a modifier on
its own (`ctrl`), or a modifier held with a key (`shift+key_j`, J while Shift is held). J and
Shift + J are two keystrokes and never both act: while a modifier is down, a key it has a chord
with answers the chord and not itself.

**Why these keys.** MuJoCo's viewer calls the user's key callback *in addition to* its own
shortcuts, and every letter is one of them — `W` flips the scene to wireframe, `G` is fog, `H`
convex hulls, Space pauses. The keypad is the set it leaves alone. The letters are there anyway
because a keyboard laid out for the hands was judged worth the flicker, and the jumper tasks lay
it out as Control-agent 3.1 does (2026-09-29): `W A S D` and the arrows walk, `I K` pitch,
`J L` turn, Shift + `J L` twist, `U O` roll, `N M` move the height, `B` lets go, and in the claw
modes Space closes the claw and Shift, Alt and Ctrl hold the arm out. `Q` and `O` were the
triggers and `M` the right stick's click until that day; `Q` is bound by nothing now. Space,
`G`, `H`, Ctrl and the digits `1` to `4` switch the `jumper` bundle's modes, each on a switch of
its own ([§5.4](#54-the-keyboard-for-switches)); the digits on the main row and the keypad
alike. The main row's `1` to `4` also flip MuJoCo's geom groups; the keypad's flip nothing.
Alt costs a browser something: pressed and let go on its own, Windows moves focus to the
browser's menu, so a page has to `preventDefault` it.

### Gestures

A control used as a switch is a pad button or a key, plus a gesture. Four read one press:

| gesture | fires |
|---|---|
| `rise` | the tick it goes down, once |
| `fall` | the tick it comes up, once |
| `hold` | every tick it is down — a dead-man switch, not a mode |
| `toggle` | on from one press until the next — the default for entering a mode |

Five count presses — **clicks**:

| gesture | fires |
|---|---|
| `single` | one press, and no second inside the click window |
| `double` | two presses, each inside the window of the one before |
| `triple` | three |
| `quadruple` | four |
| `quintuple` | five |

A control may be bound twice with two gestures — press to arm and release to go is `rise` and
`fall` on one button. The click rules are in [§5.3](#53-gestures-latches-and-clicks).

**Both edges of a key.** A browser reports a key's press and its release. MuJoCo's viewer
reports the press alone, so `rl/mjrl/viewer/keys.py` installs a GLFW key callback of its own in
front of the viewer's and hands every reader both edges, dropping auto-repeats; a key is held
from its press to its release on every host. (Until 2026-09-29 the release was inferred from
key repeats that never came, and every key let go 0.75 s after its press.)

---

## 3. How input reaches the controller

The controller — the Rust crate in `deploy/fsm` — is the same code on three hosts, and each
host hands it input its own way. A fourth path, `play --task`, does not run the controller at
all.

| host | the pad | the keyboard | clock |
|---|---|---|---|
| **board** (`runtime/board/controller`) | `RobotControlRaw_ControlRaw` over DDS from the pad service `control-pod-svc`, about 50 Hz, domain 0 — the raw pad, not an interpretation of it (`deploy/fsm/src/dds.rs`) | none: the pad is the only source | the controller's own |
| **browser** (`WebFsm`, `runtime/web/controller.wasm`) | the Gamepad API's raw values: `setAxis`, `setPad`, then `padFrame(nowUs)` | `KeyboardEvent.code` through `setKeyCode(code, down, repeat, nowUs)`; repeats are ignored | the page's `nowUs` |
| **`play --app`** (`runtime/mjlab/<platform>/controller.so`) | an Xbox pad through evdev on Linux or XInput on Windows (`controller/`), sent every step with `set_pad_frame` | the viewer's keys through `mjrl.viewer.keys`, `set_key(name, down, now_us)` | simulation time |
| **`play --task`** (no controller) | the same reader, into the Python operator | the same viewer keys, into the Python operator | wall time |

Three facts hold on every host:

- **Signs are kept as the device reports them.** `controller/xbox.py` does not flip `Ly` — "a
  robot opinion smuggled into a device driver" — and neither does the pad service. The flip
  from stick to command is the `sign` in a task's `controls.yaml` and lives nowhere else.
- **No second deadzone.** Every `controls.yaml` says `deadzone: device_reported_rescaled`: the
  device has applied one and rescaled to ±1 (the robot's service at 0.15, a bench's evdev driver
  at its own `flat`, XInput at Microsoft's documented dead zone), and a consumer that added
  another would narrow what a person can ask for without being able to tell. The Rust operator
  refuses any other value.
- **The operator going away drops everything.** Pad input older than `command_timeout_ms`
  (500 ms in the `jumper` bundle) releases every latched switch, returns every mode's command
  to its rest, and re-arms every edge, so the cascade falls back to its default mode and a
  button still held when input returns is not read as a fresh press.

`play --task` is the training side's replay, not a deployment host. There the command terms
are replaced by operator terms (`tasks/jumper/common/mdp/operator.py`) that read the same
`controls.yaml`, each device through its own bindings. The operator is inactive until touched —
the randomly sampled command passes through — and `B` hands the command back to the sampler.
Training itself never reads a pad: commands are sampled, and `controls.yaml` only decides how
a person reaches them.

On Windows the Python side reads a pad through XInput (`controller/xinput.py`): Xbox pads and
pads that present themselves as one, each reading turned into the values Linux's `xpad` driver
would give, so the signs and the button names are Linux's. On macOS no pad is read and `play`
drives from the keyboard. `python -m controller`, with no flag, streams a connected pad's axes
and buttons with the signs it expects — the bench check for a sign nobody has measured on the
robot.

---

## 4. A task's controls: `controls.yaml`

### 4.1 Who reads it

```
tasks/<task>/controls.yaml
   │  load_controls()  (tasks/jumper/common/mdp/controls.py) — every refusal below
   ├──▶ play --task: the Python operator
   └──▶ scripts/export.py: the contract's `controller` block (layout.json, operator_controller/2)
            └──▶ the controller's operator for that mode, on the board, in a browser, in play --app
```

`velocity_env_cfg` takes the file as `controls=` and raises without it unless the task passes
`operator_command=False` (`jumper.swing` does; `jump`, `ref_free_jump`, the dances and the
gestures are not built on it and are driven by mode switches alone). The exported block carries
the file's sha256 and the shaping in §4.5, and the contract beside it the command ranges the
policy trained on, so a robot never needs the file.

### 4.2 The schema

```yaml
schema_version: 2                 # 1 is refused, by name

command:                          # one entry per command term the operator drives
  - term: twist                   # its key in cfg.commands
    feeds: velocity_commands      # the observation term it lands in
    frame: body
    axes:                         # in the order the term writes them
      - { name: lin_vel_x, unit: m/s, positive: forward }
      - ...
    scaling:
      rule: full_deflection_is_range_edge

task:                             # optional: controls the task answers itself, by name
  <name>: { kind: amount | press, does: <what the task does with it> }

devices:
  gamepad:
    scheme: absolute              # the stick position is the command
    layout: xbox
    axes: { <axis>: <binding> | [<binding>, ...] }
    shift: { button: R3, gesture: hold }        # optional; hold or toggle
    reset: R3                                   # optional: a tap puts every moved axis at rest
    deadzone: device_reported_rescaled
    release_button: B
    task: { <name>: <one pad control> }         # every task control, when `task:` has any
  keyboard:
    scheme: keys
    full_after_s: 2.0             # seconds a held key takes to full deflection
    axes: { <axis>: { "+": [<keystroke>, ...], "-": [<keystroke>, ...] } | { unbound: <why> } }
    release: [key_escape]         # keystrokes that let go of everything
    task: { <name>: [<keystroke>, ...] }        # every task control, when `task:` has any
```

**Scaling.** Full deflection is the edge of the range the policy was **trained** on, each end
scaled separately — a stick means "as fast as this policy was ever asked to go" and cannot ask
for anything off-distribution. An axis whose centred value is not zero names the config field
that holds it, `rest: neutral_height`, and is scaled from there: hands off the pad is the
standing height, not a body on the floor.

**A pad binding** is `{ source, sign, travel, shifted }`:

| field | |
|---|---|
| `source` | `Lx` `Ly` `Rx` `Ry` `LT` `RT` |
| `sign` | `1` or `-1`, required — the one place a direction is decided |
| `travel` | optional, `[from, to]`: zero until `from`, full at `to`. Four numbers `[from, to, back, off]` also fall back to zero between `back` and `off` |
| `shifted` | `true`: live only while the `shift` button's layer is on |

A list of bindings on one axis is summed. Two bindings may not *climb* over the same stretch of
one control in one layer; a fall may share its stretch with the next binding's climb, which is
how one hands a stick to the other.

**A keyboard axis** is the keystrokes pushing it each way: `"+"` is the axis's own positive
direction as its `positive:` words it — forward, left, counter-clockwise, nose down, up — and
`"-"` the other. So the keyboard carries no signs: there is no stick for a sign to be relative
to. Every command axis is in `keyboard.axes`, bound or `{ unbound: <why> }`: an axis quietly
missing is a keyboard that cannot do something the pad can, and nothing would say so. A key held
pushes its direction linearly to full over `full_after_s` and is back at rest the moment it
comes up; two keystrokes on one direction are summed and clamped.

**An axis may be moved rather than placed.** `integrate_s` on an axis makes a deflection a
speed: full deflection carries the axis from its rest to either end of its range in that many
seconds, and let go it stays where it is. It is the pad's: on the keyboard the same axis is
placed, as every axis there is. `reset` names the
pad button whose tap — pressed and let go with no stick it could be reaching for moved in
between — puts every such axis back at its rest; the release does too.

### 4.3 The plain case: `jumper.tripod`

The four gait tasks carry the same file, one velocity command from two sticks:

| command axis | pad | sign | keyboard, `"+"` / `"-"` |
|---|---|---|---|
| `lin_vel_x` (forward +) | `Ly` | −1 | `W` or ↑ / `S` or ↓ |
| `lin_vel_y` (left +) | `Lx` | −1 | `A` or ← / `D` or → |
| `ang_vel_z` (counter-clockwise +) | `Rx` | −1 | `J` / `L` |
| hand the command back | `B` (`release_button`) | | `B` (`release`) |

```yaml
devices:
  gamepad:
    scheme: absolute
    layout: xbox
    axes:
      lin_vel_x: { source: Ly, sign: -1 }
      lin_vel_y: { source: Lx, sign: -1 }
      ang_vel_z: { source: Rx, sign: -1 }
    deadzone: device_reported_rescaled
    release_button: B
  keyboard:
    scheme: keys
    full_after_s: 2.0
    axes:
      lin_vel_x: { "+": [key_w, key_up], "-": [key_s, key_down] }
      lin_vel_y: { "+": [key_a, key_left], "-": [key_d, key_right] }
      ang_vel_z: { "+": [key_j], "-": [key_l] }
    release: [key_escape]
```

Note where the signs are. On the pad, `sign: -1` says that the stick's negative end is forward —
up is `Ly` negative, by evdev's convention. On the keyboard there is no sign to get wrong: `W` is
under `"+"` because `W` is forward, and forward is `lin_vel_x`'s positive direction, as the axis
says. The ranges differ by task (tripod ±0.5 m/s and ±0.75 rad/s, flat ±0.8 and ±1.0); the file
does not, because full deflection is each task's own range edge.

### 4.4 Two commands, a split stick, a shift layer and a moved axis: `jumper.posture`

`jumper.posture` drives a velocity command and a posture command, seven channels, from either
device:

| does | pad | keyboard |
|---|---|---|
| walk forward / back | left stick up / down | `W` / `S`, or ↑ / ↓ |
| step left / right | left stick left / right | `A` / `D`, or ← / → |
| nose down / up (`pitch`) | right stick up / down | `I` / `K` |
| twist counter-clockwise / clockwise | right stick left / right, first half | Shift + `J` / Shift + `L` |
| turn counter-clockwise / clockwise | right stick left / right, second half | `J` / `L` |
| roll, left side down / up | `R3` held, right stick left / right | `U` / `O` |
| height up / down — a speed; let go, it stays | `R3` held, right stick up / down | `N` / `M` |
| height back to standing | `R3` tapped | `B`, with everything else |
| let go of everything | `B` | `B` |
| stand still, level and square, at the height last set | hands off | let go |

```yaml
    axes:
      lin_vel_x: { source: Ly, sign: -1 }
      lin_vel_y: { source: Lx, sign: -1 }
      twist:     { source: Rx, sign: -1, travel: [0.0, 0.5, 0.5, 0.75] }
      ang_vel_z: { source: Rx, sign: -1, travel: [0.5, 1.0] }
      pitch:     { source: Ry, sign: -1 }
      roll:      { source: Rx, sign: 1, shifted: true }
      height:    { source: Ry, sign: -1, shifted: true }
    shift: { button: R3, gesture: hold }
    reset: R3
  keyboard:
    axes:
      ang_vel_z: { "+": [key_j], "-": [key_l] }
      twist:     { "+": [key_h], "-": [key_semicolon] }
      pitch:     { "+": [key_i], "-": [key_k] }
      roll:      { "+": [key_o], "-": [key_u] }
      height:    { "+": [key_n], "-": [key_m] }
      # ... and the walk, as in §4.3
```

The height axis carries `integrate_s: 1.0` beside its `rest: neutral_height`: one second from
standing to either end on a full stick (2026-09-29; two was too slow).

- **`travel` splits one control between axes.** `twist` climbs over the first half of the right
  stick's travel and falls back to zero between 0.5 and 0.75, while `ang_vel_z` climbs over the
  second half: the body leads into a turn and is square again before the turn is fast.
- **`shifted: true`** puts a binding on the layer `R3` brings in, for as long as it is held
  (`gesture: hold`; `toggle` would keep it in from one click to the next). With it in, the
  right stick's left-right travel rolls the body and its up-down travel moves the height;
  twist, turn and pitch are zero. Let go of `R3` and the turn and pitch are back at once.
- **On the pad the height is moved, not placed** (`integrate_s`, asked for on 2026-09-29):
  pushed, the stick raises or lowers the body at a speed — full deflection crosses from standing
  to either end of the range in one second — and let go, the body stays at that height. `R3`
  tapped, with the stick left alone, puts it back at standing (`reset: R3`); `R3` held with the
  stick pushed is the height or the roll, and that press is no tap. `B` on the pad and Esc on
  the keys put it back with everything else. Before that the height was placed: on the triggers
  until that morning, then on `R3` and the stick, back at standing the moment `R3` came up.
- **On the keys the height is placed**, as the doc has it: `N` and `M` are the high and the low
  stance — held, the body goes there over the key's ramp, and let go it is standing again, as
  `U` and `O` are the rolls. The keys never move the pad's height: while they drive, the height
  the pad left is kept, and it is the height again when the pad is touched.
- **The keyboard has no layer.** The roll and the height have keys of their own, and so does the
  twist, `H` and `;` beside `J` and `L`. Until the doc's revision of 2026-09-29 the twist was `J`
  and `L` with Shift held -- a chord, which the dictionary still spells `shift+key_j` and which
  answers alone while its modifier is down.
- **A list sums controls**: two pad bindings on one axis add, clamped. No task uses one now; the
  triggers on `height` did until 2026-09-29.

The pitch convention — **nose down positive** — is shared with `jumper.five_foot`, because both
feed the robot's one pitch channel; a posture checkpoint trained before 2026-09-26 reads it
inverted.

### 4.5 What a command may do while walking: `bands` and `max_rate`

A command term may carry a `moving` band and a `stand_threshold` — how far an axis may go
once the velocity command asks for motion — and a `max_rate`. The export writes them into the
contract, and the controller's `CommandShaper` applies them on every host, clamp then ramp,
before every observation: the policy sees the command it was trained to see, whatever the
person does with the stick. `jumper.posture` bands its posture to ±15° while walking;
`jumper.five_foot` also ramps its body pose at 30°/s.

### 4.6 Controls a task keeps for itself: `jumper.five_foot`

`jumper.five_foot` walks and leans like `jumper.posture` (no height), and keeps five controls no
command reads, declared once at the top of the file and bound on each device by name:

```yaml
task:
  claw_left:      { kind: amount, does: closes the left claw as far as it is held }
  claw_right:     { kind: amount, does: closes the right claw as far as it is held }
  arm_thumb_up:   { kind: press,  does: "holds the carried arm straight out, thumb up, while held" }
  arm_thumb_down: { kind: press,  does: "holds the carried arm straight out, thumb down, while held" }
  arm_web_up:     { kind: press,  does: "holds the carried arm straight out, thumb-web up, while held" }
  arm_to_face:    { kind: press,  does: "holds the carried arm bent in, the claw before the visor, while held" }

devices:
  gamepad:
    task: { claw_left: LT, claw_right: RT, arm_thumb_up: dpad_up,
            arm_thumb_down: dpad_down, arm_web_up: dpad_left, arm_to_face: dpad_right }
  keyboard:
    task: { claw_left: [key_space], claw_right: [key_space], arm_thumb_up: [shift],
            arm_thumb_down: [alt], arm_web_up: [ctrl], arm_to_face: [key_q] }
```

An `amount` is 0 to 1, as far as it is held: on the pad an axis — a trigger's travel — and on
the keyboard a key's ramp. A `press` is 1 while it is held: on the pad a d-pad direction, because
the ten buttons stay the bundle's for switching modes and the operator's for letting go, and on
the keyboard any keystroke. Task controls may share a keystroke with each other — both claws
close on Space, and a mode reads only its own side's — but not with an axis direction or the
release. The `does` words are what the manual prints.

What a kept control *does* is the task's own code, on the robot its deploy hook
`tasks/jumper/five_foot/deploy/lib.rs`, handed the kept controls by name and nothing else, from
whichever device drives. A hook declares the names it reads, and the controller refuses both
directions of disagreement: a hook reading a name the file does not declare, and a file
declaring controls no hook answers. In replay the task's own term answers what it can:
`mdp/gripper.py` closes the left claw on `claw_left`, through `Operator.task_control`.

`jumper.five_foot`'s arm is held out at a preset for as long as its control is held, and goes
back to its stow the moment nothing is — Control-agent 3.1's momentary switches, asked for on
2026-09-29; until then a d-pad direction was read when it was let go and pressed again to stow,
all a key through MuJoCo's viewer could deliver when the hook was written. Two held together are
the pose between them; thumb up and thumb down together are neither; `arm_to_face`, the one
bent-in preset, does not blend -- held, it is the pose. In the claw modes Shift
holds the arm out, and a modifier held for one thing cannot also modify a key for another, so
the keyboard's twist is `unbound` there and the twist is the pad's alone.

### 4.7 Which device drives

Either device drives every channel on its own, and **the one touched last drives** — the
commands and the task's controls alike. Touching the pad (a stick off centre, a button, the
d-pad) makes it the source and drops the keyboard's holds: a key still down drives nothing until
it is pressed again, so letting go of the pad does not revive a key held from before. A key
going down that the file binds takes over from an idle pad. A held key climbs linearly to full
deflection over `full_after_s` and is back at rest the moment it comes up — every axis, a moved
one included: a key places, and only the pad moves.

The release — `release_button` on the pad, a `release` keystroke on the keyboard, `B` and Esc in
every file — lets go of everything: the keyboard's holds are dropped, a toggled shift layer turns
off (a held one follows its button), every moved axis is back at its rest, and the command returns
to its rest (or, in `play --task`, to the random sampler).

Every mode's operator hears every frame and every key, whichever mode is running, so a mode
switched into has already heard the shift and the release. What differs by mode is only the
reading: each operator reads its own contract's bindings and scales to its own ranges.

### 4.8 What the loader refuses

`load_controls` and, on every host, the controller's `OperatorSpec::check` refuse the same
things, each a file that would load and then drive wrongly:

- a name the dictionary does not know — a source, a button, a key, a modifier — or a keystroke
  that is not a key, a modifier, or a modifier and a key;
- a command axis with no pad binding, or missing from the keyboard's `axes` without an
  `unbound:` and a reason; a pad binding or a keyboard entry on an axis no command has; a term or
  an axis listed twice; a key written twice in the YAML;
- a `sign` other than ±1, a `travel` out of order, two bindings climbing over one stretch;
- a `shift` that is neither `hold` nor `toggle`, a shift that is the release button, a shift
  nothing is `shifted` on, a `shifted` binding with no shift;
- an `integrate_s` that is not a positive number of seconds; a `reset` with no moved axis to put
  back, or on the release button;
- one keystroke on two things among the axis directions and the release; a modifier bound on
  its own anywhere in the keyboard block while it also modifies a key there — holding it for the
  one would do it while reaching for the other;
- a task control declared and not bound on both devices, or bound and not declared; a `kind`
  other than `amount` or `press`, or no `does`; on the pad a `press` not on a d-pad direction,
  an `amount` not on an axis, one pad control on two task controls, or one a command, the
  release, the shift or the reset also reads;
- the retired schema 1, the notched keyboard's `step`, and the `centre` key.

### 4.9 Writing one

Use the `new-task` skill for a new task, which writes all four files, or the `controls` skill
for the controls alone. The plain case is generated:

```bash
python .claude/skills/controls/scripts/write_controls.py --task jumper.<name> --print
```

`--bind AXIS=SOURCE±` changes a pad binding (the trailing `+`/`-` is the sign, and required),
`--release` the pad's release button, `--force` overwrites. The keyboard it writes is the one
every jumper task shares: `W S A D` and the arrows, `J L`, `B`. A second command, a split stick,
a shift layer, a modifier chord, a moved axis or a task's own controls is a hand edit of what
it writes, with `jumper.posture`'s and `jumper.five_foot`'s files as the worked examples.
`tests/test_controls.py` loads the existing tasks' files only, so check a new or edited file
directly:

```bash
python -c "from pathlib import Path; from tasks.jumper.common.mdp.controls import load_controls; load_controls(Path('tasks/jumper/<name>/controls.yaml'))"
```

---

## 5. Switching modes in a bundle

### 5.1 Two files

| | owns | example |
|---|---|---|
| `deploy/manifests.json` | the modes, and the switches into them — one per device — and the moments that leave them | `jump` on `A` and on Space, from `locomotion` |
| `deploy/jumper.controller.toml` | the cascade, the safety limits, states that run no model, and the `[fsm]` settings (the click window and `exclusive` included) | `feedback_stale` → `safe` first |

`scripts/deploy.py` composes the two into the bundle's `controller.toml` and refuses a cascade
file that declares a binding, an `[fsm.keyboard]` table or a model state — "two files naming
the same key is how the two come to disagree". The manifest writes no rules, so the cascade file
must carry `when = "button:<mode>"` for every mode the manifest binds; a binding no rule reads
is refused as a dead key.

### 5.2 A mode's entry

```json
"jump": {
  "task": "jumper.jump", "policy": "tasks/jumper/jump/out/<dir>",
  "pad":  {"button": "A", "on": "toggle", "from": ["locomotion"]},
  "keys": {"key": ["key_space"], "on": "toggle", "from": ["locomotion"]}
}
```

| field | |
|---|---|
| `task`, `policy` | required: the task id and the export directory (never a checkpoint) |
| `pad` | the pad's switch: `{button, on, with, from}`. `button` is a pad button or a `dpad_*` direction; `with` a pad button that must be held — `"with": "menu"` plus `"button": "dpad_up"` is Menu and up |
| `keys` | the keyboard's switch: `{key, on, with, from}`. `key` is a list — the jumper dances are on `["key_1", "keypad_1"]`, either row — and may name a modifier used as a key (`"ctrl"`); `with` is a modifier, `ctrl`, `shift` or `alt`, either of its two keys |
| `on` | in either switch, the gesture; `toggle` when omitted |
| `from` | in either switch, the modes it may be pressed in; any, when omitted |
| `hook` | the task's deploy configuration, handed to its `deploy/lib.rs` — `{"side": "right"}` |

A mode with neither `pad` nor `keys` gets no binding: it is reached by a rule that needs none,
typically the cascade's final `always`, which makes it the default mode. Each switch becomes
`[[fsm.button]]` entries named after the mode — one for the pad's, one per key of the
keyboard's — with `pad` or `key`, `on`, `with` and `from`. **Entries with one name share one
latch**, so either device switches the mode on and either switches it off. Until 2026-09-29 a
mode carried one `button`, one `key`, one gesture and one modifier for both.

`from` gates **entering** only: a latch cannot come on, and a one-shot gesture, an event or a
leave cannot fire, while the cascade is in a state `from` does not list. A latch already on can
always be switched off by its own control, from inside the mode it entered — `A` pressed again
leaves the jump, though the jump is not in its own `from`. It is how a switch says it is not
pressed in a mode whose own controls use its control (§5.9).

### 5.3 Gestures, latches and clicks

`button:<name>` is true while that binding is **active**. What makes it active is its gesture:

- **`toggle`** latches on a press and unlatches on the next: press to enter, press again to
  leave. This is what a mode switch almost always wants, since a mode has to stay entered
  through its ramp and its motion.
- **`rise`** and **`fall`** are active for one observation; **`hold`** while the control is
  down (a dead-man switch).
- **Clicks** count presses of one control — the same pad button, or the same key under the
  same modifier — each inside `click_window_ms` of the one before, and switch a mode the way
  `toggle` does: `double` to enter, `double` again to leave. As an `event` a click fires once.

How a click count is decided:

- **The longest click bound on a control fires on the press that completes it.** With only
  `double` on `A`, the second press enters at once; with only `single`, the first — a `toggle`
  as far as anyone can tell.
- **A shorter count waits the window out**, because another press could still make it longer.
  With `single` and `double` both on `A`, one press enters the `single`'s mode 300 ms later,
  and two presses enter the `double`'s at once. The wait is the price of sharing a control,
  and only then.
- **A count nothing is bound to fires nothing** — three presses on a control that has `single`
  and `double` do nothing.
- **The window is a time, not a number of ticks.** It is measured on the timestamps each host
  already hands the controller, so the board observing at the pad's 50 Hz and a browser at
  its frame rate count the same presses the same way. `click_window_ms` is required in `[fsm]`
  once any click is bound; the `jumper` cascade file sets 300 — chosen, not measured.
- **The keyboard counts its own.** A key and a pad button are two controls with two counts, so
  one press of Space and one of `A` are not a double. (Until 2026-09-29 a key in the bundle's
  `keyboard` table *was* its pad button, and they counted together.)

The crate refuses three combinations, each a click that would misfire: a control with a click
beside a single-press gesture (every click would be a `toggle` first — bind the single press as
`single` instead); a click on a control that also modifies another binding (the click would
fire as the hand reaches for the chord); and a click on a d-pad direction a task keeps
(§5.7).

**A modifier is consumed by what it modifies.** Once Menu + up fires, Menu's own bindings stay
quiet until it is released, so "Menu let go on its own leaves the dance" does not fire as the
hand lets go of the chord that chose it. Ctrl on the keyboard is spent the same way.

**A held modifier silences the bare control.** While Menu is held, d-pad up answers only the
binding chorded with Menu — the dance — and not the gesture on up alone; while Ctrl is held, `1`
is Ctrl + `1`. It is the operator's rule for a chord (`shift+key_j` answers alone while Shift
is down), for switches.

**A latch outlives nothing.** A recording that plays out releases its mode's latch and the
cascade hands back (`jumper.jump`, 1.17 s after it starts); stale pad input releases them all
(§3). A recording the cascade leaves *before* it plays out -- the tilt fallback, stale feedback
-- is over the same way, and its latch goes with it. Until 2026-10-15 it did not: a dash that
tipped the robot stayed latched through `safe`, and the moment the robot was level again the
cascade entered the dash from the start, with nobody pressing anything (a crab battle's second
round opened with one crab dashing off its mark). On the robot that is a fall, a person standing
it back up, and the move it fell on playing again. A gait left the same way keeps its latch:
`carry` is a held LB, and letting go of a dead-man switch is the operator's to do.

### 5.4 The keyboard for switches

The keyboard switches on keys of its own: a mode's `keys` names them, with a modifier as its
`with`, and nothing on the keyboard refers to a pad button. The dances are Ctrl + `1` `2` `3`
`4` on the keys and Menu + the d-pad on the pad because the manifest says both, not because one
is a copy of the other — and they differ where they have to: the keyboard's dances are not
pressed from a claw mode, whose Ctrl holds the arm out, while the pad's Menu chords are.

A key switch is judged against the keyboard block of every mode it may be pressed in: its key,
and its modifier, may not be a key or a modifier that mode's keyboard uses. There is no chord
exception on the keyboard (the pad's is §5.7); `from` is how a key switch stays out of a mode
that uses it. The board has no keyboard; these keys are for a browser and `play --app`, which
hands the viewer's keys to its controller down and up, Space pausing the viewer and `G`, `H`
drawing fog and convex hulls as they go.

### 5.5 Leaving

A bundle-level `leave` list names moments that let go of modes, after which the cascade falls
through to its `always` mode — the default:

```json
"leave": [
  {"name": "leave_motion",
   "leaves": ["dance_crab", "...", "gesture_salute"],
   "pad":  {"button": "menu", "on": "fall", "from": ["dance_crab", "...", "gesture_salute"]},
   "keys": {"key": ["ctrl"], "on": "fall", "from": ["dance_crab", "...", "gesture_salute"]}}
]
```

A leave is a moment — `rise`, `fall` or a click — never a `toggle`, a `hold` or an event; it is
not itself latched, so no rule may read it, and every name it leaves has to be a latching
binding's. In the `jumper` bundle Menu let go on its own, or Ctrl, interrupts any dance or
gesture and the robot walks again (Control-agent 3.1: 单击Menu退出舞蹈回到初始姿态). It is
`fall`, and the chord that chose a dance has spent the modifier, so reaching for Menu + up does
not leave on the way.

### 5.6 The cascade

`[[fsm.rule]]` entries are evaluated in order and **the first match wins**; the order is the
safety argument. The `jumper` cascade:

```toml
[[fsm.rule]]  when = "feedback_stale"      enter = "safe"      # no feedback, no policy
[[fsm.rule]]  when = "tilted"              enter = "safe"      # past 50 degrees
[[fsm.rule]]  when = "in_state:safe"       enter = "@initial"
[[fsm.rule]]  when = "button:jump"         enter = "jump"
[[fsm.rule]]  when = "button:dance_crab"   enter = "dance_crab"     # ... and the other three dances
[[fsm.rule]]  when = "button:gesture_hello" enter = "gesture_hello" # ... and the other three gestures
[[fsm.rule]]  when = "button:claw_right"   enter = "claw_right"
[[fsm.rule]]  when = "button:claw_left"    enter = "claw_left"
[[fsm.rule]]  when = "always"              enter = "locomotion"
```

The last rule must be `always` and no other may be, so the cascade is total and its target is
the default mode. `@stay` matches and changes nothing; `gripper_active` is refused, since no
host supplies it.

**One mode at a time.** Latches are a set unless `[fsm] exclusive = true`, and then a latch
coming on releases every other: the last switch pressed is the mode. The `jumper` cascade file
sets it (2026-09-29): without it `LB` in the right claw would latch *behind* it, and a dance
entered from a claw would hand back to the claw when it ended rather than to walking. With it,
the order among the operator's rules settles only the chord of §5.7.

**The switch-in ramp.** Entering a model's mode slides the setpoint from the measured pose to
that mode's home pose over `mode_switch_ramp_s`, at `ramp_kp`/`ramp_kd`, and the policy starts
when every joint is within `pose_reach_tol` — on the measured pose, never on a timer.

### 5.7 A d-pad direction shared by a task and a switch

`jumper.five_foot` keeps the d-pad's up, down and left for its arm, and the dances are Menu +
the d-pad. The two may share a direction only as a **chord that leaves the mode on the press**,
and `FsmConfig::chord_leaves_first` refuses anything else: the switch has to carry `with`, has to
be a `toggle` (a `rise` is true for one tick, so the cascade would leave and come back), and its
rule has to come before any rule that keeps the cascade in the mode. That is why the dance rules
sit above both claw rules. The keyboard has no such exception: Ctrl is the arm's in a claw mode,
so the keyboard's dances leave the claw modes out of their `from`.

### 5.8 Events: controls inside a mode

A top-level `buttons` list declares bindings that are not mode switches — a recorded motion's
`go`, where a person picks the moment inside the mode:

```json
"buttons": [{ "name": "<go_event>", "pad": {"button": "A", "on": "fall"} }]
```

They take the same `pad` and `keys` switches, are composed with `event = true`, and are read by
the mode whose contract names them in its `reference.go_event`; the controller refuses an event
no mode reads and a `go_event` no button declares. A recording may instead say
`starts_on_entry` — the `jumper` jump, dances and gestures do, so the bundle has no `buttons` —
and the two are exclusive.

### 5.9 Every overlap, and where it is refused

| overlap | refused by | when |
|---|---|---|
| a name outside the dictionary; a rule naming a button nothing declares; a binding no rule reads; one control, gesture and modifier bound twice; a binding with both `pad` and `key`; a `with` that is not a modifier of its binding's device; a `from` naming no state | `FsmConfig::parse` | every host, at load |
| a click beside a single-press gesture; a click on a modifier; clicks with no `click_window_ms` | `FsmConfig::parse` | every host, at load |
| a leave that is a `toggle`, a `hold` or an event, that lets go of a name nothing latches, or that a rule reads; any key `[fsm]` does not read | `FsmConfig::parse` | every host, at load |
| a switch that may be pressed in a mode, on a control that mode's pad block uses — a command's stick, the shift, the reset, the release — or on a key or modifier its keyboard block uses | `operator::check_switches`, from `Bundle::open` and `Operators::beside` | the bundler's check, the board, and each host as it loads |
| a pad switch on a d-pad direction a task keeps, other than a chord that leaves first | the same → `chord_leaves_first` | the same |
| a binding, an `[fsm.keyboard]` or a model state in the cascade file | `scripts/deploy.py` | the build |

A switch is judged in every mode it **may be pressed in** — its `from` names the mode, or it has
none — and in the mode it latches, where it can always switch itself off.

### 5.10 Adding a switch

To make a double click of the left stick (`L3`) the way into a new mode `crab`, from walking:

1. Export the mode's policy, and give it an entry in the manifest:
   `"crab": { "task": "jumper.<task>", "policy": "tasks/jumper/<task>/out/<dir>", "pad": {"button": "L3", "on": "double", "from": ["locomotion"]} }`.
   Pick a button no mode's `controls.yaml` uses where the switch may be pressed — not `B`, every
   task's release, nor `R3`, the posture tasks' shift — and not `X` or `Y`, which the dictionary
   marks unresolved.
2. Add `[[fsm.rule]] when = "button:crab" enter = "crab"` to the cascade file, above any rule it
   should win over and below the safety rules. `click_window_ms` is already there.
3. For the keyboard, give it `"keys": {"key": ["keypad_5"], "on": "double", "from":
   ["locomotion"]}`; nothing else binds `keypad_5`. It is a second binding named `crab`, with a
   count of its own.
4. Build: `python scripts/deploy.py --allow-incomplete` checks both halves without docker; the
   full build is `python scripts/deploy.py`. Read the new lines in `manual.en.json`.

### 5.11 The `jumper` bundle

Laid out as Control-agent 3.1, section 二 (按键及其功能), lays it out, asked for on 2026-09-29.
Every switch is a `toggle`; a digit is the main row's or the keypad's alike.

| mode | task | pad | keyboard | may be pressed from |
|---|---|---|---|---|
| `locomotion` | `jumper.posture` | — (the cascade's `always`: the default) | — | |
| `jump` | `jumper.jump` | `A` | Space | `locomotion` |
| `claw_left` | `jumper.five_foot`, hook side left | `LB` | `V` | `locomotion`, `claw_right` |
| `claw_right` | `jumper.five_foot`, hook side right | `RB` | `B` | `locomotion`, `claw_left` |
| `dance_crab` (螃蟹舞) | `jumper.dance` | Menu + d-pad up | Ctrl + `1` | `locomotion`, a dance or a gesture, and on the pad either claw |
| `dance_brazilian` | `jumper.dance_brazilian` | Menu + d-pad down | Ctrl + `2` | the same |
| `dance_maze` | `jumper.dance_maze` | Menu + d-pad left | Ctrl + `3` | the same |
| `dance_dream_wings` | `jumper.dance_dream_wings` | Menu + d-pad right | Ctrl + `4` | the same |
| `gesture_hello` | `jumper.gesture_hello` | d-pad up | `1` | `locomotion` |
| `gesture_bow` | `jumper.gesture_bow` | d-pad down | `2` | `locomotion` |
| `gesture_paw` | `jumper.gesture_paw` | d-pad left | `3` | `locomotion` |
| `gesture_salute` | `jumper.gesture_salute` | d-pad right | `4` | `locomotion` |
| `leave_motion` (a leave) | lets go of any dance or gesture | Menu, let go on its own | Ctrl, let go on its own | a dance or a gesture |

Inside the modes: `locomotion` is §4.4, the claw modes §4.6, and the jump, the dances and the
gestures read no stick or key — a recording drives them, from entry to its end.

- **One mode at a time** (`exclusive`). A claw's switch pressed again hands back to
  `locomotion`; `LB` in the right claw is the left claw at once, and `RB` in the left the right.
- **The dances and gestures end by themselves**, and the cascade hands back to `locomotion`.
  Menu or Ctrl, let go on its own, interrupts one; its own switch pressed again ends it too, and
  another dance's chord moves to that dance.
- **The jump cannot be interrupted**: leaving a jump in the air lands the robot out of its
  landing, so `leave_motion` does not name it and only `A` (Space) pressed again ends one early.
- **In a claw mode** `A` does nothing and Space is the claw, on the keys as on the pad (the jump
  is pressed from `locomotion` only); the d-pad alone is the arm, and 1 to 4 do nothing. The
  pad's Menu chords reach a dance from there and leave the claw on the press; the keyboard's
  cannot, since Ctrl holds the arm out.
- `jumper.dance_waist` has an export and is not in section 二, so it is not in the bundle.

---

## 6. The manual

Every bundle carries `manual.en.json` (`kk-bundle-manual/1`): each mode — its task, whether it
is the default, how it is entered and left — with what every pad control does in it
(`controls`) and, as a list of its own, what every key does in it (`keys`); and every switch,
the pad's and the keyboard's apart (`device`), with the modes each may be pressed `from`. It is
generated at build from the controller's own account of its controls, `pad_guide()`
(`deploy/fsm/src/guide.rs`), so it cannot disagree with the bindings; a key appears as what it
does, never as a pad button, and a mode lists only the task controls its hook answers — the left
claw's lists `LT`, not `RT`. A click appears as what a person does — `"pad": "double-click A"`
on the pad's switch, `"does": "switch to jump; double-click again to switch back"` on both. A
browser draws the same data live from `padGuide()`, and `play --app` prints both devices'
switches from it when it starts.

The Chinese manual is a translation of it, added with `--translate` by the `bundle-manual`
skill; the build checks that only the text differs.

---

## 7. Where each part lives

| part | file |
|---|---|
| the dictionary | `controller/vocabulary.json`; `controller/vocabulary.py`; `deploy/fsm/src/vocabulary.rs` |
| reading a pad on a bench | `controller/device.py` (Linux), `controller/xinput.py` (Windows), `controller/xbox.py`; `python -m controller` |
| both edges of a viewer key | `rl/mjrl/viewer/keys.py` |
| `controls.yaml` and its checks | `tasks/jumper/common/mdp/controls.py` |
| the replay operator | `tasks/jumper/common/mdp/operator.py` |
| the contract's `controller` block | `controller_contract` in `controls.py`, written by `scripts/export.py` |
| each mode's operator, the keyboard's keystrokes, `CommandShaper`, `check_switches` | `deploy/fsm/src/operator.rs` |
| gestures, latches, clicks, `from`, leaves, one mode at a time | `Buttons` in `deploy/fsm/src/control.rs` |
| `[fsm]`, bindings, the load-time refusals | `deploy/fsm/src/config.rs` |
| the cascade | `deploy/fsm/src/fsm.rs` |
| the bundle-level collision checks | `Bundle::open` in `deploy/fsm/src/bundle.rs` |
| a task's deploy hook | `deploy/fsm/src/hook.rs`, `tasks/<task>/deploy/lib.rs` |
| composing the manifest and the cascade file; the manual | `compose_fsm`, `write_manual` in `scripts/deploy.py` |
| the pad and the keys for a page | `pad_guide` in `deploy/fsm/src/guide.rs` |

The tests that pin these are named for the failure each one catches — `tests/test_controls.py`,
`tests/test_controller.py`, `tests/test_viewer_keys.py`, `tests/test_web_host.py`,
`tests/test_app_play.py`, and the Rust tests beside each module (`cargo test` in `deploy/fsm`),
among them `a_click_is_counted_in_time_not_in_frames`.
