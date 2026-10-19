# jumper

A robot controller and the policies it runs, exported from KingKongRobotics/jumper -- one
bundle for every host: the robot's board, a browser, and `play`.

```
runtime/web/controller.wasm       the controller for a browser, and its binding
runtime/web/controller.js           (wasm-bindgen 0.2.128)
runtime/mjlab/linux-x86_64/controller.so the controller for `play --app` on linux-x86_64
runtime/mjlab/win-amd64/controller.pyd the controller for `play --app` on win-amd64
runtime/mjlab/macosx-universal2/controller.so the controller for `play --app` on macosx-universal2
runtime/board/controller          the robot's whole program, for aarch64-unknown-linux-gnu
controller.toml                   the modes, the rules that switch between them, the keys
claw_left.json                    claw_left's contract: observation layout, joint order, gains
models/claw_left.onnx             claw_left's policy, onnx
models/claw_left.rknn             claw_left's policy, rknn
claw_right.json                   claw_right's contract: observation layout, joint order, gains
dance_brazilian.json              dance_brazilian's contract: observation layout, joint order, gains
models/dance_brazilian.onnx       dance_brazilian's policy, onnx
models/dance_brazilian.rknn       dance_brazilian's policy, rknn
brazilian.motion.trajectory.json  the recording dance_brazilian is driven by
dance_crab.json                   dance_crab's contract: observation layout, joint order, gains
models/dance_crab.onnx            dance_crab's policy, onnx
models/dance_crab.rknn            dance_crab's policy, rknn
demo.motion.trajectory.json       the recording dance_crab is driven by
dance_dream_wings.json            dance_dream_wings's contract: observation layout, joint order, gains
models/dance_dream_wings.onnx     dance_dream_wings's policy, onnx
models/dance_dream_wings.rknn     dance_dream_wings's policy, rknn
dream_wings.motion.trajectory.json the recording dance_dream_wings is driven by
dance_maze.json                   dance_maze's contract: observation layout, joint order, gains
models/dance_maze.onnx            dance_maze's policy, onnx
models/dance_maze.rknn            dance_maze's policy, rknn
maze.motion.trajectory.json       the recording dance_maze is driven by
dash.json                         dash's contract: observation layout, joint order, gains
models/dash.onnx                  dash's policy, onnx
models/dash.rknn                  dash's policy, rknn
dash_flat.trajectory.json         the recording dash is driven by
gesture_bow.json                  gesture_bow's contract: observation layout, joint order, gains
models/gesture_bow.onnx           gesture_bow's policy, onnx
models/gesture_bow.rknn           gesture_bow's policy, rknn
bow.motion.trajectory.json        the recording gesture_bow is driven by
gesture_hello.json                gesture_hello's contract: observation layout, joint order, gains
models/gesture_hello.onnx         gesture_hello's policy, onnx
models/gesture_hello.rknn         gesture_hello's policy, rknn
hello.motion.trajectory.json      the recording gesture_hello is driven by
gesture_paw.json                  gesture_paw's contract: observation layout, joint order, gains
models/gesture_paw.onnx           gesture_paw's policy, onnx
models/gesture_paw.rknn           gesture_paw's policy, rknn
paw.motion.trajectory.json        the recording gesture_paw is driven by
gesture_pump.json                 gesture_pump's contract: observation layout, joint order, gains
models/gesture_pump.onnx          gesture_pump's policy, onnx
models/gesture_pump.rknn          gesture_pump's policy, rknn
pump.motion.trajectory.json       the recording gesture_pump is driven by
gesture_salute.json               gesture_salute's contract: observation layout, joint order, gains
models/gesture_salute.onnx        gesture_salute's policy, onnx
models/gesture_salute.rknn        gesture_salute's policy, rknn
salute.motion.trajectory.json     the recording gesture_salute is driven by
gesture_sleep.json                gesture_sleep's contract: observation layout, joint order, gains
models/gesture_sleep.onnx         gesture_sleep's policy, onnx
models/gesture_sleep.rknn         gesture_sleep's policy, rknn
sleep.motion.trajectory.json      the recording gesture_sleep is driven by
gesture_threat.json               gesture_threat's contract: observation layout, joint order, gains
models/gesture_threat.onnx        gesture_threat's policy, onnx
models/gesture_threat.rknn        gesture_threat's policy, rknn
threat.motion.trajectory.json     the recording gesture_threat is driven by
gesture_wave.json                 gesture_wave's contract: observation layout, joint order, gains
models/gesture_wave.onnx          gesture_wave's policy, onnx
models/gesture_wave.rknn          gesture_wave's policy, rknn
wave.motion.trajectory.json       the recording gesture_wave is driven by
gesture_zoidberg.json             gesture_zoidberg's contract: observation layout, joint order, gains
models/gesture_zoidberg.onnx      gesture_zoidberg's policy, onnx
models/gesture_zoidberg.rknn      gesture_zoidberg's policy, rknn
zoidberg.motion.trajectory.json   the recording gesture_zoidberg is driven by
jump.json                         jump's contract: observation layout, joint order, gains
models/jump.onnx                  jump's policy, onnx
models/jump.rknn                  jump's policy, rknn
high_jump_flat.trajectory.json    the recording jump is driven by
locomotion.json                   locomotion's contract: observation layout, joint order, gains
models/locomotion.onnx            locomotion's policy, onnx
models/locomotion.rknn            locomotion's policy, rknn
sprint.json                       sprint's contract: observation layout, joint order, gains
models/sprint.onnx                sprint's policy, onnx
models/sprint.rknn                sprint's policy, rknn
reference.json                    recorded frames, so you can measure that you agree
manual.en.json                    every key and button this bundle reads, and what each does
bundle.json                       every file, its size and its SHA-256, and which host takes what
```

`manual.en.json` is written by the build from the controller's own account of
its pad and its keys, so it describes exactly the bindings here. A translation travels
beside it as `manual.<language>.json` -- `manual.zh.json` in Chinese -- listed
in `bundle.json`'s `manuals`. `deploy/app.schema` in KingKongRobotics/jumper describes every
file a bundle may hold and the format of each, and every build is checked
against it.

A reference-guided mode carries one more file, named by its contract's
`reference` block -- `high_jump_flat.trajectory.json` for `jumper.jump`. It is
the recording the mode is driven by, and the mode does not load without it.

**Each host takes its own and leaves the rest.** `bundle.json`'s `runtimes`
says which model format each host loads and where its build of the controller
is, and each mode's `models` names its file in every format:

| host | models | controller |
|---|---|---|
| `board` (RK3576) | `.rknn`, for the NPU | `runtime/board/controller`, aarch64 |
| `web` (a browser) | `.onnx` | `runtime/web/controller.wasm` and its `.js` binding -- the same bytes on any CPU |
| `mjlab` (`play --app`) | `.onnx` | `runtime/mjlab/<platform>/controller.so`, the build machine's, and `runtime/mjlab/win-amd64/controller.pyd` and `runtime/mjlab/macosx-universal2/controller.so` (arm64 and x86_64), cross-built |

A host whose build this bundle does not carry has no entry, and
`bundle.json`'s `notes` says how to add it. Everything else -- the FSM, the
contracts, the recordings, the reference -- is one copy every host reads, so
there is nothing for two hosts to disagree about.

On the board, from this directory:

```bash
./runtime/board/controller --bundle . --dry-run           # report, touch no bus
./runtime/board/controller --bundle . --check-reference   # replay the frames below
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
```

`--machine` is what this bundle cannot know: the DDS domains, the QoS XML and
the few constants that describe the actuators rather than the policy.
Everything else comes from the contracts here, and `--dry-run` prints what it
will use -- the DDS domains, the QoS file, each mode's controls as its contract
gives them -- and every note and stub.

With `play`, from KingKongRobotics/jumper -- every mode, on this bundle's own keys and a
gamepad, the controller's own joint targets driving the simulated servos:

```bash
python scripts/play.py --app <this directory, or the .app beside it>
python scripts/deploy.py --check-reference <this directory>
```

`play --app` takes the controller built for its own machine from here rather
than whatever `mjrl_fsm` is installed, because otherwise it runs a
*different build of the same source*, and the two diverge the moment somebody
rebuilds one of them.

The rest of this file is for a browser.

**The controller here is the one the robot runs.** Not the same logic
reimplemented from the same contract -- the same compiled object, built from
`deploy/fsm` at commit `372819e1fe1a7c2d77e78b16f7e4968d239542d9`. That is the point of shipping it: three
implementations of one observation agreed by people reading each other's code,
and a sign corrected in one left the other two wrong with every other number
still correct.

**And you can measure that your host runs it the same way.** `reference.json`
is a recorded run of this controller: per frame, the robot state that went in,
the observation it built, the action the policy returned and the joint targets
it published. Replay it on load:

(It says your host agrees with the machine that built the bundle. It says
nothing about whether that machine should be trusted -- see the last section.)

```js
const report = JSON.parse(fsm.checkReference(referenceJson))
// {frames, inferences, observation, target, divergent, modeMismatches}
```

`observation` and `target` should be **0**. They are the same f32 arithmetic in
the same order as every other host, so anything above about 1e-6 is a real
difference, not rounding. `divergent` names terms your host is expected to
source differently from training -- the robot measures nearly everything a
simulator computes -- with the worst difference each one actually showed, which
is normally 0 because the replay feeds every host the same recorded input. This
paragraph used to single out `joint_torque` as reconstructed on the robot from
the commanded PD and differing there by about 2 N*m. **That was wrong**: the
servo reports torque, the controller has always read it, and the term is
measured on the robot too -- the 2 N*m was the gap between a real number and a
reconstruction of it that nothing needed. `modeMismatches` is never acceptable:
a cascade landing in another state is a different robot, not a tolerance.

Your inference backend is a separate question and not covered by that call.
Run each frame's `obs` through your own model and compare against its `act`:
onnxruntime-web should match to ~1e-6, a quantised `.rknn` differs in the third
decimal. Keeping the two apart is deliberate -- a wrong observation and a
quantised model produce the same symptom, a robot that walks slightly wrong.

---

## Loading it

```js
import init, { WebFsm } from "./runtime/web/controller.js";

await init({
  module_or_path: await (await fetch("./runtime/web/controller.wasm")).arrayBuffer(),
});

const fsm = new WebFsm(
  fsmToml,                        // controller.toml, as text
  JSON.stringify(contracts),      // { "<mode>": <that mode's .json> }
  JSON.stringify(robot),          // below
  nowMicroseconds,
  JSON.stringify(trajectories),   // { "<mode>": <its *.trajectory.json text> }
);
```

`trajectories` is only for a mode whose contract carries a `reference` block --
a recorded motion the policy is driven by. Read the file that block names out of
this bundle and pass its text; omit the argument entirely for a bundle of
ordinary policies. A mode that needs one and does not get it is refused by name,
because the alternative is building its reference terms out of zeros, which is a
motionless recording that a standing robot tracks perfectly.

```js
const trajectories = {};
for (const [mode, contract] of Object.entries(contracts)) {
  if (contract.reference) {
    trajectories[mode] = await (await fetch(`./${contract.reference.file}`)).text();
  }
}
```

`robot` is what the *host* knows and no policy does:

```js
{
  joint_names: [...],             // wire_joint_order, from any mode's contract
  joint_pos_lo: [...],            // per joint; prefer the contract's joint_limits
  joint_pos_hi: [...],            //   and fall back to your own model's
  output_rate_hz: 50,             // how often you will call tick()
  gait_period: 0.32,              // optional; only for a contract whose gait_phase
  gait_gate_threshold: 0.05       //   carries no clock of its own -- each mode's
                                  //   contract names its period or its cadence law
}
```

`joint_limits` is a `{joint: [lo, hi]}` map in each mode's contract, in the
same radians as `default_joint_pos`, and it is what the robot itself clamps
to. Preferring it over your own model keeps the two hosts agreeing about the
one bound that cannot be observed going wrong -- a clamp wide enough never to
fire looks exactly like a clamp that works. Older bundles do not carry it; fall
back to your model then, and say which you used.

Every failure is thrown with the reason in it. There is no fallback: a host
that cannot build this should say so rather than run a robot on defaults
nobody chose.

## One control step

```js
fsm.set_state(q, qd, tau, quat, gyro, nowUs);
fsm.setSignal("base_lin_vel", linVelBody);        // everything else you can compute
if (stopPressed) fsm.letGo(nowUs);                // your Stop button, Escape -- once
for (const e of keyEvents) fsm.setKeyCode(e.code, e.down, e.repeat, nowUs);
if (pad) {                                        // a real pad, or on-screen sticks
  for (const [axis, v] of Object.entries(pad.axes)) fsm.setAxis(axis, v);
  for (const b of PAD_BUTTONS) fsm.setPad(b, pad.pressed.includes(b));
  fsm.padFrame(nowUs);
}
fsm.set_command(0, 0, 0, nowUs);                  // keeps the operator fresh

const mode = fsm.tick(nowUs);
if (mode !== undefined) {
  fsm.resume(await yourOnnxSession(mode).run(fsm.observation()));
}
publish(fsm.positions());         // joint targets, wire order
```

`tick` returns the mode needing an inference, or nothing when the step is
finished -- a warm start, a mode-switch ramp or a hold still publishes, and
`positions()` is ready either way.

**The split is not an abstraction.** A wasm module cannot call a JavaScript
model and return inside one synchronous call, and the loop publishes faster
than it infers: `tick` only asks when the active mode's `control_hz` says so.
Running the policy at your own rate aliases it *and* rescales the per-tick
joint rate limit.

## What must arrive exactly right, because none of it fails loudly

| argument | what it must be |
|---|---|
| `q` `qd` `tau` | **wire order**, i.e. `joint_names` above. Not your model's order and not the policy's; the three are paired by name, and pairing by index is wrong from the fifth joint on |
| `quat` | body orientation `(w, x, y, z)` |
| `gyro` | angular velocity **in the body frame**. An IMU gives this; a simulator usually gives you the world frame and a rotation matrix, so rotate: `out[a] = m[a]*w[0] + m[3+a]*w[1] + m[6+a]*w[2]` |
| `tau` | applied joint torque. A simulator has the solver's; the robot reads the servo's. If *your* host measures none, reconstruct what your PD commanded and say which you did |
| `vx` `vy` `wz` | never read by a mode with a controls block of its own (`takesRawInput()`); for a mode with none, **physical units**, m/s and rad/s |
| `setAxis` values | the pad as the robot's pad service reports it: sticks `[-1, 1]` with a 0.15 deadband already applied **and rescaled**, triggers `[0, 1]`. No sign, no scale -- those are the bundle's |
| `nowUs` | microseconds, monotonic. The module has no clock; freshness and rate are decided from what you pass |

If your simulator reports a 6-vector velocity, check its order before using it.
MuJoCo's is `(angular, linear)`; reading it the other way gives a gyro that
reports how fast the robot is walking, and nothing anywhere says so.

## Handing over a person's input

**Hand over what you see, and nothing you decided.** What a stick or a key
means is the running mode's: its `controller` block says which control drives
which command axis, how a stick's travel splits, which button holds a layer,
which key pushes which way, where a released axis rests -- and the controller
inside this wasm executes it. The pad and the keyboard are two paths, each
bound to what it does, so they may disagree on purpose. Modes may mean
different things by one control -- `jumper`'s walk raises the body on the
right stick held in and pushed up, where its claw modes tip the nose down, and
its claw modes close a claw on a trigger, or Space, that does nothing while
walking -- and the controller hands every mode everything you pass, so which
mode is running is never yours to track. A
page that mapped keys to a command itself would be a second implementation of
a contract that already exists, and a bundle with a scheme that page was never
written for would simply not be drivable.

So, every frame:

- **Every key, both edges.** `fsm.setKeyCode(event.code, down, repeat, nowUs)`
  with the browser's own `KeyboardEvent.code`. The dictionary compiled into
  this wasm knows which codes are keys it has a name for; the return value says
  whether anything here binds that one, which is what tells you to
  `preventDefault` it. `repeat` is the event's own flag, and says the key is
  still down: a key held pushes what it drives for as long as it is held --
  full after the mode's `full_after_s` -- and lets it back the moment its
  `keyup` arrives, so the up edge matters as much as the down. Shift, Alt and
  Ctrl are keys like any other, left and right each by its own code; Alt pressed
  on its own moves a Windows browser's focus to its menu unless you
  `preventDefault` it.
- **The pad, whenever there is one.** `fsm.setAxis(name, value)` for `Lx` `Ly`
  `Rx` `Ry` `LT` `RT`, `fsm.setPad(button, down)` for every button, then
  `fsm.padFrame(nowUs)`. On-screen sticks are a pad too: report them as `Lx`
  `Ly` `Rx` `Ry`. Pad and keyboard are both live and the one touched last
  drives; the controller arbitrates, so you need not.
- **Every quantity you can compute.** `fsm.setSignal(name, values)`, whether or
  not today's policy observes it -- `base_lin_vel`, body frame, m/s, is the one
  read now. A policy that needs something a robot cannot measure is refused at
  export, but a web-only bundle may, and a page that passed only what today's
  bundles need is a page tomorrow's outgrows without saying so.

- **Your own stop, once.** `fsm.letGo(nowUs)` when your page's Stop button or
  Escape is pressed: the operator let go of everything, exactly as the bundle's
  release button does. A host control rather than a pad button, because you
  have no button of the bundle's to press on somebody's behalf -- which one
  releases is the contract's to say. Call it before that step's keys, so a key
  pressed after the stop is a new press. A key still down keeps pushing until
  its `keyup`, and a page that lost focus mid-press may never be sent one, so
  without this a Stop button stops nothing.

`fsm.takesRawInput()` is true when any mode has a controls block. Those modes
never read `set_command`; a mode with none does, and sending it keeps the
operator fresh either way, so sending it every frame is harmless.
`fsm.inputs()` returns `{rawInput, keys: [{name, code}], controls, signals}` for
a page that wants to say which keys do something: `controls` is each mode's own
account of its bindings -- the lines the robot's controller prints before it
enables the motors -- on one line, the modes that read the pad alike named
together: `claw_left, claw_right: … | locomotion: …`. Show it as given;
describing the bundle yourself would be a second copy of its bindings.

`fsm.bindings()` returns `[{name, pad, key, on, with, from, leaves}]` -- the mode
switches, the events and the leaves: everything the operator can ask for
*between* modes, and which control asks it. A binding is one device's, a pad
button or a key, and two with one name -- the pad's `A` and the keyboard's
Space, both `jump` -- share one latch. Those are the same keys and buttons,
reported the same way:

```js
fsm.setKeyCode("KeyW", true, false, nowUs);   // the browser's name for it
fsm.setPad("A", down);                        // the robot pad service's names
```

Both lists come from one dictionary, `controller/vocabulary.json` -- schema
`kk-control-vocabulary/2`, compiled into this wasm, so there is no file to fetch
-- and a binding naming something outside it is refused when the bundle loads.
`fsm.setKey(name, down)` takes the dictionary's own name (`key_w`, `keypad_1`)
for a host that already has it.

Not "was pressed". `on` is one of nine gestures and two of them need the
release:

| `on` | fires |
|---|---|
| `rise` | the tick it goes down |
| `fall` | the tick it comes back up |
| `hold` | every tick it is down |
| `toggle` | down until pressed again |
| `single` `double` `triple` `quadruple` `quintuple` | that many presses, each within `click_window_ms` of the last; switches like `toggle` |

The click gestures are counted on the `nowUs` you hand `tick`, so hand it the
page's real clock (`performance.now() * 1000`): a clock that runs fast or slow
narrows or widens the window a person clicks in.

So press-to-arm and release-to-go is one control bound twice, and a host that
only reported presses could not drive half the vocabulary.

Pad names are `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3`, plus
`dpad_up` / `dpad_down` / `dpad_left` / `dpad_right`. Translate your Gamepad API
index into one of those and the two hosts agree about what was pressed. `view`
is on the pad and is deliberately not among them.

`with` names what must be held: a pad button for a pad binding, `ctrl`,
`shift` or `alt` -- either of its two keys -- for a key. A modifier is consumed
by what it modifies, and while it is held a binding without it on the same
control stays quiet, so `menu` alone, `menu`-plus-a-direction and the
direction alone can mean three things without one firing on the way to
another. `from` lists the modes a binding may be pressed in -- pressed
anywhere else it does nothing, though a latch it put on can always be switched
off by it -- and `leaves` the modes a binding lets go of, after which the
cascade falls back to its default mode.

Not every binding switches a mode, and two cases are worth expecting. **A
recorded motion ends.** A jump or a dance is a thing that finishes, and when it
does the controller hands the robot back: `tick` releases that mode's latch for
you and the cascade's last `always` rule catches it, so there is one way out of
a mode rather than two. A toggle you have drawn as lit will go out on its own --
and in a bundle that latches one mode at a time, when another switch is pressed
-- so read `fsm.mode()`, not your own copy of the latch. **And a motion's `go`,
where a bundle has one, is a moment, not a mode.** A push-off is 40 ms wide, so
the button that starts it is read by the mode itself rather than by a
transition rule. It is an ordinary binding otherwise: it is in `bindings()`, it
has a gesture, and you report it exactly like the rest. (The `jumper` bundle
has none: its jump, its dances and its gestures start on entering their modes.)

## Two things worth putting on screen

`fsm.mode()` is the active mode; `fsm.is_running_policy()` is whether it is
actually producing policy output. They differ through a mode-switch ramp,
which waits on the **measured pose** rather than a timer and can wait
indefinitely if the gains cannot converge. That is deliberate -- the
alternative is starting a policy from a pose it never trained around -- but a
robot holding still looks broken unless you say which it is.

## If you are running this from an upload

**Nothing here has checked this controller.** Your own copy, if you ship one,
is held against your own implementation by whatever test you wrote for it.
This one arrived with the upload. Say so where the person can see it.

Running an uploaded `.onnx` and running an uploaded wasm plus JavaScript are
not the same risk. A graph executed by onnxruntime is arithmetic over tensors;
`controller.js` is a program with your page's authority. If you load it,
**give it an opaque origin** -- an `<iframe sandbox="allow-scripts">` with no
`allow-same-origin`, talking over a `MessagePort`. Measured in Chrome from
inside such a frame: `origin` is `"null"`, `document.cookie`, `localStorage`,
`indexedDB` and `parent.document` all throw `SecurityError`, and a credentialed
`fetch` back to the host origin fails. A round trip of one 411-float
observation had a p95 of 0.1 ms against a 20 ms budget at 50 Hz.

Isolation does not bound CPU or memory. A watchdog that tears the frame down
when it does not answer within a few control periods is the other half.

## One source, compiled three times

`runtime/web/controller.wasm`, a native extension under `runtime/mjlab/`, an
aarch64 binary in `runtime/board/`. Same logic, same commit, which
`bundle.json`'s `runtimes` records for each. The two native ones are
**platform-specific** where the wasm is not, so each records the platform it
was built for, and a host on another one should say so rather than fail with a
loader error nobody can read.

---

Built by `scripts/deploy.py` from KingKongRobotics/jumper `372819e1fe1a7c2d77e78b16f7e4968d239542d9`.
