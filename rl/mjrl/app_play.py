"""Play an app in mjlab: every mode of it, through the deployment controller.

    python scripts/play.py --app out/bundle_<timestamp>/<name>.app

An `.app` is what `scripts/deploy.py` writes: one bundle every host
loads -- the robot, a browser, and this. `--app` opens it as the third of those
and runs all of it:

* **every mode**, entered the way the robot enters them: the bundle's own
  `controller.toml` cascade, on its own buttons and keys;
* **the person**, read through each mode's own controls: the viewer's keys and,
  when one is plugged in, a gamepad -- sticks, triggers, buttons and the d-pad;
* **the controller's own output**. The joint targets and gains it publishes are
  what the simulated servos track, so its ramps, its output filter, its joint
  stops and a task's deploy hook -- `jumper.five_foot`'s mirror, its claw on the
  trigger, its arm on the d-pad -- all reach the joints, and how wide a mode's
  action is stays the controller's business rather than the environment's.

The environment is the world and nothing more: the physics, the robot model and
its servo curve, the terrain, and a reset when the robot falls. Its own
observation, commands and rewards are not read, and it is built from the task
of the app's default mode unless `--task` names another.

This replaced `play --fsm`, which ran the controller's observation and cascade
but handed the policy's raw action back to mjlab to decode: no hook reached the
simulator, no key reached the controller, and a bundle with a recorded motion
in it did not load at all.

## Keys through MuJoCo's viewer

`mjrl.viewer.keys` hands over each key going down and coming up -- MuJoCo's
viewer itself reports the press alone, and that module says how the release is
had. They are queued on the viewer's thread and handed to the controller once a
step, on the simulation's, in the order they came: a key is held for as long as
it is held, and a chord is Ctrl held, then right. A tap shorter than a step is
down for one step and up at the next, so a `toggle` or a `rise` still sees it.
Until 2026-09-29 the release was inferred from auto-repeats the viewer never
reports, and every key came up 0.75 s after its press.

The viewer's own shortcuts fire beside these (`controller/vocabulary.json`,
`_keys_note`): Space also pauses the picture, and the arrows step a paused one.
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10, which this repository supports and which has no `tomllib`:
    # `tomli` is the same parser under its original name, and a dependency
    # there (`pyproject.toml`).
    import tomli as tomllib


class AppUnavailable(RuntimeError):
    """The app cannot be played, with what would fix it."""


@dataclass
class App:
    """An app opened for playing: the bundle directory, and what it runs first."""

    #: The directory holding `bundle.json` -- the app unzipped, or the bundle
    #: directory the build wrote beside it.
    root: Path
    #: The task of the mode the cascade falls back to, whose environment is the
    #: world the app is played in unless `--task` names another.
    default_task: str | None
    #: The fastest mode's control rate, Hz: what the board ticks the controller
    #: at (`bundle.rs::output_rate_hz`), and so what the world has to step at.
    control_hz: float
    #: Keeps an unzipped `.app` alive for as long as the app is played.
    _unzipped: tempfile.TemporaryDirectory | None = None

    def close(self) -> None:
        if self._unzipped is not None:
            self._unzipped.cleanup()
            self._unzipped = None


def open_app(path: Path) -> App:
    """An `.app`, unzipped, or the bundle directory beside it, as given.

    The build directory above both -- `out/bundle_<timestamp>/`, holding the
    bundle and its `.app` -- is refused with the app inside it named, because it
    is a path people have in hand and it is not itself a bundle.
    """
    path = Path(path)
    unzipped = None
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise AppUnavailable(f"{path} is not an app: an .app is a zip of a bundle")
        # Windows will not delete a DLL a process has loaded, and the app's
        # controller is one from the moment it is played until the interpreter
        # exits: `cleanup` raised there, from `play`'s `finally`, before the
        # environment was closed. What cannot be deleted is left in the temp
        # directory; everything else goes.
        unzipped = tempfile.TemporaryDirectory(prefix="mjrl-app-", ignore_cleanup_errors=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(unzipped.name)
        root = Path(unzipped.name)
    else:
        root = path
    if not (root / "bundle.json").is_file():
        if unzipped is not None:
            unzipped.cleanup()
            raise AppUnavailable(f"{path} holds no bundle.json, so it is not an app")
        inside = sorted(path.glob("*.app")) if path.is_dir() else []
        remedy = (
            f"       it holds {', '.join(p.name for p in inside)}. Play that:\n"
            f"       {inside[0]}"
            if inside
            else "       build one:\n       python scripts/deploy.py --manifest <name>"
        )
        raise AppUnavailable(f"{path} is not an app: no bundle.json.\n{remedy}")
    return App(root, _default_task(root), _control_hz(root), unzipped)


def _control_hz(root: Path) -> float:
    """The fastest mode's `control_hz`, as the board takes it."""
    manifest = json.loads((root / "bundle.json").read_text("utf-8"))
    return max(
        float(json.loads((root / entry["contract"]).read_text("utf-8"))["control"]["control_hz"])
        for entry in manifest["modes"].values()
    )


#: How far past the controller's tilt limit the world's own fall is moved.
#: Enough that a fall is the controller's before it is the world's, and short
#: of the robot on its side, where the world still has to take it back -- the
#: stand-in for somebody picking the robot up.
FALL_MARGIN_RAD = math.radians(30.0)


def leave_falls_to_the_app(env_cfg: Any, app: App) -> list[str]:
    """Move the world's orientation terminations past the controller's
    `tilt_limit`; returns the ones moved.

    **A fall is the controller's to handle first.** The family's world resets a
    robot tilted past 50 degrees, and the jumper controller calls it tilted
    past 0.873 rad -- 50.02 degrees -- and drops to `safe`. The world checks
    after physics and resets in the same step, so the controller was only ever
    shown the robot upright again, still in whatever mode it was in: `play
    --app` showed a jump or a dance carrying on from a teleport where the robot
    would have gone limp.
    """
    tilt = float(tomllib.loads((app.root / "controller.toml").read_text("utf-8"))["fsm"]["tilt_limit"])
    moved = []
    for name, term in (getattr(env_cfg, "terminations", None) or {}).items():
        limit = (getattr(term, "params", None) or {}).get("limit_angle")
        if limit is not None and limit <= tilt:
            term.params = {**term.params, "limit_angle": tilt + FALL_MARGIN_RAD}
            moved.append(name)
    return moved


def step_the_world_at(env_cfg: Any, hz: float) -> None:
    """Make the world step at `hz`: `decimation` so that `timestep * decimation`
    is the app's control period, the timestep left as the task set it.

    **The world's own rate is the task's choice, and an app's is its fastest
    mode's.** The controller infers each mode when that mode's period has
    passed, on the clock it is ticked with, and is ticked once a world step: a
    50 Hz world ticking an app with a 200 Hz mode -- `jumper.five_foot`'s world
    under the jumper app's posture mode -- ran that mode at 50 Hz, a quarter of
    its trained rate, with nothing said. Nothing in the world reads its own
    rate: the app writes the targets and no reward is evaluated.
    """
    dt = env_cfg.sim.mujoco.timestep
    steps = 1.0 / (hz * dt)
    if round(steps) < 1 or abs(steps - round(steps)) > 1e-6:
        raise AppUnavailable(
            f"this app ticks at {hz:g} Hz and the world's physics steps every "
            f"{dt * 1e3:g} ms, which is not a whole number of steps per tick. "
            "--physics-hz sets the physics rate"
        )
    env_cfg.decimation = round(steps)


def _default_task(root: Path) -> str | None:
    """The task of the state the cascade's `always` rule enters."""
    fsm = tomllib.loads((root / "controller.toml").read_text("utf-8"))["fsm"]
    target = next((r["enter"] for r in fsm.get("rule", []) if r["when"] == "always"), None)
    if target == "@initial":
        target = fsm.get("initial_state")
    state = next((s for s in fsm.get("state", []) if s["name"] == target), {})
    return state.get("task")


#: The key a macOS extension is filed under: one universal2 file, arm64 and
#: x86_64, for every Mac. `scripts/deploy.py`'s `PLAY_CROSS_BUILDS` writes it.
MACOS_UNIVERSAL = "macosx-universal2"


def extension_for(extensions: dict, here: str) -> str | None:
    """Which of a bundle's extensions `here` -- a `sysconfig.get_platform()` --
    loads: its own platform's, else on a Mac the universal one.

    **A Mac's platform never matches a key by itself.** It carries the
    deployment target its interpreter was built for -- `macosx-14.0-arm64` from
    Homebrew, `macosx-11.0-arm64` from conda, `macosx-10.9-universal2` from
    python.org -- and a bundle files its macOS extension once, without one.
    Looked up by the exact string, no Mac would ever find it, and `play --app`
    would say the app carries no controller for it.

    `scripts/deploy.py --check-reference` asks this too: the play host's check
    has to load what the play host loads.
    """
    if here in extensions:
        return here
    if here.startswith("macosx-") and MACOS_UNIVERSAL in extensions:
        return MACOS_UNIVERSAL
    return None


def _load_extension(bundle: Path, manifest: dict) -> tuple[Any, str]:
    """The bundle's own controller if it carries one, else whatever is installed.

    Prefer the bundle's, because otherwise this runs a *different build of the
    same source* -- whatever `mjrl_fsm` happens to be installed, which can be
    any age -- and the two diverge silently the moment somebody edits the crate
    and rebuilds one of them. A bundle that carries its own controller is the
    only way "what runs here is what would ship" is true rather than nearly true.

    Which one was loaded is always printed. An answer that depends on what is
    lying around should not have to be worked out.
    """
    # One bundle carries every host, and this one's controller as a native
    # extension per machine it was built on: `runtimes.mjlab.extensions`,
    # keyed by `sysconfig.get_platform()`.
    runtime = (manifest.get("runtimes") or {}).get("mjlab") or {}
    built = runtime.get("extensions") or {}
    if built:
        import sysconfig

        here = sysconfig.get_platform()
        key = extension_for(built, here)
        entry = built[key] if key else None
        path = bundle / entry["file"] if entry else None
        if path is None:
            print(f"[play] this app's controller is built for "
                  f"{', '.join(sorted(built))}; this is {here}")
        elif not path.is_file():
            print(f"[play] {entry['file']} is named by bundle.json and missing")
        else:
            import importlib.machinery
            import importlib.util
            import sys

            # The spec name is not free: CPython looks for `PyInit_<name>`, and
            # the crate's `#[pymodule]` is `mjrl_fsm`. Any other name fails with
            # "dynamic module does not define module export function", which
            # reads like a broken build rather than a naming rule.
            #
            # Not put into `sys.modules`, so it does not become "the" mjrl_fsm
            # for anything else in the process -- but an extension already
            # imported under that name would be returned from CPython's own
            # cache instead of this file, so that case defers to it and says so.
            if "mjrl_fsm" in sys.modules:
                cached = sys.modules["mjrl_fsm"]
                return cached, f"{cached.__file__} -- already imported, not this app's"
            # The loader is named rather than inferred from the suffix: `importlib`
            # matches a suffix against this interpreter's own list, which on
            # Windows is `.pyd` alone, and a file it does not recognise got no
            # spec -- and this fell through to the installed extension, saying
            # only that there was none.
            loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
            spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                commit = runtime.get("commit", "?")
                return module, f"{path.name} from the app ({commit[:12]})"

    try:
        import mjrl_fsm
    except ModuleNotFoundError as error:
        raise AppUnavailable(
            "this app carries no controller for this platform, and none is installed.\n"
            "       build the app on this machine: python scripts/deploy.py"
        ) from error
    return mjrl_fsm, f"{mjrl_fsm.__file__} -- installed, not this app's"


def _load_onnxruntime():
    try:
        import onnxruntime
    except ModuleNotFoundError as error:
        raise AppUnavailable(
            "an app's policies are ONNX, and onnxruntime is not installed.\n"
            "       pip install -e .   (it is a dependency; this environment "
            "predates that)"
        ) from error
    return onnxruntime


class ViewerKeys:
    """Keys as `mjrl.viewer.keys` reports them, handed on once a step.

    `report` runs on the viewer's UI thread and only queues the edge; `changes`
    runs on the simulation's, once a step, and is where a key goes down or comes
    up. The controller is driven from that one thread only.
    """

    def __init__(self, names: dict[int, str]) -> None:
        #: GLFW keycode -> the dictionary's key name.
        self._names = names
        #: Edges not yet handed on, in the order they came.
        self._queue: list[tuple[str, bool]] = []
        #: The keys the controller has been told are down.
        self._down: set[str] = set()
        self._lock = threading.Lock()

    def report(self, keycode: int, down: bool) -> None:
        name = self._names.get(keycode)
        if name is None:
            return
        with self._lock:
            self._queue.append((name, down))

    def changes(self) -> tuple[list[str], list[str]]:
        """The keys gone down and the keys come up since the last call, each in
        the order they did.

        A key that went down this step comes up at the next one at the
        earliest, and whatever it did after that waits with it: a press and a
        release in one step would otherwise be handed over as both at one
        instant, which a `toggle` or a `rise` never sees as held.
        """
        with self._lock:
            queue, self._queue = self._queue, []
        downs: list[str] = []
        ups: list[str] = []
        later: list[tuple[str, bool]] = []
        for name, down in queue:
            if name in downs or any(n == name for n, _ in later):
                later.append((name, down))
            elif down and name not in self._down:
                self._down.add(name)
                downs.append(name)
            elif not down and name in self._down:
                self._down.discard(name)
                ups.append(name)
        with self._lock:
            self._queue[:0] = later
        return downs, ups


class AppPlayer:
    """The app, wired to one running environment and to the person.

    Holds the controller, one ONNX session per mode, and the index maps between
    the simulator's joint order, its actuators' and the bundle's wire order --
    paired **by name**, since pairing by index is wrong from the fifth joint on
    and nothing raises.

    A policy for `play`'s loop: called once a control step, it runs the
    controller and returns an action of zeros, and the environment's action
    terms, taken over at construction, write the controller's targets instead.
    """

    def __init__(self, app: App, env: Any, *, pad: Any = None) -> None:
        """A key is handed to the controller stamped with the simulator's time,
        and a held key's stick deflection grows on that clock (`full_after_s`
        in the task's `controls.yaml`). At real time the two agree; at `--speed
        2`, or on a machine that cannot keep up, a key held a second of the
        person's time is held two seconds, or less, for the controller. `play`'s
        own operator times holds in the person's time; this one would need a
        second clock in the crate's operator to, and has none yet."""
        import torch

        from controller import vocabulary
        from mjrl.viewer import keys

        bundle = app.root
        manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
        mjrl_fsm, origin = _load_extension(bundle, manifest)
        ort = _load_onnxruntime()
        contracts = {
            mode: (bundle / entry["contract"]).read_text("utf-8")
            for mode, entry in manifest["modes"].items()
        }
        # A recorded motion's table travels beside its contract, and the
        # controller cannot build that mode without it.
        trajectories = {
            mode: (bundle / entry["trajectory"]).read_text("utf-8")
            for mode, entry in manifest["modes"].items()
            if entry.get("trajectory")
        }
        parsed = {mode: json.loads(text) for mode, text in contracts.items()}
        wire: list[str] = next(iter(parsed.values()))["wire_joint_order"]

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        sim_names = list(robot.joint_names)
        missing = [n for n in wire if n not in sim_names]
        if missing:
            raise AppUnavailable(f"this app drives joints this robot does not have: {missing}")
        self._wire_from_sim = [sim_names.index(n) for n in wire]

        # The clock's period and gate, from a contract that observes one, never
        # defaulted: a policy trained with `clock * (|command| > threshold)` is
        # shown `(0, 0)` while standing, and a free-running clock instead makes
        # it step in place.
        gait = next(
            (t.get("params") or {} for c in parsed.values()
             for t in c["observation"]["terms"] if t["name"] == "gait_phase"),
            {},
        )
        if abs(unwrapped.step_dt * app.control_hz - 1.0) > 1e-6:
            raise AppUnavailable(
                f"the world steps every {unwrapped.step_dt * 1e3:g} ms and this app "
                f"ticks at {app.control_hz:g} Hz, its fastest mode's rate: each "
                "mode would run at the world's rate instead of its own. "
                "`step_the_world_at` sets the world's"
            )
        limits = robot.data.joint_pos_limits[0].tolist()
        try:
            self._fsm = mjrl_fsm.Fsm(
                (bundle / "controller.toml").read_text("utf-8"),
                contracts,
                {
                    "joint_names": wire,
                    "joint_pos_lo": [limits[i][0] for i in self._wire_from_sim],
                    "joint_pos_hi": [limits[i][1] for i in self._wire_from_sim],
                    "output_rate_hz": app.control_hz,
                    "gait_period": gait.get("period"),
                    "gait_gate_threshold": gait.get("command_threshold"),
                },
                trajectories=trajectories,
                operators=True,
            )
        except TypeError as error:
            # A controller built before it read the person itself: an app
            # bundled before `play --app`, whose extension takes no `operators`.
            raise AppUnavailable(
                f"this app's controller ({origin}) predates `play --app` and cannot "
                f"read the keys itself ({error}).\n"
                "       rebuild the app: python scripts/deploy.py --manifest <name>"
            ) from error
        self._sessions = {
            mode: ort.InferenceSession(
                str(bundle / entry["models"]["onnx"]), providers=["CPUExecutionProvider"]
            )
            for mode, entry in manifest["modes"].items()
        }

        # The servos track the controller. Every action term is taken over, the
        # first to write the controller's targets on each physics substep -- as
        # the terms themselves do -- and the rest to write nothing, so no joint
        # is also driven by a decode of the zeros handed to the environment.
        self._env = env
        self._robot = robot
        self._targets = robot.data.default_joint_pos.clone()
        self._sim_ids = torch.tensor(self._wire_from_sim, device=unwrapped.device)
        manager = unwrapped.action_manager
        for i, name in enumerate(manager.active_terms):
            manager.get_term(name).apply_actions = self._apply if i == 0 else _nothing
        # And their gains: the controller's per joint and per regime -- the
        # policy's, a switch-in ramp's, a hook's arm at a preset, a hold's
        # damping alone -- by each actuator's own joint names.
        self._gains = []
        covered = set()
        for act in robot.actuators:
            if not hasattr(act, "set_gains"):
                raise AppUnavailable(
                    f"{type(act).__name__} takes no gains, and the controller sets them "
                    "every step; this robot's actuators are not ones an app can drive"
                )
            idx = [wire.index(n) for n in act.target_names]
            covered.update(idx)
            self._gains.append((act, idx))
        if len(covered) != len(wire):
            loose = sorted(set(wire) - {wire[i] for i in covered})
            raise AppUnavailable(f"no actuator drives {loose}")
        self._zero = torch.zeros(env.action_space.shape, device=unwrapped.device)

        # The person. Keys by the dictionary's names, from the GLFW codes the
        # viewer reports, down and up.
        self._viewer_keys = ViewerKeys(
            {c["glfw"]: name for name, c in vocabulary.key_codes().items()})
        set_aside = keys.claim(self._on_key)
        self._keys = keys
        self._pad = pad
        self._buttons = frozenset(b["name"] for b in vocabulary.load()["buttons"])
        self._tick = 0
        self._step_us = round(unwrapped.step_dt * 1e6)
        self._last_mode = ""

        print(f"[play] playing the app at {bundle}")
        print(f"[play]   controller: {origin}")
        # What the bundler said about it -- a mode still on ONNX, a missing
        # runtime -- is said again where somebody is looking.
        for note in manifest.get("notes", []):
            print(f"[play]   note: {note}")
        print(f"[play]   modes: {', '.join(sorted(self._sessions))}; "
              f"the world is this environment's")
        print("[play]   the person: the viewer's keys"
              + (" and the gamepad" if pad is not None else ", no gamepad")
              + (f" ({set_aside} other key reader(s) set aside)" if set_aside else ""))
        # The switches, the pad's and then the keyboard's: two paths since
        # 2026-09-29, a key never "the pad's A", so each is listed as itself.
        guide = json.loads(self._fsm.pad_guide())
        for device, lines, field in (("pad", guide.get("controls", []), "control"),
                                     ("keys", guide.get("keyboard", []), "stroke")):
            for line in lines:
                for s in (s for s in line["switches"] if not s["event"]):
                    how = (f"{s['with']} + " if s["with"] and device == "pad" else "") + line[field]
                    what = (f"leaves {'/'.join(s['leaves'])}" if s["leaves"]
                            else f"-> {s['enters']}")
                    where = f", from {'/'.join(s['from'])}" if s["from"] else ""
                    print(f"[play]   {device} {how} {what} on {s['on']}{where}")

    @property
    def fsm(self):
        return self._fsm

    def close(self) -> None:
        self._keys.unregister(self._on_key)
        if self._pad is not None and hasattr(self._pad, "close"):
            self._pad.close()

    def _apply(self) -> None:
        self._robot.set_joint_position_target(self._targets)

    def _on_key(self, keycode: int, down: bool) -> None:
        """The viewer's UI thread: queue the edge, and leave the controller alone.

        The controller is driven from the simulation's thread, once a step; a
        second thread calling into it would be two writers of one state.
        """
        self._viewer_keys.report(keycode, down)

    def _person(self, now_us: int) -> None:
        """Keys down and up since the last step, and a frame of the pad."""
        downs, ups = self._viewer_keys.changes()
        for name in downs:
            self._fsm.set_key(name, True, now_us)
        for name in ups:
            self._fsm.set_key(name, False, now_us)

        # Every step, pad or none: a frame is also what says the person is
        # there, and the controller lets go of a mode they stopped holding.
        state = self._pad.state() if self._pad is not None else None
        if state is None or not getattr(self._pad, "connected", True):
            self._fsm.set_pad_frame([], {}, connected=False, now_us=now_us)
            return
        self._fsm.set_pad_frame(
            sorted(b for b in state.buttons if b in self._buttons),
            {"Lx": state.lx, "Ly": state.ly, "Rx": state.rx, "Ry": state.ry,
             "LT": state.lt, "RT": state.rt},
            # The kernel reports the hat with down positive; the robot's pad
            # service negates it so up is +1, and the controller reads it so.
            dpad_x=int(state.hat_x),
            dpad_y=-int(state.hat_y),
            now_us=now_us,
        )

    def __call__(self, obs):
        import numpy as np
        import torch

        del obs  # the environment's own observation is not what the app sees
        unwrapped = self._env.unwrapped
        data = self._robot.data
        now = self._tick * self._step_us
        self._tick += 1

        def pick(row):
            return [float(row[i]) for i in self._wire_from_sim]

        self._fsm.set_state(
            pick(data.joint_pos[0]),
            pick(data.joint_vel[0]),
            pick(data.actuator_force[0]),
            [float(v) for v in data.root_link_quat_w[0]],
            # Already the body frame. A simulator that reported the world frame
            # would need rotating here; this one does not.
            [float(v) for v in data.root_link_ang_vel_b[0]],
            now,
        )
        self._person(now)

        mode = self._fsm.tick(now)
        if mode is not None:
            session = self._sessions[mode]
            action = session.run(
                None,
                {session.get_inputs()[0].name:
                 np.asarray([self._fsm.observation()], dtype=np.float32)},
            )[0][0]
            self._fsm.resume([float(v) for v in action])
        self._report()

        # What the controller publishes, into every environment: it read the
        # first, and the rest follow it.
        device = unwrapped.device
        n = unwrapped.num_envs
        self._targets[:, self._sim_ids] = torch.tensor(
            self._fsm.positions(), device=device).repeat(n, 1)
        kp, kd = self._fsm.kp(), self._fsm.kd()
        for act, idx in self._gains:
            act.set_gains(
                slice(None),
                kp=torch.tensor([kp[i] for i in idx], device=device).repeat(n, 1),
                kd=torch.tensor([kd[i] for i in idx], device=device).repeat(n, 1),
            )
        return self._zero

    def _report(self) -> None:
        """Say what changed, on its own line, without fighting the rate readout.

        `_loop` prints a rewritten line with no newline of its own, so anything
        here opens with one.
        """
        # Drained from the controller, not inferred from watching `mode` change:
        # the rule index is the answer to the only question an FSM raises, and
        # two rules leading to one state look identical from outside.
        for at, source, target, rule, why in self._fsm.take_log():
            print(f"\n[play] {at / 1e6:7.2f}s  {source} -> {target}   rule {rule}: {why}")
        mode = self._fsm.mode
        if mode != self._last_mode:
            state = "running" if self._fsm.running_policy else "holding"
            print(f"\n[play] mode: {mode} ({state})")
            self._last_mode = mode


def _nothing() -> None:
    """An action term taken over by the app writes nothing of its own."""
