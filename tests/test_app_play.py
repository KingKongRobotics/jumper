"""`play --app`: an app played in mjlab, and the ways that goes wrong quietly.

Three of them, each of which leaves a picture that looks like a robot running:

1. **The path.** `deploy.py --bundle` prints the build directory, which holds the
   bundle and its `.app` side by side and is neither; handed that, `--app` has to
   name the app inside rather than say only what is wrong. And the controller in
   the app is loaded by its path, not by what `importlib` makes of its suffix:
   on Windows a `controller.so` was a file with no spec, and every app fell
   through to whatever `mjrl_fsm` was installed. On a Mac it is the key: the
   platform string carries a version no bundle files its extension under.
2. **The keys.** MuJoCo's viewer reports a key's press alone, and
   `mjrl.viewer.keys` adds its release. A press and a release handed on at one
   instant are a toggle that never sees the key held, and two keys handed on in
   another order than they came are a chord read the wrong way round.
3. **The joints.** The whole point of `--app` over the `--fsm` it replaced is
   that the controller's own targets and gains drive the servos, so a task's
   deploy hook reaches the joints. Were the environment's action terms not taken
   over, it would decode the zeros it is handed into the home pose, and every
   mode would still look as if it were running.

The last is played end to end on the newest `jumper.app` under `out/` -- the
committed one on a clean checkout, a local build where there is one: the claw
mode entered on its key, the arm swung out on the d-pad and back. Skipped where
there is no app, no onnxruntime, or an app older than `play --app`.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
from mjrl.app_play import AppUnavailable, ViewerKeys, open_app

REPO = Path(__file__).resolve().parents[1]

CONTROLLER = """
[fsm]
initial_state = "walk"

[[fsm.state]]
name = "walk"
model = "walk.onnx"
task = "jumper.tripod"

[[fsm.rule]]
when = "always"
enter = "@initial"
"""


def what_bundle_writes(tmp_path: Path) -> Path:
    """What `--bundle` writes: a bundle and its archive beside it, and no bundle itself."""
    parent = tmp_path / "bundle_2026-01-01_00-00-00"
    bundle = parent / "jumper"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text(
        '{"modes": {"walk": {"contract": "walk.json"}}}\n', "utf-8")
    (bundle / "walk.json").write_text('{"control": {"control_hz": 50.0}}\n', "utf-8")
    (bundle / "controller.toml").write_text(CONTROLLER, "utf-8")
    with zipfile.ZipFile(parent / "jumper.app", "w") as app:
        for f in bundle.iterdir():
            app.write(f, f.name)
    return parent


# ── 1. The path ───────────────────────────────────────────────────────────


def test_an_app_is_unzipped_and_its_default_task_read(tmp_path: Path) -> None:
    """The `.app` itself, the way people have it: unzipped for the run, and the
    world's task read off the cascade's `always` rule -- through `@initial`
    here, which a reading of `enter` alone would have taken for a state name."""
    app = open_app(what_bundle_writes(tmp_path) / "jumper.app")
    try:
        assert (app.root / "bundle.json").is_file()
        assert app.default_task == "jumper.tripod"
        assert app.control_hz == 50.0
    finally:
        root = app.root
        app.close()
    assert not root.exists(), "the unzipped app outlived the run"


def test_the_bundle_directory_beside_it_plays_too(tmp_path: Path) -> None:
    app = open_app(what_bundle_writes(tmp_path) / "jumper")
    assert app.default_task == "jumper.tripod"
    app.close()
    assert (tmp_path / "bundle_2026-01-01_00-00-00/jumper/bundle.json").is_file(), (
        "closing an app that was not unzipped removed the bundle it was given"
    )


def test_the_build_directory_names_the_app_in_it(tmp_path: Path) -> None:
    parent = what_bundle_writes(tmp_path)
    with pytest.raises(AppUnavailable) as raised:
        open_app(parent)
    assert str(parent / "jumper.app") in str(raised.value)


def test_nothing_that_is_an_app_is_told_how_to_build_one(tmp_path: Path) -> None:
    with pytest.raises(AppUnavailable) as raised:
        open_app(tmp_path)
    # The build command, not `--bundle`: that only picks where an app goes.
    assert "python scripts/deploy.py --manifest" in str(raised.value)
    not_zip = tmp_path / "x.app"
    not_zip.write_text("not a zip", "utf-8")
    with pytest.raises(AppUnavailable, match="zip"):
        open_app(not_zip)


def test_the_app_s_controller_loads_whatever_its_suffix(tmp_path: Path, monkeypatch) -> None:
    """`importlib` picks an extension's loader by suffix, from this
    interpreter's own list -- `.pyd` alone on Windows. A file with another got
    no spec, and `play --app` fell through to the installed `mjrl_fsm`, saying
    the app carried none: on Windows, every app, whose extension was named
    `controller.so`.

    Here the same failure the other way round: this platform's extension under
    a suffix this interpreter does not use. The control shows `importlib` really
    declines that suffix here, so the load is not passing for a reason Windows
    would not share.
    """
    import importlib.util
    import sysconfig

    from mjrl.app_play import _load_extension

    app_path = _newest_app()
    if app_path is None:
        pytest.skip("no jumper.app under out/ to take a built controller from")
    platform = sysconfig.get_platform()
    foreign = "controller.so" if sys.platform == "win32" else "controller.pyd"
    source = open_app(app_path)
    try:
        built = json.loads((source.root / "bundle.json").read_text("utf-8"))
        entry = (built.get("runtimes", {}).get("mjlab") or {}).get("extensions", {}).get(platform)
        if entry is None:
            pytest.skip(f"{app_path} carries no controller for {platform}")
        renamed = tmp_path / "runtime/mjlab" / platform / foreign
        renamed.parent.mkdir(parents=True)
        shutil.copy2(source.root / entry["file"], renamed)
    finally:
        source.close()

    assert importlib.util.spec_from_file_location("mjrl_fsm", renamed) is None, (
        f"the control: {foreign} is a suffix importlib recognises here")
    monkeypatch.delitem(sys.modules, "mjrl_fsm", raising=False)
    manifest = {"runtimes": {"mjlab": {"commit": "test", "extensions": {
        platform: {"file": renamed.relative_to(tmp_path).as_posix()}}}}}
    module, origin = _load_extension(tmp_path, manifest)
    assert origin.startswith(f"{foreign} from the app"), origin
    assert hasattr(module, "Fsm")


def test_every_mac_finds_the_one_macos_controller() -> None:
    """A Mac's `sysconfig.get_platform()` carries the deployment target its
    Python was built for, and a bundle files its macOS extension once, without
    one -- so looked up by the exact string, no Mac finds it and `play --app`
    says the app carries no controller for it. Every Python a Mac has reported
    finds the universal one here.

    The control: the lookup is not a prefix match that hands anything the Mac
    file -- Linux and Windows get their own or nothing, a Mac gets nothing from
    a bundle with no macOS build, and a Mac that built the bundle itself gets
    its own exact build first.
    """
    from mjrl.app_play import extension_for

    built = {"linux-x86_64": {}, "win-amd64": {}, "macosx-universal2": {}}
    for mac in ("macosx-14.0-arm64", "macosx-11.0-arm64", "macosx-10.9-universal2",
                "macosx-10.13-x86_64"):
        assert mac not in built
        assert extension_for(built, mac) == "macosx-universal2", mac

    assert extension_for(built, "linux-x86_64") == "linux-x86_64"
    assert extension_for(built, "win-amd64") == "win-amd64"
    assert extension_for(built, "linux-aarch64") is None
    assert extension_for({"linux-x86_64": {}}, "macosx-14.0-arm64") is None
    assert extension_for({**built, "macosx-14.0-arm64": {}}, "macosx-14.0-arm64") \
        == "macosx-14.0-arm64"


def test_the_world_steps_at_the_app_s_fastest_mode() -> None:
    """The controller infers each mode when its period has passed and is
    ticked once a world step, so a world slower than the app's fastest mode
    runs that mode at the world's rate. The jumper app's posture mode is 200
    Hz; `jumper.five_foot`'s world steps at 50, and ran it at 50 with nothing
    said. `decimation` moves, the timestep does not -- and the 50 Hz app is the
    control group, left at the task's own 4."""
    from types import SimpleNamespace

    from mjrl.app_play import step_the_world_at

    def world() -> SimpleNamespace:
        return SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)),
                               decimation=4)

    for hz, decimation in ((200.0, 1), (50.0, 4), (100.0, 2)):
        cfg = world()
        step_the_world_at(cfg, hz)
        assert (cfg.decimation, cfg.sim.mujoco.timestep) == (decimation, 0.005), hz
    with pytest.raises(AppUnavailable, match="whole number"):
        step_the_world_at(world(), 300.0)


def test_a_fall_is_the_controller_s_before_it_is_the_world_s(tmp_path: Path) -> None:
    """The world reset a robot tilted past 50 degrees in the same step the
    controller would have called it tilted, past 50.02, so `tilted -> safe`
    never fired in `play --app`. Moved past the controller's limit; a
    termination already past it is the control group, and stays."""
    from types import SimpleNamespace

    from mjrl.app_play import FALL_MARGIN_RAD, leave_falls_to_the_app

    app = open_app(what_bundle_writes(tmp_path) / "jumper")
    (app.root / "controller.toml").write_text(
        CONTROLLER.replace('initial_state = "walk"', 'initial_state = "walk"\ntilt_limit = 0.873'),
        "utf-8",
    )
    world = SimpleNamespace(terminations={
        "fell_over": SimpleNamespace(params={"limit_angle": math.radians(50.0)}),
        "upside_down": SimpleNamespace(params={"limit_angle": math.radians(170.0)}),
        "time_out": SimpleNamespace(params={}),
    })
    assert leave_falls_to_the_app(world, app) == ["fell_over"]
    assert world.terminations["fell_over"].params["limit_angle"] == pytest.approx(0.873 + FALL_MARGIN_RAD)
    assert world.terminations["upside_down"].params["limit_angle"] == math.radians(170.0)


# ── 2. The keys ───────────────────────────────────────────────────────────


def test_a_key_is_down_from_its_press_to_its_release() -> None:
    """Held, a key goes down once and comes up once, however long it is held
    and however many steps read it; a key the dictionary does not name is
    nobody's. The release is the control group for the hold: the same reader
    that kept it down lets it up."""
    keys = ViewerKeys({71: "key_g"})
    keys.report(71, True)
    assert keys.changes() == (["key_g"], [])
    for _ in range(100):  # two seconds of steps
        assert keys.changes() == ([], []), "a held key read as a second press or a release"
    keys.report(71, False)
    assert keys.changes() == ([], ["key_g"])
    keys.report(999, True)  # a key the dictionary does not name
    assert keys.changes() == ([], [])


def test_a_tap_inside_one_step_is_down_for_that_step_and_up_at_the_next() -> None:
    """A press and its release between two steps. Handed over as both at one
    instant, a `toggle` or a `rise` would never see the key held; so the press
    is this step's and the release waits for the next. The control group is
    the same two edges a step apart, which come out the same -- the deferral
    only moves what would otherwise be lost."""
    keys = ViewerKeys({265: "key_up"})
    keys.report(265, True)
    keys.report(265, False)
    assert keys.changes() == (["key_up"], [])
    assert keys.changes() == ([], ["key_up"])
    assert keys.changes() == ([], [])

    keys.report(265, True)
    assert keys.changes() == (["key_up"], [])
    keys.report(265, False)
    assert keys.changes() == ([], ["key_up"])


def test_two_keys_down_in_one_step_go_down_in_the_order_they_did() -> None:
    """A chord is its modifier held and then its direction: `with` reads the
    modifier as down when the direction rises. Two keys pressed inside one
    step are handed on in the order they came, not by name -- by name,
    `key_left_ctrl` would always come first, and right-then-Ctrl would read as
    the chord."""
    keys = ViewerKeys({262: "key_right", 341: "key_left_ctrl"})
    keys.report(262, True)
    keys.report(341, True)
    assert keys.changes() == (["key_right", "key_left_ctrl"], [])


# ── 3. The joints, played ─────────────────────────────────────────────────


def _newest_app() -> Path | None:
    apps = sorted(REPO.glob("out/bundle_*/jumper.app"))
    return apps[-1] if apps else None


@pytest.fixture(scope="module")
def played():
    """The newest jumper.app under `out/`, played on the native backend: into
    the left claw on its key, then the arm held out thumb up with Shift, back to
    the stow when Shift is let go, and held out thumb down with Alt."""
    app_path = _newest_app()
    if app_path is None:
        pytest.skip("no jumper.app under out/: python scripts/deploy.py --bundle --manifest jumper")
    pytest.importorskip("onnxruntime", reason="an app's policies are ONNX")
    torch = pytest.importorskip("torch")
    import importlib

    from mjrl.app_play import AppPlayer
    from mjrl.backend.select import use_backend

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=1))
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    import tasks

    app = open_app(app_path)
    cfg = tasks.load_env_cfg(app.default_task, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=tasks.load_agent_cfg(app.default_task).clip_actions)
    try:
        player = AppPlayer(app, wrapped)
    except AppUnavailable as e:
        app.close()
        env.close()
        pytest.skip(f"{app_path} cannot be played: {e} -- "
                    "python scripts/deploy.py --bundle --manifest jumper")
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    arm = [names.index(f"LF_J{j}_joint") for j in range(4)]
    finger = names.index("LF_J4_joint")
    walking = names.index("LM_J1_joint")
    # V is the left claw's switch; in a claw mode Shift holds the arm out thumb
    # up and Alt thumb down, for as long as each is held (Control-agent 3.1's
    # layout, 2026-09-29; until then 1 and 2 were the d-pad, pressed once).
    codes = {"V": 86, "shift": 340, "alt": 342}
    obs = wrapped.get_observations()
    seen: dict[str, object] = {}

    def run(steps: int) -> None:
        nonlocal obs
        for _ in range(steps):
            with torch.inference_mode():
                obs, *_ = wrapped.step(player(obs))

    def press(key: str) -> None:
        # Held three quarters of a second and let go: a tap as a person makes one.
        player._on_key(codes[key], True)
        run(37)
        player._on_key(codes[key], False)
        run(2)

    def hold(key: str, steps: int) -> None:
        player._on_key(codes[key], True)
        run(steps)

    def let_go(key: str) -> None:
        player._on_key(codes[key], False)
        run(2)

    def snap(tag: str) -> None:
        seen[tag] = {
            "mode": player.fsm.mode,
            "running": player.fsm.running_policy,
            "arm": [float(robot.data.joint_pos[0, i]) for i in arm],
            "target": [float(robot.data.joint_pos_target[0, i]) for i in arm],
            "kp": [float(act.stiffness[0, act.target_names.index(names[i])])
                   for i in [*arm, walking] for act in robot.actuators if names[i] in act.target_names],
            "finger": float(robot.data.joint_pos_target[0, finger]),
            "height": float(robot.data.root_link_pos_w[0, 2]),
        }

    try:
        run(100)
        snap("start")
        press("V")
        # The switch-in ramp takes most of four seconds here (native CPU,
        # 2026-09-28) before the policy starts; the rest is the hook settling
        # the arm at the stow.
        run(250)
        snap("claw")
        hold("shift", 139)
        snap("up")
        let_go("shift")
        # Back from 90 degrees open to 60 the finger creeps: the hook's target
        # leads its measured angle by at most `SQUEEZE_LEAD` (0.06 rad) towards
        # shut, so the target reads 60 only once the finger is within 0.06 of
        # it. MEASURED (native CPU, 2026-09-29, five torch RNG states): that is
        # 100 to 105 steps after the preset is let go, and settled to 1e-4 by
        # 125 in every one. At 100 the snapshot sat on that edge, and which side
        # it fell depended on the RNG state this fixture started from -- so the
        # finger assertion failed whenever a test that drew random numbers ran
        # first. 200 is the settled value with a second to spare.
        run(200)
        snap("back")
        hold("alt", 139)
        snap("down")
        let_go("alt")
        yield seen
    finally:
        player.close()
        app.close()
        env.close()


def _deg(rad: list[float]) -> list[float]:
    return [math.degrees(v) for v in rad]


def test_a_key_enters_the_claw_mode_and_its_policy_runs(played) -> None:
    """V is the left claw's switch, the keyboard's own binding for the mode.
    Running, not holding: a switch-in ramp too soft to reach the claw's pose --
    the 0.15 kp this bundle shipped with until 2026-09-28 -- holds forever, with
    the base on the floor. The start is the control group: another mode."""
    assert played["start"]["mode"] == "locomotion"
    assert played["claw"]["mode"] == "claw_left" and played["claw"]["running"], played["claw"]
    assert played["claw"]["height"] > 0.08, f"the robot is down: {played['claw']['height']:.3f} m"


def test_the_hook_s_arm_preset_reaches_the_joints(played) -> None:
    """Shift held holds the arm out at rl-wbc-fsm's thumb-up preset, let go it
    stows, and Alt held holds it at thumb down. Only possible if the controller's targets are
    what the servos track: the environment's own decode would hold the stow
    whatever the hook said.

    Two readings. The servos' **target** is the preset to the degree -- the
    controller's output arrived. The arm **itself** is within the sag of a 3
    kp hold. The arm before the press is the control group -- at the stow, far
    from the preset.
    """
    up = [-90.0, -180.0, 0.0, -1.0]
    down = [-90.0, -30.0, 30.0, -1.0]
    stow = [-30.0, -90.0, -30.0, -75.0]
    def near(a, b, tol):
        return all(abs(x - y) < tol for x, y in zip(a, b, strict=True))

    assert near(_deg(played["claw"]["arm"]), stow, 5.0), _deg(played["claw"]["arm"])
    assert not near(_deg(played["claw"]["target"]), up, 20.0), "the preset was there before the press"
    assert near(_deg(played["up"]["target"]), up, 0.5), _deg(played["up"]["target"])
    assert near(_deg(played["up"]["arm"]), up, 10.0), _deg(played["up"]["arm"])
    assert near(_deg(played["back"]["arm"]), stow, 5.0), _deg(played["back"]["arm"])
    assert near(_deg(played["down"]["target"]), down, 0.5), _deg(played["down"]["target"])


def test_the_controller_s_gains_and_finger_reach_the_joints(played) -> None:
    """The gains are the controller's too: the hook's 3 on the arm, at the stow
    as at a preset -- the pitch follow holds it there for as long as the claw
    mode runs -- against the policy's 10 on a walking joint, which is the
    control group. And the finger is the hook's: 90 degrees open at a preset
    with the trigger let go, 60 at the stow."""
    for tag in ("claw", "up", "back"):
        assert played[tag]["kp"] == [3.0] * 4 + [10.0], (tag, played[tag]["kp"])
    closed = 0.10
    assert played["up"]["finger"] == pytest.approx(closed - math.radians(90), abs=1e-4)
    assert played["back"]["finger"] == pytest.approx(closed - math.radians(60), abs=1e-4)
