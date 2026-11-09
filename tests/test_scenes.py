"""The scene registry.

A scene is the ground, the sky and the light -- the axis orthogonal to the task.
The failures worth pinning here are the ones that produce a picture rather than an
error:

- a material whose `geom_names_expr` matches no geom is created, bound to nothing,
  and the ground renders exactly as before;
- a scene module nobody registered simply does not exist, with no hint that it was
  meant to.

Loading a scene imports `mjlab.utils.spec_config`, which is about 1.7 s and pulls
in no torch, so these run in the ordinary suite.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from scenes.rough import NUM_ROWS  # noqa: E402

import scenes
from scenes.registry import GROUND, Headlight, Scene

REPO = Path(__file__).resolve().parents[1]
SCENES_DIR = REPO / "scenes"


#: Modules under `scenes/` that are machinery rather than scenes. Listed rather
#: than pattern-matched: a scene that fails to register is exactly the failure the
#: check below exists for, and a loose rule would let one hide here.
#: Modules in `scenes/` that are not scenes. Shared generators live here rather
#: than in `tools/` because scene modules import them and `scenes/` is installed
#: while `tools/` is not -- see the note at the top of `skygen.py`.
#: `package` is machinery too, of a different kind: it turns an imported
#: `kk-scene-package/1` or `/2` into a `Scene` at run time, so the scenes it produces
#: are named by a path rather than by a registration. Its own checks are in
#: `tests/test_scene_package.py`.
_MACHINERY = {"__init__", "registry", "skygen", "hoardings", "package"}


def _modules() -> set[str]:
    """Scene modules on disk: everything but the machinery."""
    return {p.stem for p in SCENES_DIR.glob("*.py") if p.stem not in _MACHINERY}


# ── The registry matches the directory ──────────────────────────────────


def test_every_scene_module_is_registered() -> None:
    """A module nobody registered does not exist as far as `--scene` is concerned.

    The registration table is in `scenes/__init__.py` rather than in each module,
    because importing the modules to collect their metadata would defeat the point
    of keeping `--list` free of mjlab. The cost of that choice is exactly this
    failure mode, so it is checked here.
    """
    assert _modules() == set(scenes.list_ids()), (
        f"on disk: {sorted(_modules())}\nregistered: {sorted(scenes.list_ids())}"
    )


def test_every_registered_scene_loads() -> None:
    """Metadata pointing at something that cannot be built is worse than no entry:
    it shows up in `--list` and fails only when someone selects it."""
    for spec in scenes.all_specs():
        scene = scenes.load(spec.id)
        assert isinstance(scene, Scene), f"{spec.id} did not return a Scene"
        assert spec.description, f"{spec.id} has no description; --list would show a blank"


def test_unknown_scene_lists_the_available_ones() -> None:
    """A typo should answer the question it raises, rather than a bare KeyError."""
    with pytest.raises(KeyError, match="unknown scene"):
        scenes.get("sunset-beach")  # a hyphen where the id has an underscore
    try:
        scenes.get("nope")
    except KeyError as e:
        assert "beach" in str(e), "the message must list what is available"


def test_registering_twice_raises() -> None:
    """Silently overwriting would make which scene you get depend on import order."""
    with pytest.raises(ValueError, match="registered twice"):
        scenes.register(id="studio", use="watching")


def test_a_scene_has_to_say_what_it_is_for() -> None:
    """`use` decides whether the scene is handed to someone outside this
    repository, and both defaults are wrong in a way that looks right: exported
    ships a curriculum grid as a playground, withheld means a new scene quietly
    never arrives. So there is no default -- see `SceneSpec.use`."""
    with pytest.raises(TypeError, match="use"):
        scenes.register(id="whatever")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="must be one of"):
        scenes.register(id="whatever", use="yes")


def test_every_scene_has_decided_what_it_is_for() -> None:
    """A registration that skipped it cannot exist -- `register` raises -- but a
    value outside the vocabulary could arrive by an edit to the dataclass, and it
    would leave the exporter's one rule silently false."""
    from scenes.registry import USES

    undecided = {s: scenes.get(s).use for s in scenes.list_ids() if scenes.get(s).use not in USES}
    assert not undecided, undecided


# ── The failure that renders a picture ──────────────────────────────────


def test_ground_materials_can_actually_bind() -> None:
    """A ground material has to match the geom the ground actually is.

    mjlab names the flat ground geom `terrain`, and a material that misses it is
    created and bound to nothing -- no error, and the ground renders unchanged.
    `GROUND` is the single expression that matches it, so scenes are required to
    use it rather than spelling their own.
    """
    for spec in scenes.all_specs():
        scene = scenes.load(spec.id)
        if scene.terrain_type == "generator":
            continue  # covered by the next test
        source = (REPO / "scenes" / f"{spec.id}.py").read_text(encoding="utf-8")
        for mat in scene.materials:
            if mat.geom_names_expr:
                assert tuple(mat.geom_names_expr) == GROUND, (
                    f"scene {spec.id!r} material {mat.name!r} targets "
                    f"{mat.geom_names_expr}, which will not match the ground geom"
                )
            else:
                # The other legitimate case: bound by name to geometry the scene
                # adds itself, like the pitch's turf slab. An expression cannot
                # reach that geom -- it does not exist when the material is made.
                assert f'material="{mat.name}"' in source, (
                    f"scene {spec.id!r} material {mat.name!r} matches no geom and "
                    f"is referenced by no geom the scene adds: it will be created, "
                    f"bound to nothing, and change nothing"
                )


def test_generated_terrain_scenes_do_not_set_a_ground_material() -> None:
    """With a generator the ground is `terrain_0`, `terrain_1`, ... not `terrain`.

    A `GROUND` material binds to none of them. Matching them instead would bind,
    and would paint over the per-tile textures the generator derives from each
    tile's own height data -- which is the cue that makes relief readable. So the
    right answer is to set none, and this keeps a well-meaning edit from adding
    one back. Measured: with a `GROUND` material the scene compiled with 81 tiles
    bound to the generator's materials and the scene's own bound to nothing.
    """
    for spec in scenes.all_specs():
        scene = scenes.load(spec.id)
        if scene.terrain_type == "generator":
            assert scene.materials == (), (
                f"scene {spec.id!r} sets a ground material under generated terrain"
            )


def test_every_scene_lights_the_world() -> None:
    """No lights renders black, which reads as a broken model rather than a scene
    with nothing in it."""
    for spec in scenes.all_specs():
        assert scenes.load(spec.id).lights, f"scene {spec.id!r} has no lights"


# ── apply ───────────────────────────────────────────────────────────────


class _Terrain:
    def __init__(self) -> None:
        self.textures = ("original",)
        self.materials = ("original",)
        self.lights = ("original",)
        self.terrain_type = "plane"
        self.terrain_generator = None


class _Cfg:
    def __init__(self, sensors=()) -> None:
        self.scene = type(
            "S", (), {"terrain": _Terrain(), "sensors": sensors, "entities": {}}
        )()
        self.events: dict = {}


def test_apply_writes_the_scene_onto_the_terrain() -> None:
    cfg = _Cfg()
    scenes.apply(cfg, "studio")
    t = cfg.scene.terrain
    assert t.textures and t.textures != ("original",)
    assert t.lights and t.lights != ("original",)
    assert t.terrain_type == "plane", "a look-only scene must not touch the ground"


def test_apply_leaves_the_terrain_type_alone_unless_the_scene_sets_it() -> None:
    """`terrain_type=None` means "keep the task's choice", which is why it is the
    one field that can opt out: the task configured the ground it was designed
    for."""
    cfg = _Cfg()
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = "kept"
    scenes.apply(cfg, "daylight")
    assert cfg.scene.terrain.terrain_type == "generator"
    assert cfg.scene.terrain.terrain_generator == "kept"


def test_apply_says_so_when_there_is_no_terrain_to_apply_to() -> None:
    with pytest.raises(ValueError, match="no scene.terrain"):
        scenes.apply(type("C", (), {"scene": None})(), "studio")


# ── Wiring ──────────────────────────────────────────────────────────────


def test_scene_selection_is_wired_to_the_cli_and_env() -> None:
    """`MJRL_CPU_THREADS` was once parsed, computed, printed and never passed on.
    A config that lies is worse than no config."""
    cli = (REPO / "scripts" / "_cli.py").read_text(encoding="utf-8")
    assert '_env("MJRL_SCENE")' in cli
    assert "def apply_scene(" in cli
    assert "MJRL_SCENE=" in (REPO / ".env").read_text(encoding="utf-8")

    # Only where it can do something. export.py writes an ONNX file and a
    # contract, and nothing a scene changes reaches either; it took --scene from
    # the shared parser for a while and silently ignored it, which reads as
    # supported. `--headless` is absent from export for the same reason.
    export = (REPO / "scripts" / "export.py").read_text(encoding="utf-8")
    assert "add_scene_args" not in export, (
        "export.py must not accept a flag it cannot act on"
    )

    for entry in ("train.py", "play.py"):
        src = (REPO / "scripts" / entry).read_text(encoding="utf-8")
        assert "add_scene_args(parser)" in src, f"{entry} never offers --scene"
        assert "apply_scene(env_cfg, args" in src, f"{entry} never applies the scene"
        assert src.index("load_env_cfg") < src.index("apply_scene(env_cfg"), (
            f"{entry} must apply the scene after the task has configured itself"
        )

    # An imported scene package is refused for training, and the refusal reaches
    # both the validation pass -- so `--dry-run` sees it -- and the application.
    # Neither call site is optional: one room at the world origin trains 4095
    # robots on empty ground, and nothing in a training run reports that.
    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    play = (REPO / "scripts" / "play.py").read_text(encoding="utf-8")
    for call in ("resolve_all(args", "apply_scene(env_cfg, args"):
        assert f"{call}, for_training=True)" in train, (
            f"train.py must tell {call.split('(')[0]} it is training"
        )
        assert "for_training" not in play, "play.py accepts imported scene packages"


def test_a_misspelled_scene_is_caught_by_dry_run() -> None:
    """`--dry-run` exists to check the arguments, so it has to check this one.

    The scene is applied long after resolution, once the environment config
    exists. Validating it there only would let a typo past the very check meant to
    find it -- `--dry-run` would print a resolved run and exit 0. So the id is
    looked up during resolution, where `--model` is already resolved, and applied
    later.
    """
    cli = (REPO / "scripts" / "_cli.py").read_text(encoding="utf-8")
    body = cli.split("def resolve_all(", 1)[1].split("\n# ", 1)[0]
    assert "scenes.get(args.scene)" in body, (
        "resolve_all must validate --scene, or --dry-run cannot catch a typo"
    )


# ── The ambient floor ───────────────────────────────────────────────────


def test_every_scene_chooses_its_ambient() -> None:
    """The camera-attached headlight sets how dark a shadow can get.

    Leaving it to whatever the model brings means the scene's own lights are
    competing with an invisible flood light: mjlab's template ships 0.3 ambient
    and 0.6 diffuse, three times MuJoCo's own default, and at that strength
    halving a fill light changes almost nothing. Measured while trying exactly
    that, twice, before finding where the light was coming from.
    """
    for spec in scenes.all_specs():
        light = scenes.load(spec.id).headlight
        assert isinstance(light, Headlight), (
            f"scene {spec.id!r} leaves the ambient to whatever the model brings"
        )


def test_the_default_scene_still_reproduces_mjlab() -> None:
    """`--scene default` is the baseline to diff against, so it has to match.

    Tied to the vendored template rather than to a literal, so that a value
    changing there is noticed here instead of silently making "default" mean
    something else.
    """
    import re

    xml = (REPO / "rl" / "mjlab" / "scene" / "scene.xml").read_text(encoding="utf-8")
    line = next(ln for ln in xml.splitlines() if "<headlight" in ln)
    got = {k: tuple(float(x) for x in v.split())
           for k, v in re.findall(r'(\w+)="([\d.\s]+)"', line)}
    light = scenes.load("default").headlight
    assert light is not None
    assert light.ambient == pytest.approx(got["ambient"])
    assert light.diffuse == pytest.approx(got["diffuse"])


def test_the_headlight_goes_through_spec_fn() -> None:
    """Nothing earlier than `spec_fn` survives.

    The value comes from mjlab's `scene/scene.xml`, which is the **parent** spec,
    so a `<visual>` block in an attached entity is overridden without a word --
    measured: deleting the headlight from the jumper asset changed the compiled
    model not at all. `spec_fn` runs after everything is attached.
    """
    cfg = _Cfg()
    cfg.scene.spec_fn = None
    scenes.apply(cfg, "beach")
    assert callable(cfg.scene.spec_fn), "the scene never installed a spec_fn"

    class _Spec:
        class visual:
            class headlight:
                ambient = diffuse = specular = None

    spec = _Spec()
    cfg.scene.spec_fn(spec)
    assert spec.visual.headlight.ambient == list(
        scenes.load("beach").headlight.ambient
    )


def test_a_task_s_own_spec_fn_is_kept() -> None:
    """A task installing one has its own reason; the scene only claims lighting."""
    calls = []
    cfg = _Cfg()
    cfg.scene.spec_fn = lambda spec: calls.append("task")
    scenes.apply(cfg, "studio")

    class _Spec:
        class visual:
            class headlight:
                ambient = diffuse = specular = None

    cfg.scene.spec_fn(_Spec())
    assert calls == ["task"], "the task's callback was dropped"


def test_the_sun_in_the_sky_matches_the_sun_that_lights_the_scene() -> None:
    """The skybox draws the sun; a light casts its shadows. They are two numbers.

    They have to be negations of each other -- one points towards the sun, the
    other is the direction the light travels. Drift between them puts the bright
    part of the sky somewhere other than where the shadows say the sun is, and
    nothing anywhere reports it: both halves look entirely reasonable on their
    own. Lowering the sun means editing both files, which is exactly when this
    goes wrong.
    """
    import ast
    import math

    src = (REPO / "scenes" / "beach.py")
    line = next(ln for ln in src.read_text(encoding="utf-8").splitlines()
                if ln.startswith("SUN = "))
    towards = ast.literal_eval(line.split("np.array(", 1)[1].rstrip(")"))

    sun_light = next(light for light in scenes.load("beach").lights if light.name == "sun")
    travels = sun_light.dir

    for a, b in zip(towards, travels, strict=True):
        assert a == pytest.approx(-b, abs=1e-3), (
            f"the skybox's sun {towards} is not the negation of the light's "
            f"direction {travels}"
        )

    elevation = math.degrees(math.asin(towards[2] / math.dist((0, 0, 0), towards)))
    assert 0.0 < elevation < 20.0, (
        f"the sun is at {elevation:.1f} degrees; above the horizon and low is what "
        f"makes the shadows long, which is the point of this scene"
    )


# ── Scenes that add geometry ────────────────────────────────────────────


def test_scene_geometry_cannot_be_collided_with() -> None:
    """A scene may add geometry, and none of it may touch the robot.

    `--scene` chooses a look. A goalpost the robot can walk into changes the task
    it is being trained on -- for every task the scene is selected on, silently,
    with the reward and the observation unchanged and only the world quietly
    different. Solid geometry is a decision for a task that wants a pitch.

    The check is the rule in the sentence above, not the stricter proxy it used to
    be. It asked for `contype == 0 and conaffinity == 0` -- **cannot collide with
    anything** -- which is one way to satisfy the rule and not the only one. The
    football pitch's perimeter boards are solid to the ball and to nothing else,
    on a channel of their own, and they are exactly as unable to touch a robot as a
    geom with no masks at all.

    So the assertion is MuJoCo's own pair test against a **conventional** geom,
    `contype=1, conaffinity=1` -- what the terrain uses, what a robot's links carry
    (mjlab reserves bit 0 for the terrain channel and robots keep it), and what
    anything not deliberately doing something else will have. A goalpost with the
    default masks still fails, which is the case this test exists for.
    """
    import mujoco

    from scenes.football import _draw_pitch

    spec = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><geom type='plane' size='40 40 .1'/></worldbody></mujoco>"
    )
    _draw_pitch(spec)
    added = [g for g in spec.geoms if g.name]
    assert added, "the pitch drew nothing"

    # A stand-in for anything that is not scenery: the robot, the terrain, a prop
    # that did not ask for a private channel.
    default_contype = default_conaffinity = 1
    solid = [
        g.name
        for g in added
        if (g.contype & default_conaffinity) or (default_contype & g.conaffinity)
    ]
    assert not solid, f"these would collide with the robot: {solid}"


def test_the_perimeter_boards_stop_the_ball_and_nothing_else() -> None:
    """The boards keep the ball on the pitch without becoming a wall.

    Two halves, and the term is worthless without both: a board the ball passes
    through is not a board, and a board the *robot* cannot pass through is the
    silent task change the test above exists to prevent.
    """
    import mujoco

    from scenes.football import (
        BALL_BIT,
        BOARD_GAP,
        LENGTH,
        WIDTH,
        _ball_spec,
        _boards,
    )

    spec = mujoco.MjSpec()
    _boards(spec)
    boards = {g.name: g for g in spec.geoms if g.name.startswith("board_")}
    assert set(boards) == {"board_n", "board_s", "board_e", "board_w"}

    # The spec has to stay in a name: dropping it frees the underlying C object
    # and the geom list comes back empty rather than raising.
    ball_spec = _ball_spec()
    ball = ball_spec.worldbody.bodies[0].geoms[0]

    def collides(a, b) -> bool:
        return bool(a.contype & b.conaffinity) or bool(b.contype & a.conaffinity)

    robot_leg = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><geom name='leg' type='sphere' size='.1' "
        "contype='3' conaffinity='124'/></worldbody></mujoco>"
    ).geoms[0]

    for name, board in boards.items():
        assert collides(board, ball), f"{name} would let the ball through"
        assert not collides(board, robot_leg), f"{name} is a wall to the robot"
    assert ball.contype & 1, "the ball must keep bit 0 or it falls through the ground"
    assert ball.contype & BALL_BIT, "the ball must carry the boards' channel"

    # The enclosure is closed and sits the specified distance out. Read from the
    # geometry rather than recomputed from the constants, so that moving a board
    # without moving its neighbour is caught.
    inner_x = LENGTH / 2 + BOARD_GAP
    inner_y = WIDTH / 2 + BOARD_GAP
    assert boards["board_e"].pos[0] - boards["board_e"].size[0] == inner_x
    assert boards["board_n"].pos[1] - boards["board_n"].size[1] == inner_y
    # Corners meet: the long boards reach at least as far as the outer face of the
    # short ones, so there is no gap to escape through.
    assert boards["board_n"].size[0] >= boards["board_e"].pos[0] + boards["board_e"].size[0]


def test_the_pitch_is_a_real_pitch_divided_by_five() -> None:
    """Every dimension is a real one over five, and they have to stay in step.

    A pitch whose centre circle does not match its penalty area is not a smaller
    pitch, it is a wrong one -- and nothing about it looks wrong until you know
    what a pitch looks like.
    """
    from scenes import football as fp

    assert (fp.LENGTH, fp.WIDTH) == pytest.approx((105.0 / 5, 68.0 / 5))
    assert fp.CIRCLE_R == pytest.approx(9.15 / 5)
    assert (fp.PENALTY_W, fp.PENALTY_D) == pytest.approx((40.32 / 5, 16.5 / 5))
    assert (fp.GOAL_W, fp.GOAL_H) == pytest.approx((7.32 / 5, 2.44 / 5))
    assert fp.PENALTY_SPOT == pytest.approx(11.0 / 5)
    # The goal has to fit inside the goal area, which has to fit inside the
    # penalty area: the cheapest check that the ratios were not each edited alone.
    assert fp.GOAL_W < fp.GOAL_AREA_W < fp.PENALTY_W < fp.WIDTH
    assert fp.GOAL_AREA_D < fp.PENALTY_D < fp.LENGTH / 2


def test_the_sky_generators_share_one_face_table() -> None:
    """Which world direction each cube face covers was measured, not derived.

    Two scenes generate a sky now. A copy of that table in each is a copy that
    can drift, and a drifted one still compiles and still looks like a sky --
    with the sun in the wrong quarter. `tools/skygen.py` holds the one copy.
    """
    import importlib

    for scene_id in ("beach", "football"):
        module = importlib.import_module(f"scenes.{scene_id}")
        maps = getattr(module, "CUBEMAPS", None)
        assert maps and "sky" in maps, f"{scene_id} declares no sky in CUBEMAPS"
        text = (REPO / "scenes" / f"{scene_id}.py").read_text(encoding="utf-8")
        assert "from . import skygen" in text, f"{scene_id} does not use skygen"
        assert "FACES = {" not in text, f"{scene_id} keeps its own copy of the face table"


def test_the_skybox_driver_covers_every_sky() -> None:
    """`tools/make_skybox.py --all` has to find them by the same convention.

    The PNGs are committed rather than built on demand, so a sky whose colours
    changed and whose faces were not re-rendered goes on loading the old ones --
    silently, since the scene is perfectly valid either way.
    """
    src = (REPO / "tools" / "make_skybox.py").read_text(encoding="utf-8")
    assert 'getattr(module, "CUBEMAPS", None)' in src
    assert "import sys" in src and "sys.path" not in src, (
        "the driver must not rewrite sys.path -- see test_no_sys_path_mutation"
    )


# ── Props: the things a scene adds that are physical ────────────────────


def test_a_scene_with_props_says_so() -> None:
    """A prop has mass and it collides, so the scene is no longer only a look.

    Contacts, contact count and solver load all change because it is there, while
    the observation and the reward are untouched -- so a policy trained with it is
    not comparable to one trained without, and nothing else would ever mention it.
    Same treatment as generated terrain, for the same reason.
    """
    with pytest.warns(RuntimeWarning, match="physical prop"):
        cfg = _Cfg()
        scenes.apply(cfg, "football")
    assert "ball" in cfg.scene.entities

    # The control: a look-only scene must not warn about anything.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scenes.apply(_Cfg(), "studio")


def test_every_prop_gets_a_reset_event() -> None:
    """An entity with no reset event stays at (0, 0, 0) in **every** environment.

    mjlab applies `env_origins` inside `reset_root_state_uniform`, and nowhere
    else -- its own docstring says so. Measured before this was wired: one ball,
    at the world origin, half-buried in the pitch, reachable only by whichever
    robot happened to be near it, while the config said 0.45 m in front of each.

    So the event is installed by `apply` rather than left to whoever writes a
    prop, because the config reads correct either way.
    """
    cfg = _Cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenes.apply(cfg, "football")
    assert "reset_prop_ball" in cfg.events, "the ball has no reset event"
    assert cfg.events["reset_prop_ball"].mode == "reset"


def test_the_prop_reset_does_not_repeat_the_placement() -> None:
    """The pose range must be empty, not the prop's position.

    The entity's own `init_state` already supplies where it goes; passing it
    again as the reset's centre adds it twice. Measured: a ball configured at
    0.45 m appearing at 0.90 m, which looks like a deliberate choice rather than
    a bug.
    """
    cfg = _Cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scenes.apply(cfg, "football")
    assert cfg.events["reset_prop_ball"].params["pose_range"] == {}


def test_a_prop_cannot_replace_the_robot() -> None:
    """Names collide silently otherwise, and the loser is whatever the task
    called its robot."""
    cfg = _Cfg()
    cfg.scene.entities["ball"] = "the task's own"
    with pytest.raises(ValueError, match="overwrite the task"):
        scenes.apply(cfg, "football")


def test_the_ball_is_a_ball() -> None:
    """It has to roll, which is the one thing about a football that is instantly
    wrong when it is missing -- a sliding ball reads as a puck.

    The check is rolling itself, simulated, rather than `condim == 3` as it was
    written first. That was a proxy, and it turned out to be the *wrong* one:
    `condim=3` rolls, but it also means MuJoCo never evaluates the rolling
    friction coefficient sitting right beside it, so the ball coasted 8.5 m from a
    1 m/s tap on a pitch 21 m long. `condim=6` rolls too and makes the coefficient
    real. A test pinned to the number would have made that fix look like a
    regression.
    """
    import mujoco

    from scenes.football import (
        BALL_MASS,
        BALL_RADIUS,
        ROLLING_FRICTION,
        _ball_spec,
    )

    spec = _ball_spec()
    geom = next(g for g in spec.geoms if g.name == "ball")
    assert geom.condim >= 4, (
        "rolling and torsional friction are only evaluated at condim 4 or 6, so "
        "anything less makes geom.friction[1:] dead numbers"
    )
    assert geom.friction[1] < 0.01, "torsional friction this high stops it rolling"
    assert geom.friction[2] > 0.0, "no rolling resistance: it never stops"

    # Half a real ball, and the mass scales with **area** rather than volume: a
    # football is a thin shell, so most of its 430 g is skin. Volume scaling
    # would give 54 g, which bounces like a beach ball.
    assert BALL_RADIUS == pytest.approx(0.11 / 2)
    assert BALL_MASS == pytest.approx(0.43 / 4)

    # Kick it and watch. Two properties, and the term is wrong without either: it
    # must *roll* (v = omega * r, not slide) and it must *stop* (a ball that
    # coasts the length of the pitch is the bug this replaced).
    world = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><geom type='plane' size='0 0 1' "
        "friction='0.9 0.005 0.0001'/></worldbody></mujoco>"
    )
    body = world.worldbody.add_body(name="ball", pos=[0, 0, BALL_RADIUS])
    body.add_freejoint()
    body.add_geom(
        name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0],
        mass=BALL_MASS, condim=geom.condim, friction=list(geom.friction),
        solref=list(geom.solref),
    )
    model = mujoco.MjModel_ = world.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.0, 0.0, BALL_RADIUS]
    data.qvel[0] = 2.0

    # Rolling means the spin carries the motion: for travel along +x the spin is
    # about +y, and `omega * r` should be most of `v`.
    #
    # Most, not all. Measured, the ratio runs 0.84 -> 0.91 as it slows and never
    # reaches 1: rolling resistance is a torque, so it holds the spin slightly
    # behind the speed for as long as the ball is moving. An earlier version of
    # this test demanded `v == omega * r` to within 5% and failed on a ball that
    # was rolling perfectly well. Half is the line that separates rolling from a
    # puck, which slides with almost no spin at all.
    rolling_seen = False
    for _ in range(int(6.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        v, omega = data.qvel[0], data.qvel[4]
        if v > 0.3 and omega * BALL_RADIUS > 0.5 * v:
            rolling_seen = True
        if abs(v) < 0.02:
            break

    assert rolling_seen, "it slid instead of rolling -- that reads as a puck"
    assert abs(data.qvel[0]) < 0.05, "still moving after 6 s: it never stops"
    # The bound is loose on purpose: this pins "it stops on the pitch", not the
    # particular feel, which is a number meant to be turned. At 0.005 a 2 m/s
    # strike measures 1.9 m; the failure being guarded against is the 10.3 m of a
    # ball with no rolling resistance at all.
    assert data.qpos[0] < 4.0, (
        f"a 2 m/s kick carried it {data.qpos[0]:.1f} m on a pitch 21 m long. At "
        f"{ROLLING_FRICTION} rolling friction it should be about 2 -- this is what "
        f"a coefficient that is not being evaluated looks like"
    )


# ── The swing ───────────────────────────────────────────────────────────


def _jumper_stance() -> dict[str, tuple[float, float]]:
    """The six feet's (x, y) at `HOME`, computed from the asset rather than read
    from `scenes/swing.py`. That independence is the whole point of the test
    below."""
    import mujoco

    from tasks.jumper.common.constants import FEET, HOME, STAND_Z, get_spec

    model = get_spec().compile()
    data = mujoco.MjData(model)
    for joint, angle in HOME.items():
        data.qpos[model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        ]] = angle
    data.qpos[:3] = [0.0, 0.0, STAND_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    return {
        foot: tuple(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)][:2])
        for foot in FEET
    }


def test_the_plank_is_still_the_size_of_the_robot_that_stands_on_it() -> None:
    """The plank is derived from a measurement of the robot, and the robot moves.

    `assets/jumper/jumper.xml` is **replaced in place** when a new revision of the
    hardware lands -- that is the project's convention, not an accident -- and a
    revision that widens the stance leaves a plank sized for the previous one. The
    failure is entirely silent: the scene loads, the swing renders, the robot
    stands on it, and one foot is over the edge.

    So the stance is recomputed here from the asset and checked against the two
    constants the plank is built from. If this fails, re-measure and update
    `scenes/swing.py` -- do not widen the tolerance.
    """
    from scenes import swing

    feet = _jumper_stance()
    xs = [p[0] for p in feet.values()]
    ys = [p[1] for p in feet.values()]

    assert max(xs) - min(xs) == pytest.approx(swing.STANCE_X, abs=1e-3), (
        f"the robot's stance is {max(xs) - min(xs):.4f} m fore-aft, and the plank "
        f"is built for {swing.STANCE_X:.4f}"
    )
    assert max(ys) - min(ys) == pytest.approx(swing.STANCE_Y, abs=1e-3)
    assert (max(xs) + min(xs)) / 2 == pytest.approx(swing.STANCE_CENTRE_X, abs=1e-3)

    # And the plank really holds it, with the base placed where `STANCE_CENTRE_X`
    # says. This is the assertion the two above exist to keep honest: it is what
    # the scene claims, stated in the plank's own coordinates.
    #
    # The 1 mm slack is the rounding in the quoted constants, not slack in the
    # claim: the foot positions in `swing.py` are written to 0.1 mm, so a middle
    # foot at 0.2000314 m sits 31 micrometres outside a margin derived from
    # 0.2000 -- which is arithmetic, against a 50 mm margin.
    for foot, (x, y) in feet.items():
        clear_x = swing.BOARD_X / 2 - abs(x - swing.STANCE_CENTRE_X)
        clear_y = swing.BOARD_Y / 2 - abs(y)
        assert clear_x >= swing.FOOT_MARGIN - 1e-3, f"{foot}: {clear_x:.4f} m of plank"
        assert clear_y >= swing.FOOT_MARGIN - 1e-3, f"{foot}: {clear_y:.4f} m of plank"


def mujoco_name(model, body: int) -> str:
    """A body's name, for an assertion message. Index alone is unreadable."""
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body)


def _swing_model(tie_x: float | None = None):
    """Compile the swing on its own, over flat ground, at the simulation's step."""
    import mujoco

    from scenes import swing

    spec = swing._swing_spec(tie_x)
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05]
    )
    return spec.compile()


def _seat_pitch(model, data) -> float:
    """The plank's pitch in degrees, from its rotation matrix rather than from the
    hinge's `qpos` -- past 90 degrees the Euler parameterisation keeps counting and
    `qpos` reads 1539 degrees, which is a plank that has been round four times."""
    import mujoco
    import numpy as np

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seat")
    return float(np.degrees(np.arcsin(-np.clip(data.xmat[body].reshape(3, 3)[2, 0], -1, 1))))


def test_the_seat_hangs_taut_and_free_of_its_own_frame() -> None:
    """At rest every rope is exactly its own length and nothing is touching.

    Two failures in one, and both are quiet. A rig built with the wrong rope
    lengths is *pre-stretched*: it compiles, it hangs, and it sits a few
    millimetres low with every constraint permanently active, which shows up much
    later as a swing that damps oddly. And a seat resting against its own frame is
    a permanent contact that reads as too much damping rather than as geometry --
    that one was measured on the rigid-rod version this replaced, where the hangers
    were hinged inside the beam and a swing released at 0.35 rad came back at 0.09.
    """
    import mujoco
    import numpy as np

    model = _swing_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.ntendon == 4, f"four ropes, not {model.ntendon}"
    assert data.ncon == 0, "something is touching before the swing has even moved"
    assert data.ten_length == pytest.approx(model.tendon_range[:, 1], abs=1e-6), (
        f"the rig is not at its rest length: {np.round(data.ten_length, 5)} against "
        f"{np.round(model.tendon_range[:, 1], 5)}"
    )
    assert _seat_pitch(model, data) == pytest.approx(0.0, abs=0.1), "the plank starts tilted"


def test_a_rope_pulls_and_does_not_push() -> None:
    """The whole reason the hangers are tendons and not rods.

    Lift the plank until every rope is slack and let go: a rope contributes
    nothing, so the seat falls. A rod would hold it up there, and *that* is the
    difference the robot feels the first time it pushes off -- a rod shoves the
    plank up after it and a rope lets it drop away.

    Checked as free fall rather than as `limited == True`, which is the property
    and not the flag. It falls slightly faster than `g`: the knots are hanging
    below their own slack upper ropes, so their weight is on the taut bridle and
    rides down on the plank.
    """
    import mujoco

    model = _swing_model()
    data = mujoco.MjData(model)
    data.qpos[2] = 0.06  # seat_slide_z: 60 mm up, well above every rope's reach
    mujoco.mj_forward(model, data)

    assert (data.ten_length < model.tendon_range[:, 1] - 1e-6)[[0, 3]].all(), (
        "60 mm of lift did not put the upper ropes into slack"
    )
    assert data.nefc == 0, (
        f"{data.nefc} constraint rows on a seat that is touching nothing and hanging "
        f"on nothing -- a slack rope is still holding it"
    )

    start = data.qpos[2]
    for _ in range(20):  # 0.1 s
        mujoco.mj_step(model, data)
    fell = start - data.qpos[2]
    free_fall = 0.5 * 9.81 * 0.1 ** 2
    assert fell > 0.9 * free_fall, (
        f"it fell {fell * 1000:.1f} mm in 0.1 s against {free_fall * 1000:.1f} mm of "
        f"free fall -- something is holding the plank up"
    )


def test_the_tie_spread_is_what_stops_the_plank_tipping_over() -> None:
    """Two attachment points a side are not the mechanism; their *spread* is.

    Each side's two ropes leave the same point on the beam and land `TIE_X` fore
    and aft of the plank's centre, so the pair is a triangle. Pitch the plank and
    one of the two has to lengthen, and being a rope it can only pull -- so the
    aft one catches a nose-down pitch and the fore one catches nose-up. With the
    two ties on top of each other the triangle is a single rope, the plank is in
    neutral equilibrium about the line between its two ropes, and it flips. That
    is why a real flat swing seat flips over, and why children stand on them.

    Loaded off-centre with the robot's own weight, 30 N at **0.10 m**, which is
    past anything pumping needs (about 60 mm).

    The control group is `TIE_X = 0` and it is the whole test: "the plank did not
    flip" is something a plank that cannot flip passes for free. It is a sharper
    control than the rigging this replaced, where the knot could only be lowered
    to the deck and still left the ties their own spread.
    """
    import mujoco
    import numpy as np

    def tilt(tie_x: float | None, seconds: float) -> float:
        """The furthest the plank gets from level, in degrees, under the load.

        Read from `qpos`, which keeps counting past 90 degrees, rather than from
        the rotation matrix -- `arcsin` folds a plank that has gone right over back
        towards zero, and the control group below reads 4.6 degrees that way while
        it is in fact spinning.

        **Addressed by name, not by index.** This read `qpos[4]` until the seat's
        three hinges were reordered to move the Euler singularity off the swing's
        own axis, at which point index 4 became roll -- and the test went on
        passing, measuring a quantity that barely moves.
        """
        model = _swing_model(tie_x)
        pitch = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "seat_pitch")
        ]
        data = mujoco.MjData(model)
        seat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seat")
        worst = 0.0
        for _ in range(int(seconds / model.opt.timestep)):
            data.xfrc_applied[seat] = [0.0, 0.0, -30.0, 0.0, 0.10 * 30.0, 0.0]
            mujoco.mj_step(model, data)
            worst = max(worst, abs(float(np.degrees(data.qpos[pitch]))))
        return worst

    held = tilt(None, 6.0)
    assert held < 45.0, (
        f"the loaded plank swung {held:.1f} degrees over -- the tie spread is not "
        f"holding it. Measured peaking at 6.1 degrees when this was written, "
        f"against 29.5 for the knotted rigging it replaced"
    )

    # The control: the same plank with its ties on top of each other. One second is
    # enough -- it is already round by then, and left running it tumbles until
    # MuJoCo starts shouting about qacc.
    flipped = tilt(0.0, 1.0)
    assert flipped > 90.0, (
        f"the control group only reached {flipped:.1f} degrees, so this test would "
        f"pass on a rig with no anti-tip mechanism at all"
    )


def test_every_length_in_the_range_is_a_swing_that_can_be_rigged() -> None:
    """`HANG_L_RANGE` is a promise about geometry, and both ends can rot quietly.

    A task draws lengths from it and writes them straight into `tendon_range`.
    Nothing downstream checks them, because nothing downstream can: a rope shorter
    than the tie spread compiles, a seat resting on the ground compiles, and a deck
    hung up against the beam compiles. All three give a rig that is merely strange
    rather than one that raises.

    What can move the ends without anyone touching this constant: `TIE_X`, `BEAM_Z`
    (which is derived from `HANG_L`, so re-scaling the nominal moves the whole
    frame), `BOARD_T`, and `STAND_Z` on the robot. So this checks the promise
    directly, at both ends and at the nominal:

    - the rope is longer than the spread it has to reach across, so `hang_for` has
      something real to invert;
    - the robot standing on the deck still clears the beam, **with the swing at its
      target amplitude**, which is the end that is easy to lose -- the seat only
      ever rises from rest;
    - the deck's top face is above the ground;
    - and the seat actually settles at the length it was rigged to.
    """
    import math

    import mujoco

    from scenes import swing
    from tasks.jumper.common.constants import STAND_Z

    low, high = swing.HANG_L_RANGE
    assert low <= swing.HANG_L <= high, (
        f"the nominal {swing.HANG_L} m is outside the range {swing.HANG_L_RANGE} "
        f"that tasks draw from, so `--scene swing` builds a swing no task trains on"
    )

    model = _swing_model()
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    for hang in (low, swing.HANG_L, high):
        rope = swing.rope_for(hang)
        assert rope > swing.TIE_X * 1.5, (
            f"at {hang} m the rope is {rope:.3f} m against a tie spread of "
            f"{swing.TIE_X} m, so it is mostly sideways and barely a swing"
        )
        assert abs(swing.hang_for(rope) - hang) < 1e-9, "hang_for does not invert"

        # Headroom, at the top of the arc rather than at rest.
        risen = swing.BEAM_Z - hang + hang * (1 - math.cos(math.radians(35.0)))
        assert swing.BEAM_Z - (risen + STAND_Z) > 0.2, (
            f"at {hang} m the robot's back reaches {risen + STAND_Z:.2f} m at 35 "
            f"degrees of swing and the beam is at {swing.BEAM_Z:.2f} m"
        )
        assert swing.BEAM_Z - hang > 0.1, (
            f"at {hang} m the plank's top face is {swing.BEAM_Z - hang:.3f} m up"
        )

        # And it actually hangs there: rig the model and let it settle.
        for name in swing.ROPES:
            rope_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
            model.tendon_range[rope_id, 1] = rope
        data = mujoco.MjData(model)
        drop = swing.seat_drop(hang)
        data.qpos[model.jnt_qposadr[names.index("seat_slide_z")]] = (
            swing.seat_drop(swing.HANG_L) - drop
        )
        for _ in range(int(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        seat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seat")
        hung = swing.BEAM_Z - float(data.xpos[seat][2])
        assert abs(hung - drop) < 0.01, (
            f"rigged at {hang} m the seat settles {hung:.3f} m below the beam "
            f"instead of {drop:.3f}"
        )


def test_the_swing_swings_and_does_not_swing_for_ever() -> None:
    """Two halves, and the prop is wrong without either.

    A seat that does not swing is a shelf. A seat that swings without loss makes a
    pumping task free -- whatever amplitude a policy reaches it keeps for nothing,
    and the reward would be collected by a robot that stands still after one lucky
    push. Neither failure raises anything.

    **Released by a kick through the bottom of the arc, not by displacement.**
    Putting the seat 0.15 m out leaves it off the arc with every rope
    over-stretched, and the constraint eats 21% of the amplitude in the first
    half-swing; measured that way the swing looked heavily damped and it was the
    initial condition.

    The period is checked loosely, as "about what the docstring says": it follows
    from `HANG_L` and the plank's inertia, and pinning it tightly would turn a
    deliberate change to either into a failure that reads as a bug.
    """
    import mujoco
    import numpy as np

    from scenes import swing

    model = _swing_model()
    seat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seat")
    # **With a robot aboard**, which is the only case that matters and the one the
    # decay table in `scenes/swing.py` is chosen against. The empty seat is now a
    # third of what it was -- the plank was lightened so the robot would own the
    # pendulum -- so it decays about two and a half times faster and says nothing
    # about the task.
    total = model.body_mass[seat] + 3.0066
    model.body_ipos[seat] = [
        0.0, 0.0,
        (model.body_mass[seat] * model.body_ipos[seat][2] + 3.0066 * 0.06) / total,
    ]
    model.body_inertia[seat] += 3.0066 * 0.06 ** 2
    model.body_mass[seat] = total

    data = mujoco.MjData(model)
    data.qvel[0] = 1.0  # seat_slide_x, tangential at the bottom

    x, t = [], []
    for _ in range(int(70.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        x.append(data.xpos[seat][0])
        t.append(data.time)
    x, t = np.array(x), np.array(t)

    peaks = [
        (t[i], abs(x[i]))
        for i in range(1, len(x) - 1)
        if abs(x[i]) > abs(x[i - 1]) and abs(x[i]) > abs(x[i + 1])
    ]
    assert len(peaks) > 10, "it did not swing"
    period = 2 * float(np.mean(np.diff([ti for ti, _ in peaks[:10]])))
    assert period == pytest.approx(2.35, abs=0.25), (
        f"the swing's period is {period:.3f} s; the docstring says 2.351"
    )
    assert peaks[0][1] == pytest.approx(0.36, abs=0.06), (
        f"a 1.0 m/s kick reached {peaks[0][1]:.3f} m, not the 0.36 measured"
    )

    # It loses amplitude, and it loses it on a timescale one 20 s episode can see.
    assert peaks[-1][1] < 0.5 * peaks[0][1], (
        "the swing keeps its amplitude: nothing is asked of a policy that pumps it"
    )
    at_20s = next(a for ti, a in peaks if ti > 20.0)
    assert at_20s > 0.2 * peaks[0][1], (
        f"only {at_20s:.3f} m of {peaks[0][1]:.3f} left after one episode -- the swing "
        f"dies faster than a policy can build it. The empty seat is the harsh case; "
        f"the robot roughly doubles the inertia and halves this loss"
    )

    # And the deck follows the rope rather than staying level: standing on it means
    # standing on a tilting floor, which is the task.
    assert swing.SEAT_ANG_DAMPING < swing.SEAT_DAMPING, (
        "the deck's pitch is slaved to the swing angle, so damping it damps the "
        "swing itself -- measured, 0.02 there cost half the amplitude on its own"
    )


def test_the_plank_is_solid_to_the_robot_and_the_rigging_is_not() -> None:
    """The plank has to be standable, which is the one thing this scene is for.

    Checked as MuJoCo's own pair test against stand-ins for the hexapod's two mask
    classes, because the leg masks are not the default ones: `constants.py` gives
    every leg `contype = 1 | bit(leg)` and `conaffinity = <the other five legs>`,
    and the base `contype = 1, conaffinity = 0`. A prop that collides with a
    conventional geom does not automatically collide with either.

    The ropes need no assertion at all and that is the point of them being
    tendons: a tendon has no geometry, so the question the rigid hangers needed a
    private collision channel to answer cannot arise. Nor is there anything else
    in the rigging -- the knots that used to need `contype=0` are gone, so the
    whole rig is the plank and the frame.
    """
    from scenes import swing
    from tasks.jumper.common.constants import _ALL_LEG_BITS, _GROUND_BIT, _LEG_BIT

    spec = swing._swing_spec()
    geoms = {g.name: g for g in spec.geoms}
    assert not [g for g in geoms if g.startswith("rod")], "the rigid hangers came back"

    def collides(a, b) -> bool:
        return bool(a[0] & b[1]) or bool(b[0] & a[1])

    leg = (_GROUND_BIT | _LEG_BIT["LM"], _ALL_LEG_BITS & ~_LEG_BIT["LM"])
    base = (_GROUND_BIT, 0)

    for part in ("board", "beam", "post_lf"):
        masks = (geoms[part].contype, geoms[part].conaffinity)
        assert collides(masks, leg), f"a foot passes straight through {part}"
        assert collides(masks, base), f"the body passes straight through {part}"

    assert set(geoms) == {"board", "beam", "post_lf", "post_lb", "post_rf", "post_rb"}, (
        f"the rigging grew geometry: {sorted(set(geoms) - {'board', 'beam'})}. Ropes "
        f"are tendons and have none, which is why nothing in the rig can catch a "
        f"foot -- anything new here needs its collision masks deciding"
    )


def test_the_swing_has_no_freejoint_and_is_placed_per_environment() -> None:
    """The seat's six degrees of freedom must stay slides and hinges.

    A freejoint is the obvious way to write "the plank hangs free", and it is the
    one thing that cannot be done here. mjlab classifies an entity with a freejoint
    as floating-base, a floating-base entity is not wrapped in a mocap body, and
    without that wrap the **welded frame** has no per-world pose: it stays at the
    model origin in every environment while `reset_root_state_uniform` scatters
    something else. Nothing raises. What you see is one swing at the edge of the
    batch and 4095 robots standing next to nothing.

    So this pins the shape of the joint list, not just the outcome, because the
    outcome is only visible at more than one environment.
    """
    import mujoco

    from scenes.swing import _swing_spec, swing_entity

    spec = _swing_spec()
    free = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    assert not free, f"a freejoint appeared ({free}); the frame will stop being welded"

    entity = swing_entity().build()
    assert entity.is_fixed_base, "the swing is no longer welded to the world"
    assert entity.is_mocap, "not mocap-wrapped: every swing would stack at the origin"
    assert not entity.is_actuated, "nothing drives the swing; the robot does"
    assert entity.joint_names == (
        "seat_slide_x", "seat_slide_y", "seat_slide_z",
        "seat_yaw", "seat_roll", "seat_pitch",
    ), f"the seat's degrees of freedom changed: {entity.joint_names}"


def test_the_swing_survives_the_native_backend_stripping() -> None:
    """The native backend deletes every geom that cannot collide.

    `mjrl/backend/model_slim.py` strips non-colliding geometry so that one model
    copy per environment fits in memory. **A body whose mass lives on such a geom
    loses it**, and if that body has joints MuJoCo then refuses to compile --
    measured exactly that way on the knotted rigging this replaced: `--backend
    native` died at env construction with `mass and inertia of moving bodies must
    be larger than mjMINVAL`, on native only, with warp entirely happy. So the
    scene tests, the GPU smoke run and the whole suite passed and the failure
    waited for whoever trained on a CPU.

    Four ropes straight to the beam have no such body left: the rig is the plank
    and the frame, every geom of it collides, and there is nothing to strip. The
    assertion is therefore about the **invariant** rather than about the old fix --
    stripping must not change any mass, and every body with a degree of freedom
    must still have one. That holds for a rig with nothing to strip and would fail
    again the moment a massless-but-jointed body came back.
    """
    import numpy as np
    from mjrl.backend.model_slim import strip_noncolliding_geoms

    from scenes.swing import _swing_spec

    before = _swing_spec().compile()
    after = strip_noncolliding_geoms(_swing_spec()).compile()

    assert np.allclose(after.body_mass, before.body_mass), (
        f"stripping changed the body masses: {before.body_mass} -> {after.body_mass}"
    )
    moving = {after.body_jntnum[i] > 0 for i in range(after.nbody)}
    assert True in moving, "no body has a degree of freedom; the seat is welded"
    for i in range(after.nbody):
        if after.body_jntnum[i] > 0:
            name = mujoco_name(after, i)
            assert after.body_mass[i] > 1e-6, (
                f"body {name!r} has degrees of freedom and no mass after stripping, "
                f"which is the compile error this test exists for"
            )


def test_the_banner_does_not_call_a_prop_scene_look_only() -> None:
    """The line people skim must not contradict the warning above it.

    `--scene football` printed "look only" directly under a warning that it
    had just added a ball with mass. Whichever of the two is read second wins,
    and the banner is the one that gets read.
    """
    cli = (REPO / "scripts" / "_cli.py").read_text(encoding="utf-8")
    body = cli.split("def apply_scene(", 1)[1]
    # Code lines only. The comment explaining why the label went naturally quotes
    # it, and searching the whole passage takes the explanation for the offence --
    # the same trap `test_seam.py`'s `code()` helper exists for, walked into here
    # while writing this test.
    code = "\n".join(
        ln for ln in body.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    assert '"look only"' not in code, "the banner still has a fixed look-only label"
    assert "scene.props" in code, "the banner ignores props"

def test_a_scene_may_declare_its_own_cli_arguments() -> None:
    """The mechanism, not `rough`'s use of it.

    `scenes.load_cli_args` mirrors `tasks.load_cli_args` so that a knob belonging
    to one scene never has to appear in `scripts/_cli.py`, which is one parser
    shared by three entry points. What this pins is the contract the entry point
    relies on: absent is None rather than an error, and present returns the
    `dest` names it added, because that is how `parse_with_task_args` knows which
    parsed values to hand back to the scene.
    """
    import argparse

    assert scenes.load_cli_args("plain") is None, "a scene with no options"

    add = scenes.load_cli_args("rough")
    assert add is not None
    parser = argparse.ArgumentParser()
    names = add(parser.add_argument_group("rough"))
    assert tuple(names) == ("terrain_row", "terrain_col")
    parsed = parser.parse_args(["--terrain-row", "4"])
    assert {n: getattr(parsed, n) for n in names} == {
        "terrain_row": "4", "terrain_col": ""
    }


def test_pinning_a_row_and_column_narrows_the_same_generator() -> None:
    """A pinned tile is the tile the full grid would have made, not a lookalike.

    That is the whole value of the flag: replaying one difficulty is only a test
    of the training terrain if it *is* the training terrain. So the row's
    difficulty has to be the one the generator would have spread onto that row,
    and the column has to be the same `SubTerrainCfg` object, not a rebuilt one.
    """
    from scenes.rough import NUM_ROWS

    full = scenes.load("rough").terrain_generator
    pinned = scenes.load(
        "rough", terrain_row=str(NUM_ROWS - 1), terrain_col="random_rough"
    ).terrain_generator

    assert pinned.num_rows == 1
    assert pinned.difficulty_range == (1.0, 1.0), "the top row is difficulty 1"
    assert list(pinned.sub_terrains) == ["random_rough"]
    assert pinned.sub_terrains["random_rough"] == full.sub_terrains["random_rough"]
    # Everything else is untouched: a pin narrows the grid, it does not re-tune it.
    assert pinned.size == full.size and pinned.curriculum == full.curriculum

    middle = scenes.load("rough", terrain_row="3").terrain_generator
    assert middle.difficulty_range[0] == pytest.approx(3 / (NUM_ROWS - 1))
    assert list(middle.sub_terrains) == list(full.sub_terrains), "column left alone"


@pytest.mark.parametrize(
    "kwargs, expect",
    [
        ({"terrain_col": "staircase"}, "not a sub-terrain"),
        ({"terrain_row": "eleven"}, "not an integer"),
        # **Derived, not written.** The row count belongs to the ladder's
        # design and has been 10 and 5; a literal here passes until someone
        # halves the rungs again and then asserts against a message that no
        # longer exists.
        ({"terrain_row": str(NUM_ROWS)}, f"outside 0..{NUM_ROWS - 1}"),
    ],
)
def test_a_pin_that_names_nothing_raises(kwargs: dict, expect: str) -> None:
    """Rather than falling back to the full mix.

    A typo that quietly ran the whole terrain would be a test of something other
    than what was asked for, and its output would look exactly like a correct
    run's -- the robot walks, the numbers are plausible, and the tile it was on
    was never the one named.
    """
    with pytest.raises(SystemExit, match=expect):
        scenes.load("rough", **kwargs)


def test_the_rough_ground_gets_budgets_that_cover_what_it_needs() -> None:
    """Whether by the task's own numbers or by the scene raising them -- **measured**.

    `mjwarp.put_data` checks a single CPU `MjData` at the model's default pose --
    robot and every tile stacked at the origin -- and refuses to build with
    `nconmax overflow (nconmax must be >= N)`. It is not what the simulation
    needs (22 constraint rows a world at spawn, 146 at peak, 13 contacts), but it
    is what decides whether a run starts, and it only appears on the warp backend
    with a GPU -- so `--dry-run` and the rest of this suite pass while `--scene
    rough` cannot build.

    **The number moves with the ladder and with the robot.** Five rows wanted 184
    and 758, ten rows 117 and 490 -- on the hexa model. The jumper v1.6.1 export
    wants 131 and 546, over the task's 128 and 512, and this test did not notice:
    it compared the budgets with those two remembered numbers, and stayed green
    while every task but five_foot failed to build on the GPU. So the check itself
    is repeated here, on the CPU, exactly as `Simulation` sets it up for
    `put_data`, and the remembered numbers are only held to it.
    """
    import mujoco
    from mjlab.scene import Scene as MjlabScene

    import tasks
    from scenes.rough import NCONMAX, REQUIRED_NCON, REQUIRED_NEFC

    cfg = tasks.load_env_cfg("jumper.posture")
    scenes.apply(cfg, "rough")
    nconmax, njmax = cfg.sim.nconmax, cfg.sim.njmax

    cfg.scene.num_envs = 1
    model = MjlabScene(cfg.scene, device="cpu").spec.compile()
    cfg.sim.mujoco.apply(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert nconmax >= data.ncon, (
        f"nconmax is {nconmax} against a build check that wants {data.ncon}; "
        f"`--scene rough` raises inside mjwarp before the first step, and only on "
        f"a GPU"
    )
    assert njmax >= data.nefc, (
        f"njmax is {njmax} against a build check that wants {data.nefc}"
    )
    # `nconmax` is the expensive knob -- 256 instead of 128 cost 4.7 GB at 4096
    # environments -- so the scene raises it to exactly what the check asks, or not
    # at all. `njmax` is nearly free and may be generous.
    assert NCONMAX in (None, data.ncon), (
        f"the rough scene raises nconmax to {NCONMAX}, but the check asks for "
        f"{data.ncon}: set it to that"
    )
    # The recorded pair is what `rough.py`'s notes reason from; keep it true.
    assert (REQUIRED_NCON, REQUIRED_NEFC) == (data.ncon, data.nefc), (
        f"rough.py records ({REQUIRED_NCON}, {REQUIRED_NEFC}) and the build check now "
        f"wants ({data.ncon}, {data.nefc}): update the table there and the budgets"
    )

    # And the mechanism still works, since it is what a harder ground would use.
    tight = tasks.load_env_cfg("jumper.posture")
    tight.sim.nconmax, tight.sim.njmax = 1, 1
    scenes.apply(tight, "soft")
    assert (tight.sim.nconmax, tight.sim.njmax) == (1, 1), "a scene raised without asking"
    from scenes.registry import Scene

    raised = tasks.load_env_cfg("jumper.posture")
    raised.sim.nconmax = 1
    probe = Scene(nconmax=999)
    for name in ("nconmax", "njmax"):
        wanted = getattr(probe, name)
        cur = getattr(raised.sim, name, None)
        if wanted is not None and (cur is None or cur < wanted):
            setattr(raised.sim, name, wanted)
    assert raised.sim.nconmax == 999 and raised.sim.njmax == 512, (
        "raising to a scene's declared budget no longer works, so a ground that "
        "needs one would fail at construction with nothing to set"
    )
