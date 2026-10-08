"""Exporting a scene for another simulator to import.

Three failures are worth pinning, and all three produce a package rather than an
error:

- **the wrong scene leaves.** `rough` is a 10 x 7 grid of difficulty tiles and
  `plain` has no materials at all; either would import, render, and be a strange
  place to put a visitor. The rule that decides is `SceneSpec.use`, declared in
  `scenes/__init__.py` -- and a rule read from one place only works if the code
  actually reads it, which is what the control group below checks.
- **a decorative geom arrives solid.** Pitch markings and goalposts are
  `contype=0, conaffinity=0` here, pinned by `tests/test_scenes.py`. A package
  that lost that puts invisible walls around the robot, and a goalpost that has
  become solid renders identically to one that has not.
- **a texture does not arrive.** An unresolved texture compiles fine and renders
  flat white, which reads as a lighting problem rather than a missing file.

These import `TerrainEntity`, so torch loads. That is about 3 s once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("mujoco")

import scenes  # noqa: E402
from scenes.tools import export_web_scene as ex  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def football(tmp_path_factory) -> Path:
    """The one scene that exercises every part: decoration, a prop, file textures,
    a cube map and a private collision channel."""
    return ex.export("football", tmp_path_factory.mktemp("scenes"))


# ── Which scenes leave ───────────────────────────────────────────────────


def test_the_training_scenes_are_withheld() -> None:
    withheld = {s for s in scenes.list_ids() if ex.refuse(s) is not None}
    assert withheld == {"plain", "rough", "soft"}


def test_the_exporter_reads_the_declaration_rather_than_agreeing_with_it(monkeypatch) -> None:
    """The control group.

    Every assertion above passes against an exporter with `{"plain", "rough",
    "soft"}` written into it, which is the arrangement `SceneSpec.use` exists to
    end: a list of names in the exporter is a second place the judgement lives,
    and adding a scene without updating it is silent either way. Change what a
    scene says it is for, and the exporter must follow.
    """
    from dataclasses import replace

    studio = scenes.get("studio")
    assert ex.refuse("studio") is None

    monkeypatch.setitem(scenes.registry._REGISTRY, "studio", replace(studio, use="training"))
    denial = ex.refuse("studio")
    assert denial is not None and "use='training'" in denial.reason

    # And the other way: a withheld scene that changes its mind is still refused
    # for its *other* reason, which is the point of collecting both.
    rough = scenes.get("rough")
    monkeypatch.setitem(scenes.registry._REGISTRY, "rough", replace(rough, use="watching"))
    assert ex.refuse("rough") is None, "rough is withheld only by its declaration"

    soft = scenes.get("soft")
    monkeypatch.setitem(scenes.registry._REGISTRY, "soft", replace(soft, use="watching"))
    soft_denial = ex.refuse("soft")
    assert soft_denial is not None, "soft's Python event is a second, independent gate"
    assert "soft_ground_drag" in soft_denial.reason


def test_a_refused_scene_is_not_written(tmp_path) -> None:
    with pytest.raises(ValueError, match="will not be exported"):
        ex.export("rough", tmp_path)
    assert not (tmp_path / "rough").exists()


# ── What the package contains ────────────────────────────────────────────


def test_the_package_carries_the_look_the_physics_and_the_ball(football: Path) -> None:
    manifest = json.loads((football / "scene-package.json").read_text("utf-8"))
    assert manifest["schema"] == ex.SCHEMA
    assert manifest["use"] == "watching"

    # 99 decorative geoms, 4 boards, one ground.
    assert manifest["world"]["counts"]["geoms"] == len(manifest["world"]["decorativeGeoms"]) + len(
        manifest["world"]["collidingGeoms"]
    ) + 1

    ball = manifest["props"][0]
    assert ball["name"] == "ball"
    assert ball["mass"] == pytest.approx(0.1075)
    assert ball["freeJoint"] is True
    # condim 6, because the rolling coefficient beside it is only evaluated
    # there -- see `_ball_spec`. A package that dropped it would have a ball that
    # coasts the length of the pitch from a nudge.
    assert ball["contact"][0]["condim"] == 6
    assert ball["contact"][0]["friction"][2] == pytest.approx(0.005)
    assert (football / ball["file"]).exists()


def test_every_file_it_names_is_there_and_hashes(football: Path) -> None:
    manifest = json.loads((football / "scene-package.json").read_text("utf-8"))
    on_disk = {
        str(p.relative_to(football)) for p in football.rglob("*")
        if p.is_file() and p.name != "scene-package.json"
    }
    assert set(manifest["files"]) - {"scene-package.json"} == on_disk
    for name, entry in manifest["files"].items():
        path = football / name
        assert path.stat().st_size == entry["bytes"], name
        assert ex._sha256(path) == entry["sha256"], name


def test_the_package_does_not_refer_to_this_machine(football: Path) -> None:
    """`MjSpec` writes absolute texture paths, which are correct here and useless
    anywhere else. Checked on the text rather than on the compile, because the
    compile succeeds on the machine that wrote it and nowhere else."""
    for xml in football.rglob("*.xml"):
        text = xml.read_text("utf-8")
        # With the separator: a checkout at a one-letter path such as `/w`
        # otherwise matches `</worldbody>`, and an absolute reference into the
        # repository always continues past its root.
        assert f"{REPO}/" not in text, xml
        assert "assets/" in text, xml


# ── The rules the manifest states ────────────────────────────────────────


def test_the_decorative_geoms_really_cannot_be_collided_with(football: Path) -> None:
    """The same rule `tests/test_scenes.py` pins inside this repository, checked
    on what actually left it."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(football / "scene.xml"))
    manifest = json.loads((football / "scene-package.json").read_text("utf-8"))
    named = manifest["world"]["decorativeGeoms"]
    assert named, "a scene with no decoration would make this vacuous"
    for name in named:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert i >= 0, name
        assert not model.geom_contype[i] and not model.geom_conaffinity[i], name

    # The control: the boards are *not* in that list, and they do collide. Without
    # this the check above would pass on a package that had simply listed nothing
    # colliding as decorative -- including by listing nothing at all.
    boards = {g["geom"] for g in manifest["world"]["collidingGeoms"]}
    assert boards == {"board_n", "board_s", "board_e", "board_w"}
    for name in boards:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[i] and not (model.geom_contype[i] & 1), (
            f"{name} should collide on a private channel with bit 0 left out: "
            f"solid to the ball, transparent to the robot"
        )


def test_a_rule_is_stated_only_where_it_applies(tmp_path) -> None:
    """A rule in the wrong package is worse than no rule.

    `ice` holds its friction by carrying priority 2, and losing that leaves a
    scene that renders as ice and grips like tarmac. `football` runs on the
    default ground at priority 0 -- a line there claiming otherwise would
    contradict the number printed three keys above it.
    """
    ice = json.loads((ex.export("ice", tmp_path) / "scene-package.json").read_text("utf-8"))
    assert ice["world"]["ground"]["priority"] == 2
    assert ice["world"]["ground"]["friction"][0] == pytest.approx(0.03)
    assert "groundOutranksTheFeet" in ice["rules"]
    assert "propsArePlacedPerRobot" not in ice["rules"]

    default = json.loads((ex.export("default", tmp_path) / "scene-package.json").read_text("utf-8"))
    assert default["world"]["ground"]["priority"] == 0
    assert "groundOutranksTheFeet" not in default["rules"]


def _damaged(football: Path, tmp_path: Path, edit) -> Path:
    """A copy of the written scene with one geom altered.

    A copy, because the fixture is shared and a test that corrupts it would make
    whichever test ran next fail for a reason that has nothing to do with it --
    which is how this pair failed the first time they were written.
    """
    import shutil
    import xml.etree.ElementTree as ET

    work = tmp_path / "damaged"
    shutil.copytree(football, work)
    path = work / "scene.xml"
    tree = ET.parse(path)
    edit(tree.getroot())
    tree.write(path, encoding="utf-8")
    return path


def test_geometry_lost_on_the_way_out_is_caught(football: Path, tmp_path: Path) -> None:
    """The counts check, proven non-vacuous.

    Written the obvious way first -- delete a `<texture>` -- and that passed for
    the wrong reason: MuJoCo refuses to compile a material whose texture is gone,
    so the case never reached the counts, and the check was taking credit for
    something already loud.

    What is quiet is a *geom* going missing. 98 pitch markings instead of 99 is a
    perfectly valid model and a slightly wrong pitch, and nothing but the count
    would say so.
    """
    spec = ex._build_spec(scenes.load("football"))
    model = spec.compile()

    def drop_one_marking(root):
        world = root.find("worldbody")
        marking = next(g for g in world.findall("geom") if g.get("name") == "halfway")
        world.remove(marking)

    path = _damaged(football, tmp_path, drop_one_marking)
    with pytest.raises(AssertionError, match="did not survive being written down"):
        ex._verify(path, spec, model)


def test_a_decoration_that_came_back_solid_is_caught(football: Path, tmp_path: Path) -> None:
    """The rule that matters most, proven non-vacuous.

    A goalpost the robot can walk through and one it cannot render identically.
    This makes one solid in the written file and requires the check to say so.
    """
    spec = ex._build_spec(scenes.load("football"))
    model = spec.compile()
    name = ex._decorative_geoms(spec)[0]

    def make_it_solid(root):
        geom = next(
            g for g in root.find("worldbody").findall("geom") if g.get("name") == name
        )
        geom.set("contype", "1")

    path = _damaged(football, tmp_path, make_it_solid)
    with pytest.raises(AssertionError, match="able to collide"):
        ex._verify(path, spec, model)
