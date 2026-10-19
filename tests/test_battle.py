"""The two-crab battle world (`tools/battle/`): the failures that produce no error.

Two copies of one robot in one world **cannot touch** under the family's
collision masks -- each leg has a bit, the base has none, so two crabs' bases
and their same-numbered legs pass through each other and a shove goes through
the other crab's shell with every screen looking normal. The arena ORs a bit per
crab into the masks; the first test is MuJoCo's own pair test on the compiled
two-robot model, with the undecorated model as the control group that proves the
assertion can fail. The rest pins what `apply_arena` has to do for the second
robot to exist at all (its reset events, its start mark, its colour) and the
referee's reading of a fall, where the first version scored a crab knocked flat
on the plank as "off the plank" because its base was 3 cm above the wood.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjlab")

REPO = Path(__file__).resolve().parents[1]
BATTLE = REPO / "tools" / "battle"


def _module(name: str):
    beside = importlib.util.spec_from_file_location("_battle_beside", BATTLE / "_beside.py")
    mod = importlib.util.module_from_spec(beside)
    beside.loader.exec_module(mod)
    mod.install()
    return importlib.import_module(name)


arena = _module("arena")
referee_mod = _module("referee")


# ── the crabs can touch ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_crabs():
    """The scene as mjlab attaches it: two jumper specs under their prefixes, the
    family's collision masks on each, a terrain stand-in; compiled with and
    without the arena's edit."""
    from tasks.jumper.common.constants import HYBRID_COLLISION, get_spec

    def build(decorate: bool):
        parent = mujoco.MjSpec()
        ground = parent.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1])
        ground.contype = ground.conaffinity = 1
        for name in (arena.RED, arena.BLUE):
            spec = get_spec(None)
            HYBRID_COLLISION.edit_spec(spec)
            frame = parent.worldbody.add_frame()
            parent.attach(spec, prefix=f"{name}/", frame=frame)
        if decorate:
            arena._decorate(parent)
        return parent.compile()

    return build(True), build(False)


def _names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]


def _can_collide(model, i, j):
    return bool((model.geom_contype[i] & model.geom_conaffinity[j])
                or (model.geom_contype[j] & model.geom_conaffinity[i]))


def test_every_geom_of_one_crab_meets_every_geom_of_the_other(two_crabs):
    model, control = two_crabs
    names = _names(model)
    red = [i for i, n in enumerate(names) if n.startswith("robot/") and (model.geom_contype[i] or model.geom_conaffinity[i])]
    blue = [i for i, n in enumerate(names) if n.startswith("robot2/") and (model.geom_contype[i] or model.geom_conaffinity[i])]
    assert red and blue
    missed = [(names[i], names[j]) for i in red for j in blue if not _can_collide(model, i, j)]
    assert not missed, f"{len(missed)} cross-crab pairs cannot collide, e.g. {missed[:4]}"

    # The control group: without the arena's bits the bases pass through each other
    # and so do the same-numbered legs.
    cnames = _names(control)
    base = {n: i for i, n in enumerate(cnames) if n.endswith("base_link_meshcol")}
    assert not _can_collide(control, base["robot/base_link_meshcol"], base["robot2/base_link_meshcol"])
    lf = {n: i for i, n in enumerate(cnames) if n.endswith("LF_forearm_link_meshcol")}
    assert not _can_collide(control, lf["robot/LF_forearm_link_meshcol"], lf["robot2/LF_forearm_link_meshcol"])


def test_the_crabs_own_masks_are_untouched(two_crabs):
    """Within one crab nothing changes: same-leg pairs stay excluded (the hull
    argument in `constants.py`), different legs still meet, everything still
    meets the ground."""
    from tasks.jumper.common.constants import LEGS

    model, control = two_crabs
    names, cnames = _names(model), _names(control)
    cid = {n: i for i, n in enumerate(cnames)}
    own = [i for i, n in enumerate(names) if n.startswith("robot/") and (model.geom_contype[i] or model.geom_conaffinity[i])]
    for i, j in itertools.combinations(own, 2):
        assert _can_collide(model, i, j) == _can_collide(control, cid[names[i]], cid[names[j]]), (names[i], names[j])
    terrain = names.index("terrain")
    assert all(_can_collide(model, i, terrain) for i in own)
    leg = lambda n: next((leg for leg in LEGS if n.split("/", 1)[1].startswith(leg + "_")), None)
    same = [(i, j) for i, j in itertools.combinations(own, 2) if leg(names[i]) and leg(names[i]) == leg(names[j])]
    assert same and not any(_can_collide(model, i, j) for i, j in same)


def test_the_plank_is_solid_to_the_crabs(two_crabs):
    model, _ = two_crabs
    names = _names(model)
    plank = names.index("plank")
    foot = names.index("robot2/LF_foot_meshcol") if "robot2/LF_foot_meshcol" in names else next(
        i for i, n in enumerate(names) if n.startswith("robot2/") and model.geom_contype[i])
    assert _can_collide(model, plank, foot)


# ── the second robot in the config ─────────────────────────────────────────


@pytest.fixture
def posture_cfg():
    import tasks

    return tasks.load_env_cfg("jumper.posture", play=True)


def test_the_second_robot_is_placed_reset_and_coloured(posture_cfg, monkeypatch):
    cfg = posture_cfg
    arena.apply_arena(cfg)
    assert arena.BLUE in cfg.scene.entities
    blue, red = cfg.scene.entities[arena.BLUE], cfg.scene.entities[arena.RED]
    assert blue is not red
    assert blue.init_state.pos[0] > 0 > red.init_state.pos[0]
    assert math.isclose(blue.init_state.pos[2], red.init_state.pos[2])
    assert blue.init_state.pos[2] > arena.PLANK_TOP, "the crabs start on the plank, not in it"

    # The events that put a robot on its mark, cloned under the blue one's name
    # and pinned there: an unpinned reset scatters the crabs over +-0.5 m and any
    # yaw, and a missing one leaves the blue crab wherever it last fell.
    assert cfg.events["reset_base2"].params["asset_cfg"].name == arena.BLUE
    assert "reset_robot_joints2" in cfg.events
    for name in ("reset_base", "reset_base2"):
        pose = cfg.events[name].params["pose_range"]
        assert pose["x"] == (0.0, 0.0) and pose["y"] == (0.0, 0.0) and pose["yaw"] == (0.0, 0.0)
    assert cfg.events["reset_base"].params["asset_cfg"].name == arena.RED if "asset_cfg" in cfg.events["reset_base"].params else True

    # The world's fall reset is the referee's now.
    assert cfg.terminations["fell_over"].params["limit_angle"] == math.pi

    # Blue is blue and red is red, on the compiled specs.
    def shells(entity):
        spec = entity.spec_fn()
        return {tuple(round(float(c), 2) for c in g.rgba) for g in spec.geoms}

    assert tuple(round(c, 2) for c in arena.SHELL_BLUE) in shells(blue)
    assert tuple(round(c, 2) for c in arena.SHELL_RED) not in shells(blue)
    assert tuple(round(c, 2) for c in arena.SHELL_RED) in shells(red)


def test_the_depth_sensors_are_built_only_for_a_pilot_that_looks(posture_cfg):
    import tasks

    names = lambda cfg: [s.name for s in cfg.scene.sensors or ()]
    assert "tof" in names(posture_cfg), "the control: the task mounts a dToF in play"
    arena.apply_arena(posture_cfg)
    assert "tof" not in names(posture_cfg) and "tof2" not in names(posture_cfg)

    looking = tasks.load_env_cfg("jumper.posture", play=True)
    arena.apply_arena(looking, senses=True)
    assert "tof" in names(looking) and "tof2" in names(looking)
    tof2 = next(s for s in looking.scene.sensors if s.name == "tof2")
    assert tof2.frame.entity == arena.BLUE and tof2.pattern.camera.entity == arena.BLUE


# ── the referee ─────────────────────────────────────────────────────────────


class _Crab:
    def __init__(self, z: float, tilt_deg: float) -> None:
        import torch

        self.data = type("D", (), {})()
        self.data.default_root_state = torch.tensor([[0.0, 0.0, arena.PLANK_TOP + 0.107, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.data.default_joint_pos = torch.zeros((1, 22))
        self.set(z, tilt_deg)
        self._torch = torch
        self.restored = 0

    def set(self, z: float, tilt_deg: float) -> None:
        import torch

        self.data.root_link_pos_w = torch.tensor([[0.0, 0.0, z]])
        t = math.radians(tilt_deg)
        self.data.projected_gravity_b = torch.tensor([[math.sin(t), 0.0, -math.cos(t)]])

    def write_root_state_to_sim(self, root, env_ids=None) -> None:
        assert root.shape == (1, 13) and float(root[0, 2]) > arena.PLANK_TOP and float(root[0, 7:].abs().sum()) == 0
        self.restored += 1
        self.set(float(root[0, 2]), 0.0)

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None) -> None:
        assert pos.shape == vel.shape


class _Env:
    def __init__(self) -> None:
        self.scene = {arena.RED: _Crab(arena.PLANK_TOP + 0.107, 0.0), arena.BLUE: _Crab(arena.PLANK_TOP + 0.107, 0.0)}
        self.resets = 0


    def reset(self) -> None:
        self.resets += 1


def test_a_winner_that_went_over_with_the_loser_is_stood_back_up_for_its_piece():
    """The shove carries the shover: in two of the first three brain takes the
    winner was on the sand beside the loser when its cheer was cued, the chord
    was dropped by a controller in `safe`, and the phase ran to its ceiling with
    nothing played. A winner not standing on the plank is put back in its
    middle before the cue; one that is standing is left where it is."""
    env = _Env()
    ref = referee_mod.Referee(env, tilt_limit=0.873, best_of=3, linger_s=1.0, ready_s=1.0, down_s=0.0)
    ref.step(1.0)
    ref.step(2.1, {"red": False, "blue": False})
    assert ref.live
    env.scene[arena.RED].set(0.05, 0.0)                   # red on the sand
    ref.step(3.0)
    assert ref.state == "linger" and ref.score == {"red": 0, "blue": 1}
    env.scene[arena.BLUE].set(0.04, 110.0)                # the shove carried blue over after it: on its back on the sand
    ref.step(4.1, {})
    assert ref.state == "cheer" and ref.cue == {"blue": "cheer"}
    assert env.scene[arena.BLUE].restored == 1 and env.scene[arena.RED].restored == 0
    assert ref.standing("blue") and not ref.standing("red")
    ref.step(9.2, {})
    assert ref.round == 2
    ref.step(10.3)
    ref.step(10.4, {"red": False, "blue": False})
    env.scene[arena.RED].set(arena.PLANK_TOP + 0.107, 0.0)    # (the stub's reset moves nobody)
    env.scene[arena.BLUE].set(0.05, 0.0)                  # blue off; red standing on the plank
    ref.step(11.0, {"red": False, "blue": False})
    ref.step(11.4, {"red": False, "blue": False})
    assert ref.live
    ref.step(11.5)
    assert ref.state == "linger"
    ref.step(12.6, {})
    assert ref.cue == {"red": "cheer"} and env.scene[arena.RED].restored == 0, "a standing winner is left be"


def test_a_winner_standing_at_the_planks_end_is_moved_to_its_middle():
    """Upright on the plank is not enough: the third brain take's winner stood at
    the plank's end and its dance went off it. A performer needs plank around it."""
    env = _Env()
    ref = referee_mod.Referee(env, tilt_limit=0.873)
    red = env.scene[arena.RED]
    assert ref.standing("red")
    # (the arena's own edge: the disc's rim in both directions, the plank's end and side)
    disc = arena.ARENA == "disc"
    end_x = arena.DISC_R if disc else arena.PLANK_LEN / 2
    side_y = arena.DISC_R if disc else arena.PLANK_WIDTH / 2
    red.data.root_link_pos_w = red._torch.tensor([[end_x - 0.1, 0.0, arena.PLANK_TOP + 0.107]])
    assert not ref.standing("red"), "at the end, 10 cm from the edge"
    red.data.root_link_pos_w = red._torch.tensor([[0.0, side_y - 0.1, arena.PLANK_TOP + 0.107]])
    assert not ref.standing("red"), "at the side"


def test_a_fall_is_scored_where_it_lands_and_a_flip_where_it_stops():
    """The referee's three ways to lose, and the two states that look like one
    but are not: a crab flat on the plank (base 3 cm over the wood, the first
    referee called it "off the plank"), and a crab in mid-air below the plank's
    top, which is a fall in progress and scores only once it is on the sand."""
    env = _Env()
    ref = referee_mod.Referee(env, tilt_limit=0.873, down_s=0.5, out_s=3.0)
    crab = env.scene[arena.RED]
    crab.set(arena.PLANK_TOP + 0.03, 20.0)          # on its belly, legs splayed, under the tilt limit
    assert ref.down("red", 0.0) is None
    crab.set(arena.PLANK_TOP - 0.05, 0.0)           # over the edge, still falling
    assert ref.down("red", 0.0) is None
    crab.set(referee_mod.LANDED_Z - 0.01, 0.0)      # on the sand
    assert "on the sand" in ref.down("red", 0.0)
    crab.set(arena.PLANK_TOP + 0.05, 150.0)         # on its back, on the plank
    assert ref.down("red", 1.0) is None, "a flip is a flip once it has stayed one"
    assert "flipped" in ref.down("red", 1.6)
    crab.set(arena.PLANK_TOP + 0.10, 10.0)          # righted (a flip is past the tilt limit too: both clocks restart)
    assert ref.down("red", 1.8) is None
    crab.set(arena.PLANK_TOP + 0.08, 70.0)          # on its side: the controller let go, it cannot get up
    assert ref.down("red", 2.0) is None
    assert ref.down("red", 4.9) is None, "down and out takes `out_s`, not `down_s`"
    assert "down and out" in ref.down("red", 5.1)
    crab.set(arena.PLANK_TOP + 0.10, 10.0)          # righted: both clocks restart
    assert ref.down("red", 5.2) is None
    crab.set(arena.PLANK_TOP + 0.05, 150.0)
    assert ref.down("red", 5.3) is None


def test_the_match_is_scored_and_reset_by_the_referee():
    env = _Env()
    ref = referee_mod.Referee(env, tilt_limit=0.873, best_of=3, linger_s=1.0, ready_s=1.0, down_s=0.0, moves=False)
    ref.step(0.0)
    assert not ref.live
    ref.step(1.0)
    assert ref.live and ref.round == 1
    env.scene[arena.RED].set(0.05, 0.0)
    ref.step(2.0)
    assert ref.score == {"red": 0, "blue": 1} and ref.state == "linger" and env.resets == 0
    ref.step(3.0)
    assert env.resets == 1 and ref.round == 2 and ref.state == "ready"
    env.scene[arena.RED].set(arena.PLANK_TOP + 0.107, 0.0)
    ref.step(4.0)
    assert ref.live
    env.scene[arena.BLUE].set(arena.PLANK_TOP + 0.05, 100.0)
    ref.step(4.5)              # the flip's clock starts on the step it is first seen...
    ref.step(4.505)            # ...and `down_s` 0 scores it on the next
    assert ref.score == {"red": 1, "blue": 1}
    ref.step(5.6)
    assert env.resets == 2 and ref.round == 3
    ref.step(6.7)
    assert ref.live
    env.scene[arena.RED].set(0.05, 0.0)
    env.scene[arena.BLUE].set(0.05, 0.0)
    ref.step(7.0)
    assert ref.history[-1][1] == "draw" and ref.score == {"red": 1, "blue": 1}
    ref.step(8.1)
    assert ref.round == 4
    env.scene[arena.RED].set(arena.PLANK_TOP + 0.107, 0.0)
    env.scene[arena.BLUE].set(arena.PLANK_TOP + 0.107, 0.0)
    ref.step(9.2)
    assert ref.live
    env.scene[arena.RED].set(0.05, 0.0)
    ref.step(9.5)
    assert ref.score == {"red": 1, "blue": 2}
    with pytest.raises(referee_mod.SegmentDone, match="blue wins the match 1-2"):
        ref.step(10.6)


def test_the_set_pieces_are_cued_once_and_waited_for():
    """The taunt, the cheer and the dance are phases the referee holds until the
    crabs are done -- as the duel reports them busy -- or the ceiling; `cue_id`
    changes with every cue, so a pilot that keys on it plays each piece once.
    Without `busy` reports (a crab on the keys) the ceiling alone ends them."""
    env = _Env()
    ref = referee_mod.Referee(env, tilt_limit=0.873, best_of=3, linger_s=1.0, ready_s=1.0, down_s=0.0,
                              taunt_max_s=5.0, cheer_max_s=5.0, victory_max_s=5.0)
    ref.step(1.0)
    assert ref.state == "taunt" and ref.cue == {"red": "taunt", "blue": "taunt"}
    first = ref.cue_id
    ref.step(1.5, {"red": True, "blue": True})
    ref.step(2.5, {"red": True, "blue": False})
    assert ref.state == "taunt" and ref.cue_id == first, "one crab still taunting holds the bell"
    ref.step(2.6, {"red": False, "blue": False})
    assert ref.live and ref.cue == {} and ref.cue_id == first + 1
    env.scene[arena.RED].set(0.05, 0.0)
    ref.step(3.0)
    ref.step(4.0, {})
    assert ref.state == "cheer" and ref.cue == {"blue": "cheer"}
    ref.step(9.1, {})                                   # nobody reports: the ceiling ends it
    assert ref.state == "ready" and env.resets == 1 and ref.round == 2
    env.scene[arena.RED].set(arena.PLANK_TOP + 0.107, 0.0)
    ref.step(10.2)
    ref.step(10.3, {"red": False, "blue": False})
    ref.step(11.4, {"red": False, "blue": False})
    assert ref.live
    env.scene[arena.RED].set(0.05, 0.0)
    ref.step(12.0)
    ref.step(13.1, {})
    assert ref.state == "victory" and ref.cue == {"blue": "victory"}
    ref.step(14.2, {"blue": True})
    with pytest.raises(referee_mod.SegmentDone, match="blue wins the match 0-2"):
        ref.step(14.3, {"blue": False})


def test_a_winner_stood_back_up_faces_the_camera():
    """The dance is played to the camera (Andrew, 2026-10-14: "when a crab wins
    and does its victory dance it should be facing the camera"). A winner the
    referee stands back up is stood facing `camera_yaw` rather than at its
    mark's heading -- the mark faces the other crab, side-on to the shot -- and
    without a camera heading the mark's heading is kept, as before."""
    import torch

    env = _Env()
    seen = []
    env.scene[arena.RED].write_root_state_to_sim = lambda root, env_ids=None: seen.append(root.clone())
    yaw = math.radians(-90.0)                                   # the default camera's: azimuth 90 + 180
    ref = referee_mod.Referee(env, tilt_limit=0.873, camera_yaw=yaw)
    ref.restore("red")
    w, x, y, z = (float(v) for v in seen[-1][0, 3:7])
    heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    assert abs(heading - yaw) < 1e-6 and abs(math.hypot(w, z) - 1.0) < 1e-6
    assert float(seen[-1][0, 7:].abs().sum()) == 0 and float(seen[-1][0, 2]) > arena.PLANK_TOP
    referee_mod.Referee(env, tilt_limit=0.873).restore("red")
    assert torch.equal(seen[-1][0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0])), "no camera heading: the mark's"
