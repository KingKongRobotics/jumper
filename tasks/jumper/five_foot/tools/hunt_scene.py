"""The hunt's room, round can, and where the crab and the bugs start.

The floor the policy walks on stays a plane. These half-sizes are what it is
drawn at, and the walls are the boundary. The can is this hunt's own prop:
the square tray in ``objects.py`` is the one the objects row uses, and it is
left alone.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from tasks.jumper.five_foot.objects import (
    BIN_INNER_HALF,
    BIN_WALL_H,
    BIN_WALL_T,
    OBJECTS_NCONMAX,
    OBJECTS_NJMAX,
    PROP_CONE,
    PROP_IMPRATIO,
    _free_body,
    _place_props,
    _prop,
    _prop_geom,
    _prop_spec,
)

#: Sphere 30 mm across, a few grams. Inside the open mouth (about 73 mm) and
#: light enough that the five-foot stand does not have to relearn the carry.
BUG_RADIUS = 0.015
BUG_DENSITY = 500.0
N_BUGS = 3

#: Known fixture, to the robot's right, outside the forward cone at the spawn.
TRAY_XY = (0.0, -1.15)

#: The room, in metres. Bugs are scattered out to 1.35 m and the tray sits at
#: 1.15 m, so a 4 m square holds both with a margin. The plane's collision is
#: still infinite -- a mesh foot against a box is not the contact this policy
#: trained on -- and these half-sizes are what it is drawn at. The walls are
#: the boundary.
ROOM_HALF = 2.0
ROOM_WALL_H = 0.45
ROOM_WALL_T = 0.04
#: Twice the square tray's 20 mm rim. The room walls stay at ``ROOM_WALL_H``.
CAN_WALL_H = 2.0 * BIN_WALL_H
#: Segments in the round can's rim. The floor is one cylinder; the rim is
#: boxes because a cylinder cannot be a ring.
CAN_SEGMENTS = 24


def _bug_spec():
    spec = _prop_spec()
    body = _free_body(spec, "bug")
    _prop_geom(
        body, name="shell", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BUG_RADIUS, 0.0, 0.0], density=BUG_DENSITY,
        rgba=[0.55, 0.72, 0.18, 1.0],
    )
    return spec


def _round_bin_spec():
    """A round can whose rim is twice the square tray's.

    The room walls are unchanged. This rim is the one that grew.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dropbin", pos=[0.0, 0.0, 0.0])
    wood = [0.55, 0.42, 0.30, 1.0]
    inner, t, h = BIN_INNER_HALF, BIN_WALL_T, CAN_WALL_H
    _prop_geom(
        body, name="bin_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[inner + t, t / 2, 0.0], pos=[0.0, 0.0, t / 2],
        rgba=wood, mass=0.4,
    )
    radius = inner + t / 2
    half_arc = (inner + t) * math.sin(math.pi / CAN_SEGMENTS)
    for i in range(CAN_SEGMENTS):
        ang = 2.0 * math.pi * i / CAN_SEGMENTS
        yaw = ang / 2.0
        _prop_geom(
            body, name=f"bin_rim_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[t / 2, half_arc, h / 2],
            pos=[radius * math.cos(ang), radius * math.sin(ang), t + h / 2],
            quat=[math.cos(yaw), 0.0, 0.0, math.sin(yaw)],
            rgba=wood, mass=0.02,
        )
    return spec


def _delivered_z() -> float:
    """Rim test from ``Highlight``, using this bug's own half-height.

    ``HIGHLIGHT_DELIVERED_Z`` is half a wall above the tallest solid standing
    on the tray floor. A 15 mm bug in the claw is still under that line, so
    the same rule applied to the can would mark a carry as a drop.
    """
    return BIN_WALL_T + BUG_RADIUS + CAN_WALL_H / 2


def _scatter(rng, n: int) -> tuple[tuple[float, float, float], list[np.ndarray]]:
    """One crab pose and ``n`` bug positions, inside the room and clear of the tray."""
    tray = np.array(TRAY_XY, dtype=np.float64)
    crab = None
    for _ in range(80):
        xy = np.array([rng.uniform(-1.45, 1.45), rng.uniform(-1.45, 1.45)])
        if np.linalg.norm(xy - tray) < 0.55:
            continue
        crab = xy
        break
    if crab is None:
        raise RuntimeError("could not place the crab clear of the tray")
    yaw = float(rng.uniform(-math.pi, math.pi))

    placed: list[np.ndarray] = []
    for _ in range(n * 80):
        if len(placed) == n:
            break
        xy = np.array([rng.uniform(-1.75, 1.75), rng.uniform(-1.75, 1.75)])
        if max(abs(float(xy[0])), abs(float(xy[1]))) > ROOM_HALF - 0.15:
            continue
        if np.linalg.norm(xy - tray) < 0.45:
            continue
        if np.linalg.norm(xy - crab) < 0.50:
            continue
        if any(np.linalg.norm(xy - other) < 0.35 for other in placed):
            continue
        placed.append(xy)
    if len(placed) != n:
        raise RuntimeError(f"could only place {len(placed)} of {n} bugs")
    return (float(crab[0]), float(crab[1]), yaw), placed


def add_hunt_scene(cfg, seed: int, n: int = N_BUGS) -> tuple[list[str], tuple[float, float, float]]:
    """Bugs and the crab scattered in the room, and the tray.

    Does not call ``apply_objects``: that installs the soap row and forces a
    zero command. The friction cone is the part that has to come with any
    prop the claw is meant to hold.
    """
    rng = np.random.default_rng(seed)
    crab_pose, placed = _scatter(rng, n)
    print(
        f"[hunt] layout seed {seed}: crab ({crab_pose[0]:+.2f}, {crab_pose[1]:+.2f}) "
        f"yaw {math.degrees(crab_pose[2]):.0f} deg, bugs "
        + ", ".join(f"({float(xy[0]):+.2f}, {float(xy[1]):+.2f})" for xy in placed)
    )

    entities = cfg.scene.entities
    names = []
    props = {}
    for i, xy in enumerate(placed):
        name = f"bug_{i}"
        props[name] = _prop(_bug_spec, (float(xy[0]), float(xy[1]), BUG_RADIUS))
        entities[name] = props[name]
        names.append(name)
    props["tray"] = _prop(_round_bin_spec, (TRAY_XY[0], TRAY_XY[1], 0.0))
    entities["tray"] = props["tray"]
    # Without a reset event a free body stays at the origin in every environment,
    # which puts every bug inside the tray and ends the hunt on the first step.
    _place_props(cfg, props)

    cfg.sim.nconmax = max(int(cfg.sim.nconmax or 0), OBJECTS_NCONMAX)
    cfg.sim.njmax = max(int(cfg.sim.njmax or 0), OBJECTS_NJMAX)
    cfg.sim.mujoco.cone = PROP_CONE
    cfg.sim.mujoco.impratio = PROP_IMPRATIO
    _install_room(cfg)
    return names, crab_pose


def _install_room(cfg) -> None:
    """Draw the ground as a finite square and put walls on its edge."""
    previous = cfg.scene.spec_fn

    def spec_fn(spec):
        if previous is not None:
            previous(spec)
        _bound_room(spec)

    cfg.scene.spec_fn = spec_fn


def _bound_room(spec) -> None:
    plane = next((g for g in spec.geoms if g.name == "terrain"), None)
    if plane is None or int(plane.type) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise RuntimeError("bug hunt expected a ground plane named terrain")
    plane.size[0] = ROOM_HALF + ROOM_WALL_T
    plane.size[1] = ROOM_HALF + ROOM_WALL_T
    h, t, inner = ROOM_WALL_H, ROOM_WALL_T, ROOM_HALF
    rgba = [0.62, 0.60, 0.56, 1.0]
    # The end walls run the full side, thickness included, and the other pair
    # sits between them so the corners meet without a gap.
    for name, sx, sy, px, py in (
        ("room_xp", t / 2, inner + t, inner + t / 2, 0.0),
        ("room_xm", t / 2, inner + t, -inner - t / 2, 0.0),
        ("room_yp", inner, t / 2, 0.0, inner + t / 2),
        ("room_ym", inner, t / 2, 0.0, -inner - t / 2),
    ):
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[sx, sy, h / 2], pos=[px, py, h / 2], rgba=rgba,
            contype=1, conaffinity=1, group=2,
        )


def pin_spawn(cfg, pose: tuple[float, float, float]) -> None:
    """Hold the crab at one drawn pose for the whole run.

    A range would be drawn again on the second reset, the one after the arm
    calibration, and the hunt would start somewhere other than the pose the
    bugs were laid out around.
    """
    reset = cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("bug hunt pins the spawn, and this config has no reset_base")
    x, y, yaw = pose
    span = dict(reset.params.get("pose_range", {}))
    span["x"] = (x, x)
    span["y"] = (y, y)
    span["yaw"] = (yaw, yaw)
    reset.params["pose_range"] = span

    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.resampling_time_range = (1.0e6, 1.0e6)
    pose_cmd = cfg.commands["body_pose"]
    pose_cmd.rel_neutral_envs = 1.0
    pose_cmd.resampling_time_range = (1.0e6, 1.0e6)
    cfg.terminations.clear()
