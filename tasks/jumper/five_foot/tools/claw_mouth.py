#!/usr/bin/env python3
"""Does the claw's **collision** have a mouth, or only its picture?

    python tasks/jumper/five_foot/tools/claw_mouth.py
    python tasks/jumper/five_foot/tools/claw_mouth.py --finger -0.50 --model assets/jumper/jumper.xml
    python tasks/jumper/five_foot/tools/claw_mouth.py --pieces <dir of LF_palm_link_col*.STL / LF_finger_link_col*.STL>

MuJoCo collides a mesh by its convex hull, and the hull of a hooked jaw is filled
in across the mouth. In `jumper.five_foot` both of this claw's jaws are therefore
convex-*decomposed* (`tasks/jumper/five_foot/tools/jaw_decompose.py`, put on that task's robot by
`tasks/jumper/five_foot/jaws.py`; the shared asset keeps one hull each) -- and the
failure when one of them is not is completely silent. Nothing errors, nothing
looks wrong on screen, and the claw simply meets every object a couple of
centimetres before the part touches it: the fixed jaw shoves things it is not
touching, and the moving jaw can never push anything into a mouth that is solid.

## What it measures

The mouth is a wedge, so one number for it would be a choice of where to look.
This walks **along** it instead, in bands of distance from the finger's hinge,
and at each band takes the narrowest place across the mouth and asks how much
room a point in the middle of it has:

  * **drawn** -- to the nearest vertex of either jaw's *visual* mesh, which is
    the claw in the picture;
  * **collided** -- to the nearest collision shape, from MuJoCo itself
    (`mj_geomDistance` against a point-sized probe geom, so it is the hull the
    solver uses and not a copy of it).

A convex piece is never thinner than the band it covers, so `collided` can only
be the smaller of the two. How much smaller is the whole question.

Measured on the V1.6 asset at finger -0.90, with each jaw as one hull and as it
is cut (`LF_palm_link` in six pieces, `LF_finger_link` in three):

    from the hinge     drawn    one hull     cut
        40 mm         12.1 mm     0.0 mm     10.5 mm
        55 mm         19.7 mm     3.4 mm     18.6 mm
        70 mm         34.7 mm    23.1 mm     33.9 mm
        85 mm         33.6 mm    25.5 mm     33.3 mm
       100 mm         31.2 mm    27.3 mm     31.1 mm

With one hull the middle of the mouth is *inside* the fixed jaw at the knuckle
end and 8 to 16 mm short of the part over the grip face -- which is the bug this
check exists for, and the reason an object could be shoved by a jaw with daylight
between them. Even at the tip the hull still takes 4 mm.

The first band is the knuckle end, where the two jaws are two halves of one hinge
and a hull's own thickness has nowhere to go; `claw.py::KNUCKLE_CLEAR` is where
the mouth is taken to start, the same number `grasp_pose.py` and `mdp/grasp.py`
measure from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from tasks.jumper.common.constants import HOME, STAND_Z
from tasks.jumper.five_foot.claw import (
    FINGER_JOINT,
    GRIPPER_OPEN,
    KNUCKLE_CLEAR,
    LF_GRASP,
)
from tasks.jumper.five_foot.jaws import JAW_COLLISION, decompose_jaws

#: Only for the default `--model` path. Everything importable comes through
#: `pip install -e .` -- see the docstring in `scripts/_cli.py` for why nothing
#: here rewrites `sys.path`, and `tests/test_layout.py::test_no_sys_path_mutation`
#: for what enforces it.
REPO = Path(__file__).resolve().parents[4]

#: The V1.6 claw. The fixed jaw is the wrist link `LF_palm_link` -- the hooked upper
#: jaw -- with the two pads bolted to its tip, `LF_palm_pad_f_link` in front and
#: `LF_palm_pad_b_link` behind, the second of which is the foot when this leg walks.
#: The moving jaw is the finger, `LF_finger_link`, and its tip.
MOVING_JAW = ("LF_finger_link", "LF_finger_tip_link")
FIXED_JAW = ("LF_palm_link", "LF_palm_pad_f_link", "LF_palm_pad_b_link")

#: The two jaw links that are convex-decomposed; `--pieces` swaps cuts of these.
DECOMPOSED = ("LF_palm_link", "LF_finger_link")

#: Name of the probe geom this adds to the model.
PROBE = "mouth_probe"

#: How far along the mouth to look, in metres from the finger's hinge: one band
#: per entry, each `BAND` wide. It starts at `KNUCKLE_CLEAR` because that is
#: where the mouth is taken to start, and ends where the jaws do.
BANDS = np.arange(KNUCKLE_CLEAR, 0.106, 0.015)
BAND = 0.015

#: How much room the solver may lose against the part, in metres.
#:
#: Some loss is structural -- a convex piece contains the band it covers, so its
#: surface sits a little proud of the part -- and the question is only how much.
#: Measured with the jaws as they are cut today, at finger -0.90, the worst is
#: 1.6 mm, near the knuckle where a hull's thickness has nowhere to go, and
#: 1.1 mm or less over the grip face itself. 5 mm leaves room for a re-cut that
#: lands slightly differently and still fails a single hull, which loses 16.
ROOM_TOLERANCE = 0.005


def build(model_path: str, pieces: Path | None = None) -> mujoco.MjModel:
    """The robot with the task's collision scheme applied, plus a probe.

    `jumper.xml` is the URDF export; the contact structure is the task's and mjlab
    writes it in when it builds the entity, so a bare compile would measure a
    different robot from the one that runs.

    The probe is a point-sized sphere, `contype=0` and `conaffinity=0` so it
    takes no part in anything -- `mj_geomDistance` is a geometry query and does
    not consult either. Moving it is how a *point* gets measured against the
    solver's collision shapes, which MuJoCo has no other API for. It hangs off
    the worldbody with no joint and `room` writes its `geom_xpos` straight into
    `MjData`: a free body would need a mass and an inertia the compiler accepts,
    and there is nothing to simulate here.

    `pieces` re-points each decomposed jaw's collision geoms at whatever
    `<link>_col*.STL` a directory holds, which is how a cut is compared against
    one hull, or against another cut, with this very code. A cut with more pieces
    than the model has geoms gets the extra geoms added, copied from the first; a
    cut with fewer leaves the spare geoms on its last piece, so a coarser cut is
    measured as a coarser cut rather than as a model that fails to compile.
    `--pieces assets/jumper/meshes` is the shared asset's one hull each: its
    `<link>_col.STL` is the only file there that matches.
    """
    spec = decompose_jaws(mujoco.MjSpec.from_file(model_path))
    if pieces is not None:
        swapped = 0
        for link in DECOMPOSED:
            files = sorted(pieces.glob(f"{link}_col*.STL"))
            if not files:
                continue
            body = next(b for b in spec.bodies if b.name == link)
            for i, path in enumerate(files):
                spec.add_mesh(name=f"_{link}_swap{i}", file=str(path.resolve()))
            geoms = [g for g in body.geoms if g.name.startswith(f"{link}_meshcol")]
            first = geoms[0]
            for i in range(len(geoms), len(files)):
                extra = body.add_geom(name=f"{link}_meshcol{i}", type=mujoco.mjtGeom.mjGEOM_MESH)
                extra.contype, extra.conaffinity, extra.group = (
                    first.contype, first.conaffinity, first.group)
                geoms.append(extra)
            for i, geom in enumerate(geoms):
                geom.meshname = f"_{link}_swap{min(i, len(files) - 1)}"
            swapped += 1
        if not swapped:
            raise SystemExit(f"no {' / '.join(f'{d}_col*.STL' for d in DECOMPOSED)} in {pieces}")
    JAW_COLLISION.edit_spec(spec)
    spec.worldbody.add_geom(name=PROBE, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[1e-9, 0, 0], pos=[0.0, 0.0, 0.0],
                            contype=0, conaffinity=0, mass=0.0)
    return spec.compile()


def stand(model: mujoco.MjModel, finger: float) -> mujoco.MjData:
    """The robot standing at `STAND_Z`, five legs at HOME, the arm at LF_GRASP."""
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    data.qpos[model.jnt_qposadr[base] + 2] = STAND_Z
    for joint, value in HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    for joint, value in LF_GRASP.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[jid]] = value
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)
    data.qpos[model.jnt_qposadr[fid]] = finger
    mujoco.mj_forward(model, data)
    return data


def jaw_cloud(model: mujoco.MjModel, data: mujoco.MjData,
              bodies: tuple[str, ...]) -> np.ndarray:
    """A jaw's drawn surface, in world coordinates.

    Vertices of the visual meshes through the **geom** frame: `mesh_vert` is
    expressed in the mesh's own re-oriented frame and the compiler puts the
    compensation on the geom, so reading it against the body frame -- which this
    repository did for a while -- scatters each part tens of millimetres from
    itself. `grasp_pose.py::Claw._mesh` carries the measurements.
    """
    out = []
    for body in bodies:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_visual")
        mesh = int(model.geom_dataid[gid])
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr:adr + num]
        out.append(verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    return np.vstack(out)


def _claw_geoms(model: mujoco.MjModel) -> list[int]:
    prefixes = tuple(f"{b}_meshcol" for b in MOVING_JAW + FIXED_JAW)
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .startswith(prefixes)]


def room(model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray,
         geoms: list[int]) -> float:
    """How far `point` is from the nearest collision shape of the claw, in metres.

    Zero when the point is inside one -- which is what a jaw hulled instead of
    decomposed does to the middle of the mouth.
    """
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROBE)
    data.geom_xpos[pid] = point
    return max(0.0, min(mujoco.mj_geomDistance(model, data, pid, g, 1.0, None)
                        for g in geoms))


def profile(model: mujoco.MjModel,
            finger: float) -> list[tuple[float, float, float, float]]:
    """`(band, gap across the mouth, room drawn, room collided)` along the mouth."""
    data = stand(model, finger)
    hinge = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        MOVING_JAW[0])]
    moving = jaw_cloud(model, data, MOVING_JAW)
    fixed = jaw_cloud(model, data, FIXED_JAW)
    # Preserve the tree's rejection of non-finite mesh coordinates.
    if not np.isfinite(moving).all() or not np.isfinite(fixed).all():
        cKDTree(np.vstack([moving, fixed]))
    geoms = _claw_geoms(model)

    far = cdist(moving, hinge[None, :]).ravel()
    outside = far > KNUCKLE_CLEAR
    fixed_outside = fixed[cdist(fixed, hinge[None, :]).ravel() > KNUCKLE_CLEAR]
    across = cKDTree(fixed_outside, leafsize=64,
                     compact_nodes=False, balanced_tree=False)
    bands = []
    for lo in BANDS:
        indices = np.flatnonzero(outside & (far >= lo) & (far < lo + BAND))
        if indices.size:
            bands.append((lo, indices))
    if not bands:
        return []
    centres = []
    original_across = None
    minima = []
    for lo, band_indices in bands:
        points = moving[band_indices]
        # A measured pair bounds the band's minimum without approximating it.
        bound = across.query(points[::max(1, len(points) // 32)])[0].min()
        gap, partner = across.query(points, distance_upper_bound=np.nextafter(bound, np.inf))
        if not np.isfinite(gap).any():
            # A zero/subnormal bound can underflow when the tree squares it.
            gap, partner = across.query(points)
        k = int(gap.argmin())
        point = points[k]
        nearest = partner[k]
        nearest_distances = across.query(point, k=2)[0]
        if nearest_distances[0] == nearest_distances[1]:
            if original_across is None:
                original_across = cKDTree(fixed_outside)
            nearest = original_across.query(point)[1]
        centre = 0.5 * (point + across.data[nearest])
        centres.append(centre)
        minima.append((float(lo), float(gap[k])))
    # There are only a few centres; scan in C instead of building another tree.
    distances = np.sqrt(np.minimum(cdist(centres, moving, "sqeuclidean").min(axis=1),
                                   cdist(centres, fixed, "sqeuclidean").min(axis=1)))
    out = []
    for (lo, gap), centre, distance in zip(minima, centres, distances):
        out.append((lo, gap, float(distance), room(model, data, centre, geoms)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(REPO / "assets" / "jumper" / "jumper.xml"))
    ap.add_argument("--finger", type=float, default=GRIPPER_OPEN,
                    help="finger angle to measure at (radians)")
    ap.add_argument("--pieces", type=Path, default=None,
                    help="a directory of LF_palm_link_col*.STL and/or LF_finger_link_col*.STL "
                         "to swap in, for comparing one cut of the jaws against another")
    args = ap.parse_args()

    model = build(args.model, args.pieces)
    print(f"room in the mouth at finger {args.finger:+.2f}, along its length\n")
    print(f"{'from hinge':>11} {'gap across':>12} {'drawn':>9} {'collided':>10} {'lost':>8}")
    worst = 0.0
    for lo, gap, drawn, collided in profile(model, args.finger):
        worst = max(worst, drawn - collided)
        print(f"{lo * 1000:9.0f}mm {gap * 1000:10.1f}mm {drawn * 1000:7.1f}mm "
              f"{collided * 1000:8.1f}mm {(drawn - collided) * 1000:6.1f}mm")
    print(f"\nworst loss {worst * 1000:.1f} mm against a "
          f"{ROOM_TOLERANCE * 1000:.0f} mm bar")
    if worst > ROOM_TOLERANCE:
        print("  FAIL -- a jaw is hulled where it should be decomposed "
              "(tasks/jumper/five_foot/jaws.py)")
        return 1
    print("  ok -- what the claw is drawn holding is what it is holding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
