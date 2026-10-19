"""Convert `Coffee Mug.glb` (Y up, handle along +x) into the mug prop's embedded assets.

Writes `assets/mug_tex.png`, `assets/mug_mesh_c_handle.npz` (the model as made:
vert, face, uv in the mug's frame -- cup axis at the origin, mid-height at z = 0,
handle along -x; the cup's axis in the model, (-0.011, 0.008) across x and z, was
fitted to its bottom ring) and `assets/mug_mesh.npz`, the one the prop draws: the
same cup and artwork with the handle replaced by a loop (its own mesh: it is drawn as a grip, untextured).

Why the loop: the model's handle is a C of solid 15 x 18 mm section whose opening
sits behind the bar, so the claw can only pinch the bar from the side like a rod,
nothing of the handle lies under the finger, and the lifted mug swings cup-down
(61-77 deg in beach4/beach5). The loop's thin lower bar hangs on the finger's
underside, 18 mm below the pinch, and the mug hangs within a few degrees of level
(mug run 6, the placeholder). Its dimensions are the collision loop's in mug.py,
so what the claw meets is what the picture shows.

    .venv/bin/python tools/fetch/convert_mug.py "Coffee Mug.glb"
"""
import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import trimesh

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
from mug import BAR, BODY_R, HANDLE_OUT, LOOP_HALF_Z

CX, CZ, HALF_H = -0.011, 0.008, 0.05


def loop_handle() -> trimesh.Trimesh:
    """A loop of square section (corners rounded) swept round a rounded rectangle in
    the x-z plane, from inside the cup's wall out to the collision bar and back."""
    out_x = -(BODY_R + HANDLE_OUT)          # the outer bar's centre line
    in_x = -(BODY_R - 0.006)                # buried in the wall
    r = 0.010                                # corner radius of the path (the bars stay straight between)
    path = [[in_x, 0.0, LOOP_HALF_Z]]
    for ang in np.linspace(0, np.pi / 2, 8):   # top corner, then the outer bar, then the bottom corner
        path.append([out_x + r - r * np.sin(ang), 0.0, LOOP_HALF_Z - r + r * np.cos(ang)])
    for ang in np.linspace(0, np.pi / 2, 8)[1:]:
        path.append([out_x + r - r * np.cos(ang), 0.0, -LOOP_HALF_Z + r - r * np.sin(ang)])
    path.append([in_x, 0.0, -LOOP_HALF_Z])
    path = np.array(path)
    # The section: a square of half-side BAR with 1.5 mm corner radii, 20 points round.
    cr, pts = 0.0015, []
    for cx, cy, a0 in ((BAR - cr, BAR - cr, 0), (-(BAR - cr), BAR - cr, np.pi / 2),
                       (-(BAR - cr), -(BAR - cr), np.pi), (BAR - cr, -(BAR - cr), 3 * np.pi / 2)):
        for a in np.linspace(a0, a0 + np.pi / 2, 5):
            pts.append((cx + cr * np.cos(a), cy + cr * np.sin(a)))
    sec = np.array(pts)
    n = np.array([0.0, 1.0, 0.0])           # the path is planar: one normal for every ring
    rings = []
    for i, c in enumerate(path):
        t = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        b = np.cross(t, n)
        rings.append(c + sec[:, :1] * n + sec[:, 1:] * b)
    rings = np.array(rings)                 # (nring, nsec, 3)
    nr, ns = rings.shape[:2]
    v = rings.reshape(-1, 3)
    f = []
    for i in range(nr - 1):
        for j in range(ns):
            a, b_, c, d = i * ns + j, i * ns + (j + 1) % ns, (i + 1) * ns + (j + 1) % ns, (i + 1) * ns + j
            f += [[a, b_, c], [a, c, d]]
    for base, flip in ((0, True), ((nr - 1) * ns, False)):   # caps (fans; the section is convex)
        for j in range(1, ns - 1):
            tri = [base, base + j, base + j + 1]
            f.append(tri[::-1] if flip else tri)
    m = trimesh.Trimesh(v, np.array(f), process=False)
    if m.volume < 0:
        m.invert()
    return m


src = Path(sys.argv[1] if len(sys.argv) > 1 else "Coffee Mug.glb")
out = Path(__file__).resolve().parent / "assets"
out.mkdir(exist_ok=True)
scene = trimesh.load(src, force="scene")
(g,) = scene.geometry.values()
v = g.vertices.astype(np.float64)
p = np.c_[-(v[:, 0] - CX), v[:, 2] - CZ, v[:, 1] - HALF_H]   # (x, y, z)_glTF -> (-x, z, y): a proper rotation
faces, uv = g.faces.astype(np.int32), g.visual.uv.astype(np.float64)
np.savez_compressed(out / "mug_mesh_c_handle.npz", vert=p.astype(np.float32), face=faces, uv=uv.astype(np.float32))
g.visual.material.baseColorTexture.convert("RGB").save(out / "mug_tex.png")

# The cup alone: drop every face with a vertex beyond the cup's wall (the model's handle).
radius = np.hypot(p[:, 0], p[:, 1])
keep = (radius[faces] <= 0.0412).all(axis=1)
cup_faces = faces[keep]
used = np.unique(cup_faces)
remap = -np.ones(len(p), dtype=np.int64); remap[used] = np.arange(len(used))
cup_v, cup_uv, cup_f = p[used], uv[used], remap[cup_faces]

h = loop_handle()
hv, hf = np.asarray(h.vertices), np.asarray(h.faces, dtype=np.int64)
np.savez_compressed(out / "mug_mesh.npz", vert=cup_v.astype(np.float32), face=cup_f.astype(np.int32),
                    uv=cup_uv.astype(np.float32), hvert=hv.astype(np.float32), hface=hf.astype(np.int32))
print(f"cup: {len(cup_f)} of {len(faces)} faces kept; loop handle: {len(hf)} faces, watertight {h.is_watertight}; "
      f"handle x {hv[:, 0].min():.4f}..{hv[:, 0].max():.4f} z {hv[:, 2].min():.4f}..{hv[:, 2].max():.4f}")
