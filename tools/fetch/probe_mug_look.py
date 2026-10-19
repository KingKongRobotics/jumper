"""Render the mug prop alone, from the art side and the handle side: is the skin on
the right way up, is the text readable, and are the collision boxes hidden?"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")  # egl fails on this desktop (EGLError at context creation); glfw draws offscreen through the display
import importlib.util as _ilu
import sys
from pathlib import Path

import mujoco
from PIL import Image

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path
from mug import handle_boxes, mug_spec

spec = mug_spec()
spec.worldbody.add_light(pos=[1, 1, 2], dir=[-0.5, -0.5, -1], castshadow=False)
spec.worldbody.add_light(pos=[-1, -1, 2], dir=[0.5, 0.5, -1], castshadow=False)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], rgba=[0.8, 0.8, 0.8, 1])
model = spec.compile()
data = mujoco.MjData(model)
data.qpos[2] = 0.05
mujoco.mj_forward(model, data)
print("geoms:", [model.geom(i).name for i in range(model.ngeom)][:4], "...", model.ngeom)
print("handle boxes:", len(handle_boxes()))
print("mug mass", model.body_subtreemass[model.body("mug").id].round(4))
r = mujoco.Renderer(model, 480, 640)
opt = mujoco.MjvOption()
opt.geomgroup[5] = 0
cam = mujoco.MjvCamera()
cam.lookat = [0, 0, 0.05]; cam.distance = 0.35; cam.elevation = -15
out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
for name, az in (("art", 0), ("handle", 180), ("side", 90)):
    cam.azimuth = az
    r.update_scene(data, cam, opt)
    Image.fromarray(r.render()).save(out / f"mug_{name}.png")
# with the boxes shown, to see they sit on the skin
opt.geomgroup[5] = 1
cam.azimuth = 150
for gid in range(model.ngeom):
    if model.geom_group[gid] == 5:
        model.geom_rgba[gid] = [1, 0, 0, 0.5]
r.update_scene(data, cam, opt)
Image.fromarray(r.render()).save(out / "mug_boxes.png")
print("wrote", out)
