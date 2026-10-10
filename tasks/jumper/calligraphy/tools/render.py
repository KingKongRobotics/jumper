#!/usr/bin/env python3
"""Step 4: the shots of a `write.py` run, and the ink for the compositor.

    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u65e0/<run>
    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py <run> --check
    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py <run> --no-ink --shots low

Writes into the run's directory:

    film.mp4            real time, the camera following the character being written,
                        then pulling back to the whole text and the robot beside it
                        (its outro), and holding on it -- the shot for a video
    top.mp4, low.mp4    the overhead and the low shot, fixed, real time, ink drawn as it
                        is laid (--no-ink for clean plates to composite onto)
    result.png          the finished text from above, 1920 x 1080: a thumbnail
    wu.gif              the overhead shot sped up, small, for the README
    ink.json            every mark: time, floor xy, depth, planned press, width, and its
                        pixels in each shot; the shots' intrinsics and poses
    ink.svg             the ink from above at the character's scale

**Plain MuJoCo, no mjlab, no torch.** The run's `model.mjb` and the `qpos` in its
`log.npz` are the whole state, replayed with `mj_forward`; nothing is simulated
again. That is also what makes it run on a machine with no GPU: OSMesa renders on
the CPU, and it crashes the process when torch is loaded beside it (measured on
this container, 2026-10-08: `import mujoco; import torch; import tensordict` under
MUJOCO_GL=osmesa dies; under egl it does not), so the renderer must not share a
process with the simulation.

`--check` puts markers at known floor points, renders them from each shot and
measures where they land against `cameras.Shot.project` -- the export's pixels
are only worth anything if that agrees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco

from tasks.jumper.calligraphy import cameras, hanzi, ink

#: 地书 is water on stone: the ink is the stone darkened, not black.
INK_RGBA = (0.22, 0.21, 0.21, 1.0)
#: The paving: light stone slabs with darker joints.
STONE = np.array([0.60, 0.58, 0.55])
JOINT = np.array([0.42, 0.40, 0.38])
HAZE = (0.86, 0.87, 0.88, 1.0)


def sky_horizon(m: mujoco.MjModel) -> np.ndarray | None:
    """The colour at the horizon of the model's skybox, or None if it has none: the
    middle row of the cube's first side face."""
    for tex in range(m.ntex):
        if m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            w, h, c = m.tex_width[tex], m.tex_height[tex], m.tex_nchannel[tex]
            img = m.tex_data[m.tex_adr[tex]: m.tex_adr[tex] + w * h * c].reshape(h, w, c)
            return img[w // 2, :, :3].mean(axis=0) / 255.0
    return None


def dress(m: mujoco.MjModel) -> None:
    """The training floor -- blue, chequered, reflective -- as paving stones.

    Only what is drawn changes: the texture's pixels, the material's reflectance and
    the fog. Contact, friction and geometry are the run's, untouched.
    """
    g = m.geom("terrain").id
    mat = m.geom_matid[g]
    tex = m.mat_texid[mat][1]  # mjTEXROLE_RGB
    w, h, c = m.tex_width[tex], m.tex_height[tex], m.tex_nchannel[tex]
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    slabs = 2  # per texture repeat
    sx, sy = (xx * slabs) // w, (yy * slabs) // h
    tint = 1.0 + 0.05 * rng.standard_normal((slabs, slabs))
    img = STONE[None, None, :] * tint[sy, sx][..., None]
    img *= 1.0 + 0.025 * rng.standard_normal((h, w))[..., None]
    joint = ((xx % (w // slabs)) < 3) | ((yy % (h // slabs)) < 3)
    img[joint] = JOINT
    data = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :c]
    adr = m.tex_adr[tex]
    m.tex_data[adr: adr + w * h * c] = data.ravel()
    m.mat_rgba[mat] = (1.0, 1.0, 1.0, 1.0)
    m.mat_reflectance[mat] = 0.0
    # Haze, or with a sky (write.py --scene) the sky's own horizon, so the floor
    # fades into it rather than into grey.
    horizon = sky_horizon(m)
    m.vis.rgba.fog = HAZE if horizon is None else (*horizon, 1.0)
    m.vis.map.fogstart = 1.0
    m.vis.map.fogend = 3.5
    # The scene's light is a spot 1.5 m up; from the low camera the edge of its
    # shadow map lies on the floor as dark wedges along the horizon. A low sun
    # from the side shades the same and casts shadows the whole floor agrees on.
    sun = np.array([0.35, -0.25, -1.0])
    sun /= np.linalg.norm(sun)
    m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_dir[0] = sun
    # A directional light's shadow map is a box along its ray from its position:
    # the ray has to pass where the robot is. The daylight scene's sun stands at
    # (-3, -2, 4), and turned to this direction its shadows fell 3 m away.
    m.light_pos[0] = np.array([0.3, 0.0, 0.0]) - 2.5 * sun
    if horizon is None:
        m.vis.headlight.ambient = (0.35, 0.35, 0.35)
        m.vis.headlight.diffuse = (0.40, 0.40, 0.40)
    else:
        # The scene's sun is brighter and bluer-lit than the training light;
        # at the values above the floor washed out to white.
        m.light_ambient[0] = (0.10, 0.11, 0.13)
        m.light_diffuse[0] = (0.70, 0.68, 0.64)
        m.vis.headlight.ambient = (0.30, 0.30, 0.30)
        m.vis.headlight.diffuse = (0.30, 0.30, 0.30)


def _add_disc(scene, x: float, y: float, r: float, rgba=INK_RGBA) -> bool:
    """A flat wet patch: an ellipsoid r x r x 0.3 mm, so it has no rim."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([r, r, 0.0003]),
                        np.array([x, y, ink.LIFT]), np.eye(3).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    g.specular = 0.0
    scene.ngeom += 1
    return True


def _backdrop(scene, shot: cameras.Shot) -> None:
    """A wall of haze behind everything, square to the camera: the model has no
    sky, and the low shot would otherwise put black above the horizon."""
    f = shot.forward()
    fh = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
    centre = shot.position() + 9.0 * fh
    right = np.cross(fh, (0.0, 0.0, 1.0))
    rot = np.column_stack([right, fh, (0.0, 0.0, 1.0)])
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([30.0, 0.01, 10.0]),
                        centre, rot.ravel(), np.asarray(HAZE, dtype=np.float32))
    g.emission = 1.0
    g.specular = 0.0
    # Decor casts no shadow; as a plain geom it laid dark bands along the horizon.
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def _add_ball(scene, p, r: float, rgba) -> None:
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                        np.asarray(p, dtype=float), np.eye(3).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


class Replay:
    def __init__(self, run: Path):
        self.m = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
        dress(self.m)
        self.sky = sky_horizon(self.m) is not None
        self.d = mujoco.MjData(self.m)
        self.log = ink.Log(run / "log.npz")
        if self.log.qpos is None:
            raise SystemExit(f"{run}/log.npz has no qpos; rerun write.py")
        self.plan = hanzi.Plan.from_json(json.loads((run / "plan.json").read_text()))
        self.t = self.log["t"]

    def at(self, t: float) -> None:
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        self.d.qpos[:] = self.log.qpos[i]
        mujoco.mj_forward(self.m, self.d)

    def renderer(self, shot: cameras.Shot) -> mujoco.Renderer:
        self.m.vis.global_.fovy = shot.fovy
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, shot.width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, shot.height)
        return mujoco.Renderer(self.m, shot.height, shot.width, max_geom=20000)


def text_frame(plan) -> tuple[tuple[float, float], float]:
    """The text's centre and its larger extent: a single character's em square, or
    the whole of a longer text."""
    x0, x1, y0, y1 = plan.bounds()
    return ((x0 + x1) / 2, (y0 + y1) / 2), max(plan.size, x1 - x0, y1 - y0)


def ink_points(marks) -> np.ndarray:
    """(n, 4) rows of t, x, y, radius: every inked point once, in time order, so
    a frame at time t draws a prefix."""
    if not marks:
        return np.zeros((0, 4))
    pts = np.concatenate([np.column_stack([m.t, m.xyz[:, :2], m.width / 2]) for m in marks])
    return pts[np.argsort(pts[:, 0])]


def frame(rp: Replay, r: mujoco.Renderer, shot: cameras.Shot, cam, pts, t: float) -> np.ndarray:
    """One picture of the run at time t; the ink laid by then, unless `pts` is None."""
    rp.at(t)
    r.update_scene(rp.d, camera=cam)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    if shot.elevation > -60 and not rp.sky:
        _backdrop(r.scene, shot)
    if pts is not None:
        n = np.searchsorted(pts[:, 0], t, side="right")
        for x, y, rad in pts[:n, 1:]:
            if not _add_disc(r.scene, x, y, rad):
                break
    return r.render()


def render_shot(rp: Replay, shot: cameras.Shot, marks, out: Path, fps: float, speed: float,
                with_ink: bool, hold: float = 2.0, scale: float = 1.0) -> list[np.ndarray]:
    """Frames of one shot; written to `out` as mp4 (or returned for a gif)."""
    r = rp.renderer(shot)
    cam = shot.mjv_camera()
    # Every inked point once, in time order, so a frame draws a prefix.
    pts = ink_points(marks)
    t_end = rp.t[-1]
    times = np.arange(0.0, t_end, speed / fps)
    times = np.concatenate([times, np.full(int(hold * fps), t_end)])
    frames = []
    writer = None
    if out.suffix == ".mp4":
        import imageio.v2 as imageio

        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8)
    for k, t in enumerate(times):
        img = frame(rp, r, shot, cam, pts if with_ink else None, t)
        if scale != 1.0:
            from PIL import Image

            img = np.asarray(Image.fromarray(img).resize(
                (int(img.shape[1] * scale), int(img.shape[0] * scale)), Image.LANCZOS))
        if writer is not None:
            writer.append_data(img)
        else:
            frames.append(img)
        if k % 200 == 0:
            print(f"  {shot.name}: frame {k}/{len(times)}", flush=True)
    if writer is not None:
        writer.close()
    r.close()
    return frames


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    # One palette for the whole clip, from a frame with all the ink on it: per-frame
    # palettes make the floor flicker between neighbouring greys.
    palette = imgs[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=True)


def check(rp: Replay, shots) -> float:
    """Markers at known floor points: rendered pixel vs `Shot.project`."""
    cx, cy = rp.plan.origin
    h = rp.plan.size / 2
    # Resting on the floor rather than centred in it: half a ball sunk in the floor
    # shows only its top from a low camera, and the visible part's centroid sits
    # 2-3 px above the centre -- an error in the check, not in the projection.
    rad = 0.006
    probe = np.array([[cx, cy, rad], [cx + h, cy + h, rad], [cx - h, cy - h, rad],
                      [cx + h, cy - h, rad], [cx - h, cy + h, rad]])
    rp.at(0.0)
    worst = 0.0
    for shot in shots:
        r = rp.renderer(shot)
        for p in probe:
            r.update_scene(rp.d, camera=shot.mjv_camera())
            _add_ball(r.scene, p, rad, (1.0, 0.0, 1.0, 1.0))
            img = r.render().astype(int)
            magenta = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)
            if not magenta.any():
                print(f"  {shot.name}: marker at {p[:2]} not visible")
                continue
            ys, xs = np.nonzero(magenta)
            got = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
            want = shot.project(p[None])[0]
            err = float(np.linalg.norm(got - want))
            worst = max(worst, err)
            print(f"  {shot.name}: marker {p[:2]} rendered at {got.round(1)}, "
                  f"projected {want.round(1)}: {err:.1f} px")
        r.close()
    return worst


#: The film: following the character being written from behind the robot's
#: shoulder, then pulling back to the whole text and the robot beside it.
#: Steep, so the robot in front of the character does not hide it: at -55 deg,
#: from behind its shoulder, the trunk covered the stroke being written.
FILM_FOLLOW = {"distance": 0.78, "elevation": -72.0, "azimuth": 0.0}
FILM_REVEAL_ELEVATION = -70.0
#: Seconds of simulation the camera takes to settle on a new target (both ways in
#: time, so it moves before a cut rather than lagging after it).
FILM_TAU = 1.5
FILM_HOLD = 5.0                   # s on the finished text at the end
#: Where the robot stands for the outro, beside the text: `write.OUTRO_STAND_OFF`.
OUTRO_STAND_OFF = 0.32


def _fit_distance(x_extent: float, y_extent: float, shot: cameras.Shot) -> float:
    """How far a camera looking straight down at az 0 (+x up, -y right) has to be
    for x_extent x y_extent metres of floor to fill the frame."""
    v = math.tan(math.radians(shot.fovy) / 2)
    h = v * shot.width / shot.height
    return max(x_extent / (2 * v), y_extent / (2 * h))


def film_track(rp: Replay, times: np.ndarray) -> list[cameras.Shot]:
    """One Shot per time: the character being written, then the reveal."""
    plan, log = rp.plan, rp.log
    outro = rp.log.phases.index("outro") if "outro" in rp.log.phases else -1
    char_of = np.array([s.char for s in plan.strokes])
    centres = {}
    for c in set(char_of.tolist()):
        xy = np.concatenate([s.xy for s in plan.strokes if s.char == c])
        centres[c] = (xy.min(0) + xy.max(0)) / 2
    x0, x1, y0, y1 = plan.bounds()
    # The text and the robot beside it (its trunk at y0 - OUTRO_STAND_OFF).
    ry0 = y0 - OUTRO_STAND_OFF - 0.12
    probe = cameras.Shot("film", (0, 0, 0), 1.0, 0.0, -90.0)
    # Never nearer than a metre: the robot stands 15 cm off the floor, and nearer
    # than that it fills the frame at the edge.
    reveal_d = max(1.0, 1.3 * _fit_distance(x1 - x0 + 0.08, y1 - ry0 + 0.04, probe))
    reveal = np.array([(x0 + x1) / 2 - 0.04, (y1 + ry0) / 2, 0.0, reveal_d,
                       FILM_REVEAL_ELEVATION, 0.0])
    end = rp.t[-1]
    targets = np.zeros((len(times), 6))
    revealing = False
    for k, t in enumerate(times):
        i = int(np.clip(np.searchsorted(rp.t, t), 0, len(rp.t) - 1))
        revealing = revealing or t >= end or log["phase"][i] == outro
        if revealing:
            targets[k] = reveal
            continue
        s = int(log["stroke"][i])
        c = int(char_of[s]) if 0 <= s < len(char_of) else 0
        # Between the character and the robot, which stands 10-30 cm from it.
        cx, cy = centres[c]
        bx, by = log["bx"][i], log["by"][i]
        sep = math.hypot(cx - bx, cy - by)
        targets[k] = (0.55 * cx + 0.45 * bx, 0.55 * cy + 0.45 * by, 0.0,
                      max(FILM_FOLLOW["distance"], 1.6 * sep + 0.35),
                      FILM_FOLLOW["elevation"], FILM_FOLLOW["azimuth"])
    # Zero-phase smoothing in simulated time: forward, then backward.
    dt = np.diff(times, prepend=times[0])
    a = 1.0 - np.exp(-np.maximum(dt, 1e-3) / FILM_TAU)
    sm = targets.copy()
    for k in range(1, len(sm)):
        sm[k] = sm[k - 1] + a[k] * (targets[k] - sm[k - 1])
    for k in range(len(sm) - 2, -1, -1):
        sm[k] = sm[k + 1] + a[k + 1] * (sm[k] - sm[k + 1])
    # The shot can lag the action; never the start and the end, which hold still.
    return [cameras.Shot("film", tuple(v[:3]), float(v[3]), float(v[5]), float(v[4]))
            for v in sm]


def render_film(rp: Replay, marks, out: Path, fps: float) -> None:
    """film.mp4: real time, the camera following, then the reveal and a hold."""
    import imageio.v2 as imageio

    end = rp.t[-1]
    times = np.concatenate([np.arange(0.0, end, 1.0 / fps),
                            end + np.arange(1, int(FILM_HOLD * fps) + 1) / fps])
    shots = film_track(rp, times)
    r = rp.renderer(shots[0])
    pts = ink_points(marks)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for k, (t, shot) in enumerate(zip(times, shots)):
        writer.append_data(frame(rp, r, shot, shot.mjv_camera(), pts, min(t, end)))
        if k % 200 == 0:
            print(f"  film: frame {k}/{len(times)}", flush=True)
    writer.close()
    r.close()


def render_result(rp: Replay, marks, out: Path) -> None:
    """result.png: the finished text from straight above, 1920 x 1080, and the
    robot beside it if it stands in the frame."""
    from PIL import Image

    x0, x1, y0, y1 = rp.plan.bounds()
    probe = cameras.Shot("result", (0, 0, 0), 1.0, 0.0, -90.0, width=1920, height=1080)
    d = 1.25 * _fit_distance(x1 - x0, y1 - y0, probe)
    shot = cameras.Shot("result", ((x0 + x1) / 2, (y0 + y1) / 2, 0.0), d, 0.0, -90.0,
                        width=1920, height=1080)
    r = rp.renderer(shot)
    Image.fromarray(frame(rp, r, shot, shot.mjv_camera(), ink_points(marks),
                          float(rp.t[-1]))).save(out)
    r.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a write.py output directory")
    ap.add_argument("--shots", nargs="+", default=["film", "top", "low"],
                    help="film (following, then the reveal), top, low")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-ink", action="store_true", help="clean plates, no ink drawn")
    ap.add_argument("--gif-speed", type=float, default=6.0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--still", type=float, nargs="+", metavar="T",
                    help="write <shot>_<T>.png at these times (s) and stop")
    ap.add_argument("--check", action="store_true",
                    help="check the projection against the renderer and stop")
    args = ap.parse_args()

    rp = Replay(args.run)
    shots = {s.name: s for s in cameras.shots(*text_frame(rp.plan))}
    if args.check:
        worst = check(rp, list(shots.values()))
        print(f"worst marker error: {worst:.1f} px")
        # Measured on run7 (2026-10-08): top <= 0.5 px, low 1.4-2.1 px, a residual
        # upward bias of the low shot that is under 1 mm on the floor.
        return 0 if worst < 2.5 else 1

    ms = ink.marks(rp.log, rp.plan)
    if args.still:
        from PIL import Image

        pts = None if args.no_ink else ink_points(ms)
        for name in args.shots:
            ts = [min(t, float(rp.t[-1]) + FILM_HOLD) for t in args.still]
            track = (film_track(rp, np.arange(0.0, max(ts) + 1e-9, 1.0 / args.fps))
                     if name == "film" else None)
            for t in ts:
                shot = track[min(round(t * args.fps), len(track) - 1)] if track \
                    else shots[name]
                r = rp.renderer(shot)
                path = args.run / f"{name}_{t:05.1f}.png"
                img = frame(rp, r, shot, shot.mjv_camera(), pts, min(t, float(rp.t[-1])))
                Image.fromarray(img).save(path)
                r.close()
                print(f"[render] wrote {path}")
        return 0
    data = ink.to_json(ms, rp.plan, list(shots.values()))
    (args.run / "ink.json").write_text(json.dumps(data))
    (args.run / "ink.svg").write_text(ink.to_svg(ms, rp.plan))
    print(f"[render] {len(ms)} marks, {sum(len(m.t) for m in ms)} points -> ink.json, ink.svg")

    if not args.no_ink:
        render_result(rp, ms, args.run / "result.png")
        print(f"[render] wrote {args.run / 'result.png'}")
    suffix = "" if not args.no_ink else "_plate"
    for name in args.shots:
        path = args.run / f"{name}{suffix}.mp4"
        if name == "film":
            render_film(rp, ms if not args.no_ink else [], path, args.fps)
            print(f"[render] wrote {path}")
            continue
        render_shot(rp, shots[name], ms, path, args.fps, 1.0, not args.no_ink)
        print(f"[render] wrote {path}")
    if not args.no_gif and not args.no_ink:
        fps = 15.0
        frames = render_shot(rp, shots["top"], ms, args.run / "wu.gif", fps, args.gif_speed,
                             True, hold=2.5, scale=0.5)
        write_gif(frames, args.run / "wu.gif", fps)
        size = (args.run / "wu.gif").stat().st_size / 1e6
        print(f"[render] wrote {args.run / 'wu.gif'} ({len(frames)} frames, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
