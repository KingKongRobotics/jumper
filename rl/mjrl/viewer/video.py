"""Stream a replay to MP4, sampling control steps in simulation time.

Call ``update()`` once after each environment step. The initial state is written
on entering the context. When the requested FPS exceeds the control frequency,
the latest state is repeated; no intermediate physics states are invented.
Set ``MUJOCO_GL=egl`` before importing MuJoCo on a headless Linux GPU runtime.
"""

from __future__ import annotations

import copy
import math
from contextlib import suppress
from pathlib import Path

import mujoco
import numpy as np
from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from mjrl.viewer.live import LiveViewer

__all__ = ["ReplayVideo"]


class _ReplayRenderer(OffscreenRenderer):
    """Use the native appearance model without assigning stripped geom arrays to it."""

    def __init__(self, sim, cfg, scene):
        self._sim = sim
        physics = sim.mj_model
        model = getattr(sim, "render_model", physics)
        self._geom_map = None
        if model.ngeom != physics.ngeom:
            # Stripping deletes geoms and meshes, but keeps joints and bodies.
            # Numeric geom/mesh ids are therefore NOT interchangeable.
            ids = []
            for i in range(physics.ngeom):
                name = physics.geom(i).name
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if not name or j < 0:
                    raise ValueError("Cannot map stripped render geometry without matching names")
                ids.append(j)
            self._geom_map = np.asarray(ids)
        self._physics_model = physics
        super().__init__(model, cfg, scene, sim.model, sim.expanded_fields)
        if cfg.origin_type == ViewerConfig.OriginType.AUTO:
            # AUTO in the vendored offscreen renderer currently uses a free
            # camera. A scene with no entity metadata can still follow a body.
            moving = np.flatnonzero(model.body_dofnum)
            if len(moving):
                self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._cam.trackbodyid = int(moving[0])

    def _sync_model_fields(self, env_idx: int) -> None:
        # Query each time: expansion can happen after a renderer was constructed.
        fields = self._sim.expanded_fields & VIEWER_MODEL_FIELDS
        if self._geom_map is None:
            sync_model_fields(self._model, self._sim.model, fields, env_idx)
            self._sync_subtree_mass(fields)
            return
        for name in fields:
            src = getattr(self._sim.model, name)[env_idx].cpu().numpy()
            dst = getattr(self._model, name)
            if not name.startswith("geom_"):
                dst[:] = src.reshape(dst.shape)
                continue
            src = src.reshape((self._physics_model.ngeom, *dst.shape[1:]))
            if name == "geom_dataid":
                src = src.copy()
                for i in range(len(src)):
                    kind = int(self._physics_model.geom_type[i])
                    obj = {
                        int(mujoco.mjtGeom.mjGEOM_MESH): mujoco.mjtObj.mjOBJ_MESH,
                        int(mujoco.mjtGeom.mjGEOM_HFIELD): mujoco.mjtObj.mjOBJ_HFIELD,
                    }.get(kind)
                    if obj is not None and src[i] >= 0:
                        asset = mujoco.mj_id2name(self._physics_model, obj, int(src[i]))
                        mapped = mujoco.mj_name2id(self._model, obj, asset or "")
                        if mapped < 0:
                            raise ValueError("Cannot map stripped geometry asset to render model")
                        src[i] = mapped
            dst[self._geom_map] = src
        self._sync_subtree_mass(fields)

    def _sync_subtree_mass(self, fields):
        if "body_mass" in fields and "body_subtreemass" not in fields:
            # Native recomputes this derived field on its per-world MjModels,
            # not in the expanded input buffers. Tracking uses subtree_com;
            # leaving the nominal denominator here shifts the camera silently.
            mass = self._model.body_subtreemass
            mass[:] = self._model.body_mass
            for i in range(self._model.nbody - 1, 0, -1):
                mass[self._model.body_parentid[i]] += mass[i]

    def _sync_data_fields(self, data, env_idx: int) -> None:
        # The upstream renderer copies qpos/qvel/mocap. These remaining inputs
        # matter to mj_forward too (for example, an actuator-driven tendon).
        for name in ("ctrl", "act", "qfrc_applied", "xfrc_applied"):
            if hasattr(data, name) and getattr(self._data, name).size:
                dst = getattr(self._data, name)
                dst[:] = getattr(data, name)[env_idx].cpu().numpy().reshape(dst.shape)
        if hasattr(data, "time"):
            self._data.time = float(data.time[env_idx].cpu().numpy().reshape(-1)[0])
        super()._sync_data_fields(data, env_idx)


def _camera_config(env, model, *, width, height, env_index, camera_distance):
    cfg = copy.deepcopy(getattr(getattr(env, "cfg", None), "viewer", ViewerConfig()))
    cfg.width, cfg.height = width, height
    cfg.env_idx, cfg.max_extra_envs = env_index, 0
    if camera_distance is not None:
        cfg.distance = camera_distance
    if cfg.origin_type == ViewerConfig.OriginType.AUTO:
        entities = env.scene.entities
        robot = next((name for name, entity in entities.items()
                      if getattr(entity, "num_actuators", 0)), None)
        if robot is None and len(entities) == 1:
            robot = next(iter(entities))
        if robot is not None:
            cfg.origin_type = ViewerConfig.OriginType.ASSET_ROOT
            cfg.entity_name = robot
    stripped = getattr(env.sim, "visuals_stripped", False) and model is env.sim.mj_model
    if stripped:
        cfg.geom_group = (1, 1, 1, 1, 1, 1)
    else:
        hidden = LiveViewer._collision_only_groups(model)
        cfg.geom_group = tuple(0 if i in hidden else v for i, v in enumerate(cfg.geom_group))
    return cfg


class ReplayVideo:
    """Record one environment without retaining video frames in memory.

    ``with ReplayVideo(env, path) as video: ... video.update()`` starts with an
    initial frame and finalizes the encoder on every exit. The recorder owns its
    renderer, never ``env``. Rendering and updates run on the simulation thread.
    ``frame_count`` includes the initial frame. At time T the file contains
    ``1 + floor(T * fps)`` frames, with at most one frame of duration rounding.
    """

    def __init__(self, env, path, *, fps=30, width=640, height=480,
                 env_index=0, camera_distance=None):
        self._env = getattr(env, "unwrapped", env)
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("Replay video output must have an .mp4 extension")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or value <= 0 or value % 2:
                raise ValueError(f"Video {name} must be a positive even integer")
        if not isinstance(env_index, int) or not 0 <= env_index < self._env.num_envs:
            raise ValueError("Video environment index is out of range")
        if camera_distance is not None and (
            not math.isfinite(camera_distance) or camera_distance <= 0
        ):
            raise ValueError("Video camera distance must be finite and positive")
        self.fps = float(fps)
        self._dt = float(self._env.step_dt)
        if not math.isfinite(self._dt) or self._dt <= 0:
            raise ValueError("Environment step_dt must be finite and positive")
        self._width, self._height = width, height
        self._env_index, self._camera_distance = env_index, camera_distance
        self._renderer = self._writer = None
        self._steps = 0
        self.frame_count = 0
        self._started = False

    def __enter__(self):
        if self._started:
            raise RuntimeError("ReplayVideo instances can only be entered once")
        self._started = True
        try:
            import imageio_ffmpeg

            sim = self._env.sim
            model = getattr(sim, "render_model", sim.mj_model)
            cfg = _camera_config(
                self._env, model, width=self._width, height=self._height,
                env_index=self._env_index, camera_distance=self._camera_distance,
            )
            self._renderer = _ReplayRenderer(sim, cfg, self._env.scene)
            self._renderer.initialize()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio_ffmpeg.write_frames(
                str(self.path), (self._width, self._height), fps=self.fps,
                codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                macro_block_size=1, ffmpeg_log_level="error",
            )
            next(self._writer)
            self._write_frames(1)
        except BaseException:
            self._close_after_error()
            raise
        return self

    def _write_frames(self, count):
        self._renderer.update(self._env.sim.data)
        frame = self._renderer.render()
        for _ in range(count):
            self._writer.send(frame)
            self.frame_count += 1

    def update(self):
        """Record frames due after one completed environment control step."""
        if self._writer is None:
            raise RuntimeError("Enter the ReplayVideo context before updating")
        self._steps += 1
        # Integer frame counts prevent floating deadline accumulation. A small
        # tolerance keeps e.g. 30 * 0.1 at its mathematically exact boundary.
        target = 1 + math.floor(self._steps * self._dt * self.fps + 1e-9)
        count = target - self.frame_count
        if count:
            try:
                self._write_frames(count)
            except BaseException:
                self._close_after_error()
                raise

    def close(self):
        """Finalize MP4 and release GL resources; safe to call more than once."""
        writer, self._writer = self._writer, None
        renderer, self._renderer = self._renderer, None
        try:
            if writer is not None:
                writer.close()
        finally:
            if renderer is not None:
                renderer.close()

    def _close_after_error(self):
        # Preserve the original initialization/rendering error if cleanup fails.
        with suppress(Exception):
            self.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._close_after_error()
        return False
