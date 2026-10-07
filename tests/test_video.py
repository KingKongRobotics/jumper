"""Replay video must preserve simulated duration and the selected world's appearance.

A plausible MP4 at the wrong speed, or randomized state drawn with world zero's
model, gives no error. These checks pin those shared replay failures without a
robot task or a window.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from mjlab.viewer.viewer_config import ViewerConfig
from mjrl.viewer import video


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return _Tensor(self.value[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Frames:
    def __init__(self, *args):
        self.closed = False
        self.updates = 0

    def initialize(self):
        pass

    def update(self, data):
        self.updates += 1

    def render(self):
        return np.full((16, 32, 3), self.updates * 10 % 256, dtype=np.uint8)

    def close(self):
        self.closed = True


def _env(dt=0.02):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    return SimpleNamespace(
        num_envs=2, step_dt=dt, cfg=SimpleNamespace(viewer=ViewerConfig()),
        scene=SimpleNamespace(entities={}),
        sim=SimpleNamespace(mj_model=model, data=object()),
    )


@pytest.fixture
def recording(monkeypatch):
    encoded = []
    final = []

    def writer(*args, **kwargs):
        try:
            frame = yield
            while True:
                encoded.append(frame.copy())
                frame = yield
        finally:
            final.append(True)

    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(ffmpeg, "write_frames", writer)
    monkeypatch.setattr(video, "_ReplayRenderer", _Frames)
    return encoded, final


@pytest.mark.parametrize("dt,fps,steps,frames", [
    (0.02, 30, 50, 31),
    (0.1, 30, 10, 31),  # FPS above control Hz must repeat, not shorten playback.
    (0.02, 7, 50, 8),
    (1 / 60, 29.97, 600, 300),
])
def test_video_uses_simulation_time(recording, tmp_path, dt, fps, steps, frames):
    encoded, final = recording
    with video.ReplayVideo(_env(dt), tmp_path / "video.mp4", fps=fps,
                           width=32, height=16) as recorder:
        for _ in range(steps):
            recorder.update()
        assert recorder.frame_count == frames
        renderer = recorder._renderer
    assert len(encoded) == frames
    assert final == [True]
    assert renderer.closed
    recorder.close()
    assert final == [True]
    if fps > 1 / dt:
        # Control: distinct rendered states exist, while repeated video frames
        # fill the additional timestamps within each control step.
        assert np.array_equal(encoded[1], encoded[2])
        assert not np.array_equal(encoded[0], encoded[1])


def test_render_error_closes_encoder_and_gl(recording, tmp_path, monkeypatch):
    _, final = recording
    with (
        pytest.raises(RuntimeError, match="render failed"),
        video.ReplayVideo(_env(0.1), tmp_path / "video.mp4", width=32, height=16) as recorder,
    ):
        renderer = recorder._renderer
        monkeypatch.setattr(renderer, "render", lambda: _raise())
        recorder.update()
    assert renderer.closed
    assert final == [True]


def _raise():
    raise RuntimeError("render failed")


def test_initial_frame_failure_closes_resources(recording, tmp_path, monkeypatch):
    _, final = recording
    instances = []

    class Broken(_Frames):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)

        def render(self):
            return _raise()

    monkeypatch.setattr(video, "_ReplayRenderer", Broken)
    with (
        pytest.raises(RuntimeError, match="render failed"),
        video.ReplayVideo(_env(), tmp_path / "video.mp4", width=32, height=16),
    ):
        pass
    assert instances[0].closed
    assert final == [True]


XML = """
<mujoco>
 <worldbody><body name="robot" pos="0 0 1"><freejoint/>
  <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
  {visual}
  <geom name="collision" type="sphere" size=".2" group="4"/>
 </body></worldbody>
</mujoco>
"""


def _sim(stripped):
    full = mujoco.MjModel.from_xml_string(XML.format(
        visual='<geom name="appearance" type="sphere" size=".3" '
               'contype="0" conaffinity="0" group="1"/>'))
    physics = mujoco.MjModel.from_xml_string(XML.format(visual="")) if stripped else full
    fields = {
        name: _Tensor(np.stack([np.asarray(getattr(physics, name)).copy() for _ in range(2)]))
        for name in ("body_mass", "geom_pos", "geom_rgba", "geom_dataid")
    }
    fields["body_mass"].value[1, 1] = 3.0
    fields["geom_pos"].value[1, -1, 0] = 0.4
    fields["geom_rgba"].value[1, -1] = [1, 0, 0, 1]
    sim = SimpleNamespace(
        mj_model=physics, render_model=full, visuals_stripped=stripped,
        model=SimpleNamespace(**fields), expanded_fields=set(fields),
    )
    origins = _Tensor(np.zeros((2, 3)))
    entity = SimpleNamespace(num_actuators=1, indexing=SimpleNamespace(root_body_id=1))
    scene = SimpleNamespace(entities={"prop": SimpleNamespace(num_actuators=0),
                                     "robot": entity}, env_origins=origins)
    return sim, scene


@pytest.mark.parametrize("stripped", [False, True])
def test_model_sync_selects_world_and_preserves_appearance(stripped):
    sim, scene = _sim(stripped)
    cfg = ViewerConfig(env_idx=1, max_extra_envs=0)
    renderer = video._ReplayRenderer(sim, cfg, scene)
    renderer._sync_model_fields(1)
    model = renderer._model
    assert model.ngeom == 2  # A stripped collision-only model is not the video model.
    assert model.body_mass[1] == 3
    assert model.body_subtreemass[1] == 3  # Camera COM denominator follows randomization.
    assert model.geom_pos[-1, 0] == pytest.approx(.4)
    assert np.array_equal(model.geom_rgba[-1], [1, 0, 0, 1])
    assert model.geom_pos[0, 0] == 0  # Appearance-only geometry keeps its authored transform.
    assert sim.render_model.body_mass[1] == 1  # Render-only changes do not leak back.
    renderer._sync_model_fields(0)
    assert model.body_mass[1] == 1
    assert model.geom_pos[-1, 0] == 0  # Control: selecting world zero differs.


def test_camera_tracks_actuated_entity_and_keeps_floor_geometry():
    sim, scene = _sim(True)
    env = SimpleNamespace(cfg=SimpleNamespace(viewer=ViewerConfig()), sim=sim, scene=scene)
    cfg = video._camera_config(env, sim.render_model, width=640, height=480,
                               env_index=1, camera_distance=2.5)
    assert cfg.entity_name == "robot"
    assert cfg.origin_type == ViewerConfig.OriginType.ASSET_ROOT
    assert cfg.env_idx == 1 and cfg.max_extra_envs == 0 and cfg.distance == 2.5
    renderer = video._ReplayRenderer(sim, cfg, scene)
    assert renderer._cam.trackbodyid == 1
    # A body with only colliding geoms must remain visible (terrain is often a body).
    floor = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom type="plane" size="5 5 .1"/>'
        '</body></worldbody></mujoco>')
    cfg = video._camera_config(env, floor, width=640, height=480,
                               env_index=0, camera_distance=None)
    assert cfg.geom_group[0] == 1


def test_stripped_mesh_ids_are_mapped_by_asset_name():
    xml = """
    <mujoco><asset>
      <mesh name="visual_mesh" vertex="0 0 0 .3 0 0 0 .3 0 0 0 .3"/>
      <mesh name="collision_mesh" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/>
    </asset><worldbody><body name="robot"><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <geom name="appearance" type="mesh" mesh="visual_mesh" contype="0" conaffinity="0"/>
      <geom name="collision" type="mesh" mesh="collision_mesh"/>
    </body></worldbody></mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    full = spec.compile()
    spec.delete(spec.geom("appearance"))
    spec.delete(spec.mesh("visual_mesh"))
    physics = spec.compile()
    sim = SimpleNamespace(
        mj_model=physics, render_model=full, expanded_fields={"geom_dataid"},
        model=SimpleNamespace(geom_dataid=_Tensor(physics.geom_dataid[None])),
    )
    scene = SimpleNamespace(entities={}, env_origins=_Tensor(np.zeros((1, 3))))
    renderer = video._ReplayRenderer(sim, ViewerConfig(max_extra_envs=0), scene)
    renderer._sync_model_fields(0)
    collision = renderer._model.geom("collision").id
    expected = full.mesh("collision_mesh").id
    assert int(physics.geom_dataid[0]) != expected  # Numeric ids really did change.
    assert int(renderer._model.geom_dataid[collision]) == expected


def test_syncs_selected_batched_state_instead_of_native_thread_scratch():
    sim, scene = _sim(True)
    renderer = video._ReplayRenderer(sim, ViewerConfig(env_idx=1, max_extra_envs=0), scene)
    model = sim.mj_model
    d = mujoco.MjData(model)
    data = {}
    for name in ("qpos", "qvel", "ctrl", "act", "qfrc_applied", "xfrc_applied"):
        data[name] = _Tensor(np.stack([getattr(d, name).copy() for _ in range(2)]))
    data["qpos"].value[1, 0] = 4
    data["xfrc_applied"].value[1, 1, 0] = 17
    data = SimpleNamespace(**data, nworld=2)
    displayed = []
    renderer._renderer = SimpleNamespace(
        update_scene=lambda d, **kwargs: displayed.append(d.qpos.copy()),
    )
    renderer.update(data)
    assert displayed[0][0] == 4  # cfg.env_idx reaches model and state sync together.
    assert renderer._data.qpos[0] == 4
    assert renderer._data.xfrc_applied[1, 0] == 17
    assert renderer._data.xpos[1, 0] == 4
    renderer._sync_data_fields(data, 0)
    assert renderer._data.xpos[1, 0] == 0


def test_encoded_mp4_decodes_with_expected_dimensions_fps_and_frame_count(tmp_path, monkeypatch):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(video, "_ReplayRenderer", _Frames)
    path = tmp_path / "decoded.mp4"
    with video.ReplayVideo(_env(.1), path, fps=30, width=32, height=16) as recorder:
        for _ in range(10):
            recorder.update()
    decoded = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(decoded)
        assert metadata["size"] == (32, 16)
        assert metadata["fps"] == 30
        frames = list(decoded)
    finally:
        decoded.close()
    assert len(frames) == 31
    assert len(frames[0]) == 32 * 16 * 3
    # Encoding preserved a visible change, not 31 copies of the initial picture.
    assert frames[0] != frames[-1]


@pytest.mark.parametrize("kwargs", [
    {"fps": 0}, {"fps": float("nan")}, {"fps": float("inf")},
    {"width": 641}, {"height": 0}, {"env_index": 2}, {"env_index": -1},
    {"camera_distance": -1},
])
def test_invalid_video_configuration_fails_before_creating_resources(tmp_path, kwargs):
    with pytest.raises(ValueError):
        video.ReplayVideo(_env(), tmp_path / "video.mp4", **kwargs)
    assert not (tmp_path / "video.mp4").exists()


@pytest.fixture
def play_entrypoint(monkeypatch):
    # Entry points deliberately import _cli from their own script directory.
    # Supply that module without changing the first-party package search path.
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("_cli", scripts / "_cli.py")
    cli = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_cli", cli)
    spec.loader.exec_module(cli)
    spec = importlib.util.spec_from_file_location("_video_play_tests", scripts / "play.py")
    play = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(play)
    return play


@pytest.mark.parametrize("args,message", [
    ([], "--video requires a positive, finite --steps"),
    (["--steps", "0"], "--video requires a positive, finite --steps"),
    (["--steps", "10", "--video", "clip.gif"], "--video must name an .mp4 file"),
    (["--steps", "10", "--video-fps", "nan"], "--video-fps must be finite and positive"),
    (["--steps", "10", "--video-fps", "0"], "--video-fps must be finite and positive"),
    (["--steps", "10", "--video-width", "641"],
     "video dimensions must be positive even integers"),
    (["--steps", "10", "--video-height", "0"],
     "video dimensions must be positive even integers"),
    (["--steps", "10", "--video-env", "-1"], "--video-env must be nonnegative"),
    (["--steps", "10", "--video-distance", "inf"],
     "--video-distance must be finite and positive"),
])
def test_cli_rejects_invalid_recordings_before_resolving_or_allocating(
    play_entrypoint, monkeypatch, capsys, args, message,
):
    def should_not_resolve(_args):
        pytest.fail("Invalid video arguments reached environment resolution")

    monkeypatch.setattr(play_entrypoint, "resolve_all", should_not_resolve)
    monkeypatch.setattr(sys, "argv", ["play.py", "--video", "clip.mp4", *args])
    with pytest.raises(SystemExit) as error:
        play_entrypoint.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_rejects_out_of_range_world_but_accepts_a_valid_recording(
    play_entrypoint, monkeypatch, capsys,
):
    result = (SimpleNamespace(id="robot.motion"), SimpleNamespace(num_envs=2), None)
    monkeypatch.setattr(play_entrypoint, "resolve_all", lambda args: result)
    args = ["play.py", "--video", "clip.mp4", "--steps", "10", "--dry-run", "--video-env"]
    monkeypatch.setattr(sys, "argv", [*args, "2"])
    with pytest.raises(SystemExit) as error:
        play_entrypoint.main()
    assert error.value.code == 2
    assert "--video-env must be less than --num_envs" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*args, "1"])
    play_entrypoint.main()
    assert "--dry-run, stopping here" in capsys.readouterr().out
