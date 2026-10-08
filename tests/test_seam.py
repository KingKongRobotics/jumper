"""The contract of the simulation-backend seam.

The seam lands in vendored mjlab and is this repository's **only** functional
change to upstream -- and if an upstream sync overwrites it, it fails
**silently**: `get_simulation_cls()` goes back to `Simulation`, `--backend
native` still resolves, training still runs, it just runs on warp forever.
Nothing raises. These assertions exist to make such an overwrite show up
immediately in the tests.

All of it is source-level text checking and **imports no mjlab**, so it runs on
a machine without the simulation dependencies too. Real behavioural validation
belongs on the training machine (see "4. Smoke test" in docs/AGENT_SETUP.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SIM_INIT = REPO / "rl" / "mjlab" / "sim" / "__init__.py"
ENV_FILE = REPO / "rl" / "mjlab" / "envs" / "manager_based_rl_env.py"
NATIVE = REPO / "rl" / "mjrl" / "backend" / "native_sim.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code(text: str) -> str:
    """Keep the code lines only, dropping comments.

    These assertions frequently have to prove that some construct **no longer**
    appears, and the comment explaining why it went is exactly where it gets
    mentioned. Searching a whole passage for the keyword would take the
    explanation itself for a violation -- a trap already hit once each on
    `_CATMASK`, `_scatter_model` and the thread-count guard.
    """
    return "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")
    )


# ── Discipline for changes on the vendored side ─────────────────────────


@pytest.mark.parametrize("path", [SIM_INIT, ENV_FILE], ids=lambda p: p.name)
def test_vendored_changes_carry_the_marker(path: Path) -> None:
    """docs/VENDOR.md's discipline: every change inside vendored carries `[mjrl]`.

    That marker is not decoration; it is the **only** way to find our own changes
    again when upstream is updated.
    """
    assert "[mjrl]" in read(path), (
        f"{path.relative_to(REPO)} was changed without an [mjrl] marker"
    )


def test_seam_is_confined_to_two_files() -> None:
    """The vendored-side changes must be **a known handful**.

    VENDOR.md requires them to stay concentrated; scattered across managers and
    entities, a rebase has nowhere to start. This pins the set down: when a
    third changed file appears, either it is genuinely necessary (update this
    list and say why) or it is a change in the wrong place.
    """
    # Scan vendored only: rl/mjrl is our own code and mentions [mjrl] as a
    # matter of course (resolve.py's banner prints "[mjrl] backend=...", say).
    marked = {
        p.relative_to(REPO).as_posix()
        for d in ("mjlab", "rsl_rl")
        for p in (REPO / "rl" / d).rglob("*.py")
        if "[mjrl]" in read(p)
    }
    expected = {
        # ── The backend seam itself ──
        "rl/mjlab/sim/__init__.py",  # the implementation registry
        "rl/mjlab/envs/manager_based_rl_env.py",  # the construction point
        # ── Sensors: dispatch ray casting per backend ──
        # The rest of RayCastSensor is backend-agnostic (prepare_rays /
        # postprocess_rays are pure PyTorch, and the ray buffers allocate fine
        # on a CPU device); only raycast_kernel needs model.struct.
        # SensorContext is about skipping mjwarp's RenderContext -- native uses
        # mj_ray and needs no BVH.
        "rl/mjlab/sensor/raycast_sensor.py",
        "rl/mjlab/sensor/sensor_context.py",
        # ── The sim-device registry: which Warp device simulates ──
        # Upstream takes the torch device's name. On Apple Silicon torch has
        # no Metal device, so "cpu" tensors alias arrays that warp-metal
        # simulates on metal:0; the registry beside the seam says so, and
        # Simulation reads it where it picked its Warp device (plus a wait
        # after each launch, because CPU tensors read unified memory
        # directly). sim_data took every non-cpu array for a CUDA one.
        "rl/mjlab/utils/sim_device.py",
        "rl/mjlab/sim/sim.py",
        "rl/mjlab/sim/sim_data.py",
        # ── Unrelated to the seam ──
        "rl/mjlab/rl/config.py",  # rsl_rl's symmetry field
    }
    assert marked == expected, (
        f"the set of vendored changes moved.\nExtra: {sorted(marked - expected)}\n"
        f"Missing: {sorted(expected - marked)}"
    )


# ── The seam itself ─────────────────────────────────────────────────────


def test_registry_defaults_to_upstream_simulation() -> None:
    """With nobody registered it must be mjlab's own `Simulation`, behaving
    exactly as upstream does.

    A default of native would quietly change upstream behaviour -- and mjlab's
    own Go1 / G1 tasks are the control group for "the seam changed nothing",
    so all of those would be wasted.
    """
    assert "_SIMULATION_CLS: type = Simulation" in read(SIM_INIT)


def test_registry_exposes_both_directions() -> None:
    src = read(SIM_INIT)
    assert "def set_simulation_cls(" in src
    assert "def get_simulation_cls(" in src
    assert "Simulation if cls is None else cls" in src, (
        "set(None) has to restore the default"
    )


def test_env_constructs_through_the_registry() -> None:
    """The construction point must go through the registry, not `Simulation(...)`."""
    src = read(ENV_FILE)
    assert "self.sim = get_simulation_cls()(" in src
    assert "from mjlab.sim import get_simulation_cls" in src


def test_env_does_not_import_simulation_directly() -> None:
    """Leave the direct import in place and it is easy for someone to slip back
    to `Simulation(...)` unnoticed."""
    src = read(ENV_FILE)
    assert "from mjlab.sim.sim import Simulation" not in src


# ── The native side's interface ─────────────────────────────────────────


def test_native_constructor_matches_mjlab_signature() -> None:
    """The two signatures must line up, or swapping the class makes that one
    construction call a TypeError."""
    # Anchor on NativeSimulation: _NativeData / _NativeModel in the same file
    # have an __init__ too.
    cls = read(NATIVE).split("class NativeSimulation", 1)[1]
    sig = cls.split("def __init__", 1)[1].split(") -> None:", 1)[0]
    for param in ("num_envs", "cfg", "model", "device", "spec", "variant_info"):
        assert param in sig, f"the constructor signature is missing {param}"


def test_native_applies_the_mujoco_cfg() -> None:
    """timestep / solver / gravity all live in cfg.mujoco; miss it and the two
    backends run different physics."""
    assert "cfg.mujoco.apply(mj_model)" in read(NATIVE)


def test_sensor_context_actually_wires_both_sensor_kinds() -> None:
    """`set_sensor_context` has to wire things up, not accept them silently.

    This test's criterion used to be "must raise NotImplementedError" -- back
    when native implemented neither sensor kind. Both are implemented now, so
    the criterion became "both are wired", but **it guards the same thing**: an
    empty implementation here means `sense()` never computes the sensors, and
    the height scan and depth image stay at their initial values -- no error, no
    NaN, and the policy trains on garbage all the way through.

    The handle for each: rays need the `MjModel` / `MjData`, cameras need a
    renderer to have been built.
    """
    body = read(NATIVE).split("def set_sensor_context", 1)[1].split("\n    def ", 1)[0]
    assert "sensor._native_mj_model" in body, "the ray sensor has no MjModel"
    assert "sensor._native_datas" in body, (
        "the ray sensor has no per-environment MjData"
    )
    assert "NativeCameraRenderer" in body, "the camera sensor has no offscreen renderer"


def test_sense_runs_the_camera_renderer() -> None:
    """`sense()` has to actually render, or the depth image stays the zeros it
    was allocated with.

    This is the hardest class to spot: `get_depth()` returns a correctly-shaped
    tensor as usual, the observation assembles as usual, training runs as usual
    -- every frame is simply identical. Hence pinning it in the source.
    """
    body = read(NATIVE).split("def sense(", 1)[1].split("\n    def ", 1)[0]
    assert "render_into" in body, "sense() never calls the camera rendering"


def test_native_camera_renders_all_three_data_types() -> None:
    """All three data types must genuinely be rendered.

    If `setup_context` allocates the buffers and `render_into` misses one, that
    type stays at the value it was allocated with -- right shape, right dtype,
    the observation assembles, every frame identical.
    """
    src = read(REPO / "rl" / "mjrl" / "sensor" / "camera.py")
    body = src.split("def render_into", 1)[1].split("\n    def ", 1)[0]
    for kind in ("_rgb_cams", "_depth_cams", "_seg_cams"):
        assert kind in body, f"render_into does not handle {kind}"


def test_native_camera_matches_mjwarp_pixel_conventions() -> None:
    """Two convention differences must be handled explicitly in the code, not
    left to "it looks normal".

    - **Depth background**: rasterisation gives zfar; mjwarp's convention is
      0.0. Without the conversion, background pixels are a number tens of metres
      large and the whole depth observation's scale is off.
    - **Geoms only**: `mjCAT_ALL` draws decor elements -- sites, joints, tendons
      -- into the depth image as well, while mjwarp casts against geoms only. A
      single site in the scene is enough to make the two disagree.
    """
    src = read(REPO / "rl" / "mjrl" / "sensor" / "camera.py")
    assert "_zfar" in src, "the background is not converted from zfar to 0.0"
    # Pin this on the line defining _CATMASK -- the documentation mentions
    # mjCAT_ALL while explaining why it is not used, and searching the whole
    # file for the keyword would take the explanation for a violation.
    catmask = next(ln for ln in code(src).splitlines() if ln.startswith("_CATMASK"))
    assert "mjCAT_STATIC" in catmask and "mjCAT_DYNAMIC" in catmask
    assert "mjCAT_ALL" not in catmask, "the catmask has to exclude mjCAT_DECOR"


def test_scatter_model_keeps_the_visual_statistics() -> None:
    """When domain randomisation recomputes the model constants, `stat`'s visual
    entries must be restored.

    `mj_setConst` recomputes `stat` from the geometry along with everything else,
    throwing away an explicitly set `extent` (mjlab's `SceneCfg.extent` writes
    straight into `spec.stat.extent`, and jumper sets 2.0). MuJoCo's znear/zfar
    are both `vis.map.* x stat.extent` -- so the per-environment models' frusta
    differ from the nominal model's by a factor, camera depth is scaled as a
    whole, and it **raises nothing, produces no NaN and looks entirely normal**.
    Measured on jumper: 2.0 recomputed to 0.6322, depth 3.16x too large.

    `meanmass` / `meaninertia` are deliberately not restored: they take part in
    the solver's scale normalisation and should follow randomised masses, which
    is the whole point of calling `mj_setConst`.
    """
    body = read(NATIVE).split("def _scatter_model", 1)[1].split("\n    def ", 1)[0]
    assert "mj_setConst" in body
    # Code lines only -- the comments naturally mention meaninertia (while
    # explaining why it is not restored), and searching the whole passage would
    # take the explanation for a violation. The same trap already caught
    # _CATMASK once.
    restore = "\n".join(ln for ln in code(body).splitlines() if "mi.stat." in ln)
    assert restore, "nothing restores mi.stat.* after mj_setConst"
    for field in ("extent", "center", "meansize"):
        assert field in restore, f"stat.{field} is not restored after mj_setConst"
    for field in ("meaninertia", "meanmass"):
        assert field not in restore, (
            f"stat.{field} should be recomputed, not restored"
        )


def test_camera_guards_against_frustum_mismatch() -> None:
    """It must raise when the model the frustum was built from differs from the
    one used to linearise depth.

    The root cause is fixed (the test above), but this class of bug **never
    exposes itself** -- depth is merely multiplied by a constant. Hence a second
    line of defence: blow up on the spot if a mismatch does appear, rather than
    finding out three days into training.
    """
    cam = read(REPO / "rl" / "mjrl" / "sensor" / "camera.py")
    assert "def _check_same_frustum(" in cam
    render_one = cam.split("def _render_one", 1)[1].split("\n    def ", 1)[0]
    assert "_check_same_frustum(" in render_one, "the guard is never called"


def test_native_camera_releases_gl_explicitly() -> None:
    """The GL context has to be released explicitly, not left to `__del__`.

    EGL raises `EGLError` on the interpreter's shutdown path (rendering itself
    is entirely fine; the fault is the free inside `close()`). Left to
    `__del__`, that traceback prints as the process exits, looks like a crash,
    and sends whoever comes next off in the wrong direction.
    """
    native = read(NATIVE)
    close_body = native.split("def close(", 1)[1].split("\n    def ", 1)[0]
    assert "_camera_renderer.close()" in close_body
    cam = read(REPO / "rl" / "mjrl" / "sensor" / "camera.py")
    assert "def close(" in cam


def test_sensordata_is_gathered() -> None:
    """`sensordata` must be both allocated and gathered -- contact sensors are
    built entirely on it.

    `ContactSensor` needs no separate native implementation (it reads
    `data.sensordata` from MuJoCo's built-in contact sensor), but it depends
    completely on that buffer being refreshed each step. Allocated but never
    gathered, what it reads is forever the initial value: no error, no NaN, and
    rewards like `air_time` / `foot_slip` train on a constant contact signal.
    `xmat` was exactly this -- a buffer added without being put on the gather
    list (DESIGN 4.9).
    """
    src = read(NATIVE)
    assert '"sensordata"' in src, "sensordata is not allocated"
    # The gather list moved into _build_copy_plans (to avoid three lookups per
    # field per environment in the inner loop), but what has to hold is
    # unchanged: sensordata must be among the fields gathered every step.
    plan = src.split("def _build_copy_plans", 1)[1].split("\n    def ", 1)[0]
    assert "sensordata" in plan, "sensordata is not in the copy plan"
    gather = src.split("def _gather_range", 1)[1].split("\n    def ", 1)[0]
    assert "_gather_head" in gather and "_gather_plan" in gather, (
        "_gather_range does not use that plan"
    )
    step = src.split("def step(", 1)[1].split("\n    def ", 1)[0]
    assert "_gather()" in step, (
        "step() has to gather, or the buffers stay on the previous step"
    )


def test_thread_count_is_decoupled_from_env_count() -> None:
    """The thread count must no longer be forced equal to the environment count.

    It used to require `nthread == num_envs`, on the grounds that "with fewer
    threads than environments the MjData get reused and the derived quantities
    are only the last environment's". Once the shuffling bug was fixed that
    reasoning no longer held -- the derived quantities are recomputed by
    `_refresh_derived` on the main-thread side.

    The cost of tying them together was measured (i9-14900KF, 24 cores): 4096
    environments need 4096 threads, and at 256 environments one rollout took
    19.2 ms, **slower than single-threaded at 17.1 ms**, where 24 threads take
    2.2 ms. Worse, it turned "use every core" into "change num_envs", and
    num_envs is PPO's batch size -- tuning performance forced a change to the
    learning problem.
    """
    src = read(NATIVE)
    assert "def _resolve_nthread(" in src
    ctor = src.split("class NativeSimulation", 1)[1].split("def __init__", 1)[1]
    ctor = ctor.split("\n    def ", 1)[0]
    assert "_resolve_nthread(nthread, num_envs)" in ctor
    assert "nthread == num_envs" not in code(ctor), (
        "the forced-equality guard should be gone"
    )
    # Only nthread scratch spaces are handed over -- rollout's data length has
    # to equal nthread.
    step = src.split("def step(", 1)[1].split("\n    def ", 1)[0]
    assert "self._datas[: self._nthread]" in step, (
        "rollout's scratch has to be sliced to nthread, not handed the whole _datas"
    )


def test_cpu_threads_config_is_actually_wired() -> None:
    """`MJRL_CPU_THREADS` has to actually reach the backend.

    This was broken once: present in `.env`, documented in `USAGE.md`, parsed by
    `_cli.py`, computed by `resolve()`, even printed by the banner as
    `threads=16` -- and never passed to `NativeSimulation`, so what actually ran
    was num_envs threads. **A config that lies is worse than no config**: it
    makes people believe they have tuned something.
    """
    src = read(REPO / "rl" / "mjrl" / "backend" / "select.py")
    assert "set_default_nthread(resolution.cpu_threads)" in src
    assert "def set_default_nthread(" in read(NATIVE)


def test_non_state_writables_go_through_control_spec() -> None:
    """`xfrc_applied` / mocap have to be passed per roll, not written into
    MjData beforehand.

    Each roll of `mujoco.rollout` does `mj_setState(FULLPHYSICS)` +
    `mj_setState(control, control_spec)`, and FULLPHYSICS holds only
    time/qpos/qvel/act/history/plugin. An `xfrc_applied` written into MjData
    beforehand has **no effect at all** -- measured: add 500 N to one MjData and
    roll out, and the displacement across 8 rolls is all zero (0.002 m when
    passed correctly per roll).

    Which is to say domain randomisation based on `xfrc_applied` has been idling
    on native the whole time: no error, no NaN, that randomisation simply never
    happened. jumper happens not to hit it (its push_robot writes qvel, and
    nmocap=0); another task would.

    This is also **the precondition for decoupling the thread count**: once the
    scratch spaces no longer correspond one-to-one with environments, anything
    "written into MjData beforehand" is necessarily misplaced.
    """
    src = read(NATIVE)
    spec = src.split("_CONTROL_SPEC: int = (", 1)[1].split("\n)", 1)[0]
    for bit in ("mjSTATE_CTRL", "mjSTATE_XFRC_APPLIED",
                "mjSTATE_MOCAP_POS", "mjSTATE_MOCAP_QUAT"):
        assert bit in spec, f"_CONTROL_SPEC is missing {bit}"
    step = src.split("def step(", 1)[1].split("\n    def ", 1)[0]
    assert "control_spec=_CONTROL_SPEC" in step, (
        "the rollout call does not pass control_spec"
    )
    assert "mj_getState(self._mj_model, d, ctrl_buf, _CONTROL_SPEC)" in step, (
        "packing order for control belongs to mj_getState, not a hand-written layout"
    )


def test_warmstart_is_passed_per_roll() -> None:
    """The solver warm start is passed per roll, and it has to be **that
    environment's own**.

    `mjSTATE_FULLPHYSICS`(8223) does **not** include `WARMSTART`(32), and
    rollout zeroes `qacc_warmstart` for every roll unless `initial_warmstart`
    is given (measured on mujoco 3.11). So it is given, read per environment
    from `_datas[i]`.

    That read is only right if `_datas[i]` holds environment i's. The first
    nthread are rollout's scratch and come back holding the warm start **and
    the ctrl** of whichever roll their thread ran last. Read back unrepaired,
    the thread schedule decided whose accelerations an environment's solver
    started from: two identical runs disagreed, as did two thread counts, and
    for the first nthread environments the actuator and contact forces the
    rewards read were computed from another environment's ctrl.
    `tests/test_native_determinism.py` measures it; this pins the repair.
    """
    src = read(NATIVE)
    step = src.split("def step(", 1)[1].split("\n    def ", 1)[0]
    assert "initial_warmstart=self._warmstart_in" in step
    assert "self._warmstart_in[i] = d.qacc_warmstart" in step, (
        "the warm start has to come per environment from its own MjData"
    )
    body = code(src.split("def _refresh_derived(", 1)[1].split("\n    def ", 1)[0])
    assert "control[i, 0], _CONTROL_SPEC" in body, (
        "rollout's scratch still holds another roll's ctrl; _refresh_derived has "
        "to put environment i's back before its forward"
    )
    assert "d.qacc_warmstart[:] = warmstart[i]" in body, (
        "rollout's scratch still holds another roll's warm start; _refresh_derived "
        "has to put environment i's back"
    )


def test_derived_refresh_is_parallel() -> None:
    """Recomputing the derived quantities has to use the thread pool -- it is
    the bulk of native's time.

    Measured (the real jumper model, 64 environments): 13-17 ms serial against
    1-2 ms for the rollout itself, so serial is over 85% of the total. MuJoCo's
    bindings release the GIL during `mj_forward` (measured 10.4x on 24 threads),
    so Python threads really are parallel here.

    The pool has to be **long-lived**: creating one per step costs more than it
    saves.
    """
    src = read(NATIVE)
    ctor = src.split("class NativeSimulation", 1)[1].split("def __init__", 1)[1]
    ctor = ctor.split("\n    def ", 1)[0]
    assert "ThreadPoolExecutor" in ctor, (
        "the pool belongs in the constructor, not in every step"
    )
    body = src.split("def _refresh_derived(", 1)[1].split("\n    def ", 1)[0]
    assert "self._fwd_pool.map" in body
    assert "list(self._fwd_pool.map" in body, (
        "map is lazy; without list() consuming it nothing waits for them -- "
        "half-computed state gets read"
    )
    close = src.split("def close(", 1)[1].split("\n    def ", 1)[0]
    assert "_fwd_pool.shutdown" in close, "the pool has to be shut down in close()"


def test_visual_meshes_are_stripped_before_the_model_is_multiplied() -> None:
    """Domain randomisation copies one `MjModel` per environment, and the bulk
    of it is visual meshes nobody uses.

    Measured on jumper: 44.41 MiB per copy, mostly the 31 mesh assets, while the
    hybrid collision scheme uses only the 6 toe meshes. Stripped, 2.29 MiB
    (-94.8%): 4096 environments go from **177 GiB (does not fit, killed by the
    OOM killer)** to 9.17 GiB.

    The stripping has to happen **before** `spec.compile()`: after compilation
    `ngeom` is fixed, and stripping then would leave the per-environment models
    disagreeing with the batch buffers on the geom count.
    """
    src = read(NATIVE)
    ctor = src.split("class NativeSimulation", 1)[1].split("def __init__", 1)[1]
    ctor = ctor.split("\n    def ", 1)[0]
    assert "_maybe_strip_visuals(spec, num_envs)" in ctor
    assert ctor.index("_maybe_strip_visuals") < ctor.index("spec.compile()"), (
        "stripping has to come before compile"
    )
    body = src.split("def _maybe_strip_visuals(", 1)[1].split("\ndef ", 1)[0]
    # The criterion is not "always strip" -- what is visible changes once it is
    # stripped (ngeom 83 -> 27; the live window and the cameras see collision
    # geometry). At debugging scale that is a pointless downgrade, so it is
    # decided by memory.
    assert "_STRIP_VISUALS" in body, "MJRL_STRIP_VISUAL has to be able to force it"
    assert "_MODEL_BUDGET_BYTES" in body, (
        "auto mode decides on a memory threshold"
    )


def test_strip_visual_config_is_wired() -> None:
    """`MJRL_STRIP_VISUAL` has to actually reach the backend -- no second dead
    config.

    `MJRL_CPU_THREADS` broke exactly that way: parsed, computed and printed all
    the way through, and never passed to the backend.
    """
    assert "set_strip_visuals(resolution.strip_visual)" in read(
        REPO / "rl" / "mjrl" / "backend" / "select.py"
    )
    assert "def set_strip_visuals(" in read(NATIVE)
    cli = read(REPO / "scripts" / "_cli.py")
    assert "MJRL_STRIP_VISUAL" in cli
    assert "strip_visual=" in cli, "the resolve() call does not pass it"
    env_keys = (REPO / ".env").read_text(encoding="utf-8")
    assert "MJRL_STRIP_VISUAL=" in env_keys


def test_variant_scenes_are_rejected() -> None:
    """Per-environment variants rest on mjwarp's per-world geom_dataid, which
    has no counterpart on the native side."""
    assert "if variant_info:" in read(NATIVE)


# ── Watching while training ─────────────────────────────────────────────


def test_native_exposes_live_mjdata() -> None:
    """native has to hand out live MjData with no copy -- the viewer's fast path
    depends on it."""
    src = read(NATIVE)
    assert "def env_mjdata(" in src
    assert "def attach_viewer(" in src


def test_viewer_syncs_from_the_sim_thread() -> None:
    """The frame has to be **taken** at the end of step() (after _gather), on the
    simulation thread -- not by a thread reading live state on a timer.

    A thread reading live state catches the rollout halfway through writing it, so
    the picture tears or clips -- and **there is no way to tell a rendering
    problem from a policy problem**, which turns this tool into something that
    misleads the very judgement it exists to support. What the viewer does with the
    copy happens on its own thread; `tests/test_viewer_thread.py` pins that the
    copy really is one.
    """
    step_body = read(NATIVE).split("def step(", 1)[1].split("\n    def ", 1)[0]
    assert "self._live_viewer.maybe_sync()" in step_body
    assert step_body.index("_gather()") < step_body.index("maybe_sync()")


def test_viewer_supports_both_backends() -> None:
    """Watching while training has to work on both backends.

    The documentation used to assert that rendering on warp would break the CUDA
    graph. **That was wrong** -- a graph is invalidated when an array it holds is
    replaced (reallocated), not when one is read, and rendering only reads.
    mjlab's own play renders live on warp. The only difference is the cost of
    fetching a frame.
    """
    live = read(REPO / "rl" / "mjrl" / "viewer" / "live.py")
    assert 'hasattr(sim, "env_mjdata")' in live, "native takes the zero-copy fast path"
    assert "def _pull_state(" in live, (
        "warp needs the slow path that fetches frames from device memory"
    )
    train = read(REPO / "scripts" / "train.py")
    assert 'res.backend != "native"' not in train, "warp should no longer be blocked"


def test_viewer_restores_the_patched_step() -> None:
    """The warp side hooks in by wrapping `sim.step`, so `stop()` has to restore
    it or frames keep being fetched after the window closes."""
    live = read(REPO / "rl" / "mjrl" / "viewer" / "live.py")
    unhook = live.split("def _unhook", 1)[1].split("def ", 1)[0]
    assert "self._sim.step = self._orig_step" in unhook


def test_viewer_reuses_mjlab_field_list() -> None:
    """The visual-field list comes from mjlab rather than being copied -- two
    copies of a list inevitably drift.

    mjlab's `sync_model_fields` is no longer used: it fetches from the device and
    writes into the host model in one step, and the viewer now has to split those
    across two threads -- the fetch on the simulation thread, with the rest of the
    frame, and the write on the frame thread. The list is what must not drift.
    """
    live = read(REPO / "rl" / "mjrl" / "viewer" / "live.py")
    assert "from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS" in live
    pull = live.split("def _pull_state(", 1)[1].split("\n    def ", 1)[0]
    assert "_viewer_model_fields()" in pull, "the frame does not fetch the DR'd fields"


def test_backend_is_selected_before_env_construction() -> None:
    """Registering the backend must precede building the env -- the construction
    point consults the registry once, and later changes have no effect."""
    src = read(REPO / "scripts" / "train.py")
    assert src.index("use_backend(res)") < src.index("ManagerBasedRlEnv(cfg=")


# ── Caught in real runs; kept from coming back ──────────────────────────


def test_step_reads_state_from_the_rollout_output() -> None:
    """State has to be read back from `state_out`, not taken from
    `self._datas[i]`.

    `mujoco.rollout` dispatches rolls to threads from a work queue, and `data`
    is only per-thread scratch -- `_datas[i]` holds whatever roll that thread
    processed last, unrelated to i. Calling `_gather()` directly, as it once
    did, means **the environments are reshuffled on every step**: the physics is
    computed perfectly correctly and stored into the wrong slot. During training
    the observation comes from environment A while the action was meant for
    environment B, and it **raises nothing, produces no NaN and does not
    diverge** -- it simply learns nothing.

    Caught on 2026-09-02 on the training machine by giving 4 environments
    different base x and stepping once (see DESIGN 4.9).
    """
    src = read(NATIVE)
    step_body = src.split("def step(", 1)[1].split("\n    def ", 1)[0]
    # Recomputing the derived quantities was factored into `_refresh_derived`
    # (to parallelise it), but **the source is unchanged**: rollout's state
    # output, indexed by roll, rather than letting downstream read _datas.
    assert "_refresh_derived(state_out[" in step_body, (
        "step() has to hand over rollout's state output"
    )
    assert step_body.index("rollout(") < step_body.index("_refresh_derived")
    body = src.split("def _refresh_derived(", 1)[1].split("\n    def ", 1)[0]
    assert "mj_setState" in body, (
        "the state has to be written back into each environment's own MjData"
    )
    assert "final_state[i]" in body, "index by roll, not by thread"


def test_native_exposes_nworld() -> None:
    """mjwarp's Data/Model carry a world count, and mjlab's entities, sensors
    and renderer all read it."""
    src = read(NATIVE)
    assert 'object.__setattr__(self, "nworld"' in src, "_NativeData needs it"
    assert 'if name == "nworld"' in src, "_NativeModel needs it"


def test_native_exposes_float32() -> None:
    """Upwards it has to be float32: MuJoCo is float64, mjlab is float32
    throughout, and mixing the two raises."""
    src = read(NATIVE)
    assert "UPCAST = np.float32" in src
    buffers = src.split("buffers: dict[str, np.ndarray] = {", 1)[1].split("\n        }", 1)[0]
    zeros = [ln for ln in buffers.splitlines() if "np.zeros" in ln]
    assert zeros, "no buffer definitions found"
    missing = [ln.strip() for ln in zeros if "dtype=UPCAST" not in ln]
    assert not missing, f"these buffers do not specify float32: {missing}"
    # The arrays fed to rollout have to be mjtNum (float64). `np.zeros` defaults
    # to float64, but that is **incidental, not a guarantee** -- require it
    # spelled out, so a change cannot drift silently.
    for name in ("_state_in", "_control", "_state_out", "_sensor_out"):
        line = next(
            (ln for ln in src.splitlines() if f"self.{name} = np.zeros" in ln), None
        )
        assert line, f"cannot find the definition of {name}"
        assert "dtype=np.float64" in line, (
            f"{name} has to declare float64 explicitly: {line.strip()}"
        )


def test_single_step_residual_is_documented_as_unsettled() -> None:
    """The native/warp single-step residual (4-6e-4) is provisional, not a settled result.

    The design document lists three things that were never checked: whether the residual is
    bounded, whether it varies with contact depth, and whether it affects training. This test
    stops that section being simplified into "the two backends agree".
    """
    import re

    design = read(REPO / "docs" / "DESIGN.md")
    sec = design.split("### 9.3", 1)[1].split("\n### ", 1)[0]
    # Collapse whitespace: the assertions must survive the document being re-wrapped.
    sec = re.sub(r"\s+", " ", sec)
    assert "not yet settled" in sec or "Provisionally acceptable" in sec
    assert "comparing learning curves" in sec, "the route to settling it must be named"


def test_viewer_sets_model_before_pulling_state() -> None:
    """`start()` has to assign `self._model` before taking the first frame.

    `_take_frame` (through `_pull_state` on warp) and `_apply` use `self._model`.
    The order was once the other way round and the warp path raised
    `AttributeError: no attribute '_model'` the moment a window opened -- while
    native took the `env_mjdata` zero-copy branch, which skipped `_pull_state`
    entirely, so neither static review nor native-side validation touched it.
    **Watching while training on warp therefore never worked**, until a window was
    actually opened on the training machine on 2026-09-02 (see DESIGN 4.10). Both
    backends take a first frame now, which closes that particular gap, but the
    order is still the thing to pin.
    """
    body = read(REPO / "rl" / "mjrl" / "viewer" / "live.py")
    start = body.split("def start(", 1)[1].split("def stop(", 1)[0]
    assert "self._model, self._data = model, data" in start
    assert start.index("self._model, self._data") < start.index("self._take_frame(")


def test_viewer_counts_actual_frames() -> None:
    """The frames taken and the frames drawn both have to be observable, not just
    how often the hook was called.

    The rate limiting is inside `maybe_sync`: before the interval elapses it
    returns without taking a frame. So "what does watching cost the simulation" is
    answered by `taken` -- a copy of the state each -- and "is the window keeping
    up" by `synced`, counted on the frame thread after `handle.sync()`. Measured
    with the synchronous design: 240 calls pushed 7-18 frames.
    """
    body = read(REPO / "rl" / "mjrl" / "viewer" / "live.py")
    hook = body.split("def maybe_sync", 1)[1].split("\n    def ", 1)[0]
    assert hook.index("self._min_interval") < hook.index("self.taken += 1")
    assert "self.synced" not in hook, "frames are drawn on the frame thread, not here"
    loop = body.split("def _frame_loop", 1)[1].split("\n    def ", 1)[0]
    assert loop.index("handle.sync()") < loop.index("self.synced += 1")


def test_train_passes_runner_cfg_as_dict() -> None:
    """The runner takes a dict, not a dataclass.

    `if key in train_cfg` in `mjlab/rl/runner.py` requires something iterable;
    passing the dataclass gives `TypeError: argument of type
    'RslRlOnPolicyRunnerCfg' is not iterable`. mjlab's own train.py also does
    `asdict(cfg.agent)` before passing it. This is what the first real run of
    scripts/train.py hit on 2026-09-02.
    """
    src = read(REPO / "scripts" / "train.py")
    assert "asdict(agent_cfg)" in src
    assert "runner_cls(wrapped, runner_kwargs" in src, "pass the converted dict"
