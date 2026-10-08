# Design

## 1. Goals

1. **The simulator is always MuJoCo.** PhysX / Isaac Sim are off the training path.
2. **Use the GPU when there is one; train for real on CPU when there is not.** The CPU path is
   not a debugging prop — it has to saturate many cores.
3. **Interface with rsl_rl.** Use a mature PPO implementation rather than writing our own.

## 2. Architecture

[mjlab](https://github.com/mujocolab/mjlab) is "Isaac Lab API, powered by MuJoCo-Warp". It has
already done the MuJoCo backend and the rsl_rl integration (`rsl-rl-lib==5.4.2` is a core
dependency). What it lacks is a CPU training backend.

This framework adds a seam between mjlab's manager layer and the physics engine:

```
┌─────────────────────────────────────────────┐
│  rsl_rl · OnPolicyRunner (PPO)              │
├─────────────────────────────────────────────┤
│  VecEnv adapter                             │  mjlab's own,
├─────────────────────────────────────────────┤  backend-agnostic,
│  Manager layer                              │  unchanged
│  observation · reward · termination ·       │
│  event · command · action                   │
├─────────────────────────────────────────────┤
│  Entity / Data — batched state tensors      │
│                 [N, ...]                    │
└─────────────────────────────────────────────┘
        ╌╌╌╌╌╌╌╌ the seam ╌╌╌╌╌╌╌╌
              ↓                        ↓
   ┌────────────────────┐   ┌────────────────────┐
   │ mjlab Simulation   │   │ NativeSimulation   │
   │ mujoco_warp        │   │ native mujoco      │
   │ one kernel over N  │   │ N×MjData + threads │
   └────────────────────┘   └────────────────────┘
            GPU                    many-core CPU
```

**Why the seam sits at that height.** The four layers above it all rest on the same set of
`[N, ...]` torch tensors and do not care whether the memory behind them is device or host.
One layer up (at env level) you end up with two environment implementations; one layer down
(at `mjData` level) there is no room for two different parallelism models.

Quantified coupling: across the whole of mjlab there are only **14 real `mjwarp.<fn>()` calls
in 5 files**.

| File | Calls | How it is handled |
|---|---|---|
| `sim/sim.py` | 10 | The seam itself; `NativeSimulation` implements the same interface |
| `entity/variants.py` | 1 | `put_model`; per-world mesh variants are not supported on native |
| `sensor/sensor_context.py` | 1 | Camera rendering; native goes through `mujoco.Renderer` (§6) |
| `envs/mdp/actions/differential_ik.py` | 1 | Differential-IK action term; not supported on native |
| `utils/nan_guard.py` | 1 | Debugging utility |

`entity/data.py` reads and writes state purely through attribute access and tensor indexing:

```python
self.data.qpos[env_ids, self.indexing.free_joint_q_adr] = pose
pos_w = self.data.xpos[:, self.indexing.root_body_id]
```

It touches no mjwarp API. So a backend that offers the same attribute surface with the same
`[N, ...]` shapes can be dropped in with the Entity and Manager layers untouched.

## 3. The seam: replace `Simulation`, do not add a protocol layer

`rl/mjrl/backend/native_sim.py`'s `NativeSimulation` is not a new abstraction. It is a
**member-for-member substitute for mjlab's `Simulation`**, dropped into the same slot.

### Why no separate protocol

Managers, entities and the scene already read through mjlab's `Simulation` interface
(`sim.data`, `sim.model`, `sim.step()` …). Inserting a custom protocol between them and the
physics engine has only two outcomes:

- **Change the managers to fit the protocol** — which directly contradicts the goal of leaving
  the manager layer untouched; or
- **Write a protocol-to-`Simulation` adapter** — pure indirection, and every new field then has
  to be kept in sync in three places (protocol, warp implementation, native implementation).

mjlab's `Simulation` is already a good enough interface: what it holds is exactly the batched
`[N, ...]` tensor views, independent of whether they live in device or host memory. **Given the
interface exists, the seam is just "let this interface have a second implementation".**

### The interface surface: 14 members

The env, scene and domain-randomisation layers use 14 members of `sim`, and
`NativeSimulation` implements all of them:

| Member | Purpose | On the native side |
|---|---|---|
| `data` | Batched state tensor views | `_NativeData`; writes land in buffers and are scattered back into `MjData` before stepping |
| `model` | Batched model field views | `_NativeModel`; a measured list decides which fields carry a world dimension |
| `mj_model` | A single `MjModel`, for introspecting joint names and limits | ✅ |
| `device` | | Always `cpu` |
| `step` / `forward` / `reset` / `sense` | Advance and refresh | `step` through the `mujoco.rollout` thread pool, `forward` through a pool of its own running `mj_forward`, `reset` per environment, `sense` as in §6 |
| `expand_model_fields` / `expanded_fields` | Per-environment model fields for domain randomisation | Lazily materialises N `MjModel` copies (§5.6) |
| `get_default_field` / `per_world_default_fields` / `recompute_constants` | Randomisation baselines and derived constants | Baselines from the unmodified shared model; `mj_setConst` on a scratch `MjData` (§5.7) |
| `set_sensor_context` | Camera / raycast sensors | Rays through `mj_multiRay` / `mj_ray`, cameras through `mujoco.Renderer` (§6) |

On mjlab's side, `set_sensor_context` mainly re-captures the CUDA graph
(`self._sensor_context = ctx; self.create_graph()`). The native path has no graph to protect,
so all it does is wire the two kinds of sensor to their native implementations. **It must not
be a no-op**: silently accepting the context without wiring means `sense()` never actually
computes sensors, depth maps and height scans stay at their initial values, nothing raises and
nothing goes NaN, and the policy trains to convergence on garbage.

Conversely, the **warp-only** members of mjlab's `Simulation` (`wp_model`, `wp_data`,
`create_graph`, `apply_wp_opt`, `world_to_variant`) are never touched by the env layer and the
native side does not provide them.

### Where the seam lands

`mjlab/sim/__init__.py` gains a registry (`set_simulation_cls` / `get_simulation_cls`), and the
one construction site in `mjlab/envs/manager_based_rl_env.py` goes from `Simulation(...)` to
`get_simulation_cls()(...)`. With nothing registered, `get_simulation_cls()` returns
`Simulation` and behaviour is byte-for-byte upstream. See [`VENDOR.md`](VENDOR.md) for the full
list of vendored modifications.

## 4. The two backends

| Combination | Role | Training path |
|---|---|---|
| `warp` · `cuda` | Primary training | yes |
| `native` · `cpu` | Real training without a GPU | yes |
| `warp` · `cpu` | Kernel-logic debugging, cross-checking | no |

MuJoCo Warp does support `wp.set_device("cpu")`, but that compiles the kernels to CPU code and
runs them **serially**; upstream positions it as "development and debugging". Saturating many
cores means going back to native MuJoCo and batching through `mujoco.rollout`'s thread pool.

### Measured throughput

Host: i9-14900KF (8 P-cores + 16 E-cores, 32 threads). Raw physics, env-steps/s, `nstep=4`.

**Minimal biped** (`smoke_biped`, nv=10, 8 primitive geoms, 2 ms timestep):

| envs | 1 thread | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| 32 | 218.8k | 1.01M | 669.1k | 646.7k | 813.1k |
| 128 | 469.2k | 1.40M | 1.79M | 1.02M | 801.0k |
| 512 | 451.2k | 1.43M | 2.35M | 2.42M | 1.12M |
| 2048 | 398.4k | 2.06M | 2.50M | **3.09M** | 3.01M |

**Real hexapod**, the previous model (V1.5: nv=28, 63 geoms including 31 meshes, 1 ms timestep,
shared model):

| envs | 1 thread | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| 32 | 3.8k | 23.2k | 32.9k | 41.3k | 54.9k |
| 128 | 3.9k | 27.1k | 41.6k | 54.2k | 63.4k |
| 512 | 4.0k | 28.8k | 46.0k | 60.6k | **67.8k** |

The speedup from 1 to 24 threads is about 6.6× on the smoke model and about 17× on the hexapod
from 1 to 32. The more compute there is per environment, the thinner the thread-pool overhead
spreads and the better it scales. The smoke model falls back at 32 threads because it is
already limited by memory bandwidth and E-core scheduling; the hexapod is still compute-bound.

End-to-end training at 4096 environments (same host, RTX 5090):

| Backend | Iteration time | Memory |
|---|---|---|
| `warp:cuda` | 0.82 s | 5.4 GiB VRAM, 3.2 GiB RAM |
| `native:cpu`, 32 threads | 26.0 s | 20 GiB RAM |

## 5. The native backend

### 5.1 What `mujoco.rollout` actually guarantees

Three properties, established by measurement rather than read off the documentation, and they
determine the whole shape of this backend:

- **The `data` list must have exactly `nthread` entries, not `num_envs`.** The error is
  `ValueError: Length of data: 16 not equal to nthread: 8`. `MjData` is **per-thread scratch**,
  not a per-environment state container.
- **`model` may be a single instance or a list of length nbatch** — the latter is what makes
  per-environment domain randomisation possible.
- **Batched state lives in the `initial_state` / `state` numpy arrays**, and those can be
  preallocated and reused indefinitely.

It follows that rollout itself needs only **thread count** `MjData`, not one per environment.
The backend still keeps one per environment for derived quantities (§5.3), and the copies
between those and the batch buffers are Python loops (§13).

### 5.2 `data[i]` is not environment `i`

This is the single most dangerous property of the interface. `mujoco.rollout` dispatches rolls
to threads from a **work queue**; `data[i]` holds whatever roll that thread processed last, and
that has nothing to do with `i`. Setting `nthread == num_envs` does not make the mapping the
identity.

The failure mode is silent: nothing raises, nothing goes NaN, nothing diverges — the policy
simply does not learn. It shows up only when comparing **per environment** against warp: give
four environments base x of 0/10/20/30, take one step, and the native result is a *permutation*
of the warp result (`native[0]=warp[3]`, `[1]=[1]`, `[2]=[0]`, `[3]=[2]`), each environment
physically correct to 1e-4 but in the wrong slot.

Only the `state` and `sensordata` output arrays are indexed by roll (i.e. by environment). So
state is always read back from `state_out`, and derived quantities are then recomputed with a
per-environment `mj_forward` — `xpos`, `site_xpos` and `cvel` exist only inside `MjData` and are
not in the output arrays. The cost is one extra forward per environment per step.

### 5.3 Threads are decoupled from environments

Thread count and environment count are two different things. Forcing `nthread == num_envs`
costs badly: at 256 environments a single rollout takes **19.2 ms, slower than the 17.1 ms of a
single thread**, while 24 threads take 2.2 ms. More fundamentally it turns "use all the cores"
into "change num_envs" — and **num_envs is PPO's batch size, so tuning performance would mean
changing the learning problem.**

The measured optimum is **independent of environment count** and sits well below the core
count: eight, at 16, 64 and 256 environments alike on a 32-thread i9-14900KF, where every core
cost 16–46% more and the physical core count (24) measured no better. Roughly half of a step is
the serial gather and scatter, so by Amdahl the parallel part is spent by about eight workers;
`resolve.DEFAULT_MAX_THREADS` holds the table. `chunk_size` is best left at its
default of 1; raising it is strictly worse (with 64 rolls and chunk=32, only two threads have
work).

`self._datas` serves two roles at once: rollout's **per-thread** scratch (nthread of them) and
**per-environment** storage for derived quantities (num_envs of them). Only the first `nthread`
entries are handed to rollout.

### 5.4 Control that FULLPHYSICS does not carry

Each roll performs `mj_setState(FULLPHYSICS)` followed by
`mj_setState(control, control_spec)`, and `mjSTATE_FULLPHYSICS` contains only
time/qpos/qvel/act/history/plugin. Anything written into `MjData` in advance —
`xfrc_applied`, `mocap_pos`, `mocap_quat` — **is discarded**. Measured: apply 500 N to one
`MjData`, roll out, and all 8 rolls displace by exactly 0.0 (0.002 m when passed correctly).

So domain randomisation based on `xfrc_applied` would silently never happen: no error, no NaN,
just an absent randomisation. The fix is
`control_spec = CTRL | XFRC_APPLIED | MOCAP_POS | MOCAP_QUAT`, with the packing order left to
`mj_getState` rather than hand-written.

This is also a **precondition for decoupling thread count**: once scratch buffers no longer
correspond one-to-one with environments, anything "written into `MjData` in advance" is
necessarily misplaced.

The solver's warm start needs the same care, in the other direction. The `WARMSTART` bit is not
in FULLPHYSICS either, and rollout **zeroes** `qacc_warmstart` for every roll unless
`initial_warmstart` is given (measured on 3.11: the result is bitwise independent of what the
scratch `MjData` held). It is given, per roll, read from `_datas[i]` — and from the day threads
were decoupled from environments (2026-09-03), that read was wrong. The scratch buffers are the
first nthread of `_datas`, and rollout hands them back holding the warm start **and the
control** of whichever roll their thread ran last (§5.2). Read back as they stood, the thread
schedule decided whose accelerations an environment's solver started from, and the forward that
recomputes the derived quantities ran on another environment's `ctrl` — so for the first nthread
environments the actuator and contact forces that rewards read before `forward()`
(`standing_foot_load`, for one) were another environment's.

Nothing raised. Measured on the i9-14900KF with `jumper.tripod`, 64 environments on 8 threads
(2026-09-26): environments 0-7 recomputed with a foreign `ctrl` at 95% of physics steps, their
actuator forces off by a median of 1.26 N·m against a typical 1.13, and their contact forces by
a median of 1.93 N against a typical 4.96. Two identical runs of `jumper.tripod`, `jumper.flat`
and `jumper.swing` disagreed from the second physics step on, and in what the actor was shown
within 40 control steps. With two environments the float64 state first disagreed near round-off
(1e-16..5e-15), which the float32 rounding in `_scatter` usually erases; when a difference
survives the rounding it grows (to 4e-8 within 48 physics steps in one run), and now and then it
reaches an observation.

`_refresh_derived` now puts each environment's own control and warm start back before its
forward, and carries the solution found there into the next step as that environment's warm
start. `tests/test_native_determinism.py` holds it: two runs, and one thread against several,
agree bit for bit.

### 5.5 The model bridge

mjwarp gives a leading world dimension only to fields that **can vary per world**; pure
topology and index fields do not get one. The upper layers therefore mix two indexing
conventions:

```python
model.body_ipos[:, root_body_id]     # world dimension
model.geom_bodyid[geom_id]           # no world dimension
```

A native implementation that adds the leading dimension uniformly makes the second form
silently return an entire row instead of a scalar.

The split is a **measured list**, `WORLD_BATCHED_FIELDS` (103 with, 192 without). It cannot be
guessed: among the fields without a world dimension there are 29 float64 and 4 float32, so
"integers are topology" is wrong — `geom_matid` and `geom_rgba` carry a world dimension,
`geom_friction` does, and `geom_type` does not. `tools/checks/warp_dims.py` regenerates the
list; rerun it after upgrading mjlab.

A related trap: `expand`ed broadcast views **are writable** in torch.
`model.geom_friction[0,0,0] = 9.9` succeeds and writes shared memory, so every environment
changes at once with no indication. Someone would believe each environment had a different
friction coefficient while they were all identical, and the resulting policy would be far less
robust than expected. `_SharedView` (a `torch.Tensor` subclass) refuses all in-place writes on
such views; indexing and slicing keep pointing at the same shared memory and stay read-only.

### 5.6 Domain randomisation and the memory wall

Per-environment model parameters mean N copies of `MjModel`. Measured per copy on the
previous model (V1.5), the cost is almost entirely in things that are identical across
environments:

| Part | Size | Share |
|---|---|---|
| `bvh_*` (collision acceleration structures) | 29.88 MB | 69.4% |
| `mesh_*` | 12.80 MB | 29.7% |
| `tex_*` / `mat_*` | 0.27 MB | 0.6% |
| **actual physics parameters** | **0.08 MB** | **0.2%** |

`bvh_aabb` alone is 22.41 MB (float64). Meshes and BVHs are identical per environment and
copying them is pointless, but `deepcopy` does not know that.

`strip_visual_meshes` removes mesh geoms whose contype and conaffinity are both zero, together
with the mesh assets that are orphaned as a result; `strip_noncolliding_geoms` then removes
any non-mesh geom that does not collide. On the current `jumper.xml` a single copy goes from
75.5 MiB to 4.9 MiB (MuJoCo 3.11, `model_slim.model_nbytes`), so 4096 environments from about
300 GiB to 20 GiB; on V1.5 it was 41.8 to 3.3 MiB.

`MjData` is not the problem — 0.5 MiB per environment, 2 GiB at 4096.

Two constraints that must hold:

- **Stripping has to happen after the colliding geoms are chosen.**
  `CollisionCfg(disable_other_geoms=True)` decides which geoms collide, by zeroing
  contype/conaffinity on the rest at the spec stage. Called earlier, stripping does nothing
  useful: on V1.5 geoms dropped from 83 to 52 while nmesh stayed at 31 and the model size did
  not move.
- **Inertia has to be verified every time.** When a body has no explicit `<inertial>`, MuJoCo
  **derives** mass and inertia from its geoms, and on such a model deleting visual geoms
  silently changes the dynamics. The jumper's 41 bodies all carry explicit `<inertial>` (they
  come from URDF) so it is unaffected, but that is a property of this model rather than a rule,
  so the function compares per-body mass, centre of mass and inertia before and after and
  raises on any difference.

Whether to strip is decided by memory: stripping changes what the camera sensor sees (collision
geometry rather than appearance meshes), which is a pointless downgrade at small scale, while
the alternative at large scale is being killed by the OOM killer. The threshold is the
per-environment models exceeding 2 GiB in total; `--strip-visual on|off` forces it either way.

The live viewer is **not** affected: `_maybe_strip_visuals` compiles the pre-strip model anyway
in order to measure it, and keeps that one copy as `sim.render_model` (75.5 MiB, under 0.4%
of the 20 GiB). Stripping touches no body or joint, so qpos and qvel are interchangeable
bit-for-bit between the two models.

### 5.7 `mj_setConst` uses its `MjData` as scratch

After randomising mass or inertia, `mj_setConst` must be called to recompute derived constants.
It treats its second argument as a scratch pad: it sets `qpos` to `qpos0`, runs kinematics to
compute `body_invweight0` and friends, **and does not restore anything**.

```
before mj_setConst: qpos[:3] = [ 1.43 -1.23  0.14]
after  mj_setConst: qpos[:3] = [ 0.    0.    0.  ]
```

Passing a live per-environment `MjData` therefore wipes every environment's pose. Nothing
raises. What it looks like in practice is every robot piled at the world origin at the start of
training, then migrating to its grid cell one at a time as episodes time out and reset — which
reads as "the picture gradually fixed itself" but is in fact physics state that was erased once.
A dedicated scratch `MjData` is used instead.

`mj_setConst` also recomputes `stat`, which discards the values the scene set explicitly. Three
of those (`extent`, `center`, `meansize`) are purely visual and are restored afterwards;
`meanmass` and `meaninertia` are not, because they participate in the solver's scale
normalisation and *should* follow the randomised masses. Getting this wrong is invisible:
MuJoCo derives znear/zfar from `vis.map.* × stat.extent`, so a wrong `extent` scales every
camera depth by a constant factor while the depth map's shape, foreground mask and segmentation
all look perfectly normal.

## 6. Sensors on the native side

### 6.1 Ray casting

`RayCastSensor` has exactly one mjwarp-bound method; `prepare_rays` and `postprocess_rays` are
pure PyTorch and the ray buffers allocate fine on a CPU device. The native implementation sends
rays that share an origin — a pinhole, the jumper's dToF — through one `mujoco.mj_multiRay` per
environment per frame, and the rest (the height scans, whose rays each have their own origin)
through `mujoco.mj_ray`; the two agree bit for bit, and for the dToF's 2268 rays the one call
took 0.94 ms a frame against 10.47 for the loop (`rl/mjrl/sensor/raycast.py` has the
measurement). Agreement with warp, measured through `mj_ray`: maximum difference
**3.539e-08**.

### 6.2 Camera: depth and segmentation

Measured against mjwarp's `render()` with `mujoco.mj_ray` as an analytic third opinion
(hexapod, 2 cameras, 3 environments):

| Criterion | Result |
|---|---|
| Depth, median relative error vs warp | **0.000e+00** |
| Depth, 99th percentile relative error | 2.2e-05 |
| Depth, maximum absolute difference | 1.6e-04 (depths of order 1.19 m) |
| Segmentation, disagreeing pixels | 61/5184, **100% of them on object silhouettes** |

Two implementation details matter:

**Multisampling destroys depth.** With MSAA enabled, `mjr_readPixels` has to resolve a
multisample depth blit down to a single sample, and **how several samples combine into one is
implementation-defined**. On NVIDIA with EGL/GLFW this introduces a constant offset in
**inverse** depth (`1/z_read = 1/z_true + ε`, ε ≈ 4.9e-3 m⁻¹), which becomes a depth error
growing as z²: 1.5% at 3 m, 4.6% at 10 m. Setting `offsamples=0` takes the median relative
error from 1.0e-02 to 2.6e-05.

This is invisible on macOS, where MuJoCo bypasses that path in `mjRND_SEGMENT` mode — the same
code, the same MuJoCo version and the same scene differ by three orders of magnitude between
two machines. Disabling MSAA has no downside and is more correct anyway: depth is a geometric
quantity for which anti-aliasing is meaningless, and segmentation is per-pixel integer labels
that MSAA would blend at edges.

**Sites are not decor.** mjwarp projects only geoms, while `mjv_updateScene` also draws sites,
tendons and joint axes by default. A single site adds a phantom blob to the native depth map
that warp does not have. `mjCAT_DECOR` does not filter it — sites, like geoms, are sorted into
`STATIC`/`DYNAMIC` by their owning body — so `MjvOption.sitegroup` has to be zeroed.

### 6.3 Camera: RGB, and where the backends part ways

RGB is the one of the three that **cannot** be required to match per pixel: mjwarp does its own
ray shading, MuJoCo rasterises through OpenGL. Far more of it matches than expected — with
textures off, the two differ by **0.1/255** per channel on average, with structural correlation
**r=0.9999** in a controlled scene and a chromaticity difference of 1e-4. **The shading model,
the lighting, the material base colours and the geometry all agree.**

The difference is **entirely in texture sampling**, for two independent reasons:

1. **mjwarp does not read `texuniform`** (the string does not appear in its source). It always
   samples "repeat per unit length" while MuJoCo obeys the model. For a textured material with
   `texuniform=false` even the texture scale differs (a checkerboard alternates 3 versus 13
   times along a scanline). **It is warp that departs from the model here**, so the native
   implementation does not imitate it — but the camera warns when the context is created.
2. Even with `texuniform=true`, the two texture pipelines filter differently (rasterisation has
   mipmapping and bilinear filtering; ray casting point-samples). A textured ground plane
   correlates at 0.85 in a controlled scene where the untextured version is at 0.9999; on the
   hexapod's terrain the per-channel difference is 50/255.

**So with textures enabled, `native:cpu` and `warp:cuda` RGB are not equivalent.** For a
visual policy the texture usually dominates the frame (the hexapod's downward view is 1606 of
1728 pixels ground), so "change the backend without changing the task" does not hold for RGB.
Setting `use_textures=False` on the camera makes the two agree (per-channel difference back to
0.1/255). Depth and segmentation are unaffected.

Three further things align the native renderer with warp: `_align_render_flags` turns off the
effects mjwarp does not have (reflections, fog, haze); `use_textures=False` is implemented as
`matid = -1` (setting `texid` has no effect); MSAA stays off, since mjwarp casts one ray per
pixel and has no anti-aliasing to match.

### 6.4 Contact sensors need no separate implementation

`ContactSensor` reads `data.sensordata` — what it adds to the spec is MuJoCo's **built-in**
contact sensor, and native already allocates and gathers that buffer. mjwarp appears only in
type annotations: no kernel, no `.struct`, and `requires_sensor_context` is `False`.

Verified anyway, because it sits on two known hazards (sensordata is gathered from
`self._datas[i]`, and contact forces depend on the constraint solver, which differs between
backends):

| Quantity | Result |
|---|---|
| **Contact booleans** (`found > 0`, which is what the gait rewards use) | **0/24 differ** |
| Per-environment feet in contact (envs raised 0.5/1/1.5 m) | native `[6,0,0,0]` = warp `[6,0,0,0]` |
| Contact force magnitude | median relative difference 0.090, max 0.967, forces of order 54.7 N |
| `found` (matched contact points, not the boolean) | native **3** vs warp **4** |

The first two are the criteria; the last two are reported but not asserted. Force differences
share a root cause with the single-step residual (§9.3). The `found` difference is that the two
collision routines generate different numbers of contact points, which does not reach anything
downstream because air-time tracking uses `found > 0`.

### 6.5 Cost

3 environments × 2 cameras (48×36 and 24×18, all three data types): native 8.7 ms per step of
which `sense()` is 2.0 ms (23%); warp 6.6 ms per step of which `sense()` is 0.5 ms (8%). The
native path renders **once per environment per camera per data type** — `Renderer` is a mode
switch and produces one kind at a time — so it grows linearly with environment count. Beyond a
thousand environments it would need a batched implementation, the same shape of limit as ray
casting.

## 7. The live viewer

The viewer is on by default on both backends; `--headless` turns it off, and it falls back to
headless automatically when there is no usable display.

Rendering does not conflict with warp's CUDA graphs. A captured graph is invalidated when the
arrays it holds are **replaced** (reallocated — which is why `expand_model_fields` is followed
by `create_graph()`), not when they are **read**. Rendering only reads.

The real difference between the backends is what taking a frame costs the simulation:

| | `native:cpu` | `warp:cuda` |
|---|---|---|
| Where the state is | Host memory, already an `MjData` | Device memory |
| Taking a frame | Copy qpos/qvel/ctrl/mocap into host arrays | **One** batched device-to-host transfer |
| After DR expanded visual fields | as above | Also fetch the relevant `sim.model` fields |

**A frame is taken on the simulation thread and drawn on another.** Taking it hangs off
`sim.step()`, the point at which the state is consistent and a copy cannot tear. Everything
after that — `mj_forward`, the other environments, `handle.sync()` — runs on the viewer's own
thread, from the copy. It used to run on the simulation thread too, and `handle.sync()` alone
copies the whole model into the viewer (5.9 ms on jumper): a replay of jumper.posture on one
warp environment ran 92 control steps a second headless, 74 with the window at 30 Hz and 55 at
60. All of it releases the GIL, so off the simulation thread it costs the simulation only the
copy; frames are taken at most 30 times a second when training and 60 when replaying.

### 7.1 Why there is a hard cap on how many environments are drawn

Without a cap there are two failure modes, **neither of which reports an error**:

- **Geom capacity.** `launch_passive` gives `user_scn` room for `MAX_GEOM = 100000`. Once full,
  `mjv_addGeoms` prints **one** warning (MuJoCo prints each message once, and it scrolls away in
  the training log) and then **silently drops** the rest. Measured with warp at 4096
  environments: 253890 geoms expected, 100000 arrived — 1613 environments actually drawn (39%),
  while the picture still shows "a field of robots" and nothing looks wrong.
- **Time budget.** Drawing grows with the number drawn. Measured on native, while drawing still
  hung off `sim.step()` and came **straight out of training**: 46.6 ms per frame at 256
  environments (140% of the 30 fps budget), 185.6 ms at 1024 (557%). On the viewer's own thread
  the same numbers are a slideshow rather than a slower simulation, and the copy each frame
  starts from — taken on the simulation thread — still grows with them.

The cap is `min(num_envs, 128)` by default -- `--viewer-env-num` on `train` and `play` moves the 128 -- and the actual number is printed. An adaptive cap based
on measured frame time was tried and removed: the number it produced drifted with machine,
backend and contact density, so the same command drew a different number of environments on
different runs, and it needed hysteresis and margin to stop it grinding 1613 → 403 → 379 → 353.

Geom capacity is still computed as a **backstop**, because geoms per environment is a property
of the model and a sufficiently complex robot could fill 100000 with 128 environments. Hitting
that limit is reported rather than silently truncated.

Cost of 128 environments in a contact-dense state:

| | Per frame | Share of the 30 fps budget |
|---|---|---|
| native | 1.71 ms | 5.1% |
| warp | 5.49 ms | 16.5% |

Warp is 3× more expensive because it does not strip visual meshes and because state has to come
back from device memory. The latter is done as **one batched transfer** rather than two `.cpu()`
calls per environment; for 128 environments that is 254 small transfers dominated by launch
overhead, and batching took it from 7.70 to 5.49 ms per frame with bit-identical geom positions.

### 7.2 Drawing appearance rather than collision geometry

By default `MjvOption.geomgroup` is `[1,1,1,0,0,0]`, and a model that carries both collision
hulls and appearance meshes in visible groups draws **both**, with the hulls wrapped around the
outside of the meshes.

Which groups to turn off is derived per group, from the model rather than from hardcoded group
numbers: a group is turned off when **every geom in it is a collision geom** (contype or
conaffinity non-zero) **and every body owning those geoms also has a non-colliding geom** — an
appearance stand-in.

The second condition is not optional. mjlab's terrain is its own body (not worldbody), its plane
is a collision geom, it has group 0 to itself and it has no appearance stand-in — without that
condition **the floor disappears** and the robots hang in the void. The same protects a link
that has collision geometry but no appearance mesh.

Only viewer-local `MjvOption`s are touched, never the model: on the warp backend `mj_model` *is*
the physics model and camera sensors read it too, so changing `geom_group` there would silently
change observations. Two options need setting — `self._vopt` governs the environments appended
to `user_scn`, while the followed environment is rendered by the viewer itself through
`handle.opt`.

Appearance and collision geometry in the *same* group cannot be separated this way, and then
nothing is turned off: drawing an extra layer is better than hiding the appearance.

### 7.3 Insets: a sensor shows itself, and the robot's camera shows the world

`play` and `play --app` draw the jumper's dToF zone image in the window's top-right corner,
in every task and every scene. The viewer does not know it is a dToF: any sensor with a
`preview(env_index)` method returning an RGB image is found in the simulation's sensor
context and drawn the same way, and what the colours mean is the sensor's own business
(`tasks/jumper/common/tof.py`).

Found rather than registered because the viewer is built in `scripts/_cli.py`, which has no
task context -- the same reason keys go through a registry (`mjrl/viewer/keys.py`) rather than
a constructor argument. Nothing in `scripts/` changed for it.

The picture is taken with the frame on the simulation thread, so it shows the same step as the
robot beside it, and placed on the frame thread with `handle.set_images`, scaled by a whole
number so each zone stays a square block. A preview that raises is dropped with one message;
it must not reach the simulation, and must not fail again on every frame.
`tests/test_viewer_insets.py` pins the followed environment, the placement and that.

The top-left corner holds the robot's own camera -- the one the model names `onboard` --
whenever the window looks through anything else, and nothing while the window looks through
that camera (`]` / `[`), where it would repeat the window. It is the dToF inset's height, so
the corners line up, and the camera's own 4:3, so it shows the camera's field of view rather
than a crop of it. It is not a sensor and no policy sees it: it is rendered on the frame
thread from the window's own `MjData`, in an off-screen GL context of its own, so it costs
the simulation nothing -- `mjr_render` releases the GIL (a Python loop kept 102% of its rate
beside back-to-back renders, Mesa llvmpipe, 2026-09-29). That context has to be made off the
main thread, which EGL, OSMesa and CGL allow and GLFW does not, so under GLFW (Windows) the
inset is left out with one message. The test compares it with `mujoco.Renderer` through the
same camera -- identical to the bit on Mesa llvmpipe -- because upside down, mirrored or of
another environment it would look like any other view from a robot.

That test runs without a window, and so missed the one thing a real window changes. The
window's context is GLX and the inset's EGL, and libglvnd, by default, patches the GL entry
points for the first vendor made current; while that GLX context is current on MuJoCo's render
thread, no EGL context can be made current on any other -- `eglMakeCurrent` fails and
`eglGetError` reports success. On NVIDIA (RTX 5090 D, driver 580, X11) every first render
failed that way until `__GLVND_DISALLOW_PATCHING=1`, which libglvnd reads when a context is
first made current. So the viewer sets it before the window opens whenever it has an onboard
inset, unless it is already set.

## 8. Backend and device resolution

Precedence: **command line > environment (the shell, then `.env.local`, then `.env`) > built-in
default**. A task config has no say: backend and device are never read from one, and
`scene.num_envs` is overwritten with the resolved value.

```
start
 ├─ backend/device given explicitly? ── yes ──▶ obey; error out if unavailable
 └─ no
     └─ CUDA visible and mujoco_warp importable?
         ├─ yes ──▶ warp @ cuda:0, num_envs from the GPU tier
         └─ no  ──▶ native @ cpu, num_envs from the CPU tier, threads = min(cores, 8, num_envs)
```

**No silent fallback.** `--backend warp` on a machine without CUDA must fail, not quietly switch
to CPU and let someone believe they trained on a GPU.

```
--backend     MJRL_BACKEND       auto | warp | native
--device      MJRL_DEVICE        auto | cuda:0 | cpu
--num_envs    MJRL_NUM_ENVS      train: unset = 4096 on warp, 64 on native (resolve()'s tiers)
              MJRL_PLAY_NUM_ENVS play: unset = 1. A separate key, because a replay's
                                 count is not a batch size; see build_parser in _cli.py
--cpu_threads MJRL_CPU_THREADS   0 = available cores, at most 8, then by num_envs
```

The thread-count cap belongs in `resolve()`, not in the backend. `rollout` requires the scratch
list length to equal `nthread`, so the thread count is necessarily capped by the environment
count; capping it only inside the backend would leave the startup banner printing a number that
never took effect. **A configuration that lies is worse than no configuration** — it makes people
believe they have tuned something.

## 9. Known asymmetries

### 9.1 Domain randomisation

MuJoCo Warp batches model fields per world natively: friction, mass and motor strength can
differ per environment inside a single batched model. Native MuJoCo has no such concept, so
per-environment model parameters mean **N `MjModel` copies**.

The asymmetry is hidden below the seam by the `expand_model_fields` / `expanded_fields` pair: on
warp it is nearly a no-op, on native it **lazily** materialises N copies — only fields that
domain randomisation actually writes trigger expansion, the rest keep sharing one copy. Memory
figures and the mesh-stripping payoff are in §5.6.

`DR_NO_MODEL_COPY` / `DR_NEEDS_MODEL_COPIES` in `native_sim.py` are the configuration-side
checklist: events not in the latter work out of the box, events in it must go through
`expand_model_fields` first.

### 9.2 CPU and GPU are not seamlessly equivalent

Going from 4096 environments to 64 changes PPO's **effective batch size by two orders of
magnitude**, and one set of hyper-parameters will not produce one learning curve. The CPU tier
needs its own `num_steps_per_env` / `learning_rate` / `mini_batch` recipe.

The framework's job is to **warn clearly when the backend changes**, not to pretend the two are
interchangeable. Automatic backend selection solves portability, not reproducibility of a
specific policy across machines.

### 9.3 A single-step residual of 4–6e-4, not yet settled

Comparing native against warp per environment from the same `qpos` with `qvel=0` and `ctrl=0`,
each doing `forward()` and then advancing **one** physics step:

| Quantity | Difference | Reference |
|---|---|---|
| `qpos` after `forward()` | 0 | bit-identical |
| `xpos` after `forward()` (forward kinematics) | **1.2e-07** | magnitude 0.63 — float32 machine epsilon |
| Base pose after one step | **4–6e-04** | paired per environment with base x at 0/10/20/30 |

**Forward kinematics agreeing to 1e-07 says the model bridge is correct** — the 103
world-batched fields and the 192 topology fields all land in the right places. The residual
appears only in the integration step.

Provisionally acceptable because mjwarp is float32 internally while native MuJoCo is float64 and
the solver implementations differ; in the state measured, `qacc` reaches 1262 (deep contact),
against which 4e-4 is 3e-7 relative.

**None of the following has been verified, so do not treat the number as a conclusion:**

1. Whether the residual is **bounded**. Only one step was measured — whether it stays at 1e-4 or
   accumulates is unknown.
2. Whether it varies with **contact depth**. It was measured in one state, and that state had a
   very large `qacc`.
3. Whether it affects **training**, which is the only question that matters. Settling it means
   training both backends on the same task, hyper-parameters and seed and comparing learning
   curves. Point-wise trajectory agreement **is not achievable** and should not be the criterion:
   a hexapod's contacts are dense enough that any small difference amplifies exponentially, and
   after 20 control steps the two diverge by 2.4 against state magnitudes of 0.75.

**Until that comparison exists, a policy produced by `native:cpu` should not be assumed
equivalent to one from `warp`.**

### 9.4 RGB with textures

See §6.3. Depth and segmentation are aligned closely enough to be used as references for one
another; RGB with textures enabled is not.

### 9.5 The Apple GPU is a third device, not a second CUDA

Through [warp-metal](https://github.com/DavidDobas/warp-metal) -- a community overlay on one
exact Warp release, with a four-file patch to mujoco_warp -- the warp backend simulates on
Apple Silicon's GPU (`AGENT_SETUP.md` §1.6). It is the same mjwarp code and the same seam, and
it is not the same device:

- **torch has no Metal device.** The environment's tensors are CPU tensors aliasing the Warp
  arrays in unified memory, which is why the resolution carries a `sim_device` beside `device`
  (`metal:0` beside `cpu`) and why `Simulation` waits for the GPU after every launch
  (`utils/sim_device.synchronize`): a CPU read before the kernels finish sees the previous step, with
  no error anywhere. On CUDA the torch stream ordering does that job.
- **No conditional graph nodes.** mjwarp's solver loop ends on a condition evaluated inside the
  CUDA graph; Metal has no such node, so the solver runs its fixed iteration count
  (`graph_conditional = False`). Same answer when converged, more work when converged early.
- **No float64, and 64-bit atomics are not atomic.** mjwarp is float32 throughout, so the first
  does not bite; the second is the overlay's own caveat and nothing here has measured it.
- **Threadgroup memory, not FLOPs, is the limit.** The patch keeps one Cholesky tile per SIMD
  group and drops the blocked factorisation up to 64 DoF, and the overlay's author notes the
  gap to CUDA widens with the environment count. The measurement agrees:

| `--num_envs` | policy on | env-steps / s | iteration | vs native (1850) | vs RTX 5090 |
|---|---|---|---|---|---|
| 256 | cpu | 1948 | 3.2 s | 1.05x | |
| 1024 | mps | 3469 | 7.1 s | 1.9x | |
| 4096 | mps | 5243 | 18.8 s | 2.8x | 48x slower (§10: 0.39 s) |

M3 Max, macOS 26.5, `jumper.tripod`, 3 iterations, 2026-10-08; one sample each, and the two 4096
runs differed in collection time by a factor of two. The policy learning on MPS is what makes
the 4096 row: with it on the CPU, learning took 6.7 s of a 40.8 s iteration and collection the
rest, which says the CPU side was contended, not that the GPU was.

**The physics agrees with mjwarp's own CPU reference.** §9.3's measurement, repeated with three
engines from one state -- eight worlds on a plane, dropped 5 cm with a seeded random `ctrl`, so
the first 100 steps are free fall and the last 100 are landing (6 contacts, `qacc` 29 at step
200). Maximum base-pose difference, M3 Max, 2026-10-08:

| after | metal vs native | warp:cpu vs native | metal vs warp:cpu |
|---|---|---|---|
| `forward()`, `xpos` | 8.2e-07 | 7.8e-07 | 4.8e-07 |
| 1 step (free fall) | 5.8e-09 | 5.8e-09 | **9.8e-13** |
| 50 steps (free fall) | 1.4e-08 | 1.4e-08 | 1.2e-09 |
| 200 steps (in contact) | 4.3e-04 | 4.3e-04 | **8.8e-06** |

The third column is the control: the same kernels on two devices. In free fall they agree to
float32 round-off; after landing they are 8.8e-6 apart, two orders of magnitude inside the
4.3e-4 that separates either of them from native -- which is §9.3's residual (4–6e-4, float64
against float32 in contact), unchanged by the device. So the Metal device computes what mjwarp
computes, and whatever a policy trained here inherits from the simulator, it inherits from
mjwarp rather than from Metal. The first attempt at this probe measured nothing: the bare
`jumper.xml` has no floor (the scene adds it), the robot fell through, and `ncon` stayed 0 --
the residuals were free fall all the way down. The plane, `ncon` and `qacc` are in the table
because of that.

**And the learning curves agree.** The same-seed comparison §9.3 asks for, between native and
Metal: `jumper.tripod`, 64 environments, seed 42, 300 iterations each, the shipped
hyper-parameters (which are the GPU recipe, so neither run is a good policy -- the point is that
they are the same not-yet-good policy). M3 Max, 2026-10-08, native / Metal:

| iteration | 50 | 100 | 150 | 200 | 299 (last 10 mean) |
|---|---|---|---|---|---|
| `Train/mean_reward` | -35.1 / -32.8 | -44.1 / -43.0 | -29.5 / -30.2 | -18.7 / -21.5 | -10.2 / -11.1 |
| `Train/mean_episode_length` | 628 / 599 | 982 / 995 | 1000 / 1000 | 1000 / 1000 | 1000 / 1000 |
| `Episode_Reward/track_linear_velocity` | 0.23 / 0.32 | 0.33 / 0.28 | 0.26 / 0.22 | 0.27 / 0.42 | 0.27 / 0.29 |
| `Policy/mean_std` | 0.932 / 0.931 | 0.873 / 0.874 | 0.824 / 0.828 | 0.787 / 0.791 | 0.746 / 0.751 |

Point-wise agreement was never the criterion (§9.3: a hexapod's contacts amplify any
difference exponentially, and after 200 steps the two simulators are 4e-4 apart), and the
per-iteration numbers do scatter -- the tracking term by up to 0.15 at one mark. The curves do
not: the reward dips and recovers at the same iterations, both runs reach full-length episodes
by iteration 150 with no falls, and the policy's noise decays along one line to three decimals.
Whatever separates a native policy from a Metal one at 300 iterations is inside what separates
two iterations of the same run. The run's own `Perf/` numbers are not quoted: the Metal run
shared the machine with the residual probes above, which compile kernels for minutes, and its
throughput reads 190 env-steps/s against the 1200 measured alone (§1.6 of `AGENT_SETUP.md`).

**What has not been verified**: Metal against CUDA directly -- no CUDA machine was at hand --
and anything past 300 iterations at 64 environments, which is where the GPU recipe would
start to matter. A Metal checkpoint is still reported as a Metal checkpoint, which is what the
banner's `sim=` and the resolution's note are for.

## 10. Collision geometry

`jumper.xml` carries two geom sets side by side, `<body>_visual` for display and
`<body>_meshcol` for collision, on **one model with one set of inertias and one set of joint
limits**. There is one collision scheme, hybrid: foot meshes at source resolution, every other
link a decimated convex hull (§10.1). `CollisionCfg(disable_other_geoms=True)` zeroes
contype/conaffinity on whatever it is not given at the spec stage; nothing is deleted, so
`build_jumper.py` stays reproducible.

The table below is the measurement that settled the choice, made on the previous model when
the XML still carried all three schemes; the all-mesh and all-primitive ones have since been
deleted. Measured on a real task (flat-ground velocity tracking, 4096 envs × 300 iterations,
RTX 5090):

| Scheme | Iteration time | Relative | Mean reward at 300 | Foot fidelity |
|---|---|---|---|---|
| all meshes | 0.64 s | 1.00× | 52.61 | matches hardware |
| **hybrid** | **0.39 s** | **1.65×** | 55.47 | **matches hardware** |
| all primitives | 0.35 s | 1.78× | 55.38 | sphere approximation |

**Hybrid captures 94% of the available speedup while keeping the foot contact geometry exact.**
The extra 8% from primitives is not worth trading fidelity for. The reward curves are close and
there is no evidence that collision geometry affects learning quality.

> A performance comparison of collision geometry must be made with the robot standing under
> control with its feet on the ground. Started from `mj_resetData` with `ctrl=0` the robot is in
> free fall, and after 400 steps the hybrid and primitive variants are still airborne with zero
> contacts while the mesh variant has landed — at which point the three are not being compared
> under the same conditions at all, and what is measured is broad-phase overhead during free
> fall.

### 10.1 Links other than the feet

Links adjoining the feet and grippers take part in collision as well, so that ground contact by
a link is detectable at all — otherwise such a link passes through the floor and no criterion
for penalising it can even be expressed. (`jumper.jump` and `jumper.ref_free_jump` penalise it,
as `undesired_contacts`; the locomotion tasks do not.)

On V1.5, where this was settled, including 25 more links took the stripped model from 2.6 MiB
to 15.3 MiB, i.e. 61 GiB at 4096 environments, which does not fit. The answer is **decimated
convex hulls**:

| | Vertices | Per copy | 4096 envs |
|---|---|---|---|
| Original meshes | 111036 | 15.32 MiB | 61 GiB |
| Decimated hulls | **1527 (0.2%)** | **3.04 MiB** | **12.2 GiB** |

Capsules were rejected on measurement: a fitted capsule bulges outward, and in the standing pose
the capsules around the gripper segments sit 1.6–9.8 mm **below** the foot meshes — the robot
would stand on its palms rather than its jaws, quietly destroying the "feet make mesh contact"
design.

A convex hull contains every extreme point of the original mesh, so its lowest point is
unchanged and contact timing is preserved. Decimation (support-function sampling along
Fibonacci-distributed directions) only moves the surface **inward**, which is the safe
direction: contact triggers slightly later rather than earlier. `tools/hull_collision.py` reports
the maximum inward shrink, which is exactly how much later.

**The foot contact meshes themselves are not decimated** — they are what actually touches the
ground, and they are nearly convex already (738→738, 854→843 vertices), so there is nothing to
save.

Result: self-collisions 35 → 0, jaw clearance 3.13 → 32.10 mm, contacts only ever on the feet
(penetration 0.006–0.3 mm).

## 11. A coupling worth knowing about: `action_scale` and `action_rate_l2`

mjlab's `action_rate_l2` penalises the rate of change of the **raw policy output**, not of joint
angles ("Operates on raw policy output (before per-term scale/offset)").

Producing the same joint motion Δq therefore requires a raw action change of `Δa = Δq / scale`,
and the penalty goes as `Δa²`, i.e. `1/scale²`:

| scale | Penalty for the same physical motion |
|---|---|
| 0.25 (reference) | ×1 |
| 0.20 | ×1.6 |
| 0.05 | ×25 |

Controlled experiment (same scale=0.20, same 500 iterations, only the compensation coefficient
changed):

| | Uncompensated (weight −0.1) | Compensated (weight −0.064) |
|---|---|---|
| vx / vy / wz tracking | 1% / 2% / 13% | **94% / 91% / 93%** |
| Base displacement over 8 s | 0.015 m | **1.78 m** |

A factor of 120 in displacement. `velocity_env_cfg`'s `compensate_action_rate` (on by default)
scales the task's `action_rate_weight` by `(new/0.25)²` whenever `action_scale` is overridden:
the weight a task passes is its value at scale 0.25.

`action_scale = 0.25` is the chosen value. After compensation, 0.20 and 0.25 track equally well;
the only substantive difference is torque headroom — 0.25 spends 5.5% of the time near the
limit against 12.7% for 0.20. On hardware, derating from temperature rise lowers the usable
torque, so less time at the limit is safer.

**The general lesson**: any penalty acting in a **normalised** space has to be re-checked when
the scaling coefficient changes. The information was in the docstring all along, but it takes
working out `Δa = Δq/scale` to notice the `1/scale²` amplification.

## 12. Measuring this system

Several of the criteria in this document were wrong on their first attempt, in ways that would
have produced confident and false conclusions. The recurring failure is **a criterion with no
control group**, and the practices below are worth applying to anything new:

- **Compare against a control that isolates the variable.** "Stripping visual meshes changes the
  physics" survived one round of measurement because two separately built environments were
  compared — and two environments built *the same* way differed just as much, because startup
  randomisation resamples. Similarly, "different thread counts give different results" measured
  1e-3 until the control of rebuilding at the *same* thread count also gave 1.3e-3; swapping the
  pool inside one environment gives exactly 0.
- **Choose a statistic that is not dominated by the extreme.** Depth agreement judged by maximum
  absolute difference is dominated by grazing-angle pixels, where ∂z/∂pixel is enormous and both
  implementations are correct: 4.5 m maximum against a 5.7e-05 median.
- **Judge where disagreements are, not how many.** A segmentation agreement *rate* depends on
  how many silhouettes are in view, so it measures scene complexity. Rasterisation assigns by
  pixel-centre coverage and ray casting by the ray through the centre, so they can only differ at
  boundaries; a disagreement in an object's interior means something else is wrong.
- **A negative control has to move most of the frame.** RGB correlation first used "move one red
  sphere" as the negative control, giving 0.852 true against 0.807 control — meaningless,
  because the sphere is 5% of the pixels. Rendering a different camera pose drops the control to
  0.199.
- **Check that the parts sum to the whole before believing a breakdown.** A `sim.step` breakdown
  was 50% off because the replica drifted into a different state (an uncontrolled robot falls
  over, and contacts multiply), and then off again because a `restore()` sat inside the timed
  region but in none of the segments, leaving the parts summing to 18%.
- **Ask who else was on the machine.** A timing breakdown taken while another training run held
  about ten cores made the serial Python manager layer look like 62% of `env.step()`; on an idle
  machine it is 13.9% and the bottleneck is still in `sim`.

## 13. Where the time goes

After optimisation, `sim.step` decomposes as (self-consistent within one run; the parts sum to
99.6%):

| Segment | 64 envs | 256 envs |
|---|---|---|
| `_refresh_derived` (parallel `mj_forward`) | **67.2%** | **68.8%** |
| `pool.rollout` | 24.0% | 21.1% |
| Everything else (getState / scatter / allocation) | 8.7% | 10.1% |
| `_scatter_model` | 0.1% | 0.0% |

Both remaining blocks are real physics. The expensive part of `mj_forward` is the constraint
solver, and only `qacc` needs it — but `qacc` **is** read (`entity/data.py`'s `joint_acc`,
`utils/nan_guard.py`), so skipping the solve would silently corrupt it.

Three optimisations that got there, and one that did not:

1. **`_scatter_model` was copying the same values every step** (8.7–10.4% of `sim.step`).
   Domain randomisation only writes those buffers during startup / reset / interval events —
   measured over 100 control steps of ordinary stepping, zero accesses. A dirty flag took it from
   0.342 ms to 0.002 ms. The flag is set in two places, erring towards doing the work
   unnecessarily rather than missing it: `_NativeModel.__getattr__` (getting the tensor is the
   only entry point for reading *or* writing, so getting it marks dirty) and
   `_ExpandedView.__setitem__` (which catches holding a tensor and writing to it several steps
   later, when `__getattr__` is no longer called). In-place operators like `t.copy_()` still slip
   through, which is why `tests/test_native_dr.py` checks the end-to-end result — whether the
   randomised value can be read back off each environment's `MjModel` — rather than the flag.
2. **`_gather` / `_scatter` looked up buffers three times per field per environment** in the
   inner loop (`numpy()` called 294400 times per 20 steps under cProfile). Resolving buffer
   references and target shapes into a plan at construction time leaves one `getattr` and one
   assignment: 0.546 → 0.331 ms.
3. **`raycast_into` allocated per ray.** `mj_ray` wants contiguous float64 while the ray buffers
   are float32, so every ray did two `np.ascontiguousarray` calls — 3840 small allocations for
   64 environments × 30 rays, 1.125 ms, 38% of the loop. Converting once takes it from 2.972 to
   1.696 ms with bit-identical results. `sim.sense` went from 5.37 to 3.39 ms.
4. **Folding `_gather` into the parallel `_refresh_derived` was 17–19% slower** and was reverted.
   The prediction was that C code releasing the GIL would overlap with Python that does not; in
   practice a worker doing Python gathering holds the GIL, and 32 threads contending for it costs
   more than the overlap saves. Two clean passes — one nearly pure C, one serial — win.

Overall: 64 environments 28.51 → 23.46 ms (2245 → 2728 env-steps/s), 256 environments
83.45 → 68.48 ms (3068 → 3738 env-steps/s).

## 14. Where a parameter lives, and why the tasks are each other's controls

Three homes, and the boundary between them is the difference between a comparison
that means something and one that does not.

    tasks/jumper/common/constants.py    facts about the hardware
    tasks/jumper/common/                mechanism, and no tuning
    tasks/<task>/env_cfg.py           every parameter

**`constants.py` is for what the robot is.** One chassis, one IMU, one set of
current sensors, one servo curve — so one value, and every task in the family reads it.
`EFFORT_LIMIT` is the plateau of a measured torque-speed curve; `STAND_Z` is how
tall the robot stands; `IMU_ANG_VEL_NOISE` is what a gyro bolted to a walking
hexapod reads. None of these is a choice a gait gets to make, and duplicating them
would let one task simulate a different robot from the others.

**`common/` is for mechanism.** The reward functions, the curriculum classes, the
`ladder()` that turns a ceiling into rungs. What it must not hold is a tuning
number, because a number here is one every task in the family inherits without
choosing it. That is not hypothetical. `jumper.tetrapod` declared no command ladder
at all and ran on `curriculum.LEVELS`, which had been written for a different
gait; nothing was wrong with the values, and nothing anywhere recorded that
tetrapod had never agreed to them. It read as consensus.

So `velocity_env_cfg` **raises rather than defaulting** for the parameters a task
owns — the foot-lift target, the speed ceiling, the std ratio, the two weights that
form the stand-still barrier. A missing argument is a loud failure at build time
instead of a quiet inheritance that nobody revisits. Seven files writing these
arguments is the intended cost — the four locomotion tasks; `jumper.swing`, which
builds on the same skeleton and then subtracts the command from it; and
`jumper.posture` and `jumper.five_foot`, which build on it and choose some of the
values differently. Agreement that is written down survives one of them changing,
and agreement that is a default does not.

### The parity rule

The four locomotion tasks exist to be compared. Same robot, same reward, same
observations, same curriculum, **one variable** — the gait. Two exemptions, and
they are the only two:

- **the gait itself**: its reward term, its phase clock, the cadence every gated
  term is handed, and the load thresholds that depend on how many legs are down at
  once (a fair share over three supporting legs is not a fair share over five);
- **the speed ceiling**, because a fixed-frequency clock caps speed at stroke times
  cadence, and everything the skeleton derives from that ceiling — the command
  ranges, the ladder, the tracking stds.

`tests/test_task_parity.py` enforces this and names the offending field when it
fails. It also checks something the parity test alone cannot see: that the ladder
is **one shape scaled to each ceiling**. Before that check existed, `jumper.ripple`'s
rungs were (0.625, 0.875, 1.0) of its ceiling while `jumper.tripod`'s were
(0.5, 0.7, 1.0), because one had been built by clamping the shared absolute rungs
and the other by scaling. Both looked reasonable in isolation, and `level 1` meant
two different fractions of what each robot could do — so the level number, which is
the thing you read off a dashboard to compare two runs, was not comparable at all.

That test was written the obvious way first, recomputing each ladder from
`LEVEL_FRACTIONS`, and it passed against a deliberately broken shape: the ladders
are derived from that constant, so the assertion was comparing the derivation with
itself. It now compares the tasks against **each other**, which is the property
that matters and the one a shared constant cannot fake.
