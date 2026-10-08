# Vendored dependencies

Two upstream packages live under `rl/` as **modifiable copies** (forks) rather than pip
dependencies.

| Directory | Upstream | Version | Why fork it |
|---|---|---|---|
| `rl/mjlab/` | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 1.6.0 | We cut a seam between the manager layer and the physics engine — an invasive change |
| `rl/rsl_rl/` | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | 5.4.2 | Planned algorithm work (distillation / AMP / custom PPO variants) |

The version pairing is not arbitrary: **5.4.2 is exactly the `rsl-rl-lib` version mjlab
1.6.0 pins.** Check mjlab's constraint before changing either.

## Provenance

Both are taken unmodified from the official PyPI release artifacts:

- **mjlab 1.6.0** — unpack the wheel, take the `mjlab/` package directory
- **rsl_rl 5.4.2** — unpack the sdist, take the `rsl_rl/` package directory

Neither package directory carries its licence — that sits at the root of each upstream
repository — so each copy has a `LICENSE` added beside it, taken from the release tag:
`rl/mjlab/LICENSE` from mjlab `v1.6.0` (Apache-2.0) and `rl/rsl_rl/LICENSE` from rsl_rl
`v5.4.2` (BSD-3-Clause). Both licences require the text to travel with the code. What the
two upstreams in turn took from elsewhere — Isaac Lab in `mjlab/utils/lab_api/`, MuJoCo
Menagerie's robots in `mjlab/asset_zoo/`, pfrl in `rsl_rl/modules/normalization.py` — has
its licence in [`licenses/`](../licenses/) at the repository root, and
[`NOTICE`](../NOTICE) says which file covers what.

`rl/mjlab/asset_zoo/` is 34 MB of robot meshes (Unitree Go1 / G1 and the i2rt YAM arm). It is **kept, not pruned**:
a faithful copy makes rebasing onto upstream clean, and mjlab's own Go1 / G1 tasks are a
ready-made reference for checking that the seam did not change behaviour.

## Layout: why under `rl/`

Package locations are declared explicitly by the two package roots in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = [".", "rl"]
include = ["tasks*", "scenes*", "controller*", "mjrl*", "mjlab*", "rsl_rl*"]
```

`rl/` itself has **no** `__init__.py`, so `mjlab` and `rsl_rl` remain top-level packages —
upstream's own `import mjlab.envs` needs no edit.

The point of this layout is to make **which copy you get independent of which directory you
run from**. Put the copies at the repository root and the root is implicitly on `sys.path`:
running from the repo root picks up the copy, running from anywhere else picks up the pip
version, and the two behave differently in ways that are hard to notice. Under `rl/` that
implicit path does not exist.

**The cost is that installation goes from optional to mandatory.** `rl/` is not on
`sys.path`, so without `pip install -e .` an `import mjlab` simply fails — there is no
"happens to work" middle state. That is deliberate: an implicit directory dependency traded
for an explicit install dependency.

The shadowing risk from pip versions remains (a same-named package in `site-packages` still
competes with this copy), so **do not keep pip versions in the environment**:

```bash
pip uninstall -y mjlab rsl-rl-lib
```

Uninstalling them does not take their dependencies with them (mujoco-warp, warp-lang, torch,
tyro, viser and friends stay), and those dependencies are listed explicitly in
`pyproject.toml`, so `pip install -e .` in a fresh environment installs everything.

To check which copy is in use:

```bash
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
```

Both paths must point at `<repo>/rl/mjlab` and `<repo>/rl/rsl_rl`. If either points into
`site-packages`, a pip version is still installed or this repository was not installed as
editable.

## What was changed

All modifications to upstream are confined to **eight files**, each carrying an `[mjrl]`
marker and the reason for the change. `rsl_rl` is untouched.

| File | Change |
|---|---|
| `mjlab/sim/__init__.py` | **New**: `set_simulation_cls` / `get_simulation_cls`, making `Simulation` replaceable. This is the seam itself |
| `mjlab/envs/manager_based_rl_env.py` | Constructing the sim goes from `Simulation(...)` to `get_simulation_cls()(...)`. With nothing registered it returns `Simulation`, byte-for-byte upstream behaviour |
| `mjlab/sensor/sensor_context.py` | Dispatch the sensor context by backend. `RenderContext` is a mjwarp thing and the native path has no `model.struct`, so buffers are allocated on the CPU in the same layout instead |
| `mjlab/sensor/raycast_sensor.py` | Same dispatch. This is the only mjwarp-bound part of the whole `RayCastSensor`; the native path goes through `mujoco.mj_ray`. Its Warp device and ray buffers come from `sim_device()` as well |
| `mjlab/utils/sim_device.py` | **New**: `set_sim_device` / `sim_device` / `synchronize` -- which Warp device simulates for a torch device, registered beside the seam by `use_backend()`. Upstream's rule (the torch device's own name) is what you get with nothing registered; on Apple Silicon the resolution registers `metal:0` for torch's `cpu` |
| `mjlab/sim/sim.py` | Picks its Warp device through `sim_device()`, waits for a Metal device after every launch (CPU tensors read its unified memory directly), runs the solver for its fixed iteration count on Metal (no conditional graph nodes) and lets Metal capture graphs |
| `mjlab/sim/sim_data.py` | `TorchArray` took every non-cpu Warp array for a CUDA one and built a `torch.cuda.Stream` for it; it now asks `is_cuda` |
| `mjlab/rl/config.py` | **New field**: forwards rsl_rl's symmetry augmentation / mirror-loss parameter, which this wrapper dataclass did not expose. Delete it if upstream adds the same field |

To find every change:

```bash
grep -rn "\[mjrl\]" rl/mjlab/ rl/rsl_rl/
```

## Modification discipline

So that rebasing onto upstream stays possible:

1. **Keep changes concentrated.** Edits on the mjlab side belong near the seam
   (`rl/mjlab/sim/` and the files in the table above), not scattered across manager / entity.
2. **Mark every edit** with `[mjrl]`, saying why it was made and against which upstream
   version, e.g. `# [mjrl] reason: … (against mjlab 1.6.0)`.
3. **When upgrading upstream**: overwrite with the new official artifact, then use
   `git diff` to recover your own changes and reapply them. Which makes rule 2 not decoration
   but the only trail you have.
4. **The two `LICENSE` files are not in the artifact**, so an overwrite deletes them. Take
   them again from the new release's tag, and check that [`NOTICE`](../NOTICE) still
   describes what the new copy contains.

`tests/test_seam.py` checks that the set of files carrying the marker is exactly the one in the
table above — a marked change that spreads to a sixth file gets caught. An edit made without the
marker does not, which is why rule 2 is the one that matters.
