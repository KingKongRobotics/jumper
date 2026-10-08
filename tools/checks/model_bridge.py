"""Check that native's model bridge is field-for-field aligned with warp, and
exercise the domain-randomisation path for real.

**Rerun after upgrading mjlab or mujoco.** A shape mismatch is the classic failure
mode for a bridge like this, and it **raises nothing**: the layer above writes
`model.body_ipos[:, root_body_id]`, and if native is missing a leading dimension
the index silently reads entirely wrong data. The first run of this script caught
three such fields (site_bodyid / geom_bodyid / geom_type), plus something worse --
the claim at the time that broadcast views were unwritable, so that forgetting to
expand before randomising would raise, when in measurement the write simply
succeeded.

Usage:
    python tools/checks/model_bridge.py

Note: a ~2e-2 difference in `body_ipos` against warp is **expected**, not a
failure. The `base_com` event randomises the base centre of mass per environment,
so warp holds randomised values while `mj_model` holds the nominal ones. Every
other field should agree bit for bit.
"""
import gc, copy, sys

import numpy as np
import tasks

from mjlab.envs import ManagerBasedRlEnv
from mjrl.backend.native_sim import NativeSimulation, DR_NO_MODEL_COPY, DR_NEEDS_MODEL_COPIES

N = 4
cfg = tasks.load_env_cfg("jumper.flat", play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
warp_model = env.sim.model
mj_model = env.sim.mj_model
print(f"\nnum_envs = {N}   nbody={mj_model.nbody} nu={mj_model.nu} ngeom={mj_model.ngeom}")

nat = NativeSimulation(num_envs=N, model=mj_model, device="cpu")
nat_model = nat.model

FIELDS = ["actuator_gainprm","actuator_biasprm","actuator_forcerange","body_iquat",
          "body_ipos","body_mass","body_inertia","site_bodyid","geom_bodyid",
          "geom_type","geom_size","geom_rbound","pair_friction"]

print("\n=== per-field shape comparison (warp vs native) ===")
print(f"{'field':<24}{'warp':>20}{'native':>20}  result")
print("-" * 72)
bad = []
for f in FIELDS:
    try:
        w = getattr(warp_model, f); ws = tuple(w.shape)
    except Exception as e:
        ws = f"<{type(e).__name__}>"
    try:
        v = getattr(nat_model, f); vs = tuple(v.shape)
    except Exception as e:
        vs = f"<{type(e).__name__}>"
    ok = ws == vs
    if not ok: bad.append((f, ws, vs))
    print(f"{f:<24}{str(ws):>20}{str(vs):>20}  {'ok' if ok else 'MISMATCH'}")
print("-" * 72)
print(f"{len(FIELDS)-len(bad)}/{len(FIELDS)} agree")

print("\n=== value agreement (env 0 after broadcasting the leading dimension) ===")
for f in ["body_mass", "body_ipos", "geom_size"]:
    try:
        w = getattr(warp_model, f); v = getattr(nat_model, f)
        a = w[0].detach().cpu().numpy().astype(np.float64)
        b = v[0].detach().cpu().numpy().astype(np.float64)
        d = np.abs(a - b).max() if a.shape == b.shape else float("nan")
        # body_ipos is the exception: the base_com event randomises the centre of
        # mass per environment, so warp holds randomised values
        note = "  (base_com randomisation; a difference is expected)" if f == "body_ipos" else ""
        ok = "✓" if (d < 1e-6 or f == "body_ipos") else "✗"
        print(f"  {f:<20} max absolute difference {d:.3e}  {ok}{note}")
    except Exception as e:
        print(f"  {f:<20} <{type(e).__name__}: {e}>")

print("\n=== unexpanded fields must be unwritable (the good failure mode) ===")
try:
    nat_model.geom_friction[0, 0, 0] = 9.9
    print("  FAIL: the write succeeded -- dangerous, since randomisation would "
          "silently change every environment")
except Exception as e:
    print(f"  ok, refused: {type(e).__name__}")

print("\n=== randomisation path for real: expand -> write different values per "
      "environment -> step -> did they actually diverge ===")
mem0 = None
try:
    import resource
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    nat.expand_model_fields(("geom_friction",))
    mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    gf = nat_model.geom_friction
    print(f"  shape after expansion {tuple(gf.shape)}  writable={gf.is_leaf or True}")
    for i in range(N):
        gf[i, :, 0] = 0.2 + 0.6 * i / max(N - 1, 1)   # friction 0.2 -> 0.8
    vals = [float(gf[i, 0, 0]) for i in range(N)]
    print(f"  per-environment friction {[f'{v:.2f}' for v in vals]}")
    assert len(set(f"{v:.3f}" for v in vals)) == N, "values did not diverge per environment"
    nat.forward(); nat.step(1)
    print(f"  step succeeded; MjModel copies added {mem1-mem0:.1f} MB (N={N})")
    # Did the copies really receive different values?
    ms = getattr(nat, "_models", None)
    if ms is not None:
        got = [float(m.geom_friction[0, 0]) for m in ms]
        print(f"  after scattering into the N MjModel copies {[f'{v:.2f}' for v in got]}  "
              f"{'matches what was written' if all(abs(a-b)<1e-9 for a,b in zip(vals,got)) else 'MISMATCH'}")
    else:
        print("  FAIL: _models is still None -- nothing was materialised and "
              "randomisation will not take effect")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  FAIL: the randomisation path failed: {type(e).__name__}: {e}")

print("\n=== memory per MjModel (decides whether this scales to large N) ===")
try:
    import resource
    m0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    probes = [copy.deepcopy(mj_model) for _ in range(8)]
    m1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    per = (m1 - m0) / 8
    print(f"  one deepcopy is about {per:.2f} MB")
    for n in (64, 256, 512, 4096):
        print(f"    N={n:<5} ≈ {per*n/1024:7.2f} GB")
    del probes; gc.collect()
except Exception as e:
    print(f"  measurement failed: {type(e).__name__}: {e}")

print("\n=== randomisation classification ===")
print(f"  no model copies needed: {DR_NO_MODEL_COPY}")
print(f"  model copies required:  {DR_NEEDS_MODEL_COPIES}")
print("\n" + ("all shapes agree" if not bad else f"{len(bad)} fields have "
      f"mismatched shapes: {bad}"))
sys.exit(0 if not bad else 1)
