"""Measure which fields mjwarp actually gives a world dimension -- measurement
instead of guesswork.

Usage: python tools/checks/warp_dims.py
"""
import numpy as np
import tasks

from mjlab.envs import ManagerBasedRlEnv

N = 4
cfg = tasks.load_env_cfg("jumper.flat", play=True)
cfg.scene.num_envs = N
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
wm, mj = env.sim.model, env.sim.mj_model

batched, plain, missing = [], [], []
names = [a for a in dir(wm) if not a.startswith("_")]
for a in sorted(names):
    try:
        w = getattr(wm, a)
    except Exception:
        continue
    if not hasattr(w, "shape"):
        continue
    ws = tuple(w.shape)
    if not hasattr(mj, a):
        missing.append((a, ws)); continue
    raw = getattr(mj, a)
    if not isinstance(raw, np.ndarray):
        continue
    rs = tuple(raw.shape)
    if ws == (N,) + rs:
        batched.append(a)
    elif ws == rs:
        plain.append(a)
    else:
        missing.append((a, f"warp{ws} vs mj{rs}"))

print(f"\n=== fields with a world dimension ({len(batched)}) ===")
print("  " + " ".join(batched))
print(f"\n=== fields without one ({len(plain)}) ===")
print("  " + " ".join(plain))
print(f"\n=== shape mismatch / warp only ({len(missing)}) ===")
for a, s in missing[:40]:
    print(f"  {a:<28}{s}")

# dtype distribution among the plain fields -- tests the "integers are topology" guess
import collections
dt = collections.Counter(str(getattr(mj, a).dtype) for a in plain if hasattr(mj, a))
print(f"\ndtypes without a world dimension: {dict(dt)}")
dtb = collections.Counter(str(getattr(mj, a).dtype) for a in batched if hasattr(mj, a))
print(f"dtypes with a world dimension:    {dict(dtb)}")

print("\n=== tracing the body_ipos value difference ===")
a = wm.body_ipos[0].cpu().numpy().astype(np.float64)
b = mj.body_ipos.astype(np.float64)
d = np.abs(a - b).max(axis=1)
bad = np.argsort(-d)[:5]
for i in bad:
    if d[i] < 1e-9: break
    print(f"  body[{i}] {mj.body(int(i)).name:<22} warp={a[i]}  mj={b[i]}  diff={d[i]:.4e}")
print(f"  (bodies where env.sim.mj_model and the warp model disagree: {(d>1e-9).sum()}/{len(d)})")
