#!/usr/bin/env python3
"""Quantify the free score in tracking rewards: how much a robot that never moves
and never turns collects anyway.

**Why this is needed**: a tracking reward of the form `exp(-err^2/std^2)` gives
full marks whenever the command is near zero, so the overall mean gets diluted by
all the zero commands. It then looks high while being completely unable to
distinguish "actually tracked it" from "never moved". Measured here once: with a
soft reward at weight 2.0, training reached 1577 iterations with
track_angular_velocity = 1.3390 -- while the free score for never turning at all
is 1.3512. **The trained policy was indistinguishable from a robot that never
turns**, and the 1.34 in the log looked perfectly healthy.

So rerun this after changing std or the command ranges, and look at two things:
  1. what the free score is (the overall mean)
  2. how large the incentive gap is **when movement is actually required** --
     bucketed by command magnitude, which is the decisive view

Usage: python tools/checks/track_free_score.py
"""

import torch
import tasks

from mjlab.envs import ManagerBasedRlEnv

cfg = tasks.load_env_cfg("jumper.tetrapod", play=False)
cfg.scene.num_envs = 2048
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()

std_lin = cfg.rewards["track_linear_velocity"].params["std"]
std_ang = cfg.rewards["track_angular_velocity"].params["std"]
w_lin = cfg.rewards["track_linear_velocity"].weight
w_ang = cfg.rewards["track_angular_velocity"].weight
print(f"\nstd: linear {std_lin}  angular {std_ang}     "
      f"weights: linear {w_lin}  angular {w_ang}")

# Resample commands repeatedly to cover the real distribution (including any
# proportion of standing commands)
cmds = []
for _ in range(24):
    env.command_manager.reset(torch.arange(env.num_envs, device=env.device))
    cmds.append(env.command_manager.get_command("twist")[:, :3].clone())
c = torch.cat(cmds, 0)
print(f"sampled {c.shape[0]} commands")
for i, n in enumerate(("vx", "vy", "wz")):
    v = c[:, i]
    print(f"  {n}: mean |.| {v.abs().mean():.4f}  range [{v.min():.2f}, {v.max():.2f}]  "
          f"near zero (<0.05) {(v.abs()<0.05).float().mean()*100:.1f}%")

# What a robot that never moves (v=0, w=0) collects
lin_err = (c[:, :2] ** 2).sum(1)          # |cmd_xy - 0|²
ang_err = c[:, 2] ** 2                     # (cmd_wz - 0)²  (+ wx²+wy² = 0)
r_lin = torch.exp(-lin_err / std_lin ** 2)
r_ang = torch.exp(-ang_err / std_ang ** 2)

print(f"\n{'':24}{'normalised':>12}{'x weight':>10}")
print("-" * 44)
print(f"{'linear vel, free':<24}{r_lin.mean():>12.4f}{r_lin.mean()*w_lin:>10.4f}")
print(f"{'angular vel, free':<24}{r_ang.mean():>12.4f}{r_ang.mean()*w_ang:>10.4f}")
print(f"{'free total':<24}{'':>12}{r_lin.mean()*w_lin + r_ang.mean()*w_ang:>10.4f}")
print(f"{'maximum total':<24}{'':>12}{w_lin + w_ang:>10.4f}")
print(f"\nfree score is {(r_lin.mean()*w_lin + r_ang.mean()*w_ang)/(w_lin+w_ang)*100:.1f}% of the maximum")

print("\nmeasured reference (final values from training runs)")
print("-" * 56)
for lbl, ang, lin in [("soft reward w2.0 @1577", 1.3390, 0.3850),
                      ("bound w1.0 @541",       None,   1.1303),
                      ("bound w3.0 @381",       None,   0.3748)]:
    if ang is not None:
        print(f"  {lbl:<24} angular {ang:.4f}  (free {r_ang.mean()*w_ang:.4f}, "
              f"only {(ang/(r_ang.mean()*w_ang).item()-1)*100:+.0f}% above it)")
    else:
        print(f"  {lbl:<24} linear {lin:.4f}  (free {r_lin.mean()*w_lin:.4f})")

print(f"\nHow small std would have to be to bring the angular free score down to "
      f"the linear one ({r_lin.mean():.3f}):")
for s in (0.35, 0.30, 0.25, 0.20, 0.15):
    v = torch.exp(-(c[:, 2] ** 2) / s ** 2).mean()
    print(f"  std={s:.2f} -> free {v:.4f} (x{w_ang} = {v*w_ang:.4f})")


print('\n' + '='*68)
print('bucketed by command magnitude '
      '(the overall mean is diluted by zero commands, so buckets are essential)')
print('='*68)

import torch
cfg = tasks.load_env_cfg("jumper.tetrapod", play=False)
cfg.scene.num_envs = 2048
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
cs = []
for _ in range(24):
    env.command_manager.reset(torch.arange(env.num_envs, device=env.device))
    cs.append(env.command_manager.get_command("twist")[:, 2].clone())
c = torch.cat(cs).abs()

bands = [(0.0,0.05,"~0 (should not turn)"),(0.05,0.2,"small"),
         (0.2,0.35,"medium"),(0.35,0.51,"large")]
print(f"\n{'command bucket':<24}{'share':>7}", end="")
for s in (0.35,0.25,0.20,0.15): print(f"{'std='+str(s):>11}", end="")
print("\n" + "-"*68)
for lo,hi,lbl in bands:
    m = (c>=lo)&(c<hi)
    print(f"{lbl:<14}{m.float().mean()*100:>6.1f}%", end="")
    for s in (0.35,0.25,0.20,0.15):
        print(f"{torch.exp(-(c[m]**2)/s**2).mean():>11.3f}", end="")
    print()
print("-"*68)
print(f"{'overall mean':<22}{100.0:>6.1f}%", end="")
for s in (0.35,0.25,0.20,0.15):
    print(f"{torch.exp(-(c**2)/s**2).mean():>11.3f}", end="")
print("\n\nThe table shows what is collected **without turning**. Turning "
      "correctly always scores 1.000, so the incentive gap is 1 minus the "
      "table value.")
print("\nIncentive gap when turning is required "
      "(|cmd|>0.2, %.0f%% of commands):" % ((c>0.2).float().mean()*100))
m = c > 0.2
for s in (0.35,0.25,0.20,0.15):
    f = torch.exp(-(c[m]**2)/s**2).mean()
    print(f"  std={s:.2f}: not turning scores {f:.3f} -> gap {1-f:.3f}   "
          f"(linear velocity is about 0.90 on the same basis)")
