#!/usr/bin/env python3
"""Raise a dance clip onto a taller, straighter-legged stance.

    python tools/raise_dance.py tasks/jumper/dance_maze/media/maze.npz \
        tasks/jumper/dance_maze_tall/media/maze_tall.npz --height 0.145 --scale 0.5

The dances carry the body low (`jumper.dance`'s clip at a median of 112 mm), with
the middle and rear legs bent for most of the clip. This rewrites
the body height of every frame as

    z_new = height + scale * (z - median(z))

so the clip keeps its own ups and downs, at `scale` of their amplitude, around a
new median. The four support legs (middle and rear) are then re-solved by damped
least-squares IK, frame by frame, so that every support foot stays where it was in
the world. The front arms, the body's x, y and attitude and the phase track are
left as they were. Joints are clamped to their ranges in `jumper.xml`; the summary
printed at the end says how often a support joint ended on a limit and how far the
feet are from where they should be.

The clip is resampled to `--dt` (0.01 s by default) on the way through. Extra
arrays a source carries (`kp`, `kd`, the beat times) are copied when present. A
source's `source_sha256` is not: the content has changed. `source` records this
tool, its arguments and the original's `source` (or its file name).

At `--height 0.145 --scale 0.5` on the demo, maze, waist and dream_wings clips the
median IK residual is zero (the worst frame is 6 mm, on 0.04% of dream_wings), no
support joint reaches a limit, and the support legs' median extension goes from
67-68% to 81-83% of their full reach. The Brazilian
dance does not survive it: 29% of its frames put a support joint on a limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from tasks.jumper.common.constants import get_spec
from tasks.jumper.common.dance import motion as M

#: A support leg's full reach: J1 anchor to foot, maximised over the joint ranges.
#: The same for the four middle and rear legs. Only used for the printed summary.
REACH = 0.17107841


def solve_support_legs(model, data, names, root, quat, joints, targets, sites, iters=30):
    """Move the support legs of `joints` until their feet sit on `targets`."""
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    vadr = np.array([model.jnt_dofadr[j] for j in jid])
    rng = np.array([model.jnt_range[j] for j in jid])
    legs = {leg: [names.index(f"{leg}_J{k}_joint") for k in range(3)] for leg in M.SUPPORT_LEGS}
    jac = np.zeros((3, model.nv))

    def pose(j):
        data.qpos[:] = 0
        data.qpos[0:3] = root
        data.qpos[3:7] = quat
        data.qpos[qadr] = j
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    j = joints.copy()
    for _ in range(iters):
        pose(j)
        worst = 0.0
        for leg, cols in legs.items():
            e = targets[leg] - data.site_xpos[sites[leg]]
            worst = max(worst, float(np.linalg.norm(e)))
            mujoco.mj_jacSite(model, data, jac, None, sites[leg])
            jl = jac[:, vadr[cols]]
            dq = jl.T @ np.linalg.solve(jl @ jl.T + 1e-6 * np.eye(3), e)
            j[cols] = np.clip(j[cols] + dq, rng[cols, 0], rng[cols, 1])
        if worst < 2e-4:
            break
    pose(j)
    return j


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="the clip to raise (.npz)")
    ap.add_argument("out", type=Path, help="where to write the raised clip (.npz)")
    ap.add_argument("--height", type=float, required=True, help="new median body height, metres")
    ap.add_argument("--scale", type=float, required=True,
                    help="factor on the clip's own body-height excursion around its median")
    ap.add_argument("--dt", type=float, default=0.01, help="output timestep, seconds")
    args = ap.parse_args(argv)

    clip = M.load_source(args.src, "measured")
    jp, rp, rpy = M._resample(clip, args.dt)
    zmed = float(np.median(rp[:, 2]))
    znew = args.height + args.scale * (rp[:, 2] - zmed)
    n = len(jp)
    quat = M._rpy_to_quat(rpy)

    model = get_spec(None).compile()
    data = mujoco.MjData(model)
    names = M._entity_joint_names()
    jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in names]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    rng = np.array([model.jnt_range[j] for j in jid])
    legs = {leg: [names.index(f"{leg}_J{k}_joint") for k in range(3)] for leg in M.SUPPORT_LEGS}
    sites = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in M.SUPPORT_LEGS}
    hips = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh_link")
            for leg in M.SUPPORT_LEGS}
    support = sum(legs.values(), [])
    other = [c for c in range(len(names)) if c not in support]

    def pose(root, q, j):
        data.qpos[:] = 0
        data.qpos[0:3] = root
        data.qpos[3:7] = q
        data.qpos[qadr] = j
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    def extension():
        return [np.linalg.norm(data.site_xpos[sites[leg]] - data.xpos[hips[leg]]) / REACH for leg in legs]

    new = jp.copy()
    residual = np.zeros((n, 4))
    ext_before = np.zeros((n, 4))
    ext_after = np.zeros((n, 4))
    for k in range(n):
        pose(rp[k], quat[k], jp[k])
        targets = {leg: data.site_xpos[sites[leg]].copy() for leg in legs}
        ext_before[k] = extension()
        root = np.array([rp[k, 0], rp[k, 1], znew[k]])
        # Warm start from the previous frame's solution; the arms are the clip's.
        j = new[k - 1].copy() if k else jp[k].copy()
        j[other] = jp[k][other]
        new[k] = solve_support_legs(model, data, names, root, quat[k], j, targets, sites)
        residual[k] = [np.linalg.norm(targets[leg] - data.site_xpos[sites[leg]]) for leg in legs]
        ext_after[k] = extension()

    at_limit = (np.isclose(new[:, support], rng[support, 0], atol=1e-3)
                | np.isclose(new[:, support], rng[support, 1], atol=1e-3)).any(1)
    print(f"{n} frames; body z [{rp[:, 2].min():.3f}, {zmed:.3f}, {rp[:, 2].max():.3f}] -> "
          f"[{znew.min():.3f}, {np.median(znew):.3f}, {znew.max():.3f}] (min, median, max)")
    print(f"IK residual: median {np.median(residual) * 1e3:.2f} mm, p99 "
          f"{np.percentile(residual, 99) * 1e3:.2f} mm, max {residual.max() * 1e3:.2f} mm, "
          f"frames over 5 mm {(residual.max(1) > 0.005).mean() * 100:.2f}%")
    print(f"support-leg extension: median {np.median(ext_before) * 100:.0f}% -> "
          f"{np.median(ext_after) * 100:.0f}% (p95 {np.percentile(ext_after, 95) * 100:.0f}%)")
    print(f"frames with a support joint on a limit: {at_limit.mean() * 100:.2f}%")

    with np.load(args.src) as d:
        keep = {k: d[k] for k in d.files}
    out: dict[str, np.ndarray] = {}
    t = np.arange(n) * args.dt
    for pre in ("", "meas_"):
        for axis, col in zip("xyz", range(3)):
            out[f"{pre}body_{axis}"] = (znew if axis == "z" else rp[:, col]).astype(np.float32)
        for axis, col in zip(("roll", "pitch", "yaw"), range(3)):
            out[f"{pre}body_{axis}"] = rpy[:, col].astype(np.float32)
        for leg, joints in M.LEG_JOINTS.items():
            for i, nm in enumerate(joints):
                out[f"{pre}L{leg}_j{i}"] = new[:, names.index(nm)].astype(np.float32)
    out["time"] = t.astype(np.float32)
    phase = keep["phase"]
    out["phase"] = np.interp(t, np.arange(len(phase)) * float(keep["dt"]), phase).round().astype(np.float32)
    for k in ("kp", "kd", "beat_times_sim", "beat_times_audio", "bpm", "audio_start_in_sim"):
        if k in keep:
            out[k] = keep[k]
    origin = str(keep["source"]) if "source" in keep else args.src.name
    out["source"] = np.array(
        f"tools/raise_dance.py --height {args.height} --scale {args.scale} of {origin}"
    )
    out["dt"] = np.array(args.dt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
