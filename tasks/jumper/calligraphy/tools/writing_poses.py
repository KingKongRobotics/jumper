#!/usr/bin/env python3
"""The arm poses `jumper.calligraphy` trains with: where the brush can write.

    python tasks/jumper/calligraphy/tools/writing_poses.py               # ~3 min on CPU
    python tasks/jumper/calligraphy/tools/writing_poses.py --palm-x-min 0.10

Writes `tasks/jumper/calligraphy/data/writing_poses.npz`: for every 5 mm cell of
floor the brush can write on, in the trunk's frame, the four arm angles that put
the apex `HOVER` above it, on it and `DEPTH_MAX` below it. The training's
writing command (`mdp/writing.py`) moves the arm between these, so the policy
learns to stand still with the arm anywhere it will be when a character is
written.

The band is built wider than the one `write.py` uses with the shipped policy:
`--palm-x-min` 0.10 against 0.13. The narrower limit is where *that* policy holds
still (step 0); the training is what is meant to move it.

Committed rather than built at training time: it needs mjlab and ~3 minutes, the
training has to see exactly the same poses on every machine, and the file is a
few tens of kilobytes. Rebuild it when the brush or the arm's reach changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "tasks/jumper/calligraphy/data/writing_poses.npz"
CACHE = REPO / "logs" / "calligraphy" / "cache"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--palm-x-min", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from tasks.jumper.calligraphy import arm as armmod
    from tasks.jumper.calligraphy import sim

    sm = sim.build(1)
    arm = armmod.Arm(sm.env.sim.mj_model, palm_x_min=args.palm_x_min)
    standing = sim.standing_qpos(sm.env.sim.mj_model)
    rm = armmod.cached_reach_map(arm, standing, CACHE)
    sm.env.close()

    cells = np.argwhere(rm.ok)
    xy = cells * rm.cell + (rm.x0, rm.y0)
    arm.set_state(standing)
    heights = (armmod.HOVER, 0.0, -armmod.DEPTH_MAX)
    q = np.zeros((len(cells), len(heights), 4), dtype=np.float32)
    worst = 0.0
    for i, (a, b) in enumerate(cells):
        seed = rm.seed[a, b]
        for h, z in enumerate(heights):
            q[i, h], err = arm.ik(np.array([*xy[i], z]), seed)
            worst = max(worst, err)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, xy=xy.astype(np.float32), q=q,
                        heights=np.array(heights, dtype=np.float32),
                        palm_x_min=args.palm_x_min, cell=rm.cell)
    print(f"[poses] {len(cells)} cells ({rm.ok.sum() * rm.cell**2 * 1e4:.0f} cm^2) at "
          f"palm x >= {args.palm_x_min} m; worst IK residual {worst * 1000:.2f} mm; "
          f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
