"""The dash's reference: the high jump's recording carried forward.

`tasks/jumper/jump/ref/high_jump_flat.npz` is a crouch, a push, 0.41 s in the air
to 0.235 m, and a landing, all in place. The dash is that hop with a forward
travel `DASH_M` (Andrew, 2026-10-14: "a quick hop but in a specific direction,
not straight up like the jump"): the base's x is carried forward -- a smooth
build-up through the push, `DASH_M` over the flight, a stop through the
landing's absorb -- with the world-frame velocity to match, and each foot moves
with the body while it is in the air and stays planted where it is while it is
down, so the legs' reference bends backward under the push and the landing
feet are planted a hop ahead. Joint angles, the contact schedule and the
heights are the jump's own; the residual policy learns the thrust.

    .venv/bin/python tools/battle/make_dash.py        # writes tasks/jumper/dash/ref/dash_flat.npz
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DASH_M = float(os.environ.get("DASH_M", "0.30"))     # the hop's travel; 0.41 s in the air -> 0.73 m/s
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks/jumper/jump/ref/high_jump_flat.npz"
OUT = REPO / "tasks/jumper/dash/ref/dash_flat.npz"


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def main() -> None:
    d = dict(np.load(SRC, allow_pickle=True))
    t, c = d["t"], d["contact"]
    airborne = c.sum(1) == 0
    idx = np.nonzero(airborne)[0]
    i_off, i_land = int(idx[0]), int(idx[-1]) + 1                      # first airborne frame, first landed frame
    all_down = c.sum(1) == 6
    i_push = int(np.nonzero(all_down[:i_off])[0][0])                   # the push starts where every foot is down before the flight
    t_flight = t[i_land] - t[i_off]
    v = DASH_M / t_flight
    n = len(t)
    x = np.zeros(n)
    # the push: velocity builds smoothly from 0 at the push's start to v at take-off
    for i in range(i_push, i_off):
        u = (t[i] - t[i_push]) / max(1e-6, t[i_off] - t[i_push])
        x[i] = x[i - 1] + v * smoothstep(u) * (t[i] - t[i - 1])
    for i in range(i_off, i_land):
        x[i] = x[i - 1] + v * (t[i] - t[i - 1])
    # the landing: the absorb brings it to a stop over the frames that follow
    i_stop = min(n - 1, i_land + int(0.20 / np.median(np.diff(t))))
    for i in range(i_land, n):
        u = (t[i] - t[i_land]) / max(1e-6, t[i_stop] - t[i_land])
        x[i] = x[i - 1] + v * (1 - smoothstep(u)) * (t[i] - t[i - 1])
    vx = np.gradient(x, t)
    out = {k: v_.copy() for k, v_ in d.items()}
    out["base_pos"][:, 0] += x
    out["base_lin_vel"][:, 0] += vx
    foot = out["foot_pos"]
    for leg in range(foot.shape[1]):
        planted_x = None
        for i in range(n):
            if c[i, leg] > 0.5:
                if planted_x is None or c[i - 1, leg] < 0.5:
                    planted_x = d["foot_pos"][i, leg, 0] + x[i]            # lands a hop ahead of where it left
                foot[i, leg, 0] = planted_x
            else:
                foot[i, leg, 0] = d["foot_pos"][i, leg, 0] + x[i]
                planted_x = None
    out["meta"] = np.array(str(d["meta"]).replace("}", f', "dash_m": {DASH_M}, "source": "high_jump_flat.npz"}}', 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"[dash] push from {t[i_push]:.3f} s, airborne {t[i_off]:.3f}-{t[i_land]:.3f} s ({t_flight:.2f} s), "
          f"v {v:.2f} m/s, travel {x[-1]:.3f} m, vx peak {vx.max():.2f}; wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
