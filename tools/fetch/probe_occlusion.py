"""Does the extended claw hide the mug's handle from the dToF? Enter the claw mode,
hold the arm out (web-up) and stand still; log the handle's visibility each half
second and write the sensor's picture (range / self / mug) to OCC_DIR. Place the
mug with FETCH_MUG_X / FETCH_MUG_Y.

    OCC_DIR=out/fetch/occ_y06 FETCH_MUG_X=0.55 FETCH_MUG_Y=0.06 .venv/bin/python tools/fetch/run.py \
        --pilot tools/fetch/probe_occlusion.py --task jumper.five_foot --mug --headless --steps 1600
"""
from __future__ import annotations

import os

from sense import Senses

OCC_DIR = os.environ.get("OCC_DIR", "")


def make_pilot(env, player):
    from probe_sense import _picture

    from controller import GamepadState

    senses = Senses(env)
    last = [-1.0]
    if OCC_DIR:
        os.makedirs(OCC_DIR, exist_ok=True)

    def pilot(t):
        senses.update(t)
        s = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0, "hat_x": 0, "hat_y": 0, "buttons": frozenset()}
        if 1.0 <= t < 1.1:
            s["buttons"] = frozenset({"LB"})
        if t >= 3.0:
            s["hat_x"] = -1                      # the arm out, web up
            s["ry"] = float(os.environ.get("OCC_NOSE", "-0.6"))
        if t - last[0] >= 0.5:
            last[0] = t
            est = senses.mug()
            if est is None:
                print(f"\n[occ] t={t:4.1f} mode={player.fsm.mode:<10} no mug")
            else:
                print(f"\n[occ] t={t:4.1f} mode={player.fsm.mode:<10} bar=({est['bar'][0]:+.3f},{est['bar'][1]:+.3f}) "
                      f"handle={'seen' if est['seen_handle'] else 'ASSUMED'} n_handle={est.get('n_handle', -1)} px={est['n']}")
            if os.environ.get("OCC_SCAN") and senses.last is not None:
                sc = senses.scan(near=0.40)
                obj = senses.last["obj"]; h = senses.last["h"]; pts = senses.last["pts"]
                print(f"[occ] obj px={int(obj.sum())} far(>0.4)={int((obj.ravel() & (pts[:, 0] > 0.4)).sum())} "
                      f"hmax={float(h[obj.ravel()].max()) if obj.any() else 0:.2f} "
                      f"cols<0.8={int((sc['cols'] < 0.8).sum())} clusters={[(round(c['range'], 2), round(c['height'], 2), round(c['width'], 2), c['n']) for c in sc['clusters']]}")
            if OCC_DIR and senses.last is not None:
                _picture(senses.last, os.path.join(OCC_DIR, f"occ_{t:04.1f}.png"))
        return GamepadState(**s)

    return pilot
