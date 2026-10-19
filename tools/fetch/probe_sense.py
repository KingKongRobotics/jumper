"""Run the ground-truth fetch and, beside it, what the senses say: the detected bar
against the true bar in the base frame, every half second, plus a few range-image
pictures (range / self mask / object mask) into the directory PROBE_OUT."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fetch import _quat_rot
from fetch import make_pilot as fetch_pilot
from sense import Senses

OUT = Path(os.environ.get("PROBE_OUT", "."))


def make_pilot(env, player):
    inner = fetch_pilot(env, player)
    senses = Senses(env)
    robot, mug = env.scene["robot"], env.scene["mug"]
    from mug import FEATURE
    off = np.array([FEATURE["offset"][0], FEATURE["offset"][1], 0.0])
    last = [-1.0]
    shots = {2.0: "stowed", 6.0: "arm_out", 9.0: "fine", 13.0: "depth"}
    errs = []

    def truth_bar_b():
        bp = robot.data.root_link_pos_w[0].cpu().numpy(); bq = robot.data.root_link_quat_w[0].cpu().numpy()
        mp = mug.data.root_link_pos_w[0].cpu().numpy(); mq = mug.data.root_link_quat_w[0].cpu().numpy()
        bar_w = mp + _quat_rot(mq, off)
        w, x, y, z = bq
        return _quat_rot(np.array([w, -x, -y, -z]), bar_w - bp)

    def pilot(t):
        senses.update(t)
        s = inner(t)
        if t - last[0] >= 0.5:
            last[0] = t
            est, tb = senses.mug(), truth_bar_b()
            mouth, closing, inward, fin = senses.body()
            if est is None:
                print(f"\n[sense] t={t:6.2f} no mug  truth_bar_b={np.round(tb, 3)} mouth_b={np.round(mouth, 3)} yaw_gyro={np.degrees(senses.yaw):+.1f}")
            else:
                e = est["bar"] - tb
                errs.append(e[:2])
                lf = senses.last
                print(f"\n[sense] t={t:6.2f} bar_b={np.round(est['bar'], 3)} truth={np.round(tb, 3)} err=({e[0]*100:+.1f},{e[1]*100:+.1f}) cm "
                      f"handle={'seen' if est['seen_handle'] else 'ASSUMED'} n={lf['est']['n'] if lf and lf['est'] else 0} "
                      f"self_px={int(lf['self'].sum()) if lf else 0} mouth_b={np.round(mouth, 3)} yaw_gyro={np.degrees(senses.yaw):+.1f}")
        for ts, name in list(shots.items()):
            if t >= ts and senses.last is not None:
                del shots[ts]
                _picture(senses.last, OUT / f"tof_{name}.png")
        if t > 14.0 and errs:
            a = np.array(errs)
            print(f"\n[sense] bar error over the run: mean ({a[:,0].mean()*100:+.1f},{a[:,1].mean()*100:+.1f}) cm, "
                  f"abs max ({np.abs(a[:,0]).max()*100:.1f},{np.abs(a[:,1]).max()*100:.1f}) cm over {len(a)} samples")
            errs.clear()
        return s

    return pilot


def _picture(frame, path):
    from PIL import Image
    r = frame["range"].copy()
    img = np.zeros((42, 54, 3), dtype=np.uint8)
    valid = r > 0
    # Log scale over the sensor's whole reach, so the dunes and palms show in the
    # wander's inset while the mug keeps its contrast up close.
    g = np.clip(255 * (1 - np.log(np.clip(r, 0.05, 8.8) / 0.05) / np.log(8.8 / 0.05)), 0, 255).astype(np.uint8)
    img[..., 0] = img[..., 1] = img[..., 2] = np.where(valid, g, 0)
    img[frame["self"]] = [60, 60, 220]
    img[frame["obj"]] = [230, 60, 60]
    Image.fromarray(img).resize((54 * 8, 42 * 8), Image.NEAREST).save(path)
