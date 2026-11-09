"""Where the brush left ink, from a `write.py` log: the data the ink pass paints from.

Steps 3 and 4 of the project read the same thing -- `marks(log)` -- so the ink in
the rendered preview and the ink a compositor paints from `ink.json` are one set
of points.

## What counts as ink

The hair may go into the floor (`brush.py`). A control step leaves
ink when the hair's tip is below the floor **and** the arm is lowering, writing
or lifting; the ink is centred where the hair's axis meets the floor and is as
wide as its section there (`brush.ink_point`, `brush.section_width`, logged
by `write.py` as `ix`, `iy`, `width`). Anywhere else it would be the brush
dipping on the way somewhere; `write.py` reports that as stray ink and it is not
painted here, so a fault shows in the report rather than being quietly drawn.

Consecutive inked steps of one stretch form a **mark**: one continuous trace of
the brush -- one per stroke, when every stroke is written whole. Gaps shorter
than `GAP_S` inside a stretch are bridged at the smallest width, so a tip that
rises out of the floor for a step does not break a stroke in two.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from tasks.jumper.calligraphy import brush

#: Phases of `write.py` in which a contact is ink (lower, write, lift).
INK_PHASES = ("lower", "write", "lift")
#: Gaps in contact up to this long, inside one writing stretch, do not break a mark.
GAP_S = 0.1
#: Ink sits this far above the floor in the preview, so it does not z-fight.
LIFT = 0.0004


@dataclass
class Mark:
    stroke: int
    stretch: int
    t: np.ndarray        # (n,) s
    xyz: np.ndarray      # (n, 3) m, the tip
    depth: np.ndarray    # (n,) m, how far the apex is below the floor
    press: np.ndarray    # (n,) planned press, a fraction of the full width
    width: np.ndarray    # (n,) m, the hair's section at the floor


class Log:
    def __init__(self, path):
        z = np.load(path)
        self.rows = z["log"]
        self.col = {str(n): i for i, n in enumerate(z["columns"])}
        self.phases = [str(p) for p in z["phases"]]
        self.qpos = z["qpos"] if "qpos" in z.files else None
        self.dt = float(z["dt"]) if "dt" in z.files else 0.02

    def __getitem__(self, name: str) -> np.ndarray:
        return self.rows[:, self.col[name]]


def marks(log: Log, plan) -> list[Mark]:
    ink_ids = [log.phases.index(p) for p in INK_PHASES]
    in_phase = np.isin(log["phase"], ink_ids)
    inked = (log["contact"] > 0) & in_phase
    stretch = log["stretch"]
    # Bridge short gaps between two inked steps of the same stretch.
    gap = max(1, round(GAP_S / log.dt))
    idx = np.flatnonzero(inked)
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= gap + 1 and stretch[a] == stretch[b] and in_phase[a:b].all():
            inked[a:b] = True
    out: list[Mark] = []
    i, n = 0, len(inked)
    while i < n:
        if not inked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inked[j + 1] and stretch[j + 1] == stretch[i]:
            j += 1
        sel = slice(i, j + 1)
        stroke = int(log["stroke"][i])
        samples = np.clip(log["sample"][sel].astype(int), 0, len(plan.strokes[stroke].xy) - 1)
        width = log["width"][sel].copy()
        bridged = width <= 0
        if bridged.any() and (~bridged).any():
            width[bridged] = width[~bridged].min()
        out.append(Mark(
            stroke=stroke, stretch=int(stretch[i]), t=log["t"][sel].copy(),
            xyz=np.column_stack([log["ix"][sel], log["iy"][sel], np.zeros(j + 1 - i)]),
            depth=log["depth"][sel].copy(), press=plan.strokes[stroke].press[samples],
            width=width,
        ))
        i = j + 1
    return out


def to_json(ms: list[Mark], plan, shots) -> dict:
    out = {
        "character": plan.character,
        "size_m": plan.size,
        "frame": "world metres; character up = +x, right = -y; floor at z = 0",
        "width_rule": f"the brush's hair, {brush.HAIR_LENGTH * 1000:g} mm long, has radius "
                      f"{brush.HAIR_RADIUS * 1000:g} mm * sin(pi/2 * s / "
                      f"{brush.HAIR_LENGTH * 1000:g} mm) at s from its tip and is sunk into "
                      "the floor; width = its section where its axis meets the floor",
        "cameras": [s.to_json() for s in shots],
        "marks": [],
    }
    for m in ms:
        floor = np.column_stack([m.xyz[:, :2], np.zeros(len(m.t))])
        entry = {
            "stroke": m.stroke, "stretch": m.stretch,
            "t": np.round(m.t, 3).tolist(),
            "xy": np.round(m.xyz[:, :2], 5).tolist(),
            "depth_m": np.round(m.depth, 5).tolist(),
            "press": np.round(m.press, 3).tolist(),
            "width_m": np.round(m.width, 5).tolist(),
            "pixels": {s.name: np.round(s.project(floor), 2).tolist() for s in shots},
        }
        out["marks"].append(entry)
    return out


def to_svg(ms: list[Mark], plan, px: int = 1000) -> str:
    """The ink from above, +x up, as filled circles along each mark: the
    shape a compositor would lay down, at the character's own scale."""
    x0, x1, y0, y1 = plan.bounds()
    half = max(x1 - x0, y1 - y0, plan.size) / 2 * 1.15
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    k = px / (2 * half)
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
              f'viewBox="0 0 {px} {px}">'),
             '<rect width="100%" height="100%" fill="#d9d4cb"/>']
    for m in ms:
        u = (-(m.xyz[:, 1] - cy) + half) * k
        v = (-(m.xyz[:, 0] - cx) + half) * k
        r = m.width / 2 * k
        parts.append(f'<g fill="#2b2b2e" data-stroke="{m.stroke + 1}">')
        parts += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{c:.1f}"/>' for a, b, c in zip(u, v, r)]
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)
