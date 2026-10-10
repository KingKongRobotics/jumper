#!/usr/bin/env python3
"""Step 1: a character -> its brush plan on the floor, and a picture of it.

    python tasks/jumper/calligraphy/tools/strokes.py                    # 无, 0.30 m
    python tasks/jumper/calligraphy/tools/strokes.py --char 无 --size 0.40
    python tasks/jumper/calligraphy/tools/strokes.py --import 永 --graphics <graphics.txt>

Writes `plan.json` (what `write.py` follows) and `plan.svg` (the same plan seen
from above, +x up, so it reads the right way round) to
`logs/calligraphy/<u65e0>/`. The SVG draws each stroke's width from its `press`,
numbers the strokes at their start and marks where each one begins, which is the
quickest way to see a stroke written backwards or out of order.

Plain numpy; no simulation is built. See `hanzi.py` for the frame and the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tasks.jumper.calligraphy import hanzi

REPO = Path(__file__).resolve().parents[4]


def out_dir(char: str) -> Path:
    return REPO / "logs" / "calligraphy" / f"u{ord(char):04x}"


def svg(p: hanzi.Plan, width_m: float = 0.012, px: int = 600) -> str:
    """Top view, +x up, -y right. Stroke width = `press` * `width_m`."""
    half = p.size / 2 * 1.1
    cx, cy = p.origin
    k = px / (2 * half)

    def to_px(xy: np.ndarray) -> np.ndarray:
        # screen x = -world y (character right), screen y = -world x (character up)
        return np.column_stack([(-(xy[:, 1] - cy) + half) * k, (-(xy[:, 0] - cx) + half) * k])

    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
            f'viewBox="0 0 {px} {px}"><rect width="100%" height="100%" fill="#f4efe4"/>')]
    # The em square, for scale.
    sq = to_px(np.array([[cx + p.size / 2, cy + p.size / 2], [cx - p.size / 2, cy - p.size / 2]]))
    out.append(f'<rect x="{sq[0, 0]:.1f}" y="{sq[0, 1]:.1f}" width="{sq[1, 0] - sq[0, 0]:.1f}" '
               f'height="{sq[1, 1] - sq[0, 1]:.1f}" fill="none" stroke="#c9b99a" '
               f'stroke-dasharray="4 4"/>')
    for s in p.strokes:
        q = to_px(s.xy)
        for i in range(len(q) - 1):
            w = max(0.5, s.press[i] * width_m * k)
            out.append(f'<line x1="{q[i, 0]:.1f}" y1="{q[i, 1]:.1f}" x2="{q[i + 1, 0]:.1f}" '
                       f'y2="{q[i + 1, 1]:.1f}" stroke="#1d1d1b" stroke-width="{w:.1f}" '
                       f'stroke-linecap="round"/>')
        out.append(f'<circle cx="{q[0, 0]:.1f}" cy="{q[0, 1]:.1f}" r="5" fill="#c0392b"/>')
        out.append(f'<text x="{q[0, 0] + 8:.1f}" y="{q[0, 1] - 8:.1f}" font-size="18" '
                   f'fill="#c0392b" font-family="sans-serif">{s.index + 1}</text>')
    out.append(f'<text x="10" y="{px - 10}" font-size="14" fill="#6b5f4b" '
               f'font-family="sans-serif">{p.character}  {p.size:.2f} m  (+x up)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="无")
    ap.add_argument("--size", type=float, default=0.30, help="em square, metres")
    ap.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0),
                    help="world xy of the em square's centre, metres")
    ap.add_argument("--ds", type=float, default=0.001, help="sample spacing, metres")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--import", dest="import_char", metavar="CHAR",
                    help="copy CHAR's entry from --graphics into data/ and stop")
    ap.add_argument("--graphics", type=Path, help="a local Make Me a Hanzi graphics.txt")
    args = ap.parse_args()

    if args.import_char:
        if args.graphics is None:
            ap.error("--import needs --graphics")
        print(f"wrote {hanzi.import_entry(args.import_char, args.graphics)}")
        return 0

    p = hanzi.plan(args.char, args.size, tuple(args.origin), args.ds)
    out = args.out or out_dir(args.char)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(p.to_json()), encoding="utf-8")
    (out / "plan.svg").write_text(svg(p), encoding="utf-8")
    total = sum(s.length for s in p.strokes)
    print(f"{p.character}: {len(p.strokes)} strokes, {total:.3f} m of ink, "
          f"{args.size:.2f} m em square at {tuple(args.origin)}")
    for s in p.strokes:
        lo, hi = s.xy.min(0), s.xy.max(0)
        print(f"  stroke {s.index + 1}: {s.length:.3f} m, {len(s.xy)} samples, "
              f"x [{lo[0]:+.3f}, {hi[0]:+.3f}] y [{lo[1]:+.3f}, {hi[1]:+.3f}]")
    print(f"wrote {out / 'plan.json'} and plan.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
