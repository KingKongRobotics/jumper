#!/usr/bin/env python3
"""The moments of a `write.py` run, in seconds of its films: for cutting a video
without watching seven minutes of one.

    python tasks/jumper/calligraphy/tools/timeline.py logs/calligraphy/<text>/<run>
    python tasks/jumper/calligraphy/tools/timeline.py <run> --json > timeline.json

`film.mp4`, `top.mp4` and `low.mp4` all run in real time from the same 0 as the
log, so one time serves the three. After the log ends `film.mp4` holds on the
text, lets the ink dry and holds on the dry floor (`render.FILM_HOLD`, `DRY_S`,
`DRY_HOLD`); `top.mp4` and `low.mp4` hold 2 s.

Plain numpy: it reads `log.npz` and `plan.json`, and takes the outro's and the
film's lengths from `write.py` and `render.py` themselves, so they cannot drift.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from tasks.jumper.calligraphy import hanzi

HERE = Path(__file__).resolve().parent
#: top.mp4 and low.mp4 hold this long on the last frame (`render.render_shot`).
SHOT_HOLD = 2.0


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"_calligraphy_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timeline(run: Path) -> dict:
    write, render = _module("write"), _module("render")
    z = np.load(run / "log.npz")
    log = z["log"]
    col = {str(n): i for i, n in enumerate(z["columns"])}
    phases = [str(p) for p in z["phases"]]
    dt = float(z["dt"])
    plan = hanzi.Plan.from_json(json.loads((run / "plan.json").read_text()))
    t = log[:, col["t"]]
    ph = log[:, col["phase"]].astype(int)
    stroke = log[:, col["stroke"]].astype(int)
    stretch = log[:, col["stretch"]].astype(int)
    lower, lift = phases.index("lower"), phases.index("lift")
    outro = phases.index("outro")
    r = lambda v: round(float(v), 2)

    # Strokes: from the brush first going down to it last coming up.
    strokes = []
    seams = []
    for s in plan.strokes:
        on = (stroke == s.index) & ((ph == lower) | (ph == lift))
        if not on.any():
            continue
        down = np.where((stroke == s.index) & (ph == lower))[0]
        pieces = sorted(set(stretch[down].tolist()))
        strokes.append({"stroke": s.index + 1, "char": s.char, "start": r(t[on][0]),
                        "end": r(t[on][-1]), "pieces": len(pieces)})
        for k in pieces[1:]:
            seams.append({"stroke": s.index + 1,
                          "start": r(t[down[stretch[down] == k][0]])})
    chars = [c for c in plan.character if not c.isspace()]
    characters = []
    for i, c in enumerate(chars):
        mine = [s for s in strokes if s["char"] == i]
        characters.append({"char": c, "start": mine[0]["start"], "end": mine[-1]["end"],
                           "strokes": len(mine)})

    # The outro: the walk to the text's side, then the dance from write.OUTRO_DANCE,
    # then OUTRO_HOLD_S standing; the dance starts the last run of outro steps.
    o = np.where(ph == outro)[0]
    out = {}
    if len(o):
        breaks = np.where(np.diff(o) > 1)[0]  # the walk ends in a settle, not outro
        dance = float(t[o[breaks[-1] + 1] if len(breaks) else o[0]])
        moves, at = [], dance
        for pitch, roll, twist, seconds in write.OUTRO_DANCE:
            length = int(seconds / dt) * dt
            name = ("bow" if pitch > 0 else "twist" if twist else "roll" if roll
                    else "upright")
            moves.append({"move": name, "start": r(at), "end": r(at + length)})
            at += length
        out = {"walk": r(t[o[0]]), "dance": r(dance), "moves": moves,
               "bow": next(m["start"] for m in moves if m["move"] == "bow"),
               "still": r(at), "still_end": r(at + int(write.OUTRO_HOLD_S / dt) * dt)}

    end = float(t[-1])
    film = {"reveal": out.get("walk", r(end)), "text_end": r(end),
            "dry": r(end + render.FILM_HOLD),
            "dry_end": r(end + render.FILM_HOLD + render.DRY_S),
            "end": r(end + render.FILM_HOLD + render.DRY_S + render.DRY_HOLD),
            "end_no_dry": r(end + render.FILM_HOLD)}
    return {"run": str(run), "text": plan.character, "size_m": r(plan.size),
            "writing": {"start": strokes[0]["start"], "end": strokes[-1]["end"]},
            "characters": characters, "strokes": strokes, "seams": seams,
            "outro": out, "film": film,
            "top_low": {"end": r(end + SHOT_HOLD)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a write.py output directory")
    ap.add_argument("--json", action="store_true", help="print it as JSON")
    args = ap.parse_args()
    tl = timeline(args.run)
    if args.json:
        print(json.dumps(tl, ensure_ascii=False, indent=1))
        return 0
    f = lambda v: f"{v:7.2f}"
    print(f"{tl['text']}  ({tl['size_m'] * 100:.1f} cm a character)   seconds of film.mp4")
    print(f"{f(tl['writing']['start'])}  the brush first goes down")
    for c in tl["characters"]:
        print(f"{f(c['start'])}  {c['char']} begins ({c['strokes']} strokes), "
              f"done at {c['end']:.2f}")
    for s in tl["seams"]:
        print(f"{f(s['start'])}  seam: the arm comes back down to finish stroke {s['stroke']}")
    o = tl["outro"]
    if o:
        print(f"{f(o['walk'])}  the outro: it walks to the text's side; the film starts "
              "pulling back")
        print(f"{f(o['dance'])}  the dance: "
              + ", ".join(f"{m['move']} {m['start']:.1f}" for m in o["moves"]))
        print(f"{f(o['bow'])}  the bow")
        print(f"{f(o['still'])}  standing still, to {o['still_end']:.2f}")
    fm = tl["film"]
    print(f"{f(fm['text_end'])}  the log ends; film.mp4 holds on the text")
    print(f"{f(fm['dry'])}  the ink starts drying, gone by {fm['dry_end']:.2f}")
    print(f"{f(fm['end'])}  film.mp4 ends ({fm['end_no_dry']:.2f} with --no-dry); top.mp4 "
          f"and low.mp4 end at {tl['top_low']['end']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
