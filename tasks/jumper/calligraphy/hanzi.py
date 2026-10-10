"""A Chinese character as brush trajectories on the floor: stroke order, shape, pressure.

Plain numpy, no simulation: the same plan is read by the controller
(`tools/write.py`), by the renders and by whatever paints the ink afterwards, so it
has to be a file, not a state of the simulation.

## Where the shape comes from

Make Me a Hanzi's `graphics.txt` (https://github.com/skishore/makemeahanzi), one
JSON object per character. Its `medians` are what a brush follows: one polyline per
stroke, **in stroke order**, each running in the direction the stroke is written.
Its `strokes` are the outlines of the same strokes, kept beside them for the ink
pass but not followed by anything here.

The entries live in `data/`, one file per character, copied verbatim from
`graphics.txt`. That file is derived from Arphic's fonts and is under the Arphic
Public License, which asks for its text to travel with every copy:
`data/ARPHICPL.TXT`. `tools/strokes.py --import` adds a character from a local
`graphics.txt`.

## The frame, and the one sign that is easy to get wrong

The medians are in a 1024-unit em square with **y pointing up** -- the font's own
frame, x in [0, 1024], y in [-124, 900]. The SVG the project's site draws flips it
(`scale(1, -1) translate(0, -900)`), and a reader who copies that flip here writes
every character upside down while every stroke still looks like a stroke.

On the floor the character is laid out the way a person writes on a sheet in front
of them: the robot faces +x, so the character's **up is world +x** and its **right
is world -y**. Seen from above with +x towards the top of the picture, it reads the
right way round.

## Pressure

A brush is not a pen: a stroke is set down hard at its start and lifted off
towards its end, and that is most of what makes it look written. Each sample
carries a `press` -- the fraction of the stroke's full width, up to `HEAD` at the
head -- see `press_profile`. The controller turns it into how deep the hair goes
into the floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

#: The em square of `graphics.txt`, in font units: x in [0, 1024], y in [-124, 900].
EM = 1024.0
EM_CENTER = (512.0, 388.0)

#: Fraction of each stroke's length spent pressing the brush in, and lifting it off.
PRESS_IN = 0.06
PRESS_OUT = 0.25
#: The head of the stroke is pressed this much harder than its body, and the tail
#: leaves at this fraction of it. No harder, on purpose: the press reaches HEAD over
#: 0.1-0.15 s with the brush already moving, the arm follows 0.1 s late and goes
#: ~2 mm deeper than asked, and that alone gives the head its weight -- 12.6-13.1 mm
#: wide against a body of 8.6-8.9 (hold20-hold21, 2026-10-09). At 1.35 the heads
#: were 16-17 mm, round blots; pressing the head in place before moving did not
#: help (16-20 mm, hold18-hold19).
HEAD = 1.0
TAIL = 0.35


def entry_path(char: str) -> Path:
    """`data/u65e0.json` for 无: the code point keeps the name ASCII on every filesystem."""
    return DATA / f"u{ord(char):04x}.json"


def load_entry(char: str) -> dict:
    path = entry_path(char)
    if not path.exists():
        raise FileNotFoundError(
            f"no stroke data for {char!r} ({path.name}); add it with "
            f"`tools/strokes.py --import {char} --graphics <graphics.txt>`"
        )
    entry = json.loads(path.read_text(encoding="utf-8"))
    if entry.get("character") != char:
        raise ValueError(f"{path.name} holds {entry.get('character')!r}, not {char!r}")
    return entry


def import_entry(char: str, graphics: Path) -> Path:
    """Copy one character's line from a local `graphics.txt` into `data/`, verbatim."""
    needle = f'"character":"{char}"'
    with open(graphics, encoding="utf-8") as f:
        for line in f:
            if needle in line:
                path = entry_path(char)
                path.write_text(line if line.endswith("\n") else line + "\n", encoding="utf-8")
                return path
    raise KeyError(f"{char!r} is not in {graphics}")


def _catmull_rom(points: np.ndarray, per_segment: int = 16) -> np.ndarray:
    """Centripetal Catmull-Rom through the median's points.

    The medians are sparse -- 无's first stroke is five points -- and followed as a
    polyline every corner would be a visible kink in the ink. Centripetal
    (alpha 0.5) rather than uniform, because uniform overshoots and loops where
    two points are close, which is exactly where a hook turns.
    """
    if len(points) < 3:
        return points
    p = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = [points[0]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** 0.5
        for t in np.linspace(t1, t2, per_segment + 1)[1:]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.asarray(out)


def _resample(path: np.ndarray, ds: float) -> np.ndarray:
    """Points every `ds` along the path, the last one exactly on its end."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, path[:, k]) for k in range(path.shape[1])])


def press_profile(n: int) -> np.ndarray:
    """How hard the brush is pressed along a stroke, as a fraction of the full width.

    A person starts a stroke by setting the brush down **harder** than they carry it
    -- 顿笔, the pause-press that gives a stroke its heavy head -- then eases to the
    stroke's own weight and lifts towards the end. So: up to `HEAD` over the first
    `PRESS_IN`, settling to 1 by `2 * PRESS_IN`, and down to `TAIL` over the last
    `PRESS_OUT`. Never 0 inside the stroke: a brush lifted all the way before the
    end leaves a gap; the controller lifts it after the last sample.
    """
    u = np.linspace(0.0, 1.0, n)
    rise = np.clip(u / PRESS_IN, 0.0, 1.0)
    rise = rise * rise * (3 - 2 * rise)
    settle = np.clip((u - PRESS_IN) / PRESS_IN, 0.0, 1.0)
    head = HEAD * rise - (HEAD - 1.0) * settle * settle * (3 - 2 * settle)
    fall = 1.0 - (1.0 - TAIL) * np.clip((u - (1.0 - PRESS_OUT)) / PRESS_OUT, 0.0, 1.0)
    return head * fall


@dataclass
class Stroke:
    index: int
    #: (n, 2) floor points, world metres, `ds` apart, in writing order.
    xy: np.ndarray
    #: (n,) in [0, 1], the fraction of the full press depth at each point.
    press: np.ndarray
    #: Which character of the plan's text it belongs to.
    char: int = 0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


@dataclass
class Plan:
    #: The text: one character, or several (`plan_text`).
    character: str
    size: float
    #: The centre of the whole text.
    origin: tuple[float, float]
    ds: float
    strokes: list[Stroke] = field(default_factory=list)

    def bounds(self, pad: float = 0.0) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of every stroke, world metres."""
        xy = np.concatenate([s.xy for s in self.strokes])
        lo, hi = xy.min(0) - pad, xy.max(0) + pad
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    def label(self) -> str:
        """The text's code points, for titles and directory names: u65e0, u8df3-u8df3."""
        return "-".join(f"u{ord(c):04x}" for c in self.character)

    def to_json(self) -> dict:
        return {
            "character": self.character,
            "size_m": self.size,
            "origin_m": list(self.origin),
            "ds_m": self.ds,
            "frame": "world; character up = +x, character right = -y",
            "source": "Make Me a Hanzi graphics.txt medians (Arphic Public License)",
            "strokes": [
                {
                    "index": s.index,
                    "char": s.char,
                    "length_m": round(s.length, 5),
                    "xy": np.round(s.xy, 5).tolist(),
                    "press": np.round(s.press, 4).tolist(),
                }
                for s in self.strokes
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> Plan:
        plan = cls(data["character"], data["size_m"], tuple(data["origin_m"]), data["ds_m"])
        for s in data["strokes"]:
            plan.strokes.append(Stroke(s["index"], np.asarray(s["xy"]), np.asarray(s["press"]),
                                       s.get("char", 0)))
        return plan


def font_to_floor(p: np.ndarray, size: float, origin: tuple[float, float]) -> np.ndarray:
    """Font units -> world metres. The em square's centre lands on `origin`, and the
    em square is `size` metres across."""
    k = size / EM
    u = (p[:, 0] - EM_CENTER[0]) * k  # character right
    v = (p[:, 1] - EM_CENTER[1]) * k  # character up
    return np.column_stack([origin[0] + v, origin[1] - u])


def plan(char: str, size: float = 0.30, origin: tuple[float, float] = (0.0, 0.0),
         ds: float = 0.001) -> Plan:
    """The brush trajectory for `char`, `size` metres across, centred on `origin`."""
    entry = load_entry(char)
    out = Plan(char, size, origin, ds)
    for i, median in enumerate(entry["medians"]):
        pts = font_to_floor(np.asarray(median, dtype=float), size, origin)
        xy = _resample(_catmull_rom(pts), ds)
        out.strokes.append(Stroke(i, xy, press_profile(len(xy))))
    return out


#: The space between two characters of a text, as a fraction of their size.
GAP = 0.15


def plan_text(text: str, size: float = 0.30, origin: tuple[float, float] = (0.0, 0.0),
              layout: str = "vertical", gap: float = GAP, ds: float = 0.001) -> Plan:
    """Several characters, `size` metres each, the whole text centred on `origin`.

    `vertical` reads top to bottom, the way 地书 is usually written -- the first
    character furthest along +x, where the robot starts facing; `horizontal` left
    to right, along -y. Strokes keep the text's writing order and are numbered
    through it; each knows its character (`Stroke.char`).
    """
    if layout not in ("vertical", "horizontal"):
        raise ValueError(f"layout {layout!r}: vertical or horizontal")
    pitch = size * (1.0 + gap)
    out = Plan(text, size, origin, ds)
    for i, char in enumerate(text):
        k = (len(text) - 1) / 2 - i
        centre = ((origin[0] + k * pitch, origin[1]) if layout == "vertical"
                  else (origin[0], origin[1] + k * pitch))
        for s in plan(char, size, centre, ds).strokes:
            out.strokes.append(Stroke(len(out.strokes), s.xy, s.press, i))
    return out
