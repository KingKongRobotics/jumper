"""The control readout on a take: each crab's pad in its own corner.

Andrew (2026-10-15): "I would like to see the control output in the bottom left
and bottom right corner for each robot." The red crab's panel is bottom left and
the blue crab's bottom right, each showing what its pilot put on its virtual pad
this step -- the two sticks, the triggers, the hat and the buttons held -- with
the controller's mode and the pilot's intent above them, and the score across
the top. It is painted into the recorder's frames as they are written
(`HudVideo`, over `mjrl.viewer.video.ReplayVideo`), so a take with the readout
is one file and one pass; the fetch film's overlay (`tools/fetch/hud.py`) was
drawn as PNGs and composed afterwards, which is the other way to do it.

What is drawn is the pad the pilot sends, the way the fetch film's panel showed
its autopilot's pad: `VirtualPad.last` is the frame the controller read, so a
chord that was pressed is on the panel for exactly the ticks it was held.
"""
from __future__ import annotations

import math

import numpy as np
from mjrl.viewer.video import ReplayVideo
from PIL import Image, ImageDraw, ImageFont

W, H = 300, 150                      # one panel
MARGIN = 20
INK = (235, 235, 235)
DIM = (130, 130, 130)
BG = (16, 16, 20, 190)
COLOURS = {"red": (235, 70, 55), "blue": (70, 120, 245)}

_FONT: dict = {}


def _font(size: int):
    if size not in _FONT:
        for name in ("DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf"):
            try:
                _FONT[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]


def panel(name: str, style: str, mode: str, intent: str, pad) -> Image.Image:
    """One crab's readout: `pad` is a `GamepadState` (or None for hands off)."""
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)
    f12, f14, f16 = _font(12), _font(14), _font(16)
    accent = COLOURS.get(name, INK)
    d.rectangle((0, 0, 5, H), fill=accent)
    d.text((14, 6), f"{name.upper()} {style}", fill=accent, font=f16)
    d.text((14, 26), f"{mode}", fill=INK, font=f14)
    d.text((14, 42), intent[:30], fill=DIM, font=f12)
    lx, ly, rx, ry = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lx", "ly", "rx", "ry"))
    lt, rt = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lt", "rt"))
    hat_x, hat_y = (getattr(pad, k, 0) if pad is not None else 0 for k in ("hat_x", "hat_y"))
    buttons = sorted(pad.buttons) if pad is not None else []
    # the sticks: a dot in a ring, the ring's rim being full deflection
    for i, (lab, x, y) in enumerate((("L", lx, ly), ("R", rx, ry))):
        cx, cy, r = 50 + i * 70, 100, 28
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=DIM, width=1)
        d.line((cx - r, cy, cx + r, cy), fill=(60, 60, 64)); d.line((cx, cy - r, cx, cy + r), fill=(60, 60, 64))
        if x or y:
            d.line((cx, cy, cx + x * r, cy + y * r), fill=accent, width=2)
        px, py = cx + x * r, cy + y * r
        d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=accent if (x or y) else DIM)
        d.text((cx - 4, cy + r + 4), lab, fill=DIM, font=f12)
    # the triggers, the hat and the buttons
    x0 = 160
    for i, (lab, v) in enumerate((("LT", lt), ("RT", rt))):
        y = 72 + i * 16
        d.text((x0, y - 1), lab, fill=DIM, font=f12)
        d.rectangle((x0 + 24, y + 2, x0 + 124, y + 10), outline=DIM)
        if v > 0:
            d.rectangle((x0 + 24, y + 2, x0 + 24 + 100 * v, y + 10), fill=accent)
    hx, hy = x0 + 20, 125
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        on = (dx and dx == hat_x) or (dy and dy == hat_y)
        d.rectangle((hx + dx * 10 - 4, hy + dy * 10 - 4, hx + dx * 10 + 4, hy + dy * 10 + 4),
                    fill=accent if on else (60, 60, 64))
    d.text((x0 + 44, 112), " ".join(buttons)[:18], fill=accent, font=f14)
    return im


class Hud:
    """Paints the readouts into a frame. `crabs` is a list of (name, style, pad,
    player, pilot) in the order red, blue; `referee` gives the score."""

    def __init__(self, crabs, referee) -> None:
        self.crabs = crabs
        self.referee = referee

    def paint(self, frame: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(frame))
        height, width = frame.shape[:2]
        for i, (name, style, pad, player, pilot) in enumerate(self.crabs):
            fsm = player.fsm
            mode = fsm.mode + ("" if fsm.running_policy else " (ramping)")
            st = getattr(pilot, "state", None) or {}
            intent = st.get("intent") or ("-" if pilot is not None else "keys")
            stage = st.get("stage") or ""
            if st.get("lock"):
                intent = f"{intent} / lock {st['lock'][0]}"
            elif stage:
                intent = f"{intent} / {stage}"
            p = panel(name, style, mode, intent, getattr(pad, "last", None))
            x = MARGIN if i == 0 else width - W - MARGIN
            im.paste(p, (x, height - H - MARGIN), p)
        ref = self.referee
        score = f"RED {ref.score['red']}  -  {ref.score['blue']} BLUE"
        d = ImageDraw.Draw(im, "RGBA")
        f = _font(20)
        tw = d.textlength(score, font=f)
        d.rectangle((width / 2 - tw / 2 - 14, 14, width / 2 + tw / 2 + 14, 44), fill=BG)
        d.text((width / 2 - tw / 2, 18), score, fill=INK, font=f)
        label = {"fight": f"round {ref.round}", "taunt": "the stare-down", "ready": "on the marks",
                 "linger": "", "cheer": "", "victory": "the dance"}.get(ref.state, ref.state)
        if label:
            f2 = _font(14)
            tw2 = d.textlength(label, font=f2)
            d.rectangle((width / 2 - tw2 / 2 - 10, 46, width / 2 + tw2 / 2 + 10, 66), fill=BG)
            d.text((width / 2 - tw2 / 2, 48), label, fill=DIM, font=f2)
        return np.asarray(im)


class HudVideo(ReplayVideo):
    """`ReplayVideo` with the readouts painted into every frame it writes."""

    def __init__(self, *args, hud: Hud, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hud = hud

    def _write_frames(self, count):
        self._renderer.update(self._env.sim.data)
        frame = self._hud.paint(self._renderer.render())
        for _ in range(count):
            self._writer.send(frame)
            self.frame_count += 1


def preview(path: str) -> None:
    """One panel on a grey frame, for a look at the layout without a take."""
    from controller import GamepadState

    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    from types import SimpleNamespace as NS

    ref = NS(score={"red": 1, "blue": 0}, round=2, state="fight")
    player = NS(fsm=NS(mode="locomotion", running_policy=True))
    pad = NS(last=GamepadState(lx=0.3, ly=-0.8, rx=0.5, lt=0.7, hat_x=-1, buttons=frozenset({"L3", "menu"})))
    pilot = NS(state={"intent": "charge", "stage": "", "lock": None})
    hud = Hud([("red", "bull", pad, player, pilot), ("blue", "matador", pad, player, None)], ref)
    Image.fromarray(hud.paint(frame)).save(path)


if __name__ == "__main__":
    import sys

    preview(sys.argv[1] if len(sys.argv) > 1 else "hud_preview.png")
    print(math.pi and "ok")
