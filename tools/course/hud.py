"""The readout: what the crab sees, maps and plans, its pad, and the clock.

The panel is the race's (`tools/race/hud.py`): the depth image shaded by what the
sensing made of each zone, the map round the base with the plan, the two sticks.
The map's outline is the lane's instead of the rink's, and the top carries the
course clock, the section and the splits instead of a score.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import track
from mjrl.viewer.video import ReplayVideo
from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location("race_hud", Path(__file__).resolve().parents[1] / "race" / "hud.py")
_race = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_race)
W, H, MARGIN, INK, DIM, BG, COLOURS = _race.W, _race.H, _race.MARGIN, _race.INK, _race.DIM, _race.BG, _race.COLOURS
DEPTH_W, DEPTH_H, MAP_PX, MAP_SPAN = _race.DEPTH_W, _race.DEPTH_H, _race.MAP_PX, _race.MAP_SPAN
_font, LEGEND = _race._font, _race.LEGEND
BALL_C = (90, 230, 120)


def depth_image(frame) -> Image.Image:
    """The race's, with the ball the pilot is after in green (Andrew: "the ball
    never shows green in the depth view" -- only the map's marker was)."""
    return _race.depth_image(frame, extra=(("ball_px", BALL_C),))
CLOCK_PX, LABEL_PX = 44, 16       # the clock's and its label's type, in 720p pixels


def map_image(mapper, off: float, target=None, accent=INK, marks=(), ball=None, balls=()) -> Image.Image:
    """The map round the base: the lane's kerbs in white, what the sensor has mapped
    in grey, the plan in cyan, the ball being pushed in green and the ones still to
    push as green rings where they were told, the target ringed in yellow."""
    from mapper import FREE, OCC, RES, N

    k = int(MAP_SPAN / 2 / RES)
    r0, c0 = mapper.cell(mapper.x, mapper.y)
    r0, c0 = min(max(r0, k), N - k - 1), min(max(c0, k), N - k - 1)
    sub = mapper.grid[r0 - k: r0 + k, c0 - k: c0 + k]
    img = np.full(sub.shape + (3,), 150, dtype=np.uint8)
    img[sub < FREE] = (200, 196, 185)
    img[sub > OCC] = (70, 70, 70)
    scale = MAP_PX / sub.shape[1]
    im = Image.fromarray(img[::-1]).resize((MAP_PX, int(sub.shape[0] * scale)), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def px(wx, wy):
        r, c = mapper.cell(wx, wy)
        return (c - (c0 - k)) * scale + scale / 2, (2 * k - 1 - (r - (r0 - k))) * scale + scale / 2

    hy = track.LANE_W / 2
    x0, x1 = mapper.x - MAP_SPAN, mapper.x + MAP_SPAN
    for sy in (-hy, hy):
        d.line([px(x0, sy), px(x1, sy)], fill=(250, 250, 250), width=2)
    for wx in marks:
        d.line([px(wx - off, -hy), px(wx - off, hy)], fill=(250, 250, 250), width=1)
    if len(mapper.path) > 1:
        d.line([px(*p) for p in mapper.path], fill=(0, 200, 220), width=2)
    for b in balls:
        x, y = px(*b)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=BALL_C, width=1)
    if ball is not None:
        x, y = px(*ball)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=BALL_C)
    if target is not None:
        x, y = px(*target)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 220, 0), width=2)
    x, y = px(mapper.x, mapper.y)
    hx, hy2 = px(mapper.x + 0.25 * math.cos(mapper.yaw), mapper.y + 0.25 * math.sin(mapper.yaw))
    d.line([(x, y), (hx, hy2)], fill=accent, width=3)
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=accent)
    return im


PAD, IMG_H, COL_W = _race.PAD, _race.IMG_H, _race.COL_W


def panel(name: str, mode: str, intent: str, pad, senses, mapper, st: dict, scale: float = 1.0, pilot=None) -> Image.Image:
    """The race module's panel (`wide_panel`) with the lane's map and the ball tinted."""
    m = None if mapper is None else map_image(mapper, st.get("off", 0.0), st.get("target"), COLOURS.get(name, INK),
                                               st.get("marks", ()), st.get("ball"), st.get("balls", ()))
    return _race.wide_panel(name, mode, intent, pad, senses, m, scale=scale, pilot=pilot, depth=depth_image)


class Hud:
    def __init__(self, name: str, pad, player, pilot, referee, scale: float | None = None) -> None:
        self.name, self.pad, self.player, self.pilot, self.referee = name, pad, player, pilot, referee
        self.scale = scale                              # None: from the frame's height, 720p being 1

    def paint(self, frame: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(frame))
        width = frame.shape[1]
        s = self.scale if self.scale is not None else frame.shape[0] / 720.0

        def P(v):
            return round(v * s)
        fsm = self.player.fsm
        mode = fsm.mode + ("" if fsm.running_policy else " (ramping)")
        st = getattr(self.pilot, "state", None) or {}
        intent = st.get("intent") or ("-" if self.pilot is not None else "keys")
        if st.get("stage"):
            intent = f"{intent}: {st['stage']}"
        p = panel(self.name, mode, intent, getattr(self.pad, "last", None), getattr(self.pilot, "senses", None),
                  getattr(self.pilot, "mapper", None), st, scale=s, pilot=self.pilot)
        im.paste(p, (P(MARGIN), P(MARGIN)), p)
        ref = self.referee
        d = ImageDraw.Draw(im, "RGBA")
        # the clock, big, centred on the frame, the section and the splits under it in one box
        # (Andrew: "the timer is still far too small")
        f = _font(P(CLOCK_PX))
        clock = f"{ref.clock(ref.t):5.1f} s"
        tw = d.textlength(clock, font=f)
        label = {"ready": "on the mark", "run": ref.section, "finish": "the dance", "over": ""}.get(ref.state, "")
        if ref.splits:
            label += "   " + "  ".join(f"{k} {v:.0f}" for k, v in ref.splits.items())
        f2 = _font(P(LABEL_PX))
        tw2 = d.textlength(label, font=f2) if label else 0
        bw = max(tw, tw2) + P(40)
        cx = width - P(MARGIN) - bw / 2                   # top right, the panel's mirror: centred it met the panel
        d.rectangle((cx - bw / 2, P(MARGIN), cx + bw / 2, P(MARGIN + CLOCK_PX + LABEL_PX + 26)), fill=BG)
        d.text((cx - tw / 2, P(MARGIN + 6)), clock, fill=INK, font=f)
        if label:
            d.text((cx - tw2 / 2, P(MARGIN + CLOCK_PX + 12)), label, fill=DIM, font=f2)
        return np.asarray(im)


class HudVideo(ReplayVideo):
    def __init__(self, *args, hud: Hud, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hud = hud

    def _write_frames(self, count):
        self._renderer.update(self._env.sim.data)
        frame = self._hud.paint(self._renderer.render())
        for _ in range(count):
            self._writer.send(frame)
            self.frame_count += 1
