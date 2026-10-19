"""The readout on a race take: what each crab sees, maps and plans, in its corner.

Andrew (2026-10-15): "the depth vision and cost map with the path planner visible
for each robot during this test because it will be interesting to see what
objects a robot sees and how each robot prioritizes object collection."

Each crab's panel (red top left, blue top right -- at the bottom they covered the
near half of the rink, Andrew: "the map/depth needs to be at the top") shows its depth image as
the sensor reads it (grey by range, red where the pilot's senses call something
an object, blue where they call it the crab itself), its map (`mapper.Mapper`:
dark occupied, light free, grey unknown, the plan in cyan, the object it is after
in yellow, the line in white), its intent, and the pad it is driving. Across the
top: the count on each half and the clock. Painted into the frames as they are
written (`HudVideo`, as `tools/battle/hud.py` does), from the pilot's own state
(`pilot.state`, `pilot.senses`, `pilot.mapper`) and nothing the referee knows,
bar the score.
"""
from __future__ import annotations

import math

import numpy as np
from arena import BLUE, RED
from room import CHAMFER, ROOM_X, ROOM_Y
from mjrl.viewer.video import ReplayVideo
from PIL import Image, ImageDraw, ImageFont

W, H = 400, 206                      # one panel
MARGIN = 16
INK = (235, 235, 235)
DIM = (130, 130, 130)
BG = (16, 16, 20, 190)
COLOURS = {"red": (235, 70, 55), "blue": (70, 120, 245)}
DEPTH_W, DEPTH_H = 162, 126          # the 54 x 42 image at 3 px a zone
MAP_PX, MAP_SPAN = 160, 2.4          # the map inset: this many px across, this many metres

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


SELF_C, THING_C, KERB_C, CRAB_C = (70, 110, 255), (255, 70, 60), (255, 200, 40), (220, 70, 230)
LEGEND = (("me", SELF_C), ("things", THING_C), ("wall", KERB_C), ("crab", CRAB_C))


def depth_image(frame) -> Image.Image:
    """The range image, grey by range (log-scaled over the sensor's reach), with what
    the sensing made of each zone: the crab's own legs and claws, the things, the
    kerb, the other crab (Andrew: "shade the claws one color, the objects another and
    the walls a third")."""
    r = frame["range"]
    img = np.zeros(r.shape + (3,), dtype=np.uint8)
    valid = r > 0
    g = np.clip(255 * (1 - np.log(np.clip(r, 0.05, 8.8) / 0.05) / np.log(8.8 / 0.05)), 0, 255).astype(np.uint8)
    img[..., 0] = img[..., 1] = img[..., 2] = np.where(valid, g, 0)
    for key, colour in (("kerb_px", KERB_C), ("crab_px", CRAB_C), ("thing_px", THING_C), ("self", SELF_C)):
        if key in frame:
            img[frame[key]] = colour
    return Image.fromarray(img).resize((DEPTH_W, DEPTH_H), Image.NEAREST)


def map_image(mapper, target=None, line=None, accent=INK, things=(), foe=None) -> Image.Image:
    """The map round the base: the room the crab was given (kerb and chamfers) in
    white, what the sensor has mapped lately in grey, the plan in cyan, the
    remembered things in yellow with the target ringed, the other crab in magenta."""
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

    hx, hy, c = ROOM_X / 2, ROOM_Y / 2, CHAMFER
    outline = [(-hx + c, -hy), (hx - c, -hy), (hx, -hy + c), (hx, hy - c), (hx - c, hy), (-hx + c, hy), (-hx, hy - c), (-hx, -hy + c)]
    d.line([px(*q) for q in outline + outline[:1]], fill=(250, 250, 250), width=2)
    if line is not None:
        d.line([px(*line[0]), px(*line[1])], fill=(250, 250, 250), width=2)
    if foe is not None:
        fx, fy = px(*foe)
        d.ellipse([fx - 4, fy - 4, fx + 4, fy + 4], fill=CRAB_C)
    if len(mapper.path) > 1:
        d.line([px(*p) for p in mapper.path], fill=(0, 200, 220), width=2)
    for tx, ty in things:
        x, y = px(tx, ty)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 220, 0))
    if target is not None:
        x, y = px(*target)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 220, 0), width=2)
    x, y = px(mapper.x, mapper.y)
    hx, hy = px(mapper.x + 0.25 * math.cos(mapper.yaw), mapper.y + 0.25 * math.sin(mapper.yaw))
    d.line([(x, y), (hx, hy)], fill=accent, width=3)
    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=accent)
    return im


def panel(name: str, mode: str, intent: str, pad, senses, mapper, target, line, things=(), foe=None) -> Image.Image:
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)
    f12, f16 = _font(12), _font(16)
    accent = COLOURS.get(name, INK)
    d.rectangle((0, 0, 5, H), fill=accent)
    d.text((14, 6), f"{name.upper()}  {mode}", fill=accent, font=f16)
    d.text((14, 26), intent[:52], fill=INK, font=f12)
    x0, y0 = 14, 46
    frame = getattr(senses, "last", None) if senses is not None else None
    if frame is not None:
        im.paste(depth_image(frame), (x0, y0))
        lx_ = x0
        for word, colour in LEGEND:
            d.rectangle((lx_, y0 + DEPTH_H + 5, lx_ + 7, y0 + DEPTH_H + 12), fill=colour)
            d.text((lx_ + 10, y0 + DEPTH_H + 2), word, fill=DIM, font=f12)
            lx_ += 12 + int(d.textlength(word, font=f12)) + 6
    else:
        d.rectangle((x0, y0, x0 + DEPTH_W, y0 + DEPTH_H), outline=DIM)
        d.text((x0 + 30, y0 + 55), "no senses" if senses is None else "no frame", fill=DIM, font=f12)
    mx, my = x0 + DEPTH_W + 12, y0
    if mapper is not None:
        m = map_image(mapper, target, line, accent, things, foe)
        im.paste(m, (mx, my))
        d.text((mx, my + m.height + 2), "map + plan", fill=DIM, font=f12)
    else:
        d.rectangle((mx, my, mx + MAP_PX, my + MAP_PX * 0.66), outline=DIM)
        d.text((mx + 60, my + 50), "no map", fill=DIM, font=f12)
    # the pad: two stick rings and the buttons, in the panel's lower right
    lx, ly, rx, ry = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lx", "ly", "rx", "ry"))
    buttons = sorted(pad.buttons) if pad is not None else []
    for i, (lab, x, y) in enumerate((("L", lx, ly), ("R", rx, ry))):
        cx, cy, r = W - 70 + i * 36 - 18, H - 30, 13
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=DIM, width=1)
        if x or y:
            d.line((cx, cy, cx + x * r, cy + y * r), fill=accent, width=2)
        pxx, pyy = cx + x * r, cy + y * r
        d.ellipse((pxx - 3, pyy - 3, pxx + 3, pyy + 3), fill=accent if (x or y) else DIM)
        d.text((cx - 4, cy + r + 2), lab, fill=DIM, font=f12)
    d.text((W - 110, H - 62), " ".join(buttons)[:14], fill=accent, font=f12)
    return im


class Hud:
    """`crabs` is a list of (name, pad, player, pilot) in the order red, blue."""

    def __init__(self, crabs, referee) -> None:
        self.crabs = crabs
        self.referee = referee

    def paint(self, frame: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(frame))
        width = frame.shape[1]
        for i, (name, pad, player, pilot) in enumerate(self.crabs):
            fsm = player.fsm
            mode = fsm.mode + ("" if fsm.running_policy else " (ramping)")
            st = getattr(pilot, "state", None) or {}
            intent = st.get("intent") or ("-" if pilot is not None else "keys")
            if st.get("stage"):
                intent = f"{intent}: {st['stage']}"
            p = panel(name, mode, intent, getattr(pad, "last", None), getattr(pilot, "senses", None),
                      getattr(pilot, "mapper", None), st.get("target_w"), st.get("line_w"), st.get("things_w", ()),
                      st.get("foe_w"))
            x = MARGIN if i == 0 else width - W - MARGIN
            im.paste(p, (x, MARGIN), p)
        ref = self.referee
        tally = ref.score()
        d = ImageDraw.Draw(im, "RGBA")
        f = _font(22)
        score = f"RED {tally[RED]}   {tally[BLUE]} BLUE"
        tw = d.textlength(score, font=f)
        d.rectangle((width / 2 - tw / 2 - 16, 12, width / 2 + tw / 2 + 16, 44), fill=BG)
        d.text((width / 2 - tw / 2, 16), score, fill=INK, font=f)
        clock = ref.clock(ref.t)
        label = {"ready": "on the marks", "race": f"{clock:5.1f} s", "linger": "time", "victory": "the dance"}.get(ref.state, "")
        if tally[None]:
            label += f"   ({tally[None]} on the line)"
        f2 = _font(14)
        tw2 = d.textlength(label, font=f2)
        d.rectangle((width / 2 - tw2 / 2 - 10, 46, width / 2 + tw2 / 2 + 10, 66), fill=BG)
        d.text((width / 2 - tw2 / 2, 48), label, fill=DIM, font=f2)
        return np.asarray(im)


class HudVideo(ReplayVideo):
    """`ReplayVideo` with the readout painted into every frame it writes."""

    def __init__(self, *args, hud: Hud, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hud = hud

    def _write_frames(self, count):
        self._renderer.update(self._env.sim.data)
        frame = self._hud.paint(self._renderer.render())
        for _ in range(count):
            self._writer.send(frame)
            self.frame_count += 1
