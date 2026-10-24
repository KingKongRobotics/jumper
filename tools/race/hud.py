"""The readout on a race take: what each crab sees, maps and plans, in its corner.

Andrew (2026-10-15): "the depth vision and cost map with the path planner visible
for each robot during this test because it will be interesting to see what
objects a robot sees and how each robot prioritizes object collection."

Each crab's panel (red top left, blue top right -- at the bottom they covered the
near half of the rink, Andrew: "the map/depth needs to be at the top") is the
obstacle course's (`wide_panel`, shared with `tools/course/hud.py`): its intent,
its depth image as the sensor reads it (grey by range, red where the pilot's senses
call something an object, yellow the kerb, magenta the other crab, blue the crab
itself), its map (`mapper.Mapper`: dark occupied, light free, grey unknown, the
plan in cyan, the object it is after in yellow, the line in white), and a column of
instruments: the IMU's horizon, the two sticks, the legs' joints. Between the two
panels: the count on each half and the clock. Painted into the frames as they are
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


def depth_image(frame, extra=()) -> Image.Image:
    """The range image, grey by range (log-scaled over the sensor's reach), with what
    the sensing made of each zone: the crab's own legs and claws, the things, the
    kerb, the other crab (Andrew: "shade the claws one color, the objects another and
    the walls a third")."""
    r = frame["range"]
    img = np.zeros(r.shape + (3,), dtype=np.uint8)
    valid = r > 0
    g = np.clip(255 * (1 - np.log(np.clip(r, 0.05, 8.8) / 0.05) / np.log(8.8 / 0.05)), 0, 255).astype(np.uint8)
    img[..., 0] = img[..., 1] = img[..., 2] = np.where(valid, g, 0)
    for key, colour in (("kerb_px", KERB_C), ("crab_px", CRAB_C), ("thing_px", THING_C), ("self", SELF_C)) + tuple(extra):
        if key in frame and frame[key] is not None:               # `extra`: (key, colour) pairs a caller adds (the course's ball)
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


# The panel's layout, in 720p pixels, scaled with the frame (Andrew: "the map and depth camera
# are not aligned vertically and the L and R controls are just out on their own ... this needs
# to be tightened up and balanced properly"): two lines of text, then one row -- the depth
# camera and the map at the same height, no captions (Andrew: "remove the legend and the map
# and plan text ... make the depth and map a little bigger"), and a column of instruments at
# the right: the IMU's horizon, the two stick dials, the legs' joints as bars -- in one box.
PAD = 14                          # the panel's inner margin
IMG_H = 170                       # the depth image's and the map's height; the depth image keeps its aspect
DIAL_R, DIAL_GAP = 17, 12         # the stick dials' radius and the gap between them
COL_W = 150                       # the instruments' column
HORIZON_R = 24                    # the IMU dial's radius
LEG_BAR_H, LEG_BAR_RAD = 50, 0.5  # the joint bars' full height, and the deviation from the standing pose (rad) that fills it


def _joints(pilot):
    """The legs' joints, as deviations from the standing pose in radians, or None."""
    eyes = getattr(pilot, "eyes", None)
    me = getattr(eyes, "me", None) or getattr(getattr(eyes, "sight", None), "robot", None) or \
        getattr(getattr(pilot, "senses", None), "robot", None)            # (the race's pilot has senses, no eyes)
    try:
        q = me.data.joint_pos[0].cpu().numpy()
        q0 = me.data.default_joint_pos[0].cpu().numpy()
        return (q - q0).astype(float)
    except (AttributeError, TypeError, IndexError):
        return None


def _attitude(pilot):
    """The body's pitch and roll from the IMU's up, radians (nose up and left side up positive), or None."""
    sight = getattr(pilot, "senses", None)
    try:
        up = sight.up()
        return math.atan2(up[0], up[2]), math.atan2(up[1], up[2])
    except (AttributeError, TypeError, IndexError):
        return None


def wide_panel(name: str, mode: str, intent: str, pad, senses, map_im, scale: float = 1.0, pilot=None, depth=None) -> Image.Image:
    """One crab's panel: two lines of text, the depth camera and the map (`map_im`, a
    square map image or None) side by side, the instruments' column. `depth` draws the
    depth image (the course's tints its ball); `scale` is the frame's size over 720p."""
    depth = depth or depth_image
    def P(v):
        return round(v * scale)
    depth_w = round(IMG_H * DEPTH_W / DEPTH_H)
    map_w = IMG_H
    pw = PAD + depth_w + PAD + map_w + PAD + COL_W + PAD
    row_y = 6 + 20 + 20 + 8
    ph = row_y + IMG_H + 10
    im = Image.new("RGBA", (P(pw), P(ph)), BG)
    d = ImageDraw.Draw(im)
    f11, f12, f16 = _font(P(11)), _font(P(12)), _font(P(16))
    accent = COLOURS.get(name, INK)
    d.rectangle((0, 0, P(5), P(ph)), fill=accent)
    d.text((P(PAD), P(6)), f"{name.upper()}  {mode}", fill=accent, font=f16)
    d.text((P(PAD), P(26)), intent[:70], fill=INK, font=f12)
    # the depth camera
    x0, y0 = P(PAD), P(row_y)
    frame = getattr(senses, "last", None) if senses is not None else None
    if frame is not None:
        im.paste(depth(frame).resize((P(depth_w), P(IMG_H)), Image.NEAREST), (x0, y0))
    else:
        d.rectangle((x0, y0, x0 + P(depth_w), y0 + P(IMG_H)), outline=DIM)
        d.text((x0 + P(30), y0 + P(60)), "no senses" if senses is None else "no frame", fill=DIM, font=f12)
    # the map, in a square beside it
    mx = x0 + P(depth_w + PAD)
    if map_im is not None:
        m = map_im
        k = min(P(map_w) / m.width, P(IMG_H) / m.height)
        m = m.resize((max(1, round(m.width * k)), max(1, round(m.height * k))), Image.NEAREST)
        im.paste(m, (mx + (P(map_w) - m.width) // 2, y0 + (P(IMG_H) - m.height) // 2))
    else:
        d.rectangle((mx, y0, mx + P(map_w), y0 + P(IMG_H)), outline=DIM)
    # the instruments' column: the horizon at the top, the sticks in the middle, the legs below
    cx0 = mx + P(map_w + PAD)
    ccx = cx0 + P(COL_W) // 2
    # the IMU: a horizon dial
    hr = P(HORIZON_R)
    hcy = y0 + hr + P(2)
    d.ellipse((ccx - hr, hcy - hr, ccx + hr, hcy + hr), outline=DIM, width=max(1, P(1)))
    att = _attitude(pilot)
    if att is not None:
        pitch, roll = att
        dy = max(-hr * 0.8, min(hr * 0.8, -pitch / math.radians(30.0) * hr))   # nose up: the horizon drops
        c, sn = math.cos(-roll), math.sin(-roll)
        ex, ey = (hr - P(3)) * c, (hr - P(3)) * sn
        d.line((ccx - ex, hcy + dy - ey, ccx + ex, hcy + dy + ey), fill=accent, width=P(2))
        d.text((ccx + hr + P(6), hcy - P(14)), f"p {math.degrees(pitch):+.0f}", fill=DIM, font=f11)
        d.text((ccx + hr + P(6), hcy + P(1)), f"r {math.degrees(roll):+.0f}", fill=DIM, font=f11)
    d.line((ccx - P(5), hcy, ccx + P(5), hcy), fill=INK, width=max(1, P(1)))
    d.text((ccx - hr, hcy - hr - P(13)), "imu", fill=DIM, font=f11)
    # the sticks
    lx, ly, rx, ry = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lx", "ly", "rx", "ry"))
    buttons = sorted(pad.buttons) if pad is not None else []
    r = P(DIAL_R)
    scy = hcy + hr + P(14) + r + P(8)
    for i, (lab, x, y) in enumerate((("L", lx, ly), ("R", rx, ry))):
        cx = ccx - P(DIAL_R + DIAL_GAP // 2) + i * P(2 * DIAL_R + DIAL_GAP)
        d.ellipse((cx - r, scy - r, cx + r, scy + r), outline=DIM, width=max(1, P(1)))
        if x or y:
            d.line((cx, scy, cx + x * r, scy + y * r), fill=accent, width=P(2))
        pxx, pyy = cx + x * r, scy + y * r
        d.ellipse((pxx - P(3), pyy - P(3), pxx + P(3), pyy + P(3)), fill=accent if (x or y) else DIM)
        d.text((cx - P(4), scy + r + P(2)), lab, fill=DIM, font=f11)
    if buttons:
        d.text((cx0, scy - r - P(13)), " ".join(buttons)[:12], fill=accent, font=f11)
    # the legs: one bar a joint, the standing pose the midline, grouped by leg
    q = _joints(pilot)
    by1 = y0 + P(IMG_H)
    by0 = by1 - P(LEG_BAR_H)
    mid = (by0 + by1) // 2
    d.text((cx0, by0 - P(13)), "legs", fill=DIM, font=f11)
    d.line((cx0, mid, cx0 + P(COL_W) - P(4), mid), fill=DIM, width=max(1, P(1)))
    if q is not None and len(q):
        n = len(q)
        per_leg = 3 if n % 3 == 0 else n
        legs = n // per_leg
        gap, bw, lgap = P(2), max(P(3), (P(COL_W) - P(4) - P(2) * (n - 1) - P(4) * (legs - 1)) // n), P(4)
        x = cx0
        for i, v in enumerate(q):
            h = max(-1.0, min(1.0, float(v) / LEG_BAR_RAD)) * (P(LEG_BAR_H) // 2 - P(2))
            d.rectangle((x, min(mid, mid - h), x + bw, max(mid, mid - h)), fill=accent)
            x += bw + gap + (lgap if (i + 1) % per_leg == 0 else 0)
    return im


PANEL_K = 0.78                     # both crabs' panels at the course's layout, scaled to fit a frame side by side
CLOCK_PX, SCORE_PX, LABEL_PX = 40, 22, 14   # the clock's, the score's and the label's type, in 720p pixels


class Hud:
    """`crabs` is a list of (name, pad, player, pilot) in the order red, blue.

    Each crab's panel is the obstacle course's (`wide_panel`: the depth camera, the
    map and plan, the IMU, the sticks and the legs), red in the top left and blue in
    the top right, scaled by PANEL_K to leave the clock room between them (Andrew,
    2026-10-23: "Like we had it for the obstacle course except we display both crabs
    at the same time ... We should also make the clock a little bigger"). Sizes are in
    720p pixels and scale with the frame's height, so a 4K take reads the same."""

    def __init__(self, crabs, referee, scale: float | None = None) -> None:
        self.crabs = crabs
        self.referee = referee
        self.scale = scale                              # None: from the frame's height, 720p being 1

    def paint(self, frame: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(frame))
        width = frame.shape[1]
        s = self.scale if self.scale is not None else frame.shape[0] / 720.0

        def P(v):
            return round(v * s)
        for i, (name, pad, player, pilot) in enumerate(self.crabs):
            fsm = player.fsm
            mode = fsm.mode + ("" if fsm.running_policy else " (ramping)")
            st = getattr(pilot, "state", None) or {}
            intent = st.get("intent") or ("-" if pilot is not None else "keys")
            if st.get("stage"):
                intent = f"{intent}: {st['stage']}"
            mapper = getattr(pilot, "mapper", None)
            m = None if mapper is None else map_image(mapper, st.get("target_w"), st.get("line_w"), COLOURS.get(name, INK),
                                                       st.get("things_w", ()), st.get("foe_w"))
            p = wide_panel(name, mode, intent, getattr(pad, "last", None), getattr(pilot, "senses", None), m,
                           scale=s * PANEL_K, pilot=pilot)
            x = P(MARGIN) if i == 0 else width - p.width - P(MARGIN)
            im.paste(p, (x, P(MARGIN)), p)
        # between the panels: the score, the clock under it, big, and what the referee is doing
        ref = self.referee
        tally = ref.score()
        d = ImageDraw.Draw(im, "RGBA")
        fs, fc, fl = _font(P(SCORE_PX)), _font(P(CLOCK_PX)), _font(P(LABEL_PX))
        score = f"RED {tally[RED]}   {tally[BLUE]} BLUE"
        clock = {"ready": "on the marks", "race": f"{ref.clock(ref.t):5.1f} s", "linger": "time",
                 "victory": "the dance"}.get(ref.state, "")
        label = f"{tally[None]} on the line" if tally[None] else ""
        widths = [d.textlength(score, font=fs), d.textlength(clock, font=fc), d.textlength(label, font=fl) if label else 0]
        bw = max(widths) + P(32)
        top, cx = P(MARGIN), width / 2
        bottom = top + P(8 + SCORE_PX + 6 + CLOCK_PX + 8) + (P(LABEL_PX + 6) if label else 0)
        d.rectangle((cx - bw / 2, top, cx + bw / 2, bottom), fill=BG)
        y = top + P(6)
        d.text((cx - widths[0] / 2, y), score, fill=INK, font=fs)
        y += P(SCORE_PX + 6)
        d.text((cx - widths[1] / 2, y), clock, fill=INK, font=fc)
        if label:
            d.text((cx - widths[2] / 2, y + P(CLOCK_PX + 6)), label, fill=DIM, font=fl)
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
