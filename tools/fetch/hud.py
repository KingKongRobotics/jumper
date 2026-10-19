"""The sensor overlay for the film: what the crab knows, drawn once per video frame
beside the depth inset. Everything on it is read from the robot's own sensors as
`sense.py` reads them (the IMU's up-vector and gyro, the joint encoders, the claw's
finger) plus the autopilot's state and the pad it is driving -- no ground truth.

`draw(path, info)` writes one PNG; `tools/fetch/compose.py` lays the frames over a
take. `info` keys: t, state, mode, yaw (rad, gyro-integrated), up (3, base frame),
q (22 joint angles, rad, by name), names, aperture_mm, pad (lx ly rx ry lt rt hat_x
hat_y buttons), note (one line, free).
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

W, H = 440, 420
INK = (235, 235, 235)
DIM = (120, 120, 120)
ACCENT = (230, 90, 70)
TEAL = (80, 200, 190)
BG = (18, 18, 22, 215)

_FONT = {}


def _font(size: int):
    if size not in _FONT:
        for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf", "LiberationMono-Regular.ttf"):
            try:
                _FONT[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _FONT[size] = ImageFont.load_default()
    return _FONT[size]


def draw(path: str, info: dict) -> None:
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)
    f12, f14, f18 = _font(12), _font(14), _font(18)
    d.text((12, 8), f"{info.get('state', '?'):<12}", fill=ACCENT, font=f18)
    d.text((180, 10), f"mode {info.get('mode', '?')}", fill=INK, font=f14)
    d.text((360, 10), f"{info.get('t', 0.0):6.1f}s", fill=DIM, font=f14)

    # IMU: a tilt bubble (the up-vector's x, y in the base frame) and the gyro heading.
    cx, cy, r = 70, 95, 44
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=DIM, width=1)
    d.line((cx - r, cy, cx + r, cy), fill=DIM); d.line((cx, cy - r, cx, cy + r), fill=DIM)
    up = info.get("up", (0.0, 0.0, 1.0))
    bx, by = cx - up[1] * r * 3.0, cy - up[0] * r * 3.0        # nose down -> bubble up the screen
    d.ellipse((bx - 6, by - 6, bx + 6, by + 6), fill=TEAL)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
    d.text((cx - r - 10, cy + r + 4), f"tilt {tilt:4.1f} deg", fill=INK, font=f12)
    hx, hy = 190, 95
    d.ellipse((hx - r, hy - r, hx + r, hy + r), outline=DIM, width=1)
    yaw = info.get("yaw", 0.0)
    d.line((hx, hy, hx + r * math.sin(-yaw) * 0.9, hy - r * math.cos(yaw) * 0.9), fill=ACCENT, width=3)
    d.text((hx - r, hy + r + 4), f"yaw {math.degrees(yaw):+6.1f}", fill=INK, font=f12)

    # The claw and the pad.
    ap = info.get("aperture_mm")
    d.text((260, 60), "claw", fill=DIM, font=f12)
    if ap is not None:
        d.rectangle((260, 76, 420, 92), outline=DIM)
        d.rectangle((260, 76, 260 + max(0, min(160, ap / 60.0 * 160)), 92), fill=TEAL)
        d.text((260, 96), f"{ap:5.1f} mm open", fill=INK, font=f12)
    pad = info.get("pad", {})
    d.text((260, 118), "pad", fill=DIM, font=f12)
    for i, (lab, key) in enumerate((("L", ("lx", "ly")), ("R", ("rx", "ry")))):
        sx, sy = 285 + i * 70, 158
        d.ellipse((sx - 22, sy - 22, sx + 22, sy + 22), outline=DIM)
        px, py = sx + pad.get(key[0], 0.0) * 20, sy + pad.get(key[1], 0.0) * 20
        d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=ACCENT)
        d.text((sx - 4, sy + 24), lab, fill=DIM, font=f12)
    pressed = sorted(pad.get("buttons", ())) + ([f"hat({pad.get('hat_x', 0):+d},{pad.get('hat_y', 0):+d})"]
                                                 if pad.get("hat_x") or pad.get("hat_y") else [])
    d.text((260, 196), " ".join(pressed)[:24], fill=ACCENT, font=f12)
    lt = pad.get("lt", 0.0)
    d.rectangle((260, 214, 420, 222), outline=DIM); d.rectangle((260, 214, 260 + 160 * lt, 222), fill=ACCENT)
    d.text((260, 224), f"trigger {lt:.2f}", fill=INK, font=f12)

    # The 22 joint encoders as a bar strip, by leg.
    q, names = info.get("q"), info.get("names", [])
    y0 = 250
    d.text((12, y0 - 2), "joint encoders (rad)", fill=DIM, font=f12)
    if q is not None:
        n = len(q); bw = (W - 24) / max(1, n)
        for i, v in enumerate(q):
            x = 12 + i * bw
            mid = y0 + 90
            top = mid - max(-70, min(70, v / 3.0 * 70))
            col = TEAL if "F_J" in names[i] else INK
            d.rectangle((x + 1, min(mid, top), x + bw - 1, max(mid, top)), fill=col)
        d.line((12, y0 + 90, W - 12, y0 + 90), fill=DIM)
        for leg, i in (("LF", 0), ("RF", 5), ("LM", 10), ("RM", 13), ("LR", 16), ("RR", 19)):
            d.text((12 + i * bw, y0 + 2 + 10), leg, fill=DIM, font=f12)
    note = info.get("note", "")
    if note:
        d.text((12, H - 22), note[:60], fill=INK, font=f12)
    im.save(path)
