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

import numpy as np
from mjrl.viewer.video import ReplayVideo
from PIL import Image, ImageDraw, ImageFont

W, H = 420, 176                      # one panel, in 720p pixels (scaled with the frame's height)
MARGIN = 20
INK = (235, 235, 235)
DIM = (130, 130, 130)
GRID = (60, 60, 64)
BG = (16, 16, 20, 190)
SCORE_BG = (16, 16, 20, 225)          # the score's box, nearly opaque: the pennant line runs behind it
COLOURS = {"red": (235, 70, 55), "blue": (70, 120, 245)}
LEGS = ("LF", "RF", "LM", "RM", "LR", "RR")   # the joint bars' groups, in the order drawn
LEG_BAR_H, LEG_BAR_RAD = 44, 0.6     # the joint bars' full height, and the deviation (rad) that fills it
SCORE_PX, LABEL_PX = 54, 20          # the score's and the round label's type, in 720p pixels

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


def joints(env, entity: str):
    """The crab's joints as {leg: [deviation from the standing pose, rad]}, or None."""
    try:
        robot = env.scene[entity]
        q = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).cpu().numpy()
        names = list(robot.joint_names)
    except (AttributeError, KeyError, TypeError, IndexError):
        return None
    out = {leg: [] for leg in LEGS}
    for n, v in zip(names, q):
        if n[:2] in out:
            out[n[:2]].append(float(v))
    return out


def panel(name: str, style: str, mode: str, intent: str, pad, legs=None, scale: float = 1.0) -> Image.Image:
    """One crab's readout: `pad` is a `GamepadState` (or None for hands off), `legs`
    what `joints` returns. The control surfaces on the left -- the two sticks, the
    triggers, the hat, the buttons held -- and the joints on the right, one bar a
    joint, the standing pose the midline, grouped by leg (the front legs' two extra
    are the claw arms)."""
    def P(v):
        return round(v * scale)
    im = Image.new("RGBA", (P(W), P(H)), BG)
    d = ImageDraw.Draw(im)
    f12, f14, f18 = _font(P(12)), _font(P(14)), _font(P(18))
    lw = max(1, P(1))
    accent = COLOURS.get(name, INK)
    d.rectangle((0, 0, P(5), P(H)), fill=accent)
    d.text((P(14), P(6)), f"{name.upper()} {style}", fill=accent, font=f18)
    d.text((P(14), P(28)), f"{mode}  {intent}"[:52], fill=DIM, font=f12)
    lx, ly, rx, ry = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lx", "ly", "rx", "ry"))
    lt, rt = (getattr(pad, k, 0.0) if pad is not None else 0.0 for k in ("lt", "rt"))
    hat_x, hat_y = (getattr(pad, k, 0) if pad is not None else 0 for k in ("hat_x", "hat_y"))
    buttons = sorted(pad.buttons) if pad is not None else []
    # the sticks: a dot in a ring, the ring's rim being full deflection
    r = P(26)
    for i, (lab, x, y) in enumerate((("L", lx, ly), ("R", rx, ry))):
        cx, cy = P(44) + i * P(66), P(84)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=DIM, width=lw)
        d.line((cx - r, cy, cx + r, cy), fill=GRID, width=lw)
        d.line((cx, cy - r, cx, cy + r), fill=GRID, width=lw)
        if x or y:
            d.line((cx, cy, cx + x * r, cy + y * r), fill=accent, width=P(2))
        px, py = cx + x * r, cy + y * r
        d.ellipse((px - P(5), py - P(5), px + P(5), py + P(5)), fill=accent if (x or y) else DIM)
        d.text((cx - P(4), cy + r + P(2)), lab, fill=DIM, font=f12)
    # the triggers, the hat and the buttons under the sticks
    for i, (lab, v) in enumerate((("LT", lt), ("RT", rt))):
        x0, y = P(14) + i * P(80), P(134)
        d.text((x0, y - P(2)), lab, fill=DIM, font=f12)
        d.rectangle((x0 + P(20), y, x0 + P(70), y + P(8)), outline=DIM, width=lw)
        if v > 0:
            d.rectangle((x0 + P(20), y, x0 + P(20) + P(50) * v, y + P(8)), fill=accent)
    hx, hy = P(28), P(158)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        on = (dx and dx == hat_x) or (dy and dy == hat_y)
        d.rectangle((hx + dx * P(8) - P(3), hy + dy * P(8) - P(3), hx + dx * P(8) + P(3), hy + dy * P(8) + P(3)),
                    fill=accent if on else GRID)
    d.text((P(48), P(150)), " ".join(buttons)[:14], fill=accent, font=f14)
    # the joints
    x0, x1 = P(176), P(W - 12)
    by0 = P(64)
    d.line((x0 - P(10), P(50), x0 - P(10), P(H - 10)), fill=GRID, width=lw)
    d.text((x0, P(48)), "joints", fill=DIM, font=f12)
    gw = (x1 - x0) // 3
    for k, leg in enumerate(LEGS):
        col, row = k // 2, k % 2                  # left legs on the top row, right on the bottom
        gx = x0 + col * gw
        top = by0 + row * (P(LEG_BAR_H) + P(14))
        mid = top + P(LEG_BAR_H) // 2
        d.text((gx, top + P(LEG_BAR_H) // 2 - P(7)), leg, fill=DIM, font=f12)
        bx0 = gx + P(22)
        d.line((bx0, mid, gx + gw - P(6), mid), fill=GRID, width=lw)
        qs = (legs or {}).get(leg) or []
        if not qs:
            continue
        bw = max(P(3), (gx + gw - P(6) - bx0 - P(2) * (len(qs) - 1)) // len(qs))
        for j, v in enumerate(qs):
            h = max(-1.0, min(1.0, v / LEG_BAR_RAD)) * (P(LEG_BAR_H) // 2 - P(2))
            bx = bx0 + j * (bw + P(2))
            d.rectangle((bx, min(mid, mid - h), bx + bw, max(mid, mid - h)), fill=accent)
    return im


class Hud:
    """Paints the readouts into a frame. `crabs` is a list of (name, style, pad,
    player, pilot, entity) in the order red, blue; `referee` gives the score.

    Andrew (2026-10-24), for the 4K take: "just the control surfaces and joints. Red
    on the bottom left and blue on the bottom right. Like you find in the hud1
    attempt ... make sure the score is nice and visible as well in the top center."
    Sizes are in 720p pixels and scale with the frame's height, so a 4K take reads
    as a 720p one does (hud1's were fixed pixels: a third the size at 4K)."""

    def __init__(self, crabs, referee, env=None, scale: float | None = None) -> None:
        self.crabs = crabs
        self.referee = referee
        self.env = env
        self.scale = scale                              # None: from the frame's height, 720p being 1

    def paint(self, frame: np.ndarray) -> np.ndarray:
        im = Image.fromarray(np.ascontiguousarray(frame))
        height, width = frame.shape[:2]
        s = self.scale if self.scale is not None else height / 720.0

        def P(v):
            return round(v * s)
        for i, (name, style, pad, player, pilot, entity) in enumerate(self.crabs):
            fsm = player.fsm
            mode = fsm.mode + ("" if fsm.running_policy else " (ramping)")
            st = getattr(pilot, "state", None) or {}
            intent = st.get("intent") or ("-" if pilot is not None else "keys")
            stage = st.get("stage") or ""
            if st.get("lock"):
                intent = f"{intent} / lock {st['lock'][0]}"
            elif stage:
                intent = f"{intent} / {stage}"
            legs = joints(self.env, entity) if self.env is not None else None
            p = panel(name, style, mode, intent, getattr(pad, "last", None), legs, scale=s)
            x = P(MARGIN) if i == 0 else width - p.width - P(MARGIN)
            im.paste(p, (x, height - p.height - P(MARGIN)), p)
        # the score, top centre: the two counts big in their crabs' colours, the round under them
        ref = self.referee
        d = ImageDraw.Draw(im, "RGBA")
        f, fl = _font(P(SCORE_PX)), _font(P(LABEL_PX))
        parts = (("RED ", COLOURS["red"]), (f"{ref.score['red']}", INK), ("  -  ", DIM),
                 (f"{ref.score['blue']}", INK), (" BLUE", COLOURS["blue"]))
        tw = sum(d.textlength(t, font=f) for t, _ in parts)
        label = {"fight": f"round {ref.round}", "taunt": "the stare-down", "ready": f"round {ref.round + 1}",
                 "victory": "the dance"}.get(ref.state, "")
        label = f"best of {ref.best_of}" + (f"   {label}" if label else "")
        tl = d.textlength(label, font=fl)
        bw = max(tw, tl) + P(48)
        top, bot = P(MARGIN), P(MARGIN + SCORE_PX + LABEL_PX + 28)
        d.rounded_rectangle((width / 2 - bw / 2, top, width / 2 + bw / 2, bot), radius=P(10), fill=SCORE_BG)
        x = width / 2 - tw / 2
        for t, c in parts:
            d.text((x, top + P(6)), t, fill=c, font=f)
            x += d.textlength(t, font=f)
        d.text((width / 2 - tl / 2, top + P(SCORE_PX + 14)), label, fill=DIM, font=fl)
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
    import sys

    from controller import GamepadState

    height = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    frame = np.full((height, height * 16 // 9, 3), 90, dtype=np.uint8)

    from types import SimpleNamespace as NS

    ref = NS(score={"red": 1, "blue": 0}, round=2, state="fight", best_of=5)
    player = NS(fsm=NS(mode="locomotion", running_policy=True))
    pad = NS(last=GamepadState(lx=0.3, ly=-0.8, rx=0.5, lt=0.7, hat_x=-1, buttons=frozenset({"L3", "menu"})))
    pilot = NS(state={"intent": "charge", "stage": "", "lock": None})
    import torch

    names = [f"{leg}_J{j}_joint" for leg in LEGS for j in range(5 if leg[1] == "F" else 3)]
    q = torch.tensor([[0.4 * np.sin(1.7 * i) for i in range(len(names))]])
    robot = NS(joint_names=names, data=NS(joint_pos=q, default_joint_pos=torch.zeros_like(q)))
    env = NS(scene={"robot": robot, "robot2": robot})
    hud = Hud([("red", "bull", pad, player, pilot, "robot"), ("blue", "matador", pad, player, None, "robot2")], ref, env)
    Image.fromarray(hud.paint(frame)).save(path)


if __name__ == "__main__":
    import sys

    preview(sys.argv[1] if len(sys.argv) > 1 else "hud_preview.png")
