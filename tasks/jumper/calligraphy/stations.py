"""Cut each stroke into stretches the arm can write without the trunk moving.

The arm writes inside a band (`arm.ReachMap`), and a character is several times
larger than the band is thick. So the robot writes a stretch, lifts the brush,
walks, puts it down where it stopped and writes the next. Where a stretch ends is
a visible seam in the ink, so the planner makes stretches as long as the band
allows, and makes the robot stand where every point of the stretch is at least
`margin` inside the band -- the margin is what absorbs the trunk not arriving
exactly where it was sent, and drifting while it writes.

## The search

For one heading, a trunk position `b` can write points P when every `p - b`, in
the trunk's frame, lands on a cell of the band at least `margin` from its edge.
Starting from one point the set of such `b` is the band itself, mirrored; each
further point intersects it with its own copy. The stretch grows point by point
until the set is empty, and the position kept is the one furthest inside the band
over the whole stretch. That is done for each heading in `yaws`, and the heading
that writes furthest wins -- ties go to the heading nearest the previous one, so
the robot does not turn for nothing.

Positions are searched on the band's own grid, so a stretch is exact to a cell
(5 mm) and the margin covers that too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import ReachMap
from .hanzi import Plan

#: Planning looks at every PLAN_STEP-th sample: 5 mm, the band's own cell.
PLAN_STEP = 5
#: A remainder of at most this many planning points (20 mm) is folded into the
#: stretch before it at half the margin, when it fits there.
TAIL = 4


@dataclass
class Stretch:
    stroke: int
    #: Sample indices [start, end] into the stroke's `xy`, both written.
    start: int
    end: int
    #: Where the trunk stands, world xy, and its heading, rad.
    base: np.ndarray
    yaw: float
    #: How far inside the band the stretch's closest point is, metres.
    margin: float


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _grow(pts: np.ndarray, inside: np.ndarray, dist: np.ndarray, rm: ReachMap):
    """Longest prefix of `pts` (trunk-frame orientation, world offsets) that one
    trunk position fits. Returns (count, best position offset, its margin)."""
    cells = np.argwhere(inside)  # (n, 2) band cells, as indices
    ip = np.round(pts / rm.cell).astype(int)
    # Candidates: b such that ip[0] - b lands on a band cell.
    cand = ip[0] - cells
    count = 1
    for k in range(1, len(ip)):
        idx = ip[k] - cand
        good = (
            (idx[:, 0] >= 0) & (idx[:, 0] < inside.shape[0])
            & (idx[:, 1] >= 0) & (idx[:, 1] < inside.shape[1])
        )
        good[good] = inside[idx[good, 0], idx[good, 1]]
        if not good.any():
            break
        cand = cand[good]
        count = k + 1
    # The candidate furthest inside the band over the whole stretch.
    worst = np.full(len(cand), np.inf)
    for k in range(count):
        idx = ip[k] - cand
        worst = np.minimum(worst, dist[idx[:, 0], idx[:, 1]])
    best = int(np.argmax(worst))
    return count, cand[best], float(worst[best])


def cut_stroke(s, start: int, rm: ReachMap, margin: float, yaws, yaw_prev: float,
               dist: np.ndarray | None = None, tail: bool = True) -> list[Stretch]:
    """Stretches covering `s` from sample `start` to its end."""
    if dist is None:
        dist = rm.margin()
    inside = dist >= margin
    if not inside.any():
        raise ValueError(f"no cell of the band is {margin * 1000:.0f} mm inside it")
    origin = np.array([rm.x0, rm.y0])
    n = len(s.xy)
    sub = list(range(start, n, PLAN_STEP))
    if sub[-1] != n - 1:
        sub.append(n - 1)
    out: list[Stretch] = []
    i = 0  # index into `sub`
    while i < len(sub) - 1:
        best = None
        for yaw in sorted(yaws, key=lambda a: abs(a - yaw_prev)):
            R = _rot(yaw)
            # World points, rotated into the trunk's orientation, relative to the
            # band's origin: then p_trunk - origin = q - b' on the grid.
            q = (s.xy[sub[i:]] @ R) - origin
            count, b_idx, mgn = _grow(q, inside, dist, rm)
            # Ties within one sample go to the nearer heading (tried first).
            if best is None or count > best[0] + 1:
                best = (count, yaw, b_idx, mgn)
        count, yaw, b_idx, mgn = best
        left = len(sub) - i - count
        if tail and 0 < left <= TAIL:
            # A short tail would cost a walk and a seam for a few millimetres of
            # ink: take it into this stretch at half the margin if that fits.
            R = _rot(yaw)
            q = (s.xy[sub[i:]] @ R) - origin
            c2, b2, m2 = _grow(q, dist >= margin / 2, dist, rm)
            if c2 == len(sub) - i:
                count, b_idx, mgn = c2, b2, m2
        if count < 2:
            raise ValueError(
                f"stroke {s.index + 1} cannot be written from anywhere at sample {sub[i]}"
            )
        # b' (trunk-orientation frame) back to a world trunk position.
        base = _rot(yaw) @ (b_idx * rm.cell)
        j = i + count - 1
        out.append(Stretch(s.index, sub[i], sub[j], base, float(yaw), mgn))
        yaw_prev = yaw
        i = j  # the next stretch starts where this one ended: the seam
    return out


YAWS = tuple(np.deg2rad(a) for a in (0, -20, 20, -40, 40))


def plan_stretches(plan: Plan, rm: ReachMap, margin: float = 0.012,
                   yaws: tuple[float, ...] = YAWS) -> list[Stretch]:
    dist = rm.margin()
    out: list[Stretch] = []
    yaw_prev = 0.0
    for s in plan.strokes:
        out += cut_stroke(s, 0, rm, margin, yaws, yaw_prev, dist)
        yaw_prev = out[-1].yaw
    return out


def reachable_until(s, start: int, end: int, base: np.ndarray, yaw: float, rm: ReachMap,
                    margin: float, dist: np.ndarray | None = None) -> int:
    """The last sample in [start, end] up to which every sample is at least
    `margin` inside the band, with the trunk at `base`, `yaw` -- where it really
    is, rather than where it was sent. `start - 1` if not even the first one."""
    if dist is None:
        dist = rm.margin()
    c, sn = np.cos(yaw), np.sin(yaw)
    d = s.xy[start: end + 1] - base[:2]
    local = np.column_stack([c * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + c * d[:, 1]])
    ij = rm.index(local)
    ok = (
        (ij[:, 0] >= 0) & (ij[:, 0] < dist.shape[0])
        & (ij[:, 1] >= 0) & (ij[:, 1] < dist.shape[1])
    )
    ok[ok] = dist[ij[ok, 0], ij[ok, 1]] >= margin
    bad = np.flatnonzero(~ok)
    return end if len(bad) == 0 else start + int(bad[0]) - 1


#: Headings tried when every stroke has to be written from one place: finer than
#: `YAWS`, since a stroke that just fails at one heading often fits at the next.
FIT_YAWS = tuple(np.deg2rad(a) for a in range(-60, 61, 10))


def whole_strokes(plan: Plan, rm: ReachMap, margin: float,
                  yaws: tuple[float, ...] = FIT_YAWS) -> list[Stretch] | None:
    """One stretch per stroke -- each stroke written by the arm alone, the trunk
    still -- or None if some stroke does not fit the band whole."""
    dist = rm.margin()
    out: list[Stretch] = []
    yaw_prev = 0.0
    for s in plan.strokes:
        try:
            # No half-margin tail here: a stroke written whole needs the full
            # margin all along it, or the tail is where it fails -- 无's fourth
            # stroke fitted with 5 mm at its hook and split there four times.
            cut = cut_stroke(s, 0, rm, margin, yaws, yaw_prev, dist, tail=False)
        except ValueError:
            return None
        if len(cut) != 1:
            return None
        out += cut
        yaw_prev = cut[0].yaw
    return out


#: `fit_size` nudges the text by these fractions of a band cell before it gives a
#: size up: which sizes fit depends on where the strokes fall on the band's grid.
FIT_SHIFTS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))


def fit_size(make_plan, rm: ReachMap, margin: float, lo: float = 0.05, hi: float = 0.40,
             tol: float = 0.0025, yaws: tuple[float, ...] = FIT_YAWS):
    """The largest character size at which every stroke is written whole.

    A stroke cut into stretches is written in pieces with the robot walking in
    between, and every seam shows: the trunk never stops exactly where it was
    sent, and the second piece starts a few millimetres off the first. So the
    longest stroke sets the size -- it is fitted whole into the band -- and the rest
    of the character is scaled with it. `make_plan(size, shift)` builds the plan,
    moved by `shift` metres. Returns (size, plan, stretches).

    Sizes are tried from `hi` down, `tol` apart, each at the FIT_SHIFTS: whether a
    size fits is not monotonic in it, nor the same a fraction of a cell away. A
    bisection assumed it was and found 跳跳 9.1 cm, where 11.5 fits and 跳 alone
    fitted at 12.1 (whole_strokes: 跳 fits at 9, 10, 11.5 and 12.1 cm, not at 8,
    11 or 12; 2026-10-10).
    """
    for size in np.arange(hi, lo - 1e-9, -tol):
        for fx, fy in FIT_SHIFTS:
            plan = make_plan(float(size), (fx * rm.cell, fy * rm.cell))
            cut = whole_strokes(plan, rm, margin, yaws)
            if cut is not None:
                return float(size), plan, cut
    raise ValueError(f"not even a {lo * 100:.0f} cm character fits stroke by stroke")
