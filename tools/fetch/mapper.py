"""An occupancy map of the room from the depth camera, as a lidar: each frame's object
returns mark cells occupied and the rays to them (and to the floor) mark cells free,
in a world frame anchored where the crab woke up. The heading is the gyro's (integrated
about the world's up); the position is dead-reckoned from the pad's sticks and pulled
back onto the map each frame by matching the scan against what is already there
(a small search over offsets, the heading trusted). The map answers what the rules
could not: is the way behind clear before backing, has this pillar been passed, and
which way round the obstacles to a goal (A* on the inflated grid).

Measured on the room walk (w72, 2026-10-14): see `step`'s docstring for the speed
model the dead reckoning starts from; the scan match takes it the rest of the way.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from sense import COLS, FLOOR_CLEAR, ROWS, TOF_ORIGIN

RES = 0.05                 # m per cell
SIZE = 20.0                # m across; the wake pose sits at the centre (14 put the edge 1 m past the
                           # tray; stalled against a pillar the stick-reckoned pose ran 1.4 m ahead of
                           # the body and walked off the grid, film72)
N = int(round(SIZE / RES))
HIT, MISS = 0.9, 0.25      # log-odds steps per frame
CLAMP = 6.0
OCC, FREE = 1.0, -1.0      # thresholds on the log-odds
MAP_RANGE = 3.0            # returns past this are not mapped (the picture's edge is noisy)
MAP_NEAR = 0.45            # nor inside this: the carried mug hangs 0.25-0.35 m ahead and, mapped, it
                           # smeared a trail of "obstacles" that stopped the retreat 9 cm behind (w82)
MAP_SELF = 0.35            # cells within this of the base are freed every step: the body stands there, and
                           # the specks the legs and the swinging mug left round the crab at the tray read
                           # as "0.30 m behind" and stopped the retreat (film58)
INFLATE = 0.40             # m: half the body plus clearance, for planning and corridors (0.28 -> 0.33 with the
                           # pillar's face mapped and its sides unknown, the path passed 27 cm from a pillar's
                           # centre and the crab wedged on it, carry10; 0.40 since)
                           # walk allowed 17 deg off the path, film42)
SPEED_FWD = 0.42           # m/s per unit of walk stick (0.5 -> 0.21 measured, CREEP_SPEED)
SPEED_SIDE = 0.35          # m/s per unit of strafe stick (0.5 -> ~0.17 m/s, the sidestep tests)
DEAD_BAND = 0.40           # a stick under this moves nothing
MATCH_WIN, MATCH_STEP = 0.15, 0.05   # the scan match's search, m
STALL_COLS, STALL_M, STALL_S = 8, 0.01, 1.0   # the stall test: this many columns' nearest objects moved less than
                                           # STALL_M between frames (0.2 s at 0.2 m/s is 4 cm) for STALL_S
MATCH_GAIN = 0.5           # how much of the best offset is applied per frame


class Mapper:
    def __init__(self):
        self.grid = np.zeros((N, N), dtype=np.float32)      # log-odds, row = y cell, col = x cell
        self.x, self.y = 0.0, 0.0                            # the base in the wake frame
        self.yaw = 0.0
        self.last_t = None
        self.last_frame_t = None
        self.marks = {}                                      # named world points (the flag, the mug)
        self.fixed = set()                                   # ...of which these are known, not seen (`prior`)
        self.soft = {}                                       # mark name -> radius (m) the planner walks through
        self._cols = None                                    # the last frame's nearest object per column (stall test)
        self._still_t = None                                 # since when the scan has not changed under a walk stick
        self.stalled = False                                 # the scan still for STALL_S while walking: no reckoning
        self._carry = [0.0, 0.0]                             # a fix's shift not yet applied to the grid, m
        self.path = []                                       # the last plan, world xy
        self.inflate_m = INFLATE                             # the planner's clearance; a smaller room asks for less
        self.model = None                                    # (ly, lx) -> (v forward, v left) m/s, in place of the
                                                             # stick model below, for a mode measured differently
        self.match_win = MATCH_WIN                           # the scan match's search, m each way
        self.stall_test = True                               # off for a reckoning that knows when the body moves (the legs):
                                                             # a thing shoved along fills the picture and never changes, which
                                                             # this test reads as wedged, and the pose froze through every shove
        self.decay_s = None                                  # log-odds half-life toward unknown, s (None: never): a room
                                                             # of things that move is a map of where they were
        self._known = None                                   # cells `known` set, re-asserted through the decay

    # -- frames -------------------------------------------------------------------
    def cell(self, wx, wy):
        return int(round(wy / RES)) + N // 2, int(round(wx / RES)) + N // 2

    def world(self, r, c):
        return (c - N // 2) * RES, (r - N // 2) * RES

    def to_world(self, pts_b):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x = self.x + cy * pts_b[:, 0] - sy * pts_b[:, 1]
        y = self.y + sy * pts_b[:, 0] + cy * pts_b[:, 1]
        return x, y

    def to_body(self, wx, wy):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = wx - self.x, wy - self.y
        return cy * dx + sy * dy, -sy * dx + cy * dy

    # -- the step ----------------------------------------------------------------
    def prior(self, marks):
        """The staged scene's known points in the wake frame -- the tray's flag, the mug --
        placed on the map before the first step, as a robot starts with its dock on the
        map (Andrew, 2026-10-14: "pre-mark the flag on the cost map like a dock... the
        tray and the cup marked, and the other obstacles are the unknowns"). A sighting
        of a marked point then corrects the odometry (`fix`), never the mark."""
        for name, (x, y) in marks.items():
            self.marks[name] = (float(x), float(y))
            self.fixed.add(name)

    def known(self, segments, thick=0.0):
        """Walls the robot was given before it looked: each segment ((x0, y0), (x1, y1))
        in the wake frame is set occupied at the clamp, so the scan match has something
        to hold the pose against from the first frame. The race's kerb is the case
        (tools/race, 2026-10-15): with nothing known the dead-reckoned pose was 0.2-0.4 m
        off within 10 s and the things it remembered doubled up; the kerb is always in
        view and does not move, which is what a landmark is."""
        if self._known is None:
            self._known = np.zeros_like(self.grid, dtype=bool)
        for (x0, y0), (x1, y1) in segments:
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / (RES * 0.5)))
            for x, y in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)):
                r, c = self.cell(float(x), float(y))
                k = int(math.ceil(thick / RES))
                self.grid[max(0, r - k): min(N, r + k + 1), max(0, c - k): min(N, c + k + 1)] = CLAMP
                self._known[max(0, r - k): min(N, r + k + 1), max(0, c - k): min(N, c + k + 1)] = True

    def fix(self, name, bx, by, gain=0.5, max_m=1.5):
        """A landmark fix: the mark `name` seen at (bx, by) in the body frame. The pose
        moves by `gain` of the discrepancy, and the grid moves with it, so what was
        mapped from the drifted pose keeps its place against the body. Returns the
        discrepancy in metres, or None when it is past `max_m` (not this landmark)."""
        if name not in self.marks:
            return None
        sx, sy = self.to_world(np.array([[bx, by]]))
        dx, dy = self.marks[name][0] - float(sx[0]), self.marks[name][1] - float(sy[0])
        d = math.hypot(dx, dy)
        if d > max_m:
            return None
        self.shift(gain * dx, gain * dy)
        return d

    def shift(self, dx: float, dy: float) -> None:
        """Move the pose by (dx, dy) and the grid with it, so what was mapped from the
        drifted pose keeps its place against the body (a fix's second half, on its own
        for a fix that is not a mark's: the race's known walls, `tools/race`)."""
        self.x += dx
        self.y += dy
        self._carry[0] += dx
        self._carry[1] += dy
        # the grid shifts by whole cells; the remainder waits for the next fix
        sc, sr = int(round(self._carry[0] / RES)), int(round(self._carry[1] / RES))
        if sc or sr:
            self.grid = np.roll(self.grid, (sr, sc), axis=(0, 1))
            if self._known is not None:
                self._known = np.roll(self._known, (sr, sc), axis=(0, 1))
            if sr > 0:
                self.grid[:sr, :] = 0.0                      # (the wrapped edge is unknown again)
            elif sr < 0:
                self.grid[sr:, :] = 0.0
            if sc > 0:
                self.grid[:, :sc] = 0.0
            elif sc < 0:
                self.grid[:, sc:] = 0.0
            self._carry[0] -= sc * RES
            self._carry[1] -= sr * RES
            self.path = [(x + sc * RES, y + sr * RES) for x, y in self.path]

    def step(self, senses, t, ly, lx, scale=1.0):
        """Once per control step. `ly`/`lx` are the pad's sticks (ly < 0 walks forward,
        lx > 0 strafes right); `scale` is the mode's speed against the claw mode's
        (SPEED_FWD was measured there). Dead-reckons the pose, then on a fresh exposure
        matches the scan to the map and folds the frame in."""
        if self.last_t is not None:
            dt = t - self.last_t
            if self.model is not None:
                v, vy = self.model(ly, lx)
            else:
                v = -SPEED_FWD * scale * ly if abs(ly) >= DEAD_BAND else 0.0
                vy = -SPEED_SIDE * scale * lx if abs(lx) >= DEAD_BAND else 0.0
            if self.stalled:
                v = vy = 0.0                                 # (wedged: the stick moves nothing)
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            self.x += (cy * v - sy * vy) * dt
            self.y += (sy * v + cy * vy) * dt
        if self.decay_s and self.last_t is not None:
            self.grid *= float(np.exp(-math.log(2.0) * (t - self.last_t) / self.decay_s))
            if self._known is not None:
                self.grid[self._known] = CLAMP
        self.last_t = t
        self.yaw = senses.yaw
        # the body's own cells, freed every step (before the frame test: a step without a
        # fresh exposure still moves the base)
        k = int(math.ceil(MAP_SELF / RES))
        r0, c0 = self.cell(self.x, self.y)
        rr, cc = np.ogrid[-k:k + 1, -k:k + 1]
        disk = (rr * rr + cc * cc) * RES * RES <= MAP_SELF * MAP_SELF
        rs, cs = slice(max(0, r0 - k), min(N, r0 + k + 1)), slice(max(0, c0 - k), min(N, c0 + k + 1))
        if rs.stop > rs.start and cs.stop > cs.start:      # (off the grid: an empty slice and a mask that
            sub = self.grid[rs, cs]                        # does not match raised IndexError and ended film72)
            d = disk[k - (r0 - rs.start): k + (rs.stop - r0), k - (c0 - cs.start): k + (cs.stop - c0)]
            sub[d] = np.minimum(sub[d], FREE - 0.5)
        self.yaw = senses.yaw
        f = senses.last
        if f is None or f["t"] == self.last_frame_t:
            return
        self.last_frame_t = f["t"]
        r = f["range"].ravel()
        pts, h = f["pts"], f["h"]
        valid = (r > 0) & ~f["self"].ravel() & (pts[:, 0] > 0.10) & (r < MAP_RANGE) & (r > MAP_NEAR)
        obj = valid & (h > FLOOR_CLEAR)
        # The stall test: the stick says walk and the scan does not change. Wedged on a
        # pillar the stick-reckoned pose ran 0.6 m ahead in 10 s and the approach "arrived"
        # 0.8 m from the tray (carry10). Judged on the nearest object in each column, with
        # at least STALL_COLS of them in view: an empty picture says nothing.
        cols = np.where(obj, pts[:, 0], np.inf).reshape(ROWS, COLS).min(axis=0)
        walking = abs(ly) >= DEAD_BAND or abs(lx) >= DEAD_BAND
        if self._cols is not None and walking and self.stall_test:
            both = np.isfinite(cols) & np.isfinite(self._cols)
            if both.sum() >= STALL_COLS and float(np.median(np.abs(cols[both] - self._cols[both]))) < STALL_M:
                if self._still_t is None:
                    self._still_t = t
            else:
                self._still_t = None
        else:
            self._still_t = None
        self.stalled = self._still_t is not None and t - self._still_t > STALL_S
        self._cols = cols
        floor = valid & (h <= FLOOR_CLEAR)
        # the scan match: object points against the map's occupied cells
        if obj.sum() >= 20 and (self.grid > OCC).sum() >= 40:
            ox, oy = self.to_world(pts[obj])
            best, best_s = (0.0, 0.0), -1
            base = None
            for dx in np.arange(-self.match_win, self.match_win + 1e-6, MATCH_STEP):
                for dy in np.arange(-self.match_win, self.match_win + 1e-6, MATCH_STEP):
                    rr = np.clip(np.round((oy + dy) / RES).astype(int) + N // 2, 0, N - 1)
                    cc = np.clip(np.round((ox + dx) / RES).astype(int) + N // 2, 0, N - 1)
                    sc = int((self.grid[rr, cc] > OCC).sum())
                    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                        base = sc
                    if sc > best_s or (sc == best_s and dx * dx + dy * dy < best[0] ** 2 + best[1] ** 2):
                        best, best_s = (float(dx), float(dy)), sc
            if base is not None and best_s >= base + max(4, 0.15 * obj.sum()):
                self.x += MATCH_GAIN * best[0]
                self.y += MATCH_GAIN * best[1]
        # fold the frame in: free rays to each column's nearest object (or farthest floor), hits at the objects
        ex, ey = self.to_world(TOF_ORIGIN[None, :2])
        ex, ey = float(ex[0]), float(ey[0])
        x_img = np.where(obj, pts[:, 0], np.inf).reshape(ROWS, COLS)
        near_i = x_img.argmin(axis=0)
        near_x = x_img[near_i, np.arange(COLS)]
        fl_img = np.where(floor, pts[:, 0], -np.inf).reshape(ROWS, COLS)
        far_i = fl_img.argmax(axis=0)
        far_x = fl_img[far_i, np.arange(COLS)]
        ends = []
        for c in range(COLS):
            if np.isfinite(near_x[c]):
                ends.append(near_i[c] * COLS + c)
            elif np.isfinite(far_x[c]):
                ends.append(far_i[c] * COLS + c)
        if ends:
            wx, wy = self.to_world(pts[ends])
            for px, py in zip(wx, wy):
                n = max(2, int(math.hypot(px - ex, py - ey) / (RES * 0.7)))
                xs = np.linspace(ex, px, n, endpoint=False)[1:]
                ys = np.linspace(ey, py, n, endpoint=False)[1:]
                rr = np.clip(np.round(ys / RES).astype(int) + N // 2, 0, N - 1)
                cc = np.clip(np.round(xs / RES).astype(int) + N // 2, 0, N - 1)
                self.grid[rr, cc] -= MISS
        if obj.any():
            ox, oy = self.to_world(pts[obj])
            rr = np.clip(np.round(oy / RES).astype(int) + N // 2, 0, N - 1)
            cc = np.clip(np.round(ox / RES).astype(int) + N // 2, 0, N - 1)
            np.add.at(self.grid, (rr, cc), HIT)
        np.clip(self.grid, -CLAMP, CLAMP, out=self.grid)

    # -- queries -----------------------------------------------------------------
    def mark(self, name, bx, by):
        """Remember a point seen in the body frame under a name, in world coordinates."""
        wx, wy = self.to_world(np.array([[bx, by]]))
        self.marks[name] = (float(wx[0]), float(wy[0]))

    def mark_body(self, name):
        """A remembered point, in the body frame now (or None)."""
        if name not in self.marks:
            return None
        return self.to_body(*self.marks[name])

    def occupied_cells(self):
        return self.grid > OCC

    def inflated(self):
        """The occupied cells grown by INFLATE, for planning. Cells within `soft[name]`
        of a mark are not obstacles: the mug the crab is walking to is a thing to reach,
        never a thing to go round (Andrew, 2026-10-14: the map "so that it cannot
        possibly confuse the object for an obstacle")."""
        from scipy import ndimage
        occ = self.grid > OCC
        for name, radius in self.soft.items():
            if name in self.marks:
                r0, c0 = self.cell(*self.marks[name])
                k = int(math.ceil(radius / RES))
                rr, cc = np.ogrid[-k:k + 1, -k:k + 1]
                d = (rr * rr + cc * cc) * RES * RES <= radius * radius
                rs, cs = slice(max(0, r0 - k), min(N, r0 + k + 1)), slice(max(0, c0 - k), min(N, c0 + k + 1))
                if rs.stop > rs.start and cs.stop > cs.start:
                    sub = occ[rs, cs]
                    sub[d[k - (r0 - rs.start): k + (rs.stop - r0), k - (c0 - cs.start): k + (cs.stop - c0)]] = False
        k = int(math.ceil(self.inflate_m / RES))
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        disk = (xx * xx + yy * yy) * RES * RES <= self.inflate_m * self.inflate_m
        return ndimage.binary_dilation(occ, structure=disk)

    def clearance(self, heading_b, length, half_width=0.25):
        """The nearest occupied cell inside a corridor from the base along the body-frame
        bearing `heading_b` (0 ahead, pi behind), in metres, or `length` if clear."""
        occ = self.grid > OCC
        rr, cc = np.nonzero(occ)
        if len(rr) == 0:
            return length
        wx, wy = self.world(rr, cc)
        bx, by = self.to_body(wx, wy)
        ch, sh = math.cos(heading_b), math.sin(heading_b)
        along = bx * ch + by * sh
        across = -bx * sh + by * ch
        sel = (along > 0.0) & (along < length) & (np.abs(across) < half_width)
        return float(along[sel].min()) if sel.any() else length

    def plan(self, goal_w, inflate=None):
        """A* over the inflated grid from the base to `goal_w` (world xy); unknown cells
        count as free. Returns the path as world xy (empty when unreachable)."""
        blocked = self.inflated() if inflate is None else inflate
        sr, sc = self.cell(self.x, self.y)
        gr, gc = self.cell(*goal_w)
        sr, sc, gr, gc = (min(max(v, 0), N - 1) for v in (sr, sc, gr, gc))
        blocked[sr, sc] = False
        if blocked[gr, gc]:
            # the goal inside an obstacle's inflation (the flag's pole): take the nearest free cell
            rr, cc = np.nonzero(~blocked)
            k = int(np.argmin((rr - gr) ** 2 + (cc - gc) ** 2))
            gr, gc = int(rr[k]), int(cc[k])
        came = {}
        g = {(sr, sc): 0.0}
        pq = [(0.0, (sr, sc))]
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        found = False
        n_pop = 0
        while pq and n_pop < 60000:
            _, cur = heapq.heappop(pq)
            n_pop += 1
            if cur == (gr, gc):
                found = True
                break
            for dr, dc, w in steps:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < N and 0 <= nc < N) or blocked[nr, nc]:
                    continue
                ng = g[cur] + w
                if ng < g.get((nr, nc), 1e9):
                    g[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    heapq.heappush(pq, (ng + math.hypot(nr - gr, nc - gc), (nr, nc)))
        if not found:
            self.path = []
            return []
        cells = [(gr, gc)]
        while cells[-1] != (sr, sc):
            cells.append(came[cells[-1]])
        cells.reverse()
        self.path = [self.world(r, c) for r, c in cells]
        return self.path

    def next_point(self, goal_w, lookahead=0.5):
        """Plan to `goal_w` and return the world point about `lookahead` metres along the
        path (None when no path). Planning costs ~20 ms; call it a few times a second
        and steer at the returned point in between (`bearing_to`)."""
        path = self.plan(goal_w)
        if not path:
            return None
        d = 0.0
        tx, ty = path[-1]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            d += math.hypot(x1 - x0, y1 - y0)
            if d >= lookahead:
                tx, ty = x1, y1
                break
        return tx, ty

    def bearing_to(self, wx, wy):
        bx, by = self.to_body(wx, wy)
        return math.atan2(by, bx)

    def next_bearing(self, goal_w, lookahead=0.5):
        pt = self.next_point(goal_w, lookahead)
        return None if pt is None else self.bearing_to(*pt)

    # -- pictures ----------------------------------------------------------------
    def picture(self, path, span=4.0):
        """A PNG of the map around the base: dark = occupied, light = free, grey = unknown,
        the base as a red arrow, marks in yellow, the plan in cyan."""
        from PIL import Image, ImageDraw
        k = int(span / RES)
        r0, c0 = self.cell(self.x, self.y)
        r0, c0 = min(max(r0, k), N - k - 1), min(max(c0, k), N - k - 1)
        sub = self.grid[r0 - k: r0 + k, c0 - k: c0 + k]
        img = np.full(sub.shape + (3,), 128, dtype=np.uint8)
        img[sub > OCC] = (30, 30, 30)
        img[sub < FREE] = (220, 215, 200)
        im = Image.fromarray(img[::-1]).resize((sub.shape[1] * 3, sub.shape[0] * 3), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        def px(wx, wy):
            r, c = self.cell(wx, wy)
            return (c - (c0 - k)) * 3 + 1, (2 * k - 1 - (r - (r0 - k))) * 3 + 1
        if len(self.path) > 1:
            dr.line([px(*p) for p in self.path], fill=(0, 200, 220), width=2)
        for name, (wx, wy) in self.marks.items():
            x, y = px(wx, wy)
            dr.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 220, 0), width=2)
            dr.text((x + 6, y - 6), name, fill=(255, 220, 0))
        x, y = px(self.x, self.y)
        hx, hy = px(self.x + 0.3 * math.cos(self.yaw), self.y + 0.3 * math.sin(self.yaw))
        dr.line([(x, y), (hx, hy)], fill=(230, 40, 40), width=3)
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 40, 40))
        im.save(path)
