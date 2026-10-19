"""Who has more on their side when the clock runs out, and the set pieces around it.

The count is the referee's and only the referee's: it reads every object's true
position (`room.side_of`) and says, as they cross, which half each one is on. A
pilot that senses rather than cheats never reads it (`pilots/sweeper.py`).

Phases: `ready` (the crabs stand on their marks), `race` (the clock runs; it ends
at `race_s`, or early when every object is on one half), `linger` (the last shove
plays out), then `victory` for the winner (cued through `cue`, as the battle's
referee cues its set pieces, and the pilot turns to the camera first) and
`SegmentDone`. A tie is a tie: nobody dances.
"""
from __future__ import annotations

import math

from arena import BLUE, RED
from room import side_of


class SegmentDone(Exception):
    pass


class Referee:
    def __init__(self, env, objects: list[dict], *, ready_s: float = 2.0, race_s: float = 120.0,
                 linger_s: float = 2.0, moves: bool = True, camera_yaw: float | None = None,
                 tilt_limit: float = math.radians(50)) -> None:
        self.env = env
        self.names = [o["name"] for o in objects]
        self.kinds = {o["name"]: o["kind"] for o in objects}
        self.ready_s, self.race_s, self.linger_s = ready_s, race_s, linger_s
        self.moves = moves
        self.camera_yaw = camera_yaw
        self.tilt_limit = tilt_limit
        self.state = "ready"
        self.t = 0.0                         # the last time stepped (the readout's clock)
        self._since = 0.0
        self.race_since = 0.0
        self.cue: dict[str, str] = {}
        self.cue_id = 0
        self.winner: str | None = None
        self.sides: dict[str, str | None] = {}
        self.crossings: list[tuple[float, str, str | None, str | None]] = []
        self.ceiling = {"victory": 12.0}

    # -- what the referee reads -------------------------------------------------
    @property
    def live(self) -> bool:
        return self.state == "race"

    def clock(self, t: float) -> float:
        """Seconds left on the race clock (the whole clock before the start)."""
        if self.state == "ready":
            return self.race_s
        return max(0.0, self.race_s - (t - self.race_since)) if self.state == "race" else self._final_clock

    def positions(self) -> dict[str, tuple[float, float, float]]:
        out = {}
        for name in self.names:
            p = self.env.scene[name].data.root_link_pos_w[0]
            out[name] = (float(p[0]), float(p[1]), float(p[2]))
        return out

    def score(self) -> dict[str | None, int]:
        tally: dict[str | None, int] = {RED: 0, BLUE: 0, None: 0}
        for x, _, _ in self.positions().values():
            tally[side_of(x)] += 1
        return tally

    def _count(self, t: float) -> None:
        for name, (x, _, _) in self.positions().items():
            s = side_of(x)
            was = self.sides.get(name, s)
            if s != was:
                self.crossings.append((t, name, was, s))
                where = {RED: "the red half", BLUE: "the blue half", None: "the line"}
                tally = self.score()
                print(f"\n[referee] {t - self.race_since:6.1f}s  the {self.kinds[name]} ({name}) crosses to {where[s]}"
                      f"  --  red {tally[RED]} : {tally[BLUE]} blue, {tally[None]} on the line")
            self.sides[name] = s

    # -- the phases --------------------------------------------------------------
    def _cue(self, cue: dict[str, str]) -> None:
        self.cue = cue
        self.cue_id += 1

    def _phase(self, state: str, t: float, cue: dict[str, str] | None = None) -> None:
        self.state, self._since = state, t
        self._cue(cue or {})

    def step(self, t: float, busy: dict[str, bool] | None = None) -> None:
        busy = busy or {}
        self.t = t
        if self.state == "ready":
            self.sides = {name: side_of(x) for name, (x, _, _) in self.positions().items()}
            if t - self._since >= self.ready_s:
                self.race_since = t
                tally = self.score()
                self.start_tally = dict(tally)
                print(f"\n[referee] go!  red {tally[RED]} : {tally[BLUE]} blue, {self.race_s:g} s on the clock")
                self._phase("race", t)
        elif self.state == "race":
            self._count(t)
            tally = self.score()
            total = len(self.names)
            # swept: every object on one half, which had to have been short of that at the
            # start (a probe staging one object on one half is not a race already won)
            swept = total > 0 and any(tally[c] == total and self.start_tally[c] < total for c in (RED, BLUE))
            if t - self.race_since >= self.race_s or swept:
                self._final_clock = self.clock(t) if swept else 0.0
                self.winner = RED if tally[RED] > tally[BLUE] else (BLUE if tally[BLUE] > tally[RED] else None)
                word = {RED: "red wins", BLUE: "blue wins", None: "a tie"}[self.winner]
                why = "the room is swept" if swept else "time"
                print(f"\n[referee] {why}: {word}  --  red {tally[RED]} : {tally[BLUE]} blue, {tally[None]} on the line")
                self._phase("linger", t)
        elif self.state == "linger" and t - self._since >= self.linger_s:
            if self.moves and self.winner is not None:
                name = "red" if self.winner == RED else "blue"
                print(f"\n[referee] {name} dances")
                self._phase("victory", t, {name: "victory"})
            else:
                self._finish()
        elif self.state == "victory":
            since = t - self._since
            if since >= self.ceiling["victory"] or (since >= 1.0 and not any(busy.get(n, False) for n in self.cue)):
                self._finish()

    def _finish(self) -> None:
        tally = self.score()
        self._phase("over", 0.0)
        word = {RED: "red wins the race", BLUE: "blue wins the race", None: "the race is a tie"}[self.winner]
        raise SegmentDone(f"{word} {tally[RED]}-{tally[BLUE]} ({tally[None]} on the line)")
