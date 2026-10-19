"""Who fell, who won, the set pieces around each round, and the reset between.

A round is lost by the crab that is **flipped** -- tilted past `FLIP_DEG`, on
its head or its back, for `down_s` -- or that is **off the plank and on the
sand**: its base below `LANDED_Z`, which a standing crab is twice as high as
and a falling one passes within a tenth of a second of landing, so the fall
itself is still in the shot when the round is called. A crab knocked over
short of a flip, on its side with the controller dropped to `safe`, cannot get
up; it loses as **down and out** once its tilt has been past the controller's
own limit for `out_s`, which keeps a round from running its clock out over a
limp crab. (A crab knocked flat onto the plank keeps its base 3 cm above the
wood, so a height rule alone cannot tell a fall off the plank from a fall on
it; the base below the plank's top surface is a fall in progress and not yet a
loss.) Both down in the same step is a draw and the round is fought again.

Around the fight are the set pieces (`moves.py`): the crabs taunt each other
on their marks before the bell, the round's winner cheers while the loser lies
where it fell, and the match's winner dances. The referee cues them (`cue`: a
colour -> piece dict, with `cue_id` ticking on every change, so a pilot plays
each cue once) and waits for the crabs to finish, `busy` as the duel reports it
per crab, with a ceiling per phase since a crab on the keys has nobody to
report for it. The fall plays out for `linger_s` before the cheer -- the winner
walks back to the middle of the plank meanwhile, and one that went over with
the loser (the shove carries the shover: in two of the first three brain
takes the winner was on the sand too) is stood back up in the middle of the
plank, the loser left where it fell -- the world is reset after it, and the
crabs stand on their marks for `ready_s` while the controllers settle. First
to a majority of `best_of` wins the match. A cheer and the dance are played
to the camera: `camera_yaw` is the heading that faces it (the duel works it
out from the recorder's azimuth), a crab stood back up is stood facing it,
and a pilot turns its crab to it before it presses the piece.

The pilots read `live` (the round is on), `fight_since` and `cue`, and nothing
else: the referee is the only thing that resets the world.
"""
from __future__ import annotations

import math

from arena import BLUE, PLANK_TOP, RED
from arena import margin as arena_margin

#: Tilt past which a crab is flipped (on its head, or its back) rather than knocked over.
FLIP_DEG = 90.0
#: Plank a performer needs on every side of it: the third brain take's winner
#: stood upright at the plank's end, was left there, and its dance went off it.
PERFORM_MARGIN = 0.25
#: Base height under which a crab is on the ground: the stance is 0.107 m, the
#: plank's top 0.25, so 0.16 is within 5 cm of the sand and below any crab still
#: standing on the plank.
LANDED_Z = 0.16


class SegmentDone(Exception):
    """The match is over; `duel.py` ends the take on it."""


class Referee:
    def __init__(self, env, *, tilt_limit: float, best_of: int = 3, linger_s: float = 3.0,
                 ready_s: float = 2.0, round_s: float = 45.0, down_s: float = 0.5, out_s: float = 3.0,
                 plank_top: float = PLANK_TOP, moves: bool = True, taunt_max_s: float = 16.0,
                 cheer_max_s: float = 12.0, victory_max_s: float = 24.0, camera_yaw: float | None = None) -> None:
        self.env = getattr(env, "unwrapped", env)
        self.crabs = {"red": self.env.scene[RED], "blue": self.env.scene[BLUE]}
        self.best_of = best_of
        self.need = best_of // 2 + 1
        self.linger_s, self.ready_s, self.round_s = linger_s, ready_s, round_s
        self.down_s, self.out_s = down_s, out_s
        self.tilt_limit = tilt_limit          # radians, the app's `fsm.tilt_limit`
        self.plank_top = plank_top
        self.moves = moves
        self.camera_yaw = camera_yaw          # radians, the heading that faces the camera; None for the mark's
        self.ceiling = {"taunt": taunt_max_s, "cheer": cheer_max_s, "victory": victory_max_s}
        self._tilted_since: dict[str, float | None] = {"red": None, "blue": None}
        self._flipped_since: dict[str, float | None] = {"red": None, "blue": None}
        self.score = {"red": 0, "blue": 0}
        self.round = 1
        self.history: list[tuple[int, str, str]] = []     # (round, winner or "draw", why)
        self.state = "ready"
        self.fight_since = 0.0
        self._since = 0.0
        self._over = ""
        self._winner = ""
        self.cue: dict[str, str] = {}
        self.cue_id = 0

    @property
    def live(self) -> bool:
        return self.state == "fight"

    def down(self, name: str, t: float) -> str | None:
        """Why `name` has lost the round, or None while it is still in it."""
        data = self.crabs[name].data
        z = float(data.root_link_pos_w[0, 2])
        g = data.projected_gravity_b[0]
        tilt = math.acos(max(-1.0, min(1.0, -float(g[2]))))
        if z < LANDED_Z:
            return f"off the plank and on the sand (base at {z:.2f} m)"
        flipped = self._persisted(self._flipped_since, name, tilt > math.radians(FLIP_DEG), t)
        if flipped is not None and flipped >= self.down_s:
            return f"flipped ({math.degrees(tilt):.0f} deg for {flipped:.1f} s)"
        over = self._persisted(self._tilted_since, name, tilt > self.tilt_limit, t)
        if over is not None and over >= self.out_s:
            return f"down and out ({math.degrees(tilt):.0f} deg for {over:.1f} s)"
        return None

    @staticmethod
    def _persisted(clocks: dict, name: str, condition: bool, t: float) -> float | None:
        """How long `condition` has held for `name`, or None; the clock starts on
        the first step it is seen, so a single step past a limit never scores."""
        if not condition:
            clocks[name] = None
            return None
        if clocks[name] is None:
            clocks[name] = t
        return t - clocks[name]

    def standing(self, name: str) -> bool:
        """Upright on the plank with `PERFORM_MARGIN` of it on every side: fit to perform."""
        data = self.crabs[name].data
        g = data.projected_gravity_b[0]
        tilt = math.acos(max(-1.0, min(1.0, -float(g[2]))))
        x, y, z = (float(v) for v in data.root_link_pos_w[0])
        room = arena_margin(x, y)
        return z > self.plank_top and tilt < self.tilt_limit and room >= PERFORM_MARGIN

    def restore(self, name: str) -> None:
        """Stand `name` in the middle of the plank, facing the camera (or at its
        mark's heading, without one), still."""
        import torch

        crab = self.crabs[name]
        root = crab.data.default_root_state[:1].clone()
        root[0, 0:2] = 0.0
        if self.camera_yaw is not None:
            root[0, 3:7] = torch.tensor([math.cos(self.camera_yaw / 2), 0.0, 0.0, math.sin(self.camera_yaw / 2)])
        root[0, 7:] = 0.0
        ids = torch.tensor([0], device=root.device)
        crab.write_root_state_to_sim(root, ids)
        pos = crab.data.default_joint_pos[:1]
        crab.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=ids)
        print(f"\n[referee] {name} is stood back up on the plank")

    def _perform(self, phase: str, t: float, name: str) -> None:
        if not self.standing(name):
            self.restore(name)
        self._phase(phase, t, {name: phase})

    def _cue(self, cue: dict[str, str]) -> None:
        self.cue = cue
        self.cue_id += 1

    def _phase(self, state: str, t: float, cue: dict[str, str] | None = None) -> None:
        self.state, self._since = state, t
        self._cue(cue or {})

    def _done(self, phase: str, t: float, busy: dict[str, bool]) -> bool:
        """A set-piece phase is over when nobody cued is still busy (after a
        second, for the press to register) or at the phase's ceiling."""
        since = t - self._since
        if since >= self.ceiling[phase]:
            return True
        return since >= 1.0 and not any(busy.get(name, False) for name in self.cue)

    def step(self, t: float, busy: dict[str, bool] | None = None) -> None:
        busy = busy or {}
        if self.state == "ready":
            if t - self._since >= self.ready_s:
                if self.moves:
                    print(f"\n[referee] round {self.round}: the stare-down")
                    self._phase("taunt", t, {"red": "taunt", "blue": "taunt"})
                else:
                    self._fight(t)
        elif self.state == "taunt":
            if self._done("taunt", t, busy):
                self._fight(t)
        elif self.state == "fight":
            why = {name: self.down(name, t) for name in self.crabs}
            losers = [name for name, reason in why.items() if reason]
            if len(losers) == 1:
                loser = losers[0]
                winner = "blue" if loser == "red" else "red"
                self.score[winner] += 1
                self._end(t, winner, f"{loser} {why[loser]} at {t - self.fight_since:.1f} s")
            elif losers:
                self._end(t, "draw", "both down at once: " + "; ".join(f"{n} {why[n]}" for n in losers))
            elif t - self.fight_since > self.round_s:
                self._end(t, "draw", f"nobody down in {self.round_s:g} s")
        elif self.state == "linger" and t - self._since >= self.linger_s:
            if self._over and self.moves and self._winner:
                print(f"\n[referee] {self._winner} dances")
                self._perform("victory", t, self._winner)
            elif self._over:
                self._finish()
            elif self.moves and self._winner:
                self._perform("cheer", t, self._winner)
            else:
                self._next_round(t)
        elif self.state == "cheer":
            if self._done("cheer", t, busy):
                self._next_round(t)
        elif self.state == "victory" and self._done("victory", t, busy):
            self._finish()

    def _fight(self, t: float) -> None:
        self.fight_since = t
        self._tilted_since = {name: None for name in self.crabs}
        self._flipped_since = {name: None for name in self.crabs}
        print(f"\n[referee] round {self.round}: fight!  (red {self.score['red']} - {self.score['blue']} blue)")
        self._phase("fight", t)

    def _next_round(self, t: float) -> None:
        self.env.reset()
        self.round += 1
        self._phase("ready", t)

    def _finish(self) -> None:
        self._phase("over", 0.0)
        raise SegmentDone(self._over)

    def _end(self, t: float, winner: str, why: str) -> None:
        self.history.append((self.round, winner, why))
        word = "a draw" if winner == "draw" else f"{winner} takes it"
        print(f"\n[referee] round {self.round}: {word} -- {why}.  red {self.score['red']} - {self.score['blue']} blue")
        self._winner = "" if winner == "draw" else winner
        self._phase("linger", t)
        champion = next((n for n, s in self.score.items() if s >= self.need), None)
        if champion:
            self._over = f"{champion} wins the match {self.score['red']}-{self.score['blue']}"
        elif self.round >= 2 * self.best_of:
            self._over = f"no winner in {self.round} rounds ({self.score['red']}-{self.score['blue']})"
            self._winner = ""
