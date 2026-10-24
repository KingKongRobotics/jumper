"""The clock, the sections and the finish: what the readout shows and when the
run is over.

The referee reads the crab's true position (the pilot never does, unless it is
the `--truth` control): the clock starts at the bell, each section's split is
taken as the base passes its start, the finish is the base past the line, and
after the crab has turned to the camera and danced (cued through `cue`, as the
battle's and the race's referees cue their set pieces) the run ends with
`SegmentDone`. A crab that goes over on the course is not reset -- it is on
camera -- but the run is called off after `TIMEOUT_S` wherever it is.
"""
from __future__ import annotations

import math

from arena import RED


class SegmentDone(Exception):
    pass


SECTIONS = ("stairs", "ramp", "forest", "pit", "leap", "ball", "finish")


class Referee:
    def __init__(self, env, layout: dict, *, crab: str = "red", ready_s: float = 2.0, moves: bool = True,
                 camera_yaw: float | None = None, timeout_s: float = 240.0) -> None:
        self.env = env
        self.L = layout
        self.crab = crab                     # whose dance the finish cues (blue1 ran the course and did not dance:
                                             # the cue said "red")
        self.ready_s, self.moves, self.camera_yaw, self.timeout_s = ready_s, moves, camera_yaw, timeout_s
        h = layout["hill"]
        # where each section starts, along the lane, for the splits and the readout
        self.starts = {"stairs": h["flight1"][0] - 0.3, "ramp": h["ramp2"][0] - 0.6, "forest": layout["forest"][0],
                       "pit": layout["ramp"][0], "leap": layout["plat2"][1] - 0.5, "ball": layout["plat2"][1] + 0.6,
                       "finish": layout["finish"]}
        self.blocks_in: dict[int, float] = {}
        self.state = "ready"
        self.t = 0.0
        self._since = 0.0
        self.go_t = 0.0
        self.section = "start"
        self.splits: dict[str, float] = {}
        self.cue: dict[str, str] = {}
        self.cue_id = 0
        self.finish_t: float | None = None
        self.ball_over_t: float | None = None     # when the last ball crossed
        self.balls_over: dict = {}                # ball index -> when it crossed
        self.ceiling = {"victory": 16.0}

    @property
    def live(self) -> bool:
        return self.state in ("run", "finish")

    def clock(self, t: float) -> float:
        if self.state == "ready":
            return 0.0
        if self.finish_t is not None:
            return self.finish_t - self.go_t
        return t - self.go_t

    def pose(self) -> tuple[float, float, float]:
        d = self.env.scene[RED].data
        p = d.root_link_pos_w[0]
        w, x, y, z = (float(v) for v in d.root_link_quat_w[0])
        return float(p[0]), float(p[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def ball(self, i: int = 0) -> tuple[float, float]:
        p = self.env.scene["ball" if i == 0 else f"ball{i}"].data.root_link_pos_w[0]
        return float(p[0]), float(p[1])

    def _cue(self, cue: dict[str, str]) -> None:
        self.cue = cue
        self.cue_id += 1

    def _phase(self, state: str, t: float, cue: dict[str, str] | None = None) -> None:
        self.state, self._since = state, t
        self._cue(cue or {})

    def step(self, t: float, busy: bool = False) -> None:
        self.t = t
        x, _y, _yaw = self.pose()
        if self.state == "ready":
            if t - self._since >= self.ready_s:
                self.go_t = t
                print("\n[referee] go!")
                self._phase("run", t)
            return
        if self.state == "run":
            for name in SECTIONS:
                if name not in self.splits and x >= self.starts[name]:
                    self.splits[name] = t - self.go_t
                    self.section = name
                    print(f"\n[referee] {t - self.go_t:6.1f}s  {name}")
            for i in range(len(self.L["balls"])):
                bx, _ = self.ball(i)
                if i not in self.balls_over and bx >= self.L["goal"]:
                    self.balls_over[i] = t - self.go_t
                    print(f"\n[referee] {t - self.go_t:6.1f}s  ball {i + 1} is over the finish line"
                          f" ({len(self.balls_over)} of {len(self.L['balls'])})")
                    if len(self.balls_over) == len(self.L["balls"]):
                        self.ball_over_t = t - self.go_t
            import track
            for i in range(len(self.L["blocks"])):
                if i not in self.blocks_in:
                    p = self.env.scene[f"block{i}"].data.root_link_pos_w[0]
                    px0, px1 = self.L["pit"]
                    if px0 - 0.05 < float(p[0]) < px1 + 0.05 and float(p[2]) < track.PLAT_H - 0.02:
                        self.blocks_in[i] = t - self.go_t
                        print(f"\n[referee] {t - self.go_t:6.1f}s  block {i + 1} is in the pit"
                              f" ({'the bridge is built' if len(self.blocks_in) == len(self.L['blocks']) else 'one to go'})")
            if x >= self.L["finish"] and len(self.balls_over) == len(self.L["balls"]):
                # the course is complete only once every ball is over: a crab past the line with
                # balls to fetch is not finished (Andrew: "push all three across the finish line
                # before they can complete the course themselves")
                self.finish_t = t
                print(f"\n[referee] {t - self.go_t:6.1f}s  FINISH" + ("" if self.ball_over_t is not None else
                                                                        f" ({len(self.balls_over)} of {len(self.L['balls'])} balls over)")
                      + ("" if len(self.blocks_in) == len(self.L["blocks"]) else f" ({len(self.blocks_in)} block(s) in the pit)"))
                self._phase("finish", t, {self.crab: "victory"} if self.moves else {})
                if not self.moves:
                    self._finish()
            elif t - self.go_t > self.timeout_s:
                print(f"\n[referee] {t - self.go_t:6.1f}s  time: the run is called off at x {x:.1f}")
                self._finish()
        elif self.state == "finish":
            since = t - self._since
            if since >= self.ceiling["victory"] or (since >= 1.5 and not busy):
                self._finish()

    def _finish(self) -> None:
        self._phase("over", 0.0)
        splits = ", ".join(f"{k} {v:.1f}" for k, v in self.splits.items())
        total = f"{self.finish_t - self.go_t:.1f} s" if self.finish_t is not None else "no finish"
        raise SegmentDone(f"{total}; splits: {splits}")
