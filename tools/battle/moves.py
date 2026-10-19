"""The crabs' set pieces: a taunt at the bell, a cheer after a round, a victory
dance after the match -- each colour its own -- played through the app's own
modes.

A set piece is one of the bundle's motion modes. The bundle says which pad
press enters each (`controller.toml`'s `[[fsm.button]]` rows: a button, or a
d-pad direction, with or without Menu held) and that Menu let go leaves any of
them, so `pad_presses` reads the table from the app rather than this file
repeating it, and a mode the app does not have is refused when the duel starts
rather than pressed into nothing at the bell.

Which mode is which crab's set piece is `MOVES`, overridable per colour and
piece by environment variable (`BATTLE_RED_TAUNT=gesture_hello`), so a trained
threat display or dance replaces a shipped one by name and nothing else moves.
The shipped choices: the red crab's taunt is the Zoidberg -- arms up, claws
snapping, the one shipped gesture that reads as a threat; the blue crab's is
the paw swipe. The cheers are the two short gestures (hello, salute: 7.3 and
6.0 s of clip), and the victory dances the Brazilian, cut short with Menu
since its clip is 66 s, and the maze, 10 s whole.

`Performer` presses a mode on a virtual pad the way `tools/fetch/auto.py`'s
routine does (the chord held for `PRESS_S`, then nothing while the clip plays,
then Menu held and let go to cut it) and says whether the crab is still busy:
the mode switch ramps for a second before the clip starts and the hand-back to
walking takes another two, so busy is "pressed, and not yet back in locomotion
with the policy running". The press waits for the crab to be walking with its
policy running -- the pieces are entered from locomotion only, and a chord
pressed while the controller is in `safe` or ramping is dropped (the second
brain take's cheer was pressed over a crab that had just gone over) -- and
gives up after `PRESS_WAIT_S`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tomllib

#: Each colour's set pieces, by the bundle's mode names.
MOVES = {
    # the battle's own displays, trained 2026-10-14 (threat: blue's rear-and-snap taunt;
    # pump: red's Zoidberg-armed bounce; wave: blue's spread-armed roll), in the bundle
    # since the rebuild of 2026-10-14 ~14:50 (Menu+X / Menu+Y / Menu+A, keypad 7/8/9)
    "red": {"taunt": "gesture_zoidberg", "cheer": "gesture_hello", "victory": "gesture_pump"},
    "blue": {"taunt": "gesture_threat", "cheer": "gesture_salute", "victory": "gesture_wave"},
}
for _colour, _pieces in MOVES.items():
    for _piece in _pieces:
        _pieces[_piece] = os.environ.get(f"BATTLE_{_colour.upper()}_{_piece.upper()}", _pieces[_piece])

#: How long a chord is held: a click, long enough for one 200 Hz tick of the
#: controller to see it rise and the next to see it fall (auto.py's 0.3 s).
PRESS_S = 0.3
#: How long a piece waits for the crab to be walking before it is given up.
PRESS_WAIT_S = 6.0
#: The kernel's hat: d-pad down is +1 on `hat_y`, right +1 on `hat_x`.
_HAT = {"dpad_up": (0, -1), "dpad_down": (0, +1), "dpad_left": (-1, 0), "dpad_right": (+1, 0)}


@dataclass(frozen=True)
class Press:
    """One chord on the pad: the buttons held and the hat, and the modes it works from."""

    buttons: frozenset[str]
    hat_x: int
    hat_y: int
    from_modes: tuple[str, ...]


def pad_presses(app) -> dict[str, Press]:
    """Every mode the pad can enter, from the app's own FSM table."""
    table = tomllib.loads((Path(app.root) / "controller.toml").read_text("utf-8"))["fsm"]
    presses: dict[str, Press] = {}
    for row in table.get("button", ()):
        if "pad" not in row or row.get("on") != "toggle":
            continue
        buttons = set()
        hat = (0, 0)
        if row["pad"] in _HAT:
            hat = _HAT[row["pad"]]
        else:
            buttons.add(row["pad"])
        if row.get("with"):
            buttons.add(row["with"])
        presses[row["name"]] = Press(frozenset(buttons), hat[0], hat[1], tuple(row.get("from", ())))
    return presses


def check_moves(presses: dict[str, Press], colours=("red", "blue")) -> None:
    """Refuse a set piece the app cannot enter from locomotion, before the bell."""
    for colour in colours:
        for piece, mode in MOVES[colour].items():
            if mode not in presses:
                raise SystemExit(f"error: {colour}'s {piece} `{mode}` is not a mode the pad can enter "
                                 f"(the app has {', '.join(sorted(presses))})")
            if "locomotion" not in presses[mode].from_modes:
                raise SystemExit(f"error: {colour}'s {piece} `{mode}` cannot be entered from locomotion")


#: The modes a crab drives in, from which a set piece can be pressed and to
#: which it hands back: the walk, and the two claw modes a brain may fight in.
WALKING = ("locomotion", "claw_left", "claw_right", "sprint")


class Performer:
    """Plays one set piece at a time on a crab's virtual pad."""

    def __init__(self, player, presses: dict[str, Press], label: str = "") -> None:
        self.player = player
        self.presses = presses
        self.label = label
        self.mode: str | None = None
        self._since = 0.0            # when the piece was asked for
        self._pressed = None         # when the chord went down
        self._cut_s = None
        self._cutting_since = None

    def start(self, mode: str, t: float, *, cut_s: float | None = None) -> None:
        """Press `mode` now; with `cut_s`, let go of it (Menu) that long after the press."""
        self.mode, self._since, self._pressed, self._cut_s, self._cutting_since = mode, t, None, cut_s, None
        print(f"\n[{self.label}] {t:6.2f}s  plays {mode}" + (f" for {cut_s:g} s" if cut_s else ""))

    def leave(self, t: float) -> None:
        """Menu, let go: leaves a motion mode. A set piece that ended in the tilt
        fallback used to keep its toggle latched, and the latch re-entered the
        piece as soon as the crab was upright again (the salute of the first brain
        take played twice, the second time into the next round's bell); since
        2026-10-15 the controller releases the latch of a motion it leaves early
        (README, "The reset"), and this press covers whatever mode is left."""
        self.mode, self._since, self._pressed, self._cut_s, self._cutting_since = "_leave", t, t, None, None
        print(f"\n[{self.label}] {t:6.2f}s  leaves {self.player.fsm.mode}")

    def busy(self, t: float) -> bool:
        """Pressed, and not yet back walking with the policy running."""
        if self.mode is None:
            return False
        fsm = self.player.fsm
        if self._pressed is None:
            if t - self._since > PRESS_WAIT_S:
                print(f"\n[{self.label}] {t:6.2f}s  gives up {self.mode}: not walking ({fsm.mode})")
                self.mode = None
                return False
            return True                                # waiting to be walking
        if t - self._pressed < 1.0:
            return True                                # the switch has not registered yet
        if fsm.mode in WALKING and fsm.running_policy:
            self.mode = None
            return False
        return True

    def state(self, t: float):
        """This step's frame of the pad, or None when the piece is playing on its own."""
        from controller import GamepadState

        if self.mode is None:
            return None
        fsm = self.player.fsm
        if self._pressed is None:
            if not (fsm.mode in WALKING and fsm.running_policy):
                return None
            self._pressed = t
        since = t - self._pressed
        if self.mode == "_leave":
            return GamepadState(buttons=frozenset({"menu"})) if since < PRESS_S else None
        if since < PRESS_S:
            press = self.presses[self.mode]
            return GamepadState(hat_x=press.hat_x, hat_y=press.hat_y, buttons=press.buttons)
        if self._cut_s is not None and since >= self._cut_s and self.player.fsm.mode == self.mode:
            if self._cutting_since is None:
                self._cutting_since = t
            if t - self._cutting_since < PRESS_S:
                return GamepadState(buttons=frozenset({"menu"}))
        return None
