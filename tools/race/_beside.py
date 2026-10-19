"""Sibling imports for the race tools without touching `sys.path`.

`race.py` loads each crab's pilot by its file path, and the pilots, the room, the
referee and the readout import each other by bare name (`from referee import
Referee`); `race.py` borrows `play`'s helpers from `scripts/`, the second crab and
the set pieces from `tools/battle/` (`arena`, `moves`), and the senses and the map
from `tools/fetch/` (`sense`, `mapper`). A `sys.path` insert would resolve all of
those, and the repository forbids it (`tests/test_layout.py::test_no_sys_path_mutation`:
which directory you ran from must never decide which code you get). This finder
resolves exactly those names to exactly these files, wherever the process was
started -- the same finder `tools/battle/_beside.py` is, pointed at this directory.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
FETCH = HERE.parent / "fetch"
BATTLE = HERE.parent / "battle"
FROM_SCRIPTS = {"_cli", "play"}
FROM_FETCH = {"sense", "mapper"}        # the crab's senses and its map are the fetch's
FROM_BATTLE = {"arena", "moves", "duel"}   # the second crab, the set pieces and the two-player loop are the battle's
FROM_TOOLS: set[str] = set()


class _Beside(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if "." in name:
            return None
        if name in FROM_SCRIPTS:
            file = SCRIPTS / f"{name}.py"
        elif name in FROM_FETCH:
            file = FETCH / f"{name}.py"
        elif name in FROM_BATTLE:
            file = BATTLE / f"{name}.py"
        elif name in FROM_TOOLS:
            file = HERE.parent / f"{name}.py"
        else:
            file = HERE / f"{name}.py"
        if not file.is_file():
            return None
        return importlib.util.spec_from_file_location(name, file)


def install() -> None:
    if not any(isinstance(f, _Beside) for f in sys.meta_path):
        sys.meta_path.append(_Beside())      # after the normal finders: installed packages win
