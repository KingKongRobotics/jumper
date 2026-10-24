"""Sibling imports for the course tools without touching `sys.path`.

`course.py` loads the pilot by its file path, and the pilot, the track, the
referee and the readout import each other by bare name; the runner borrows
`play`'s helpers from `scripts/`, the recoloured crab, the set pieces and the
virtual pad from `tools/battle/` (`arena`, `moves`, `duel`), the dToF reading and
the readout's drawing from `tools/race/` (`sight`, `hud`) and the map from
`tools/fetch/` (`mapper`). A `sys.path` insert would resolve all of those, and the
repository forbids it (`tests/test_layout.py::test_no_sys_path_mutation`). This
finder resolves exactly those names to exactly these files, wherever the process
was started -- the same finder `tools/race/_beside.py` is, pointed at this directory.
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
RACE = HERE.parent / "race"
FROM_SCRIPTS = {"_cli", "play"}
FROM_FETCH = {"sense", "mapper"}
FROM_BATTLE = {"arena", "moves", "duel"}
FROM_RACE = {"sight", "room"}           # the dToF reading is the race's (`room` is what `sight` imports beside it)


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
        elif name in FROM_RACE:
            file = RACE / f"{name}.py"
        else:
            file = HERE / f"{name}.py"
        if not file.is_file():
            return None
        return importlib.util.spec_from_file_location(name, file)


def install() -> None:
    if not any(type(f).__name__ == "_Beside" and getattr(f, "_here", None) == HERE for f in sys.meta_path):
        finder = _Beside()
        finder._here = HERE
        sys.meta_path.append(finder)      # after the normal finders: installed packages win
