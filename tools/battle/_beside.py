"""Sibling imports for the battle tools without touching `sys.path`.

`duel.py` loads each crab's pilot by its file path, and the pilots, the arena and
the referee import each other by bare name (`from referee import Referee`);
`duel.py` also borrows `play`'s helpers from `scripts/`, and a pilot that senses
rather than cheats borrows `sense` from `tools/fetch/`. A `sys.path` insert would
resolve all of those, and the repository forbids it
(`tests/test_layout.py::test_no_sys_path_mutation`: which directory you ran from
must never decide which code you get). This finder resolves exactly those names
to exactly these files, wherever the process was started -- the same finder
`tools/fetch/_beside.py` is, pointed at this directory.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
FETCH = HERE.parent / "fetch"
FROM_SCRIPTS = {"_cli", "play"}
FROM_FETCH = {"sense"}     # the crab's senses are the fetch's: the dToF image, the IMU, the encoders
FROM_TOOLS = {"import_wbc_dances", "import_wbc_gestures"}     # the clip makers borrow the wbc importers


class _Beside(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if "." in name:
            return None
        if name in FROM_SCRIPTS:
            file = SCRIPTS / f"{name}.py"
        elif name in FROM_FETCH:
            file = FETCH / f"{name}.py"
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
