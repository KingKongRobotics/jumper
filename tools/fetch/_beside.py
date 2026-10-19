"""Sibling imports for the fetch tools without touching `sys.path`.

`run.py` loads a pilot by its file path, and the pilot, the props and the probes
import each other by bare name (`from sense import Senses`); `run.py` also borrows
`play`'s helpers from `scripts/`. A `sys.path` insert would resolve both, and the
repository forbids it (`tests/test_layout.py::test_no_sys_path_mutation`: which
directory you ran from must never decide which code you get). This finder resolves
exactly those names to exactly these files, wherever the process was started.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1] / "scripts"
FROM_SCRIPTS = {"_cli", "play"}
FROM_TOOLS = {"import_wbc_dances", "import_wbc_gestures"}     # the gesture makers borrow the wbc importers


class _Beside(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if "." in name:
            return None
        if name in FROM_SCRIPTS:
            file = SCRIPTS / f"{name}.py"
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
