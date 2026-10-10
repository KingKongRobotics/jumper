"""jumper.calligraphy -- Jumper writes a Chinese character on the floor with a brush.

Two things live here. The **tools** (`tools/`, with `hanzi.py`, `brush.py`,
`arm.py`, `stations.py`, `ink.py`, `cameras.py`, `sim.py`) plan a character,
drive a five-legged policy and the carried arm to write it, and render it. The
**task** (`env_cfg.py`, `rl_cfg.py`, `mdp/`) fine-tunes `jumper.five_foot` so that
policy stands still while the arm writes -- which the shipped five_foot policy,
never trained with the arm out, does not reliably do. See `README.md`.

This module registers the task's name and nothing else: no config imports, so
`--list` stays free of simulation dependencies.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.calligraphy",
    assets=JUMPER_ASSETS,
    description="jumper.five_foot fine-tuned to stand still while the carried arm writes "
    "with a brush",
    tags=("jumper", "flat", "five_foot", "manipulation", "calligraphy"),
)
