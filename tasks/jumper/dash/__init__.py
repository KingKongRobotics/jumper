"""jumper.dash -- the high jump carried forward: a hop of 0.3 m in the body's direction.

`jumper.jump`'s task in every respect but its reference: `ref/dash_flat.npz` is
the high jump's recording with the base carried 0.3 m forward through the
flight (`tools/battle/make_dash.py` writes it). Trained by resuming the jump
policy; the crab battle's dash (Andrew, 2026-10-14: "a quick hop but in a
specific direction, not straight up like the jump").
"""
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dash",
    assets=JUMPER_ASSETS,
    description="jumper hexapod reference-guided forward hop (the high jump carried 0.3 m forward): the battle's dash",
    tags=("locomotion", "jumper", "jump", "reference-tracking", "dash"),
)
