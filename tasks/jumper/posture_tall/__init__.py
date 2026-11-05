"""jumper.posture_tall -- jumper.posture with the body carried high.

The same task as `jumper.posture`, with the height command raised to 0.10-0.18 m,
most height draws taken from the top of that range, and a reward for keeping the
middle and rear legs extended when a tall stance is asked for. `env_cfg.py` lists
every difference; the rest is imported from `tasks/jumper/posture/`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.posture_tall",
    assets=JUMPER_ASSETS,
    description=(
        "jumper.posture with the height command raised to 0.10-0.18 m and biased high: "
        "legs near straight, body carried high; tripod gait at 5 Hz, 0.8 m/s"
    ),
    tags=("locomotion", "jumper", "flat", "tripod", "posture"),
)
