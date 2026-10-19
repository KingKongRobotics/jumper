"""jumper.sprint -- jumper.posture with a higher speed ceiling, for the battle's dash.

The same task as `jumper.posture` in every respect but the one decision that
sets the gait's speed: the linear ceiling (1.2 m/s against 0.8) and the stride
the cadence law assumes (0.08 m against 0.072), from which the clock's
frequency and the angular ceiling follow. `env_cfg.py` builds it from the
posture task's own source with those two constants replaced, so there is one
posture task to maintain; it is trained by resuming the posture policy
(`tasks/jumper/posture/out/example/model_74800.pt`) rather than from nothing.
The battle's brain switches into it for a charge and back out
(`tools/battle/pilots/brain.py`).
"""
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.sprint",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod velocity tracking with a commanded body posture, "
        "the posture task at a 1.2 m/s ceiling: the battle's dash"
    ),
    tags=("locomotion", "jumper", "flat", "tripod", "posture", "sprint"),
)
