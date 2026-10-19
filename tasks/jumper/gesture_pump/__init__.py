"""jumper.gesture_pump -- the claw pump: both claws straight up, the body bouncing twice a second, the claws snapping at the bottom of every bounce -- the red crab's victory dance in the crab battle (`tools/battle/`). Learned by imitation of a keyframed
clip, `media/pump.npz`, written by `tools/battle/make_pump.py`.

A one-shot gesture on the dance machinery, as the imported gestures are: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers. The numbers are `jumper.gesture_bow`'s, as a starting point.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_pump",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed claw pump (claws up, bouncing, snapping); "
                "the clip is committed in tasks/jumper/gesture_pump/media/",
)
