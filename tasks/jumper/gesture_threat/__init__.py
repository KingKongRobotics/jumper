"""jumper.gesture_threat -- the body reared nose-up, both claws spread wide and
high, sweeping in and snapping shut on the beat: the blue crab's threat display at
the bell of the crab battle (`tools/battle/`). Learned by imitation of a keyframed
clip, `media/threat.npz`, written by `tools/battle/make_threat.py`.

A one-shot gesture on the dance machinery, as the imported gestures are: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers. The numbers are `jumper.gesture_bow`'s, as a starting point.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_threat",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed threat display (reared up, claws "
                "spread and snapping); the clip is committed in tasks/jumper/gesture_threat/media/",
)
