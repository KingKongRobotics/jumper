"""jumper.gesture_wave -- the wave: both claws spread, the body rolling side to side and the arm on the high side rising with each roll, the claws snapping as they come up -- the blue crab's victory dance in the crab battle (`tools/battle/`). Learned by imitation of a keyframed
clip, `media/wave.npz`, written by `tools/battle/make_wave.py`.

A one-shot gesture on the dance machinery, as the imported gestures are: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers. The numbers are `jumper.gesture_bow`'s, as a starting point.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_wave",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed wave (claws spread, rolling, one claw up at a time); "
                "the clip is committed in tasks/jumper/gesture_wave/media/",
)
