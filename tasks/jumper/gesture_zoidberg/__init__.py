"""jumper.gesture_zoidberg -- both claws straight up, the body swaying, the claws
snapping in alternation: the "woop woop" dance. Learned by imitation of a keyframed
clip, `media/zoidberg.npz`, written by `tools/fetch/make_zoidberg.py`.

A one-shot gesture on the dance machinery, as the imported gestures are: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers. The numbers are `jumper.gesture_bow`'s, as a starting point.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_zoidberg",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed Zoidberg dance (both claws up, "
                "snapping); the clip is committed in tasks/jumper/gesture_zoidberg/media/",
)
