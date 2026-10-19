"""jumper.gesture_sleep -- the body settles to its lowest stance, nose down, on all
six feet, and breathes (a slow bob of a few millimetres), then rises. Learned by
imitation of a keyframed clip, `media/sleep.npz`, written by `tools/fetch/make_sleep.py`.

A one-shot gesture on the dance machinery, as the imported gestures are: the
environment is `tasks/jumper/common/dance/env.py`, and what this directory adds is
the clip and the numbers. The numbers are `jumper.gesture_zoidberg`'s, as a starting
point. Why trained rather than driven through the posture mode: the posture policy
keeps balancing at its lowest height and the robot jiggles; a gesture policy tracks a
still reference and is still -- a resting pose for anything that needs one.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_sleep",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a keyframed sleep (settle low, nose down, "
                "breathe, rise); the clip is committed in tasks/jumper/gesture_sleep/media/",
)
