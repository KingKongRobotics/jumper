"""jumper.dance_dream_wings_tall -- jumper.dance_dream_wings danced on a taller, straighter-legged stance.

The same task as `jumper.dance_dream_wings`, with its clip raised by `tools/raise_dance.py`
(see `env_cfg.py` for the parameters and what they do to the clip). The
environment is `tasks/jumper/common/dance/env.py`; every number is
`jumper.dance_dream_wings`'s, so the stance is the one thing that differs.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_dream_wings_tall",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating jumper.dance_dream_wings's clip raised to a 145 mm median body "
                "height; the clip is committed in tasks/jumper/dance_dream_wings_tall/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
