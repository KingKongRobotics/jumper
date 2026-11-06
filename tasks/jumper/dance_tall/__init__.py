"""jumper.dance_tall -- jumper.dance's choreography on a taller, straighter-legged stance.

The same task as `jumper.dance`, with its clip raised by `tools/raise_dance.py
--height 0.145 --scale 0.5`: the body's median height goes from 112 mm to 145 mm,
its own excursion is halved (141-157 mm), and the middle and rear legs are
re-solved by IK so every support foot stays where it was. Arms, attitude, timing,
the beat track and the face animation (`media/demo_tall.mp4`, a copy of the
demo's) are the original's.

Every number in `env_cfg.py` and `rl_cfg.py` is `jumper.dance`'s, so the stance is
the one thing that differs. What `jumper.dance`'s docstring says about material,
music and deployment holds here unchanged.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_tall",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating jumper.dance's choreography raised to a 145 mm median "
                "body height; the clip is committed in tasks/jumper/dance_tall/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
