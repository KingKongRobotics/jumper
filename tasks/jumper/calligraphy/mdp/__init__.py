"""MDP terms specific to jumper.calligraphy.

`writing` is the command that takes the carried arm out to write and back, and
the reward for holding the trunk where it stood while it does. Everything else is
`jumper.five_foot`'s, which this task fine-tunes (see `../env_cfg.py`).
"""

from . import writing

__all__ = ["writing"]
