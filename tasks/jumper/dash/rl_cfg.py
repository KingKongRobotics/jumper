"""jumper.dash's training config: jumper.jump's, logged under its own name."""
from __future__ import annotations

from dataclasses import replace

from ..jump import rl_cfg as _jump


def agent_cfg():
    cfg = _jump.agent_cfg()
    # (the literal `experiment_name="jumper.dash"` is what tests/test_log_layout.py reads)
    cfg = replace(cfg, experiment_name="jumper.dash")
    return cfg


for _name in ("runner_cls", "play_status"):
    if hasattr(_jump, _name):
        globals()[_name] = getattr(_jump, _name)
