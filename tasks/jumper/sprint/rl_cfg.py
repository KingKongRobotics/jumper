"""jumper.sprint's training config: jumper.posture's, logged under its own name."""
from __future__ import annotations

from dataclasses import replace

from ..posture import rl_cfg as _posture
from ..posture.rl_cfg import play_status, runner_cls  # noqa: F401

NUM_STEPS_PER_ENV = _posture.NUM_STEPS_PER_ENV


def agent_cfg():
    cfg = _posture.agent_cfg()
    # (the literal `experiment_name="jumper.sprint"` is what tests/test_log_layout.py reads)
    cfg = replace(cfg, experiment_name="jumper.sprint")
    return cfg
