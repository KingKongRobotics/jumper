"""Hyper-parameters for jumper.posture_tall: jumper.posture's, value for value.

Imported rather than written out because this task is defined as a diff against
`jumper.posture`; a second change in the optimiser would make the comparison
between the two say less. Only the experiment name differs.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..posture import rl_cfg as parent
from ..posture.rl_cfg import NUM_STEPS_PER_ENV, play_status, runner_cls

__all__ = ["NUM_STEPS_PER_ENV", "agent_cfg", "play_status", "runner_cls"]


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """`logs/<model>/jumper.posture_tall/<date-time>`."""
    return replace(parent.agent_cfg(), experiment_name="jumper.posture_tall")
