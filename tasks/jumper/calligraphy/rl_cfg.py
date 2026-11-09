"""jumper.calligraphy's rsl_rl config: five_foot's, renamed, and shorter.

Five_foot's network and PPO settings, because this task resumes from its
checkpoint and the network has to be the same to load (see `env_cfg.py`). Only the
experiment name and the length of the run are this task's.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """`logs/<model>/jumper.calligraphy/<date-time>`.

    `max_iterations` 3000: a fine-tune of a policy that already walks, not a
    training from scratch (five_foot's checkpoint is at iteration 86600). A first
    figure, to be judged on `Episode_Reward/hold_position` and
    `Metrics/writing/hold_drift` flattening; `--max-iterations` overrides it.
    """
    from dataclasses import replace

    from ..five_foot.rl_cfg import agent_cfg as five_foot_agent_cfg

    return replace(five_foot_agent_cfg(), experiment_name="jumper.calligraphy",
                   max_iterations=3000)


def runner_cls() -> type:
    """Five_foot's: the curriculum levels travel in the checkpoint, so the warm
    start begins at the command range and payload five_foot had reached."""
    from ..five_foot.rl_cfg import runner_cls as five_foot_runner_cls

    return five_foot_runner_cls()
