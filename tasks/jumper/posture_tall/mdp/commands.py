"""The posture command, with its height draws biased towards a tall stance.

`jumper.posture` draws the height uniformly over the curriculum's current range.
With the range raised to 0.10-0.18 m a uniform draw still spends most samples
below 0.15, and on the first 1500-iteration run the policy traded height for
speed. So after the parent's draw, `tall_share` of the resampled environments
have their height redrawn from `[tall_low, current upper bound]`. The upper bound
follows the curriculum; until the ladder has climbed above `tall_low` the bias
is off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...posture.mdp.commands import PostureCommand, PostureCommandCfg
from ...posture.mdp.teleop import TeleopPostureCommand, TeleopPostureCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class TallPostureCommand(PostureCommand):
    """`PostureCommand` with `tall_share` of its height draws taken from the top."""

    cfg: "TallPostureCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        low, high = self.cfg.ranges.height
        if high <= self.cfg.tall_low:
            return
        # Applied after the parent's draw, so environments it made neutral keep the
        # neutral height: those are what teach the robot to stand at HOME.
        tall = (torch.rand(len(env_ids), device=self.device) < self.cfg.tall_share) & ~self.is_neutral_env[env_ids]
        n = int(tall.sum())
        if n:
            self.posture_command[env_ids[tall], 3] = torch.empty(n, device=self.device).uniform_(
                max(low, self.cfg.tall_low), high
            )


@dataclass(kw_only=True)
class TallPostureCommandCfg(PostureCommandCfg):
    tall_share: float
    """Fraction of resampled environments whose height is redrawn from the top."""

    tall_low: float
    """Lower edge of the tall band, metres."""

    def build(self, env: "ManagerBasedRlEnv") -> TallPostureCommand:
        return TallPostureCommand(self, env)


class TallTeleopPostureCommand(TeleopPostureCommand, TallPostureCommand):
    """The operator's term on top of the biased sampler, so replay matches training
    until a control is touched."""


@dataclass(kw_only=True)
class TallTeleopPostureCommandCfg(TeleopPostureCommandCfg, TallPostureCommandCfg):
    def build(self, env: "ManagerBasedRlEnv") -> TallTeleopPostureCommand:
        return TallTeleopPostureCommand(self, env)


__all__ = [
    "TallPostureCommand",
    "TallPostureCommandCfg",
    "TallTeleopPostureCommand",
    "TallTeleopPostureCommandCfg",
]
