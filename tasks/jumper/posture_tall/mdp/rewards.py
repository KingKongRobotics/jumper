"""What jumper.posture_tall is paid for on top of jumper.posture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...posture.mdp.rewards import _command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def leg_straightness(
    env: "ManagerBasedRlEnv",
    min_height: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    reach: tuple[float, ...],
) -> torch.Tensor:
    """Mean straightness of the listed legs, scored only when the commanded height
    is at least `min_height` (crouching is never penalised).

    Straightness is the distance from the thigh body's origin (the J1 joint) to
    the foot, divided by that leg's fully extended length, clipped at 1.
    `asset_cfg.body_ids` are pairs `(hip0, foot0, hip1, foot1, ...)` and `reach`
    holds one extended length per pair.
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    d = (pos[:, 1::2] - pos[:, 0::2]).norm(dim=-1)
    ratio = d / torch.tensor(reach, device=d.device)
    cmd = _command(env, command_name)
    gate = (cmd[:, 3] >= min_height).float()
    return ratio.clamp(max=1.0).mean(dim=-1) * gate


__all__ = ["leg_straightness"]
