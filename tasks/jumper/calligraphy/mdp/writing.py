"""The arm writing while the trunk stands still: a command term and its reward.

## Why a command term

`jumper.five_foot` holds the carried arm stowed, sampled per episode inside a
small box (`five_foot/claw.py::LF_GRASP_BOX`). The shipped policy therefore never
saw the arm out, and it shows: with the arm held over the writing band it drifts
up to 36 mm and 67 deg (step 0), it cannot walk (0.001 m/s at 0.12 commanded), the
trunk settles 5-10 mm from where it stopped when the arm unfolds, and at the
band's ends it does not hold still at all (13 of 57 probes, `tools/stability.py`).

This term does to the arm, in training, what `tools/write.py` does to it when
writing: segments of the episode take it out -- unfold from the stow, a chain of
moves between floor poses a few centimetres apart, at writing speed, some hovering,
some on the floor, some with the hair sunk in -- and fold it back. It is a command
term rather than an event because it advances every step and owns state that the
reward reads. The poses are the committed table `data/writing_poses.npz`
(`tools/writing_poses.py`).

## The trunk is told to stand while the arm is out

Writing is done standing, so while the arm is out the velocity command is zeroed
and the body-pose command levelled, for those robots, every step -- after their
own terms have updated (this term is registered after them). The policy observes
both, so it is told to stand exactly as `tools/write.py` will tell it.

## Holding still is position, not velocity

The velocity tracking terms reward a zero command with a small speed, which a
slow creep satisfies and a 5 mm settling step does not cost. Writing needs the
trunk where it was, so `hold_position` rewards distance from where the trunk stood
when it last became still -- the arm going out, or a standing command -- in xy
and in heading, after a short grace for the step that is ending.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ...five_foot.claw import ARM_JOINTS, FINGER_JOINT, LF_GRASP

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

CARRY, UNFOLD, WRITE, FOLD = range(4)
POSES = Path(__file__).resolve().parents[1] / "data" / "writing_poses.npz"


@dataclass(kw_only=True)
class WritingArmCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    velocity_command_name: str = "twist"
    pose_command_name: str = "body_pose"
    poses: Path = POSES
    #: Each time the term resamples, the chance a robot's arm is out until the next.
    rel_writing_envs: float
    #: Seconds to unfold from the stow to the first pose, and to fold back.
    unfold_s: tuple[float, float]
    fold_s: tuple[float, float]
    #: Each move goes this far over the floor, at this speed.
    move_reach: tuple[float, float]
    move_speed: tuple[float, float]
    #: Probability of hovering, touching and sinking the hair, per move.
    height_probs: tuple[float, float, float]
    #: The finger, shut on the handle while the arm is out.
    finger_hold: float
    #: Noise around the stow when folding back, rad.
    stow_noise: float
    #: Seconds of a still spell before `hold_position` counts.
    grace_s: float

    def build(self, env: ManagerBasedRlEnv) -> WritingArmCommand:
        return WritingArmCommand(self, env)


class WritingArmCommand(CommandTerm):
    cfg: WritingArmCommandCfg

    def __init__(self, cfg: WritingArmCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        names = list(self.robot.joint_names)
        # Tensors, not lists: `set_joint_position_target` outer-indexes with them.
        self.arm_ids = torch.tensor([names.index(j) for j in ARM_JOINTS], device=self.device)
        self.finger_id = torch.tensor([names.index(FINGER_JOINT)], device=self.device)
        table = np.load(cfg.poses)
        self.xy = torch.tensor(table["xy"], device=self.device)
        self.q = torch.tensor(table["q"], device=self.device)  # (cells, heights, 4)
        self.stow = torch.tensor([LF_GRASP[j] for j in ARM_JOINTS], device=self.device)
        self.height_probs = torch.tensor(cfg.height_probs, device=self.device)

        n, d = self.num_envs, self.device
        self.phase = torch.zeros(n, dtype=torch.long, device=d)
        self.want = torch.zeros(n, dtype=torch.bool, device=d)
        self.t = torch.zeros(n, device=d)
        self.T = torch.ones(n, device=d)
        self.q_from = self.stow.repeat(n, 1)
        self.q_to = self.stow.repeat(n, 1)
        self.cell = torch.zeros(n, dtype=torch.long, device=d)
        self.still = torch.zeros(n, dtype=torch.bool, device=d)
        self.still_t = torch.zeros(n, device=d)
        self.anchor_xy = torch.zeros(n, 2, device=d)
        self.anchor_yaw = torch.zeros(n, device=d)
        self.metrics["arm_out"] = torch.zeros(n, device=d)
        self.metrics["hold_drift"] = torch.zeros(n, device=d)

    @property
    def command(self) -> torch.Tensor:
        return self.phase.float().unsqueeze(1)

    # ── state ────────────────────────────────────────────────────────────────

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self.phase[env_ids] = CARRY
        self.t[env_ids] = 0.0
        self.still[env_ids] = False
        self.still_t[env_ids] = 0.0
        return super().reset(env_ids)

    def _uniform(self, k: int, lo_hi: tuple[float, float]) -> torch.Tensor:
        return torch.empty(k, device=self.device).uniform_(*lo_hi)

    def _current(self, ids: torch.Tensor) -> torch.Tensor:
        return self.robot.data.joint_pos_target[ids][:, self.arm_ids].clone()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.rand(len(env_ids), device=self.device)
        self.want[env_ids] = r < self.cfg.rel_writing_envs
        # Out from the stow: unfold to a random pose, hovering.
        go = env_ids[self.want[env_ids] & (self.phase[env_ids] == CARRY)]
        if len(go):
            cell = torch.randint(0, len(self.xy), (len(go),), device=self.device)
            self.cell[go] = cell
            self._start(go, self._current(go), self.q[cell, 0], self.cfg.unfold_s, UNFOLD)
        # Back to the stow.
        back = env_ids[~self.want[env_ids] & (self.phase[env_ids] != CARRY)
                       & (self.phase[env_ids] != FOLD)]
        if len(back):
            noise = (torch.rand(len(back), 4, device=self.device) * 2 - 1) * self.cfg.stow_noise
            self._start(back, self._current(back), self.stow + noise, self.cfg.fold_s, FOLD)

    def _start(self, ids, q_from, q_to, seconds, phase) -> None:
        self.q_from[ids] = q_from
        self.q_to[ids] = q_to
        self.t[ids] = 0.0
        self.T[ids] = seconds if isinstance(seconds, torch.Tensor) else self._uniform(
            len(ids), seconds)
        self.phase[ids] = phase

    def _next_move(self, ids: torch.Tensor) -> None:
        """A move to a cell `move_reach` away, at a height drawn from `height_probs`."""
        k = 16
        here = self.xy[self.cell[ids]]                                    # (m, 2)
        cand = torch.randint(0, len(self.xy), (len(ids), k), device=self.device)
        dist = torch.linalg.norm(self.xy[cand] - here[:, None], dim=-1)   # (m, k)
        lo, hi = self.cfg.move_reach
        ok = (dist >= lo) & (dist <= hi)
        # The first candidate in range, else the first one drawn.
        pick = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(ok[:, 0],
                                                                                dtype=torch.long))
        cell = cand[torch.arange(len(ids), device=self.device), pick]
        d = dist[torch.arange(len(ids), device=self.device), pick]
        height = torch.multinomial(self.height_probs, len(ids), replacement=True)
        speed = self._uniform(len(ids), self.cfg.move_speed)
        self.cell[ids] = cell
        self._start(ids, self.q_to[ids].clone(), self.q[cell, height],
                    torch.clamp(d / speed, 0.3, 4.0), WRITE)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is not None:
            return  # the reset path: `reset` has put every arm back in the stow
        dt = self._env.step_dt
        out = self.phase != CARRY
        self.t[out] += dt
        done = out & (self.t >= self.T)
        moving_on = done & (self.phase != FOLD)
        if moving_on.any():
            self._next_move(moving_on.nonzero().flatten())
        folded = done & (self.phase == FOLD)
        self.phase[folded] = CARRY

        out_ids = (self.phase != CARRY).nonzero().flatten()
        if len(out_ids):
            f = torch.clamp(self.t[out_ids] / self.T[out_ids], 0.0, 1.0)[:, None]
            f = f * f * (3 - 2 * f)
            q = self.q_from[out_ids] + f * (self.q_to[out_ids] - self.q_from[out_ids])
            self.robot.set_joint_position_target(q, joint_ids=self.arm_ids, env_ids=out_ids)
            self.robot.set_joint_position_target(
                torch.full((len(out_ids), 1), self.cfg.finger_hold, device=self.device),
                joint_ids=self.finger_id, env_ids=out_ids)
            # Standing, level, while the arm is out -- the way write.py drives it.
            vel = self._env.command_manager.get_term(self.cfg.velocity_command_name)
            vel.vel_command_b[out_ids] = 0.0
            vel.vel_command_w[out_ids] = 0.0
            vel.is_standing_env[out_ids] = True
            pose = self._env.command_manager.get_term(self.cfg.pose_command_name)
            pose.pose_target_b[out_ids] = 0.0
            pose.is_neutral_env[out_ids] = True

        # Still spells, and where each one began.
        vel = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        still_now = (self.phase != CARRY) | vel.is_standing_env
        began = still_now & ~self.still
        if began.any():
            self.anchor_xy[began] = self.robot.data.root_link_pos_w[began, :2]
            self.anchor_yaw[began] = self.robot.data.heading_w[began]
            self.still_t[began] = 0.0
        self.still_t[still_now] += dt
        self.still = still_now

    def _update_metrics(self) -> None:
        self.metrics["arm_out"] = (self.phase != CARRY).float()
        drift = torch.linalg.norm(self.robot.data.root_link_pos_w[:, :2] - self.anchor_xy, dim=1)
        self.metrics["hold_drift"] = torch.where(self.still, drift, torch.zeros_like(drift))


def hold_position(env: ManagerBasedRlEnv, command_name: str, std_xy: float,
                  std_yaw: float) -> torch.Tensor:
    """`exp(-(d^2 / std_xy^2 + dyaw^2 / std_yaw^2))` while still, past the grace.

    `d` and `dyaw` are how far the trunk is from where it stood when the still
    spell began. Zero while walking, and for the first `grace_s` of a spell -- the
    step that was under way when the command stopped is allowed to land.
    """
    term = env.command_manager.get_term(command_name)
    robot = term.robot
    d2 = torch.sum((robot.data.root_link_pos_w[:, :2] - term.anchor_xy) ** 2, dim=1)
    dyaw = torch.atan2(torch.sin(robot.data.heading_w - term.anchor_yaw),
                       torch.cos(robot.data.heading_w - term.anchor_yaw))
    r = torch.exp(-(d2 / std_xy**2 + dyaw**2 / std_yaw**2))
    counts = term.still & (term.still_t > term.cfg.grace_s)
    return torch.where(counts, r, torch.zeros_like(r))



def feet_slide_standing(env: ManagerBasedRlEnv, sensor_name: str, command_name: str,
                        command_threshold: float, asset_cfg) -> torch.Tensor:
    """Summed horizontal speed of the feet **on the floor**, while told to stand. m/s.

    five_foot's `feet_still` charges foot speed while standing, on the floor and in
    the air alike, and `feet_planted` pays every foot that is down. So a foot moved
    by a step paid what a foot dragged the same distance paid, and lost
    `feet_planted` while it was up: dragging was always the cheaper way to move a
    foot, and every policy here did it -- 0 lift-offs, 14-55 mm of slide per foot
    over a 4.8 s unfold, the same with MuJoCo's elliptic cone at impratio 10, so not
    contact creep (`model_92598`, 2026-10-09, native:cpu). This charges only the
    drag, so a step becomes the cheap way. The contact test is the sensor's `found`,
    the one the gait terms read.
    """
    from ...common.mdp.rewards import contact_state, moving_gate

    asset = env.scene[asset_cfg.name]
    speed = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1)  # [B, F]
    touching = contact_state(env, sensor_name)
    standing = 1.0 - moving_gate(env, command_name, command_threshold)
    return (speed * touching).sum(dim=1) * standing


__all__ = ["WritingArmCommand", "WritingArmCommandCfg", "feet_slide_standing", "hold_position"]
