"""Jump command term: a single jump per episode.

The go instant is expressed uniformly through `_go_step` (a floating-point step
number on the episode step axis):

- Normal spawn: `_go_step = delay/step_dt`; during the wait `time_since_go < 0`,
  the reference is clamped to the go frame (standing), phase = 0.
- RSI spawn (the reset event records the spawn phase in
  `env._jump_rsi_phase`): go already happened before the episode started,
  `_go_step` is **negative**, and `time_since_go = phase×span` at the start.

The reset event runs before `command_manager.reset()` (see the ordering comment
in `ManagerBasedRlEnv._reset_idx`), so the event writes the phase and this term
consumes it in `_resample_command` without it being overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

from .reference import JUMP_REF_NPZ, get_reference

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JumpMotionCommandCfg(CommandTermCfg):
    class_type: type = None  # filled in __post_init__ (forward reference)

    # Override the parent field that has no default: resample only happens at
    # episode reset (called explicitly in reset()), so give a time far larger
    # than an episode to keep time_left from running out on the step path and
    # triggering an unexpected resample.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    entity_name: str = "robot"
    """The entity the reference motion is attached to."""

    motion_file: str = str(JUMP_REF_NPZ)
    """Path to the reference motion npz."""

    command_delay_range: tuple[float, float] = (0.2, 1.0)
    """Random delay (s) of the go command for non-RSI spawns."""

    rsi_fraction: float = 1.0
    """Fraction of spawns using RSI (Reference State Initialization).

    An RSI env spawns directly at a random phase (0~0.9) of the reference
    trajectory -- joints and base pose/velocity all come from the reference.
    1.0 means every env spawns on the trajectory, so the policy learns
    "finish the motion from some intermediate state" throughout, which is the
    key setting for reference-guided jumping.
    """

    rsi_phase_range: tuple[float, float] = (0.0, 0.9)
    """RSI spawn phase range. Excludes 1.0 -- spawning at the endpoint has no learning value."""

    force_threshold: float = 1.0
    """Force threshold (N) for deciding foot lift-off/contact, used by the flight metrics."""

    def __post_init__(self):
        self.class_type = JumpMotionCommand

    def build(self, env: "ManagerBasedRlEnv") -> "JumpMotionCommand":
        return JumpMotionCommand(self, env)


class JumpMotionCommand(CommandTerm):
    cfg: JumpMotionCommandCfg

    def __init__(self, cfg: JumpMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.reference = get_reference(env, cfg.motion_file)

        # Go step number on the episode step axis; negative = RSI (go happened early).
        self.go_step = torch.zeros(self.num_envs, device=self.device)

        # Unlatched flight metrics (RSI, delay and not having learned to jump yet do not
        # affect their meaning).
        self._stance_z = self.reference.stand_base_z
        self._apex_rise = torch.zeros(self.num_envs, device=self.device)
        self._flight_s = torch.zeros(self.num_envs, device=self.device)
        self._takeoff_vz = torch.zeros(self.num_envs, device=self.device)
        self._travel = torch.zeros(self.num_envs, device=self.device)
        self._contact_sensor: ContactSensor | None = None
        self.metrics["apex_rise"] = self._apex_rise
        self.metrics["flight_s"] = self._flight_s
        self.metrics["takeoff_vz"] = self._takeoff_vz
        self.metrics["travel"] = self._travel

    # ── Queries ────────────────────────────────────────────────────────
    @property
    def time_since_go(self) -> torch.Tensor:
        return (self._env.episode_length_buf - self.go_step).float() * self._env.step_dt

    @property
    def phase(self) -> torch.Tensor:
        return self.reference.phase(self.time_since_go)

    @property
    def command(self) -> torch.Tensor:
        return self.phase.unsqueeze(1)

    # ── CommandTerm interface ──────────────────────────────────────────
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        delay = r.uniform_(*self.cfg.command_delay_range)

        rsi = getattr(self._env, "_jump_rsi_phase", None)
        if rsi is not None:
            phase = rsi[env_ids]
            consumed = phase.clone()
            rsi[env_ids] = 0.0
            # Phase p: go happened p*span seconds ago -> go step is negative
            rsi_go = -consumed * self.reference.span / self._env.step_dt
            is_rsi = consumed > 0.0
            self.go_step[env_ids] = torch.where(is_rsi, rsi_go, delay / self._env.step_dt)
        else:
            self.go_step[env_ids] = delay / self._env.step_dt

        self._apex_rise[env_ids] = 0.0
        self._flight_s[env_ids] = 0.0
        self._takeoff_vz[env_ids] = 0.0
        self._travel[env_ids] = 0.0

    def _update_metrics(self) -> None:
        self.metrics["apex_rise"] = self._apex_rise
        self.metrics["flight_s"] = self._flight_s
        self.metrics["takeoff_vz"] = self._takeoff_vz
        self.metrics["travel"] = self._travel

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            # reset path: the spawn state was just written, so no physical metrics accumulate.
            return
        robot = self.robot
        z = robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        torch.maximum(self._apex_rise, z - self._stance_z, out=self._apex_rise)
        # Ground gained along the spawn's x since the reset: the jump's is drift,
        # the dash's (`jumper.dash`) is the point.
        x = robot.data.root_link_pos_w[:, 0] - self._env.scene.env_origins[:, 0]
        torch.maximum(self._travel, x, out=self._travel)

        if self._contact_sensor is None:
            try:
                self._contact_sensor = self._env.scene.sensors["feet_ground_contact"]
            except KeyError:
                return
        forces = self._contact_sensor.data.force
        if forces is None:
            return
        airborne = (forces.norm(dim=-1) <= self.cfg.force_threshold).all(dim=1)
        self._flight_s += airborne.float() * self._env.step_dt
        vz = robot.data.root_link_lin_vel_w[:, 2]
        grounded = ~airborne
        # Boolean indexing returns a copy, so scatter back into the original tensor
        # (out= to the copy is silently lost).
        self._takeoff_vz[grounded] = torch.maximum(
            self._takeoff_vz[grounded], vz[grounded]
        )


__all__ = ["JumpMotionCommand", "JumpMotionCommandCfg"]
