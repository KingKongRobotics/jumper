"""Environment config for jumper.gesture_pump: the pump, `media/pump.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. The numbers are `jumper.gesture_zoidberg`'s, a clip of the same kind (both
front arms up, the four legs standing, the claws snapping), which trained; the two
stds are this clip's own (`tools/battle/make_pump.py` prints them: 10.5 s at
50 Hz with the leads, RMS amplitude 0.838 rad, velocity RMS 1.53 rad/s; the base bounces between 92 and 107 mm).
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/import_wbc_gestures.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. Half the 10.5 s clip, jumper.dance_maze's rule for a
#: short clip: the command teleports the robot back onto the reference wherever an
#: episode runs off the end, and at half the clip an episode sampled in its first
#: half runs clean.
EPISODE_S = 5.3


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude. Here: 0.44 x 0.838 -> 0.37.
        joint_pos_std=0.37,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 1.53 -> 1.02.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=1.02,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.03,     # jumper.gesture_bow's -0.1: the first run held the claws open rather than
                                      # pay for snapping them (joint error plateaued at 0.94 = the two claws)
        action_acc_weight=-0.02,
        # Power: 22 joints at ~0.25 N*m and this clip's ~0.56 rad/s is ~3.1 W, so
        # -0.02 is about -0.06 a step, 0.9% of the budget -- jumper.dance's weight,
        # which a gentler clip leaves smaller in proportion.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference stays inside the soft joint limits throughout,
        # so this term charges a faithful policy nothing.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        #
        # **This clip asks for contact.** From 3.0 to 5.1 s the table presses the two
        # claws into each other, up to 0.27 rad past where they meet on the v1.6.1
        # model (the tool's docstring has the run that found it). The rule is kept:
        # counted per substep, pressing through all 2.1 s would cost up to 4 a step,
        # so the policy will bring the claws together and stop at touching. That is
        # the gesture, and it is the one a servo pressing a claw into a claw should
        # make; the joint-position term pays for the difference.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 92-107 mm, a 15 mm band, so 40 mm exceeds all of it and cannot fire on a policy merely tracking
        # badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over"; the clip stays level.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
