"""Environment config for jumper.dance_waist_tall: the waist (waist-twist) dance on a taller stance, `media/waist_tall.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## The clip

`media/waist_tall.npz` is `jumper.dance_waist`'s clip passed through
`tools/raise_dance.py --height 0.145 --scale 0.5`: the body's median height moves to
145 mm and its own excursion is halved, so it runs 140-155 mm; the middle and rear
legs are re-solved by IK to keep every support foot where it was. Arms, attitude
and timing are the original's.

## The original clip, measured

Every number in this file is `jumper.dance_waist`'s, unchanged, and the measurements
quoted beside them are of the original clip:

At the 50 Hz control rate, after conversion (56.6 s): per-joint position std 0.140
rad at the median and 0.488 at the most active, RMS amplitude 0.219 rad; joint
velocity RMS 1.465 rad/s, per-joint std 0.958 at the median and 2.609 at the most
active, peaking at 8.7 rad/s; the base between 107 and 138 mm; tilt at most 7.6
degrees.

Kept as they are on purpose, so that the only variable against `jumper.dance_waist` is
the stance. The policy trained here was warm-started from that task's committed
model.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/raise_dance.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is 56.6 s, so an episode is a window into it
#: and the adaptive sampler chooses where it starts. 10 s, as jumper.dance.
EPISODE_S = 10.0


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
        # amplitude, which scores a robot standing still through the clip at
        # 0.006. Here: 0.44 x 0.219 -> 0.10, standing still 0.008.
        joint_pos_std=0.10,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS, standing still
        # about 0.10. Here: 1.465 -> 1.0, standing still 0.12. Half
        # the position weight, so when they disagree the pose wins.
        joint_vel_std=1.0,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~1.5 rad/s is ~8 W, so -0.02 is
        # about -0.16 a step, 2.5% of the budget -- jumper.dance's figure.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. The reference stays inside the soft joint limits (at most 1 mrad
        # past, on 1.4% of frames), so this term charges a faithful policy nothing.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 107-138 mm, a 31 mm band, so 40 mm
        # exceeds all of it and cannot fire on a policy merely tracking badly --
        # jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". The dance itself tilts at
        # most 7.6 degrees.
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
