"""Environment config for jumper.dance_dream_wings_tall: the dream_wings dance on a taller stance, `media/dream_wings_tall.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## The clip

`media/dream_wings_tall.npz` is `jumper.dance_dream_wings`'s clip passed through
`tools/raise_dance.py --height 0.145 --scale 0.5`: the body's median height moves to
145 mm and its own excursion is halved, so it runs 138-153 mm; the middle and rear
legs are re-solved by IK to keep every support foot where it was. Arms, attitude
and timing are the original's.

## The original clip, measured

Every number in this file is `jumper.dance_dream_wings`'s, unchanged, and the measurements
quoted beside them are of the original clip:

At the 50 Hz control rate, after conversion (55.6 s): per-joint position std 0.238
rad at the median and 0.565 at the most active, RMS amplitude 0.285 rad; joint
velocity RMS 2.417 rad/s, per-joint std 2.513 at the median and 4.162 at the most
active, peaking at 15.7 rad/s; the base between 97 and 127 mm; tilt at most 3.5
degrees.

Kept as they are on purpose, so that the only variable against `jumper.dance_dream_wings` is
the stance. The policy trained here was warm-started from that task's committed
model.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/raise_dance.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is 55.6 s, so an episode is a window into it
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
        # 0.006. Here: 0.44 x 0.285 -> 0.13, standing still 0.008.
        joint_pos_std=0.13,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS, standing still
        # about 0.10. Here: 2.417 -> 1.6, standing still 0.10. Half
        # the position weight, so when they disagree the pose wins.
        joint_vel_std=1.6,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~2.4 rad/s is ~13 W, so -0.02 is
        # about -0.27 a step, 4.1% of the budget.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. **The reference crosses the soft joint limits** (90% of each range,
        # where `joint_limit` starts charging): the rear ankles `LR_J2_joint` /
        # `RR_J2_joint` by up to 0.20 rad on 17% of frames, and the middle hips
        # `LM_J0_joint` / `RM_J0_joint` by 0.057 rad on 20%. Tracking the clip exactly
        # therefore costs 0.061 rad of violation a step on average, -0.61 at this weight,
        # about 9% of the positive budget (jumper.dance's clip: 0.0009). Kept at -10 all the
        # same: the soft limit is what keeps the robot off its stops. If tracking plateaus,
        # this is the first term to look at.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 97-127 mm, a 30 mm band, so 40 mm
        # exceeds all of it and cannot fire on a policy merely tracking badly --
        # jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". The dance itself tilts at
        # most 3.5 degrees.
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
