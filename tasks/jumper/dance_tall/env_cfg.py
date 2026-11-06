"""Environment config for jumper.dance_tall: jumper.dance on a taller stance, `media/demo_tall.npz`.

The clip is `jumper.dance`'s raised by `tools/raise_dance.py --height 0.145 --scale
0.5` (body 141-157 mm, median 145; the original is 105-137, median 112).

Every number below is `jumper.dance`'s, unchanged, with the reasoning that file
gives for it; where a comment quotes a clip measurement, it is the original
clip's. Kept that way on purpose, so the stance is the only variable against
`jumper.dance`. The policy trained here was warm-started from that task's
committed model.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/raise_dance.py`. See `jumper.dance`'s
#: `media/README.md` for what may go here.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is far longer than this (the sample is
#: 235.7 s), so an episode is a window into it and the command's adaptive sampler
#: chooses where the window starts -- weighting the passages that have been failing.
#: 10 s at 129.88 bpm is about 21 beats, enough for several complete moves.
EPISODE_S = 10.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
            The clip is written against the default asset's joints, so a
            structurally different model fails during conversion, by name.
        play: replay mode -- observation noise and disturbances off, the clip
            played from its first frame rather than sampled, unbounded episodes.
    """
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # mjlab's ranges put the root up to 0.05 m and 0.2 rad off the reference
        # at every reset, which for a machine 0.105 m tall and 0.3 m across is
        # most of a body length and a visible tilt -- an episode would begin
        # closer to falling than to dancing. These are the same fractions of
        # *this* robot: 10 mm is a tenth of its standing height, and 0.05 rad is
        # under half the pitch the dance itself uses.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        # Joint jitter at reset. mjlab's default is +/-0.52 rad and its G1 config
        # narrows that to +/-0.1; +/-0.05 is a fifth of this robot's action scale,
        # so a reset perturbation is something one control step can correct rather
        # than a pose the policy has to recover from before it can start dancing.
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, inherited as they are. Neither is measured on
        # this robot or in `constants.py`, and the walking tasks inherit a
        # different joint-velocity default (+/-1.5) from their skeleton.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # mjlab's six tracking terms keep its weights and stds unchanged; they are
        # the BeyondMimic recipe and there is no measurement here that would
        # justify moving them. What follows replaces the regularisers.
        #
        # Positive budget: 0.5 + 0.5 + 1 + 1 + 1 + 1 = 5.0 from the six, plus 1.5
        # from the two joint-space terms below = **6.5** per step, which is the
        # scale every negative weight further down is set against.
        #
        # That number changed. It was 5.0, and the negative weights were chosen
        # against 5.0, so adding 1.5 of positive reward makes every one of them 23%
        # weaker in relative terms -- `torque_headroom` and `joint_limit` most
        # noticeably, since they are the two doing real constraining. Recorded here
        # rather than silently absorbed: if the robot starts reaching its stops or
        # sitting on the torque ceiling after this change, this paragraph is why,
        # and the fix is those weights and not these.
        #
        # 跟对姿势 -- joint-space position tracking.
        #
        # MEASURED, from the clip itself (235.7 s, 129.88 bpm, all 22 joints
        # moving): per-joint position std is 0.213 rad at the median and 0.734 at
        # the most active; the RMS amplitude across joints is 0.342 rad.
        #
        # std 0.15 is a little under half that RMS (0.44 of it). The kernel is
        # exp(-mse/std^2), so this reads: full marks inside about 0.15 rad (8.6
        # degrees) RMS, and 0.006 for a robot standing still through the whole
        # clip. A std at the motion's own amplitude would score 1/e for standing
        # still, which is a term that pays for doing nothing.
        joint_pos_std=0.15,
        joint_pos_weight=1.0,
        # 跟上节拍 -- the timing, which position alone does not carry: every pose
        # reached late and slowly satisfies the term above and is not a dance.
        #
        # MEASURED: joint velocity RMS 1.517 rad/s, per-joint std 1.269 at the
        # median and 2.850 at the most active, peaking at 12.9 rad/s.
        #
        # std 1.0 is about two thirds of that RMS -- 0.10 for standing still, so it
        # still separates, but looser than the position term because velocity is
        # the noisier signal of the two and because this one argues with
        # `action_rate_l2` / `action_acc_l2` below. Those suppress motion that is
        # fast for no reason; this asks for motion that is fast for a reason. Half
        # the position weight, so when they disagree the pose wins.
        joint_vel_std=1.0,
        joint_vel_weight=0.5,
        # 平滑不抖动 -- smoothness, in two orders.
        #
        # -0.1 is the velocity skeleton's calibrated value **at this action scale**
        # (0.25); the relation is quadratic, so it transfers unchanged only because
        # the scale is the same. See `_ACTION_RATE_REF` in `common/velocity_env.py`.
        action_rate_weight=-0.1,
        # Starting point, not a measurement. Half of `action_rate_l2`: the second
        # difference of a bounded signal is the larger of the two for anything
        # jittery and the smaller for anything smooth, so an equal weight would
        # dominate early while the policy is still noisy and suppress exploration.
        action_acc_weight=-0.05,
        # 能耗低 -- electrical power, positive part only (regeneration is not
        # rewarded).
        #
        # Starting point. Sized from the clip: 22 joints at the standing torque of
        # ~0.25 N*m and the reference's typical joint speed of ~1.5 rad/s is on
        # the order of 8 W, so -0.02 puts a typical step near -0.16, about 3% of
        # the positive budget -- present enough to break ties between two ways of
        # hitting the same pose, small enough not to buy its way out of dancing.
        power_weight=-0.02,
        # 关节扭矩峰值小 -- torque above what the servo can hold.
        #
        # Starting point, but the scale is anchored: one joint at the plateau
        # (1.7464 N*m) is 0.546 N*m over the continuous rating, so it contributes
        # 0.298 and costs 0.6 at this weight -- 12% of the positive budget for a
        # single saturated actuator, and nothing at all for a policy that stays
        # under 1.2.
        torque_headroom_weight=-2.0,
        # Joint limits and self-collision, kept from mjlab but re-weighted for a
        # robot whose limbs are 0.16 m on a 1.75 N*m actuator: its worst-case
        # leg-on-leg push is about 12.5 N, so mjlab's 10 N threshold would fire
        # only at near-saturation and let every ordinary collision through. 1.0 N
        # is the velocity skeleton's measured choice, ~8% of that ceiling.
        joint_limit_weight=-10.0,
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations: 不跌倒 ──────────────────────────────────────────
        # Every threshold here replaces one chosen for a 1.3 m humanoid.
        #
        # Root height: the dance moves the body through 105-137 mm, a 32 mm band.
        # 40 mm of error therefore exceeds the entire vertical range of the
        # choreography, so it cannot fire on a policy that is merely tracking
        # badly.
        anchor_height_error=0.04,
        # Orientation: the term compares the z component of projected gravity, so
        # the threshold is 1 - cos(tilt error) and 0.3 is 45.6 degrees. Measured by
        # rolling the base against an upright reference: 30 and 45 degrees do not
        # fire, 60 does. The dance itself reaches 16 degrees of pitch (0.04), so
        # this is well clear of the choreography and squarely at "has fallen over".
        #
        # It is the backstop rather than the usual cause: under a flailing policy
        # the support-foot term fires first almost every time, because a hexapod
        # loses foot contact well before it tilts 45 degrees. What this catches is
        # the case that one cannot -- a foot staying planted while the body goes
        # over.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the robot's standing height
        # and means it is no longer standing on that leg.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # mjlab pushes every 1-3 s with up to 0.5 m/s. On a 2 kg machine holding a
        # precise standing pose that is a shove, not a disturbance, and it would
        # arrive roughly twice per move. Gentler and rarer: enough to keep the
        # policy from learning an open-loop replay, not so much that the
        # choreography is noise.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
