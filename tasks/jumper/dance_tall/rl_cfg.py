"""Hyper-parameters for jumper.dance_tall: jumper.dance's, value for value.

**Only this file changes this task's training hyper-parameters**, and nothing it
does not state is inherited silently.

Two departures from the four velocity tasks' shared starting point, one of which
is not optional.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path:
    `logs/<model>/jumper.dance_tall/<date-time>`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.dance_tall",
        # ── Symmetry: OFF, and this one is not a tuning choice ──
        #
        # The baseline turns left-right symmetry on, and for the velocity tasks it
        # is sound: the hexapod has strict sagittal mirror symmetry, so a policy
        # that walks forward should mirror onto a policy that walks forward, and
        # `mdp/symmetry.py` augments every mini-batch with the mirrored samples.
        #
        # **A choreography has no such symmetry.** Mirroring this task's samples
        # asserts that when the reference raises the left arm, raising the right
        # one is equally correct -- so the augmented half of every mini-batch is
        # labelled with a dance the reference never asks for, and the mirror loss
        # additionally penalises the policy for telling the two apart. The dance is
        # visibly left-right asymmetric (the front arms' yaw ranges are mirror
        # images, but they are traversed at different times), so this is not a
        # borderline case.
        #
        # It would not raise, and it would not obviously diverge: the policy would
        # train to the average of the dance and its mirror image, which is a
        # smaller, blander motion, and the tracking reward would plateau below what
        # the same run without symmetry reaches with no indication why.
        #
        # It also would not even load: `symmetry.py` looks each observation term up
        # by name to know how to mirror it and raises on one it does not recognise,
        # which none of `clip_phase`, `ref_joint_pos`, `ref_joint_vel`, `ref_future`
        # or `ref_tilt_error` is. That is a good failure -- loud, and at startup -- but
        # it is the second reason, not the first.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        # ── Network ──
        # The baseline's shape. The actor is wider at the input than the velocity
        # tasks' (the reference trajectory and its three future horizons), but the
        # mapping it has to learn is not obviously harder -- most of the input is
        # the answer -- so there is no reason yet to depart from a shape that is
        # working on the same robot.
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # `entropy_coef` is the one to watch, and the velocity tasks' note applies
        # with more force here. Measured on jumper.tripod, a rising `Policy/mean_std`
        # cost the precision terms first -- the gait reward fell 21% and the
        # action-rate penalty grew 77% while velocity tracking was flat. This task
        # is *entirely* precision terms, so the same drift would show up as the
        # whole reward sagging. 0.005 rather than the baseline's 0.01 on that
        # reasoning: imitation needs less exploration than locomotion does, because
        # the reference already says what to do.
        #
        # A starting point, not a measurement. If the dance comes out mushy, this
        # is the first thing to halve again -- but establish it on a deterministic
        # replay (`play.py`) first, since `Train/mean_reward` is collected under
        # the exploration noise and falls mechanically as sigma grows.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's motion-tracking runner.

    It is not the velocity tasks' runner and not the default. Two things it does
    that matter here: it logs the tracking metrics the `MotionCommand` accumulates
    (per-body position and orientation error, and the adaptive sampler's entropy,
    which is how you see *which* passages of the dance are still failing), and it
    **bakes the clip into the exported ONNX** so a deployed policy carries the
    reference it was trained against rather than depending on a file beside it.
    """
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
