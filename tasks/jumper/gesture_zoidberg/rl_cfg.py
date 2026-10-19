"""Hyper-parameters for jumper.gesture_zoidberg.

jumper.dance's, value for value, and for the reasons written out in
`tasks/jumper/dance/rl_cfg.py`: the network, PPO and runner settings were chosen for
imitating a clip on this robot, not for that one clip. Written out again rather
than imported, so that changing one clip's training is a change to that clip.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config: `logs/<model>/jumper.gesture_zoidberg/<date-time>`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.gesture_zoidberg",
        # Off, and not a tuning choice: the two arms meet but do not mirror each
        # other exactly, and mirroring the samples would train the average of the
        # bow and its mirror image. `symmetry.py` would also refuse the reference
        # terms.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # Half the baseline's: imitation needs less exploration than locomotion,
        # and a rising `Policy/mean_std` costs precision terms first.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's motion-tracking runner, as jumper.dance: it logs the tracking metrics
    and the adaptive sampler's entropy, and bakes the clip into the exported ONNX."""
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
