"""Environment config for jumper.posture_tall: jumper.posture carried high.

`jumper.posture` lets the operator command a body height of 0.07-0.15 m. Its
default stance keeps the middle and rear legs well bent, and at 0.15 m the body
is still close to the ground for a robot whose legs reach about 0.17 m. This task
is `jumper.posture` built unchanged and then edited in four places, so the diff
against it is the whole of what this task is:

    the height range          0.07-0.15 -> 0.10-0.18 m, with the curriculum's
                              height rungs, the height std and `pose`'s height
                              reach rederived from it exactly as the parent
                              derives them
    the height draw           `TALL_SHARE` of the draws come from
                              `[TALL_LOW, current top]` (`mdp/commands.py`)
    track_height              weight 1.0 -> 3.0, plus a coarse twin with a 3 cm std
    leg_straightness          new: the middle and rear legs' extension, scored
                              only when the command asks for >= 0.15 m

Everything not listed is `jumper.posture`'s, imported rather than copied, so a
fix there reaches this task.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..posture import env_cfg as parent
from ..posture.mdp.curriculum import POSTURE_STD_SCALES, height_std, posture_levels
from ..posture.mdp.teleop import TeleopPostureCommandCfg
from .mdp.commands import TallPostureCommandCfg, TallTeleopPostureCommandCfg
from .mdp.rewards import leg_straightness

POSTURE_COMMAND = parent.POSTURE_COMMAND

#: The commanded height range, metres. The parent's is (0.07, 0.15).
HEIGHT_RANGE = (0.10, 0.18)

#: The parent's ladder recomputed for the new range; the angle rungs do not depend
#: on the height and come out identical to the parent's.
POSTURE_ANGLES, POSTURE_HEIGHTS = posture_levels(parent.MOVE_LEAN, HEIGHT_RANGE, parent.STAND_Z)

#: Share of resampled height commands redrawn from `[TALL_LOW, current top]`.
#: 0.7 because the first run at a uniform draw spent most of its time below
#: 0.15 m and learned to give up height for speed.
TALL_SHARE = 0.7
TALL_LOW = 0.16

#: Weight on the parent's `track_height` (1.0 there). On the first 1500-iteration
#: run at 1.0 the policy traded height for speed.
TRACK_HEIGHT_WEIGHT = 3.0

#: A second height term with a 3 cm std. With 1-2 cm of error left, the parent's
#: 1 cm std has almost no gradient; this one still pulls.
TRACK_HEIGHT_COARSE_STD = 0.03
TRACK_HEIGHT_COARSE_WEIGHT = 1.5

#: Legs scored by `leg_straightness`, as (thigh body, foot body) pairs. The front
#: legs are arms with a claw and are left alone.
STRAIGHT_LEGS = (
    ("LM_thigh_link", "LM_foot_tip_link"),
    ("RM_thigh_link", "RM_foot_tip_link"),
    ("LR_thigh_link", "LR_foot_tip_link"),
    ("RR_thigh_link", "RR_foot_tip_link"),
)

#: Fully extended length of each leg in `STRAIGHT_LEGS`, metres: the largest
#: distance from the J1 joint anchor to the foot body over the joint ranges in
#: `assets/jumper/jumper.xml`, found by bounded L-BFGS-B from 80 random starts
#: per leg. 0.1711 for all four (0.1211 / 0.1194 at HOME, a ratio of 0.70).
STRAIGHT_REACH = (0.1711, 0.1711, 0.1711, 0.1711)

#: Commanded height from which `leg_straightness` is scored. Below it the robot
#: is being asked to crouch and bent legs are correct.
STRAIGHT_MIN_HEIGHT = 0.15
STRAIGHT_WEIGHT = 1.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.posture`'s for the arguments."""
    cfg = parent.env_cfg(asset=asset, play=play)

    # ── The command: new range, biased draw ───────────────────────────────
    old = cfg.commands[POSTURE_COMMAND]
    kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
    kwargs["ranges"] = replace(old.ranges, height=HEIGHT_RANGE)
    cls = TallTeleopPostureCommandCfg if isinstance(old, TeleopPostureCommandCfg) else TallPostureCommandCfg
    cfg.commands[POSTURE_COMMAND] = cls(**kwargs, tall_share=TALL_SHARE, tall_low=TALL_LOW)

    # ── Height tracking ───────────────────────────────────────────────────
    track = cfg.rewards["track_height"]
    track.weight = TRACK_HEIGHT_WEIGHT
    track.params["std"] = height_std(-1, POSTURE_HEIGHTS, POSTURE_STD_SCALES)
    cfg.rewards["track_height_coarse"] = RewardTermCfg(
        func=track.func,
        weight=TRACK_HEIGHT_COARSE_WEIGHT,
        params={**track.params, "std": TRACK_HEIGHT_COARSE_STD},
    )
    cfg.rewards["leg_straightness"] = RewardTermCfg(
        func=leg_straightness,
        weight=STRAIGHT_WEIGHT,
        params={
            "min_height": STRAIGHT_MIN_HEIGHT,
            "command_name": POSTURE_COMMAND,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[b for pair in STRAIGHT_LEGS for b in pair], preserve_order=True
            ),
            "reach": STRAIGHT_REACH,
        },
    )

    # ── `pose`'s height reach is half the height range, as in the parent ──
    reach = cfg.rewards["pose"].params["posture_reach"]
    cfg.rewards["pose"].params["posture_reach"] = (*reach[:3], (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) / 2.0)

    # ── The posture curriculum climbs the new height rungs ────────────────
    if POSTURE_COMMAND in cfg.curriculum:
        cfg.curriculum[POSTURE_COMMAND].params["angles"] = POSTURE_ANGLES
        cfg.curriculum[POSTURE_COMMAND].params["heights"] = POSTURE_HEIGHTS

    return cfg
