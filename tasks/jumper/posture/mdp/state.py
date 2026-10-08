"""What the posture command is measured against: twist, pitch, roll, height.

Four scalars, one function each, and every reward and observation in this task
reads them from here. They are kept apart from the rewards deliberately -- the
same four numbers are scored, logged as metrics by the command term, and handed
to the critic as an observation, and three copies of "how the body's twist is
computed" would be three chances to disagree.

## The frame each one is measured in, which is the whole design

    twist    body yaw **relative to the feet**, by least-squares aligning the
             current footprint onto the HOME footprint
    pitch    body pitch relative to **gravity**
    roll     body roll relative to gravity
    height   base height above the **mean of the feet**

Twist and height are foot-relative because there is nothing else they could mean:
a hexapod standing still can yaw its body over planted feet, and that is the
quantity being commanded, not its heading in the world. `heading` already exists
for the world one and is what `twist`'s `ang_vel_z` integrates to.

Pitch and roll are gravity-relative rather than foot-relative, and that is a
choice with a cost. On the plane these tasks train on the two agree exactly -- the
feet are on z = 0, so the foot plane *is* the horizontal plane. On a slope they
would not, and a foot-plane version is the one that keeps meaning "lean relative
to the ground you are standing on". Gravity is chosen anyway because it is the
one a real robot can measure: an IMU gives pitch and roll directly, and a foot
plane has to be reconstructed from leg kinematics and contact state. When this
task moves onto `--scene rough`, this is the first paragraph to come back to.

## Why the twist needs a reference footprint at all

The feet move. A yaw of the body over a footprint is only defined against some
footprint that counts as untwisted, and the natural one is the stance the robot
holds at HOME -- `constants.NOMINAL_FOOT_XY`, measured off the model and pinned
by `tests/test_posture_stance.py`.

Given that reference, the estimate is a two-dimensional orthogonal Procrustes
problem, which in the plane has a closed form rather than needing an SVD:

    theta = atan2( sum_i (n_i x p_i),  sum_i (n_i . p_i) )

where `p_i` is foot `i` in the body frame with the footprint's centroid removed
and `n_i` is the same for the reference. `theta` is the yaw of the *stance*
relative to the body, so the body's twist relative to the stance is `-theta`.

**Centroid removal is what makes it a rotation estimate rather than a translation
one.** Without it, a robot that has simply shifted its body forward over its feet
-- which is most of what walking is -- reads as a twist, because the whole
footprint is then offset in x and the cross terms do not cancel.

**Swinging legs are included, and that is deliberate.** The obvious refinement is
to weight by contact, so only planted feet vote; it was not done, for two reasons.
A tripod gait has three feet in the air half the time and their positions are a
continuous function of the same joints, so they carry the same yaw information
with a lead rather than a bias -- and a contact-weighted estimate changes its
divisor six times a cycle, which puts a step discontinuity into a reward the
policy is being asked to hold constant. What the airborne feet do add is cycle
ripple, which is real and is the reason `track_twist`'s std is not tight; see the
note there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common.constants import FOOT_SITE_Z, LEGS, NOMINAL_FOOT_XY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The reference footprint, cached per device. A [6, 2] tensor with the centroid
#: already removed, which is the only form anything here uses.
_REFERENCE: dict[torch.device, torch.Tensor] = {}


def reference_footprint(device: torch.device) -> torch.Tensor:
    """`NOMINAL_FOOT_XY` as a centred [6, 2] tensor on `device`."""
    cached = _REFERENCE.get(device)
    if cached is None:
        ref = torch.tensor(NOMINAL_FOOT_XY, dtype=torch.float32, device=device)
        cached = ref - ref.mean(dim=0, keepdim=True)
        _REFERENCE[device] = cached
    return cached


#: Where "this config's site order has been checked" is remembered: **on the
#: config object**, not in a set of `id()`s.
#:
#: The set came first and was unsound, for the reason `common/mdp/rewards.py`
#: records in full: CPython reuses a collected object's address -- measured, five
#: `SceneEntityCfg`s created and dropped in a loop all reported the same `id` --
#: so a fresh config can find itself already in the set and skip the check.
#:
#: Here that only skips a check, where the same bug in the actuator terms paired
#: torques with another entity's joints. It is fixed the same way anyway: an
#: attribute lives and dies with the object it describes.
_CHECKED_ATTR = "_mjrl_foot_order_checked"


def _check_foot_order(asset, asset_cfg: SceneEntityCfg) -> None:
    """The selected sites must be the six feet in `LEGS` order.

    **This is the one way the twist estimate fails silently.** It aligns the
    footprint onto `NOMINAL_FOOT_XY` row by row, so a permuted selection is a
    perfectly well-formed least-squares fit of the robot's feet onto a different
    robot's -- same shapes, same units, an answer that looks like an angle. The
    two ways to get there are a `SceneEntityCfg` built without
    `preserve_order=True` and a model whose sites are reordered or renamed.
    """
    names = tuple(asset.site_names)
    if getattr(asset_cfg, _CHECKED_ATTR, None) == names:
        return
    ids = asset_cfg.site_ids
    selected = tuple(names) if isinstance(ids, slice) else tuple(names[i] for i in ids)
    if selected != tuple(LEGS):
        raise ValueError(
            f"the posture terms need the six foot sites in LEGS order "
            f"{tuple(LEGS)}, got {selected}. Build the SceneEntityCfg with "
            f"site_names=LEGS and preserve_order=True."
        )
    # Remembered with the site list it was checked against, so the same config
    # reused on a different entity is checked again rather than trusted.
    setattr(asset_cfg, _CHECKED_ATTR, names)


def foot_pos_w(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The selected sites' world positions, [num_envs, n_sites, 3].

    **`EntityData.site_pos_w` is the obvious way to write this and it raises on
    the native backend.** That property is `site_pose_w[..., 0:3]`, and
    `site_pose_w` builds the pose by running `site_xmat` through
    `quat_from_matrix` -- which wants `[..., 3, 3]`. mjwarp supplies that shape;
    `mjrl/backend/native_sim.py` allocates `site_xmat` as `(N, nsite, 9)`, the
    flat layout MuJoCo itself uses, so the call fails with "Invalid rotation
    matrix shape [N, nsite, 9]". Nothing in this repository had used a site pose
    before, so the divergence had never been reached.

    `site_xpos` is a 3-vector on both backends and is what the property reads
    anyway, so taking it directly is the same number with none of that in the
    way. When the backends agree on `site_xmat` this can go back to `site_pos_w`.
    """
    data = env.scene[asset_cfg.name].data
    return data.data.site_xpos[:, data.indexing.site_ids][:, asset_cfg.site_ids]


def foot_pos_b(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The six foot sites in the body frame, [num_envs, 6, 3].

    `site_pos_w` is in the world, so this rotates by the base's orientation
    rather than subtracting a yaw: a pitched or rolled body is exactly the case
    this task creates, and a yaw-only transform would leak that lean into the
    footprint's x and y.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    asset = env.scene[asset_cfg.name]
    _check_foot_order(asset, asset_cfg)
    feet_w = foot_pos_w(env, asset_cfg)  # [B, 6, 3]
    base_w = asset.data.root_link_pos_w.unsqueeze(1)  # [B, 1, 3]
    rot = matrix_from_quat(asset.data.root_link_quat_w)  # [B, 3, 3]
    return torch.einsum("bij,bfj->bfi", rot.transpose(1, 2), feet_w - base_w)


def twist_from_footprint(xy: torch.Tensor) -> torch.Tensor:
    """The body's yaw over a footprint, in radians, from [.., 6, 2] body-frame
    foot positions.

    Split out from `body_twist` so that it can be tested without a simulator --
    `tests/test_posture.py` feeds it footprints rotated by known angles.
    The estimator is the closed-form planar Procrustes fit the module docstring
    derives; the centroid is removed here, so callers pass raw body-frame
    positions.
    """
    p = xy - xy.mean(dim=-2, keepdim=True)
    n = reference_footprint(p.device)
    cross = (n[:, 0] * p[..., 1] - n[:, 1] * p[..., 0]).sum(dim=-1)
    dot = (n[:, 0] * p[..., 0] + n[:, 1] * p[..., 1]).sum(dim=-1)
    # `atan2(cross, dot)` is the stance yawed relative to the body; the body
    # relative to the stance is its negative.
    return -torch.atan2(cross, dot)


def body_twist(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """How far the body is yawed over its feet, in radians, [num_envs].

    Positive is the body turned to its **left** (counter-clockwise seen from
    above), which is the right-hand rule about +z and the same sense as the
    `twist` command's `ang_vel_z`. The two are independent all the same: yawing
    the body over planted feet is this, and yawing the whole robot including its
    feet is that.

    See the module docstring for the estimator and for why the centroid is
    removed.
    """
    return twist_from_footprint(foot_pos_b(env, asset_cfg)[..., :2])


def body_tilt(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pitch and roll from the gravity direction, in radians, [num_envs, 2].

    `projected_gravity_b` is the world's down direction expressed in the body
    frame, `g = (sin, -cos p sin r, -cos p cos r)` for pitch `p` and roll `r`, so

        pitch = atan2(g_x, hypot(g_y, g_z))       nose down is positive
        roll  = atan2(-g_y, -g_z)                 left side up is positive

    **Both signs follow the right-hand rule about the body's own axes** (+y is to
    the left, +z is up): a positive turn about +y takes +x towards -z, which is
    the nose going down. The mirror rule in `common/mdp/symmetry.py` -- negate
    roll, keep pitch -- holds under any sign convention. The pitch is the opposite
    of the aerospace one, where z points down; getting that backwards costs
    nothing at training time and inverts the stick on the robot.

    **Pitch was nose-up positive until 2026-09-26**, which the paragraph above
    claimed was the right-hand rule and was not. It was flipped to agree with
    `jumper.five_foot`, whose `mdp/pose_command.py::base_pitch_roll` is `asin(g_x)`
    -- the same number as this for a unit `g` -- because the two tasks' pose
    commands reach the robot through one channel, `Command::base_pitch`, and a
    sign that differs by mode is a stick that tips the nose one way in one policy
    and the other way in the next. Measured on the built env, a base turned +10
    degrees about +y (nose down) reads +10 here and +10 there; before the flip it
    read -10 here. **Every posture checkpoint trained before the flip reads its
    pitch channel inverted**, and re-exporting one with this code hands it a
    contract whose stick leans it backwards.

    `atan2` rather than `asin(g_x)` so that the expression is stable at large
    lean and does not need the clamp a bare `asin` needs against a unit vector
    that is only unit to float precision.
    """
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(g[:, 0], torch.hypot(g[:, 1], g[:, 2]))
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    return torch.stack((pitch, roll), dim=-1)


def body_height(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Base height above the ground the feet are on, in metres, [num_envs].

    The mean of the six foot sites' world z, subtracted from the base's, plus
    `FOOT_SITE_Z`. That last term is not cosmetic: the site is at the centre of
    the silicone pad rather than at its contact point, so without it the
    measurement reads 8.9 mm short and a command of 0.107 m would not be the
    robot standing at `STAND_Z`. With it, HOME measures `STAND_Z` exactly, and
    the range in `env_cfg.py` can be quoted in the units every other height in
    this repository is quoted in.

    **Above the feet rather than above the terrain, on purpose.** They agree on a
    plane and diverge the moment the ground is not one, and on a slope it is the
    feet that the body's height is a posture of. It also needs no height sensor,
    which keeps the same quantity available to a reward and to a metric without
    either depending on `foot_height_scan`'s configuration.

    Airborne feet are included, for the reason the module docstring gives for the
    twist: they are half the feet in a tripod gait, they move continuously with
    the same joints, and excluding them would put a six-times-a-cycle step into
    the measurement. The cost is the same cycle ripple, and `track_height`'s std
    is sized for it.
    """
    asset = env.scene[asset_cfg.name]
    feet_z = foot_pos_w(env, asset_cfg)[..., 2]
    return asset.data.root_link_pos_w[:, 2] - feet_z.mean(dim=1) + FOOT_SITE_Z


def posture_state(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """All four, in the command's own order: (twist, pitch, roll, height).

    [num_envs, 4], and the order is load-bearing -- it is the layout the command
    term produces, the reward terms index into and `posture4` in
    `common/mdp/symmetry.py` mirrors by position.
    """
    tilt = body_tilt(env, asset_cfg)
    return torch.stack(
        (
            body_twist(env, asset_cfg),
            tilt[:, 0],
            tilt[:, 1],
            body_height(env, asset_cfg),
        ),
        dim=-1,
    )


__all__ = [
    "body_height",
    "body_tilt",
    "body_twist",
    "foot_pos_b",
    "foot_pos_w",
    "posture_state",
    "reference_footprint",
    "twist_from_footprint",
]
