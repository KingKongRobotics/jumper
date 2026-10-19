"""jumper.dash's environment: jumper.jump's, playing `ref/dash_flat.npz`.

The reference is the one thing that differs, and it is named in two places:
the jump's command term loads the file this config names (every reward and
observation reads the reference through that term, so that is the whole of
training), and the contract the export writes loads a recording of its own
(`reference_contract`, which the jump's config closes over its own file). The
first dash export forgot the second and shipped `high_jump_flat` under the
dash's policy -- the bundler refused it only because the jump ships a file
of the same name, so the contract here is built from the dash's recording
by the same call. Resume from the jump's checkpoint:

    python scripts/train.py --task jumper.dash --headless --max-iterations 1500 --num_envs 2048 \\
        --checkpoint tasks/jumper/jump/out/example/model_3100.pt
"""
from __future__ import annotations

from pathlib import Path

from ..jump import env_cfg as _jump

DASH_REF_NPZ = Path(__file__).resolve().parent / "ref" / "dash_flat.npz"


def env_cfg(*args, **kwargs):
    cfg = _jump.env_cfg(*args, **kwargs)
    cfg.commands["jump"].motion_file = str(DASH_REF_NPZ)
    # The jump's weights hold the dash to the jump's shape: trained with them as they
    # were (1500 iterations resumed from the jump's policy, 2048 envs, RTX 3060),
    # the hop gained 0.10 m a time in the duel against the 0.47 m of its reference
    # (tools/battle/pilots/brain.py logs the range before and after). The reference's
    # joint angles are the jump's, so the joint and foot terms score standing in
    # place as well as hopping forward, and only the velocity term, at sigma_sq 0.30
    # (0.72 m/s short scores 0.18) over the 0.42 s flight of a 3.5 s episode, asks
    # for the travel. The dash owns these three numbers; everything else is the jump's.
    cfg.rewards["ref_base_lin_vel"].weight = 6.0
    cfg.rewards["ref_base_lin_vel"].params["sigma_sq"] = 0.08
    cfg.rewards["ref_joint_pos"].weight = 1.5

    def _reference_contract() -> dict:
        # The jump's, with the dash's recording: see `_jump.env_cfg` for why the
        # joint order is this robot's rather than the file's.
        ref = _jump.jump_mdp.reference.JumpReference(DASH_REF_NPZ, "cpu", list(_jump.HOME))
        return ref.deploy_contract(
            _jump._FUTURE_OFFSETS, _jump._SIM_TIMESTEP * _jump._DECIMATION,
            starts_on_entry=_jump._STARTS_ON_ENTRY,
        )

    cfg.reference_contract = _reference_contract
    return cfg
