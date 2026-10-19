# Deployment bundle: jumper.dash

Source checkpoint: `/home/andrew/Documents/jumper/logs/jumper/jumper.dash/2026-10-14_20-53-48/model_6598.pt`

Generated from a live environment by `scripts/export.py`.
**Do not edit by hand** -- re-export whenever the config changes; values
edited by hand always drift.

## Files

| File | Contents |
|---|---|
| `actor.onnx` | `[1, 167] -> [1, 20]`; takes raw observations, returns the mean action |
| `layout.json` | The deployment contract. Read every number below from it rather than copying them into code |

## Assembling the observation

Length 167, concatenated in the order below. Joint quantities are always ordered by `action_joint_order` (20 of them), **not** by `obs_joint_order`.

| Range | Term | Dim |
|---|---|---|
| `[0:3]` | base_ang_vel | 3 |
| `[3:6]` | projected_gravity | 3 |
| `[6:26]` | joint_pos | 20 |
| `[26:46]` | joint_vel | 20 |
| `[46:66]` | actions | 20 |
| `[66:67]` | jump_phase | 1 |
| `[67:167]` | ref_future | 100 |

Observation normalisation is already inside the ONNX (`normalization = baked_into_onnx`), so **feed raw values and do not normalise again**.

## Applying the action

```
for i, joint in enumerate(contract['action_joint_order']):
    target[joint] = action[i] * action_scale + default_joint_pos[joint]
```

Then track with PD: kp = 20.0, kd = 0.5, torque limit 1.746399998664856 N*m, control rate 200.0 Hz.

Joints in `unactuated_joints` are not policy-controlled and **must be locked at the given positions**: they take part in collision, and freeing them changes the foot contact geometry.

## Read before deploying

- Action transform: target = action * action_scale + default_joint_pos[joint], paired joint by joint in action_joint_order.
- In simulation, apply_actions also subtracts encoder_bias (a domain randomisation term modelling encoder offset). Do not reproduce it on hardware.
- Joints in unactuated_joints are not policy-controlled and must be locked at the given positions: they take part in collision, and freeing them changes the foot contact geometry.
- The ONNX already contains observation normalisation; feed raw observations.
