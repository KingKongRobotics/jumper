# Deployment bundle: jumper.gesture_sleep

Source checkpoint: `/home/andrew/Documents/jumper/logs/jumper/jumper.gesture_sleep/2026-10-13_08-02-39/model_600.pt`

Generated from a live environment by `scripts/export.py`.
**Do not edit by hand** -- re-export whenever the config changes; values
edited by hand always drift.

## Files

| File | Contents |
|---|---|
| `actor.onnx` | `[1, 209] -> [1, 22]`; takes raw observations, returns the mean action |
| `layout.json` | The deployment contract. Read every number below from it rather than copying them into code |

## Assembling the observation

Length 209, concatenated in the order below. Joint quantities are always ordered by `action_joint_order` (22 of them), **not** by `obs_joint_order`.

| Range | Term | Dim |
|---|---|---|
| `[0:2]` | clip_phase | 2 |
| `[2:24]` | ref_joint_pos | 22 |
| `[24:46]` | ref_joint_vel | 22 |
| `[46:112]` | ref_future | 66 |
| `[112:115]` | ref_tilt_error | 3 |
| `[115:118]` | base_ang_vel | 3 |
| `[118:121]` | projected_gravity | 3 |
| `[121:143]` | joint_pos | 22 |
| `[143:165]` | joint_vel | 22 |
| `[165:187]` | actions | 22 |
| `[187:209]` | joint_torque | 22 |

Observation normalisation is already inside the ONNX (`normalization = baked_into_onnx`), so **feed raw values and do not normalise again**.

## Applying the action

```
for i, joint in enumerate(contract['action_joint_order']):
    target[joint] = action[i] * action_scale + default_joint_pos[joint]
```

Then track with PD: kp = 10.0, kd = 0.5, torque limit 1.746399998664856 N*m, control rate 50.0 Hz.

Joints in `unactuated_joints` are not policy-controlled and **must be locked at the given positions**: they take part in collision, and freeing them changes the foot contact geometry.

## Read before deploying

- Action transform: target = action * action_scale + default_joint_pos[joint], paired joint by joint in action_joint_order.
- In simulation, apply_actions also subtracts encoder_bias (a domain randomisation term modelling encoder offset). Do not reproduce it on hardware.
- Joints in unactuated_joints are not policy-controlled and must be locked at the given positions: they take part in collision, and freeing them changes the foot contact geometry.
- The ONNX already contains observation normalisation; feed raw observations.
