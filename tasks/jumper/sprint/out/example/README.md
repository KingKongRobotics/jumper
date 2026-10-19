# Deployment bundle: jumper.sprint

Source checkpoint: `/home/andrew/Documents/jumper-battle/logs/jumper/jumper.sprint/2026-10-14_14-51-08/model_77799.pt`

Generated from a live environment by `scripts/export.py`.
**Do not edit by hand** -- re-export whenever the config changes; values
edited by hand always drift.

## Files

| File | Contents |
|---|---|
| `actor.onnx` | `[1, 415] -> [1, 20]`; takes raw observations, returns the mean action |
| `layout.json` | The deployment contract. Read every number below from it rather than copying them into code |

## Assembling the observation

Length 415, concatenated in the order below. Joint quantities are always ordered by `action_joint_order` (20 of them), **not** by `obs_joint_order`.

| Range | Term | Dim |
|---|---|---|
| `[0:3]` | base_ang_vel | 3 |
| `[3:6]` | projected_gravity | 3 |
| `[6:106]` | joint_pos | 100 |
| `[106:206]` | joint_vel | 100 |
| `[206:306]` | actions | 100 |
| `[306:309]` | velocity_commands | 3 |
| `[309:409]` | joint_torque | 100 |
| `[409:411]` | gait_phase | 2 |
| `[411:415]` | posture_command | 4 |

### The history is strided -- read this before building it

These terms stack several frames, and **the frames are not consecutive control steps**:

| Term | Frames | Stride | Spans |
|---|---|---|---|
| joint_pos | 5 | 4 | 80 ms (16 control steps) |
| joint_vel | 5 | 4 | 80 ms (16 control steps) |
| actions | 5 | 4 | 80 ms (16 control steps) |
| joint_torque | 5 | 4 | 80 ms (16 control steps) |

Frame `i` of a term is the value **`i * stride` control steps before now**, oldest first, newest last. Build it by keeping a ring of `stride * (frames - 1) + 1` frames and emitting every `stride`-th, with the newest always included.

**Why it matters.** The frame count was chosen at 50 Hz to span 100 ms. The policy now runs at 200 Hz, where the same count of *consecutive* frames spans 20 ms. Striding restores the window without widening the network's input.

**What goes wrong if this is ignored.** A builder that shifts one frame per inference emits a tensor of exactly the right length, the ONNX accepts it, the robot walks -- on a window 4x shorter than the one the policy was trained against. There is no error and no log line. `rl-wbc-fsm`'s `ObservationBuilder` shifts one frame per inference as of this writing and needs the change above.

Observation normalisation is already inside the ONNX (`normalization = baked_into_onnx`), so **feed raw values and do not normalise again**.

## Applying the action

```
for i, joint in enumerate(contract['action_joint_order']):
    target[joint] = action[i] * action_scale + default_joint_pos[joint]
```

Then track with PD: kp = 10.0, kd = 0.5, torque limit 1.746399998664856 N*m, control rate 200.0 Hz.

Joints in `unactuated_joints` are not policy-controlled and **must be locked at the given positions**: they take part in collision, and freeing them changes the foot contact geometry.

## Read before deploying

- Action transform: target = action * action_scale + default_joint_pos[joint], paired joint by joint in action_joint_order.
- In simulation, apply_actions also subtracts encoder_bias (a domain randomisation term modelling encoder offset). Do not reproduce it on hardware.
- Joints in unactuated_joints are not policy-controlled and must be locked at the given positions: they take part in collision, and freeing them changes the foot contact geometry.
- The ONNX already contains observation normalisation; feed raw observations.
- posture_command is 4 floats ordered [twist, pitch, roll, height - params.neutral_height]: the command term's own order, the height centred on standing. Not the builder's `base_pose`, which is three of the same quantities ordered [pitch, roll, twist].
