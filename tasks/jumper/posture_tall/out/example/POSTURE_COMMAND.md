# `posture_command`: what the C++ has to build

Bundle: `jumper.posture_tall`, from `model_76299.pt`.

The observation contains a term the on-robot `ObservationBuilder` does not
construct yet. This is its specification. **Nothing else in the bundle needs a
change** -- the action transform, the joint orders, the gains and the gait clock
are all the same as any other hexa locomotion policy.

## The four channels

`posture_command` is **4 floats, in this order**:

| i | channel | unit | range | sign |
|---|---|---|---|---|
| 0 | twist  | rad | +/-0.5236 parked, +/-0.2618 walking | + yaws the body **left** over its planted feet |
| 1 | pitch  | rad | +/-0.3491 parked, +/-0.2618 walking | + is **nose down** |
| 2 | roll   | rad | +/-0.2618 parked, +/-0.2618 walking | + is **left side up** |
| 3 | height | m   | -0.00647 to +0.07353 | + is taller |

All three angles are right-hand-rule about the **body's own axes** (x forward,
y left, z up), which is the same convention `base_pose` already uses. "Walking"
is a velocity command whose `[vx, vy, wz]` has a norm of at least
0.06; the controller holds the angles to the walking band then.

## The two things that are silent if you get them wrong

**1. The order is not `base_pose`'s.** The builder already has
```cpp
} else if (term == "base_pose") {        // trunk-attitude command
    frame.push_back(command.base_pitch);
    frame.push_back(command.base_roll);
    frame.push_back(command.base_twist);
}
```
-- the same three quantities as channels 0-2 here, in the order **(pitch, roll,
twist)** against this term's **(twist, pitch, roll)**. Wiring `base_pose`'s three
floats into the first three slots compiles, runs, and puts pitch where the policy
reads twist. Nothing reports it; the robot leans in a direction nobody asked for.

**2. Channel 3 is centred and the operator's command is not.** The observation
carries the height **relative to the standing height**:

    posture_command[3] = height_command_m - 0.10647

So an operator asking for 0.18 m sends +0.07353, and "stand
normally" is 0.0, not 0.10647. Feeding the raw height instead is a constant
0.10647 m offset on an input whose whole useful range is 0.08 m wide --
the policy would see a command it never met in training and hold something near
the top of its range permanently.

There is no per-term offset in `layout.json`, and that is why this file exists:
the contract can express a scale and cannot express this. (The scale is 1.0 here,
like every other term in this bundle.)

## Two ways to implement it

**A. A new term, matching this contract.** Add `posture_command` to
`term_width()` (returns 4) and to the builder's chain:
```cpp
} else if (term == "posture_command") {
    frame.push_back(command.base_twist);
    frame.push_back(command.base_pitch);
    frame.push_back(command.base_roll);
    frame.push_back(command.height - 0.10647f);   // centred, see above
}
```
The command struct already carries all four fields (`base_twist`, `base_pitch`,
`base_roll`, `height`), so this is the builder and nothing else -- no DDS change,
no new operator channel.

**B. Retrain the policy into the shape the controller already has** -- split into
`base_pose` (pitch, roll, twist) plus `height` as a fourth channel of the
`commands` block, and add `"height"` to `command_terms` in
`config/rl-wbc-fsm.toml`. That needs no C++ at all and leaves one spelling for
one quantity, which is the better end state. It costs a full retrain, which is
why this bundle exists in shape A.

**If B ever happens, delete A rather than keeping both.** Two names for the same
four numbers is how a contract vocabulary rots.

## While you are in there: the observation scales have two sources

Not this task's bug, and worth knowing because it bites the same way. `export.py`
writes each term's training `scale` into `layout.json`, and `contract.cpp` never
reads it -- the robot takes `[observation.scales]` from the hand-written
`config/rl-wbc-fsm.toml`, whose comment says *"Policies are trained WITHOUT
observation normalization -> all scales 1.0"*.

That conclusion is right and the reason given is not: these policies **are**
trained with `EmpiricalNormalization`, and 1.0 is correct only because
`export.py` bakes the normaliser into the ONNX graph (`normalization =
"baked_into_onnx"` in `layout.json`, which is why the README says to feed raw
values). Export a policy some day with the normaliser outside the graph and the
TOML will go on saying 1.0, silently.

The fix is to read the scales from the contract. It is a few lines in
`contract.cpp`, and it removes a second source of truth for a number that
training owns.

## Checking it on the bench

With the robot on a stand, feed a static command and read the body back:

| command | expect |
|---|---|
| all zero | stands at 0.10647 m, level, square |
| pitch +0.26 | nose down 15 deg, feet stay put |
| roll +0.26 | left side up 15 deg |
| twist +0.26 | body yaws left 15 deg **over stationary feet** |
| height +0.074 | stands 0.18 m tall |
| height -0.006 | squats to 0.10 m |

The twist row is the one that catches a mis-ordered wiring: it is the only
channel whose effect cannot be confused with another's, because pitch and roll
both tilt the body and twist does not.

Measured in simulation on a stand, holding 0.18 m costs 0.29 N*m at the worst
joint (17% of the peak rating) and 0.10 m costs 0.35 N*m with 2.5 degrees of
travel left at the tightest joint -- **the squat is the demanding end**, not the
extension.
