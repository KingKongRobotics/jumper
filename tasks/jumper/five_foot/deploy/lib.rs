//! `jumper.five_foot` on the robot: the claw on either side, from one policy,
//! closed by the control on its side and held out on three more.
//!
//! The policy is trained with the **left**-front leg carried as the claw and the
//! other five walking. An operator may want the claw on the right. Nothing about
//! the right is trained -- the robot is not asymmetric enough for that to be
//! worth a second policy -- so the right is the left, mirrored: the controller
//! shows the policy a robot reflected in its own x-z plane, and reflects the
//! answer back. The policy sees the claw on the left, exactly as it was trained,
//! while the robot carries it on the right.
//!
//! This is `rl-wbc-fsm`'s `carry_mirror`, moved out of the controller into this
//! task's own library, and with its per-joint signs taken from the robot's joint
//! axes rather than guessed from the home pose.
//!
//! ## What the mirror does, at each hook point
//!
//! Reflecting in the x-z plane (`y -> -y`):
//!
//! * **joints**: each joint is presented as its left-right twin (`LF_*` as
//!   `RF_*`, `LM_*` as `RM_*`, `LR_*` as `RR_*`), times a sign -- `+1` for the
//!   front arms' shoulder pitch, whose axis is y, and `-1` for every other joint,
//!   whose axis is x or z. Positions, velocities and torques alike; gains move
//!   with their joint and keep their sign.
//! * **IMU**: a polar vector (linear velocity, gravity) flips its y; an axial one
//!   (angular velocity, attitude) flips the components about x and z and keeps
//!   the one about y. So `gyro -> (-x, y, -z)`, `lin_vel -> (x, -y, z)`, and the
//!   orientation `(w, x, y, z) -> (w, -x, y, -z)`, which is gravity's y flipped
//!   once the observation projects it.
//! * **command**: `lin_vel_y`, the yaw rate, the roll and the twist flip; forward
//!   speed, pitch and height do not -- pitch is about y, perpendicular to the
//!   mirror, which is why both jumper pose commands measure it nose-down positive
//!   now and neither needs a special case here.
//!
//! The map is its own inverse, so one function serves both directions:
//! `before_observation` takes the robot into the policy's frame and
//! `after_decode` brings the decoded command back, and `home_pose` sends the
//! policy's home pose -- claw on the left, held stowed -- to the
//! robot as the same pose with the claw on the right, which the controller ramps
//! to before this mode's policy takes over.
//!
//! ## What it needs of the robot
//!
//! That the right half is the mirror image of the left, joint for joint. It is
//! measured to be, for every pair: the axis-derived sign maps each joint's range
//! onto its twin's exactly (`tests/test_five_foot_deploy.py`). The home poses
//! agree to 0.043 rad, the worst being `J3` on the front arms -- the V1.6.1 right
//! half is a rotated copy of the left rather than a mirrored one, open with the
//! mechanical team -- so the mirrored home is that much off the robot's own, well
//! inside the ramp's 0.10 rad tolerance.
//!
//! ## The claw's finger
//!
//! The policy does not drive the claw. Its finger is out of the action, and the
//! decoder holds it at the contract's home pose -- open -- like every joint no
//! policy drives. This library drives it instead, from the task control on the
//! carried claw's side: `claw_left` with the claw on the left, `claw_right` on
//! the right -- `LT` and `RT` on the pad, which is how `control-agent` has
//! always handed a gripper to the arm it is on, and Space on the keyboard for
//! either, as Control-agent 3.1 has it. The control's travel is the claw's: let
//! go is 60 degrees open, all the way is `GRIPPER_CLOSED`, and halfway is
//! halfway; let go while the arm is held out at a preset is 90 degrees open --
//! rl-wbc-fsm V3.1's two numbers, both from the shut end. The target never leads
//! the finger's measured angle by more than `SQUEEZE_LEAD` towards shut, so a
//! finger stopped by an object squeezes with the servo's `kp * lead` however
//! hard the control is held -- `claw.py` has what that is worth to the gait,
//! measured.
//!
//! **The policy trained on less of that travel**: `GRIPPER_OPEN` to shut, 37
//! degrees of it, and the finger is one of the joints it observes. Wider than
//! `GRIPPER_OPEN` it is shown the finger at `GRIPPER_OPEN`, still; inside that,
//! as it is -- the angle an object stops the finger at is the object's, and the
//! policy trained on exactly that.
//!
//! It is set in the policy's frame, on the carried claw's own joint
//! (`LF_J4_joint`), just before the mirror: on the right, the mirror carries it
//! to `RF_J4_joint` with the sign every finger takes. Only while the policy
//! runs -- the ramp and the hold keep the finger at the mode's home pose, open.
//! The controls are the task's by its `controls.yaml` (`task:`), by name, and
//! each device puts them where it likes; what the controller holds
//! [`ModeHook::reads`] to is the names.
//!
//! ## The arm, held out while its control is
//!
//! The arm stows -- the contract's home pose, `claw.py::LF_GRASP` -- and three
//! controls hold it out straight, one way each, for as long as they are held:
//! `arm_thumb_up`, `arm_thumb_down` and `arm_web_up`, the d-pad's up, down and
//! left on the pad and Shift, Alt and Ctrl on the keyboard; a fourth,
//! `arm_to_face` (d-pad right, Q), bends it in instead, the claw before the
//! visor. Let go, it goes back to the stow. That is Control-agent 3.1's layout, asked for on 2026-09-29
//! ("开关量 ... 松开后回位"); until then a direction was a press, read when it
//! was let go, pressed again to stow -- all a key through MuJoCo's viewer could
//! deliver when it was written. The presets, the gains, the interpolation and
//! the stow's pitch follow are rl-wbc-fsm's `config/hexa/rl-wbc-fsm.toml`
//! `[arm]` as of V3.1 (`fix/control_agent_v3.1`, 15e8d7d), verbatim, which is
//! what the robot runs them with: angles in degrees for shoulder yaw, pitch,
//! elbow and wrist, `kp` 3.0 and `kd` 0.3, and the setpoint moved `ARM_INTERP`
//! of the way to its target on every decode (50 Hz, where rl-wbc-fsm ran it on
//! every inference too).
//!
//! ```text
//!     arm_thumb_up     straight out, the thumb up       [-90, -180,   0,  -1]
//!     arm_thumb_down   straight out, the thumb down     [-90,  -30,  30,  -1]
//!     arm_web_up       straight out, the thumb-web up   [-90,  -90,   0,  -1]
//!     arm_to_face      folded in, the claw before the visor [-22.8, -90.0, -28.7, -84.7]
//! ```
//!
//! Thumb up and down are 3.0's right and left moved, and the web is 3.0's up
//! recalibrated on the robot. **Two held at once is the pose between them**, the
//! mean of the two: the thumb at 45 degrees between up and the web, or between
//! the web and down -- 3.1's "混合时，角度为45度". Thumb up and thumb down
//! together are neither, as the d-pad's two ends cannot both be down: a keyboard
//! can hold Shift and Alt at once. V3.1's diagonals -- up and a side, the arm
//! swung 45 degrees out towards it -- are gone with the press they were chosen on.
//! `arm_to_face` is not one of V3.1's and does not blend: held, it is the pose,
//! whatever else is held with it. It is for a thing carried in the claw -- a mug
//! by its handle -- brought to where a mouth would be, so what it must clear is
//! the load as much as the arm; a mean of it and a straight-out pose would sweep
//! that load through the trunk. Its shoulder pitch is the web's, exactly, so the
//! hand keeps the web's orientation and the load hangs as it hung: the shoulder
//! yaw swings the arm in and the elbow and wrist fold it, in the plane the web
//! holds them in. Measured 2026-10-12 (`grasp_pose.py`'s `Claw` at 1500 points a
//! body, the trunk level, a 40 mm cup 100 mm tall hanging plumb 86 mm off the
//! pinch): the pinch point at (0.206, 0.040, -0.025) m of the base, the cup's
//! centre at (0.271, -0.017, -0.025) -- on the centreline, as near as the
//! fold reaches -- the cup 34 mm from trunk and legs and 141 from the visor, the
//! forearm and claw 26 mm from them, the arm 62 mm above the floor; along the
//! straight joint-space path from `web_up` the cup never comes under 34 mm, the
//! arm under 14, the floor under 62.
//!
//! The dances are the pad's `menu` and a direction. That chord enters the dance
//! on the press, the claw mode stops ticking, and the direction never reaches
//! the arm (`deploy/fsm/src/config.rs::FsmConfig::chord_leaves_first` holds the
//! bundle to that). The keyboard's Ctrl, the web preset here, is not a dance's
//! modifier in the claw modes at all: the manifest keeps those switches out of
//! them with `from`.
//!
//! **The operator's pitch moves the arm at the stow**, as in V3.1: the shoulder
//! pitch follows it -- `SHOULDER_FOLLOW_NOSE_DOWN` rad at a full stick nose
//! down, `SHOULDER_FOLLOW_NOSE_UP` nose up, which is 57 degrees of shoulder. The
//! stick is the command this library is handed divided by its full standing
//! deflection, which is the stick exactly while standing; walking, the command
//! is held to 15 degrees and the follow to three quarters, where rl-wbc-fsm
//! reads the stick itself. **A held preset does not follow**: 3.1 gives the
//! preset priority over the stick ("优先级高于Ry对手臂的控制"). Until
//! 2026-09-29 the elbow followed at up and down, V3.1's `elbow_pitch_follow`.
//!
//! So, as there, **this library holds the arm for as long as the mode runs**,
//! at the stow as well: at `ARM_KP` / `ARM_KD`, not the 10 the policy trained
//! its hold at. Were both shoulder gains zero, it would hand the arm back to the
//! decoder once its setpoint was home within `RETURN_TOL`, as rl-wbc-fsm does.
//!
//! **The policy is shown the arm at home** while this library holds it: the
//! four joints' angle and torque in the observation are the ones measured on
//! the last tick before it took the arm, and the velocity zero, until the arm
//! is handed back **and measured** inside `GRASP_BOX` of the stow -- the setpoint
//! gets there well before the arm, at these gains. The policy was trained with
//! the arm inside that box; the presets are about a radian away, which in an
//! input that has never varied by more is not a measurement but noise of
//! unknown effect on every leg. Hidden, the observation stays in distribution
//! and what the policy meets is the mass moving -- the kind of disturbance the
//! payload curriculum trains against.
//!
//! **On the right** the table is the left's, mirrored with everything else. The
//! thumb's direction is the hand's own, and so is a blend of two, so nothing
//! swaps: the right arm's thumb-up is its own thumb up. The stow's follow is
//! mirrored too, as the shoulder pitch keeps its sign under the mirror.
//!
//! Measured on V1.6.1 (`grasp_pose.py`'s `Claw`, on the visual meshes, the
//! trunk level unless said), 2026-09-28. At a level stick every preset is
//! inside the soft limits, the forearm 13.7 mm or more from the trunk and its
//! neighbouring legs and 61.6 mm or more above the floor; the finger at 90
//! degrees brings `down` to 34.5 mm. The straight joint-space path from the
//! stow to `down` dips to 7.7 mm above the floor on the way. The follow is
//! where it is tight:
//!
//! * nose up, a full stick: the stow's shoulder brings the finger, at 60
//!   degrees, to 1.0 mm of the trunk or a neighbouring leg (15.8 mm at 37). With
//!   V3.1's elbow follow, which a held preset no longer has, `down`'s elbow
//!   reached 1.52 rad, past the 1.46 soft limit, 9.3 mm from them;
//! * nose down, it takes the claw towards the floor. With the trunk turned 20
//!   degrees about its origin at the standing height -- a crude stand-in for how
//!   the robot leans -- the arm's lowest point is under the floor by 2 mm at the
//!   stow without the follow and 28 mm with it, 19 and 137 mm at `up`, 25 and
//!   105 mm at `down`. The elbow's sign turned round would leave `up` 36 mm
//!   above it. Whether reaching down as the nose goes down is what the robot's
//!   operators want is theirs to say; this is what they tuned.
//!
//! ## Configuring it
//!
//! A mode in `deploy/manifests.json` names this task and carries
//! `"hook": {"side": "right"}` or `{"side": "left"}` -- the second is the trained
//! side and mirrors nothing, and saying so is what makes the pair of modes read
//! as a choice. `side` is required; any other key is refused.

use crate::hook::{HookSetup, ModeHook};
use crate::types::{Command, MotorCommand, RobotState, TaskControls};

/// The joints whose value keeps its sign under the x-z mirror: the front arms'
/// shoulder pitch, the only y axes on this robot. Every other joint -- the
/// claws' fingers included -- is an x or z axis and negates.
///
/// `tasks/jumper/common/mdp/symmetry.py::_SIGN_KEEP` is the same table for the
/// twenty gait joints the six-legged tasks' mirror augmentation covers. This one
/// covers all twenty-two, because the fingers are on the wire and five_foot
/// observes one. `tests/test_five_foot_deploy.py` reads both and the model's
/// joint axes, and fails when any of the three parts.
const SIGN_KEEP: [&str; 2] = ["LF_J1_joint", "RF_J1_joint"];

/// The carried claw's finger, in the policy's frame: the left-front one.
const FINGER: &str = "LF_J4_joint";

/// The finger's shut end and the squeeze limit: `claw.py`'s `GRIPPER_CLOSED`
/// and `SQUEEZE_LEAD`, where each is measured. On this joint open is the
/// **lower** number. `tests/test_five_foot_deploy.py` holds both to `claw.py`.
const GRIPPER_CLOSED: f32 = 0.10;
const SQUEEZE_LEAD: f32 = 0.06;
/// Seconds of the finger's closing speed added to the lead: `claw.py`'s
/// `LEAD_PER_SPEED`, which has why and what it measured. A finger closing on
/// nothing is led further the faster it goes; stopped by an object, it is led
/// by `SQUEEZE_LEAD` alone again, and squeezes as hard as it did.
const LEAD_PER_SPEED: f32 = 0.08;

/// The open end of the travel the policy was **trained** on: `claw.py`'s
/// `GRIPPER_OPEN`, which is also where the contract's home pose holds the
/// finger. The robot opens wider than this (below); the policy is never shown
/// the finger past it.
const GRIPPER_OPEN: f32 = -0.65;

/// How far open the finger goes, in degrees from `GRIPPER_CLOSED`: with the
/// trigger let go, and with the trigger let go while the arm is out at a
/// preset. rl-wbc-fsm's `[command]` `gripper_max_angle_deg` and
/// `gripper_pose_angle_deg` (V3.1), which measure from the shut end as these
/// do; there the shut end is 0 rad, which is the jaw before V1.6.1's.
const FINGER_RELEASED_DEG: f32 = 60.0;
const FINGER_POSED_DEG: f32 = 90.0;

/// The carried arm's four joints in the policy's frame, shoulder outwards: the
/// order the presets are written in.
const ARM: [&str; 4] = ["LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint"];

/// The four task controls that hold the arm at a preset, by `controls.yaml`'s
/// names: thumb up, thumb down, thumb-web up, to the face -- the order `PRESETS`
/// lists them in.
const ARM_CONTROLS: [&str; 4] = ["arm_thumb_up", "arm_thumb_down", "arm_web_up", "arm_to_face"];

/// rl-wbc-fsm's `config/hexa/rl-wbc-fsm.toml`, `[arm.left]` (V3.1), in degrees,
/// in `ARM_CONTROLS`' order: straight out with the thumb up, the thumb down, the
/// thumb-web up -- and the fourth, this crate's own: bent in, the pinch point
/// about 5 cm before the visor and a little under it (see the module doc).
const PRESETS: [[f32; 4]; 4] = [
    [-90.0, -180.0, 0.0, -1.0],
    [-90.0, -30.0, 30.0, -1.0],
    [-90.0, -90.0, 0.0, -1.0],
    [-22.8, -90.0, -28.7, -84.7],
];

/// The arm's gains whenever this library holds it, and how far the setpoint
/// moves towards its target on each decode: `[arm.left]`'s `kp` and `kd` and
/// `[arm]`'s `interp`.
const ARM_KP: f32 = 3.0;
const ARM_KD: f32 = 0.3;
const ARM_INTERP: f32 = 0.30;
/// The fraction per decode while `arm_to_face` is held, and until the arm has
/// settled after it is let go: a fifth of the presets' rate, a time constant of
/// a third of a second instead of a fifteenth. The straight-out presets swap
/// with nothing in the claw; the bend in is made with a mug hanging on the
/// pinch's friction, and at `ARM_INTERP` the swing flung it off the bar in two
/// of three runs (`tools/fetch`, 2026-10-12: the mug on the sand 0.5 s into the
/// bend); at this rate it stayed in every one.
const FACE_INTERP: f32 = 0.06;

/// The operator's pitch carried onto the arm at the stow, in rad of shoulder
/// per unit of stick: `[arm]`'s `shoulder_pitch_follow_up` and
/// `shoulder_pitch_follow_down`, one for each way the stick goes. V3.1's third,
/// `elbow_pitch_follow`, was the elbow at up and down; a held preset has
/// priority over the stick since 2026-09-29, and follows nothing.
///
/// Named here for the way the nose goes, which is what decides between them:
/// rl-wbc-fsm takes `_up` for a stick at or above zero, and a positive
/// `attitude.pitch` is nose **down** (`control-agent`: "positive = leaning
/// forward", `-ry`). Its own comment on the two keys says the reverse and is
/// the later of two; the one written beside the numbers when they were tuned
/// on the robot agrees with the code.
const SHOULDER_FOLLOW_NOSE_DOWN: f32 = 0.2;
const SHOULDER_FOLLOW_NOSE_UP: f32 = 1.0;

/// The pitch a full stick asks for while standing, in degrees:
/// `mdp/pose_command.py`'s `stand_pitch`, which the operator's full deflection
/// is the edge of. What turns the command this library is handed back into the
/// stick's travel, `-1..1`, which is what the follow gains multiply.
const PITCH_FULL_DEG: f32 = 20.0;

/// How close to the stow, in rad, every arm joint's setpoint has to be before
/// the decoder holds the arm again: rl-wbc-fsm's `kReturnTol`.
const RETURN_TOL: f32 = 0.01;

/// How close to the stow, in rad, every arm joint has to be **measured** before
/// the policy is shown the arm again: `claw.py`'s `GRASP_BOX`, the box the arm
/// is held in in training.
const GRASP_BOX: f32 = 0.10;

/// Each leg's prefix and its mirror image.
const LEG_TWIN: [(&str, &str); 6] = [
    ("LF", "RF"),
    ("RF", "LF"),
    ("LM", "RM"),
    ("RM", "LM"),
    ("LR", "RR"),
    ("RR", "LR"),
];

/// This task's hook for one mode. See the module docstring.
pub fn build(setup: &HookSetup<'_>) -> Result<Box<dyn ModeHook>, String> {
    Ok(Box::new(five_foot(setup)?))
}

fn five_foot(setup: &HookSetup<'_>) -> Result<FiveFoot, String> {
    if let Some(key) = setup.config.keys().find(|k| k.as_str() != "side") {
        return Err(format!("unknown key `{key}`; the only one is `side`"));
    }
    let side = match setup.config.get("side") {
        None => {
            return Err(
                "`side` is required: \"left\" for the side the policy was trained on, \
                 \"right\" for its mirror"
                    .into(),
            )
        }
        Some(v) => v.as_str().ok_or_else(|| format!("`side` is a string, got {v}"))?,
    };
    let names = &setup.contract.joint_names;
    let (mirror, control) = match side {
        "left" => (None, "claw_left"),
        "right" => (Some(Mirror::new(names)?), "claw_right"),
        other => {
            return Err(format!(
                "side = {other:?}; the claw is \"left\" (as trained) or \"right\" (mirrored)"
            ))
        }
    };
    let finger = names
        .iter()
        .position(|n| n == FINGER)
        .ok_or_else(|| format!("the wire has no `{FINGER}`, the carried claw's finger"))?;
    let mut joints = [0usize; 4];
    for (k, joint) in ARM.iter().enumerate() {
        joints[k] = names
            .iter()
            .position(|n| n == joint)
            .ok_or_else(|| format!("the wire has no `{joint}`, a joint of the carried arm"))?;
    }
    let home = joints.map(|w| setup.contract.default_wire[w]);
    Ok(FiveFoot {
        claw: Claw { finger, control, squeeze: 0.0, angle: GRIPPER_OPEN, speed: 0.0 },
        arm: Arm::new(joints, home),
        mirror,
    })
}

/// This task's hook for one mode: the claw, the arm, and on the right the mirror.
struct FiveFoot {
    /// `None` on the side the policy was trained on.
    mirror: Option<Mirror>,
    claw: Claw,
    arm: Arm,
}

impl ModeHook for FiveFoot {
    fn reads(&self) -> Vec<String> {
        std::iter::once(self.claw.control).chain(ARM_CONTROLS).map(str::to_string).collect()
    }

    fn reset(&mut self) {
        // A policy takes over from the home pose the ramp just drove to, so the
        // arm is stowed; a control still held takes it out again on the tick.
        self.arm = Arm::new(self.arm.joints, self.arm.home);
    }

    fn home_pose(&self, default_wire: &[f32]) -> Vec<f32> {
        let mut home = default_wire.to_vec();
        if let Some(m) = &self.mirror {
            m.joints(&mut home);
        }
        home
    }

    fn before_observation(
        &mut self,
        state: &mut RobotState,
        command: &mut Command,
        controls: &TaskControls,
    ) {
        if let Some(m) = &self.mirror {
            m.state(state);
            m.command(command);
        }
        let squeeze = controls.get(self.claw.control);
        self.claw.squeeze = if squeeze.is_finite() { squeeze.clamp(0.0, 1.0) } else { 0.0 };
        self.claw.observe(state);
        let pitch = command.base_pitch / PITCH_FULL_DEG.to_radians();
        self.arm.pitch = if pitch.is_finite() { pitch.clamp(-1.0, 1.0) } else { 0.0 };
        // Nobody there to let go of a control is nobody holding one.
        let held = ARM_CONTROLS.map(|c| controls.connected && controls.get(c) > 0.5);
        self.arm.observe(state, held);
    }

    fn after_decode(&mut self, motor: &mut MotorCommand) {
        motor.pos[self.claw.finger] = self.claw.target(self.arm.preset.is_some());
        self.arm.drive(motor);
        if let Some(m) = &self.mirror {
            m.motor(motor);
        }
    }
}

/// The carried arm, held out while a control is. Policy frame. See the module
/// docstring.
struct Arm {
    /// `ARM`'s wire indices.
    joints: [usize; 4],
    /// The contract's home pose for those joints: the stow.
    home: [f32; 4],
    /// The pose held out to while its controls are held, in rad; `None` is the
    /// stow.
    preset: Option<[f32; 4]>,
    /// `arm_to_face` is held, or was and the arm has not settled since: the
    /// setpoint moves at `FACE_INTERP`.
    slow: bool,
    face_held: bool,
    /// The operator's pitch as a stick, `-1..1`, nose down positive, as of the
    /// latest observation.
    pitch: f32,
    /// The shoulder's follow gains at the stow, nose down and nose up:
    /// `SHOULDER_FOLLOW_NOSE_DOWN` and `SHOULDER_FOLLOW_NOSE_UP`. Both zero is
    /// rl-wbc-fsm's other configuration, where the stow is the decoder's.
    shoulder_follow: (f32, f32),
    /// The hook, not the decoder, is holding the arm: at a preset, at the stow
    /// following the pitch, or on its way back to the decoder.
    driving: bool,
    /// The policy is being shown the arm at home rather than as it is: from the
    /// tick the hook takes the arm until it is measured back inside `GRASP_BOX`.
    hidden: bool,
    /// The setpoint, moved `ARM_INTERP` of the way to its target per decode.
    setpoint: [f32; 4],
    /// The arm as measured this tick, before anything is hidden.
    measured: [f32; 4],
    /// What the policy is shown of the arm while it is hidden: angle and torque
    /// from the last tick it was shown as it is.
    shown_q: [f32; 4],
    shown_tau: [f32; 4],
}

impl Arm {
    fn new(joints: [usize; 4], home: [f32; 4]) -> Self {
        Self {
            joints,
            home,
            preset: None,
            slow: false,
            face_held: false,
            pitch: 0.0,
            shoulder_follow: (SHOULDER_FOLLOW_NOSE_DOWN, SHOULDER_FOLLOW_NOSE_UP),
            driving: false,
            hidden: false,
            setpoint: home,
            measured: home,
            shown_q: home,
            shown_tau: [0.0; 4],
        }
    }

    /// The arm controls held, in `ARM_CONTROLS`' order, and the arm the policy
    /// is about to be shown.
    fn observe(&mut self, state: &mut RobotState, held: [bool; 4]) {
        self.measured = self.joints.map(|w| state.q[w]);
        self.preset = Self::pose(held);
        self.face_held = held[3];
        if self.face_held {
            self.slow = true;
        }
        self.show(state);
    }

    /// Where the held controls put the arm, in rad: one held is its preset, two
    /// the mean of theirs, none the stow. Thumb up and thumb down cancel -- the
    /// two ends of the d-pad's one rocker cannot both be down, and on the
    /// keyboard, where they can, neither of the two is what was meant. To the
    /// face held is that pose outright, whatever else is held: a mean of it and
    /// a straight-out pose sweeps the carried load through the trunk.
    fn pose([up, down, web, face]: [bool; 4]) -> Option<[f32; 4]> {
        if face {
            return Some(PRESETS[3].map(f32::to_radians));
        }
        let vertical = match (up, down) {
            (true, false) => Some(PRESETS[0]),
            (false, true) => Some(PRESETS[1]),
            _ => None,
        };
        let web = web.then_some(PRESETS[2]);
        let deg = match (vertical, web) {
            (Some(a), Some(b)) => std::array::from_fn(|k| 0.5 * (a[k] + b[k])),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return None,
        };
        Some(deg.map(f32::to_radians))
    }

    /// The arm as the policy is shown it: as it is, or as it was before this
    /// library took it.
    fn show(&mut self, state: &mut RobotState) {
        let back = self.measured.iter().zip(&self.home).all(|(q, h)| (q - h).abs() <= GRASP_BOX);
        if self.hidden && !self.driving && back {
            self.hidden = false;
        }
        if self.hidden {
            for (k, &w) in self.joints.iter().enumerate() {
                state.q[w] = self.shown_q[k];
                state.qd[w] = 0.0;
                state.tau[w] = self.shown_tau[k];
            }
        } else {
            self.shown_q = self.measured;
            self.shown_tau = self.joints.map(|w| state.tau[w]);
        }
    }

    /// Where the arm is to go this tick, or `None` for the decoder's hold: the
    /// preset held, as it is -- a held preset has priority over the stick -- or
    /// the stow, with the operator's pitch carried onto the shoulder.
    fn target(&self) -> Option<[f32; 4]> {
        let p = self.pitch;
        match self.preset {
            Some(rad) => Some(rad),
            None if self.shoulder_follow != (0.0, 0.0) => {
                let (down, up) = self.shoulder_follow;
                let gain = if p >= 0.0 { down } else { up };
                let mut rad = self.home;
                rad[1] += gain * p;
                Some(rad)
            }
            None => None,
        }
    }

    /// Overwrite the decoder's hold of the arm while the hook has it.
    fn drive(&mut self, motor: &mut MotorCommand) {
        let target = match self.target() {
            Some(t) => {
                if !self.driving {
                    // From where the arm is, so it does not jump: rl-wbc-fsm
                    // seeds the same way on the edge into a preset.
                    self.setpoint = self.measured;
                    self.driving = true;
                    self.hidden = true;
                }
                t
            }
            None if self.driving => self.home,
            None => return,
        };
        let rate = if self.slow { FACE_INTERP } else { ARM_INTERP };
        for (k, &w) in self.joints.iter().enumerate() {
            self.setpoint[k] += (target[k] - self.setpoint[k]) * rate;
            motor.pos[w] = self.setpoint[k];
            motor.kp[w] = ARM_KP;
            motor.kd[w] = ARM_KD;
        }
        if self.slow && !self.face_held {
            let settled = self.setpoint.iter().zip(&target).all(|(s, t)| (s - t).abs() <= 0.02);
            if settled {
                self.slow = false;
            }
        }
        let home = self.setpoint.iter().zip(&self.home).all(|(s, h)| (s - h).abs() <= RETURN_TOL);
        if self.target().is_none() && home {
            self.driving = false;
        }
    }
}

/// The carried claw's finger, driven from its task control. Policy frame.
struct Claw {
    /// `FINGER`'s wire index.
    finger: usize,
    /// The task control at the carried claw's side: `claw_left` or `claw_right`.
    control: &'static str,
    /// How far that control is held, `0..1`, where the finger is, and how fast
    /// it is closing -- all as of the latest observation, which precedes every
    /// decode. The speed as measured, before the policy's view of it is held
    /// still (`observe`).
    squeeze: f32,
    angle: f32,
    speed: f32,
}

impl Claw {
    /// Where the finger is, and the finger the policy is shown: no wider open
    /// than it was trained on, and still while it is held wider.
    fn observe(&mut self, state: &mut RobotState) {
        self.angle = state.q[self.finger];
        let speed = state.qd[self.finger];
        self.speed = if speed.is_finite() { speed } else { 0.0 };
        if self.angle < GRIPPER_OPEN {
            state.q[self.finger] = GRIPPER_OPEN;
            state.qd[self.finger] = 0.0;
        }
    }

    /// The control's travel laid onto the claw's, landing on both ends exactly,
    /// and never more than `SQUEEZE_LEAD` shut of where the finger is -- plus
    /// `LEAD_PER_SPEED` of its closing speed, so a finger closing on nothing is
    /// not held to the squeeze's pace. An opening is never held back: the limit
    /// is `min`, and open is the lower number.
    /// Let go is `FINGER_RELEASED_DEG` open, or `FINGER_POSED_DEG` while the
    /// arm is `posed` -- held out at a preset; any squeeze at all is back on the
    /// released travel, as rl-wbc-fsm has it.
    fn target(&self, posed: bool) -> f32 {
        let released = GRIPPER_CLOSED - FINGER_RELEASED_DEG.to_radians();
        let wanted = if posed && self.squeeze <= 0.0 {
            GRIPPER_CLOSED - FINGER_POSED_DEG.to_radians()
        } else if self.squeeze >= 1.0 {
            GRIPPER_CLOSED
        } else {
            released + self.squeeze * (GRIPPER_CLOSED - released)
        };
        wanted.min(self.angle + SQUEEZE_LEAD + LEAD_PER_SPEED * self.speed.max(0.0))
    }
}

/// The claw on the right: the robot reflected in its x-z plane, both ways.
struct Mirror {
    /// `twin[w]` is the wire index of joint `w`'s left-right image.
    twin: Vec<usize>,
    /// `sign[w]` multiplies joint `w`'s value as its twin is presented.
    sign: Vec<f32>,
}

impl Mirror {
    fn new(names: &[String]) -> Result<Self, String> {
        let mut twin = Vec::with_capacity(names.len());
        let mut sign = Vec::with_capacity(names.len());
        for name in names {
            let image = LEG_TWIN
                .iter()
                .find(|(leg, _)| name.starts_with(leg))
                .map(|(leg, other)| format!("{other}{}", &name[leg.len()..]))
                .ok_or_else(|| {
                    format!("joint `{name}` is on no leg the mirror knows (LF RF LM RM LR RR)")
                })?;
            let w = names.iter().position(|n| *n == image).ok_or_else(|| {
                format!("joint `{name}` has no mirror image `{image}` on the wire")
            })?;
            twin.push(w);
            sign.push(if SIGN_KEEP.contains(&name.as_str()) { 1.0 } else { -1.0 });
        }
        Ok(Self { twin, sign })
    }

    /// `v[w] = sign[w] * v[twin[w]]`, for positions, velocities and torques. Its
    /// own inverse: twins share a sign.
    fn joints(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.twin.len(), "a joint vector is not in wire order");
        let src = v.to_vec();
        for (w, out) in v.iter_mut().enumerate() {
            *out = self.sign[w] * src[self.twin[w]];
        }
    }

    /// `v[w] = v[twin[w]]`, for gains: they move with their joint, and a gain has
    /// no direction to flip.
    fn gains(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.twin.len(), "a gain vector is not in wire order");
        let src = v.to_vec();
        for (w, out) in v.iter_mut().enumerate() {
            *out = src[self.twin[w]];
        }
    }
}

impl Mirror {
    /// The robot's state, reflected: joints and IMU.
    fn state(&self, state: &mut RobotState) {
        self.joints(&mut state.q);
        self.joints(&mut state.qd);
        self.joints(&mut state.tau);
        let [w, x, y, z] = state.imu.quat;
        state.imu.quat = [w, -x, y, -z];
        state.imu.gyro[0] = -state.imu.gyro[0];
        state.imu.gyro[2] = -state.imu.gyro[2];
        state.imu.lin_vel[1] = -state.imu.lin_vel[1];
    }

    /// The operator's command, reflected: what is about x or z flips.
    fn command(&self, command: &mut Command) {
        command.lin_vel_y = -command.lin_vel_y;
        command.yaw_rate = -command.yaw_rate;
        command.base_roll = -command.base_roll;
        command.base_twist = -command.base_twist;
    }

    /// A motor command, reflected: targets with their signs, gains without.
    fn motor(&self, motor: &mut MotorCommand) {
        self.joints(&mut motor.pos);
        self.joints(&mut motor.vel);
        self.joints(&mut motor.tau);
        self.gains(&mut motor.kp);
        self.gains(&mut motor.kd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The robot's joints, leg by leg: five on each front arm, three on each
    /// walking leg. The order is deliberately not left-right paired, so a mirror
    /// that assumed adjacency would be caught.
    fn names() -> Vec<String> {
        let mut out = Vec::new();
        for leg in ["LF", "RF", "LM", "RM", "LR", "RR"] {
            let n = if leg.ends_with('F') { 5 } else { 3 };
            for j in 0..n {
                out.push(format!("{leg}_J{j}_joint"));
            }
        }
        out
    }

    fn at(name: &str) -> usize {
        names().iter().position(|x| x == name).unwrap()
    }

    fn mirror() -> Mirror {
        Mirror::new(&names()).unwrap()
    }

    fn ramp(n: usize, from: f32) -> Vec<f32> {
        (0..n).map(|i| from + 0.1 * i as f32).collect()
    }

    fn state() -> RobotState {
        let n = names().len();
        let mut s = RobotState::new(n);
        s.q = ramp(n, 0.3);
        s.qd = ramp(n, -1.0);
        s.tau = ramp(n, 0.05);
        s.imu.quat = [0.9, 0.1, 0.2, 0.3];
        s.imu.gyro = [0.4, 0.5, 0.6];
        s.imu.lin_vel = [0.7, 0.8, 0.9];
        s
    }

    fn command() -> Command {
        Command {
            lin_vel_x: 0.3,
            lin_vel_y: 0.2,
            yaw_rate: 0.5,
            height: 0.1,
            base_pitch: 0.15,
            base_roll: -0.1,
            base_twist: 0.25,
            ..Default::default()
        }
    }

    /// The task's controls, held at the given values, somebody there.
    fn held(controls: &[(&str, f32)]) -> TaskControls {
        TaskControls::new(true, controls.iter().map(|(n, v)| (n.to_string(), *v)))
    }

    /// Reflecting twice is the identity, for everything the mirror touches. This
    /// is what lets one map serve both directions; a sign that differed between
    /// two twins would break it on exactly that pair.
    #[test]
    fn the_mirror_is_its_own_inverse() {
        let m = mirror();
        let (mut s, mut c) = (state(), command());
        m.state(&mut s);
        m.command(&mut c);
        assert_ne!(s.q, state().q, "the control group: one reflection moves the joints");
        m.state(&mut s);
        m.command(&mut c);
        assert_eq!(s.q, state().q);
        assert_eq!(s.qd, state().qd);
        assert_eq!(s.tau, state().tau);
        assert_eq!(s.imu.quat, state().imu.quat);
        assert_eq!(s.imu.gyro, state().imu.gyro);
        assert_eq!(s.imu.lin_vel, state().imu.lin_vel);
        let c0 = command();
        assert_eq!(
            (c.lin_vel_x, c.lin_vel_y, c.yaw_rate, c.base_pitch, c.base_roll, c.base_twist),
            (c0.lin_vel_x, c0.lin_vel_y, c0.yaw_rate, c0.base_pitch, c0.base_roll, c0.base_twist)
        );

        let n = names().len();
        let mut motor = MotorCommand::new(n);
        motor.pos = ramp(n, 0.2);
        motor.kp = ramp(n, 10.0);
        let before = motor.clone();
        m.motor(&mut motor);
        m.motor(&mut motor);
        assert_eq!((motor.pos, motor.kp), (before.pos, before.kp));
    }

    /// Each joint shows up as its twin with the axis-derived sign: the front
    /// arms' J1 unsigned, everything else -- a walking leg's J1 and the fingers
    /// included -- negated.
    #[test]
    fn a_joint_is_presented_as_its_twin() {
        let mut s = state();
        mirror().state(&mut s);
        let q = state().q;
        assert_eq!(s.q[at("LF_J1_joint")], q[at("RF_J1_joint")], "a y axis keeps its sign");
        assert_eq!(s.q[at("LF_J0_joint")], -q[at("RF_J0_joint")], "a z axis negates");
        assert_eq!(s.q[at("LF_J4_joint")], -q[at("RF_J4_joint")], "the finger negates");
        assert_eq!(s.q[at("LM_J1_joint")], -q[at("RM_J1_joint")], "a walking knee negates");
        assert_eq!(s.q[at("RR_J2_joint")], -q[at("LR_J2_joint")]);
    }

    /// The body rolled one way, yawing one way and drifting one way looks, in the
    /// policy's frame, like the same body doing each the other way; pitching and
    /// moving forward look the same.
    #[test]
    fn the_imu_and_the_command_flip_what_is_about_x_and_z() {
        let (mut s, mut c) = (state(), command());
        mirror().state(&mut s);
        mirror().command(&mut c);
        assert_eq!(s.imu.quat, [0.9, -0.1, 0.2, -0.3]);
        assert_eq!(s.imu.gyro, [-0.4, 0.5, -0.6]);
        assert_eq!(s.imu.lin_vel, [0.7, -0.8, 0.9]);
        assert_eq!((c.lin_vel_x, c.lin_vel_y, c.yaw_rate), (0.3, -0.2, -0.5));
        assert_eq!((c.base_pitch, c.base_roll, c.base_twist), (0.15, 0.1, -0.25));
        assert_eq!(c.height, 0.1);
    }

    /// The policy's home pose holds the left arm at the grasp pose; the robot is
    /// sent the right arm there and the left one walking. Written against a pose
    /// that is not symmetric, because a symmetric one maps onto itself and would
    /// pass against a mirror that did nothing.
    #[test]
    fn the_home_pose_carries_the_claw_on_the_right() {
        let mut policy_home = vec![0.0; names().len()];
        policy_home[at("LF_J2_joint")] = -0.6; // carried, the grasp pose
        policy_home[at("RF_J2_joint")] = 0.44; // walking
        let home = hook("right").home_pose(&policy_home);
        assert_eq!(home[at("RF_J2_joint")], 0.6, "the right arm takes the grasp pose");
        assert_eq!(home[at("LF_J2_joint")], -0.44, "the left arm takes the walking one");
        assert_eq!(hook("left").home_pose(&policy_home), policy_home, "the left side is as trained");
    }

    /// A decoded target lands on the twin with the sign; gains move unsigned.
    #[test]
    fn the_decoded_command_goes_back_to_the_robot() {
        let mut motor = MotorCommand::new(names().len());
        motor.pos[at("RF_J0_joint")] = 0.5;
        motor.kp[at("RF_J0_joint")] = 12.0;
        mirror().motor(&mut motor);
        assert_eq!(motor.pos[at("LF_J0_joint")], -0.5);
        assert_eq!(motor.kp[at("LF_J0_joint")], 12.0);
        assert_eq!(motor.pos[at("RF_J0_joint")], 0.0);
    }

    /// What one tick is handed: the robot's joints, the task's controls by
    /// name, and the operator's pitch as a command, in rad. Everything unnamed
    /// reads zero.
    #[derive(Default)]
    struct In<'a> {
        q: &'a [(&'a str, f32)],
        qd: &'a [(&'a str, f32)],
        tau: &'a [(&'a str, f32)],
        controls: &'a [(&'a str, f32)],
        pitch: f32,
    }

    /// One tick of a hook, with a decoded command of zeros at the decoder's
    /// gains, 10 / 0.5: what the policy was shown, and what the robot is sent.
    fn run(hook: &mut dyn ModeHook, i: In<'_>) -> (RobotState, MotorCommand) {
        let mut s = RobotState::new(names().len());
        for (v, set) in [(&mut s.q, i.q), (&mut s.qd, i.qd), (&mut s.tau, i.tau)] {
            for (name, x) in set {
                v[at(name)] = *x;
            }
        }
        let mut command = Command { base_pitch: i.pitch, ..Default::default() };
        hook.before_observation(&mut s, &mut command, &held(i.controls));
        let mut motor = MotorCommand::new(names().len());
        motor.kp.fill(10.0);
        motor.kd.fill(0.5);
        hook.after_decode(&mut motor);
        (s, motor)
    }

    /// One tick of a hook: the robot at `q` with `controls` held. Returns what
    /// the robot is sent.
    fn tick(hook: &mut dyn ModeHook, q: &[(&str, f32)], controls: &[(&str, f32)]) -> MotorCommand {
        run(hook, In { q, controls, ..Default::default() }).1
    }

    /// Long enough, with `controls` held, for the setpoint to be where it is
    /// going: 0.7^60 of the way is left. Returns the tick after.
    fn settle_with(hook: &mut dyn ModeHook, controls: &[(&str, f32)], pitch: f32) -> MotorCommand {
        // 300 decodes: `arm_to_face` moves at FACE_INTERP, and 60 leaves it 2% short.
        for _ in 0..300 {
            run(hook, In { controls, pitch, ..Default::default() });
        }
        run(hook, In { controls, pitch, ..Default::default() }).1
    }

    /// The same with nothing held: where the arm goes home to.
    fn settle(hook: &mut dyn ModeHook, pitch: f32) -> MotorCommand {
        settle_with(hook, &[], pitch)
    }

    /// The four arm joints of the claw at `leg`, in the order the presets list them.
    fn arm(motor: &MotorCommand, leg: &str) -> [f32; 4] {
        [0, 1, 2, 3].map(|j| motor.pos[at(&format!("{leg}_J{j}_joint"))])
    }

    fn near(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    /// A preset in rad, by its control's name.
    fn preset(control: &str) -> [f32; 4] {
        PRESETS[ARM_CONTROLS.iter().position(|c| *c == control).unwrap()].map(f32::to_radians)
    }

    /// The pose between two presets.
    fn between(a: &str, b: &str) -> [f32; 4] {
        let (a, b) = (preset(a), preset(b));
        std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
    }

    const UP: (&str, f32) = ("arm_thumb_up", 1.0);
    const DOWN: (&str, f32) = ("arm_thumb_down", 1.0);
    const WEB: (&str, f32) = ("arm_web_up", 1.0);
    const FACE: (&str, f32) = ("arm_to_face", 1.0);

    /// A stow that is not zero and not symmetric -- `LF_GRASP`, near enough --
    /// so that a hook seeding from it, or holding zero, is told apart from one
    /// holding the stow.
    const STOW: [f32; 4] = [-0.52, -1.57, -0.52, -1.31];

    /// The left claw's hook, holding the arm at `STOW`, as its concrete type so a
    /// test can set rl-wbc-fsm's other configuration.
    fn stowed(side: &str) -> FiveFoot {
        let config: toml::Table = toml::from_str(&format!("side = {side:?}")).unwrap();
        let home: Vec<(String, f32)> = names()
            .into_iter()
            .map(|n| {
                let k = ARM.iter().position(|a| *a == n);
                (n, k.map_or(0.0, |k| STOW[k]))
            })
            .collect();
        let contract = contract_with(&home);
        five_foot(&HookSetup { task: "jumper.five_foot", mode: "claw", config: &config, contract: &contract })
            .unwrap()
    }

    fn released() -> f32 {
        GRIPPER_CLOSED - FINGER_RELEASED_DEG.to_radians()
    }

    /// The claw control at the claw's side is the claw's travel: let go is 60
    /// degrees open, all the way is shut, halfway is halfway. The finger is held
    /// where the lead cannot bind, so what is measured is the mapping. The
    /// other side's control is the control group: held as hard, it closes
    /// nothing.
    #[test]
    fn the_claw_control_at_its_side_is_the_claws_travel() {
        let mut left = hook("left");
        assert_eq!(left.reads(), ["claw_left", "arm_thumb_up", "arm_thumb_down", "arm_web_up", "arm_to_face"]);
        let shut = [("LF_J4_joint", GRIPPER_CLOSED)];
        let half = 0.5 * (released() + GRIPPER_CLOSED);
        for (lt, want) in [(0.0, released()), (0.5, half), (1.0, GRIPPER_CLOSED), (1.7, GRIPPER_CLOSED)] {
            let pos = tick(left.as_mut(), &shut, &[("claw_left", lt), ("claw_right", 1.0 - lt.min(1.0))]).pos;
            assert!((pos[at("LF_J4_joint")] - want).abs() < 1e-6, "claw_left {lt}: {}", pos[at("LF_J4_joint")]);
            assert_eq!(pos[at("RF_J4_joint")], 0.0, "the other claw moved");
        }
        assert!((released() - (0.10 - 1.0472)).abs() < 1e-4, "60 degrees from shut: {}", released());
    }

    /// Stopped by an object, the finger is sent no more than `SQUEEZE_LEAD` past
    /// where it is, however hard the control is held -- the policy sees that
    /// finger's torque in simulation and never trained with a squeeze. The
    /// opposite direction is the control group: an opening goes out whole, or
    /// the limit is a claw that cannot let go.
    #[test]
    fn a_finger_stopped_by_an_object_is_squeezed_no_harder_than_the_lead() {
        let mut left = hook("left");
        let stalled = [("LF_J4_joint", -0.4)];
        let sent = tick(left.as_mut(), &stalled, &[("claw_left", 1.0)]).pos[at("LF_J4_joint")];
        assert!((sent - (-0.4 + SQUEEZE_LEAD)).abs() < 1e-6, "sent {sent}");
        let sent = tick(left.as_mut(), &stalled, &[("claw_left", 0.0)]).pos[at("LF_J4_joint")];
        assert_eq!(sent, released(), "the limit held back an opening");
    }

    /// On the right, `claw_right` closes the right claw: the robot's `RF_J4` is
    /// sent the claw's target with the finger's mirror sign (open is the higher
    /// number there), measured from the robot's own right finger. `claw_left`
    /// is the control group, and closes nothing on this side.
    #[test]
    fn on_the_right_claw_right_closes_the_right_claw() {
        let mut right = hook("right");
        assert_eq!(right.reads(), ["claw_right", "arm_thumb_up", "arm_thumb_down", "arm_web_up", "arm_to_face"]);
        // The robot's right finger at +0.2 is the policy's left one at -0.2.
        let finger = [("RF_J4_joint", 0.2)];
        let pos = tick(right.as_mut(), &finger, &[("claw_right", 1.0)]).pos;
        assert!((pos[at("RF_J4_joint")] - -(-0.2 + SQUEEZE_LEAD)).abs() < 1e-6, "{}", pos[at("RF_J4_joint")]);
        assert_eq!(pos[at("LF_J4_joint")], 0.0, "the walking claw's finger moved");
        let pos = tick(right.as_mut(), &finger, &[("claw_left", 1.0)]).pos;
        assert_eq!(pos[at("RF_J4_joint")], -released(), "claw_left closed the right claw");
    }

    /// With the arm held out at a preset and the claw let go the finger opens to
    /// 90 degrees; the first touch of the claw is back on the 60-degree travel,
    /// as rl-wbc-fsm has it. The stow is the control group, before the preset is
    /// held and after it is let go.
    #[test]
    fn the_claw_opens_wider_while_a_preset_is_held_until_it_is_touched() {
        let posed = GRIPPER_CLOSED - FINGER_POSED_DEG.to_radians();
        let finger = |h: &mut dyn ModeHook, controls: &[(&str, f32)]| tick(h, &[], controls).pos[at("LF_J4_joint")];
        let mut left = hook("left");
        assert_eq!(finger(left.as_mut(), &[]), released(), "the stow");
        assert_eq!(finger(left.as_mut(), &[WEB]), posed, "held out, let go");
        let touched = finger(left.as_mut(), &[WEB, ("claw_left", 0.01)]);
        assert!((touched - (released() + 0.01 * (GRIPPER_CLOSED - released()))).abs() < 1e-6, "{touched}");
        assert_eq!(finger(left.as_mut(), &[]), released(), "let go of the preset, it is the stow's");
    }

    /// Opened wider than it trained on, the finger is shown to the policy at the
    /// trained travel's open end, and still -- while the claw's own squeeze limit
    /// keeps working from where the finger really is. Inside the travel it is
    /// shown as it is: the control group.
    #[test]
    fn the_policy_is_shown_no_more_finger_than_it_trained_on() {
        let mut left = hook("left");
        let (seen, sent) =
            run(left.as_mut(), In { q: &[("LF_J4_joint", -1.2)], qd: &[("LF_J4_joint", 3.0)],
                                    controls: &[("claw_left", 1.0)], ..Default::default() });
        assert_eq!((seen.q[at("LF_J4_joint")], seen.qd[at("LF_J4_joint")]), (GRIPPER_OPEN, 0.0));
        // Led from where the finger really is and how fast it really closes --
        // not the still, narrower finger the policy is shown.
        let want = -1.2 + SQUEEZE_LEAD + 3.0 * LEAD_PER_SPEED;
        assert!((sent.pos[at("LF_J4_joint")] - want).abs() < 1e-6, "lead from the shown finger");
        let (seen, _) =
            run(left.as_mut(), In { q: &[("LF_J4_joint", -0.3)], qd: &[("LF_J4_joint", 3.0)], ..Default::default() });
        assert_eq!((seen.q[at("LF_J4_joint")], seen.qd[at("LF_J4_joint")]), (-0.3, 3.0));
    }

    /// A preset is held out to from the tick its control goes down, from where
    /// the arm is measured -- the setpoint starts there, not at the stow -- at
    /// `ARM_KP` / `ARM_KD`, and the arm goes home the tick it is let go.
    /// Pinned because until 2026-09-29 a preset was chosen on the release and
    /// stayed until pressed again; Control-agent 3.1 holds it only while held.
    #[test]
    fn a_preset_is_held_out_to_while_its_control_is_held_and_no_longer() {
        let mut left = stowed("left");
        let off = [("LF_J0_joint", STOW[0]), ("LF_J1_joint", STOW[1] + 0.2),
                   ("LF_J2_joint", STOW[2]), ("LF_J3_joint", STOW[3])];
        let measured = [STOW[0], STOW[1] + 0.2, STOW[2], STOW[3]];
        let up = preset("arm_thumb_up");
        let first = run(&mut left, In { q: &off, controls: &[UP], ..Default::default() }).1;
        let want: [f32; 4] = std::array::from_fn(|k| measured[k] + ARM_INTERP * (up[k] - measured[k]));
        assert!(near(arm(&first, "LF"), want), "{:?} against {want:?}", arm(&first, "LF"));
        assert_eq!((first.kp[at("LF_J1_joint")], first.kd[at("LF_J1_joint")]), (ARM_KP, ARM_KD));
        assert_eq!(first.kp[at("LM_J1_joint")], 10.0, "a walking joint's gains were taken");
        let there = settle_with(&mut left, &[UP], 0.0);
        assert!(near(arm(&there, "LF"), up), "held out at {:?}", arm(&there, "LF"));
        assert_eq!(arm(&there, "RF"), [0.0; 4], "the walking arm moved");
        let let_go = run(&mut left, In::default()).1;
        let back: [f32; 4] = std::array::from_fn(|k| up[k] + ARM_INTERP * (STOW[k] - up[k]));
        assert!(near(arm(&let_go, "LF"), back), "let go, it did not turn for home: {:?}", arm(&let_go, "LF"));
        assert!(near(arm(&settle(&mut left, 0.0), "LF"), STOW));
    }

    /// Two presets held are the pose between them -- the thumb at 45 degrees
    /// between up and the web, or between the web and down -- and thumb up with
    /// thumb down is neither, as the d-pad's two ends cannot both be down. Each
    /// preset alone is the control group for the blends.
    #[test]
    fn two_presets_held_are_the_pose_between_them() {
        let cases: [(&[(&str, f32)], [f32; 4]); 10] = [
            (&[UP], preset("arm_thumb_up")),
            (&[DOWN], preset("arm_thumb_down")),
            (&[WEB], preset("arm_web_up")),
            (&[UP, WEB], between("arm_thumb_up", "arm_web_up")),
            (&[DOWN, WEB], between("arm_thumb_down", "arm_web_up")),
            (&[UP, DOWN], STOW),
            (&[UP, DOWN, WEB], preset("arm_web_up")),
            // To the face does not blend: held with anything, it is the pose.
            (&[FACE], preset("arm_to_face")),
            (&[WEB, FACE], preset("arm_to_face")),
            (&[UP, DOWN, FACE], preset("arm_to_face")),
        ];
        for (controls, want) in cases {
            let mut left = stowed("left");
            let got = arm(&settle_with(&mut left, controls, 0.0), "LF");
            assert!(near(got, want), "{controls:?}: {got:?} against {want:?}");
        }
        // 45 degrees: the thumb's joint half way from up's -180 to the web's -90.
        assert!((between("arm_thumb_up", "arm_web_up")[1] - (-135f32).to_radians()).abs() < 1e-6);
    }

    /// Nobody there to let go of a control -- the pad unplugged, the operator
    /// gone -- is nobody holding one, and the arm goes home. The same control
    /// held with somebody there is the control group.
    #[test]
    fn nobody_there_holds_no_preset() {
        for (connected, out) in [(false, false), (true, true)] {
            let mut left = stowed("left");
            let controls = TaskControls::new(connected, [("arm_web_up".to_string(), 1.0)]);
            let mut last = MotorCommand::new(names().len());
            for _ in 0..61 {
                let mut s = RobotState::new(names().len());
                left.before_observation(&mut s, &mut Command::default(), &controls);
                last = MotorCommand::new(names().len());
                left.after_decode(&mut last);
            }
            assert_eq!(near(arm(&last, "LF"), preset("arm_web_up")), out, "connected {connected}");
        }
    }

    /// Let go, the arm goes back to the stow, where the library keeps holding it
    /// -- the pitch follow needs it.
    #[test]
    fn letting_go_takes_the_arm_home_and_keeps_it() {
        let mut left = stowed("left");
        settle_with(&mut left, &[WEB], 0.0);
        let home = settle(&mut left, 0.0);
        assert!(near(arm(&home, "LF"), STOW), "{:?}", arm(&home, "LF"));
        assert_eq!(home.kp[at("LF_J1_joint")], ARM_KP, "the library let go of the follow");
    }

    /// rl-wbc-fsm's other configuration, both shoulder gains zero: the stow is
    /// the decoder's, and the library hands the arm back once its setpoint is
    /// home. The policy is shown the arm again only once it is **measured** back
    /// inside `GRASP_BOX` -- the setpoint gets home long before the arm does.
    #[test]
    fn with_no_follow_the_arm_is_handed_back_and_shown_once_it_is_measured_home() {
        let mut left = stowed("left");
        left.arm.shoulder_follow = (0.0, 0.0);
        let at_stow: Vec<(&str, f32)> = ARM.iter().copied().zip(STOW).collect();
        let (seen, sent) = run(&mut left, In { q: &at_stow, ..Default::default() });
        assert_eq!(sent.kp[at("LF_J1_joint")], 10.0, "the decoder's, before anything is held");
        assert_eq!(seen.q[at("LF_J1_joint")], STOW[1]);
        for _ in 0..3 {
            run(&mut left, In { q: &at_stow, controls: &[UP], ..Default::default() });
        }
        let lagging = [("LF_J1_joint", STOW[1] - 0.3)];
        let mut sent = run(&mut left, In { q: &lagging, ..Default::default() }).1;
        for _ in 0..60 {
            if sent.kp[at("LF_J1_joint")] == 10.0 {
                break;
            }
            sent = run(&mut left, In { q: &lagging, ..Default::default() }).1;
        }
        assert_eq!(sent.kp[at("LF_J1_joint")], 10.0, "never handed back");
        let lagging: Vec<(&str, f32)> =
            ARM.iter().copied().zip(STOW).map(|(n, h)| (n, if n == "LF_J1_joint" { h - 0.3 } else { h })).collect();
        let (seen, _) = run(&mut left, In { q: &lagging, ..Default::default() });
        assert_eq!(seen.q[at("LF_J1_joint")], STOW[1], "shown while still 0.3 rad out");
        let inside: Vec<(&str, f32)> =
            ARM.iter().copied().zip(STOW).map(|(n, h)| (n, if n == "LF_J1_joint" { h - 0.05 } else { h })).collect();
        let (seen, _) = run(&mut left, In { q: &inside, ..Default::default() });
        assert_eq!(seen.q[at("LF_J1_joint")], STOW[1] - 0.05, "hidden once back inside the box");
    }

    /// While the library holds the arm the policy is shown it as it was on the
    /// tick before: the angles and the torque of then, and no velocity. The
    /// first tick is the control group -- the arm is passed through, torque and
    /// all -- and so is a walking joint, on every tick.
    #[test]
    fn the_policy_is_shown_the_arm_at_home_while_the_library_holds_it() {
        let mut left = stowed("left");
        let first = [("LF_J1_joint", -1.56), ("LF_J3_joint", -1.30), ("LM_J1_joint", 0.1)];
        let first_tau = [("LF_J1_joint", 0.7), ("LF_J3_joint", -0.2)];
        let (seen, _) = run(&mut left, In { q: &first, tau: &first_tau, qd: &[("LF_J1_joint", 0.4)],
                                             ..Default::default() });
        assert_eq!((seen.q[at("LF_J1_joint")], seen.qd[at("LF_J1_joint")], seen.tau[at("LF_J1_joint")]),
                   (-1.56, 0.4, 0.7), "the first tick is passed through");
        let away = [("LF_J1_joint", -2.9), ("LF_J3_joint", -0.3), ("LM_J1_joint", 0.4)];
        let (seen, _) = run(&mut left, In {
            q: &away,
            qd: &[("LF_J1_joint", 2.5), ("LM_J1_joint", 1.5)],
            tau: &[("LF_J1_joint", -1.1), ("LF_J3_joint", 0.9), ("LM_J1_joint", 0.3)],
            controls: &[UP],
            ..Default::default()
        });
        assert_eq!(seen.q[at("LF_J1_joint")], -1.56, "the policy saw the arm leave");
        assert_eq!(seen.q[at("LF_J3_joint")], -1.30);
        assert_eq!(seen.qd[at("LF_J1_joint")], 0.0, "the arm's velocity");
        assert_eq!((seen.tau[at("LF_J1_joint")], seen.tau[at("LF_J3_joint")]), (0.7, -0.2), "the arm's torque");
        assert_eq!((seen.q[at("LM_J1_joint")], seen.qd[at("LM_J1_joint")], seen.tau[at("LM_J1_joint")]),
                   (0.4, 1.5, 0.3), "a walking joint was hidden too");
    }

    /// At the stow the shoulder pitch follows the operator's pitch: a fifth of a
    /// radian at a full stick nose down, a whole one nose up, in proportion
    /// between, and the stick is the command over the standing band's 20
    /// degrees -- held at full past it. Level is the control group: the stow.
    #[test]
    fn the_stow_follows_the_pitch_on_the_shoulder() {
        let full = PITCH_FULL_DEG.to_radians();
        for (pitch, shoulder) in [(0.0, 0.0), (full, 0.2), (-full, -1.0), (0.5 * full, 0.1),
                                  (-0.75 * full, -0.75), (2.0 * full, 0.2)] {
            let mut left = stowed("left");
            let got = arm(&settle(&mut left, pitch), "LF");
            let want = [STOW[0], STOW[1] + shoulder, STOW[2], STOW[3]];
            assert!(near(got, want), "pitch {pitch}: {got:?} against {want:?}");
        }
    }

    /// A held preset has priority over the stick: at a full pitch either way it
    /// is exactly the preset, where V3.1's elbow followed at up and down. The
    /// stow, which does follow, is the control group.
    #[test]
    fn a_held_preset_does_not_follow_the_pitch() {
        let full = PITCH_FULL_DEG.to_radians();
        for control in ARM_CONTROLS {
            for pitch in [full, -full] {
                let mut left = hook("left");
                let got = arm(&settle_with(left.as_mut(), &[(control, 1.0)], pitch), "LF");
                assert!(near(got, preset(control)), "{control} at pitch {pitch}: {got:?}");
            }
        }
        let mut left = stowed("left");
        assert!(!near(arm(&settle(&mut left, full), "LF"), STOW), "the control group: the stow follows");
    }

    /// On the right the arm is the left's, mirrored -- a preset, a blend and the
    /// stow's follow alike, nothing swapped: the thumb's direction is the hand's
    /// own. Checked on the robot's right arm, in the robot's frame.
    #[test]
    fn on_the_right_the_arm_is_the_left_s_mirrored() {
        let mirrored = |rad: [f32; 4]| [-rad[0], rad[1], -rad[2], -rad[3]];
        for (controls, policy) in [(&[UP][..], preset("arm_thumb_up")),
                                   (&[UP, WEB][..], between("arm_thumb_up", "arm_web_up")),
                                   (&[WEB][..], preset("arm_web_up"))] {
            let mut right = hook("right");
            let got = arm(&settle_with(right.as_mut(), controls, 0.0), "RF");
            assert!(near(got, mirrored(policy)), "{controls:?}: {got:?}");
        }
        let full = PITCH_FULL_DEG.to_radians();
        let mut right = hook("right");
        assert!(near(arm(&settle(right.as_mut(), -full), "RF"), [0.0, -1.0, 0.0, 0.0]),
                "the stow's follow keeps the shoulder pitch's sign");
    }

    /// A policy taking over finds the arm stowed: the ramp has just driven it
    /// home. A control still held takes it out again on the next tick, as a
    /// held control does.
    #[test]
    fn a_policy_taking_over_starts_from_the_stow() {
        let mut left = stowed("left");
        settle_with(&mut left, &[UP], 0.0);
        left.reset();
        assert!(near(arm(&settle(&mut left, 0.0), "LF"), STOW), "the preset outlived the takeover");
        assert!(!near(arm(&settle_with(&mut left, &[UP], 0.0), "LF"), STOW), "held again, it did not go out");
    }

    /// A contract over `names()` with the given home pose.
    fn contract_with(home: &[(String, f32)]) -> crate::layout::Contract {
        let entries: Vec<String> = home.iter().map(|(n, v)| format!("{n:?}: {v}")).collect();
        let text = format!(
            r#"{{"obs_joint_order": [], "action_joint_order": [], "action_scale": 0.25,
                "default_joint_pos": {{{}}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": 0, "history_length": 1, "terms": []}},
                "action": {{"dim": 0}}}}"#,
            entries.join(", ")
        );
        let names: Vec<String> = home.iter().map(|(n, _)| n.clone()).collect();
        crate::layout::Contract::from_str(&text, &names).unwrap()
    }

    fn setup_with(config: &str, names: Vec<String>) -> Result<Box<dyn ModeHook>, String> {
        let config: toml::Table = toml::from_str(config).unwrap();
        let contract = contract_with(&names.into_iter().map(|n| (n, 0.0)).collect::<Vec<_>>());
        build(&HookSetup { task: "jumper.five_foot", mode: "claw_right", config: &config, contract: &contract })
    }

    fn hook(side: &str) -> Box<dyn ModeHook> {
        setup_with(&format!("side = {side:?}"), names()).unwrap()
    }

    /// Each refusal is a mode that would otherwise run with the claw on the side
    /// nobody chose, or with a trigger that closes nothing.
    #[test]
    fn a_configuration_it_cannot_honour_is_refused() {
        assert!(setup_with(r#"side = "right""#, names()).is_ok(), "the control group builds");
        assert!(setup_with(r#"side = "left""#, names()).is_ok());
        let err = |c: &str, n: Vec<String>| setup_with(c, n).err().expect("refused");
        assert!(err("", names()).contains("required"));
        assert!(err(r#"side = "up""#, names()).contains("\"up\""));
        assert!(err(r#"side = 1"#, names()).contains("string"));
        assert!(err("side = \"right\"\nflip = true", names()).contains("unknown key `flip`"));
        let mut lonely = names();
        lonely.retain(|n| n != "RR_J2_joint");
        assert!(err(r#"side = "right""#, lonely).contains("no mirror image `RR_J2_joint`"));
        let mut stray = names();
        stray.push("waist_joint".into());
        assert!(err(r#"side = "right""#, stray).contains("on no leg"));
        let mut fingerless = names();
        fingerless.retain(|n| n != "LF_J4_joint" && n != "RF_J4_joint");
        assert!(err(r#"side = "left""#, fingerless).contains("no `LF_J4_joint`"));
        let mut armless = names();
        armless.retain(|n| n != "LF_J3_joint" && n != "RF_J3_joint");
        assert!(err(r#"side = "left""#, armless).contains("no `LF_J3_joint`"));
    }
}
