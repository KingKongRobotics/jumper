//! One tick, in two phases, for whichever mode the FSM has chosen.
//!
//! `fsm.rs` decides *which* mode. This decides what a tick does inside one, and
//! it is where the three regimes live:
//!
//! | regime | when | what it commands |
//! |---|---|---|
//! | warm start | the state sets `warm_start` | a slide to the hand-off pose, at the state's own soft gains |
//! | ramp | a model state was just entered, and the robot is not at its default pose | a slide to **that model's** default pose, at the ramp gains |
//! | policy | a model state, at its default pose | observation -> inference -> joint targets |
//! | hold | anything else, including a model that cannot run | damping around the measured pose |
//!
//! ## Why the tick is split
//!
//! In a browser the model runs in onnxruntime-web, which is JavaScript: a wasm
//! module cannot call it and return inside one synchronous call. So `tick`
//! returns either finished targets or a request, and the host answers the
//! request with `resume`.
//!
//! That is not a concession. The loop was already going to be split, because it
//! publishes faster than it infers -- `rl-wbc-fsm` runs the output at 1 kHz and
//! the policy at the contract's `control_hz` (50 for locomotion, 200 for a jump),
//! holding the last decoded target in between. Making the split a return type
//! rather than a timer comparison buys the property this whole design is for:
//! `tick` is a pure function of `(state, inputs, now)`, so one recorded episode
//! replays identically in a test, in a browser and on a bench.
//!
//! ## Two things that look like details and are not
//!
//! **The hand-off waits on the measured pose, not on the clock.**
//! `mode_switch_ramp_s` is how fast the setpoint slides; the policy starts only
//! once every joint is within `pose_reach_tol` of the model's default pose. Gains
//! too soft to converge mean the hand-off waits indefinitely and the robot
//! soft-holds. That is deliberate -- the alternative is starting a policy from a
//! pose it was never trained around -- and the fix is firmer gains, not a timer.
//!
//! **A model that momentarily cannot run must not re-ramp.** Holding is right;
//! clearing the active mode as well would re-enter the ramp on the next tick and
//! do it again, giving a robot that twitches once per failed inference instead of
//! standing still. Only a real hold state clears it.

use std::collections::HashMap;

use crate::action::{commanded_gains, ema_alpha_from_cutoff, ActionDecoder, GainTrim, RobotLimits};
use crate::config::{FsmConfig, StateConfig};
use crate::fsm::{Fsm, Inputs, Micros};
use crate::layout::Contract;
use crate::obs::{self, ObsConfig, ObservationBuilder};
use crate::hook::{HookSetup, ModeHook};
use crate::operator::{CommandShaper, Operators};
use crate::source::{self, Divergence, Provenance, SourceCapability};
use crate::types::{Command, MotorCommand, RobotState};

/// What the host must answer, or nothing at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Targets are final; read them from `command()`.
    Hold,
    /// Run `observation()` through this mode's model and call `resume`.
    ///
    /// The name rather than an index: a host that loaded its models by name
    /// cannot mispair them, and an index would be this crate's ordering leaking
    /// into an interface that has no reason to know it.
    Infer { mode: String },
}

/// What only the host knows. Everything derivable from the robot's state --
/// tilt, whether the warm-start pose is reached -- is derived here instead, so a
/// host cannot get it subtly wrong in three different ways.
pub struct TickInput<'a> {
    pub state: &'a RobotState,
    /// The command the host built itself -- `play`'s, a recorded frame's, a
    /// page's own mapping -- for a mode with no operator in `operators`.
    pub command: &'a Command,
    /// Each mode's own reading of the person (`operator::Operators`), or `None`
    /// for a host that builds every command itself. A mode found here takes its
    /// command and its task's controls from its own operator, after the cascade
    /// has picked it, and never `command` or `controls`: different policies take
    /// different controls, at their own ranges.
    pub operators: Option<&'a Operators>,
    /// A state sample arrived within `state_timeout_ms`.
    pub state_fresh: bool,
    /// An operator command arrived within `command_timeout_ms`.
    pub command_fresh: bool,
    /// The operator's latched mode, matched against `button:<name>` rules.
    pub buttons: &'a std::collections::BTreeSet<String>,
    pub gripper_active: bool,
    /// The controls a task answers itself -- `controls.yaml`'s `task:`, by
    /// name -- for a mode with no operator in `operators`, as the host read
    /// them. Handed to a mode's hook and to nothing else; a host with no such
    /// reading passes `TaskControls::default()`.
    pub controls: &'a crate::types::TaskControls,
}

/// One mode that runs a model. Hold states have none of this.
struct ModeRuntime {
    contract: Contract,
    obs: ObservationBuilder,
    act: ActionDecoder,
    kp: f32,
    kd: f32,
    infer_period_us: u64,
    alpha: f32,
    /// The bands and the ramp the contract's controller block declares, or
    /// `None` for a mode whose command reaches the policy as the operator set it.
    shaper: Option<CommandShaper>,
    /// The pose this mode ramps to and holds before its policy takes over, robot
    /// frame: the contract's home pose, or what the mode's hook makes of it.
    home: Vec<f32>,
    /// The task's hook, for a mode whose manifest entry asked for one.
    hook: Option<HookRuntime>,
}

/// A mode's hook (`hook.rs`), and what the decoder carries between ticks in the
/// frame the hook presents to the policy. A mode without a hook has one frame
/// and keeps all of this on the controller, as it always has.
struct HookRuntime {
    hook: Box<dyn ModeHook>,
    /// The last decoded command, policy frame: the decoder's previous command.
    cmd: MotorCommand,
    prev_target: Vec<f32>,
    /// What the decoder applied, policy frame: the `joint_torque` a robot that
    /// measures none is shown.
    applied_torque: Vec<f32>,
}

pub struct Controller {
    fsm: Fsm,
    joints: usize,
    /// For every regime that is not a policy: warm start, ramp, hold.
    hold: ActionDecoder,
    modes: HashMap<String, ModeRuntime>,
    /// Home pose, for a warm start that names no target of its own.
    default_wire: Vec<f32>,

    // Carried across ticks.
    active_mode: Option<String>,
    ramping: bool,
    ramp_from: Vec<f32>,
    ramp_t0: Micros,
    prev_target: Vec<f32>,
    last_action: Vec<f32>,
    /// Fed into the next observation's `joint_torque`. The feedback edge: on a
    /// robot this is the only torque there is, because the servo reports none.
    applied_torque: Vec<f32>,
    last_infer: Option<Micros>,
    /// The mode whose recording has played out and has not been handed back
    /// yet. Read and cleared by the host, which owns the latch.
    completed: Option<String>,
    /// Modes whose policy must not run, whatever the cascade asks for.
    ///
    /// Set by `interlock_stub_engine`. Separate from "this mode has no model"
    /// -- it has one, the host just cannot run it -- so it takes the same path
    /// as a stale-feedback tick: keep the mode, hold steadily, say why.
    interlocked: std::collections::BTreeSet<String>,
    /// Which motion `go` names were active last tick, so a held or latched
    /// binding starts the motion once rather than restarting it every frame.
    ///
    /// A `rise` or `fall` is active for a single frame and an edge is not
    /// needed for it; `hold` and `toggle` stay active, and without this the
    /// clock would be re-zeroed every tick and the robot would crouch forever.
    go_was_active: std::collections::BTreeSet<String>,
    /// The mode awaiting `resume`. `Some` is the only state in which `resume`
    /// means anything, so calling it out of turn is an error rather than a
    /// silently mis-attributed action.
    pending: Option<String>,
    observation: Vec<f32>,
    /// True when this environment measures joint torque. The robot does not --
    /// the servo reports none, so the controller reconstructs it from the PD it
    /// commanded and feeds that back. A simulator does, and handing it the
    /// reconstruction instead would silently feed the policy zeros on the first
    /// tick of every mode and a one-step-stale value thereafter. Found by
    /// `tests/simulator-fsm-parity.test.ts`, which is what it is for.
    torque_measured: bool,
    /// Every term this host supplies differently than training did, per mode.
    /// Reported, never acted on: the numbers are what they are, and a runner
    /// that cannot see this list cannot know why a policy behaves differently
    /// here than it did where it was trained.
    divergences: HashMap<String, Vec<Divergence>>,
    cmd: MotorCommand,
    /// What the last decode asked for. Kept apart from what is published,
    /// because the filter has to keep converging toward it through every tick
    /// that does not infer -- and `cmd.pos` cannot be both, which the first
    /// version of this tried and got a filter that moved once and then stopped.
    target: Vec<f32>,
    /// The published position, after the output-rate filter.
    pos_des: Vec<f32>,
    alpha: f32,
}

#[derive(Debug)]
pub enum Error {
    Obs(obs::Error),
    /// This environment cannot supply a term the policy observes.
    Source(source::Error),
    /// `resume` was called when nothing was pending, or with the wrong width.
    NotPending,
    ActionWidth { got: usize, want: usize },
    /// A state names a model, but no contract was installed for it.
    NoContract(String),
    /// A mode's recorded motion names a `go` button the cascade does not
    /// declare, or the cascade declares an `event` button no mode reads. Both
    /// directions, because each alone is a control that does nothing and looks
    /// exactly like one that works.
    GoEvent(String),
    /// A mode's hook could not be built: its task compiled none, or refused the
    /// configuration the manifest gave it.
    Hook(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::GoEvent(m) => write!(f, "{m}"),
            Error::Hook(m) => write!(f, "{m}"),
            Error::Obs(e) => write!(f, "{e}"),
            Error::Source(e) => write!(f, "{e}"),
            Error::NotPending => f.write_str(
                "resume() with no inference outstanding: the action would be applied to \
                 whatever mode happens to be active now, which is not the one that asked",
            ),
            Error::ActionWidth { got, want } => write!(
                f,
                "the model returned {got} values and this mode's action is {want} wide"
            ),
            Error::NoContract(s) => write!(
                f,
                "state '{s}' names a model but no contract was installed for it; it would \
                 hold forever while the panel said it was running"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// What a mode needs besides its contract, and what is global to the robot.
pub struct Setup {
    pub robot: RobotLimits,
    pub scales: obs::Scales,
    pub obs_clip: f32,
    pub action_clip: f32,
    pub action_smoothing: f32,
    pub command_terms: Vec<String>,
    pub gait_period: f64,
    pub gait_gate_threshold: Option<f32>,
    /// Seconds from entering a reference-guided mode to its `go`. A placeholder
    /// for the button that should trigger it; see `ObsConfig`.
    pub reference_go_delay_s: f64,
    /// How often the host will call `tick`. Sets the output filter and nothing
    /// else -- the policy rate comes from each mode's own contract.
    pub output_rate_hz: f64,
    /// Where this environment's numbers come from. Checked against every
    /// contract at construction, and consulted for `joint_torque`, which is the
    /// one term whose *source* differs between hosts rather than only its value.
    pub source: SourceCapability,
}

impl Controller {
    /// Build every mode up front from contracts the host has already parsed.
    ///
    /// Contracts arrive parsed because reading a file is I/O and this crate has
    /// none: a browser has no filesystem, and making the core able to read one
    /// would make it unable to run in the place it most needs to.
    pub fn new(
        cfg: FsmConfig,
        setup: Setup,
        mut contracts: HashMap<String, Contract>,
        now: Micros,
    ) -> Result<Self, Error> {
        let joints = setup.robot.default_wire.len();
        let default_wire = setup.robot.default_wire.clone();
        let output_dt_s = if setup.output_rate_hz > 0.0 { 1.0 / setup.output_rate_hz } else { 0.0 };

        let torque_measured = !matches!(
            setup.source.terms.get("joint_torque"),
            Some(Provenance::Reconstructed) | None
        );
        let mut divergences = HashMap::new();
        let mut modes = HashMap::new();
        for state in cfg.states.iter().filter(|s| s.has_model()) {
            // Taken, not borrowed: a contract belongs to exactly one mode, and
            // `Contract` is deliberately not `Clone` -- two modes sharing one
            // would share its observation history too.
            let contract = contracts
                .remove(&state.name)
                .ok_or_else(|| Error::NoContract(state.name.clone()))?;
            let terms: Vec<String> = contract
                .layout
                .observation
                .terms
                .iter()
                .map(|t| t.name.clone())
                .collect();
            divergences.insert(
                state.name.clone(),
                source::check(&terms, &Default::default(), &setup.source).map_err(Error::Source)?,
            );
            modes.insert(
                state.name.clone(),
                build_mode(state, contract, &setup, output_dt_s)?,
            );
        }

        // Both directions of the `go` binding, here because this is the one
        // place that sees the cascade and the contracts at once. Either half
        // alone is a control that loads and does nothing: a motion waiting for
        // a press that can never arrive, or a button whose press goes nowhere.
        let declared: std::collections::BTreeSet<&str> =
            cfg.buttons.iter().filter(|b| b.event).map(|b| b.name.as_str()).collect();
        let mut wanted: std::collections::BTreeSet<&str> = Default::default();
        for (mode, m) in &modes {
            let Some(go) = m.contract.reference.as_ref().and_then(|r| r.go_event()) else {
                continue;
            };
            wanted.insert(go);
            if !declared.contains(go) {
                return Err(Error::GoEvent(format!(
                    "mode '{mode}' starts its motion on a button named '{go}', and the \
                     cascade declares no `[[fsm.button]]` with that name and `event = true`. \
                     Nothing would ever start it.\n\
                     Add it to the manifest's `buttons`, or drop `go_event` from the task \
                     and let the motion run on the timer."
                )));
            }
        }
        for name in declared.difference(&wanted) {
            return Err(Error::GoEvent(format!(
                "`[[fsm.button]]` '{name}' is declared as an event, and no mode reads it. \
                 An event button is read by a contract's `go_event`; no contract in this \
                 bundle names this one, so pressing it would do nothing."
            )));
        }

        Ok(Self {
            fsm: Fsm::new(cfg, now),
            joints,
            hold: ActionDecoder::new(
                crate::action::ActionConfig {
                    scale: 0.0,
                    clip: 0.0,
                    smoothing: 0.0,
                    effort_limit: f32::INFINITY,
                    action_wire_idx: vec![],
                },
                setup.robot,
            ),
            modes,
            default_wire: default_wire.clone(),
            active_mode: None,
            ramping: false,
            ramp_from: vec![0.0; joints],
            ramp_t0: now,
            prev_target: default_wire.clone(),
            last_action: vec![],
            applied_torque: vec![0.0; joints],
            last_infer: None,
            go_was_active: Default::default(),
            completed: None,
            interlocked: Default::default(),
            pending: None,
            observation: Vec::new(),
            torque_measured,
            divergences,
            cmd: MotorCommand::new(joints),
            target: default_wire.clone(),
            pos_des: default_wire,
            alpha: 1.0,
        })
    }

    pub fn state(&self) -> &StateConfig {
        self.fsm.current()
    }

    pub fn config(&self) -> &FsmConfig {
        self.fsm.config()
    }

    /// The task controls each mode's hook answers (`ModeHook::reads`), for the
    /// modes that have one: what `guide::pad_guide` lists for them.
    pub fn hook_reads(&self) -> std::collections::BTreeMap<String, Vec<String>> {
        self.modes
            .iter()
            .filter_map(|(name, m)| m.hook.as_ref().map(|h| (name.clone(), h.hook.reads())))
            .collect()
    }

    /// Every transition since the last call, with the rule that caused each.
    pub fn take_log(&mut self) -> Vec<crate::fsm::Transition> {
        self.fsm.take_log()
    }

    /// The joint stops every commanded target is clamped to, in wire order.
    pub fn limits(&self) -> (&[f32], &[f32]) {
        self.hold.limits()
    }

    pub fn observation(&self) -> &[f32] {
        &self.observation
    }

    /// The command to publish. Valid after either phase.
    pub fn command(&self) -> &MotorCommand {
        &self.cmd
    }

    /// Whether the active mode is producing policy output, as opposed to holding
    /// or ramping toward it. A panel that reads "running" off the mode name alone
    /// says so through a two-second ramp it is not running through.
    /// What this host supplies differently than training did, for one mode.
    pub fn divergences(&self, mode: &str) -> &[Divergence] {
        self.divergences.get(mode).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn is_running_policy(&self) -> bool {
        self.active_mode.is_some() && !self.ramping
    }

    pub fn tick(&mut self, input: &TickInput<'_>, now: Micros) -> Result<Step, Error> {
        // An outstanding request is abandoned rather than carried: the world has
        // moved on by a tick, and an action decoded against a stale observation
        // is worse than one not decoded at all.
        self.pending = None;

        let before = self.fsm.current().name.clone();
        let mode_name = {
            let cfg = self.fsm.config();
            let warm_target = self.warm_target();
            let inputs = Inputs {
                state_fresh: input.state_fresh,
                command_fresh: input.command_fresh,
                tilt: input.state.imu.valid.then(|| tilt_angle(input.state.imu.quat)),
                buttons: input.buttons.clone(),
                gripper_active: input.gripper_active,
                pose_reached: warm_target
                    .map(|t| pose_reached(&input.state.q, t, cfg.pose_reach_tol))
                    .unwrap_or(false),
            };
            self.fsm.update(&inputs, now).name.clone()
        };
        let state = self.fsm.current().clone();

        // A recorded motion the cascade left before it played out -- the tilt
        // fallback, stale feedback -- is over all the same. It is reported
        // completed so the host releases its latch; otherwise the latch outlives
        // the fall, and `button:<motion>` re-enters the motion the moment the
        // robot is level again, from the start. Measured in the crab battle
        // (2026-10-15, `tools/battle`, seed 3): a dash that flipped the crab at
        // 11.7 s stayed latched through `safe`, and when the next round stood
        // the crab on its mark at 24.4 s it dashed into the middle of the arena
        // before anybody pressed anything -- which on the robot is a fall, a
        // person standing it back up, and the move it fell on playing again.
        // A gait left the same way says nothing: its latch is the operator's
        // (`carry` on a held LB), and a mode is not a move.
        if before != mode_name && self.modes.get(&before).is_some_and(|m| m.contract.reference.is_some()) {
            self.completed = Some(before);
        }

        if state.warm_start && input.state_fresh {
            let target = self.warm_target().unwrap_or(&self.default_wire).to_vec();
            self.enter_hold(&mode_name);
            self.slide(&target, state.kp, state.kd, input.state, now, self.fsm.config().warm_start_duration_s);
            return Ok(Step::Hold);
        }

        let runs_model = state.has_model()
            && input.state_fresh
            && !state.hold_current
            && !self.interlocked.contains(&mode_name);
        if !runs_model {
            // A real hold state clears the active mode, so re-entering a model
            // re-ramps. See the module note: a model that merely could not run
            // this tick keeps it, and holds steadily instead.
            if !state.has_model() {
                self.active_mode = None;
                self.ramping = false;
            }
            self.hold_still(input.state, state.kd);
            return Ok(Step::Hold);
        }

        if self.active_mode.as_deref() != Some(mode_name.as_str()) {
            self.begin_mode(&mode_name, input.state, now);
        }

        if self.ramping {
            let target = self.modes[&mode_name].home.clone();
            let ramp_s = self.fsm.config().mode_switch_ramp_s;
            let (kp, kd) = (self.fsm.config().ramp_kp, self.fsm.config().ramp_kd);
            self.slide(&target, kp, kd, input.state, now, ramp_s);
            if pose_reached(&input.state.q, &target, self.fsm.config().pose_reach_tol) {
                self.ramping = false;
                self.seed(&mode_name);
            }
            return Ok(Step::Hold);
        }

        // Every tick, not only the ones that infer: a `rise` or `fall` is
        // active for one host frame, and the host runs far faster than the
        // policy, so checking at inference time would miss most presses.
        //
        // After the ramp, deliberately. A `go` arriving while the robot is
        // still sliding into the mode's home pose is a motion asked for before
        // there is a pose to start it from.
        for (name, m) in self.modes.iter_mut() {
            let Some(go) = m.contract.reference.as_ref().and_then(|r| r.go_event()) else {
                continue;
            };
            let now_active = input.buttons.contains(go);
            let was = self.go_was_active.contains(go);
            if now_active && !was && *name == mode_name {
                m.obs.trigger_go();
            }
            if now_active {
                self.go_was_active.insert(go.to_string());
            } else {
                self.go_was_active.remove(go);
            }
        }

        // A recorded motion is a thing that ends. When it does, the mode has
        // nothing left to play, and holding its last frame is a robot frozen in
        // a landing pose. Reported rather than acted on here: the latch that
        // put the FSM in this mode belongs to the host, and releasing it is
        // what lets the cascade fall through to its `always` rule -- the same
        // path, and the same ordering, as every other way of leaving a mode.
        if self.modes[&mode_name].obs.motion_finished() {
            self.completed = Some(mode_name.clone());
        }

        // Between policy ticks the last decoded target stands and the output
        // filter keeps moving toward it. Running the policy at the host's rate
        // instead would alias it *and* rescale `joint_pos_rate_limit`, which is
        // per-tick.
        let period = self.modes[&mode_name].infer_period_us;
        let due = self.last_infer.is_none_or(|t| now.saturating_sub(t) >= period);
        if !due {
            self.publish();
            return Ok(Step::Hold);
        }

        let mode = self.modes.get_mut(&mode_name).expect("built at construction");
        // This mode's own reading of the person where its contract describes one
        // -- its sticks, its ranges, its task's controls -- and the host's
        // command where it does not. Read here, once the cascade has picked the
        // mode, because which operator applies is exactly what it decided.
        let (target, controls) = match input.operators.and_then(|o| o.get(&mode_name)) {
            Some(operator) => (operator.command(), operator.task_controls()),
            None => (*input.command, input.controls.clone()),
        };
        // What the policy was trained to see of the operator: held to its moving
        // band while walking and ramped at its rate, where the contract says so.
        let mut command = match mode.shaper.as_mut() {
            Some(shaper) => shaper.shape(&target),
            None => target,
        };
        let frame = match mode.hook.as_mut() {
            None => {
                let torque: &[f32] =
                    if self.torque_measured { &input.state.tau } else { &self.applied_torque };
                mode.obs.build(input.state, &command, &self.last_action, torque)
            }
            // The robot the hook presents: its state and the operator's command
            // rewritten into the policy frame, and -- for a robot that measures no
            // torque -- the decoder's own applied torque, already in that frame.
            Some(h) => {
                let mut seen = input.state.clone();
                h.hook.before_observation(&mut seen, &mut command, &controls);
                let torque: &[f32] =
                    if self.torque_measured { &seen.tau } else { &h.applied_torque };
                mode.obs.build(&seen, &command, &self.last_action, torque)
            }
        }
        .map_err(Error::Obs)?;
        self.observation.clear();
        self.observation.extend_from_slice(frame);
        self.last_infer = Some(now);
        self.pending = Some(mode_name.clone());
        Ok(Step::Infer { mode: mode_name })
    }

    /// Apply the action the host inferred. Returns the command to publish.
    pub fn resume(&mut self, action: &[f32]) -> Result<&MotorCommand, Error> {
        let name = self.pending.take().ok_or(Error::NotPending)?;
        let mode = self.modes.get(&name).expect("pending names a built mode");
        let want = mode.contract.action_dim();
        if action.len() != want {
            return Err(Error::ActionWidth { got: action.len(), want });
        }
        self.last_action.clear();
        self.last_action.extend_from_slice(action);
        self.alpha = mode.alpha;
        let (kp, kd) = (mode.kp, mode.kd);
        let m = self.modes.get_mut(&name).expect("pending names a built mode");
        // From the same tick's observation, not recomputed: one clock reading
        // for the vector the policy saw and the pose its answer sits on.
        let baseline = m.obs.reference_baseline().map(|b| b.to_vec());
        match m.hook.as_mut() {
            None => m.act.decode(
                action,
                // The state the observation was built from is the state this action
                // answers; the host has not stepped the world in between.
                &RobotState { q: self.cmd.pos.clone(), qd: vec![0.0; self.joints], tau: vec![0.0; self.joints], imu: Default::default() },
                kp,
                kd,
                &mut self.prev_target,
                &mut self.cmd,
                &mut self.applied_torque,
                baseline.as_deref(),
            ),
            // The decoder answers in the frame the policy lives in, from its own
            // previous command in that frame; only what the hook makes of the
            // result reaches the robot.
            Some(h) => {
                let mut raw = action.to_vec();
                h.hook.after_inference(&mut raw);
                let previous = RobotState {
                    q: h.cmd.pos.clone(),
                    qd: vec![0.0; self.joints],
                    tau: vec![0.0; self.joints],
                    imu: Default::default(),
                };
                m.act.decode(
                    &raw,
                    &previous,
                    kp,
                    kd,
                    &mut h.prev_target,
                    &mut h.cmd,
                    &mut h.applied_torque,
                    baseline.as_deref(),
                );
                self.cmd.clone_from(&h.cmd);
                h.hook.after_decode(&mut self.cmd);
                // The robot's own stops, once more: the decoder clamped in the
                // policy's frame, and what the hook made of that is what reaches
                // the joints. The clamp is the last thing between a policy and the
                // robot, so no hook gets to be after it.
                let (lo, hi) = m.act.limits();
                for (w, pos) in self.cmd.pos.iter_mut().enumerate() {
                    *pos = pos.clamp(lo[w], hi[w]);
                }
            }
        }
        self.publish_new_target();
        Ok(&self.cmd)
    }

    // ── regimes ─────────────────────────────────────────────────────────

    fn warm_target(&self) -> Option<&[f32]> {
        let cfg = self.fsm.config();
        if !self.fsm.current().warm_start {
            return None;
        }
        self.modes
            .get(&cfg.warm_start_ref)
            .map(|m| m.home.as_slice())
            .or(Some(&self.default_wire))
    }

    fn enter_hold(&mut self, _mode: &str) {
        self.active_mode = None;
        self.ramping = false;
    }

    fn begin_mode(&mut self, name: &str, state: &RobotState, now: Micros) {
        self.active_mode = Some(name.to_string());
        self.ramp_from.copy_from_slice(&state.q);
        self.ramp_t0 = now;
        self.last_infer = None;
        let target = &self.modes[name].home;
        let tol = self.fsm.config().pose_reach_tol;
        self.ramping = self.fsm.config().mode_switch_ramp_s > 0.0
            && !pose_reached(&state.q, target, tol);
        if !self.ramping {
            self.seed(name);
        }
    }

    /// Hand off to a policy from a known pose.
    ///
    /// Everything carried across ticks is reset, because none of it belongs to
    /// this mode: a zeroed action, the target at its own default pose, and the
    /// history re-padded from the first frame rather than from another policy's.
    fn seed(&mut self, name: &str) {
        let mode = self.modes.get_mut(name).expect("built at construction");
        self.last_action = vec![0.0; mode.contract.action_dim()];
        self.prev_target.copy_from_slice(&mode.home);
        mode.obs.reset();
        if let Some(shaper) = mode.shaper.as_mut() {
            shaper.reset();
        }
        // The decoder's state in the policy frame starts where the policy's home
        // pose is -- the robot has just been ramped to the hook's image of it.
        if let Some(h) = mode.hook.as_mut() {
            h.prev_target.copy_from_slice(&mode.contract.default_wire);
            h.cmd.pos.copy_from_slice(&mode.contract.default_wire);
            h.applied_torque.fill(0.0);
            h.hook.reset();
        }
        self.last_infer = None;
    }

    /// Direct PD toward a pose, the setpoint sliding over `duration_s`.
    fn slide(
        &mut self,
        target: &[f32],
        kp: f32,
        kd: f32,
        state: &RobotState,
        now: Micros,
        duration_s: f32,
    ) {
        let elapsed = now.saturating_sub(self.ramp_t0) as f32 / 1e6;
        let t = if duration_s > 0.0 { (elapsed / duration_s).clamp(0.0, 1.0) } else { 1.0 };
        let mut pose = vec![0.0f32; self.joints];
        for w in 0..self.joints {
            let from = self.ramp_from.get(w).copied().unwrap_or(state.q[w]);
            let to = target.get(w).copied().unwrap_or(state.q[w]);
            pose[w] = from + (to - from) * t;
        }
        self.hold.command_pose(&pose, state, kp, kd, &mut self.cmd, &mut self.applied_torque);
        self.alpha = 1.0;
        self.publish_new_target();
    }

    fn hold_still(&mut self, state: &RobotState, kd: f32) {
        self.hold.hold(state, kd, &mut self.prev_target, &mut self.cmd, &mut self.applied_torque);
        self.alpha = 1.0;
        self.publish_new_target();
    }

    /// Adopt what the regime just wrote into `cmd` as the standing target, then
    /// publish toward it.
    fn publish_new_target(&mut self) {
        self.target.copy_from_slice(&self.cmd.pos);
        self.publish();
    }

    /// The output-rate filter. `alpha == 1` is a pure hold, which is what every
    /// regime but the policy uses -- and is why `pos_des` tracks what was
    /// published there, so entering the policy regime starts the filter from the
    /// current pose with no seed and no step.
    fn publish(&mut self) {
        for w in 0..self.joints {
            self.pos_des[w] += (self.target[w] - self.pos_des[w]) * self.alpha;
            self.cmd.pos[w] = self.pos_des[w];
        }
    }
}

fn build_mode(
    state: &StateConfig,
    contract: Contract,
    setup: &Setup,
    output_dt_s: f64,
) -> Result<ModeRuntime, Error> {
    let control = &contract.layout.control;
    let (kp, kd) = commanded_gains(
        control.kp,
        control.kd,
        &GainTrim {
            kp_scale: state.kp_scale,
            kd_scale: state.kd_scale,
            kp_offset: state.kp_offset,
            kd_offset: state.kd_offset,
        },
    );
    let command_terms = if state.command_terms.is_empty() {
        setup.command_terms.clone()
    } else {
        state.command_terms.clone()
    };
    let hz = control.control_hz.max(1.0);
    // The clock is this mode's own, from its own contract: a bundle mixing a
    // fixed-tempo gait with one whose tempo follows the command has two clocks,
    // and taking the first mode's for both is a policy stepping to the other's
    // tempo. The host's `gait_period` is the fallback for a contract that
    // predates the params, which is where it came from before.
    let terms = &contract.layout.observation.terms;
    let gait = terms.iter().find(|t| t.name == "gait_phase");
    let cadence = gait.and_then(obs::Cadence::from_term).transpose().map_err(Error::Obs)?;
    let gait_period = gait.and_then(|t| t.param_f64("period")).unwrap_or(setup.gait_period);
    let gait_gate_threshold = gait
        .and_then(|t| t.param_f64("command_threshold"))
        .map(|v| v as f32)
        .or(setup.gait_gate_threshold);
    let posture_neutral_height = terms
        .iter()
        .find(|t| t.name == "posture_command")
        .and_then(|t| t.param_f64("neutral_height"))
        .map(|v| v as f32);
    let obs_cfg = ObsConfig {
        terms: terms.iter().map(|t| t.name.clone()).collect(),
        term_history: contract.term_history.clone(),
        term_stride: contract.term_stride.clone(),
        command_terms,
        scales: setup.scales.clone(),
        clip: setup.obs_clip,
        obs_wire_idx: contract.obs_wire_idx.clone(),
        gait_period,
        control_dt: 1.0 / hz,
        gait_gate_threshold,
        cadence,
        posture_neutral_height,
        reference_go_delay_s: setup.reference_go_delay_s,
    };
    let obs = ObservationBuilder::new(obs_cfg, &contract, setup.robot.default_wire.len())
        .map_err(Error::Obs)?;
    // The mode's own home pose, not the robot's. `setup.robot.default_wire` is the
    // first contract's, and the decoder holds every joint the policy does not
    // drive at its default and adds every action to it: a mode trained around
    // another pose was being held at the first mode's. `jumper.five_foot` is the
    // case that showed it -- its carried arm is at the grasp pose, and the first
    // mode's HOME would have folded it back the moment the policy took over.
    let act = ActionDecoder::new(
        crate::action::ActionConfig {
            scale: contract.layout.action_scale,
            clip: setup.action_clip,
            smoothing: setup.action_smoothing,
            effort_limit: control.effort_limit,
            action_wire_idx: contract.action_wire_idx.clone(),
        },
        RobotLimits { default_wire: contract.default_wire.clone(), ..setup.robot.clone() },
    );
    let shaper = contract
        .layout
        .controller
        .as_ref()
        .and_then(|spec| CommandShaper::of(spec, (1.0 / hz) as f32));
    let hook = match &state.hook {
        None => None,
        Some(config) => {
            let hook = crate::hook::build(&HookSetup {
                task: &state.task,
                mode: &state.name,
                config,
                contract: &contract,
            })
            .map_err(Error::Hook)?;
            let joints = contract.default_wire.len();
            Some(HookRuntime {
                hook,
                cmd: MotorCommand::new(joints),
                prev_target: contract.default_wire.clone(),
                applied_torque: vec![0.0; joints],
            })
        }
    };
    // What the task declares for itself (`controls.yaml`'s `task:`) and what
    // its hook answers have to be one story. A hook reading a name the file
    // does not declare reads zero forever, on either device; a file declaring
    // controls no hook answers is a squeeze that does nothing on the robot.
    // Both look, from the pad or the keys, exactly like a control that works.
    let kept: std::collections::BTreeSet<&str> = contract
        .layout
        .controller
        .as_ref()
        .map(|spec| spec.task().collect())
        .unwrap_or_default();
    match &hook {
        Some(h) => {
            if let Some(control) = h.hook.reads().into_iter().find(|c| !kept.contains(c.as_str())) {
                return Err(Error::Hook(format!(
                    "mode '{}': {}'s hook reads {control}, which the task's controls.yaml does \
                     not declare (`task:` declares {kept:?}). It would read zero on every \
                     device, with nothing to say so.",
                    state.name, state.task
                )));
            }
        }
        None if !kept.is_empty() => {
            return Err(Error::Hook(format!(
                "mode '{}': its controls.yaml declares {kept:?} for the task, and the mode asks \
                 for no hook to answer them, so on the robot they would do nothing. Give the \
                 mode its task's hook in the manifest (`\"hook\": {{ ... }}`).",
                state.name
            )));
        }
        None => {}
    }
    let home = match &hook {
        Some(h) => h.hook.home_pose(&contract.default_wire),
        None => contract.default_wire.clone(),
    };
    if home.len() != contract.default_wire.len() {
        return Err(Error::Hook(format!(
            "mode '{}': the hook's home pose has {} joints and the robot {}",
            state.name,
            home.len(),
            contract.default_wire.len()
        )));
    }
    Ok(ModeRuntime {
        infer_period_us: (1e6 / hz) as u64,
        alpha: ema_alpha_from_cutoff(control.action_filter_cutoff_hz, output_dt_s),
        shaper,
        home,
        hook,
        contract,
        obs,
        act,
        kp,
        kd,
    })
}

impl Controller {
    /// The host cannot run this mode's model; decide whether that is fatal.
    ///
    /// Returns whether the mode is now interlocked, so the host can say so
    /// where it already says the engine is a stub.
    ///
    /// **A zero action is only safe for a policy whose decode baseline is the
    /// home pose.** A stub engine returns zeros, and for an ordinary policy
    /// that is the home pose held -- legible, and what `stub_modes` promises.
    /// A reference-residual policy decodes against a recording instead, so the
    /// same zeros mean "track the recording exactly": the recorded motion
    /// played open loop, from a model that never loaded. `jumper.jump` would run
    /// its push-off at 200 Hz with kp=20 on a bundle whose `.rknn` conversion
    /// had simply not been done.
    ///
    /// The host owns the fact (this engine is a stub) and this owns the rule
    /// (a residual must not run on one), because the contract is here.
    pub fn interlock_stub_engine(&mut self, mode: &str) -> bool {
        let residual = self.modes.get(mode).is_some_and(|m| {
            m.contract
                .layout
                .reference
                .as_ref()
                .is_some_and(|r| r.residual_action)
        });
        if residual {
            self.interlocked.insert(mode.to_string());
        }
        residual
    }

    /// Whether this mode's policy is barred from running. For a host reporting
    /// what it will and will not do before anybody enables the motors.
    pub fn is_interlocked(&self, mode: &str) -> bool {
        self.interlocked.contains(mode)
    }

    /// The mode whose recording has just played out, once.
    ///
    /// A host calls this after `tick` and releases the latch of the same name,
    /// which is how a timed policy hands back to the default. Returns `None`
    /// for a gait, which does not end, and for a motion still waiting on its
    /// `go`.
    pub fn take_completed(&mut self) -> Option<String> {
        self.completed.take()
    }
}

/// Angle between the body's up axis and the world's, rad.
pub fn tilt_angle(q: [f32; 4]) -> f32 {
    let up = -obs::projected_gravity(q)[2];
    up.clamp(-1.0, 1.0).acos()
}

pub fn pose_reached(q: &[f32], target: &[f32], tol: f32) -> bool {
    !target.is_empty()
        && q.iter().zip(target).all(|(a, b)| (a - b).abs() <= tol)
        && q.len() >= target.len()
}

#[cfg(test)]
mod tests {
    use crate::config::{ButtonBinding, Gesture};

    fn bind(name: &str, pad: &str, on: Gesture, with: Option<&str>) -> ButtonBinding {
        ButtonBinding {
            event: false,
            name: name.into(),
            pad: Some(pad.into()),
            key: None,
            on,
            with: with.map(str::to_string),
            from: None,
            leaves: Vec::new(),
            click_window_us: WINDOW,
        }
    }

    /// The keyboard's binding: `key` a key or a modifier, `with` a modifier.
    fn key(name: &str, key: &str, on: Gesture, with: Option<&str>) -> ButtonBinding {
        ButtonBinding { pad: None, key: Some(key.into()), ..bind(name, "A", on, with) }
    }

    /// The state the cascade is in, for bindings whose `from` does not care.
    const S: &str = "walk";

    /// The click window the tests count in: the jumper bundle's 300 ms.
    const WINDOW: u64 = 300_000;

    fn pressing(names: &[&str]) -> crate::types::Pad {
        let mut p = crate::types::Pad::default();
        for n in names {
            if let Some(i) = crate::vocabulary::pad_buttons().iter().position(|b| b == n) {
                p.buttons[i] = true;
            }
        }
        p
    }

    fn active(b: &Buttons) -> Vec<&str> {
        b.active().iter().map(String::as_str).collect()
    }

    /// The four gestures, driven through one press of one control.
    ///
    /// `rise` and `fall` are one tick each and `hold` is every tick, so the
    /// interesting frames are the edges and the ones between them. A `toggle`
    /// that acted on the level rather than the edge would be on for the whole
    /// press and off after -- indistinguishable from `hold`, which is why both
    /// are here against the same frames.
    #[test]
    fn each_gesture_fires_where_it_says() {
        let all = [
            bind("r", "A", Gesture::Rise, None),
            bind("f", "A", Gesture::Fall, None),
            bind("h", "A", Gesture::Hold, None),
            bind("t", "A", Gesture::Toggle, None),
        ];
        let (up, down) = (pressing(&[]), pressing(&["A"]));
        let mut b = Buttons::default();
        let no_keys = |_: &str| false;

        b.observe(&all, &up, &no_keys, 0, S);
        assert_eq!(active(&b), Vec::<&str>::new(), "nothing pressed");

        b.observe(&all, &down, &no_keys, 0, S);
        assert_eq!(active(&b), ["h", "r", "t"], "the press edge");

        b.observe(&all, &down, &no_keys, 0, S);
        assert_eq!(active(&b), ["h", "t"], "still held: rise is spent, toggle remembers");

        b.observe(&all, &up, &no_keys, 0, S);
        assert_eq!(active(&b), ["f", "t"], "the release edge, and the toggle survives it");

        b.observe(&all, &up, &no_keys, 0, S);
        assert_eq!(active(&b), ["t"], "only the toggle is left");

        // Pressed again: the toggle goes off, the others fire as before.
        b.observe(&all, &down, &no_keys, 0, S);
        assert_eq!(active(&b), ["h", "r"], "the second press clears the toggle");
    }

    /// The pad's A and the keyboard's Space are two bindings with one name,
    /// and a name is one latch: either switches it on and either switches it
    /// off. Each alone is the control group for the other.
    #[test]
    fn a_pad_binding_and_a_key_binding_share_one_latch() {
        let all = [bind("jump", "A", Gesture::Toggle, None), key("jump", "key_space", Gesture::Toggle, None)];
        let (none, space) = (|_: &str| false, |k: &str| k == "key_space");
        let mut b = Buttons::default();
        b.observe(&all, &pressing(&["A"]), &none, 0, S);
        assert_eq!(active(&b), ["jump"], "the pad alone");
        b.observe(&all, &pressing(&[]), &none, 0, S);
        b.observe(&all, &pressing(&[]), &space, 0, S);
        assert_eq!(active(&b), Vec::<&str>::new(), "Space switched off what A switched on");
        b.observe(&all, &pressing(&[]), &none, 0, S);
        b.observe(&all, &pressing(&[]), &space, 0, S);
        assert_eq!(active(&b), ["jump"], "the key alone");
    }

    /// A modifier is consumed by what it modifies.
    ///
    /// The case this exists for: `menu` alone leaves dance mode, `menu` plus a
    /// direction picks a dance. Without consumption, reaching for the direction
    /// fires the exit first -- which is what the C++ had to solve too.
    #[test]
    fn a_modifier_is_spent_by_what_it_modifies() {
        let all = [
            bind("exit", "menu", Gesture::Fall, None),
            bind("crab", "dpad_up", Gesture::Rise, Some("menu")),
        ];
        let no_keys = |_: &str| false;
        let mut up = crate::types::Pad::default();
        up.dpad_y = 1;
        let mut menu_and_up = pressing(&["menu"]);
        menu_and_up.dpad_y = 1;

        // Menu held, then a direction: the dance fires and the exit does not.
        let mut b = Buttons::default();
        b.observe(&all, &pressing(&["menu"]), &no_keys, 0, S);
        assert_eq!(active(&b), Vec::<&str>::new(), "holding menu asks for nothing yet");
        b.observe(&all, &menu_and_up, &no_keys, 0, S);
        assert_eq!(active(&b), ["crab"]);
        b.observe(&all, &pressing(&[]), &no_keys, 0, S);
        assert_eq!(active(&b), Vec::<&str>::new(), "releasing a spent menu does not exit");

        // ...and it is spent for that press only. Consumed forever would mean
        // one dance selection disables the exit for the rest of the session,
        // which reads as a menu button that stopped working.
        b.observe(&all, &pressing(&["menu"]), &no_keys, 0, S);
        b.observe(&all, &pressing(&[]), &no_keys, 0, S);
        assert_eq!(active(&b), ["exit"], "the next press of menu is its own again");

        // The control group: menu alone, pressed and released, does exit.
        let mut b = Buttons::default();
        b.observe(&all, &pressing(&["menu"]), &no_keys, 0, S);
        b.observe(&all, &pressing(&[]), &no_keys, 0, S);
        assert_eq!(active(&b), ["exit"]);

        // And a direction without the modifier asks for nothing.
        let mut b = Buttons::default();
        b.observe(&all, &up, &no_keys, 0, S);
        assert_eq!(active(&b), Vec::<&str>::new());
    }

    /// The keyboard's modifier is spent the way the pad's is: Ctrl held and 4
    /// pressed is the dance, and Ctrl let go after it does not leave the dance,
    /// while Ctrl pressed and let go on its own does. Either Ctrl is Ctrl. Four
    /// without Ctrl is the control group, and asks for nothing.
    #[test]
    fn a_key_modifier_is_spent_by_what_it_modifies() {
        let mut leave = key("out", "ctrl", Gesture::Fall, None);
        leave.leaves = vec!["dance".into()];
        let all = [key("dance", "key_4", Gesture::Toggle, Some("ctrl")), leave];
        let observe = |b: &mut Buttons, held: &[&str]| {
            b.observe(&all, &pressing(&[]), &|k: &str| held.contains(&k), 0, S);
        };

        let mut b = Buttons::default();
        observe(&mut b, &["key_left_ctrl"]);
        assert_eq!(active(&b), Vec::<&str>::new(), "holding Ctrl asks for nothing yet");
        observe(&mut b, &["key_left_ctrl", "key_4"]);
        assert_eq!(active(&b), ["dance"]);
        observe(&mut b, &[]);
        assert_eq!(active(&b), ["dance"], "letting go of a spent Ctrl left the dance");
        observe(&mut b, &["key_right_ctrl"]);
        observe(&mut b, &[]);
        assert_eq!(active(&b), Vec::<&str>::new(), "Ctrl on its own leaves the dance");
        assert!(b.latched.is_empty(), "and the latch is gone, not only quiet");

        let mut b = Buttons::default();
        observe(&mut b, &["key_4"]);
        assert_eq!(active(&b), Vec::<&str>::new(), "4 without Ctrl");
    }

    /// Up alone is a gesture and `menu` + up a dance, and a press of up with
    /// `menu` held is the dance only: a bare binding on a control another binding
    /// chords with a held modifier stays quiet. The keyboard's the same, 1 and
    /// Ctrl + 1. Up alone, and 1 alone, are the control groups.
    #[test]
    fn a_chord_silences_the_bare_press_of_its_control() {
        let all = [
            bind("hello", "dpad_up", Gesture::Toggle, None),
            bind("crab", "dpad_up", Gesture::Toggle, Some("menu")),
            key("hello", "key_1", Gesture::Toggle, None),
            key("crab", "key_1", Gesture::Toggle, Some("ctrl")),
        ];
        let mut up = pressing(&[]);
        up.dpad_y = 1;
        let mut menu_up = pressing(&["menu"]);
        menu_up.dpad_y = 1;
        let frame = |b: &mut Buttons, pad: &crate::types::Pad, keys: &[&str]| {
            b.observe(&all, pad, &|k: &str| keys.contains(&k), 0, S);
        };
        for (pad, keys, want) in [
            (&menu_up, &[][..], "crab"),
            (&up, &[][..], "hello"),
            (&pressing(&[]), &["key_left_ctrl", "key_1"][..], "crab"),
            (&pressing(&[]), &["key_1"][..], "hello"),
        ] {
            // Not exclusive: there, the chord's latch coming on last would hide
            // the bare one's coming on with it.
            let mut b = Buttons::new(false);
            if keys.contains(&"key_left_ctrl") {
                frame(&mut b, &pressing(&[]), &["key_left_ctrl"]);
            } else if pad.button("menu") {
                frame(&mut b, &pressing(&["menu"]), &[]);
            }
            frame(&mut b, pad, keys);
            assert_eq!(active(&b), [want], "{keys:?} / dpad {}", pad.dpad_y);
        }
    }

    /// `from` gates entering and nothing else: A does not switch the jump on in
    /// the claw mode, does in the default one, and switches it off from inside
    /// the jump, which is not in its `from` -- a latch nobody could switch off
    /// would be the failure the C++ learned about the hard way.
    #[test]
    fn from_gates_entering_and_not_leaving() {
        let mut jump = bind("jump", "A", Gesture::Toggle, None);
        jump.from = Some(vec!["walk".into()]);
        let all = [jump];
        let press = |b: &mut Buttons, state: &str| {
            b.observe(&all, &pressing(&["A"]), &|_| false, 0, state);
            b.observe(&all, &pressing(&[]), &|_| false, 0, state);
        };
        let mut b = Buttons::default();
        press(&mut b, "claw");
        assert_eq!(active(&b), Vec::<&str>::new(), "A entered the jump from the claw mode");
        press(&mut b, "walk");
        assert_eq!(active(&b), ["jump"], "the control group: from walk it does");
        press(&mut b, "jump");
        assert_eq!(active(&b), Vec::<&str>::new(), "and from inside the jump it leaves");
    }

    /// With `exclusive`, a latch coming on releases the rest: RB in the left
    /// claw mode is the right claw, and LB there the left again. Without it --
    /// the control group -- both stay latched and the cascade's order would
    /// decide, which is how LB in the right claw mode used to do nothing.
    #[test]
    fn one_latch_at_a_time_when_exclusive() {
        let all = [bind("claw_left", "LB", Gesture::Toggle, None), bind("claw_right", "RB", Gesture::Toggle, None)];
        let press = |b: &mut Buttons, button: &str| {
            b.observe(&all, &pressing(&[button]), &|_| false, 0, S);
            b.observe(&all, &pressing(&[]), &|_| false, 0, S);
        };
        let mut b = Buttons::new(true);
        press(&mut b, "LB");
        press(&mut b, "RB");
        assert_eq!(active(&b), ["claw_right"]);
        press(&mut b, "LB");
        assert_eq!(active(&b), ["claw_left"]);
        b.toggle("claw_right");
        assert_eq!(active(&b), ["claw_right"], "a host's toggle is one latch at a time too");

        let mut b = Buttons::new(false);
        press(&mut b, "LB");
        press(&mut b, "RB");
        assert_eq!(active(&b), ["claw_left", "claw_right"]);
    }

    /// A leave releases what it names and is never active itself; it fires only
    /// from its own `from`, so Menu let go in the default mode leaves nothing
    /// behind; and the chord that entered the dance keeps Menu's fall quiet.
    #[test]
    fn a_leave_releases_its_latches_and_is_never_on() {
        let mut out = bind("out", "menu", Gesture::Fall, None);
        out.leaves = vec!["dance".into()];
        out.from = Some(vec!["dance".into()]);
        let all = [bind("dance", "dpad_up", Gesture::Toggle, Some("menu")), out];
        let mut up = pressing(&["menu"]);
        up.dpad_y = 1;
        let frame = |b: &mut Buttons, pad: &crate::types::Pad, state: &str| {
            b.observe(&all, pad, &|_| false, 0, state);
        };
        let mut b = Buttons::default();
        frame(&mut b, &pressing(&["menu"]), "walk");
        frame(&mut b, &up, "walk");
        assert_eq!(active(&b), ["dance"]);
        frame(&mut b, &pressing(&[]), "dance");
        assert_eq!(active(&b), ["dance"], "the menu that chose the dance left it");
        frame(&mut b, &pressing(&["menu"]), "dance");
        frame(&mut b, &pressing(&[]), "dance");
        assert_eq!(active(&b), Vec::<&str>::new(), "menu alone did not leave the dance");

        // Not from `walk`: the latch it names stays, as it would if menu were
        // bound to nothing.
        b.toggle("dance");
        frame(&mut b, &pressing(&["menu"]), "walk");
        frame(&mut b, &pressing(&[]), "walk");
        assert_eq!(active(&b), ["dance"], "a leave fired outside its from");
    }

    /// The operator going away drops everything and re-arms every edge, so a
    /// control still down when they return is not read as a fresh press.
    #[test]
    fn forgetting_re_arms_the_edges() {
        let all = [bind("t", "A", Gesture::Toggle, None)];
        let mut b = Buttons::default();
        b.observe(&all, &pressing(&["A"]), &|_| false, 0, S);
        assert_eq!(active(&b), ["t"]);

        b.forget();
        b.observe(&all, &pressing(&["A"]), &|_| false, 0, S);
        assert_eq!(active(&b), ["t"], "a still-held control reads as one fresh press");
        b.observe(&all, &pressing(&["A"]), &|_| false, 0, S);
        assert_eq!(active(&b), ["t"], "...and not as one per tick");
    }

    // ── Clicks ──────────────────────────────────────────────────────────

    /// The same binding as an event: fires once, so a test can count firings.
    fn ev(name: &str, pad: &str, on: Gesture) -> ButtonBinding {
        ButtonBinding { event: true, ..bind(name, pad, on, None) }
    }

    const MS: u64 = 1000;

    /// `A` pressed at each of `presses` (µs) and held 40 ms, observed every
    /// `step` µs up to `until`: each name active on a frame, with its time.
    fn clicking(all: &[ButtonBinding], presses: &[u64], step: u64, until: u64) -> Vec<(u64, String)> {
        let mut b = Buttons::default();
        let mut seen = vec![];
        let mut t = 0;
        while t <= until {
            let down = presses.iter().any(|&p| t >= p && t < p + 40 * MS);
            b.observe(all, &pressing(if down { &["A"] } else { &[] }), &|_| false, t, S);
            seen.extend(b.active().iter().map(|n| (t, n.clone())));
            t += step;
        }
        seen
    }

    fn names(seen: &[(u64, String)]) -> Vec<&str> {
        seen.iter().map(|(_, n)| n.as_str()).collect()
    }

    /// One control, three counts bound, and each count fires its own binding --
    /// one, once.
    ///
    /// A count below the longest bound waits out the window, because another
    /// press could still make it longer; the longest fires on the press that
    /// reaches it. The control group is two presses further apart than the
    /// window: two singles, never a double.
    #[test]
    fn a_count_fires_the_click_it_comes_to() {
        let all = [
            ev("once", "A", Gesture::Single),
            ev("twice", "A", Gesture::Double),
            ev("thrice", "A", Gesture::Triple),
        ];
        let run = |presses: &[u64]| clicking(&all, presses, MS, 1500 * MS);

        let one = run(&[0]);
        assert_eq!(names(&one), ["once"]);
        assert!(one[0].0 > WINDOW && one[0].0 <= WINDOW + MS, "fires as the window runs out: {one:?}");

        let two = run(&[0, 150 * MS]);
        assert_eq!(names(&two), ["twice"]);
        assert!(two[0].0 > 150 * MS + WINDOW, "a triple could still come, so it waits: {two:?}");

        let three = run(&[0, 150 * MS, 300 * MS]);
        assert_eq!(three, [(300 * MS, "thrice".to_string())], "the longest does not wait");

        // Control group: the second press comes after the window.
        assert_eq!(names(&run(&[0, 400 * MS])), ["once", "once"]);
    }

    /// With nothing longer bound, a click fires on the press that completes it.
    ///
    /// The wait is the price of sharing a control with a longer click and only
    /// that: a `double` alone fires on its second press, and a `single` alone
    /// on its first -- a `toggle`, as far as anyone holding the pad can tell.
    #[test]
    fn the_longest_click_on_a_control_does_not_wait() {
        let double = [ev("twice", "A", Gesture::Double)];
        assert_eq!(clicking(&double, &[0, 150 * MS], MS, 900 * MS), [(150 * MS, "twice".to_string())]);
        assert_eq!(clicking(&double, &[0], MS, 900 * MS), [], "one press of a double is nothing");

        let single = [ev("once", "A", Gesture::Single)];
        assert_eq!(clicking(&single, &[0], MS, 900 * MS), [(0, "once".to_string())]);
    }

    /// The window is a time, so how often a host observes does not move it.
    ///
    /// This is the objection that kept clicks out of the dictionary: a count
    /// kept in frames would be a different gesture at the board's pad rate
    /// (50 Hz) than in a browser's frame loop. The same presses at 50 Hz and at
    /// 1 kHz have to come to the same clicks. The control group is the window
    /// itself: 250 ms apart is a double at both rates, 350 ms apart is two
    /// singles at both -- so the answer does change, with time and only time.
    #[test]
    fn a_click_is_counted_in_time_not_in_frames() {
        let all = [ev("once", "A", Gesture::Single), ev("twice", "A", Gesture::Double)];
        for (gap, expect) in [(250 * MS, vec!["twice"]), (350 * MS, vec!["once", "once"])] {
            for step in [20 * MS, MS] {
                let seen = clicking(&all, &[0, gap], step, 1500 * MS);
                assert_eq!(names(&seen), expect, "{gap} µs apart, observed every {step} µs");
            }
        }
    }

    /// A click switches a mode the way `toggle` does: click to enter, the same
    /// click again to leave, and the latch holds between.
    #[test]
    fn a_click_switches_a_mode_as_toggle_does() {
        let all = [bind("dance", "A", Gesture::Double, None)];
        let seen = clicking(&all, &[0, 150 * MS, 1000 * MS, 1150 * MS, 2000 * MS], MS, 2600 * MS);
        let on: Vec<u64> = seen.iter().map(|(t, _)| *t).collect();
        assert_eq!(on.first(), Some(&(150 * MS)), "entered on the second press");
        assert_eq!(on.last(), Some(&(1150 * MS - MS)), "left on the second press of the next double");
        assert_eq!(on.len() as u64, (1000 * MS) / MS, "held on without a gap in between");
        // The lone press at 2 s is one click, and nothing is bound to one.
    }

    /// A key clicks on its own, and the pad on its own: Space twice is a
    /// double, and Space and then A are one press of each -- two controls, two
    /// counts, since the keyboard stopped being a copy of the pad.
    #[test]
    fn a_key_clicks_on_its_own() {
        let all = [bind("dance", "A", Gesture::Double, None), key("dance", "key_space", Gesture::Double, None)];
        let frame = |b: &mut Buttons, pad: &[&str], keys: &[&str], t: u64| {
            b.observe(&all, &pressing(pad), &|k: &str| keys.contains(&k), t, S);
        };

        let mut b = Buttons::default();
        frame(&mut b, &[], &["key_space"], 0);
        frame(&mut b, &[], &[], 50 * MS);
        frame(&mut b, &[], &["key_space"], 150 * MS);
        assert_eq!(active(&b), ["dance"], "Space twice");

        let mut b = Buttons::default();
        frame(&mut b, &[], &["key_space"], 0);
        frame(&mut b, &[], &[], 50 * MS);
        frame(&mut b, &["A"], &[], 150 * MS);
        frame(&mut b, &[], &[], 200 * MS);
        frame(&mut b, &[], &[], 600 * MS);
        assert_eq!(active(&b), Vec::<&str>::new(), "Space and then A made a double");
    }

    /// Clicks under a modifier count only while it is held, and spend it as
    /// any modified binding does.
    #[test]
    fn a_click_under_a_modifier_counts_while_it_is_held() {
        let all = [
            bind("exit", "menu", Gesture::Fall, None),
            bind("dance", "dpad_right", Gesture::Double, Some("menu")),
        ];
        let right = |menu: bool| {
            let mut p = pressing(if menu { &["menu"] } else { &[] });
            p.dpad_x = 1;
            p
        };
        let menu = pressing(&["menu"]);
        let none = pressing(&[]);

        let mut b = Buttons::default();
        for (pad, t) in [(&menu, 0), (&right(true), 10), (&menu, 60), (&right(true), 150)] {
            b.observe(&all, pad, &|_| false, t * MS, S);
        }
        assert_eq!(active(&b), ["dance"]);
        b.observe(&all, &none, &|_| false, 200 * MS, S);
        assert_eq!(active(&b), ["dance"], "letting go of the spent menu does not exit");

        // Control group: the same two clicks without menu count nothing.
        let mut b = Buttons::default();
        for (pad, t) in [(&right(false), 0), (&none, 60), (&right(false), 150), (&none, 600)] {
            b.observe(&all, pad, &|_| false, t * MS, S);
        }
        assert_eq!(active(&b), Vec::<&str>::new());
    }

    /// The operator going away drops a count in progress, so the first press
    /// after they return is a first click, not a second.
    #[test]
    fn forgetting_drops_a_count() {
        let all = [ev("twice", "A", Gesture::Double)];
        let mut b = Buttons::default();
        let at = |b: &mut Buttons, down: bool, t: u64| {
            b.observe(&all, &pressing(if down { &["A"] } else { &[] }), &|_| false, t * MS, S);
            active(b).iter().map(|s| s.to_string()).collect::<Vec<_>>()
        };
        at(&mut b, true, 0);
        at(&mut b, false, 50);
        b.forget();
        assert!(at(&mut b, true, 100).is_empty(), "one press since the operator came back");
        at(&mut b, false, 150);
        assert_eq!(at(&mut b, true, 200), ["twice"], "and the next one is the second");
    }

    use super::*;
    use crate::config::{FsmConfig, EXAMPLE};

    const JOINTS: [&str; 2] = ["a", "b"];

    /// A two-joint contract. `home` is its default pose, and it is deliberately
    /// **not** zero: a ramp toward zero and a ramp toward the right pose are hard
    /// to tell apart when the right pose is zero.
    fn contract(home: [f32; 2], hz: f64, cutoff: f64) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": {}, "b": {}}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0,
                             "control_hz": {hz}, "action_filter_cutoff_hz": {cutoff}}},
                "observation": {{"dim": 6, "history_length": 1,
                                 "terms": [{{"name":"joint_pos","dim":2}},
                                           {{"name":"joint_vel","dim":2}},
                                           {{"name":"actions","dim":2}}]}},
                "action": {{"dim": 2}}}}"#,
            home[0], home[1]
        );
        Contract::from_str(&text, &JOINTS.map(String::from)).unwrap()
    }

    fn setup() -> Setup {
        Setup {
            reference_go_delay_s: crate::trajectory::DEFAULT_GO_DELAY_S,
            robot: RobotLimits {
                default_wire: vec![0.0, 0.0],
                joint_pos_lo: vec![-3.3, -3.3],
                joint_pos_hi: vec![3.3, 3.3],
                joint_pos_rate_limit: 0.0,
            },
            scales: obs::Scales::default(),
            obs_clip: 100.0,
            action_clip: 100.0,
            action_smoothing: 0.0,
            command_terms: vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()],
            gait_period: 0.32,
            gait_gate_threshold: Some(0.05),
            output_rate_hz: 1000.0,
            // Simulated: the torque comes from the solver. `device.rs`'s tests
            // use the robot's capability instead, and the two take different
            // paths through `tick` because of it.
            source: SourceCapability::mjlab(),
        }
    }

    fn controller(home: [f32; 2], hz: f64, cutoff: f64) -> Controller {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let mut contracts = HashMap::new();
        for name in ["locomotion", "carry", "jump"] {
            contracts.insert(name.to_string(), contract(home, hz, cutoff));
        }
        Controller::new(cfg, setup(), contracts, 0).unwrap()
    }

    /// A contract whose policy is a residual on a recording, optionally naming
    /// the button that starts it.
    fn jump_contract(go_event: Option<&str>) -> Contract {
        let go = go_event.map_or("null".to_string(), |g| format!("\"{g}\""));
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.1, "b": 0.1}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0,
                             "control_hz": 200.0}},
                "observation": {{"dim": 1, "history_length": 1,
                                 "terms": [{{"name":"jump_phase","dim":1}}]}},
                "action": {{"dim": 2}},
                "reference": {{"file": "t.json", "residual_action": true, "rec_hz": 100.0,
                               "control_hz": 50.0, "t_go": 0.1, "go_frame": 10,
                               "duration": 0.3, "span": 0.2, "lookahead_s": [0.01],
                               "baseline": "q_cmd", "baseline_lead_s": 0.005,
                               "n_frames": 30, "n_cmd": 15, "go_event": {go}}}}}"#
        );
        let wire = JOINTS.map(String::from);
        let mut c = Contract::from_str(&text, &wire).unwrap();
        let q: Vec<Vec<f32>> = (0..30).map(|f| vec![f as f32, 0.0]).collect();
        let qc: Vec<Vec<f32>> = (0..15).map(|f| vec![100.0 + f as f32, 0.0]).collect();
        let table = serde_json::json!({"joint_order": ["a","b"], "q": q, "q_cmd": qc});
        c.attach_reference(&table.to_string(), &wire).unwrap();
        c
    }

    /// `EXAMPLE` plus one event binding, or plus nothing.
    fn cascade_with_event(name: Option<&str>) -> FsmConfig {
        let mut text = EXAMPLE.to_string();
        if let Some(n) = name {
            text.push_str(&format!(
                "\n[[fsm.button]]\nname = \"{n}\"\npad = \"A\"\non = \"fall\"\nevent = true\n"
            ));
        }
        FsmConfig::parse(&text).unwrap()
    }

    fn with_jump(cascade: Option<&str>, contract_event: Option<&str>) -> Result<Controller, Error> {
        let mut contracts = HashMap::new();
        for name in ["locomotion", "carry"] {
            contracts.insert(name.to_string(), contract([0.1, 0.1], 200.0, 0.0));
        }
        contracts.insert("jump".to_string(), jump_contract(contract_event));
        Controller::new(cascade_with_event(cascade), setup(), contracts, 0)
    }

    /// Drive a latched jump for six seconds and return every mode that was
    /// asked to infer.
    fn inferred_modes(c: &mut Controller) -> std::collections::BTreeSet<String> {
        let (home, cmd) = ([0.1f32, 0.1], Command::default());
        let s = at(home);
        let latched: std::collections::BTreeSet<String> = ["jump".to_string()].into();
        let mut seen = std::collections::BTreeSet::new();
        for tick in 0..6_000u64 {
            let mut i = input(&s, &cmd);
            i.buttons = &latched;
            if let Step::Infer { mode } = c.tick(&i, tick * 1_000).unwrap() {
                seen.insert(mode);
                c.resume(&[0.0, 0.0]).unwrap();
            }
        }
        seen
    }

    #[test]
    fn a_residual_policy_will_not_run_on_a_stub_engine() {
        // The hazard, in one sentence: a stub returns zeros, and for a residual
        // policy zeros mean "track the recording exactly" -- the recorded jump
        // played open loop, at 200 Hz and kp=20, from a model that never
        // loaded. For an ordinary policy the same zeros are the home pose held,
        // which is what `stub_modes` promises and is true.
        //
        // `device.rs` carried that warning in its module docstring for as long
        // as there was no reference support, and the design doc claimed the
        // interlock existed while it did not.
        let mut c = with_jump(Some("jump_go"), Some("jump_go")).unwrap();
        assert!(c.interlock_stub_engine("jump"), "jump's action is a residual");
        assert!(c.is_interlocked("jump"));

        let ran = inferred_modes(&mut c);
        assert!(!ran.contains("jump"), "a barred mode inferred anyway: {ran:?}");

        // Two controls, because "never infers" would also be true of a
        // controller that is simply broken.
        //
        // One: the same bundle without the interlock does reach the mode, so the
        // loop above is capable of getting there.
        let mut open = with_jump(Some("jump_go"), Some("jump_go")).unwrap();
        assert!(inferred_modes(&mut open).contains("jump"), "the loop cannot reach jump");

        // Two: a stub on a mode that is *not* a residual is not interlocked --
        // its zeros are the home pose, and barring it would make a bundle
        // pending conversion useless for no safety gain.
        let mut gait = with_jump(None, None).unwrap();
        assert!(!gait.interlock_stub_engine("locomotion"), "a gait needs no interlock");
        assert!(!gait.is_interlocked("locomotion"));
        assert!(inferred_modes(&mut gait).contains("locomotion"));
    }

    #[test]
    fn a_recorded_motion_hands_back_when_it_ends_and_a_gait_never_does() {
        // The difference between the two kinds of policy, and the reason this
        // exists: a gait runs until somebody stops it, a recording is a thing
        // that ends. Without the hand-back the robot holds the motion's last
        // frame -- a landing pose, indefinitely -- and the only sign is that it
        // stopped moving, which looks like every other way a robot stops.
        let mut c = with_jump(Some("jump_go"), Some("jump_go")).unwrap();
        let (home, cmd) = ([0.1f32, 0.1], Command::default());
        let s = at(home);

        // `EXAMPLE` warm-starts for 2 s and ramps for 1 s before any policy
        // runs, and the recording is 0.3 s at 200 Hz; six seconds of 1 kHz
        // ticks clears all three with room to spare.
        //
        // `jump_go` is pressed **after** the mode is running, which is the only
        // way it counts: a control already down when a mode starts is not a
        // fresh press, the same rule `Buttons::forget` applies when an operator
        // comes back. Holding it from tick zero -- which this test did first --
        // reads as held-since-before and correctly triggers nothing.
        let latched: std::collections::BTreeSet<String> = ["jump".to_string()].into();
        let with_go: std::collections::BTreeSet<String> =
            ["jump".to_string(), "jump_go".to_string()].into_iter().collect();
        let (mut completed, mut ran_jump, mut pressed_at) = (vec![], false, None);
        for tick in 0..6_000u64 {
            let mut i = input(&s, &cmd);
            // Down for five ticks once the mode is up, then released.
            i.buttons = match pressed_at {
                Some(t) if tick < t + 5 => &with_go,
                _ => &latched,
            };
            if let Step::Infer { mode } = c.tick(&i, tick * 1_000).unwrap() {
                if mode == "jump" && !ran_jump {
                    ran_jump = true;
                    pressed_at = Some(tick + 1);
                }
                c.resume(&[0.0, 0.0]).unwrap();
            }
            if let Some(done) = c.take_completed() {
                completed.push(done);
            }
        }
        // Said separately so a failure names the stage that stalled rather than
        // only the symptom.
        assert!(ran_jump, "the jump mode never ran; warm start or ramp did not clear");
        assert!(!completed.is_empty(), "the recording ended and nothing said so");
        assert!(completed.iter().all(|m| m == "jump"), "{completed:?}");

        // A gait in the same loop never reports itself finished, however long
        // it runs. Without this the assertion above would pass for a controller
        // that simply reported every mode complete on every tick.
        let mut g = with_jump(None, None).unwrap();
        let only_walk: std::collections::BTreeSet<String> = Default::default();
        for tick in 0..6_000u64 {
            let mut i = input(&s, &cmd);
            i.buttons = &only_walk;
            if let Step::Infer { .. } = g.tick(&i, tick * 1_000).unwrap() {
                g.resume(&[0.0, 0.0]).unwrap();
            }
            assert_eq!(g.take_completed(), None, "tick {tick}: a gait does not end");
        }
    }

    /// `EXAMPLE` with the jump answerable to the tilt check, as the shipped
    /// jumper bundle's every motion is ("No state sets `skip_tilt_check`").
    fn with_jump_under_the_tilt_check() -> Controller {
        let text = EXAMPLE.replace("skip_tilt_check = true\n", "");
        assert_ne!(text, EXAMPLE, "the exemption was not where this expected it");
        let mut contracts = HashMap::new();
        for name in ["locomotion", "carry"] {
            contracts.insert(name.to_string(), contract([0.1, 0.1], 200.0, 0.0));
        }
        contracts.insert("jump".to_string(), jump_contract(None));
        Controller::new(FsmConfig::parse(&text).unwrap(), setup(), contracts, 0).unwrap()
    }

    /// Drive `c` for `ticks` at 1 kHz with `latched` pressed: level until the
    /// mode `cut` has inferred for 50 ms, then tipped past the tilt limit for
    /// half a second, then level again. Returns every mode that inferred
    /// after the robot was back on its feet, and what `take_completed` said.
    /// With `honour`, a completion releases the latch of that name, which is
    /// what every host does; without it the latch stays, which is the control.
    fn cut_short_by_a_tilt(
        c: &mut Controller,
        cut: &str,
        latched: &mut std::collections::BTreeSet<String>,
        honour: bool,
    ) -> (std::collections::BTreeSet<String>, Vec<String>) {
        let (home, cmd) = ([0.1f32, 0.1], Command::default());
        let level = at(home);
        let mut tipped = at(home);
        tipped.imu.quat = [0.707, 0.707, 0.0, 0.0]; // a quarter turn about x: past 0.8 rad either way round
        let (mut running_since, mut tipped_at) = (None, None);
        let (mut after, mut completed) = (std::collections::BTreeSet::new(), vec![]);
        for tick in 0..12_000u64 {
            let fallen = matches!(tipped_at, Some(t) if tick >= t && tick < t + 500);
            let mut i = input(if fallen { &tipped } else { &level }, &cmd);
            i.buttons = latched;
            let step = c.tick(&i, tick * 1_000).unwrap();
            if let Some(done) = c.take_completed() {
                completed.push(done.clone());
                if honour {
                    latched.remove(&done);
                }
            }
            if let Step::Infer { mode } = step {
                c.resume(&[0.0, 0.0]).unwrap();
                if mode == cut && running_since.is_none() {
                    running_since = Some(tick);
                }
                if matches!(tipped_at, Some(t) if tick >= t + 500) {
                    after.insert(mode);
                }
            }
            if tipped_at.is_none() && matches!(running_since, Some(t) if tick >= t + 50) {
                tipped_at = Some(tick);
            }
        }
        assert!(running_since.is_some(), "{cut} never ran; warm start or ramp did not clear");
        assert!(tipped_at.is_some());
        (after, completed)
    }

    #[test]
    fn a_motion_the_tilt_fallback_cuts_short_is_over_and_a_gait_is_not() {
        // The failure this pins: a latched jump interrupted by a fall stayed
        // latched through `safe`, and the cascade re-entered the jump the
        // moment the robot was level -- from the start, with nobody pressing
        // anything. The completion the host releases latches on was only
        // reported when a recording played out, and a fall cut it short.
        let mut c = with_jump_under_the_tilt_check();
        let mut latched: std::collections::BTreeSet<String> = ["jump".to_string()].into();
        let (after, completed) = cut_short_by_a_tilt(&mut c, "jump", &mut latched, true);
        assert_eq!(completed, ["jump"], "the cut-short jump was not reported over, exactly once");
        assert!(latched.is_empty(), "the host released it on that report");
        assert!(after.contains("locomotion"), "the robot never got back on its feet: {after:?}");
        assert!(!after.contains("jump"), "the jump played again after the fall: {after:?}");

        // Control one: the same fall with the report ignored -- the latch kept
        // -- and the cascade does re-enter the jump, so the assertion above is
        // about the release and not about a loop that cannot reach the mode.
        let mut c = with_jump_under_the_tilt_check();
        let mut kept: std::collections::BTreeSet<String> = ["jump".to_string()].into();
        let (after, _) = cut_short_by_a_tilt(&mut c, "jump", &mut kept, false);
        assert!(after.contains("jump"), "with the latch kept the jump should have played again: {after:?}");

        // Control two: a gait cut short the same way reports nothing. Its latch
        // is the operator's -- `carry` is LB held -- and releasing it here would
        // be the controller letting go of a dead-man switch on its own.
        let mut c = with_jump_under_the_tilt_check();
        let mut carry: std::collections::BTreeSet<String> = ["carry".to_string()].into();
        let (after, completed) = cut_short_by_a_tilt(&mut c, "carry", &mut carry, true);
        assert!(completed.is_empty(), "a gait does not end: {completed:?}");
        assert!(after.contains("carry"), "and LB still held, the carry resumes: {after:?}");
    }

    #[test]
    fn releasing_a_latch_does_not_set_one() {
        // `release` rather than `toggle`, which would latch a button that
        // happened to be clear -- re-entering the mode the tick after it ended,
        // forever.
        let mut b = Buttons::default();
        b.toggle("jump");
        assert_eq!(active(&b), ["jump"]);
        b.release("jump");
        assert!(active(&b).is_empty());
        b.release("jump");
        assert!(active(&b).is_empty(), "releasing twice must not latch it");
    }

    #[test]
    fn a_go_binding_missing_either_half_is_refused() {
        // Each half alone is a control that loads and does nothing, and the two
        // failures look completely different on a bench: a motion that never
        // starts, and a button that never does anything. Neither reports
        // itself, so both are refused at construction.
        with_jump(Some("jump_go"), Some("jump_go")).expect("both halves present");

        let e = with_jump(None, Some("jump_go")).err().expect("refused");
        assert!(
            matches!(&e, Error::GoEvent(m) if m.contains("Nothing would ever start it")),
            "{e}"
        );

        let e = with_jump(Some("jump_go"), None).err().expect("refused");
        assert!(
            matches!(&e, Error::GoEvent(m) if m.contains("no mode reads it")),
            "{e}"
        );

        // And a cascade that declares no event at all is the ordinary case,
        // which must keep working: the motion falls back to the timer.
        with_jump(None, None).expect("no reference button anywhere is fine");
    }

    #[test]
    fn an_event_button_is_not_a_dead_key() {
        // `[[fsm.button]]` is otherwise refused when no rule latches it -- that
        // check is what `event = true` opts out of, and it must still hold for
        // everything else.
        let mut text = EXAMPLE.to_string();
        text.push_str("\n[[fsm.button]]\nname = \"nobody\"\npad = \"A\"\non = \"fall\"\n");
        assert!(FsmConfig::parse(&text).is_err(), "a button no rule reacts to is still dead");
        assert!(cascade_with_event(Some("nobody")).buttons.iter().any(|b| b.event));
    }

    fn at(q: [f32; 2]) -> RobotState {
        let mut s = RobotState::new(2);
        s.q = q.to_vec();
        s.imu.quat = [1.0, 0.0, 0.0, 0.0];
        s.imu.valid = true;
        s
    }

    /// Nothing pressed. A `static` so the borrow outlives the call -- these
    /// tests never press anything, and the FSM's own tests cover the buttons.
    static NONE: std::sync::LazyLock<std::collections::BTreeSet<String>> =
        std::sync::LazyLock::new(Default::default);

    fn input<'a>(state: &'a RobotState, cmd: &'a Command) -> TickInput<'a> {
        TickInput {
            state,
            command: cmd,
            operators: None,
            state_fresh: true,
            command_fresh: true,
            buttons: &NONE,
            gripper_active: false,
            controls: &IDLE,
        }
    }

    /// No reading of a task's controls: what a host without one hands over.
    static IDLE: crate::types::TaskControls = crate::types::TaskControls::NONE;

    /// A pad frame with nothing touched.
    const PAD_IDLE: crate::types::Pad =
        crate::types::Pad { connected: false, buttons: [false; 10], dpad_x: 0, dpad_y: 0, axes: [0.0; 6] };

    /// Drive until the controller asks for an inference, or give up.
    fn run_to_infer(c: &mut Controller, q: [f32; 2], from_us: Micros) -> (Step, Micros) {
        let cmd = Command::default();
        let mut now = from_us;
        for _ in 0..8000 {
            let s = at(q);
            let step = c.tick(&input(&s, &cmd), now).unwrap();
            if let Step::Infer { .. } = step {
                return (step, now);
            }
            now += 1_000;
        }
        panic!("never asked for an inference");
    }

    /// A two-joint contract whose observation is the pose command, carrying the
    /// controller block `jumper.five_foot` exports: the walk, and the body pose
    /// held to 15 degrees while walking and ramped at `rate` rad/s.
    fn posed_contract(home: [f32; 2], rate: f32) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": {a}, "b": {b}}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": 5, "history_length": 1,
                                 "terms": [{{"name":"base_pose","dim":3}},
                                           {{"name":"actions","dim":2}}]}},
                "action": {{"dim": 2}},
                "command_ranges": {{
                  "twist": {{"lin_vel_x": [-0.5, 0.5], "lin_vel_y": [-0.5, 0.5],
                             "ang_vel_z": [-2.0, 2.0]}},
                  "body_pose": {{"pitch": [-0.35, 0.35], "roll": [-0.26, 0.26],
                                 "twist": [-0.52, 0.52]}}}},
                "controller": {{"schema": "operator_controller/2",
                  "command": [
                    {{"term": "twist", "feeds": "velocity_commands",
                      "axes": [{{"name": "lin_vel_x"}}, {{"name": "lin_vel_y"}},
                               {{"name": "ang_vel_z"}}]}},
                    {{"term": "body_pose", "feeds": "base_pose",
                      "axes": [{{"name": "pitch"}}, {{"name": "roll"}}, {{"name": "twist"}}],
                      "bands": {{"standing_below": 0.06, "velocity_term": "twist",
                                 "moving": {{"pitch": [-0.26, 0.26], "roll": [-0.26, 0.26],
                                             "twist": [-0.26, 0.26]}}}},
                      "max_rate": {rate}}}],
                  "devices": {{
                    "gamepad": {{"scheme": "absolute", "layout": "xbox",
                      "axes": {{"lin_vel_x": {{"source": "Ly", "sign": -1}},
                                "lin_vel_y": {{"source": "Lx", "sign": -1}},
                                "twist": {{"source": "Rx", "sign": -1, "travel": [0.0, 0.5, 0.5, 0.75]}},
                                "ang_vel_z": {{"source": "Rx", "sign": -1, "travel": [0.5, 1.0]}},
                                "pitch": {{"source": "Ry", "sign": -1}},
                                "roll": {{"source": "Rx", "sign": 1, "shifted": true}}}},
                      "shift": {{"button": "R3", "gesture": "toggle"}},
                      "deadzone": "device_reported_rescaled", "release_button": "B"}},
                    "keyboard": {{"scheme": "keys", "full_after_s": 2.0,
                      "axes": {{"lin_vel_x": {{"+": ["key_w"], "-": ["key_s"]}},
                                "lin_vel_y": {{"+": ["key_a"], "-": ["key_d"]}},
                                "ang_vel_z": {{"+": ["key_j"], "-": ["key_l"]}},
                                "twist": {{"+": ["shift+key_j"], "-": ["shift+key_l"]}},
                                "pitch": {{"+": ["key_i"], "-": ["key_k"]}},
                                "roll": {{"+": ["key_o"], "-": ["key_u"]}}}},
                      "release": ["key_b"]}}}}}}}}"#,
            a = home[0],
            b = home[1],
        );
        Contract::from_str(&text, &JOINTS.map(String::from)).unwrap()
    }

    /// The pose the policy sees, from the controller's own observation, at each
    /// of the next `n` inferences with the operator holding `cmd`.
    fn poses_seen(c: &mut Controller, cmd: &Command, from_us: Micros, n: usize) -> Vec<f32> {
        let mut seen = Vec::new();
        let mut now = from_us;
        while seen.len() < n {
            now += 1_000;
            let s = at([0.1, 0.2]);
            if let Step::Infer { .. } = c.tick(&input(&s, cmd), now).unwrap() {
                seen.push(c.observation()[0]);
            }
        }
        seen
    }

    /// The contract's bands and ramp reach the policy on the robot, not only in
    /// `play`: a stick at the standing edge while walking is ramped from the rest
    /// at the declared rate and stops at the walking band. The control group is
    /// the same stick parked, which has to climb past it to the standing edge --
    /// otherwise the stop at 15 degrees could be something other than the band.
    #[test]
    fn the_policy_sees_the_command_its_contract_shapes() {
        let rate = 0.5;
        for (walk, edge) in [(0.3, 0.26), (0.0, 0.35)] {
            let cfg = FsmConfig::parse(EXAMPLE).unwrap();
            let mut contracts = HashMap::new();
            contracts.insert("locomotion".to_string(), posed_contract([0.1, 0.2], rate));
            for name in ["carry", "jump"] {
                contracts.insert(name.to_string(), contract([0.1, 0.2], 50.0, 0.0));
            }
            let mut c = Controller::new(cfg, setup(), contracts, 0).unwrap();
            let (_, now) = run_to_infer(&mut c, [0.1, 0.2], 0);
            assert_eq!(c.observation()[0], 0.0, "a policy taking over starts from level");

            let cmd = Command { lin_vel_x: walk, base_pitch: 0.35, ..Default::default() };
            let seen = poses_seen(&mut c, &cmd, now, 60);
            let step = rate / 50.0;
            assert!((seen[0] - step).abs() < 1e-6, "the first step is rate x dt: {}", seen[0]);
            assert!(seen.windows(2).all(|w| w[1] >= w[0] - 1e-6), "the ramp went backwards");
            let last = *seen.last().unwrap();
            assert!((last - edge).abs() < 1e-6, "walking {walk}: settled at {last}, not {edge}");
        }
    }

    /// A two-joint contract whose policy drives `a` only, so `b` is held.
    fn held_contract(home: [f32; 2]) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": {}, "b": {}}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": 3, "history_length": 1,
                                 "terms": [{{"name":"joint_pos","dim":2}},
                                           {{"name":"actions","dim":1}}]}},
                "action": {{"dim": 1}}}}"#,
            home[0], home[1]
        );
        Contract::from_str(&text, &JOINTS.map(String::from)).unwrap()
    }

    /// Every mode holds what its policy does not drive at its **own** home pose,
    /// and adds its actions to it. The decoder used to be built on the robot's
    /// default -- the first contract's -- so a mode trained around another pose
    /// was held at somebody else's, the moment its policy took over:
    /// `jumper.five_foot`'s carried arm, folded back to HOME. The robot default
    /// here is zero, which is what that version would have commanded.
    #[test]
    fn a_joint_the_policy_does_not_drive_is_held_at_its_modes_home() {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let mut contracts = HashMap::new();
        contracts.insert("locomotion".to_string(), held_contract([0.1, 0.9]));
        for name in ["carry", "jump"] {
            contracts.insert(name.to_string(), contract([0.1, 0.2], 50.0, 0.0));
        }
        let mut c = Controller::new(cfg, setup(), contracts, 0).unwrap();
        run_to_infer(&mut c, [0.1, 0.9], 0);
        let cmd = c.resume(&[0.0]).unwrap();
        assert!((cmd.pos[1] - 0.9).abs() < 1e-6, "b is held at {}, not this mode's 0.9", cmd.pos[1]);
        assert!((cmd.pos[0] - 0.1).abs() < 1e-6, "a zero action sits at {}, not 0.1", cmd.pos[0]);
    }

    /// `EXAMPLE` with its `locomotion` state asking `task` for a hook.
    fn with_hook(task: &str, hook: &str) -> FsmConfig {
        let mut cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let loco = cfg.states.iter_mut().find(|s| s.name == "locomotion").unwrap();
        loco.task = task.to_string();
        loco.hook = Some(toml::from_str(hook).unwrap());
        cfg
    }

    fn hooked(task: &str, hook: &str) -> Result<Controller, Error> {
        let mut contracts = HashMap::new();
        for name in ["locomotion", "carry", "jump"] {
            contracts.insert(name.to_string(), contract([0.1, 0.2], 50.0, 0.0));
        }
        Controller::new(with_hook(task, hook), setup(), contracts, 0)
    }

    /// The next inference with the robot held at `q`.
    fn next_infer(c: &mut Controller, q: [f32; 2], from_us: Micros) -> Micros {
        let cmd = Command::default();
        let mut now = from_us;
        for _ in 0..1000 {
            now += 1_000;
            if let Step::Infer { .. } = c.tick(&input(&at(q), &cmd), now).unwrap() {
                return now;
            }
        }
        panic!("no second inference");
    }

    /// Each point is called, in its place, with what the frame table in
    /// `hook.rs` says. `test.visible` makes every effect a number: home `+1`,
    /// the joints the policy sees `+10`, the action doubled, the targets `-1`.
    #[test]
    fn a_hook_stands_between_the_robot_and_the_policy() {
        let before = crate::hook::testing::RESETS.with(|r| r.get());
        let mut c = hooked("test.visible", "offset = 1.0\nshift = -1.0").unwrap();
        // The robot waits at the hook's home, not the contract's. Its joints never
        // move here, so at the contract's home the warm start and the ramp would
        // never end and no inference would come.
        let (_, now) = run_to_infer(&mut c, [1.1, 1.2], 0);
        assert_eq!(crate::hook::testing::RESETS.with(|r| r.get()) - before, 1);

        let obs = c.observation().to_vec();
        for (i, want) in [(0, 11.0), (1, 11.0)] {
            assert!((obs[i] - want).abs() < 1e-5, "joint_pos[{i}] = {}, not the hooked state", obs[i]);
        }

        let cmd = c.resume(&[0.1, -0.2]).unwrap().clone();
        // Doubled, around the contract's home, then shifted: 0.1 + 0.25 * 0.2 - 1.
        assert!((cmd.pos[0] + 0.85).abs() < 1e-5, "{:?}", cmd.pos);
        assert!((cmd.pos[1] + 0.9).abs() < 1e-5, "{:?}", cmd.pos);

        next_infer(&mut c, [1.1, 1.2], now);
        let obs = c.observation();
        assert_eq!(&obs[4..6], &[0.1, -0.2], "the policy is shown its own action, not the doubled one");
    }

    /// The ramp into a hooked mode drives the robot to the hook's home. Held one
    /// radian away from it, the robot is let go by the warm start's timeout and
    /// then ramped by the mode, and where that ramp's setpoint comes to rest is
    /// the pose the mode holds -- for a mirror, the claw on the right. Written
    /// after a first version that started the robot at the hook's home: the mode
    /// then found its pose already reached, never ramped, and passed against a
    /// ramp aimed at the contract's home instead.
    #[test]
    fn the_ramp_into_a_hooked_mode_ends_at_the_hooks_home() {
        let mut c = hooked("test.visible", "offset = 1.0\nshift = 0.0").unwrap();
        let cmd = Command::default();
        let mut now = 0;
        for _ in 0..4000 {
            now += 1_000;
            c.tick(&input(&at([0.1, 0.2]), &cmd), now).unwrap();
        }
        let pos = &c.command().pos;
        assert!(
            (pos[0] - 1.1).abs() < 1e-5 && (pos[1] - 1.2).abs() < 1e-5,
            "the ramp came to rest at {pos:?}, not the hook's home [1.1, 1.2]"
        );
    }

    /// The clamp is the last thing between a policy and the robot, so it is after
    /// the hook as well as before it.
    #[test]
    fn no_hook_gets_past_the_robots_stops() {
        let mut c = hooked("test.visible", "offset = 0.0\nshift = -100.0").unwrap();
        run_to_infer(&mut c, [0.1, 0.2], 0);
        let cmd = c.resume(&[0.0, 0.0]).unwrap().clone();
        let (lo, _) = c.limits();
        assert_eq!(cmd.pos, lo.to_vec(), "a hook pushed the targets past the stops");
    }

    /// A hook that cannot be built is a mode that would run without it -- for a
    /// mirror, a claw on the side nobody chose -- so the controller does not build.
    #[test]
    fn a_hook_that_cannot_be_built_is_refused() {
        match hooked("no.such_task", "") {
            Err(Error::Hook(m)) => assert!(m.contains("'no.such_task', which compiled none"), "{m}"),
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("built a mode whose hook does not exist"),
        }
        match hooked("test.visible", "offset = 1.0") {
            Err(Error::Hook(m)) => assert!(m.contains("`shift` is a float"), "{m}"),
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("built a hook from a configuration it refused"),
        }
        assert!(hooked("test.visible", "offset = 1.0\nshift = 0.0").is_ok(), "the control group");
    }

    /// `contract`'s two joints, carrying a controls block whose task keeps the
    /// controls `kept`, by name -- each an amount, on a trigger of the pad and a
    /// key of the keyboard in turn -- the way `jumper.five_foot`'s keeps its
    /// claw's.
    fn keeping(kept: &[&str]) -> Contract {
        let task: Vec<String> =
            kept.iter().map(|c| format!(r#""{c}": {{"kind": "amount", "does": "closes a claw"}}"#)).collect();
        let pad: Vec<String> = kept.iter().zip(["LT", "RT"]).map(|(c, t)| format!(r#""{c}": "{t}""#)).collect();
        let keys: Vec<String> =
            kept.iter().zip(["key_q", "key_o"]).map(|(c, k)| format!(r#""{c}": ["{k}"]"#)).collect();
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.1, "b": 0.2}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": 6, "history_length": 1,
                                 "terms": [{{"name":"joint_pos","dim":2}},
                                           {{"name":"joint_vel","dim":2}},
                                           {{"name":"actions","dim":2}}]}},
                "action": {{"dim": 2}},
                "command_ranges": {{"twist": {{"lin_vel_x": [-0.8, 0.8], "lin_vel_y": [-0.8, 0.8],
                                              "ang_vel_z": [-4.0, 4.0]}}}},
                "controller": {{
                  "schema": "operator_controller/2",
                  "command": [{{"term": "twist", "feeds": "velocity_commands",
                               "axes": [{{"name": "lin_vel_x"}}, {{"name": "lin_vel_y"}},
                                        {{"name": "ang_vel_z"}}]}}],
                  "task": {{{task}}},
                  "devices": {{
                    "gamepad": {{"scheme": "absolute", "deadzone": "device_reported_rescaled",
                                "release_button": "B",
                                "axes": {{"lin_vel_x": {{"source": "Ly", "sign": -1}},
                                         "lin_vel_y": {{"source": "Lx", "sign": -1}},
                                         "ang_vel_z": {{"source": "Rx", "sign": -1}}}},
                                "task": {{{pad}}}}},
                    "keyboard": {{"scheme": "keys", "full_after_s": 2.0,
                                 "axes": {{"lin_vel_x": {{"+": ["key_w"], "-": ["key_s"]}},
                                          "lin_vel_y": {{"+": ["key_a"], "-": ["key_d"]}},
                                          "ang_vel_z": {{"+": ["key_j"], "-": ["key_l"]}}}},
                                 "release": ["key_b"],
                                 "task": {{{keys}}}}}}}}}}}"#,
            task = task.join(", "),
            pad = pad.join(", "),
            keys = keys.join(", "),
        );
        Contract::from_str(&text, &JOINTS.map(String::from)).unwrap()
    }

    /// `hooked`, with the locomotion mode's task keeping `kept`.
    fn hooked_keeping(kept: &[&str], hook: Option<&str>) -> Result<Controller, Error> {
        let mut cfg = FsmConfig::parse(EXAMPLE).unwrap();
        if let Some(hook) = hook {
            cfg = with_hook("test.visible", hook);
        }
        let mut contracts = HashMap::new();
        contracts.insert("locomotion".to_string(), keeping(kept));
        for name in ["carry", "jump"] {
            contracts.insert(name.to_string(), contract([0.1, 0.2], 50.0, 0.0));
        }
        Controller::new(cfg, setup(), contracts, 0)
    }

    /// The hook is handed the controls its task keeps, as the host read them
    /// this tick. The control group is the tick before: the same controller
    /// handed nothing records nothing, so the recorded squeeze is the input's.
    #[test]
    fn a_hook_is_handed_the_controls_its_task_keeps() {
        let hook = "offset = 1.0\nshift = 0.0\nreads = [\"claw_left\"]";
        let mut c = hooked_keeping(&["claw_left"], Some(hook)).unwrap();
        let (_, now) = run_to_infer(&mut c, [1.1, 1.2], 0);
        let seen = crate::hook::testing::CONTROLS.with(|c| c.borrow().clone());
        assert_eq!(seen.get("claw_left"), 0.0, "the control group: nothing squeezed, nothing seen");

        c.resume(&[0.0, 0.0]).unwrap();
        let squeezed = crate::types::TaskControls::new(true, [("claw_left".to_string(), 0.6)]);
        let cmd = Command::default();
        let mut now = now;
        loop {
            now += 1_000;
            let robot = at([1.1, 1.2]);
            let input = TickInput { controls: &squeezed, ..input(&robot, &cmd) };
            if let Step::Infer { .. } = c.tick(&input, now).unwrap() {
                break;
            }
        }
        let seen = crate::hook::testing::CONTROLS.with(|c| c.borrow().clone());
        assert_eq!(seen.get("claw_left"), 0.6, "the hook was not handed the squeeze");
    }

    /// With operators, a hook is handed what its own mode's operator reads of
    /// the controls its task keeps -- not the host's `controls`, which only a
    /// mode without an operator reads. The control group is the same tick with
    /// no operators, which hands over the host's.
    #[test]
    fn a_hook_is_handed_its_own_operators_reading() {
        let hook = "offset = 1.0\nshift = 0.0\nreads = [\"claw_left\"]";
        let mut c = hooked_keeping(&["claw_left"], Some(hook)).unwrap();
        let (_, mut now) = run_to_infer(&mut c, [1.1, 1.2], 0);
        let lt = crate::vocabulary::pad_axes().iter().position(|a| *a == "LT").unwrap();
        let mut ops = Operators::of([("locomotion", &keeping(&["claw_left"]))].into_iter()).unwrap();
        let mut squeeze = PAD_IDLE;
        squeeze.connected = true;
        squeeze.axes[lt] = 0.9;
        ops.pad(&squeeze);
        let host = crate::types::TaskControls::new(true, [("claw_left".to_string(), 0.2)]);
        let cmd = Command::default();
        for (operators, want) in [(Some(&ops), 0.9), (None, 0.2)] {
            loop {
                now += 1_000;
                let robot = at([1.1, 1.2]);
                let input = TickInput { operators, controls: &host, ..input(&robot, &cmd) };
                if let Step::Infer { .. } = c.tick(&input, now).unwrap() {
                    break;
                }
            }
            let seen = crate::hook::testing::CONTROLS.with(|c| c.borrow().clone());
            assert_eq!(seen.get("claw_left"), want, "operators: {}", operators.is_some());
        }
    }

    /// A two-joint contract that observes the velocity command, steered by a
    /// block whose full stick forward is `top` m/s: the edge of that policy's
    /// trained range.
    fn steered(top: f32) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
                "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.1, "b": 0.2}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": 5, "history_length": 1,
                                 "terms": [{{"name":"commands","dim":3}},
                                           {{"name":"actions","dim":2}}]}},
                "action": {{"dim": 2}},
                "command_ranges": {{"twist": {{"lin_vel_x": [-{top}, {top}],
                                              "lin_vel_y": [-{top}, {top}],
                                              "ang_vel_z": [-2.0, 2.0]}}}},
                "controller": {{
                  "schema": "operator_controller/2",
                  "command": [{{"term": "twist", "feeds": "velocity_commands",
                               "axes": [{{"name": "lin_vel_x"}}, {{"name": "lin_vel_y"}},
                                        {{"name": "ang_vel_z"}}]}}],
                  "devices": {{
                    "gamepad": {{"scheme": "absolute", "deadzone": "device_reported_rescaled",
                                "release_button": "B",
                                "axes": {{"lin_vel_x": {{"source": "Ly", "sign": -1}},
                                         "lin_vel_y": {{"source": "Lx", "sign": -1}},
                                         "ang_vel_z": {{"source": "Rx", "sign": -1}}}}}},
                    "keyboard": {{"scheme": "keys", "full_after_s": 2.0,
                                 "axes": {{"lin_vel_x": {{"+": ["key_w"], "-": ["key_s"]}},
                                          "lin_vel_y": {{"+": ["key_a"], "-": ["key_d"]}},
                                          "ang_vel_z": {{"+": ["key_j"], "-": ["key_l"]}}}},
                                 "release": ["key_b"]}}}}}}}}"#
        );
        Contract::from_str(&text, &JOINTS.map(String::from)).unwrap()
    }

    /// Each mode is driven through its own operator, the one for the mode the
    /// cascade picked: one pad frame, full stick forward, is 0.8 m/s to the
    /// walking policy and 0.5 m/s once `carry` is latched, each the edge of the
    /// range that policy trained on. A bundle-wide operator handed both the
    /// same number -- the claw policy, on the robot, 1.6 times its speed. The
    /// control group is the same two modes with no operators, where both see
    /// the host's own command, so the two numbers are the operators' doing.
    #[test]
    fn each_mode_is_driven_through_its_own_operator() {
        let contracts = || -> HashMap<String, Contract> {
            [("locomotion", steered(0.8)), ("carry", steered(0.5)),
             ("jump", contract([0.1, 0.2], 50.0, 0.0))]
                .into_iter()
                .map(|(n, c)| (n.to_string(), c))
                .collect()
        };
        let built = contracts();
        let mut ops = Operators::of(built.iter().map(|(n, c)| (n.as_str(), c))).unwrap();
        let mut forward = PAD_IDLE;
        forward.connected = true;
        forward.axes[crate::vocabulary::pad_axes().iter().position(|a| *a == "Ly").unwrap()] = -1.0;
        ops.pad(&forward);

        let carry: std::collections::BTreeSet<String> = ["carry".to_string()].into();
        let seen = |operators: Option<&Operators>| -> Vec<f32> {
            let mut c = Controller::new(FsmConfig::parse(EXAMPLE).unwrap(), setup(), contracts(), 0)
                .unwrap();
            let host = Command { lin_vel_x: 0.3, ..Default::default() };
            let mut now = 0;
            let mut out = Vec::new();
            for (buttons, want) in [(&*NONE, "locomotion"), (&carry, "carry")] {
                loop {
                    now += 1_000;
                    let s = at([0.1, 0.2]);
                    let input = TickInput { operators, buttons, ..input(&s, &host) };
                    if let Step::Infer { mode } = c.tick(&input, now).unwrap() {
                        if mode == want {
                            out.push(c.observation()[0]);
                            break;
                        }
                    }
                    assert!(now < 20_000_000, "never ran {want}");
                }
            }
            out
        };
        let with = seen(Some(&ops));
        assert!((with[0] - 0.8).abs() < 1e-6 && (with[1] - 0.5).abs() < 1e-6, "{with:?}");
        let without = seen(None);
        assert!(without.iter().all(|v| (v - 0.3).abs() < 1e-6), "control group: {without:?}");
    }

    /// Both halves of "one story": a hook reading a control the file does not
    /// keep, and a file keeping controls for a mode with no hook to answer
    /// them, are each a squeeze that does nothing on the robot.
    #[test]
    fn what_a_hook_reads_is_what_its_task_keeps() {
        assert!(
            hooked_keeping(&["claw_left"], Some("offset = 1.0\nshift = 0.0\nreads = [\"claw_left\"]")).is_ok(),
            "the control group builds"
        );
        match hooked_keeping(&["claw_left"], Some("offset = 1.0\nshift = 0.0\nreads = [\"LT\"]")) {
            Err(Error::Hook(m)) => assert!(m.contains("reads LT") && m.contains("\"claw_left\""), "{m}"),
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("built a hook reading a control its task does not declare"),
        }
        match hooked_keeping(&["claw_left", "claw_right"], None) {
            Err(Error::Hook(m)) => assert!(m.contains("asks for no hook"), "{m}"),
            Err(e) => panic!("refused for another reason: {e}"),
            Ok(_) => panic!("built a mode whose task's controls nothing answers"),
        }
    }

    /// A contract missing for a state that names a model is a load error, not a
    /// robot that holds forever while the panel reads "running".
    #[test]
    fn a_mode_with_no_contract_refuses_to_build() {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let mut contracts = HashMap::new();
        contracts.insert("locomotion".to_string(), contract([0.1, 0.2], 50.0, 0.0));
        let err = match Controller::new(cfg, setup(), contracts, 0) {
            Err(e) => e,
            Ok(_) => panic!("a state naming a model with no contract must not build"),
        };
        assert!(err.to_string().contains("no contract was installed"));
    }

    /// The hand-off waits on the measured pose, not on the clock. With the robot
    /// held away from the model's default pose, `mode_switch_ramp_s` elapses many
    /// times over and the policy still never starts.
    #[test]
    fn the_policy_does_not_start_until_the_pose_is_reached() {
        let home = [0.5, -0.5];
        let mut c = controller(home, 50.0, 0.0);
        let cmd = Command::default();

        // Far from home, for ten seconds -- ten times mode_switch_ramp_s.
        let away = at([0.0, 0.0]);
        let mut now = 0;
        let mut asked = false;
        for _ in 0..10_000 {
            if let Step::Infer { .. } = c.tick(&input(&away, &cmd), now).unwrap() {
                asked = true;
                break;
            }
            now += 1_000;
        }
        assert!(!asked, "a ramp that gives up on time would start the policy from the wrong pose");
        assert!(!c.is_running_policy());

        // Reaching the pose is what releases it.
        let (step, _) = run_to_infer(&mut c, home, now);
        assert_eq!(step, Step::Infer { mode: "locomotion".into() });
        assert!(c.is_running_policy());
    }

    /// The observation is built once per policy period, not once per tick. At
    /// 50 Hz on a 1 kHz loop that is one tick in twenty, and getting it wrong
    /// aliases the policy while everything still looks like it is running.
    #[test]
    fn inference_runs_at_the_contract_s_rate_not_the_host_s() {
        let home = [0.0, 0.0];
        let mut c = controller(home, 50.0, 0.0);
        let cmd = Command::default();
        let s = at(home);

        let mut asks = 0;
        for tick in 0..1_000u64 {
            let step = c.tick(&input(&s, &cmd), tick * 1_000).unwrap();
            if let Step::Infer { .. } = step {
                asks += 1;
                c.resume(&[0.0, 0.0]).unwrap();
            }
        }
        // One second of 1 kHz ticks at 50 Hz: 50 inferences, give or take the
        // first.
        assert!((49..=51).contains(&asks), "asked {asks} times in one second at 50 Hz");

        // The same loop with a 200 Hz contract asks four times as often, which is
        // the check that the number comes from the contract rather than a constant.
        let mut fast = controller(home, 200.0, 0.0);
        let mut fast_asks = 0;
        for tick in 0..1_000u64 {
            if let Step::Infer { .. } = fast.tick(&input(&s, &cmd), tick * 1_000).unwrap() {
                fast_asks += 1;
                fast.resume(&[0.0, 0.0]).unwrap();
            }
        }
        assert!((199..=201).contains(&fast_asks), "asked {fast_asks} times at 200 Hz");
    }

    /// `resume` out of turn is refused. An action decoded against an observation
    /// nobody asked for would be applied to whatever mode is active now, which is
    /// not the one that produced it.
    #[test]
    fn an_action_nobody_asked_for_is_refused() {
        let mut c = controller([0.0, 0.0], 50.0, 0.0);
        assert!(matches!(c.resume(&[0.0, 0.0]), Err(Error::NotPending)));

        let (_, _) = run_to_infer(&mut c, [0.0, 0.0], 0);
        assert!(matches!(
            c.resume(&[0.0]),
            Err(Error::ActionWidth { got: 1, want: 2 })
        ));
    }

    /// A pending request does not survive a tick. The world has moved on, and an
    /// action decoded against last tick's observation is worse than none.
    #[test]
    fn a_request_the_host_ignored_expires() {
        let mut c = controller([0.0, 0.0], 50.0, 0.0);
        let (_, now) = run_to_infer(&mut c, [0.0, 0.0], 0);
        let cmd = Command::default();
        let s = at([0.0, 0.0]);
        c.tick(&input(&s, &cmd), now + 1_000).unwrap();
        assert!(matches!(c.resume(&[0.0, 0.0]), Err(Error::NotPending)));
    }

    /// Stale feedback drops to the safe state from inside a running policy, and
    /// the safe state damps rather than driving anywhere.
    #[test]
    fn stale_feedback_drops_out_of_a_running_policy() {
        let mut c = controller([0.0, 0.0], 50.0, 0.0);
        let (_, now) = run_to_infer(&mut c, [0.0, 0.0], 0);
        c.resume(&[0.3, -0.3]).unwrap();
        assert_eq!(c.state().name, "locomotion");

        let s = at([0.2, -0.2]);
        let cmd = Command::default();
        let step = c
            .tick(&TickInput { state_fresh: false, ..input(&s, &cmd) }, now + 1_000)
            .unwrap();
        assert_eq!(step, Step::Hold);
        assert_eq!(c.state().name, "safe");
        assert!(!c.is_running_policy());
        // kp 0, kd 0.01: damping around where the robot is, not a drive to a pose.
        assert_eq!(c.command().kp[0], 0.0);
        assert!(c.command().kd[0] > 0.0);
        assert!((c.command().pos[0] - 0.2).abs() < 1e-6);
    }

    /// The output filter is rate-invariant and starts from the current pose.
    ///
    /// Every regime but the policy publishes with alpha = 1, so `pos_des` tracks
    /// what was published and entering the policy regime cannot step. Checked by
    /// requiring the first filtered output to be nearer the old pose than the new
    /// target, and to arrive there without overshoot.
    #[test]
    fn the_output_filter_starts_where_the_robot_is() {
        let home = [0.0, 0.0];
        let mut c = controller(home, 50.0, 20.0);
        let cmd = Command::default();
        let s = at(home);

        let (_, now) = run_to_infer(&mut c, home, 0);
        let before = c.command().pos[0];
        c.resume(&[1.0, 0.0]).unwrap();
        let first = c.command().pos[0];
        assert!(
            (first - before).abs() < 0.25 * 0.25,
            "the filter moved {} of the way on its first sample", first
        );

        let mut last = first;
        for tick in 1..400u64 {
            c.tick(&input(&s, &cmd), now + tick * 1_000).unwrap();
            let p = c.command().pos[0];
            assert!(p >= last - 1e-6, "monotone toward the target");
            last = p;
        }
        // action_scale 0.25, use_default_offset around home 0: the target is 0.25.
        assert!((last - 0.25).abs() < 0.01, "converged to {last}, not the decoded target");
    }
}

/// What the operator is asking for, this tick.
///
/// > From the C++, which learned it the useful way: a latched mode that
/// > outlives the operator keeps the FSM in a state that may suspend the tilt
/// > fallback, with nobody left to release it.
///
/// A **set**, not one slot -- unless the config says `exclusive`. Bindings are
/// independent -- `LB` held while `A` is pressed is two things asked for at
/// once -- so which one wins is the cascade's order rather than whichever was
/// pressed last. That is where every other priority decision in this crate
/// lives. With `exclusive` (the jumper bundle, since 2026-09-29) a latch coming
/// on releases every other, and the last switch pressed is the mode.
///
/// Names, not controls, because which control means what is the robot's
/// decision and the pad's names are the hardware's. `[[fsm.button]]` maps
/// between them, one device per binding: the pad's A and the keyboard's Space
/// are two bindings named `jump`, and a name is one latch.
#[derive(Debug, Default)]
pub struct Buttons {
    active: std::collections::BTreeSet<String>,
    /// Last frame's state, per binding, so a held control is not read as a
    /// fresh press every tick.
    was_down: Vec<bool>,
    /// Which latching bindings are on, by name.
    latched: std::collections::BTreeSet<String>,
    /// Modifiers that have already modified something during this press -- a
    /// pad button (`menu`) or a keyboard modifier (`ctrl`). A consumed modifier
    /// stays quiet until released, so a single `menu` bound to "leave dance
    /// mode" does not fire the instant you reach for a direction.
    consumed: std::collections::BTreeSet<String>,
    /// Presses counted so far on each control a click gesture reads, and when
    /// the last one came. Keyed by the control rather than by the binding:
    /// `single` and `double` on `A` are one count, which one fires is what the
    /// count comes to.
    clicks: std::collections::BTreeMap<ClickControl, Clicks>,
    /// One latch at a time: `[fsm] exclusive`.
    exclusive: bool,
}

/// A control as clicks are counted on it: pad button, key, modifier.
type ClickControl = (Option<String>, Option<String>, Option<String>);

#[derive(Debug, Default, Clone, Copy)]
struct Clicks {
    count: u32,
    last_us: u64,
}

impl Buttons {
    /// Latches as the config says: a set, or one at a time (`exclusive`).
    pub fn new(exclusive: bool) -> Self {
        Self { exclusive, ..Default::default() }
    }

    /// One frame. `pad` is the gamepad; `key` answers whether a named key is
    /// down, for a host driven by a keyboard; `now_us` is the host's clock;
    /// `state` is the cascade's state as this tick starts, which a binding's
    /// `from` is read against.
    ///
    /// The pad and the keys feed different bindings -- a binding is one
    /// device's -- which share a latch when they share a name.
    ///
    /// The clock is only read by the click gestures, and read as a time, never
    /// as a count of calls: a press counts when it comes within the binding's
    /// window of the one before, however many frames lie between them. So a
    /// host observing at 50 Hz and one at 1 kHz count the same presses the
    /// same way, which is what used to be the reason there were no clicks.
    pub fn observe(
        &mut self,
        bindings: &[crate::config::ButtonBinding],
        pad: &crate::types::Pad,
        key: &dyn Fn(&str) -> bool,
        now_us: u64,
        state: &str,
    ) {
        use crate::config::{is_down, Gesture};
        self.was_down.resize(bindings.len(), false);
        self.active.clear();

        // Which controls are down, before any of them is consumed.
        let down: Vec<bool> = bindings.iter().map(|b| b.down(pad, key)).collect();
        let held = |name: &str| is_down(name, pad, key);
        // A binding with no modifier, on a control some other binding takes with
        // a modifier that is held now, is the chord's: `menu` and up is a dance
        // and not also the gesture on up, Ctrl + 1 the same on the keyboard --
        // J and Shift + J, as the operator has it (since 2026-09-29).
        let shadowed = |b: &crate::config::ButtonBinding| {
            b.with.is_none()
                && bindings.iter().any(|o| {
                    o.control_name() == b.control_name() && o.with.as_deref().is_some_and(|m| held(m))
                })
        };

        let clicked = self.count_clicks(bindings, &down, &held, &shadowed, now_us);

        let mut fired_once = Vec::new();
        let mut leaves = Vec::new();
        let mut switched_on: Option<String> = None;
        for (i, b) in bindings.iter().enumerate() {
            let now = down[i];
            let edge_up = now && !self.was_down[i];
            let edge_down = !now && self.was_down[i];
            self.was_down[i] = now;

            // A modifier that is not held disables the binding entirely, and
            // one that fires here is spent for the rest of its press; a held
            // modifier another binding chords this control with disables the
            // bare one.
            let gated = b.with.as_deref().is_some_and(|m| !held(m)) || shadowed(b);
            let fires = match b.on {
                Gesture::Rise => !gated && edge_up,
                Gesture::Fall => !gated && edge_down,
                Gesture::Hold => !gated && now,
                Gesture::Toggle => !gated && edge_up,
                // Gated when counted, not when fired: a count that settles
                // after the modifier came up was still clicked under it.
                _ => clicked.contains(&i),
            };
            if !fires {
                continue;
            }
            let on = self.latched.contains(&b.name);
            // `from` gates entering: switching a latch on, and anything that
            // fires once. Switching a latch off is its own control's, from
            // wherever the cascade is -- inside the mode it entered, usually.
            let may = b.live_in(state) || (b.latches() && on);
            if !may {
                continue;
            }
            if let Some(m) = b.with.as_deref() {
                self.consumed.insert(m.to_string());
            }
            if !b.leaves.is_empty() {
                leaves.push(i);
            } else if b.latches() {
                // A click switches a mode as a `toggle` does -- click again to
                // leave.
                if on {
                    self.latched.remove(&b.name);
                } else {
                    self.latched.insert(b.name.clone());
                    switched_on = Some(b.name.clone());
                }
            } else {
                fired_once.push(b.name.clone());
            }
        }
        if let (true, Some(name)) = (self.exclusive, &switched_on) {
            self.latched.retain(|n| n == name);
        }

        // A consumed modifier's own bindings stay quiet. Done after the loop
        // because the binding that consumes it may come later in the file than
        // the one it silences, and file order is not meant to matter here --
        // only rule order is.
        let consumed = self.consumed.clone();
        let quiet = |name: &str| {
            bindings.iter().filter(|b| b.name == name).all(|b| consumed.contains(b.control_name()))
        };
        for i in leaves {
            let b = &bindings[i];
            if consumed.contains(b.control_name()) {
                continue;
            }
            for target in &b.leaves {
                self.latched.remove(target);
            }
        }
        self.active.extend(fired_once.into_iter().filter(|n| !quiet(n)));
        self.latched.retain(|name| !quiet(name));
        self.active.extend(self.latched.iter().cloned());

        // A modifier stays spent until its release has been *processed*, not
        // until it comes up. Dropping it at the top of the frame would clear
        // the flag before the `fall` it is meant to silence is evaluated, and
        // releasing a spent `menu` would leave the dance after all -- which is
        // exactly what this is for.
        self.consumed.retain(|m| held(m));
    }

    /// Which click bindings fire this frame, by index.
    ///
    /// Per control: every press not gated by its modifier adds one to the
    /// count if it came within the window of the last, and starts a new count
    /// otherwise. A count fires the binding for it the moment it reaches the
    /// longest click bound on that control -- nothing longer could follow --
    /// and otherwise once the window has run out with no further press. A
    /// count nothing is bound to fires nothing.
    fn count_clicks(
        &mut self,
        bindings: &[crate::config::ButtonBinding],
        down: &[bool],
        held: &dyn Fn(&str) -> bool,
        shadowed: &dyn Fn(&crate::config::ButtonBinding) -> bool,
        now_us: u64,
    ) -> std::collections::BTreeSet<usize> {
        let mut fired = std::collections::BTreeSet::new();
        let mut seen = std::collections::BTreeSet::new();
        for (i, b) in bindings.iter().enumerate() {
            if b.on.clicks().is_none() || !seen.insert(b.control()) {
                continue;
            }
            // Everything on this control is a click: the parse refuses a
            // control that mixes them with the single-press gestures.
            let members: Vec<(usize, u32)> = bindings
                .iter()
                .enumerate()
                .filter(|(_, o)| o.control() == b.control())
                .filter_map(|(j, o)| o.on.clicks().map(|n| (j, n)))
                .collect();
            let longest = members.iter().map(|&(_, n)| n).max().unwrap_or(0);
            let window = b.click_window_us;
            let fire = |count: u32, fired: &mut std::collections::BTreeSet<usize>| {
                fired.extend(members.iter().filter(|&&(_, n)| n == count).map(|&(j, _)| j));
            };

            let key: ClickControl = (b.pad.clone(), b.key.clone(), b.with.clone());
            let st = self.clicks.entry(key).or_default();
            // A count whose window ran out settles first, so a press arriving
            // after it starts a count of its own.
            if st.count > 0 && now_us.saturating_sub(st.last_us) > window {
                fire(st.count, &mut fired);
                st.count = 0;
            }
            let press = down[i] && !self.was_down[i];
            let gated = b.with.as_deref().is_some_and(|m| !held(m)) || shadowed(b);
            if press && !gated {
                st.count += 1;
                st.last_us = now_us;
                if st.count >= longest {
                    fire(st.count, &mut fired);
                    st.count = 0;
                }
            }
        }
        fired
    }

    /// Release one latch by name, whatever put it there.
    ///
    /// For a timed policy that has finished: the mode is over, and the rule
    /// holding the FSM in it reads this latch. Not `toggle`, which would latch
    /// a button that happened to be clear -- re-entering the mode the tick
    /// after it ended.
    pub fn release(&mut self, button: &str) {
        self.latched.remove(button);
        self.active.remove(button);
    }

    /// Latch or release one button by name, for a host with no pad frame to
    /// find an edge in. Only meaningful for a latching binding; with
    /// `exclusive`, latching one releases the rest, as a press would.
    pub fn toggle(&mut self, button: &str) {
        if !self.latched.remove(button) {
            if self.exclusive {
                for other in std::mem::take(&mut self.latched) {
                    self.active.remove(&other);
                }
            }
            self.latched.insert(button.to_string());
            self.active.insert(button.to_string());
        } else {
            self.active.remove(button);
        }
    }

    /// The operator is gone. Drop everything, and re-arm every edge so a
    /// control still down when they return is not read as a fresh press.
    pub fn forget(&mut self) {
        self.active.clear();
        self.latched.clear();
        self.consumed.clear();
        self.clicks.clear();
        self.was_down.iter_mut().for_each(|d| *d = false);
    }

    pub fn active(&self) -> &std::collections::BTreeSet<String> {
        &self.active
    }
}
