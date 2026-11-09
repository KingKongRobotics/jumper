"""The four locomotion tasks are each other's controls, and that has to hold.

They exist to be compared: same robot, same reward, same observations, same
curriculum shape, **one variable changed** -- the gait. A difference that is not
the gait makes every comparison between them mean something else, and it does so
silently, because each task on its own looks perfectly reasonable.

That has already happened twice. `jumper.tetrapod` declared no command ladder and
ran on a module default nobody chose for it. `jumper.ripple` and `jumper.flat` built
their ladders by clamping and appending to the shared *absolute* rungs, which
squeezed one to (0.625, 0.875, 1.0) of its own ceiling and stretched the other to
(0.312, 0.437, 0.625, 1.0) -- three different curricula wearing one name, so
"level 1" meant three different fractions of what each robot could do.

So this pins the parity rather than the values: whatever the numbers are, the four
have to agree on them unless the field is in one of the two exemptions below.

## Where a parameter belongs

    constants.py   facts about the hardware. One robot, one IMU, one set of
                   current sensors, one 2 kg chassis -- so one value, shared.
                   EFFORT_LIMIT, STAND_Z, the noise levels, PUSH_VELOCITY_RANGE.

    common/        mechanism, and no tuning. The reward functions, the curriculum
                   classes, the ladder's *shape*. A number here is a number four
                   tasks inherit without choosing.

    <task>/        every parameter. Even when all four agree -- especially then,
                   because agreement that is written down survives one of them
                   changing, and agreement that is a default does not.

Adding a fifth locomotion task means adding it to `TASKS` here, and
`test_every_jumper_task_is_accounted_for` is what makes that happen rather than
something to remember. `jumper.dance` is a jumper task and is deliberately **not** in
`TASKS`: it has no command, no gait prior and a different reward budget, so it is
not one of these tasks' controls and forcing parity on it would only mean the
comparison stops saying anything.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab", reason="building the configs needs mjlab")

import tasks

TASKS = ("jumper.flat", "jumper.tripod", "jumper.ripple", "jumper.tetrapod")

#: Jumper tasks that are **not** each other's controls, and so are not compared here.
#: Kept as an explicit list rather than a rule about the id, so that leaving a task
#: out of `TASKS` is a decision someone wrote down.
NOT_LOCOMOTION = {
    # Imitates a fixed choreography: no command, no gait prior, 22 joints instead
    # of 20, and its reward budget is the tracking terms rather than the velocity
    # skeleton's. Nothing it could agree with the four on would mean anything.
    "jumper.dance",
    # The same, each with a clip of its own, imported from rl-wbc-fsm.
    "jumper.dance_brazilian",
    "jumper.dance_dream_wings",
    "jumper.dance_maze",
    "jumper.dance_waist",
    # One-shot gestures on the same machinery, each with a clip of its own, imported
    # from rl-wbc-fsm.
    "jumper.gesture_bow",
    "jumper.gesture_hello",
    "jumper.gesture_paw",
    "jumper.gesture_salute",
    # Walks on five legs with the left-front arm carried as a claw: 16 driven joints
    # instead of 20, five feet in every per-foot term, symmetry off, and a reward
    # set of its own. Set beside the four it would be measuring a leg count, not a
    # gait.
    "jumper.five_foot",
    # jumper.five_foot fine-tuned to stand still while the arm writes: its config
    # is five_foot's (it resumes from five_foot's checkpoint, so the network has to
    # match), with the brush, the writing command and one reward added. Its control
    # is five_foot, not the four.
    "jumper.calligraphy",
    # Stands on a swing and pumps it: no command, so none of the tracking,
    # gait-gated or curriculum terms the four share survive, and the three terms
    # it does share had to be rewritten against the deck because the floor tilts.
    # It builds a prop of its own into the scene as well. There is no number it
    # could agree with the four on that would mean the same thing.
    "jumper.swing",
    # Imitates a single recorded jump: one-shot, no command, no gait prior, and
    # its reward budget is reference tracking. Same reasoning as jumper.dance.
    "jumper.jump",
    # The same jump with the recording only as a fading training-time prior:
    # one-shot, no command, no gait prior, and a reward budget of height, not of
    # velocity tracking.
    "jumper.ref_free_jump",
    # Walks the tripod gait with the body's posture commanded on top: a second
    # command term, three tracking rewards for it, two observations, `upright`
    # replaced by its command-relative version, and both ceilings raised (which
    # forces a different cadence). It is not a fifth control -- its control is
    # `jumper.tripod`, one task, which is why tripod is left untouched by it.
    # Every field it would differ on is one of those deliberate changes, so
    # listing them as exemptions would empty the parity test rather than
    # describe it.
    "jumper.posture",
}

#: Fields a task is allowed to differ on because **the gait fixes them**.
#: Matched as substrings of the field name.
GAIT_EXEMPT = (
    "_gait",           # the gait reward itself
    "gait_phase",      # its clock, in the observation
    "freq_hz",         # the cadence every gated term is handed
    "force_threshold",  # a fair share of load depends on how many legs are down
    "stance_",         # the stance terms exist only where a clock says which legs
)

#: Fields a task is allowed to differ on because **the speed ceiling fixes them**,
#: and the ceiling is a gait property: a fixed-frequency clock caps speed at stroke
#: times cadence. Everything here is derived from `command_*_ceiling` by the
#: skeleton, so these differ *as a consequence* rather than by choice.
CEILING_EXEMPT = (
    "cmd/lin", "cmd/ang",
    "cur/command/levels", "cur/command/ang_levels",
    "track_linear_velocity/std", "track_angular_velocity/std",
)


def snapshot(task: str) -> dict:
    """Every scalar a task configures, flattened to `name -> value`."""
    cfg = tasks.load_env_cfg(task)
    out: dict[str, object] = {}
    for name, term in cfg.rewards.items():
        out[f"reward/{name}"] = term.weight
        for key, value in (term.params or {}).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[f"reward/{name}/{key}"] = value
    for group, grp in cfg.observations.items():
        out[f"obs/{group}/corrupt"] = grp.enable_corruption
        for key, term in grp.terms.items():
            noise = getattr(term, "noise", None)
            out[f"obs/{group}/{key}"] = (
                getattr(noise, "n_max", None) if noise is not None else "no-noise"
            )
    twist = cfg.commands["twist"]
    out["cmd/lin"] = twist.ranges.lin_vel_x[1]
    out["cmd/ang"] = twist.ranges.ang_vel_z[1]
    for field in ("rel_standing_envs", "rel_heading_envs", "rel_forward_envs"):
        out[f"cmd/{field}"] = getattr(twist, field, None)
    for name, term in cfg.curriculum.items():
        for key, value in (getattr(term, "params", None) or {}).items():
            out[f"cur/{name}/{key}"] = str(value)
    out["sim/decimation"] = cfg.decimation
    out["sim/dt"] = cfg.sim.mujoco.timestep
    agent = tasks.load_agent_cfg(task)
    out["ppo/num_steps_per_env"] = agent.num_steps_per_env
    for field in ("gamma", "lam", "learning_rate", "entropy_coef", "clip_param",
                  "num_learning_epochs", "num_mini_batches", "desired_kl"):
        out[f"ppo/{field}"] = getattr(agent.algorithm, field, None)
    return out


def exempt(field: str) -> str | None:
    """Which exemption covers `field`, or None."""
    if any(token in field for token in GAIT_EXEMPT):
        return "gait"
    if any(field.startswith(token) or token in field for token in CEILING_EXEMPT):
        return "ceiling"
    return None


def test_the_tasks_differ_only_in_their_gait_and_its_ceiling() -> None:
    """Anything else that differs is an accident, and reads as a finding."""
    snaps = {t: snapshot(t) for t in TASKS}
    fields = sorted(set().union(*(set(s) for s in snaps.values())))

    offenders = []
    for field in fields:
        if exempt(field):
            continue
        values = {t: snaps[t].get(field, "<absent>") for t in TASKS}
        if len({str(v) for v in values.values()}) > 1:
            offenders.append(f"{field}: " + ", ".join(
                f"{t.split('.')[1]}={v}" for t, v in values.items()
            ))
    assert not offenders, (
        "these differ between tasks and are neither the gait nor its ceiling:\n  "
        + "\n  ".join(offenders)
        + "\n\nEither the difference is deliberate -- then say so by adding the "
        "field to GAIT_EXEMPT or CEILING_EXEMPT with a reason -- or it is the "
        "accident this test exists to catch."
    )


def test_the_ladder_is_one_shape_scaled_to_each_ceiling() -> None:
    """Level 1 has to mean the same fraction of capability in every task.

    The parity test above cannot see this: the ladders are ceiling-exempt, so three
    differently *shaped* ladders pass it while making every cross-task comparison of
    `Curriculum/command/level` meaningless. Which is what the code did -- ripple's
    rungs were (0.625, 0.875, 1.0) of its ceiling against tripod's (0.5, 0.7, 1.0).

    **The tasks are compared against each other, not against `LEVEL_FRACTIONS`.**
    Written the obvious way first -- recomputing the expected rungs from that
    constant -- and it passed against a deliberately broken shape, because the
    ladders are derived from the same constant and the assertion was comparing the
    derivation with itself. Changing `LEVEL_FRACTIONS` is a global decision and this
    should not object to it; a single task departing from the others is the failure.
    """
    shapes: dict[str, dict[str, tuple]] = {}
    for task in TASKS:
        cfg = tasks.load_env_cfg(task)
        params = getattr(cfg.curriculum["command"], "params", None) or {}
        shapes[task] = {}
        for axis, key in (("linear", "levels"), ("angular", "ang_levels")):
            rungs = params[key]
            ceiling = max(rungs)
            shapes[task][axis] = tuple(round(r / ceiling, 4) for r in rungs)

    for axis in ("linear", "angular"):
        distinct = {shapes[t][axis]: t for t in TASKS}
        assert len(distinct) == 1, (
            f"the {axis} ladder is not one shape across the tasks:\n  "
            + "\n  ".join(f"{t.split('.')[1]}: {shapes[t][axis]}" for t in TASKS)
            + "\n\nEach is its rungs as a fraction of its own ceiling, so these "
            "have to match whatever the ceilings are."
        )


def test_the_exemptions_are_not_vacuous() -> None:
    """The control group: both exemption lists must actually be covering something.

    An exemption list that matches nothing would make the parity test look strict
    while it silently had no exceptions to make -- and one that matched everything
    would make it pass unconditionally. Both are failures the assertions above
    cannot see.
    """
    snaps = {t: snapshot(t) for t in TASKS}
    fields = sorted(set().union(*(set(s) for s in snaps.values())))
    differing = [
        f for f in fields
        if len({str(snaps[t].get(f, "<absent>")) for t in TASKS}) > 1
    ]
    assert differing, "nothing differs at all, so the exemptions prove nothing"

    reasons = {exempt(f) for f in differing}
    assert "gait" in reasons, "no differing field is covered by GAIT_EXEMPT"
    assert "ceiling" in reasons, "no differing field is covered by CEILING_EXEMPT"
    assert len(differing) < len(fields) / 2, (
        f"{len(differing)} of {len(fields)} fields differ, which is too many for "
        f"these tasks to be each other's controls"
    )


def test_every_jumper_task_is_accounted_for() -> None:
    """`TASKS` has to keep up with the repository, and nothing else makes it.

    The docstring above says a fifth locomotion task must be added to `TASKS`. A
    sentence in a docstring is not a mechanism: the task that ignores it trains,
    exports and ships, and the only symptom is that the parity assertions quietly
    stop covering it. That is the same shape as the two failures this file was
    written for -- a task running on something nobody chose for it.

    So every registered `jumper.*` id must be either compared or explicitly excused.
    Adding a locomotion task fails here until it joins `TASKS`; adding another
    task like `jumper.dance` fails until someone writes down why it is not a
    control.
    """
    registered = {i for i in tasks.list_ids() if i.startswith("jumper.")}
    unaccounted = registered - set(TASKS) - NOT_LOCOMOTION
    assert not unaccounted, (
        f"jumper tasks in neither TASKS nor NOT_LOCOMOTION: {sorted(unaccounted)}. "
        f"Add it to TASKS if it is one of the four's controls -- same robot, same "
        f"reward skeleton, the gait as the one variable -- and to NOT_LOCOMOTION "
        f"with a reason if it is not."
    )

    # Both lists have to name tasks that exist, or an id renamed upstream leaves a
    # dead entry here and the task it used to name goes uncompared.
    stale = (set(TASKS) | NOT_LOCOMOTION) - registered
    assert not stale, f"listed but not registered: {sorted(stale)}"
