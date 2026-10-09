#!/usr/bin/env python3
"""Search for bugs, pick one up, and drop it in the tray. Then search again.

    python tasks/jumper/five_foot/tools/bug_hunt.py --probe
    python tasks/jumper/five_foot/tools/bug_hunt.py --checkpoint <run>/model_N.pt

A replay on ``jumper.five_foot``. The policy only tracks a velocity command.
The hunt decides when a bug has been seen, walks the body there, holds the
arm out, closes the claw, and carries what it kept to the tray. It is not a
training task and not a deploy app. The soap row (``--objects``) is untouched.

Bugs are still. Their poses are read only to test whether they lie in the
onboard camera's forward cone, out to 1.2 m. The tray is a fixture, so its
pose is known the whole time. The ground is a walled room: the floor the
policy walks on stays a plane, drawn finite, with walls at its edge. Nothing
moves except the robot and whatever the claw pushes.

The room and the can are ``hunt_scene``. Bearings, the camera cone and the
keep-out around the can are ``hunt_motion``. Each state of the hunt is a
function in ``hunt_mission``; ``tick`` only dispatches and then steers.

The arm pose and the lift below are from ``--probe`` (native CPU, one
environment, no policy, 2026-10-08). Each preset endpoint leaves the mouth
between 77 and 158 mm up. A 30 mm bug on the floor has its centre at 15 mm,
and the shoulder lift carried none of them: ratio 0.00 at every endpoint.

The straight path from the stow to thumb-down is the one that dips. At 0.40
of the way, with the shoulder another 0.26 rad in the direction that lowers
the mouth, the mouth is at 18 mm (``GRASP_POSE``, -54, -51, -6, -45 degrees).
The bug was seated on the floor at the horizontal position ``Claw.seat``
gives it, the finger closed, and the shoulder lifted 0.25 rad the other way.
That carried it on 3 of 3 trials: the bug rose 19.1 to 19.7 mm against the
anvil's 21.7 to 22.4, ratio 0.88. The shoulder that raises the anvil is the
negative direction.

Grasping is friction, under ``objects.py``'s elliptic cone. A squeeze on the
training solver lets the bug creep out. Picked up means the bug rose at least
``FOLLOWED`` of the way the anvil rose. Delivered means the claw opened with
the bug's origin inside the tray's footprint. The height line is only how
long the open claw waits for it to fall: a bug in the can, stacked or still
settling, sits above that line, and the search resumes beside the can, so
the height test was retiring nothing and the same bug was the next target.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from tasks.jumper.five_foot.claw import GRIPPER_CLOSED, GRIPPER_OPEN
from tasks.jumper.five_foot.mdp.grasp import Claw
from tasks.jumper.five_foot.objects import PROP_CONE
from tasks.jumper.five_foot.tools.grasp_objects import FOLLOWED
from tasks.jumper.five_foot.tools.hunt_mission import (
    GRASP_ALPHA,
    GRASP_POSE,
    GRASP_SHOULDER,
    LIFT_RAD,
    LIFT_SIGN,
    PRESETS,
    STOW,
    Arm,
    Hunt,
    ShownArm,
    install_command,
    install_pose,
    set_command,
    tick,
)
from tasks.jumper.five_foot.tools.hunt_motion import _base, _into, _root, mouth_in_base
from tasks.jumper.five_foot.tools.hunt_scene import (
    BUG_RADIUS,
    N_BUGS,
    add_hunt_scene,
    pin_spawn,
)


def _place(entity, pos: torch.Tensor) -> None:
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=pos.device)
    entity.write_root_link_pose_to_sim(torch.cat([pos.view(1, 3), quat], dim=1))
    entity.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=pos.device))


def _zero_action(env) -> torch.Tensor:
    dim = env.action_manager.total_action_dim
    return torch.zeros(env.num_envs, dim, device=env.device)


def _hold_step(env, arm: Arm, angles: np.ndarray, finger: float) -> None:
    arm.goal = np.array(angles, dtype=np.float64)
    arm.cmd = arm.goal.copy()
    arm.finger = finger
    arm.step_targets()
    env.step(_zero_action(env))


def _anvil_sign(env, arm: Arm, claw: Claw, preset: np.ndarray) -> float:
    """+1 if increasing the shoulder angle raises the anvil, else -1.

    Leaves the arm back at ``preset``. The nudge is a teleport, so it has to
    happen before a bug is sitting in the mouth.
    """
    arm.write_state(preset, GRIPPER_OPEN)
    env.sim.forward()
    z0 = float(claw.mouth()[0, 2])
    nudged = preset.copy()
    nudged[1] = preset[1] + 0.05
    arm.write_state(nudged, GRIPPER_OPEN)
    env.sim.forward()
    sign = 1.0 if float(claw.mouth()[0, 2]) > z0 else -1.0
    arm.write_state(preset, GRIPPER_OPEN)
    env.sim.forward()
    return sign


def measure_preset(env, arm: Arm, claw: Claw, name: str) -> dict:
    """Floor bug under this preset's mouth: does the lift carry it?"""
    angles = PRESETS[name]
    env.reset()
    arm.write_state(angles, GRIPPER_OPEN)
    env.sim.forward()
    for _ in range(30):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    env.sim.forward()
    robot = env.scene["robot"]
    sign = _anvil_sign(env, arm, claw, angles)
    local = mouth_in_base(robot, claw)
    mouth = claw.mouth()[0]
    origin_z = float(env.scene.env_origins[0, 2])
    pos = mouth.detach().clone()
    pos[0] = mouth[0]
    pos[1] = mouth[1]
    pos[2] = origin_z + BUG_RADIUS
    bug = env.scene["bug_0"]
    _place(bug, pos)
    for _ in range(20):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    gap = float((bug.data.root_link_pos_w[0] - claw.mouth()[0]).norm())
    for _ in range(80):
        _hold_step(env, arm, angles, GRIPPER_CLOSED)
    z0 = float(bug.data.root_link_pos_w[0, 2])
    a0 = float(claw.mouth()[0, 2])
    for k in range(40):
        mid = angles.copy()
        mid[1] = angles[1] + sign * LIFT_RAD * (k + 1) / 40.0
        _hold_step(env, arm, mid, GRIPPER_CLOSED)
    rise_b = float(bug.data.root_link_pos_w[0, 2]) - z0
    rise_a = float(claw.mouth()[0, 2]) - a0
    ratio = rise_b / rise_a if rise_a > 1.0e-3 else 0.0
    return {
        "name": name,
        "mouth_mm": (local * 1000.0).round(1),
        "mouth_z_mm": float(mouth[2] - origin_z) * 1000.0,
        "gap_mm": gap * 1000.0,
        "rise_bug_mm": rise_b * 1000.0,
        "rise_anvil_mm": rise_a * 1000.0,
        "ratio": ratio,
        "sign": sign,
        "carried": rise_a > 0.005 and rise_b >= FOLLOWED * rise_a,
    }


def measure_grasp(env, arm: Arm, claw: Claw) -> dict:
    """Bug on the floor at ``Claw.seat``'s horizontal position, then the lift.

    This is the pose the hunt uses. The preset endpoints are measured by
    ``measure_preset``, which parks the bug under the anvil; that is where a
    floor bug is not.
    """
    angles = GRASP_POSE
    env.reset()
    for _ in range(25):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    origin_z = float(env.scene.env_origins[0, 2])
    seat = claw.seat(env, "bug_0")[0].detach().clone()
    pos = seat.clone()
    pos[2] = origin_z + BUG_RADIUS
    bug = env.scene["bug_0"]
    _place(bug, pos)
    for _ in range(12):
        _hold_step(env, arm, angles, GRIPPER_OPEN)
    mouth = claw.mouth()[0]
    local = mouth_in_base(env.scene["robot"], claw)
    for _ in range(60):
        _hold_step(env, arm, angles, GRIPPER_CLOSED)
    z0 = float(bug.data.root_link_pos_w[0, 2])
    a0 = float(claw.mouth()[0, 2])
    lifted = angles.copy()
    lifted[1] = angles[1] + LIFT_SIGN * LIFT_RAD
    for _ in range(30):
        _hold_step(env, arm, lifted, GRIPPER_CLOSED)
    rise_b = float(bug.data.root_link_pos_w[0, 2]) - z0
    rise_a = float(claw.mouth()[0, 2]) - a0
    ratio = rise_b / rise_a if rise_a > 1.0e-3 else 0.0
    return {
        "mouth_mm": (local * 1000.0).round(1),
        "mouth_z_mm": float(mouth[2] - origin_z) * 1000.0,
        "rise_bug_mm": rise_b * 1000.0,
        "rise_anvil_mm": rise_a * 1000.0,
        "ratio": ratio,
        "carried": rise_a > 0.005 and rise_b >= FOLLOWED * rise_a,
    }


def probe(env, arm: Arm) -> None:
    claw = Claw(env)
    env.claw = claw
    degrees = np.rad2deg(GRASP_POSE)
    print(f"probe  cone={PROP_CONE}  bug diameter {BUG_RADIUS * 2000:.0f} mm  "
          f"lift {LIFT_SIGN:+.0f} * {LIFT_RAD:.2f} rad")
    print(f"grasp pose  {GRASP_ALPHA:.2f} of the way stow -> thumb_down, "
          f"shoulder {GRASP_SHOULDER:+.2f} rad"
          f"  ({degrees[0]:.0f}, {degrees[1]:.0f}, {degrees[2]:.0f}, {degrees[3]:.0f}) deg\n")
    rows = [measure_preset(env, arm, claw, name) for name in PRESETS]
    for row in rows:
        flag = "carried" if row["carried"] else "left behind"
        mouth = row["mouth_mm"]
        print(f"{row['name']:<14} mouth ({mouth[0]:+.1f}, {mouth[1]:+.1f}, "
              f"{row['mouth_z_mm']:+.1f}) mm   gap {row['gap_mm']:.0f} mm   "
              f"bug {row['rise_bug_mm']:+.1f} / anvil {row['rise_anvil_mm']:+.1f} mm   "
              f"ratio {row['ratio']:.2f}  lift {row['sign']:+.0f}   {flag}")
    print()
    carried = 0
    for trial in range(3):
        row = measure_grasp(env, arm, claw)
        carried += int(row["carried"])
        flag = "carried" if row["carried"] else "left behind"
        mouth = row["mouth_mm"]
        print(f"grasp {trial}        mouth ({mouth[0]:+.1f}, {mouth[1]:+.1f}, "
              f"{row['mouth_z_mm']:+.1f}) mm   "
              f"bug {row['rise_bug_mm']:+.1f} / anvil {row['rise_anvil_mm']:+.1f} mm   "
              f"ratio {row['ratio']:.2f}   {flag}")
    print(f"\ngrasp pose carried the floor bug on {carried} of 3 trials")


def _calibrate(env, arm: Arm, claw: Claw, preset: np.ndarray):
    """Where the grasp pose seats a bug, and which way the shoulder lifts.

    Both are read after the arm has been held at ``preset`` while the robot
    stands, then the arm goes back to the stow. A teleport of the joints does
    not leave the mouth where the PD holds it.
    """
    for _ in range(25):
        _hold_step(env, arm, preset, GRIPPER_OPEN)
    robot = env.scene["robot"]
    seat = claw.seat(env, "bug_0")[0].detach().cpu().numpy()
    pos, _, quat = _base(robot)
    seat_b = _into(quat, seat - pos)
    z0 = float(claw.mouth()[0, 2])
    nudged = preset.copy()
    nudged[1] = preset[1] + 0.08
    for _ in range(12):
        _hold_step(env, arm, nudged, GRIPPER_OPEN)
    sign = 1.0 if float(claw.mouth()[0, 2]) > z0 + 0.004 else -1.0
    for _ in range(20):
        _hold_step(env, arm, STOW, GRIPPER_OPEN)
    return seat_b[:2].copy(), sign


def run_hunt(wrapped, env, policy, viewer, hunt: Hunt, arm: Arm, shown: ShownArm,
             held: dict, max_steps: int | None, speed: float) -> None:
    from mjrl.viewer.stats import Pacer

    pacer = Pacer(env.step_dt, speed)
    wrapped.reset()
    set_command(held, 0.0, 0.0, 0.0)
    hunt.seat_xy, hunt.lift_sign = _calibrate(env, arm, env.claw, hunt.preset)
    obs, _ = wrapped.reset()
    set_command(held, 0.0, 0.0, 0.0)
    twist = env.command_manager.get_term("twist")
    twist.vel_command_b[:] = 0.0
    twist.vel_command_w[:] = 0.0
    env.observation_manager._obs_buffer = None
    obs = wrapped.get_observations()
    print(f"[hunt] grasp seat offset "
          f"({hunt.seat_xy[0] * 1000:.0f}, {hunt.seat_xy[1] * 1000:.0f}) mm, "
          f"lift sign {hunt.lift_sign:+.0f}")
    crab, crab_yaw, _ = _base(env.scene["robot"])
    print(f"[hunt] crab at ({crab[0]:+.2f}, {crab[1]:+.2f}) "
          f"yaw {math.degrees(crab_yaw):.0f} deg")
    print("[hunt] bugs: " + ", ".join(
        f"{name} ({_root(env.scene[name])[0]:+.2f}, {_root(env.scene[name])[1]:+.2f})"
        for name in hunt.bugs
    ))
    step = 0
    while hunt.state != "done" or viewer is not None:
        if viewer is not None and not viewer.is_running:
            print("\n[hunt] the live viewer was closed; stopping")
            break
        if max_steps is not None and step >= max_steps:
            print(f"\n[hunt] reached --steps {max_steps} in state {hunt.state}")
            break
        if viewer is None and hunt.state == "done":
            break
        pacer.wait()
        shown.apply(obs)
        with torch.inference_mode():
            action = policy(obs)
        command, pitch = tick(env, hunt, arm, env.sim.mj_model)
        set_command(held, *command)
        held["pitch"] = pitch
        arm.step_targets()
        with torch.inference_mode():
            obs, *_ = wrapped.step(action)
        step += 1


def build(args, bug_names_out: list):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend=args.backend, device=args.device, num_envs=1)
    use_backend(res)
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True, task_args={"objects": False})
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    names, crab_pose = add_hunt_scene(cfg, args.seed, 1 if args.probe else N_BUGS)
    pin_spawn(cfg, crab_pose)
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    bug_names_out.extend(names)
    return env, res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true",
                        help="try the three arm presets on one floor bug and exit")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backend", choices=("warp", "native"), default="native")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="layout of the crab and the bugs; omit it for a new one each run",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()
    if not args.probe and args.checkpoint is None:
        parser.error("--checkpoint is required to hunt; --probe measures the arm without one")
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0] % (2**31))

    names: list[str] = []
    env, res = build(args, names)
    arm = Arm(env.scene["robot"], env.device)
    held = install_command(env)
    if not args.probe:
        install_pose(env, held)
    try:
        if args.probe:
            probe(env, arm)
            return 0
        from dataclasses import asdict

        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.rl.runner import MjlabOnPolicyRunner

        import tasks

        agent = tasks.load_agent_cfg("jumper.five_foot")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (tasks.load_runner_cls("jumper.five_foot") or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), device=res.device
        )
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)
        env.claw = Claw(env)
        shown = ShownArm(env, arm)
        hunt = Hunt(bugs=names, preset=GRASP_POSE.copy())

        from tasks.paths import REPO_ROOT

        cli_path = Path(REPO_ROOT) / "scripts" / "_cli.py"
        spec = importlib.util.spec_from_file_location("jumper_play_cli", cli_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load the play viewer from {cli_path}")
        play_cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play_cli)
        maybe_viewer = play_cli.maybe_viewer

        viewer_args = argparse.Namespace(
            headless=args.headless, viewer_env=0, viewer_fps=60.0,
            viewer_env_num=1, viewer_ui=False,
        )
        with maybe_viewer(env, viewer_args) as viewer:
            speed = args.speed
            if speed is None:
                speed = 0.0 if viewer is None else 1.0
            run_hunt(wrapped, env, policy, viewer, hunt, arm, shown, held,
                     args.steps, speed)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
