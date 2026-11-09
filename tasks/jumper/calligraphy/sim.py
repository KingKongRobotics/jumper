"""The simulation the tools drive: jumper.five_foot's replay environment with the
brush, its shipped policy, and the two commands taken over from the operator.

Shared by `tools/write.py` (one robot, writing) and `tools/stability.py` (many,
holding the arm out), so the two measure the same robot under the same policy.
Heavy imports happen inside `build`, after the backend is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO / "tasks/jumper/five_foot/out/example/model_86600.pt"
#: This task's replay config: five_foot's with the brush, and no writing command
#: (here the tools drive the arm). Its policy is five_foot's network, so the shipped
#: five_foot checkpoint and a fine-tuned calligraphy one both load.
TASK = "jumper.calligraphy"


class Operator:
    """Stands in for the pad on one command term: returns `values` every step.

    The replay config hands `twist` (vx, vy, wz) and `body_pose` (pitch, roll,
    twist) to an operator; with no pad they fall back to random sampling. Putting
    this in its place makes them exactly what the caller sets, for every robot.
    """

    def __init__(self, n: int = 3):
        self.values = [0.0] * n

    def command(self, term, stamp=None):
        return list(self.values)

    def task_control(self, name, stamp=None):
        return None


@dataclass
class Sim:
    env: object
    wrapped: object
    policy: object
    robot: object
    joint_ids: list[int]
    steer: Operator
    pose: Operator
    dt: float
    #: The policy that walks; `policy` unless `build` was given a `walk_checkpoint`.
    walk_policy: object = None


def build(num_envs: int = 1, checkpoint: Path = CHECKPOINT,
          walk_checkpoint: Path | None = None, scene: str | None = None) -> Sim:
    """With `walk_checkpoint`, a second policy for walking: both are five_foot's
    network, a stateless MLP that carries its own observation normaliser, so the
    caller may switch between them from one step to the next.

    `scene`: a look-only scene from `scenes/` (`daylight`, `beach`) for the sky,
    the light and the ground the renders start from; its physics is the default
    scene's."""
    import warnings

    warnings.filterwarnings("ignore")
    from dataclasses import asdict

    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=num_envs))

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.five_foot.claw import ARM_JOINTS

    from . import arm as armmod

    cfg = tasks.load_env_cfg(TASK, play=True)
    if scene is not None:
        import scenes

        scenes.apply(cfg, scene)
    cfg.scene.num_envs = num_envs
    cfg.commands["body_pose"].rel_neutral_envs = 1.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    agent = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner_cls = tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent), device="cpu")
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
    walk_policy = None
    if walk_checkpoint is not None:
        walker = runner_cls(wrapped, asdict(agent), device="cpu")
        walker.load(str(walk_checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cpu")
        walk_policy = walker.get_inference_policy(device="cpu")
    steer, pose = Operator(), Operator()
    env.command_manager.get_term("twist")._operator = steer
    env.command_manager.get_term("body_pose")._operator = pose
    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(j) for j in (*ARM_JOINTS, armmod.FINGER_JOINT)]
    policy = runner.get_inference_policy(device="cpu")
    return Sim(env, wrapped, policy, robot, joint_ids, steer, pose, env.step_dt,
               walk_policy or policy)


def standing_qpos(mjm) -> np.ndarray:
    """Standing at the origin, level, legs at HOME, the brush in the shut claw --
    fixed, not read from the simulation, whose reset randomises the joints."""
    from tasks.jumper.common.constants import HOME, STAND_Z

    from . import arm as armmod
    from . import brush

    q = mjm.qpos0.astype(np.float64).copy()
    q[0:3] = (0.0, 0.0, STAND_Z)
    q[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in {**HOME, armmod.FINGER_JOINT: brush.FINGER_HOLD}.items():
        q[mjm.jnt_qposadr[mjm.joint(armmod.PREFIX + name).id]] = value
    return q
