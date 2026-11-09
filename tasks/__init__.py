"""The tasks package.

The top level is the registry. Importing this package triggers every task's
registration, so the entry points under `scripts/` need only `import tasks` to see
all available tasks.

## Directory convention

Grouped by robot family, one directory per task, and **a task id is its module
path relative to tasks/**:

    tasks/jumper/                  the jumper family
      common/                      shared within the family -- **not a task**
        assets.py                    asset declarations (lightweight, no mjlab)
        constants.py                 joint names / HOME / collision / EntityCfg
        velocity_env.py              the environment skeleton
        ppo.py                       the PPO baseline
        mdp/                         shared MDP pieces (gating, phase matching, mirroring)
      tripod/                      one task, id = "jumper.tripod"
        __init__.py                  register(...): the fixed name and supported assets
        env_cfg.py                   def env_cfg(asset, play)
        rl_cfg.py                    def agent_cfg() / def runner_cls()  <- this task's hyper-parameters
        mdp/                         MDP terms **only this task uses** (its gait reward)
      flat/  tetrapod/  ripple/    the same shape

`common/` is **not a task**: it is never registered and must not appear in
`--list`. When a task needs its own MDP terms, create an `mdp/` in that task's
directory -- something only one task uses does not belong in `common/`.

## No simulation dependencies at import time

This module is imported by lightweight commands such as `--list`, so each task's
`__init__.py` may touch only `registry` and lightweight asset declarations (like
`jumper/common/assets.py`). The real configs live in `env_cfg.py` / `rl_cfg.py` and
are imported on demand by `load_env_cfg` / `load_agent_cfg`, which is when mjlab,
torch and mujoco get pulled in. `tests/test_registry.py` watches this.
"""

from .registry import (
    AssetSpec,
    TaskSpec,
    all_specs,
    get,
    list_ids,
    load_agent_cfg,
    load_cli_args,
    load_env_cfg,
    load_export_media,
    load_play_status,
    load_runner_cls,
    register,
)

# ── Import each task here to trigger registration ─────────────────────────
# Adding a task means adding one line here; nothing under scripts/ changes.
from .jumper import calligraphy  # noqa: F401,E402
from .jumper import dance  # noqa: F401,E402
from .jumper import dance_brazilian  # noqa: F401,E402
from .jumper import dance_dream_wings  # noqa: F401,E402
from .jumper import dance_maze  # noqa: F401,E402
from .jumper import dance_waist  # noqa: F401,E402
from .jumper import five_foot  # noqa: F401,E402
from .jumper import flat  # noqa: F401,E402
from .jumper import gesture_bow  # noqa: F401,E402
from .jumper import gesture_hello  # noqa: F401,E402
from .jumper import gesture_paw  # noqa: F401,E402
from .jumper import gesture_salute  # noqa: F401,E402
from .jumper import jump  # noqa: F401,E402
from .jumper import posture  # noqa: F401,E402
from .jumper import ref_free_jump  # noqa: F401,E402
from .jumper import ripple  # noqa: F401,E402
from .jumper import swing  # noqa: F401,E402
from .jumper import tetrapod  # noqa: F401,E402
from .jumper import tripod  # noqa: F401,E402

__all__ = [
    "AssetSpec",
    "TaskSpec",
    "register",
    "get",
    "list_ids",
    "all_specs",
    "load_env_cfg",
    "load_agent_cfg",
    "load_cli_args",
    "load_play_status",
    "load_runner_cls",
    "load_export_media",
]
