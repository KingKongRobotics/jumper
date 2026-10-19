"""jumper.sprint's environment: jumper.posture's source with two constants changed.

`COMMAND_LIN_CEILING` and `GAIT_STRIDE` are module constants of the posture
task, read at import to derive the clock's frequency and the angular ceiling,
so the clean way to a faster copy is to execute that module's source again
with the two lines replaced, under this package's name. Everything else --
the rewards, the observations, the ladder's shape, the controls -- is the
posture task's, unchanged and uncopied.

Measured (RTX 3060, warp, 4096 envs): a posture-shaped task trains at 2.8 s per
iteration, and the shipped posture policy is 74 800 iterations deep, so this
task resumes from it rather than starting over:

    python scripts/train.py --task jumper.sprint --headless --max-iterations 3000 \\
        --checkpoint tasks/jumper/posture/out/example/model_74800.pt
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

#: The two decisions this task makes for itself (see the package docstring).
COMMAND_LIN_CEILING = 1.2
GAIT_STRIDE = 0.08

_POSTURE = Path(__file__).resolve().parents[1] / "posture" / "env_cfg.py"
_REPLACE = {
    "COMMAND_LIN_CEILING = 0.8\n": f"COMMAND_LIN_CEILING = {COMMAND_LIN_CEILING}\n",
    "GAIT_STRIDE = 0.072\n": f"GAIT_STRIDE = {GAIT_STRIDE}\n",
}


def _load():
    source = _POSTURE.read_text()
    for old, new in _REPLACE.items():
        if source.count(old) != 1:
            raise RuntimeError(f"jumper.sprint: expected exactly one `{old.strip()}` in {_POSTURE}; "
                               "the posture task changed and this task's replacement must follow")
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location("tasks.jumper.posture._sprint_source", str(_POSTURE))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "tasks.jumper.posture"      # its relative imports (`.mdp`, `..common`) resolve as posture's
    exec(compile(source, str(_POSTURE), "exec"), module.__dict__)  # noqa: S102 -- posture's own source, re-run with this task's ceiling
    return module


_source = _load()
env_cfg = _source.env_cfg
GAIT_FREQ_HZ = _source.GAIT_FREQ_HZ
COMMAND_ANG_CEILING = _source.COMMAND_ANG_CEILING
