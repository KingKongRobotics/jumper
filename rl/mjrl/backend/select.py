"""Choose which `Simulation` implementation the environment is built with.

This is the **using side** of the seam. The seam itself is in the vendored
`rl/mjlab/sim/__init__.py` (`set_simulation_cls` / `get_simulation_cls`) and in
the one construction in `rl/mjlab/envs/manager_based_rl_env.py`; both carry
`[mjrl]` markers.

Timing matters: **this must run before the env is built.** The construction in
`ManagerBasedRlEnv.__init__` consults the registry once, and changing it after
the env exists has no effect at all.
"""

from __future__ import annotations

from .resolve import Resolution

__all__ = ["use_backend"]


def use_backend(resolution: Resolution) -> None:
    """Register `resolution.backend` with mjlab for the env built next.

    `warp` restores mjlab's own `Simulation` (i.e. does nothing); `native`
    installs `NativeSimulation` and hands over the resolved thread count with it.

    Raises:
        ValueError: unknown backend name.
    """
    from mjlab.sim import set_simulation_cls
    from mjlab.utils.sim_device import set_sim_device

    if resolution.backend == "warp":
        set_simulation_cls(None)
        # Where physics runs. Upstream takes the torch device's name; on Apple
        # Silicon the resolution says metal:0 for torch's "cpu", and this is
        # the one place the vendored side learns that -- before the env is
        # built, like the class above, because the construction reads it once.
        set_sim_device(None if resolution.sim_device == resolution.device
                       else resolution.sim_device)
    elif resolution.backend == "native":
        from .native_sim import (
            NativeSimulation,
            set_default_nthread,
            set_strip_visuals,
        )

        set_simulation_cls(NativeSimulation)
        set_sim_device(None)  # native never asks Warp for a device
        # The thread count is handed over here too. `NativeSimulation` is
        # constructed by the vendored manager_based_rl_env through
        # `get_simulation_cls()(...)`, whose signature is fixed and leaves no room
        # for extra arguments -- so it follows the same pattern as
        # set_simulation_cls.
        #
        # A configuration that lies is worse than no configuration: MJRL_CPU_THREADS
        # existed in .env, _cli.py parsed it, resolve() computed it and the banner
        # printed "threads=16", while the backend never received it and actually ran
        # num_envs threads.
        set_default_nthread(resolution.cpu_threads)
        # Visual-mesh stripping goes the same way (None = decide by memory)
        set_strip_visuals(resolution.strip_visual)
    else:
        raise ValueError(f"unknown backend {resolution.backend!r}")
