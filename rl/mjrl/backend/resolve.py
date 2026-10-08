"""Backend and device resolution.

Precedence: **command line > environment variables > task cfg defaults**.

The governing principle is **no silent fallback**: asking explicitly for `warp` on
a machine without CUDA must fail, not quietly switch to CPU and let someone
believe they are training on a GPU. That is the easiest trap for a framework like
this to set.

Of the four combinations three are training paths:

    warp   · cuda   primary training
    warp   · cpu    on Apple Silicon with the `metal` extra: the torch device
      (sim metal:0)   stays "cpu" (torch has no Metal device, and unified memory
                    lets CPU tensors alias the Warp arrays), the simulation runs
                    on the Apple GPU. Measured on an M3 Max, jumper.tripod:
                    5243 env-steps/s at 4096 environments against native's
                    1850 ceiling -- and about 48x slower than an RTX 5090
                    (docs/DESIGN.md §9.5).
    native · cpu    real training without a GPU
    warp   · cpu    kernel-logic debugging and cross-checking only, when Warp
      (sim cpu)       has no Metal device or MJRL_SIM_DEVICE=cpu asks for it
                    (very slow -- measured at roughly 1/40 of multi-threaded
                    native; do not train with it)

The torch device and the Warp device are two things, and the resolution names
both: `device` is where the environment's tensors live and the policy runs by
default, `sim_device` is where physics is computed. They coincide everywhere
except on Apple Silicon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

__all__ = ["Resolution", "resolve", "BackendUnavailable"]

Backend = Literal["warp", "native"]

_ENV_BACKEND = "MJRL_BACKEND"
_ENV_DEVICE = "MJRL_DEVICE"
_ENV_SIM_DEVICE = "MJRL_SIM_DEVICE"
_ENV_AGENT_DEVICE = "MJRL_AGENT_DEVICE"


class BackendUnavailable(RuntimeError):
    """An explicitly requested backend/device is unavailable here. Fail, do not
    fall back."""


@dataclass(frozen=True)
class Resolution:
    backend: Backend
    device: str
    num_envs: int
    cpu_threads: int
    forced: bool
    """True when this came from an explicit choice rather than detection."""
    sim_device: str = ""
    """The Warp device physics runs on. The torch `device` everywhere but on
    Apple Silicon, where it is `metal:0` for `device="cpu"`. Native ignores it.
    Empty only while `resolve()` is building the object; see `__post_init__`."""
    agent_device: str = ""
    """The torch device the policy learns on. `device`, except `mps` when the
    simulation is on Metal and torch has MPS: the environment's tensors stay
    on the CPU (unified memory) and the runner copies them across. Measured on
    an M3 Max at 4096 environments: learning 6.7 s -> 2.0 s per iteration."""
    strip_visual: bool | None = None
    """Whether native strips visual-only meshes. None = decide from the memory the
    per-environment models would take."""
    notes: tuple[str, ...] = ()
    """Things the user should be told, printed before training starts."""

    def __post_init__(self) -> None:
        # The two extra devices default to the torch one, so a Resolution built
        # by hand (the tests, a probe) means what upstream meant.
        if not self.sim_device:
            object.__setattr__(self, "sim_device", self.device)
        if not self.agent_device:
            object.__setattr__(self, "agent_device", self.device)

    def banner(self) -> str:
        how = "forced" if self.forced else "auto"
        head = (
            f"[mjrl] backend={self.backend} device={self.device} "
            f"num_envs={self.num_envs}"
        )
        if self.backend == "native":
            head += f" threads={self.cpu_threads}"
        # Named only when they differ from the torch device, which is the one
        # case somebody has to know about: the GPU is being used and it is not
        # the one `device` says.
        if self.backend == "warp" and self.sim_device != self.device:
            head += f" sim={self.sim_device}"
        if self.agent_device != self.device:
            head += f" agent={self.agent_device}"
        return "\n".join(
            [f"{head} ({how})", *(f"[mjrl] warning: {n}" for n in self.notes)]
        )


# ── Detection ─────────────────────────────────────────────────────────────


def _torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _mujoco_warp_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("mujoco_warp") is not None


def _warp_sees_cuda() -> bool:
    try:
        import warp as wp
    except ImportError:
        return False
    try:
        wp.init()
        return any(d.is_cuda for d in wp.get_devices())
    except Exception:  # noqa: BLE001 - a failed warp init means unavailable
        return False


def _warp_sees_metal() -> bool:
    """Whether Warp lists an Apple GPU -- true only with the `warp-metal` overlay
    installed beside a matching warp-lang (the `metal` extra). Stock Warp has no
    `is_metal` on its devices, so this is False there without an exception."""
    try:
        import warp as wp
    except ImportError:
        return False
    try:
        wp.init()
        return any(getattr(d, "is_metal", False) for d in wp.get_devices())
    except Exception:  # noqa: BLE001 - a failed warp init means unavailable
        return False


def _torch_mps() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


#: Ceiling on the automatic thread count.
#:
#: "Every core" is the obvious default and it is **measurably wrong**. Roughly
#: half of a native step is serial -- the gather and scatter between the batch
#: buffers and the per-environment MjData is Python-side -- so by Amdahl the
#: parallel part is already spent by about eight workers, and past that each
#: extra thread only adds synchronisation and scheduling overhead.
#:
#: Measured on an i9-14900KF (8 performance + 16 efficiency cores, 32 logical),
#: ms per policy step, native:cpu, jumper.tetrapod:
#:
#:     envs    2      4      8     16     24     32 (= every core)
#:       16  10.63   8.10   7.17   9.70    --    9.85
#:       64  24.14  16.31  15.27  21.18  22.39  22.36
#:      256  92.84  69.79  59.09  62.84  67.39  68.82
#:
#: Eight is the optimum at every environment count, and "every core" costs
#: 16-46%. Pinning does not rescue it -- eight threads on eight dedicated
#: performance cores measured 21.44 ms against 15.71 unpinned, because the main
#: thread needs somewhere to run too. Nor is the physical core count the answer
#: (24 here): it measured no better than 32.
#:
#: A machine with fewer cores than this simply uses what it has; the ceiling
#: only bites on many-core hosts, which is exactly where the old default hurt.
#: `MJRL_CPU_THREADS` overrides it in either direction.
DEFAULT_MAX_THREADS = 8


def _available_cores() -> int:
    """Cores this process may actually use.

    `sched_getaffinity` respects cgroup / taskset limits, while `os.cpu_count()`
    inside a container reports the host's cores.
    """
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _default_threads() -> int:
    return min(_available_cores(), DEFAULT_MAX_THREADS)


# ── Resolution ────────────────────────────────────────────────────────────


def resolve(
    backend: str = "auto",
    device: str = "auto",
    num_envs: int | None = None,
    cpu_threads: int = 0,
    strip_visual: bool | None = None,
    num_envs_gpu: int = 4096,
    num_envs_cpu: int = 64,
    training: bool = True,
) -> Resolution:
    """Resolve the final backend / device / num_envs / thread count.

    When `backend` or `device` is ``"auto"``, environment variables are consulted
    first and capabilities detected second. Explicit values (from the command line
    or the environment) never fall back.

    `training` says `num_envs` is PPO's batch size, which is what makes a count
    other than the backend's default worth a warning. A replay's count is how many
    robots are on screen and has no batch to change, so there the warning is
    noise -- printed on every replay, it teaches people to skip the one that means
    something.
    """
    env_backend = os.environ.get(_ENV_BACKEND)
    env_device = os.environ.get(_ENV_DEVICE)

    if backend == "auto" and env_backend:
        backend = env_backend
    if device == "auto" and env_device:
        device = env_device

    forced = backend != "auto" or device != "auto"

    if backend == "auto":
        if _mujoco_warp_importable() and _torch_cuda() and _warp_sees_cuda():
            backend, auto_device = "warp", "cuda:0"
        elif _mujoco_warp_importable() and _warp_sees_metal():
            # Apple Silicon with the `metal` extra: torch's device is "cpu", the
            # simulation's is the Metal GPU (below).
            backend, auto_device = "warp", "cpu"
        else:
            backend, auto_device = "native", "cpu"
        if device == "auto":
            device = auto_device

    if device == "auto":
        device = "cuda:0" if backend == "warp" else "cpu"

    if device.startswith("metal"):
        # torch has no Metal device, so the tensors cannot live there; what the
        # person means is the simulation on the Apple GPU, and that is what
        # device=cpu does whenever Warp lists one. Said rather than fixed up:
        # a device string that resolves to something else is a silent fallback.
        raise BackendUnavailable(
            f"device={device} is not a torch device. On Apple Silicon use "
            "--backend warp --device cpu: the environment's tensors stay on the "
            "CPU and the simulation runs on metal:0 (MJRL_SIM_DEVICE=cpu keeps it "
            "off the GPU)."
        )

    # ── Availability check for the resolved combination: fail, never fall back ──
    if backend == "warp":
        if not _mujoco_warp_importable():
            raise BackendUnavailable(
                "backend=warp was requested but mujoco_warp is not installed.\n"
                "Install it with: pip install \"mjlab[cu128]\", or use "
                "--backend native."
            )
        if device.startswith("cuda") and not (_torch_cuda() and _warp_sees_cuda()):
            raise BackendUnavailable(
                f"device={device} was requested but this machine has no usable "
                f"CUDA device.\n"
                "Diagnose with `python -c \"import warp as wp; wp.init(); "
                "print(wp.get_devices())\"`, or use --backend native --device cpu."
            )
    elif backend == "native":
        if device != "cpu":
            raise BackendUnavailable(
                f"the native backend only supports device=cpu, got {device!r}.\n"
                "For a GPU use --backend warp --device cuda:0."
            )
    else:
        raise BackendUnavailable(
            f"unknown backend {backend!r}; choose warp / native / auto"
        )

    default_envs = num_envs_gpu if backend == "warp" else num_envs_cpu
    resolved_envs = num_envs if num_envs is not None else default_envs
    # The thread count is capped by the environment count here, **so that the
    # banner tells the truth**. `mujoco.rollout` requires the scratch list to be
    # exactly nthread long and the backend only has num_envs MjData copies, so a
    # cap is inevitable; capping only inside the backend would leave the banner
    # printing "threads=32" while 16 actually ran.
    threads = min(cpu_threads or _default_threads(), resolved_envs)

    # ── Where physics runs, and where the policy learns ──
    sim_device = device
    agent_device = device
    if backend == "warp" and device == "cpu":
        env_sim = os.environ.get(_ENV_SIM_DEVICE)
        if env_sim:
            sim_device = env_sim
            if sim_device.startswith("metal") and not _warp_sees_metal():
                raise BackendUnavailable(
                    f"{_ENV_SIM_DEVICE}={env_sim} but Warp lists no Metal device. "
                    "Install the `metal` extra (pip install -e \".[metal]\") on "
                    "Apple Silicon, or unset it."
                )
        elif _warp_sees_metal():
            sim_device = "metal:0"
    if backend == "warp" and sim_device.startswith("metal"):
        env_agent = os.environ.get(_ENV_AGENT_DEVICE)
        if env_agent:
            agent_device = env_agent
        elif _torch_mps():
            agent_device = "mps"

    notes: list[str] = []
    if backend == "warp" and device == "cpu" and sim_device == "cpu":
        notes.append(
            "warp's cpu device is a serial debugging path, measured at roughly "
            "1/40 of multi-threaded native. Do not train with it; for CPU training "
            "use --backend native."
        )
    if backend == "warp" and sim_device.startswith("metal"):
        notes.append(
            "the simulation runs on the Apple GPU through warp-metal, a community "
            "backend. Measured on an M3 Max at 4096 environments: 2.8x the native "
            "CPU backend, and about 48x slower than an RTX 5090. Whether a policy "
            "from it matches one from CUDA has not been checked (docs/DESIGN.md "
            "§9.5)."
        )
    if training and resolved_envs != default_envs:
        notes.append(
            f"num_envs={resolved_envs} differs from this backend's default of "
            f"{default_envs}. PPO's effective batch size changes with it and the "
            "hyper-parameters need re-tuning -- changing backend is not the same as "
            "changing machine and reproducing the same policy."
        )

    return Resolution(
        backend=backend,  # type: ignore[arg-type]
        device=device,
        num_envs=resolved_envs,
        cpu_threads=threads,
        strip_visual=strip_visual,
        forced=forced,
        sim_device=sim_device,
        agent_device=agent_device,
        notes=tuple(notes),
    )
