# ── [mjrl] new (against mjlab 1.6.0) ──────────────────────────────────
# Reason: which Warp device simulates is, upstream, the torch device's own
# name: `wp.get_device(self.device)`. That holds on CUDA, where the two are
# one card, and on a CPU-only machine, where both are "cpu". It does not hold
# on Apple Silicon: torch has no Metal device, so the environment's tensors
# are "cpu" tensors, while Warp -- with the `warp-metal` overlay -- lists
# "metal:0" and can simulate there. Unified memory is what makes the split
# work: a Warp array on the Metal device is aliased by a CPU tensor with no
# copy, and the only thing the CPU side has to do is wait for the GPU before
# reading.
#
# The mapping is registered here, by `mjrl.backend.select.use_backend()`,
# before the environment is built -- the same shape as `set_simulation_cls`
# in `sim/__init__.py`, and for the same reason: the construction reads it
# once. It lives under utils/ rather than beside the registry because the
# sensors import it, and importing anything from the `mjlab.sim` package runs
# `sim/__init__.py`, which loads `sim.py`, which imports the managers, which
# import the sensors: a cycle (measured as 15 collection errors).
# With nothing registered the behaviour is upstream's, byte for byte.
#
# When upgrading upstream: this file is new, keep it; the call sites are the
# four `[mjrl]` marks in sim.py, sim_data.py, sensor_context.py and
# raycast_sensor.py.
"""The Warp device that simulates for a torch device (`[mjrl]`, see above)."""

from __future__ import annotations

import warp as wp

_SIM_DEVICE: str | None = None


def set_sim_device(device: str | None) -> None:
  """Register the Warp device the next environment simulates on.

  `None` restores upstream's rule, the torch device's own name. Called by
  `mjrl.backend.select.use_backend()` **before** the environment is built.
  """
  global _SIM_DEVICE
  _SIM_DEVICE = device


def sim_device(torch_device: str) -> str:
  """The Warp device for `torch_device`: the registered one, else the same name."""
  return torch_device if _SIM_DEVICE is None else _SIM_DEVICE


def synchronize(device: wp.Device) -> None:
  """Wait for a Metal device before CPU tensors aliasing its arrays are read.

  On CUDA the torch stream ordering does this (`sim_data.WarpBridge`); on
  Metal the CPU reads unified memory directly, and a read before the GPU has
  finished sees the previous step with no error anywhere.
  """
  if getattr(device, "is_metal", False):
    wp.synchronize_device(device)
