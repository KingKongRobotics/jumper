"""Observation history that spans a duration rather than a step count.

`OBS_HISTORY` is 5 frames, and the 5 was chosen at 50 Hz -- "a third of a 3.125
Hz gait cycle, enough to carry the direction a contact is developing rather than
only its present value", as `common/velocity_env.py` puts it. The frame count
then survived two rate changes that the *spacing* did not.

**Reach here means how far back the oldest frame is**, `(frames - 1) * stride /
rate`, and saying which convention is in use matters because the other one is
also in this repository: `velocity_env.py` says five frames "spans 0.1 s",
counting each frame as covering a control period. The same five frames are 100 ms
under that convention and 80 ms under this one. The first draft of this module
used both at once -- the docstring said the window was restored to 100 ms while
the README it generates said 80 -- so everything here and in the bundle quotes
reach, which is the number that says what the policy can still see.

    control rate   stride   spacing   reach
      50 Hz          1       20 ms     80 ms   <- where the 5 was chosen
     100 Hz          1       10 ms     40 ms
     200 Hz          1        5 ms     20 ms   <- a quarter of what the 5 assumes
     200 Hz          4       20 ms     80 ms   <- this module

The note in `env_cfg.py` that recorded the loss when the rate first doubled
offered one fix -- 10 frames, and an actor input of about 800 rather than 415 --
and called it "an architecture change rather than a rate change".

**Striding is the other fix, and it costs nothing in width.** Four control steps
at 200 Hz is 20 ms, which is one control step at 50 Hz: the strided history is
not a similar window, it is **the same frames at the same spacing** the count was
calibrated on, for an actor input that does not change.

    frames  stride   depth   reach at 200 Hz   actor input
      5       1        5          20 ms            415
      5       4       17          80 ms            415
     10       1       10          45 ms           ~800

## What is given up

Resolution at the recent end. Consecutive frames used to be 5 ms apart and are
now 20 ms, so a contact that began within the last 15 ms is visible in the newest
frame alone -- its *rate* of development is not, until the next strided frame
lands. That is the trade the table makes explicit: 80 ms of reach at 20 ms
spacing against 20 ms of reach at 5 ms spacing. The reach is what the 5 was
chosen for, and 20 ms spacing is what it was chosen *at*.

A non-uniform spacing -- two consecutive frames and then 2, 4, 8 -- buys back the
recency at a similar reach. It is not what this does, because there is no
measurement here to choose the spacing with and an even stride keeps the mirror
below trivially correct.

## Where it sits in the pipeline

mjlab's own pipeline is `compute -> noise -> clip -> scale -> delay -> history`,
and this class replaces the last stage while **keeping the noise ahead of it**.
That ordering is not a detail:

    buffer the raw value and let the manager add noise afterwards, and every
    frame's noise is resampled on every step.

A policy seeing five independently-jittering copies of one past reading can
average the noise away, which no robot can do -- on hardware a past reading is a
number that was written down once, with whatever error it had then. The sim would
be quietly *cleaner* than the machine, in the direction that flatters a policy
and fails on deployment. So this class applies `noise.apply` itself, exactly as
`ObservationManager._compute_group_obs` does, and `env_cfg.py` clears the term's
own noise so it is not applied twice.

`clip` and `scale` are elementwise and frame-independent, so the manager applying
them to the stacked vector is identical to applying them per frame. Neither is
set on any term this wraps, but the equivalence is why it does not have to be.

## The buffer is mjlab's

`CircularBuffer` already has the two behaviours this needs and would otherwise be
reimplemented: `buffer` returns frames chronologically, oldest first, which is
the layout `common/mdp/symmetry.py` mirrors against; and a reset row is
backfilled with the *first* value appended after it rather than left at zero, so
an episode starts with the history full of its own first observation instead of
17 steps of zeros that occur nowhere else in training.

Reading every `stride`-th frame of a `stride * (frames - 1) + 1` buffer with
`[:, ::stride]` lands on index 0 and index `depth - 1` exactly, so the oldest and
the newest frames are both kept and the ordering convention is inherited.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.utils.buffers import CircularBuffer

if TYPE_CHECKING:
    import torch
    from mjlab.envs import ManagerBasedRlEnv


class StridedHistory:
    """Stack `frames` copies of a term, `stride` control steps apart.

    Wraps another observation term rather than being one: `func` and `params` are
    what it would have been configured with, and the output is the same width the
    manager's own `history_length = frames` would have produced.

    Set the manager's `history_length` to 0 on any term this wraps, or the history
    is taken twice -- and tell `common/mdp/symmetry.py` how many frames there are,
    which is what `history_frames` is for. It reads `history_length` from the
    manager to mirror frame by frame, and a 0 there makes it treat five frames as
    one wide observation: the joint permutation would then shuffle values *across*
    frames, so the mirrored sample is a robot whose left legs are several control
    steps out of step with its right. Nothing raises; the shapes match.
    """

    #: Prefix reserved for this wrapper's own parameters. Everything else in
    #: `cfg.params` belongs to the term being wrapped and is passed through.
    PREFIX = "_history_"

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        # **The wrapped term's params stay at the top level**, and that is the
        # whole of this class's interface with the rest of the repository.
        #
        # The obvious spelling nests them -- `params = {"func": ..., "params":
        # {...}}` -- and it broke three things that read a term's config directly,
        # one at a time, each silently:
        #
        #   * `ManagerBase._resolve_common_term_cfg` resolves `SceneEntityCfg`s at
        #     the top level only, so `joint_pos` returned all 22 joints instead of
        #     the 20 the policy drives;
        #   * `scripts/export.py::_observed_joint_order` reads
        #     `params["asset_cfg"]` to size every per-joint block in the
        #     deployment contract, and fell back to "unrestricted" -- 22 again;
        #   * `common/mdp/symmetry.py` reads `params["mirror_kind"]`, which no
        #     wrapped term currently sets, so that one was luck.
        #
        # Patching each consumer is how the fourth one gets missed. A reserved
        # prefix keeps the term's config looking exactly like the term's config.
        params = cfg.params
        p = self.PREFIX
        self._func = params[p + "func"]
        self._frames = int(params[p + "frames"])
        self._stride = int(params[p + "stride"])
        self._noise = params.get(p + "noise")
        self._params = {k: v for k, v in params.items() if not k.startswith(p)}
        assert self._frames > 0 and self._stride > 0

        self._buffer = CircularBuffer(
            max_len=self._stride * (self._frames - 1) + 1,
            batch_size=env.num_envs,
            device=env.device,
        )
        # The step the buffer was last appended for, and what that append
        # produced. `ObservationManager.compute` is called more than once in some
        # steps -- it has its own `update_history` flag for the same reason -- and
        # an unguarded append would advance this history once per call, so the
        # oldest frame would be a different age depending on how often anything
        # inspected the observation.
        self._step: int | None = None
        self._out: "torch.Tensor | None" = None

    @property
    def history_frames(self) -> int:
        """How many frames the flattened output holds.

        Read by `common/mdp/symmetry.py`, to mirror frame by frame, and by
        `scripts/export.py`, to write the block's shape into the deployment
        contract. Both read `history_length` off the config first and find 0,
        because the manager is not the thing stacking these.
        """
        return self._frames

    @property
    def history_stride(self) -> int:
        """Control steps between consecutive frames.

        Read by `scripts/export.py`, which writes it into `layout.json` and into
        the bundle README **only when it is not 1** -- so an unstrided bundle is
        unchanged and a strided one carries a field an old reader fails on rather
        than silently ignores. What it is protecting against is not a crash: a
        builder that shifts one frame per inference emits exactly the right
        number of floats, from a window `stride` times too short.
        """
        return self._stride

    def __call__(self, env: "ManagerBasedRlEnv", **params) -> "torch.Tensor":
        del params  # consumed in __init__; the manager passes them back
        step = int(env.common_step_counter)
        if self._out is None or step != self._step:
            obs = self._func(env, **self._params).clone()
            if self._noise is not None:
                obs = self._noise.apply(obs)
            self._buffer.append(obs)
            self._out = self._buffer.buffer[:, :: self._stride].reshape(
                env.num_envs, -1
            )
            self._step = step
        return self._out

    def reset(self, env_ids=None) -> None:
        """Drop the reset rows, and the cache that would outlive them."""
        self._buffer.reset(batch_ids=None if isinstance(env_ids, slice) else env_ids)
        # The manager invalidates its own `_obs_buffer` here for the same reason:
        # a reset happens inside a step, so the value cached for that step is no
        # longer the value this term would produce.
        self._out = None


__all__ = ["StridedHistory"]
