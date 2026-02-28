"""Compatibility shim for removed legacy barrier utility module."""

_REMOVAL_MESSAGE = (
    "ml4t.engineer.labeling.barrier_utils has been removed. "
    "Use ml4t.engineer.labeling.triple_barrier.triple_barrier_labels for labeling. "
    "Legacy helpers apply_triple_barrier, calculate_returns, and compute_barrier_touches "
    "are no longer supported."
)

raise ImportError(_REMOVAL_MESSAGE)
