"""Compatibility shim for removed labeling.core module."""

_REMOVAL_MESSAGE = (
    "ml4t.engineer.labeling.core has been removed. "
    "Use ml4t.engineer.labeling.triple_barrier for triple_barrier_labels "
    "(with ml4t.engineer.config.LabelingConfig); "
    "ml4t.engineer.labeling.horizon_labels for fixed_time_horizon_labels and "
    "trend_scanning_labels; "
    "ml4t.engineer.labeling.uniqueness for build_concurrency, "
    "calculate_label_uniqueness, calculate_sample_weights, and sequential_bootstrap. "
    "Legacy helpers apply_triple_barrier, calculate_returns, and "
    "compute_barrier_touches are no longer supported."
)

raise ImportError(_REMOVAL_MESSAGE)
