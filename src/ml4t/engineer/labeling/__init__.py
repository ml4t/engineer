# mypy: disable-error-code="arg-type"
"""Labeling module for ml4t.engineer.

Provides generalized labeling functionality including triple-barrier method.
"""

from ml4t.engineer.core.registry import FeatureMetadata
from ml4t.engineer.labeling.atr_barriers import atr_triple_barrier_labels
from ml4t.engineer.labeling.barriers import BarrierConfig
from ml4t.engineer.labeling.calendar import (
    PandasMarketCalendar,
    SimpleTradingCalendar,
    TradingCalendar,
    calendar_aware_labels,
)
from ml4t.engineer.labeling.core import (
    build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
    fixed_time_horizon_labels,
    sequential_bootstrap,
    trend_scanning_labels,
    triple_barrier_labels,
)
from ml4t.engineer.labeling.meta_labels import (
    apply_meta_model,
    compute_bet_size,
    meta_labels,
)
from ml4t.engineer.labeling.percentile_labels import (
    compute_label_statistics,
    rolling_percentile_binary_labels,
    rolling_percentile_multi_labels,
)

__all__ = [
    "ALL_LABELING_FEATURES",
    "BarrierConfig",
    "PandasMarketCalendar",
    "SimpleTradingCalendar",
    "TradingCalendar",
    "apply_meta_model",
    "atr_triple_barrier_labels",
    "build_concurrency",
    "calculate_label_uniqueness",
    "calculate_sample_weights",
    "calendar_aware_labels",
    "compute_bet_size",
    "compute_label_statistics",
    "fixed_time_horizon_feature",
    "fixed_time_horizon_labels",
    "meta_labels",
    "register_labeling_features",
    "rolling_percentile_binary_labels",
    "rolling_percentile_multi_labels",
    "sequential_bootstrap",
    "trend_scanning_feature",
    "trend_scanning_labels",
    "triple_barrier_feature",
    "triple_barrier_labels",
]

# Create feature metadata objects for backward compatibility
triple_barrier_feature = FeatureMetadata(
    name="triple_barrier",
    func=triple_barrier_labels,
    category="labeling",
    description="Triple barrier labeling method",
    lookback=lambda **_kwargs: 0,  # Forward-looking, not backward-looking
)

# Feature metadata for the new labeling methods
fixed_time_horizon_feature = FeatureMetadata(
    name="fixed_time_horizon",
    func=fixed_time_horizon_labels,
    category="labeling",
    description="Fixed time horizon labeling",
    lookback=lambda **_kwargs: 0,  # Forward-looking
)

trend_scanning_feature = FeatureMetadata(
    name="trend_scanning",
    func=trend_scanning_labels,
    category="labeling",
    description="Trend scanning labeling (De Prado)",
    lookback=lambda **_kwargs: 0,  # Forward-looking
)

# List of all labeling features for testing
ALL_LABELING_FEATURES = [
    triple_barrier_feature,
    fixed_time_horizon_feature,
    trend_scanning_feature,
]


def register_labeling_features(registry: object = None) -> int:
    """
    Register labeling features.

    Parameters
    ----------
    registry : FeatureRegistry, optional
        Registry to register features with. If None, uses global registry.

    Returns
    -------
    int
        Number of features registered
    """
    from contextlib import suppress

    if registry is None:
        from ml4t.engineer.core.registry import get_registry

        registry = get_registry()

    # Register all labeling features
    for feature in ALL_LABELING_FEATURES:
        with suppress(ValueError):
            # Already registered errors are expected and can be ignored
            registry.register(feature)  # type: ignore[attr-defined]

    return len(ALL_LABELING_FEATURES)
