"""Feature-outcome analysis and visualization (Module C).

This module provides comprehensive analysis of how features relate to outcomes.
"""

from ml4t.engineer.outcome.drift import DriftSummaryResult, analyze_drift
from ml4t.engineer.outcome.feature_outcome import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcome,
    FeatureOutcomeResult,
)
from ml4t.engineer.outcome.plot_ic import plot_ic_comparison, plot_ic_time_series
from ml4t.engineer.outcome.plot_importance import (
    plot_feature_importance,
    plot_importance_comparison,
)

__all__ = [
    # Core analysis
    "FeatureOutcome",
    "FeatureOutcomeResult",
    "FeatureICResults",
    "FeatureImportanceResults",
    # Drift detection
    "analyze_drift",
    "DriftSummaryResult",
    # Plotting
    "plot_feature_importance",
    "plot_importance_comparison",
    "plot_ic_time_series",
    "plot_ic_comparison",
]
