"""Feature selection for ML pipelines.

This module provides systematic feature selection with multiple criteria:
- IC filtering (predictive power)
- Importance filtering (MDI/permutation/SHAP)
- Correlation filtering (redundancy removal)
- Drift filtering (stability)

Example:
    >>> from ml4t.engineer.selection import FeatureSelector
    >>> from ml4t.engineer.outcome import FeatureOutcome
    >>> from ml4t.engineer.relationships import compute_correlation_matrix
    >>>
    >>> # Analyze features
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, returns_df)
    >>> corr_matrix = compute_correlation_matrix(features_df)
    >>>
    >>> # Select features
    >>> selector = FeatureSelector(results, corr_matrix)
    >>> selector.run_pipeline([
    ...     ("ic", {"threshold": 0.02}),
    ...     ("correlation", {"threshold": 0.8}),
    ...     ("importance", {"threshold": 0.01, "method": "mdi"})
    ... ])
    >>> selected = selector.get_selected_features()
"""

from ml4t.engineer.selection.systematic import (
    FeatureSelector,
    SelectionReport,
    SelectionStep,
)

__all__ = [
    "FeatureSelector",
    "SelectionReport",
    "SelectionStep",
]
