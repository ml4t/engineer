"""Validation module for ml4t.engineer.

IMPORTANT: Cross-validation with purging and embargo for financial time series
is implemented in the ml4t.eval library, not ml4t.engineer.

Please use:
    from ml4t.eval.splitters import PurgedWalkForwardCV, CombinatorialPurgedKFold

See the ml4t.eval documentation for proper cross-validation in financial ML.
"""

# No exports - see ml4t.eval for cross-validation utilities
__all__: list[str] = []
