"""Cross-validation utilities notice.

IMPORTANT: Cross-validation with purging and embargo for financial time series
is implemented in the ml4t.eval library, not ml4t.engineer.

The ml4t.eval library provides proper implementations of:
- PurgedWalkForwardCV: Walk-forward cross-validation with purging and embargo
- CombinatorialPurgedKFold: Combinatorial purged K-fold cross-validation

These implementations correctly handle:
- Purging: Removing training samples that are too close to test samples
- Embargo: Adding a gap after test samples to prevent information leakage
- Label horizons: Accounting for the forward-looking nature of labels

For cross-validation in financial machine learning, please use:
    from ml4t.evaluation.splitters import PurgedWalkForwardCV, CombinatorialPurgedKFold

See the qeval documentation for usage examples.
"""

__all__: list[str] = []
