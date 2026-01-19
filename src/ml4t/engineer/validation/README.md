# Cross-Validation for Financial Time Series

## Important Notice

Cross-validation with purging and embargo for financial time series is implemented in the **ml4t-backtest** library, not ml4t-engineer.

## Why Not in ml4t-engineer?

The ml4t-engineer library focuses on feature engineering, while ml4t-backtest specializes in backtesting and model evaluation. Proper cross-validation for financial time series requires:

1. **Purging**: Removing training samples that are too close to test samples to prevent information leakage
2. **Embargo**: Adding a gap after test samples to account for the forward-looking nature of labels
3. **Label Horizons**: Accounting for how far into the future labels look

These requirements are tightly coupled with backtesting and evaluation logic, making ml4t-backtest the natural home for these utilities.

## Using Cross-Validation with ml4t-engineer Data

To use proper cross-validation with data processed by ml4t-engineer:

```python
# 1. Engineer features with ml4t-engineer
from ml4t.engineer import compute_features
from ml4t.engineer.labeling import triple_barrier_labels

# Create features
result = compute_features(df, ["rsi", "adx"])

# Apply labeling
labeled_df = triple_barrier_labels(
    df,
    upper_barrier=0.02,
    lower_barrier=0.01,
    max_holding=10,
)

# 2. Use ml4t-backtest for cross-validation (when available)
# See ml4t-backtest documentation for PurgedWalkForwardCV
```

## Available Cross-Validators in ml4t-backtest

1. **PurgedWalkForwardCV**: Walk-forward cross-validation with purging and embargo
   - Best for time series with strong temporal dependencies
   - Supports expanding and rolling windows

2. **CombinatorialPurgedKFold**: Combinatorial purged K-fold cross-validation
   - Generates more training/test combinations
   - Better for limited data scenarios

## References

- López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7: Cross-Validation in Finance
- Bailey, D. H., & López de Prado, M. (2012). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"

## See Also

- [ml4t-backtest documentation](https://pypi.org/project/ml4t-backtest/) for detailed usage examples
- [ml4t-engineer labeling module](../labeling/) for creating labels with proper horizons
