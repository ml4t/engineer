# Cross-Validation for Financial Time Series

## Important Notice

Cross-validation with purging and embargo for financial time series is implemented in the **qeval** library, not qfeatures.

## Why Not in QFeatures?

The qfeatures library focuses on feature engineering, while qeval specializes in model evaluation and backtesting. Proper cross-validation for financial time series requires:

1. **Purging**: Removing training samples that are too close to test samples to prevent information leakage
2. **Embargo**: Adding a gap after test samples to account for the forward-looking nature of labels
3. **Label Horizons**: Accounting for how far into the future labels look

These requirements are tightly coupled with backtesting and evaluation logic, making qeval the natural home for these utilities.

## Using Cross-Validation with QFeatures Data

To use proper cross-validation with data processed by qfeatures:

```python
# 1. Engineer features with qfeatures
import qfeatures as qf
from qfeatures.labeling import BarrierConfig, triple_barrier_labels

# Create features
df = df.with_columns([
    qf.ta.rsi("close", 14).alias("rsi"),
    qf.ta.adx("high", "low", "close", 14).alias("adx"),
])

# Apply labeling
config = BarrierConfig(
    upper_barrier=0.02,
    lower_barrier=-0.01,
    max_holding_period=10,
)
labeled_df = triple_barrier_labels(df, config)

# 2. Use qeval for cross-validation
from qeval.splitters import PurgedWalkForwardCV

# Create cross-validator
cv = PurgedWalkForwardCV(
    n_splits=5,
    label_horizon=10,  # Match your labeling horizon
    embargo_size=2,    # Additional safety buffer
)

# Split your data
X = labeled_df.select(feature_columns)
y = labeled_df.select("label")

for train_idx, test_idx in cv.split(X, y):
    # Train and evaluate your model
    pass
```

## Available Cross-Validators in qeval

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

- [qeval documentation](https://github.com/quantlab/qeval) for detailed usage examples
- [qfeatures labeling module](../labeling/) for creating labels with proper horizons
