# Labeling Methods

ML4T Engineer provides advanced labeling methods for supervised learning in finance.

## Triple-Barrier Labeling

The triple-barrier method from *Advances in Financial Machine Learning* creates labels based on which barrier is touched first:

- **Upper barrier**: Profit target (label = 1)
- **Lower barrier**: Stop loss (label = -1)
- **Vertical barrier**: Time limit (label = 0)

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=0.01,  # 1% stop loss
    max_holding_period=20,  # 20 bar horizon
)

labels = triple_barrier_labels(
    df,
    config=config,
)
```

## ATR-Based Dynamic Barriers

Use volatility-adjusted barriers that adapt to market conditions:

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import atr_triple_barrier_labels

config = LabelingConfig.atr_barrier(
    atr_period=14,
    atr_tp_multiple=2.0,  # 2x ATR profit target
    atr_sl_multiple=1.0,  # 1x ATR stop loss
    max_holding_period=20,
)

labels = atr_triple_barrier_labels(df, config=config)
```

## Performance

- **Speed**: 50,000 labels/second
- **Memory**: Efficient vectorized implementation
- **Accuracy**: Exact match with López de Prado's reference

## Best Practices

1. **Avoid overlapping labels**: Use `min_return` to filter small moves
2. **Handle class imbalance**: Triple-barrier often creates imbalanced labels
3. **Account for transaction costs**: Barriers should exceed expected costs
