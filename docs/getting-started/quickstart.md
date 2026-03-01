# Quickstart

Get up and running with ML4T Engineer in 5 minutes.

## Basic Feature Computation

```python
import polars as pl
from ml4t.engineer import compute_features

# Create sample OHLCV data
df = pl.DataFrame({
    "open": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,
    "high": [102.0, 103.0, 104.0, 105.0, 106.0] * 20,
    "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 20,
    "close": [101.0, 102.0, 103.0, 104.0, 105.0] * 20,
    "volume": [1000, 1100, 1200, 1300, 1400] * 20,
})

# Compute features
result = compute_features(df, ["rsi", "macd", "atr"])
print(result.columns)
```

## Custom Parameters

```python
# Use dict format for custom parameters
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
    {"name": "bollinger_bands", "params": {"period": 20, "std_dev": 2.5}},
])
```

## YAML Configuration

Create a `features.yaml` file:

```yaml
features:
  - name: rsi
    params:
      period: 14
  - name: macd
    params:
      fast: 12
      slow: 26
      signal: 9
  - name: atr
    params:
      period: 14
```

Then use it:

```python
result = compute_features(df, "features.yaml")
```

## Triple-Barrier Labeling

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=0.01,  # 1% stop loss
    max_holding_period=20,  # 20 bar maximum holding
)

labels = triple_barrier_labels(
    df,
    config=config,
)
```

## Explore Available Features

```python
from ml4t.engineer import feature_catalog

# List all categories
print(feature_catalog.categories())
# ['momentum', 'trend', 'volatility', ...]

# List features in a category
print(feature_catalog.list(category="momentum"))
# ['rsi', 'macd', 'stoch', 'cci', ...]

# Get feature details
info = feature_catalog.describe("rsi")
print(info)
# {'name': 'rsi', 'category': 'momentum', 'normalized': True, ...}
```

## Next Steps

- [Features Guide](../user-guide/features.md) - Deep dive into all indicators
- [Labeling Guide](../user-guide/labeling.md) - Triple-barrier and other labeling methods
- [API Reference](../api/index.md) - Complete API documentation
