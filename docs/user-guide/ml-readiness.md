# ML-Readiness: Feature Normalization

This guide explains the `normalized` field in feature metadata and how to prepare features for machine learning models.

## Overview

Features in ml4t-engineer have a `normalized` metadata field that indicates whether a feature produces bounded outputs suitable for direct ML consumption:

- **`normalized=True`** (33 features): Output is bounded (typically 0-100 or -1 to 1). ML-ready without preprocessing.
- **`normalized=False`** (87 features): Output is unbounded (raw prices, volumes, spreads). Requires preprocessing.

## Why Normalization Matters

Machine learning models (especially neural networks and distance-based methods) perform better when features are on similar scales:

- **Gradient descent** converges faster with normalized inputs
- **Distance metrics** (k-NN, SVM) are biased toward high-magnitude features
- **Regularization** works correctly only when features have similar scales

## Normalized Features (ML-Ready)

These 33 features produce bounded outputs and can be used directly in ML models:

### Momentum (17 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `rsi` | 0-100 | Relative Strength Index |
| `stochastic` | 0-100 | Stochastic Oscillator |
| `stochf` | 0-100 | Fast Stochastic |
| `stochrsi` | 0-100 | Stochastic RSI |
| `cci` | typically -200 to 200 | Commodity Channel Index |
| `mfi` | 0-100 | Money Flow Index |
| `ultosc` | 0-100 | Ultimate Oscillator |
| `willr` | -100 to 0 | Williams %R |
| `cmo` | -100 to 100 | Chande Momentum Oscillator |
| `adx` | 0-100 | Average Directional Index |
| `adxr` | 0-100 | ADX Rating |
| `plus_di` | 0-100 | Plus Directional Indicator |
| `minus_di` | 0-100 | Minus Directional Indicator |
| `aroonosc` | -100 to 100 | Aroon Oscillator |
| `bop` | -1 to 1 | Balance of Power |
| `imi` | 0-100 | Intraday Momentum Index |

### Volatility (1 feature)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `natr` | 0-100% | Normalized ATR (percentage of price) |

### Regime (4 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `hurst_exponent` | 0-1 | Hurst exponent for trend detection |
| `choppiness_index` | 0-100 | Market choppiness indicator |
| `fractal_efficiency` | 0-1 | Price efficiency ratio |
| `trend_intensity_index` | 0-100 | Trend strength indicator |

### Statistics (5 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `coefficient_of_variation` | 0-inf (typically 0-1) | CV ratio |
| `variance_ratio` | typically 0.5-1.5 | VR statistic |
| `rolling_cv_zscore` | typically -3 to 3 | Z-score of CV |
| `rolling_drift` | typically -1 to 1 | Drift measure |
| `rolling_kl_divergence` | 0-inf (typically 0-2) | KL divergence |

### Microstructure (2 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `bid_ask_imbalance` | -1 to 1 | Order flow imbalance |
| `book_depth_ratio` | 0-1 | Depth asymmetry |

### ML Utilities (3 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `rolling_entropy` | 0-1 | Shannon entropy (normalized) |
| `rolling_entropy_lz` | 0-1 | Lempel-Ziv complexity |
| `rolling_entropy_plugin` | 0-1 | Plugin entropy estimator |

### Risk (2 features)
| Feature | Output Range | Description |
|---------|-------------|-------------|
| `downside_deviation` | 0-inf (similar scale to returns) | Downside volatility |
| `tail_ratio` | typically 0.5-2.0 | Tail risk ratio |

## Using the Registry API

```python
from ml4t.engineer.core.registry import get_registry

registry = get_registry()

# List all normalized features
normalized_features = [
    name for name in registry.list_all()
    if registry.get(name).normalized is True
]

# Check if a specific feature is normalized
meta = registry.get("rsi")
print(f"RSI normalized: {meta.normalized}")  # True

# Get features by category that are normalized
momentum_normalized = [
    name for name in registry.list_features(category="momentum")
    if registry.get(name).normalized is True
]
```

## Preprocessing Non-Normalized Features

For features with `normalized=False`, use the preprocessing module:

```python
from ml4t.engineer.preprocessing import StandardScaler, RobustScaler

# Standard scaling (mean=0, std=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# Robust scaling (resistant to outliers)
scaler = RobustScaler()
features_scaled = scaler.fit_transform(features_df)
```

### When to Use Each Scaler

| Scaler | Use When |
|--------|----------|
| `StandardScaler` | Data is approximately Gaussian |
| `RobustScaler` | Data has outliers or fat tails |
| `MinMaxScaler` | Need bounded 0-1 range |

## Best Practices

1. **Don't double-normalize**: If using normalized features (RSI, Stochastic), don't apply additional scaling.

2. **Consistent preprocessing**: Use `fit()` on training data, `transform()` on test data to prevent leakage.

3. **Group similar features**: When mixing normalized and non-normalized features, scale only the non-normalized ones.

4. **Consider feature ranges**: Even normalized features may have different effective ranges (RSI rarely hits 0 or 100 in practice).

## Example: Mixed Feature Pipeline

```python
from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry
from ml4t.engineer.preprocessing import StandardScaler

# Compute mixed features
result = compute_features(ohlcv, ["rsi", "sma", "atr", "macd"])

# Separate normalized and non-normalized
registry = get_registry()
normalized_cols = [
    col for col in result.columns
    if any(registry.get(f).normalized for f in ["rsi"]
           if col.startswith(f))
]
non_normalized_cols = [col for col in result.columns if col not in normalized_cols]

# Scale only non-normalized features
scaler = StandardScaler()
result = result.with_columns(
    scaler.fit_transform(result.select(non_normalized_cols))
)
```

## Reference

- See [Feature Registry](../api/registry.md) for full metadata API
- See [Preprocessing](../api/preprocessing.md) for scaler documentation
