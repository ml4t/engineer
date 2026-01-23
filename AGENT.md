# ml4t-engineer

Feature engineering for financial ML. 120 indicators, triple-barrier labeling, alternative bars.

## Quick Start

```python
from ml4t.engineer import compute_features
result = compute_features(df, ["rsi", "macd", "atr"])
```

## Directory Map

| Path | Purpose | Key Exports |
|------|---------|-------------|
| `features/` | 120 technical indicators | `compute_features()`, `list_features()` |
| `labeling/` | ML label generation | `triple_barrier_labels()`, `rolling_percentile_binary_labels()` |
| `bars/` | Alternative bar sampling | `volume_bars()`, `dollar_bars()`, `tick_imbalance_bars()` |
| `core/` | Registry, types, validation | `get_registry()`, `FeatureMetadata` |
| `config/` | Pydantic configurations | `LabelingConfig`, `PreprocessingConfig` |
| `preprocessing.py` | Scalers, transforms | `StandardScaler`, `RobustScaler` |

## Entry Points

```python
from ml4t.engineer import compute_features, list_features, describe_feature
from ml4t.engineer.labeling import triple_barrier_labels, atr_triple_barrier_labels
from ml4t.engineer.labeling import rolling_percentile_binary_labels
from ml4t.engineer.bars import volume_bars, dollar_bars, tick_imbalance_bars
```

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| momentum | 31 | rsi, macd, adx, stoch, cci, mfi |
| trend | 10 | sma, ema, kama, tema, dema |
| volatility | 15 | atr, yang_zhang, bollinger_bands, garch |
| microstructure | 12 | kyle_lambda, amihud, vpin, roll_spread |
| ml | 11 | lag, fractional_diff, entropy, cyclical_encode |
| statistics | 8 | stddev, var, linear_regression |
| volume | 3 | obv, ad, adosc |
| price_transform | 5 | avgprice, typprice, medprice |
| math | 3 | max_, min_, sum_ |

## Core API

```python
# List available features
list_features()              # All 120 features
list_features("momentum")    # 31 momentum indicators

# Compute with custom parameters
compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
])

# Get feature metadata
describe_feature("rsi")  # Returns dict with params, formula, range
```

## Labeling Methods

| Method | Use Case |
|--------|----------|
| `triple_barrier_labels()` | Path-dependent with PT/SL barriers |
| `atr_triple_barrier_labels()` | ATR-based dynamic barriers |
| `rolling_percentile_binary_labels()` | Adaptive percentile thresholds |
| `trend_scanning_labels()` | De Prado's t-stat method |
| `fixed_time_horizon_labels()` | Simple forward returns |

## Performance

- **Speed**: Polars-native implementation, ~1x TA-Lib for RSI, streaming at 11M rows/second
- **Validation**: 60 TA-Lib compatible features at 1e-6 tolerance
- **Throughput**: ~480K indicator calculations/second (batch mode)

## Navigation

See `src/ml4t/engineer/AGENT.md` for package-level detail with function signatures.

## References

- Lopez de Prado (2018). *Advances in Financial Machine Learning*
- Lopez de Prado (2020). *Machine Learning for Asset Managers*
