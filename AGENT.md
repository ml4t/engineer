# ml4t-engineer Agent Reference

## Purpose

High-performance feature engineering library for financial machine learning. Provides 107+ technical indicators, triple-barrier labeling, and alternative bar sampling with Polars-first implementation.

## Installation

```bash
pip install ml4t-engineer
```

## Quick Start

```python
import polars as pl
from ml4t.engineer import compute_features

# Load OHLCV data
df = pl.DataFrame({
    "open": [100.0, 101.0, 102.0, 103.0, 104.0],
    "high": [102.0, 103.0, 104.0, 105.0, 106.0],
    "low": [99.0, 100.0, 101.0, 102.0, 103.0],
    "close": [101.0, 102.0, 103.0, 104.0, 105.0],
    "volume": [1000, 1100, 1200, 1300, 1400],
})

# Compute features
result = compute_features(df, ["rsi", "macd", "atr"])
```

## Core API

### compute_features(data, features) -> DataFrame
Compute technical indicators on OHLCV data.
- data: pl.DataFrame - OHLCV data with columns: open, high, low, close, volume
- features: list[str] | list[dict] | Path - Feature specifications
- returns: pl.DataFrame - Original data with feature columns added

```python
# Default parameters
result = compute_features(df, ["rsi", "sma", "atr"])

# Custom parameters
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
])

# YAML config file
result = compute_features(df, "features.yaml")
```

### list_features(category=None) -> list[str]
List available features.
- category: str | None - Filter by category (momentum, trend, volatility, etc.)
- returns: list[str] - Sorted list of feature names

```python
from ml4t.engineer import list_features

all_features = list_features()  # All 107+ features
momentum = list_features("momentum")  # 31 momentum indicators
```

### list_categories() -> list[str]
Get available feature categories.
- returns: list[str] - ["math", "microstructure", "ml", "momentum", "price_transform", "statistics", "trend", "volatility", "volume"]

### describe_feature(name) -> dict
Get feature metadata.
- name: str - Feature name
- returns: dict - Description, parameters, formula, value range, etc.

```python
from ml4t.engineer import describe_feature

info = describe_feature("rsi")
# {'name': 'rsi', 'category': 'momentum', 'normalized': True,
#  'value_range': (0, 100), 'ta_lib_compatible': True, ...}
```

## Available Features

### Momentum (31 indicators)
| Name | Description | TA-Lib |
|------|-------------|--------|
| rsi | Relative Strength Index | Yes |
| macd | Moving Average Convergence/Divergence | Yes |
| stoch | Stochastic Oscillator | Yes |
| cci | Commodity Channel Index | Yes |
| willr | Williams %R | Yes |
| adx | Average Directional Index | Yes |
| mfi | Money Flow Index | Yes |
| roc | Rate of Change | Yes |
| mom | Momentum | Yes |
| trix | Triple Exponential Average | Yes |

### Trend (10 indicators)
| Name | Description | TA-Lib |
|------|-------------|--------|
| sma | Simple Moving Average | Yes |
| ema | Exponential Moving Average | Yes |
| wma | Weighted Moving Average | Yes |
| dema | Double EMA | Yes |
| tema | Triple EMA | Yes |
| kama | Kaufman Adaptive MA | Yes |
| t3 | Triple Exponential T3 | Yes |

### Volatility (15 measures)
| Name | Description |
|------|-------------|
| atr | Average True Range (TA-Lib) |
| natr | Normalized ATR (TA-Lib) |
| bollinger_bands | Bollinger Bands (TA-Lib) |
| yang_zhang | Yang-Zhang volatility (most efficient) |
| parkinson | Parkinson high-low volatility |
| garman_klass | Garman-Klass OHLC volatility |
| rogers_satchell | Rogers-Satchell drift-adjusted |
| realized | Realized volatility (squared returns) |
| ewma | EWMA volatility |
| garch | GARCH(1,1) forecast |

### Microstructure (12 metrics)
| Name | Description |
|------|-------------|
| kyle_lambda | Kyle's Lambda (price impact) |
| amihud_illiquidity | Amihud illiquidity ratio |
| vpin | Volume-Synchronized PIN |
| roll_spread | Roll implied spread |
| corwin_schultz | Corwin-Schultz high-low spread |
| hasbrouck_lambda | Hasbrouck's Lambda |

### ML Features (11)
| Name | Description |
|------|-------------|
| fractional_diff | Fractionally differenced series |
| lag | Lagged values |
| rolling_stats | Rolling statistics (mean, std, skew, kurt) |
| entropy | Shannon entropy |
| hurst | Hurst exponent |
| autocorr | Autocorrelation |

## Common Patterns

### Pattern: Complete Feature Pipeline

```python
import polars as pl
from ml4t.engineer import compute_features
from ml4t.engineer.labeling import triple_barrier_labels

# 1. Load data
df = pl.read_parquet("ohlcv.parquet")

# 2. Compute features
features = compute_features(df, [
    "rsi", "macd", "atr", "obv",
    {"name": "sma", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
])

# 3. Create labels
labels = triple_barrier_labels(
    df,
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=0.01,  # 1% stop loss
    max_holding=20,       # 20 bar horizon
)

# 4. Combine for ML
dataset = features.with_columns(labels.alias("label"))
```

### Pattern: Alternative Bar Sampling

```python
from ml4t.engineer.bars import volume_bars, dollar_bars, tick_imbalance_bars

# Volume bars (equal volume per bar)
vol_bars = volume_bars(ticks_df, volume_threshold=1000)

# Dollar bars (equal dollar volume per bar)
dollar_bars = dollar_bars(ticks_df, dollar_threshold=1_000_000)

# Tick imbalance bars (information-driven)
imb_bars = tick_imbalance_bars(ticks_df, expected_imbalance=100)
```

### Pattern: Triple-Barrier Labeling

```python
from ml4t.engineer.labeling import triple_barrier_labels, atr_barriers

# Fixed barriers
labels = triple_barrier_labels(
    df,
    upper_barrier=0.02,
    lower_barrier=0.01,
    max_holding=20,
)

# ATR-based dynamic barriers
labels = atr_barriers(
    df,
    atr_period=14,
    upper_multiplier=2.0,
    lower_multiplier=1.0,
    max_holding=20,
)
```

### Pattern: Fractional Differencing

```python
from ml4t.engineer.features.fdiff import fractional_diff

# Preserve stationarity while keeping memory
fd_close = fractional_diff(df["close"], d=0.4)

# Find optimal d via ADF test
from ml4t.engineer.features.fdiff import find_min_d
optimal_d = find_min_d(df["close"], threshold=0.05)
```

## Configuration

### YAML Config Format

```yaml
# features.yaml
features:
  - name: rsi
    params:
      period: 14
  - name: macd
    params:
      fast: 12
      slow: 26
      signal: 9
  - name: bollinger_bands
    params:
      period: 20
      std_dev: 2.0
```

### Preprocessing

```python
from ml4t.engineer import Preprocessor, StandardScaler, RobustScaler

# Leakage-safe preprocessing
preprocessor = Preprocessor([
    StandardScaler(),  # or RobustScaler() for outliers
])

# Fit on train, transform both
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Feature 'xyz' not found` | Unknown feature name | Use `list_features()` to see available |
| `ValueError: DataFrame missing required columns` | OHLCV columns missing | Ensure columns: open, high, low, close, volume |
| `InsufficientDataError` | Not enough data for lookback | Provide more historical data |
| `InvalidParameterError` | Invalid parameter value | Check `describe_feature()` for valid ranges |

## What NOT To Do

- DON'T compute features on train+test together (causes leakage)
- DON'T use future data in feature computation
- DON'T ignore the warmup period (first N bars are NaN)
- DON'T use pandas when Polars is available (10-100x slower)
- DON'T create features without understanding their lookback requirements

## Performance Notes

- **Speed**: 10-100x faster than pandas equivalents, ~0.8x TA-Lib C
- **TA-Lib Validation**: 59 indicators validated at 1e-6 tolerance
- **Labeling**: 50,000 labels/second (triple-barrier)
- **Memory**: Lazy evaluation with Polars minimizes memory

## Integration with ML4T Libraries

```python
# Complete workflow: data -> engineer -> diagnostic -> backtest
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.engineer.labeling import triple_barrier_labels
from ml4t.diagnostic import Evaluator
from ml4t.diagnostic.splitters import CombinatorialPurgedCV
from ml4t.backtest import Engine, Strategy

# 1. Load data
data = DataManager(config).fetch("SPY", "2020-01-01", "2023-12-31")

# 2. Engineer features
features = compute_features(data, ["rsi", "macd", "atr"])

# 3. Create labels
labels = triple_barrier_labels(data, upper=0.02, lower=0.01, horizon=20)

# 4. Evaluate with proper CV
cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)
results = Evaluator(splitter=cv).evaluate(model, features, labels)

# 5. Backtest
engine = Engine(data, MyStrategy(model), config)
backtest_results = engine.run()
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- López de Prado, M. (2020). *Machine Learning for Asset Managers*
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity"
