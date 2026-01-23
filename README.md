# ml4t-engineer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-performance feature engineering for financial machine learning.**

ml4t-engineer provides 120 technical indicators, triple-barrier labeling, and alternative bar sampling. Built on Polars with Numba JIT compilation, achieving performance comparable to TA-Lib's C implementation.

## Features

- **120 Technical Indicators**: Momentum, trend, volatility, microstructure, and more
- **TA-Lib Validated**: 60 indicators validated against TA-Lib at 1e-6 tolerance
- **Triple-Barrier Labeling**: AFML-compliant labeling with fixed or ATR-based barriers
- **Alternative Bars**: Volume, dollar, tick, and imbalance bar sampling
- **Microstructure Metrics**: Kyle's Lambda, VPIN, Amihud illiquidity, Roll spread
- **ML-Specific Features**: Fractional differencing, entropy, Hurst exponent
- **Polars + Numba**: Optimized for large datasets with lazy evaluation support

## Installation

```bash
pip install ml4t-engineer
```

With optional dependencies:

```bash
pip install ml4t-engineer[ta]        # TA-Lib backend for validation
pip install ml4t-engineer[store]     # DuckDB feature store
pip install ml4t-engineer[viz]       # Plotting (matplotlib, plotly)
pip install ml4t-engineer[calendars] # Trading calendars
pip install ml4t-engineer[all]       # All optional dependencies
```

## Quick Start

```python
import polars as pl
from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry

# See available features
registry = get_registry()
print(registry.list_all())                    # All 120 features
print(registry.list_by_category("momentum"))  # 31 momentum indicators

# Load OHLCV data
df = pl.read_parquet("ohlcv.parquet")

# Compute features with default parameters
result = compute_features(df, ["rsi", "macd", "atr", "obv"])

# Or with custom parameters
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
    {"name": "bollinger_bands", "params": {"period": 20, "std_dev": 2.0}},
])
```

## Feature Categories

| Category        | Count | Examples                                      |
|-----------------|-------|-----------------------------------------------|
| Momentum        | 31    | RSI, MACD, Stochastic, CCI, ADX, MFI          |
| Microstructure  | 15    | Kyle Lambda, VPIN, Amihud, Roll spread        |
| Volatility      | 15    | ATR, Bollinger, Yang-Zhang, Parkinson, GARCH  |
| Statistics      | 14    | Variance, Linear Regression, Correlation      |
| ML              | 14    | Fractional Diff, Entropy, Lag features        |
| Trend           | 10    | SMA, EMA, WMA, DEMA, TEMA, KAMA               |
| Risk            | 6     | Max Drawdown, Sortino, CVaR, Tail Ratio       |
| Price Transform | 5     | Typical Price, Weighted Close, Median Price   |
| Regime          | 4     | Hurst Exponent, Choppiness Index              |
| Volume          | 3     | OBV, AD, ADOSC                                |
| Math            | 3     | MAX, MIN, SUM                                 |

## Triple-Barrier Labeling

```python
from ml4t.engineer.labeling import triple_barrier_labels, atr_triple_barrier_labels

# Fixed barriers
labels = triple_barrier_labels(
    df,
    upper_barrier=0.02,    # 2% profit target
    lower_barrier=0.01,    # 1% stop loss
    max_holding_period=20, # 20 bar horizon
)

# Dynamic ATR-based barriers
labels = atr_triple_barrier_labels(
    df,
    atr_period=14,
    upper_atr_mult=2.0,    # 2x ATR profit target
    lower_atr_mult=1.0,    # 1x ATR stop loss
    max_holding_period=20,
)

# Time-based horizons (new in 0.1.0a7)
labels = triple_barrier_labels(
    df,
    upper_barrier=0.02,
    lower_barrier=0.01,
    max_holding_period="4h",  # 4 hours instead of bar count
)
```

## Alternative Bar Sampling

```python
from ml4t.engineer.bars import VolumeBarSampler, DollarBarSampler, TickImbalanceBarSampler

# Volume bars (equal volume per bar)
sampler = VolumeBarSampler(volume_threshold=1000)
vbars = sampler.sample(tick_data)

# Dollar bars (equal dollar volume per bar)
sampler = DollarBarSampler(dollar_threshold=1_000_000)
dbars = sampler.sample(tick_data)

# Tick imbalance bars (information-driven)
sampler = TickImbalanceBarSampler(expected_imbalance=100)
ibars = sampler.sample(tick_data)
```

## Preprocessing

```python
from ml4t.engineer import Preprocessor, StandardScaler, RobustScaler

# Leakage-safe preprocessing pipeline
preprocessor = Preprocessor([
    StandardScaler(),
])

# Fit on training data only, transform both
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
```

## Configuration via YAML

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

```python
result = compute_features(df, "features.yaml")
```

## Performance

Performance is comparable to TA-Lib's C implementation:

| Indicator | 100K rows | vs TA-Lib |
|-----------|-----------|-----------|
| RSI       | 0.4ms     | ~1x       |
| MACD      | 0.8ms     | ~1x       |
| SMA       | 0.2ms     | ~1x       |

For 1M rows, RSI computes in ~5ms. The implementation uses Numba JIT compilation for numerical operations and Polars for data handling.

*Benchmarks on Linux with Intel i7. Times may vary based on hardware and JIT warm-up.*

## ML-Ready Features

33 features produce bounded outputs suitable for direct ML consumption without preprocessing:

```python
# Get normalized (ML-ready) features
normalized = registry.list_normalized()
# Returns: ['rsi', 'stochastic', 'mfi', 'willr', 'cci', 'adx', ...]

# Check if a feature is normalized
meta = registry.get("rsi")
print(meta.normalized)  # True (0-100 range)
```

See [ML-Readiness Guide](docs/user-guide/ml-readiness.md) for the complete list.

## API Reference

### Core Functions

```python
from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry

# Registry for feature discovery
registry = get_registry()
registry.list_all()                    # All features
registry.list_by_category("momentum")  # By category
registry.list_ta_lib_compatible()      # TA-Lib validated
registry.list_normalized()             # ML-ready features
registry.get("rsi")                    # Feature metadata
```

### Labeling

```python
from ml4t.engineer.labeling import (
    triple_barrier_labels,        # Triple-barrier method
    atr_triple_barrier_labels,    # ATR-based barriers
    fixed_time_horizon_labels,    # Simple forward returns
    rolling_percentile_binary_labels,  # Adaptive thresholds
    meta_labels,                  # Meta-labeling
    calculate_sample_weights,     # Sample uniqueness weights
    sequential_bootstrap,         # Sequential bootstrap
)
```

### Bars

```python
from ml4t.engineer.bars import (
    VolumeBarSampler,             # Volume bars
    DollarBarSampler,             # Dollar bars
    TickImbalanceBarSampler,      # Tick imbalance bars
    VolumeRunBarSampler,          # Volume run bars
)
```

### Preprocessing

```python
from ml4t.engineer import (
    Preprocessor,      # Preprocessing pipeline
    StandardScaler,    # Z-score normalization
    RobustScaler,      # Robust scaling (median/IQR)
    MinMaxScaler,      # Min-max scaling
)
```

## Integration with ML4T Libraries

ml4t-engineer is part of the ML4T library ecosystem:

```python
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.engineer.labeling import triple_barrier_labels
from ml4t.diagnostic.evaluation import FeatureOutcome  # Feature analysis
from ml4t.backtest import Engine

# Complete workflow
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")
features = compute_features(data, ["rsi", "macd", "atr"])
labels = triple_barrier_labels(data, 0.02, 0.01, 20)
# ... train model, evaluate with diagnostic, backtest
```

## Validation

The library includes comprehensive validation:

- **60 TA-Lib Compatible Features**: Validated at 1e-6 tolerance against TA-Lib C library
- **Labeling Validation**: Triple-barrier and sample weights verified against AFML formulas and mlfinpy
- **3,000+ Unit Tests**: Covering edge cases and numerical accuracy

## Development

```bash
# Clone repository
git clone https://github.com/applied-ai/ml4t-engineer.git
cd ml4t-engineer

# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -q

# Type checking
uv run ty check

# Linting
uv run ruff check src/
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World."

## License

MIT License - see [LICENSE](LICENSE) for details.
