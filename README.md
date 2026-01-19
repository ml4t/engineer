# ml4t-engineer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-performance feature engineering for financial machine learning.**

ml4t-engineer provides 107+ technical indicators, triple-barrier labeling, and alternative bar sampling with a Polars-first implementation that's 10-100x faster than pandas alternatives.

## Features

- **107+ Technical Indicators**: Momentum, trend, volatility, volume, and more
- **TA-Lib Validated**: 59 indicators validated against TA-Lib at 1e-6 tolerance
- **Triple-Barrier Labeling**: AFML-compliant labeling with ATR-based barriers
- **Alternative Bars**: Volume, dollar, tick, and imbalance bars
- **Microstructure Metrics**: Kyle's Lambda, VPIN, Amihud, Roll spread
- **ML-Specific Features**: Fractional differencing, entropy, Hurst exponent
- **Polars-First**: 10-100x faster than pandas, ~0.8x TA-Lib C speed
- **Type-Safe**: Type hints throughout

## Installation

```bash
pip install ml4t-engineer
```

With optional dependencies:

```bash
pip install ml4t-engineer[talib]      # TA-Lib backend
pip install ml4t-engineer[numba]      # Numba acceleration
pip install ml4t-engineer[all]        # All optional dependencies
```

## Quick Start

```python
import polars as pl
from ml4t.engineer import compute_features, list_features

# See available features
print(list_features("momentum"))  # RSI, MACD, Stochastic, etc.

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

| Category | Count | Examples |
|----------|-------|----------|
| Momentum | 31 | RSI, MACD, Stochastic, CCI, ADX, MFI |
| Trend | 10 | SMA, EMA, WMA, DEMA, TEMA, KAMA |
| Volatility | 15 | ATR, Bollinger, Yang-Zhang, GARCH |
| Volume | 3 | OBV, AD, ADOSC |
| Statistics | 8 | Variance, Linear Regression, Correlation |
| Math | 3 | MAX, MIN, SUM |
| Price Transform | 5 | Typical Price, Weighted Close |
| Microstructure | 12 | Kyle Lambda, VPIN, Amihud, Roll |
| ML | 11 | Fractional Diff, Entropy, Hurst |

## Triple-Barrier Labeling

```python
from ml4t.engineer.labeling import triple_barrier_labels, atr_barriers

# Fixed barriers
labels = triple_barrier_labels(
    df,
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=0.01,  # 1% stop loss
    max_holding=20,       # 20 bar horizon
)

# Dynamic ATR-based barriers
labels = atr_barriers(
    df,
    atr_period=14,
    upper_multiplier=2.0,  # 2x ATR profit target
    lower_multiplier=1.0,  # 1x ATR stop loss
    max_holding=20,
)
```

## Alternative Bar Sampling

```python
from ml4t.engineer.bars import volume_bars, dollar_bars, tick_imbalance_bars

# Volume bars (equal volume per bar)
vbars = volume_bars(tick_data, volume_threshold=1000)

# Dollar bars (equal dollar volume per bar)
dbars = dollar_bars(tick_data, dollar_threshold=1_000_000)

# Tick imbalance bars (information-driven)
ibars = tick_imbalance_bars(tick_data, expected_imbalance=100)
```

## Preprocessing

```python
from ml4t.engineer import Preprocessor, StandardScaler, RobustScaler

# Leakage-safe preprocessing
preprocessor = Preprocessor([
    StandardScaler(),
])

# Fit on train only, transform both
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

| Benchmark | ml4t-engineer | pandas-ta | Speedup |
|-----------|---------------|-----------|---------|
| RSI (1M rows) | 12ms | 850ms | 70x |
| MACD (1M rows) | 18ms | 1200ms | 67x |
| Bollinger (1M rows) | 15ms | 920ms | 61x |
| Triple-barrier (1M rows) | 20ms | N/A | - |

*Benchmarks on M1 MacBook Pro with Polars 0.20+*

## API Reference

### Core Functions

```python
from ml4t.engineer import (
    compute_features,   # Compute features from config
    list_features,      # List available features
    list_categories,    # List feature categories
    describe_feature,   # Get feature metadata
)
```

### Labeling

```python
from ml4t.engineer.labeling import (
    triple_barrier_labels,  # Triple-barrier method
    atr_barriers,           # ATR-based barriers
    meta_labels,            # Meta-labeling
)
```

### Bars

```python
from ml4t.engineer.bars import (
    volume_bars,           # Volume bars
    dollar_bars,           # Dollar bars
    tick_imbalance_bars,   # Tick imbalance bars
    volume_imbalance_bars, # Volume imbalance bars
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
from ml4t.diagnostic import Evaluator
from ml4t.backtest import Engine

# Complete workflow
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")
features = compute_features(data, ["rsi", "macd", "atr"])
labels = triple_barrier_labels(data, 0.02, 0.01, 20)
# ... train model, evaluate, backtest
```

## Development

```bash
# Clone repository
git clone https://github.com/applied-ai/ml4t-engineer.git
cd ml4t-engineer

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Type checking
uv run ty check src/

# Linting
uv run ruff check src/
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest tests/ --cov=ml4t.engineer

# TA-Lib validation tests (requires TA-Lib)
uv run pytest tests/test_talib_validation.py
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World."

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.
