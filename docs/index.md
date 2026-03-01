# ML4T Engineer

High-performance feature engineering library for financial machine learning.

## Overview

ML4T Engineer provides **120 technical indicators**, triple-barrier labeling, and alternative bar sampling with a Polars-first implementation that's 10-100x faster than pandas alternatives.

## Key Features

- **120 Technical Indicators** across 11 categories (momentum, trend, volatility, etc.)
- **60 TA-Lib Validated** indicators with 1e-6 tolerance matching
- **Triple-Barrier Labeling** at 50,000 labels/second
- **Alternative Bar Sampling** (volume, dollar, tick imbalance bars)
- **Polars-First** implementation for maximum performance

## Quick Example

```python
import polars as pl
from ml4t.engineer import compute_features

# Load OHLCV data
df = pl.read_parquet("ohlcv.parquet")

# Compute features with a single function call
result = compute_features(df, ["rsi", "macd", "atr", "bollinger_bands"])
```

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Momentum | 31 | RSI, MACD, Stochastic, CCI, ADX, MFI |
| Microstructure | 15 | Kyle Lambda, VPIN, Amihud, Roll spread |
| Volatility | 15 | ATR, Bollinger, Yang-Zhang, Parkinson |
| Statistics | 14 | Variance, Linear Regression, Correlation |
| ML | 14 | Fractional Diff, Entropy, Lag features |
| Trend | 10 | SMA, EMA, WMA, DEMA, TEMA, KAMA |
| Risk | 6 | Max Drawdown, Sortino, CVaR |
| Price Transform | 5 | Typical Price, Weighted Close |
| Regime | 4 | Hurst Exponent, Choppiness Index |
| Volume | 3 | OBV, AD, ADOSC |
| Math | 3 | MAX, MIN, SUM |

## Installation

```bash
pip install ml4t-engineer
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed setup instructions
- [Quickstart](getting-started/quickstart.md) - Get running in 5 minutes
- [API Reference](api/index.md) - Complete API documentation

## Part of the ML4T Library Suite

ML4T Engineer integrates seamlessly with other ML4T libraries:

```
ml4t-data → ml4t-engineer → ml4t-diagnostic → ml4t-backtest → ml4t-live
```
