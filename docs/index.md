# ML4T Engineer

High-performance feature engineering library for financial machine learning.

## Overview

ML4T Engineer provides **107+ technical indicators**, triple-barrier labeling, and alternative bar sampling with a Polars-first implementation that's 10-100x faster than pandas alternatives.

## Key Features

- **107+ Technical Indicators** across 10 categories (momentum, trend, volatility, etc.)
- **59 TA-Lib Validated** indicators with 1e-6 tolerance matching
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
| Momentum | 31 | RSI, MACD, Stochastic, CCI, ADX |
| Trend | 10 | SMA, EMA, KAMA, T3 |
| Volatility | 15 | ATR, Bollinger, Yang-Zhang, GARCH |
| Volume | 3 | OBV, AD, CMF |
| Statistics | 8 | Rolling stats, Z-score |
| Microstructure | 12 | Kyle's Lambda, VPIN, Amihud |
| ML Features | 11 | Fractional diff, Hurst, Entropy |

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
