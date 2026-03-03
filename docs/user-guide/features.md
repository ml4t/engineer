# Technical Indicators

ML4T Engineer provides 120 technical indicators across 11 categories, built on Polars with Numba JIT for performance-critical kernels. 59 indicators are validated against TA-Lib at 1e-6 tolerance.

## Overview

| Category | Count | TA-Lib | Key Indicators |
|----------|-------|--------|----------------|
| Momentum | 31 | 19 | RSI, MACD, Stochastic, CCI, ADX, MFI |
| Volatility | 15 | 4 | ATR, Bollinger, Yang-Zhang, Parkinson, GARCH |
| Microstructure | 15 | 0 | Kyle Lambda, Amihud, Roll Spread, Realized Spread |
| Trend | 10 | 9 | SMA, EMA, WMA, DEMA, TEMA, KAMA |
| ML Features | 14 | 0 | Lag, Entropy, Fourier, Cyclical Encode |
| Statistics | 14 | 7 | STDDEV, Linear Regression, TSF, Variance Ratio |
| Risk | 6 | 0 | Max Drawdown, Downside Deviation, Sharpe, Sortino |
| Price Transform | 5 | 5 | Typical Price, Weighted Close, Average Price |
| Regime | 4 | 0 | Hurst Exponent, Choppiness, Fractal Efficiency |
| Volume | 3 | 3 | OBV, AD, ADOSC |
| Math | 3 | 3 | Maximum, Minimum, Summation |
| Cross-Asset | 10 | 0 | Beta, Correlation, Cointegration (standalone functions) |

> **Book**: *ML for Trading, 3rd ed.* — Ch8 notebooks (`01_price_volume_features.py` through `04_fundamentals_macro_calendar.py`) build features manually to explain the economics. Case studies (ETFs, US Equities Panel, CME Futures) then use `compute_features()` in production pipelines.

## Computation API

`compute_features()` accepts three input formats:

```python
from ml4t.engineer import compute_features

# 1. List of names (default parameters)
result = compute_features(df, ["rsi", "macd", "atr"])

# 2. List of dicts (custom parameters)
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
    {"name": "bollinger_bands", "params": {"period": 20, "std_dev": 2.5}},
])

# 3. YAML config file (production pipelines)
result = compute_features(df, "features.yaml")
```

Features are computed in dependency order (topological sort). Circular dependencies raise `ValueError`. The return type matches the input: `DataFrame` in, `DataFrame` out; `LazyFrame` in, `LazyFrame` out.

> **Book**: Ch7 `10_ml4t_library_ecosystem.py` demonstrates all three input formats on SPY data, including a comparison between library and manual RSI implementations.

## Category Reference

### Momentum (31 indicators)

Price momentum and oscillator indicators. Most produce bounded (normalized) output suitable for direct ML use.

| Name | Description | TA-Lib | Normalized | Default Period |
|------|-------------|--------|------------|----------------|
| `rsi` | Relative Strength Index | Yes | 0-100 | 14 |
| `macd` | Moving Average Convergence/Divergence | Yes | No | 12/26/9 |
| `stochastic` | Stochastic Oscillator (%K, %D) | No | 0-100 | 14/3/3 |
| `stochf` | Fast Stochastic | Yes | 0-100 | 5/3 |
| `stochrsi` | Stochastic RSI | Yes | 0-100 | 14 |
| `cci` | Commodity Channel Index | Yes | ~-200 to 200 | 14 |
| `willr` | Williams %R | Yes | -100 to 0 | 14 |
| `adx` | Average Directional Index | Yes | 0-100 | 14 |
| `adxr` | ADX Rating | Yes | 0-100 | 14 |
| `dx` | Directional Movement Index | Yes | 0-100 | 14 |
| `plus_di` | Plus Directional Indicator | Yes | 0-100 | 14 |
| `minus_di` | Minus Directional Indicator | Yes | 0-100 | 14 |
| `mfi` | Money Flow Index | Yes | 0-100 | 14 |
| `roc` | Rate of Change | Yes | No | 10 |
| `rocp` | Rate of Change (%) | Yes | No | 10 |
| `mom` | Momentum | Yes | No | 10 |
| `trix` | Triple Exponential Average | Yes | No | 30 |
| `cmo` | Chande Momentum Oscillator | Yes | -100 to 100 | 14 |
| `ultosc` | Ultimate Oscillator | Yes | 0-100 | 7/14/28 |
| `bop` | Balance of Power | Yes | -1 to 1 | — |
| `imi` | Intraday Momentum Index | No | 0-100 | 14 |
| `aroon` | Aroon (up/down) | Yes | 0-100 | 14 |
| `aroonosc` | Aroon Oscillator | Yes | -100 to 100 | 14 |
| `apo` | Absolute Price Oscillator | Yes | No | 12/26 |
| `ppo` | Percentage Price Oscillator | Yes | No | 12/26 |
| `sar` | Parabolic SAR | Yes | No | 0.02/0.2 |

> **Book**: Ch8 `01_price_volume_features.py` constructs momentum indicators on ETF data, explaining the economic rationale for each. ETFs and US Equities Panel case studies use these in `03_features.py`.

### Trend (10 indicators)

Moving averages that produce price-scale outputs. Require preprocessing for ML models.

| Name | Description | TA-Lib | Default Period |
|------|-------------|--------|----------------|
| `sma` | Simple Moving Average | Yes | 20 |
| `ema` | Exponential Moving Average | Yes | 20 |
| `wma` | Weighted Moving Average | Yes | 20 |
| `dema` | Double Exponential MA | Yes | 20 |
| `tema` | Triple Exponential MA | Yes | 20 |
| `t3` | Triple Exponential T3 | Yes | 5 |
| `kama` | Kaufman Adaptive MA | Yes | 30 |
| `trima` | Triangular MA | Yes | 20 |
| `midpoint` | Midpoint over period | Yes | 14 |
| `donchian_channels` | Donchian Channels (highest high/lowest low) | No | 20 |

### Volatility (15 indicators)

Volatility estimators ranging from simple (ATR) to advanced (GARCH). Includes range-based estimators that are more efficient than close-to-close.

| Name | Description | TA-Lib | Normalized |
|------|-------------|--------|------------|
| `atr` | Average True Range | Yes | No |
| `natr` | Normalized ATR (% of price) | Yes | 0-100 |
| `trange` | True Range | Yes | No |
| `bollinger_bands` | Bollinger Bands (upper/middle/lower) | Yes | No |
| `yang_zhang_volatility` | Yang-Zhang (overnight + intraday) | No | No |
| `parkinson_volatility` | Parkinson range-based | No | No |
| `garman_klass_volatility` | Garman-Klass OHLC-based | No | No |
| `rogers_satchell_volatility` | Rogers-Satchell drift-independent | No | No |
| `realized_volatility` | Standard deviation of returns | No | No |
| `ewma_volatility` | EWMA of variance | No | No |
| `garch_forecast` | GARCH(1,1) conditional volatility | No | No |
| `conditional_volatility_ratio` | Up-market vs down-market vol ratio | No | No |
| `volatility_percentile_rank` | Current vol vs historical distribution | No | 0-100 |
| `volatility_of_volatility` | Second-order volatility measure | No | No |
| `volatility_regime_probability` | High/low vol regime probability | No | No |

**Efficiency ranking**: Yang-Zhang > Garman-Klass ~ Rogers-Satchell > Parkinson > Close-to-Close. See Molnar (2012) for theoretical efficiency ratios.

> **Book**: Ch9 `08_garch_volatility.py` and `09_har_rough_volatility.py` compare volatility estimators on real data. Ch8 `01_price_volume_features.py` covers range-based estimators with efficiency analysis.

### Microstructure (15 indicators)

Market microstructure features from De Prado (2018) and empirical market microstructure literature.

| Name | Description |
|------|-------------|
| `kyle_lambda` | Kyle's Lambda (price impact coefficient) |
| `amihud_illiquidity` | Amihud illiquidity ratio |
| `roll_spread_estimator` | Roll implied bid-ask spread |
| `realized_spread` | Realized spread |
| `effective_tick_rule` | Effective tick rule classification |
| `order_flow_imbalance` | Order flow imbalance |
| `price_impact_ratio` | Price impact ratio |
| `volume_weighted_price_momentum` | Volume-weighted price momentum |
| `bid_ask_imbalance` | Bid-ask imbalance (normalized -1 to 1) |
| `book_depth_ratio` | Book depth ratio (normalized 0 to 1) |
| `quote_stuffing_indicator` | Quote stuffing detection |
| `trade_intensity` | Trade intensity |
| `volume_at_price_ratio` | Volume at price ratio |
| `volume_synchronicity` | Volume synchronicity |
| `weighted_mid_price` | Weighted mid price |

> **Book**: Ch8 `02_microstructure_features.py` builds microstructure features from tick and minute data. The NASDAQ-100 Microstructure case study (`03_features.py`) implements Kyle's Lambda, Amihud, and VPIN manually for pedagogical purposes — the ml4t-engineer implementations are production-ready equivalents.

### ML Features (14 indicators)

Features designed specifically for machine learning pipelines.

| Name | Description | Normalized |
|------|-------------|------------|
| `create_lag_features` | Multiple lag columns at once | No |
| `cyclical_encode` | Cyclical time encoding (sin/cos) | No |
| `fourier_features` | Fourier transform features | No |
| `rolling_entropy` | Shannon entropy | 0-10 |
| `rolling_entropy_lz` | Lempel-Ziv entropy | 0-10 |
| `rolling_entropy_plugin` | Plugin entropy estimator | 0-10 |
| `percentile_rank_features` | Rank-based normalization | 0-100 |
| `interaction_features` | Feature interaction terms | No |
| `multi_horizon_returns` | Returns at multiple horizons | No |
| `directional_targets` | Directional movement targets | No |
| `volatility_adjusted_returns` | Returns scaled by volatility | No |
| `regime_conditional_features` | Regime-conditional transforms | No |
| `time_decay_weights` | Exponential time decay | No |
| `ffdiff` | Fractional differencing | No |

> **Book**: Ch8 `04_fundamentals_macro_calendar.py` covers feature construction patterns including lag features and calendar encodings.

### Risk (6 indicators)

Risk and risk-adjusted return metrics.

| Name | Description | Normalized |
|------|-------------|------------|
| `maximum_drawdown` | Maximum drawdown | No |
| `downside_deviation` | Downside volatility | 0-2 |
| `tail_ratio` | Right tail / left tail ratio | 0-10 |
| `higher_moments` | Skewness and kurtosis | No |
| `risk_adjusted_returns` | Sharpe, Sortino, Calmar, Omega | No |
| `ulcer_index` | Ulcer Index (drawdown-based risk) | No |

### Cross-Asset (10 functions)

Multi-asset relationship features. These are standalone functions in `ml4t.engineer.features.cross_asset` rather than registry entries, since they require two or more price series as input.

| Function | Description |
|----------|-------------|
| `rolling_correlation` | Rolling Pearson correlation |
| `beta_to_market` | Rolling beta vs market index |
| `correlation_regime_indicator` | Low/medium/high correlation regimes |
| `lead_lag_correlation` | Lead-lag cross-correlation |
| `multi_asset_dispersion` | Cross-sectional return dispersion |
| `correlation_matrix_features` | Mean/min/max of correlation matrix |
| `relative_strength_index_spread` | RSI spread between two assets |
| `volatility_ratio` | Volatility ratio between assets |
| `co_integration_score` | Rolling cointegration score |
| `cross_asset_momentum` | Rank-based cross-asset momentum |

These are called directly (not via `compute_features`) since they require multi-asset DataFrames.

> **Book**: Ch8 `03_structural_cross_instrument_features.py` constructs cross-asset features. Ch9 `14_panel_features.py` applies cross-sectional features to equity panels.

### Regime (4 indicators)

Market regime detection features. All produce bounded outputs suitable for direct ML use.

| Name | Description | Range |
|------|-------------|-------|
| `hurst_exponent` | Hurst exponent (R/S analysis) | 0-1 |
| `choppiness_index` | Market choppiness | 0-100 |
| `fractal_efficiency` | Price path efficiency | 0-1 |
| `trend_intensity_index` | Trend strength | 0-100 |

> **Book**: Ch9 `11_hmm_regimes.py` and `13_regime_as_feature.py` apply regime detection to equity indices.

### Statistics (14 indicators)

Statistical features including TA-Lib standard and rolling distribution metrics.

| Name | Description | TA-Lib | Normalized |
|------|-------------|--------|------------|
| `stddev` | Standard Deviation | Yes | No |
| `var` | Variance | Yes | No |
| `avgdev` | Average Deviation | No | No |
| `linearreg` | Linear Regression Value | Yes | No |
| `linearreg_slope` | Linear Regression Slope | Yes | No |
| `linearreg_angle` | Linear Regression Angle | Yes | No |
| `linearreg_intercept` | Linear Regression Intercept | Yes | No |
| `tsf` | Time Series Forecast | Yes | No |
| `coefficient_of_variation` | Rolling coefficient of variation | No | 0-10 |
| `variance_ratio` | Variance ratio test | No | 0-5 |
| `rolling_cv_zscore` | Cross-validated z-score | No | -10 to 10 |
| `rolling_drift` | Rolling drift estimate | No | -10 to 10 |
| `rolling_kl_divergence` | KL divergence vs reference | No | 0-10 |
| `rolling_wasserstein` | Wasserstein distance | No | No |

### Price Transform (5 indicators)

| Name | Description | TA-Lib |
|------|-------------|--------|
| `avgprice` | Average Price (O+H+L+C)/4 | Yes |
| `typprice` | Typical Price (H+L+C)/3 | Yes |
| `medprice` | Median Price (H+L)/2 | Yes |
| `wclprice` | Weighted Close (H+L+2C)/4 | Yes |
| `midprice` | Midpoint Price (H+L)/2 | Yes |

### Volume (3 indicators)

| Name | Description | TA-Lib |
|------|-------------|--------|
| `obv` | On Balance Volume | Yes |
| `ad` | Accumulation/Distribution | Yes |
| `adosc` | A/D Oscillator | Yes |

> **Book**: ETFs case study `03_features.py` uses volume features in a multi-asset pipeline alongside momentum and volatility.

### Math (3 indicators)

O(n) rolling operations using monotonic deque.

| Name | Description | TA-Lib |
|------|-------------|--------|
| `maximum` | Rolling maximum | Yes |
| `minimum` | Rolling minimum | Yes |
| `summation` | Rolling sum | Yes |

### Fractional Differencing (4 functions)

See the dedicated [Fractional Differencing guide](fractional-differencing.md) for the full workflow.

## Feature Discovery

The `FeatureCatalog` provides filtering and full-text search over all 120 features:

```python
from ml4t.engineer import feature_catalog

# List by category
momentum = feature_catalog.list(category="momentum")

# Filter by multiple criteria
ml_ready = feature_catalog.list(normalized=True, ta_lib_compatible=True)

# Full-text search
results = feature_catalog.search("volatility estimator")
# Returns: [("parkinson_volatility", 0.65), ("garman_klass_volatility", 0.45), ...]

# Detailed feature info
info = feature_catalog.describe("yang_zhang_volatility")
# {'name': 'yang_zhang_volatility', 'category': 'volatility', ...}

# List all categories
print(feature_catalog.categories())
# ['cross_asset', 'math', 'microstructure', 'ml', 'momentum', ...]

# List all tags
print(feature_catalog.tags())
```

See the dedicated [Feature Discovery guide](discovery.md) for complete examples.

> **Book**: Ch7 `10_ml4t_library_ecosystem.py` explores the registry metadata for RSI, ATR, and Garman-Klass, then demonstrates `feature_catalog.search()` and filtered listing.

## YAML Configuration

For reproducible feature pipelines, store configurations in YAML:

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

  - name: yang_zhang_volatility
```

Load with `compute_features(df, "features.yaml")`. The YAML format supports version comments and parameter documentation inline.

## Input Requirements

### OHLCV DataFrame

Most features expect a DataFrame with standardized column names (lowercase):

| Column | Type | Required By |
|--------|------|-------------|
| `open` | float | OHLCV, OHLC features |
| `high` | float | OHLCV, OHLC, HLC, HL features |
| `low` | float | OHLCV, OHLC, HLC, HL features |
| `close` | float | All features |
| `volume` | float | OHLCV, volume features |
| `returns` | float | Return-based features (auto-computed if missing) |

Features declare their `input_type` metadata (e.g., `"OHLCV"`, `"close"`, `"returns"`), and `compute_features` validates that required columns are present.

### Missing Columns

If a feature requires a column that's missing, `compute_features` raises a clear error:

```
ValueError: Feature 'mfi' requires column 'volume' (input_type='OHLCV') but it was not found.
```

## Custom Parameters

Override default parameters per feature:

```python
# Check defaults
from ml4t.engineer.core.registry import get_registry
meta = get_registry().get("rsi")
print(meta.parameters)  # {'period': 14}

# Override
result = compute_features(df, [{"name": "rsi", "params": {"period": 20}}])
```

Invalid parameters raise `ValueError` with the valid parameter names.

## Performance

- **Polars-native**: All computations use Polars expressions for automatic parallelism
- **Numba JIT**: Numerical kernels (volatility estimators, microstructure) are Numba-accelerated
- **Throughput**: ~480K indicator calculations/second, 11M rows/second streaming
- **TA-Lib parity**: RSI computed at ~1x TA-Lib speed via Polars native implementation
- **Dependency ordering**: `compute_features` resolves feature dependencies via topological sort

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Molnar, P. (2012). Properties of range-based volatility estimators. *International Review of Financial Analysis*.
