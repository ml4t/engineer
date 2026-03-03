# Labeling Methods

ML4T Engineer provides 7 labeling methods for supervised learning in finance, implementing the full workflow from *Advances in Financial Machine Learning* (De Prado, 2018).

## Overview

| Method | Function | Use Case |
|--------|----------|----------|
| Triple-barrier | `triple_barrier_labels()` | Fixed profit/loss targets with time limit |
| ATR-based barriers | `atr_triple_barrier_labels()` | Volatility-adjusted targets |
| Rolling percentile | `rolling_percentile_binary_labels()` | Adaptive threshold from return distribution |
| Fixed time horizon | `fixed_time_horizon_labels()` | Simple forward returns |
| Trend scanning | `trend_scanning_labels()` | Optimal horizon via t-statistic |
| Meta-labeling | `meta_labels()` + `compute_bet_size()` | Bet sizing for primary model |
| Calendar-aware | `calendar_aware_labels()` | Session-break handling for futures |

All methods return a Polars DataFrame with standardized output columns. Performance is ~50,000 labels/second via Numba-accelerated kernels.

> **Book**: *ML for Trading, 3rd ed.* — Ch7 `03_label_methods.py` walks through all 7 methods on real ETF data with visualizations. All case study `02_labels.py` notebooks apply these methods in production pipelines.

## Choosing a Method

```
Need fixed profit/loss targets?
├── Yes → Do barriers scale with volatility?
│   ├── Yes → atr_triple_barrier_labels()
│   └── No  → triple_barrier_labels()
├── No  → Need directional labels (long/short)?
│   ├── Yes → rolling_percentile_binary_labels()
│   └── No  → Need optimal holding period?
│       ├── Yes → trend_scanning_labels()
│       └── No  → fixed_time_horizon_labels()

Have a primary model? → meta_labels() for bet sizing
Trading futures with session breaks? → calendar_aware_labels()
```

## LabelingConfig

All barrier-based methods accept a `LabelingConfig` object created via factory methods. This provides serialization, validation, and a bridge to `DataContractConfig` for pipeline integration.

### Factory Methods

```python
from ml4t.engineer.config import LabelingConfig

# Fixed barriers
config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,       # 2% take profit
    lower_barrier=0.01,       # 1% stop loss
    max_holding_period=20,    # 20 bars (or "4h" for time-based)
    side=1,                   # 1=long, -1=short, 0=symmetric
    trailing_stop=False,      # Enable trailing stop loss
)

# ATR-based barriers
config = LabelingConfig.atr_barrier(
    atr_tp_multiple=2.0,      # 2x ATR take profit
    atr_sl_multiple=1.0,      # 1x ATR stop loss
    atr_period=14,
    max_holding_period=20,
)

# Fixed horizon (simple forward returns)
config = LabelingConfig.fixed_horizon(
    horizon=10,
    return_method="returns",  # "returns" | "log_returns" | "binary"
    threshold=None,
)

# Trend scanning
config = LabelingConfig.trend_scanning(
    min_horizon=5,
    max_horizon=20,
    t_value_threshold=2.0,
)
```

### Serialization

Store labeling configurations for experiment reproducibility:

```python
# Save to YAML
config.to_yaml("labeling_config.yaml")

# Reload
config = LabelingConfig.from_yaml("labeling_config.yaml")
```

## Triple-Barrier Labels

The triple-barrier method from AFML Chapter 3 creates labels based on which of three barriers is touched first: upper (profit target), lower (stop loss), or vertical (time limit).

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,       # 2% profit target
    lower_barrier=0.01,       # 1% stop loss
    max_holding_period=20,    # 20 bar horizon
    side=1,                   # Long-only signals
)

result = triple_barrier_labels(
    data=df,
    config=config,
    price_col="close",
    high_col="high",                  # For intrabar barrier touches
    low_col="low",                    # For intrabar barrier touches
    timestamp_col="timestamp",        # Required for time-based max_holding
    calculate_uniqueness=False,       # Compute sample weights
    uniqueness_weight_scheme="returns_uniqueness",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | OHLCV data |
| `config` | `LabelingConfig` | required | Barrier configuration |
| `price_col` | `str` | `"close"` | Price column for barrier calculations |
| `high_col` | `str \| None` | `None` | High column for intrabar touch detection |
| `low_col` | `str \| None` | `None` | Low column for intrabar touch detection |
| `timestamp_col` | `str \| None` | `None` | Required when `max_holding_period` is time-based |
| `calculate_uniqueness` | `bool` | `False` | Compute label uniqueness and sample weights |
| `uniqueness_weight_scheme` | `str` | `"returns_uniqueness"` | Weight scheme (see Sample Weighting) |

### Output Columns

| Column | Description |
|--------|-------------|
| `label` | +1 (upper hit), -1 (lower hit), 0 (vertical hit) |
| `label_time` | Timestamp when barrier was hit |
| `label_price` | Price at barrier touch |
| `label_return` | Return from entry to barrier |
| `label_bars` | Number of bars until barrier |
| `label_duration` | Time duration until barrier |
| `barrier_hit` | Which barrier: `"upper"`, `"lower"`, `"vertical"` |
| `label_uniqueness` | Average uniqueness (when `calculate_uniqueness=True`) |
| `sample_weight` | Sample weight (when `calculate_uniqueness=True`) |

### The `side` Parameter

Controls directional bias:

- `side=1`: Long-only. Upper barrier = profit, lower barrier = loss.
- `side=-1`: Short-only. Upper barrier = loss, lower barrier = profit. Labels are flipped.
- `side=0`: Symmetric. Both barriers treated equally. Label is +1 or -1 based on direction.

### Trailing Stop

Enable a trailing stop loss that ratchets up as price moves favorably:

```python
config = LabelingConfig.triple_barrier(
    upper_barrier=0.03,
    lower_barrier=0.01,
    max_holding_period=20,
    trailing_stop=True,       # Stop loss follows highest price
)
```

With `trailing_stop=True`, the lower barrier moves up as the trade moves in favor. This reduces the time spent in losing positions.

> **Book**: Ch7 `03_label_methods.py` applies triple-barrier labeling to SPY with visualization of barrier touches.

## ATR-Based Dynamic Barriers

Volatility-adjusted barriers adapt to changing market conditions. The barriers scale with the Average True Range (ATR), so they're wider in volatile markets and tighter in calm markets.

```python
from ml4t.engineer.labeling import atr_triple_barrier_labels

result = atr_triple_barrier_labels(
    data=df,
    atr_tp_multiple=2.0,      # Take profit at 2x ATR
    atr_sl_multiple=1.0,      # Stop loss at 1x ATR
    atr_period=14,             # ATR lookback
    max_holding_bars=20,       # Can also be "4h" for time-based
    side=1,
    price_col="close",
    timestamp_col="timestamp",
    trailing_stop=False,
)

# Or via LabelingConfig
config = LabelingConfig.atr_barrier(
    atr_tp_multiple=2.0,
    atr_sl_multiple=1.0,
    atr_period=14,
    max_holding_period=20,
)
result = atr_triple_barrier_labels(df, config=config)
```

### When to Use ATR Barriers

| Scenario | Recommendation |
|----------|---------------|
| Single asset, stable volatility | Fixed barriers sufficient |
| Multi-asset (different volatility levels) | ATR barriers adapt per asset |
| Regime changes (calm → volatile) | ATR barriers avoid premature stops |
| Futures with varying contract sizes | ATR normalizes across contracts |

> **Book**: CME Futures case study `02_labels.py` applies ATR barriers on ES, NQ, and CL futures with session-aware horizons.

## Rolling Percentile Labels

Adaptive labeling where thresholds are computed from the rolling return distribution. Produces binary long/short signals based on whether forward returns exceed a historical percentile.

```python
from ml4t.engineer.labeling import rolling_percentile_binary_labels

result = rolling_percentile_binary_labels(
    data=df,
    horizon=10,                # Forward return horizon (bars or "1h")
    percentile=95,             # 95th percentile for long signals
    direction="long",          # "long" or "short"
    lookback_window=252 * 24,  # ~1 year of hourly bars
    price_col="close",
    session_col=None,          # Session-aware forward returns
    min_samples=None,          # Minimum samples for percentile
    timestamp_col=None,        # Required for time-based horizons
    tolerance=None,            # E.g., "2m" for time-based horizons
)
```

### Output Columns

| Column | Example | Description |
|--------|---------|-------------|
| `forward_return_10` | 0.0123 | Forward return over horizon |
| `threshold_p95_h10` | 0.0089 | Rolling 95th percentile threshold |
| `label_long_p95_h10` | 1 | 1 if return exceeds threshold, 0 otherwise |

### Multiple Horizons and Percentiles

Generate labels for multiple combinations simultaneously:

```python
from ml4t.engineer.labeling import rolling_percentile_multi_labels

result = rolling_percentile_multi_labels(
    data=df,
    horizons=[5, 10, 20],
    percentiles=[90, 95],
    direction="long",
    lookback_window=252,
)
# Produces: label_long_p90_h5, label_long_p95_h5, label_long_p90_h10, ...
```

> **Book**: Ch7 `03_label_methods.py` compares rolling percentile labels against triple-barrier on SPY. ETFs case study `02_labels.py` uses percentile labels in its production pipeline.

## Fixed Time Horizon

The simplest labeling method: compute forward returns over a fixed horizon. Supports both bar-count and time-based horizons.

```python
from ml4t.engineer.labeling import fixed_time_horizon_labels

# Bar-based horizon
result = fixed_time_horizon_labels(
    data=df,
    horizon=10,           # 10 bars forward
    price_col="close",
)

# Time-based horizon
result = fixed_time_horizon_labels(
    data=df,
    horizon="1h",         # 1 hour forward
    timestamp_col="timestamp",
    price_col="close",
)
```

Output includes a `forward_return` column. For binary labels, use the `threshold` parameter or `rolling_percentile_binary_labels` for adaptive thresholds.

## Trend Scanning

De Prado's trend-scanning method finds the optimal holding period for each observation by fitting linear regressions over a range of horizons and selecting the one with the highest t-statistic.

```python
from ml4t.engineer.labeling import trend_scanning_labels

result = trend_scanning_labels(
    data=df,
    min_horizon=5,
    max_horizon=20,
    t_value_threshold=2.0,
    price_col="close",
)
```

Output includes `trend_label` (+1/-1/0), `optimal_horizon`, and `t_statistic`. Observations with |t-statistic| below the threshold receive label 0 (no trend).

> **Book**: Ch7 `03_label_methods.py` demonstrates trend scanning alongside triple-barrier and percentile methods, showing how the optimal horizon varies with market conditions.

## Meta-Labeling & Bet Sizing

Meta-labeling is a two-stage workflow (AFML Chapter 3):

1. A **primary model** generates directional signals (+1/-1/0)
2. A **meta-model** predicts whether the primary signal will be profitable (1/0)
3. **Bet sizing** converts meta-model probability into position sizes

### Step 1: Generate Meta-Labels

```python
from ml4t.engineer.labeling import meta_labels

meta_result = meta_labels(
    data=df,
    signal_col="primary_signal",  # +1/-1/0 from primary model
    return_col="forward_return",  # Actual forward returns
    threshold=0.0,                # Minimum return for "correct"
)
# Adds "meta_label" column: 1 if signal was correct, 0 otherwise
```

### Step 2: Train Meta-Model

Train any classifier on the meta-labels to predict P(primary signal is correct).

### Step 3: Compute Bet Sizes

```python
from ml4t.engineer.labeling import compute_bet_size, apply_meta_model

# Low-level: get bet size expression
bet_expr = compute_bet_size(
    probability="meta_probability",   # Column name or pl.Expr
    method="sigmoid",                 # "linear" | "sigmoid" | "discrete"
    scale=5.0,                        # Sigmoid steepness
    threshold=0.5,                    # Minimum probability to bet
)

# High-level: apply meta-model to size positions
result = apply_meta_model(
    data=df,
    primary_signal_col="signal",
    meta_probability_col="meta_prob",
    bet_size_method="sigmoid",
    scale=5.0,
    threshold=0.5,
    output_col="sized_signal",        # signal * bet_size
)
```

### Bet Sizing Methods

| Method | Formula | When to Use |
|--------|---------|-------------|
| `"linear"` | `max(0, p - threshold) / (1 - threshold)` | Simple, interpretable |
| `"sigmoid"` | `2 / (1 + exp(-scale * (p - 0.5))) - 1` | Smooth, differentiable |
| `"discrete"` | `1 if p >= threshold else 0` | Binary position sizing |

> **Book**: Ch7 `03_label_methods.py` implements the complete meta-labeling workflow: primary model signals → meta-labels → bet sizing on SPY.

## Calendar-Aware Labels

For futures and other instruments with defined trading sessions, calendar-aware labeling respects session boundaries. A label that would span an overnight gap is handled correctly.

```python
from ml4t.engineer.labeling import calendar_aware_labels

result = calendar_aware_labels(
    data=df,
    config=config,                          # LabelingConfig
    calendar="CME_Equity",                  # "NYSE", "CME_Equity", etc.
    price_col="close",
    timestamp_col="timestamp",
)
```

The calendar prevents forward returns from crossing session breaks (e.g., CME overnight gap from 4:00 PM to 5:00 PM CT).

## Sample Weighting

AFML Chapter 4 addresses the problem of overlapping labels creating correlated samples. The library provides the full toolkit:

### Label Uniqueness

```python
from ml4t.engineer.labeling import (
    build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
)

# Count how many labels overlap each bar
concurrency = build_concurrency(
    event_indices=starts,     # Label start indices
    label_indices=ends,       # Label end indices
    n_bars=len(df),
)

# Average uniqueness per label (range [0, 1])
uniqueness = calculate_label_uniqueness(
    event_indices=starts,
    label_indices=ends,
    n_bars=len(df),
)

# Combine uniqueness with return magnitude
weights = calculate_sample_weights(
    uniqueness=uniqueness,
    returns=returns_array,
    weight_scheme="returns_uniqueness",
    # Options: "returns_uniqueness", "uniqueness_only", "returns_only", "equal"
)
```

### Sequential Bootstrap

The sequential bootstrap (AFML Chapter 4) draws samples while accounting for label overlap, producing a more independent training set:

```python
from ml4t.engineer.labeling import sequential_bootstrap

selected = sequential_bootstrap(
    starts=start_indices,
    ends=end_indices,
    n_bars=len(df),
    n_draws=1000,             # Number of samples to draw
    with_replacement=True,
    random_state=42,
)
# selected: array of indices for training
```

### Integrated Computation

Triple-barrier labels can compute uniqueness and weights in a single call:

```python
result = triple_barrier_labels(
    df,
    config=config,
    calculate_uniqueness=True,
    uniqueness_weight_scheme="returns_uniqueness",
)
# Result includes label_uniqueness and sample_weight columns
```

### Label Statistics

Quick summary of label balance:

```python
from ml4t.engineer.labeling import compute_label_statistics

stats = compute_label_statistics(df, label_col="label")
# Returns: {"n_samples", "n_positive", "n_negative", "n_neutral",
#           "positive_ratio", "negative_ratio", "neutral_ratio"}
```

> **Book**: Ch7 `03_label_methods.py` demonstrates sequential bootstrap applied to triple-barrier labels, showing how it reduces effective sample size while improving independence.

## Time-Based Durations

All labeling functions that accept `max_holding_period` or `horizon` support duration strings in addition to bar counts:

```python
# Bar-based (integer)
config = LabelingConfig.triple_barrier(max_holding_period=20)

# Time-based (duration string)
config = LabelingConfig.triple_barrier(max_holding_period="4h")
config = LabelingConfig.triple_barrier(max_holding_period="1d")
config = LabelingConfig.triple_barrier(max_holding_period="30m")
```

### Supported Duration Formats

| Format | Example | Meaning |
|--------|---------|---------|
| Minutes | `"30m"` | 30 minutes |
| Hours | `"4h"` | 4 hours |
| Days | `"1d"` | 1 day |
| Combined | `"1h30m"` | 1 hour 30 minutes |

### Utility Functions

```python
from ml4t.engineer.labeling.utils import (
    is_duration_string,       # Check: is_duration_string("4h") → True
    parse_duration,           # Parse: parse_duration("1h30m") → timedelta(hours=1, minutes=30)
    time_horizon_to_bars,     # Convert to per-event bar counts using timestamps
    get_future_price_at_time, # Price at exact time offset
)
```

Time-based horizons require a `timestamp_col` in the input DataFrame.

## Performance

- **Speed**: ~50,000 labels/second (Numba-accelerated)
- **Memory**: Efficient vectorized implementation via Polars
- **Accuracy**: Exact match with AFML reference (validated at 1e-10 tolerance against mlfinpy)

## Best Practices

1. **Match barriers to transaction costs**: Barriers should exceed expected round-trip costs. A 0.1% barrier with 0.05% commission leaves little net profit.

2. **Handle class imbalance**: Triple-barrier often creates imbalanced labels (many vertical barrier hits). Check with `compute_label_statistics()` and consider adjusting barrier levels or using sample weights.

3. **Prevent leakage with sample weighting**: Overlapping labels create correlated training samples. Use `calculate_uniqueness=True` or `sequential_bootstrap()` to address this.

4. **Use ATR barriers for multi-asset**: Fixed barriers work for single-asset studies but fail across assets with different volatility levels.

5. **Time-based horizons for irregular data**: If your bars are not equally spaced (e.g., volume bars), use duration strings (`"4h"`) instead of bar counts to ensure consistent holding periods.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 3-4.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
