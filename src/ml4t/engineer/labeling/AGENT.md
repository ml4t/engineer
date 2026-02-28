# labeling/ - ML Label Generation

Methods for creating supervised learning labels from price data.

## Triple-Barrier (Recommended)

### triple_barrier_labels(data, config, ...) -> DataFrame
Path-dependent labeling with profit target, stop loss, and time horizon.
- Returns: +1 (profit), -1 (loss), 0 (timeout)
- Performance: 50,000 labels/second
Reference: López de Prado (2018), Chapter 3

### atr_triple_barrier_labels(data, ..., config=None, contract=None) -> DataFrame
Triple barrier with ATR-based dynamic barriers.
Barriers adapt to volatility.

## Percentile-Based (Adaptive)

### rolling_percentile_binary_labels(data, horizon, percentile, direction="long", lookback_window=252*24*12) -> DataFrame
Rolling window percentile thresholds. Adapts to volatility regimes.
- `percentile=95, direction="long"`: Top 5% returns → label=1
- `percentile=5, direction="short"`: Bottom 5% returns → label=1
Natural class balance (~5% positives for p95).

### rolling_percentile_multi_labels(data, horizons, percentiles, direction) -> DataFrame
Generate labels for multiple horizon/percentile combinations.

## Fixed Horizon (Simple)

### fixed_time_horizon_labels(data, horizon=1, method="returns") -> DataFrame
Simple forward return labels.
- `method="returns"`: (price[t+h] - price[t]) / price[t]
- `method="log_returns"`: log(price[t+h] / price[t])
- `method="binary"`: +1 up, -1 down

### trend_scanning_labels(data, min_window=5, max_window=50) -> DataFrame
De Prado's t-statistic method. Selects optimal window per observation.

## Meta Labels

### meta_labels(data, primary_model_predictions, side) -> DataFrame
Secondary model for bet sizing. Filters primary model predictions.

### compute_bet_size(data, probability, max_position=1.0) -> DataFrame
Position sizing from prediction probability.

## Sample Weighting

### calculate_sample_weights(data, label_col) -> DataFrame
Sample weights based on label uniqueness.

### sequential_bootstrap(data, n_samples) -> DataFrame
Sequential bootstrap sampling for overlapping labels.

### calculate_label_uniqueness(data, label_col) -> DataFrame
Uniqueness score per label.

## Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `triple_barrier.py` | 350 | High-level triple barrier API |
| `atr_barriers.py` | 303 | ATR-based barrier calculation |
| `percentile_labels.py` | 322 | Rolling percentile labels |
| `horizon_labels.py` | 259 | Fixed horizon + trend scanning |
| `meta_labels.py` | - | Meta-labeling for bet sizing |
| `uniqueness.py` | 389 | Sample weights and bootstrap |
| `numba_ops.py` | 424 | Numba-compiled operations |

Legacy module notes:
- `core.py`, `barriers.py`, and `barrier_utils.py` are removed compatibility paths that raise import-time migration errors.
