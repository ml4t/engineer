# ml/ - 11 ML-Specific Features

Features designed for machine learning pipelines.

## Lag Features

### lag(data, columns, periods=[1, 2, 3, 5, 10]) -> DataFrame
Create lagged versions of columns.
```python
df = lag(df, columns=["close", "volume"], periods=[1, 5, 10])
# Creates: close_lag_1, close_lag_5, ..., volume_lag_1, ...
```

### create_lag_features(data, price_col="close", lags=[1,2,3]) -> DataFrame
Convenience wrapper for price lags.

## Temporal Encoding

### cyclical_encode(data, column, period) -> DataFrame
Cyclical encoding for periodic features.
```python
df = cyclical_encode(df, "hour", period=24)
# Creates: hour_sin, hour_cos columns
```

### time_features(data, datetime_col="datetime") -> DataFrame
Extract time components: hour, dayofweek, month, quarter.

## Frequency Domain

### fourier_features(data, column, periods=[5, 10, 20]) -> DataFrame
Fourier transform features for seasonality detection.

### rolling_fft(data, column, period=20) -> DataFrame
Rolling FFT for frequency analysis.

## Information Theory

### rolling_entropy(data, column, period=20) -> DataFrame
Shannon entropy over rolling window. Measures randomness.
High entropy = unpredictable, low = more structured.

### mutual_information(data, columns, period=20) -> DataFrame
Rolling mutual information between columns.

## Statistical

### rolling_stats(data, column, period=20) -> DataFrame
Rolling mean, std, skew, kurtosis.

### percentile_rank(data, column, period=252) -> DataFrame
Current value as percentile of history.

### z_score(data, column, period=20) -> DataFrame
Rolling z-score normalization.

## Target Creation

### directional_targets(data, horizon=5) -> DataFrame
Binary direction labels. +1 up, -1 down, 0 flat.
