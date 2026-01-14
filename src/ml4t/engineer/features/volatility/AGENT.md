# volatility/ - 15 Volatility Measures

Historical and realized volatility estimators.

## TA-Lib Compatible

### atr(data, period=14) -> DataFrame
Average True Range. Absolute volatility measure.

### natr(data, period=14) -> DataFrame
Normalized ATR. ATR as percentage of close.

### trange(data) -> DataFrame
True Range. Single-bar volatility.

### bollinger_bands(data, period=20, std_dev=2.0) -> DataFrame
Bollinger Bands. Returns: bb_upper, bb_middle, bb_lower, bb_bandwidth.

## Academic Estimators

### yang_zhang(data, period=20) -> DataFrame
Yang-Zhang volatility. Most efficient OHLC estimator. Drift-independent.

### parkinson(data, period=20) -> DataFrame
Parkinson high-low volatility. Uses log(H/L).

### garman_klass(data, period=20) -> DataFrame
Garman-Klass OHLC volatility. More efficient than close-to-close.

### rogers_satchell(data, period=20) -> DataFrame
Rogers-Satchell. Drift-adjusted, handles trending markets.

### close_to_close(data, period=20) -> DataFrame
Close-to-close volatility. Traditional standard deviation.

## Realized Measures

### realized_volatility(data, period=20) -> DataFrame
Realized volatility. Sum of squared returns.

### ewma_volatility(data, span=20, min_periods=10) -> DataFrame
EWMA volatility. Exponentially weighted.

### range_volatility(data, period=20) -> DataFrame
High-low range based volatility.

## Conditional

### garch(data, p=1, q=1) -> DataFrame
GARCH(p,q) volatility forecast.

### conditional_volatility_ratio(data, short=5, long=20) -> DataFrame
Ratio of short-term to long-term volatility.

### volatility_percentile_rank(data, period=252) -> DataFrame
Current volatility percentile vs history.

## Efficiency Comparison

| Estimator | Efficiency | Drift Robust | Notes |
|-----------|------------|--------------|-------|
| Yang-Zhang | Best | Yes | Preferred for most uses |
| Garman-Klass | Good | No | Assumes zero drift |
| Rogers-Satchell | Good | Yes | Handles trending |
| Parkinson | Medium | No | High-low only |
| Close-to-Close | Baseline | No | Simple but inefficient |
