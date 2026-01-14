# trend/ - 10 Trend Indicators

Moving averages and trend-following indicators. All TA-Lib compatible.

## Simple Averages

### sma(data, period=20, price_col="close") -> DataFrame
Simple Moving Average.

### ema(data, period=20, price_col="close") -> DataFrame
Exponential Moving Average.

### wma(data, period=20, price_col="close") -> DataFrame
Weighted Moving Average. Linear weights.

## Double/Triple Smoothing

### dema(data, period=20, price_col="close") -> DataFrame
Double EMA. Reduces lag vs single EMA.

### tema(data, period=20, price_col="close") -> DataFrame
Triple EMA. Further reduces lag.

### t3(data, period=5, vfactor=0.7, price_col="close") -> DataFrame
Tillson T3. Smooth with minimal lag.

## Adaptive

### kama(data, period=10, fast=2, slow=30, price_col="close") -> DataFrame
Kaufman Adaptive MA. Adapts to volatility.

### trima(data, period=20, price_col="close") -> DataFrame
Triangular Moving Average. Double-smoothed SMA.

## Other

### midpoint(data, period=14) -> DataFrame
Midpoint over period. (highest + lowest) / 2.

### midprice(data, period=14) -> DataFrame
Mid Price. (highest_high + lowest_low) / 2.

## Common Usage

```python
# Crossover signals
df = sma(df, period=20).rename({"sma_20": "sma_fast"})
df = sma(df, period=50).rename({"sma_50": "sma_slow"})
signal = df["sma_fast"] > df["sma_slow"]
```
