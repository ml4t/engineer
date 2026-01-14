# volume/ - 3 Volume Indicators

Volume-based indicators. All TA-Lib compatible.

## On Balance Volume

### obv(data) -> DataFrame
On Balance Volume. Cumulative volume based on price direction.
- Price up: +volume
- Price down: -volume
Trend confirmation: OBV trend should match price trend.

## Accumulation/Distribution

### ad(data) -> DataFrame
Accumulation/Distribution Line.
Uses CLV (Close Location Value) * volume.
CLV = ((close - low) - (high - close)) / (high - low)

### adosc(data, fast=3, slow=10) -> DataFrame
A/D Oscillator. EMA difference of A/D line.
Fast/slow crossover signals.

## Common Usage

```python
# Trend confirmation
df = obv(df)
price_trend = df["close"].diff() > 0
volume_confirms = df["obv"].diff() > 0
divergence = price_trend != volume_confirms
```
