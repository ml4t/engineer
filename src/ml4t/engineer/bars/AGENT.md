# bars/ - Alternative Bar Sampling

Information-driven bar types for tick data. Reference: López de Prado (2018), Chapter 2.

## Time vs Information Bars

Standard time bars (1-min, daily) sample at fixed intervals regardless of market activity.
Alternative bars sample based on market information flow.

## Volume Bars

### volume_bars(ticks, volume_threshold=1000) -> DataFrame
Equal volume per bar. New bar when cumulative volume reaches threshold.
More bars during high activity, fewer during quiet periods.

### dollar_bars(ticks, dollar_threshold=1_000_000) -> DataFrame
Equal dollar volume per bar. Normalizes for price level changes.
Preferred over volume bars for most applications.

## Tick Bars

### tick_bars(ticks, tick_threshold=100) -> DataFrame
Equal number of trades per bar.

## Information Bars (Imbalance)

### tick_imbalance_bars(ticks, expected_imbalance=100) -> DataFrame
New bar when cumulative tick imbalance (buy vs sell) exceeds threshold.
Uses tick rule for trade classification.

### volume_imbalance_bars(ticks, expected_imbalance=10000) -> DataFrame
New bar when cumulative volume imbalance exceeds threshold.

### dollar_imbalance_bars(ticks, expected_imbalance=1_000_000) -> DataFrame
New bar when cumulative dollar imbalance exceeds threshold.

## Run Bars

### tick_run_bars(ticks, expected_run=50) -> DataFrame
New bar when run length (consecutive same-direction trades) exceeds threshold.

### volume_run_bars(ticks, expected_run=5000) -> DataFrame
Volume-weighted run bars.

### dollar_run_bars(ticks, expected_run=500_000) -> DataFrame
Dollar-weighted run bars.

## Input Requirements

All functions expect tick data with columns:
- `datetime`: Timestamp
- `price`: Trade price
- `volume`: Trade size

## Output Format

Returns OHLCV DataFrame:
- `datetime`: Bar timestamp (first tick)
- `open`, `high`, `low`, `close`: Price OHLC
- `volume`: Total volume in bar
- `tick_count`: Number of ticks in bar

## Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `base.py` | - | Base sampler interface |
| `vectorized.py` | 557 | Vectorized implementations |
| `volume.py` | - | Volume/dollar bars |
| `tick.py` | - | Tick bars |
| `imbalance.py` | - | Imbalance bars |
| `run.py` | 673 | Run bars |

## Example

```python
from ml4t.engineer.bars import dollar_bars, tick_imbalance_bars

# Load tick data
ticks = pl.read_parquet("ticks.parquet")

# Create dollar bars
bars = dollar_bars(ticks, dollar_threshold=1_000_000)

# Create information bars
info_bars = tick_imbalance_bars(ticks, expected_imbalance=100)
```
