# Alternative Bar Sampling

Transform tick data into information-driven bars instead of time-based bars.

## Why Alternative Bars?

Time bars (1min, 1h, daily) have problems:

- Unequal information content per bar
- Autocorrelation in returns
- Poor statistical properties

Alternative bars sample based on market activity, producing bars with more uniform information content.

## Volume Bars

Equal volume per bar:

```python
from ml4t.engineer.bars import volume_bars

bars = volume_bars(ticks_df, volume_threshold=1000)
```

## Dollar Bars

Equal dollar volume per bar:

```python
from ml4t.engineer.bars import dollar_bars

bars = dollar_bars(ticks_df, dollar_threshold=1_000_000)
```

## Tick Imbalance Bars

Information-driven bars based on order flow imbalance:

```python
from ml4t.engineer.bars import tick_imbalance_bars

bars = tick_imbalance_bars(ticks_df, expected_imbalance=100)
```

## Volume Imbalance Bars

Similar to tick imbalance but weighted by volume:

```python
from ml4t.engineer.bars import volume_imbalance_bars

bars = volume_imbalance_bars(ticks_df, expected_imbalance=10000)
```

## Input Format

All bar functions expect a DataFrame with:

- `timestamp`: Datetime of tick
- `price`: Trade price
- `volume`: Trade volume

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 2
