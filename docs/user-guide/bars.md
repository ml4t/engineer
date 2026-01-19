# Alternative Bar Sampling

Transform tick data into information-driven bars instead of time-based bars.

## Why Alternative Bars?

Time bars (1min, 1h, daily) have problems:

- Unequal information content per bar
- Autocorrelation in returns
- Poor statistical properties (non-normal, heteroskedastic)

Alternative bars sample based on market activity, producing bars with more uniform
information content and better statistical properties for ML models.

## Quick Start

```python
from ml4t.engineer.bars import (
    TickBarSampler,
    VolumeBarSampler,
    DollarBarSampler,
    # Adaptive imbalance bars (AFML algorithm - requires careful calibration)
    ImbalanceBarSampler,       # Volume Imbalance Bars (VIBs)
    TickImbalanceBarSampler,   # Tick Imbalance Bars (TIBs)
    # Fixed threshold bars (recommended for production)
    FixedTickImbalanceBarSampler,
    FixedVolumeImbalanceBarSampler,
    # Window-based bars (bounded adaptation via rolling windows)
    WindowTickImbalanceBarSampler,
    WindowVolumeImbalanceBarSampler,
    # Run bars
    TickRunBarSampler,
)

# Tick bars: fixed number of trades per bar
tick_bars = TickBarSampler(ticks_per_bar=100).sample(trades_df)

# Volume bars: fixed volume per bar
volume_bars = VolumeBarSampler(volume_per_bar=10_000).sample(trades_df)

# Dollar bars: fixed dollar volume per bar
dollar_bars = DollarBarSampler(dollars_per_bar=1_000_000).sample(trades_df)

# Tick Imbalance Bars (TIBs): θ = Σ b_t
tick_imbalance_bars = TickImbalanceBarSampler(
    expected_ticks_per_bar=1000,
    alpha=0.001,           # CRITICAL: Use slow adaptation
    min_bars_warmup=100,   # Longer warmup for stability
).sample(trades_df)

# Volume Imbalance Bars (VIBs): θ = Σ b_t × v_t
volume_imbalance_bars = ImbalanceBarSampler(
    expected_ticks_per_bar=10000,
    alpha=0.001,           # CRITICAL: Use slow adaptation
    min_bars_warmup=100,   # Longer warmup for stability
).sample(trades_df)

# Run bars: adaptive threshold based on directional dominance
run_bars = TickRunBarSampler(expected_ticks_per_bar=50).sample(trades_df)

# RECOMMENDED: Fixed threshold imbalance bars (stable, no calibration issues)
fixed_tick_bars = FixedTickImbalanceBarSampler(threshold=100).sample(trades_df)
fixed_volume_bars = FixedVolumeImbalanceBarSampler(threshold=50_000).sample(trades_df)
```

## Input Format

All samplers expect a DataFrame with:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Trade timestamp |
| `price` | float | Trade price |
| `volume` | float | Trade volume |
| `side` | float | Trade direction: +1 (buy), -1 (sell). Required for imbalance/run bars. |

## Information-Driven Bars

AFML defines two variants of imbalance bars that sample when cumulative order
flow imbalance exceeds an adaptive threshold.

### Tick Imbalance Bars (TIBs)

Accumulate signed trade direction (each trade counts ±1).

**Threshold Formula:**
```
θ = Σ b_t
E[θ_T] = E[T] × |2P[b=1] - 1|
```

Where:
- `E[T]` = Expected bar length (ticks per bar), updated via EWMA
- `P[b=1]` = Probability a trade is a buy
- `b_t ∈ {+1, -1}` = Trade direction

**Usage:**

```python
from ml4t.engineer.bars import TickImbalanceBarSampler

sampler = TickImbalanceBarSampler(
    expected_ticks_per_bar=1000,  # Initial E[T]
    alpha=0.001,                  # EWMA decay factor (SLOW!)
    initial_p_buy=0.5,            # Initial P[b=1]
    min_bars_warmup=100,          # Bars before EWMA updates start
)

bars = sampler.sample(trades_df)
```

### Volume Imbalance Bars (VIBs)

Accumulate signed volume (each trade weighted by size).

**Threshold Formula:**
```
θ = Σ b_t × v_t
E[θ_T] = E[T] × |2v⁺ - E[v]|
```

Where:
- `E[T]` = Expected bar length (ticks per bar), updated via EWMA
- `v⁺ = P[b=1] × E[v|b=1]` = Expected buy volume contribution
- `P[b=1]` = Probability a trade is a buy
- `E[v|b=1]` = Expected volume given trade is a buy
- `E[v]` = Unconditional mean volume per tick

**Usage:**

```python
from ml4t.engineer.bars import ImbalanceBarSampler

sampler = ImbalanceBarSampler(
    expected_ticks_per_bar=10000,  # Initial E[T] (higher than TIBs!)
    alpha=0.001,                   # EWMA decay factor (SLOW!)
    initial_p_buy=0.5,             # Initial P[b=1]
    min_bars_warmup=100,           # Bars before EWMA updates start
)

bars = sampler.sample(trades_df)
```

### TIBs vs VIBs

**Key Difference:** TIBs count trades equally; VIBs weight by size. For the
same `E[T]`, TIBs produce 100-1000x more bars because:

- TIB threshold ≈ `E[T] × 0.1` (for 55% buy fraction)
- VIB threshold ≈ `E[T] × volume_per_trade × 0.1`

**Output columns (both):**
- Standard OHLCV: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`
- Imbalance: `buy_volume`, `sell_volume`, `imbalance`
- Threshold: `expected_imbalance`, `cumulative_theta`
- EWMA state: `expected_t`, `p_buy`, `v_plus`, `e_v`

### Run Bars

Sample when one side of the market dominates, measured by cumulative counts.

**Key Point:** Run bars count **cumulative** trades on each side within a bar,
NOT consecutive same-direction trades. Direction changes within a bar do NOT
reset the counts.

**Threshold Formula:**
```
θ = max(cumulative_buys, cumulative_sells)
E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]}
```

**Usage:**

```python
from ml4t.engineer.bars import TickRunBarSampler

sampler = TickRunBarSampler(
    expected_ticks_per_bar=50,  # Initial E[T]
    alpha=0.1,                   # EWMA decay factor
    initial_p_buy=0.5,          # Initial P[b=1]
    min_bars_warmup=10,         # Bars before EWMA updates start
)

bars = sampler.sample(trades_df)
```

**Variants:**
- `TickRunBarSampler`: Counts number of trades
- `VolumeRunBarSampler`: Sums volumes
- `DollarRunBarSampler`: Sums dollar volumes

## EWMA Parameter Updates

After the warmup period, all parameters update via exponentially weighted
moving averages:

```python
E[T]_new = α × actual_bar_length + (1-α) × E[T]_old
P[b=1]_new = α × bar_buy_fraction + (1-α) × P[b=1]_old
```

The `alpha` parameter controls adaptation speed:
- Higher α (e.g., 0.3): Faster adaptation, more responsive to recent data
- Lower α (e.g., 0.05): Slower adaptation, more stable thresholds

## ⚠️ Critical: Avoiding Threshold Spiral

**The default α=0.1 from AFML causes threshold spiral with real market data!**

The adaptive EWMA algorithm is sensitive to persistent order flow imbalance.
Most stocks show systematic buy/sell bias (not 50/50), causing a positive feedback loop:

1. Bars form when imbalance exceeds threshold
2. P[b=1] estimate drifts toward actual buy fraction (e.g., 55%)
3. Larger P[b=1] → larger threshold → larger bars
4. E[T] adapts upward to match larger bars
5. Threshold grows exponentially!

**Empirical Evidence** (NVDA and SPY, 2024):

| Symbol | Buy Fraction | α=0.1 Spiral | α=0.001 Spiral |
|--------|--------------|--------------|----------------|
| NVDA   | 60%          | 32x ⚠️       | 2.9x ✓         |
| SPY    | 49%          | 6.6x ⚠️      | 1.2x ✓         |

Even SPY with near-balanced order flow exhibits 6x threshold spiral with α=0.1!

**Recommended Settings:**

```python
# CORRECT: Use slow adaptation
sampler = TickImbalanceBarSampler(
    expected_ticks_per_bar=1000,
    alpha=0.001,           # NOT 0.1!
    min_bars_warmup=100,   # NOT 10
)

# WRONG: Will cause threshold spiral
sampler = TickImbalanceBarSampler(
    expected_ticks_per_bar=1000,
    alpha=0.1,             # Too fast!
    min_bars_warmup=10,    # Too short!
)
```

**Detection:** Monitor the `expected_imbalance` column in output. If it grows
by >3x over the dataset, threshold spiral is occurring.

## 🎯 Fixed-Threshold Imbalance Bars (Recommended)

**For production use, we recommend fixed-threshold imbalance bars** which avoid
all the issues with adaptive algorithms.

```python
from ml4t.engineer.bars import (
    FixedTickImbalanceBarSampler,
    FixedVolumeImbalanceBarSampler,
)

# Fixed Tick Imbalance Bars
# Bar forms when |Σ b_t| >= threshold
tick_imbalance_bars = FixedTickImbalanceBarSampler(
    threshold=100,  # Fixed threshold (no adaptation)
).sample(trades_df)

# Fixed Volume Imbalance Bars
# Bar forms when |Σ b_t × v_t| >= threshold
volume_imbalance_bars = FixedVolumeImbalanceBarSampler(
    threshold=50_000,  # Fixed threshold (no adaptation)
).sample(trades_df)
```

### Why Fixed Thresholds?

Advantages over adaptive (AFML) algorithm:
- **No threshold spiral** - stable by construction
- **Predictable bar count** - based on imbalance statistics
- **No feedback loops** - works consistently across all market conditions
- **Simpler to calibrate** - one parameter instead of three

### Calibration

To get approximately N bars per day:

1. **For tick imbalance**: `threshold ≈ ticks_per_day / N × |2P[b=1] - 1|`
2. **For volume imbalance**: `threshold ≈ daily_volume / N × order_flow_asymmetry`

Or empirically: test thresholds `[50, 100, 200, 500, 1000]` and pick the one
giving your desired bar frequency.

### Output Columns

**FixedTickImbalanceBarSampler:**
- Standard OHLCV: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`
- Imbalance: `buy_volume`, `sell_volume`, `buy_count`, `sell_count`, `tick_imbalance`
- Threshold: `cumulative_theta`, `threshold`

**FixedVolumeImbalanceBarSampler:**
- Standard OHLCV: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`
- Imbalance: `buy_volume`, `sell_volume`, `volume_imbalance`
- Threshold: `cumulative_theta`, `threshold`

## 🔄 Window-Based Imbalance Bars

Window-based imbalance bars offer a middle ground between AFML's exponential decay
and fixed thresholds. They use rolling windows for parameter estimation, providing
bounded adaptation without threshold spiral.

```python
from ml4t.engineer.bars import (
    WindowTickImbalanceBarSampler,
    WindowVolumeImbalanceBarSampler,
)

# Window-based Tick Imbalance Bars
tick_imbalance_bars = WindowTickImbalanceBarSampler(
    initial_expected_t=1000,  # Initial E[T]
    bar_window=10,            # E[T] from last 10 bars
    tick_window=5000,         # P[b=1] from last 5000 ticks
).sample(trades_df)

# Window-based Volume Imbalance Bars
volume_imbalance_bars = WindowVolumeImbalanceBarSampler(
    initial_expected_t=5000,  # Initial E[T]
    bar_window=10,            # E[T] from last 10 bars
    tick_window=5000,         # Imbalance from last 5000 ticks
).sample(trades_df)
```

### How Window-Based Works

Instead of exponential decay (EWMA), window-based uses rolling means:

1. **E[T]** = mean of last `bar_window` bar lengths
2. **P[b=1]** = mean of last `tick_window` tick signs (for TIBs)
3. **Imbalance factor** = rolling signed volume (for VIBs)

**Key feature:** Old data falls out of the window, preventing unbounded parameter drift.

### Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `initial_expected_t` | Initial E[T] before first bar | 1000-5000 |
| `bar_window` | Bars to average for E[T] | 10-50 |
| `tick_window` | Ticks to average for P[b=1] | 2000-10000 |

### Warmup Behavior

**Important:** Window-based bars wait until `tick_window` ticks accumulate before
forming bars. The first `tick_window` ticks are used only for initialization.

This warmup period means fewer bars for short datasets.

### Comparison: Three Approaches

| Method | Bars (100K ticks) | E[T] Drift | Best For |
|--------|-------------------|------------|----------|
| Fixed threshold=100 | 198 | N/A | Production, simplicity |
| α-based α=0.001 | 106 | 1.01x | AFML fidelity |
| α-based α=0.1 | 97 | 1.24x | ⚠️ Avoid (spiral) |
| Window tick_win=2000 | 113 | 1.24x | Bounded adaptation |
| Window tick_win=5000 | 77 | 1.27x | Stable estimation |

**Recommendations:**
- **Production:** Use `FixedTickImbalanceBarSampler` (simplest, no drift)
- **Research:** Use `WindowTickImbalanceBarSampler` (bounded adaptation)
- **AFML replication:** Use `TickImbalanceBarSampler` with α=0.001

### Output Columns

Both window-based samplers output:
- Standard OHLCV: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `tick_count`
- Imbalance: `buy_volume`, `sell_volume`, `imbalance`
- Threshold: `expected_imbalance`, `cumulative_theta`
- Window state: `expected_t`, `p_buy` (TIBs) or `imbalance_factor` (VIBs)

## Handling Incomplete Bars

By default, the final bar (which may not have reached the threshold) is excluded:

```python
# Exclude incomplete final bar (default)
bars = sampler.sample(trades_df, include_incomplete=False)

# Include incomplete final bar
bars = sampler.sample(trades_df, include_incomplete=True)
```

## Performance Notes

The default implementations use vectorized Polars operations with Numba-accelerated
inner loops for optimal performance. For very large datasets, consider:

1. Processing data in daily chunks
2. Using the streaming API for out-of-memory datasets
3. Caching computed bars rather than recomputing

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  John Wiley & Sons. Chapter 2.3: Information-Driven Bars.
