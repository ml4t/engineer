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

All methods return a Polars DataFrame with standardized output columns.

[Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls Engineer's labeling APIs on ETF data. The ETF case-study label notebook builds labels without Engineer, so the [Book Guide](../book-guide/index.md) classifies it as a related workflow rather than a library example.

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

<!-- ml4t-exec -->
```python
from datetime import datetime, timedelta

import polars as pl
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

close = [100.0 + i * 0.05 + (i % 6) * 0.2 for i in range(60)]
df = pl.DataFrame({
    "timestamp": [
        datetime(2024, 1, 1) + timedelta(minutes=i)
        for i in range(60)
    ],
    "open": close,
    "high": [price + 0.5 for price in close],
    "low": [price - 0.5 for price in close],
    "close": close,
})

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
    open_col="open",                  # Executes gaps at the open
    high_col="high",                  # For intrabar barrier touches
    low_col="low",                    # For intrabar barrier touches
    timestamp_col="timestamp",        # Required for time-based max_holding
    calculate_uniqueness=False,
    uniqueness_weight_scheme="returns_uniqueness",
)

assert {"label", "label_return", "barrier_hit"} <= set(result.columns)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | OHLCV data |
| `config` | `LabelingConfig` | required | Barrier configuration |
| `price_col` | `str` | `"close"` | Price column for barrier calculations |
| `open_col` | `str \| None` | `None` | Open column for gap execution |
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
| `barrier_hit` | Which barrier: `"upper"`, `"lower"`, `"time"` |
| `label_uniqueness` | Average uniqueness (when `calculate_uniqueness=True`) |
| `sample_weight` | Sample weight (when `calculate_uniqueness=True`) |

### The `side` Parameter

Controls directional bias:

- `side=1`: Long-only. Upper barrier = profit, lower barrier = loss.
- `side=-1`: Short-only. Upper barrier = profit, lower barrier = loss.
- `side=0`: Symmetric. Both barriers treated equally. Label is +1 or -1 based on direction.

Barrier distances and holding periods must be positive. Dynamic side columns accept
only -1, 0, and 1.

With high and low columns, an intrabar touch exits at the barrier price. Supplying
`open_col` makes a gap beyond a barrier exit at the open. If one bar touches both
barriers, the function selects the stop-loss exit because OHLC data does not show
which extreme occurred first.

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

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls `triple_barrier_labels` and visualizes barrier touches.

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

ATR is computed independently within each asset group. The returned
`upper_barrier_distance` and `lower_barrier_distance` columns are price-unit
distances. Internally, the function divides them by the event price before
calling triple-barrier labeling. ATR warm-up rows and rows with zero ATR have
null labels. High and low prices determine intrabar touches.

### When to Use ATR Barriers

| Scenario | Recommendation |
|----------|---------------|
| Single asset, stable volatility | Fixed barriers sufficient |
| Multi-asset (different volatility levels) | ATR barriers adapt per asset |
| Regime changes (calm → volatile) | ATR barriers avoid premature stops |
| Futures with varying contract sizes | ATR normalizes across contracts |

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls Engineer's ATR-barrier labeling API.

## Rolling Percentile Labels

Adaptive labeling computes each threshold from outcomes whose forward horizons have ended.
An unresolved outcome never contributes to an earlier threshold.

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

Decimal percentiles use `p` in column names. For example, percentile `95.5`
produces `threshold_p95p5_h10` and `label_long_p95p5_h10`.

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

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls Engineer's percentile and triple-barrier functions. [ETFs: Label Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/02_labels.ipynb) constructs its case-study labels without calling Engineer.

## Fixed Time Horizon

The simplest labeling method: compute forward returns over a fixed horizon. Supports both bar-count and time-based horizons.

```python
from ml4t.engineer.labeling import fixed_time_horizon_labels

# Bar-based horizon
result = fixed_time_horizon_labels(
    data=df,
    horizon=10,           # 10 bars forward
    method="binary",
    threshold=0.01,       # +1 above 1%, -1 below -1%, otherwise 0
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

Output names identify the method and horizon, such as `label_return_10p`,
`label_log_return_1h`, or `label_direction_10p`. A supplied `LabelingConfig`
controls the horizon, method, threshold, and column mapping. Conflicting explicit
arguments fail before calculation.

## Prepare keyed multi-horizon outcomes

The unreleased `forward_returns` API prepares realized outcomes for training or supplied-return
analysis. It requires Polars 2 and Specs 0.2. It does not replace barrier labels or Diagnostic's
Alphalens cleaner. Existing fixed-horizon defaults and output names remain unchanged. Integer
fixed-horizon arithmetic now avoids unsigned subtraction wrap and preserves small integer ticks.
Decimal price differences also retain their original scale until conversion to fractional returns.
Decimal inputs now produce Float64 returns rather than a scale-truncated Decimal ratio; price
columns are unchanged. Float32/Float64 calculations retain their established formula and dtype.

Use exact entry keys, declare the price basis, and select the horizon unit. This example counts
observations with `calendar=None`; it makes no schedule-completeness claim.

<!-- ml4t-exec -->
```python
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.artifacts import LabelDefinition, LabelSchema, LabelSpec
from ml4t.engineer.labeling import forward_returns

times = [datetime(2025, 7, 3, tzinfo=UTC) + timedelta(hours=i) for i in range(3)]
prices = pl.DataFrame({
    "timestamp": times,
    "asset": ["A"] * 3,
    "close": [100, 110, 121],
    "known_at": [time + timedelta(seconds=30) for time in times],
})
feed = FeedSpec(time_contract_version=2, calendar=None, entity_col="asset")
outcomes = forward_returns(
    prices,
    horizons={"one": 1, "two": 2},
    feed_spec=feed,
    window_unit="observation",
    endpoint_method="cumulative",
    price_basis="raw",
    available_at_col="known_at",
)
two = outcomes.filter(pl.col("horizon") == "two")
assert abs(two["forward_return"][0] - 0.21) < 1e-15
assert two["outcome_status"].to_list() == ["ok", "insufficient_tail", "insufficient_tail"]
assert two["exit_time"][0] == times[2]
assert two["outcome_available_at"][0] == prices["known_at"][2]

spec = LabelSpec(
    artifact_id="two-observation-outcomes",
    schema=LabelSchema(
        label_col="forward_return", entry_time_col="timestamp",
        requested_exit_time_col="requested_exit_time", exit_time_col="exit_time",
        outcome_available_at_col="outcome_available_at",
    ),
    definition=LabelDefinition(
        horizon="2", window_unit="observation", endpoint_method="cumulative",
        return_method="returns", missing_price_policy="raise", price_basis="raw",
        index_mode="per_asset", tail_policy="null", feed_spec=feed,
    ),
)
with TemporaryDirectory() as directory:
    path = Path(directory)
    two.write_parquet(path / "outcomes.parquet")
    (path / "outcomes.json").write_text(json.dumps(spec.to_dict()))
    restored = pl.read_parquet(path / "outcomes.parquet")
    restored_spec = LabelSpec.from_mapping(json.loads((path / "outcomes.json").read_text()))
    assert restored.equals(two)
    assert restored_spec.definition.feed_spec.resolve_calendar() is None
```

The result is a long table, one row per entry and named horizon. It retains entry keys,
requested and actual exits, the reference-price time, outcome availability, method, unit,
indexing mode, price basis and an exclusion status.
The horizon name is a selector. `horizon_value` stores a count, or a canonical fixed-nanosecond
duration for elapsed windows, so persisted values can replay without guessing a format.
Select one horizon before sending the table to Diagnostic's `prepare_signal_panel`, and pass `return_columns=["forward_return"]`,
`date_col="timestamp"` and the declared asset column. Metadata must not become return columns.
Diagnostic also accepts independently prepared equivalent tables without Engineer.

The installed handoff has been qualified against an unpublished Diagnostic build at
[`643b94b`](https://github.com/ml4t/diagnostic/tree/643b94b6636b0371319da84653fbe65cb95b56df),
not the published 0.1.9 package. Keep the original outcome artifact for endpoint lineage and join
it to retained analysis keys; do not backshift a sparse array. A manual join to older frames may
require normalizing their instant keys to UTC nanoseconds first. Session-label dates remain
identifiers, not midnight instants. Exact producer artifact and qualification revisions are recorded
in [the owning pull request](https://github.com/ml4t/engineer/pull/114).

| Choice | Endpoint definition |
|--------|---------------------|
| `window_unit="observation"` | N following chronological observations on the selected clock. |
| `window_unit="expected_bar"` | N following bars after validating the complete declared schedule/grid. |
| `window_unit="session"` | Final scheduled bar of the Nth following trading session; daily session labels remain dates. |
| `window_unit="elapsed"` | First timestamp at or after an exact physical-duration target, optionally bounded by `tolerance`. Session-label dates cannot define elapsed time. |
| `index_mode="per_asset"` | Each asset's own chronological price clock. |
| `index_mode="common_grid"` | Union of supplied timestamps; an absent asset quote is missing, never the next asset observation. |
| `endpoint_method="cumulative"` | Selected exit price divided by entry price, minus one. |
| `endpoint_method="exit_bar"` | Selected exit price divided by the preceding price on the selected clock, minus one. Thus 100/110/121 at horizon two gives 0.1 rather than 0.21. |

Calendar selection is explicit argument, declared feed metadata, then NYSE. A named calendar
requires a complete v2 grid declaration even when counting observations. A named 24/7 calendar
validates continuous scheduled bars; `None` disables schedule validation. Expected-bar and session
units require a named calendar. No frequency or holiday is inferred from observed gaps.
For intraday grids declare frequency, timezone, timestamp meaning, session-open anchoring, label
side and partial-bar policy using FeedSpec. Durations support composite and subsecond targets,
including `30m`, `1h30m`, `1d1h1ns`, `1ms` and `1ns`. Instants normalize to UTC nanoseconds.
Out-of-range inputs or targets fail explicitly rather than wrapping into another date. Naive inputs need
an explicit feed timezone; ambiguous/nonexistent local times fail.

The default missing-price policy raises for null, nonfinite or nonpositive prices. Explicit
`missing_price_policy="null"` retains unavailable outcomes and actual keys without filling quotes.
Absent scheduled rows still fail completeness validation. `tail_policy="null"` retains tail rows
with `insufficient_tail`; `drop` excludes those rows, and `raise` rejects the operation.
Missing in-range endpoints receive `missing_endpoint`, not zero returns.
If a covered session has no eligible full bar under the declared partial-bar policy, its outcome
also has `missing_endpoint`; it is not skipped in the session count.
`entries` may select a sparse ordered table of exact entity/time keys; missing or duplicate entry keys fail.
Input role names cannot use the reserved `_out_` prefix or conflict with output key metadata.

Engineer never constructs a filled price view. If a caller supplies an explicit analytical
research view, pass both `original_row_col` and `original_price_col` Boolean columns. They
identify original rows and valid observed quotes before the caller's preparation. Both must
be nonnull and consistent; an observed quote cannot belong to a synthetic row. Entry,
reference and exit row/price flags remain in the outcome table. Without a supplied view,
provided rows are original and finite positive quotes are observed. A numeric return with
false observation flags is a research outcome, not evidence of an observed executable price.
Named-calendar completeness is checked on original rows, so inserting and filling a missing
scheduled row does not clear the failure. Each asset must have original coverage. Preserve
the two input role declarations in LabelDefinition when persisting a research view's results.

Availability is the later known-at time of the two prices used. If either is unknown, availability
remains null. Session-label dates do not assert information availability. These realized outcomes
contain future information and must not enter a contemporaneous feature or strategy decision.
Keep the original feed and effective options in LabelSpec when persisting the table. Calendar
omission and explicit null retain their distinct shared serialization semantics.

## Trend Scanning

De Prado's trend-scanning method finds the optimal holding period for each observation by fitting linear regressions over a range of horizons and selecting the one with the highest t-statistic.

```python
from ml4t.engineer.labeling import trend_scanning_labels

result = trend_scanning_labels(
    data=df,
    min_window=5,
    max_window=20,
    step=1,
    t_value_threshold=2.0,
    price_col="close",
)
```

Output includes `label` (+1, -1, or null), `optimal_window`, and `t_value`.
The function selects the tested window with the largest absolute slope t-statistic.
Labels below `t_value_threshold` are null while `optimal_window` and `t_value`
remain available. A supplied trend-scanning config controls all four numerical settings.
Constant windows have null outputs. Exact nonconstant linear fits use the largest
finite float as the signed `t_value`.

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls `trend_scanning_labels` alongside the barrier and percentile methods.

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

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls the meta-labeling and bet-sizing APIs on SPY.

## Calendar-Aware Labels

For futures and other instruments with defined trading sessions, calendar-aware labeling respects session boundaries. A label that would span an overnight gap is handled correctly.

In the next minor version, `calendar_aware_labels` uses the shared temporal contract.
An explicit `calendar` argument wins over a declared `LabelingConfig.calendar`, then
`FeedSpec` metadata, then NYSE. Omitting a selector differs from explicit `None`.
Config JSON/YAML output preserves that distinction and emits only explicitly declared
column overrides, so a round trip does not replace feed column mappings with defaults.

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

The calendar prevents forward returns from crossing genuine session breaks, such as the
`CME_Equity` daily maintenance period from 4:00 PM to 5:00 PM CT. Its historical 3:15 PM to
3:30 PM pause applies only before trade date June 28, 2021. Modern labels can cross that
former pause because trading is continuous until 4:00 PM.
See the [CME exchange notice](https://www.cmegroup.com/notices/electronic-trading/2021/06/20210621.html)
for the affected equity-index contracts.

<a id="observation-rows-expected-bars-and-elapsed-durations"></a>

### Observation rows, expected bars, sessions and elapsed durations

Use `calendar=None` for observation-row labeling, including timestamp-indexed data:

```python
row_config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,
    lower_barrier=0.02,
    max_holding_period=10,
    calendar=None,
    window_unit="observation",
)
labels = calendar_aware_labels(df, row_config)
```

Row mode retains input order, processes each asset independently, does not require
timestamps and never loads a schedule provider. Its `label_bars` and `label_time` are
observation counts and per-asset row offsets; `label_duration` is unset. Timestamp
spacing and weekend placement do not alter its numerical labels. Duration holding
periods are incompatible with this mode.

Named calendars segment authoritative sessions and breaks. New shared-feed calls and
omitted selectors default count windows to `"expected_bar"`; an undeclared or incomplete
grid fails before labeling. Supply the complete shared grid
declarations described in the [feature guide](features.md#calendar-and-bar-grid) to reject
missing scheduled bars before labeling. Named `"24/7"` still has daily schedule boundaries
and is not equivalent to row mode. Duration holding periods select `"elapsed"` and remain truncated at session
boundaries, rather than counting sessions or observed dates.

Pass `feed_spec=FeedSpec(...)` directly or through `LabelingConfig.feed_spec`. Shared
`timestamp_semantics` must distinguish intraday instants from `session_label` identifiers.
Naive feed timestamps require an explicit timezone, and ambiguous/nonexistent local
times fail. Right-labeled/bar-close observations at a break boundary end the preceding
tradable interval. Input timestamp representations are retained in the output.
Instant horizons and `label_duration` use physical elapsed time, including across
daylight-saving transitions. Shared `session_label` identifiers do not identify instants;
their `label_time` retains the session identifier and `label_duration` is unset.

Legacy calls without FeedSpec keep naive-UTC handling. Custom protocol objects and
explicit `calendar="auto"` remain compatibility modes without shared feed metadata.
An explicit legacy calendar selector without FeedSpec retains observation-count horizons.
New calls can also select `window_unit="observation"` explicitly to count received rows
within each authoritative segment, without claiming scheduled-bar completeness.
Auto-gap segmentation does not establish market holidays, session completeness or a
bar grid. Standalone labeling kernels retain their documented row-count or elapsed-time
definitions and chronological sorting; they do not acquire calendar interpretation from
timestamps. Use this wrapper when selecting the shared calendar/row contract.
Standalone kernels reject configs declaring calendar/feed/window semantics instead
of silently ignoring those declarations.
Standalone kernels retain date-valued label times and calendar-day differences in
`label_duration` for legacy `Date` inputs. These standalone calls do not validate
the dates against an exchange schedule.

For `window_unit="session"`, an integer `max_holding_period=N` expires at the final
bar of the Nth scheduled session after the entry session. A positive integral count
column can select N per event. July 3, 2025 on NYSE therefore reaches July 7 for N=1,
not the July 4 holiday. Intraday entries reach the following session's last bar,
including an early close; lunch and overnight breaks do not truncate this horizon.
Profit, stop and trailing barriers still use the existing triple-barrier calculation.
The full declared grid is checked before any labels are computed, including whole-session
outages. No prices are filled or rows inserted.

With `partial_bar_policy="exclude"`, a shortened session can have no bars. It still
counts as a scheduled session. The exit uses the last available bar no later than the
deadline, even if that is the entry itself with zero holding bars. The calculation
does not extend the deadline into a later session to obtain another observation.

Daily input uses `FeedSpec(time_contract_version=2, calendar="NYSE",
timestamp_semantics="session_label", data_frequency="1d", bar_type="time")`.
Do not supply intraday anchor/label/partial-bar fields. Date-valued and midnight datetime
identifiers keep their displayed session date; they are not converted to UTC instants.
Daily observation and expected-bar horizons scan across session rows rather than
truncating every label to its entry row. Elapsed horizons reject these identifiers.

As with existing count horizons, a deadline beyond the available dataset is censored
at the last available bar. `label_bars` reports the actual observed holding length and
`label_time` the actual exit, not an inferred future observation. These changes target
the next 0.2 development release, not the historical 0.1.7 patch.

<!-- ml4t-exec -->
```python
from datetime import date
from io import BytesIO

import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import calendar_aware_labels

prices = pl.DataFrame({
    "timestamp": [date(2025, 7, day) for day in (3, 7, 8)],
    "close": [100.0, 101.0, 103.0],
})
config = LabelingConfig.triple_barrier(
    upper_barrier=None, lower_barrier=None, max_holding_period=1, window_unit="session",
    feed_spec=FeedSpec(time_contract_version=2, calendar="NYSE", data_frequency="1d",
                       timestamp_semantics="session_label", bar_type="time"),
)
labels = calendar_aware_labels(prices, config)
assert labels["label_time"].to_list() == [date(2025, 7, 7), date(2025, 7, 8), date(2025, 7, 8)]
assert labels["label_bars"].to_list() == [1, 1, 0]
labels.write_parquet(BytesIO())
labels.head(0).write_parquet(BytesIO())
```
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

> **Book**: [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls the sample-weighting and sequential-bootstrap APIs.

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

## Runtime behavior

Benchmark with your row count, horizon, labeling method, and hardware. Runtime and
memory use depend on those inputs.

## Best Practices

1. **Match barriers to transaction costs**: Barriers should exceed expected round-trip costs. A 0.1% barrier with 0.05% commission leaves little net profit.

2. **Handle class imbalance**: Triple-barrier often creates imbalanced labels (many vertical barrier hits). Check with `compute_label_statistics()` and consider adjusting barrier levels or using sample weights.

3. **Prevent leakage with sample weighting**: Overlapping labels create correlated training samples. Use `calculate_uniqueness=True` or `sequential_bootstrap()` to address this.

4. **Use ATR barriers for multi-asset**: Fixed barriers work for single-asset studies but fail across assets with different volatility levels.

5. **Time-based horizons for irregular data**: If your bars are not equally spaced (e.g., volume bars), use duration strings (`"4h"`) instead of bar counts to ensure consistent holding periods.

## See It In The Book

- [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) calls Engineer's labeling APIs.
- [ETFs: Label Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/02_labels.ipynb) is a related workflow that does not call Engineer.
- [Book Guide](../book-guide/index.md) records the pinned revision and task mapping.

## Next Steps

- Read [Dataset Builder](dataset-builder.md) when labeled data moves into training
  and CV workflows.
- Read [Preprocessing](preprocessing.md) if labels are part of a broader feature
  preparation pipeline.
- Use the [API Reference](../api/index.md) for the full labeling surface and config
  objects.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 3-4.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
