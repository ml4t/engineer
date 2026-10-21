# Technical Indicators

ML4T Engineer provides 120 registry features across 11 categories, built on
Polars with Numba JIT for performance-critical kernels. 60 indicators are
validated against TA-Lib at 1e-6 tolerance.

## Overview

| Category | Count | TA-Lib | Key Indicators |
|----------|-------|--------|----------------|
| Momentum | 31 | 19 | RSI, MACD, Stochastic, CCI, ADX, MFI |
| Volatility | 15 | 4 | ATR, Bollinger, Yang-Zhang, Parkinson, GARCH |
| Microstructure | 15 | 0 | Kyle Lambda, Amihud, Roll Spread, Realized Spread |
| Trend | 10 | 9 | SMA, EMA, WMA, DEMA, TEMA, KAMA |
| ML Features | 14 | 0 | Lag, Entropy, Fourier, Cyclical Encode |
| Statistics | 14 | 7 | STDDEV, Linear Regression, TSF, Variance Ratio |
| Risk | 6 | 0 | Max Drawdown, Downside Deviation, Sharpe, Sortino |
| Price Transform | 5 | 5 | Typical Price, Weighted Close, Average Price |
| Regime | 4 | 0 | Hurst Exponent, Choppiness, Fractal Efficiency |
| Volume | 3 | 3 | OBV, AD, ADOSC |
| Math | 3 | 3 | Maximum, Minimum, Summation |

Standalone cross-asset utilities such as beta, rolling correlation, and
cointegration are documented below, but they are not part of the 120-feature
registry count.

The pinned book notebooks combine direct Engineer calls with manual teaching implementations. Start with [The ml4t Library Ecosystem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) for `compute_features()`, then use the [Book Guide](../book-guide/index.md) to check each notebook's exact relationship to the API.

## Computation API

`compute_features()` accepts three input formats:

<!-- ml4t-exec -->
```python
from datetime import date, timedelta

import polars as pl
from ml4t.engineer import compute_features

close = [100.0 + i * 0.1 + (i % 5) * 0.05 for i in range(100)]
df = pl.DataFrame({
    "timestamp": [date(2024, 1, 1) + timedelta(days=i) for i in range(100)],
    "open": close,
    "high": [price + 1.0 for price in close],
    "low": [price - 1.0 for price in close],
    "close": close,
    "volume": [100_000 + i * 100 for i in range(100)],
})

# 1. List of names (default parameters)
result = compute_features(df, ["rsi", "macd", "atr"], calendar=None)

# 2. List of dicts (custom parameters)
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
    {
        "name": "bollinger_bands",
        "params": {"period": 20, "nbdevup": 2.5, "nbdevdn": 2.5},
    },
], calendar=None)

assert {"rsi", "sma", "bollinger_bands"} <= set(result.columns)
```

Features are computed in dependency order (topological sort). Circular dependencies raise `ValueError`. The return type matches the input: `DataFrame` in, `DataFrame` out; `LazyFrame` in, `LazyFrame` out.

Common asset columns such as `asset_id`, `symbol`, and `ticker` are detected automatically,
and every rolling or lagged feature is evaluated independently for each asset.
The examples explicitly select `calendar=None`: their periods count observations and
input order is preserved, even when timestamps are irregular. Sort prices yourself before
deriving returns if chronological order is intended. Row mode does not require timestamps
or an additional `assume_sorted=True` assertion. A disabled calendar in shared feed metadata
has the same behavior, even when the feed's timestamp column is absent.

Use `output` when requesting more than one configuration of the same feature:

```python
result = compute_features(df, [
    {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
    {"name": "sma", "params": {"period": 50}, "output": "sma_50"},
], calendar=None)
```

Unknown parameters, repeated output names, and output names that replace input columns
raise `ValueError` before feature execution.

### Derive precise price returns

The next minor version requires Polars 2.0 or newer. Return expressions select their
native arithmetic from the column dtype while planning a query, without collecting
lazy data. Keep integer ticks as integers until after the expression evaluates;
casting large prices to floating point first can erase a one-tick change.

<!-- ml4t-exec -->
```python
import polars as pl
from ml4t.engineer.features.utils.arithmetic import returns, log_returns

prices = pl.DataFrame({
    "asset": ["A", "A", "A"],
    "close": pl.Series([2**53, 2**53 + 1, 2**53 + 2], dtype=pl.Int64),
})
result = prices.lazy().with_columns(
    returns("close").over("asset").alias("return"),
    log_returns("close").over("asset").alias("log_return"),
).collect()
assert result["return"][0] is None
assert result["return"][1] == 1 / 2**53
assert result["log_return"][1] > 0

from decimal import Decimal
from math import isclose

decimal_prices = pl.DataFrame({"close": [
    Decimal("1000000.000000000000"),
    Decimal("1000000.000000000001"),
    Decimal("1000000.000000000002"),
]})
decimal_returns = decimal_prices.select(returns("close"))
assert decimal_returns.schema["close"] == pl.Float64
assert isclose(decimal_returns["close"][1], 1e-18, rel_tol=1e-15, abs_tol=0)
```

Both expressions retain the input row count and leave the first `periods` rows
undefined in each asset. Integer differences are computed before floating division;
floating input keeps its existing ratio formula and dtype. Log returns retain tiny
changes without turning a near-total loss into an artificial negative infinity.
`percentage_change` uses the same simple-return calculation multiplied by 100.
Decimal inputs retain their exact scale through subtraction, then produce Float64 fractional
returns. This changes their former Decimal output dtype, whose division could truncate a nonzero
tick to zero. Original prices stay unchanged; Float32/Float64 input behavior is unchanged.

When replacing an Empyrical price-return call, note that its output omits the first
undefined observation. Trim that documented leading row only if the consuming
workflow needs that shape; do not drop every null or compress gaps. These are price
expressions, not portfolio statistics. Use explicit grouping for multiple assets and
validate the declared bar grid before deriving returns as described below. Row-only
data must already have the intended observation order.

### Explicit finite-window methods

The next minor version adds the following registered methods for workflows that need
finite observation windows rather than the native recursive or TA-Lib definitions.
Existing `rsi`, `ema`, `macd`, `aroon`, `bollinger_bands` and risk features keep their
defaults. These additions do not provide a Pipeline graph interpreter.

| Feature | Definition and missingness |
|---|---|
| `rsi_cutler` | Simple gains/losses over `period` changes (`period + 1` prices). Omit changes adjacent to missing quotes; a flat window is undefined. |
| `absolute_percent_change` | Fractional change over `period` changes with an absolute reference denominator. This accepts signed values and does not multiply by 100. |
| `finite_mean` | Require `period` rows, then divide by the count of nonmissing prices. |
| `finite_weighted_mean` | Sum valid price-times-volume products, divided by the independently summed volume. Missing prices do not remove denominator weights. |
| `average_dollar_volume` | Sum valid price-times-volume products, divided by the fixed `period` row count. Input price, share and currency basis must already agree. |
| `linear_weighted_mean` | Weights `1..period`, oldest to newest. Omit missing products but retain every denominator weight. |
| `finite_ewma` | Normalize `decay_rate**age` over exactly `period` prices. Missing prices propagate; this is not recursive EMA. |
| `finite_ewm_std` | The same finite weights, with weighted bias correction. Insufficient effective degrees of freedom yield NaN. |
| `annualized_volatility` | Population standard deviation (`ddof=0`) of supplied `returns`, omitting missing values after the full window. Multiply by `sqrt(annualization_factor)`, default 252. Calendar choice does not set this scale. |
| `finite_max_drawdown` | Select the first largest absolute price drop, then divide by its trough price. This explicit legacy definition is not a peak-denominator portfolio drawdown. |
| `finite_bollinger_bands` | NaN-omitting mean and population standard deviation; fields `lower`, `middle`, `upper`, with multiplier `k`. |
| `finite_aroon` | First extreme wins ties; fields `down`, `up`. Scale the position in `period` prices by `100 / (period - 1)`. |
| `fast_stochastic` | Unsmoothed K from the latest close and NaN-omitting high/low extrema. Zero range remains undefined. |
| `finite_ichimoku` | Unshifted `tenkan_sen`, `kijun_sen`, `senkou_span_a`, `senkou_span_b`. `chikou_position` indexes from the oldest price in the window, unlike a chart displacement. |
| `finite_macd_signal` | Finite normalized fast/slow exponential means, followed by a finite signal mean. Length-one means use their sole observation. |
| `finite_true_range` | Two-price true range with NaN-omitting high/low/previous-close components. There is no implicit larger-window reduction. |

Except for Cutler RSI, absolute percent change and the composed MACD signal, `period` counts input rows and the
first structurally usable row is `period - 1`. Missing observations still occupy their
original row positions. Numeric results are Float64 and retain every input row;
Ichimoku's selected `chikou_span` price retains its original dtype. Multi-output
features use the usual `<output>_<field>` column names. Ichimoku
propagates missing high/low values within each field's subwindow; all fields require
the declared full window. Its zero tenkan/kijun lengths select that full window,
matching the legacy `[-0:]` slice, while `chikou_position` must be in `[0, period)`.

<!-- ml4t-exec -->
```python
import math
import polars as pl
from ml4t.engineer import compute_features

prices = pl.DataFrame({"close": [10.0, 13.0, 12.0, 14.0, 13.0]}).with_columns(
    pl.col("close").alias("open"),
    pl.col("close").alias("high"),
    pl.col("close").alias("low"),
    pl.lit(100.0).alias("volume"),
)
result = compute_features(prices.lazy(), [
    {"name": "rsi_cutler", "params": {"period": 3}, "output": "cutler"},
    {"name": "rsi", "params": {"period": 3}, "output": "wilder"},
], calendar=None).collect()
assert math.isclose(result["cutler"][4], 50.0, rel_tol=1e-14, abs_tol=0.0)
assert result["wilder"][4] != result["cutler"][4]
assert result.height == prices.height
```

Use explicit grouping or `compute_features` for multiple assets. A `calendar=None`
call counts the supplied observation rows without checking schedules. With a named
calendar, declare the complete grid before computing any feature. The finite methods
do not infer frequency, select source revisions, fill gaps or make future information
eligible. They currently support observation/expected-bar windows; do not reinterpret
their kernels as session or elapsed windows. Variance methods subtract an original
window-local typed price anchor before floating conversion, so adjacent large integer ticks do
not collapse to zero variance. Output remains Float64.

### Factor-name expression recipes

`compat.zipline.factor_expression` translates supported built-in names into the
native methods above. It returns a Polars expression or a dictionary of named output
expressions, not a Pipeline Term. It never imports Zipline or evaluates an opaque
legacy graph. Use native column operators for arithmetic/comparisons and the existing
feature decorator for custom calculations.

Recipe `window_length` counts prices/inputs as in Pipeline. Return and Cutler recipes
translate N prices into N-1 changes. The default annualized-volatility recipe derives
daily returns from `close` and keeps that extra dependency row of warmup. Explicit
`inputs=["returns"]` instead consumes supplied returns. EWMA/EWMSTD accept one explicit
`decay_rate`, `span`, `halflife` or `center_of_mass`, including the three legacy
constructor suffixes. Statistical recipes can consume already aligned column pairs;
their legacy correlation/regression lengths and beta missing percentage are explicit.

<!-- ml4t-exec -->
```python
import math
import polars as pl
from ml4t.engineer.compat.zipline import factor_expression

frame = pl.DataFrame({"close": [1.0, 3.0, 9.0], "eligible": [True, False, True]})
expression = factor_expression(
    "EWMA.from_span", inputs=["close"], window_length=3, span=3, mask="eligible",
)
result = frame.lazy().with_columns(expression.alias("finite_mean")).collect()
assert math.isclose(result["finite_mean"][2], 43 / 7, rel_tol=1e-14)
assert result.height == frame.height
```

The output mask in this example does not erase the middle observation from history.
Apply `.over(entity_col)` for independent asset histories. These expression recipes
do not validate a market schedule. For bar data, validate the declared grid once with
`compute_features(data, [], feed_spec=...)` before deriving returns/features, or
explicitly choose `calendar=None` for observation rows. Recipe output masks are not
dated universe/knowledge selection or final output screens. Named fields stay explicit;
for example, select `factor_expression("BollingerBands", window_length=20)["middle"]`.

### Eligible cross-sectional transforms and boolean windows

`features.selection.cross_section` computes `rank`, `top`, `bottom`, `demean`,
`zscore`, `winsorize`, `quantiles`, `percentile_between` or `peer_count` separately
at each supplied timestamp and optional classification group. Its `mask` selects
the input population before the transform. A later dataframe filter is a final
screen and does not change those inputs. Unknown groups, nulls and NaNs are
excluded. Map legacy classifier sentinels such as -1 to null explicitly.

Ranks support ordinal/min/max/dense/average ties. Ordinal ties and top/bottom
selection use ascending stable entity IDs, independently of input row order.
Supply unique timestamp/entity keys; display symbols are not stable identities.
Z-scores use population standard deviation by default (`ddof=0`), not a fitted
training scaler. Constant groups have undefined z-scores. Typed local differences
retain adjacent large integer ticks, and scaled moments avoid variance overflow
or underflow for representable outputs.

<!-- ml4t-exec -->
```python
from datetime import date
import polars as pl
from ml4t.engineer.features.selection import cross_section

signals = pl.DataFrame({
    "timestamp": [date(2025, 7, 7)] * 4, "asset": ["D", "B", "A", "C"],
    "value": [3., 1., 1., 999.], "eligible": [True, True, True, False],
})
ranked = signals.with_columns(
    cross_section("value", method="rank", mask="eligible").alias("rank"),
    cross_section("value", method="bottom", n=1, mask="eligible").alias("selected"),
)
assert ranked["rank"].to_list() == [3., 2., 1., None]
assert ranked.filter("selected")["asset"].to_list() == ["A"]
```

Winsorization uses the legacy floor/ceil order-statistic cutoffs, not linear
percentile clipping. Quantile bins and inclusive percentile filters use linear
edges. Duplicate quantile edges raise instead of silently changing the number of
bins. Native quantile labels are zero-based integers with null missing values.
These feature selectors do not invoke Diagnostic's analytical factor cleaner.

`features.selection.rolling_filter` supplies full-row `any`, `all` and `at_least`
windows. Apply `.over(entity_col)` to isolate histories. Missing predicates count
false; leading incomplete windows are null. `is_missing` recognizes native null
and floating NaN, without treating infinity or a classifier sentinel as missing.
Ordinary Polars arithmetic, comparisons, boolean operators and `is_in` cover the
corresponding elementwise factor/filter/classifier operations. None of these
expressions selects Data revisions or establishes availability by itself.

### Decision-time feature preparation

`features.preparation.prepare_feature_panel` consumes already prepared native
histories and supplied universe facts. Data or the caller selects source revisions
and adjustment perspective first. Engineer validates keys, preserves pre-entry
warmup and computes feature eligibility at each requested decision. It does not
choose the latest revision, apply corporate actions or fetch provider history.

Supply a canonical `FeedSpec` instance. For a canonical serialized handoff, restore
it with `FeedSpec.from_mapping(...)` once. Unresolved config objects are rejected;
explicit column arguments cannot conflict with the declaration. A named calendar
requires the existing complete bar-grid declaration before any return/feature is
derived. Explicit `calendar=None` keeps observation-row windows.

`decisions` has the observation cutoff, a separate aware `decision_time` and an
explicit decision-effective universe key. The default `universe_time_col` selects
`effective_session`, a supplied session identifier. It need not equal the prior
price session used for a morning decision. For instant-effective universes, set
`universe_time_col` to the explicitly supplied instant column, such as
`decision_time`. No effective session is inferred from a UTC date or from prices.
Universe rows have stable entity keys, `effective_from`, exclusive
`effective_to` (null means open), aware `available_at`, nullable boolean `member`
and optional classification columns. Effective intervals use that declared universe
clock. Date identifiers stay dates; instant comparisons preserve physical timezone
equivalence and ns boundaries. Missing/unknown facts never imply
membership. Overlapping known active facts fail with an upstream-resolution error.

<!-- ml4t-exec -->
```python
from datetime import UTC, date, datetime
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.features.preparation import prepare_feature_panel
from ml4t.engineer.features.rolling import finite_mean

days = [date(2025, 7, d) for d in (7, 8, 9)]
history = pl.DataFrame({
    "timestamp": days, "asset": ["A"] * 3, "close": [1., 2., 3.],
    "available_at": [datetime(2025, 7, d.day, 16, tzinfo=UTC) for d in days],
})
decisions = pl.DataFrame({
    "timestamp": [days[-1]], "decision_time": [datetime(2025, 7, 9, 17, tzinfo=UTC)],
    "effective_session": [days[-1]],
})
universe = pl.DataFrame({
    "asset": ["A"], "effective_from": [days[-1]],
    "effective_to": pl.Series([None], dtype=pl.Date), "member": [True],
    "available_at": [datetime(2025, 7, 8, 12, tzinfo=UTC)],
})
result = prepare_feature_panel(
    history.lazy(), features={"average": finite_mean("close", period=3)},
    lookbacks={"average": 3}, input_columns=["close"],
    decisions=decisions, universe=universe, price_basis="raw",
    feed_spec=FeedSpec(time_contract_version=2, calendar=None, entity_col="asset"),
).collect()
assert result["average"].to_list() == [2.]
assert result["eligible"].to_list() == [True]
```

This asset enters the universe on July 9 but retains July 7-8 warmup. Expressions
are bare per-asset expressions; the preparer applies entity grouping. Declare each
finite row lookback and all input columns. Only complete, known, usable required
windows become eligible. Unavailable or input-masked rows remain as null inputs,
not shorter windows. A missing endpoint does not carry an earlier feature forward.
For kernels with deliberate omission rules, select
`missing_input_policy="known_missing"` to allow confirmed null/NaN or input-masked
values within the full original window. Every row still needs a publication time
known by the decision. Unknown/future rows and infinite inputs remain ineligible.
This option neither adds pre-listing rows nor fills absent scheduled bars; consume
an explicitly supplied complete panel when that is what the original loader emits.
Undefined/nonfinite outputs, including undefined fields in regression Structs,
cannot become eligible. The output retains the observation/decision clocks,
conservative contributing availability, membership, eligibility, the declared
universe key and price basis. Membership is evaluated at that explicit key using
only facts known at the decision, not at the historical price cutoff.

For per-decision restated histories, Data prepares each perspective and the caller
adds an aware snapshot column. Pass its name as `history_decision_time_col`; the
preparer requires an exact matching snapshot for each decision. Without that
column, duplicate history keys fail. It never selects or forward-fills a source
revision/perspective implicitly. Preserve the Data snapshot's provenance alongside
the resulting table.

`input_mask` affects history before computation. Membership does not erase
pre-entry history. `screen` runs after all features are computed. Alternatively,
apply `cross_section(..., timestamp_col="decision_time", mask="eligible")` to the
output, then filter final selections. If one instant has multiple declared universe
contexts, include their universe key in the group keys too. Cross-sectional masks determine the contemporary population and are
not a substitute for information-time validation.

Owned `FeatureSchema` records optional `available_at_col`, `decision_time_col` and
`eligible_col`, plus `universe_time_col`; `entity_col` preserves one column name
or a list of compound identity columns. `FeatureDefinition` records the canonical `feed_spec`,
`window_unit`, `price_basis`, finite `lookbacks` and optional `missing_input_policy`, with existing source-artifact
lineage and join keys. `FeatureSpec.to_dict()` preserves omitted calendar versus
explicit null; unused fields leave the old wire keys unchanged. Store actual
typed frames separately, for example with Parquet. These records do not select
source revisions or add a mandatory dataframe wrapper.

### Paired relationships and supplied event dates

`features.paired.rolling_pearson`, `rolling_spearman`, `rolling_beta` and
`rolling_regression` take explicit dependent/independent columns or expressions.
Align their actual time/entity keys before computing and apply `.over(entity_col)`
for asset-isolated windows. `period` counts original rows. `allowed_missing` is a
count, not a percentage; a missing pair is removed from both sides without
compressing the timeline. At least two valid pairs remain necessary. Spearman
recomputes average tie ranks inside each window. Pearson/beta/regression propagate
undefined nonmissing infinities. Constant independent regression windows produce
NaN fields instead of aborting the whole asset panel.

Regression returns a Struct with `alpha`, `beta`, `r_value`, `p_value`, `stderr`.
Its inferential fields require the optional `stats` extra (`uv add
'ml4t-engineer[stats]'`). The other paired methods and base imports do not need SciPy.
This is predictive rolling regression on supplied observations, not trend-on-time
regression or portfolio attribution.

For price panels, `features.paired.target_return_features` performs the exact
time/identity join and derives returns before applying the paired statistic.
Declare the target as a mapping of physical identity columns to stable values,
including all components for a compound identity. `price_basis` records the
supplied Data/caller declaration; this function does not adjust prices.

The default `index_mode="common_grid"` temporarily retains absent quotes on the
supplied union clock through both return derivation and rolling calculation.
`per_asset` instead counts each asset's own observations. Neither mode fills
quotes, compresses missing pairs out of rolling windows, or emits synthetic rows.
A named calendar validates the original declared grid before calculating any
return. Explicit `calendar=None` selects observation rows. These finite statistics
support observation/expected-bar units, not session or elapsed windows.

<!-- ml4t-exec -->
```python
from datetime import date
import math
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.features.paired import target_return_features

days = [date(2025, 7, day) for day in (7, 8, 9, 10)]
prices = pl.DataFrame({
    "timestamp": days * 2, "asset": ["A"] * 4 + ["B"] * 4,
    "close": [100., 120., 80., 112., 100., 110., 99., 118.8],
})
result = target_return_features(
    prices.lazy(), target={"asset": "B"}, period=3, method="beta",
    price_basis="raw", feed_spec=FeedSpec(time_contract_version=2, calendar=None),
).collect()
assert math.isclose(result.filter(pl.col("asset") == "A")["target_beta"][-1], 52 / 21)
assert result.select(prices.columns).equals(prices)
assert result["target_available_at"].null_count() == prices.height
```

Added return columns expose the actual paired inputs. `_available_at` is the latest
publication instant across both assets' price dependencies in the full window.
Without complete supplied timezone-aware `available_at_col` values it stays
unknown. Observation timestamps are not fabricated publication times. The caller
must still enforce decision-time eligibility before using these features.

`compat.zipline.return_factor_features` maps `RollingPearsonOfReturns`,
`RollingSpearmanOfReturns`, `RollingLinearRegressionOfReturns` and `SimpleBeta`
onto this native preparation. It translates `returns_length=N` to an N-1-price
lag, accepts the corresponding correlation/regression length, and converts
SimpleBeta's allowed-missing percentage to a count. Its optional output mask
does not remove the target's prices or warmup history. Pass stable target IDs,
not opaque Zipline Asset objects. This recipe returns a native frame, not a Term.

<!-- ml4t-exec -->
```python
import math
from importlib.util import find_spec
import polars as pl
from ml4t.engineer.features.paired import rolling_regression

paired = pl.DataFrame({"market": [1.0, 2.0, 3.0, 4.0], "feature": [1.0, 2.0, 2.0, 4.0]})
if find_spec("scipy") is None:
    try:
        rolling_regression("feature", "market", period=4)
    except ImportError as error:
        assert "ml4t-engineer[stats]" in str(error)
    else:
        raise AssertionError("Inferential regression must require its optional dependency")
else:
    result = paired.lazy().select(
        rolling_regression("feature", "market", period=4).alias("regression")
    ).unnest("regression").collect()
    assert math.isclose(result["beta"][3], 0.9, rel_tol=1e-14)
    assert math.isclose(result["stderr"][3], math.sqrt(0.07), rel_tol=1e-14)
    assert math.isclose(result["p_value"][3], 1 - 9 / math.sqrt(95), rel_tol=1e-14)
```

`features.events.event_business_days` counts Monday through Friday, including
exchange holidays, with the end date excluded. Supply separate event/reference
Date labels and timezone-aware publication/decision instants. Unknown publication
or publication after the decision yields null. This method neither selects an
event provider nor infers exchange sessions. `direction="until"` permits a known
future event; `direction="since"` uses the same supplied date with reversed endpoints.

<!-- ml4t-exec -->
```python
from datetime import date, datetime, timezone
import polars as pl
from ml4t.engineer.features.events import event_business_days

events = pl.DataFrame({
    "event_date": [date(2025, 7, 4)],
    "session": [date(2025, 7, 7)],
    "known_at": [datetime(2025, 7, 1, tzinfo=timezone.utc)],
    "decision": [datetime(2025, 7, 7, 14, tzinfo=timezone.utc)],
})
result = events.select(event_business_days(
    "event_date", reference_date_col="session", known_at_col="known_at",
    decision_time_col="decision", direction="since",
).alias("weekdays"))
assert result["weekdays"][0] == 1.0  # July 4 is a weekday, despite the NYSE closure.
```

### Calendar and bar grid

The calendar-aware default targets the next minor version (0.2), separate from the 0.1.7 patch.
The shared temporal contract requires published `ml4t-specs>=0.2.0,<0.3.0`, also declared
in built package metadata. Specs 0.1.5 does not provide this API. Engineer's development
branch uses the published dependency; the temporal migration is not part of the 0.1.7 patch.

Calendar selection
uses the explicit `calendar` argument, then declared feed metadata, then NYSE. Explicit
`calendar=None` disables schedule interpretation. A named `"24/7"` calendar instead defines
expected bars and detects outages, including on weekends.

Calendar-aware execution requires a shared `FeedSpec(time_contract_version=2)` with declared
frequency and timestamp meaning. Fixed intraday time bars require an explicit bar-grid policy,
anchored at each tradable interval's open. The selected schedule supplies holidays, early closes
and breaks. Naive instants need a declared timezone; aware instants declare their timezone.

Daily bars instead declare `timestamp_semantics="session_label"`, `data_frequency="1d"` and
`bar_type="time"`. Do not supply intraday anchor, label-side or partial-bar fields for session
labels. Date values and datetime representations identify the displayed session date, without
timezone conversion. Validation checks every scheduled session between each asset's first and
last label, including completely missing sessions. A named `24/7` schedule includes weekends;
an NYSE schedule does not. No prices are filled or rows inserted. A daily frequency alone does
not turn `event_time` instants into session labels. Temporal feature windows below use the
same completeness checks; choosing a unit does not permit missing scheduled observations.

<!-- ml4t-exec -->
```python
from datetime import date

import polars as pl
from ml4t.engineer import compute_features
from ml4t.specs import FeedSpec

close = [100.0, 103.0, 106.0, 109.0]
daily_prices = pl.DataFrame({
    "timestamp": [date(2025, 7, day) for day in (3, 7, 8, 9)],
    "open": close, "high": close, "low": close, "close": close, "volume": 1.0,
})
daily_feed = FeedSpec(
    time_contract_version=2, calendar="NYSE", data_frequency="1d",
    timestamp_semantics="session_label", bar_type="time",
)
checked = compute_features(daily_prices, [], feed_spec=daily_feed)
returns = checked.with_columns(pl.col("close").log().diff().alias("returns"))
features = compute_features(returns, [{"name": "sma", "params": {"period": 2}}], feed_spec=daily_feed)
assert features["sma"].to_list() == [None, 101.5, 104.5, 107.5]
assert features["timestamp"].equals(daily_prices["timestamp"])
```

`CryptoCalendar` also changes in the next minor version. Install the `calendars` extra
to use its authoritative `24/7` schedule. It enumerates UTC daily session opens, not
hourly observations. `next_open` and `previous_close` return strictly adjacent daily
session boundaries; a maintenance break or reopen is not a new session boundary.
The optional `(start_hour, end_hour)` maintenance window uses UTC, is half-open and
may cross midnight, such as `(23, 1)`. Session duration excludes that downtime.
Annotations use UTC session dates and times regardless of an instant's display timezone;
naive instants mean UTC. Existing `CRYPTO_24_7` names and maintenance arguments remain
accepted. A concrete calendar object cannot represent row mode: use `calendar=None`
on the consuming feature API instead.

Validate prices before deriving returns. An empty feature list performs the same grid
validation and chronological sorting without adding feature columns:

<!-- ml4t-exec -->
```python
from datetime import datetime, timedelta, timezone

import polars as pl
from ml4t.engineer import compute_features
from ml4t.specs import FeedSpec

contract = FeedSpec(
    time_contract_version=2,
    calendar="24/7",
    entity_col="symbol",
    data_frequency="1h",
    timezone="UTC",
    bar_type="time",
    timestamp_semantics="bar_close",
    bar_grid_anchor="session_open",
    bar_label="right",
    partial_bar_policy="include",
)
close = [100.0 + i for i in range(8)]
prices = pl.DataFrame({
    "timestamp": [datetime(2025, 7, 11, tzinfo=timezone.utc) + timedelta(hours=i)
                  for i in range(8)],
    "symbol": "BTC",
    "open": close, "high": close, "low": close, "close": close, "volume": 1.0,
})
checked = compute_features(prices, [], feed_spec=contract)
with_returns = checked.with_columns(
    pl.col("close").log().diff().over("symbol").alias("returns")
)
result = compute_features(with_returns, [{
    "name": "realized_volatility",
    "params": {"period": 3, "annualize": False},
}], feed_spec=contract.to_dict(), window_unit="expected_bar")
assert result.height == prices.height
assert result["realized_volatility"][3] is not None
```

Missing, duplicate or off-grid bars fail before any feature executes. The validator neither
fills prices nor changes row counts. If normalization is needed, explicitly construct the
intended schedule grid with null prices first, then derive returns. Reindexing an already-derived
return series leaves a cross-gap return intact; finite returns adjacent to missing prices are
rejected. Completeness is checked between each asset's observed endpoints, not outside unknown
dataset coverage.

Use the canonical shared `to_dict()` methods for JSON/YAML handoffs. Omitted selectors and
explicit null remain different in version 2. Legacy version-1 null metadata is unspecified,
not an intentional opt-out; migrate it explicitly or pass `calendar=None` for row mode.
This does not change legacy Specs bridges or the numerical feature kernels.

`window_unit="observation"` belongs to row mode; `"expected_bar"` belongs to a validated grid.
Neither means sessions or elapsed hours. A shortened final bar still counts as one expected
bar when `partial_bar_policy="include"`. Annualization remains a separate convention: for
example, `realized_volatility(trading_periods=252)` retains its documented numerical default.

### Session and elapsed feature windows

`sma`, `stddev` and `realized_volatility` accept `window_unit="session"` or `"elapsed"`.
They retain original bars and row counts. They do not first aggregate to session closes.
For sessions, `params.period=N` includes the current session's observations through the
current row and the preceding N-1 scheduled sessions. Earlier rows of the first N-1
observed sessions are null. July 4 holidays and lunch breaks do not create extra sessions.
A one-session average of intraday prices100/110 is therefore100 at the first row and105
at the second row, not the session's eventual closing price110 at both rows.

Elapsed periods use a positive duration such as `params.period="2h"` and the physical
UTC interval `(timestamp - duration, timestamp]`, including through DST changes.
Outputs remain null until observed coverage spans the requested duration. A half-hour
final bar can put three observations inside two elapsed hours. Daily `session_label`
identifiers cannot define elapsed windows; they do support session-count windows.
Windows with insufficient samples for the numerical statistic also remain null.
Annualization parameters and all other supported feature parameters retain their
numerical definitions. Recursive EMA/RSI and other unqualified window definitions
reject temporal units explicitly rather than silently resetting or reseeding.

<!-- ml4t-exec -->
```python
from datetime import date, datetime, timedelta, timezone

import polars as pl
from ml4t.engineer import compute_features
from ml4t.specs import FeedSpec

daily = FeedSpec(time_contract_version=2, calendar="NYSE", data_frequency="1d",
                 timestamp_semantics="session_label", bar_type="time")
values = [100.0, 101.0, 104.0]
prices = pl.DataFrame({
    "timestamp": [date(2025, 7, day) for day in (2, 3, 7)],
    "open": values, "high": values, "low": values, "close": values, "volume": 1.0,
})
result = compute_features(prices, [{"name": "sma", "params": {"period": 2}}],
                          feed_spec=daily, window_unit="session")
assert result["sma"].to_list() == [None, 100.5, 102.5]

hourly = FeedSpec(time_contract_version=2, calendar="24/7", data_frequency="1h",
                  timestamp_semantics="event_time", timezone="UTC", bar_type="time",
                  bar_grid_anchor="session_open", bar_label="left", partial_bar_policy="include")
prices = prices.with_columns(pl.Series("timestamp", [
    datetime(2025, 7, 5, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(3)
]))
result = compute_features(prices, [{"name": "sma", "params": {"period": "2h"}}],
                          feed_spec=hourly, window_unit="elapsed")
assert result["sma"].to_list() == [None, None, 102.5]
```
No calendar-derived annualization factor is silently substituted.

### Equity schedule helpers

`ml4t.engineer.core.EquityCalendar(calendar="NYSE")` uses the same authoritative
schedule adapter as labeling and grid validation. `exchange` and the existing positional
selector remain compatible aliases; contradictory selectors fail. Its naive timestamps use
the selected exchange timezone unless `timezone` explicitly overrides it. Ambiguous and
nonexistent naive local times are rejected; pass an aware instant to disambiguate them.

Membership excludes scheduled breaks. `sessions_between` returns session opens inside the
inclusive instant interval. `session_duration` takes a session-label date, not an event
instant, and returns tradable time excluding breaks, with zero duration on holidays.
`add_session_info` uses the authoritative session label, including overnight sessions,
and leaves that label null off-session. Its clock time uses the exchange timezone;
the input timestamps remain unchanged. Boundary lookups return UTC-aware
instants and fail explicitly if no scheduled boundary exists within 366 days.

The next minor version removes the approximate weekday fallback when the calendar provider
is absent or a schedule is empty. Install `ml4t-engineer[calendars]` for these helpers.
An `EquityCalendar` object requires a named schedule; it cannot represent calendar-free
execution. Use `compute_features(..., calendar=None)` for observation-row features.

> **Book**: [The ml4t Library Ecosystem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) calls `compute_features()` with names and parameter dictionaries, and compares the library RSI with a manual implementation.

## Category Reference

### Momentum (31 indicators)

Price momentum and oscillator indicators. Most produce bounded (normalized) output suitable for direct ML use.

| Name | Description | TA-Lib | Normalized | Default Period |
|------|-------------|--------|------------|----------------|
| `rsi` | Relative Strength Index | Yes | 0-100 | 14 |
| `macd` | Moving Average Convergence/Divergence | Yes | No | 12/26 |
| `stochastic` | Stochastic Oscillator (%K, %D) | No | 0-100 | 14/3/3 |
| `stochf` | Fast Stochastic | Yes | 0-100 | 5/3 |
| `stochrsi` | Stochastic RSI | Yes | 0-100 | 14 |
| `cci` | Commodity Channel Index | Yes | ~-200 to 200 | 14 |
| `willr` | Williams %R | Yes | -100 to 0 | 14 |
| `adx` | Average Directional Index | Yes | 0-100 | 14 |
| `adxr` | ADX Rating | Yes | 0-100 | 14 |
| `dx` | Directional Movement Index | Yes | 0-100 | 14 |
| `plus_di` | Plus Directional Indicator | Yes | 0-100 | 14 |
| `minus_di` | Minus Directional Indicator | Yes | 0-100 | 14 |
| `mfi` | Money Flow Index | Yes | 0-100 | 14 |
| `roc` | Rate of Change | Yes | No | 10 |
| `rocp` | Rate of Change (%) | Yes | No | 10 |
| `mom` | Momentum | Yes | No | 10 |
| `trix` | Triple Exponential Average | Yes | No | 30 |
| `cmo` | Chande Momentum Oscillator | Yes | -100 to 100 | 14 |
| `ultosc` | Ultimate Oscillator | Yes | 0-100 | 7/14/28 |
| `bop` | Balance of Power | Yes | -1 to 1 | - |
| `imi` | Intraday Momentum Index | No | 0-100 | 14 |
| `aroon` | Aroon (up/down) | Yes | 0-100 | 14 |
| `aroonosc` | Aroon Oscillator | Yes | -100 to 100 | 14 |
| `apo` | Absolute Price Oscillator | Yes | No | 12/26 |
| `ppo` | Percentage Price Oscillator | Yes | No | 12/26 |
| `sar` | Parabolic SAR | Yes | No | 0.02/0.2 |

> **Book**: [Price and Volume Feature Families](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/01_price_volume_features.ipynb) calls Engineer for selected indicators and derives others manually. [ETFs: Feature Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/03_financial_features.ipynb) calls individual Engineer feature functions in a case-study pipeline.

### Trend (10 indicators)

Moving averages that produce price-scale outputs. Require preprocessing for ML models.

| Name | Description | TA-Lib | Default Period |
|------|-------------|--------|----------------|
| `sma` | Simple Moving Average | Yes | 20 |
| `ema` | Exponential Moving Average | Yes | 20 |
| `wma` | Weighted Moving Average | Yes | 20 |
| `dema` | Double Exponential MA | Yes | 20 |
| `tema` | Triple Exponential MA | Yes | 20 |
| `t3` | Triple Exponential T3 | Yes | 5 |
| `kama` | Kaufman Adaptive MA | Yes | 30 |
| `trima` | Triangular MA | Yes | 20 |
| `midpoint` | Midpoint over period | Yes | 14 |
| `donchian_channels` | Donchian Channels (highest high/lowest low) | No | 20 |

### Volatility (15 indicators)

Volatility estimators ranging from simple (ATR) to advanced (GARCH). Includes range-based estimators that are more efficient than close-to-close.

| Name | Description | TA-Lib | Normalized |
|------|-------------|--------|------------|
| `atr` | Average True Range | Yes | No |
| `natr` | Normalized ATR (% of price) | Yes | 0-100 |
| `trange` | True Range | Yes | No |
| `bollinger_bands` | Bollinger Bands (upper/middle/lower) | Yes | No |
| `yang_zhang_volatility` | Yang-Zhang (overnight + intraday) | No | No |
| `parkinson_volatility` | Parkinson range-based | No | No |
| `garman_klass_volatility` | Garman-Klass OHLC-based | No | No |
| `rogers_satchell_volatility` | Rogers-Satchell drift-independent | No | No |
| `realized_volatility` | Standard deviation of returns | No | No |
| `ewma_volatility` | EWMA of variance | No | No |
| `garch_forecast` | GARCH(1,1) conditional volatility | No | No |
| `conditional_volatility_ratio` | Up-market vs down-market vol ratio | No | No |
| `volatility_percentile_rank` | Current vol vs historical distribution | No | 0-100 |
| `volatility_of_volatility` | Second-order volatility measure | No | No |
| `volatility_regime_probability` | High/low vol regime probability | No | No |

**Efficiency ranking**: Yang-Zhang > Garman-Klass ~ Rogers-Satchell > Parkinson > Close-to-Close. See Molnar (2012) for theoretical efficiency ratios.

> **Book**: [Price and Volume Feature Families](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/01_price_volume_features.ipynb) calls Engineer's volatility functions and compares selected estimators on ETF data.

### Microstructure (15 indicators)

Market microstructure features from De Prado (2018) and empirical market microstructure literature.

| Name | Description |
|------|-------------|
| `kyle_lambda` | Kyle's Lambda (price impact coefficient) |
| `amihud_illiquidity` | Amihud illiquidity ratio |
| `roll_spread_estimator` | Roll implied bid-ask spread |
| `realized_spread` | Realized spread |
| `effective_tick_rule` | Effective tick rule classification |
| `order_flow_imbalance` | Order flow imbalance |
| `price_impact_ratio` | Price impact ratio |
| `volume_weighted_price_momentum` | Volume-weighted price momentum |
| `bid_ask_imbalance` | Bid-ask imbalance (normalized -1 to 1) |
| `book_depth_ratio` | Book depth ratio (normalized 0 to 1) |
| `quote_stuffing_indicator` | Quote stuffing detection |
| `trade_intensity` | Trade intensity |
| `volume_at_price_ratio` | Volume at price ratio |
| `volume_synchronicity` | Volume synchronicity |
| `weighted_mid_price` | Weighted mid price |

> **Book**: [Microstructure Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/02_microstructure_features.ipynb) calls Engineer's tick-rule and liquidity functions while teaching the wider workflow. [NASDAQ-100 Microstructure: Feature Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/nasdaq100_microstructure/03_financial_features.ipynb) calls `amihud_illiquidity` and implements other case-study features separately.

### ML Features (14 indicators)

Features designed specifically for machine learning pipelines.

| Name | Description | Normalized |
|------|-------------|------------|
| `create_lag_features` | Multiple lag columns at once | No |
| `cyclical_encode` | Cyclical time encoding (sin/cos) | No |
| `fourier_features` | Fourier transform features | No |
| `rolling_entropy` | Shannon entropy | 0-10 |
| `rolling_entropy_lz` | Lempel-Ziv entropy | 0-10 |
| `rolling_entropy_plugin` | Plugin entropy estimator | 0-10 |
| `percentile_rank_features` | Rank-based normalization | 0-100 |
| `interaction_features` | Feature interaction terms | No |
| `multi_horizon_returns` | Returns at multiple horizons | No |
| `directional_targets` | Directional movement targets | No |
| `volatility_adjusted_returns` | Returns scaled by volatility | No |
| `regime_conditional_features` | Regime-conditional transforms | No |
| `time_decay_weights` | Exponential time decay | No |
| `ffdiff` | Fractional differencing | No |

> **Book**: [Slow Features and Context](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/04_fundamentals_macro_calendar.ipynb) calls Engineer's `cyclical_encode` and teaches the point-in-time data joins manually.

### Risk (6 indicators)

Risk and risk-adjusted return metrics.

| Name | Description | Normalized |
|------|-------------|------------|
| `maximum_drawdown` | Maximum drawdown | No |
| `downside_deviation` | Downside volatility | 0-2 |
| `tail_ratio` | Right tail / left tail ratio | 0-10 |
| `higher_moments` | Skewness and kurtosis | No |
| `risk_adjusted_returns` | Sharpe, Sortino, Calmar, Omega | No |
| `ulcer_index` | Ulcer Index (drawdown-based risk) | No |

`risk_adjusted_returns` accepts `trading_periods` to match the annualization
frequency to the data, such as `252` for daily observations or `52` for weekly
observations. Its annual `risk_free_rate` is converted to a per-period threshold
using the same value.

### Cross-Asset (10 functions)

Multi-asset relationship features. These are standalone functions in `ml4t.engineer.features.cross_asset` rather than registry entries, since they require two or more price series as input.

| Function | Description |
|----------|-------------|
| `rolling_correlation` | Rolling Pearson correlation |
| `beta_to_market` | Rolling beta vs market index |
| `correlation_regime_indicator` | Low/medium/high correlation regimes |
| `lead_lag_correlation` | Lead-lag cross-correlation |
| `multi_asset_dispersion` | Cross-sectional return dispersion |
| `correlation_matrix_features` | Mean/min/max of correlation matrix |
| `relative_strength_index_spread` | RSI spread between two assets |
| `volatility_ratio` | Volatility ratio between assets |
| `co_integration_score` | Rolling cointegration score |
| `cross_asset_momentum` | Rank-based cross-asset momentum |

These are called directly (not via `compute_features`) since they require multi-asset DataFrames.

> **Book**: [Structural and Cross-Instrument Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/03_structural_cross_instrument_features.ipynb) calls `beta_to_market`. [Panel Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/09_model_based_features/14_panel_features.ipynb) calls several cross-asset functions and combines them with manual analysis.

### Regime (4 indicators)

Market regime detection features. All produce bounded outputs suitable for direct ML use.

| Name | Description | Range |
|------|-------------|-------|
| `hurst_exponent` | Hurst exponent (R/S analysis) | 0-1 |
| `choppiness_index` | Market choppiness | 0-100 |
| `fractal_efficiency` | Price path efficiency | 0-1 |
| `trend_intensity_index` | Trend strength | 0-100 |


### Statistics (14 indicators)

Statistical features including TA-Lib standard and rolling distribution metrics.

| Name | Description | TA-Lib | Normalized |
|------|-------------|--------|------------|
| `stddev` | Standard Deviation | Yes | No |
| `var` | Variance | Yes | No |
| `avgdev` | Average Deviation | No | No |
| `linearreg` | Linear Regression Value | Yes | No |
| `linearreg_slope` | Linear Regression Slope | Yes | No |
| `linearreg_angle` | Linear Regression Angle | Yes | No |
| `linearreg_intercept` | Linear Regression Intercept | Yes | No |
| `tsf` | Time Series Forecast | Yes | No |
| `coefficient_of_variation` | Rolling coefficient of variation | No | 0-10 |
| `variance_ratio` | Variance ratio test | No | 0-5 |
| `rolling_cv_zscore` | Cross-validated z-score | No | -10 to 10 |
| `rolling_drift` | Rolling drift estimate | No | -10 to 10 |
| `rolling_kl_divergence` | KL divergence vs reference | No | 0-10 |
| `rolling_wasserstein` | Wasserstein distance | No | No |

### Price Transform (5 indicators)

| Name | Description | TA-Lib |
|------|-------------|--------|
| `avgprice` | Average Price (O+H+L+C)/4 | Yes |
| `typprice` | Typical Price (H+L+C)/3 | Yes |
| `medprice` | Median Price (H+L)/2 | Yes |
| `wclprice` | Weighted Close (H+L+2C)/4 | Yes |
| `midprice` | Midpoint Price (H+L)/2 | Yes |

### Volume (3 indicators)

| Name | Description | TA-Lib |
|------|-------------|--------|
| `obv` | On Balance Volume | Yes |
| `ad` | Accumulation/Distribution | Yes |
| `adosc` | A/D Oscillator | Yes |

> **Book**: [ETFs: Feature Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/03_financial_features.ipynb) calls Engineer's volume, momentum, trend, volatility, and regime feature functions.

### Math (3 indicators)

O(n) rolling operations using monotonic deque.

| Name | Description | TA-Lib |
|------|-------------|--------|
| `maximum` | Rolling maximum | Yes |
| `minimum` | Rolling minimum | Yes |
| `summation` | Rolling sum | Yes |

### Fractional Differencing (4 functions)

See the dedicated [Fractional Differencing guide](fractional-differencing.md) for the full workflow.

## Feature Discovery

The `FeatureCatalog` provides filtering and full-text search over all 120 features:

```python
from ml4t.engineer import feature_catalog

# List by category
momentum = feature_catalog.list(category="momentum")

# Filter by multiple criteria
ml_ready = feature_catalog.list(normalized=True, ta_lib_compatible=True)

# Full-text search
results = feature_catalog.search("volatility estimator")
# Returns: [("parkinson_volatility", 0.65), ("garman_klass_volatility", 0.45), ...]

# Detailed feature info
info = feature_catalog.describe("yang_zhang_volatility")
# {'name': 'yang_zhang_volatility', 'category': 'volatility', ...}

# List all categories
print(feature_catalog.categories())
# ['cross_asset', 'math', 'microstructure', 'ml', 'momentum', ...]

# List all tags
print(feature_catalog.tags())
```

See the dedicated [Feature Discovery guide](discovery.md) for complete examples.

> **Book**: [The ml4t Library Ecosystem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) inspects registry metadata for RSI, ATR, and Garman-Klass before computing features.

## YAML Configuration

For reproducible feature pipelines, store configurations in YAML:

```yaml
# features.yaml
features:
  - name: rsi
    params:
      period: 14

  - name: macd
    params:
      fast_period: 12
      slow_period: 26

  - name: bollinger_bands
    params:
      period: 20
      nbdevup: 2.0
      nbdevdn: 2.0

  - name: yang_zhang_volatility
```

Load row-based features with `compute_features(df, "features.yaml", calendar=None)`.
The YAML format supports version comments and parameter documentation inline.

## Input Requirements

### OHLCV DataFrame

Most features expect a DataFrame with standardized column names (lowercase):

| Column | Type | Required By |
|--------|------|-------------|
| `timestamp` | datetime | Required for calendar-aware grids; optional in observation-row mode |
| `asset_id` | string | Optional panel grouping, auto-detected when present |
| `open` | float | OHLCV, OHLC features |
| `high` | float | OHLCV, OHLC, HLC, HL features |
| `low` | float | OHLCV, OHLC, HLC, HL features |
| `close` | float | All features |
| `volume` | float | OHLCV, volume features |
| `returns` | float | Return-based features (auto-computed if missing) |

Features declare their `input_type` metadata (e.g., `"OHLCV"`, `"close"`, `"returns"`), and `compute_features` validates that required columns are present.

### Missing Columns

If a feature requires a column that's missing, `compute_features` raises a clear error:

```
ValueError: Feature 'mfi' requires column 'volume' (input_type='OHLCV') but it was not found.
```

## Custom Parameters

Override default parameters per feature:

```python
# Check defaults
from ml4t.engineer.core.registry import get_registry
meta = get_registry().get("rsi")
print(meta.parameters)  # {'period': 14}

# Override
result = compute_features(df, [{"name": "rsi", "params": {"period": 20}}], calendar=None)
```

Invalid parameters raise `ValueError` with the valid parameter names.

## Performance

- **Polars-native**: All computations use Polars expressions for automatic parallelism
- **Numba JIT**: Numerical kernels (volatility estimators, microstructure) are Numba-accelerated
- `compute_features` resolves feature dependencies before execution.
- LazyFrame input remains lazy until you collect the result.
- Measure throughput with your feature set, row count, grouping columns, and hardware.

## See It In The Book

- [The ml4t Library Ecosystem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) calls the registry and `compute_features()`.
- [Price and Volume Feature Families](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/01_price_volume_features.ipynb) combines Engineer calls with manual teaching implementations.
- [ETFs: Feature Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/03_financial_features.ipynb) calls individual Engineer functions in a case-study pipeline.
- [Book Guide](../book-guide/index.md) records the pinned revision and relationship for every selected notebook.

## Next Steps

- Read [Feature Discovery](discovery.md) to choose features through metadata and
  search instead of hardcoding.
- Read [ML Readiness](ml-readiness.md) to separate normalized from non-normalized
  outputs.
- Read [Dataset Builder](dataset-builder.md) when features are ready to move into
  training workflows.
- Use the [API Reference](../api/index.md) for exact function and module locations.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Molnar, P. (2012). Properties of range-based volatility estimators. *International Review of Financial Analysis*.
