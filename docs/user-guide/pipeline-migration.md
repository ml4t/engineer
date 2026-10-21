# Migrate Pipeline feature selection

Port finite factors and dated screens to native Polars expressions. Keep source
history preparation in ML4T Data and order/account processing in ML4T Backtest.
Engineer does not run a Zipline graph or import Zipline in production.

## Replace the calculation and retain its input contract

The original momentum example constructs `RSI().top(3)` and `RSI().bottom(3)` in a
Pipeline. Its pricing loader supplies the previous session's close before the
current session opens. Preserve those different clocks when replacing the graph.

Supply a long price panel with stable IDs, an observation timestamp and actual
publication times. Supply decisions with an observation cutoff, an aware
`decision_time` and an `effective_session` for membership. Supply dated universe
facts with `effective_from`, exclusive `effective_to`, `available_at`, `member`
and any classifications. Unknown membership is not membership. Data selects
source revisions, identities and adjustment perspectives before this boundary.

<!-- ml4t-exec -->
```python
from datetime import UTC, date, datetime
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.compat.zipline import factor_expression
from ml4t.engineer.features.preparation import prepare_feature_panel
from ml4t.engineer.features.selection import cross_section

days = [date(2025, 7, d) for d in (7, 8, 9)]
prices = pl.DataFrame({
    "timestamp": days * 3,
    "asset": [17] * 3 + [23] * 3 + [31] * 3,
    "close": [10., 11., 10., 10., 12., 13., 10., 9., 8.],
    "available_at": [datetime(2025, 7, d.day, 20, tzinfo=UTC) for d in days] * 3,
})
decisions = pl.DataFrame({
    "timestamp": [days[-1]],
    "decision_time": [datetime(2025, 7, 10, 13, tzinfo=UTC)],
    "effective_session": [date(2025, 7, 10)],
})
universe = pl.DataFrame({
    "asset": [17, 23, 31], "effective_from": [date(2025, 7, 10)] * 3,
    "effective_to": pl.Series([None] * 3, dtype=pl.Date),
    "available_at": [datetime(2025, 7, 1, tzinfo=UTC)] * 3,
    "member": [True] * 3,
})
# This small example deliberately counts observation rows, not scheduled bars.
feed = FeedSpec(time_contract_version=2, calendar=None, entity_col="asset")
rsi = factor_expression("RSI", inputs=["close"], window_length=3)
assert isinstance(rsi, pl.Expr)
signals = prepare_feature_panel(
    prices.lazy(), features={"rsi": rsi}, lookbacks={"rsi": 3},
    input_columns=["close"], decisions=decisions, universe=universe,
    price_basis="raw", feed_spec=feed, missing_input_policy="known_missing",
).with_columns(
    cross_section("rsi", method="top", n=1, timestamp_col="decision_time",
                  mask="eligible").alias("longs"),
    cross_section("rsi", method="bottom", n=1, timestamp_col="decision_time",
                  mask="eligible").alias("shorts"),
).filter(pl.col("member").fill_null(False)).collect()
assert signals["rsi"].to_list() == [50., 100., 0.]
assert signals.filter("longs")["asset"].to_list() == [23]
assert signals.filter("shorts")["asset"].to_list() == [31]
assert signals["timestamp"].to_list() == [date(2025, 7, 9)] * 3
assert signals["effective_session"].to_list() == [date(2025, 7, 10)] * 3
```

The default legacy RSI uses 15 prices, so use `window_length=15` and
`lookbacks={"rsi": 15}` for that workflow. Native `rsi` remains Wilder RSI;
the recipe selects the separately named `rsi_cutler` definition. Select `n=3`
to reproduce the original selection count.

For scheduled daily inputs, install `ml4t-engineer[calendars]` and declare
`calendar="NYSE"`, `timestamp_semantics="session_label"`,
`data_frequency="1d"`, and `bar_type="time"` in the feed instead of selecting
row mode. Omitted calendar follows feed metadata, then NYSE. Explicit `None`
disables calendars and does not mean a continuous schedule. Intraday bars also
need timezone, anchor, label side and partial-bar policy. Do not infer them from
the observations. Missing scheduled rows fail before a return or feature is
interpreted as one-bar data.

The `known_missing` option permits a kernel's declared null/NaN omission rules
over a full, supplied row window. It does not invent pre-listing rows or treat
unknown publication as confirmed absence. A loader that emits a complete panel
with explicit pre-listing null cells must hand over those actual cells. Keep the
strict default when every input must be usable. Apply an input mask before the
calculation and a final screen after it. Filtering history to current membership
loses pre-entry warmup. Partition cross-sections by decision time, not the prior
price session; add the universe key if one instant has multiple contexts.

## Port custom terms

Run `uv run python examples/pipeline_migration.py` in the source checkout. The
example includes a finite custom mean, a positive-value custom filter and an
explicit sector-code classifier alongside the momentum screen. Replace a
`CustomFactor.compute` loop with a bare per-entity expression and declare its
finite lookback and all input columns. The preparer supplies entity grouping.
For reusable calculations, use the existing feature decorator and registry.
Replace `CustomFilter` with a Boolean expression, and `CustomClassifier` with a
supplied classification or an explicit native mapping. Retain missing categories
as null; do not convert them into a real category. Select multi-output Struct
fields explicitly. Opaque legacy objects, loaders and graph execution are not
accepted as expressions.

## Adapt presentation only where needed

Keep the native table for subsequent feature or label preparation. For a caller
that still expects a pandas MultiIndex, select `effective_session` and stable
asset IDs as its keys, then convert at this explicit eager boundary. With
PyArrow installed, `signals.select("effective_session", "asset", "longs",
"shorts").to_pandas(date_as_object=True).set_index(["effective_session",
"asset"])` preserves civil Date labels rather than claiming they are instants.
Rename the index levels to `date` and `asset` if that is the caller's contract.
For instant keys, preserve their timezone and nanoseconds. Never go through
Python `datetime` values to reconstruct a nanosecond index.

If a retained caller needs Zipline Asset objects, map stable IDs through that
caller's AssetFinder explicitly. Do not stringify compound IDs or introduce a
Zipline dependency to native calculations. Native compound entity columns and
Struct IDs can remain unchanged; an old pandas caller must declare its own
hashable compound-key representation. This is a caller presentation edit, not
an automatic emulation of `Pipeline.run_pipeline`.

## Preserve prepared-history metadata

Consume `AdjustedHistory.prices` and the canonical feed in its
`metadata["feed_spec"]`. Use the declared price column and price basis, not a
hard-coded raw close. Data's daily frequency declaration is not by itself a
complete Engineer time-bar grid; add the fixture/feed owner's explicit bar-grid
declarations. For per-decision adjustment perspectives, retain the snapshot
decision key and pass `history_decision_time_col`. Do not reuse retrospectively
adjusted history as a causal snapshot. Store Data provenance alongside the
FeatureSpec and typed frame.

## Port the complete preparation workflow

Run `uv run python examples/reloaded_workflows.py` from the source checkout for
precise wide-price conversion, a combined screen and a multi-output custom
factor port. The example captures Data-produced split-adjusted observations
across OLD/NEW symbols for stable asset 17. Its source as-of is July 7, 2025 at
21:00 UTC; its signal-effective session is July 8. It does not claim the later
adjusted history was tradable on earlier observation dates.

| Existing workflow | Native route | Required input or material difference |
|---|---|---|
| Wide prices to trailing returns | `precise_wide_returns` recipe using `market_to_long` and native `returns` | Preserve integer ticks before division. Native output retains the leading null row; legacy simple-return presentation drops the first row. This recipe explicitly counts rows. |
| RSI top/bottom selection | `momentum_screen` in `examples/pipeline_migration.py` | Retain input warmup and decision-effective membership; a screen must not erase feature history. |
| Momentum/volatility/liquidity/sector screen | `snapshot_screen` in `examples/reloaded_workflows.py` | Source-selected split-compatible prices/volume, currency, multiplier, actual publication/completion and dated universe facts. Optional volume must not invalidate independent price-only features. |
| Multi-output CustomFactor | `custom_bands`, then `pl.col("bands").struct.field("mid")` | Each required-input group keeps its own eligibility. Missing band inputs do not silently become usable outputs or remove another feature's history. |
| CustomFilter/CustomClassifier | Typed Boolean/category expressions and supplied dated facts | Rewrite callbacks explicitly; preserve missing categories and known-at times. |
| `term.alias(name)` | `expression.alias(name)` | Rename the native expression/output; no legacy Term object is created. |
| `term.downsample(frequency)` | Explicit sampling sessions and `sampled_values` | Use the full declared exchange schedule, including the preceding sample when output starts mid-period. Carry sampled feature rows, not raw quotes. |
| Factor keys plus prices for analysis | [Native multi-horizon outcomes](labeling.md#prepare-keyed-multi-horizon-outcomes) | Diagnostic owns the Alphalens facade/cleaner. Select the return columns explicitly and retain actual endpoints, availability, units and basis. |
| Pyfolio-style market/loading containers | [Optional input preparation](optional-input-migration.md) | Data/callers supply source revisions, compatible units and knowledge times. Diagnostic owns capacity, exposures and attribution. |

Numeric recipes can replace individual built-in constructor calls. An opaque
Pipeline, custom callback or mutable execution environment requires a native
table/expression port, not a global import replacement. Orders, fills, costs,
account state and strategy delivery remain outside Engineer.

Empyrical Reloaded 0.7.0 with pandas 2.3.3 has container-dependent return behavior:
its pandas `pct_change` path can round away large-integer ticks and implicitly
fill missing prices. Native preparation preserves the integer differences and
does not fill quotes. The qualified integer-ndarray path and independent integer
quotients establish the precision expectation; rounded or implicitly filled
pandas results are not the native correctness target.

The combined example deliberately uses `calendar=None` for its standalone
base-dependency run. Omitting that argument preserves the captured NYSE feed
and validates its declared daily grid with the real calendar extra. Neither
route infers July 4 from missing observations.

<!-- ml4t-exec -->
```python
import runpy

recipes = runpy.run_path("examples/reloaded_workflows.py")
prices, decisions, universe, feed = recipes["captured_snapshot"]()
result = recipes["snapshot_screen"](
    prices, decisions, universe, feed_spec=feed, calendar=None,
)
assert result["asset"].to_list() == [17]
assert result["screen"].to_list() == [True]
assert result["lagged_mean_notional_volume"].to_list() == [100343.0]
```

For sampled terms, select actual week-, month-, quarter- or year-start exchange
sessions before the backward join. Do not choose the first observed quote in
a partial period. Preserve a missing sample row: filling only nonnull values
would retain a stale earlier sample. Keep input warmup before sampling, and
retain `sample_session` and `sample_available_at` separately from the receiving
decision. `sampled_values` rejects future-known values and ambiguous keys.

## Qualification profiles

The original workflow reference is Zipline Reloaded 3.2.0 at
[`cf649ac`](https://github.com/stefan-jansen/zipline-reloaded/tree/cf649ac06fddce76f6caa2403cd8494ac84f72ad),
with pandas 2.3.3 and NumPy 2.5.4. Original/custom PipelineEngine profiles use
the real shifted pricing loader, missing quotes, listing/delisting and supplied
lifetimes. Separate alias/cadence profiles cover all four sampling frequencies.
The comparisons do not establish order/account/performance equivalence.

The Data producer reference is unpublished `0.2.2.dev41+gd48d849f7` at
[`d48d849`](https://github.com/ml4t/data/tree/d48d849f77d3ff75d05383cbaaed9b42f6ca95b7).
Data selects revisions/adjustments and retains stable IDs. Actual installed
handoffs cover future corrections, raw/split-compatible products and rename
history. The Pyfolio input-method reference is Reloaded 0.10.0, not an Engineer
capacity implementation. These are qualification profiles, not dependency pins
or published install recommendations. Immutable Engineer artifact identities,
exact recipe revisions and final checks are recorded in the owning
[factor/screen PR](https://github.com/ml4t/engineer/pull/115) and
[input/migration PR](https://github.com/ml4t/engineer/pull/116).

See the [operation reference](../api/pipeline-ports.md) for individual methods,
defaults and material differences, and [native features](features.md) for the
preparation and expression contracts.
