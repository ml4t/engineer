# Prepare optional market, sector and loading inputs

Use these helpers when an existing research workflow supplies Pyfolio-style
market/loadings containers or already keyed Polars tables. Engineer prepares the
inputs. Diagnostic owns capacity, sector exposures and supplied-loading analysis;
Data owns revision selection, corporate adjustments and source identity history.
No Reloaded package is required by these helpers.

## Convert the container before preparing derived values

`market_to_long` maps a pandas frame indexed by `(date, field)`, with assets in
columns, to the timestamp/entity/price/volume columns declared by `feed_spec`.
The default field selectors are `price` and `volume`. `loadings_to_long` maps
`(date, asset)` rows and factor columns to keyed `factor`/`loading` rows. Both
accept native Polars tables, validate keys and preserve their eager/lazy form.

Legacy containers do not declare publication time, monetary units, corporate
adjustment basis or completed bars. Supply those facts from the source contract;
the conversion does not invent them. Opaque legacy Asset objects require an
explicit `entity_map` to stable integer/string IDs. Compound entity mappings use
tuples matching `FeedSpec.entity_col`. Numeric and timestamp dtypes are retained;
incompatible asset/factor dtypes must be normalized explicitly, rather than
silently casting precise integers to floats. These narrow container conversions
preserve numeric nullable arrays and physical datetime keys without requiring
PyArrow. They do not provide a generic arbitrary-object serialization service.

Run `uv run python examples/optional_input_migration.py` for a complete container
conversion, liquidity preparation, supplied-loading selection and static-sector
recipe. The example declares a synthetic source's publication/completion facts.
It does not claim that real daily prices are known before their close.

## Prepare completed, known market inputs

This executable native example uses observation rows deliberately. With a named
calendar, declare the actual time-bar grid; omitted `calendar` resolves the
explicit feed metadata, then NYSE. Explicit `calendar=None` disables schedules
and counts rows, including gaps. A named continuous schedule still requires its
scheduled bars and is not row mode.

<!-- ml4t-exec -->
```python
from datetime import UTC, date, datetime
import polars as pl
from ml4t.specs import FeedSpec
from ml4t.engineer.features.inputs import prepare_market_inputs

days = [date(2025, 7, d) for d in (7, 8, 9)]
raw = pl.DataFrame({
    "timestamp": days, "asset": [17] * 3,
    "close": [10.0, 20.0, 30.0], "volume": [100.0, 0.0, 200.0],
    "currency": ["USD"] * 3, "multiplier": [1.0] * 3,
    "volume_unit": ["shares"] * 3,
    "available_at": [datetime(2025, 7, d.day, 20, tzinfo=UTC) for d in days],
    "completed_at": [datetime(2025, 7, d.day, 20, tzinfo=UTC) for d in days],
    "decision_time": [datetime(2025, 7, d.day, 13, tzinfo=UTC) for d in days],
})
result = prepare_market_inputs(
    raw.lazy(),
    feed_spec=FeedSpec(time_contract_version=2, calendar=None, entity_col="asset"),
    price_basis="raw", volume_basis="raw", purpose="causal",
    liquidity_window=2, completed_at_col="completed_at",
).collect()
assert result["lagged_mean_notional_volume"].to_list() == [None, None, 500.0]
assert result["notional_volume"].to_list() == [None, None, None]
assert result["liquidity_covered"].to_list() == [False, False, True]
assert result.select(raw.columns).equals(raw)
```

The third decision can use the two completed prior products, 1,000 and zero. It
cannot use that day's unpublished price/volume product. Missing volume remains
missing; a genuine zero is retained. Unknown currency, volume unit or monetary
multiplier leaves dependent liquidity uncovered. A contract multiplier is
explicit, not an assumed equity unit. No FX conversion occurs.
Each stable entity's liquidity history must use one quote currency. Mixed
currencies fail atomically and require explicit normalization upstream; units
are not made comparable by averaging. Different entities may use different
currencies. With no requested liquidity feature, their raw tables remain unchanged.

## Method and output reference

| Input or output | Contract |
|---|---|
| `purpose` | Required `causal` or `retrospective`; retrospective values are not signals. |
| `price_basis`, `volume_basis` | Required compatible declarations for derived liquidity. Data/callers prepare adjustments upstream. |
| `currency`, `multiplier`, `volume_unit` | Explicit per-row supplied units; column roles are configurable. |
| `liquidity_window=None` | Validate and return the native market table unchanged. |
| `notional_volume` | Float64 `price * volume * multiplier`; original financial columns remain unchanged. Causal output requires known completed input. |
| `lagged_mean_notional_volume` | Strict complete-window mean of those products, shifted once by `completed_lag` (default one). |
| `liquidity_available_at`, `liquidity_covered` | Latest contributing publication/completion instant and actual usable coverage. Unknown clocks cannot qualify causal output. |
| `liquidity_window`, `liquidity_lag`, `liquidity_window_unit` | Explicit observation/expected-bar window and lag, not elapsed duration. |
| `liquidity_price_basis`, `liquidity_volume_basis`, `preparation_purpose` | Retained financial definition and information-set declaration. |

Pyfolio Reloaded's capacity denominator is `mean(price * volume).shift(1)` for
unit equity multipliers. Current price times lagged average volume gives a
different answer. Engineer prepares the denominator only; it does not normalize
monetary holdings into weights or calculate days to liquidate. A second
liquidity preparation rejects already derived output columns to prevent double
lagging. A zero denominator remains covered zero here; the receiving capacity
calculation decides that liquidation is undefined.

## Select supplied loadings and classifications

`prepare_loading_inputs(facts, requests=..., factors=[...], feed_spec=...,
purpose=...)` retains every explicitly requested asset/date/factor, even when no
loading exists. Facts have the feed's timestamp/entity keys, string `factor`,
numeric `loading` and aware `available_at`. Requests have observation keys and
aware `decision_time`. `effective_time_col` selects a separate request key when
the applicable session differs from the observation cutoff. No earlier loading
is carried to an absent date; missing loading is not zero exposure.

`prepare_classification_inputs` accepts supplied start-inclusive/end-exclusive
`effective_from`/`effective_to` facts, or a mapping with `static=True`. Causal
static maps also require a real `static_known_at` instant. Static maps describe
a fixed classification, not historical sector truth. Expired or unknown sectors
remain uncovered. A declared request effective clock is used exactly, without
UTC date flooring or observation-spacing inference.

Both preparers retain the original request columns plus the selected value and
its `_available_at`, `_covered`, `_known_at_decision`, `_effective_from`,
`_effective_to`, `_purpose` and `_static` columns. The suffixes use the selected
`output` name (`loading` or `sector` by default). Causal selection excludes
future publication; retrospective selection retains the actual time and a false
known-at-decision flag where appropriate. Multiple applicable known source
revisions fail explicitly; Data/callers must select the intended revision.
Output names cannot overwrite a factor key. Compound static-map tuple keys must
match the declared entity columns exactly; extra tuple elements are not discarded.
`None` optional facts produce uncovered request rows and do not remove unrelated
price features. Explicit requested factors define loading coverage, not future
factor names or assets discovered in the supplied history.

## Compatibility and qualification boundary

These APIs are unreleased development additions after Engineer 0.1.6. The
native examples need no Pyfolio, Data or Diagnostic installation. The original
capacity method reference is Pyfolio Reloaded 0.10.0; matching its denominator
does not establish equivalence of a portfolio tear sheet, intraday estimate or
order simulation. Do not relabel retrospective external tables as causal, or
silently reinterpret total-return prices with unadjusted share volumes.
