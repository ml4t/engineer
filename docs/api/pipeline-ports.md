# Pipeline operation ports

This reference maps the factor, filter and classifier surface inspected in
Zipline Reloaded 3.2.0 source `cf649ac06fddce76f6caa2403cd8494ac84f72ad`.
Routes are native Polars operations or explicit recipes, not legacy graph
objects. `factor_expression` and `return_factor_features` live under
`ml4t.engineer.compat.zipline`; rolling methods live under `features.rolling`,
relationships under `features.paired`, and selections under `features.selection`.
Use the [migration how-to](../user-guide/pipeline-migration.md) to prepare inputs.

## Named factors and aliases

| Legacy name | Native route | Definition and choices |
|---|---|---|
| `Returns` | `factor_expression` or precise `features.utils.arithmetic.returns` | N prices become N-1 changes; signed reference denominator; fraction, not percentage points. |
| `DailyReturns` | Same recipe with two prices | One change; retain both endpoint quote masks. |
| `PercentChange` | `absolute_percent_change` or recipe | Absolute reference denominator; fractional result. |
| `RateOfChangePercentage` | `features.utils.arithmetic.percentage_change` or recipe | Signed reference denominator, percentage points (100 times the fractional return). |
| `SimpleMovingAverage` | `finite_mean` or recipe | Full row window, omit null/NaN values from mean denominator. |
| `WeightedAverageValue`, `VWAP` | `finite_weighted_mean` or recipes | Independent NaN sums of value*weight and weights; a missing value does not erase its weight. |
| `AverageDollarVolume` | `average_dollar_volume` or recipe | Sum of price*volume divided by fixed N; distinguish quote units and volume basis. |
| `LinearWeightedMovingAverage` | `linear_weighted_mean` or recipe | Weights 1..N and fixed weight denominator; omit missing numerator terms. |
| `ExponentialWeightedMovingAverage`, `EWMA` | `finite_ewma` or recipes | Finite normalized weights, missing values propagate. Not native recursive EMA. |
| `ExponentialWeightedMovingStdDev`, `EWMSTD` | `finite_ewm_std` or recipes | Same finite weights plus unbiased weighted variance correction. |
| EWMA/EWMSTD `.from_span`, `.from_halflife`, `.from_center_of_mass` | Recipe suffix or `exponential_decay` | Exactly one weight selector; finite-window semantics remain unchanged. |
| `AnnualizedVolatility` | `annualized_volatility` or recipe | Population stddev, default annualization factor 252 independent of calendar; default price recipe derives daily returns with extra warmup. Supplied returns avoid that dependency. |
| `MaxDrawdown` | `finite_max_drawdown` or recipe | Legacy selected largest absolute fall and trough denominator; not ordinary peak-denominator portfolio drawdown. |
| `RSI` | `rsi_cutler` or recipe | 15-price default, 14 finite changes, omit changes adjacent to missing prices; flat windows undefined. Native `rsi` is Wilder. |
| `BollingerBands` | `finite_bollinger_bands` or recipe | Population stddev, default multiplier 2, upper/middle/lower named outputs. |
| `Aroon` | `finite_aroon` or recipe | First extrema ties; indices divided by N-1. Recipe preserves legacy low/high input order. |
| `FastStochasticOscillator` | `fast_stochastic` or recipe | Unsmoothed K, no D output; full price window. |
| `IchimokuKinkoHyo` | `finite_ichimoku` or recipe | Unshifted named outputs; legacy zero Tenkan/Kijun options select the entire window, not an inferred default. |
| `MovingAverageConvergenceDivergenceSignal`, `MACDSignal` | `finite_macd_signal` or recipes | Finite normalized exponential windows; not recursively seeded TA-Lib MACD. |
| `TrueRange` | `finite_true_range` or recipe | Two-price dependency, NaN-max of high-low and gaps; not ATR. |
| `RollingPearson`, `RollingSpearman` | Recipes or paired rolling functions | Full paired windows, explicit missing-count allowance; Spearman averages ties inside each window. |
| Factor `.pearsonr`, `.spearmanr`, `.linear_regression`; `RollingLinearRegression` | Paired functions or recipes | Explicit target alignment; regression Struct contains alpha, beta, r_value, p_value, stderr and requires the optional `stats` extra. |
| `RollingPearsonOfReturns`, `RollingSpearmanOfReturns`, `RollingLinearRegressionOfReturns` | `return_factor_features` | Align common grid before deriving both price returns; `returns_length` counts prices, correlation/regression length counts return rows. |
| `SimpleBeta` | `return_factor_features` | Two-price daily returns, default maximum missing fraction .25 converted to a row count. Target is declared stable ID, not opaque Asset. |
| `BusinessDaysSincePreviousEvent`, `BusinessDaysUntilNextEvent` | `event_business_days` or recipes | Supplied known event Dates, end-exclusive Monday-Friday arithmetic including exchange holidays. Not exchange-session windows. |
| `PeerCount` | `cross_section(method="peer_count", groupby=...)` | Eligible peers with known classifier; explicit missing sentinel to null first. |
| Numeric, categorical or Boolean `Latest` | `pl.col(...)` or `Latest` recipe | Preserve dtype and missingness; no implicit historical fill. |
| `Clip` and Factor `.clip` | `pl.col(...).clip(...)` or recipe | Explicit bounds, nulls preserved; apply output mask after history. |
| `Factor`, `CustomFactor`, `RecarrayField` | Native expressions/feature registry, Struct field selection | Supporting graph roles are ports, not automatically executable legacy objects. |

## Factor methods and expression operators

| Operation | Native route | Material choices |
|---|---|---|
| Arithmetic `+`, `-`, `*`, `/`, `**`, `%`, reflected forms, unary minus | Corresponding `pl.Expr` operations | Native null and integer overflow rules; do not cast large tick prices before differencing. |
| Comparisons `<`, `<=`, `>`, `>=`, `.eq`, `!=`; Boolean `&`, `|`, `~` | Native expressions over explicitly normalized missing inputs | Do not use Python `and`/`or`. Polars orders NaNs differently from NumExpr comparisons; normalize NaN to null and declare the Boolean null policy, or preserve a specifically required legacy NaN comparison explicitly. Final eligibility still excludes undefined features. |
| `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` | Same-named Polars expression functions | Domain-invalid real values remain undefined. |
| `log`, `log10`, `log1p`, `exp`, `expm1`, `sqrt`, `abs` | `.log()`, `.log(10)`, `.log1p()`, `.exp()`, `numpy.expm1(expr)`, `.sqrt()`, `.abs()` | Use expm1 directly for tiny inputs, not subtraction from a rounded exponential. |
| `.isnan`, `.notnan`, `.isfinite` | `is_missing`, inverse `is_missing`, `.is_finite().fill_null(False)` | Native nulls join NaNs in missingness; infinities are not missing but are nonfinite. |
| `.rank` | `cross_section(method="rank")` | ordinal/min/max/dense/average, ascending direction; ordinal ties use stable ascending IDs, not input row order. |
| `.top`, `.bottom` | Same-named cross-section methods | Exact N, stable ordinal ties; mask determines input population. |
| `.demean`, `.zscore` | Same-named cross-section methods | Contemporary date/group population, population ddof=0 default; zero variance undefined. Not a train-fitted scaler. |
| `.winsorize` | Same-named cross-section method | Floor/ceil legacy order-statistic bounds, not linear quantile clipping. |
| `.quantiles`, `.quartiles`, `.quintiles`, `.deciles` | `cross_section(method="quantiles", bins=...)` | 0-based labels, stable linear interpolation, duplicate-edge error, null exclusions. Nonmissing infinite values fail explicitly. Aliases use 4/5/10 bins. Floating values retain their local scale; integer/Decimal differences precede float conversion. |
| `.percentile_between` | Same-named cross-section method | Inclusive linear bounds; convert legacy percent values 0..100 to fractions 0..1. |
| `.mean`, `.stddev`, `.max`, `.min`, `.median`, `.sum`, `.notnull_count` daily summaries | Mask input then aggregate with native expressions over decision-time keys | Omit null/NaN, population stddev; sum of all missing values is zero, count is zero. Return one row per decision, not one per asset. |
| Asset slicing and named multi-output fields | Exact keyed filter/join; `.struct.field(name)` | Stable keys and explicit output fields replace opaque Asset lookup/record wrappers. |

## Filters and classifiers

| Legacy surface | Native route | Material choices |
|---|---|---|
| `All`, `Any`, `AtLeastN` | `rolling_filter` with all/any/at_least | Full original windows, missing predicates false; apply `.over(entity_col)`. |
| `AllPresent` | `rolling_filter(~is_missing(value), method="all")` | Known row presence is separate from actual value usability. |
| `MaximumFilter` | `cross_section(method="top", n=1)` | One maximum per eligible group, stable-ID first tie. The pinned legacy implementation reinterprets float storage as signed integers and selects the wrong maximum for all-negative groups. Native numeric ordering deliberately corrects that defect. |
| `NullFilter`, `NotNullFilter` | `is_missing`, inverse | Normalize legacy sentinels explicitly; no automatic -1 interpretation. |
| `PercentileFilter` | Cross-section percentile_between | Inclusive bounds and eligible population. |
| `SingleAsset`, `StaticAssets`, `StaticSids` | Stable-ID equality or `.is_in(...)` | Explicitly resolve legacy Asset.sid through its owner; static membership is not a historical universe. Legacy SingleAsset additionally raises unless its asset is alive throughout the requested range. Check that lifetime condition explicitly if retaining its strict behavior; a native equality filter alone does not raise. |
| `ArrayPredicate`, `NumExprFilter` | Native elementwise Boolean expressions | Preserve declared null behavior; do not execute arbitrary legacy expression strings. |
| `Filter.if_else` | `pl.when(predicate).then(left).otherwise(right)` | Declare unknown-predicate handling explicitly. |
| `CustomFilter`, base `Filter` | Boolean expressions or reusable registered features | No graph interpreter; supply required history and knowledge clocks. |
| Classifier `eq`, `!=` (`__ne__`), `element_of` | Equality, inequality, `.is_in(...)` | Map missing sentinel to null; fill null Boolean outputs as intended. |
| `startswith`, `endswith`, `has_substring` | `.str.starts_with`, `.str.ends_with`, `.str.contains(..., literal=True)` | Literal substring, not accidental regex interpretation. |
| `matches` | Explicit `re.compile(pattern).match(value)` mapping | Legacy Python re.match starts at the beginning and supports Python regex features. Polars unanchored Rust regex is not a general equivalent. |
| `relabel` | `.replace_strict(mapping, default=None)` or explicit mapping function | Preserve nulls and declare behavior for unknown labels. |
| `Quantiles` | Cross-section quantiles | Same bins/interpolation/missing choices as Factor method. |
| `Everything` | A declared constant valid group | Missing classifications require a separate policy, not automatic inclusion. |
| `Classifier.peer_count` | Cross-section peer_count | Contemporary eligible group count, not full static universe size. |
| `CustomClassifier`, base `Classifier` | Supplied categories or typed native expression | Retain dtype, nulls and known-at information; encode numeric sentinels only at an explicit receiver boundary. |

Daily summaries, elementary operators and custom ports use existing Polars
functionality rather than duplicate Engineer kernels. The reference identifies
routes and material differences; listing a route alone is not a claim that every
possible legacy graph or regular expression has been qualified.
