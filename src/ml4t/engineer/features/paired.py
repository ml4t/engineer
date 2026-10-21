"""Finite paired-observation correlations and regression features.

Inputs must already be aligned by their actual keys. Apply expressions within
asset groups. Missing pairs occupy their original window positions and are
removed from both sides together, never by compressing the input timeline.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from ml4t.specs import UNSET_CALENDAR, FeedSpec, WindowUnit

from ml4t.engineer._numba import jit
from ml4t.engineer.features.cross_asset import beta_to_market, rolling_correlation
from ml4t.engineer.features.rolling import _period, _ready
from ml4t.engineer.utils.dependencies import require_dependency


def target_return_features(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    target: Mapping[str, Any],
    period: int,
    price_basis: str,
    returns_periods: int = 1,
    method: str = "pearson",
    feed_spec: FeedSpec | None = None,
    timestamp_col: str | None = None,
    entity_col: str | list[str] | None = None,
    price_col: str | None = None,
    calendar: Any = UNSET_CALENDAR,
    window_unit: str | WindowUnit | None = None,
    index_mode: str = "common_grid",
    allowed_missing: int = 0,
    missing_price_policy: str = "raise",
    available_at_col: str | None = None,
    output: str = "target",
) -> pl.DataFrame | pl.LazyFrame:
    """Prepare keyed price returns and a rolling statistic against one stable target.

    The default common_grid uses the supplied union clock, retaining absent quotes
    through both return derivation and the rolling window. per_asset instead counts
    each asset's observations before joining target returns on exact timestamps.
    Neither mode fills prices. A named calendar validates the original declared grid
    before calculation. calendar=None selects observation rows, not a 24/7 schedule.

    Original columns, keys, order and row count survive. Added columns are
    <output>_asset_return, _target_return, _<method>, _available_at and _price_basis.
    Availability is unknown unless all price dependencies in the full window have
    supplied timezone-aware publication instants. It is not a decision-time screen.
    price_basis records a caller declaration; this function does not adjust prices.
    """
    from ml4t.engineer.core.bar_grid import validate_declared_grid
    from ml4t.engineer.features.events import _instant_ticks
    from ml4t.engineer.features.utils.arithmetic import returns

    _validate(period, allowed_missing)
    _period(returns_periods)
    if period > 2**63 - 1 or returns_periods > 2**63 - 1:
        raise ValueError("Window counts must fit signed 64-bit integers")
    methods = {
        "pearson": rolling_pearson,
        "spearman": rolling_spearman,
        "beta": rolling_beta,
        "regression": rolling_regression,
    }
    if method not in methods:
        raise ValueError(f"method must be one of {tuple(methods)}")
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("Declare a nonempty price_basis")
    if not isinstance(output, str) or not output:
        raise ValueError("output must be a nonempty prefix")
    if index_mode not in {"common_grid", "per_asset"}:
        raise ValueError("index_mode must be common_grid or per_asset")
    if missing_price_policy not in {"raise", "null"}:
        raise ValueError("missing_price_policy must be raise or null")
    if feed_spec is not None and not isinstance(feed_spec, FeedSpec):
        raise TypeError("feed_spec must be a canonical FeedSpec instance, not an unresolved config")
    contract = FeedSpec(time_contract_version=2) if feed_spec is None else feed_spec

    def bind(explicit: Any, declared: Any, default: Any, role: str) -> Any:
        if explicit is not None and feed_spec is not None and declared is not None:
            left = [explicit] if isinstance(explicit, str) else list(explicit)
            right = [declared] if isinstance(declared, str) else list(declared)
            if left != right:
                raise ValueError(f"Conflicting {role} and feed_spec declaration")
        return explicit if explicit is not None else declared if declared is not None else default

    timestamp = bind(timestamp_col, contract.timestamp_col, "timestamp", "timestamp_col")
    identity = bind(entity_col, contract.entity_col, "asset", "entity_col")
    entities = [identity] if isinstance(identity, str) else list(identity)
    price = bind(price_col, contract.price_col, "close", "price_col")
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Declare distinct entity columns")
    if not isinstance(target, Mapping) or set(target) != set(entities):
        raise ValueError("target must supply exactly the declared stable entity columns")
    if any(value is None for value in target.values()):
        raise ValueError("target identity cannot be null")
    if len({timestamp, *entities, price}) != len(entities) + 2:
        raise ValueError("Timestamp, entity and price roles must be distinct")
    resolved_calendar = contract.resolve_calendar(calendar=calendar)
    unit = (
        WindowUnit(window_unit)
        if window_unit is not None
        else (WindowUnit.OBSERVATION if resolved_calendar is None else WindowUnit.EXPECTED_BAR)
    )
    expected_unit = WindowUnit.OBSERVATION if resolved_calendar is None else WindowUnit.EXPECTED_BAR
    if unit != expected_unit:
        raise ValueError(f"This finite-row statistic requires window_unit={expected_unit.value!r}")
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    required = [timestamp, *entities, price, *([available_at_col] if available_at_col else [])]
    if set(required) - set(schema):
        raise ValueError(f"Missing input columns: {sorted(set(required) - set(schema))}")
    if schema[timestamp].base_type() not in {pl.Date, pl.Datetime}:
        raise ValueError("timestamp_col must contain Date labels or Datetime instants")
    if not schema[price].is_numeric():
        raise ValueError("price_col must be numeric")
    names = [
        f"{output}_{name}"
        for name in ("asset_return", "target_return", method, "available_at", "price_basis")
    ]
    if set(names) & set(schema) or any(c.startswith("__ml4t_target_") for c in schema):
        raise ValueError("Output or reserved preparation columns already exist")
    result_schema = dict(schema)
    result_schema.update(dict.fromkeys(names[:2], pl.Float64()))
    result_schema[names[2]] = _REGRESSION_DTYPE if method == "regression" else pl.Float64()
    result_schema[names[3]] = pl.Datetime("ns", "UTC")
    result_schema[names[4]] = pl.String()
    keys = [*entities, timestamp]
    prefix = "__ml4t_target_"
    quote, known, pair_known, row_id = (prefix + s for s in ("quote", "known", "pair", "row"))

    def calculate(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame.with_columns(
                [pl.lit(None, dtype=result_schema[c]).alias(c) for c in names]
            )
        if frame.select(pl.any_horizontal(pl.col(c).is_null() for c in keys).any()).item():
            raise ValueError("Timestamp and entity keys cannot be null")
        if frame.select(pl.struct(keys).is_duplicated().any()).item():
            raise ValueError("Duplicate timestamp/entity keys")
        predicate = pl.all_horizontal(
            pl.col(c) == pl.lit(target[c], dtype=schema[c]) for c in entities
        )
        if frame.filter(predicate).is_empty():
            raise ValueError("target identity is absent from supplied input")
        valid_price = (pl.col(price) > 0) & pl.col(price).cast(pl.Float64).is_finite()
        if missing_price_policy == "raise" and frame.filter(~valid_price.fill_null(False)).height:
            raise ValueError("Prices must be finite, positive and nonmissing")
        if resolved_calendar is not None:
            validate_declared_grid(
                frame,
                feed_spec=contract,
                calendar=resolved_calendar,
                timestamp_col=timestamp,
                group_cols=entities,
            )
        known_expr = (
            _instant_ticks(available_at_col) if available_at_col else pl.lit(None, pl.Int128)
        )
        prepared = frame.select(
            *keys, pl.when(valid_price).then(pl.col(price)).alias(quote), known_expr.alias(known)
        )
        if (
            available_at_col
            and prepared.filter((pl.col(known) < -(2**63)) | (pl.col(known) > 2**63 - 1)).height
        ):
            raise ValueError("Publication instants exceed Datetime(ns) range")
        if index_mode == "common_grid":
            grid = (
                frame.select(entities).unique().join(frame.select(timestamp).unique(), how="cross")
            )
            prepared = grid.join(prepared, on=keys, how="left", validate="1:1")
        prepared = prepared.sort(keys)
        lag = min(returns_periods, prepared.height)
        return_known = pl.when(
            pl.col(known).is_not_null() & pl.col(known).shift(lag).is_not_null()
        ).then(pl.max_horizontal(pl.col(known), pl.col(known).shift(lag)))
        prepared = prepared.with_columns(
            returns(quote, lag).cast(pl.Float64).over(entities).alias(names[0]),
            return_known.over(entities).alias(known),
        )
        benchmark = prepared.filter(predicate).select(
            timestamp, pl.col(names[0]).alias(names[1]), pl.col(known).alias(pair_known)
        )
        prepared = prepared.join(benchmark, on=timestamp, how="left", validate="m:1")
        prepared = prepared.with_columns(
            pl.when(pl.col(known).is_not_null() & pl.col(pair_known).is_not_null())
            .then(pl.max_horizontal(known, pair_known))
            .cast(pl.Int64)
            .alias(pair_known)
        )
        statistic = methods[method](names[0], names[1], period, allowed_missing=allowed_missing)
        warmup = returns_periods + period
        availability = (
            pl.when(pl.col(pair_known).is_not_null().cast(pl.Int64).rolling_sum(period) == period)
            .then(pl.col(pair_known).rolling_max(period))
            .cast(pl.Datetime("ns", "UTC"))
        )
        prepared = prepared.with_columns(
            _ready(statistic, warmup).over(entities).alias(names[2]),
            _ready(availability, warmup).over(entities).alias(names[3]),
            pl.lit(price_basis).alias(names[4]),
        )
        return (
            frame.with_row_index(row_id)
            .join(prepared.select(*keys, *names), on=keys, how="left", validate="1:1")
            .sort(row_id)
            .drop(row_id)
        )

    if isinstance(data, pl.LazyFrame):
        return data.map_batches(
            calculate,
            schema=result_schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    return calculate(data)


def _validate(period: int, allowed_missing: int) -> None:
    _period(period, minimum=2)
    if (
        isinstance(allowed_missing, bool)
        or not isinstance(allowed_missing, int)
        or not 0 <= allowed_missing <= period
    ):
        raise ValueError("allowed_missing must be an integer in [0, period]")


def _expressions(
    dependent: str | pl.Expr, independent: str | pl.Expr
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    y = pl.col(dependent) if isinstance(dependent, str) else dependent
    x = pl.col(independent) if isinstance(independent, str) else independent
    y, x = y.cast(pl.Float64), x.cast(pl.Float64)
    missing = y.is_null() | x.is_null() | y.is_nan() | x.is_nan()
    return pl.when(~missing).then(y), pl.when(~missing).then(x), y.is_infinite() | x.is_infinite()


def rolling_pearson(
    dependent: str | pl.Expr,
    independent: str | pl.Expr,
    period: int = 20,
    *,
    allowed_missing: int = 0,
) -> pl.Expr:
    """Pearson correlation over a full row window with a paired missing-count limit."""
    _validate(period, allowed_missing)
    y, x, infinite = _expressions(dependent, independent)
    result = rolling_correlation(y, x, window=period, min_periods=max(2, period - allowed_missing))
    return _ready(
        pl.when(infinite.fill_null(False).rolling_max(period)).then(float("nan")).otherwise(result),
        period,
    )


def rolling_beta(
    dependent: str | pl.Expr,
    independent: str | pl.Expr,
    period: int = 60,
    *,
    allowed_missing: int = 0,
) -> pl.Expr:
    """Slope Cov(independent,dependent)/Var(independent) using identical paired rows."""
    _validate(period, allowed_missing)
    y, x, infinite = _expressions(dependent, independent)
    result = beta_to_market(y, x, window=period, min_periods=max(2, period - allowed_missing))
    return _ready(
        pl.when(infinite.fill_null(False).rolling_max(period)).then(float("nan")).otherwise(result),
        period,
    )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values)
    result = np.empty(len(values))
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2
        for j in range(start, stop):
            result[order[j]] = rank
        start = stop
    return result


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _spearman_windows(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64], period: int, allowed_missing: int
) -> npt.NDArray[np.float64]:
    result = np.full(len(y), np.nan)
    for end in range(period - 1, len(y)):
        yy, xx = y[end - period + 1 : end + 1], x[end - period + 1 : end + 1]
        valid = ~np.isnan(yy) & ~np.isnan(xx)
        count = valid.sum()
        if count < 2 or period - count > allowed_missing:
            continue
        yr, xr = _ranks(yy[valid]), _ranks(xx[valid])
        yr, xr = yr - yr.mean(), xr - xr.mean()
        denominator = np.sqrt((yr * yr).sum() * (xr * xr).sum())
        if denominator > 0:
            result[end] = (yr * xr).sum() / denominator
    return result


def rolling_spearman(
    dependent: str | pl.Expr,
    independent: str | pl.Expr,
    period: int = 20,
    *,
    allowed_missing: int = 0,
) -> pl.Expr:
    """Spearman correlation with average ties recomputed inside each paired window."""
    _validate(period, allowed_missing)
    y, x, _ = _expressions(dependent, independent)
    return pl.map_batches(
        [y, x],
        lambda values: pl.Series(
            _spearman_windows(values[0].to_numpy(), values[1].to_numpy(), period, allowed_missing)
        ),
        return_dtype=pl.Float64,
    )


_REGRESSION_FIELDS = ("alpha", "beta", "r_value", "p_value", "stderr")
_REGRESSION_DTYPE = pl.Struct(dict.fromkeys(_REGRESSION_FIELDS, pl.Float64))


def rolling_regression(
    dependent: str | pl.Expr,
    independent: str | pl.Expr,
    period: int = 20,
    *,
    allowed_missing: int = 0,
) -> pl.Expr:
    """OLS of dependent on independent, returning a five-field Struct expression.

    Requires the stats extra for SciPy's inferential outputs. Constant independent
    windows are undefined and yield NaN fields instead of failing the whole panel.
    Nonfinite nonmissing inputs propagate undefined fields. No trend-on-time or
    portfolio attribution is substituted for the supplied independent observations.
    """
    _validate(period, allowed_missing)
    require_dependency("scipy", "rolling regression inferential fields")
    from scipy.stats import linregress

    y, x, _ = _expressions(dependent, independent)

    def compute(values: Sequence[pl.Series]) -> pl.Series:
        yy, xx = values[0].to_numpy(), values[1].to_numpy()
        output = np.full((len(yy), 5), np.nan)
        for end in range(period - 1, len(yy)):
            ys, xs = yy[end - period + 1 : end + 1], xx[end - period + 1 : end + 1]
            valid = ~np.isnan(ys) & ~np.isnan(xs)
            count = np.count_nonzero(valid)
            if count < 2 or period - count > allowed_missing:
                continue
            ys, xs = ys[valid], xs[valid]
            if not np.isfinite(ys).all() or not np.isfinite(xs).all() or np.ptp(xs) == 0:
                continue
            result = linregress(xs, ys)
            output[end] = (
                result.intercept,
                result.slope,
                result.rvalue,
                result.pvalue,
                result.stderr,
            )
        return pl.DataFrame(
            {name: output[:, i] for i, name in enumerate(_REGRESSION_FIELDS)}
        ).to_struct()

    return pl.map_batches([y, x], compute, return_dtype=_REGRESSION_DTYPE)
