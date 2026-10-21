"""Contemporary eligible-group transforms and full-window boolean filters."""

import math

import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer.features.rolling import _paired_difference, _period


def _value(value: str | pl.Expr) -> pl.Expr:
    return pl.col(value) if isinstance(value, str) else value


def is_missing(value: str | pl.Expr) -> pl.Expr:
    """True for native null or floating NaN, without sentinel or infinity inference."""

    def build(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        return (
            expression.is_null() | expression.is_nan().fill_null(False)
            if dtype.is_float()
            else expression.is_null()
        )

    return _value(value).pipe_with_dtype(build)


def _linear_edges(
    values: npt.NDArray[np.float64], fractions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    ordered = np.sort(values)
    if not np.isfinite(ordered).all():
        raise ValueError("Quantile and percentile inputs require finite nonmissing values")
    positions = (len(ordered) - 1) * np.asarray(fractions, dtype=np.float64)
    a = ordered[np.floor(positions).astype(np.int64)]
    b = ordered[np.ceil(positions).astype(np.int64)]
    weight = positions - np.floor(positions)
    opposite = np.signbit(a) != np.signbit(b)
    edges = np.empty_like(positions)
    # Avoid subtracting opposite finite extremes or losing small local values
    # by centering every floating observation around a distant first value.
    edges[opposite] = a[opposite] * (1 - weight[opposite]) + b[opposite] * weight[opposite]
    edges[~opposite] = a[~opposite] + (b[~opposite] - a[~opposite]) * weight[~opposite]
    return edges


def cross_section(
    value: str | pl.Expr,
    *,
    method: str,
    timestamp_col: str = "timestamp",
    entity_col: str | list[str] = "asset",
    groupby: str | list[str] | None = None,
    mask: str | pl.Expr | None = None,
    ascending: bool = True,
    ties: str = "ordinal",
    n: int | None = None,
    bins: int | None = None,
    min_percentile: float = 0.0,
    max_percentile: float = 1.0,
    ddof: int = 0,
) -> pl.Expr:
    """Transform the eligible contemporary cross-section, independently per group.

    mask determines the input population, not a final screen. Null/NaN group labels,
    null values and NaNs are excluded; legacy classifier sentinels must be mapped
    explicitly. Ordinal ties use ascending stable entity IDs, not input row order.
    Input keys must identify at most one observation per timestamp/entity. This
    expression does not select source revisions or establish causal availability.

    Winsorization uses floor(n*lower)/ceil(n*upper)-1 order statistics. Quantile
    edges and percentile bounds use linear interpolation. Duplicate quantile edges
    raise rather than silently reducing bins. Percentile bounds are inclusive.
    """
    methods = {
        "rank",
        "top",
        "bottom",
        "demean",
        "zscore",
        "winsorize",
        "quantiles",
        "percentile_between",
        "peer_count",
    }
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    if not isinstance(ascending, bool) or ties not in {"ordinal", "min", "max", "dense", "average"}:
        raise ValueError("Declare boolean ascending and a supported tie method")
    if ties != "ordinal" and method != "rank":
        raise ValueError("Only rank accepts a nonordinal tie method")
    for name, count, applicable, minimum in (
        ("n", n, method in {"top", "bottom"}, 0),
        ("bins", bins, method == "quantiles", 1),
    ):
        if applicable:
            if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        elif count is not None:
            raise ValueError(f"{name} is not applicable to {method}")
    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof must be a nonnegative integer")
    if ddof and method != "zscore":
        raise ValueError("ddof applies only to zscore")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
            for v in (min_percentile, max_percentile)
        )
        or not 0 <= min_percentile < max_percentile <= 1
    ):
        raise ValueError("Declare 0 <= min_percentile < max_percentile <= 1")
    if (min_percentile, max_percentile) != (0.0, 1.0) and method not in {
        "winsorize",
        "percentile_between",
    }:
        raise ValueError("Percentile bounds do not apply to this method")
    entities = [entity_col] if isinstance(entity_col, str) else list(entity_col)
    groups = [] if groupby is None else [groupby] if isinstance(groupby, str) else list(groupby)
    if not entities or any(
        not isinstance(c, str) or not c for c in [timestamp_col, *entities, *groups]
    ):
        raise ValueError("Declare nonempty timestamp, entity and grouping column names")
    if mask is not None and not isinstance(mask, str | pl.Expr):
        raise TypeError("mask must be a column name or expression")
    original = _value(value)
    eligible = ~is_missing(original)
    if groups:
        eligible &= pl.all_horizontal(~is_missing(c) for c in groups)
    if mask is not None:
        eligible &= _value(mask).fill_null(False)
    x = pl.when(eligible).then(original)
    partition = list(dict.fromkeys([timestamp_col, *groups]))
    if method == "peer_count":
        return pl.when(eligible).then(x.count().over([*partition, x]).cast(pl.Float64))
    if method in {"rank", "top", "bottom"}:
        descending = method == "top" or (method == "rank" and not ascending)
        rank = x.rank(method=ties, descending=descending).over(partition, order_by=entities)
        return rank.cast(pl.Float64) if method == "rank" else (rank <= n).fill_null(False)
    centered = _paired_difference(x, x.drop_nulls().first())
    if method in {"demean", "zscore"}:
        # Retain typed ticks near a large offset, then scale residuals before
        # moments. Opposite extreme floats need scaling before subtraction too.
        fallback = centered.abs().max().is_infinite().fill_null(False)
        raw_scale = x.abs().max().cast(pl.Float64)
        working = pl.when(fallback).then(x / raw_scale).otherwise(centered)
        scale = working.abs().max()
        normalized = working / pl.when(scale > 0).then(scale).otherwise(1.0)
        result = normalized - normalized.mean()
        if method == "zscore":
            result = result / normalized.std(ddof=ddof)
        else:
            result = result * scale * pl.when(fallback).then(raw_scale).otherwise(1.0)
    elif method == "winsorize":
        ordered = x.drop_nulls().sort()
        lower = ordered.get((x.count() * min_percentile).floor().cast(pl.Int64), null_on_oob=True)
        upper = ordered.get(
            (x.count() * max_percentile).ceil().cast(pl.Int64) - 1, null_on_oob=True
        )
        result = x.clip(lower, upper)
    else:

        def coordinates(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
            return (
                value.cast(pl.Float64)
                if dtype.is_float()
                else _paired_difference(value, value.drop_nulls().first())
            )

        def select_quantiles(series: pl.Series) -> pl.Series:
            values = series.to_numpy()
            valid = ~np.isnan(values)
            if not valid.any():
                return (
                    pl.Series([None] * len(values), dtype=pl.Int64)
                    if method == "quantiles"
                    else pl.Series([False] * len(values), dtype=pl.Boolean)
                )
            if method == "percentile_between":
                lower, upper = _linear_edges(
                    values[valid], np.array([min_percentile, max_percentile])
                )
                return pl.Series(valid & (values >= lower) & (values <= upper))
            assert bins is not None
            edges = _linear_edges(values[valid], np.linspace(0.0, 1.0, bins + 1))
            if bins > 1 and len(np.unique(edges)) != len(edges):
                raise ValueError("Duplicate quantile edges; declare a different bin count")
            labels = np.searchsorted(edges, values, side="left") - 1
            labels[values == edges[0]] = 0
            return pl.Series(labels, dtype=pl.Int64).set(pl.Series(~valid), None)

        result = x.pipe_with_dtype(coordinates).map_batches(
            select_quantiles, return_dtype=pl.Int64 if method == "quantiles" else pl.Boolean
        )
    return result.over(partition, order_by=entities)


def rolling_filter(
    value: str | pl.Expr,
    *,
    period: int,
    method: str,
    minimum: int | None = None,
) -> pl.Expr:
    """Any/all/at_least over full observation rows; missing predicates count false.

    Apply over(entity_col) to isolate histories. Leading incomplete windows are
    null. Keep input eligibility and final screens separate from this history.
    """
    _period(period)
    if method not in {"any", "all", "at_least"}:
        raise ValueError("method must be any, all or at_least")
    if method == "at_least":
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 0 <= minimum <= period:
            raise ValueError("minimum must be an integer in [0,period]")
    elif minimum is not None:
        raise ValueError("minimum applies only to at_least")

    def build(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if dtype != pl.Boolean:
            raise ValueError("rolling_filter requires boolean predicates")
        return expression.fill_null(False).cast(pl.Int64)

    count = _value(value).pipe_with_dtype(build).rolling_sum(period)
    threshold = 1 if method == "any" else period if method == "all" else minimum
    return count >= threshold
