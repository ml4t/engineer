"""Finite paired-observation correlations and regression features.

Inputs must already be aligned by their actual keys. Apply expressions within
asset groups. Missing pairs occupy their original window positions and are
removed from both sides together, never by compressing the input timeline.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer._numba import jit
from ml4t.engineer.features.cross_asset import beta_to_market, rolling_correlation
from ml4t.engineer.features.rolling import _period, _ready
from ml4t.engineer.utils.dependencies import require_dependency


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
