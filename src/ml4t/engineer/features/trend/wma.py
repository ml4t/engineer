# mypy: disable-error-code="no-any-return"
"""
Weighted Moving Average (WMA) - TA-Lib compatible implementation.

The Weighted Moving Average gives more weight to recent close, with weights
decreasing linearly as close get older.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def wma_numba(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """
    Weighted Moving Average using Numba JIT compilation.

    Parameters
    ----------
    close : npt.NDArray
        Price data
    period : int
        Number of periods for the moving average

    Returns
    -------
    npt.NDArray
        WMA close with NaN for insufficient data
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 1 or period > n:
        return result

    # Pre-calculate weight sum for efficiency
    weight_sum = period * (period + 1) // 2

    for i in range(period - 1, n):
        weighted_sum = 0.0
        for j in range(period):
            weight = period - j
            weighted_sum += close[i - j] * weight
        result[i] = weighted_sum / weight_sum

    return result


def wma_polars(column: str, period: int) -> pl.Expr:
    """
    Weighted Moving Average using Polars expressions.

    Parameters
    ----------
    column : str
        Column name to apply WMA to
    period : int
        Number of periods for the moving average

    Returns
    -------
    pl.Expr
        Polars expression for WMA calculation
    """
    # Create weights (1, 2, 3, ..., period)
    weights = list(range(1, period + 1))
    weight_sum = sum(weights)

    return pl.col(column).rolling_map(
        lambda s: sum(val * weight for val, weight in zip(s, weights, strict=False)) / weight_sum,
        window_size=period,
    )


@feature(
    name="wma",
    category="trend",
    description="WMA - Weighted Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def wma(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Weighted Moving Average with automatic implementation selection.

    Parameters
    ----------
    close : array-like or column name
        Price data or column name (for Polars expressions)
    period : int
        Number of periods for the moving average
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        WMA close
    """
    if isinstance(close, str):
        return wma_polars(close, period)

    if implementation == "polars":
        raise ValueError("Polars implementation requires column name (string) input")
    # Use Numba (default for best performance)
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return wma_numba(close, period)


# Export the main function
__all__ = ["wma"]
