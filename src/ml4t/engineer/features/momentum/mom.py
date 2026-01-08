"""
Momentum (MOM) - TA-Lib compatible implementation.

MOM is the simplest momentum indicator. It calculates the difference
between the current price and the price n periods ago.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.deprecation import resolve_period_parameter
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def mom_numba(close: npt.NDArray[np.float64], period: int = 10) -> npt.NDArray[np.float64]:
    """
    Momentum calculation optimized for performance.

    Formula: MOM = price - price[n periods ago]

    Parameters
    ----------
    close : npt.NDArray
        Price data (typically closing close)
    period : int, default 10
        Number of periods for momentum calculation

    Returns
    -------
    npt.NDArray
        Momentum close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least period + 1 close
    if n <= period:
        return result

    # Vectorized calculation - much faster than loop
    result[period:] = close[period:] - close[:-period]

    return result


def mom_polars(column: str, period: int = 10) -> pl.Expr:
    """
    Momentum using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    column : str
        Column name for price data
    period : int, default 10
        Number of periods for momentum calculation

    Returns
    -------
    pl.Expr
        Polars expression for momentum
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(mom_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="mom",
    category="momentum",
    description="Momentum - rate of price change",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def mom(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int | None = None,
    *,
    timeperiod: int | None = None,  # Deprecated alias for period
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Momentum exactly matching TA-Lib.

    The momentum indicator measures the amount that a security's price has
    changed over a given time span. Unlike ROC which is normalized, MOM
    returns the raw price difference.

    Parameters
    ----------
    close : array-like or str
        Price data or column name
    period : int, default 10
        Number of periods for momentum calculation
    timeperiod : int, optional
        Deprecated alias for period. Use period instead.

    Returns
    -------
    array or Polars expression
        Momentum close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> mom_values = mom(close, period=5)

    Notes
    -----
    - MOM is not normalized and should not be used to compare different securities
    - For normalized momentum, use ROC or ROCP
    - First 'period' close will be NaN
    - Simple calculation: price - price[n periods ago]
    """
    # Resolve period with deprecation handling
    period = resolve_period_parameter(
        period=period,
        timeperiod=timeperiod,
        default=10,
        func_name="mom",
    )

    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if isinstance(close, str):
        # Column name provided for Polars
        return mom_polars(close, period)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Ensure we have a numpy array with minimal copying
    if not isinstance(close, np.ndarray):
        close = np.asarray(close, dtype=np.float64)
    elif close.dtype != np.float64:
        close = close.astype(np.float64, copy=False)

    n = len(close)
    if n <= period:
        return np.full(n, np.nan)

    # Ultra-minimal allocation approach for best performance
    result = np.empty(n, dtype=np.float64)
    result[:period] = np.nan

    # Direct array subtraction - NumPy's C code is highly optimized for this
    result[period:] = close[period:] - close[:-period]
    return result


# Export all functions
__all__ = ["mom", "mom_numba", "mom_polars"]
