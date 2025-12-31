# mypy: disable-error-code="no-any-return"
"""
STDDEV (Standard Deviation) - TA-Lib compatible implementation.

Calculates standard deviation over a rolling window, exactly matching TA-Lib's algorithm.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def stddev_numba(
    close: npt.NDArray[np.float64],
    period: int = 5,
    nbdev: float = 1.0,
    ddof: int = 0,
) -> npt.NDArray[np.float64]:
    """
    Standard Deviation calculation exactly matching TA-Lib.

    TA-Lib uses a rolling window approach with optional scaling factor (nbdev).

    Parameters
    ----------
    close : npt.NDArray
        Input close (typically close close)
    period : int, default 5
        Number of periods for calculation
    nbdev : float, default 1.0
        Number of standard deviations (scaling factor)
    ddof : int, default 0
        Delta degrees of freedom (0 for population, 1 for sample)
        TA-Lib uses ddof=0 by default

    Returns
    -------
    npt.NDArray
        Standard deviation close scaled by nbdev
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 2 or n < period:
        return result

    # Calculate standard deviation for each window
    for i in range(period - 1, n):
        window_start = i - period + 1
        window = close[window_start : i + 1]

        # Check for NaN close in window
        has_nan = False
        for j in range(len(window)):
            if np.isnan(window[j]):
                has_nan = True
                break

        if has_nan:
            continue

        # Calculate mean
        mean = np.sum(window) / period

        # Calculate variance
        variance = 0.0
        for j in range(len(window)):
            diff = window[j] - mean
            variance += diff * diff

        # Apply degrees of freedom correction
        if period - ddof > 0:
            variance = variance / (period - ddof)
        else:
            # Avoid division by zero
            result[i] = 0.0
            continue

        # Standard deviation = sqrt(variance) * nbdev
        result[i] = np.sqrt(variance) * nbdev

    return result


def stddev_polars(
    col: str,
    period: int = 5,
    nbdev: float = 1.0,
    ddof: int = 0,
) -> pl.Expr:
    """
    Standard Deviation using Polars expressions.

    Parameters
    ----------
    col : str
        Column name for close
    period : int, default 5
        Number of periods
    nbdev : float, default 1.0
        Number of standard deviations
    ddof : int, default 0
        Delta degrees of freedom

    Returns
    -------
    pl.Expr
        Polars expression for standard deviation
    """
    # Use Polars rolling standard deviation with scaling
    return pl.col(col).rolling_std(window_size=period, ddof=ddof) * nbdev


@feature(
    name="stddev",
    category="statistics",
    description="STDDEV - Standard Deviation",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def stddev(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 5,
    nbdev: float = 1.0,
    ddof: int = 0,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Standard Deviation exactly matching TA-Lib.

    Calculates the standard deviation of close over a rolling window,
    with optional scaling by nbdev (number of standard deviations).

    Parameters
    ----------
    close : array-like or str
        Input close or column name
    period : int, default 5
        Number of periods for calculation
    nbdev : float, default 1.0
        Number of standard deviations (scaling factor)
    ddof : int, default 0
        Delta degrees of freedom (0 for population, 1 for sample)
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        Standard deviation close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    >>> stddev_values = stddev(close, period=3, nbdev=2.0)

    Notes
    -----
    - First 'period-1' close are NaN (need full window)
    - TA-Lib uses ddof=0 (population standard deviation) by default
    - nbdev scales the result (e.g., 2.0 for 2 standard deviations)
    """
    if isinstance(close, str):
        # Column name provided for Polars
        return stddev_polars(close, period, nbdev, ddof)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return stddev_numba(close, period, nbdev, ddof)


# Export all functions
__all__ = ["stddev", "stddev_numba", "stddev_polars"]
