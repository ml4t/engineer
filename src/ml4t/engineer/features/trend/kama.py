# mypy: disable-error-code="no-any-return"
"""
KAMA - Kaufman's Adaptive Moving Average.

This is an adaptive moving average that adjusts its smoothing constant based on
market efficiency (directional movement vs volatility).
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def kama_numba(close: npt.NDArray[np.float64], timeperiod: int = 30) -> npt.NDArray[np.float64]:
    """
    KAMA calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 30
        Time period for efficiency ratio calculation

    Returns
    -------
    npt.NDArray
        KAMA close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Constants for smoothing (SC = smoothing constant)
    # Fast SC = 2/(2+1) = 0.6667
    # Slow SC = 2/(30+1) = 0.0645
    const_max = 2.0 / (30.0 + 1.0)  # Slow SC
    const_diff = 2.0 / (2.0 + 1.0) - const_max  # Fast SC - Slow SC

    # Need at least timeperiod close
    if n < timeperiod:
        return result

    # Initialize variables
    sum_roc1 = 0.0  # Sum of absolute 1-period changes

    # Calculate initial sum of 1-period absolute changes
    for i in range(timeperiod - 1):
        sum_roc1 += abs(close[i + 1] - close[i])

    # Skip until we have enough data (TA-Lib uses unstable period)
    # We'll start from timeperiod-1 for simplicity
    start_idx = timeperiod - 1

    # Initialize KAMA with the first available price
    prev_kama = close[start_idx]

    # Process each value
    for today in range(start_idx + 1, n):
        trailing_idx = today - timeperiod

        # Calculate period change (direction)
        period_roc = close[today] - close[trailing_idx]

        # Update sum of 1-period changes
        # Remove the oldest 1-period change
        if trailing_idx > 0:
            sum_roc1 -= abs(close[trailing_idx] - close[trailing_idx - 1])
        # Add the newest 1-period change
        sum_roc1 += abs(close[today] - close[today - 1])

        # Calculate Efficiency Ratio (ER)
        # ER = Direction / Volatility
        if sum_roc1 <= abs(period_roc) or sum_roc1 == 0:
            efficiency_ratio = 1.0
        else:
            efficiency_ratio = abs(period_roc) / sum_roc1

        # Calculate smoothing constant
        # SC = (ER * (Fast SC - Slow SC) + Slow SC)^2
        smoothing_constant = efficiency_ratio * const_diff + const_max
        smoothing_constant = smoothing_constant * smoothing_constant

        # Calculate KAMA using EMA formula with adaptive smoothing
        # KAMA = KAMA_prev + SC * (Price - KAMA_prev)
        prev_kama = prev_kama + smoothing_constant * (close[today] - prev_kama)
        result[today] = prev_kama

    return result


def kama_polars(col: str, timeperiod: int = 30) -> pl.Expr:
    """
    KAMA using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 30
        Time period for efficiency ratio calculation

    Returns
    -------
    pl.Expr
        Polars expression for KAMA calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(kama_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="kama",
    category="trend",
    description="KAMA - Kaufman Adaptive Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def kama(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 30,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    KAMA - Kaufman's Adaptive Moving Average.

    Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to
    account for market noise or volatility. KAMA will closely follow close
    when the price swings are relatively small and the noise is low. KAMA
    will adjust when the price swings widen and follow close from a greater
    distance. This trend-following indicator can be used to identify the
    overall trend, time turning points and filter price movements.

    Parameters
    ----------
    close : array-like or column name
        Input close (typically close close)
    timeperiod : int, default 30
        Time period for efficiency ratio calculation

    Returns
    -------
    array or Polars expression
        KAMA close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 11, 10, 11, 13, 14, 13, 12])
    >>> kama_values = kama(close, timeperiod=5)

    Notes
    -----
    The KAMA calculation involves:
    1. Efficiency Ratio (ER) = Direction / Volatility
       - Direction = ABS(Close - Close[n periods ago])
       - Volatility = Sum[ABS(Close - Close[1 period ago])]
    2. Smoothing Constant (SC) = [ER * (fastest SC - slowest SC) + slowest SC]^2
    3. KAMA = Previous KAMA + SC * (Price - Previous KAMA)

    TA-Lib uses an unstable period for KAMA, but we start outputting
    immediately after the minimum required period for simplicity.
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return kama_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return kama_numba(close, timeperiod)


# Export the main function
__all__ = ["kama"]
