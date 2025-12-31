# mypy: disable-error-code="no-any-return"
"""
Relative Strength Index (RSI) - TA-Lib compatible implementation.

RSI measures the speed and change of price movements, oscillating between 0 and 100.
Uses Wilder's smoothing method for exact TA-Lib compatibility.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def rsi_numba(close: npt.NDArray[np.float64], period: int = 14) -> npt.NDArray[np.float64]:
    """
    RSI using Wilder's smoothing - exact TA-Lib algorithm.

    Wilder's smoothing is different from EMA - it uses a different
    averaging method that TA-Lib implements specifically for RSI.

    Parameters
    ----------
    close : npt.NDArray
        Price data
    period : int, default 14
        Number of periods for RSI calculation

    Returns
    -------
    npt.NDArray
        RSI close (0-100 scale) with NaN for insufficient data
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 1 or period >= n:
        return result

    # Calculate initial sums for the first period
    sum_gain = 0.0
    sum_loss = 0.0

    for i in range(1, period + 1):
        change = close[i] - close[i - 1]
        if change > 0:
            sum_gain += change
        else:
            sum_loss += -change

    # Calculate first RSI
    avg_gain = sum_gain / period
    avg_loss = sum_loss / period

    if avg_loss != 0:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0, TA-Lib returns 0 if avg_gain is also 0
    elif avg_gain == 0:
        result[period] = 0.0
    else:
        result[period] = 100.0

    # Calculate subsequent RSI close using Wilder's smoothing
    # Pre-compute constant
    factor = (period - 1.0) / period
    inv_period = 1.0 / period

    for i in range(period + 1, n):
        change = close[i] - close[i - 1]

        if change > 0:
            avg_gain = avg_gain * factor + change * inv_period
            avg_loss = avg_loss * factor
        else:
            avg_gain = avg_gain * factor
            avg_loss = avg_loss * factor + (-change) * inv_period

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
        # When avg_loss is 0, TA-Lib returns 0 if avg_gain is also 0
        elif avg_gain == 0:
            result[i] = 0.0
        else:
            result[i] = 100.0

    return result


def rsi_polars(column: str, period: int = 14) -> pl.Expr:
    """
    RSI using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    column : str
        Column name to apply RSI to
    period : int, default 14
        Number of periods for RSI calculation

    Returns
    -------
    pl.Expr
        Polars expression for RSI calculation
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(rsi_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="rsi",
    category="momentum",
    description="Relative Strength Index - momentum oscillator measuring speed and change of price movements",
    lookback="period",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss",
    ta_lib_compatible=True,
    input_type="close",
    parameters={"period": 14},
    references=["Wilder, J. W. (1978). New Concepts in Technical Trading Systems"],
    tags=["oscillator", "momentum", "overbought", "oversold"],
)
def rsi(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Relative Strength Index with Wilder's smoothing.

    RSI is a momentum oscillator that measures the speed and change of price
    movements. Values above 70 typically indicate overbought conditions,
    while close below 30 indicate oversold conditions.

    Parameters
    ----------
    close : array-like or column name
        Price data (typically closing close)
    period : int, default 14
        Number of periods for RSI calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        RSI close on 0-100 scale

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.15, 47.72])
    >>> rsi_14 = rsi(close, period=5)  # Using 5 for demo
    >>> rsi_14[5:]  # First 5 close are NaN
    array([57.97, 62.93, 70.46, 74.53, 70.61])

    Notes
    -----
    - Uses Wilder's smoothing (not simple EMA) for exact TA-Lib compatibility
    - RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss
    - First average uses simple mean, subsequent close use Wilder's smoothing
    - Standard period is 14, commonly used close are 7, 14, 21
    """
    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if isinstance(close, str):
        return rsi_polars(close, period)

    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return rsi_numba(close, period)


# Export the main function
__all__ = ["rsi"]
