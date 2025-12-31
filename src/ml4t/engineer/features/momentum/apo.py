# mypy: disable-error-code="no-any-return"
"""
Absolute Price Oscillator (APO) - TA-Lib compatible implementation.

APO is a momentum oscillator that measures the absolute difference between two
moving averages, unlike PPO which shows the percentage difference.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.trend.ema import ema_numba
from ml4t.engineer.features.trend.sma import sma_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def apo_numba(
    close: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> npt.NDArray[np.float64]:
    """
    APO calculation exactly replicating TA-Lib algorithm.

    APO = Fast MA - Slow MA

    Parameters
    ----------
    close : npt.NDArray
        Price data (typically closing close)
    fast_period : int, default 12
        Period for the fast moving average
    slow_period : int, default 26
        Period for the slow moving average
    matype : int, default 0
        Type of moving average (0=SMA, 1=EMA, etc.)
        Currently only EMA (1) is implemented

    Returns
    -------
    npt.NDArray
        APO close (absolute difference)
    """
    n = len(close)

    # Swap if fast is slower than slow
    if slow_period < fast_period:
        fast_period, slow_period = slow_period, fast_period

    # Calculate fast and slow MAs based on matype
    if matype == 0:  # SMA
        fast_ma = sma_numba(close, fast_period)
        slow_ma = sma_numba(close, slow_period)
    elif matype == 1:  # EMA
        fast_ma = ema_numba(close, fast_period)
        slow_ma = ema_numba(close, slow_period)
    else:
        # Return NaN array for unsupported MA types
        return np.full(n, np.nan)

    # Initialize result array
    result = np.full(n, np.nan)

    # APO calculation: fast - slow
    # Only calculate where both MAs are valid
    for i in range(n):
        if not np.isnan(slow_ma[i]) and not np.isnan(fast_ma[i]):
            result[i] = fast_ma[i] - slow_ma[i]

    return result


def apo_polars(
    column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> pl.Expr:
    """
    APO using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    column : str
        Column name for price data
    fast_period : int, default 12
        Period for the fast moving average
    slow_period : int, default 26
        Period for the slow moving average
    matype : int, default 1
        Type of moving average (currently only 1=EMA is supported)

    Returns
    -------
    pl.Expr
        Polars expression for APO
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(apo_numba(s.to_numpy(), fast_period, slow_period, matype)),
        return_dtype=pl.Float64,
    )


@feature(
    name="apo",
    category="momentum",
    description="APO - Absolute Price Oscillator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def apo(
    close: npt.NDArray[np.float64] | pl.Series | str,
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Absolute Price Oscillator exactly matching TA-Lib.

    APO is a momentum oscillator that shows the absolute difference between two
    moving averages. Unlike PPO which shows percentage differences, APO shows
    the raw point difference, making it suitable for price-level analysis.

    Parameters
    ----------
    close : array-like or str
        Price data or column name
    fast_period : int, default 12
        Period for the fast moving average
    slow_period : int, default 26
        Period for the slow moving average
    matype : int, default 0
        Type of moving average (0=SMA, 1=EMA)
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        APO close (absolute difference)

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> apo_values = apo(close, fast_period=3, slow_period=5)

    Notes
    -----
    - APO = Fast MA - Slow MA
    - Positive close indicate fast MA is above slow MA (upward momentum)
    - Negative close indicate fast MA is below slow MA (downward momentum)
    - Zero line crossovers can signal trend changes
    - Unlike PPO, APO close are in the same units as the price data
    - Supports SMA (matype=0) and EMA (matype=1)
    """
    # Validate parameters
    if fast_period < 2:
        raise InvalidParameterError(f"fast_period must be >= 2, got {fast_period}")
    if slow_period < 2:
        raise InvalidParameterError(f"slow_period must be >= 2, got {slow_period}")
    if matype not in (0, 1):
        raise InvalidParameterError(f"matype must be 0 (SMA) or 1 (EMA), got {matype}")

    if isinstance(close, str):
        # Column name provided for Polars
        return apo_polars(close, fast_period, slow_period, matype)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return apo_numba(close, fast_period, slow_period, matype)


# Export all functions
__all__ = ["apo", "apo_numba", "apo_polars"]
