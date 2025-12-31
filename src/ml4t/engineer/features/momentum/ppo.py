# mypy: disable-error-code="no-any-return"
"""
Percentage Price Oscillator (PPO) - TA-Lib compatible implementation.

PPO is a momentum oscillator that measures the difference between two
moving averages as a percentage of the slower moving average.
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
def ppo_numba(
    close: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> npt.NDArray[np.float64]:
    """
    PPO calculation exactly replicating TA-Lib algorithm.

    PPO = ((Fast MA - Slow MA) / Slow MA) * 100

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
        Currently supports SMA (0) and EMA (1)

    Returns
    -------
    npt.NDArray
        PPO close (percentage)
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

    # PPO calculation: ((fast - slow) / slow) * 100
    # Only calculate where both MAs are valid
    for i in range(n):
        if not np.isnan(slow_ma[i]) and not np.isnan(fast_ma[i]):
            if slow_ma[i] != 0.0:
                result[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
            else:
                result[i] = 0.0

    return result


def ppo_polars(
    column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> pl.Expr:
    """
    PPO using Polars - delegates to Numba for exact TA-Lib compatibility.

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
        Polars expression for PPO
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(ppo_numba(s.to_numpy(), fast_period, slow_period, matype)),
        return_dtype=pl.Float64,
    )


@feature(
    name="ppo",
    category="momentum",
    description="PPO - Percentage Price Oscillator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def ppo(
    close: npt.NDArray[np.float64] | pl.Series | str,
    fast_period: int = 12,
    slow_period: int = 26,
    matype: int = 0,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Percentage Price Oscillator exactly matching TA-Lib.

    PPO is a momentum oscillator that shows the relationship between two
    moving averages in percentage terms. It's similar to MACD but normalized
    as a percentage, making it easier to compare across different securities.

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
        PPO close in percentage terms

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> ppo_values = ppo(close, fast_period=3, slow_period=5)

    Notes
    -----
    - PPO = ((Fast MA - Slow MA) / Slow MA) * 100
    - Positive close indicate upward momentum
    - Negative close indicate downward momentum
    - Zero line crossovers can signal trend changes
    - More suitable than MACD for comparing different securities
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
        return ppo_polars(close, fast_period, slow_period, matype)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return ppo_numba(close, fast_period, slow_period, matype)


# Export all functions
__all__ = ["ppo", "ppo_numba", "ppo_polars"]
