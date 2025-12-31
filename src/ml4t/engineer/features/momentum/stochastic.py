# mypy: disable-error-code="no-any-return"
"""
STOCHASTIC - TA-Lib compatible implementation.

The Stochastic Oscillator measures the closing price relative to the high-low range
over a specified period. %K is the raw stochastic, %D is a moving average of %K.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.trend.sma import sma_numba


@jit(nopython=True, cache=False, fastmath=False)  # type: ignore[misc]
def stochastic_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    fastk_period: int = 14,
    slowk_period: int = 1,
    slowd_period: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Stochastic calculation exactly matching TA-Lib.

    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    Slow %K = SMA of %K over slowk_period
    Slow %D = SMA of Slow %K over slowd_period

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    fastk_period : int, default 14
        Period for %K calculation
    slowk_period : int, default 1
        Period for slow %K (smoothing, 1 = no smoothing = fast %K)
    slowd_period : int, default 3
        Period for slow %D (smoothing of smoothed %K)

    Returns
    -------
    tuple of npt.NDArray
        (slowk, slowd) - The smoothed stochastic close
    """
    n = len(high)

    # Calculate raw %K (fast %K)
    fastk = np.full(n, np.nan)

    if fastk_period < 1 or n < fastk_period:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Calculate %K for each period
    # TA-Lib starts fastk at index (fastk_period - 1) for the rolling window,
    # then SMAs add (slowk_period - 1) and (slowd_period - 1) respectively.
    # Total NaNs = (fastk_period - 1) + (slowk_period - 1) + (slowd_period - 1)
    # For slowk: (fastk_period - 1) + (slowk_period - 1)
    start_idx = fastk_period + slowk_period - 2
    for i in range(start_idx, n):
        # Find highest high and lowest low in the fastk_period window
        window_start = i - fastk_period + 1
        highest_high = high[window_start]
        lowest_low = low[window_start]

        for j in range(window_start + 1, i + 1):
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])

        # Calculate %K
        diff = highest_high - lowest_low
        if diff > 0:
            fastk[i] = 100.0 * (close[i] - lowest_low) / diff
        else:
            # When high == low for entire period, %K = 0
            fastk[i] = 0.0

    # Calculate slow %K (smoothed fast %K)
    slowk = sma_numba(fastk, slowk_period)

    # Calculate slow %D (smoothed slow %K)
    # TA-Lib's STOCH starts slowd at the same index as slowk by computing
    # the SMA with whatever window is available (partial window initially)
    slowd = np.full(n, np.nan)

    # Start slowd calculation as soon as we have slowk values
    slowk_start = np.where(~np.isnan(slowk))[0]
    if len(slowk_start) > 0:
        first_slowk_idx = slowk_start[0]

        # TA-Lib computes slowd starting at first_slowk_idx using available data
        for i in range(first_slowk_idx, n):
            # Use standard window [i-period+1:i+1], but start from first_slowk_idx
            window_start = max(first_slowk_idx, i - slowd_period + 1)
            window_end = i + 1

            # Get slowk values in window
            window_vals = slowk[window_start:window_end]

            # Compute mean of non-NaN values (TA-Lib behavior)
            valid_vals = window_vals[~np.isnan(window_vals)]
            if len(valid_vals) > 0:
                slowd[i] = np.mean(valid_vals)

    return slowk, slowd


def stochastic_polars(
    high: str,
    low_col: str,
    close_col: str,
    fastk_period: int = 14,
    slowk_period: int = 1,
    slowd_period: int = 3,
) -> pl.Expr:
    """
    Stochastic using Polars - delegates to Numba for exact TA-Lib compatibility.

    Returns only the %K value (not %D) for single-value compatibility.

    Parameters
    ----------
    high : str
        Column name for high close
    low_col : str
        Column name for low close
    close_col : str
        Column name for close close
    fastk_period : int, default 14
        Period for %K calculation
    slowk_period : int, default 1
        Period for slow %K
    slowd_period : int, default 3
        Period for slow %D

    Returns
    -------
    pl.Expr
        Polars expression for slow %K
    """
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda s: pl.Series(
            stochastic_numba(
                s.struct.field(high).to_numpy(),
                s.struct.field(low_col).to_numpy(),
                s.struct.field(close_col).to_numpy(),
                fastk_period,
                slowk_period,
                slowd_period,
            )[0],  # Return only slowk
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="stochastic",
    category="momentum",
    description="Stochastic Oscillator - %K and %D",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=False,
)
def stochastic(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    fastk_period: int = 14,
    slowk_period: int = 1,
    slowd_period: int = 3,
    return_pair: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | pl.Expr:
    """
    Stochastic Oscillator exactly matching TA-Lib.

    Calculates the Stochastic Oscillator (%K and %D lines).
    %K measures where the close is relative to the recent trading range.
    %D is a moving average of %K.

    Parameters
    ----------
    high : array-like or str
        High close or column name
    low : array-like or str
        Low close or column name
    close : array-like or str
        Close close or column name
    fastk_period : int, default 14
        Period for %K calculation
    slowk_period : int, default 1
        Period for slow %K (smoothing, 1 = no smoothing = fast %K)
    slowd_period : int, default 3
        Period for slow %D (smoothing of smoothed %K)
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    return_pair : bool, default False
        If True, return tuple (slowk, slowd). If False, return only slowk.
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array, tuple of arrays, or Polars expression
        If return_pair=False: slow %K close only
        If return_pair=True: (slowk, slowd) tuple
        For Polars: expression for slow %K only

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([127.01, 127.62, 126.59, 127.35, 128.17])
    >>> low = np.array([125.36, 126.16, 124.93, 126.09, 126.82])
    >>> close = np.array([125.83, 126.48, 126.03, 126.39, 127.18])
    >>> slowk, slowd = stochastic(high, low, close, 14, 3, 3)

    Notes
    -----
    - Values range from 0 to 100
    - Above 80 is considered overbought
    - Below 20 is considered oversold
    - When high == low for entire period, %K = 0
    """
    # Validate parameters
    if fastk_period < 1:
        raise InvalidParameterError(f"fastk_period must be >= 1, got {fastk_period}")
    if slowk_period < 1:
        raise InvalidParameterError(f"slowk_period must be >= 1, got {slowk_period}")
    if slowd_period < 1:
        raise InvalidParameterError(f"slowd_period must be >= 1, got {slowd_period}")

    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        # Column names provided for Polars
        return stochastic_polars(
            high,
            low,
            close,
            fastk_period,
            slowk_period,
            slowd_period,
        )

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    slowk, slowd = stochastic_numba(
        high,
        low,
        close,
        fastk_period,
        slowk_period,
        slowd_period,
    )
    if return_pair:
        return slowk, slowd
    return slowk


# Export all functions
__all__ = ["stochastic", "stochastic_numba", "stochastic_polars"]
