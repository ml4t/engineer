"""
Stochastic RSI (STOCHRSI) - TA-Lib compatible implementation.

STOCHRSI applies the Stochastic Oscillator formula to RSI close
instead of price data, creating a more sensitive momentum indicator.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.momentum.rsi import rsi_numba
from ml4t.engineer.features.utils.ma_types import apply_ma


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def stochrsi_fastk_numba(
    close: npt.NDArray[np.float64],
    timeperiod: int = 14,
    fastk_period: int = 5,
) -> npt.NDArray[np.float64]:
    """
    Calculate STOCHRSI %K using Numba for performance.

    Parameters
    ----------
    close : npt.NDArray
        Price data (typically closing close)
    timeperiod : int, default 14
        Period for RSI calculation
    fastk_period : int, default 5
        Period for Stochastic calculation on RSI

    Returns
    -------
    npt.NDArray
        The STOCHRSI %K close (0-100 scale)
    """
    n = len(close)

    # Calculate RSI first
    rsi_values = rsi_numba(close, timeperiod)

    # Initialize output array
    fastk = np.full(n, np.nan)

    # Find first valid RSI
    first_rsi = -1
    for i in range(n):
        if not np.isnan(rsi_values[i]):
            first_rsi = i
            break

    if first_rsi == -1 or first_rsi + fastk_period > n:
        return fastk

    # Calculate Stochastic on RSI close
    # TA-Lib uses RSI as high, low, and close for STOCHF
    for i in range(first_rsi + fastk_period - 1, n):
        # Get window of RSI close
        start_idx = i - fastk_period + 1

        # Find highest and lowest RSI in period
        highest_rsi = rsi_values[start_idx]
        lowest_rsi = rsi_values[start_idx]

        for j in range(start_idx + 1, i + 1):
            highest_rsi = max(highest_rsi, rsi_values[j])
            lowest_rsi = min(lowest_rsi, rsi_values[j])

        # Calculate Stochastic RSI (FastK)
        diff = highest_rsi - lowest_rsi
        if diff > 0:
            fastk[i] = 100.0 * (rsi_values[i] - lowest_rsi) / diff
        else:
            # When range is 0, TA-Lib returns 0
            fastk[i] = 0.0

    return fastk


def stochrsi_numba(
    close: npt.NDArray[np.float64],
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate STOCHRSI %K and %D.

    Parameters
    ----------
    close : npt.NDArray
        Price data
    timeperiod : int, default 14
        Period for RSI
    fastk_period : int, default 5
        Period for Stochastic on RSI
    fastd_period : int, default 3
        Period for %D smoothing
    fastd_matype : int, default 0
        Moving average type for %D

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (%K, %D) arrays
    """
    # Calculate %K
    fastk = stochrsi_fastk_numba(close, timeperiod, fastk_period)

    # Calculate %D using specified MA type
    fastd = apply_ma(fastk, fastd_period, fastd_matype)

    # TA-Lib compatibility: Apply lookback adjustment
    # Total lookback = RSI lookback + STOCH lookback
    total_lookback = timeperiod + fastk_period + fastd_period - 2
    if total_lookback > 0:
        fastk[:total_lookback] = np.nan
        fastd[:total_lookback] = np.nan

    return fastk, fastd


def stochrsi_polars(
    column: str,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> pl.Expr:
    """
    STOCHRSI using Polars - delegates to Numba for exact TA-Lib compatibility.

    Returns only the %K value for single-value compatibility.

    Parameters
    ----------
    column : str
        Column name for price data
    timeperiod : int, default 14
        Period for RSI calculation
    fastk_period : int, default 5
        Period for Stochastic calculation on RSI
    fastd_period : int, default 3
        Smoothing period for %D

    Returns
    -------
    pl.Expr
        Polars expression for STOCHRSI %K
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(
            stochrsi_numba(s.to_numpy(), timeperiod, fastk_period, fastd_period, fastd_matype)[
                0
            ],  # Return only fastk
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="stochrsi",
    category="momentum",
    description="Stochastic RSI - applies stochastic formula to RSI",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def stochrsi(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
    return_pair: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | pl.Expr:
    """
    Stochastic RSI exactly matching TA-Lib.

    STOCHRSI combines RSI and Stochastic Oscillator to create a more
    sensitive indicator. It measures the level of RSI relative to its
    high-low range over a given period.

    Parameters
    ----------
    close : array-like or str
        Price data or column name
    timeperiod : int, default 14
        Period for RSI calculation
    fastk_period : int, default 5
        Period for Stochastic calculation on RSI
    fastd_period : int, default 3
        Smoothing period for %D
    return_pair : bool, default False
        If True, return tuple (fastk, fastd). If False, return only fastk.
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array, tuple of arrays, or Polars expression
        If return_pair=False: STOCHRSI %K close only
        If return_pair=True: (fastk, fastd) tuple
        For Polars: expression for %K only

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85])
    >>> stochrsi_k, stochrsi_d = stochrsi(close, return_pair=True)

    Notes
    -----
    - First calculate RSI for the price data
    - Then apply Stochastic formula to RSI close
    - %K = 100 × (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
    - %D = SMA of %K
    - Values range from 0 to 100
    - More sensitive than regular RSI or Stochastic
    """
    # Validate parameters
    if timeperiod < 1:
        raise InvalidParameterError(f"timeperiod must be >= 1, got {timeperiod}")
    if fastk_period < 1:
        raise InvalidParameterError(f"fastk_period must be >= 1, got {fastk_period}")
    if fastd_period < 1:
        raise InvalidParameterError(f"fastd_period must be >= 1, got {fastd_period}")

    if isinstance(close, str):
        # Column name provided for Polars
        return stochrsi_polars(close, timeperiod, fastk_period, fastd_period, fastd_matype)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    fastk, fastd = stochrsi_numba(close, timeperiod, fastk_period, fastd_period, fastd_matype)

    if return_pair:
        return fastk, fastd
    return fastk


# Export all functions
__all__ = ["stochrsi", "stochrsi_numba", "stochrsi_polars"]
