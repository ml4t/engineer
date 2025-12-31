"""
BOLLINGER_BANDS - TA-Lib compatible implementation.

Bollinger Bands consist of a middle band (SMA), an upper band (SMA + k*stddev),
and a lower band (SMA - k*stddev).
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.statistics.stddev import stddev_numba
from ml4t.engineer.features.trend.sma import sma_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def bollinger_bands_numba(
    close: npt.NDArray[np.float64],
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Bollinger Bands calculation exactly matching TA-Lib.

    Parameters
    ----------
    close : npt.NDArray
        Input close (typically close close)
    period : int, default 20
        Number of periods for moving average
    nbdevup : float, default 2.0
        Number of standard deviations for upper band
    nbdevdn : float, default 2.0
        Number of standard deviations for lower band

    Returns
    -------
    tuple of npt.NDArray
        (upper_band, middle_band, lower_band)
    """
    n = len(close)
    upper_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    if period < 2 or n < period:
        return upper_band, middle_band, lower_band

    # Calculate SMA (middle band)
    middle_band = sma_numba(close, period)

    # Calculate standard deviation
    stddev_up = stddev_numba(close, period, nbdevup, ddof=0)
    stddev_dn = stddev_numba(close, period, nbdevdn, ddof=0)

    # Calculate upper and lower bands
    for i in range(n):
        if not np.isnan(middle_band[i]):
            if not np.isnan(stddev_up[i]):
                upper_band[i] = middle_band[i] + stddev_up[i]
            if not np.isnan(stddev_dn[i]):
                lower_band[i] = middle_band[i] - stddev_dn[i]

    return upper_band, middle_band, lower_band


def bollinger_bands_polars(
    col: str,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> pl.Expr:
    """
    Bollinger Bands using Polars expressions.

    Returns a struct with upper, middle, and lower bands.

    Parameters
    ----------
    col : str
        Column name for close
    period : int, default 20
        Number of periods
    nbdevup : float, default 2.0
        Number of standard deviations for upper band
    nbdevdn : float, default 2.0
        Number of standard deviations for lower band

    Returns
    -------
    pl.Expr
        Polars expression returning struct with upper, middle, lower
    """
    # Calculate components
    middle = pl.col(col).rolling_mean(window_size=period)
    stddev = pl.col(col).rolling_std(window_size=period, ddof=0)
    upper = middle + (stddev * nbdevup)
    lower = middle - (stddev * nbdevdn)

    # Return as struct
    return pl.struct(
        [upper.alias("upper"), middle.alias("middle"), lower.alias("lower")],
    )


@feature(
    name="bollinger_bands",
    category="volatility",
    description="Bollinger Bands - volatility bands around MA",
    lookback=20,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def bollinger_bands(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] | pl.Expr:
    """
    Bollinger Bands exactly matching TA-Lib.

    Calculates upper band, middle band (SMA), and lower band based on
    standard deviations from the middle band.

    Parameters
    ----------
    close : array-like or str
        Input close or column name
    period : int, default 20
        Number of periods for moving average
    nbdevup : float, default 2.0
        Number of standard deviations for upper band
    nbdevdn : float, default 2.0
        Number of standard deviations for lower band
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    tuple of arrays or Polars expression
        For arrays: (upper_band, middle_band, lower_band)
        For Polars: struct expression with upper, middle, lower fields

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83])
    >>> upper, middle, lower = bollinger_bands(close, period=5, nbdevup=2, nbdevdn=2)

    Notes
    -----
    - First 'period-1' close are NaN (need full window)
    - Middle band is a simple moving average
    - Upper band = middle + (nbdevup * stddev)
    - Lower band = middle - (nbdevdn * stddev)
    - Uses population standard deviation (ddof=0)
    """
    # Validate parameters
    if period < 2:
        raise InvalidParameterError(f"period must be >= 2, got {period}")
    if nbdevup <= 0:
        raise InvalidParameterError(f"std_dev must be > 0, got {nbdevup}")
    if nbdevdn <= 0:
        raise InvalidParameterError(f"std_dev must be > 0, got {nbdevdn}")

    if isinstance(close, str):
        # Column name provided for Polars
        return bollinger_bands_polars(close, period, nbdevup, nbdevdn)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return tuple(bollinger_bands_numba(close, period, nbdevup, nbdevdn))


# Export all functions
__all__ = ["bollinger_bands", "bollinger_bands_numba", "bollinger_bands_polars"]
