# mypy: disable-error-code="no-any-return"
"""
Chaikin A/D Line (AD) - TA-Lib compatible implementation.

The Accumulation/Distribution Line is a volume-based indicator designed to measure
the cumulative flow of money into and out of a security.

Formula:
1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
2. Money Flow Volume = Money Flow Multiplier * Volume
3. A/D Line = Previous A/D Line + Current Money Flow Volume

When High == Low, the Money Flow Multiplier is set to 0.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def ad_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Chaikin A/D Line calculation using Numba.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    volume : npt.NDArray
        Volume close

    Returns
    -------
    npt.NDArray
        A/D Line close
    """
    n = len(high)
    result = np.zeros(n)
    ad_value = 0.0

    for i in range(n):
        # Check for NaN in any input
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or np.isnan(volume[i]):
            # Once we hit a NaN, all subsequent close are NaN
            for j in range(i, n):
                result[j] = np.nan
            break

        # Calculate Money Flow Multiplier
        hl_diff = high[i] - low[i]

        # MFM = ((Close - Low) - (High - Close)) / (High - Low) if hl_diff != 0 else 0
        mfm = ((close[i] - low[i]) - (high[i] - close[i])) / hl_diff if hl_diff != 0 else 0.0

        # Calculate Money Flow Volume
        mfv = mfm * volume[i]

        # Accumulate
        ad_value += mfv
        result[i] = ad_value

    return result


def ad_polars(high: str, low_col: str, close_col: str, volume_col: str) -> pl.Expr:
    """
    Chaikin A/D Line using Polars expressions.

    Parameters
    ----------
    high : str
        Name of high price column
    low_col : str
        Name of low price column
    close_col : str
        Name of close price column
    volume_col : str
        Name of volume column

    Returns
    -------
    pl.Expr
        Polars expression for A/D Line calculation
    """
    # Calculate Money Flow Multiplier
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    # When High == Low, use 0
    hl_diff = pl.col(high) - pl.col(low_col)

    mfm = (
        pl.when(hl_diff != 0)
        .then(
            ((pl.col(close_col) - pl.col(low_col)) - (pl.col(high) - pl.col(close_col))) / hl_diff,
        )
        .otherwise(0.0)
    )

    # Money Flow Volume
    mfv = mfm * pl.col(volume_col)

    # Cumulative sum for A/D Line
    return mfv.cum_sum()


@feature(
    name="ad",
    category="volume",
    description="AD - Accumulation/Distribution",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def ad(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    volume: npt.NDArray[np.float64] | pl.Series | str,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Chaikin A/D Line.

    Measures the cumulative flow of money into and out of a security by
    analyzing the relationship between closing price and the high-low range,
    weighted by volume.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    volume : array-like or column name
        Volume close

    Returns
    -------
    array or Polars expression
        A/D Line close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([102.0, 103.0, 104.0])
    >>> low = np.array([98.0, 99.0, 100.0])
    >>> close = np.array([101.0, 100.0, 103.0])
    >>> volume = np.array([1000.0, 2000.0, 1500.0])
    >>> ad_line = ad(high, low, close, volume)

    Notes
    -----
    - The A/D Line is a cumulative indicator
    - When High == Low, the Money Flow Multiplier is 0
    - NaN close propagate forward in the cumulative sum
    - Useful for confirming price trends with volume
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if (
        isinstance(high, str)
        and isinstance(low, str)
        and isinstance(close, str)
        and isinstance(volume, str)
    ):
        return ad_polars(high, low, close, volume)
    if implementation == "polars":
        raise ValueError(
            "Polars implementation requires all inputs to be column names (strings)",
        )

    # Convert to numpy arrays
    high = high.to_numpy() if isinstance(high, pl.Series) else np.asarray(high, dtype=np.float64)
    low = low.to_numpy() if isinstance(low, pl.Series) else np.asarray(low, dtype=np.float64)
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    else:
        close = np.asarray(close, dtype=np.float64)
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()
    else:
        volume = np.asarray(volume, dtype=np.float64)

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("high, low, close, and volume must have the same length")

    # Use Numba implementation by default or when requested
    if implementation in {"numba", "auto"}:
        return ad_numba(high, low, close, volume)
    # Pure NumPy fallback would go here if needed
    return ad_numba(high, low, close, volume)


# Export the main function
__all__ = ["ad"]
