# mypy: disable-error-code="no-any-return"
"""
True Range (TRANGE) - TA-Lib compatible implementation.

True Range is the greatest of:
- Current High minus current Low
- Absolute value of current High minus previous Close
- Absolute value of current Low minus previous Close

Used as a volatility measure and component of ATR.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def trange_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    True Range calculation using Numba.

    True Range = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close

    Returns
    -------
    npt.NDArray
        True Range close with first value as NaN
    """
    n = len(high)
    result = np.full(n, np.nan)

    # Need at least 2 close (current and previous)
    if n < 1:
        return result

    # First value is NaN (no previous close)
    # Starting from second value
    for i in range(1, n):
        prev_close = close[i - 1]

        # Three components of True Range
        hl = high[i] - low[i]  # High-Low range
        hc = abs(high[i] - prev_close)  # High-Close range
        lc = abs(low[i] - prev_close)  # Low-Close range

        # True Range is the maximum
        result[i] = max(hl, hc, lc)

    return result


def trange_polars(high: str, low_col: str, close_col: str) -> pl.Expr:
    """
    True Range using Polars expressions.

    Parameters
    ----------
    high : str
        Name of high price column
    low_col : str
        Name of low price column
    close_col : str
        Name of close price column

    Returns
    -------
    pl.Expr
        Polars expression for True Range calculation
    """
    # Get previous close
    prev_close = pl.col(close_col).shift(1)

    # Calculate three components
    hl_range = pl.col(high) - pl.col(low_col)
    hc_range = (pl.col(high) - prev_close).abs()
    lc_range = (pl.col(low_col) - prev_close).abs()

    # Return maximum of the three, but with first value as null (no previous close)
    return (
        pl.when(prev_close.is_null())
        .then(None)
        .otherwise(pl.max_horizontal(hl_range, hc_range, lc_range))
    )


@feature(
    name="trange",
    category="volatility",
    description="TRANGE - True Range",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def trange(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    True Range (TRANGE) - Volatility indicator.

    True Range is the greatest of:
    - Current High minus current Low
    - Absolute value of current High minus previous Close
    - Absolute value of current Low minus previous Close

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close

    Returns
    -------
    array or Polars expression
        True Range close with first value as NaN

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10.0, 11.0, 12.0, 11.5])
    >>> low = np.array([9.0, 9.5, 10.5, 10.0])
    >>> close = np.array([9.5, 10.5, 11.0, 10.5])
    >>> tr = trange(high, low, close)
    >>> tr[1:]  # First value is NaN
    array([1.5, 1.5, 1.5])

    Notes
    -----
    - First value is always NaN (no previous close)
    - Used as component in ATR (Average True Range)
    - Measures volatility by considering gaps between periods
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return trange_polars(high, low, close)
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

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    # Use Numba implementation by default or when requested
    if implementation in {"numba", "auto"}:
        return trange_numba(high, low, close)
    # Pure NumPy fallback
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    result[0] = np.nan  # First value is always NaN

    if n < 2:
        return result

    # Vectorized calculation for all close from index 1 onwards
    prev_close = close[:-1]  # Previous close close [0 to n-2]
    curr_high = high[1:]  # Current high close [1 to n-1]
    curr_low = low[1:]  # Current low close [1 to n-1]

    # Three components of True Range (vectorized)
    hl_range = curr_high - curr_low  # High-Low range
    hc_range = np.abs(curr_high - prev_close)  # High-Close range
    lc_range = np.abs(curr_low - prev_close)  # Low-Close range

    # True Range is the maximum of the three components
    result[1:] = np.maximum(hl_range, np.maximum(hc_range, lc_range))

    return result


# Export the main function
__all__ = ["trange"]
