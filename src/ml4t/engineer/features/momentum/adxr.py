# mypy: disable-error-code="no-any-return,type-arg,attr-defined,arg-type,union-attr"
"""
ADXR - Average Directional Movement Index Rating.

ADXR is a smoothed version of ADX that provides a more stable measure of trend strength.
It's calculated as the average of the current ADX and the ADX from n periods ago.

ADXR[i] = (ADX[i] + ADX[i-n]) / 2
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature

from .adx import adx_numba


@njit  # type: ignore[misc]
def adxr_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod: int,
) -> npt.NDArray[np.float64]:
    """Calculate ADXR using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    # First calculate ADX
    adx_values = adx_numba(high, low, close, timeperiod)

    # ADXR is the average of current ADX and ADX from timeperiod ago
    # Find first valid ADX value
    first_valid_adx = -1
    for i in range(n):
        if not np.isnan(adx_values[i]):
            first_valid_adx = i
            break

    if first_valid_adx >= 0:
        # ADXR starts at first_valid_adx + timeperiod - 1
        start_idx = first_valid_adx + timeperiod - 1
        for i in range(start_idx, n):
            if not np.isnan(adx_values[i]) and not np.isnan(
                adx_values[i - timeperiod + 1],
            ):
                result[i] = (adx_values[i] + adx_values[i - timeperiod + 1]) / 2.0

    return result


@feature(
    name="adxr",
    category="momentum",
    description="ADXR - Average Directional Movement Rating",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def adxr(
    high: npt.NDArray[np.float64] | pl.Expr | str,
    low: npt.NDArray[np.float64] | pl.Expr | str,
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Average Directional Movement Index Rating.

    ADXR provides a more stable measure of trend strength by averaging
    the current ADX with the ADX from n periods ago.

    Parameters
    ----------
    high : array-like or Polars expression
        High close
    low : array-like or Polars expression
        Low close
    close : array-like or Polars expression
        Close close
    timeperiod : int, default 14
        Period for ADX calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        ADXR close

    Notes
    -----
    ADXR calculation:
    1. Calculate ADX using the specified timeperiod
    2. ADXR[i] = (ADX[i] + ADX[i-timeperiod]) / 2

    ADXR provides:
    - Smoother trend strength measurement than ADX
    - Reduced sensitivity to short-term price fluctuations
    - Values between 0 and 100, same interpretation as ADX

    Interpretation:
    - ADXR > 25: Strong trend
    - ADXR < 20: Weak trend or sideways market
    - Rising ADXR: Strengthening trend
    - Falling ADXR: Weakening trend

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116, 118])
    >>> low = np.array([108, 109, 107, 110, 112, 111, 113, 115])
    >>> close = np.array([109, 111, 108, 112, 114, 112, 115, 117])
    >>>
    >>> adxr_values = qta.adxr(high, low, close, timeperiod=14)

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "high": high, "low": low, "close": close
    ... })
    >>> result = df.with_columns([
    ...     qta.adxr("high", "low", "close", 14).alias("adxr")
    ... ])
    """
    if isinstance(high, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        if isinstance(high, str):
            high = pl.col(high)
        if isinstance(low, str):
            low = pl.col(low)
        if isinstance(close, str):
            close = pl.col(close)
        # Type narrowing: at this point all must be Expr
        assert isinstance(low, pl.Expr) and isinstance(close, pl.Expr)

        def calc_adxr(s: pl.Series) -> float:
            if isinstance(s[0], dict):
                # When using struct, we get a list of dicts
                h = np.array([x["high"] for x in s])
                lo = np.array([x["low"] for x in s])
                c = np.array([x["close"] for x in s])
            else:
                # Direct array input
                h = s.struct.field("high").to_numpy()
                lo = s.struct.field("low").to_numpy()
                c = s.struct.field("close").to_numpy()
            return adxr_numba(h, lo, c, timeperiod)

        return pl.struct(
            [high.alias("high"), low.alias("low"), close.alias("close")],
        ).map_batches(calc_adxr, return_dtype=pl.Float64)

    # Handle numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(high) < timeperiod * 2:
        return np.full(len(high), np.nan)

    return adxr_numba(high, low, close, timeperiod)


# Make available at package level
__all__ = ["adxr"]
