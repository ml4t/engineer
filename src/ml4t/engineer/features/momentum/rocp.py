# mypy: disable-error-code="no-any-return"
"""
ROCP - Rate of Change Percentage.

ROCP calculates the percentage change in price over a specified period.
It's similar to ROC but expressed as a percentage.

ROCP = (price[i] - price[i-n]) / price[i-n] * 100
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def rocp_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """Calculate Rate of Change Percentage using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(timeperiod, n):
        prev_value = close[i - timeperiod]
        curr_value = close[i]

        if abs(prev_value) > 1e-10:  # Avoid division by zero
            result[i] = (curr_value - prev_value) / prev_value
        else:
            # TA-Lib returns 0.0 when dividing by zero in ROCP
            result[i] = 0.0

    return result


@feature(
    name="rocp",
    category="momentum",
    description="ROCP - Rate of Change Percentage",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def rocp(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 10,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Rate of Change Percentage.

    ROCP measures the percentage change in price over a specified time period.
    It's useful for comparing momentum across different price levels.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    timeperiod : int, default 10
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Rate of Change Percentage close

    Notes
    -----
    ROCP calculation:
    ROCP[i] = (price[i] - price[i-n]) / price[i-n]

    Interpretation:
    - Positive close: Upward momentum
    - Negative close: Downward momentum
    - Values expressed as decimals (0.04 = 4% change)
    - Larger absolute close indicate stronger momentum

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> rocp_values = qta.rocp(close, timeperiod=3)
    >>> print(rocp_values[3:])  # First 3 close are NaN
    [3.0 3.92 2.88 2.91 2.86]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": close})
    >>> result = df.with_columns([
    ...     qta.rocp("price", 3).alias("rocp")
    ... ])
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        # Pure Polars expression optimization - much faster than map_batches
        past_value = close.shift(timeperiod)
        return (
            pl.when(past_value.abs() <= 1e-10)
            .then(0.0)  # TA-Lib returns 0.0 when dividing by zero
            .otherwise((close - past_value) / past_value)
        )

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) <= timeperiod:
        return np.full(len(close), np.nan)

    return rocp_numba(close, timeperiod)


# Make available at package level
__all__ = ["rocp"]
