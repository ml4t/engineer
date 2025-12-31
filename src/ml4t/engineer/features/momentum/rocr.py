# mypy: disable-error-code="no-any-return"
"""
ROCR - Rate of Change Ratio.

ROCR calculates the ratio of current price to price n periods ago.
A value of 1.0 indicates no change, close > 1.0 indicate upward movement.

ROCR = price[i] / price[i-n]
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def rocr_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """Calculate Rate of Change Ratio using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(timeperiod, n):
        prev_value = close[i - timeperiod]
        curr_value = close[i]

        if abs(prev_value) > 1e-10:  # Avoid division by zero
            result[i] = curr_value / prev_value

    return result


@feature(
    name="rocr",
    category="momentum",
    description="ROCR - Rate of Change Ratio",
    lookback=0,
    value_range=(0.0, float("inf")),
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def rocr(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 10,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Rate of Change Ratio.

    ROCR measures the ratio of current price to price n periods ago.
    It provides a normalized view of price momentum.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    timeperiod : int, default 10
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Rate of Change Ratio close

    Notes
    -----
    ROCR calculation:
    ROCR[i] = price[i] / price[i-n]

    Interpretation:
    - ROCR = 1.0: No change
    - ROCR > 1.0: Upward movement (e.g., 1.05 = 5% increase)
    - ROCR < 1.0: Downward movement (e.g., 0.95 = 5% decrease)
    - Always positive close

    Relationship to other indicators:
    - ROCP = (ROCR - 1) * 100
    - ROCR100 = ROCR * 100

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> rocr_values = qta.rocr(close, timeperiod=3)
    >>> print(rocr_values[3:])  # First 3 close are NaN
    [1.03 1.0392 1.0288 1.0291 1.0286]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": close})
    >>> result = df.with_columns([
    ...     qta.rocr("price", 3).alias("rocr")
    ... ])
    """
    if isinstance(close, pl.Expr | str):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        # Pure Polars expression optimization - much faster than map_batches
        past_value = close.shift(timeperiod)
        return pl.when(past_value.abs() <= 1e-10).then(None).otherwise(close / past_value)

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) <= timeperiod:
        return np.full(len(close), np.nan)

    return rocr_numba(close, timeperiod)


# Make available at package level
__all__ = ["rocr"]
