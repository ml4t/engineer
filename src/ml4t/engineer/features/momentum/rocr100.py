# mypy: disable-error-code="no-any-return"
"""
ROCR100 - Rate of Change Ratio 100 scale.

ROCR100 is the same as ROCR but scaled by 100 for easier interpretation.
A value of 100 indicates no change, close > 100 indicate upward movement.

ROCR100 = (price[i] / price[i-n]) * 100
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def rocr100_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """Calculate Rate of Change Ratio 100 scale using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(timeperiod, n):
        prev_value = close[i - timeperiod]
        curr_value = close[i]

        if abs(prev_value) > 1e-10:  # Avoid division by zero
            result[i] = (curr_value / prev_value) * 100.0

    return result


@feature(
    name="rocr100",
    category="momentum",
    description="ROCR100 - Rate of Change Ratio * 100",
    lookback=0,
    value_range=(0.0, float("inf")),
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def rocr100(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 10,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Rate of Change Ratio 100 scale.

    ROCR100 measures the ratio of current price to price n periods ago,
    scaled by 100 for easier interpretation.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    timeperiod : int, default 10
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Rate of Change Ratio 100 scale close

    Notes
    -----
    ROCR100 calculation:
    ROCR100[i] = (price[i] / price[i-n]) * 100

    Interpretation:
    - ROCR100 = 100: No change
    - ROCR100 > 100: Upward movement (e.g., 105 = 5% increase)
    - ROCR100 < 100: Downward movement (e.g., 95 = 5% decrease)
    - Always positive close

    Relationship to other indicators:
    - ROCR100 = ROCR * 100
    - ROCP = ROCR100 - 100

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> rocr100_values = qta.rocr100(close, timeperiod=3)
    >>> print(rocr100_values[3:])  # First 3 close are NaN
    [103.0 103.92 102.88 102.91 102.86]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": close})
    >>> result = df.with_columns([
    ...     qta.rocr100("price", 3).alias("rocr100")
    ... ])
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        # Pure Polars expression optimization - much faster than map_batches
        past_value = close.shift(timeperiod)
        return pl.when(past_value.abs() <= 1e-10).then(None).otherwise((close / past_value) * 100.0)

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) <= timeperiod:
        return np.full(len(close), np.nan)

    return rocr100_numba(close, timeperiod)


# Make available at package level
__all__ = ["rocr100"]
