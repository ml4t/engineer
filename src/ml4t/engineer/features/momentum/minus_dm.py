# mypy: disable-error-code="no-any-return,union-attr"
"""
MINUS_DM - Minus Directional Movement.

Minus Directional Movement measures downward price movement. It's a component
used in the calculation of the Minus Directional Indicator (-DI).

MINUS_DM = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1], else 0
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def minus_dm_numba(
    high: npt.NDArray[np.float64], low: npt.NDArray[np.float64], timeperiod: int
) -> npt.NDArray[np.float64]:
    """Calculate Minus Directional Movement using Numba for performance."""
    n = len(high)
    dm_minus = np.full(n, np.nan)

    # Calculate raw directional movements starting at index 0
    raw_dm_minus = np.zeros(n)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if down_move > up_move and down_move > 0:
            raw_dm_minus[i] = down_move

    # Apply Wilder's smoothing exactly as in PLUS_DM
    if n <= timeperiod:
        return dm_minus

    # TA-Lib compatibility: Sum from index 1 to timeperiod-1 (exclusive end)
    sum_val = 0.0
    for i in range(1, timeperiod):
        sum_val += raw_dm_minus[i]

    # Store running sum for Wilder's calculation
    prev_sum = sum_val

    # First output at index timeperiod-1 (TA-Lib style)
    dm_minus[timeperiod - 1] = sum_val

    # Wilder's smoothing: new_sum = prev_sum - prev_sum/period + new_value
    for i in range(timeperiod, n):
        prev_sum = prev_sum - prev_sum / timeperiod + raw_dm_minus[i]
        dm_minus[i] = prev_sum

    return dm_minus


@feature(
    name="minus_dm",
    category="momentum",
    description="Minus Directional Movement",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def minus_dm(
    high: npt.NDArray[np.float64] | pl.Expr | str,
    low: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Minus Directional Movement.

    Minus Directional Movement is a component of the Directional Movement System.
    It measures the portion of price movement in the downward direction.

    Parameters
    ----------
    high : array-like or Polars expression
        High close
    low : array-like or Polars expression
        Low close
    timeperiod : int, default 14
        Period for Wilder's smoothing

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Minus Directional Movement close

    Notes
    -----
    Minus DM calculation:
    1. Calculate upward movement: up_move = high[i] - high[i-1]
    2. Calculate downward movement: down_move = low[i-1] - low[i]
    3. If down_move > up_move AND down_move > 0: -DM = down_move, else -DM = 0
    4. Apply Wilder's smoothing over timeperiod

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116])
    >>> low = np.array([108, 109, 107, 110, 112, 111, 113])
    >>>
    >>> minus_dm_values = qta.minus_dm(high, low, timeperiod=14)

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"high": high, "low": low})
    >>> result = df.with_columns([
    ...     qta.minus_dm("high", "low", 14).alias("minus_dm")
    ... ])
    """
    if isinstance(high, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        if isinstance(high, str):
            high = pl.col(high)
        if isinstance(low, str):
            low = pl.col(low)
        # Type narrowing: at this point low must be Expr
        assert isinstance(low, pl.Expr)

        def calc_minus_dm(s: pl.Series) -> pl.Series:
            if isinstance(s[0], dict):
                # When using struct, we get a list of dicts
                h = np.array([x["high"] for x in s])
                lo = np.array([x["low"] for x in s])
            else:
                # Direct array input
                h = s.struct.field("high").to_numpy()
                lo = s.struct.field("low").to_numpy()
            return pl.Series(minus_dm_numba(h, lo, timeperiod))

        return pl.struct([high.alias("high"), low.alias("low")]).map_batches(
            calc_minus_dm,
            return_dtype=pl.Float64,
        )

    # Handle numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(high) < timeperiod + 1:
        return np.full(len(high), np.nan)

    return minus_dm_numba(high, low, timeperiod)


# Make available at package level
__all__ = ["minus_dm"]
