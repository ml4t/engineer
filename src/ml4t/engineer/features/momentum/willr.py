# mypy: disable-error-code="no-any-return"
"""Williams %R implementation."""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def willr_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """Williams %R calculation using Numba for performance."""
    n = len(high)
    result = np.full(n, np.nan)

    if period > n:
        return result

    for i in range(period - 1, n):
        # Get window of highs and lows
        high_window = high[i - period + 1 : i + 1]
        low_window = low[i - period + 1 : i + 1]

        # Calculate highest high and lowest low
        highest_high: float = float(np.max(high_window))
        lowest_low: float = float(np.min(low_window))

        # Calculate Williams %R
        if highest_high != lowest_low:
            result[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            result[i] = -50.0  # Neutral value when no range

    return result


def willr_polars(
    high: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pl.Expr:
    """Williams %R using Polars expressions."""
    # Calculate rolling highest high and lowest low
    highest_high = pl.col(high).rolling_max(window_size=period)
    lowest_low = pl.col(low_col).rolling_min(window_size=period)

    # Handle division by zero case
    return (
        pl.when((highest_high - lowest_low) == 0)
        .then(-50.0)
        .otherwise(
            -100 * (highest_high - pl.col(close_col)) / (highest_high - lowest_low),
        )
    )


@feature(
    name="willr",
    category="momentum",
    description="Williams %R - momentum indicator",
    lookback=0,
    value_range=(-100.0, 0.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def willr(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """Williams %R.

    Williams %R is a momentum indicator that measures overbought and oversold
    levels. It's similar to the Stochastic Oscillator but plotted upside down.

    Formula:
        %R = -100 × (Highest High - Close) / (Highest High - Lowest Low)

    Args:
        high: High close (numpy array, Polars Series, or column name)
        low: Low close (numpy array, Polars Series, or column name)
        close: Close close (numpy array, Polars Series, or column name)
        period: Number of periods for the calculation (default: 14)
        implementation: Implementation to use ('auto', 'numba', 'polars')

    Returns:
        Williams %R close (-100 to 0) as numpy array or Polars expression

    Raises:
        InvalidParameterError: If period < 1
    """
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        if low is None or close is None:
            raise ValueError(
                "low and close must be provided when high is a column name",
            )
        return willr_polars(high, low, close, period)

    # Handle numpy/Series inputs
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return willr_numba(high, low, close, period)
