# mypy: disable-error-code="no-any-return"
"""Rate of Change (ROC) implementation."""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def roc_numba(close: npt.NDArray[np.float64], period: int = 10) -> npt.NDArray[np.float64]:
    """ROC calculation using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    if period >= n:
        return result

    for i in range(period, n):
        past_value = close[i - period]
        if past_value != 0:
            result[i] = 100 * (close[i] - past_value) / past_value
        else:
            result[i] = 0.0

    return result


def roc_polars(column: str, period: int = 10) -> pl.Expr:
    """ROC using Polars expressions."""
    # Get the value n periods ago
    past_value = pl.col(column).shift(period)

    # Calculate ROC
    return (
        pl.when(past_value == 0)
        .then(0.0)
        .otherwise(100 * (pl.col(column) - past_value) / past_value)
    )


@feature(
    name="roc",
    category="momentum",
    description="ROC - Rate of Change",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def roc(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 10,
) -> npt.NDArray[np.float64] | pl.Expr:
    """Rate of Change.

    The ROC indicator measures the percentage change in price from the current
    period to n periods ago. It's a momentum oscillator that oscillates above
    and below zero.

    Formula:
        ROC = 100 × (Close - Close[n]) / Close[n]

    Args:
        close: Price data (numpy array, Polars Series, or column name)
        period: Number of periods for the calculation (default: 10)
        implementation: Implementation to use ('auto', 'numba', 'polars')

    Returns:
        ROC close as numpy array or Polars expression

    Raises:
        InvalidParameterError: If period < 1
    """
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return roc_polars(close, period)

    # Handle numpy/Series inputs
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return roc_numba(close, period)
