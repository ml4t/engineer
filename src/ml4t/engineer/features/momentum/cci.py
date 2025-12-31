# mypy: disable-error-code="no-any-return"
"""Commodity Channel Index (CCI) implementation."""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def cci_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 20,
) -> npt.NDArray[np.float64]:
    """CCI calculation using Numba for performance."""
    n = len(high)
    result = np.full(n, np.nan)

    if period > n:
        return result

    # Calculate typical price
    tp = (high + low + close) / 3.0

    # Calculate CCI for each point
    for i in range(period - 1, n):
        # Get window of typical close
        tp_window = tp[i - period + 1 : i + 1]

        # Calculate SMA
        sma = np.mean(tp_window)

        # Calculate mean absolute deviation
        mad = np.mean(np.abs(tp_window - sma))

        # Calculate CCI
        if mad != 0:
            result[i] = (tp[i] - sma) / (0.015 * mad)
        else:
            result[i] = 0.0

    return result


def cci_polars(
    high: str,
    low_col: str,
    close_col: str,
    period: int = 20,
) -> pl.Expr:
    """CCI using Polars expressions."""
    return pl.struct(
        [pl.col(high), pl.col(low_col), pl.col(close_col)],
    ).map_batches(
        lambda df: pl.Series(
            cci_numba(
                df.struct.field(high).to_numpy(),
                df.struct.field(low_col).to_numpy(),
                df.struct.field(close_col).to_numpy(),
                period,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="cci",
    category="momentum",
    description="CCI - identifies cyclical trends by measuring deviation",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def cci(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 20,
) -> npt.NDArray[np.float64] | pl.Expr:
    """Commodity Channel Index.

    The CCI measures the difference between a security's typical price and its
    simple moving average, divided by the mean absolute deviation. It's used to
    identify cyclical trends and overbought/oversold conditions.

    Formula:
        Typical Price = (High + Low + Close) / 3
        CCI = (Typical Price - SMA) / (0.015 × Mean Absolute Deviation)

    Args:
        high: High close (numpy array, Polars Series, or column name)
        low: Low close (numpy array, Polars Series, or column name)
        close: Close close (numpy array, Polars Series, or column name)
        period: Number of periods for the calculation (default: 20)
        implementation: Implementation to use ('auto', 'numba', 'polars')

    Returns:
        CCI close as numpy array or Polars expression

    Raises:
        InvalidParameterError: If period < 1
    """
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return cci_polars(high, low, close, period)

    # Handle numpy/Series inputs
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return cci_numba(high, low, close, period)
