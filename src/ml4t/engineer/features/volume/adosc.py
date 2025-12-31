# mypy: disable-error-code="no-any-return"
"""
Chaikin A/D Oscillator (ADOSC) - TA-Lib compatible implementation.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def adosc_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> npt.NDArray[np.float64]:
    """
    Chaikin A/D Oscillator calculation exactly matching TA-Lib.

    Key differences from standard approach:
    1. EMAs are initialized with first AD value (not SMA)
    2. AD is calculated cumulatively within the function
    3. Lookback period is based on slowest EMA period
    """
    n = len(high)
    result = np.full(n, np.nan)

    # Determine slowest period for lookback
    slowest_period = max(fastperiod, slowperiod)

    # TA-Lib lookback for EMA is period - 1
    lookback = slowest_period - 1

    if n <= lookback:
        return result

    # Calculate k close (alpha) for EMAs
    fastk = 2.0 / (fastperiod + 1.0)
    one_minus_fastk = 1.0 - fastk

    slowk = 2.0 / (slowperiod + 1.0)
    one_minus_slowk = 1.0 - slowk

    # Initialize AD accumulator
    ad = 0.0

    # Calculate first AD value
    high_val = high[0]
    low_val = low[0]
    close_val = close[0]
    hl_diff = high_val - low_val

    if hl_diff > 0.0:
        ad += (((close_val - low_val) - (high_val - close_val)) / hl_diff) * volume[0]

    # Initialize both EMAs with the first AD value
    fast_ema = ad
    slow_ema = ad

    # Process through lookback period
    for i in range(1, lookback):
        # Calculate AD
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        hl_diff = high_val - low_val

        if hl_diff > 0.0:
            ad += (((close_val - low_val) - (high_val - close_val)) / hl_diff) * volume[i]

        # Update EMAs
        fast_ema = (fastk * ad) + (one_minus_fastk * fast_ema)
        slow_ema = (slowk * ad) + (one_minus_slowk * slow_ema)

    # Calculate output starting from lookback position
    for i in range(lookback, n):
        # Calculate AD
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        hl_diff = high_val - low_val

        if hl_diff > 0.0:
            ad += (((close_val - low_val) - (high_val - close_val)) / hl_diff) * volume[i]

        # Update EMAs
        fast_ema = (fastk * ad) + (one_minus_fastk * fast_ema)
        slow_ema = (slowk * ad) + (one_minus_slowk * slow_ema)

        # Calculate ADOSC
        result[i] = fast_ema - slow_ema

    return result


def adosc_polars(
    high: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> pl.Expr:
    """Chaikin A/D Oscillator using Polars expressions."""
    return pl.struct([high, low_col, close_col, volume_col]).map_batches(
        lambda x: pl.Series(
            adosc_numba(
                x.struct.field(high).to_numpy(),
                x.struct.field(low_col).to_numpy(),
                x.struct.field(close_col).to_numpy(),
                x.struct.field(volume_col).to_numpy(),
                fastperiod,
                slowperiod,
            ),
        ),
    )


@feature(
    name="adosc",
    category="volume",
    description="ADOSC - Chaikin A/D Oscillator",
    lookback="slowperiod",
    normalized=False,  # Oscillator of unbounded A/D line is not stationary
    formula="",
    ta_lib_compatible=True,
    parameters={"fastperiod": 3, "slowperiod": 10},
)
def adosc(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    volume: npt.NDArray[np.float64] | pl.Series | str,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Chaikin A/D Oscillator.

    This version exactly matches TA-Lib's implementation.
    """
    # Handle string inputs (Polars column names)
    if (
        isinstance(high, str)
        and isinstance(low, str)
        and isinstance(close, str)
        and isinstance(volume, str)
    ):
        return adosc_polars(high, low, close, volume, fastperiod, slowperiod)

    # Validate periods
    if fastperiod <= 0:
        raise ValueError("fastperiod must be > 0")
    if slowperiod <= 0:
        raise ValueError("slowperiod must be > 0")

    # Convert to numpy arrays
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("high, low, close, and volume must have the same length")

    return adosc_numba(high, low, close, volume, fastperiod, slowperiod)


# Export the main function
__all__ = ["adosc"]
