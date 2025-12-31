# mypy: disable-error-code="no-any-return,misc"
"""
ADX (Average Directional Movement Index) - TA-Lib compatible implementation.

ADX measures the strength of a trend, regardless of direction.
Range: 0-100, where higher close indicate stronger trends.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.logging import get_logger, logged_feature

# Module logger
_logger = get_logger("ta.adx")


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_directional_movement(
    prev_high: float,
    current_high: float,
    prev_low: float,
    current_low: float,
) -> tuple[float, float]:
    """Calculate plus and minus directional movements.

    Parameters
    ----------
    prev_high : float
        Previous period high price
    current_high : float
        Current period high price
    prev_low : float
        Previous period low price
    current_low : float
        Current period low price

    Returns
    -------
    tuple[float, float]
        (plus_dm, minus_dm) directional movements
    """
    diff_p = current_high - prev_high  # Plus Delta
    diff_m = prev_low - current_low  # Minus Delta

    # Determine which DM to use based on TA-Lib logic
    if diff_m > 0 and diff_p < diff_m:
        # Case 2 and 4: +DM=0, -DM=diffM
        return 0.0, diff_m
    if diff_p > 0 and diff_p > diff_m:
        # Case 1 and 3: +DM=diffP, -DM=0
        return diff_p, 0.0
    # Case 5: Both zero
    return 0.0, 0.0


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """Calculate True Range for a single period.

    Parameters
    ----------
    high : float
        Current period high price
    low : float
        Current period low price
    prev_close : float
        Previous period close price

    Returns
    -------
    float
        True Range value
    """
    tr_hl = high - low
    tr_hc = abs(high - prev_close)
    tr_lc = abs(low - prev_close)
    return max(tr_hl, tr_hc, tr_lc)


@jit(nopython=True, cache=True, fastmath=True)
def _apply_wilders_smoothing(
    prev_smoothed: float,
    new_value: float,
    period: int,
) -> float:
    """Apply Wilder's smoothing to a value.

    Wilder's smoothing formula: new_smoothed = prev_smoothed - (prev_smoothed / period) + new_value

    Parameters
    ----------
    prev_smoothed : float
        Previous smoothed value
    new_value : float
        New raw value to incorporate
    period : int
        Smoothing period

    Returns
    -------
    float
        New smoothed value
    """
    return prev_smoothed - (prev_smoothed / period) + new_value


@jit(nopython=True, cache=True, fastmath=True)
def _calculate_dx(plus_di: float, minus_di: float) -> float:
    """Calculate Directional Index (DX) from Directional Indicators.

    Parameters
    ----------
    plus_di : float
        Plus Directional Indicator (+DI)
    minus_di : float
        Minus Directional Indicator (-DI)

    Returns
    -------
    float
        Directional Index (DX) value
    """
    sum_di = minus_di + plus_di
    if sum_di == 0.0:
        return 0.0
    return 100.0 * (abs(minus_di - plus_di) / sum_di)


@jit(nopython=True, cache=True, fastmath=True)
def adx_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    ADX calculation exactly replicating TA-Lib algorithm - refactored version.

    This refactored version breaks down the monolithic ADX calculation into
    smaller, testable components while maintaining exact numerical compatibility.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    period : int, default 14
        Time period for ADX calculation

    Returns
    -------
    npt.NDArray
        ADX close exactly matching TA-Lib
    """
    n = len(high)

    # ADX lookback is 2*period - 1 (without unstable period)
    lookback = 2 * period - 1

    if n <= lookback:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    # Initialize accumulators
    prev_minus_dm = 0.0
    prev_plus_dm = 0.0
    prev_tr = 0.0

    # Start from the beginning of data
    today = 0
    prev_high = high[today]
    prev_low = low[today]
    prev_close = close[today]

    # Accumulate initial DM and TR close (period-1 close)
    for _i in range(period - 1):
        today += 1

        # Calculate directional movements using helper function
        current_high = high[today]
        current_low = low[today]
        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )
        prev_plus_dm += plus_dm
        prev_minus_dm += minus_dm

        # Calculate True Range using helper function
        tr = _calculate_true_range(current_high, current_low, prev_close)
        prev_tr += tr

        prev_high = current_high
        prev_low = current_low
        prev_close = close[today]

    # Calculate initial DX close
    sum_dx = 0.0

    for _i in range(period):
        today += 1

        # Calculate directional movements
        current_high = high[today]
        current_low = low[today]
        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        # Apply Wilder's smoothing to DM using helper function
        prev_plus_dm = _apply_wilders_smoothing(prev_plus_dm, plus_dm, period)
        prev_minus_dm = _apply_wilders_smoothing(prev_minus_dm, minus_dm, period)

        # Calculate and smooth True Range
        tr = _calculate_true_range(current_high, current_low, prev_close)
        prev_tr = _apply_wilders_smoothing(prev_tr, tr, period)

        prev_high = current_high
        prev_low = current_low
        prev_close = close[today]

        # Calculate DX using helper function
        if prev_tr != 0:
            minus_di = 100.0 * (prev_minus_dm / prev_tr)
            plus_di = 100.0 * (prev_plus_dm / prev_tr)
            dx = _calculate_dx(plus_di, minus_di)
            sum_dx += dx

    # First ADX is average of first 'period' DX close
    prev_adx = sum_dx / period

    # First output at index lookback
    result[lookback] = prev_adx

    # Calculate subsequent ADX close
    for out_idx in range(lookback + 1, n):
        today += 1

        # Calculate directional movements
        current_high = high[today]
        current_low = low[today]
        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        # Apply Wilder's smoothing to DM
        prev_plus_dm = _apply_wilders_smoothing(prev_plus_dm, plus_dm, period)
        prev_minus_dm = _apply_wilders_smoothing(prev_minus_dm, minus_dm, period)

        # Calculate and smooth True Range
        tr = _calculate_true_range(current_high, current_low, prev_close)
        prev_tr = _apply_wilders_smoothing(prev_tr, tr, period)

        prev_high = current_high
        prev_low = current_low
        prev_close = close[today]

        # Calculate DX and smooth to ADX
        if prev_tr != 0:
            minus_di = 100.0 * (prev_minus_dm / prev_tr)
            plus_di = 100.0 * (prev_plus_dm / prev_tr)

            dx = _calculate_dx(plus_di, minus_di)
            # Wilder's smoothing for ADX
            prev_adx = ((prev_adx * (period - 1)) + dx) / period

        result[out_idx] = prev_adx

    return result


def adx_polars(
    high: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pl.Expr:
    """
    ADX using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    high : str
        Column name for high close
    low_col : str
        Column name for low close
    close_col : str
        Column name for close close
    period : int, default 14
        Time period for ADX calculation

    Returns
    -------
    pl.Expr
        Polars expression for ADX calculation
    """
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda s: pl.Series(
            adx_numba(
                s.struct.field(high).to_numpy(),
                s.struct.field(low_col).to_numpy(),
                s.struct.field(close_col).to_numpy(),
                period,
            ),
        ),
        return_dtype=pl.Float64,
    )


@logged_feature("ADX", warn_threshold_ms=500.0, log_data_quality=True)
@feature(
    name="adx",
    category="momentum",
    description="ADX - measures trend strength regardless of direction",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def adx(
    high: npt.NDArray[np.float64] | pl.Series,
    low: npt.NDArray[np.float64] | pl.Series,
    close: npt.NDArray[np.float64] | pl.Series,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    ADX (Average Directional Movement Index) exactly matching TA-Lib.

    ADX measures trend strength on a scale of 0-100. Values:
    - < 25: Weak or no trend
    - 25-50: Moderate trend
    - 50-75: Strong trend
    - > 75: Very strong trend

    Parameters
    ----------
    high : array-like
        High close
    low : array-like
        Low close
    close : array-like
        Close close
    period : int, default 14
        Time period for ADX calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        ADX close exactly matching TA-Lib

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([48.70, 48.72, 48.90, 48.87, 48.82])
    >>> low = np.array([47.79, 48.14, 48.39, 48.37, 48.24])
    >>> close = np.array([48.16, 48.61, 48.75, 48.63, 48.74])
    >>> adx_values = adx(high, low, close, period=14)

    Notes
    -----
    - First ADX value appears at index 2*period-1 (e.g., 27 for period=14)
    - Uses Wilder's smoothing throughout (+DM, -DM, TR, and ADX)
    - Based on Directional Movement System from Wilder's book
    - Exact replication of TA-Lib ta_ADX.c algorithm
    """
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        # Column names provided for Polars
        return adx_polars(high, low, close, period)

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return adx_numba(high, low, close, period)


# Export all functions
__all__ = ["adx", "adx_numba", "adx_polars"]
