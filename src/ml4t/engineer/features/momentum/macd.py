# mypy: disable-error-code="no-any-return"
"""
MACD (Moving Average Convergence/Divergence) - TA-Lib compatible implementation.

MACD is a trend-following momentum indicator that shows the relationship
between two moving averages of a security's price.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.trend.ema import ema_numba


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def _int_ema_talib_nb(
    close: npt.NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    period: int,
    k: float,
) -> npt.NDArray[np.float64]:
    """
    Exact replication of TA-Lib's INT_EMA function.

    Based on ta_EMA.c lines 274-376 with exact algorithm matching.
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Calculate lookback exactly like TA-Lib
    lookback_total = period - 1

    # Move up start if not enough data (line 291-292)
    start_idx = max(start_idx, lookback_total)

    # Check if anything to evaluate (line 295-300)
    if start_idx > end_idx:
        return result

    # Calculate seed exactly like TA-Lib (lines 327-333)
    # today = startIdx-lookbackTotal
    today = start_idx - lookback_total
    i = period
    temp_real = 0.0
    while i > 0:
        temp_real += close[today]
        today += 1
        i -= 1
    prev_ma = temp_real / period  # This is the seed

    # Output the seed at start_idx (this is what TA-Lib does)
    result[start_idx] = prev_ma

    # Skip unstable period (lines 359-360)
    # while( today <= startIdx )
    while today <= start_idx:
        prev_ma = ((close[today] - prev_ma) * k) + prev_ma
        today += 1

    # Calculate remaining range (lines 366-371)
    # Output starting from start_idx + 1
    out_idx = start_idx + 1
    while today <= end_idx:
        prev_ma = ((close[today] - prev_ma) * k) + prev_ma
        result[out_idx] = prev_ma
        today += 1
        out_idx += 1

    return result


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def macd_numba(
    close: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> npt.NDArray[np.float64]:
    """
    MACD calculation exactly replicating TA-Lib algorithm.

    Based on ta_MACD.c lines 361-551 with exact buffer management
    and EMA calculation timing.

    Parameters
    ----------
    close : npt.NDArray
        Price data (typically closing close)
    fast_period : int, default 12
        Period for fast EMA (TA-Lib default: 12)
    slow_period : int, default 26
        Period for slow EMA (TA-Lib default: 26)
    signal_period : int, default 9
        Period for signal EMA (TA-Lib default: 9)

    Returns
    -------
    npt.NDArray
        MACD line close exactly matching TA-Lib
    """
    n = len(close)

    # Ensure slow > fast (TA-Lib swaps if needed) - lines 402-408
    if slow_period < fast_period:
        temp_integer = slow_period
        slow_period = fast_period
        fast_period = temp_integer

    # Calculate k close exactly like TA-Lib (PER_TO_K macro) - lines 412, 420
    k1 = 2.0 / (slow_period + 1)  # slow EMA k
    k2 = 2.0 / (fast_period + 1)  # fast EMA k

    # Calculate lookbacks exactly like TA-Lib - lines 427, 432-433
    lookback_signal = signal_period - 1
    lookback_total = lookback_signal + (slow_period - 1)

    # Adjust start index - lines 435-436
    start_idx = lookback_total
    end_idx = n - 1

    if start_idx > end_idx:
        return np.full(n, np.nan)

    # Calculate tempInteger exactly like TA-Lib - line 476
    temp_integer = start_idx - lookback_signal

    # Calculate slow EMA using INT_EMA - lines 477-479
    slow_ema_buffer = _int_ema_talib_nb(close, temp_integer, end_idx, slow_period, k1)

    # Calculate fast EMA using INT_EMA - lines 491-493
    fast_ema_buffer = _int_ema_talib_nb(close, temp_integer, end_idx, fast_period, k2)

    # Calculate difference exactly like TA-Lib - lines 518-519
    for i in range(temp_integer, end_idx + 1):
        if not np.isnan(fast_ema_buffer[i]) and not np.isnan(slow_ema_buffer[i]):
            fast_ema_buffer[i] = fast_ema_buffer[i] - slow_ema_buffer[i]

    # Copy result exactly like TA-Lib - line 523
    # ARRAY_MEMMOVE( outMACD, 0, fastEMABuffer, lookbackSignal, (endIdx-startIdx)+1 );
    # This copies from the buffer starting at temp_integer + lookback_signal
    result = np.full(n, np.nan)
    copy_start_src = temp_integer + lookback_signal
    for i in range(start_idx, end_idx + 1):
        src_idx = copy_start_src + (i - start_idx)
        if src_idx < n and not np.isnan(fast_ema_buffer[src_idx]):
            result[i] = fast_ema_buffer[src_idx]

    return result


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def macd_signal_numba(
    close: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> npt.NDArray[np.float64]:
    """
    MACD Signal line (EMA of MACD line) exactly matching TA-Lib.

    This applies our standard EMA to the MACD line output, but the key
    insight is that we need to match TA-Lib's exact timing.

    Parameters
    ----------
    close : npt.NDArray
        Price data
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA

    Returns
    -------
    npt.NDArray
        MACD signal line close
    """
    # First get the MACD line
    macd_line = macd_numba(close, fast_period, slow_period, signal_period)

    # Apply standard EMA to MACD line
    return ema_numba(macd_line, signal_period)


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def macd_histogram_numba(
    close: npt.NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> npt.NDArray[np.float64]:
    """
    MACD Histogram (MACD line - Signal line).

    Parameters
    ----------
    close : npt.NDArray
        Price data
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA

    Returns
    -------
    npt.NDArray
        MACD histogram close
    """
    macd_line = macd_numba(close, fast_period, slow_period, signal_period)
    signal_line = macd_signal_numba(close, fast_period, slow_period, signal_period)

    n = len(close)
    result = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            result[i] = macd_line[i] - signal_line[i]

    return result


def macd_polars(column: str, fast_period: int = 12, slow_period: int = 26) -> pl.Expr:
    """
    MACD using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    column : str
        Column name to apply MACD to
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA

    Returns
    -------
    pl.Expr
        Polars expression for MACD calculation
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(macd_numba(s.to_numpy(), fast_period, slow_period, 9)),
        return_dtype=pl.Float64,
    )


@feature(
    name="macd",
    category="momentum",
    description="MACD - trend-following momentum indicator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def macd(
    close: npt.NDArray[np.float64] | pl.Series | str,
    fast_period: int = 12,
    slow_period: int = 26,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    MACD line with TA-Lib compatible unstable period.

    MACD shows the relationship between two moving averages of close.
    When MACD is above zero, it indicates upward momentum.

    Parameters
    ----------
    close : array-like or column name
        Price data (typically closing close)
    fast_period : int, default 12
        Period for fast EMA (commonly 12)
    slow_period : int, default 26
        Period for slow EMA (commonly 26)
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        MACD line close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.random.randn(100).cumsum() + 100
    >>> macd_line = macd(close)
    >>> # First ~33 close will be NaN due to unstable period

    Notes
    -----
    - MACD = EMA(close, fast_period) - EMA(close, slow_period)
    - TA-Lib uses 8-period unstable period before outputting close
    - Standard parameters: fast=12, slow=26, signal=9
    """
    # Validate parameters
    if fast_period < 1:
        raise InvalidParameterError(f"fast_period must be >= 1, got {fast_period}")
    if slow_period < 1:
        raise InvalidParameterError(f"slow_period must be >= 1, got {slow_period}")
    if fast_period >= slow_period:
        raise InvalidParameterError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})",
        )

    if isinstance(close, str):
        return macd_polars(close, fast_period, slow_period)

    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return macd_numba(close, fast_period, slow_period, 9)


def macd_signal(
    close: npt.NDArray[np.float64] | pl.Series | str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    MACD Signal Line (EMA of MACD).

    The signal line is used to generate buy/sell signals when it crosses
    the MACD line.

    Parameters
    ----------
    close : array-like or column name
        Price data
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        MACD signal line close
    """
    # Validate parameters
    if fast_period < 1:
        raise InvalidParameterError(f"fast_period must be >= 1, got {fast_period}")
    if slow_period < 1:
        raise InvalidParameterError(f"slow_period must be >= 1, got {slow_period}")
    if fast_period >= slow_period:
        raise InvalidParameterError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})",
        )
    if signal_period < 1:
        raise InvalidParameterError(f"signal_period must be >= 1, got {signal_period}")

    if isinstance(close, str):
        return pl.col(close).map_batches(
            lambda s: pl.Series(
                macd_signal_numba(s.to_numpy(), fast_period, slow_period, signal_period),
            ),
            return_dtype=pl.Float64,
        )

    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return macd_signal_numba(close, fast_period, slow_period, signal_period)


def macd_full(
    close: npt.NDArray[np.float64] | pl.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Complete MACD calculation returning all three components.

    Parameters
    ----------
    close : array-like
        Price data
    fast_period : int, default 12
        Period for fast EMA
    slow_period : int, default 26
        Period for slow EMA
    signal_period : int, default 9
        Period for signal line EMA

    Returns
    -------
    tuple of npt.NDArray
        (macd_line, signal_line, histogram)
    """
    # Validate parameters
    if fast_period < 1:
        raise InvalidParameterError(f"fast_period must be >= 1, got {fast_period}")
    if slow_period < 1:
        raise InvalidParameterError(f"slow_period must be >= 1, got {slow_period}")
    if fast_period >= slow_period:
        raise InvalidParameterError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})",
        )
    if signal_period < 1:
        raise InvalidParameterError(f"signal_period must be >= 1, got {signal_period}")

    if isinstance(close, pl.Series):
        close = close.to_numpy()

    macd_line = macd_numba(close, fast_period, slow_period)
    signal_line = macd_signal_numba(close, fast_period, slow_period, signal_period)
    histogram = macd_histogram_numba(close, fast_period, slow_period, signal_period)

    return macd_line, signal_line, histogram


# Export all functions
__all__ = ["macd", "macd_full", "macd_signal"]
