# mypy: disable-error-code="no-any-return"
"""
MACDFIX - Moving Average Convergence/Divergence Fix 12/26.

MACDFIX is a simplified version of MACD with fixed periods of 12 and 26.
It's equivalent to MACD(12, 26, signal_period) but only the signal period
can be customized.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.features.trend.ema import ema_numba


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def _int_ema_fixed_k(
    close: npt.NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    period: int,
    k: float,
) -> npt.NDArray[np.float64]:
    """
    Calculate EMA with fixed k constant (for MACDFIX).

    This is the same as _int_ema_talib_nb but takes k directly
    instead of calculating it from period.
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Calculate lookback
    lookback_total = period - 1

    # Move up start if not enough data
    start_idx = max(start_idx, lookback_total)

    # Check if anything to evaluate
    if start_idx > end_idx:
        return result

    # Calculate seed (SMA)
    today = start_idx - lookback_total
    i = period
    temp_real = 0.0
    while i > 0:
        temp_real += close[today]
        today += 1
        i -= 1
    prev_ma = temp_real / period

    # Output the seed at start_idx
    result[start_idx] = prev_ma

    # Skip unstable period
    while today <= start_idx:
        prev_ma = ((close[today] - prev_ma) * k) + prev_ma
        today += 1

    # Calculate remaining range
    out_idx = start_idx + 1
    while today <= end_idx:
        prev_ma = ((close[today] - prev_ma) * k) + prev_ma
        result[out_idx] = prev_ma
        today += 1
        out_idx += 1

    return result


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def macdfix_numba(
    close: npt.NDArray[np.float64],
    signalperiod: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate MACDFIX using exact TA-Lib algorithm.

    IMPORTANT: MACDFIX uses FIXED k constants (0.15 and 0.075),
    NOT the standard k = 2/(period+1) formula!

    From TA-Lib source (ta_MACD.c):
    - Fast period 12: k = 0.15 (NOT 2/(12+1) = 0.1538...)
    - Slow period 26: k = 0.075 (NOT 2/(26+1) = 0.0740...)

    This is why it's called "FIX" - fixed decimal constants.
    """
    n = len(close)

    # MACDFIX fixed constants from TA-Lib source
    fast_period = 12
    slow_period = 26
    fast_k = 0.15  # Fixed constant, NOT 2/(12+1)
    slow_k = 0.075  # Fixed constant, NOT 2/(26+1)

    # Calculate lookback
    lookback_signal = signalperiod - 1
    lookback_total = lookback_signal + (slow_period - 1)

    # Adjust start index
    start_idx = lookback_total
    end_idx = n - 1

    if start_idx > end_idx:
        nan_array = np.full(n, np.nan)
        return nan_array, nan_array, nan_array

    # Calculate temp_integer
    temp_integer = start_idx - lookback_signal

    # Calculate EMAs with FIXED k close
    slow_ema_buffer = _int_ema_fixed_k(close, temp_integer, end_idx, slow_period, slow_k)
    fast_ema_buffer = _int_ema_fixed_k(close, temp_integer, end_idx, fast_period, fast_k)

    # Calculate difference (MACD line)
    for i in range(temp_integer, end_idx + 1):
        if not np.isnan(fast_ema_buffer[i]) and not np.isnan(slow_ema_buffer[i]):
            fast_ema_buffer[i] = fast_ema_buffer[i] - slow_ema_buffer[i]

    # Copy result to output
    macd_line = np.full(n, np.nan)
    copy_start_src = temp_integer + lookback_signal
    for i in range(start_idx, end_idx + 1):
        src_idx = copy_start_src + (i - start_idx)
        if src_idx < n and not np.isnan(fast_ema_buffer[src_idx]):
            macd_line[i] = fast_ema_buffer[src_idx]

    # Calculate signal line (EMA of MACD)
    signal_line = ema_numba(macd_line, signalperiod)

    # Calculate histogram
    histogram = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


@feature(
    name="macdfix",
    category="momentum",
    description="MACD Fix - MACD with fixed 12/26 periods",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def macdfix(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    signalperiod: int = 9,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate MACD Fix 12/26.

    MACDFIX uses fixed fast (12) and slow (26) periods, with only the
    signal period customizable. Returns the MACD line value.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    signalperiod : int, default 9
        Period for signal line EMA

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        MACD line close

    See Also
    --------
    macdfix_signal : Get signal line
    macdfix_full : Get all three components (MACD, signal, histogram)

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> macd_line = qta.macdfix(close, signalperiod=9)
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        return close.map_batches(
            lambda x: macdfix_numba(x.to_numpy(), signalperiod)[0],
            return_dtype=pl.Float64,
        )

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if signalperiod <= 0:
        raise ValueError("signalperiod must be > 0")

    if len(close) < 26:  # Minimum for slow EMA
        return np.full(len(close), np.nan)

    macd_line, _, _ = macdfix_numba(close, signalperiod)
    return macd_line


def macdfix_signal(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    signalperiod: int = 9,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Get MACDFIX signal line.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    signalperiod : int, default 9
        Period for signal line EMA

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Signal line close
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        return close.map_batches(
            lambda x: macdfix_numba(x.to_numpy(), signalperiod)[1],
            return_dtype=pl.Float64,
        )

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if signalperiod <= 0:
        raise ValueError("signalperiod must be > 0")

    if len(close) < 26:  # Minimum for slow EMA
        return np.full(len(close), np.nan)

    _, signal_line, _ = macdfix_numba(close, signalperiod)
    return signal_line


def macdfix_full(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    signalperiod: int = 9,
) -> tuple[
    npt.NDArray[np.float64] | pl.Expr,
    npt.NDArray[np.float64] | pl.Expr,
    npt.NDArray[np.float64] | pl.Expr,
]:
    """
    Get all MACDFIX components.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    signalperiod : int, default 9
        Period for signal line EMA

    Returns
    -------
    tuple of (macd, signal, histogram)
        All three MACDFIX components

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> close = np.array([100, 102, 104, 103, 105, 107, 106, 108])
    >>> macd, signal, hist = qta.macdfix_full(close, signalperiod=9)
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expressions
        if isinstance(close, str):
            close = pl.col(close)

        macd = close.map_batches(
            lambda x: macdfix_numba(x.to_numpy(), signalperiod)[0],
            return_dtype=pl.Float64,
        )
        signal = close.map_batches(
            lambda x: macdfix_numba(x.to_numpy(), signalperiod)[1],
            return_dtype=pl.Float64,
        )
        hist = close.map_batches(
            lambda x: macdfix_numba(x.to_numpy(), signalperiod)[2],
            return_dtype=pl.Float64,
        )
        return macd, signal, hist

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if signalperiod <= 0:
        raise ValueError("signalperiod must be > 0")

    if len(close) < 26:  # Minimum for slow EMA
        nan_array = np.full(len(close), np.nan)
        return nan_array, nan_array, nan_array

    return tuple(macdfix_numba(close, signalperiod))


# Make available at package level
__all__ = ["macdfix", "macdfix_full", "macdfix_signal"]
