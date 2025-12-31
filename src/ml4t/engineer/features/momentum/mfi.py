# mypy: disable-error-code="no-any-return"
"""
Money Flow Index (MFI) - TA-Lib compatible implementation.

MFI is a momentum indicator that uses price and volume to identify
overbought or oversold conditions. It's similar to RSI but incorporates volume.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def mfi_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    volume: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    MFI calculation exactly replicating TA-Lib algorithm.

    Based on ta_MFI.c with circular buffer for money flow tracking.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    volume : npt.NDArray
        Volume data
    period : int, default 14
        Number of periods for MFI calculation

    Returns
    -------
    npt.NDArray
        MFI close (0-100 scale) exactly matching TA-Lib
    """
    n = len(high)
    result = np.full(n, np.nan)

    if period < 1 or period >= n:
        return result

    # Need at least period + 1 close to calculate first MFI
    if n < period + 1:
        return result

    # Circular buffers for positive and negative money flow
    pos_mf_buffer = np.zeros(period)
    neg_mf_buffer = np.zeros(period)
    buffer_idx = 0

    # Calculate initial typical price
    today = 0
    prev_tp = (high[today] + low[today] + close[today]) / 3.0

    # Accumulate initial money flows
    pos_sum_mf = 0.0
    neg_sum_mf = 0.0
    today += 1

    for _i in range(period):
        # Calculate typical price
        tp = (high[today] + low[today] + close[today]) / 3.0

        # Calculate money flow
        mf = tp * volume[today]

        # Determine direction based on typical price change
        tp_change = tp - prev_tp

        if tp_change > 0:
            pos_mf_buffer[buffer_idx] = mf
            neg_mf_buffer[buffer_idx] = 0.0
            pos_sum_mf += mf
        elif tp_change < 0:
            neg_mf_buffer[buffer_idx] = mf
            pos_mf_buffer[buffer_idx] = 0.0
            neg_sum_mf += mf
        else:
            pos_mf_buffer[buffer_idx] = 0.0
            neg_mf_buffer[buffer_idx] = 0.0

        prev_tp = tp
        buffer_idx = (buffer_idx + 1) % period
        today += 1

    # Calculate first MFI
    total_mf = pos_sum_mf + neg_sum_mf
    if total_mf < 1.0:
        result[period] = 0.0
    else:
        result[period] = 100.0 * (pos_sum_mf / total_mf)

    # Calculate subsequent MFI close
    for out_idx in range(period + 1, n):
        # Remove old money flow
        pos_sum_mf -= pos_mf_buffer[buffer_idx]
        neg_sum_mf -= neg_mf_buffer[buffer_idx]

        # Calculate new typical price and money flow
        tp = (high[today] + low[today] + close[today]) / 3.0
        mf = tp * volume[today]

        # Determine direction
        tp_change = tp - prev_tp

        if tp_change > 0:
            pos_mf_buffer[buffer_idx] = mf
            neg_mf_buffer[buffer_idx] = 0.0
            pos_sum_mf += mf
        elif tp_change < 0:
            neg_mf_buffer[buffer_idx] = mf
            pos_mf_buffer[buffer_idx] = 0.0
            neg_sum_mf += mf
        else:
            pos_mf_buffer[buffer_idx] = 0.0
            neg_mf_buffer[buffer_idx] = 0.0

        # Calculate MFI
        total_mf = pos_sum_mf + neg_sum_mf
        if total_mf < 1.0:
            result[out_idx] = 0.0
        else:
            result[out_idx] = 100.0 * (pos_sum_mf / total_mf)

        prev_tp = tp
        buffer_idx = (buffer_idx + 1) % period
        today += 1

    return result


def mfi_polars(
    high: str,
    low_col: str,
    close_col: str,
    volume_col: str,
    period: int = 14,
) -> pl.Expr:
    """
    MFI using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    high : str
        Column name for high close
    low_col : str
        Column name for low close
    close_col : str
        Column name for close close
    volume_col : str
        Column name for volume
    period : int, default 14
        Number of periods for MFI calculation

    Returns
    -------
    pl.Expr
        Polars expression for MFI calculation
    """
    return pl.struct([high, low_col, close_col, volume_col]).map_batches(
        lambda s: pl.Series(
            mfi_numba(
                s.struct.field(high).to_numpy(),
                s.struct.field(low_col).to_numpy(),
                s.struct.field(close_col).to_numpy(),
                s.struct.field(volume_col).to_numpy(),
                period,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="mfi",
    category="momentum",
    description="MFI - volume-weighted RSI measuring buying/selling pressure",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def mfi(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    volume: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Money Flow Index exactly matching TA-Lib.

    MFI is a volume-weighted RSI that measures buying and selling pressure.
    Values above 80 indicate overbought conditions, below 20 indicate oversold.

    Parameters
    ----------
    high : array-like or str
        High close or column name
    low : array-like or str
        Low close or column name
    close : array-like or str
        Close close or column name
    volume : array-like or str
        Volume data or column name
    period : int, default 14
        Number of periods for MFI calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        MFI close (0-100 scale)

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([127.01, 127.62, 126.59, 127.35, 128.17])
    >>> low = np.array([125.36, 126.16, 124.93, 126.09, 126.82])
    >>> close = np.array([125.83, 126.48, 126.03, 126.39, 127.18])
    >>> volume = np.array([5000, 5500, 4800, 5200, 6000])
    >>> mfi_values = mfi(high, low, close, volume, period=14)

    Notes
    -----
    - Typical Price = (High + Low + Close) / 3
    - Money Flow = Typical Price × Volume
    - Positive MF = Money Flow when TP increases
    - Negative MF = Money Flow when TP decreases
    - MFI = 100 × (Positive MF / (Positive MF + Negative MF))
    """
    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if (
        isinstance(high, str)
        and isinstance(low, str)
        and isinstance(close, str)
        and isinstance(volume, str)
    ):
        # Column names provided for Polars
        return mfi_polars(high, low, close, volume, period)

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()

    return mfi_numba(high, low, close, volume, period)


# Export all functions
__all__ = ["mfi", "mfi_numba", "mfi_polars"]
