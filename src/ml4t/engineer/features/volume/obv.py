# mypy: disable-error-code="no-any-return"
"""
On Balance Volume (OBV) - TA-Lib compatible implementation.

OBV is a momentum indicator that uses volume flow to predict changes in stock price.
The theory is that volume precedes price movement.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def obv_numba(
    close: npt.NDArray[np.float64], volume: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    OBV calculation exactly replicating TA-Lib algorithm.

    OBV accumulates volume based on price direction:
    - If close > previous close: add volume
    - If close < previous close: subtract volume
    - If close = previous close: no change

    Parameters
    ----------
    close : npt.NDArray
        Closing close
    volume : npt.NDArray
        Volume data

    Returns
    -------
    npt.NDArray
        On Balance Volume close
    """
    n = len(close)
    if n == 0:
        return np.zeros(0)

    if len(volume) != n:
        return np.full(n, np.nan)

    result = np.zeros(n)

    # Initialize with first volume
    prev_obv = volume[0]
    prev_close = close[0]
    result[0] = prev_obv

    # Calculate OBV
    for i in range(1, n):
        if close[i] > prev_close:
            prev_obv += volume[i]
        elif close[i] < prev_close:
            prev_obv -= volume[i]
        # If close[i] == prev_close, OBV remains unchanged

        result[i] = prev_obv
        prev_close = close[i]

    return result


def obv_polars(close_column: str, volume_column: str) -> pl.Expr:
    """
    OBV using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    close_column : str
        Column name for closing close
    volume_column : str
        Column name for volume data

    Returns
    -------
    pl.Expr
        Polars expression for OBV
    """
    return pl.struct([close_column, volume_column]).map_batches(
        lambda s: pl.Series(
            obv_numba(
                s.struct.field(close_column).to_numpy(),
                s.struct.field(volume_column).to_numpy(),
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="obv",
    category="volume",
    description="OBV - On-Balance Volume",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def obv(
    close: npt.NDArray[np.float64] | pl.Series | str,
    volume: npt.NDArray[np.float64] | pl.Series | str,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    On Balance Volume exactly matching TA-Lib.

    OBV is a technical trading momentum indicator that uses volume flow to predict
    changes in stock price. The theory behind OBV is that volume precedes price
    movement, so if a security is seeing an increasing OBV, it is a signal that
    volume is growing on upward price moves.

    Parameters
    ----------
    close : array-like or str
        Closing close or column name
    volume : array-like or str
        Volume data or column name
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        On Balance Volume close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 11, 13, 12, 14])
    >>> volume = np.array([1000, 1100, 1200, 900, 1300, 800, 1500])
    >>> obv_values = obv(close, volume)

    Notes
    -----
    - OBV starts with the first volume value
    - Rising OBV suggests positive volume pressure (accumulation)
    - Falling OBV suggests negative volume pressure (distribution)
    - OBV can be used to confirm price trends or spot divergences
    - No lookback period - first value is the first volume
    """
    if isinstance(close, str) and isinstance(volume, str):
        # Column names provided for Polars
        return obv_polars(close, volume)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()

    # Ensure both arrays have the same length
    if len(close) != len(volume):
        raise ValueError(
            f"close and volume must have the same length. Got {len(close)} and {len(volume)}",
        )

    return obv_numba(close, volume)


# Export all functions
__all__ = ["obv", "obv_numba", "obv_polars"]
