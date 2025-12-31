# mypy: disable-error-code="no-any-return"
"""
CMO - Chande Momentum Oscillator.

The CMO is a technical momentum indicator developed by Tushar Chande.
It is created by calculating the difference between the sum of all recent
gains and the sum of all recent losses and then dividing the result by
the sum of all price movement over the period.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def cmo_numba(close: npt.NDArray[np.float64], timeperiod: int = 14) -> npt.NDArray[np.float64]:
    """
    CMO calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        CMO close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # CMO needs at least timeperiod + 1 close
    if n <= timeperiod:
        return result

    # Initialize gains and losses
    prev_gain = 0.0
    prev_loss = 0.0

    # Calculate initial sums over the period
    prev_value = close[0]
    for i in range(1, timeperiod + 1):
        diff = close[i] - prev_value
        if diff > 0:
            prev_gain += diff
        else:
            prev_loss -= diff  # Make positive
        prev_value = close[i]

    # Average the initial gains and losses
    prev_gain /= timeperiod
    prev_loss /= timeperiod

    # Calculate first CMO value
    total = prev_gain + prev_loss
    if total != 0:
        result[timeperiod] = 100.0 * ((prev_gain - prev_loss) / total)
    else:
        result[timeperiod] = 0.0

    # Calculate remaining CMO close using Wilder's smoothing
    for i in range(timeperiod + 1, n):
        diff = close[i] - close[i - 1]

        # Update smoothed gains and losses
        prev_gain = (prev_gain * (timeperiod - 1)) / timeperiod
        prev_loss = (prev_loss * (timeperiod - 1)) / timeperiod

        if diff > 0:
            prev_gain += diff / timeperiod
        else:
            prev_loss += (-diff) / timeperiod

        # Calculate CMO
        total = prev_gain + prev_loss
        if total != 0:
            result[i] = 100.0 * ((prev_gain - prev_loss) / total)
        else:
            result[i] = 0.0

    return result


def cmo_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    CMO using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for CMO calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(cmo_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="cmo",
    category="momentum",
    description="CMO - Chande Momentum Oscillator",
    lookback=0,
    value_range=(-100.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def cmo(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    CMO - Chande Momentum Oscillator.

    The CMO is a technical momentum indicator that calculates the difference
    between the sum of recent gains and the sum of recent losses, then divides
    by the sum of all price movements over the period. The result is multiplied
    by 100 to give a percentage value that oscillates between -100 and +100.

    Formula:
    CMO = 100 * (Sum of Gains - Sum of Losses) / (Sum of Gains + Sum of Losses)

    Parameters
    ----------
    close : array-like or column name
        Input close (typically close close)
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    array or Polars expression
        CMO close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([44, 44.25, 44.5, 43.75, 44.75, 45.5, 45.25, 46, 47])
    >>> cmo_values = cmo(close, timeperiod=5)

    Notes
    -----
    - CMO close range from -100 to +100
    - Values above +50 indicate strong upward momentum
    - Values below -50 indicate strong downward momentum
    - Zero line crossovers can signal trend changes
    - Similar to RSI but uses (gain-loss)/(gain+loss) instead of gain/(gain+loss)

    Reference: "The New Technical Trader" by Tushar Chande and Stanley Kroll
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return cmo_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return cmo_numba(close, timeperiod)


# Export the main function
__all__ = ["cmo"]
