# mypy: disable-error-code="no-any-return,union-attr"
"""
IMI - Intraday Momentum Index.

IMI is a variation of RSI that uses the relationship between open and close close
instead of consecutive closes. It measures the momentum of intraday price movements.

Formula:
IMI = 100 * Sum(Gains) / (Sum(Gains) + Sum(Losses))
Where:
- Gains = Close - Open when Close > Open, else 0
- Losses = Open - Close when Close < Open, else 0
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def imi_numba(
    open: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod: int,
) -> npt.NDArray[np.float64]:
    """Calculate Intraday Momentum Index using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(timeperiod - 1, n):
        sum_gains = 0.0
        sum_losses = 0.0

        # Calculate gains and losses over the period
        for j in range(i - timeperiod + 1, i + 1):
            diff = close[j] - open[j]

            if diff > 0:
                sum_gains += diff
            else:
                sum_losses += abs(diff)

        # Calculate IMI
        total = sum_gains + sum_losses
        if total > 1e-10:
            result[i] = 100.0 * sum_gains / total
        # When no movement, leave as NaN (matches TA-Lib)

    return result


@feature(
    name="imi",
    category="momentum",
    description="IMI - Intraday Momentum Index",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=False,
)
def imi(
    open: npt.NDArray[np.float64] | pl.Expr | str,
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Intraday Momentum Index.

    IMI measures the relationship between intraday gains and losses,
    providing insight into intraday momentum patterns.

    Parameters
    ----------
    open : array-like or Polars expression
        Open close
    close : array-like or Polars expression
        Close close
    timeperiod : int, default 14
        Period for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        IMI close between 0 and 100

    Notes
    -----
    IMI calculation:
    1. For each period, calculate gain = close - open (if positive)
    2. For each period, calculate loss = open - close (if positive)
    3. IMI = 100 * Sum(gains) / (Sum(gains) + Sum(losses))

    Interpretation:
    - IMI > 70: Overbought condition
    - IMI < 30: Oversold condition
    - IMI = 50: Equal gains and losses

    Key differences from RSI:
    - Uses intraday (open to close) instead of interday changes
    - No smoothing applied (simple sums)
    - Better for intraday trading analysis

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> open_prices = np.array([100, 102, 101, 103, 105, 104, 106])
    >>> close_prices = np.array([102, 101, 103, 104, 104, 106, 105])
    >>>
    >>> imi_values = qta.imi(open_prices, close_prices, timeperiod=5)

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "open": open_prices,
    ...     "close": close_prices
    ... })
    >>> result = df.with_columns([
    ...     qta.imi("open", "close", 14).alias("imi")
    ... ])
    """
    if isinstance(open, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        open_expr = pl.col(open) if isinstance(open, str) else open
        if isinstance(close, str):
            close = pl.col(close)
        # Type narrowing: at this point close must be Expr
        assert isinstance(close, pl.Expr)

        def calc_imi(s: pl.Series) -> pl.Series:
            if isinstance(s[0], dict):
                # When using struct, we get a list of dicts
                o = np.array([x["open"] for x in s])
                c = np.array([x["close"] for x in s])
            else:
                # Direct array input
                o = s.struct.field("open").to_numpy()
                c = s.struct.field("close").to_numpy()
            return pl.Series(imi_numba(o, c, timeperiod))

        return pl.struct([open_expr.alias("open"), close.alias("close")]).map_batches(
            calc_imi,
            return_dtype=pl.Float64,
        )

    # Handle numpy arrays
    open_array = np.asarray(open, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if len(open_array) != len(close):
        raise ValueError("open and close must have the same length")

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(open_array) < timeperiod:
        return np.full(len(open_array), np.nan)

    return imi_numba(open_array, close, timeperiod)


# Make available at package level
__all__ = ["imi"]
