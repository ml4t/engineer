# mypy: disable-error-code="no-any-return,arg-type,union-attr"
"""
BOP - Balance of Power.

BOP measures the strength of buyers versus sellers by assessing the ability
of each to push price to extreme levels. The indicator relates the close
price to the trading range.

Formula:
BOP = (Close - Open) / (High - Low)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def bop_numba(
    open: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate Balance of Power using Numba for performance."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(n):
        high_low_diff = high[i] - low[i]

        # Avoid division by zero
        if abs(high_low_diff) > 1e-10:
            result[i] = (close[i] - open[i]) / high_low_diff
        else:
            # When high equals low, BOP is 0
            result[i] = 0.0

    return result


@feature(
    name="bop",
    category="momentum",
    description="BOP - Balance of Power",
    lookback=0,
    value_range=(-1.0, 1.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def bop(
    open: npt.NDArray[np.float64] | pl.Expr | str,
    high: npt.NDArray[np.float64] | pl.Expr | str,
    low: npt.NDArray[np.float64] | pl.Expr | str,
    close: npt.NDArray[np.float64] | pl.Expr | str,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Balance of Power.

    BOP measures market strength by comparing the ability of buyers and sellers
    to push close to extreme levels within the trading range.

    Parameters
    ----------
    open : array-like or Polars expression
        Open close
    high : array-like or Polars expression
        High close
    low : array-like or Polars expression
        Low close
    close : array-like or Polars expression
        Close close

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        BOP close between -1 and 1

    Notes
    -----
    BOP calculation:
    BOP = (Close - Open) / (High - Low)

    Interpretation:
    - BOP > 0: Buyers in control (close > open)
    - BOP < 0: Sellers in control (close < open)
    - BOP near +1: Strong buying pressure
    - BOP near -1: Strong selling pressure
    - BOP near 0: Balance between buyers and sellers

    Special cases:
    - When High = Low (no range), BOP = 0
    - Values are bounded between -1 and +1

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> open_prices = np.array([100, 102, 101, 103, 105])
    >>> high_prices = np.array([102, 103, 103, 105, 106])
    >>> low_prices = np.array([99, 101, 100, 102, 104])
    >>> close_prices = np.array([101, 101, 102, 104, 105])
    >>>
    >>> bop_values = qta.bop(open_prices, high_prices, low_prices, close_prices)
    >>> print(bop_values)
    [0.333 -0.5 0.333 0.333 0.0]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "open": open_prices,
    ...     "high": high_prices,
    ...     "low": low_prices,
    ...     "close": close_prices
    ... })
    >>> result = df.with_columns([
    ...     qta.bop("open", "high", "low", "close").alias("bop")
    ... ])
    """
    if isinstance(open, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        open_expr = pl.col(open) if isinstance(open, str) else open
        if isinstance(high, str):
            high = pl.col(high)
        if isinstance(low, str):
            low = pl.col(low)
        if isinstance(close, str):
            close = pl.col(close)
        # Type narrowing: at this point all must be Expr
        assert isinstance(high, pl.Expr) and isinstance(low, pl.Expr) and isinstance(close, pl.Expr)

        def calc_bop(s: pl.Series) -> pl.Series:
            if isinstance(s[0], dict):
                # When using struct, we get a list of dicts
                o = np.array([x["open"] for x in s])
                h = np.array([x["high"] for x in s])
                lo = np.array([x["low"] for x in s])
                c = np.array([x["close"] for x in s])
            else:
                # Direct struct access
                o = s.struct.field("open").to_numpy()
                h = s.struct.field("high").to_numpy()
                lo = s.struct.field("low").to_numpy()
                c = s.struct.field("close").to_numpy()
            return pl.Series(bop_numba(o, h, lo, c))

        return pl.struct(
            [
                open_expr.alias("open"),
                high.alias("high"),
                low.alias("low"),
                close.alias("close"),
            ],
        ).map_batches(calc_bop, return_dtype=pl.Float64)

    # Handle numpy arrays
    open = np.asarray(open, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if not (len(open) == len(high) == len(low) == len(close)):
        raise ValueError("open, high, low, and close must have the same length")

    if len(open) == 0:
        return np.array([])

    return bop_numba(open, high, low, close)


# Make available at package level
__all__ = ["bop"]
