import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="parkinson_volatility",
    category="volatility",
    description="Parkinson Volatility - range-based volatility estimator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def parkinson_volatility(
    high: pl.Expr | str,
    low: pl.Expr | str,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pl.Expr:
    """Calculate Parkinson volatility estimator using high-low range.

    The Parkinson estimator is 5x more efficient than close-to-close volatility
    and captures intraday volatility that close close miss.

    Formula: sqrt(1/(4*n*ln(2)) * sum(ln(H/L)^2))

    Parameters
    ----------
    high : pl.Expr | str
        High price column
    low : pl.Expr | str
        Low price column
    period : int, default 20
        Rolling window period
    annualize : bool, default True
        Whether to annualize the volatility
    trading_periods : int, default 252
        Number of trading periods per year

    Returns
    -------
    pl.Expr
        Parkinson volatility estimate

    Raises
    ------
    ValueError
        If period or trading_periods are not positive
    TypeError
        If period or trading_periods are not integers, or annualize is not boolean
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")
    validate_window(trading_periods, min_window=1, name="trading_periods")
    if not isinstance(annualize, bool):
        raise TypeError(f"annualize must be a boolean, got {type(annualize).__name__}")

    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low

    # Calculate log of high/low ratio
    log_hl = (high / low).log()

    # Parkinson volatility
    factor = 1.0 / (4.0 * np.log(2))
    parkinson: pl.Expr = (factor * log_hl.pow(2).rolling_mean(period)).sqrt()

    if annualize:
        parkinson = parkinson * np.sqrt(trading_periods)

    return parkinson.cast(pl.Float64)
