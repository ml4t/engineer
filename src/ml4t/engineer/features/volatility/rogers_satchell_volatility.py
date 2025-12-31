import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="rogers_satchell_volatility",
    category="volatility",
    description="Rogers-Satchell Volatility - drift-independent OHLC estimator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def rogers_satchell_volatility(
    open: pl.Expr | str,
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pl.Expr:
    """Calculate Rogers-Satchell volatility estimator.

    The Rogers-Satchell estimator handles non-zero drift and is particularly
    useful for trending markets.

    Parameters
    ----------
    open, high, low, close : pl.Expr | str
        OHLC price columns
    period : int, default 20
        Rolling window period
    annualize : bool, default True
        Whether to annualize the volatility
    trading_periods : int, default 252
        Number of trading periods per year

    Returns
    -------
    pl.Expr
        Rogers-Satchell volatility estimate

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

    open = pl.col(open) if isinstance(open, str) else open
    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low
    close = pl.col(close) if isinstance(close, str) else close

    # Rogers-Satchell formula
    log_hc = (high / close).log()
    log_ho = (high / open).log()
    log_lc = (low / close).log()
    log_lo = (low / open).log()

    rs = (log_hc * log_ho + log_lc * log_lo).rolling_mean(period).sqrt()

    if annualize:
        rs = rs * np.sqrt(trading_periods)

    return rs
