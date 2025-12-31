import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)

# Helper functions


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


# Main feature function


@feature(
    name="yang_zhang_volatility",
    category="volatility",
    description="Yang-Zhang Volatility - combines overnight and intraday volatility",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def yang_zhang_volatility(
    open: pl.Expr | str,
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pl.Expr:
    """Calculate Yang-Zhang volatility estimator.

    The Yang-Zhang estimator is the most accurate, combining overnight and
    intraday volatility with minimal bias.

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
        Yang-Zhang volatility estimate

    Raises
    ------
    ValueError
        If period or trading_periods are not positive
    TypeError
        If period or trading_periods are not integers, or annualize is not boolean
    """
    # Validate inputs
    validate_window(
        period,
        min_window=3,
        name="period",
    )  # Need extra data for variance calculation
    validate_window(trading_periods, min_window=1, name="trading_periods")
    if not isinstance(annualize, bool):
        raise TypeError(f"annualize must be a boolean, got {type(annualize).__name__}")

    open = pl.col(open) if isinstance(open, str) else open
    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low
    close = pl.col(close) if isinstance(close, str) else close

    # Overnight volatility (close-to-open)
    log_co = (open / close.shift(1)).log()
    overnight_var = log_co.rolling_var(window_size=period, ddof=1)

    # Open-to-close volatility
    log_oc = (close / open).log()
    openclose_var = log_oc.rolling_var(window_size=period, ddof=1)

    # Rogers-Satchell volatility
    rs_vol = rogers_satchell_volatility(open, high, low, close, period, False)

    # Yang-Zhang formula with k factor
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    yz = (overnight_var + k * openclose_var + (1 - k) * rs_vol.pow(2)).sqrt()

    if annualize:
        yz = yz * np.sqrt(trading_periods)

    return yz
