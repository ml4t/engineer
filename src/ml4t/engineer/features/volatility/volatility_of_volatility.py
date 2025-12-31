import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)

# Helper functions


def realized_volatility(
    returns: pl.Expr | str,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pl.Expr:
    """Calculate realized volatility from returns.

    Standard realized volatility calculation with optional annualization.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column (not close)
    period : int, default 20
        Rolling window period
    annualize : bool, default True
        Whether to annualize the volatility
    trading_periods : int, default 252
        Number of trading periods per year

    Returns
    -------
    pl.Expr
        Realized volatility

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

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Rolling standard deviation
    vol = returns.rolling_std(period)

    if annualize:
        vol = vol * np.sqrt(trading_periods)

    return vol


# Main feature function


@feature(
    name="volatility_of_volatility",
    category="volatility",
    description="Volatility of Volatility - second-order volatility measure",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volatility_of_volatility(
    close: pl.Expr | str,
    vol_period: int = 20,
    vov_period: int = 20,
    annualize: bool = True,
) -> pl.Expr:
    """Calculate volatility of volatility (vol-of-vol).

    Measures how volatile the volatility itself is, useful for detecting
    regime changes and volatility clustering.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    vol_period : int, default 20
        Period for calculating base volatility
    vov_period : int, default 20
        Period for calculating volatility of volatility
    annualize : bool, default True
        Whether to annualize

    Returns
    -------
    pl.Expr
        Volatility of volatility

    Raises
    ------
    ValueError
        If vol_period or vov_period are not positive
    TypeError
        If vol_period or vov_period are not integers, or annualize is not boolean
    """
    # Validate inputs
    validate_window(vol_period, min_window=2, name="vol_period")
    validate_window(vov_period, min_window=2, name="vov_period")
    if not isinstance(annualize, bool):
        raise TypeError(f"annualize must be a boolean, got {type(annualize).__name__}")

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate returns
    returns = close.pct_change()

    # Calculate rolling volatility
    vol = realized_volatility(returns, vol_period, annualize)

    # Calculate volatility of the volatility
    vol_returns = vol.pct_change()
    vov = vol_returns.rolling_std(vov_period)

    if annualize:
        vov = vov * np.sqrt(252)

    return vov
