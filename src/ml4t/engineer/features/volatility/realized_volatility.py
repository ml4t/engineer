import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="realized_volatility",
    category="volatility",
    description="Realized Volatility - standard deviation of returns",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
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
