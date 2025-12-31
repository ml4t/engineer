import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="garman_klass_volatility",
    category="volatility",
    description="Garman-Klass Volatility - OHLC-based estimator",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def garman_klass_volatility(
    open: pl.Expr | str,
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    period: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
) -> pl.Expr:
    """Calculate Garman-Klass volatility estimator.

    The Garman-Klass estimator is 7.4x more efficient than close-to-close
    and incorporates opening jumps.

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
        Garman-Klass volatility estimate

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

    # Components
    log_hl = (high / low).log()
    log_co = (close / open).log()

    # Garman-Klass formula
    term1 = 0.5 * log_hl.pow(2)
    term2 = (2 * np.log(2) - 1) * log_co.pow(2)

    gk: pl.Expr = (term1 - term2).rolling_mean(period).sqrt()

    if annualize:
        gk = gk * np.sqrt(trading_periods)

    return gk.cast(pl.Float64)
