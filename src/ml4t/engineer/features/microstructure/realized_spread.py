import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="realized_spread",
    category="microstructure",
    description="Realized Spread - measures post-trade price reversion",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def realized_spread(
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    period: int = 20,
) -> pl.Expr:
    """Calculate realized spread based on high-low range.

    A simple but effective measure of trading costs and liquidity.

    Parameters
    ----------
    high, low, close : pl.Expr | str
        Price columns
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Realized spread estimate

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")

    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low
    close = pl.col(close) if isinstance(close, str) else close

    # Realized spread = 2 * |close - midpoint|
    midpoint = (high + low) / 2
    spread = 2 * (close - midpoint).abs() / midpoint

    # Rolling average
    return spread.rolling_mean(period)
