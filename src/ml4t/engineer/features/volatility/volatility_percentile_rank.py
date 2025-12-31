import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="volatility_percentile_rank",
    category="volatility",
    description="Volatility Percentile Rank - current vol vs historical distribution",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volatility_percentile_rank(
    close: pl.Expr | str,
    period: int = 20,
    lookback: int = 252,
) -> pl.Expr:
    """Calculate percentile rank of current volatility.

    Shows where current volatility stands relative to recent history,
    useful for volatility regime identification.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    period : int, default 20
        Period for volatility calculation
    lookback : int, default 252
        Lookback period for percentile calculation

    Returns
    -------
    pl.Expr
        Volatility percentile rank (0-100)

    Raises
    ------
    ValueError
        If period or lookback are not positive
    TypeError
        If period or lookback are not integers
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")
    validate_window(
        lookback,
        min_window=period + 1,
        name="lookback",
    )  # Need more data than volatility period

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate volatility
    returns = close.pct_change()
    vol = returns.rolling_std(period)

    # Calculate rolling percentile rank
    # Use rolling_map to properly calculate percentile rank within each lookback window
    rank = vol.rolling_map(
        lambda x: ((x[-1] > x[:-1]).sum() + 0.5 * (x[-1] == x[:-1]).sum()) / len(x) * 100
        if len(x) > 0
        else None,
        window_size=lookback,
    )

    return rank
