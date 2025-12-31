import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="volume_weighted_price_momentum",
    category="microstructure",
    description="VWPM - volume-weighted price momentum",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volume_weighted_price_momentum(
    close: pl.Expr | str,
    volume: pl.Expr | str,
    period: int = 20,
) -> pl.Expr:
    """Calculate volume-weighted price momentum.

    Measures momentum weighted by volume, giving more weight to
    high-volume price moves.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    volume : pl.Expr | str
        Volume column
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Volume-weighted momentum

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")

    close = pl.col(close) if isinstance(close, str) else close
    volume = pl.col(volume) if isinstance(volume, str) else volume

    # Calculate returns
    returns = close.pct_change()

    # Volume-weighted returns
    vw_returns = (returns * volume).rolling_sum(period) / volume.rolling_sum(period)

    return vw_returns
