import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="volume_at_price_ratio",
    category="microstructure",
    description="Volume at Price Ratio - volume profile concentration",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volume_at_price_ratio(
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    volume: pl.Expr | str,
    n_bins: int = 10,
    period: int = 20,
) -> pl.Expr:
    """Calculate volume concentration at different price levels.

    Measures how volume is distributed across the price range,
    useful for identifying support/resistance levels.

    Parameters
    ----------
    high, low, close : pl.Expr | str
        Price columns
    volume : pl.Expr | str
        Volume column
    n_bins : int, default 10
        Number of price bins
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Volume concentration ratio (0-1)

    Raises
    ------
    ValueError
        If n_bins or period are not positive
    TypeError
        If n_bins or period are not integers
    """
    # Validate inputs
    validate_window(n_bins, min_window=2, name="n_bins")
    validate_window(period, min_window=1, name="period")

    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low
    close = pl.col(close) if isinstance(close, str) else close
    volume = pl.col(volume) if isinstance(volume, str) else volume

    # Calculate price position within daily range
    price_position = (close - low) / (high - low + 1e-10)

    # Calculate volume concentration
    # High concentration = volume focused in few price levels
    # Using simplified measure based on volume at high price levels
    bin_volumes = volume.rolling_sum(period)

    # Concentration ratio (simplified)
    # Measures proportion of volume at high price levels (>80% of range)
    concentration = (
        pl.when(price_position > 0.8).then(volume).otherwise(0).rolling_sum(period) / bin_volumes
    )

    return concentration
