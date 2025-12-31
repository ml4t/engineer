import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_positive,
    validate_window,
)


@feature(
    name="price_impact_ratio",
    category="microstructure",
    description="Price Impact Ratio - realized price impact per trade",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def price_impact_ratio(
    returns: pl.Expr | str,
    volume: pl.Expr | str,
    period: int = 20,
    impact_threshold: float = 0.001,
) -> pl.Expr:
    """Calculate ratio of high-impact to low-impact trades.

    Identifies periods where trades have unusually high price impact,
    potentially indicating informed trading or low liquidity.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column
    volume : pl.Expr | str
        Volume column
    period : int, default 20
        Rolling window period
    impact_threshold : float, default 0.001
        Threshold for high impact (0.1%)

    Returns
    -------
    pl.Expr
        High/low impact ratio

    Raises
    ------
    ValueError
        If period is not positive or impact_threshold is not positive
    TypeError
        If period is not an integer or impact_threshold is not numeric
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")
    validate_positive(impact_threshold, name="impact_threshold")

    returns = pl.col(returns) if isinstance(returns, str) else returns
    volume = pl.col(volume) if isinstance(volume, str) else volume

    # Calculate price impact per trade
    impact = returns.abs() / (volume + 1)

    # Classify as high or low impact
    high_impact = (impact > impact_threshold).cast(pl.Int32)
    low_impact = (impact <= impact_threshold).cast(pl.Int32)

    # Calculate ratio
    high_count = high_impact.rolling_sum(period)
    low_count = low_impact.rolling_sum(period)

    ratio = pl.when(low_count > 0).then(high_count / low_count).otherwise(1.0)

    return ratio
