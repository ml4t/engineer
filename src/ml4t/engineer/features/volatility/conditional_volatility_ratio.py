import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="conditional_volatility_ratio",
    category="volatility",
    description="Conditional Volatility Ratio - vol in up markets vs down markets",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def conditional_volatility_ratio(
    returns: pl.Expr | str,
    threshold: float = 0.0,
    period: int = 20,
) -> pl.Expr:
    """Calculate ratio of upside to downside volatility.

    Measures asymmetry in volatility, useful for detecting different
    behaviors in up vs down moves.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column
    threshold : float, default 0.0
        Threshold for separating upside/downside
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Upside/downside volatility ratio

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer or threshold is not numeric
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Separate positive and negative returns
    upside_returns = pl.when(returns > threshold).then(returns).otherwise(0)
    downside_returns = pl.when(returns < threshold).then(returns).otherwise(0)

    # Calculate conditional volatilities
    upside_vol = upside_returns.rolling_std(period)
    downside_vol = downside_returns.abs().rolling_std(period)

    # Ratio (handle division by zero)
    ratio = pl.when(downside_vol > 0).then(upside_vol / downside_vol).otherwise(1.0)

    return ratio
