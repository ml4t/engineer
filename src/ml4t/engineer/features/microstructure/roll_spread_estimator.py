import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="roll_spread_estimator",
    category="microstructure",
    description="Roll Spread Estimator - bid-ask spread from price covariances",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def roll_spread_estimator(close: pl.Expr | str, period: int = 20) -> pl.Expr:
    """Estimate bid-ask spread using Roll's model.

    Roll's model estimates the effective spread from price changes,
    useful when quote data is not available.

    Formula: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Estimated bid-ask spread

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

    # Calculate price changes
    price_changes = close.diff()

    # Calculate rolling covariance of price changes with lagged changes
    # Note: Polars doesn't have rolling covariance, so we approximate
    lag_changes = price_changes.shift(1)

    # Calculate components for covariance
    mean_change = price_changes.rolling_mean(period)
    mean_lag = lag_changes.rolling_mean(period)

    # Approximate covariance
    cov_approx = ((price_changes - mean_change) * (lag_changes - mean_lag)).rolling_mean(period)

    # Roll spread = 2 * sqrt(max(-cov, 0))
    spread = 2 * pl.when(cov_approx < 0).then((-cov_approx).sqrt()).otherwise(0)

    return spread
