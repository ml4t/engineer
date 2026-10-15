import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="roll_spread_estimator",
    category="microstructure",
    description="Roll Spread Estimator - bid-ask spread from price covariances",
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
        Estimated bid-ask spread, null for an unobserved covariance window

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

    # Serial covariance of price changes. Demeaning each point by its own
    # trailing mean and then averaging those products is not the covariance of
    # the window; cov = E[xy] - E[x]E[y] over the same window is.
    price_changes = close.diff()
    lag_changes = price_changes.shift(1)
    mean_change = price_changes.rolling_mean(period)
    mean_lag = lag_changes.rolling_mean(period)
    cov_approx = (price_changes * lag_changes).rolling_mean(period) - mean_change * mean_lag

    # Roll spread = 2 * sqrt(max(-cov, 0))
    spread = 2 * pl.when(cov_approx.is_null() | cov_approx.is_nan()).then(None).when(
        cov_approx < 0
    ).then((-cov_approx).sqrt()).otherwise(0)

    return spread
