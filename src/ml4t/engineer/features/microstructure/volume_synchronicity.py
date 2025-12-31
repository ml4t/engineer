import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="volume_synchronicity",
    category="microstructure",
    description="Volume Synchronicity - correlation of volume across assets",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volume_synchronicity(
    volume: pl.Expr | str,
    returns: pl.Expr | str,
    period: int = 20,
) -> pl.Expr:
    """Calculate volume-return synchronicity.

    Measures how synchronized volume and price movements are,
    useful for detecting accumulation/distribution patterns.

    Parameters
    ----------
    volume : pl.Expr | str
        Volume column
    returns : pl.Expr | str
        Returns column
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Synchronicity measure (-1 to 1)

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")

    volume = pl.col(volume) if isinstance(volume, str) else volume
    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Normalize volume changes
    volume_change = volume.pct_change()

    # Calculate rolling correlation between volume changes and returns
    # Simplified correlation calculation
    mean_vol = volume_change.rolling_mean(period)
    mean_ret = returns.rolling_mean(period)

    covariance = ((volume_change - mean_vol) * (returns - mean_ret)).rolling_mean(
        period,
    )
    vol_std = volume_change.rolling_std(period)
    ret_std = returns.rolling_std(period)

    correlation = covariance / (vol_std * ret_std + 1e-10)

    return correlation
