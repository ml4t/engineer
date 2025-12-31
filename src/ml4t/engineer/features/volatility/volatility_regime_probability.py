import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_positive,
    validate_window,
)


@feature(
    name="volatility_regime_probability",
    category="volatility",
    description="Volatility Regime Probability - probability of high/low vol regime",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volatility_regime_probability(
    close: pl.Expr | str,
    low_vol_threshold: float = 0.01,
    high_vol_threshold: float = 0.02,
    period: int = 20,
    lookback: int = 100,
) -> dict[str, pl.Expr]:
    """Calculate probability of being in low/medium/high volatility regime.

    Uses a simple threshold-based approach to classify volatility regimes
    and calculate rolling probabilities.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    low_vol_threshold : float, default 0.01
        Threshold for low volatility regime (1% daily)
    high_vol_threshold : float, default 0.02
        Threshold for high volatility regime (2% daily)
    period : int, default 20
        Period for volatility calculation
    lookback : int, default 100
        Lookback for regime probability calculation

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary with regime probabilities

    Raises
    ------
    ValueError
        If thresholds are negative, period/lookback are not positive, or threshold ordering is incorrect
    TypeError
        If period/lookback are not integers or thresholds are not numeric
    """
    # Validate inputs
    validate_positive(low_vol_threshold, name="low_vol_threshold")
    validate_positive(high_vol_threshold, name="high_vol_threshold")
    validate_window(period, min_window=2, name="period")
    validate_window(lookback, min_window=2, name="lookback")

    if low_vol_threshold >= high_vol_threshold:
        raise ValueError(
            f"low_vol_threshold ({low_vol_threshold}) must be less than high_vol_threshold ({high_vol_threshold})",
        )

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate daily volatility
    returns = close.pct_change()
    vol = returns.rolling_std(period)

    # Classify regimes
    low_vol = (vol < low_vol_threshold).cast(pl.Int32)
    med_vol = ((vol >= low_vol_threshold) & (vol < high_vol_threshold)).cast(pl.Int32)
    high_vol = (vol >= high_vol_threshold).cast(pl.Int32)

    # Calculate rolling probabilities
    return {
        "prob_low_vol": low_vol.rolling_mean(lookback),
        "prob_med_vol": med_vol.rolling_mean(lookback),
        "prob_high_vol": high_vol.rolling_mean(lookback),
        "current_vol": vol,
    }
