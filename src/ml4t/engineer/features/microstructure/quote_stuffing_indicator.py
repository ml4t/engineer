import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="quote_stuffing_indicator",
    category="microstructure",
    description="Quote Stuffing Indicator - detects quote manipulation",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def quote_stuffing_indicator(
    volume: pl.Expr | str,
    num_trades: pl.Expr | str | None = None,
    period: int = 5,
) -> pl.Expr:
    """Detect potential quote stuffing or spoofing activity.

    High volume with low actual trades may indicate manipulative behavior.

    Parameters
    ----------
    volume : pl.Expr | str
        Volume column
    num_trades : pl.Expr | str, optional
        Number of trades (if available)
    period : int, default 5
        Short period for detection

    Returns
    -------
    pl.Expr
        Quote stuffing indicator (0-1)

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")

    volume = pl.col(volume) if isinstance(volume, str) else volume

    if num_trades is not None:
        num_trades_col: pl.Expr = pl.col(num_trades) if isinstance(num_trades, str) else num_trades

        # Average trade size
        avg_trade_size = volume / (num_trades_col + 1)

        # Unusually small average trade size may indicate stuffing
        historical_avg = avg_trade_size.rolling_mean(period * 4)
        current_avg = avg_trade_size.rolling_mean(period)

        # Indicator: current average much smaller than historical
        indicator = pl.when(current_avg < 0.5 * historical_avg).then(1.0).otherwise(0.0)
    else:
        # Without trade count, use volume spike detection
        vol_mean = volume.rolling_mean(period * 4)
        vol_std = volume.rolling_std(period * 4)

        # Detect unusual volume spikes
        z_score = (volume - vol_mean) / (vol_std + 1e-10)
        indicator = pl.when(z_score > 3).then(1.0).otherwise(0.0)

    return indicator
