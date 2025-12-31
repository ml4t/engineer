import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="trade_intensity",
    category="microstructure",
    description="Trade Intensity - trade arrival rate",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def trade_intensity(
    volume: pl.Expr | str,
    time_interval: int = 1,
    period: int = 20,
) -> pl.Expr:
    """Calculate trade intensity (volume per time unit).

    Measures the rate of trading activity, useful for identifying
    periods of high market activity.

    Parameters
    ----------
    volume : pl.Expr | str
        Volume column
    time_interval : int, default 1
        Time interval in minutes
    period : int, default 20
        Rolling window period

    Returns
    -------
    pl.Expr
        Trade intensity

    Raises
    ------
    ValueError
        If time_interval or period are not positive
    TypeError
        If time_interval or period are not integers
    """
    # Validate inputs
    validate_window(time_interval, min_window=1, name="time_interval")
    validate_window(period, min_window=1, name="period")

    volume = pl.col(volume) if isinstance(volume, str) else volume

    # Volume per time interval
    volume_rate = volume / time_interval

    # Normalized by rolling average
    avg_rate = volume_rate.rolling_mean(period)
    intensity = volume_rate / (avg_rate + 1e-10)

    return intensity
