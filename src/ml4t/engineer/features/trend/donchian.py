# mypy: disable-error-code="operator,assignment,union-attr,call-arg"
"""
Donchian Channels - Price channel indicator.

Donchian Channels track the highest high and lowest low over a rolling window,
creating a price channel that helps identify breakouts and support/resistance levels.

Developed by Richard Donchian, these channels are commonly used in trend-following
and breakout trading strategies.
"""

import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@feature(
    name="donchian_channels",
    category="trend",
    description="Donchian Channel - highest high / lowest low over period",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def donchian_channels(
    high: str = "high",
    low_col: str = "low",
    period: int = 20,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """
    Calculate Donchian Channels.

    Returns upper band (highest high), lower band (lowest low), and middle band
    (average of upper and lower).

    Parameters
    ----------
    high : str, default "high"
        Column name containing high close
    low_col : str, default "low"
        Column name containing low close
    period : int, default 20
        Lookback period for channel calculation

    Returns
    -------
    tuple[pl.Expr, pl.Expr, pl.Expr]
        (upper_band, lower_band, middle_band) as Polars expressions

    Raises
    ------
    InvalidParameterError
        If period < 1

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.trend import donchian_channels
    >>>
    >>> df = pl.DataFrame({
    ...     "high": [102, 104, 106, 105, 107, 109, 108],
    ...     "low": [98, 100, 102, 101, 103, 105, 104],
    ... })
    >>>
    >>> upper, lower, middle = donchian_channels("high", "low", period=3)
    >>> result = df.with_columns([
    ...     upper.alias("donchian_upper"),
    ...     lower.alias("donchian_lower"),
    ...     middle.alias("donchian_middle"),
    ... ])

    References
    ----------
    .. [1] Donchian, R. (1960). "Donchian's 5- and 20-Day Moving Averages".
           Futures Magazine.
    """
    if period < 1:
        raise InvalidParameterError(
            f"period must be >= 1, got {period}",
            context={"parameter": "period", "value": period},
        )

    # Upper band: Highest high over period
    upper = pl.col(high).rolling_max(window_size=period)

    # Lower band: Lowest low over period
    lower = pl.col(low_col).rolling_min(window_size=period)

    # Middle band: Average of upper and lower
    middle = (upper + lower) / 2

    return upper, lower, middle


def donchian_upper(
    high: str = "high",
    period: int = 20,
) -> pl.Expr:
    """
    Calculate Donchian Channel upper band only.

    Parameters
    ----------
    high : str, default "high"
        Column name containing high close
    period : int, default 20
        Lookback period

    Returns
    -------
    pl.Expr
        Upper band expression

    Examples
    --------
    >>> df = df.with_columns(
    ...     donchian_upper("high", 20).alias("donchian_20_upper")
    ... )
    """
    if period < 1:
        raise InvalidParameterError(
            f"period must be >= 1, got {period}",
            context={"parameter": "period", "value": period},
        )

    return pl.col(high).rolling_max(window_size=period)


def donchian_lower(
    low_col: str = "low",
    period: int = 20,
) -> pl.Expr:
    """
    Calculate Donchian Channel lower band only.

    Parameters
    ----------
    low_col : str, default "low"
        Column name containing low close
    period : int, default 20
        Lookback period

    Returns
    -------
    pl.Expr
        Lower band expression

    Examples
    --------
    >>> df = df.with_columns(
    ...     donchian_lower("low", 20).alias("donchian_20_lower")
    ... )
    """
    if period < 1:
        raise InvalidParameterError(
            f"period must be >= 1, got {period}",
            context={"parameter": "period", "value": period},
        )

    return pl.col(low_col).rolling_min(window_size=period)


def donchian_middle(
    high: str = "high",
    low_col: str = "low",
    period: int = 20,
) -> pl.Expr:
    """
    Calculate Donchian Channel middle band only.

    Parameters
    ----------
    high : str, default "high"
        Column name containing high close
    low_col : str, default "low"
        Column name containing low close
    period : int, default 20
        Lookback period

    Returns
    -------
    pl.Expr
        Middle band expression

    Examples
    --------
    >>> df = df.with_columns(
    ...     donchian_middle("high", "low", 20).alias("donchian_20_middle")
    ... )
    """
    if period < 1:
        raise InvalidParameterError(
            f"period must be >= 1, got {period}",
            context={"parameter": "period", "value": period},
        )

    upper = pl.col(high).rolling_max(window_size=period)
    lower = pl.col(low_col).rolling_min(window_size=period)

    return (upper + lower) / 2
