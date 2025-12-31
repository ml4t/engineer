"""Arithmetic Operations - Basic mathematical transformations and calculations.

These functions provide fundamental mathematical operations commonly used
in technical analysis and financial calculations.
"""

import polars as pl

from ml4t.engineer.core.exceptions import InvalidParameterError


def returns(column: str, periods: int = 1) -> pl.Expr:
    """Price returns over N periods.

    Calculates the percentage change in price over a specified number of periods.
    This is one of the most fundamental calculations in quantitative finance.

    Formula:
        Return = (Price[t] / Price[t-periods]) - 1

    Args:
        column: Name of the column containing price data
        periods: Number of periods to look back (default: 1 for daily returns)

    Returns:
        Polars expression for the returns (as decimal, e.g., 0.05 = 5%)

    Raises:
        InvalidParameterError: If periods < 1

    Example:
        >>> import polars as pl
        >>> import ml4t.engineer.features.ta as qta
        >>>
        >>> df = pl.DataFrame({
        ...     "close": [100, 105, 103, 107, 110]
        ... })
        >>>
        >>> result = df.with_columns([
        ...     qta.returns("close", 1).alias("daily_returns"),
        ...     qta.returns("close", 2).alias("2day_returns")
        ... ])
    """
    if periods < 1:
        raise InvalidParameterError(f"periods must be >= 1, got {periods}")

    return (pl.col(column) / pl.col(column).shift(periods)) - 1


def log_returns(column: str, periods: int = 1) -> pl.Expr:
    """Logarithmic returns over N periods.

    Calculates the natural logarithm of the price ratio. Log returns have
    better statistical properties than simple returns and are additive
    over time periods.

    Formula:
        Log Return = ln(Price[t] / Price[t-periods])

    Args:
        column: Name of the column containing price data
        periods: Number of periods to look back (default: 1)

    Returns:
        Polars expression for the log returns

    Raises:
        InvalidParameterError: If periods < 1

    Example:
        >>> import polars as pl
        >>> import ml4t.engineer.features.ta as qta
        >>>
        >>> df = pl.DataFrame({
        ...     "close": [100, 105, 103, 107, 110]
        ... })
        >>>
        >>> result = df.with_columns([
        ...     qta.log_returns("close", 1).alias("log_returns")
        ... ])
    """
    if periods < 1:
        raise InvalidParameterError(f"periods must be >= 1, got {periods}")

    return (pl.col(column) / pl.col(column).shift(periods)).log()


def percentage_change(column: str, periods: int = 1) -> pl.Expr:
    """Percentage change over N periods.

    Similar to returns() but expressed as percentage (multiplied by 100).
    Useful for display purposes when you want percentage close.

    Formula:
        Percentage Change = ((Price[t] / Price[t-periods]) - 1) × 100

    Args:
        column: Name of the column containing price data
        periods: Number of periods to look back (default: 1)

    Returns:
        Polars expression for the percentage change (e.g., 5.0 = 5%)

    Raises:
        InvalidParameterError: If periods < 1

    Example:
        >>> import polars as pl
        >>> import ml4t.engineer.features.ta as qta
        >>>
        >>> df = pl.DataFrame({
        ...     "close": [100, 105, 103, 107, 110]
        ... })
        >>>
        >>> result = df.with_columns([
        ...     qta.percentage_change("close", 1).alias("pct_change")
        ... ])
    """
    return returns(column, periods) * 100
