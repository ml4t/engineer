"""Arithmetic Operations - Basic mathematical transformations and calculations.

These functions provide fundamental mathematical operations commonly used
in technical analysis and financial calculations.
"""

import polars as pl

from ml4t.engineer.core.exceptions import InvalidParameterError


def _price_return(
    column: str | pl.Expr,
    periods: int,
    *,
    logarithmic: bool,
    reference: pl.Expr | None = None,
    float_difference: bool = False,
) -> pl.Expr:
    def build(price: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        previous = price.shift(periods) if reference is None else reference
        decimal = isinstance(dtype, pl.Decimal)
        divisor = previous.cast(pl.Float64) if decimal else previous
        ratio = price.cast(pl.Float64) / divisor if decimal else price / previous
        if not dtype.is_integer() and not decimal:
            if logarithmic:
                return ratio.log()
            return (price - previous) / previous if float_difference else ratio - 1

        wide = price if decimal else price.cast(pl.Int128)
        lagged = previous if decimal else previous.cast(pl.Int128)
        if dtype == pl.Int128 or decimal:
            # Opposite-sign differences can exceed the widest integer/decimal representation.
            same_sign = ((wide >= 0) & (lagged >= 0)) | ((wide <= 0) & (lagged <= 0))
            safe_price = pl.when(same_sign).then(wide).otherwise(0)
            safe_previous = pl.when(same_sign).then(lagged).otherwise(0)
            difference = (safe_price - safe_previous).cast(pl.Float64)
            gain = pl.when(same_sign).then(difference / divisor).otherwise(ratio - 1)
        else:
            gain = (wide - lagged).cast(pl.Float64) / divisor
        if not logarithmic:
            return gain
        # log1p retains tiny ticks, while the ratio retains near-total losses and domains.
        return pl.when(gain.abs() < 0.5).then(gain.log1p()).otherwise(ratio.log())

    price = pl.col(column) if isinstance(column, str) else column
    return price.pipe_with_dtype(build)


def returns(column: str, periods: int = 1) -> pl.Expr:
    """Price returns over N periods.

    Calculates the percentage change in price over a specified number of periods.
    This is one of the most fundamental calculations in quantitative finance.

    Formula:
        Return = (Price[t] - Price[t-periods]) / Price[t-periods]

    Integer and Decimal differences are retained before Float64 conversion.
    Floating inputs retain their existing ratio calculation and output dtype.

    Args:
        column: Name of the column containing price data
        periods: Number of periods to look back (default: 1 for daily returns)

    Returns:
        Polars expression for the returns (as decimal, e.g., 0.05 = 5%)

    Raises:
        InvalidParameterError: If periods < 1

    Example:
        >>> import polars as pl
        >>> from ml4t.engineer.features.utils import arithmetic as qta
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

    return _price_return(column, periods, logarithmic=False)


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
        >>> from ml4t.engineer.features.utils import arithmetic as qta
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

    return _price_return(column, periods, logarithmic=True)


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
        >>> from ml4t.engineer.features.utils import arithmetic as qta
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
