"""Helper functions for batch indicator processing and utilities."""

import polars as pl


def add_indicators(df: pl.DataFrame, indicators: dict[str, pl.Expr]) -> pl.DataFrame:
    """Batch add multiple indicators to a DataFrame.

    This is a convenience function that allows you to add multiple technical
    indicators to a DataFrame in a single operation. It's equivalent to calling
    df.with_columns() with a list of aliased expressions.

    Args:
        df: Input DataFrame containing price/volume data
        indicators: Dictionary mapping indicator names to Polars expressions

    Returns:
        DataFrame with the new indicator columns added

    Example:
        >>> import polars as pl
        >>> import ml4t.engineer.features.ta as ta as qta
        >>>
        >>> df = pl.DataFrame({
        ...     "close": [100, 102, 104, 103, 105, 107, 106],
        ...     "high": [101, 103, 105, 104, 106, 108, 107],
        ...     "low": [99, 101, 103, 102, 104, 106, 105],
        ... })
        >>>
        >>> indicators = {
        ...     "sma_20": qta.sma("close", 20),
        ...     "rsi_14": qta.rsi("close", 14),
        ...     "macd": qta.macd("close", 12, 26),
        ... }
        >>>
        >>> result = qta.add_indicators(df, indicators)
    """
    expressions = [expr.alias(name) for name, expr in indicators.items()]
    return df.with_columns(expressions)


def validate_ohlcv_columns(df: pl.DataFrame, required_columns: list[str]) -> None:
    """Validate that required OHLCV columns exist in DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {sorted(df.columns)}",
        )


def get_trading_session_filter(
    start_time: str = "09:30",
    end_time: str = "16:00",
) -> pl.Expr:
    """Create a filter expression for trading session hours.

    Useful for filtering intraday data to only include regular trading hours.

    Args:
        start_time: Start of trading session in HH:MM format (default: "09:30")
        end_time: End of trading session in HH:MM format (default: "16:00")

    Returns:
        Polars expression that filters for trading session hours

    Example:
        >>> import polars as pl
        >>> import ml4t.engineer.features.ta as ta.utils as utils
        >>>
        >>> # Assuming your data has a datetime column
        >>> trading_hours_filter = utils.get_trading_session_filter("09:30", "16:00")
        >>> filtered_df = df.filter(trading_hours_filter)
    """
    from datetime import datetime

    start_dt = datetime.strptime(start_time, "%H:%M").time()
    end_dt = datetime.strptime(end_time, "%H:%M").time()
    return (pl.col("datetime").dt.time() >= start_dt) & (pl.col("datetime").dt.time() <= end_dt)
