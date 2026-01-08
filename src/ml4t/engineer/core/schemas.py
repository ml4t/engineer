"""Data schemas for ml4t-engineer.

Defines the canonical schemas for input and output data.
"""

import polars as pl

# Canonical OHLCV schema for time series data
OHLCV_SCHEMA = {
    "event_time": pl.Datetime("ns"),
    "asset_id": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Extended schema with additional fields
EXTENDED_OHLCV_SCHEMA = {
    **OHLCV_SCHEMA,
    "vwap": pl.Float64,
    "trade_count": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "open_interest": pl.Float64,  # For futures
}

# Schema for labeled data
LABELED_DATA_SCHEMA = {
    "event_time": pl.Datetime("ns"),
    "asset_id": pl.Utf8,
    "label": pl.Int8,  # -1, 0, 1
    "label_time": pl.Datetime("ns"),
    "label_price": pl.Float64,
    "label_return": pl.Float64,
    "weight": pl.Float64,
}

# Schema for feature output
FEATURE_SCHEMA = {
    "event_time": pl.Datetime("ns"),
    "asset_id": pl.Utf8,
    # Features are added dynamically
}


def validate_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> None:
    """Validate that a DataFrame conforms to a schema.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate
    schema : dict[str, pl.DataType]
        The expected schema

    Raises
    ------
    ValueError
        If the DataFrame doesn't match the schema
    """
    # Check for missing columns
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check data types
    for col, expected_type in schema.items():
        if col in df.columns:
            actual_type = df[col].dtype
            if actual_type != expected_type:
                raise ValueError(
                    f"Column '{col}' has type {actual_type}, expected {expected_type}",
                )


def validate_ohlcv_schema(
    df: pl.DataFrame | pl.LazyFrame,
    require_asset_id: bool = True,
    allow_flexible_time: bool = True,
) -> None:
    """Validate OHLCV schema compatibility with qdata.

    This validator ensures that input data conforms to the expected OHLCV
    format for ml4t-engineer. It's more flexible than strict schema validation,
    allowing for common variations while ensuring core requirements are met.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The DataFrame to validate
    require_asset_id : bool, default True
        Whether to require the asset_id column
    allow_flexible_time : bool, default True
        Whether to allow timestamp/event_time/date variations

    Raises
    ------
    DataSchemaError
        If schema doesn't match expected format

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.core.schemas import validate_ohlcv_schema
    >>>
    >>> # Valid OHLCV data
    >>> df = pl.DataFrame({
    ...     "event_time": pl.datetime_range(...),
    ...     "asset_id": ["AAPL"] * 100,
    ...     "open": [...],
    ...     "high": [...],
    ...     "low": [...],
    ...     "close": [...],
    ...     "volume": [...],
    ... })
    >>> validate_ohlcv_schema(df)  # Passes silently
    """
    from ml4t.engineer.core.exceptions import DataSchemaError

    # Materialize schema if LazyFrame
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

    # Check for required OHLCV columns
    ohlcv_cols = {"open", "high", "low", "close", "volume"}
    missing = ohlcv_cols - set(schema.keys())
    if missing:
        raise DataSchemaError(
            f"Missing required OHLCV columns: {sorted(missing)}. "
            f"Available columns: {sorted(schema.keys())}",
        )

    # Check for time column (flexible)
    time_cols = {"timestamp", "event_time", "date", "datetime"}
    has_time = any(col in schema for col in time_cols)
    if not has_time and not allow_flexible_time:
        raise DataSchemaError(
            f"No time column found. Expected one of: {sorted(time_cols)}. "
            f"Available columns: {sorted(schema.keys())}",
        )

    # Check for asset_id if required
    if require_asset_id and "asset_id" not in schema:
        raise DataSchemaError(
            "Missing required 'asset_id' column. "
            f"Available columns: {sorted(schema.keys())}. "
            "Set require_asset_id=False if working with single-asset data.",
        )

    # Validate OHLCV column types (must be numeric)
    for col in ohlcv_cols:
        if col in schema:
            dtype = schema[col]
            if dtype not in [
                pl.Float32,
                pl.Float64,
                pl.Int32,
                pl.Int64,
                pl.UInt32,
                pl.UInt64,
            ]:
                raise DataSchemaError(
                    f"Column '{col}' must be numeric, got {dtype}. "
                    "OHLCV columns must be numeric types (float or int).",
                )

    # Validate time column type if present
    for time_col in time_cols:
        if time_col in schema:
            dtype = schema[time_col]
            if dtype not in [pl.Datetime, pl.Date]:
                raise DataSchemaError(
                    f"Time column '{time_col}' must be Datetime or Date, got {dtype}",
                )
            break  # Only check first found time column

    # Validate data integrity (only for materialized DataFrames)
    if isinstance(df, pl.DataFrame) and len(df) > 0:
        # Check high >= low
        try:
            invalid_hl = df.filter(pl.col("high") < pl.col("low"))
            if len(invalid_hl) > 0:
                raise DataSchemaError(
                    f"Found {len(invalid_hl)} rows where high < low. "
                    "This violates OHLCV constraints. First invalid row: "
                    f"{invalid_hl.head(1).to_dict(as_series=False)}",
                )
        except Exception as e:
            if isinstance(e, DataSchemaError):
                raise
            # If comparison fails (e.g., due to nulls), just warn

        # Check for negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                try:
                    negative = df.filter(pl.col(col) < 0)
                    if len(negative) > 0:
                        raise DataSchemaError(
                            f"Found {len(negative)} rows with negative prices in '{col}'. "
                            f"First invalid row: {negative.head(1).to_dict(as_series=False)}",
                        )
                except Exception as e:
                    if isinstance(e, DataSchemaError):
                        raise


__all__ = [
    "EXTENDED_OHLCV_SCHEMA",
    "FEATURE_SCHEMA",
    "LABELED_DATA_SCHEMA",
    "OHLCV_SCHEMA",
    "validate_ohlcv_schema",
    "validate_schema",
]
