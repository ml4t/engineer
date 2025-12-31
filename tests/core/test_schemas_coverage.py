"""
Coverage tests for core schemas module.

Tests validation functions and schema definitions.
"""

import polars as pl
import pytest

from ml4t.engineer.core.schemas import (
    EXTENDED_OHLCV_SCHEMA,
    FEATURE_SCHEMA,
    LABELED_DATA_SCHEMA,
    OHLCV_SCHEMA,
    validate_ohlcv_schema,
    validate_schema,
)


class TestSchemaDefinitions:
    """Tests for schema constants."""

    def test_ohlcv_schema_has_required_columns(self):
        """Test OHLCV schema has all required columns."""
        required = {"event_time", "asset_id", "open", "high", "low", "close", "volume"}
        assert required == set(OHLCV_SCHEMA.keys())

    def test_extended_ohlcv_schema_extends_base(self):
        """Test extended schema includes base schema."""
        for col in OHLCV_SCHEMA:
            assert col in EXTENDED_OHLCV_SCHEMA

    def test_extended_ohlcv_schema_has_extra_columns(self):
        """Test extended schema has additional columns."""
        extra = {"vwap", "trade_count", "bid", "ask", "open_interest"}
        for col in extra:
            assert col in EXTENDED_OHLCV_SCHEMA

    def test_labeled_data_schema_has_required_columns(self):
        """Test labeled data schema has required columns."""
        required = {
            "event_time",
            "asset_id",
            "label",
            "label_time",
            "label_price",
            "label_return",
            "weight",
        }
        assert required == set(LABELED_DATA_SCHEMA.keys())

    def test_feature_schema_has_required_columns(self):
        """Test feature schema has required columns."""
        assert "event_time" in FEATURE_SCHEMA
        assert "asset_id" in FEATURE_SCHEMA


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_schema_valid(self):
        """Test validate_schema with valid data."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )
        schema = {"a": pl.Float64, "b": pl.Float64}

        # Should not raise
        validate_schema(df, schema)

    def test_validate_schema_missing_column(self):
        """Test validate_schema with missing column."""
        df = pl.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
            }
        )
        schema = {"a": pl.Float64, "b": pl.Float64}

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, schema)

    def test_validate_schema_wrong_type(self):
        """Test validate_schema with wrong column type."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],  # Int64 instead of Float64
            }
        )
        schema = {"a": pl.Float64}

        with pytest.raises(ValueError, match="has type"):
            validate_schema(df, schema)


class TestValidateOHLCVSchema:
    """Tests for validate_ohlcv_schema function."""

    @pytest.fixture
    def valid_ohlcv_df(self):
        """Create valid OHLCV DataFrame."""
        return pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 10),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000.0 + i * 100 for i in range(10)],
            }
        )

    def test_validate_ohlcv_valid(self, valid_ohlcv_df):
        """Test validate_ohlcv_schema with valid data."""
        # Should not raise
        validate_ohlcv_schema(valid_ohlcv_df)

    def test_validate_ohlcv_missing_column(self):
        """Test validate_ohlcv_schema with missing OHLCV column."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                # Missing close and volume
            }
        )

        with pytest.raises(DataSchemaError, match="Missing required OHLCV columns"):
            validate_ohlcv_schema(df)

    def test_validate_ohlcv_missing_asset_id(self):
        """Test validate_ohlcv_schema with missing asset_id."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        with pytest.raises(DataSchemaError, match="Missing required 'asset_id'"):
            validate_ohlcv_schema(df, require_asset_id=True)

    def test_validate_ohlcv_no_asset_id_optional(self):
        """Test validate_ohlcv_schema with asset_id optional."""
        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        # Should not raise when asset_id not required
        validate_ohlcv_schema(df, require_asset_id=False)

    def test_validate_ohlcv_wrong_column_type(self):
        """Test validate_ohlcv_schema with wrong column type."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": ["a", "b", "c", "d", "e"],  # String instead of numeric
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        with pytest.raises(DataSchemaError, match="must be numeric"):
            validate_ohlcv_schema(df)

    def test_validate_ohlcv_high_less_than_low(self):
        """Test validate_ohlcv_schema with high < low."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "high": [99.0] * 5,  # High less than low!
                "low": [101.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        with pytest.raises(DataSchemaError, match="high < low"):
            validate_ohlcv_schema(df)

    def test_validate_ohlcv_negative_prices(self):
        """Test validate_ohlcv_schema with negative prices."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [-100.0, 100.0, 100.0, 100.0, 100.0],  # Negative price
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        with pytest.raises(DataSchemaError, match="negative prices"):
            validate_ohlcv_schema(df)

    def test_validate_ohlcv_lazyframe(self, valid_ohlcv_df):
        """Test validate_ohlcv_schema with LazyFrame."""
        lazy_df = valid_ohlcv_df.lazy()

        # Should not raise (but can't check data integrity on LazyFrame)
        validate_ohlcv_schema(lazy_df)

    def test_validate_ohlcv_alternative_time_column(self):
        """Test validate_ohlcv_schema with alternative time column names."""
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        # Should accept "timestamp" as valid time column
        validate_ohlcv_schema(df)

    def test_validate_ohlcv_date_column(self):
        """Test validate_ohlcv_schema with date column."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(
                    start=pl.date(2023, 1, 1),
                    end=pl.date(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        # Should accept "date" as valid time column
        validate_ohlcv_schema(df)

    def test_validate_ohlcv_empty_df(self):
        """Test validate_ohlcv_schema with empty DataFrame."""
        df = pl.DataFrame(
            {
                "event_time": pl.Series([], dtype=pl.Datetime("ns")),
                "asset_id": pl.Series([], dtype=pl.Utf8),
                "open": pl.Series([], dtype=pl.Float64),
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )

        # Should not raise for empty DataFrame
        validate_ohlcv_schema(df)

    def test_validate_ohlcv_int_columns(self):
        """Test validate_ohlcv_schema with integer OHLCV columns."""
        df = pl.DataFrame(
            {
                "event_time": pl.datetime_range(
                    start=pl.datetime(2023, 1, 1),
                    end=pl.datetime(2023, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_id": ["AAPL"] * 5,
                "open": [100, 101, 102, 103, 104],  # Int64
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        # Should accept integer columns
        validate_ohlcv_schema(df)

    def test_validate_ohlcv_wrong_time_type(self):
        """Test validate_ohlcv_schema with wrong time column type."""
        from ml4t.engineer.core.exceptions import DataSchemaError

        df = pl.DataFrame(
            {
                "event_time": [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                ],  # String
                "asset_id": ["AAPL"] * 5,
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000.0] * 5,
            }
        )

        with pytest.raises(DataSchemaError, match="must be Datetime or Date"):
            validate_ohlcv_schema(df)
