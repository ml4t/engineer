"""Tests for helper functions."""

from datetime import datetime

import polars as pl
import pytest

from ml4t.engineer.features.utils.helpers import (
    add_indicators,
    get_trading_session_filter,
    validate_ohlcv_columns,
)


class TestAddIndicators:
    """Tests for add_indicators function."""

    def test_add_single_indicator(self):
        """Test adding a single indicator."""
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})

        indicators = {"sma_3": pl.col("close").rolling_mean(3)}

        result = add_indicators(df, indicators)

        assert "sma_3" in result.columns
        assert len(result) == len(df)

    def test_add_multiple_indicators(self):
        """Test adding multiple indicators at once."""
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})

        indicators = {
            "sma_2": pl.col("close").rolling_mean(2),
            "std_2": pl.col("close").rolling_std(2),
            "diff": pl.col("close").diff(),
        }

        result = add_indicators(df, indicators)

        assert "sma_2" in result.columns
        assert "std_2" in result.columns
        assert "diff" in result.columns
        assert len(result) == len(df)

    def test_add_empty_indicators(self):
        """Test adding no indicators (empty dict)."""
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})

        result = add_indicators(df, {})

        assert result.columns == df.columns
        assert len(result) == len(df)

    def test_preserves_original_columns(self):
        """Test that original columns are preserved."""
        df = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [1000, 1100, 1200],
            }
        )

        indicators = {"sma": pl.col("close").rolling_mean(2)}

        result = add_indicators(df, indicators)

        assert "close" in result.columns
        assert "volume" in result.columns
        assert "sma" in result.columns


class TestValidateOhlcvColumns:
    """Tests for validate_ohlcv_columns function."""

    def test_all_columns_present(self):
        """Test validation passes when all columns are present."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        # Should not raise
        validate_ohlcv_columns(df, ["open", "high", "low", "close", "volume"])

    def test_subset_of_columns(self):
        """Test validation with subset of columns."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
            }
        )

        # Should not raise - only requiring close
        validate_ohlcv_columns(df, ["close"])

    def test_missing_single_column(self):
        """Test validation fails when single column is missing."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "close": [100.5],
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_columns(df, ["open", "high", "low", "close"])

    def test_missing_multiple_columns(self):
        """Test validation fails when multiple columns are missing."""
        df = pl.DataFrame({"close": [100.5]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_columns(df, ["open", "high", "low", "close", "volume"])

    def test_error_message_contains_missing_columns(self):
        """Test error message includes missing column names."""
        df = pl.DataFrame({"close": [100.5], "volume": [1000]})

        with pytest.raises(ValueError, match="open") as exc_info:
            validate_ohlcv_columns(df, ["open", "high", "close"])

        # Verify error message contains available columns
        assert "close" in str(exc_info.value)
        assert "volume" in str(exc_info.value)


class TestGetTradingSessionFilter:
    """Tests for get_trading_session_filter function."""

    def test_default_trading_hours(self):
        """Test filter with default trading hours (9:30 to 16:00)."""
        df = pl.DataFrame(
            {
                "datetime": [
                    datetime(2024, 1, 1, 9, 0),  # Before open
                    datetime(2024, 1, 1, 9, 30),  # At open
                    datetime(2024, 1, 1, 12, 0),  # During session
                    datetime(2024, 1, 1, 16, 0),  # At close
                    datetime(2024, 1, 1, 17, 0),  # After close
                ],
                "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        filter_expr = get_trading_session_filter()
        result = df.filter(filter_expr)

        assert len(result) == 3  # 9:30, 12:00, 16:00

    def test_custom_trading_hours(self):
        """Test filter with custom trading hours."""
        df = pl.DataFrame(
            {
                "datetime": [
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 1, 15, 0),
                ],
                "price": [100.0, 101.0, 102.0, 103.0],
            }
        )

        filter_expr = get_trading_session_filter(start_time="10:00", end_time="14:00")
        result = df.filter(filter_expr)

        assert len(result) == 2  # 10:00 and 14:00

    def test_filter_all_within_session(self):
        """Test when all data is within session."""
        df = pl.DataFrame(
            {
                "datetime": [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 11, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "price": [100.0, 101.0, 102.0],
            }
        )

        filter_expr = get_trading_session_filter()
        result = df.filter(filter_expr)

        assert len(result) == 3

    def test_filter_all_outside_session(self):
        """Test when all data is outside session."""
        df = pl.DataFrame(
            {
                "datetime": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 18, 0),
                    datetime(2024, 1, 1, 20, 0),
                ],
                "price": [100.0, 101.0, 102.0],
            }
        )

        filter_expr = get_trading_session_filter()
        result = df.filter(filter_expr)

        assert len(result) == 0
