"""Tests for core validation utilities."""

import polars as pl
import pytest

from ml4t.engineer.core.validation import (
    validate_column_exists,
    validate_lag,
    validate_list_length,
    validate_percentage,
    validate_period,
    validate_positive,
    validate_probability,
    validate_threshold,
    validate_window,
)


class TestValidateWindow:
    """Tests for validate_window function."""

    def test_valid_window(self):
        """Test valid window values."""
        # Should not raise
        validate_window(10)
        validate_window(1)
        validate_window(100)

    def test_window_below_minimum(self):
        """Test window below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_window(0)

        with pytest.raises(ValueError, match="must be at least 5"):
            validate_window(4, min_window=5)

    def test_window_type_error(self):
        """Test non-integer window raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_window(10.5)  # type: ignore

        with pytest.raises(TypeError, match="must be an integer"):
            validate_window("10")  # type: ignore

    def test_custom_name_in_error(self):
        """Test custom parameter name appears in error."""
        with pytest.raises(ValueError, match="period"):
            validate_window(0, name="period")


class TestValidatePeriod:
    """Tests for validate_period function (alias for validate_window)."""

    def test_valid_period(self):
        """Test valid period values."""
        # Should not raise
        validate_period(14)
        validate_period(1)

    def test_period_below_minimum(self):
        """Test period below minimum raises ValueError."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_period(0)

    def test_period_type_error(self):
        """Test non-integer period raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_period(14.0)  # type: ignore


class TestValidateThreshold:
    """Tests for validate_threshold function."""

    def test_valid_threshold(self):
        """Test valid threshold values."""
        validate_threshold(0.5)
        validate_threshold(0.0)
        validate_threshold(1.0)

    def test_threshold_out_of_range(self):
        """Test threshold outside range raises ValueError."""
        with pytest.raises(ValueError, match="must be between"):
            validate_threshold(-0.1)

        with pytest.raises(ValueError, match="must be between"):
            validate_threshold(1.5)

    def test_custom_range(self):
        """Test threshold with custom range."""
        validate_threshold(50, min_val=0, max_val=100)

        with pytest.raises(ValueError):
            validate_threshold(150, min_val=0, max_val=100)

    def test_threshold_type_error(self):
        """Test non-numeric threshold raises TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            validate_threshold("0.5")  # type: ignore

        with pytest.raises(TypeError, match="must be numeric"):
            validate_threshold(None)  # type: ignore


class TestValidateLag:
    """Tests for validate_lag function."""

    def test_valid_lag(self):
        """Test valid lag values."""
        validate_lag(0)
        validate_lag(1)
        validate_lag(100)

    def test_negative_lag(self):
        """Test negative lag raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_lag(-1)

    def test_lag_exceeds_max(self):
        """Test lag exceeding max raises ValueError."""
        with pytest.raises(ValueError, match="must not exceed"):
            validate_lag(10, max_lag=5)

    def test_lag_at_max(self):
        """Test lag at max is valid."""
        validate_lag(5, max_lag=5)

    def test_lag_type_error(self):
        """Test non-integer lag raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_lag(1.5)  # type: ignore


class TestValidateListLength:
    """Tests for validate_list_length function."""

    def test_valid_list(self):
        """Test valid list."""
        validate_list_length([1, 2, 3])
        validate_list_length([1])

    def test_empty_list(self):
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="must have at least"):
            validate_list_length([])

    def test_list_below_min(self):
        """Test list below minimum length raises ValueError."""
        with pytest.raises(ValueError, match="must have at least 3"):
            validate_list_length([1, 2], min_length=3)

    def test_list_exceeds_max(self):
        """Test list exceeding max length raises ValueError."""
        with pytest.raises(ValueError, match="must have at most"):
            validate_list_length([1, 2, 3, 4, 5], max_length=3)

    def test_list_at_max(self):
        """Test list at max length is valid."""
        validate_list_length([1, 2, 3], max_length=3)

    def test_list_type_error(self):
        """Test non-list raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            validate_list_length((1, 2, 3))  # type: ignore

        with pytest.raises(TypeError, match="must be a list"):
            validate_list_length("abc")  # type: ignore


class TestValidateColumnExists:
    """Tests for validate_column_exists function."""

    def test_column_exists(self):
        """Test column that exists."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        validate_column_exists(df, "a")
        validate_column_exists(df, "b")

    def test_column_not_exists(self):
        """Test column that doesn't exist raises ValueError."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="Column 'c' not found"):
            validate_column_exists(df, "c")

    def test_error_shows_available_columns(self):
        """Test error message shows available columns."""
        df = pl.DataFrame({"close": [1], "volume": [100]})
        with pytest.raises(ValueError, match="close") as exc_info:
            validate_column_exists(df, "open")
        assert "volume" in str(exc_info.value)

    def test_many_columns_truncated(self):
        """Test many columns are truncated in error message."""
        df = pl.DataFrame({f"col_{i}": [i] for i in range(10)})
        with pytest.raises(ValueError, match="total") as exc_info:
            validate_column_exists(df, "missing")
        assert "10 total" in str(exc_info.value)


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_valid_positive(self):
        """Test valid positive values."""
        validate_positive(1)
        validate_positive(0.001)
        validate_positive(100)

    def test_zero_raises(self):
        """Test zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0)

    def test_negative_raises(self):
        """Test negative raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-1)

    def test_type_error(self):
        """Test non-numeric raises TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            validate_positive("1")  # type: ignore


class TestValidateProbability:
    """Tests for validate_probability function."""

    def test_valid_probability(self):
        """Test valid probability values."""
        validate_probability(0.0)
        validate_probability(0.5)
        validate_probability(1.0)

    def test_invalid_probability(self):
        """Test invalid probability raises ValueError."""
        with pytest.raises(ValueError):
            validate_probability(-0.1)

        with pytest.raises(ValueError):
            validate_probability(1.1)


class TestValidatePercentage:
    """Tests for validate_percentage function."""

    def test_valid_percentage(self):
        """Test valid percentage values."""
        validate_percentage(0)
        validate_percentage(50)
        validate_percentage(100)

    def test_invalid_percentage(self):
        """Test invalid percentage raises ValueError."""
        with pytest.raises(ValueError):
            validate_percentage(-1)

        with pytest.raises(ValueError):
            validate_percentage(101)
