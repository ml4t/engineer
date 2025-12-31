"""Coverage tests for math features (MAX, MIN, SUM)."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.math.max import max_numba, maximum
from ml4t.engineer.features.math.min import min_numba, minimum
from ml4t.engineer.features.math.sum import sum_numba, summation


class TestMaxNumba:
    """Tests for max_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic rolling maximum."""
        close = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0])
        result = max_numba(close, timeperiod=3)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Expected: [nan, nan, 3, 5, 5, 7, 7, 8]
        assert result[2] == 3.0
        assert result[3] == 5.0
        assert result[4] == 5.0
        assert result[5] == 7.0
        assert result[6] == 7.0
        assert result[7] == 8.0

    def test_insufficient_data(self):
        """Test with data shorter than period."""
        close = np.array([1.0, 2.0])
        result = max_numba(close, timeperiod=5)

        assert all(np.isnan(result))

    def test_constant_values(self):
        """Test with constant values."""
        close = np.array([5.0] * 10)
        result = max_numba(close, timeperiod=3)

        valid = result[~np.isnan(result)]
        assert all(v == 5.0 for v in valid)

    def test_decreasing_values(self):
        """Test with decreasing values."""
        close = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        result = max_numba(close, timeperiod=3)

        # Max in decreasing sequence is the first value in window
        assert result[2] == 10.0
        assert result[3] == 9.0
        assert result[4] == 8.0

    def test_increasing_values(self):
        """Test with increasing values."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = max_numba(close, timeperiod=3)

        # Max in increasing sequence is the last value in window
        assert result[2] == 3.0
        assert result[3] == 4.0
        assert result[4] == 5.0
        assert result[5] == 6.0


class TestMinNumba:
    """Tests for min_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic rolling minimum."""
        close = np.array([5.0, 3.0, 4.0, 1.0, 2.0, 0.0, 2.0, 1.0])
        result = min_numba(close, timeperiod=3)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Check valid values
        assert result[2] == 3.0  # min(5, 3, 4)
        assert result[3] == 1.0  # min(3, 4, 1)
        assert result[4] == 1.0  # min(4, 1, 2)
        assert result[5] == 0.0  # min(1, 2, 0)

    def test_insufficient_data(self):
        """Test with data shorter than period."""
        close = np.array([1.0, 2.0])
        result = min_numba(close, timeperiod=5)

        assert all(np.isnan(result))

    def test_constant_values(self):
        """Test with constant values."""
        close = np.array([5.0] * 10)
        result = min_numba(close, timeperiod=3)

        valid = result[~np.isnan(result)]
        assert all(v == 5.0 for v in valid)


class TestSumNumba:
    """Tests for sum_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic rolling sum."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sum_numba(close, timeperiod=3)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Check valid values
        assert result[2] == 6.0  # 1 + 2 + 3
        assert result[3] == 9.0  # 2 + 3 + 4
        assert result[4] == 12.0  # 3 + 4 + 5

    def test_insufficient_data(self):
        """Test with data shorter than period."""
        close = np.array([1.0, 2.0])
        result = sum_numba(close, timeperiod=5)

        assert all(np.isnan(result))


class TestMaximumWrapper:
    """Tests for maximum wrapper function."""

    def test_numpy_array_auto(self):
        """Test with numpy array using auto implementation."""
        close = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = maximum(close, timeperiod=3, implementation="auto")

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_numpy_array_numba(self):
        """Test with numpy array forcing Numba."""
        close = np.array([1.0, 3.0, 2.0, 5.0, 4.0] * 100)
        result = maximum(close, timeperiod=3, implementation="numba")

        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression."""
        result = maximum("close", timeperiod=3)

        assert isinstance(result, pl.Expr)

    def test_polars_column_string(self):
        """Test with column name string."""
        df = pl.DataFrame({"close": [1.0, 3.0, 2.0, 5.0, 4.0]})
        result = df.select(maximum("close", timeperiod=3).alias("max"))

        assert "max" in result.columns

    def test_invalid_timeperiod(self):
        """Test with invalid timeperiod."""
        close = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            maximum(close, timeperiod=0)

    def test_short_data(self):
        """Test with data shorter than period."""
        close = np.array([1.0, 2.0])
        result = maximum(close, timeperiod=5)

        assert all(np.isnan(result))


class TestMinimumWrapper:
    """Tests for minimum wrapper function."""

    def test_numpy_array_auto(self):
        """Test with numpy array using auto implementation."""
        close = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        result = minimum(close, timeperiod=3, implementation="auto")

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_numpy_array_numba(self):
        """Test with numpy array forcing Numba."""
        close = np.array([5.0, 3.0, 4.0, 1.0, 2.0] * 100)
        result = minimum(close, timeperiod=3, implementation="numba")

        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression."""
        result = minimum("close", timeperiod=3)

        assert isinstance(result, pl.Expr)

    def test_invalid_timeperiod(self):
        """Test with invalid timeperiod."""
        close = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            minimum(close, timeperiod=0)


class TestSummationWrapper:
    """Tests for summation wrapper function."""

    def test_numpy_array_auto(self):
        """Test with numpy array using auto implementation."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = summation(close, timeperiod=3, implementation="auto")

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_numpy_array_numba(self):
        """Test with numpy array forcing Numba."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        result = summation(close, timeperiod=3, implementation="numba")

        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression."""
        result = summation("close", timeperiod=3)

        assert isinstance(result, pl.Expr)

    def test_invalid_timeperiod(self):
        """Test with invalid timeperiod."""
        close = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            summation(close, timeperiod=0)


class TestMathAccuracy:
    """Accuracy tests for math functions."""

    def test_max_matches_numpy(self):
        """Test MAX matches numpy rolling max."""
        np.random.seed(42)
        close = np.random.randn(100)

        result = maximum(close, timeperiod=5)

        # Compare with numpy sliding window
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(close, window_shape=5)
        expected = np.full(len(close), np.nan)
        expected[4:] = np.max(windows, axis=1)

        valid = ~np.isnan(expected)
        np.testing.assert_allclose(result[valid], expected[valid])

    def test_min_matches_numpy(self):
        """Test MIN matches numpy rolling min."""
        np.random.seed(42)
        close = np.random.randn(100)

        result = minimum(close, timeperiod=5)

        # Compare with numpy sliding window
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(close, window_shape=5)
        expected = np.full(len(close), np.nan)
        expected[4:] = np.min(windows, axis=1)

        valid = ~np.isnan(expected)
        np.testing.assert_allclose(result[valid], expected[valid])

    def test_sum_matches_numpy(self):
        """Test SUM matches numpy rolling sum."""
        np.random.seed(42)
        close = np.random.randn(100)

        result = summation(close, timeperiod=5)

        # Compare with numpy sliding window
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(close, window_shape=5)
        expected = np.full(len(close), np.nan)
        expected[4:] = np.sum(windows, axis=1)

        valid = ~np.isnan(expected)
        np.testing.assert_allclose(result[valid], expected[valid])
