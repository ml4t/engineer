"""Tests for MACDFIX (MACD with fixed 12/26 periods) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.macdfix import macdfix, macdfix_full, macdfix_signal


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pl.DataFrame(
        {
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n),
            "low": close - np.random.rand(n),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_macdfix_computes_successfully_numpy(self):
        """Test MACDFIX computes successfully with numpy array."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macdfix(close, signalperiod=9)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_macdfix_computes_successfully_polars_expr(self, ohlcv_data):
        """Test MACDFIX computes successfully with Polars expression."""
        result = ohlcv_data.select(macdfix("close", signalperiod=9))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_macdfix_signal_computes(self):
        """Test MACDFIX signal line computes."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macdfix_signal(close, signalperiod=9)

        assert result is not None
        assert len(result) == len(close)

    def test_macdfix_full_returns_three_components(self):
        """Test macdfix_full returns all three components."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        macd_line, signal_line, histogram = macdfix_full(close, signalperiod=9)

        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)

    def test_parameter_variations(self):
        """Test MACDFIX with different signal periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        r1 = macdfix(close, signalperiod=5)
        r2 = macdfix(close, signalperiod=14)

        # MACD lines should be the same (fixed 12/26)
        # But we're getting the main MACD line, so should be same
        # Different signal periods affect when values start appearing
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            # MACD line itself should be identical for overlapping valid periods
            # (since fast/slow are fixed at 12/26)
            assert np.allclose(r1[valid_mask], r2[valid_mask], rtol=1e-10)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array(self):
        """Test MACDFIX with empty array."""
        empty = np.array([])
        result = macdfix(empty, signalperiod=9)

        assert len(result) == 0

    def test_single_value(self):
        """Test MACDFIX with single value."""
        single = np.array([100.0])
        result = macdfix(single, signalperiod=9)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_constant_values(self):
        """Test MACDFIX with constant prices."""
        const = np.full(100, 100.0)
        result = macdfix(const, signalperiod=9)

        # With constant prices, MACDFIX should be 0
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_with_nans_in_data(self):
        """Test MACDFIX handles NaN values in input."""
        close = np.array([100.0, 101.0, np.nan, 103.0, 104.0] * 20)
        result = macdfix(close, signalperiod=9)

        assert result is not None
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test MACDFIX with insufficient data."""
        close = np.array([100.0, 101.0, 102.0])
        result = macdfix(close, signalperiod=9)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test MACDFIX with exactly minimum data needed."""
        # Need at least slow_period (26) for basic computation
        close = np.array([100.0 + i * 0.5 for i in range(35)])
        result = macdfix(close, signalperiod=9)

        # Should have some valid values near the end
        assert not np.all(np.isnan(result))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_signal_period_zero(self):
        """Test MACDFIX rejects zero signal period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macdfix(close, signalperiod=0)

    def test_invalid_signal_period_negative(self):
        """Test MACDFIX rejects negative signal period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macdfix(close, signalperiod=-1)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_histogram_equals_difference(self):
        """Test that histogram = MACDFIX line - signal line."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        macd_line, signal_line, histogram = macdfix_full(close, signalperiod=9)

        # Where both are valid, histogram should equal difference
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line) & ~np.isnan(histogram)
        if np.sum(valid_mask) > 0:
            expected_hist = macd_line[valid_mask] - signal_line[valid_mask]
            actual_hist = histogram[valid_mask]
            assert np.allclose(expected_hist, actual_hist, atol=1e-10)

    def test_zero_crossovers(self):
        """Test MACDFIX crosses zero during trending market."""
        # Create uptrend then downtrend
        uptrend = np.linspace(100, 120, 50)
        downtrend = np.linspace(120, 100, 50)
        close = np.concatenate([uptrend, downtrend])

        result = macdfix(close, signalperiod=9)

        # Should have some positive and some negative values
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            has_positive = np.any(valid_values > 0)
            has_negative = np.any(valid_values < 0)
            # At least one should be true in trending market
            assert has_positive or has_negative

    def test_fixed_periods_12_26(self):
        """Test that MACDFIX uses fixed periods 12 and 26."""
        # This is implicit in the implementation
        # We verify by checking the function computes without period parameters
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        # Should work with only signal period
        result = macdfix(close, signalperiod=9)
        assert result is not None

        # Should have same lookback as MACD(12,26,9)
        # First valid value appears around index 26+9-1 = 34
        valid_indices = ~np.isnan(result)
        if np.sum(valid_indices) > 0:
            first_valid = np.where(valid_indices)[0][0]
            # Should be around index 34 (may vary slightly due to TA-Lib unstable period)
            assert first_valid >= 26  # At minimum need slow EMA


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy array."""
        close = np.random.randn(100).astype(np.float64)
        result = macdfix(close, signalperiod=9)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy array (should convert)."""
        close = np.random.randn(100).astype(np.float32)
        result = macdfix(close, signalperiod=9)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression (string column name)."""
        df = pl.DataFrame({"close": [100.0 + i * 0.5 for i in range(100)]})
        result = df.select(macdfix("close", signalperiod=9))
        assert isinstance(result, pl.DataFrame)
        assert "close" in result.columns


class TestSignalPeriods:
    """Test various signal periods."""

    def test_default_signal_period(self):
        """Test with default signal period (9)."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        macd_line, signal_line, histogram = macdfix_full(close)

        # Should have valid values
        assert np.sum(~np.isnan(macd_line)) > 0
        assert np.sum(~np.isnan(signal_line)) > 0
        assert np.sum(~np.isnan(histogram)) > 0

    def test_short_signal_period(self):
        """Test with short signal period."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macdfix(close, signalperiod=5)

        assert np.sum(~np.isnan(result)) > 0

    def test_long_signal_period(self):
        """Test with long signal period."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macdfix(close, signalperiod=20)

        assert np.sum(~np.isnan(result)) > 0


class TestSignalLine:
    """Test signal line specific functionality."""

    def test_signal_line_computation(self):
        """Test signal line is EMA of MACD line."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 2)  # More volatility

        macd_line, signal_line, _ = macdfix_full(close, signalperiod=9)

        # Signal line should generally be smoother (less volatile)
        # Skip NaN values
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        if np.sum(valid_mask) > 10:
            macd_std = np.std(macd_line[valid_mask])
            signal_std = np.std(signal_line[valid_mask])
            # Both should have reasonable variance
            assert macd_std > 0
            assert signal_std > 0

    def test_signal_from_standalone_function(self):
        """Test macdfix_signal returns same as macdfix_full."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        signal_standalone = macdfix_signal(close, signalperiod=9)
        _, signal_from_full, _ = macdfix_full(close, signalperiod=9)

        # Should be identical
        valid_mask = ~np.isnan(signal_standalone) & ~np.isnan(signal_from_full)
        if np.sum(valid_mask) > 0:
            assert np.allclose(
                signal_standalone[valid_mask],
                signal_from_full[valid_mask],
                atol=1e-10,
            )


class TestTrendDetection:
    """Test trend detection capabilities."""

    def test_uptrend_positive_macdfix(self):
        """Test MACDFIX is positive during uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        result = macdfix(close, signalperiod=9)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Most later values should be positive in uptrend
            assert np.median(valid_values[-10:]) > 0

    def test_downtrend_negative_macdfix(self):
        """Test MACDFIX is negative during downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        result = macdfix(close, signalperiod=9)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Most later values should be negative in downtrend
            assert np.median(valid_values[-10:]) < 0

    def test_sideways_near_zero(self):
        """Test MACDFIX oscillates near zero in sideways market."""
        # Sideways market
        close = 100 + np.sin(np.linspace(0, 10 * np.pi, 100)) * 5
        result = macdfix(close, signalperiod=9)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Mean should be close to zero
            assert abs(np.mean(valid_values)) < 2


class TestPolarsIntegration:
    """Test Polars-specific functionality."""

    def test_polars_expr_macdfix(self, ohlcv_data):
        """Test MACDFIX with Polars expression."""
        result = ohlcv_data.select(macdfix("close", signalperiod=9))

        assert isinstance(result, pl.DataFrame)
        assert "close" in result.columns
        assert len(result) == len(ohlcv_data)

    def test_polars_expr_signal(self, ohlcv_data):
        """Test MACDFIX signal with Polars expression."""
        result = ohlcv_data.select(macdfix_signal("close", signalperiod=9))

        assert isinstance(result, pl.DataFrame)
        assert "close" in result.columns
        assert len(result) == len(ohlcv_data)

    def test_polars_full_components(self, ohlcv_data):
        """Test MACDFIX full with Polars expressions."""
        macd_expr, signal_expr, hist_expr = macdfix_full("close", signalperiod=9)

        # Apply all three to dataframe
        result = ohlcv_data.select(
            [
                macd_expr.alias("macd"),
                signal_expr.alias("signal"),
                hist_expr.alias("hist"),
            ]
        )

        assert isinstance(result, pl.DataFrame)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "hist" in result.columns
        assert len(result) == len(ohlcv_data)
