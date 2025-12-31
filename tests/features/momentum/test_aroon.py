"""Tests for Aroon indicators (AROON, AROONOSC)."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.aroon import aroon, aroonosc


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    return pl.DataFrame(
        {
            "open": close - np.random.rand(n) * 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestAroon:
    """Tests for Aroon indicator."""

    def test_aroon_computes_successfully_numpy(self):
        """Test Aroon computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)

        assert aroon_down is not None
        assert aroon_up is not None
        assert len(aroon_down) == n
        assert len(aroon_up) == n
        assert isinstance(aroon_down, np.ndarray)
        assert isinstance(aroon_up, np.ndarray)

    def test_aroon_computes_successfully_polars_expr(self, ohlcv_data):
        """Test Aroon computes successfully with Polars expressions."""
        result = ohlcv_data.select(aroon("high", "low", timeperiod=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_parameter_variations(self):
        """Test Aroon with different periods."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        down1, up1 = aroon(high, low, timeperiod=7)
        down2, up2 = aroon(high, low, timeperiod=21)

        # Different periods should give different results
        valid_mask = ~np.isnan(down1) & ~np.isnan(down2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(down1[valid_mask], down2[valid_mask], rtol=0.01)

    def test_output_range(self):
        """Test Aroon output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)

        # Check valid values are in range [0, 100]
        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]

        if len(valid_down) > 0:
            assert np.all(valid_down >= 0)
            assert np.all(valid_down <= 100)

        if len(valid_up) > 0:
            assert np.all(valid_up >= 0)
            assert np.all(valid_up <= 100)


class TestAroonOsc:
    """Tests for Aroon Oscillator."""

    def test_aroonosc_computes_successfully_numpy(self):
        """Test Aroon Oscillator computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = aroonosc(high, low, timeperiod=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_aroonosc_computes_successfully_polars_expr(self, ohlcv_data):
        """Test Aroon Oscillator computes successfully with Polars expressions."""
        result = ohlcv_data.select(aroonosc("high", "low", timeperiod=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_output_range(self):
        """Test Aroon Oscillator output is within -100 to 100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = aroonosc(high, low, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= -100)
            assert np.all(valid_values <= 100)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty = np.array([])

        aroon_down, aroon_up = aroon(empty, empty, timeperiod=14)
        assert len(aroon_down) == 0
        assert len(aroon_up) == 0

        osc = aroonosc(empty, empty, timeperiod=14)
        assert len(osc) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])

        aroon_down, aroon_up = aroon(single + 1, single - 1, timeperiod=14)
        assert len(aroon_down) == 1
        assert len(aroon_up) == 1
        assert np.isnan(aroon_down[0])
        assert np.isnan(aroon_up[0])

        osc = aroonosc(single + 1, single - 1, timeperiod=14)
        assert len(osc) == 1
        assert np.isnan(osc[0])

    def test_constant_values(self):
        """Test with constant prices."""
        high = np.full(100, 101.0)
        low = np.full(100, 99.0)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        assert aroon_down is not None
        assert aroon_up is not None

        osc = aroonosc(high, low, timeperiod=14)
        assert osc is not None

    def test_with_nans_in_data(self):
        """Test handling of NaN values in input."""
        high = np.array([101.0, 102.0, np.nan, 104.0, 105.0] * 20)
        low = np.array([99.0, 100.0, np.nan, 102.0, 103.0] * 20)

        aroon_down, aroon_up = aroon(high, low, timeperiod=5)
        assert aroon_down is not None
        assert aroon_up is not None

        osc = aroonosc(high, low, timeperiod=5)
        assert osc is not None

    def test_insufficient_data(self):
        """Test with insufficient data."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0, 100.0])

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        # Should return all NaN
        assert np.all(np.isnan(aroon_down))
        assert np.all(np.isnan(aroon_up))

        osc = aroonosc(high, low, timeperiod=14)
        assert np.all(np.isnan(osc))

    def test_exact_minimum_data(self):
        """Test with exactly minimum data needed."""
        # Need timeperiod + 1 values
        n = 15
        close = np.array([100.0 + i * 0.5 for i in range(n)])
        high = close + 1
        low = close - 1

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        # Should have first valid value at index 14
        assert not np.all(np.isnan(aroon_down))
        assert not np.all(np.isnan(aroon_up))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_period_zero(self):
        """Test rejection of zero period."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            aroon(high, low, timeperiod=0)

        with pytest.raises((ValueError, Exception)):
            aroonosc(high, low, timeperiod=0)

    def test_invalid_period_one(self):
        """Test rejection of period=1."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            aroon(high, low, timeperiod=1)

        with pytest.raises((ValueError, Exception)):
            aroonosc(high, low, timeperiod=1)

    def test_invalid_period_negative(self):
        """Test rejection of negative period."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            aroon(high, low, timeperiod=-1)

        with pytest.raises((ValueError, Exception)):
            aroonosc(high, low, timeperiod=-1)

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        high = np.random.randn(100) + 100
        low = np.random.randn(50) + 99  # Different length

        with pytest.raises((ValueError, IndexError)):
            aroon(high, low, timeperiod=14)

        with pytest.raises((ValueError, IndexError)):
            aroonosc(high, low, timeperiod=14)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_aroonosc_equals_difference(self):
        """Test that Aroon Osc = Aroon Up - Aroon Down."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        osc = aroonosc(high, low, timeperiod=14)

        # Where all are valid, osc should equal up - down
        valid_mask = ~np.isnan(aroon_down) & ~np.isnan(aroon_up) & ~np.isnan(osc)
        if np.sum(valid_mask) > 0:
            expected_osc = aroon_up[valid_mask] - aroon_down[valid_mask]
            actual_osc = osc[valid_mask]
            assert np.allclose(expected_osc, actual_osc, atol=1e-10)

    def test_uptrend_aroon_up_high(self):
        """Test Aroon Up is high during uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)

        valid_values = aroon_up[~np.isnan(aroon_up)]
        if len(valid_values) > 10:
            # Later values should be high (close to 100)
            assert np.median(valid_values[-10:]) > 70

    def test_downtrend_aroon_down_high(self):
        """Test Aroon Down is high during downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)

        valid_values = aroon_down[~np.isnan(aroon_down)]
        if len(valid_values) > 10:
            # Later values should be high (close to 100)
            assert np.median(valid_values[-10:]) > 70

    def test_positive_osc_in_uptrend(self):
        """Test Aroon Oscillator is positive in uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        osc = aroonosc(high, low, timeperiod=14)

        valid_values = osc[~np.isnan(osc)]
        if len(valid_values) > 10:
            # Most values should be positive in uptrend
            assert np.median(valid_values[-10:]) > 0

    def test_negative_osc_in_downtrend(self):
        """Test Aroon Oscillator is negative in downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        osc = aroonosc(high, low, timeperiod=14)

        valid_values = osc[~np.isnan(osc)]
        if len(valid_values) > 10:
            # Most values should be negative in downtrend
            assert np.median(valid_values[-10:]) < 0


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        high = (np.random.randn(100) + 100).astype(np.float64)
        low = (high - 1).astype(np.float64)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        assert isinstance(aroon_down, np.ndarray)
        assert isinstance(aroon_up, np.ndarray)

        osc = aroonosc(high, low, timeperiod=14)
        assert isinstance(osc, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        high = pl.Series([100.0 + i + 1 for i in range(100)])
        low = pl.Series([100.0 + i - 1 for i in range(100)])

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        assert isinstance(aroon_down, np.ndarray)
        assert isinstance(aroon_up, np.ndarray)

        osc = aroonosc(high, low, timeperiod=14)
        assert isinstance(osc, np.ndarray)

    def test_polars_expressions(self):
        """Test with Polars expressions."""
        df = pl.DataFrame(
            {
                "high": [100.0 + i + 1 for i in range(100)],
                "low": [100.0 + i - 1 for i in range(100)],
            }
        )

        # Aroon returns struct with Polars expressions
        result_aroon = df.select(aroon("high", "low", timeperiod=14))
        assert isinstance(result_aroon, pl.DataFrame)

        result_osc = df.select(aroonosc("high", "low", timeperiod=14))
        assert isinstance(result_osc, pl.DataFrame)


class TestPeriodSizes:
    """Test various period sizes."""

    def test_small_period(self):
        """Test with small period (5)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=5)
        osc = aroonosc(high, low, timeperiod=5)

        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]
        valid_osc = osc[~np.isnan(osc)]

        assert len(valid_down) > 0
        assert len(valid_up) > 0
        assert len(valid_osc) > 0

    def test_medium_period(self):
        """Test with medium period (14 - default)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=14)
        osc = aroonosc(high, low, timeperiod=14)

        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]
        valid_osc = osc[~np.isnan(osc)]

        assert len(valid_down) > 0
        assert len(valid_up) > 0
        assert len(valid_osc) > 0

    def test_large_period(self):
        """Test with large period (30)."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        aroon_down, aroon_up = aroon(high, low, timeperiod=30)
        osc = aroonosc(high, low, timeperiod=30)

        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]
        valid_osc = osc[~np.isnan(osc)]

        assert len(valid_down) > 0
        assert len(valid_up) > 0
        assert len(valid_osc) > 0
