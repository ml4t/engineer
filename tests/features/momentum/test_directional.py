"""Tests for Directional Movement indicators (PLUS_DI, MINUS_DI, DX)."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.directional import dx, minus_di, plus_di


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


class TestPlusDI:
    """Tests for Plus Directional Indicator (+DI)."""

    def test_computes_successfully_numpy(self):
        """Test +DI computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = plus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_expr(self, ohlcv_data):
        """Test +DI computes successfully with Polars expressions."""
        result = ohlcv_data.select(plus_di("high", "low", "close", timeperiod=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_output_range(self):
        """Test +DI output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = plus_di(high, low, close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)

    def test_uptrend_high_values(self):
        """Test +DI is high during uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        result = plus_di(high, low, close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should be relatively high
            assert np.median(valid_values[-10:]) > 20


class TestMinusDI:
    """Tests for Minus Directional Indicator (-DI)."""

    def test_computes_successfully_numpy(self):
        """Test -DI computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = minus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_expr(self, ohlcv_data):
        """Test -DI computes successfully with Polars expressions."""
        result = ohlcv_data.select(minus_di("high", "low", "close", timeperiod=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_output_range(self):
        """Test -DI output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = minus_di(high, low, close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)

    def test_downtrend_high_values(self):
        """Test -DI is high during downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        result = minus_di(high, low, close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should be relatively high
            assert np.median(valid_values[-10:]) > 20


class TestDX:
    """Tests for Directional Movement Index (DX)."""

    def test_computes_successfully_numpy(self):
        """Test DX computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = dx(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_expr(self, ohlcv_data):
        """Test DX computes successfully with Polars expressions."""
        result = ohlcv_data.select(dx("high", "low", "close", timeperiod=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_output_range(self):
        """Test DX output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = dx(high, low, close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)


class TestEdgeCases:
    """Edge case tests for all indicators."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty = np.array([])

        result_plus = plus_di(empty, empty, empty, timeperiod=14)
        result_minus = minus_di(empty, empty, empty, timeperiod=14)
        result_dx = dx(empty, empty, empty, timeperiod=14)

        assert len(result_plus) == 0
        assert len(result_minus) == 0
        assert len(result_dx) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])

        result_plus = plus_di(single + 1, single - 1, single, timeperiod=14)
        result_minus = minus_di(single + 1, single - 1, single, timeperiod=14)
        result_dx = dx(single + 1, single - 1, single, timeperiod=14)

        assert len(result_plus) == 1
        assert len(result_minus) == 1
        assert len(result_dx) == 1
        assert np.isnan(result_plus[0])
        assert np.isnan(result_minus[0])
        assert np.isnan(result_dx[0])

    def test_constant_values(self):
        """Test with constant prices."""
        const = np.full(100, 100.0)
        high = const + 1
        low = const - 1

        result_plus = plus_di(high, low, const, timeperiod=14)
        result_minus = minus_di(high, low, const, timeperiod=14)
        result_dx = dx(high, low, const, timeperiod=14)

        # Should handle constant prices gracefully
        assert result_plus is not None
        assert result_minus is not None
        assert result_dx is not None

    def test_insufficient_data(self):
        """Test with insufficient data."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])

        result_plus = plus_di(high, low, close, timeperiod=14)
        result_minus = minus_di(high, low, close, timeperiod=14)
        result_dx = dx(high, low, close, timeperiod=14)

        # Should return all NaN
        assert np.all(np.isnan(result_plus))
        assert np.all(np.isnan(result_minus))
        assert np.all(np.isnan(result_dx))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_period_zero(self):
        """Test rejection of zero period."""
        high = np.random.randn(100) + 100
        low = high - 1
        close = high - 0.5

        with pytest.raises((ValueError, Exception)):
            plus_di(high, low, close, timeperiod=0)

        with pytest.raises((ValueError, Exception)):
            minus_di(high, low, close, timeperiod=0)

        with pytest.raises((ValueError, Exception)):
            dx(high, low, close, timeperiod=0)

    def test_invalid_period_negative(self):
        """Test rejection of negative period."""
        high = np.random.randn(100) + 100
        low = high - 1
        close = high - 0.5

        with pytest.raises((ValueError, Exception)):
            plus_di(high, low, close, timeperiod=-1)

        with pytest.raises((ValueError, Exception)):
            minus_di(high, low, close, timeperiod=-1)

        with pytest.raises((ValueError, Exception)):
            dx(high, low, close, timeperiod=-1)

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        high = np.random.randn(100) + 100
        low = np.random.randn(50) + 99  # Different length
        close = np.random.randn(100) + 100

        with pytest.raises((ValueError, IndexError)):
            plus_di(high, low, close, timeperiod=14)

        with pytest.raises((ValueError, IndexError)):
            minus_di(high, low, close, timeperiod=14)

        with pytest.raises((ValueError, IndexError)):
            dx(high, low, close, timeperiod=14)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_dx_from_plus_minus_di(self):
        """Test that DX is calculated from +DI and -DI."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        pdi = plus_di(high, low, close, timeperiod=14)
        mdi = minus_di(high, low, close, timeperiod=14)
        dx_result = dx(high, low, close, timeperiod=14)

        # DX = 100 * |+DI - -DI| / (+DI + -DI)
        valid_mask = ~np.isnan(pdi) & ~np.isnan(mdi) & ~np.isnan(dx_result)
        if np.sum(valid_mask) > 0:
            sum_di = pdi[valid_mask] + mdi[valid_mask]
            # Only check where sum is not zero
            nonzero_mask = sum_di != 0
            if np.sum(nonzero_mask) > 0:
                expected_dx = (
                    100.0
                    * np.abs(pdi[valid_mask][nonzero_mask] - mdi[valid_mask][nonzero_mask])
                    / sum_di[nonzero_mask]
                )
                actual_dx = dx_result[valid_mask][nonzero_mask]
                assert np.allclose(expected_dx, actual_dx, rtol=0.01)

    def test_uptrend_plus_di_greater(self):
        """Test that +DI > -DI during uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        pdi = plus_di(high, low, close, timeperiod=14)
        mdi = minus_di(high, low, close, timeperiod=14)

        valid_mask = ~np.isnan(pdi) & ~np.isnan(mdi)
        if np.sum(valid_mask) > 10:
            # In uptrend, +DI should generally be higher
            # Check last values
            last_indices = np.where(valid_mask)[0][-10:]
            plus_higher_count = np.sum(pdi[last_indices] > mdi[last_indices])
            assert plus_higher_count > 5  # Most should have +DI > -DI

    def test_downtrend_minus_di_greater(self):
        """Test that -DI > +DI during downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        pdi = plus_di(high, low, close, timeperiod=14)
        mdi = minus_di(high, low, close, timeperiod=14)

        valid_mask = ~np.isnan(pdi) & ~np.isnan(mdi)
        if np.sum(valid_mask) > 10:
            # In downtrend, -DI should generally be higher
            # Check last values
            last_indices = np.where(valid_mask)[0][-10:]
            minus_higher_count = np.sum(mdi[last_indices] > pdi[last_indices])
            assert minus_higher_count > 5  # Most should have -DI > +DI


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        high = (np.random.randn(100) + 100).astype(np.float64)
        low = (high - 1).astype(np.float64)
        close = (high - 0.5).astype(np.float64)

        result_plus = plus_di(high, low, close, timeperiod=14)
        result_minus = minus_di(high, low, close, timeperiod=14)
        result_dx = dx(high, low, close, timeperiod=14)

        assert isinstance(result_plus, np.ndarray)
        assert isinstance(result_minus, np.ndarray)
        assert isinstance(result_dx, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        high = pl.Series([100.0 + i + 1 for i in range(100)])
        low = pl.Series([100.0 + i - 1 for i in range(100)])
        close = pl.Series([100.0 + i for i in range(100)])

        result_plus = plus_di(high, low, close, timeperiod=14)
        result_minus = minus_di(high, low, close, timeperiod=14)
        result_dx = dx(high, low, close, timeperiod=14)

        assert isinstance(result_plus, np.ndarray)
        assert isinstance(result_minus, np.ndarray)
        assert isinstance(result_dx, np.ndarray)

    def test_polars_expressions(self):
        """Test with Polars expressions."""
        df = pl.DataFrame(
            {
                "high": [100.0 + i + 1 for i in range(100)],
                "low": [100.0 + i - 1 for i in range(100)],
                "close": [100.0 + i for i in range(100)],
            }
        )

        result_plus = df.select(plus_di("high", "low", "close", timeperiod=14))
        result_minus = df.select(minus_di("high", "low", "close", timeperiod=14))
        result_dx = df.select(dx("high", "low", "close", timeperiod=14))

        assert isinstance(result_plus, pl.DataFrame)
        assert isinstance(result_minus, pl.DataFrame)
        assert isinstance(result_dx, pl.DataFrame)
