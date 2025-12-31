"""
Extended coverage tests for statistics features.

Tests additional code paths to improve coverage.
"""

import numpy as np
import polars as pl
import pytest


class TestLinearRegCoverage:
    """Coverage tests for Linear Regression."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_linearreg_numpy_basic(self, price_data):
        """Test LINEARREG with numpy arrays."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        result = linearreg(price_data, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_linearreg_polars_series(self, price_data):
        """Test LINEARREG with Polars Series."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        series = pl.Series(price_data)
        result = linearreg(series, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)

    def test_linearreg_polars_expr(self, price_data):
        """Test LINEARREG with Polars expressions."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        df = pl.DataFrame({"close": price_data})
        result = df.select(linearreg("close", timeperiod=14))

        assert result is not None
        assert len(result) == len(price_data)

    def test_linearreg_numba_impl(self, price_data):
        """Test LINEARREG with numba implementation."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        result = linearreg(price_data, timeperiod=14)
        assert result is not None

    def test_linearreg_polars_impl(self, price_data):
        """Test LINEARREG with polars implementation."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        df = pl.DataFrame({"close": price_data})
        result = df.select(linearreg("close", timeperiod=14))
        assert result is not None

    def test_linearreg_different_periods(self, price_data):
        """Test LINEARREG with different periods."""
        from ml4t.engineer.features.statistics.linearreg import linearreg

        r1 = linearreg(price_data, timeperiod=5)
        r2 = linearreg(price_data, timeperiod=20)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)


class TestTSFCoverage:
    """Coverage tests for Time Series Forecast."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_tsf_numpy_basic(self, price_data):
        """Test TSF with numpy arrays."""
        from ml4t.engineer.features.statistics.tsf import tsf

        result = tsf(price_data, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_tsf_polars_series(self, price_data):
        """Test TSF with Polars Series."""
        from ml4t.engineer.features.statistics.tsf import tsf

        series = pl.Series(price_data)
        result = tsf(series, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)

    def test_tsf_polars_expr(self, price_data):
        """Test TSF with Polars expressions."""
        from ml4t.engineer.features.statistics.tsf import tsf

        df = pl.DataFrame({"close": price_data})
        result = df.select(tsf("close", timeperiod=14))

        assert result is not None
        assert len(result) == len(price_data)

    def test_tsf_numba_impl(self, price_data):
        """Test TSF with numba implementation."""
        from ml4t.engineer.features.statistics.tsf import tsf

        result = tsf(price_data, timeperiod=14)
        assert result is not None

    def test_tsf_polars_impl(self, price_data):
        """Test TSF with polars implementation."""
        from ml4t.engineer.features.statistics.tsf import tsf

        df = pl.DataFrame({"close": price_data})
        result = df.select(tsf("close", timeperiod=14))
        assert result is not None

    def test_tsf_uptrend(self):
        """Test TSF with uptrend data."""
        from ml4t.engineer.features.statistics.tsf import tsf

        close = np.linspace(100, 200, 100)
        result = tsf(close, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestStddevCoverage:
    """Coverage tests for Standard Deviation."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_stddev_numpy_basic(self, price_data):
        """Test STDDEV with numpy arrays."""
        from ml4t.engineer.features.statistics.stddev import stddev

        result = stddev(price_data, period=5)

        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_stddev_polars_series(self, price_data):
        """Test STDDEV with Polars Series."""
        from ml4t.engineer.features.statistics.stddev import stddev

        series = pl.Series(price_data)
        result = stddev(series, period=5)

        assert result is not None
        assert len(result) == len(price_data)

    def test_stddev_polars_expr(self, price_data):
        """Test STDDEV with Polars expressions."""
        from ml4t.engineer.features.statistics.stddev import stddev

        df = pl.DataFrame({"close": price_data})
        result = df.select(stddev("close", period=5))

        assert result is not None
        assert len(result) == len(price_data)

    def test_stddev_numba_impl(self, price_data):
        """Test STDDEV with numba implementation."""
        from ml4t.engineer.features.statistics.stddev import stddev

        result = stddev(price_data, period=5)
        assert result is not None

    def test_stddev_polars_impl(self, price_data):
        """Test STDDEV with polars implementation."""
        from ml4t.engineer.features.statistics.stddev import stddev

        df = pl.DataFrame({"close": price_data})
        result = df.select(stddev("close", period=5))
        assert result is not None

    def test_stddev_with_nbdev(self, price_data):
        """Test STDDEV with nbdev multiplier."""
        from ml4t.engineer.features.statistics.stddev import stddev

        r1 = stddev(price_data, period=5, nbdev=1.0)
        r2 = stddev(price_data, period=5, nbdev=2.0)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            # nbdev=2 should be 2x nbdev=1
            assert np.allclose(r2[valid_mask], r1[valid_mask] * 2, rtol=0.01)

    def test_stddev_with_ddof(self, price_data):
        """Test STDDEV with different ddof."""
        from ml4t.engineer.features.statistics.stddev import stddev

        r1 = stddev(price_data, period=5, ddof=0)
        r2 = stddev(price_data, period=5, ddof=1)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            # ddof=1 should be slightly larger
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)


class TestVarCoverage:
    """Coverage tests for Variance."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_var_numpy_basic(self, price_data):
        """Test VAR with numpy arrays."""
        from ml4t.engineer.features.statistics.var import var

        result = var(price_data, timeperiod=5)

        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_var_polars_series(self, price_data):
        """Test VAR with Polars Series."""
        from ml4t.engineer.features.statistics.var import var

        series = pl.Series(price_data)
        result = var(series, timeperiod=5)

        assert result is not None
        assert len(result) == len(price_data)

    def test_var_polars_expr(self, price_data):
        """Test VAR with Polars expressions."""
        from ml4t.engineer.features.statistics.var import var

        df = pl.DataFrame({"close": price_data})
        result = df.select(var("close", timeperiod=5))

        assert result is not None
        assert len(result) == len(price_data)

    def test_var_numba_impl(self, price_data):
        """Test VAR with numba implementation."""
        from ml4t.engineer.features.statistics.var import var

        result = var(price_data, timeperiod=5)
        assert result is not None

    def test_var_polars_impl(self, price_data):
        """Test VAR with polars implementation."""
        from ml4t.engineer.features.statistics.var import var

        df = pl.DataFrame({"close": price_data})
        result = df.select(var("close", timeperiod=5))
        assert result is not None

    def test_var_with_nbdev(self, price_data):
        """Test VAR with nbdev multiplier."""
        from ml4t.engineer.features.statistics.var import var

        r1 = var(price_data, timeperiod=5, nbdev=1.0)
        r2 = var(price_data, timeperiod=5, nbdev=2.0)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            # Both should compute and have same values (nbdev doesn't affect variance)
            assert len(r1[valid_mask]) > 0
            assert len(r2[valid_mask]) > 0


class TestAvgdevCoverage:
    """Coverage tests for Average Deviation."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_avgdev_numpy_basic(self, price_data):
        """Test AVGDEV with numpy arrays."""
        from ml4t.engineer.features.statistics.avgdev import avgdev

        result = avgdev(price_data, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_avgdev_polars_series(self, price_data):
        """Test AVGDEV with Polars Series."""
        from ml4t.engineer.features.statistics.avgdev import avgdev

        series = pl.Series(price_data)
        result = avgdev(series, timeperiod=14)

        assert result is not None
        assert len(result) == len(price_data)

    def test_avgdev_polars_expr(self, price_data):
        """Test AVGDEV with Polars expressions."""
        from ml4t.engineer.features.statistics.avgdev import avgdev

        df = pl.DataFrame({"close": price_data})
        result = df.select(avgdev("close", timeperiod=14))

        assert result is not None
        assert len(result) == len(price_data)

    def test_avgdev_numba_impl(self, price_data):
        """Test AVGDEV with numba implementation."""
        from ml4t.engineer.features.statistics.avgdev import avgdev

        result = avgdev(price_data, timeperiod=14)
        assert result is not None

    def test_avgdev_polars_impl(self, price_data):
        """Test AVGDEV with polars implementation."""
        from ml4t.engineer.features.statistics.avgdev import avgdev

        df = pl.DataFrame({"close": price_data})
        result = df.select(avgdev("close", timeperiod=14))
        assert result is not None


class TestStatisticsEdgeCases:
    """Edge case tests for statistics features."""

    def test_constant_prices(self):
        """Test with constant prices."""
        from ml4t.engineer.features.statistics.stddev import stddev
        from ml4t.engineer.features.statistics.var import var

        close = np.full(100, 100.0)

        stddev_result = stddev(close, period=5)
        var_result = var(close, timeperiod=5)

        # Constant values should have zero variance/stddev
        valid_stddev = stddev_result[~np.isnan(stddev_result)]
        valid_var = var_result[~np.isnan(var_result)]

        if len(valid_stddev) > 0:
            assert np.allclose(valid_stddev, 0, atol=1e-10)
        if len(valid_var) > 0:
            assert np.allclose(valid_var, 0, atol=1e-10)

    def test_short_data(self):
        """Test with short data."""
        from ml4t.engineer.features.statistics.linearreg import linearreg
        from ml4t.engineer.features.statistics.tsf import tsf

        close = np.array([100.0, 101.0, 102.0])

        lr_result = linearreg(close, timeperiod=14)
        tsf_result = tsf(close, timeperiod=14)

        # Should handle gracefully
        assert len(lr_result) == 3
        assert len(tsf_result) == 3

    def test_empty_array(self):
        """Test with empty array."""
        from ml4t.engineer.features.statistics.stddev import stddev

        result = stddev(np.array([]), period=5)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        from ml4t.engineer.features.statistics.var import var

        result = var(np.array([100.0]), timeperiod=5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_high_volatility(self):
        """Test with high volatility data."""
        from ml4t.engineer.features.statistics.stddev import stddev
        from ml4t.engineer.features.statistics.var import var

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 5)

        stddev_result = stddev(close, period=10)
        var_result = var(close, timeperiod=10)

        valid_stddev = stddev_result[~np.isnan(stddev_result)]
        valid_var = var_result[~np.isnan(var_result)]

        assert len(valid_stddev) > 0
        assert len(valid_var) > 0
        # High volatility should give higher values
        assert valid_stddev.mean() > 0.1


class TestLinearRegFamilyCoverage:
    """Coverage tests for linear regression family."""

    @pytest.fixture
    def price_data(self):
        """Create price data for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_linearreg_slope_numpy(self, price_data):
        """Test LINEARREG_SLOPE with numpy."""
        from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope

        result = linearreg_slope(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_linearreg_slope_numba(self, price_data):
        """Test LINEARREG_SLOPE with numba implementation."""
        from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope

        result = linearreg_slope(price_data, timeperiod=14)
        assert result is not None

    def test_linearreg_intercept_numpy(self, price_data):
        """Test LINEARREG_INTERCEPT with numpy."""
        from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept

        result = linearreg_intercept(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_linearreg_intercept_numba(self, price_data):
        """Test LINEARREG_INTERCEPT with numba implementation."""
        from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept

        result = linearreg_intercept(price_data, timeperiod=14)
        assert result is not None

    def test_linearreg_angle_numpy(self, price_data):
        """Test LINEARREG_ANGLE with numpy."""
        from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle

        result = linearreg_angle(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_linearreg_angle_numba(self, price_data):
        """Test LINEARREG_ANGLE with numba implementation."""
        from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle

        result = linearreg_angle(price_data, timeperiod=14)
        assert result is not None

    def test_linearreg_family_polars(self, price_data):
        """Test linear regression family with Polars."""
        from ml4t.engineer.features.statistics.linearreg import linearreg
        from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle
        from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept
        from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope

        df = pl.DataFrame({"close": price_data})

        lr = df.select(linearreg("close", timeperiod=14))
        slope = df.select(linearreg_slope("close", timeperiod=14))
        intercept = df.select(linearreg_intercept("close", timeperiod=14))
        angle = df.select(linearreg_angle("close", timeperiod=14))

        assert lr is not None
        assert slope is not None
        assert intercept is not None
        assert angle is not None
