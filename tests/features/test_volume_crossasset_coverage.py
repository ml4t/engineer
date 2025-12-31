"""
Comprehensive coverage tests for volume and cross_asset modules.

This test file targets missing coverage in:
1. features/volume/adosc.py - Numba paths and edge cases
2. features/volume/obv.py - Numba edge cases
3. features/volume/ad.py - NaN handling and validation
4. features/cross_asset.py - min_periods, validation, Numba functions
"""

import numpy as np
import polars as pl
import pytest


class TestADOSCNumbaDirectCalls:
    """Test ADOSC Numba function directly to cover missing lines."""

    def test_adosc_numba_basic(self):
        """Test adosc_numba directly."""
        from ml4t.engineer.features.volume.adosc import adosc_numba

        np.random.seed(42)
        n = 100
        high = np.random.randn(n) + 101
        low = np.random.randn(n) + 99
        close = np.random.randn(n) + 100
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)

        assert isinstance(result, np.ndarray)
        assert len(result) == n
        # First 9 values should be NaN (lookback = slowperiod - 1 = 9)
        assert np.isnan(result[:9]).all()

    def test_adosc_numba_insufficient_data(self):
        """Test adosc_numba with data shorter than lookback."""
        from ml4t.engineer.features.volume.adosc import adosc_numba

        # Only 5 bars, but slowperiod=10 requires lookback=9
        high = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        low = np.array([99.0, 100.0, 101.0, 102.0, 103.0])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = np.array([5000.0, 5100.0, 5200.0, 5300.0, 5400.0])

        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)

        # All values should be NaN (n=5 <= lookback=9)
        assert np.isnan(result).all()

    def test_adosc_numba_zero_hl_diff(self):
        """Test adosc_numba when high == low (zero division case)."""
        from ml4t.engineer.features.volume.adosc import adosc_numba

        n = 50
        # All bars have high == low
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)
        volume = np.full(n, 5000.0)

        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)

        assert isinstance(result, np.ndarray)
        assert len(result) == n
        # Should handle gracefully without errors
        # AD line should remain 0 throughout, so ADOSC should be 0
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0)

    def test_adosc_numba_fast_equals_slow(self):
        """Test adosc_numba when fastperiod == slowperiod."""
        from ml4t.engineer.features.volume.adosc import adosc_numba

        np.random.seed(42)
        n = 50
        high = np.random.randn(n) + 101
        low = np.random.randn(n) + 99
        close = np.random.randn(n) + 100
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = adosc_numba(high, low, close, volume, fastperiod=10, slowperiod=10)

        # When both periods are equal, ADOSC should be 0
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0)

    def test_adosc_numba_fast_greater_than_slow(self):
        """Test adosc_numba when fastperiod > slowperiod (unusual but valid)."""
        from ml4t.engineer.features.volume.adosc import adosc_numba

        np.random.seed(42)
        n = 50
        high = np.random.randn(n) + 101
        low = np.random.randn(n) + 99
        close = np.random.randn(n) + 100
        volume = np.random.randint(1000, 10000, n).astype(float)

        # Fast > slow (unusual configuration)
        result = adosc_numba(high, low, close, volume, fastperiod=10, slowperiod=3)

        assert isinstance(result, np.ndarray)
        # Lookback should be based on max(fastperiod, slowperiod) = 10
        assert np.isnan(result[:9]).all()

    def test_adosc_validation_errors(self):
        """Test ADOSC validation errors."""
        from ml4t.engineer.features.volume.adosc import adosc

        # Test invalid fastperiod
        with pytest.raises(ValueError, match="fastperiod must be > 0"):
            adosc(
                np.array([101.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
                fastperiod=0,
            )

        # Test negative fastperiod
        with pytest.raises(ValueError, match="fastperiod must be > 0"):
            adosc(
                np.array([101.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
                fastperiod=-1,
            )

        # Test invalid slowperiod
        with pytest.raises(ValueError, match="slowperiod must be > 0"):
            adosc(
                np.array([101.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
                slowperiod=0,
            )

        # Test mismatched lengths
        with pytest.raises(
            ValueError,
            match="high, low, close, and volume must have the same length",
        ):
            adosc(
                np.array([101.0, 102.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
            )


class TestOBVNumbaDirectCalls:
    """Test OBV Numba function directly to cover missing lines."""

    def test_obv_numba_basic(self):
        """Test obv_numba directly."""
        from ml4t.engineer.features.volume.obv import obv_numba

        close = np.array([100.0, 101.0, 102.0, 101.0, 103.0])
        volume = np.array([5000.0, 5100.0, 5200.0, 5300.0, 5400.0])

        result = obv_numba(close, volume)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        # First value should be first volume
        assert result[0] == 5000.0

    def test_obv_numba_empty_arrays(self):
        """Test obv_numba with empty arrays."""
        from ml4t.engineer.features.volume.obv import obv_numba

        result = obv_numba(np.array([]), np.array([]))

        assert len(result) == 0

    def test_obv_numba_mismatched_lengths(self):
        """Test obv_numba with mismatched array lengths."""
        from ml4t.engineer.features.volume.obv import obv_numba

        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([5000.0, 5100.0])  # Different length

        result = obv_numba(close, volume)

        # Should return all NaN when lengths don't match
        assert np.isnan(result).all()

    def test_obv_numba_price_increase(self):
        """Test obv_numba with increasing prices."""
        from ml4t.engineer.features.volume.obv import obv_numba

        # All prices increase
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        result = obv_numba(close, volume)

        # OBV should be cumulative sum of volume
        expected = np.cumsum(volume)
        np.testing.assert_array_equal(result, expected)

    def test_obv_numba_price_decrease(self):
        """Test obv_numba with decreasing prices."""
        from ml4t.engineer.features.volume.obv import obv_numba

        # All prices decrease
        close = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
        volume = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        result = obv_numba(close, volume)

        # First value is volume[0]
        # Then subtract each subsequent volume
        expected = np.array([1000.0, -1000.0, -4000.0, -8000.0, -13000.0])
        np.testing.assert_array_equal(result, expected)

    def test_obv_numba_price_unchanged(self):
        """Test obv_numba with unchanged prices."""
        from ml4t.engineer.features.volume.obv import obv_numba

        # All prices stay the same
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        result = obv_numba(close, volume)

        # OBV should remain unchanged after first value
        # First value = volume[0], then no changes
        expected = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        np.testing.assert_array_equal(result, expected)

    def test_obv_numba_mixed_price_movements(self):
        """Test obv_numba with mixed price movements."""
        from ml4t.engineer.features.volume.obv import obv_numba

        close = np.array([100.0, 101.0, 101.0, 100.0, 102.0])
        volume = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        result = obv_numba(close, volume)

        # Manual calculation:
        # [0]: 1000 (first volume)
        # [1]: 1000 + 2000 = 3000 (price up)
        # [2]: 3000 (price unchanged)
        # [3]: 3000 - 4000 = -1000 (price down)
        # [4]: -1000 + 5000 = 4000 (price up)
        expected = np.array([1000.0, 3000.0, 3000.0, -1000.0, 4000.0])
        np.testing.assert_array_equal(result, expected)

    def test_obv_validation_error(self):
        """Test OBV validation errors."""
        from ml4t.engineer.features.volume.obv import obv

        # Test mismatched lengths
        with pytest.raises(
            ValueError,
            match="close and volume must have the same length",
        ):
            obv(
                np.array([100.0, 101.0]),
                np.array([5000.0]),
            )


class TestADNumbaDirectCalls:
    """Test AD Numba function directly to cover missing lines."""

    def test_ad_numba_basic(self):
        """Test ad_numba directly."""
        from ml4t.engineer.features.volume.ad import ad_numba

        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, 99.0, 100.0])
        close = np.array([101.0, 100.0, 103.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad_numba(high, low, close, volume)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        # AD should be cumulative
        assert result[0] != 0  # First value should be non-zero

    def test_ad_numba_with_nans(self):
        """Test ad_numba with NaN values."""
        from ml4t.engineer.features.volume.ad import ad_numba

        high = np.array([102.0, 103.0, np.nan, 104.0, 105.0])
        low = np.array([98.0, 99.0, 100.0, 101.0, 102.0])
        close = np.array([101.0, 100.0, 103.0, 103.0, 104.0])
        volume = np.array([1000.0, 2000.0, 1500.0, 1800.0, 2000.0])

        result = ad_numba(high, low, close, volume)

        # Once NaN is hit, all subsequent values should be NaN
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_ad_numba_nan_in_low(self):
        """Test ad_numba with NaN in low array."""
        from ml4t.engineer.features.volume.ad import ad_numba

        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, np.nan, 100.0])
        close = np.array([101.0, 100.0, 103.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad_numba(high, low, close, volume)

        # NaN at index 1 should propagate
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_ad_numba_nan_in_close(self):
        """Test ad_numba with NaN in close array."""
        from ml4t.engineer.features.volume.ad import ad_numba

        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, 99.0, 100.0])
        close = np.array([101.0, np.nan, 103.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad_numba(high, low, close, volume)

        # NaN at index 1 should propagate
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_ad_numba_nan_in_volume(self):
        """Test ad_numba with NaN in volume array."""
        from ml4t.engineer.features.volume.ad import ad_numba

        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, 99.0, 100.0])
        close = np.array([101.0, 100.0, 103.0])
        volume = np.array([1000.0, np.nan, 1500.0])

        result = ad_numba(high, low, close, volume)

        # NaN at index 1 should propagate
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_ad_numba_high_equals_low(self):
        """Test ad_numba when high == low (MFM = 0)."""
        from ml4t.engineer.features.volume.ad import ad_numba

        # All bars have high == low
        high = np.array([100.0, 100.0, 100.0])
        low = np.array([100.0, 100.0, 100.0])
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad_numba(high, low, close, volume)

        # MFM should be 0 when high == low, so MFV = 0, AD line stays at 0
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))

    def test_ad_validation_error(self):
        """Test AD validation errors."""
        from ml4t.engineer.features.volume.ad import ad

        # Test mismatched lengths
        with pytest.raises(
            ValueError,
            match="high, low, close, and volume must have the same length",
        ):
            ad(
                np.array([101.0, 102.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
            )

    def test_ad_polars_implementation_error(self):
        """Test AD error when requesting polars impl with numpy arrays."""
        from ml4t.engineer.features.volume.ad import ad

        # Request polars implementation with numpy arrays (should fail)
        with pytest.raises(
            ValueError,
            match="Polars implementation requires all inputs to be column names",
        ):
            ad(
                np.array([101.0]),
                np.array([99.0]),
                np.array([100.0]),
                np.array([5000.0]),
                implementation="polars",
            )


class TestCrossAssetMinPeriods:
    """Test cross_asset functions with min_periods parameter."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 100
        returns1 = np.random.normal(0.0005, 0.01, n)
        returns2 = 0.8 * returns1 + 0.2 * np.random.normal(0.0003, 0.008, n)

        return pl.DataFrame(
            {
                "asset1_returns": returns1,
                "asset2_returns": returns2,
            }
        )

    def test_rolling_correlation_min_periods(self, sample_data):
        """Test rolling_correlation with custom min_periods."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        # Test with min_periods = 5
        result = sample_data.select(
            rolling_correlation(
                "asset1_returns",
                "asset2_returns",
                window=20,
                min_periods=5,
            ).alias("corr"),
        )

        assert len(result) == len(sample_data)
        # Should have values starting from index 5
        corr_values = result["corr"].drop_nulls()
        assert len(corr_values) > 0

    def test_rolling_correlation_min_periods_greater_than_window(self, sample_data):
        """Test rolling_correlation with min_periods > window."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        # min_periods > window should raise error
        with pytest.raises(
            ValueError,
            match="min_periods.*cannot exceed window",
        ):
            sample_data.select(
                rolling_correlation(
                    "asset1_returns",
                    "asset2_returns",
                    window=20,
                    min_periods=25,
                ).alias("corr"),
            )

    def test_rolling_correlation_invalid_min_periods(self, sample_data):
        """Test rolling_correlation with invalid min_periods."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        # min_periods = 0 should raise error
        with pytest.raises(ValueError, match="min_periods must be at least 1"):
            sample_data.select(
                rolling_correlation(
                    "asset1_returns",
                    "asset2_returns",
                    window=20,
                    min_periods=0,
                ).alias("corr"),
            )

    def test_beta_to_market_min_periods(self, sample_data):
        """Test beta_to_market with custom min_periods."""
        from ml4t.engineer.features.cross_asset import beta_to_market

        result = sample_data.select(
            beta_to_market(
                "asset1_returns",
                "asset2_returns",
                window=30,
                min_periods=10,
            ).alias("beta"),
        )

        assert len(result) == len(sample_data)
        beta_values = result["beta"].drop_nulls()
        assert len(beta_values) > 0

    def test_beta_to_market_min_periods_greater_than_window(self, sample_data):
        """Test beta_to_market with min_periods > window."""
        from ml4t.engineer.features.cross_asset import beta_to_market

        with pytest.raises(
            ValueError,
            match="min_periods.*cannot exceed window",
        ):
            sample_data.select(
                beta_to_market(
                    "asset1_returns",
                    "asset2_returns",
                    window=30,
                    min_periods=40,
                ).alias("beta"),
            )

    def test_correlation_matrix_features_min_periods(self, sample_data):
        """Test correlation_matrix_features with custom min_periods."""
        from ml4t.engineer.features.cross_asset import correlation_matrix_features

        # Add a third asset
        data = sample_data.with_columns(
            (pl.col("asset1_returns") * 0.5).alias("asset3_returns"),
        )

        features = correlation_matrix_features(
            ["asset1_returns", "asset2_returns", "asset3_returns"],
            window=20,
            min_periods=10,
        )

        result = data.with_columns(
            [features["avg_correlation"].alias("avg_corr")],
        )

        assert len(result) == len(data)
        avg_corr = result["avg_corr"].drop_nulls()
        assert len(avg_corr) > 0


class TestCrossAssetValidation:
    """Test cross_asset validation edge cases."""

    def test_rolling_correlation_invalid_window_type(self):
        """Test rolling_correlation with non-integer window."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        with pytest.raises(TypeError):
            rolling_correlation("asset1", "asset2", window=20.5)

    def test_rolling_correlation_negative_window(self):
        """Test rolling_correlation with negative window."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        with pytest.raises(ValueError, match="window must be at least 2"):
            rolling_correlation("asset1", "asset2", window=-10)

    def test_correlation_regime_invalid_thresholds(self):
        """Test correlation_regime_indicator with invalid thresholds."""
        from ml4t.engineer.features.cross_asset import correlation_regime_indicator

        # Test threshold > 1
        with pytest.raises(ValueError, match="low_threshold must be between 0.0 and 1.0"):
            correlation_regime_indicator("corr", low_threshold=1.5, high_threshold=2.0)

        # Test threshold < 0
        with pytest.raises(ValueError, match="low_threshold must be between 0.0 and 1.0"):
            correlation_regime_indicator("corr", low_threshold=-0.5, high_threshold=0.5)

        # Test high_threshold > 1
        with pytest.raises(ValueError, match="high_threshold must be between 0.0 and 1.0"):
            correlation_regime_indicator("corr", low_threshold=0.3, high_threshold=1.5)

    def test_correlation_regime_invalid_lookback(self):
        """Test correlation_regime_indicator with invalid lookback."""
        from ml4t.engineer.features.cross_asset import correlation_regime_indicator

        with pytest.raises(ValueError, match="lookback must be at least 1"):
            correlation_regime_indicator("corr", lookback=0)

    def test_lead_lag_invalid_window(self):
        """Test lead_lag_correlation with invalid window."""
        from ml4t.engineer.features.cross_asset import lead_lag_correlation

        with pytest.raises(ValueError, match="window must be at least 2"):
            lead_lag_correlation("asset1", "asset2", max_lag=5, window=1)

    def test_multi_asset_dispersion_single_asset(self):
        """Test multi_asset_dispersion with too few assets."""
        from ml4t.engineer.features.cross_asset import multi_asset_dispersion

        with pytest.raises(
            ValueError,
            match="returns_list must have at least 2 elements",
        ):
            multi_asset_dispersion(["asset1"])


class TestTransferEntropyNumba:
    """Test transfer_entropy_nb Numba function directly."""

    def test_transfer_entropy_nb_basic(self):
        """Test transfer_entropy_nb directly."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        result = transfer_entropy_nb(x, y, lag=1, bins=10)

        assert isinstance(result, float | np.floating)
        # Should be finite
        assert np.isfinite(result)

    def test_transfer_entropy_nb_insufficient_data(self):
        """Test transfer_entropy_nb with insufficient data."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        # Only 5 samples
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = transfer_entropy_nb(x, y, lag=1, bins=10)

        # Should return NaN (n < 10)
        assert np.isnan(result)

    def test_transfer_entropy_nb_constant_x(self):
        """Test transfer_entropy_nb with constant x."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        # Constant x (no variation)
        x = np.full(100, 1.0)
        y = np.random.randn(100)

        result = transfer_entropy_nb(x, y, lag=1, bins=10)

        # Should return 0.0 (no information in constant series)
        assert result == 0.0

    def test_transfer_entropy_nb_constant_y(self):
        """Test transfer_entropy_nb with constant y."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        x = np.random.randn(100)
        y = np.full(100, 1.0)

        result = transfer_entropy_nb(x, y, lag=1, bins=10)

        # Should return 0.0
        assert result == 0.0

    def test_transfer_entropy_nb_different_lags(self):
        """Test transfer_entropy_nb with different lag values."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        # Test different lags
        result_lag1 = transfer_entropy_nb(x, y, lag=1, bins=10)
        result_lag5 = transfer_entropy_nb(x, y, lag=5, bins=10)

        # Both should be finite
        assert np.isfinite(result_lag1)
        assert np.isfinite(result_lag5)

    def test_transfer_entropy_nb_different_bins(self):
        """Test transfer_entropy_nb with different bin counts."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        # Test different bin counts
        result_5bins = transfer_entropy_nb(x, y, lag=1, bins=5)
        result_20bins = transfer_entropy_nb(x, y, lag=1, bins=20)

        # Both should be finite
        assert np.isfinite(result_5bins)
        assert np.isfinite(result_20bins)

    def test_transfer_entropy_raises_not_implemented(self):
        """Test that transfer_entropy Polars wrapper raises NotImplementedError."""
        from ml4t.engineer.features.cross_asset import transfer_entropy

        with pytest.raises(
            NotImplementedError,
            match="Transfer entropy calculation is not yet implemented",
        ):
            transfer_entropy("asset1", "asset2")


class TestCrossAssetEdgeCases:
    """Test edge cases in cross_asset functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 50
        returns1 = np.random.normal(0.0005, 0.01, n)
        returns2 = np.random.normal(0.0003, 0.008, n)
        prices1 = 100 * np.exp(np.cumsum(returns1))
        prices2 = 100 * np.exp(np.cumsum(returns2))

        return pl.DataFrame(
            {
                "asset1_returns": returns1,
                "asset2_returns": returns2,
                "asset1_price": prices1,
                "asset2_price": prices2,
            }
        )

    def test_co_integration_with_short_data(self, sample_data):
        """Test co_integration_score with data shorter than window."""
        from ml4t.engineer.features.cross_asset import co_integration_score

        # Use window larger than data
        result = sample_data.select(
            co_integration_score("asset1_price", "asset2_price", window=100).alias(
                "coint",
            ),
        )

        # Should handle gracefully
        assert len(result) == len(sample_data)

    def test_cross_asset_momentum_mad_method(self):
        """Test multi_asset_dispersion with MAD method."""
        from ml4t.engineer.features.cross_asset import multi_asset_dispersion

        np.random.seed(42)
        n = 100
        returns1 = np.random.normal(0.0005, 0.01, n)
        returns2 = np.random.normal(0.0003, 0.008, n)
        returns3 = np.random.normal(0.0002, 0.012, n)

        df = pl.DataFrame(
            {
                "r1": returns1,
                "r2": returns2,
                "r3": returns3,
            }
        )

        result = df.select(
            multi_asset_dispersion(["r1", "r2", "r3"], window=20, method="mad").alias(
                "dispersion",
            ),
        )

        disp_values = result["dispersion"].drop_nulls()
        assert len(disp_values) > 0
        assert (disp_values > 0).all()

    def test_relative_strength_index_spread_edge_case(self):
        """Test RSI spread with extreme values."""
        from ml4t.engineer.features.cross_asset import relative_strength_index_spread

        # RSI values at extremes
        rsi1 = np.full(50, 90.0)  # Overbought
        rsi2 = np.full(50, 10.0)  # Oversold

        df = pl.DataFrame(
            {
                "rsi1": rsi1,
                "rsi2": rsi2,
            }
        )

        result = df.select(
            relative_strength_index_spread("rsi1", "rsi2", smooth_period=5).alias(
                "spread",
            ),
        )

        spread_values = result["spread"].drop_nulls()
        # Spread should be around 80
        assert spread_values.mean() > 70

    def test_volatility_ratio_with_zero_vol(self):
        """Test volatility_ratio when one volatility is near zero."""
        from ml4t.engineer.features.cross_asset import volatility_ratio

        vol1 = np.full(50, 0.01)
        vol2 = np.full(50, 1e-12)  # Near zero

        df = pl.DataFrame(
            {
                "vol1": vol1,
                "vol2": vol2,
            }
        )

        # Should handle gracefully with 1e-10 epsilon
        result = df.select(
            volatility_ratio("vol1", "vol2", log_ratio=False).alias("ratio"),
        )

        ratio_values = result["ratio"].drop_nulls()
        # Should be very large but finite
        assert (ratio_values > 0).all()
        assert ratio_values.is_finite().all()
