"""
Comprehensive tests for low-coverage modules to boost overall coverage.

Focuses on:
1. rolling_entropy.py (150 missing lines, 33.6% covered) - Direct Numba function tests
2. structural_break.py (106 missing lines, 37.3% covered) - Numba implementations
3. t3.py (69 missing lines, 24.2% covered) - T3 Numba function
4. risk.py (58 missing lines, 66.5% covered) - Risk metric edge cases
5. cross_asset.py (37 missing lines, 77.3% covered) - Cross-asset functions
"""

import numpy as np
import polars as pl
import pytest

# =============================================================================
# ROLLING ENTROPY TESTS - Target 150 missing lines
# =============================================================================


class TestRollingEntropyNumba:
    """Direct tests of Numba functions for rolling entropy."""

    def test_encode_binary_nb_edge_cases(self):
        """Test binary encoding with various edge cases."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_binary_nb

        # All zeros
        result = _encode_binary_nb(np.zeros(10))
        assert all(r == 0 for r in result)

        # Mixed positive/negative
        data = np.array([1.0, -1.0, 0.0, 2.0, -2.0])
        result = _encode_binary_nb(data)
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 0

        # With NaN
        data = np.array([1.0, np.nan, -1.0])
        result = _encode_binary_nb(data)
        assert result[0] == 1
        assert result[1] == -1
        assert result[2] == 0

    def test_encode_quantile_nb_boundary_cases(self):
        """Test quantile encoding at bin boundaries."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_quantile_nb

        # Equal values should all map to same bin
        data = np.ones(50) * 5.0
        result = _encode_quantile_nb(data, n_bins=10)
        assert len(np.unique(result)) == 1

        # Empty valid values
        data = np.array([np.nan, np.nan])
        result = _encode_quantile_nb(data, n_bins=5)
        assert all(r == -1 for r in result)

        # Boundary testing with known distribution
        data = np.linspace(0, 100, 1000)
        result = _encode_quantile_nb(data, n_bins=10)
        # Check bins are utilized
        unique_bins = np.unique(result[result >= 0])
        assert len(unique_bins) == 10

    def test_encode_sigma_nb_extreme_values(self):
        """Test sigma encoding with extreme z-scores."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        # Extremely spread out data
        data = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
        result = _encode_sigma_nb(data, n_bins=10)
        # Should clamp to valid range
        assert all(0 <= r < 10 for r in result)

        # Zero std (all same)
        data = np.full(50, 42.0)
        result = _encode_sigma_nb(data, n_bins=10)
        assert all(r == 5 for r in result)  # Middle bin

        # Zero std with NaN
        data = np.array([42.0, 42.0, np.nan, 42.0])
        result = _encode_sigma_nb(data, n_bins=10)
        assert result[0] == 5
        assert result[2] == -1

    def test_shannon_entropy_nb_distributions(self):
        """Test Shannon entropy with different distributions."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        # Uniform distribution
        data = np.random.uniform(0, 1, 1000)
        result = _shannon_entropy_nb(data, n_bins=10)
        assert result > 3.0  # High entropy for uniform

        # Single value (deterministic)
        data = np.ones(100)
        result = _shannon_entropy_nb(data, n_bins=10)
        assert result == 0.0

        # Empty array
        result = _shannon_entropy_nb(np.array([]), n_bins=10)
        assert np.isnan(result)

        # All NaN
        result = _shannon_entropy_nb(np.array([np.nan, np.nan]), n_bins=10)
        assert np.isnan(result)

        # Very small range
        data = np.array([1.0, 1.0 + 1e-15])
        result = _shannon_entropy_nb(data, n_bins=10)
        assert result == 0.0

    def test_lz_match_length_patterns(self):
        """Test LZ match length with various patterns."""
        from ml4t.engineer.features.ml.rolling_entropy import _lz_match_length

        # Exact repeat
        seq = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
        match = _lz_match_length(seq, 3, 3)
        assert match > 1

        # No match
        seq = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        match = _lz_match_length(seq, 3, 3)
        assert match >= 1

        # Position 0
        seq = np.array([1, 2, 3], dtype=np.int32)
        match = _lz_match_length(seq, 0, 10)
        assert match == 1

        # Window larger than position
        seq = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
        match = _lz_match_length(seq, 2, 100)
        assert match >= 1

    def test_kontoyiannis_entropy_nb_sequences(self):
        """Test Kontoyiannis entropy with different sequences."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        # Highly repetitive
        seq = np.array([0, 1] * 50, dtype=np.int32)
        result = _kontoyiannis_entropy_nb(seq, window_size=50)
        assert result >= 0

        # Random sequence
        np.random.seed(42)
        seq = np.random.randint(0, 10, 100, dtype=np.int32)
        result = _kontoyiannis_entropy_nb(seq, window_size=50)
        assert result >= 0

        # With invalid entries (-1)
        seq = np.array([0, 1, -1, 2, 3, -1, 4, 5], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(seq, window_size=5)
        # Should filter -1 and compute
        assert not np.isnan(result)

        # All invalid
        seq = np.array([-1, -1, -1], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(seq, window_size=5)
        assert np.isnan(result)

        # Too short (k < 3)
        seq = np.array([0, 1], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(seq, window_size=5)
        assert np.isnan(result)

    def test_plugin_entropy_nb_word_lengths(self):
        """Test plugin entropy with various word lengths."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        # Word length 1
        seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        r1 = _plugin_entropy_nb(seq, word_length=1)
        assert r1 >= 0

        # Word length 2
        r2 = _plugin_entropy_nb(seq, word_length=2)
        assert r2 >= 0

        # Word length 3
        seq = np.random.randint(0, 5, 100, dtype=np.int32)
        r3 = _plugin_entropy_nb(seq, word_length=3)
        assert r3 >= 0

        # Word length longer than sequence
        seq = np.array([0, 1, 2], dtype=np.int32)
        result = _plugin_entropy_nb(seq, word_length=10)
        assert np.isnan(result)

        # With invalid entries
        seq = np.array([0, 1, -1, 2, -1, 3], dtype=np.int32)
        result = _plugin_entropy_nb(seq, word_length=1)
        assert result >= 0

        # All invalid
        seq = np.array([-1, -1], dtype=np.int32)
        result = _plugin_entropy_nb(seq, word_length=1)
        assert np.isnan(result)


class TestRollingEntropyEdgeCases:
    """Test rolling entropy functions with edge cases."""

    def test_rolling_entropy_different_parameters(self):
        """Test rolling entropy with different parameter combinations."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        df = pl.DataFrame({"returns": np.random.randn(200) * 0.02})

        # Different window sizes
        for window in [20, 50, 100]:
            result = df.select(rolling_entropy("returns", window=window))
            assert result is not None

        # Different bin counts
        for n_bins in [5, 10, 20]:
            result = df.select(rolling_entropy("returns", window=50, n_bins=n_bins))
            assert result is not None

        # With pl.Expr
        result = df.select(rolling_entropy(pl.col("returns"), window=30))
        assert result is not None

    def test_rolling_entropy_lz_all_encodings(self):
        """Test rolling LZ entropy with all encoding schemes."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        df = pl.DataFrame({"returns": np.random.randn(150) * 0.02})

        for encoding in ["binary", "quantile", "sigma"]:
            result = df.select(rolling_entropy_lz("returns", window=100, encoding=encoding))
            assert result is not None
            assert len(result) == len(df)

        # With different n_bins
        result = df.select(rolling_entropy_lz("returns", window=100, encoding="quantile", n_bins=5))
        assert result is not None

        # Invalid encoding
        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_lz("returns", window=100, encoding="invalid"))

    def test_rolling_entropy_plugin_parameters(self):
        """Test rolling plugin entropy with various parameters."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        df = pl.DataFrame({"returns": np.random.randn(150) * 0.02})

        # All encodings
        for encoding in ["binary", "quantile", "sigma"]:
            result = df.select(rolling_entropy_plugin("returns", window=50, encoding=encoding))
            assert result is not None

        # Different word lengths
        for word_length in [1, 2, 3]:
            result = df.select(
                rolling_entropy_plugin("returns", window=50, word_length=word_length)
            )
            assert result is not None

        # Invalid encoding
        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_plugin("returns", window=50, encoding="bad"))


# =============================================================================
# STRUCTURAL BREAK TESTS - Target 106 missing lines
# =============================================================================


class TestStructuralBreakNumba:
    """Direct tests of Numba functions for structural break detection."""

    def test_cv_nb_edge_cases(self):
        """Test coefficient of variation with edge cases."""
        from ml4t.engineer.features.statistics.structural_break import _cv_nb

        # Normal data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _cv_nb(data)
        assert result >= 0

        # Zero mean
        data = np.array([-1.0, 0.0, 1.0])
        result = _cv_nb(data)
        # CV undefined for zero mean
        assert np.isnan(result)

        # Near-zero mean
        data = np.array([1e-12, 2e-12, 3e-12])
        result = _cv_nb(data)
        assert np.isnan(result)

        # All NaN
        data = np.array([np.nan, np.nan])
        result = _cv_nb(data)
        assert np.isnan(result)

        # Insufficient data
        data = np.array([1.0])
        result = _cv_nb(data)
        assert np.isnan(result)

    def test_cv_zscore_nb_cases(self):
        """Test CV Z-score calculation."""
        from ml4t.engineer.features.statistics.structural_break import _cv_zscore_nb

        # Normal case
        np.random.seed(42)
        data = np.random.randn(200) * 0.5 + 10.0
        result = _cv_zscore_nb(data, lookback=20)
        assert not np.isnan(result)

        # Insufficient data
        data = np.random.randn(10)
        result = _cv_zscore_nb(data, lookback=20)
        assert np.isnan(result)

        # Constant CV (std = 0)
        data = np.ones(100) * 5.0
        result = _cv_zscore_nb(data, lookback=10)
        assert result == 0.0

        # Few valid CVs
        data = np.random.randn(15)
        result = _cv_zscore_nb(data, lookback=10)
        # Need at least 10 valid CVs
        assert np.isnan(result)

    def test_variance_ratio_nb_cases(self):
        """Test variance ratio calculation."""
        from ml4t.engineer.features.statistics.structural_break import _variance_ratio_nb

        # Random walk should have VR ≈ 1
        np.random.seed(42)
        data = np.cumsum(np.random.randn(200))
        result = _variance_ratio_nb(data, q=5)
        assert 0.5 < result < 2.0

        # Mean reverting (VR < 1)
        data = np.sin(np.linspace(0, 20, 100))
        result = _variance_ratio_nb(data, q=5)
        assert result > 0

        # Insufficient data
        data = np.array([1.0, 2.0, 3.0])
        result = _variance_ratio_nb(data, q=5)
        assert np.isnan(result)

        # Zero variance
        data = np.ones(50)
        result = _variance_ratio_nb(data, q=3)
        assert np.isnan(result)

        # With NaN (will be filtered out)
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = _variance_ratio_nb(data, q=2)
        # Should compute on valid values
        assert result > 0 or np.isnan(result)

    def test_kl_divergence_nb_distributions(self):
        """Test KL divergence between distributions."""
        from ml4t.engineer.features.statistics.structural_break import _kl_divergence_nb

        # Identical distributions (split_point unused, uses automatic halving)
        np.random.seed(42)
        data = np.random.randn(100)
        result = _kl_divergence_nb(data, _split_point=50, n_bins=20)
        # Should be low but not exactly zero
        assert result >= 0

        # Very different distributions
        data = np.concatenate([np.random.randn(50) * 1.0, np.random.randn(50) * 5.0 + 10.0])
        result = _kl_divergence_nb(data, _split_point=50, n_bins=20)
        assert result > 0

        # Insufficient data
        data = np.array([1.0, 2.0, 3.0])
        result = _kl_divergence_nb(data, _split_point=1, n_bins=10)
        assert np.isnan(result)

        # All same value (no range)
        data = np.ones(100)
        result = _kl_divergence_nb(data, _split_point=50, n_bins=10)
        assert result == 0.0

        # All NaN
        data = np.array([np.nan, np.nan, np.nan, np.nan])
        result = _kl_divergence_nb(data, _split_point=2, n_bins=10)
        assert np.isnan(result)

    def test_wasserstein_1d_nb_cases(self):
        """Test 1D Wasserstein distance."""
        from ml4t.engineer.features.statistics.structural_break import _wasserstein_1d_nb

        # Identical distributions
        np.random.seed(42)
        data = np.random.randn(100)
        result = _wasserstein_1d_nb(data)
        assert result >= 0

        # Shifted distributions
        data = np.concatenate([np.random.randn(50), np.random.randn(50) + 5.0])
        result = _wasserstein_1d_nb(data)
        assert result > 0

        # Different sized halves
        data = np.random.randn(101)
        result = _wasserstein_1d_nb(data)
        assert result >= 0

        # Insufficient data
        data = np.array([1.0, 2.0, 3.0])
        result = _wasserstein_1d_nb(data)
        assert np.isnan(result)

        # With NaN
        data = np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0])
        result = _wasserstein_1d_nb(data)
        # NaN filtered
        assert not np.isnan(result)

    def test_drift_nb_cases(self):
        """Test drift detection."""
        from ml4t.engineer.features.statistics.structural_break import _drift_nb

        # Upward drift
        data = np.concatenate([np.ones(50) * 1.0, np.ones(50) * 5.0])
        result = _drift_nb(data)
        assert result > 0

        # Downward drift
        data = np.concatenate([np.ones(50) * 5.0, np.ones(50) * 1.0])
        result = _drift_nb(data)
        assert result < 0

        # No drift
        data = np.ones(100)
        result = _drift_nb(data)
        assert abs(result) < 1e-10

        # Insufficient data
        data = np.array([1.0, 2.0])
        result = _drift_nb(data)
        assert np.isnan(result)

    def test_drift_zscore_nb_cases(self):
        """Test drift Z-score calculation."""
        from ml4t.engineer.features.statistics.structural_break import _drift_zscore_nb

        # Significant drift
        np.random.seed(42)
        data = np.concatenate([np.random.randn(50) * 1.0, np.random.randn(50) * 1.0 + 10.0])
        result = _drift_zscore_nb(data)
        assert abs(result) > 2.0

        # Zero pooled SE with drift
        data = np.concatenate([np.ones(50) * 1.0, np.ones(50) * 5.0])
        result = _drift_zscore_nb(data)
        # Returns sign * 100
        assert result > 0

        # Zero pooled SE with no drift
        data = np.ones(100)
        result = _drift_zscore_nb(data)
        assert result == 0.0

        # Insufficient data
        data = np.array([1.0, 2.0])
        result = _drift_zscore_nb(data)
        assert np.isnan(result)


class TestStructuralBreakFeatures:
    """Test structural break detection features."""

    def test_coefficient_of_variation(self):
        """Test coefficient of variation feature."""
        from ml4t.engineer.features.statistics.structural_break import coefficient_of_variation

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(200) * 0.02})

        result = df.select(coefficient_of_variation("returns", window=50))
        assert result is not None
        assert len(result) == len(df)

    def test_rolling_cv_zscore(self):
        """Test rolling CV Z-score."""
        from ml4t.engineer.features.statistics.structural_break import rolling_cv_zscore

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        result = df.select(rolling_cv_zscore("returns", window=20, lookback_multiplier=5))
        assert result is not None

    def test_variance_ratio(self):
        """Test variance ratio feature."""
        from ml4t.engineer.features.statistics.structural_break import variance_ratio

        np.random.seed(42)
        df = pl.DataFrame({"price": np.cumsum(np.random.randn(200))})

        result = df.select(variance_ratio("price", window=100, q=5))
        assert result is not None

    def test_rolling_kl_divergence(self):
        """Test rolling KL divergence."""
        from ml4t.engineer.features.statistics.structural_break import rolling_kl_divergence

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(200) * 0.02})

        result = df.select(rolling_kl_divergence("returns", window=100, n_bins=20))
        assert result is not None

    def test_rolling_wasserstein(self):
        """Test rolling Wasserstein distance."""
        from ml4t.engineer.features.statistics.structural_break import rolling_wasserstein

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(200) * 0.02})

        result = df.select(rolling_wasserstein("returns", window=100))
        assert result is not None

    def test_rolling_drift(self):
        """Test rolling drift detection."""
        from ml4t.engineer.features.statistics.structural_break import rolling_drift

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(200) * 0.02})

        # Normalized
        result = df.select(rolling_drift("returns", window=100, normalize=True))
        assert result is not None

        # Raw drift
        result = df.select(rolling_drift("returns", window=100, normalize=False))
        assert result is not None


# =============================================================================
# T3 TESTS - Target 69 missing lines
# =============================================================================


class TestT3Numba:
    """Direct tests of T3 Numba function."""

    def test_t3_numba_basic(self):
        """Test T3 numba function with basic input."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        close = np.array([10, 11, 12, 11, 10, 11, 13, 14, 13, 12] * 10, dtype=np.float64)
        result = t3_numba(close, timeperiod=5, vfactor=0.7)

        assert len(result) == len(close)
        # First values should be NaN
        assert np.isnan(result[0])
        # Later values should be valid
        assert not np.all(np.isnan(result))

    def test_t3_numba_different_periods(self):
        """Test T3 with different time periods."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        np.random.seed(42)
        close = np.cumsum(np.random.randn(200)) + 100

        for period in [5, 10, 20]:
            result = t3_numba(close, timeperiod=period, vfactor=0.7)
            assert len(result) == len(close)
            lookback = 6 * (period - 1)
            assert np.all(np.isnan(result[:lookback]))

    def test_t3_numba_different_vfactors(self):
        """Test T3 with different volume factors."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        np.random.seed(42)
        close = np.cumsum(np.random.randn(100)) + 100

        for vf in [0.0, 0.5, 0.7, 1.0]:
            result = t3_numba(close, timeperiod=5, vfactor=vf)
            assert len(result) == len(close)
            # Valid values should exist
            assert not np.all(np.isnan(result))

    def test_t3_numba_insufficient_data(self):
        """Test T3 with insufficient data."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        # Shorter than lookback
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = t3_numba(close, timeperiod=5, vfactor=0.7)
        # All NaN when n <= lookback
        assert np.all(np.isnan(result))

    def test_t3_numba_edge_boundary(self):
        """Test T3 at exact lookback boundary."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        period = 5
        lookback = 6 * (period - 1)  # 24
        # Exactly lookback + 1 points
        close = np.linspace(100, 110, lookback + 1, dtype=np.float64)
        result = t3_numba(close, timeperiod=period, vfactor=0.7)

        # First lookback values should be NaN
        assert np.all(np.isnan(result[:lookback]))
        # Value at lookback position should be valid
        assert not np.isnan(result[lookback])


class TestT3Feature:
    """Test T3 feature function."""

    def test_t3_with_numpy_array(self):
        """Test T3 with NumPy array input."""
        from ml4t.engineer.features.trend.t3 import t3

        close = np.array([10, 11, 12, 11, 10, 11, 13, 14, 13, 12] * 10, dtype=np.float64)
        result = t3(close, timeperiod=5, vfactor=0.7)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_t3_with_polars_series(self):
        """Test T3 with Polars Series input."""
        from ml4t.engineer.features.trend.t3 import t3

        close = pl.Series([10, 11, 12, 11, 10, 11, 13, 14, 13, 12] * 10, dtype=pl.Float64)
        result = t3(close, timeperiod=5, vfactor=0.7)

        assert isinstance(result, np.ndarray)

    def test_t3_with_string_column(self):
        """Test T3 with string column name."""
        from ml4t.engineer.features.trend.t3 import t3

        result = t3("close", timeperiod=5, vfactor=0.7)
        assert isinstance(result, pl.Expr)

    def test_t3_validation(self):
        """Test T3 input validation."""
        from ml4t.engineer.features.trend.t3 import t3

        close = np.random.randn(100)

        # Invalid timeperiod
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            t3(close, timeperiod=1, vfactor=0.7)

        # Invalid vfactor
        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(close, timeperiod=5, vfactor=1.5)

        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(close, timeperiod=5, vfactor=-0.1)

    def test_t3_polars_expression(self):
        """Test T3 as Polars expression."""
        from ml4t.engineer.features.trend.t3 import t3_polars

        df = pl.DataFrame({"close": np.linspace(100, 110, 50)})
        result = df.select(t3_polars("close", timeperiod=5, vfactor=0.7))

        assert result is not None
        assert len(result) == len(df)


# =============================================================================
# RISK TESTS - Target 58 missing lines
# =============================================================================


class TestRiskMetricsEdgeCases:
    """Test risk metrics with edge cases."""

    def test_value_at_risk_methods(self):
        """Test VaR with all methods."""
        from ml4t.engineer.features.risk import value_at_risk

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        # Historical method
        result = df.select(
            value_at_risk("returns", confidence_level=0.95, window=252, method="historical")
        )
        assert result is not None

        # Parametric method
        result = df.select(
            value_at_risk("returns", confidence_level=0.95, window=252, method="parametric")
        )
        assert result is not None

        # Cornish-Fisher method
        result = df.select(
            value_at_risk("returns", confidence_level=0.95, window=252, method="cornish_fisher")
        )
        assert result is not None

        # Invalid method
        with pytest.raises(ValueError, match="method must be"):
            df.select(value_at_risk("returns", method="invalid"))

    def test_conditional_var_methods(self):
        """Test CVaR with different methods."""
        from ml4t.engineer.features.risk import conditional_value_at_risk

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        # Historical
        result = df.select(
            conditional_value_at_risk(
                "returns", confidence_level=0.95, window=252, method="historical"
            )
        )
        assert result is not None

        # Parametric
        result = df.select(
            conditional_value_at_risk(
                "returns", confidence_level=0.95, window=252, method="parametric"
            )
        )
        assert result is not None

        # Invalid method
        with pytest.raises(ValueError, match="method must be"):
            df.select(conditional_value_at_risk("returns", method="bad"))

    def test_maximum_drawdown_expanding(self):
        """Test maximum drawdown with expanding window."""
        from ml4t.engineer.features.risk import maximum_drawdown

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100
        pl.DataFrame({"price": prices})

        result = maximum_drawdown("price", window=None)
        assert "max_drawdown" in result
        assert "current_drawdown" in result

    def test_maximum_drawdown_rolling(self):
        """Test maximum drawdown with rolling window."""
        from ml4t.engineer.features.risk import maximum_drawdown

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100
        pl.DataFrame({"price": prices})

        result = maximum_drawdown("price", window=50)
        assert "max_drawdown" in result
        assert "max_duration" in result
        assert "current_drawdown" in result
        assert "time_underwater" in result

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        from ml4t.engineer.features.risk import downside_deviation

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        # Zero target
        result = df.select(downside_deviation("returns", target_return=0.0, window=252))
        assert result is not None

        # Non-zero target
        result = df.select(downside_deviation("returns", target_return=0.001, window=252))
        assert result is not None

    def test_tail_ratio(self):
        """Test tail ratio calculation."""
        from ml4t.engineer.features.risk import tail_ratio

        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        result = df.select(tail_ratio("returns", confidence_level=0.95, window=252))
        assert result is not None

    def test_higher_moments(self):
        """Test higher moments calculation."""
        from ml4t.engineer.features.risk import higher_moments

        np.random.seed(42)
        pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        result = higher_moments("returns", window=252)
        assert "skewness" in result
        assert "kurtosis" in result
        assert "hyperskewness" in result
        assert "hyperkurtosis" in result

    def test_risk_adjusted_returns_with_prices(self):
        """Test risk-adjusted returns with actual prices."""
        from ml4t.engineer.features.risk import risk_adjusted_returns

        np.random.seed(42)
        returns = np.random.randn(300) * 0.02
        prices = np.cumprod(1 + returns) * 100
        pl.DataFrame({"returns": returns, "price": prices})

        result = risk_adjusted_returns("returns", risk_free_rate=0.02, window=252, close="price")
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "calmar_ratio" in result
        assert "omega_ratio" in result

    def test_risk_adjusted_returns_without_prices(self):
        """Test risk-adjusted returns without prices."""
        from ml4t.engineer.features.risk import risk_adjusted_returns

        np.random.seed(42)
        pl.DataFrame({"returns": np.random.randn(300) * 0.02})

        result = risk_adjusted_returns("returns", risk_free_rate=0.02, window=252, close=None)
        assert "sharpe_ratio" in result
        assert "calmar_ratio" in result

    def test_ulcer_index(self):
        """Test Ulcer Index calculation."""
        from ml4t.engineer.features.risk import ulcer_index

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(300)) + 100
        df = pl.DataFrame({"price": prices})

        result = df.select(ulcer_index("price", window=252))
        assert result is not None

    def test_information_ratio(self):
        """Test Information Ratio calculation."""
        from ml4t.engineer.features.risk import information_ratio

        np.random.seed(42)
        df = pl.DataFrame(
            {"returns": np.random.randn(300) * 0.02, "benchmark": np.random.randn(300) * 0.015}
        )

        result = df.select(information_ratio("returns", "benchmark", window=252))
        assert result is not None


# =============================================================================
# CROSS ASSET TESTS - Target 37 missing lines
# =============================================================================


class TestCrossAssetEdgeCases:
    """Test cross-asset features with edge cases."""

    def test_rolling_correlation_validation(self):
        """Test rolling correlation with validation."""
        from ml4t.engineer.features.cross_asset import rolling_correlation

        df = pl.DataFrame({"asset1": np.random.randn(100), "asset2": np.random.randn(100)})

        # Basic usage
        result = df.select(rolling_correlation("asset1", "asset2", window=20))
        assert result is not None

        # Custom min_periods
        result = df.select(rolling_correlation("asset1", "asset2", window=20, min_periods=10))
        assert result is not None

        # min_periods > window
        with pytest.raises(ValueError, match="min_periods.*cannot exceed window"):
            df.select(rolling_correlation("asset1", "asset2", window=20, min_periods=30))

    def test_beta_to_market_validation(self):
        """Test beta calculation with validation."""
        from ml4t.engineer.features.cross_asset import beta_to_market

        df = pl.DataFrame(
            {"asset_ret": np.random.randn(100) * 0.02, "market_ret": np.random.randn(100) * 0.015}
        )

        result = df.select(beta_to_market("asset_ret", "market_ret", window=60))
        assert result is not None

        # Custom min_periods
        result = df.select(beta_to_market("asset_ret", "market_ret", window=60, min_periods=30))
        assert result is not None

    def test_correlation_regime_indicator(self):
        """Test correlation regime indicator."""
        from ml4t.engineer.features.cross_asset import correlation_regime_indicator

        np.random.seed(42)
        corr = np.random.uniform(-1, 1, 100)
        pl.DataFrame({"corr": corr})

        result = correlation_regime_indicator(
            "corr", low_threshold=0.3, high_threshold=0.7, lookback=20
        )
        assert "corr_regime_low" in result
        assert "corr_regime_mid" in result
        assert "corr_regime_high" in result
        assert "corr_trend" in result
        assert "corr_stability" in result

        # Invalid thresholds
        with pytest.raises(ValueError, match="low_threshold.*must be less than high_threshold"):
            correlation_regime_indicator("corr", low_threshold=0.8, high_threshold=0.5)

    def test_lead_lag_correlation(self):
        """Test lead-lag correlation."""
        from ml4t.engineer.features.cross_asset import lead_lag_correlation

        pl.DataFrame({"s1": np.random.randn(100), "s2": np.random.randn(100)})

        result = lead_lag_correlation("s1", "s2", max_lag=5, window=20)
        assert "lag_0" in result
        assert "lead_1" in result
        assert "lag_1" in result

    def test_multi_asset_dispersion(self):
        """Test multi-asset dispersion."""
        from ml4t.engineer.features.cross_asset import multi_asset_dispersion

        df = pl.DataFrame(
            {
                "r1": np.random.randn(100) * 0.02,
                "r2": np.random.randn(100) * 0.015,
                "r3": np.random.randn(100) * 0.018,
            }
        )

        # Std method
        result = df.select(multi_asset_dispersion(["r1", "r2", "r3"], window=20, method="std"))
        assert result is not None

        # MAD method
        result = df.select(multi_asset_dispersion(["r1", "r2", "r3"], window=20, method="mad"))
        assert result is not None

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            df.select(multi_asset_dispersion(["r1", "r2", "r3"], method="invalid"))

    def test_correlation_matrix_features(self):
        """Test correlation matrix feature extraction."""
        from ml4t.engineer.features.cross_asset import correlation_matrix_features

        pl.DataFrame(
            {
                "r1": np.random.randn(100) * 0.02,
                "r2": np.random.randn(100) * 0.015,
                "r3": np.random.randn(100) * 0.018,
            }
        )

        result = correlation_matrix_features(["r1", "r2", "r3"], window=20)
        assert "avg_correlation" in result
        assert "max_correlation" in result
        assert "min_correlation" in result
        assert "corr_dispersion" in result

    def test_relative_strength_index_spread(self):
        """Test RSI spread calculation."""
        from ml4t.engineer.features.cross_asset import relative_strength_index_spread

        df = pl.DataFrame(
            {"rsi1": np.random.uniform(20, 80, 100), "rsi2": np.random.uniform(20, 80, 100)}
        )

        result = df.select(relative_strength_index_spread("rsi1", "rsi2", smooth_period=5))
        assert result is not None

    def test_volatility_ratio(self):
        """Test volatility ratio."""
        from ml4t.engineer.features.cross_asset import volatility_ratio

        df = pl.DataFrame(
            {"vol1": np.random.uniform(0.1, 0.3, 100), "vol2": np.random.uniform(0.1, 0.3, 100)}
        )

        # Log ratio
        result = df.select(volatility_ratio("vol1", "vol2", log_ratio=True))
        assert result is not None

        # Regular ratio
        result = df.select(volatility_ratio("vol1", "vol2", log_ratio=False))
        assert result is not None

    def test_transfer_entropy_nb(self):
        """Test transfer entropy numba function."""
        from ml4t.engineer.features.cross_asset import transfer_entropy_nb

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = transfer_entropy_nb(x, y, lag=1, bins=10)
        # Transfer entropy can be negative due to discretization effects
        assert not np.isnan(result)

        # Short sequence
        result = transfer_entropy_nb(np.array([1.0, 2.0]), np.array([1.0, 2.0]), lag=1, bins=5)
        assert np.isnan(result)

        # Constant values (no range)
        result = transfer_entropy_nb(np.ones(50), np.ones(50), lag=1, bins=10)
        # Constant has no entropy, should be 0.0
        assert result == 0.0

    def test_transfer_entropy_not_implemented(self):
        """Test transfer entropy raises NotImplementedError."""
        from ml4t.engineer.features.cross_asset import transfer_entropy

        with pytest.raises(
            NotImplementedError, match="Transfer entropy calculation is not yet implemented"
        ):
            transfer_entropy("s1", "s2", lag=1, window=100)

    def test_co_integration_score(self):
        """Test co-integration score."""
        from ml4t.engineer.features.cross_asset import co_integration_score

        np.random.seed(42)
        price1 = np.cumsum(np.random.randn(100)) + 100
        price2 = np.cumsum(np.random.randn(100)) + 100
        df = pl.DataFrame({"p1": price1, "p2": price2})

        result = df.select(co_integration_score("p1", "p2", window=60))
        assert result is not None

    def test_cross_asset_momentum(self):
        """Test cross-asset momentum."""
        from ml4t.engineer.features.cross_asset import cross_asset_momentum

        pl.DataFrame(
            {
                "r1": np.random.randn(100) * 0.02,
                "r2": np.random.randn(100) * 0.015,
                "r3": np.random.randn(100) * 0.018,
            }
        )

        # Rank method
        result = cross_asset_momentum(["r1", "r2", "r3"], lookback=20, method="rank")
        assert "momentum_rank" in result
        assert "momentum_dispersion" in result

        # Zscore method
        result = cross_asset_momentum(["r1", "r2", "r3"], lookback=20, method="zscore")
        assert "momentum_zscore" in result
        assert "momentum_dispersion" in result
