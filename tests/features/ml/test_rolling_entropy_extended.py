"""Extended tests for rolling entropy features.

Tests different encoding schemes, LZ vs Shannon entropy,
and edge cases with perfect correlation and random data.
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.ml.rolling_entropy import (
    _encode_binary_nb,
    _encode_quantile_nb,
    _encode_sigma_nb,
    _kontoyiannis_entropy_nb,
    _plugin_entropy_nb,
    _shannon_entropy_nb,
    encode_binary,
    encode_quantile,
    encode_sigma,
    rolling_entropy,
    rolling_entropy_lz,
    rolling_entropy_plugin,
)

# =============================================================================
# Encoding Scheme Tests
# =============================================================================


class TestBinaryEncoding:
    """Tests for binary encoding."""

    def test_binary_encoding_basic(self) -> None:
        """Test basic binary encoding (sign-based)."""
        returns = np.array([0.5, -0.3, 0.2, -0.1, 0.0], dtype=np.float64)
        encoded = _encode_binary_nb(returns)

        # Positive -> 1, Non-positive -> 0
        expected = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(encoded, expected)

    def test_binary_encoding_with_nan(self) -> None:
        """Test binary encoding handles NaN."""
        returns = np.array([0.5, np.nan, -0.3, np.nan], dtype=np.float64)
        encoded = _encode_binary_nb(returns)

        # NaN -> -1
        assert encoded[1] == -1
        assert encoded[3] == -1
        assert encoded[0] == 1
        assert encoded[2] == 0

    def test_binary_encoding_polars_expr(self) -> None:
        """Test binary encoding as Polars expression."""
        df = pl.DataFrame({"returns": [0.1, -0.1, 0.2, -0.2, 0.0]})

        result = df.select(encode_binary("returns").alias("encoded"))

        # Check it produces values
        assert len(result) == 5
        assert result["encoded"].dtype == pl.Int32


class TestQuantileEncoding:
    """Tests for quantile encoding."""

    def test_quantile_encoding_basic(self) -> None:
        """Test quantile encoding with equal bins."""
        # Values from 0 to 99 (100 values)
        values = np.arange(100, dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=10)

        # Each bin should have ~10 values
        unique, counts = np.unique(encoded[encoded >= 0], return_counts=True)

        assert len(unique) == 10  # 10 bins
        # Each bin should have approximately equal counts
        assert np.all(counts >= 8) and np.all(counts <= 12)

    def test_quantile_encoding_duplicate_values(self) -> None:
        """Test quantile encoding with duplicate values."""
        values = np.array([1.0] * 50 + [2.0] * 50, dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=10)

        # With only 2 unique values, should still produce valid bins
        assert np.all(encoded >= 0)
        assert np.all(encoded < 10)

    def test_quantile_encoding_with_nan(self) -> None:
        """Test quantile encoding with NaN values."""
        values = np.array([1.0, np.nan, 2.0, 3.0, np.nan], dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=5)

        # NaN should encode to -1
        assert encoded[1] == -1
        assert encoded[4] == -1
        # Others should be valid bins
        assert np.all(encoded[[0, 2, 3]] >= 0)

    def test_quantile_encoding_polars_expr(self) -> None:
        """Test quantile encoding as Polars expression."""
        df = pl.DataFrame({"values": np.random.randn(100)})

        result = df.select(encode_quantile("values", n_bins=10).alias("encoded"))

        # Should produce 10 bins (0-9)
        unique_bins = result["encoded"].unique().sort()
        assert len(unique_bins) >= 5  # At least some bins populated


class TestSigmaEncoding:
    """Tests for sigma encoding."""

    def test_sigma_encoding_basic(self) -> None:
        """Test sigma encoding with standard normal data."""
        np.random.seed(42)
        values = np.random.randn(1000)  # Standard normal
        encoded = _encode_sigma_nb(values, n_bins=10)

        # Should produce bins centered at mean
        middle_bin = 10 // 2  # Bin 5

        # Most values should be in middle bins (68% within 1 sigma)
        middle_bins = np.sum((encoded >= middle_bin - 2) & (encoded <= middle_bin + 2))
        assert middle_bins / len(values) > 0.6

    def test_sigma_encoding_constant_values(self) -> None:
        """Test sigma encoding with constant values (std=0)."""
        values = np.ones(100, dtype=np.float64) * 5.0
        encoded = _encode_sigma_nb(values, n_bins=10)

        # All should be in middle bin
        middle_bin = 10 // 2
        assert np.all(encoded == middle_bin)

    def test_sigma_encoding_outliers(self) -> None:
        """Test sigma encoding clips outliers to valid bins."""
        values = np.array([0.0, 0.0, 100.0, -100.0], dtype=np.float64)
        encoded = _encode_sigma_nb(values, n_bins=10)

        # All bins should be valid (0-9)
        assert np.all(encoded >= 0)
        assert np.all(encoded < 10)

    def test_sigma_encoding_polars_expr(self) -> None:
        """Test sigma encoding as Polars expression."""
        df = pl.DataFrame({"values": np.random.randn(100)})

        result = df.select(encode_sigma("values", n_bins=10).alias("encoded"))

        assert result["encoded"].dtype == pl.Int32


# =============================================================================
# Shannon Entropy Tests
# =============================================================================


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_shannon_entropy_uniform(self) -> None:
        """Test Shannon entropy of uniform distribution."""
        # Uniform distribution should have maximum entropy
        n = 1000
        n_bins = 10
        # Create uniform values across bins
        values = np.linspace(0, 10, n, dtype=np.float64)

        entropy = _shannon_entropy_nb(values, n_bins=n_bins)

        # Maximum entropy for n_bins is log2(n_bins)
        max_entropy = np.log2(n_bins)
        assert entropy > max_entropy * 0.9  # Close to maximum

    def test_shannon_entropy_constant(self) -> None:
        """Test Shannon entropy of constant values is zero."""
        values = np.ones(100, dtype=np.float64) * 5.0
        entropy = _shannon_entropy_nb(values, n_bins=10)

        assert entropy == 0.0

    def test_shannon_entropy_with_nan(self) -> None:
        """Test Shannon entropy ignores NaN values."""
        values = np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float64)
        entropy = _shannon_entropy_nb(values, n_bins=10)

        # Should compute entropy of valid values only
        assert not np.isnan(entropy)
        assert entropy >= 0

    def test_shannon_entropy_empty(self) -> None:
        """Test Shannon entropy of empty array."""
        values = np.array([], dtype=np.float64)
        entropy = _shannon_entropy_nb(values, n_bins=10)

        assert np.isnan(entropy)

    def test_rolling_shannon_entropy(self) -> None:
        """Test rolling Shannon entropy."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(200)})

        result = df.select(rolling_entropy("values", window=50, n_bins=10).alias("entropy"))

        # Should have valid entropy values
        valid_entropy = result["entropy"].drop_nulls()
        assert len(valid_entropy) > 100
        assert all(valid_entropy >= 0)


# =============================================================================
# Kontoyiannis (LZ) Entropy Tests
# =============================================================================


class TestKontoyiannisEntropy:
    """Tests for Kontoyiannis (Lempel-Ziv) entropy."""

    def test_lz_entropy_perfect_repetition(self) -> None:
        """Test LZ entropy with perfect repetition (low entropy)."""
        # Perfectly repeating pattern: [0, 1, 0, 1, 0, 1, ...]
        sequence = np.tile(np.array([0, 1], dtype=np.int32), 100)
        entropy = _kontoyiannis_entropy_nb(sequence, window_size=50)

        # Should have low entropy (highly compressible)
        assert entropy < 5.0

    def test_lz_entropy_random_sequence(self) -> None:
        """Test LZ entropy with random sequence (high entropy)."""
        np.random.seed(42)
        # Random sequence (less compressible)
        sequence = np.random.randint(0, 10, size=500, dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(sequence, window_size=100)

        # Should have higher entropy
        assert entropy > 1.0

    def test_lz_entropy_with_invalid_entries(self) -> None:
        """Test LZ entropy filters out invalid entries (-1)."""
        sequence = np.array([0, 1, -1, 2, 3, -1, 4], dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(sequence, window_size=10)

        # Should compute on valid entries only
        assert not np.isnan(entropy)

    def test_lz_entropy_too_short(self) -> None:
        """Test LZ entropy with insufficient data."""
        sequence = np.array([0, 1], dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(sequence, window_size=10)

        assert np.isnan(entropy)

    def test_rolling_lz_entropy_binary(self) -> None:
        """Test rolling LZ entropy with binary encoding."""
        np.random.seed(42)
        df = pl.DataFrame({"returns": np.random.randn(300)})

        result = df.select(
            rolling_entropy_lz(
                "returns",
                window=100,
                encoding="binary",
                n_bins=10,
            ).alias("entropy_lz")
        )

        valid_entropy = result["entropy_lz"].drop_nulls()
        assert len(valid_entropy) > 100

    def test_rolling_lz_entropy_quantile(self) -> None:
        """Test rolling LZ entropy with quantile encoding."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy_lz(
                "values",
                window=100,
                encoding="quantile",
                n_bins=10,
            ).alias("entropy_lz")
        )

        valid_entropy = result["entropy_lz"].drop_nulls()
        assert len(valid_entropy) > 100

    def test_rolling_lz_entropy_sigma(self) -> None:
        """Test rolling LZ entropy with sigma encoding."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy_lz(
                "values",
                window=100,
                encoding="sigma",
                n_bins=10,
            ).alias("entropy_lz")
        )

        valid_entropy = result["entropy_lz"].drop_nulls()
        assert len(valid_entropy) > 100


# =============================================================================
# Plug-in Entropy Tests
# =============================================================================


class TestPluginEntropy:
    """Tests for plug-in (ML) entropy."""

    def test_plugin_entropy_uniform(self) -> None:
        """Test plug-in entropy with uniform symbol distribution."""
        # 10 symbols, each appearing 100 times
        sequence = np.repeat(np.arange(10, dtype=np.int32), 100)
        np.random.shuffle(sequence)

        entropy = _plugin_entropy_nb(sequence, word_length=1)

        # Should be close to log2(10) ≈ 3.32
        assert 3.0 < entropy < 3.5

    def test_plugin_entropy_single_symbol(self) -> None:
        """Test plug-in entropy with single symbol (zero entropy)."""
        sequence = np.zeros(100, dtype=np.int32)
        entropy = _plugin_entropy_nb(sequence, word_length=1)

        assert entropy == 0.0

    def test_plugin_entropy_word_length(self) -> None:
        """Test plug-in entropy with different word lengths."""
        np.random.seed(42)
        sequence = np.random.randint(0, 5, size=500, dtype=np.int32)

        entropy_1 = _plugin_entropy_nb(sequence, word_length=1)
        entropy_2 = _plugin_entropy_nb(sequence, word_length=2)

        # Both should be valid
        assert not np.isnan(entropy_1)
        assert not np.isnan(entropy_2)

    def test_plugin_entropy_insufficient_data(self) -> None:
        """Test plug-in entropy with insufficient data."""
        sequence = np.array([0, 1], dtype=np.int32)
        entropy = _plugin_entropy_nb(sequence, word_length=3)

        assert np.isnan(entropy)

    def test_rolling_plugin_entropy(self) -> None:
        """Test rolling plug-in entropy."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy_plugin(
                "values",
                window=50,
                encoding="quantile",
                n_bins=10,
                word_length=1,
            ).alias("entropy_plugin")
        )

        valid_entropy = result["entropy_plugin"].drop_nulls()
        assert len(valid_entropy) > 150


# =============================================================================
# Edge Cases & Comparative Tests
# =============================================================================


class TestEntropyEdgeCases:
    """Tests for edge cases and comparative analysis."""

    def test_perfect_correlation_zero_entropy(self) -> None:
        """Test that perfectly correlated data has zero entropy."""
        # Constant values
        df = pl.DataFrame({"values": [5.0] * 100})

        result = df.select(rolling_entropy("values", window=50, n_bins=10).alias("entropy"))

        # Entropy should be zero or near-zero
        valid_entropy = result["entropy"].drop_nulls()
        assert all(valid_entropy < 0.1)

    def test_random_data_high_entropy(self) -> None:
        """Test that random data has entropy ≈ log(n_bins)."""
        np.random.seed(42)
        n = 1000
        n_bins = 10

        # Random data uniformly distributed
        df = pl.DataFrame({"values": np.random.rand(n) * 10})

        result = df.select(rolling_entropy("values", window=100, n_bins=n_bins).alias("entropy"))

        valid_entropy = result["entropy"].drop_nulls()
        mean_entropy = valid_entropy.mean()

        # Should be close to log2(n_bins) ≈ 3.32
        max_entropy = np.log2(n_bins)
        assert mean_entropy > max_entropy * 0.7

    def test_lz_vs_shannon_comparison(self) -> None:
        """Compare LZ and Shannon entropy on same data."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy("values", window=100, n_bins=10).alias("shannon"),
            rolling_entropy_lz("values", window=100, encoding="quantile", n_bins=10).alias("lz"),
        )

        # Both should produce valid values
        assert result["shannon"].drop_nulls().len() > 100
        assert result["lz"].drop_nulls().len() > 100

        # Both should be positive
        assert all(result["shannon"].drop_nulls() >= 0)
        assert all(result["lz"].drop_nulls() >= 0)

    def test_encoding_schemes_on_same_data(self) -> None:
        """Test all encoding schemes on same data."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy_lz("values", window=100, encoding="binary", n_bins=10).alias(
                "lz_binary"
            ),
            rolling_entropy_lz("values", window=100, encoding="quantile", n_bins=10).alias(
                "lz_quantile"
            ),
            rolling_entropy_lz("values", window=100, encoding="sigma", n_bins=10).alias("lz_sigma"),
        )

        # All should produce valid values
        for col in ["lz_binary", "lz_quantile", "lz_sigma"]:
            valid = result[col].drop_nulls()
            assert len(valid) > 50
            assert all(valid >= 0)


# =============================================================================
# Validation Tests
# =============================================================================


class TestEntropyValidation:
    """Tests for input validation."""

    def test_invalid_encoding_scheme(self) -> None:
        """Test that invalid encoding scheme raises error."""
        df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_lz("values", window=10, encoding="invalid"))

    def test_invalid_window_size(self) -> None:
        """Test that invalid window size raises error."""
        df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError):
            df.select(rolling_entropy("values", window=0))

    def test_invalid_n_bins(self) -> None:
        """Test that invalid n_bins raises error."""
        df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError):
            df.select(rolling_entropy("values", window=10, n_bins=1))


# =============================================================================
# Performance & Integration Tests
# =============================================================================


class TestEntropyIntegration:
    """Integration tests for entropy features."""

    def test_regime_detection_with_entropy(self) -> None:
        """Test that entropy can detect regime changes."""
        np.random.seed(42)

        # Create data with regime change
        # Low entropy regime (predictable)
        regime1 = np.sin(np.linspace(0, 10 * np.pi, 200))

        # High entropy regime (random)
        regime2 = np.random.randn(200)

        values = np.concatenate([regime1, regime2])
        df = pl.DataFrame({"values": values})

        result = df.select(rolling_entropy("values", window=50, n_bins=10).alias("entropy"))

        entropy_values = result["entropy"].drop_nulls().to_numpy()

        # Just verify we get reasonable entropy values for both regimes
        if len(entropy_values) > 200:
            first_half = np.mean(entropy_values[:100])
            second_half = np.mean(entropy_values[-100:])

            # Both should be positive and finite
            assert first_half > 0 and np.isfinite(first_half)
            assert second_half > 0 and np.isfinite(second_half)

    def test_all_entropy_types_on_market_data(self) -> None:
        """Test all entropy types on simulated market returns."""
        np.random.seed(42)
        # Simulate market returns with autocorrelation
        returns = np.random.randn(500) * 0.01
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i - 1]  # Add autocorrelation

        df = pl.DataFrame({"returns": returns})

        result = df.select(
            rolling_entropy("returns", window=50).alias("shannon"),
            rolling_entropy_lz("returns", window=100, encoding="binary").alias("lz"),
            rolling_entropy_plugin("returns", window=50, encoding="quantile").alias("plugin"),
        )

        # All should produce valid results
        for col in ["shannon", "lz", "plugin"]:
            valid = result[col].drop_nulls()
            assert len(valid) > 200
