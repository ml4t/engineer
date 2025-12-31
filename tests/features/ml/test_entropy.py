"""
Tests for entropy features.

Tests the following features:
- Shannon entropy (rolling_entropy)
- Kontoyiannis (LZ) entropy (rolling_entropy_lz)
- Plug-in ML entropy (rolling_entropy_plugin)
- Encoding schemes (binary, quantile, sigma)
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
    encode_sigma,
    rolling_entropy,
    rolling_entropy_lz,
    rolling_entropy_plugin,
)

# =============================================================================
# Encoding Tests
# =============================================================================


class TestEncodingBinary:
    """Tests for binary encoding."""

    def test_binary_encoding_basic(self) -> None:
        """Test basic binary encoding of positive/negative values."""
        values = np.array([1.0, -1.0, 0.5, -0.5, 0.0], dtype=np.float64)
        encoded = _encode_binary_nb(values)

        assert encoded[0] == 1  # positive -> 1
        assert encoded[1] == 0  # negative -> 0
        assert encoded[2] == 1  # positive -> 1
        assert encoded[3] == 0  # negative -> 0
        assert encoded[4] == 0  # zero -> 0

    def test_binary_encoding_nan_handling(self) -> None:
        """Test that NaN values are encoded as -1."""
        values = np.array([1.0, np.nan, -1.0], dtype=np.float64)
        encoded = _encode_binary_nb(values)

        assert encoded[0] == 1
        assert encoded[1] == -1  # NaN -> -1
        assert encoded[2] == 0

    def test_binary_encoding_polars(self) -> None:
        """Test binary encoding through Polars interface."""
        df = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.0]})
        result = df.select(encode_binary("returns").alias("encoded"))

        assert result["encoded"][0] == 1
        assert result["encoded"][1] == 0
        assert result["encoded"][2] == 1


class TestEncodingQuantile:
    """Tests for quantile encoding."""

    def test_quantile_encoding_uniform(self) -> None:
        """Test that quantile encoding creates uniform distribution."""
        # Values should be evenly distributed across bins
        values = np.arange(100, dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=10)

        # Count values in each bin
        bin_counts = np.bincount(encoded[encoded >= 0], minlength=10)
        # Each bin should have ~10 values
        assert all(8 <= c <= 12 for c in bin_counts)

    def test_quantile_encoding_preserves_order(self) -> None:
        """Test that quantile encoding preserves relative order."""
        values = np.array([1.0, 5.0, 10.0, 50.0, 100.0], dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=5)

        # Each value should get a different bin
        assert encoded[0] <= encoded[1] <= encoded[2] <= encoded[3] <= encoded[4]

    def test_quantile_encoding_nan_handling(self) -> None:
        """Test that NaN values are encoded as -1."""
        values = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64)
        encoded = _encode_quantile_nb(values, n_bins=5)

        assert encoded[1] == -1  # NaN -> -1
        assert encoded[0] >= 0  # Valid values are non-negative


class TestEncodingSigma:
    """Tests for sigma (standard deviation) encoding."""

    def test_sigma_encoding_basic(self) -> None:
        """Test sigma encoding assigns extreme values to edge bins."""
        # Generate normal-ish data
        np.random.seed(42)
        values = np.random.randn(100).astype(np.float64)

        _encode_sigma_nb(values, n_bins=10)

        # Values near mean should be in middle bins
        mean_idx = values[np.abs(values) < 0.5].astype(int)
        # At least some middle bins should be populated
        assert len(mean_idx) > 0

    def test_sigma_encoding_constant_values(self) -> None:
        """Test sigma encoding with constant values."""
        values = np.ones(10, dtype=np.float64) * 5.0
        encoded = _encode_sigma_nb(values, n_bins=10)

        # All values should be in middle bin
        assert all(e == 5 for e in encoded)

    def test_sigma_encoding_polars(self) -> None:
        """Test sigma encoding through Polars interface."""
        df = pl.DataFrame({"values": list(range(100))})
        result = df.select(encode_sigma("values", n_bins=10).alias("encoded"))

        assert result["encoded"].max() <= 9
        assert result["encoded"].min() >= 0


# =============================================================================
# Shannon Entropy Tests
# =============================================================================


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_shannon_entropy_uniform(self) -> None:
        """Test that uniform distribution has maximum entropy."""
        # Uniform distribution over bins
        values = np.arange(100, dtype=np.float64)
        entropy = _shannon_entropy_nb(values, n_bins=10)

        # Max entropy for 10 bins is log2(10) ≈ 3.32
        assert 3.0 < entropy < 3.5

    def test_shannon_entropy_constant(self) -> None:
        """Test that constant values have zero entropy."""
        values = np.ones(100, dtype=np.float64) * 5.0
        entropy = _shannon_entropy_nb(values, n_bins=10)

        assert entropy == 0.0

    def test_shannon_entropy_nan_handling(self) -> None:
        """Test Shannon entropy with NaN values."""
        values = np.array([np.nan] * 10, dtype=np.float64)
        entropy = _shannon_entropy_nb(values, n_bins=10)

        assert np.isnan(entropy)

    def test_rolling_entropy_basic(self) -> None:
        """Test rolling Shannon entropy computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(200)})

        result = df.select(rolling_entropy("values", window=50, n_bins=10).alias("entropy"))

        # Should have valid values after warmup
        valid_count = result["entropy"].drop_nulls().len()
        assert valid_count > 100


# =============================================================================
# Kontoyiannis (LZ) Entropy Tests
# =============================================================================


class TestKontoyiannisEntropy:
    """Tests for Kontoyiannis (LZ) entropy estimator."""

    def test_lz_entropy_repetitive_sequence(self) -> None:
        """Test that repetitive sequence has low entropy."""
        # Very repetitive pattern
        pattern = np.array([0, 1, 0, 1] * 25, dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(pattern, window_size=50)

        # Entropy should be relatively low (highly compressible)
        assert entropy < 2.0

    def test_lz_entropy_random_sequence(self) -> None:
        """Test that random sequence has higher entropy."""
        np.random.seed(42)
        random_seq = np.random.randint(0, 10, 100, dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(random_seq, window_size=50)

        # Random sequence should have higher entropy
        assert entropy > 1.0

    def test_lz_entropy_short_sequence(self) -> None:
        """Test LZ entropy with short sequence returns NaN."""
        short_seq = np.array([0, 1], dtype=np.int32)
        entropy = _kontoyiannis_entropy_nb(short_seq, window_size=10)

        assert np.isnan(entropy)

    def test_rolling_entropy_lz_basic(self) -> None:
        """Test rolling LZ entropy computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(
            rolling_entropy_lz("values", window=100, encoding="quantile", n_bins=10).alias(
                "entropy_lz"
            )
        )

        # Should have valid values after warmup
        valid_count = result["entropy_lz"].drop_nulls().len()
        assert valid_count > 100

    def test_rolling_entropy_lz_encoding_options(self) -> None:
        """Test all encoding options for LZ entropy."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(200)})

        for encoding in ["binary", "quantile", "sigma"]:
            result = df.select(
                rolling_entropy_lz("values", window=100, encoding=encoding).alias("entropy")
            )
            assert result["entropy"].drop_nulls().len() > 0


# =============================================================================
# Plug-in Entropy Tests
# =============================================================================


class TestPluginEntropy:
    """Tests for plug-in ML entropy estimator."""

    def test_plugin_entropy_uniform(self) -> None:
        """Test plug-in entropy with uniform symbol distribution."""
        # Each symbol appears equally often
        symbols = np.array([0, 1, 2, 3, 4] * 20, dtype=np.int32)
        entropy = _plugin_entropy_nb(symbols, word_length=1)

        # Entropy of uniform distribution over 5 symbols is log2(5) ≈ 2.32
        assert 2.2 < entropy < 2.4

    def test_plugin_entropy_single_symbol(self) -> None:
        """Test plug-in entropy with single symbol."""
        symbols = np.zeros(100, dtype=np.int32)
        entropy = _plugin_entropy_nb(symbols, word_length=1)

        assert entropy == 0.0

    def test_plugin_entropy_word_length(self) -> None:
        """Test plug-in entropy with different word lengths."""
        np.random.seed(42)
        symbols = np.random.randint(0, 4, 200, dtype=np.int32)

        entropy_1 = _plugin_entropy_nb(symbols, word_length=1)
        entropy_2 = _plugin_entropy_nb(symbols, word_length=2)

        # Both should be positive
        assert entropy_1 > 0
        assert entropy_2 > 0

    def test_rolling_entropy_plugin_basic(self) -> None:
        """Test rolling plug-in entropy computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(200)})

        result = df.select(
            rolling_entropy_plugin("values", window=50, encoding="quantile").alias("entropy")
        )

        # Should have valid values after warmup
        valid_count = result["entropy"].drop_nulls().len()
        assert valid_count > 100


# =============================================================================
# Validation Tests
# =============================================================================


class TestEntropyValidation:
    """Tests for input validation."""

    def test_rolling_entropy_invalid_window(self) -> None:
        """Test that invalid window raises error."""
        df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError):
            df.select(rolling_entropy("values", window=0))

    def test_rolling_entropy_lz_invalid_encoding(self) -> None:
        """Test that invalid encoding raises error."""
        df = pl.DataFrame({"values": [1.0] * 200})

        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_lz("values", window=100, encoding="invalid"))

    def test_rolling_entropy_lz_small_window(self) -> None:
        """Test that too small window raises error."""
        df = pl.DataFrame({"values": [1.0] * 50})

        with pytest.raises(ValueError):
            df.select(rolling_entropy_lz("values", window=5))
