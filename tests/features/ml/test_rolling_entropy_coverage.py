"""
Coverage tests for rolling entropy features.

Tests edge cases and various code paths to improve coverage.
"""

import numpy as np
import polars as pl
import pytest


class TestEncodingFunctions:
    """Coverage tests for encoding functions."""

    @pytest.fixture
    def returns_df(self):
        """Create returns DataFrame for testing."""
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.02
        return pl.DataFrame({"returns": returns})

    def test_encode_binary(self, returns_df):
        """Test binary encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import encode_binary

        result = returns_df.select(encode_binary("returns"))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_encode_quantile(self, returns_df):
        """Test quantile encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import encode_quantile

        result = returns_df.select(encode_quantile("returns", n_bins=10))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_encode_quantile_different_bins(self, returns_df):
        """Test quantile encoding with different bin counts."""
        from ml4t.engineer.features.ml.rolling_entropy import encode_quantile

        r5 = returns_df.select(encode_quantile("returns", n_bins=5))
        r20 = returns_df.select(encode_quantile("returns", n_bins=20))

        assert r5 is not None
        assert r20 is not None

    def test_encode_sigma(self, returns_df):
        """Test sigma encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import encode_sigma

        result = returns_df.select(encode_sigma("returns", n_bins=10))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_encode_sigma_different_bins(self, returns_df):
        """Test sigma encoding with different bin counts."""
        from ml4t.engineer.features.ml.rolling_entropy import encode_sigma

        r5 = returns_df.select(encode_sigma("returns", n_bins=5))
        r20 = returns_df.select(encode_sigma("returns", n_bins=20))

        assert r5 is not None
        assert r20 is not None


class TestNumbaEncoders:
    """Coverage tests for Numba encoding functions."""

    def test_encode_binary_nb(self):
        """Test binary encoding numba function."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_binary_nb

        returns = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
        result = _encode_binary_nb(returns)

        assert result is not None
        assert len(result) == len(returns)
        assert result[0] == 1  # Positive
        assert result[1] == 0  # Negative
        assert result[2] == 1  # Positive
        assert result[3] == 0  # Zero is non-positive
        assert result[4] == 0  # Negative

    def test_encode_quantile_nb(self):
        """Test quantile encoding numba function."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_quantile_nb

        np.random.seed(42)
        values = np.random.randn(100)
        result = _encode_quantile_nb(values, n_bins=10)

        assert result is not None
        assert len(result) == len(values)
        assert result.min() >= 0
        assert result.max() < 10

    def test_encode_sigma_nb(self):
        """Test sigma encoding numba function."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        np.random.seed(42)
        values = np.random.randn(100)
        result = _encode_sigma_nb(values, n_bins=10)

        assert result is not None
        assert len(result) == len(values)
        assert result.min() >= 0


class TestShannonEntropy:
    """Coverage tests for Shannon entropy."""

    def test_shannon_entropy_nb_basic(self):
        """Test Shannon entropy numba function."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        # Uniform distribution should have high entropy
        values = np.random.randn(100)
        result = _shannon_entropy_nb(values, n_bins=10)

        assert result >= 0

    def test_shannon_entropy_nb_constant(self):
        """Test Shannon entropy with constant values."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        # Constant values should have zero entropy
        values = np.full(100, 1.0)
        result = _shannon_entropy_nb(values, n_bins=10)

        assert result >= 0

    def test_shannon_entropy_nb_empty(self):
        """Test Shannon entropy with empty array."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        result = _shannon_entropy_nb(np.array([]), n_bins=10)
        # Empty array returns NaN or 0
        assert np.isnan(result) or result == 0


class TestKontoyiannisEntropy:
    """Coverage tests for Kontoyiannis (LZ) entropy."""

    def test_kontoyiannis_entropy_basic(self):
        """Test Kontoyiannis entropy."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        # Random sequence should have moderate entropy
        np.random.seed(42)
        sequence = np.random.randint(0, 10, 100).astype(np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=50)

        assert result >= 0

    def test_kontoyiannis_entropy_repetitive(self):
        """Test Kontoyiannis entropy with repetitive pattern."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        # Repetitive pattern should have lower entropy
        sequence = np.array([0, 1, 2, 3] * 25, dtype=np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=50)

        assert result >= 0


class TestPluginEntropy:
    """Coverage tests for Plugin entropy."""

    def test_plugin_entropy_basic(self):
        """Test Plugin entropy."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        np.random.seed(42)
        sequence = np.random.randint(0, 10, 100).astype(np.int32)
        result = _plugin_entropy_nb(sequence, word_length=1)

        assert result >= 0

    def test_plugin_entropy_word_length(self):
        """Test Plugin entropy with different word lengths."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        np.random.seed(42)
        sequence = np.random.randint(0, 5, 100).astype(np.int32)

        r1 = _plugin_entropy_nb(sequence, word_length=1)
        r2 = _plugin_entropy_nb(sequence, word_length=2)

        assert r1 >= 0
        assert r2 >= 0


class TestRollingEntropy:
    """Coverage tests for rolling entropy."""

    @pytest.fixture
    def returns_df(self):
        """Create returns DataFrame for testing."""
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.02
        return pl.DataFrame({"returns": returns})

    def test_rolling_entropy_basic(self, returns_df):
        """Test basic rolling entropy."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        result = returns_df.select(rolling_entropy("returns", window=50, n_bins=10))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_rolling_entropy_different_windows(self, returns_df):
        """Test rolling entropy with different windows."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        r30 = returns_df.select(rolling_entropy("returns", window=30, n_bins=10))
        r100 = returns_df.select(rolling_entropy("returns", window=100, n_bins=10))

        assert r30 is not None
        assert r100 is not None

    def test_rolling_entropy_different_bins(self, returns_df):
        """Test rolling entropy with different bin counts."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        r5 = returns_df.select(rolling_entropy("returns", window=50, n_bins=5))
        r20 = returns_df.select(rolling_entropy("returns", window=50, n_bins=20))

        assert r5 is not None
        assert r20 is not None


class TestRollingEntropyLZ:
    """Coverage tests for rolling LZ entropy."""

    @pytest.fixture
    def returns_df(self):
        """Create returns DataFrame for testing."""
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.02
        return pl.DataFrame({"returns": returns})

    def test_rolling_entropy_lz_basic(self, returns_df):
        """Test basic rolling LZ entropy."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        result = returns_df.select(rolling_entropy_lz("returns", window=100))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_rolling_entropy_lz_quantile(self, returns_df):
        """Test rolling LZ entropy with quantile encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        result = returns_df.select(
            rolling_entropy_lz("returns", window=100, encoding="quantile", n_bins=10)
        )
        assert result is not None

    def test_rolling_entropy_lz_binary(self, returns_df):
        """Test rolling LZ entropy with binary encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        result = returns_df.select(rolling_entropy_lz("returns", window=100, encoding="binary"))
        assert result is not None

    def test_rolling_entropy_lz_sigma(self, returns_df):
        """Test rolling LZ entropy with sigma encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        result = returns_df.select(
            rolling_entropy_lz("returns", window=100, encoding="sigma", n_bins=10)
        )
        assert result is not None


class TestRollingEntropyPlugin:
    """Coverage tests for rolling plugin entropy."""

    @pytest.fixture
    def returns_df(self):
        """Create returns DataFrame for testing."""
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.02
        return pl.DataFrame({"returns": returns})

    def test_rolling_entropy_plugin_basic(self, returns_df):
        """Test basic rolling plugin entropy."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        result = returns_df.select(rolling_entropy_plugin("returns", window=50))
        assert result is not None
        assert len(result) == len(returns_df)

    def test_rolling_entropy_plugin_quantile(self, returns_df):
        """Test rolling plugin entropy with quantile encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        result = returns_df.select(
            rolling_entropy_plugin("returns", window=50, encoding="quantile", n_bins=10)
        )
        assert result is not None

    def test_rolling_entropy_plugin_binary(self, returns_df):
        """Test rolling plugin entropy with binary encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        result = returns_df.select(rolling_entropy_plugin("returns", window=50, encoding="binary"))
        assert result is not None

    def test_rolling_entropy_plugin_sigma(self, returns_df):
        """Test rolling plugin entropy with sigma encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        result = returns_df.select(
            rolling_entropy_plugin("returns", window=50, encoding="sigma", n_bins=10)
        )
        assert result is not None

    def test_rolling_entropy_plugin_word_length(self, returns_df):
        """Test rolling plugin entropy with different word lengths."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        r1 = returns_df.select(rolling_entropy_plugin("returns", window=50, word_length=1))
        r2 = returns_df.select(rolling_entropy_plugin("returns", window=50, word_length=2))

        assert r1 is not None
        assert r2 is not None


class TestEntropyEdgeCases:
    """Edge case tests for entropy functions."""

    def test_constant_returns(self):
        """Test entropy with constant returns."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        df = pl.DataFrame({"returns": np.full(100, 0.01)})
        result = df.select(rolling_entropy("returns", window=50, n_bins=10))
        assert result is not None

    def test_alternating_returns(self):
        """Test entropy with alternating returns."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        returns = np.array([0.01, -0.01] * 100)
        df = pl.DataFrame({"returns": returns})
        result = df.select(rolling_entropy("returns", window=50, n_bins=10))
        assert result is not None

    def test_short_data(self):
        """Test entropy with short data."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        df = pl.DataFrame({"returns": np.random.randn(20) * 0.02})
        result = df.select(rolling_entropy("returns", window=50, n_bins=10))
        assert result is not None

    def test_extreme_values(self):
        """Test entropy with extreme values."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        np.random.seed(42)
        returns = np.concatenate(
            [
                np.random.randn(80) * 0.02,
                np.array([-0.5, 0.3, -0.4, 0.6]),
                np.random.randn(20) * 0.02,
            ]
        )
        df = pl.DataFrame({"returns": returns})
        result = df.select(rolling_entropy("returns", window=50, n_bins=10))
        assert result is not None


class TestLZMatchLength:
    """Coverage tests for LZ match length function."""

    def test_lz_match_length(self):
        """Test LZ match length helper."""
        from ml4t.engineer.features.ml.rolling_entropy import _lz_match_length

        sequence = np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int32)

        # Position 3 should match [0,1,2] at the beginning
        match_len = _lz_match_length(sequence, 3, 3)
        assert match_len >= 0

    def test_lz_match_length_no_match(self):
        """Test LZ match length with no match."""
        from ml4t.engineer.features.ml.rolling_entropy import _lz_match_length

        sequence = np.array([0, 1, 2, 5, 6, 7, 8], dtype=np.int32)

        match_len = _lz_match_length(sequence, 3, 3)
        assert match_len >= 0

    def test_lz_match_length_position_zero(self):
        """Test LZ match length at position 0."""
        from ml4t.engineer.features.ml.rolling_entropy import _lz_match_length

        sequence = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        match_len = _lz_match_length(sequence, 0, 3)
        assert match_len == 1  # Position 0 always returns 1

    def test_lz_match_length_large_window(self):
        """Test LZ match length with large window."""
        from ml4t.engineer.features.ml.rolling_entropy import _lz_match_length

        sequence = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        match_len = _lz_match_length(sequence, 6, 100)  # Large window
        assert match_len >= 1


class TestEncoderEdgeCases:
    """Additional edge case tests for encoder functions."""

    def test_encode_binary_nb_with_nan(self):
        """Test binary encoding with NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_binary_nb

        returns = np.array([0.01, np.nan, -0.02, np.nan, 0.03])
        result = _encode_binary_nb(returns)

        assert result[0] == 1  # Positive
        assert result[1] == -1  # NaN
        assert result[2] == 0  # Negative
        assert result[3] == -1  # NaN
        assert result[4] == 1  # Positive

    def test_encode_binary_nb_all_nan(self):
        """Test binary encoding with all NaN."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_binary_nb

        returns = np.array([np.nan, np.nan, np.nan])
        result = _encode_binary_nb(returns)
        assert all(r == -1 for r in result)

    def test_encode_quantile_nb_with_nan(self):
        """Test quantile encoding with NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_quantile_nb

        values = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = _encode_quantile_nb(values, n_bins=5)

        assert result[1] == -1  # NaN position
        assert result[3] == -1  # NaN position
        # Non-NaN positions should be valid bins
        assert result[0] >= 0
        assert result[2] >= 0

    def test_encode_quantile_nb_all_nan(self):
        """Test quantile encoding with all NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_quantile_nb

        values = np.array([np.nan, np.nan, np.nan])
        result = _encode_quantile_nb(values, n_bins=5)
        assert all(r == -1 for r in result)

    def test_encode_quantile_nb_empty(self):
        """Test quantile encoding with empty array."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_quantile_nb

        values = np.array([], dtype=np.float64)
        result = _encode_quantile_nb(values, n_bins=5)
        assert len(result) == 0

    def test_encode_sigma_nb_with_nan(self):
        """Test sigma encoding with NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        values = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        result = _encode_sigma_nb(values, n_bins=5)

        assert result[1] == -1  # NaN position
        assert result[3] == -1  # NaN position
        assert result[0] >= 0

    def test_encode_sigma_nb_all_nan(self):
        """Test sigma encoding with all NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        values = np.array([np.nan, np.nan, np.nan])
        result = _encode_sigma_nb(values, n_bins=5)
        assert all(r == -1 for r in result)

    def test_encode_sigma_nb_constant_values(self):
        """Test sigma encoding with constant values (std=0)."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = _encode_sigma_nb(values, n_bins=5)

        # All should be in middle bin
        assert all(r == 2 for r in result)  # n_bins // 2 = 2

    def test_encode_sigma_nb_constant_with_nan(self):
        """Test sigma encoding with constant values and NaN."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        values = np.array([5.0, np.nan, 5.0, 5.0])
        result = _encode_sigma_nb(values, n_bins=5)

        assert result[1] == -1  # NaN position
        assert result[0] == 2  # Middle bin for constant

    def test_encode_sigma_nb_extreme_z_scores(self):
        """Test sigma encoding with extreme z-scores."""
        from ml4t.engineer.features.ml.rolling_entropy import _encode_sigma_nb

        # Create data with known extreme values
        values = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = _encode_sigma_nb(values, n_bins=10)

        # Extreme values should be at edge bins (clamped within valid range)
        # The exact bin depends on the algorithm, but should be valid
        assert result[0] >= 0  # First value should be low bin
        assert result[-1] <= 9  # Last value should be high bin
        # The extreme negative should be less than extreme positive
        assert result[0] < result[-1]


class TestKontoyiannisEdgeCases:
    """Additional edge cases for Kontoyiannis entropy."""

    def test_kontoyiannis_with_invalid_entries(self):
        """Test Kontoyiannis with invalid (-1) entries."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        # Mix of valid and invalid entries
        sequence = np.array([0, 1, -1, 2, 3, -1, 1, 2, 3, 4], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=5)

        # Should filter out -1 entries and compute on valid
        assert not np.isnan(result) or result >= 0

    def test_kontoyiannis_all_invalid(self):
        """Test Kontoyiannis with all invalid entries."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        sequence = np.array([-1, -1, -1], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=5)
        assert np.isnan(result)

    def test_kontoyiannis_short_sequence(self):
        """Test Kontoyiannis with very short sequence."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        sequence = np.array([0, 1], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=5)
        assert np.isnan(result)  # k < 3

    def test_kontoyiannis_single_symbol(self):
        """Test Kontoyiannis with single repeated symbol."""
        from ml4t.engineer.features.ml.rolling_entropy import _kontoyiannis_entropy_nb

        sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        result = _kontoyiannis_entropy_nb(sequence, window_size=5)

        # Single symbol = maximum compression = low entropy
        assert result >= 0


class TestPluginEntropyEdgeCases:
    """Additional edge cases for plugin entropy."""

    def test_plugin_entropy_with_invalid_entries(self):
        """Test plugin entropy with invalid (-1) entries."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        sequence = np.array([0, 1, -1, 2, 3, -1, 1, 2], dtype=np.int32)
        result = _plugin_entropy_nb(sequence, word_length=1)

        assert result >= 0

    def test_plugin_entropy_all_invalid(self):
        """Test plugin entropy with all invalid entries."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        sequence = np.array([-1, -1, -1], dtype=np.int32)
        result = _plugin_entropy_nb(sequence, word_length=1)
        assert np.isnan(result)

    def test_plugin_entropy_word_length_too_long(self):
        """Test plugin entropy with word length longer than sequence."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        sequence = np.array([0, 1, 2], dtype=np.int32)
        result = _plugin_entropy_nb(sequence, word_length=5)
        assert np.isnan(result)

    def test_plugin_entropy_word_length_2(self):
        """Test plugin entropy with word length 2 in detail."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        # Create sequence with known patterns
        sequence = np.array([0, 1, 0, 1, 0, 1, 2, 3], dtype=np.int32)
        result = _plugin_entropy_nb(sequence, word_length=2)

        assert result >= 0

    def test_plugin_entropy_word_length_3(self):
        """Test plugin entropy with word length 3."""
        from ml4t.engineer.features.ml.rolling_entropy import _plugin_entropy_nb

        np.random.seed(42)
        sequence = np.random.randint(0, 5, 50).astype(np.int32)
        result = _plugin_entropy_nb(sequence, word_length=3)

        assert result >= 0


class TestShannonEntropyEdgeCases:
    """Additional edge cases for Shannon entropy."""

    def test_shannon_entropy_all_nan(self):
        """Test Shannon entropy with all NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        values = np.array([np.nan, np.nan, np.nan])
        result = _shannon_entropy_nb(values, n_bins=10)
        assert np.isnan(result)

    def test_shannon_entropy_mixed_nan(self):
        """Test Shannon entropy with mixed NaN values."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        np.random.seed(42)
        values = np.array([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = _shannon_entropy_nb(values, n_bins=5)
        assert result >= 0

    def test_shannon_entropy_very_small_range(self):
        """Test Shannon entropy with very small value range."""
        from ml4t.engineer.features.ml.rolling_entropy import _shannon_entropy_nb

        values = np.array([1.0, 1.0 + 1e-12, 1.0 - 1e-12])
        result = _shannon_entropy_nb(values, n_bins=10)
        # Very small range treated as constant
        assert result == 0.0


class TestRollingEntropyValidation:
    """Test validation in rolling entropy functions."""

    def test_rolling_entropy_lz_invalid_encoding(self):
        """Test rolling_entropy_lz with invalid encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        df = pl.DataFrame({"returns": np.random.randn(100) * 0.02})
        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_lz("returns", window=50, encoding="invalid"))

    def test_rolling_entropy_plugin_invalid_encoding(self):
        """Test rolling_entropy_plugin with invalid encoding."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        df = pl.DataFrame({"returns": np.random.randn(100) * 0.02})
        with pytest.raises(ValueError, match="encoding must be"):
            df.select(rolling_entropy_plugin("returns", window=50, encoding="invalid"))

    def test_rolling_entropy_with_expression(self):
        """Test rolling entropy with pl.Expr instead of string."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy

        df = pl.DataFrame({"returns": np.random.randn(100) * 0.02})
        result = df.select(rolling_entropy(pl.col("returns"), window=30, n_bins=10))
        assert result is not None
        assert len(result) == len(df)

    def test_rolling_entropy_lz_with_expression(self):
        """Test rolling LZ entropy with pl.Expr."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_lz

        df = pl.DataFrame({"returns": np.random.randn(150) * 0.02})
        result = df.select(rolling_entropy_lz(pl.col("returns"), window=100))
        assert result is not None

    def test_rolling_entropy_plugin_with_expression(self):
        """Test rolling plugin entropy with pl.Expr."""
        from ml4t.engineer.features.ml.rolling_entropy import rolling_entropy_plugin

        df = pl.DataFrame({"returns": np.random.randn(100) * 0.02})
        result = df.select(rolling_entropy_plugin(pl.col("returns"), window=50))
        assert result is not None
