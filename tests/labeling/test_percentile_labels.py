"""Tests for rolling percentile-based binary labels."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.labeling.percentile_labels import (
    compute_label_statistics,
    rolling_percentile_binary_labels,
    rolling_percentile_multi_labels,
)


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 500
    # Create trending price data
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "close": prices,
            "session_date": [f"2024-01-{(i // 100) + 1:02d}" for i in range(n)],
        }
    )


class TestRollingPercentileBinaryLabels:
    """Tests for rolling_percentile_binary_labels function."""

    def test_basic_long_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic long label generation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
        )

        # Check that expected columns are added
        assert "forward_return_10" in result.columns
        assert "threshold_p90_h10" in result.columns
        assert "label_long_p90_h10" in result.columns

        # Labels should be 0 or 1 (or null)
        labels = result["label_long_p90_h10"].drop_nulls()
        assert all(label in [0, 1] for label in labels)

    def test_basic_short_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic short label generation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=10,
            direction="short",
            lookback_window=100,
        )

        # Check that expected columns are added
        assert "forward_return_10" in result.columns
        assert "threshold_p10_h10" in result.columns
        assert "label_short_p10_h10" in result.columns

        # Labels should be 0 or 1 (or null)
        labels = result["label_short_p10_h10"].drop_nulls()
        assert all(label in [0, 1] for label in labels)

    def test_session_aware_returns(self, sample_price_data: pl.DataFrame) -> None:
        """Test session-aware forward return computation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
            session_col="session_date",
        )

        # Should have forward returns column
        assert "forward_return_10" in result.columns

    def test_invalid_direction_raises(self, sample_price_data: pl.DataFrame) -> None:
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            rolling_percentile_binary_labels(
                sample_price_data,
                horizon=10,
                percentile=90,
                direction="invalid",  # type: ignore
                lookback_window=100,
            )

    def test_custom_min_samples(self, sample_price_data: pl.DataFrame) -> None:
        """Test custom min_samples parameter."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
            min_samples=50,
        )

        # Should complete without error
        assert "label_long_p90_h10" in result.columns

    def test_high_percentile_long_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test high percentile creates sparse long labels."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=95,
            direction="long",
            lookback_window=100,
        )

        labels = result["label_long_p95_h10"].drop_nulls()
        # With 95th percentile, should have ~5% positive labels
        positive_rate = labels.mean()
        # Allow some tolerance due to rolling window effects
        assert 0 < positive_rate < 0.30

    def test_low_percentile_short_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test low percentile creates sparse short labels."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=5,
            direction="short",
            lookback_window=100,
        )

        labels = result["label_short_p5_h10"].drop_nulls()
        # With 5th percentile, should have ~5% positive labels
        positive_rate = labels.mean()
        assert 0 < positive_rate < 0.30


class TestRollingPercentileMultiLabels:
    """Tests for rolling_percentile_multi_labels function."""

    def test_multiple_horizons(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple horizons."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10, 20],
            percentiles=[90],
            direction="long",
            lookback_window=100,
        )

        # Should have columns for both horizons
        assert "label_long_p90_h10" in result.columns
        assert "label_long_p90_h20" in result.columns

    def test_multiple_percentiles(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple percentiles."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10],
            percentiles=[90, 95],
            direction="long",
            lookback_window=100,
        )

        # Should have columns for both percentiles
        assert "label_long_p90_h10" in result.columns
        assert "label_long_p95_h10" in result.columns

    def test_multiple_horizons_and_percentiles(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple horizons and percentiles."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10, 20],
            percentiles=[90, 95],
            direction="long",
            lookback_window=100,
        )

        # Should have 4 label columns (2 horizons × 2 percentiles)
        label_cols = [c for c in result.columns if c.startswith("label_")]
        assert len(label_cols) == 4


class TestComputeLabelStatistics:
    """Tests for compute_label_statistics function."""

    def test_basic_statistics(self) -> None:
        """Test basic label statistics computation."""
        df = pl.DataFrame(
            {
                "label": [1, 0, 1, 0, 0, 1, None, None],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 8
        assert stats["positive_labels"] == 3
        assert stats["negative_labels"] == 3
        assert stats["null_labels"] == 2
        assert stats["positive_rate"] == 50.0
        assert stats["null_rate"] == 25.0

    def test_all_nulls(self) -> None:
        """Test statistics with all null labels."""
        df = pl.DataFrame(
            {
                "label": [None, None, None],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 3
        assert stats["positive_labels"] == 0
        assert stats["negative_labels"] == 0
        assert stats["null_labels"] == 3
        assert stats["positive_rate"] == 0.0
        assert stats["null_rate"] == 100.0

    def test_no_nulls(self) -> None:
        """Test statistics with no nulls."""
        df = pl.DataFrame(
            {
                "label": [1, 0, 1, 1, 0],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 5
        assert stats["null_labels"] == 0
        assert stats["null_rate"] == 0.0
        assert stats["positive_rate"] == 60.0
