"""Tests for newly implemented labeling methods.

Tests fixed_time_horizon_labels and trend_scanning_labels.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.labeling import (
    fixed_time_horizon_labels,
    trend_scanning_labels,
)


@pytest.fixture
def sample_prices():
    """Generate simple price series for testing."""
    np.random.seed(42)
    n = 100
    base_price = 100.0
    returns = np.random.normal(0.001, 0.01, n)
    prices = base_price * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)],
            "close": prices,
        }
    )


@pytest.fixture
def trending_up():
    """Generate clear uptrend."""
    prices = np.linspace(100, 120, 50)
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)],
            "close": prices,
        }
    )


@pytest.fixture
def trending_down():
    """Generate clear downtrend."""
    prices = np.linspace(120, 100, 50)
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)],
            "close": prices,
        }
    )


class TestFixedTimeHorizonLabels:
    """Test fixed time horizon labeling."""

    def test_returns_basic(self, sample_prices):
        """Test basic returns calculation."""
        result = fixed_time_horizon_labels(
            sample_prices,
            horizon=1,
            method="returns",
        )

        # Should have label column
        assert "label_return_1p" in result.columns

        # Last value should be null (no future data)
        assert result["label_return_1p"][-1] is None

        # First N-1 values should be non-null
        valid_labels = result["label_return_1p"][:-1].drop_nulls()
        assert len(valid_labels) > 0

    def test_returns_calculation_correctness(self):
        """Test that returns are calculated correctly."""
        # Simple test data: prices go 100 -> 102 -> 101
        df = pl.DataFrame(
            {
                "close": [100.0, 102.0, 101.0, 103.0],
            }
        )

        result = fixed_time_horizon_labels(df, horizon=1, method="returns")

        # Check specific return calculations
        # Row 0: (102 - 100) / 100 = 0.02
        assert abs(result["label_return_1p"][0] - 0.02) < 1e-10

        # Row 1: (101 - 102) / 102 ≈ -0.0098
        expected = (101 - 102) / 102
        assert abs(result["label_return_1p"][1] - expected) < 1e-10

        # Row 2: (103 - 101) / 101 ≈ 0.0198
        expected = (103 - 101) / 101
        assert abs(result["label_return_1p"][2] - expected) < 1e-10

        # Row 3: null (no future data)
        assert result["label_return_1p"][3] is None

    def test_log_returns(self, sample_prices):
        """Test log returns method."""
        result = fixed_time_horizon_labels(
            sample_prices,
            horizon=5,
            method="log_returns",
        )

        assert "label_log_return_5p" in result.columns

        # Last 5 values should be null
        nulls = result["label_log_return_5p"][-5:].null_count()
        assert nulls == 5

    def test_binary_labels(self, trending_up):
        """Test binary classification labels."""
        result = fixed_time_horizon_labels(
            trending_up,
            horizon=1,
            method="binary",
        )

        assert "label_direction_1p" in result.columns

        # In uptrend, most labels should be 1 (up)
        labels = result["label_direction_1p"][:-1].drop_nulls()
        up_count = (labels == 1).sum()
        total = len(labels)

        # Should be mostly upward labels
        assert up_count > total * 0.9, (
            f"Expected mostly up labels in uptrend, got {up_count}/{total}"
        )

    def test_binary_tail_is_null(self, sample_prices):
        """Binary labels should be null when future data is unavailable."""
        result = fixed_time_horizon_labels(
            sample_prices,
            horizon=5,
            method="binary",
        )
        assert result["label_direction_5p"][-5:].null_count() == 5

    def test_binary_downtrend(self, trending_down):
        """Test binary labels detect downtrend."""
        result = fixed_time_horizon_labels(
            trending_down,
            horizon=1,
            method="binary",
        )

        labels = result["label_direction_1p"][:-1].drop_nulls()
        down_count = (labels == -1).sum()
        total = len(labels)

        # Should be mostly downward labels
        assert down_count > total * 0.9, (
            f"Expected mostly down labels in downtrend, got {down_count}/{total}"
        )

    def test_different_horizons(self, sample_prices):
        """Test with different horizon lengths."""
        for horizon in [1, 5, 10, 20]:
            result = fixed_time_horizon_labels(
                sample_prices,
                horizon=horizon,
                method="returns",
            )

            # Should have correct number of nulls at end
            col_name = f"label_return_{horizon}p"
            null_count = result[col_name][-horizon:].null_count()
            assert null_count == horizon

    def test_invalid_parameters(self, sample_prices):
        """Test error handling for invalid parameters."""
        # Negative horizon
        with pytest.raises(ValueError, match="horizon must be positive"):
            fixed_time_horizon_labels(sample_prices, horizon=-1)

        # Zero horizon
        with pytest.raises(ValueError, match="horizon must be positive"):
            fixed_time_horizon_labels(sample_prices, horizon=0)

        # Invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            fixed_time_horizon_labels(sample_prices, method="invalid")

        # Missing column
        with pytest.raises(Exception):  # DataValidationError
            fixed_time_horizon_labels(sample_prices, price_col="nonexistent")


class TestTrendScanningLabels:
    """Test trend scanning labeling."""

    def test_basic_trend_scanning(self, sample_prices):
        """Test basic trend scanning."""
        result = trend_scanning_labels(
            sample_prices,
            min_window=5,
            max_window=20,
        )

        # Should have three output columns
        assert "label" in result.columns
        assert "t_value" in result.columns
        assert "optimal_window" in result.columns

        # Labels should be ±1 or null
        labels = result["label"].drop_nulls()
        assert labels.is_in([1, -1]).all()

    def test_uptrend_detection(self, trending_up):
        """Test that uptrend is detected."""
        result = trend_scanning_labels(
            trending_up,
            min_window=3,
            max_window=20,
        )

        # In clear uptrend, most labels should be 1
        labels = result["label"].drop_nulls()
        up_count = (labels == 1).sum()
        total = len(labels)

        assert up_count > total * 0.8, (
            f"Expected mostly up labels in uptrend, got {up_count}/{total}"
        )

        # T-values should be mostly positive
        t_values = result["t_value"].drop_nulls()
        positive_t = (t_values > 0).sum()
        assert positive_t > len(t_values) * 0.8

    def test_downtrend_detection(self, trending_down):
        """Test that downtrend is detected."""
        result = trend_scanning_labels(
            trending_down,
            min_window=3,
            max_window=20,
        )

        # In clear downtrend, most labels should be -1
        labels = result["label"].drop_nulls()
        down_count = (labels == -1).sum()
        total = len(labels)

        assert down_count > total * 0.8, (
            f"Expected mostly down labels in downtrend, got {down_count}/{total}"
        )

        # T-values should be mostly negative
        t_values = result["t_value"].drop_nulls()
        negative_t = (t_values < 0).sum()
        assert negative_t > len(t_values) * 0.8

    def test_optimal_window_range(self, sample_prices):
        """Test that optimal windows are within specified range."""
        min_w, max_w = 5, 30
        result = trend_scanning_labels(
            sample_prices,
            min_window=min_w,
            max_window=max_w,
        )

        windows = result["optimal_window"].drop_nulls()

        # All windows should be in range
        assert (windows >= min_w).all()
        assert (windows <= max_w).all()

    def test_uses_config_column_contract(self):
        """Trend scanning should read price/timestamp columns from config."""
        df = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)],
                "px": np.linspace(100, 110, 50),
            }
        )
        config = LabelingConfig.trend_scanning(
            min_horizon=5,
            max_horizon=20,
            price_col="px",
            timestamp_col="ts",
        )
        result = trend_scanning_labels(
            df,
            min_window=5,
            max_window=20,
            config=config,
        )

        assert "label" in result.columns
        assert "t_value" in result.columns

    def test_uses_shared_contract_column_contract(self):
        """Trend scanning should read price/timestamp columns from DataContractConfig."""
        df = pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(50)],
                "px": np.linspace(100, 110, 50),
            }
        )
        contract = DataContractConfig(timestamp_col="ts", price_col="px", symbol_col=None)
        result = trend_scanning_labels(
            df,
            min_window=5,
            max_window=20,
            contract=contract,
        )

        assert "label" in result.columns
        assert "t_value" in result.columns

    def test_step_parameter(self, sample_prices):
        """Test that step parameter works."""
        # Using larger steps should be faster but less precise
        result = trend_scanning_labels(
            sample_prices,
            min_window=5,
            max_window=50,
            step=5,  # Skip every 5 windows
        )

        # Should still produce valid labels
        assert "label" in result.columns
        labels = result["label"].drop_nulls()
        assert len(labels) > 0

    def test_invalid_parameters(self, sample_prices):
        """Test error handling for invalid parameters."""
        # min_window too small
        with pytest.raises(ValueError, match="min_window must be at least 2"):
            trend_scanning_labels(sample_prices, min_window=1)

        # max_window <= min_window
        with pytest.raises(ValueError, match="max_window must be greater"):
            trend_scanning_labels(sample_prices, min_window=10, max_window=10)

        # Invalid step
        with pytest.raises(ValueError, match="step must be at least 1"):
            trend_scanning_labels(sample_prices, step=0)

        # Missing column
        with pytest.raises(Exception):  # DataValidationError
            trend_scanning_labels(sample_prices, price_col="nonexistent")

    def test_null_handling_at_end(self, sample_prices):
        """Test that end of series has nulls (insufficient data)."""
        result = trend_scanning_labels(
            sample_prices,
            min_window=10,
            max_window=20,
        )

        # Last few rows should have nulls (can't scan forward)
        last_labels = result["label"][-10:]
        null_count = last_labels.null_count()

        # Should have some nulls at the end
        assert null_count > 0


class TestLabelingComparison:
    """Compare different labeling methods."""

    def test_fixed_vs_trend_scanning(self, trending_up):
        """Compare fixed horizon vs trend scanning on uptrend."""
        # Fixed horizon
        fixed = fixed_time_horizon_labels(trending_up, horizon=5, method="binary")

        # Trend scanning
        trend = trend_scanning_labels(trending_up, min_window=3, max_window=15)

        # Both should detect uptrend
        fixed_labels = fixed["label_direction_5p"].drop_nulls()
        trend_labels = trend["label"].drop_nulls()

        fixed_up_pct = (fixed_labels == 1).sum() / len(fixed_labels)
        trend_up_pct = (trend_labels == 1).sum() / len(trend_labels)

        # Both should have high upward percentage
        assert fixed_up_pct > 0.8
        assert trend_up_pct > 0.7  # Trend scanning is more conservative

    def test_consistency_across_methods(self, sample_prices):
        """Test that all methods produce valid output."""
        # Test all three methods work without errors
        fixed_returns = fixed_time_horizon_labels(sample_prices, horizon=5, method="returns")
        fixed_log = fixed_time_horizon_labels(sample_prices, horizon=5, method="log_returns")
        fixed_binary = fixed_time_horizon_labels(sample_prices, horizon=5, method="binary")
        trend = trend_scanning_labels(sample_prices, min_window=5, max_window=20)

        # All should produce valid dataframes
        assert len(fixed_returns) == len(sample_prices)
        assert len(fixed_log) == len(sample_prices)
        assert len(fixed_binary) == len(sample_prices)
        assert len(trend) == len(sample_prices)
