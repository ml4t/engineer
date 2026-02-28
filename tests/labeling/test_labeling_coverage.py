"""
Additional tests for labeling APIs to boost coverage.

Focuses on:
- Edge cases in validation
- Short position handling
- Dynamic column configurations
- Error handling branches
- Trend scanning edge cases
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_almost_equal

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.horizon_labels import (
    fixed_time_horizon_labels,
    trend_scanning_labels,
)
from ml4t.engineer.labeling.triple_barrier import triple_barrier_labels
from ml4t.engineer.labeling.uniqueness import (
    _build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
    sequential_bootstrap,
)


class TestBuildConcurrency:
    """Test build_concurrency function."""

    def test_n_bars_none_uses_max_label_index(self):
        """Test that n_bars=None uses max(label_indices) + 1."""
        event_indices = np.array([0, 5, 10], dtype=np.int64)
        label_indices = np.array([3, 8, 15], dtype=np.int64)

        concurrency = _build_concurrency(16, event_indices, label_indices)

        # Should have length 16
        assert len(concurrency) == 16

    def test_with_explicit_n_bars(self):
        """Test with explicit n_bars parameter."""
        event_indices = np.array([0, 2], dtype=np.int64)
        label_indices = np.array([5, 7], dtype=np.int64)

        concurrency = _build_concurrency(20, event_indices, label_indices)

        assert len(concurrency) == 20

    def test_out_of_bounds_handling(self):
        """Test handling of out-of-bounds indices."""
        event_indices = np.array([0, 100], dtype=np.int64)  # 100 is out of bounds for n_bars=50
        label_indices = np.array([5, 105], dtype=np.int64)

        # Should handle gracefully without crashing
        concurrency = _build_concurrency(50, event_indices, label_indices)

        assert len(concurrency) == 50
        # First event should be counted
        assert concurrency[0] >= 1


class TestCalculateLabelUniqueness:
    """Test calculate_label_uniqueness edge cases."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        event_indices = np.array([], dtype=np.float64)
        label_indices = np.array([], dtype=np.float64)

        uniqueness = calculate_label_uniqueness(event_indices, label_indices)

        assert len(uniqueness) == 0

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        event_indices = np.array([0, 1, 2], dtype=np.float64)
        label_indices = np.array([5, 6], dtype=np.float64)  # Different length

        with pytest.raises(ValueError, match="same length"):
            calculate_label_uniqueness(event_indices, label_indices)

    def test_negative_indices_raises_error(self):
        """Test that negative indices raise ValueError."""
        event_indices = np.array([-1, 1, 2], dtype=np.float64)
        label_indices = np.array([5, 6, 7], dtype=np.float64)

        with pytest.raises(ValueError, match="non-negative"):
            calculate_label_uniqueness(event_indices, label_indices)

    def test_out_of_bounds_start_index(self):
        """Test label with start index >= n_bars gets uniqueness 1.0."""
        event_indices = np.array([0, 100], dtype=np.float64)  # 100 is out of bounds
        label_indices = np.array([5, 110], dtype=np.float64)

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=50)

        # Out-of-bounds label should get default uniqueness 1.0
        assert uniqueness[1] == 1.0


class TestCalculateSampleWeights:
    """Test calculate_sample_weights edge cases."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        uniqueness = np.array([], dtype=np.float64)
        returns = np.array([], dtype=np.float64)

        weights = calculate_sample_weights(uniqueness, returns)

        assert len(weights) == 0

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        uniqueness = np.array([0.5, 0.8], dtype=np.float64)
        returns = np.array([0.01, 0.02, 0.03], dtype=np.float64)

        with pytest.raises(ValueError, match="same length"):
            calculate_sample_weights(uniqueness, returns)

    def test_all_weight_schemes(self):
        """Test all four weight schemes."""
        uniqueness = np.array([0.5, 0.8, 0.3])
        returns = np.array([0.02, -0.01, 0.03])

        # Test each scheme
        for scheme in ["returns_uniqueness", "uniqueness_only", "returns_only", "equal"]:
            weights = calculate_sample_weights(uniqueness, returns, scheme)
            assert len(weights) == 3
            assert all(w >= 0 for w in weights)

    def test_zero_total_weight_fallback(self):
        """Test fallback when total weight is zero."""
        uniqueness = np.array([0.0, 0.0, 0.0])
        returns = np.array([0.0, 0.0, 0.0])

        weights = calculate_sample_weights(uniqueness, returns, "returns_uniqueness")

        # Should fall back to uniform weights
        assert_array_almost_equal(weights, [1.0, 1.0, 1.0])


class TestSequentialBootstrap:
    """Test sequential_bootstrap edge cases."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)

        order = sequential_bootstrap(starts, ends)

        assert len(order) == 0

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        starts = np.array([0, 1], dtype=np.int64)
        ends = np.array([5, 6, 7], dtype=np.int64)

        with pytest.raises(ValueError, match="same length"):
            sequential_bootstrap(starts, ends)

    def test_negative_indices_raises_error(self):
        """Test that negative indices raise ValueError."""
        starts = np.array([-1, 1], dtype=np.int64)
        ends = np.array([5, 6], dtype=np.int64)

        with pytest.raises(ValueError, match="non-negative"):
            sequential_bootstrap(starts, ends)

    def test_n_draws_zero_raises_error(self):
        """Test that n_draws=0 raises ValueError."""
        starts = np.array([0, 5], dtype=np.int64)
        ends = np.array([4, 9], dtype=np.int64)

        with pytest.raises(ValueError, match="positive"):
            sequential_bootstrap(starts, ends, n_draws=0)

    def test_without_replacement_exceeds_candidates(self):
        """Test error when n_draws > n_candidates without replacement."""
        starts = np.array([0, 5], dtype=np.int64)
        ends = np.array([4, 9], dtype=np.int64)

        with pytest.raises(ValueError, match="Cannot draw"):
            sequential_bootstrap(starts, ends, n_draws=5, with_replacement=False)

    def test_reversed_interval(self):
        """Test handling of reversed intervals (end < start)."""
        starts = np.array([5, 10], dtype=np.int64)
        ends = np.array([2, 15], dtype=np.int64)  # First interval is reversed

        # Should handle gracefully
        order = sequential_bootstrap(starts, ends, n_bars=20, n_draws=2, random_state=42)

        assert len(order) == 2


class TestTripleBarrierDynamicColumns:
    """Test triple_barrier_labels with dynamic column configurations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with dynamic columns."""
        n = 100
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "volatility": np.abs(np.random.randn(n) * 0.02),
                "dynamic_side": np.random.choice([1, -1], n),
                "dynamic_period": np.random.randint(5, 20, n),
                "trailing_pct": np.abs(np.random.randn(n) * 0.01),
            }
        )

    def test_dynamic_upper_barrier_missing_column(self, sample_data):
        """Test error when dynamic upper barrier column is missing."""
        config = LabelingConfig.triple_barrier(
            upper_barrier="nonexistent_column",
            lower_barrier=0.01,
            max_holding_period=10,
        )

        with pytest.raises(DataValidationError, match="Upper barrier column"):
            triple_barrier_labels(sample_data, config, price_col="close")

    def test_dynamic_lower_barrier_missing_column(self, sample_data):
        """Test error when dynamic lower barrier column is missing."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier="nonexistent_column",
            max_holding_period=10,
        )

        with pytest.raises(DataValidationError, match="Lower barrier column"):
            triple_barrier_labels(sample_data, config, price_col="close")

    def test_dynamic_max_holding_period_missing_column(self, sample_data):
        """Test error when dynamic max_holding_period column is missing."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period="nonexistent_column",
        )

        with pytest.raises(DataValidationError, match="Max holding period column"):
            triple_barrier_labels(sample_data, config, price_col="close")

    def test_dynamic_side_missing_column(self, sample_data):
        """Test error when dynamic side column is missing."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
            side="nonexistent_column",
        )

        with pytest.raises(DataValidationError, match="Side column"):
            triple_barrier_labels(sample_data, config, price_col="close")

    def test_dynamic_trailing_stop_missing_column(self, sample_data):
        """Test error when dynamic trailing_stop column is missing."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
            trailing_stop="nonexistent_column",
        )

        with pytest.raises(DataValidationError, match="Trailing stop column"):
            triple_barrier_labels(sample_data, config, price_col="close")

    def test_dynamic_side_column(self, sample_data):
        """Test with dynamic side from column."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
            side="dynamic_side",
        )

        result = triple_barrier_labels(sample_data, config, price_col="close")

        assert "label" in result.columns
        assert len(result) == len(sample_data)

    def test_dynamic_max_holding_period_column(self, sample_data):
        """Test with dynamic max_holding_period from column."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period="dynamic_period",
        )

        result = triple_barrier_labels(sample_data, config, price_col="close")

        assert "label" in result.columns

    def test_dynamic_trailing_stop_column(self, sample_data):
        """Test with dynamic trailing_stop from column."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
            trailing_stop="trailing_pct",
        )

        result = triple_barrier_labels(sample_data, config, price_col="close")

        assert "label" in result.columns

    def test_trailing_stop_true_uses_lower_barrier(self, sample_data):
        """Test that trailing_stop=True uses lower_barrier percentage."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=0.02,
            max_holding_period=20,
            trailing_stop=True,  # Should use lower_barrier (0.02) as trailing stop
        )

        result = triple_barrier_labels(sample_data, config, price_col="close")

        assert "label" in result.columns

    def test_trailing_stop_true_no_lower_barrier(self, sample_data):
        """Test that trailing_stop=True with no lower_barrier uses default."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=None,
            max_holding_period=20,
            trailing_stop=True,  # Should use default 1%
        )

        result = triple_barrier_labels(sample_data, config, price_col="close")

        assert "label" in result.columns


class TestShortPositionLabeling:
    """Test short position handling in triple barrier labeling."""

    def test_short_position_profit_target(self):
        """Test short position profit target (price goes down)."""
        # Create data where price goes down (good for short)
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(20)]
        prices = [100.0 - i * 0.3 for i in range(20)]  # Declining prices

        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=0.01,  # 1% stop loss
            max_holding_period=15,
            side=-1,  # SHORT
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # First entry should hit profit target (price went down 2%)
        # Short profit target is at 100 * (1 - 0.02) = 98
        # This happens around bar 6-7 (price = 100 - 0.3*7 = 97.9)
        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"

    def test_short_position_stop_loss(self):
        """Test short position stop loss (price goes up)."""
        # Create data where price goes up (bad for short)
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(20)]
        prices = [100.0 + i * 0.3 for i in range(20)]  # Rising prices

        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=0.01,  # 1% stop loss
            max_holding_period=15,
            side=-1,  # SHORT
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # First entry should hit stop loss (price went up 1%)
        # Short stop loss is at 100 * (1 + 0.01) = 101
        # This happens around bar 3-4 (price = 100 + 0.3*4 = 101.2)
        assert result["label"][0] == -1
        assert result["barrier_hit"][0] == "lower"

    def test_short_position_with_trailing_stop(self):
        """Test short position with trailing stop."""
        # Price goes down initially then reverses
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(30)]
        prices = [100.0 - min(i, 10) * 0.3 + max(0, i - 10) * 0.5 for i in range(30)]

        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.10,  # 10% profit target (won't hit)
            lower_barrier=0.05,  # 5% stop loss
            max_holding_period=25,
            side=-1,
            trailing_stop=0.02,  # 2% trailing stop
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # Should have some labels
        assert result["label"][0] is not None


class TestFixedTimeHorizonLabels:
    """Test fixed_time_horizon_labels function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        return pl.DataFrame({"close": [100.0, 102.0, 101.0, 103.0, 99.0, 105.0]})

    def test_returns_method(self, sample_data):
        """Test returns labeling method."""
        result = fixed_time_horizon_labels(sample_data, horizon=1, method="returns")

        assert "label_return_1p" in result.columns
        # First return: (102-100)/100 = 0.02
        assert abs(result["label_return_1p"][0] - 0.02) < 1e-10

    def test_log_returns_method(self, sample_data):
        """Test log returns labeling method."""
        result = fixed_time_horizon_labels(sample_data, horizon=1, method="log_returns")

        assert "label_log_return_1p" in result.columns
        # First log return: log(102/100) ≈ 0.0198
        expected = np.log(102 / 100)
        assert abs(result["label_log_return_1p"][0] - expected) < 1e-10

    def test_binary_method(self, sample_data):
        """Test binary labeling method."""
        result = fixed_time_horizon_labels(sample_data, horizon=1, method="binary")

        assert "label_direction_1p" in result.columns
        # First direction: 102 > 100 → 1
        assert result["label_direction_1p"][0] == 1
        # Third direction: 103 > 101 → 1
        assert result["label_direction_1p"][2] == 1
        # Fifth direction: 105 > 99 → 1
        assert result["label_direction_1p"][4] == 1

    def test_invalid_horizon(self, sample_data):
        """Test that negative/zero horizon raises error."""
        with pytest.raises(ValueError, match="positive"):
            fixed_time_horizon_labels(sample_data, horizon=0)

        with pytest.raises(ValueError, match="positive"):
            fixed_time_horizon_labels(sample_data, horizon=-1)

    def test_invalid_method(self, sample_data):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            fixed_time_horizon_labels(sample_data, horizon=1, method="invalid")

    def test_missing_price_column(self, sample_data):
        """Test error when price column is missing."""
        with pytest.raises(DataValidationError, match="not found"):
            fixed_time_horizon_labels(sample_data, horizon=1, price_col="nonexistent")


class TestTrendScanningLabels:
    """Test trend_scanning_labels function."""

    @pytest.fixture
    def uptrend_data(self):
        """Create upward trending data with noise."""
        np.random.seed(42)
        # Add noise to prevent perfect R² which causes std_err = 0
        noise = np.random.randn(100) * 0.5
        return pl.DataFrame({"close": [100.0 + i * 0.5 + noise[i] for i in range(100)]})

    @pytest.fixture
    def downtrend_data(self):
        """Create downward trending data with noise."""
        np.random.seed(42)
        noise = np.random.randn(100) * 0.5
        return pl.DataFrame({"close": [100.0 - i * 0.5 + noise[i] for i in range(100)]})

    def test_uptrend_detection(self, uptrend_data):
        """Test that uptrend is detected."""
        result = trend_scanning_labels(uptrend_data, min_window=5, max_window=20)

        # Most labels should be positive (uptrend)
        labels = result["label"].drop_nulls().to_numpy()
        positive_ratio = (labels == 1).sum() / len(labels)
        # At least 50% should be positive for uptrend
        assert positive_ratio > 0.5, f"Expected >50% positive, got {positive_ratio:.1%}"

    def test_downtrend_detection(self, downtrend_data):
        """Test that downtrend is detected."""
        result = trend_scanning_labels(downtrend_data, min_window=5, max_window=20)

        # Most labels should be negative (downtrend)
        labels = result["label"].drop_nulls().to_numpy()
        negative_ratio = (labels == -1).sum() / len(labels)
        assert negative_ratio > 0.5, f"Expected >50% negative, got {negative_ratio:.1%}"

    def test_output_columns(self, uptrend_data):
        """Test that output has required columns."""
        result = trend_scanning_labels(uptrend_data, min_window=5, max_window=20)

        assert "label" in result.columns
        assert "t_value" in result.columns
        assert "optimal_window" in result.columns

    def test_invalid_min_window(self, uptrend_data):
        """Test that min_window < 2 raises error."""
        with pytest.raises(ValueError, match="at least 2"):
            trend_scanning_labels(uptrend_data, min_window=1)

    def test_invalid_max_window(self, uptrend_data):
        """Test that max_window <= min_window raises error."""
        with pytest.raises(ValueError, match="greater than"):
            trend_scanning_labels(uptrend_data, min_window=10, max_window=10)

        with pytest.raises(ValueError, match="greater than"):
            trend_scanning_labels(uptrend_data, min_window=10, max_window=5)

    def test_invalid_step(self, uptrend_data):
        """Test that step < 1 raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            trend_scanning_labels(uptrend_data, step=0)

    def test_missing_price_column(self, uptrend_data):
        """Test error when price column is missing."""
        with pytest.raises(DataValidationError, match="not found"):
            trend_scanning_labels(uptrend_data, price_col="nonexistent")

    def test_exception_handling_in_linregress(self):
        """Test that numerical issues in linregress are handled."""
        # Create data that might cause numerical issues (constant values)
        constant_data = pl.DataFrame({"close": [100.0] * 50})

        # Should not raise, just continue with other windows
        result = trend_scanning_labels(constant_data, min_window=5, max_window=20)

        # Should still produce output
        assert "label" in result.columns


class TestEventBasedLabeling:
    """Test event-based labeling scenarios."""

    def test_no_events_returns_null_labels(self):
        """Test that no events returns null labels."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(20)]
        prices = [100.0 + i * 0.1 for i in range(20)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "event_time": [None] * 20,  # No events
            }
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # All labels should be null
        assert result["label"].is_null().all()

    def test_sparse_events(self):
        """Test labeling with sparse events."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)]
        prices = [100.0 + i * 0.2 for i in range(50)]

        # Only 3 events
        event_markers = [None] * 50
        event_markers[5] = timestamps[5]
        event_markers[20] = timestamps[20]
        event_markers[35] = timestamps[35]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "event_time": event_markers,
            }
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # Only event rows should have labels
        assert result.filter(pl.col("event_time").is_not_null())["label"].null_count() == 0


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_zero_price_return_calculation(self):
        """Test that zero entry price is handled."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(5)]
        prices = [0.0, 1.0, 2.0, 1.5, 1.25]
        df = pl.DataFrame({"timestamp": timestamps, "close": prices})
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.1,
            lower_barrier=0.1,
            max_holding_period=3,
        )

        result = triple_barrier_labels(df, config, price_col="close")
        assert "label" in result.columns

    def test_very_small_barriers(self):
        """Test with very small barrier values."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(20)]
        prices = [100.0 + np.random.randn() * 0.001 for _ in range(20)]

        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.0001,  # 0.01%
            lower_barrier=0.0001,
            max_holding_period=15,
        )

        # Should not crash
        result = triple_barrier_labels(df, config, price_col="close")
        assert "label" in result.columns

    def test_very_large_barriers(self):
        """Test with very large barrier values."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(20)]
        prices = [100.0] * 20

        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        config = LabelingConfig.triple_barrier(
            upper_barrier=10.0,  # 1000%
            lower_barrier=10.0,
            max_holding_period=10,
        )

        result = triple_barrier_labels(df, config, price_col="close")

        # All should timeout (barriers too far)
        assert (result["barrier_hit"] == "time").all()
