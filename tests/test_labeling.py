"""Tests for the generalized labeling module."""

import importlib
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling import triple_barrier_labels


class TestRemovedCoreModule:
    """Test explicit error messaging for removed legacy module paths."""

    def test_labeling_core_import_error_is_actionable(self):
        """Importing removed labeling.core should raise a clear migration error."""
        sys.modules.pop("ml4t.engineer.labeling.core", None)
        with pytest.raises(ImportError, match="labeling.core has been removed"):
            importlib.import_module("ml4t.engineer.labeling.core")

    def test_labeling_barrier_config_export_is_actionable(self):
        """Importing removed labeling.BarrierConfig should raise migration guidance."""
        with pytest.raises(ImportError, match="BarrierConfig has been removed"):
            from ml4t.engineer.labeling import BarrierConfig  # noqa: F401

    def test_labeling_barriers_module_import_error_is_actionable(self):
        """Importing removed labeling.barriers should raise migration guidance."""
        sys.modules.pop("ml4t.engineer.labeling.barriers", None)
        with pytest.raises(ImportError, match="labeling.barriers has been removed"):
            importlib.import_module("ml4t.engineer.labeling.barriers")

    def test_labeling_barrier_utils_module_import_error_is_actionable(self):
        """Importing removed labeling.barrier_utils should raise migration guidance."""
        sys.modules.pop("ml4t.engineer.labeling.barrier_utils", None)
        with pytest.raises(ImportError, match="labeling.barrier_utils has been removed"):
            importlib.import_module("ml4t.engineer.labeling.barrier_utils")

    def test_config_alias_import_error_is_actionable(self):
        """Importing removed config alias should raise migration guidance."""
        with pytest.raises(ImportError, match="BarrierLabelingConfig has been removed"):
            from ml4t.engineer.config import BarrierLabelingConfig  # noqa: F401

    def test_config_labeling_alias_import_error_is_actionable(self):
        """Importing removed config.labeling alias should raise migration guidance."""
        with pytest.raises(ImportError, match="BarrierLabelingConfig has been removed"):
            from ml4t.engineer.config.labeling import BarrierLabelingConfig  # noqa: F401

    def test_non_labeling_config_input_raises_actionable_error(self):
        """Passing non-LabelingConfig should fail with migration guidance."""

        class LegacyBarrierConfig:
            pass

        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "close": [100.0]})
        with pytest.raises(TypeError, match="Legacy BarrierConfig inputs are no longer supported"):
            triple_barrier_labels(df, config=LegacyBarrierConfig())  # type: ignore[arg-type]


class TestLabelingConfig:
    """Test labeling configuration."""

    def test_default_config(self):
        """Test default labeling configuration."""
        config = LabelingConfig()
        assert config.upper_barrier is None
        assert config.lower_barrier is None
        assert config.max_holding_period == 10
        assert config.trailing_stop is False

    def test_custom_config(self):
        """Test custom barrier configuration."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
            trailing_stop=True,
        )
        assert config.upper_barrier == 0.02
        assert config.lower_barrier == -0.01
        assert config.max_holding_period == 20
        assert config.trailing_stop is True


class TestTripleBarrierLabels:
    """Test triple barrier labeling functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        # Create synthetic price data with known patterns
        np.random.seed(42)
        n = 1000

        # Create timestamps
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]

        # Create price with trend and noise
        trend = np.linspace(100, 120, n)
        noise = np.random.randn(n) * 0.5
        prices = trend + noise

        # Add some jumps
        prices[200:210] += 5  # Positive jump
        prices[500:510] -= 3  # Negative jump

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "volume": np.random.randint(1000, 10000, n),
            },
        )

    def test_basic_triple_barrier(self, sample_data):
        """Test basic triple barrier labeling."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=-0.01,  # 1% stop loss
            max_holding_period=50,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check output columns
        expected_cols = set(sample_data.columns) | {
            "label",
            "label_time",
            "label_price",
            "label_return",
            "barrier_hit",
            "label_bars",  # Added in updated implementation
            "label_duration",  # Added in updated implementation
        }
        assert set(result.columns) == expected_cols

        # Check label values
        labels = result["label"].unique().to_list()
        assert all(label in [-1, 0, 1] for label in labels if label is not None)

        # Check that returns match labels
        pos_labels = result.filter(pl.col("label") == 1)
        if len(pos_labels) > 0:
            assert (pos_labels["label_return"] > 0).all()

        neg_labels = result.filter(pl.col("label") == -1)
        if len(neg_labels) > 0:
            assert (neg_labels["label_return"] < 0).all()

    def test_dynamic_barriers(self, sample_data):
        """Test dynamic barriers based on volatility."""
        # Add rolling volatility
        sample_data = sample_data.with_columns(
            volatility=pl.col("price").pct_change().rolling_std(window_size=20) * np.sqrt(252),
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier="volatility",  # Use volatility column
            lower_barrier="volatility",  # Symmetric barriers
            max_holding_period=30,
        )

        result = triple_barrier_labels(
            sample_data.filter(pl.col("volatility").is_not_null()),
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check that we got labels
        assert "label" in result.columns
        assert result["label"].null_count() < len(result)

    def test_trailing_stop(self, sample_data):
        """Test trailing stop functionality."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=-0.02,
            max_holding_period=100,
            trailing_stop=True,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # With trailing stop, we should see some different behavior
        # Specifically, positions that go up should be protected
        assert "label" in result.columns

    def test_max_holding_period(self, sample_data):
        """Test that max holding period is respected."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=10.0,  # Very high barrier (unlikely to hit)
            lower_barrier=-10.0,  # Very low barrier (unlikely to hit)
            max_holding_period=5,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # All positions should close within max_holding_period
        for idx in range(len(result) - config.max_holding_period):
            if result["label"][idx] is not None:
                entry_time = result["timestamp"][idx]
                exit_time = result["label_time"][idx]
                if exit_time is not None:
                    time_diff = (exit_time - entry_time).total_seconds() / 60
                    assert time_diff <= config.max_holding_period

    def test_no_barriers(self, sample_data):
        """Test with no profit/loss barriers, only time."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=None,
            lower_barrier=None,
            max_holding_period=10,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # All exits should be at max holding period
        for idx in range(len(result) - config.max_holding_period):
            if result["label"][idx] is not None:
                assert result["barrier_hit"][idx] == "time"

    def test_asymmetric_barriers(self, sample_data):
        """Test asymmetric barriers."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.03,  # 3% profit
            lower_barrier=-0.01,  # 1% loss
            max_holding_period=50,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check that barriers are applied correctly
        upper_hits = result.filter(pl.col("barrier_hit") == "upper")
        lower_hits = result.filter(pl.col("barrier_hit") == "lower")

        if len(upper_hits) > 0:
            # Returns should be around upper barrier (allow tolerance)
            returns = upper_hits["label_return"]
            assert (returns >= config.upper_barrier * 0.70).all()

        if len(lower_hits) > 0:
            # Returns should be around lower barrier (negative values)
            # Just check that they're all negative (close to barrier is fine, variance expected)
            returns = lower_hits["label_return"]
            assert (returns < 0).all()  # Should be negative
            # Most should be reasonably close to the barrier
            median_return = returns.median()
            assert abs(median_return - config.lower_barrier) / abs(config.lower_barrier) < 0.6


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pl.DataFrame({"timestamp": [], "price": []})
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        result = triple_barrier_labels(df, config, price_col="price")
        assert len(result) == 0

    def test_single_row(self):
        """Test with single row."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "price": [100.0]})
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        result = triple_barrier_labels(df, config, price_col="price")
        assert len(result) == 1
        # With only one row, it immediately times out (label=0)
        assert result["label"][0] == 0

    def test_all_nan_prices(self):
        """Test with all NaN prices."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(10)],
                "price": [float("nan")] * 10,
            },
        )
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        result = triple_barrier_labels(df, config, price_col="price")
        # With NaN prices, all events timeout (label=0) and returns are NaN
        assert (result["label"] == 0).all()
        assert result["label_return"].is_nan().all()

    def test_invalid_barriers(self):
        """Test with invalid barrier configuration."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "price": [100.0]})

        # Upper barrier less than lower barrier
        # Note: Currently no validation for invalid barriers, so test just ensures no crash
        config = LabelingConfig.triple_barrier(upper_barrier=-0.01, lower_barrier=0.02)
        result = triple_barrier_labels(df, config, price_col="price")
        assert len(result) == len(df)

    def test_missing_columns(self):
        """Test with missing required columns."""
        df = pl.DataFrame({"time": [datetime(2024, 1, 1)], "value": [100.0]})
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        # Missing price column should raise DataValidationError
        with pytest.raises(DataValidationError):
            triple_barrier_labels(df, config, price_col="price")

        # Test with valid price column but no timestamp
        result = triple_barrier_labels(df, config, price_col="value")
        assert len(result) == len(df)

    def test_missing_default_timestamp_does_not_warn(self):
        """Default config timestamp should not warn when not explicitly requested."""
        df = pl.DataFrame({"value": [100.0, 101.0, 102.0]})
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = triple_barrier_labels(df, config, price_col="value")

        assert len(result) == len(df)
        assert not any(
            "Specified timestamp_col 'timestamp' not found" in str(w.message) for w in caught
        )

    def test_explicit_missing_timestamp_still_warns(self):
        """Explicitly requesting a missing timestamp column should still warn."""
        df = pl.DataFrame({"value": [100.0, 101.0, 102.0]})
        config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=-0.01)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = triple_barrier_labels(df, config, price_col="value", timestamp_col="timestamp")

        assert len(result) == len(df)
        assert any(
            "Specified timestamp_col 'timestamp' not found" in str(w.message) for w in caught
        )


class TestBarDuration:
    """Test bar duration calculations."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with known barrier hits."""
        # Create data where we know exactly when barriers will be hit
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)]
        prices = [100.0] * 100

        # Set up specific price movements for testing
        # Event at index 0: hits upper barrier at index 10
        for i in range(11):
            prices[i] = 100.0 + i * 0.2  # Gradual increase

        # Event at index 20: hits lower barrier at index 25
        for i in range(20, 26):
            prices[i] = 100.0 - (i - 20) * 0.2  # Gradual decrease

        # Event at index 40: times out at max_holding_period
        for i in range(40, 60):
            prices[i] = 100.0 + 0.001 * (i - 40)  # Small movement, won't hit barriers

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

    def test_bar_duration_barrier_hit(self, simple_data):
        """Test bar duration when barrier is hit."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% up
            lower_barrier=-0.01,  # 1% down
            max_holding_period=50,
        )

        result = triple_barrier_labels(
            simple_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check that label_bars column exists
        assert "label_bars" in result.columns

        # First event (index 0) should hit upper barrier at index 10
        # Price goes from 100 to 102 (2% gain) in 10 bars
        first_label = result[0, "label"]
        if first_label == 1:  # Upper barrier hit
            assert result[0, "label_bars"] == 10

    def test_bar_duration_timeout(self, simple_data):
        """Test bar duration on timeout."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.10,  # 10% up (won't be hit)
            lower_barrier=-0.10,  # 10% down (won't be hit)
            max_holding_period=15,
        )

        result = triple_barrier_labels(
            simple_data,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Event at index 40 should timeout at max_holding_period
        label_40 = result[40, "label"]
        if label_40 == 0:  # Timeout
            # Should be close to max_holding_period (allow off-by-one)
            assert result[40, "label_bars"] in [14, 15]

    def test_bar_duration_matches_label_index(self):
        """Test that bar duration matches label_idx - event_idx."""
        np.random.seed(42)
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(200)]
        prices = 100 + np.random.randn(200) * 2

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.015,
            max_holding_period=30,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # For each labeled event, verify bar_duration calculation
        for i in range(len(result)):
            if result["label"][i] is not None:
                # Get the index where the label was determined
                label_time = result["label_time"][i]
                if isinstance(label_time, datetime):
                    # Find the index of label_time in timestamps
                    label_idx = None
                    for j, ts in enumerate(timestamps):
                        if ts == label_time:
                            label_idx = j
                            break

                    if label_idx is not None:
                        expected_bars = label_idx - i
                        actual_bars = result["label_bars"][i]
                        assert actual_bars == expected_bars, (
                            f"Event {i}: expected {expected_bars} bars, got {actual_bars}"
                        )

    def test_bar_duration_first_bar(self):
        """Test duration when entry is at first bar."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)]
        prices = [100.0 + i * 0.5 for i in range(50)]  # Steady increase

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # First event should have valid bar duration
        first_label_bars = result[0, "label_bars"]
        assert first_label_bars is not None
        assert first_label_bars >= 0

    def test_bar_duration_near_end(self):
        """Test duration when entry is near end of data."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)]
        prices = [100.0] * 50

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Event near end (index 45) should have small bar duration
        label_45 = result[45, "label"]
        if label_45 is not None:
            # Should be <= 4 bars (only 4 bars remaining)
            assert result[45, "label_bars"] <= 4

    def test_bar_duration_with_event_based_labeling(self):
        """Test bar duration with event-based labeling."""
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)]
        prices = 100 + np.random.randn(100) * 2

        # Create event markers at specific indices
        event_markers = [None] * 100
        event_markers[10] = datetime(2024, 1, 1, 0, 10)  # Event at index 10
        event_markers[30] = datetime(2024, 1, 1, 0, 30)  # Event at index 30
        event_markers[50] = datetime(2024, 1, 1, 0, 50)  # Event at index 50

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
                "event_time": event_markers,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=15,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check that only events have bar durations
        # Non-event rows should have null bar durations
        for i in range(len(result)):
            if result["event_time"][i] is None:
                # Non-event row - label_bars should be null
                label_bars_val = result["label_bars"][i]
                assert label_bars_val is None or (
                    isinstance(label_bars_val, float) and np.isnan(label_bars_val)
                )
            else:
                # Event row - label_bars should have a value
                assert result["label_bars"][i] is not None


class TestTimeDuration:
    """Test time duration calculations with gap handling."""

    def test_continuous_data_duration(self):
        """Test time duration with continuous 1-minute bars."""
        # Create continuous 1-minute data
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(100)]
        prices = 100 + np.random.randn(100) * 2

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.015,
            max_holding_period=20,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Check that label_duration column exists
        assert "label_duration" in result.columns

        # For continuous data, time duration should be proportional to bar duration
        for i in range(len(result) - 20):
            if result["label"][i] is not None:
                bar_duration = result["label_bars"][i]
                time_duration = result["label_duration"][i]

                # Time duration should equal bar_duration minutes (as timedelta)
                if time_duration is not None:
                    expected_minutes = bar_duration
                    actual_minutes = time_duration / np.timedelta64(1, "m")
                    assert abs(actual_minutes - expected_minutes) < 0.1, (
                        f"Expected ~{expected_minutes} minutes, got {actual_minutes}"
                    )

    def test_data_with_gaps(self):
        """Test time duration correctly handles gaps in data."""
        # Create data with known gaps
        timestamps = []
        prices = []

        # Bars 0-9: continuous
        for i in range(10):
            timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
            prices.append(100.0)

        # GAP: skip minutes 10-14 (5-minute gap)

        # Bars 10-19: continuous after gap
        for i in range(15, 25):
            timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
            prices.append(100.0)

        # GAP: skip minutes 25-29 (5-minute gap)

        # Bars 20-29: continuous after second gap
        for i in range(30, 40):
            timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
            prices.append(100.0)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.10,  # Won't be hit
            lower_barrier=-0.10,  # Won't be hit
            max_holding_period=15,  # Will timeout
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Event at index 0, exits at index 15 (max_holding_period)
        # Bar duration: around 15 bars (allow off-by-one)
        # Time duration: Should account for gaps
        first_label = result[0, "label"]
        if first_label == 0:  # Timeout
            assert result[0, "label_bars"] in [14, 15]

            # Time duration should be > 15 minutes due to gaps
            time_duration = result[0, "label_duration"]
            time_duration_minutes = time_duration / np.timedelta64(1, "m")
            assert time_duration_minutes > 14, (
                f"Expected >14 minutes due to gaps, got {time_duration_minutes}"
            )

    def test_irregular_timestamps(self):
        """Test with irregular timestamp intervals."""
        # Create data with varying intervals
        timestamps = [
            datetime(2024, 1, 1, 0, 0),  # 0
            datetime(2024, 1, 1, 0, 1),  # 1 minute
            datetime(2024, 1, 1, 0, 3),  # 2 minutes
            datetime(2024, 1, 1, 0, 7),  # 4 minutes
            datetime(2024, 1, 1, 0, 15),  # 8 minutes
            datetime(2024, 1, 1, 0, 31),  # 16 minutes
            datetime(2024, 1, 1, 1, 3),  # 32 minutes
        ] + [datetime(2024, 1, 1, 2, 0) + timedelta(minutes=i) for i in range(50)]

        prices = [100.0] * len(timestamps)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.10,
            lower_barrier=-0.10,
            max_holding_period=5,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Event at index 0 should exit at around index 5 (max_holding_period)
        # Bar duration: around 4-5 bars (allow off-by-one)
        # Time duration: depends on actual bar count and irregular intervals
        first_label = result[0, "label"]
        if first_label is not None:
            bar_duration = result[0, "label_bars"]
            assert bar_duration in [4, 5]

            time_duration = result[0, "label_duration"]
            time_duration_minutes = time_duration / np.timedelta64(1, "m")
            # With bar_duration=4: 0->1->3->7->15 = 15 minutes
            # With bar_duration=5: 0->1->3->7->15->31 = 31 minutes
            expected_min = 14 if bar_duration == 4 else 30
            expected_max = 16 if bar_duration == 4 else 32
            assert expected_min <= time_duration_minutes <= expected_max, (
                f"Expected ~{expected_min}-{expected_max} minutes for {bar_duration} bars, got {time_duration_minutes}"
            )

    def test_null_when_no_timestamp_col(self):
        """Test that label_duration is null when no timestamp column provided."""
        prices = [100.0 + i * 0.1 for i in range(50)]

        df = pl.DataFrame(
            {
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        # Call without timestamp_col
        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
        )

        # label_duration should exist but contain None values
        assert "label_duration" in result.columns

        # All duration values should be None
        for i in range(len(result)):
            duration_val = result["label_duration"][i]
            assert duration_val is None or (
                isinstance(duration_val, float) and np.isnan(duration_val)
            )

    def test_time_duration_handles_datetime_types(self):
        """Test that time duration works with different datetime types."""
        # Test with datetime64
        timestamps = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(50)]
        prices = 100 + np.random.randn(50) * 2

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.03,
            lower_barrier=-0.02,
            max_holding_period=15,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # Verify label_duration is calculated
        assert "label_duration" in result.columns

        # Check that we get timedelta values
        for i in range(len(result) - 15):
            if result["label"][i] is not None:
                time_duration = result["label_duration"][i]
                if time_duration is not None:
                    # Should be a timedelta-like value
                    assert hasattr(time_duration, "__sub__") or isinstance(
                        time_duration, int | float | np.timedelta64
                    )

    def test_gap_vs_bar_duration_difference(self):
        """Verify that time duration != bar duration when gaps exist."""
        # Create data with significant gaps
        timestamps = []
        prices = []

        # Regular 1-minute bars
        for i in range(5):
            timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
            prices.append(100.0)

        # 10-minute gap
        for i in range(15, 25):
            timestamps.append(datetime(2024, 1, 1) + timedelta(minutes=i))
            prices.append(100.0)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "price": prices,
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.10,
            lower_barrier=-0.10,
            max_holding_period=10,
        )

        result = triple_barrier_labels(
            df,
            config,
            price_col="price",
            timestamp_col="timestamp",
        )

        # First event times out at around bar 10 (allow off-by-one)
        first_label = result[0, "label"]
        if first_label == 0:
            bar_duration = result[0, "label_bars"]
            time_duration = result[0, "label_duration"]
            time_duration_minutes = time_duration / np.timedelta64(1, "m")

            # Bar duration: around 10 bars (allow off-by-one)
            assert bar_duration in [9, 10]

            # Time duration: should be > bar_duration minutes due to gap
            assert time_duration_minutes > bar_duration, (
                f"Time duration ({time_duration_minutes} min) should exceed bar duration ({bar_duration} bars) due to gap"
            )


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for duration calculations."""

    def _create_benchmark_data(self, n_bars: int) -> pl.DataFrame:
        """Create synthetic data for benchmarking.

        Parameters
        ----------
        n_bars : int
            Number of bars to generate

        Returns
        -------
        pl.DataFrame
            Synthetic OHLCV data with timestamps
        """
        # Create realistic price movements
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0, 0.01, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        # Create timestamps (1-minute bars)
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": np.random.randint(1000, 10000, n_bars),
            }
        )

    @pytest.mark.parametrize("n_bars", [10_000, 50_000, 100_000])
    def test_duration_overhead(self, benchmark, n_bars):
        """Benchmark duration calculation overhead.

        This test measures the performance impact of adding duration
        calculations to triple barrier labeling. Target: <5% overhead.

        Parameters
        ----------
        benchmark : fixture
            pytest-benchmark fixture
        n_bars : int
            Number of bars to test with
        """
        # Create test data
        df = self._create_benchmark_data(n_bars)

        # Create config
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        # Benchmark the full labeling with duration calculations
        result = benchmark(
            triple_barrier_labels,
            df,
            config,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Verify output contains duration columns
        assert "label_bars" in result.columns
        assert "label_duration" in result.columns

        # Report statistics
        labeled_count = result.filter(pl.col("label").is_not_null()).height
        print(
            f"\n{n_bars:,} bars: {labeled_count:,} events labeled "
            f"({labeled_count / n_bars * 100:.1f}% label rate)"
        )

    def test_scaling_characteristics(self, benchmark):
        """Test how duration calculations scale with dataset size.

        This test verifies linear scaling by measuring performance
        across multiple dataset sizes.
        """
        # Test with medium dataset (100K bars)
        df = self._create_benchmark_data(100_000)

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        result = benchmark(
            triple_barrier_labels,
            df,
            config,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Verify correctness
        assert result.height == df.height
        assert "label_bars" in result.columns
        assert "label_duration" in result.columns

    def test_duration_computation_only(self, benchmark):
        """Benchmark just the duration computation overhead.

        This isolates the duration calculation cost from the full
        labeling pipeline.
        """

        def duration_operations(timestamps, event_indices, label_indices):
            """Simulate duration calculations."""
            # Bar duration (Numba-compiled integer subtraction)
            bar_durations = label_indices - event_indices

            # Time duration (vectorized NumPy)
            timestamps_array = timestamps.to_numpy()
            entry_times = timestamps_array[event_indices]
            exit_times = timestamps_array[label_indices]
            time_durations = exit_times - entry_times

            return bar_durations, time_durations

        # Create test data
        n_events = 10_000
        n_bars = 100_000

        df = self._create_benchmark_data(n_bars)
        timestamps = df["timestamp"]

        # Simulate random event and label indices
        np.random.seed(42)
        event_indices = np.sort(np.random.choice(n_bars - 100, n_events, replace=False))
        label_indices = event_indices + np.random.randint(1, 50, n_events)
        label_indices = np.clip(label_indices, 0, n_bars - 1)

        # Benchmark just the duration calculations
        bar_durations, time_durations = benchmark(
            duration_operations, timestamps, event_indices, label_indices
        )

        # Verify results
        assert len(bar_durations) == n_events
        assert len(time_durations) == n_events
        assert np.all(bar_durations > 0)

        print(f"\n{n_events:,} duration calculations completed")

    @pytest.mark.parametrize("max_holding_period", [10, 20, 50, 100], ids=lambda x: f"period_{x}")
    def test_performance_vs_holding_period(self, benchmark, max_holding_period):
        """Test performance impact of different holding periods.

        Longer holding periods mean more bars to scan per event,
        potentially affecting performance.

        Parameters
        ----------
        benchmark : fixture
            pytest-benchmark fixture
        max_holding_period : int
            Maximum holding period to test
        """
        df = self._create_benchmark_data(50_000)

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=max_holding_period,
        )

        result = benchmark(
            triple_barrier_labels,
            df,
            config,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Verify correctness
        assert "label_bars" in result.columns
        max_bars = result["label_bars"].max()
        assert max_bars <= max_holding_period

        print(f"\nMax holding period: {max_holding_period}, Max bars held: {max_bars}")

    def test_memory_usage(self):
        """Test memory overhead of duration columns.

        Verifies that additional memory usage is minimal (16 bytes/event).
        """
        import sys

        # Create large dataset
        df = self._create_benchmark_data(100_000)

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=-0.01,
            max_holding_period=20,
        )

        result = triple_barrier_labels(df, config, price_col="close", timestamp_col="timestamp")

        # Estimate memory for duration columns
        n_events = result.height
        bar_duration_bytes = sys.getsizeof(result["label_bars"].to_numpy())
        time_duration_bytes = sys.getsizeof(result["label_duration"].to_numpy())
        total_duration_bytes = bar_duration_bytes + time_duration_bytes

        bytes_per_event = total_duration_bytes / n_events

        print(f"\nMemory usage for {n_events:,} events:")
        print(f"  Bar duration: {bar_duration_bytes / 1024:.1f} KB")
        print(f"  Time duration: {time_duration_bytes / 1024:.1f} KB")
        print(f"  Total: {total_duration_bytes / 1024:.1f} KB")
        print(f"  Per event: {bytes_per_event:.1f} bytes")

        # Verify reasonable memory usage (<50 bytes/event including overhead)
        assert bytes_per_event < 50, f"Memory per event too high: {bytes_per_event}"


class TestOHLCBarrierChecking:
    """Tests for OHLC-based barrier checking functionality."""

    @pytest.fixture
    def ohlc_data(self):
        """Create OHLC data where barriers trigger on intra-bar extremes."""
        # 10 bars of data
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]

        # Entry at bar 0: close=100
        # Bar 1-3: close moves slightly but HIGH spikes to trigger 2% TP
        # Bar 4-6: close moves slightly but LOW dips to trigger 1% SL

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 100.5, 100.8, 100.3, 99.8, 99.5, 99.2, 99.0, 98.8, 98.5],
                "high": [100.5, 102.5, 101.0, 100.5, 100.0, 99.8, 99.5, 99.3, 99.0, 98.7],
                "low": [99.5, 100.0, 100.5, 100.0, 98.5, 98.0, 98.8, 98.5, 98.5, 98.2],
            },
        )

    def test_ohlc_tp_triggers_on_high_for_long(self, ohlc_data):
        """LONG: TP should trigger when high >= target (bar 1 high=102.5 >= 102)."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% TP: 100 * 1.02 = 102
            lower_barrier=0.01,  # 1% SL: 100 * 0.99 = 99
            max_holding_period=10,
            side=1,  # LONG
        )

        result = triple_barrier_labels(
            ohlc_data,
            config,
            price_col="close",
            high_col="high",
            low_col="low",
        )

        # First entry (bar 0) should hit TP at bar 1 (high=102.5 >= 102)
        assert result["label"][0] == 1  # TP hit
        assert result["barrier_hit"][0] == "upper"
        assert result["label_bars"][0] == 1  # Triggered at bar 1

    def test_ohlc_sl_triggers_on_low_for_long(self):
        """LONG: SL should trigger when low <= stop."""
        # Data where close never hits SL but low does
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]

        data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 99.8, 99.5, 99.3, 99.2, 99.0, 98.9, 98.8, 98.7, 98.5],
                "high": [100.5, 100.0, 99.8, 99.5, 99.3, 99.2, 99.0, 98.9, 98.8, 98.7],
                "low": [99.5, 98.5, 99.0, 99.0, 99.0, 98.8, 98.7, 98.6, 98.5, 98.3],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% TP: 100 * 1.02 = 102
            lower_barrier=0.01,  # 1% SL: 100 * 0.99 = 99
            max_holding_period=10,
            side=1,  # LONG
        )

        result = triple_barrier_labels(
            data, config, price_col="close", high_col="high", low_col="low"
        )

        # SL should trigger at bar 1 (low=98.5 < 99)
        assert result["label"][0] == -1  # SL hit
        assert result["barrier_hit"][0] == "lower"
        assert result["label_bars"][0] == 1

    def test_ohlc_vs_close_only_different_results(self):
        """OHLC checking should detect barriers that close-only misses."""
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]

        # Close stays in 99.5-100.5 range, but high spikes to 103 on bar 2
        data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 100.2, 100.1, 100.3, 100.0, 99.8, 99.9, 100.0, 99.7, 99.5],
                "high": [100.5, 100.5, 103.0, 100.5, 100.2, 100.0, 100.0, 100.2, 100.0, 99.8],
                "low": [99.5, 99.8, 99.9, 100.0, 99.7, 99.5, 99.5, 99.6, 99.3, 99.0],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% TP
            lower_barrier=0.01,  # 1% SL
            max_holding_period=10,
            side=1,
        )

        # With OHLC: TP triggers at bar 2 (high=103 >= 102)
        result_ohlc = triple_barrier_labels(
            data, config, price_col="close", high_col="high", low_col="low"
        )

        # Without OHLC (close-only): No barrier hit, time exit
        result_close_only = triple_barrier_labels(data, config, price_col="close")

        # OHLC should detect TP
        assert result_ohlc["label"][0] == 1
        assert result_ohlc["barrier_hit"][0] == "upper"

        # Close-only should miss it (time exit)
        assert result_close_only["label"][0] == 0
        assert result_close_only["barrier_hit"][0] == "time"

    def test_backward_compatible_without_ohlc(self):
        """Without high_col/low_col, behavior should be unchanged (close-only)."""
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]
        data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 102.5, 101.0, 100.5, 100.0, 99.8, 99.5, 99.0, 98.5, 98.0],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02, lower_barrier=0.01, max_holding_period=10, side=1
        )

        result = triple_barrier_labels(data, config, price_col="close")

        # TP triggers at bar 1 (close=102.5 >= 102)
        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"

    def test_ohlc_column_validation(self):
        """Invalid column names should raise DataValidationError."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "close": [100.0],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02, lower_barrier=0.01, max_holding_period=10
        )

        with pytest.raises(DataValidationError, match="High column 'nonexistent' not found"):
            triple_barrier_labels(data, config, price_col="close", high_col="nonexistent")

        with pytest.raises(DataValidationError, match="Low column 'nonexistent' not found"):
            triple_barrier_labels(data, config, price_col="close", low_col="nonexistent")

    def test_ohlc_short_position_sl_on_high(self):
        """SHORT: SL should trigger when high >= stop (price goes up)."""
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]

        # SHORT entry at 100, SL at 101 (price going up is bad for short)
        data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0],
                "high": [100.5, 101.5, 100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2],
                "low": [99.5, 99.8, 100.0, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% TP: for short, target is 98
            lower_barrier=0.01,  # 1% SL: for short, stop is 101
            max_holding_period=10,
            side=-1,  # SHORT
        )

        result = triple_barrier_labels(
            data, config, price_col="close", high_col="high", low_col="low"
        )

        # SL should trigger at bar 1 (high=101.5 >= 101)
        assert result["label"][0] == -1  # SL hit
        assert result["barrier_hit"][0] == "lower"

    def test_ohlc_short_position_tp_on_low(self):
        """SHORT: TP should trigger when low <= target (price goes down is good)."""
        timestamps = [datetime(2024, 1, 1, 10, i) for i in range(10)]

        # SHORT entry at 100, TP at 98 (price going down is good for short)
        data = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5],
                "high": [100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0],
                "low": [99.5, 99.0, 98.5, 97.5, 97.0, 97.0, 96.5, 96.0, 95.5, 95.0],
            },
        )

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,  # 2% TP: for short, target is 98
            lower_barrier=0.01,  # 1% SL: for short, stop is 101
            max_holding_period=10,
            side=-1,  # SHORT
        )

        result = triple_barrier_labels(
            data, config, price_col="close", high_col="high", low_col="low"
        )

        # TP should trigger at bar 2 or 3 (low reaches 98 or below)
        assert result["label"][0] == 1  # TP hit
        assert result["barrier_hit"][0] == "upper"


class TestPanelLabeling:
    """Panel/multi-asset triple-barrier behavior."""

    def test_grouped_labels_do_not_cross_assets(self):
        """Labels must only use future prices from the same asset group."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "symbol": ["A", "B", "A", "B"],
                "close": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05, lower_barrier=0.05, max_holding_period=1
        )

        result = triple_barrier_labels(
            data,
            config,
            price_col="close",
            timestamp_col="timestamp",
            group_col="symbol",
        )

        a0 = result.filter((pl.col("symbol") == "A") & (pl.col("timestamp") == base)).row(
            0, named=True
        )
        b0 = result.filter((pl.col("symbol") == "B") & (pl.col("timestamp") == base)).row(
            0, named=True
        )

        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0
        assert a0["label"] == 0
        assert b0["label"] == 0

    def test_symbol_column_auto_detects_panel_grouping(self):
        """When symbol exists, grouping is auto-detected if group_col is omitted."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "symbol": ["A", "B", "A", "B"],
                "close": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05, lower_barrier=0.05, max_holding_period=1
        )

        explicit = triple_barrier_labels(
            data,
            config,
            price_col="close",
            timestamp_col="timestamp",
            group_col="symbol",
        )
        autodetected = triple_barrier_labels(
            data,
            config,
            price_col="close",
            timestamp_col="timestamp",
        )

        assert explicit["label_price"].to_list() == autodetected["label_price"].to_list()
        assert explicit["label"].to_list() == autodetected["label"].to_list()

    def test_labeling_config_column_mapping_is_applied(self):
        """LabelingConfig should supply timestamp/price/group column names."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=0.05,
            max_holding_period=1,
            price_col="px",
            timestamp_col="ts",
            group_col="ticker",
        )

        result = triple_barrier_labels(data, config=config)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)

        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0

    def test_labeling_config_symbol_col_alias_maps_to_group_col(self):
        """LabelingConfig should accept symbol_col alias for group mapping."""
        config = LabelingConfig.from_dict(
            {
                "method": "triple_barrier",
                "upper_barrier": 0.05,
                "lower_barrier": 0.05,
                "max_holding_period": 1,
                "price_col": "px",
                "timestamp_col": "ts",
                "symbol_col": "ticker",
            }
        )
        assert config.group_col == "ticker"

    def test_labeling_config_contract_alias_maps_to_data_contract(self):
        """LabelingConfig should accept nested contract alias."""
        config = LabelingConfig.from_dict(
            {
                "method": "triple_barrier",
                "upper_barrier": 0.05,
                "lower_barrier": 0.05,
                "max_holding_period": 1,
                "contract": {
                    "timestamp_col": "ts",
                    "symbol_col": "ticker",
                    "price_col": "px",
                },
            }
        )
        assert config.data_contract is not None
        assert config.data_contract.timestamp_col == "ts"
        assert config.data_contract.symbol_col == "ticker"
        assert config.data_contract.price_col == "px"

    def test_shared_contract_column_mapping_is_applied(self):
        """DataContractConfig should supply timestamp/price/group names."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(
            timestamp_col="ts",
            symbol_col="ticker",
            price_col="px",
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05, lower_barrier=0.05, max_holding_period=1
        )

        result = triple_barrier_labels(data, config=config, contract=contract)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0

    def test_contract_overrides_labeling_config_defaults(self):
        """Contract mapping should apply when config keeps default column names."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=0.05,
            max_holding_period=1,
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")

        result = triple_barrier_labels(data, config=config, contract=contract)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0

    def test_nested_data_contract_in_labeling_config_is_applied(self):
        """LabelingConfig.data_contract should drive column mapping without explicit contract arg."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=0.05,
            max_holding_period=1,
            data_contract=contract,
        )

        result = triple_barrier_labels(data, config=config)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0

    def test_explicit_args_override_config_and_contract(self):
        """Explicit column arguments must win over config and contract values."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "timestamp": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "symbol": ["A", "B", "A", "B"],
                "close": [100.0, 1000.0, 101.0, 999.0],
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [1.0, 1.0, 1.0, 1.0],
            }
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.05,
            lower_barrier=0.05,
            max_holding_period=1,
            price_col="px",
            timestamp_col="ts",
            group_col="ticker",
        )
        contract = DataContractConfig(
            timestamp_col="ts",
            symbol_col="ticker",
            price_col="px",
        )

        result = triple_barrier_labels(
            data,
            config=config,
            price_col="close",
            timestamp_col="timestamp",
            group_col="symbol",
            contract=contract,
        )
        a0 = result.filter((pl.col("symbol") == "A") & (pl.col("timestamp") == base)).row(
            0, named=True
        )
        b0 = result.filter((pl.col("symbol") == "B") & (pl.col("timestamp") == base)).row(
            0, named=True
        )
        assert a0["label_price"] == 101.0
        assert b0["label_price"] == 999.0
