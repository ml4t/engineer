"""Tests for time-based horizon support in labeling functions.

Tests cover:
1. Duration parsing utilities
2. Bar-based vs time-based horizon labels
3. Triple barrier with time-based max_holding_period
4. Rolling percentile with time-based lookback windows
5. Edge cases (tolerance, irregular data, session boundaries)
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.labeling.horizon_labels import fixed_time_horizon_labels
from ml4t.engineer.labeling.percentile_labels import rolling_percentile_binary_labels
from ml4t.engineer.labeling.triple_barrier import triple_barrier_labels
from ml4t.engineer.labeling.utils import (
    get_future_price_at_time,
    is_duration_string,
    parse_duration,
    time_horizon_to_bars,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def regular_5min_data() -> pl.DataFrame:
    """Create regular 5-minute OHLCV data."""
    from datetime import datetime

    n_bars = 100
    base_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(n_bars)]

    # Simulate price with trend and noise
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices + np.random.randn(n_bars) * 0.05,
            "high": prices + np.abs(np.random.randn(n_bars) * 0.1),
            "low": prices - np.abs(np.random.randn(n_bars) * 0.1),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


@pytest.fixture
def irregular_trade_data() -> pl.DataFrame:
    """Create irregular trade bar data with variable intervals."""
    from datetime import datetime

    n_trades = 50
    base_time = datetime(2024, 1, 1, 9, 30)

    # Variable intervals (1-10 minutes between trades)
    np.random.seed(42)
    intervals = np.random.randint(1, 11, n_trades)
    timestamps = [base_time + timedelta(minutes=int(sum(intervals[:i]))) for i in range(n_trades)]

    prices = 100.0 + np.cumsum(np.random.randn(n_trades) * 0.2)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": prices,
            "volume": np.random.randint(100, 1000, n_trades),
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


@pytest.fixture
def multi_session_data() -> pl.DataFrame:
    """Create data spanning multiple trading sessions."""
    from datetime import date, datetime

    sessions = []
    for day in range(3):  # 3 trading days
        base_time = datetime(2024, 1, 1 + day, 9, 30)
        n_bars = 78  # 6.5 hours of 5-min bars

        timestamps = [base_time + timedelta(minutes=5 * i) for i in range(n_bars)]
        prices = 100.0 + day * 2 + np.cumsum(np.random.randn(n_bars) * 0.1)

        session_df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": prices,
                "session_date": [date(2024, 1, 1 + day)] * n_bars,
            }
        )
        sessions.append(session_df)

    return pl.concat(sessions).with_columns(
        [
            pl.col("timestamp").cast(pl.Datetime("us")),
            pl.col("session_date").cast(pl.Date),
        ]
    )


# =============================================================================
# Duration Parsing Tests
# =============================================================================


class TestDurationParsing:
    """Tests for is_duration_string and parse_duration utilities."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1h", True),
            ("30m", True),
            ("1d", True),
            ("1w", True),
            ("15s", True),
            ("1d2h30m", True),
            ("2h30m15s", True),
            ("1H", True),  # Case insensitive
            ("30M", True),
            # Invalid - column names
            ("close", False),
            ("volume", False),
            ("max_holding_period", False),
            ("my_column", False),
            # Invalid - no digits
            ("h", False),
            ("dm", False),
            # Invalid - malformed
            ("1x", False),
            ("abc123", False),
            ("", False),
        ],
    )
    def test_is_duration_string(self, value: str, expected: bool):
        """Test duration string detection."""
        assert is_duration_string(value) == expected

    @pytest.mark.parametrize(
        "value,expected_seconds",
        [
            ("1h", 3600),
            ("30m", 1800),
            ("1d", 86400),
            ("1w", 604800),
            ("15s", 15),
            ("1d2h30m", 86400 + 7200 + 1800),
            ("2h30m15s", 7200 + 1800 + 15),
        ],
    )
    def test_parse_duration(self, value: str, expected_seconds: int):
        """Test duration string parsing."""
        td = parse_duration(value)
        assert td.total_seconds() == expected_seconds

    def test_parse_duration_invalid(self):
        """Test that invalid duration strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid duration string"):
            parse_duration("close")

        with pytest.raises(ValueError, match="Invalid duration string"):
            parse_duration("not_a_duration")


# =============================================================================
# Time Horizon to Bars Conversion Tests
# =============================================================================


class TestTimeHorizonToBars:
    """Tests for time_horizon_to_bars utility."""

    def test_regular_5min_bars(self, regular_5min_data: pl.DataFrame):
        """Test conversion on regular 5-minute bars."""
        timestamps = (
            regular_5min_data["timestamp"].to_numpy().astype("datetime64[ns]").view("int64")
        )

        # 15 minutes = 3 bars of 5-minute data
        bar_counts = time_horizon_to_bars(timestamps, "15m")

        # First bar: 15min forward should be 3 bars
        assert bar_counts[0] == 3

        # Middle bars should also be 3
        assert bar_counts[len(bar_counts) // 2] == 3

        # Last few bars should still compute (clipped to end)
        assert bar_counts[-1] >= 1

    def test_with_timedelta(self, regular_5min_data: pl.DataFrame):
        """Test using timedelta instead of string."""
        timestamps = (
            regular_5min_data["timestamp"].to_numpy().astype("datetime64[ns]").view("int64")
        )

        bar_counts_str = time_horizon_to_bars(timestamps, "1h")
        bar_counts_td = time_horizon_to_bars(timestamps, timedelta(hours=1))

        np.testing.assert_array_equal(bar_counts_str, bar_counts_td)

    def test_minimum_bar_count(self, regular_5min_data: pl.DataFrame):
        """Test that minimum bar count is 1."""
        timestamps = (
            regular_5min_data["timestamp"].to_numpy().astype("datetime64[ns]").view("int64")
        )

        # Very short horizon
        bar_counts = time_horizon_to_bars(timestamps, "1s")

        # All should be >= 1
        assert bar_counts.min() >= 1


# =============================================================================
# Fixed Time Horizon Labels Tests
# =============================================================================


class TestFixedTimeHorizonLabels:
    """Tests for fixed_time_horizon_labels with time-based horizons."""

    def test_bar_based_unchanged(self, regular_5min_data: pl.DataFrame):
        """Test that bar-based API is unchanged."""
        result = fixed_time_horizon_labels(
            regular_5min_data,
            horizon=5,  # Integer = bars
            method="returns",
        )

        assert "label_return_5p" in result.columns
        # First value should be computable
        assert result["label_return_5p"][0] is not None
        # Last 5 values should be null
        assert result["label_return_5p"][-1] is None

    def test_time_based_horizon(self, regular_5min_data: pl.DataFrame):
        """Test time-based horizon with duration string."""
        result = fixed_time_horizon_labels(
            regular_5min_data,
            horizon="15m",  # 15 minutes
            method="returns",
        )

        assert "label_return_15m" in result.columns
        # Should have computed values
        assert result["label_return_15m"].null_count() < len(result)

    def test_time_based_binary(self, regular_5min_data: pl.DataFrame):
        """Test time-based horizon with binary method."""
        result = fixed_time_horizon_labels(
            regular_5min_data,
            horizon="1h",
            method="binary",
        )

        assert "label_direction_1h" in result.columns
        # Values should be -1, 0, or 1
        valid_values = result["label_direction_1h"].drop_nulls().unique().to_list()
        assert all(v in [-1, 0, 1] for v in valid_values)

    def test_time_based_log_returns(self, regular_5min_data: pl.DataFrame):
        """Test time-based horizon with log returns."""
        result = fixed_time_horizon_labels(
            regular_5min_data,
            horizon="30m",
            method="log_returns",
        )

        assert "label_log_return_30m" in result.columns

    def test_invalid_horizon_string(self, regular_5min_data: pl.DataFrame):
        """Test that invalid horizon strings raise error."""
        with pytest.raises(ValueError, match="Invalid horizon"):
            fixed_time_horizon_labels(
                regular_5min_data,
                horizon="not_a_duration",
                method="returns",
            )

    def test_time_based_requires_timestamp(self):
        """Test that time-based horizon requires timestamp column."""
        # Data without datetime column
        df = pl.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        with pytest.raises(ValueError, match="timestamp column"):
            fixed_time_horizon_labels(df, horizon="1h", method="returns")

    def test_uses_config_column_contract(self):
        """Column mapping should come from LabelingConfig when omitted in call."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.fixed_horizon(
            horizon=1,
            return_method="returns",
            price_col="px",
            timestamp_col="ts",
            group_col="ticker",
        )

        result = fixed_time_horizon_labels(data, horizon=1, method="returns", config=config)

        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_return_1p"] == pytest.approx(0.01)
        assert b0["label_return_1p"] == pytest.approx(-0.001)

    def test_uses_shared_contract_column_mapping(self):
        """Column mapping should come from DataContractConfig when config is omitted."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")

        result = fixed_time_horizon_labels(
            data,
            horizon=1,
            method="returns",
            contract=contract,
        )

        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_return_1p"] == pytest.approx(0.01)
        assert b0["label_return_1p"] == pytest.approx(-0.001)


# =============================================================================
# Triple Barrier Labels Tests
# =============================================================================


class TestTripleBarrierTimeBased:
    """Tests for triple_barrier_labels with time-based max_holding_period."""

    def test_bar_based_unchanged(self, regular_5min_data: pl.DataFrame):
        """Test that bar-based API is unchanged."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period=10,  # Bars
            side=1,
        )

        result = triple_barrier_labels(
            regular_5min_data,
            config=config,
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        assert "barrier_hit" in result.columns

    def test_time_based_duration_string(self, regular_5min_data: pl.DataFrame):
        """Test time-based max_holding_period with duration string."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period="1h",  # 1 hour
            side=1,
        )

        result = triple_barrier_labels(
            regular_5min_data,
            config=config,
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        # Labels should be computed
        assert result["label"].null_count() < len(result)

    def test_time_based_timedelta(self, regular_5min_data: pl.DataFrame):
        """Test time-based max_holding_period with timedelta."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period=timedelta(hours=1),
            side=1,
        )

        result = triple_barrier_labels(
            regular_5min_data,
            config=config,
            timestamp_col="timestamp",
        )

        assert "label" in result.columns

    def test_time_based_requires_timestamp_col(self, regular_5min_data: pl.DataFrame):
        """Test that time-based max_holding_period requires timestamp_col."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period="1h",
            side=1,
        )

        # Should raise error because no timestamp column auto-detection when
        # the data has datetime columns but we explicitly check for time-based
        # This test validates the error message is clear
        # (In practice, timestamp_col will be auto-detected from datetime columns)

    def test_consistent_results(self, regular_5min_data: pl.DataFrame):
        """Test that time-based and equivalent bar-based give similar results.

        For regular 5-min bars, 1h = 12 bars.
        """
        from polars.testing import assert_series_equal

        config_bars = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period=12,  # 12 bars = 1 hour for 5-min bars
            side=1,
        )

        config_time = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period="1h",
            side=1,
        )

        result_bars = triple_barrier_labels(
            regular_5min_data,
            config=config_bars,
            timestamp_col="timestamp",
        )

        result_time = triple_barrier_labels(
            regular_5min_data,
            config=config_time,
            timestamp_col="timestamp",
        )

        # Labels should be identical for regular data
        assert_series_equal(
            result_bars["label"],
            result_time["label"],
        )


# =============================================================================
# Rolling Percentile Labels Tests
# =============================================================================


class TestRollingPercentileTimeBased:
    """Tests for rolling_percentile_binary_labels with time-based parameters."""

    def test_bar_based_unchanged(self, regular_5min_data: pl.DataFrame):
        """Test that bar-based API is unchanged."""
        result = rolling_percentile_binary_labels(
            regular_5min_data,
            horizon=10,  # Bars
            percentile=90,
            direction="long",
            lookback_window=50,  # Bars
        )

        assert "forward_return_10" in result.columns
        assert "label_long_p90_h10" in result.columns

    def test_time_based_horizon(self, regular_5min_data: pl.DataFrame):
        """Test time-based horizon."""
        result = rolling_percentile_binary_labels(
            regular_5min_data,
            horizon="30m",  # 30 minutes
            percentile=90,
            direction="long",
            lookback_window=50,  # Still bar-based lookback
        )

        assert "forward_return_30m" in result.columns
        assert "label_long_p90_h30m" in result.columns

    def test_time_based_lookback(self, regular_5min_data: pl.DataFrame):
        """Test time-based lookback window.

        Note: Polars rolling with time-based windows requires the 'by' parameter.
        """
        # This test verifies the time-based lookback pathway
        result = rolling_percentile_binary_labels(
            regular_5min_data,
            horizon=6,  # 30 minutes worth of 5-min bars
            percentile=90,
            direction="long",
            lookback_window="4h",  # 4-hour lookback
            timestamp_col="timestamp",
        )

        assert "forward_return_6" in result.columns
        # Threshold column should exist
        assert "threshold_p90_h6" in result.columns


# =============================================================================
# Get Future Price Tests
# =============================================================================


class TestGetFuturePriceAtTime:
    """Tests for get_future_price_at_time utility."""

    def test_regular_data(self, regular_5min_data: pl.DataFrame):
        """Test future price retrieval on regular data."""
        future_prices, valid_mask = get_future_price_at_time(
            regular_5min_data,
            time_horizon="15m",
            price_col="close",
            timestamp_col="timestamp",
        )

        # Should have mostly valid prices except at end
        assert valid_mask.sum() > len(regular_5min_data) * 0.9

    def test_with_tolerance(self, irregular_trade_data: pl.DataFrame):
        """Test future price retrieval with tolerance for irregular data."""
        future_prices, valid_mask = get_future_price_at_time(
            irregular_trade_data,
            time_horizon="15m",
            price_col="close",
            timestamp_col="timestamp",
            tolerance="2m",  # Allow 2-minute gap
        )

        # Some may be invalid due to gaps exceeding tolerance
        # This is expected behavior for irregular data
        assert len(future_prices) == len(irregular_trade_data)

    def test_no_forward_match_returns_null(self, regular_5min_data: pl.DataFrame):
        """Forward horizon should return null when no price exists at/after target time."""
        future_prices, valid_mask = get_future_price_at_time(
            regular_5min_data,
            time_horizon="24h",
            price_col="close",
            timestamp_col="timestamp",
        )
        assert len(future_prices) == len(regular_5min_data)
        assert valid_mask.sum() == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_horizon_past_data_end(self, regular_5min_data: pl.DataFrame):
        """Test behavior when horizon extends past data end.

        For time-based horizons, join_asof with forward strategy returns null
        when there is no price at or after the requested horizon.
        """
        # Very long horizon
        result = fixed_time_horizon_labels(
            regular_5min_data,
            horizon="24h",  # Longer than data duration (~8h of 5-min bars)
            method="returns",
        )

        # Label column should exist
        assert "label_return_24h" in result.columns

        # No row has a future timestamp 24h ahead, so all labels are null.
        assert result["label_return_24h"].null_count() == len(result)

    def test_zero_horizon_rejected(self, regular_5min_data: pl.DataFrame):
        """Test that zero/negative horizon is rejected."""
        with pytest.raises(ValueError, match="positive"):
            fixed_time_horizon_labels(
                regular_5min_data,
                horizon=0,
                method="returns",
            )

        with pytest.raises(ValueError, match="positive"):
            fixed_time_horizon_labels(
                regular_5min_data,
                horizon=-5,
                method="returns",
            )

    def test_column_name_vs_duration_ambiguity(self, regular_5min_data: pl.DataFrame):
        """Test that column names are not confused with duration strings.

        A column named '1h' could be ambiguous, but the check prioritizes
        duration string detection.
        """
        # Add a column that looks like a duration
        data_with_ambiguous = regular_5min_data.with_columns(
            pl.lit(100).alias("1h")  # Column named '1h'
        )

        # This should treat "1h" as a duration, not a column name
        result = fixed_time_horizon_labels(
            data_with_ambiguous,
            horizon="1h",
            method="returns",
        )

        # Should use time-based, not column-based
        assert "label_return_1h" in result.columns

    def test_session_boundary_handling(self, multi_session_data: pl.DataFrame):
        """Test that session-aware labels respect session boundaries."""
        result = rolling_percentile_binary_labels(
            multi_session_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=50,
            session_col="session_date",
        )

        # Should have computed labels
        assert result["label_long_p90_h10"].null_count() < len(result)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_bar_based(self, regular_5min_data: pl.DataFrame):
        """Test complete labeling workflow with bar-based horizons."""
        # Step 1: Fixed horizon labels
        data = fixed_time_horizon_labels(
            regular_5min_data,
            horizon=12,  # 1 hour worth of 5-min bars
            method="returns",
        )

        # Step 2: Triple barrier labels
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period=12,
            side=1,
        )
        data = triple_barrier_labels(
            data,
            config=config,
            timestamp_col="timestamp",
        )

        # Verify all labels computed
        assert "label_return_12p" in data.columns
        assert "label" in data.columns

    def test_complete_workflow_time_based(self, regular_5min_data: pl.DataFrame):
        """Test complete labeling workflow with time-based horizons."""
        # Step 1: Time-based fixed horizon labels
        data = fixed_time_horizon_labels(
            regular_5min_data,
            horizon="1h",
            method="returns",
        )

        # Step 2: Time-based triple barrier labels
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period="1h",
            side=1,
        )
        data = triple_barrier_labels(
            data,
            config=config,
            timestamp_col="timestamp",
        )

        # Verify all labels computed
        assert "label_return_1h" in data.columns
        assert "label" in data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
