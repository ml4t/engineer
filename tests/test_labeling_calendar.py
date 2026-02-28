"""Tests for calendar-aware labeling functionality."""

from datetime import UTC

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.labeling.calendar import (
    PandasMarketCalendar,
    SimpleTradingCalendar,
    calendar_aware_labels,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_intraday_data():
    """Sample intraday data with overnight gaps."""
    from datetime import datetime

    # Create data with market hours (9:30-16:00) and overnight gaps
    dates = []
    prices = []

    # Day 1: 9:30-16:00 (6.5 hours)
    for hour in range(9, 17):
        for minute in [0, 30] if hour < 16 else [0]:
            if hour == 9 and minute == 0:
                continue  # Skip 9:00, start at 9:30
            dates.append(datetime(2024, 1, 2, hour, minute))
            prices.append(100 + np.random.randn() * 0.5)

    # Overnight gap

    # Day 2: 9:30-16:00
    for hour in range(9, 17):
        for minute in [0, 30] if hour < 16 else [0]:
            if hour == 9 and minute == 0:
                continue
            dates.append(datetime(2024, 1, 3, hour, minute))
            prices.append(100 + np.random.randn() * 0.5)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
        }
    )


@pytest.fixture
def sample_daily_data():
    """Sample daily data for testing."""
    from datetime import datetime, timedelta

    # Create 30 days of data
    dates = [datetime(2024, 1, 2) + timedelta(days=i) for i in range(30)]

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": 100 + np.random.randn(30) * 2,
        }
    )


# ============================================================================
# SimpleTradingCalendar Tests
# ============================================================================


class TestSimpleTradingCalendar:
    """Tests for SimpleTradingCalendar class."""

    def test_initialization(self):
        """Test basic initialization."""
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)  # 8 hours
        assert cal.gap_threshold.total_seconds() == 480 * 60
        assert cal._data is None  # Not fitted yet

    def test_fit_detects_gaps(self, sample_intraday_data):
        """Test that fit() detects session breaks."""
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)  # 8 hours
        cal.fit(sample_intraday_data, timestamp_col="timestamp")

        assert cal._session_breaks is not None
        assert len(cal._session_breaks) > 0  # Should detect overnight gap

    def test_fit_no_gaps_daily_data(self, sample_daily_data):
        """Test fit on daily data (no intraday gaps)."""
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)
        cal.fit(sample_daily_data, timestamp_col="timestamp")

        # Daily data may have weekend gaps > 8 hours
        assert cal._session_breaks is not None

    def test_is_trading_time_always_true(self, sample_intraday_data):
        """Test that is_trading_time always returns True for SimpleTradingCalendar."""
        from datetime import datetime

        cal = SimpleTradingCalendar(gap_threshold_minutes=480)
        cal.fit(sample_intraday_data, timestamp_col="timestamp")

        # Always True (data defines trading times)
        result = cal.is_trading_time(datetime(2024, 1, 2, 10, 0))
        assert result is True

    def test_next_session_break_after_fit(self, sample_intraday_data):
        """Test next_session_break after fitting."""
        from datetime import datetime

        cal = SimpleTradingCalendar(gap_threshold_minutes=480)
        cal.fit(sample_intraday_data, timestamp_col="timestamp")

        # Get next break
        next_break = cal.next_session_break(datetime(2024, 1, 2, 10, 0))

        if next_break is not None:
            assert isinstance(next_break, datetime)

    def test_next_session_break_no_breaks_found(self, sample_daily_data):
        """Test next_session_break when timestamp is after all breaks."""
        from datetime import datetime

        cal = SimpleTradingCalendar(gap_threshold_minutes=10000)  # Very large gap
        cal.fit(sample_daily_data, timestamp_col="timestamp")

        # Request break far in future
        next_break = cal.next_session_break(datetime(2025, 1, 1, 0, 0))

        assert next_break is None  # No breaks after this date

    def test_fit_with_custom_column(self, sample_intraday_data):
        """Test fit with custom timestamp column name."""
        # Rename column
        data = sample_intraday_data.rename({"timestamp": "time"})

        cal = SimpleTradingCalendar()
        cal.fit(data, timestamp_col="time")

        assert cal._session_breaks is not None

    def test_fit_returns_self(self, sample_intraday_data):
        """Test that fit() returns self for chaining."""
        cal = SimpleTradingCalendar()
        result = cal.fit(sample_intraday_data, timestamp_col="timestamp")

        assert result is cal


# ============================================================================
# PandasMarketCalendar Tests
# ============================================================================


class TestPandasMarketCalendar:
    """Tests for PandasMarketCalendar adapter."""

    def test_initialization_requires_library(self):
        """Test that initialization requires pandas_market_calendars."""
        try:
            cal = PandasMarketCalendar("NYSE")
            # If no error, library is installed
            assert cal.calendar_name == "NYSE"
        except ImportError as e:
            # Expected if library not installed
            assert "pandas_market_calendars" in str(e)

    def test_initialization_with_invalid_name(self):
        """Test initialization with invalid calendar name."""
        try:
            import pandas_market_calendars as mcal  # noqa: F401

            with pytest.raises(Exception):  # May raise different errors
                PandasMarketCalendar("INVALID_EXCHANGE")
        except ImportError:
            pytest.skip("pandas_market_calendars not installed")

    def test_is_trading_time(self):
        """Test is_trading_time method."""
        from datetime import datetime

        try:
            cal = PandasMarketCalendar("NYSE")

            # Test known trading time (Tuesday 10:00 AM ET) - use UTC
            result = cal.is_trading_time(datetime(2024, 1, 2, 15, 0, tzinfo=UTC))
            assert isinstance(result, bool)

        except ImportError:
            pytest.skip("pandas_market_calendars not installed")

    def test_next_session_break(self):
        """Test next_session_break method."""
        from datetime import datetime

        try:
            cal = PandasMarketCalendar("NYSE")

            # Get next break from trading time (use UTC)
            next_break = cal.next_session_break(datetime(2024, 1, 2, 15, 0, tzinfo=UTC))

            if next_break is not None:
                assert isinstance(next_break, datetime)

        except ImportError:
            pytest.skip("pandas_market_calendars not installed")


# ============================================================================
# ============================================================================
# calendar_aware_labels Tests
# ============================================================================


class TestCalendarAwareLabels:
    """Tests for calendar_aware_labels function."""

    def test_basic_functionality_auto_calendar(self, sample_daily_data):
        """Test basic labeling with auto calendar detection."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        result = calendar_aware_labels(
            sample_daily_data,
            config,
            calendar="auto",  # Required, no None support
            price_col="close",
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        assert "label_price" in result.columns
        assert len(result) == len(sample_daily_data)

    def test_non_labeling_config_input_raises_actionable_error(self, sample_daily_data):
        """Passing non-LabelingConfig should fail with migration guidance."""
        class LegacyBarrierConfig:
            pass

        with pytest.raises(TypeError, match="Legacy BarrierConfig inputs are no longer supported"):
            calendar_aware_labels(
                sample_daily_data,
                config=LegacyBarrierConfig(),  # type: ignore[arg-type]
                calendar="auto",
            )

    def test_with_simple_calendar(self, sample_intraday_data):
        """Test with SimpleTradingCalendar."""
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)  # 8 hours
        cal.fit(sample_intraday_data, timestamp_col="timestamp")

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        result = calendar_aware_labels(
            sample_intraday_data,
            config,
            calendar=cal,
            price_col="close",
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        assert len(result) == len(sample_intraday_data)

    def test_auto_calendar_detection(self, sample_intraday_data):
        """Test automatic calendar detection from data."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        result = calendar_aware_labels(
            sample_intraday_data,
            config,
            calendar="auto",
            price_col="close",
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        assert len(result) == len(sample_intraday_data)

    def test_string_calendar_name(self, sample_daily_data):
        """Test with string calendar name (NYSE)."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        try:
            result = calendar_aware_labels(
                sample_daily_data,
                config,
                calendar="NYSE",
                price_col="close",
                timestamp_col="timestamp",
            )

            assert "label" in result.columns

        except ImportError:
            # Expected if pandas_market_calendars not installed
            pytest.skip("pandas_market_calendars not installed")

    # Test removed: calendar_library parameter no longer exists
    # Library standardized on pandas-market-calendars only

    def test_respects_session_boundaries(self, sample_intraday_data):
        """Test that labels respect session boundaries."""
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)  # 8 hours
        cal.fit(sample_intraday_data, timestamp_col="timestamp")

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,  # Long enough to cross sessions
        )

        result = calendar_aware_labels(
            sample_intraday_data,
            config,
            calendar=cal,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Should have labels (may timeout at session breaks)
        labels = result["label"].drop_nulls()
        assert len(labels) > 0

    def test_with_side_parameter(self, sample_daily_data):
        """Test calendar-aware labels with position side."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
            side=1,  # Long position
        )

        result = calendar_aware_labels(
            sample_daily_data,
            config,
            calendar="auto",
            price_col="close",
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
        labels = result["label"].drop_nulls()
        assert len(labels) > 0

    def test_uses_config_column_contract_for_panel_data(self):
        """Config-driven price/timestamp/group mapping should be honored."""
        from datetime import datetime, timedelta

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

        result = calendar_aware_labels(data, config=config, calendar="auto")
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == pytest.approx(101.0)
        assert b0["label_price"] == pytest.approx(999.0)

    def test_uses_shared_contract_column_contract_for_panel_data(self):
        """Shared DataContractConfig mapping should be honored."""
        from datetime import datetime, timedelta

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
        )

        result = calendar_aware_labels(data, config=config, calendar="auto", contract=contract)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == pytest.approx(101.0)
        assert b0["label_price"] == pytest.approx(999.0)

    def test_uses_nested_data_contract_from_config_for_panel_data(self):
        """LabelingConfig.data_contract should drive calendar-aware mapping."""
        from datetime import datetime, timedelta

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

        result = calendar_aware_labels(data, config=config, calendar="auto")
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == pytest.approx(101.0)
        assert b0["label_price"] == pytest.approx(999.0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestCalendarIntegration:
    """Integration tests for calendar functionality."""

    def test_multiple_calendar_types(self, sample_daily_data):
        """Test that different calendar types produce valid results."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        # Test with auto calendar
        result_auto = calendar_aware_labels(
            sample_daily_data,
            config,
            calendar="auto",
            price_col="close",
            timestamp_col="timestamp",
        )

        # Test with simple calendar
        cal = SimpleTradingCalendar(gap_threshold_minutes=480)
        cal.fit(sample_daily_data, timestamp_col="timestamp")

        result_simple = calendar_aware_labels(
            sample_daily_data,
            config,
            calendar=cal,
            price_col="close",
            timestamp_col="timestamp",
        )

        # Both should produce valid results
        assert "label" in result_auto.columns
        assert "label" in result_simple.columns

    def test_calendar_with_nan_prices(self, sample_intraday_data):
        """Test calendar-aware labels with NaN prices."""
        # Add some NaN values
        data_with_nan = sample_intraday_data.with_columns(
            pl.when(pl.col("close") > 100.5).then(None).otherwise(pl.col("close")).alias("close")
        )

        cal = SimpleTradingCalendar(gap_threshold_minutes=480)
        cal.fit(data_with_nan, timestamp_col="timestamp")

        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=5,
        )

        # Should not crash
        result = calendar_aware_labels(
            data_with_nan,
            config,
            calendar=cal,
            price_col="close",
            timestamp_col="timestamp",
        )

        assert "label" in result.columns
