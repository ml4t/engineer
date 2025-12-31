"""Tests for bars module - base, tick, and volume samplers."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.tick import TickBarSampler
from ml4t.engineer.bars.volume import DollarBarSampler, VolumeBarSampler
from ml4t.engineer.core.exceptions import DataValidationError


@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing."""
    n = 100
    np.random.seed(42)

    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 10000) for i in range(n)]
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.randint(10, 100, n).astype(float)
    sides = np.random.choice([-1, 1], n)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.fixture
def large_tick_data():
    """Create larger tick data for edge case testing."""
    n = 1000
    np.random.seed(42)

    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 1000) for i in range(n)]
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.randint(10, 200, n).astype(float)
    sides = np.random.choice([-1, 1], n)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


class TestBarSamplerBase:
    """Tests for BarSampler base class validation."""

    def test_validate_data_missing_timestamp(self):
        """Test validation fails when timestamp column is missing."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "price": [100.0, 101.0],
                "volume": [10.0, 20.0],
            }
        )

        with pytest.raises(DataValidationError, match="Missing required columns"):
            sampler._validate_data(data)

    def test_validate_data_missing_price(self):
        """Test validation fails when price column is missing."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "volume": [10.0],
            }
        )

        with pytest.raises(DataValidationError, match="Missing required columns"):
            sampler._validate_data(data)

    def test_validate_data_missing_volume(self):
        """Test validation fails when volume column is missing."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
            }
        )

        with pytest.raises(DataValidationError, match="Missing required columns"):
            sampler._validate_data(data)

    def test_validate_data_empty_df(self):
        """Test validation passes for empty DataFrame."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )
        # Should not raise - empty data is valid
        sampler._validate_data(data)

    def test_validate_data_non_numeric_price(self):
        """Test validation fails for non-numeric price."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": ["invalid"],
                "volume": [10.0],
            }
        )

        with pytest.raises(DataValidationError, match="Price column must be numeric"):
            sampler._validate_data(data)

    def test_validate_data_non_numeric_volume(self):
        """Test validation fails for non-numeric volume."""
        sampler = TickBarSampler(ticks_per_bar=10)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": ["invalid"],
            }
        )

        with pytest.raises(DataValidationError, match="Volume column must be numeric"):
            sampler._validate_data(data)

    def test_create_ohlcv_bar_empty(self):
        """Test _create_ohlcv_bar returns empty dict for empty ticks."""
        sampler = TickBarSampler(ticks_per_bar=10)
        empty_ticks = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        bar = sampler._create_ohlcv_bar(empty_ticks)
        assert bar == {}

    def test_create_ohlcv_bar_with_additional_cols(self, sample_tick_data):
        """Test _create_ohlcv_bar with additional columns."""
        sampler = TickBarSampler(ticks_per_bar=10)
        ticks = sample_tick_data.head(10)

        bar = sampler._create_ohlcv_bar(
            ticks, additional_cols={"custom_col": 123, "another_col": "value"}
        )

        assert "custom_col" in bar
        assert bar["custom_col"] == 123
        assert "another_col" in bar
        assert bar["another_col"] == "value"


class TestTickBarSampler:
    """Tests for TickBarSampler."""

    def test_init_valid_ticks_per_bar(self):
        """Test initialization with valid ticks_per_bar."""
        sampler = TickBarSampler(ticks_per_bar=50)
        assert sampler.ticks_per_bar == 50

    def test_init_invalid_ticks_per_bar_zero(self):
        """Test initialization fails with zero ticks_per_bar."""
        with pytest.raises(ValueError, match="ticks_per_bar must be positive"):
            TickBarSampler(ticks_per_bar=0)

    def test_init_invalid_ticks_per_bar_negative(self):
        """Test initialization fails with negative ticks_per_bar."""
        with pytest.raises(ValueError, match="ticks_per_bar must be positive"):
            TickBarSampler(ticks_per_bar=-10)

    def test_sample_basic(self, sample_tick_data):
        """Test basic tick bar sampling."""
        sampler = TickBarSampler(ticks_per_bar=10)
        bars = sampler.sample(sample_tick_data)

        # 100 ticks / 10 per bar = 10 complete bars
        assert len(bars) == 10

        # Check bar structure
        assert "timestamp" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns
        assert "tick_count" in bars.columns

    def test_sample_with_incomplete_bar(self, sample_tick_data):
        """Test sampling with include_incomplete=True."""
        # 100 ticks / 15 per bar = 6 complete + 10 remaining
        sampler = TickBarSampler(ticks_per_bar=15)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_without) == 6
        assert len(bars_with) == 7  # 6 complete + 1 incomplete

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = TickBarSampler(ticks_per_bar=10)
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        bars = sampler.sample(empty_data)
        assert len(bars) == 0

    def test_sample_insufficient_data(self):
        """Test sampling with fewer ticks than ticks_per_bar."""
        sampler = TickBarSampler(ticks_per_bar=100)
        small_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
                "price": [100.0 + i for i in range(5)],
                "volume": [10.0] * 5,
            }
        )

        # Without incomplete
        bars = sampler.sample(small_data, include_incomplete=False)
        assert len(bars) == 0

        # With incomplete
        bars_with = sampler.sample(small_data, include_incomplete=True)
        assert len(bars_with) == 1

    def test_sample_tick_count_correct(self, sample_tick_data):
        """Test that tick_count is correct in each bar."""
        sampler = TickBarSampler(ticks_per_bar=10)
        bars = sampler.sample(sample_tick_data)

        # All complete bars should have exactly 10 ticks
        tick_counts = bars["tick_count"].to_list()
        assert all(tc == 10 for tc in tick_counts)

    def test_sample_ohlcv_values(self):
        """Test OHLCV values are calculated correctly."""
        sampler = TickBarSampler(ticks_per_bar=5)

        # Create data with known OHLCV
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
                "price": [100.0, 105.0, 95.0, 102.0, 103.0],  # O=100, H=105, L=95, C=103
                "volume": [10.0, 20.0, 30.0, 40.0, 50.0],  # Total = 150
            }
        )

        bars = sampler.sample(data)

        assert len(bars) == 1
        assert bars["open"][0] == 100.0
        assert bars["high"][0] == 105.0
        assert bars["low"][0] == 95.0
        assert bars["close"][0] == 103.0
        assert bars["volume"][0] == 150.0


class TestVolumeBarSampler:
    """Tests for VolumeBarSampler."""

    def test_init_valid_volume_per_bar(self):
        """Test initialization with valid volume_per_bar."""
        sampler = VolumeBarSampler(volume_per_bar=1000.0)
        assert sampler.volume_per_bar == 1000.0

    def test_init_invalid_volume_per_bar_zero(self):
        """Test initialization fails with zero volume_per_bar."""
        with pytest.raises(ValueError, match="volume_per_bar must be positive"):
            VolumeBarSampler(volume_per_bar=0)

    def test_init_invalid_volume_per_bar_negative(self):
        """Test initialization fails with negative volume_per_bar."""
        with pytest.raises(ValueError, match="volume_per_bar must be positive"):
            VolumeBarSampler(volume_per_bar=-100.0)

    def test_sample_missing_side_column(self):
        """Test sampling fails without side column."""
        sampler = VolumeBarSampler(volume_per_bar=100.0)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": [50.0],
            }
        )

        with pytest.raises(DataValidationError, match="Volume bars require 'side' column"):
            sampler.sample(data)

    def test_sample_basic(self, sample_tick_data):
        """Test basic volume bar sampling."""
        # Total volume is roughly 55*100 = 5500
        sampler = VolumeBarSampler(volume_per_bar=500.0)
        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0

        # Check bar structure
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = VolumeBarSampler(volume_per_bar=100.0)
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
                "side": [],
            }
        )

        bars = sampler.sample(empty_data)
        assert len(bars) == 0

    def test_sample_with_incomplete_bar(self, sample_tick_data):
        """Test sampling with include_incomplete=True."""
        sampler = VolumeBarSampler(volume_per_bar=1000.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        # Should have at least one more bar with incomplete
        assert len(bars_with) >= len(bars_without)

    def test_sample_buy_sell_breakdown(self):
        """Test buy/sell volume breakdown is correct."""
        sampler = VolumeBarSampler(volume_per_bar=100.0)

        # Create data with known buy/sell volumes
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(4)],
                "price": [100.0, 101.0, 99.0, 100.5],
                "volume": [30.0, 40.0, 20.0, 50.0],  # Total = 140 > 100
                "side": [1, 1, -1, -1],  # Buy: 70, Sell: 70
            }
        )

        bars = sampler.sample(data)

        assert len(bars) >= 1
        # First bar should have buy_volume and sell_volume
        assert bars["buy_volume"][0] >= 0
        assert bars["sell_volume"][0] >= 0


class TestDollarBarSampler:
    """Tests for DollarBarSampler."""

    def test_init_valid_dollars_per_bar(self):
        """Test initialization with valid dollars_per_bar."""
        sampler = DollarBarSampler(dollars_per_bar=1_000_000.0)
        assert sampler.dollars_per_bar == 1_000_000.0

    def test_init_invalid_dollars_per_bar_zero(self):
        """Test initialization fails with zero dollars_per_bar."""
        with pytest.raises(ValueError, match="dollars_per_bar must be positive"):
            DollarBarSampler(dollars_per_bar=0)

    def test_init_invalid_dollars_per_bar_negative(self):
        """Test initialization fails with negative dollars_per_bar."""
        with pytest.raises(ValueError, match="dollars_per_bar must be positive"):
            DollarBarSampler(dollars_per_bar=-1000.0)

    def test_sample_basic(self, sample_tick_data):
        """Test basic dollar bar sampling."""
        # Avg price ~100, avg volume ~55, so avg dollar ~5500 per tick
        # Total dollar volume roughly 550000
        sampler = DollarBarSampler(dollars_per_bar=50000.0)
        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0

        # Check bar structure
        assert "dollar_volume" in bars.columns
        assert "vwap" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = DollarBarSampler(dollars_per_bar=1000.0)
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        bars = sampler.sample(empty_data)
        assert len(bars) == 0

    def test_sample_with_incomplete_bar(self, sample_tick_data):
        """Test sampling with include_incomplete=True."""
        sampler = DollarBarSampler(dollars_per_bar=100000.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        # Should have at least one more bar with incomplete
        assert len(bars_with) >= len(bars_without)

    def test_sample_vwap_calculation(self):
        """Test VWAP is calculated correctly."""
        sampler = DollarBarSampler(dollars_per_bar=5000.0)

        # Create data with known VWAP
        # VWAP = sum(price * volume) / sum(volume)
        # = (100*10 + 110*20 + 90*30) / (10+20+30)
        # = (1000 + 2200 + 2700) / 60 = 5900 / 60 = 98.333...
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(3)],
                "price": [100.0, 110.0, 90.0],
                "volume": [10.0, 20.0, 30.0],  # Dollar volume = 5900
            }
        )

        bars = sampler.sample(data)

        assert len(bars) == 1
        assert abs(bars["vwap"][0] - 98.333333) < 0.001
        assert abs(bars["dollar_volume"][0] - 5900.0) < 0.01

    def test_sample_zero_volume_vwap(self):
        """Test VWAP calculation with zero total volume."""
        sampler = DollarBarSampler(dollars_per_bar=0.01)  # Very small threshold

        # Create data where volume is very small but non-zero
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": [0.001],  # Dollar volume = 0.1 > 0.01
            }
        )

        bars = sampler.sample(data)

        assert len(bars) == 1
        # VWAP should be computable
        assert not np.isnan(bars["vwap"][0])


class TestBarsIntegration:
    """Integration tests for bars module."""

    def test_tick_bar_volume_consistency(self, large_tick_data):
        """Test that tick bar total volume equals input data volume."""
        sampler = TickBarSampler(ticks_per_bar=50)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        # Total volume should match
        input_volume = large_tick_data["volume"].sum()
        output_volume = bars["volume"].sum()

        assert abs(input_volume - output_volume) < 0.01

    def test_volume_bar_consistency(self, large_tick_data):
        """Test volume bar consistency."""
        sampler = VolumeBarSampler(volume_per_bar=5000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        # Each complete bar should have approximately volume_per_bar
        if len(bars) > 1:
            # Skip last bar (may be incomplete)
            complete_bars = bars.head(len(bars) - 1)
            for vol in complete_bars["volume"]:
                assert vol >= 5000.0  # At least threshold

    def test_dollar_bar_consistency(self, large_tick_data):
        """Test dollar bar consistency."""
        sampler = DollarBarSampler(dollars_per_bar=500000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        # Each complete bar should have approximately dollars_per_bar
        if len(bars) > 1:
            # Skip last bar (may be incomplete)
            complete_bars = bars.head(len(bars) - 1)
            for dol_vol in complete_bars["dollar_volume"]:
                assert dol_vol >= 500000.0  # At least threshold
