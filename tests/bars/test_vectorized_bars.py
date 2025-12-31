"""Tests for vectorized bar samplers.

Tests for VolumeBarSamplerVectorized, DollarBarSamplerVectorized,
TickBarSamplerVectorized, and ImbalanceBarSamplerVectorized.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.vectorized import (
    DollarBarSamplerVectorized,
    ImbalanceBarSamplerVectorized,
    TickBarSamplerVectorized,
    VolumeBarSamplerVectorized,
    _assign_dollar_bar_ids,
    _assign_volume_bar_ids,
)
from ml4t.engineer.core.exceptions import DataValidationError


@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing."""
    n = 200
    np.random.seed(42)

    # Use millisecond increments that stay within microsecond range
    timestamps = [datetime(2024, 1, 1, 9, 30, i // 60, (i % 60) * 1000) for i in range(n)]
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

    # Spread timestamps across minutes to stay within microsecond range
    timestamps = [datetime(2024, 1, 1, 9 + i // 3600, (i // 60) % 60, i % 60, 0) for i in range(n)]
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


class TestNumbaFunctions:
    """Tests for Numba-compiled helper functions."""

    def test_assign_volume_bar_ids_basic(self):
        """Test volume bar ID assignment."""
        volumes = np.array([50.0, 60.0, 40.0, 100.0, 50.0])
        threshold = 100.0

        bar_ids = _assign_volume_bar_ids(volumes, threshold)

        assert len(bar_ids) == 5
        assert bar_ids[0] == 0  # First tick, bar 0
        assert bar_ids[1] == 0  # 50+60=110 >= 100, stays in bar 0
        # After 110, reset to 0, then 40, bar 1
        assert bar_ids[2] == 1
        # 40 + 100 = 140 >= 100, bar 1
        assert bar_ids[3] == 1
        # Reset, 50 < 100, bar 2
        assert bar_ids[4] == 2

    def test_assign_volume_bar_ids_large(self):
        """Test volume bar ID assignment with large data."""
        np.random.seed(42)
        volumes = np.random.randint(10, 100, 10000).astype(float)
        threshold = 1000.0

        bar_ids = _assign_volume_bar_ids(volumes, threshold)

        assert len(bar_ids) == 10000
        # Bar IDs should be monotonically non-decreasing
        assert np.all(np.diff(bar_ids) >= 0)

    def test_assign_dollar_bar_ids_basic(self):
        """Test dollar bar ID assignment."""
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        volumes = np.array([10.0, 20.0, 15.0, 25.0])
        threshold = 2500.0  # Dollar threshold

        bar_ids = _assign_dollar_bar_ids(prices, volumes, threshold)

        assert len(bar_ids) == 4
        # Dollar volumes: 1000, 2000, 1500, 2500
        # Cumulative: 1000, 3000 >= 2500 (bar 0), reset
        # Then: 1500, 4000 >= 2500 (bar 1)

    def test_assign_dollar_bar_ids_varying_prices(self):
        """Test dollar bar ID assignment with varying prices."""
        prices = np.array([100.0, 110.0, 90.0, 105.0])
        volumes = np.array([10.0, 10.0, 10.0, 10.0])
        threshold = 2000.0

        bar_ids = _assign_dollar_bar_ids(prices, volumes, threshold)

        assert len(bar_ids) == 4
        # Dollar volumes: 1000, 1100, 900, 1050
        # Cumulative: 1000, 2100 >= 2000, reset
        # Then: 900, 1950, next tick would exceed


class TestVolumeBarSamplerVectorized:
    """Tests for VolumeBarSamplerVectorized."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=1000.0)
        assert sampler.volume_per_bar == 1000.0

    def test_init_invalid_zero(self):
        """Test initialization fails with zero volume."""
        with pytest.raises(ValueError, match="volume_per_bar must be positive"):
            VolumeBarSamplerVectorized(volume_per_bar=0)

    def test_init_invalid_negative(self):
        """Test initialization fails with negative volume."""
        with pytest.raises(ValueError, match="volume_per_bar must be positive"):
            VolumeBarSamplerVectorized(volume_per_bar=-100.0)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=500.0)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Volume bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=500.0)
        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0
        assert "timestamp" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns
        assert "tick_count" in bars.columns
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=100.0)
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

    def test_sample_include_incomplete(self, sample_tick_data):
        """Test sampling with incomplete bar."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=10000.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_sample_all_incomplete(self):
        """Test when all data forms incomplete bar."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=100000.0)

        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(5)],
                "price": [100.0] * 5,
                "volume": [10.0] * 5,  # Total = 50, far less than threshold
                "side": [1, -1, 1, -1, 1],
            }
        )

        bars = sampler.sample(data, include_incomplete=False)
        assert len(bars) == 0

    def test_empty_result_schema(self):
        """Test _empty_volume_bars_df returns correct schema."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=100.0)
        empty_df = sampler._empty_volume_bars_df()

        expected_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "buy_volume",
            "sell_volume",
        ]
        assert list(empty_df.columns) == expected_cols


class TestDollarBarSamplerVectorized:
    """Tests for DollarBarSamplerVectorized."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=1000000.0)
        assert sampler.dollars_per_bar == 1000000.0

    def test_init_invalid_zero(self):
        """Test initialization fails with zero dollars."""
        with pytest.raises(ValueError, match="dollars_per_bar must be positive"):
            DollarBarSamplerVectorized(dollars_per_bar=0)

    def test_init_invalid_negative(self):
        """Test initialization fails with negative dollars."""
        with pytest.raises(ValueError, match="dollars_per_bar must be positive"):
            DollarBarSamplerVectorized(dollars_per_bar=-100.0)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=50000.0)
        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0
        assert "dollar_volume" in bars.columns
        assert "vwap" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=100.0)
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        bars = sampler.sample(empty_data)
        assert len(bars) == 0

    def test_sample_include_incomplete(self, sample_tick_data):
        """Test sampling with incomplete bar."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=1000000.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_vwap_calculation(self):
        """Test VWAP is calculated correctly."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=5000.0)

        # VWAP = sum(price * volume) / sum(volume)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(3)],
                "price": [100.0, 110.0, 90.0],
                "volume": [10.0, 20.0, 30.0],  # Dollar volume = 5900
            }
        )

        bars = sampler.sample(data)

        assert len(bars) == 1
        expected_vwap = 5900.0 / 60.0  # 98.333...
        assert abs(bars["vwap"][0] - expected_vwap) < 0.001

    def test_empty_result_schema(self):
        """Test _empty_dollar_bars_df returns correct schema."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=100.0)
        empty_df = sampler._empty_dollar_bars_df()

        expected_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "dollar_volume",
            "vwap",
        ]
        assert list(empty_df.columns) == expected_cols


class TestTickBarSamplerVectorized:
    """Tests for TickBarSamplerVectorized."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=50)
        assert sampler.ticks_per_bar == 50

    def test_init_invalid_zero(self):
        """Test initialization fails with zero ticks."""
        with pytest.raises(ValueError, match="ticks_per_bar must be positive"):
            TickBarSamplerVectorized(ticks_per_bar=0)

    def test_init_invalid_negative(self):
        """Test initialization fails with negative ticks."""
        with pytest.raises(ValueError, match="ticks_per_bar must be positive"):
            TickBarSamplerVectorized(ticks_per_bar=-10)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data)

        # 200 ticks / 20 per bar = 10 complete bars
        assert len(bars) == 10
        assert "tick_count" in bars.columns
        # All complete bars should have exactly 20 ticks
        tick_counts = bars["tick_count"].to_list()
        assert tick_counts == [20] * 10

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=10)
        empty_data = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        bars = sampler.sample(empty_data)
        assert len(bars) == 0

    def test_sample_include_incomplete(self, sample_tick_data):
        """Test sampling with incomplete bar."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=30)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        # 200 / 30 = 6 complete + 20 remaining
        assert len(bars_without) == 6
        assert len(bars_with) == 7

    def test_sample_insufficient_data(self):
        """Test sampling with fewer ticks than threshold."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=100)

        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(10)],
                "price": [100.0] * 10,
                "volume": [10.0] * 10,
            }
        )

        bars_without = sampler.sample(data, include_incomplete=False)
        bars_with = sampler.sample(data, include_incomplete=True)

        assert len(bars_without) == 0
        assert len(bars_with) == 1

    def test_empty_result_schema(self):
        """Test _empty_tick_bars_df returns correct schema."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=10)
        empty_df = sampler._empty_tick_bars_df()

        expected_cols = ["timestamp", "open", "high", "low", "close", "volume", "tick_count"]
        assert list(empty_df.columns) == expected_cols


class TestImbalanceBarSamplerVectorized:
    """Tests for ImbalanceBarSamplerVectorized."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_custom_alpha(self):
        """Test initialization with custom alpha."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=100, alpha=0.2)
        assert sampler.alpha == 0.2

    def test_init_invalid_ticks_zero(self):
        """Test initialization fails with zero expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            ImbalanceBarSamplerVectorized(expected_ticks_per_bar=0)

    def test_init_invalid_ticks_negative(self):
        """Test initialization fails with negative expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            ImbalanceBarSamplerVectorized(expected_ticks_per_bar=-10)

    def test_init_invalid_alpha_zero(self):
        """Test initialization fails with zero alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSamplerVectorized(expected_ticks_per_bar=100, alpha=0)

    def test_init_invalid_alpha_greater_than_one(self):
        """Test initialization fails with alpha > 1."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSamplerVectorized(expected_ticks_per_bar=100, alpha=1.5)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=50)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Imbalance bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, large_tick_data):
        """Test basic sampling with large data."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=100)
        bars = sampler.sample(large_tick_data)

        # Should produce some bars
        assert len(bars) >= 0
        if len(bars) > 0:
            assert "imbalance" in bars.columns
            assert "expected_imbalance" in bars.columns
            assert "cumulative_theta" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=50)
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

    def test_sample_include_incomplete(self, sample_tick_data):
        """Test sampling with incomplete bar."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=50)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_initial_expectation_estimation(self, sample_tick_data):
        """Test initial expectation is estimated when not provided."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=50, initial_expectation=None)

        sampler.sample(sample_tick_data)
        # Should estimate initial_expectation from data
        assert sampler.initial_expectation is not None
        assert sampler.initial_expectation > 0

    def test_initial_expectation_provided(self, sample_tick_data):
        """Test sampling with provided initial expectation."""
        sampler = ImbalanceBarSamplerVectorized(
            expected_ticks_per_bar=50, initial_expectation=500.0
        )

        sampler.sample(sample_tick_data)
        # Should use provided initial_expectation
        assert sampler.initial_expectation == 500.0

    def test_empty_result_schema(self):
        """Test _empty_imbalance_bars_df returns correct schema."""
        sampler = ImbalanceBarSamplerVectorized(expected_ticks_per_bar=50)
        empty_df = sampler._empty_imbalance_bars_df()

        expected_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "buy_volume",
            "sell_volume",
            "imbalance",
            "cumulative_theta",
            "expected_imbalance",
        ]
        assert list(empty_df.columns) == expected_cols

    def test_create_incomplete_imbalance_bar(self, sample_tick_data):
        """Test _create_incomplete_imbalance_bar method."""
        sampler = ImbalanceBarSamplerVectorized(
            expected_ticks_per_bar=50, initial_expectation=1000.0
        )

        incomplete_bar = sampler._create_incomplete_imbalance_bar(sample_tick_data)

        assert len(incomplete_bar) == 1
        assert "imbalance" in incomplete_bar.columns
        assert "expected_imbalance" in incomplete_bar.columns
        assert incomplete_bar["expected_imbalance"][0] == 1000.0


class TestVectorizedBarsIntegration:
    """Integration tests for vectorized bars."""

    def test_volume_bar_volume_consistency(self, large_tick_data):
        """Test that total volume is preserved."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=5000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        input_volume = large_tick_data["volume"].sum()
        output_volume = bars["volume"].sum()

        assert abs(input_volume - output_volume) < 0.01

    def test_dollar_bar_dollar_consistency(self, large_tick_data):
        """Test that total dollar volume is preserved."""
        sampler = DollarBarSamplerVectorized(dollars_per_bar=500000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        input_dollars = (large_tick_data["price"] * large_tick_data["volume"]).sum()
        output_dollars = bars["dollar_volume"].sum()

        assert abs(input_dollars - output_dollars) < 0.01

    def test_tick_bar_tick_consistency(self, sample_tick_data):
        """Test that total tick count is preserved."""
        sampler = TickBarSamplerVectorized(ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data, include_incomplete=True)

        input_ticks = len(sample_tick_data)
        output_ticks = bars["tick_count"].sum()

        assert input_ticks == output_ticks

    def test_buy_sell_volume_sum(self, large_tick_data):
        """Test buy + sell volume equals total volume for volume bars."""
        sampler = VolumeBarSamplerVectorized(volume_per_bar=5000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        for i in range(len(bars)):
            total_vol = bars["volume"][i]
            buy_vol = bars["buy_volume"][i]
            sell_vol = bars["sell_volume"][i]

            assert abs(total_vol - (buy_vol + sell_vol)) < 0.01
