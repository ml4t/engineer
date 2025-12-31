"""Comprehensive tests for bar sampling methods.

Tests all bar types for correctness, edge cases, and invariants.
Created: 2025-11-04
Purpose: Increase coverage of bars module and ensure correctness
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars import (
    DollarBarSampler,
    DollarRunBarSampler,
    ImbalanceBarSampler,
    TickBarSampler,
    TickRunBarSampler,
    VolumeBarSampler,
    VolumeRunBarSampler,
)


@pytest.fixture
def tick_data():
    """Generate realistic tick data for testing."""
    n = 10000
    base_time = datetime(2024, 1, 1)
    np.random.seed(42)

    # Simulate realistic price movement
    base_price = 100.0
    returns = np.random.normal(0, 0.0001, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Realistic volume distribution
    volumes = np.random.lognormal(mean=4, sigma=1, size=n).astype(int)

    # Buy/sell imbalance
    sides = np.random.choice([1, -1], n, p=[0.52, 0.48])

    return pl.DataFrame(
        {
            "timestamp": [base_time + timedelta(seconds=i * 0.1) for i in range(n)],
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


class TestTickBarSampler:
    """Test tick bar sampling."""

    def test_basic_tick_bars(self, tick_data):
        """Test that tick bars aggregate correctly."""
        sampler = TickBarSampler(ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        # Should have exactly n/100 bars
        assert len(bars) == len(tick_data) // 100

        # Each bar should have correct tick count
        assert (bars["tick_count"] == 100).all()

    def test_tick_bar_ohlc_invariants(self, tick_data):
        """Test OHLC relationships hold."""
        sampler = TickBarSampler(ticks_per_bar=50)
        bars = sampler.sample(tick_data)

        # High >= max(Open, Close)
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()

        # Low <= min(Open, Close)
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()

        # High >= Low (always)
        assert (bars["high"] >= bars["low"]).all()

    def test_tick_bar_volume_sum(self, tick_data):
        """Test that volume sums correctly."""
        sampler = TickBarSampler(ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        # Total volume should match (within floating point)
        expected_bars = len(tick_data) // 100
        expected_volume = tick_data["volume"][: expected_bars * 100].sum()
        actual_volume = bars["volume"].sum()

        assert actual_volume == expected_volume

    def test_variable_tick_sizes(self, tick_data):
        """Test different tick sizes."""
        for ticks in [10, 50, 100, 500]:
            sampler = TickBarSampler(ticks_per_bar=ticks)
            bars = sampler.sample(tick_data)

            assert len(bars) == len(tick_data) // ticks
            assert (bars["tick_count"] == ticks).all()


class TestVolumeBarSampler:
    """Test volume bar sampling."""

    def test_basic_volume_bars(self, tick_data):
        """Test that volume bars aggregate correctly."""
        volume_per_bar = 10000
        sampler = VolumeBarSampler(volume_per_bar=volume_per_bar)
        bars = sampler.sample(tick_data)

        # Each bar should have volume >= threshold
        assert (bars["volume"] >= volume_per_bar).all()

    def test_volume_bar_count(self, tick_data):
        """Test expected number of bars."""
        total_volume = tick_data["volume"].sum()
        volume_per_bar = 5000

        sampler = VolumeBarSampler(volume_per_bar=volume_per_bar)
        bars = sampler.sample(tick_data)

        # Should have roughly total_volume / volume_per_bar bars
        expected_bars = total_volume // volume_per_bar
        # Allow ±5% tolerance since volume per bar varies slightly
        assert abs(len(bars) - expected_bars) <= expected_bars * 0.05

    def test_volume_bars_ohlc_valid(self, tick_data):
        """Test OHLC relationships."""
        sampler = VolumeBarSampler(volume_per_bar=8000)
        bars = sampler.sample(tick_data)

        assert (bars["high"] >= bars["low"]).all()
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()


class TestDollarBarSampler:
    """Test dollar bar sampling."""

    def test_basic_dollar_bars(self, tick_data):
        """Test that dollar bars aggregate correctly."""
        dollars_per_bar = 1_000_000
        sampler = DollarBarSampler(dollars_per_bar=dollars_per_bar)
        bars = sampler.sample(tick_data)

        # Each bar should have dollar volume close to threshold
        dollar_volumes = bars["close"] * bars["volume"]
        # Allow some variance since we use close price
        assert (dollar_volumes >= dollars_per_bar * 0.9).all()

    def test_dollar_bar_consistency(self, tick_data):
        """Test consistency across runs."""
        sampler = DollarBarSampler(dollars_per_bar=500_000)
        bars1 = sampler.sample(tick_data)
        bars2 = sampler.sample(tick_data)

        # Should produce identical results
        assert bars1.equals(bars2)


class TestRunBarSamplers:
    """Test run bar samplers (detect directional runs)."""

    def test_tick_run_bars(self, tick_data):
        """Test tick run bars detect price runs."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)
        bars = sampler.sample(tick_data)

        # Should have proper structure
        assert "tick_count" in bars.columns
        assert len(bars) > 0

    def test_volume_run_bars(self, tick_data):
        """Test volume run bars."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        assert "volume" in bars.columns
        assert len(bars) > 0

    def test_dollar_run_bars(self, tick_data):
        """Test dollar run bars."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        assert "close" in bars.columns
        assert len(bars) > 0


class TestImbalanceBarSampler:
    """Test imbalance bar sampler."""

    def test_imbalance_bars(self, tick_data):
        """Test that imbalance bars sample on order flow imbalance."""
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        # Should have buy/sell volume split
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns

        # Buy + sell = total
        assert np.allclose(bars["buy_volume"] + bars["sell_volume"], bars["volume"], rtol=0.01)


class TestBarSamplerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty dataframes."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
                "side": [],
            }
        )

        sampler = TickBarSampler(ticks_per_bar=10)
        bars = sampler.sample(empty_df)
        assert len(bars) == 0

    def test_insufficient_data(self, tick_data):
        """Test when data has fewer ticks than threshold."""
        small_data = tick_data[:50]
        sampler = TickBarSampler(ticks_per_bar=100)
        bars = sampler.sample(small_data)

        # Should produce no bars (incomplete bar not included by default)
        assert len(bars) == 0

    def test_zero_volume_handling(self):
        """Test handling of zero volume ticks."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(100)],
                "price": [100.0] * 100,
                "volume": [0] * 100,
                "side": [1] * 100,
            }
        )

        sampler = VolumeBarSampler(volume_per_bar=100)
        bars = sampler.sample(data)

        # Should produce no bars (zero volume never reaches threshold)
        assert len(bars) == 0


class TestBarSamplerInvariants:
    """Test mathematical invariants that should always hold."""

    def test_timestamps_monotonic(self, tick_data):
        """Test that bar timestamps are monotonically increasing."""
        sampler = TickBarSampler(ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        timestamps = bars["timestamp"].to_list()
        assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

    def test_no_data_loss(self, tick_data):
        """Test that we don't lose data in sampling."""
        sampler = TickBarSampler(ticks_per_bar=100)
        bars = sampler.sample(tick_data)

        # Number of ticks processed
        ticks_processed = len(bars) * 100

        # Should process all complete bars worth of data
        assert ticks_processed == (len(tick_data) // 100) * 100

    def test_price_bounds(self, tick_data):
        """Test that bar prices stay within tick data bounds."""
        sampler = VolumeBarSampler(volume_per_bar=10000)
        bars = sampler.sample(tick_data)

        tick_min = tick_data["price"].min()
        tick_max = tick_data["price"].max()

        assert bars["low"].min() >= tick_min
        assert bars["high"].max() <= tick_max
