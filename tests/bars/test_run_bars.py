"""Tests for run bar samplers.

Tests for TickRunBarSampler, VolumeRunBarSampler, and DollarRunBarSampler.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.run import (
    DollarRunBarSampler,
    TickRunBarSampler,
    VolumeRunBarSampler,
    _calculate_run_bars_nb,
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
def consecutive_buys_data():
    """Create data with consecutive buy runs."""
    n = 50
    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 10000) for i in range(n)]
    prices = [100.0 + i * 0.1 for i in range(n)]
    volumes = [10.0] * n
    # Create alternating runs: 10 buys, 10 sells, 10 buys, 10 sells, 10 buys
    sides = [1] * 10 + [-1] * 10 + [1] * 10 + [-1] * 10 + [1] * 10

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


class TestCalculateRunBarsNb:
    """Tests for Numba-compiled run bar calculation."""

    def test_basic_run_detection(self):
        """Test basic run detection."""
        volumes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        sides = np.array([1.0, 1.0, 1.0, -1.0, 1.0])
        initial_expectation = 3.0  # Expect runs of 3

        bar_indices, expected_runs, run_lengths = _calculate_run_bars_nb(
            volumes, sides, initial_expectation, alpha=0.1
        )

        # First 3 buys should form a run of 3, triggering a bar
        assert len(bar_indices) > 0
        assert bar_indices[0] == 2  # Index of 3rd buy (0-indexed)

    def test_no_runs_exceed_expectation(self):
        """Test when no runs exceed expectation."""
        volumes = np.array([10.0, 10.0, 10.0, 10.0])
        sides = np.array([1.0, -1.0, 1.0, -1.0])  # Alternating, run length = 1
        initial_expectation = 10.0  # Very high expectation

        bar_indices, expected_runs, run_lengths = _calculate_run_bars_nb(
            volumes, sides, initial_expectation, alpha=0.1
        )

        # No bars should be created
        assert len(bar_indices) == 0

    def test_ewma_update(self):
        """Test EWMA expectation update."""
        volumes = np.array([10.0] * 20)
        sides = np.array([1.0] * 10 + [-1.0] * 10)  # 10 buys then 10 sells
        initial_expectation = 5.0
        alpha = 0.5  # High alpha for more noticeable update

        bar_indices, expected_runs, run_lengths = _calculate_run_bars_nb(
            volumes, sides, initial_expectation, alpha=alpha
        )

        # Should trigger bars when runs exceed expectation
        # First bar should trigger after run of 5 or more
        assert len(bar_indices) >= 1
        # Run lengths should be at least initial_expectation
        assert run_lengths[0] >= initial_expectation


class TestTickRunBarSampler:
    """Tests for TickRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_with_initial_expectation(self):
        """Test initialization with custom initial expectation."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=15)
        assert sampler.initial_run_expectation == 15

    def test_init_invalid_ticks_zero(self):
        """Test initialization fails with zero expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            TickRunBarSampler(expected_ticks_per_bar=0)

    def test_init_invalid_ticks_negative(self):
        """Test initialization fails with negative expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            TickRunBarSampler(expected_ticks_per_bar=-10)

    def test_init_invalid_alpha_zero(self):
        """Test initialization fails with zero alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            TickRunBarSampler(expected_ticks_per_bar=100, alpha=0)

    def test_init_invalid_alpha_greater_than_one(self):
        """Test initialization fails with alpha > 1."""
        with pytest.raises(ValueError, match="alpha must be in"):
            TickRunBarSampler(expected_ticks_per_bar=100, alpha=1.5)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Run bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            assert "run_length" in bars.columns
            assert "expected_run" in bars.columns
            assert "buy_volume" in bars.columns
            assert "sell_volume" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)
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
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_sample_consecutive_runs(self, consecutive_buys_data):
        """Test sampling with consecutive runs."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50, initial_run_expectation=5)

        bars = sampler.sample(consecutive_buys_data)

        # Should detect runs of 10
        assert len(bars) > 0
        # Each detected run should have run_length >= expected
        for i in range(len(bars)):
            assert bars["run_length"][i] >= sampler.initial_run_expectation

    def test_initial_expectation_estimation(self, sample_tick_data):
        """Test initial expectation is estimated when not provided."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)

        sampler.sample(sample_tick_data)
        # Should estimate initial_run_expectation from expected_ticks_per_bar
        # Default: ~15% of expected_ticks_per_bar
        assert sampler.initial_run_expectation is not None

    def test_empty_result_schema(self):
        """Test _empty_run_bars_df returns correct schema."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)
        empty_df = sampler._empty_run_bars_df()

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
            "run_length",
            "expected_run",
        ]
        assert list(empty_df.columns) == expected_cols


class TestVolumeRunBarSampler:
    """Tests for VolumeRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_with_initial_expectation(self):
        """Test initialization with custom initial expectation."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=1000.0)
        assert sampler.initial_run_expectation == 1000.0

    def test_init_invalid_ticks_zero(self):
        """Test initialization fails with zero expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            VolumeRunBarSampler(expected_ticks_per_bar=0)

    def test_init_invalid_alpha_zero(self):
        """Test initialization fails with zero alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            VolumeRunBarSampler(expected_ticks_per_bar=100, alpha=0)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=50)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Run bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=200.0)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            assert "run_volume" in bars.columns
            assert "expected_run" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=50)
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
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=200.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_initial_expectation_estimation(self, sample_tick_data):
        """Test initial expectation is estimated when not provided."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=50)

        sampler.sample(sample_tick_data)
        # Should estimate from avg_volume * expected_ticks * 0.15
        assert sampler.initial_run_expectation is not None

    def test_calculate_volume_runs(self):
        """Test _calculate_volume_runs method."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=50)

        # Create simple test case
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        sides = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
        initial_expectation = 250.0

        bar_indices, expected_runs, run_volumes = sampler._calculate_volume_runs(
            volumes, sides, initial_expectation, alpha=0.1
        )

        # First 3 buys = 300 volume, exceeds 250, should trigger bar
        assert len(bar_indices) > 0
        assert run_volumes[0] >= initial_expectation

    def test_empty_result_schema(self):
        """Test _empty_run_bars_df returns correct schema."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=50)
        empty_df = sampler._empty_run_bars_df()

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
            "run_volume",
            "expected_run",
        ]
        assert list(empty_df.columns) == expected_cols


class TestDollarRunBarSampler:
    """Tests for DollarRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_with_initial_expectation(self):
        """Test initialization with custom initial expectation."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=100000.0)
        assert sampler.initial_run_expectation == 100000.0

    def test_init_invalid_ticks_zero(self):
        """Test initialization fails with zero expected ticks."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            DollarRunBarSampler(expected_ticks_per_bar=0)

    def test_init_invalid_alpha_zero(self):
        """Test initialization fails with zero alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            DollarRunBarSampler(expected_ticks_per_bar=100, alpha=0)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=50)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Run bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=20000.0)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            assert "run_dollars" in bars.columns
            assert "expected_run" in bars.columns
            assert "dollar_volume" in bars.columns
            assert "vwap" in bars.columns

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=50)
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
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=20000.0)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_initial_expectation_estimation(self, sample_tick_data):
        """Test initial expectation is estimated when not provided."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=50)

        sampler.sample(sample_tick_data)
        # Should estimate from avg_dollar_volume * expected_ticks * 0.15
        assert sampler.initial_run_expectation is not None

    def test_vwap_calculation(self, sample_tick_data):
        """Test VWAP is calculated correctly in bars."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=20000.0)
        bars = sampler.sample(sample_tick_data, include_incomplete=True)

        if len(bars) > 0:
            # VWAP should be positive and reasonable
            for i in range(len(bars)):
                assert bars["vwap"][i] > 0
                # VWAP = dollar_volume / volume
                if bars["volume"][i] > 0:
                    expected_vwap = bars["dollar_volume"][i] / bars["volume"][i]
                    assert abs(bars["vwap"][i] - expected_vwap) < 0.01

    def test_calculate_dollar_runs(self):
        """Test _calculate_dollar_runs method."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=50)

        # Create simple test case
        dollar_volumes = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        sides = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
        initial_expectation = 2500.0

        bar_indices, expected_runs, run_dollars = sampler._calculate_dollar_runs(
            dollar_volumes, sides, initial_expectation, alpha=0.1
        )

        # First 3 buys = 3000 dollars, exceeds 2500, should trigger bar
        assert len(bar_indices) > 0
        assert run_dollars[0] >= initial_expectation

    def test_empty_result_schema(self):
        """Test _empty_run_bars_df returns correct schema."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=50)
        empty_df = sampler._empty_run_bars_df()

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
            "dollar_volume",
            "vwap",
            "run_dollars",
            "expected_run",
        ]
        assert list(empty_df.columns) == expected_cols


class TestRunBarsIntegration:
    """Integration tests for run bars."""

    def test_tick_run_bar_volume_consistency(self, large_tick_data):
        """Test that total volume is preserved."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_volume = large_tick_data["volume"].sum()
            output_volume = bars["volume"].sum()

            assert abs(input_volume - output_volume) < 0.01

    def test_volume_run_bar_volume_consistency(self, large_tick_data):
        """Test that total volume is preserved for volume run bars."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=500.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_volume = large_tick_data["volume"].sum()
            output_volume = bars["volume"].sum()

            assert abs(input_volume - output_volume) < 0.01

    def test_dollar_run_bar_dollar_consistency(self, large_tick_data):
        """Test that total dollar volume is preserved."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=50000.0)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_dollars = (large_tick_data["price"] * large_tick_data["volume"]).sum()
            output_dollars = bars["dollar_volume"].sum()

            assert abs(input_dollars - output_dollars) < 0.01

    def test_buy_sell_volume_sum(self, large_tick_data):
        """Test buy + sell volume equals total volume."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            for i in range(len(bars)):
                total_vol = bars["volume"][i]
                buy_vol = bars["buy_volume"][i]
                sell_vol = bars["sell_volume"][i]

                assert abs(total_vol - (buy_vol + sell_vol)) < 0.01
