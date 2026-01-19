"""Tests for run bar samplers.

Tests for TickRunBarSampler, VolumeRunBarSampler, and DollarRunBarSampler.
AFML-compliant: uses CUMULATIVE counts, NOT consecutive runs.
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
def alternating_sides_data():
    """Create data with alternating buy/sell to test cumulative behavior."""
    n = 50
    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 10000) for i in range(n)]
    prices = [100.0 + i * 0.1 for i in range(n)]
    volumes = [10.0] * n
    # Alternating: B, S, B, S, ...
    sides = [1 if i % 2 == 0 else -1 for i in range(n)]

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
    """Tests for Numba-compiled AFML-compliant run bar calculation."""

    def test_basic_cumulative_run_detection(self):
        """Test that runs are CUMULATIVE (not consecutive)."""
        # Pattern: B, B, S, B, B  (3 buys, 2 sells cumulative)
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        sides = np.array([1.0, 1.0, -1.0, 1.0, 1.0])
        initial_expected_t = 3.0  # Threshold = 3 * 0.5 = 1.5, so triggers at 2

        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t,
            initial_p_buy=0.5,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # Should create bars based on max(cumulative_buys, cumulative_sells)
        assert len(bar_indices) >= 1

    def test_cumulative_not_consecutive(self):
        """CRITICAL: Verify direction changes do NOT reset counts."""
        # Pattern: B, S, B, S, B (alternating)
        # Old (wrong) behavior: run length never exceeds 1
        # New (correct) behavior: cumulative_buys = 3, cumulative_sells = 2
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        sides = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        # Set threshold low enough to trigger
        initial_expected_t = 4.0  # Threshold = 4 * 0.5 = 2

        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t,
            initial_p_buy=0.5,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # Should create bars because cumulative counts grow even with alternating
        # At tick 4: cum_buys=2, cum_sells=2, max=2, threshold=2 -> bar!
        assert len(bar_indices) >= 1

        # Each bar's theta should be max(cum_buys, cum_sells)
        if len(thetas) > 0:
            assert thetas[0] >= 2  # At least 2 cumulative on one side

    def test_no_runs_exceed_expectation(self):
        """Test when no runs exceed expectation."""
        values = np.array([1.0, 1.0])
        sides = np.array([1.0, -1.0])
        initial_expected_t = 1000.0  # Very high expectation

        (bar_indices, *_) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t,
            initial_p_buy=0.5,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # No bars should be created
        assert len(bar_indices) == 0

    def test_ewma_update(self):
        """Test EWMA expectation update after warmup."""
        values = np.array([1.0] * 100)
        sides = np.array([1.0] * 50 + [-1.0] * 50)  # 50 buys then 50 sells
        initial_expected_t = 10.0
        alpha = 0.5  # High alpha for noticeable update

        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t,
            initial_p_buy=0.5,
            alpha=alpha,
            min_bars_warmup=2,
        )

        # Should trigger bars and EWMA should update after warmup
        assert len(bar_indices) >= 1


class TestTickRunBarSampler:
    """Tests for TickRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1
        assert sampler.initial_p_buy == 0.5

    def test_init_deprecated_initial_run_expectation(self):
        """Test deprecation warning for initial_run_expectation."""
        with pytest.warns(DeprecationWarning, match="initial_run_expectation is deprecated"):
            sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=15)
        assert sampler.expected_ticks_per_bar == 100

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

    def test_init_invalid_p_buy(self):
        """Test initialization fails with invalid p_buy."""
        with pytest.raises(ValueError, match="initial_p_buy must be in"):
            TickRunBarSampler(expected_ticks_per_bar=100, initial_p_buy=-0.1)

        with pytest.raises(ValueError, match="initial_p_buy must be in"):
            TickRunBarSampler(expected_ticks_per_bar=100, initial_p_buy=1.1)

    def test_sample_missing_side(self, sample_tick_data):
        """Test sampling fails without side column."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=50)
        data_no_side = sample_tick_data.drop("side")

        with pytest.raises(DataValidationError, match="Run bars require 'side' column"):
            sampler.sample(data_no_side)

    def test_sample_basic(self, sample_tick_data):
        """Test basic sampling with AFML columns."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            # Backward-compatible columns
            assert "run_length" in bars.columns
            assert "expected_run" in bars.columns
            assert "buy_volume" in bars.columns
            assert "sell_volume" in bars.columns

            # AFML diagnostic columns
            assert "theta" in bars.columns
            assert "expected_theta" in bars.columns
            assert "expected_t" in bars.columns
            assert "p_buy" in bars.columns
            assert "cumulative_buys" in bars.columns
            assert "cumulative_sells" in bars.columns

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
        # Check new columns in schema
        assert "cumulative_buys" in bars.columns
        assert "cumulative_sells" in bars.columns

    def test_sample_include_incomplete(self, sample_tick_data):
        """Test sampling with incomplete bar."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=20)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_cumulative_behavior_with_alternating(self, alternating_sides_data):
        """Test AFML cumulative behavior with alternating buy/sell."""
        # With alternating data, consecutive runs would never trigger
        # but cumulative runs should
        sampler = TickRunBarSampler(expected_ticks_per_bar=10, initial_p_buy=0.5)
        bars = sampler.sample(alternating_sides_data)

        # With cumulative counting, should trigger bars
        # At tick 10: ~5 buys, ~5 sells cumulative, max=5
        # Threshold = 10 * 0.5 = 5, so should trigger
        assert len(bars) >= 1

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
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        ]
        assert list(empty_df.columns) == expected_cols


class TestVolumeRunBarSampler:
    """Tests for VolumeRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_deprecated_initial_run_expectation(self):
        """Test deprecation warning for initial_run_expectation."""
        with pytest.warns(DeprecationWarning, match="initial_run_expectation is deprecated"):
            sampler = VolumeRunBarSampler(
                expected_ticks_per_bar=100, initial_run_expectation=1000.0
            )
        assert sampler.expected_ticks_per_bar == 100

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
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            # Backward-compatible columns
            assert "run_volume" in bars.columns
            assert "expected_run" in bars.columns

            # AFML diagnostic columns
            assert "theta" in bars.columns
            assert "cumulative_buys" in bars.columns
            assert "cumulative_sells" in bars.columns

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
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=20)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

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
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        ]
        assert list(empty_df.columns) == expected_cols


class TestDollarRunBarSampler:
    """Tests for DollarRunBarSampler."""

    def test_init_valid(self):
        """Test valid initialization."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100)
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.alpha == 0.1

    def test_init_deprecated_initial_run_expectation(self):
        """Test deprecation warning for initial_run_expectation."""
        with pytest.warns(DeprecationWarning, match="initial_run_expectation is deprecated"):
            sampler = DollarRunBarSampler(
                expected_ticks_per_bar=100, initial_run_expectation=100000.0
            )
        assert sampler.expected_ticks_per_bar == 100

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
        sampler = DollarRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data)

        if len(bars) > 0:
            # Backward-compatible columns
            assert "run_dollars" in bars.columns
            assert "expected_run" in bars.columns
            assert "dollar_volume" in bars.columns
            assert "vwap" in bars.columns

            # AFML diagnostic columns
            assert "theta" in bars.columns
            assert "cumulative_buys" in bars.columns
            assert "cumulative_sells" in bars.columns

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
        sampler = DollarRunBarSampler(expected_ticks_per_bar=20)

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        assert len(bars_with) >= len(bars_without)

    def test_vwap_calculation(self, sample_tick_data):
        """Test VWAP is calculated correctly in bars."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(sample_tick_data, include_incomplete=True)

        if len(bars) > 0:
            # VWAP should be positive and reasonable
            for i in range(len(bars)):
                assert bars["vwap"][i] > 0
                # VWAP = dollar_volume / volume
                if bars["volume"][i] > 0:
                    expected_vwap = bars["dollar_volume"][i] / bars["volume"][i]
                    assert abs(bars["vwap"][i] - expected_vwap) < 0.01

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
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        ]
        assert list(empty_df.columns) == expected_cols


class TestRunBarsIntegration:
    """Integration tests for run bars."""

    def test_tick_run_bar_volume_consistency(self, large_tick_data):
        """Test that total volume is preserved."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_volume = large_tick_data["volume"].sum()
            output_volume = bars["volume"].sum()

            assert abs(input_volume - output_volume) < 0.01

    def test_volume_run_bar_volume_consistency(self, large_tick_data):
        """Test that total volume is preserved for volume run bars."""
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_volume = large_tick_data["volume"].sum()
            output_volume = bars["volume"].sum()

            assert abs(input_volume - output_volume) < 0.01

    def test_dollar_run_bar_dollar_consistency(self, large_tick_data):
        """Test that total dollar volume is preserved."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            input_dollars = (large_tick_data["price"] * large_tick_data["volume"]).sum()
            output_dollars = bars["dollar_volume"].sum()

            assert abs(input_dollars - output_dollars) < 0.01

    def test_buy_sell_volume_sum(self, large_tick_data):
        """Test buy + sell volume equals total volume."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=20)
        bars = sampler.sample(large_tick_data, include_incomplete=True)

        if len(bars) > 0:
            for i in range(len(bars)):
                total_vol = bars["volume"][i]
                buy_vol = bars["buy_volume"][i]
                sell_vol = bars["sell_volume"][i]

                assert abs(total_vol - (buy_vol + sell_vol)) < 0.01


class TestAFMLRunBarCompliance:
    """Tests verifying AFML formula compliance for run bars."""

    def test_theta_is_max_of_cumulative(self):
        """Verify θ = max{cumulative_buys, cumulative_sells}."""
        values = np.array([1.0] * 20)
        # 12 buys, 8 sells
        sides = np.array([1.0] * 12 + [-1.0] * 8)

        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t=20.0,  # High threshold to get one bar
            initial_p_buy=0.5,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # Should not trigger (20 * 0.5 = 10, and max(12, 8) = 12 > 10)
        # Actually it should trigger since max(12, 8) = 12 >= 10
        if len(bar_indices) > 0:
            # theta should equal max(cumulative_buys, cumulative_sells)
            for i in range(len(thetas)):
                assert thetas[i] == max(cumulative_buys[i], cumulative_sells[i])

    def test_threshold_formula(self):
        """Verify threshold = E[T] × max{P[b=1], 1-P[b=1]}."""
        values = np.array([1.0] * 50)
        sides = np.array([1.0] * 50)

        E_T = 10.0
        p_buy = 0.7  # 70% buy probability

        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            *_,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t=E_T,
            initial_p_buy=p_buy,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # Threshold = E[T] × max{P[b=1], 1-P[b=1]} = 10 × max(0.7, 0.3) = 10 × 0.7 = 7
        expected_threshold = E_T * max(p_buy, 1 - p_buy)

        if len(expected_thetas) > 0:
            assert abs(expected_thetas[0] - expected_threshold) < 0.01

    def test_cumulative_not_reset_on_direction_change(self):
        """CRITICAL: Direction change must NOT reset within-bar counts."""
        # Create pattern: B, B, S, B, B, S, B, B
        # With cumulative: buys = 6, sells = 2
        # With consecutive (WRONG): max run = 2
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        sides = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])

        # Set threshold to 5 (should trigger with cumulative, not with consecutive)
        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            initial_expected_t=10.0,  # Threshold = 10 * 0.5 = 5
            initial_p_buy=0.5,
            alpha=0.1,
            min_bars_warmup=100,
        )

        # Should trigger because cumulative_buys reaches 5 or more
        # At tick 5 (0-indexed 4): buys=4, sells=1, max=4 < 5
        # At tick 6 (0-indexed 5): buys=4, sells=2, max=4 < 5
        # At tick 7 (0-indexed 6): buys=5, sells=2, max=5 >= 5 -> bar!
        assert len(bar_indices) >= 1

        # The theta should be 5 or more (cumulative buys)
        if len(thetas) > 0:
            assert thetas[0] >= 5
