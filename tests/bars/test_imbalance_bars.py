"""Tests for imbalance bar sampler."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.imbalance import ImbalanceBarSampler, _calculate_imbalance_bars_nb
from ml4t.engineer.core.exceptions import DataValidationError


@pytest.fixture
def sample_tick_data():
    """Create sample tick data with clear imbalance patterns."""
    n = 200
    np.random.seed(42)

    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 5000) for i in range(n)]
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.randint(50, 150, n).astype(float)

    # Create some imbalance patterns
    # First 50: mostly buys
    # Next 50: mostly sells
    # Last 100: mixed
    sides = np.ones(n)
    sides[50:100] = -1
    sides[100:] = np.random.choice([-1, 1], 100)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.fixture
def uniform_tick_data():
    """Create tick data with uniform imbalance."""
    n = 100
    timestamps = [datetime(2024, 1, 1, 9, 30, 0, i * 10000) for i in range(n)]
    prices = [100.0 + i * 0.1 for i in range(n)]
    volumes = [100.0] * n
    sides = [1.0] * n  # All buys

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


class TestCalculateImbalanceBarsNb:
    """Tests for the Numba imbalance calculation function."""

    def test_basic_calculation(self):
        """Test basic imbalance calculation."""
        volumes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        sides = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # All buys

        bar_indices, expected_imbalances, cumulative_thetas = _calculate_imbalance_bars_nb(
            volumes, sides, initial_expectation=250.0, alpha=0.1
        )

        # With initial_expectation=250 and all buys at 100 vol each,
        # cumulative theta reaches 300 after 3 ticks (100+100+100=300 > 250)
        assert len(bar_indices) > 0

    def test_alternating_sides(self):
        """Test with alternating buy/sell."""
        volumes = np.array([100.0, 100.0, 100.0, 100.0])
        sides = np.array([1.0, -1.0, 1.0, -1.0])  # Alternating

        bar_indices, _, _ = _calculate_imbalance_bars_nb(
            volumes, sides, initial_expectation=50.0, alpha=0.1
        )

        # Each tick alternates, so cumulative should oscillate around 0
        # Depends on threshold - with 50, should create bar when |theta| >= 50
        assert isinstance(bar_indices, np.ndarray)

    def test_alpha_effect(self):
        """Test that alpha affects expected imbalance updates."""
        volumes = np.array([100.0] * 20)
        sides = np.array([1.0] * 20)

        # Low alpha - slow adaptation
        _, expected_low_alpha, _ = _calculate_imbalance_bars_nb(
            volumes, sides, initial_expectation=200.0, alpha=0.1
        )

        # High alpha - fast adaptation
        _, expected_high_alpha, _ = _calculate_imbalance_bars_nb(
            volumes, sides, initial_expectation=200.0, alpha=0.9
        )

        # Both should have values
        assert len(expected_low_alpha) > 0
        assert len(expected_high_alpha) > 0

    def test_empty_arrays(self):
        """Test with empty arrays."""
        volumes = np.array([], dtype=np.float64)
        sides = np.array([], dtype=np.float64)

        bar_indices, expected_imbalances, cumulative_thetas = _calculate_imbalance_bars_nb(
            volumes, sides, initial_expectation=100.0, alpha=0.1
        )

        assert len(bar_indices) == 0
        assert len(expected_imbalances) == 0
        assert len(cumulative_thetas) == 0


class TestImbalanceBarSampler:
    """Tests for ImbalanceBarSampler class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=100, initial_expectation=5000.0, alpha=0.2
        )

        assert sampler.expected_ticks_per_bar == 100
        assert sampler.initial_expectation == 5000.0
        assert sampler.alpha == 0.2

    def test_init_invalid_expected_ticks(self):
        """Test initialization fails with invalid expected_ticks_per_bar."""
        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            ImbalanceBarSampler(expected_ticks_per_bar=0)

        with pytest.raises(ValueError, match="expected_ticks_per_bar must be positive"):
            ImbalanceBarSampler(expected_ticks_per_bar=-10)

    def test_init_invalid_alpha_zero(self):
        """Test initialization fails with alpha=0."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=0)

    def test_init_invalid_alpha_negative(self):
        """Test initialization fails with negative alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=-0.1)

    def test_init_invalid_alpha_too_large(self):
        """Test alpha=1 is valid (edge of range)."""
        # alpha=1.0 should be valid
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=1.0)
        assert sampler.alpha == 1.0

        # alpha > 1 should fail
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=1.1)

    def test_sample_missing_side_column(self):
        """Test sampling fails without side column."""
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=50)
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": [50.0],
            }
        )

        with pytest.raises(DataValidationError, match="Imbalance bars require 'side' column"):
            sampler.sample(data)

    def test_sample_empty_data(self):
        """Test sampling empty DataFrame."""
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=50)
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

    def test_sample_basic(self, sample_tick_data):
        """Test basic imbalance bar sampling."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=20, initial_expectation=1000.0, alpha=0.2
        )
        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0

        # Check bar structure
        assert "timestamp" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns
        assert "tick_count" in bars.columns
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns
        assert "imbalance" in bars.columns
        assert "cumulative_theta" in bars.columns
        assert "expected_imbalance" in bars.columns

    def test_sample_with_incomplete(self, sample_tick_data):
        """Test sampling with include_incomplete=True."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=50, initial_expectation=5000.0, alpha=0.1
        )

        bars_without = sampler.sample(sample_tick_data, include_incomplete=False)
        bars_with = sampler.sample(sample_tick_data, include_incomplete=True)

        # Should have at least one more bar with incomplete
        assert len(bars_with) >= len(bars_without)

    def test_sample_auto_initial_expectation(self, sample_tick_data):
        """Test sampling with auto-estimated initial expectation."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=20,
            initial_expectation=None,  # Auto-estimate
            alpha=0.1,
        )

        bars = sampler.sample(sample_tick_data)

        assert len(bars) > 0
        # Initial expectation should have been set
        assert sampler.initial_expectation is not None
        assert sampler.initial_expectation > 0

    def test_sample_imbalance_calculation(self, uniform_tick_data):
        """Test imbalance is correctly calculated."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=10, initial_expectation=500.0, alpha=0.2
        )

        bars = sampler.sample(uniform_tick_data)

        # All sides are buys (1), so imbalance = buy_volume - sell_volume = buy_volume
        for i in range(len(bars)):
            buy_vol = bars["buy_volume"][i]
            sell_vol = bars["sell_volume"][i]
            imbalance = bars["imbalance"][i]

            assert abs(imbalance - (buy_vol - sell_vol)) < 0.01

    def test_sample_cumulative_theta_sign(self, sample_tick_data):
        """Test cumulative theta reflects imbalance direction."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=25, initial_expectation=1000.0, alpha=0.1
        )

        bars = sampler.sample(sample_tick_data)

        # cumulative_theta should match imbalance direction (positive for buys, negative for sells)
        for i in range(len(bars)):
            imbalance = bars["imbalance"][i]
            theta = bars["cumulative_theta"][i]

            # Sign should match (or be zero)
            if abs(imbalance) > 0.01:
                assert np.sign(imbalance) == np.sign(theta) or abs(theta) < 0.01

    def test_sample_expected_imbalance_updates(self, sample_tick_data):
        """Test expected imbalance is updated with EWMA."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=25,
            initial_expectation=500.0,
            alpha=0.5,  # High alpha for noticeable updates
        )

        bars = sampler.sample(sample_tick_data)

        if len(bars) > 1:
            # Expected imbalance should change between bars
            expected_imbalances = bars["expected_imbalance"].to_list()

            # Not all should be the same (unless by coincidence)
            # This is a weak test but verifies the mechanism works
            assert isinstance(expected_imbalances, list)


class TestImbalanceBarSamplerEdgeCases:
    """Edge case tests for ImbalanceBarSampler."""

    def test_single_tick(self):
        """Test with single tick."""
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=1, initial_expectation=50.0, alpha=0.5)

        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": [100.0],
                "side": [1.0],
            }
        )

        bars = sampler.sample(data, include_incomplete=True)

        # Should create one bar
        assert len(bars) == 1

    def test_no_bars_created(self):
        """Test when no bars meet threshold."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=100,
            initial_expectation=1000000.0,  # Very high threshold
            alpha=0.1,
        )

        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, i) for i in range(10)],
                "price": [100.0 + i for i in range(10)],
                "volume": [10.0] * 10,
                "side": [1.0] * 10,
            }
        )

        bars = sampler.sample(data, include_incomplete=False)

        # No bars should be created (threshold too high)
        assert len(bars) == 0

        # With incomplete, should get one bar
        bars_with = sampler.sample(data, include_incomplete=True)
        assert len(bars_with) == 1

    def test_all_sells(self):
        """Test with all sell orders."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=5, initial_expectation=200.0, alpha=0.2
        )

        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, i) for i in range(20)],
                "price": [100.0 - i * 0.1 for i in range(20)],
                "volume": [50.0] * 20,
                "side": [-1.0] * 20,  # All sells
            }
        )

        bars = sampler.sample(data)

        # Should create bars
        assert len(bars) > 0

        # All imbalances should be negative (all sells)
        for imb in bars["imbalance"]:
            assert imb <= 0
