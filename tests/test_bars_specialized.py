"""Tests for specialized bar sampling (tick, volume, dollar, imbalance bars)."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.bars.imbalance import ImbalanceBarSampler
from ml4t.engineer.bars.tick import TickBarSampler
from ml4t.engineer.bars.volume import DollarBarSampler, VolumeBarSampler
from ml4t.engineer.core.exceptions import DataValidationError

# =============================================================================
# Test Data Generators
# =============================================================================


def generate_tick_data(
    n_ticks: int = 1000,
    base_price: float = 100.0,
    volatility: float = 1.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic tick data for testing.

    Parameters
    ----------
    n_ticks : int
        Number of ticks to generate
    base_price : float
        Starting price level
    volatility : float
        Price volatility (std of returns)
    seed : int
        Random seed

    Returns
    -------
    pl.DataFrame
        Tick data with columns: timestamp, price, volume, side
    """
    np.random.seed(seed)

    # Generate timestamps (1 second apart)
    start_time = datetime(2024, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_ticks)]

    # Generate prices (random walk)
    returns = np.random.randn(n_ticks) * volatility * 0.01
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate volumes (log-normal)
    volumes = np.exp(np.random.randn(n_ticks) * 0.5 + 5)

    # Generate sides (buy/sell) with some autocorrelation
    sides = np.zeros(n_ticks)
    sides[0] = np.random.choice([-1, 1])
    for i in range(1, n_ticks):
        # 70% chance to continue same direction (creates runs)
        if np.random.rand() < 0.7:
            sides[i] = sides[i - 1]
        else:
            sides[i] = -sides[i - 1]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


# =============================================================================
# Tick Bar Tests
# =============================================================================


class TestTickBarSampler:
    """Tests for TickBarSampler."""

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        # Valid
        sampler = TickBarSampler(ticks_per_bar=100)
        assert sampler.ticks_per_bar == 100

        # Invalid (non-positive)
        with pytest.raises(ValueError, match="must be positive"):
            TickBarSampler(ticks_per_bar=0)

        with pytest.raises(ValueError, match="must be positive"):
            TickBarSampler(ticks_per_bar=-10)

    def test_empty_data(self) -> None:
        """Test sampling with empty data."""
        sampler = TickBarSampler(ticks_per_bar=100)
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
            }
        )

        result = sampler.sample(empty_df)
        assert len(result) == 0

    def test_missing_columns(self) -> None:
        """Test that missing required columns raise error."""
        sampler = TickBarSampler(ticks_per_bar=100)

        # Missing 'price'
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "volume": [100.0],
            }
        )

        with pytest.raises(DataValidationError, match="Missing required columns"):
            sampler.sample(df)

    def test_basic_sampling(self) -> None:
        """Test basic tick bar sampling."""
        tick_data = generate_tick_data(n_ticks=500, seed=42)
        sampler = TickBarSampler(ticks_per_bar=100)

        bars = sampler.sample(tick_data, include_incomplete=False)

        # Should have 5 complete bars (500 / 100)
        assert len(bars) == 5

        # Check bar structure
        assert "timestamp" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns
        assert "tick_count" in bars.columns

        # Each bar should have 100 ticks
        assert all(bars["tick_count"] == 100)

    def test_include_incomplete_bar(self) -> None:
        """Test including incomplete final bar."""
        tick_data = generate_tick_data(n_ticks=550, seed=42)
        sampler = TickBarSampler(ticks_per_bar=100)

        bars = sampler.sample(tick_data, include_incomplete=True)

        # Should have 6 bars (5 complete + 1 incomplete)
        assert len(bars) == 6

        # Last bar should have 50 ticks
        assert bars["tick_count"][-1] == 50

    def test_ohlcv_correctness(self) -> None:
        """Test that OHLCV values are computed correctly."""
        tick_data = generate_tick_data(n_ticks=100, seed=42)
        sampler = TickBarSampler(ticks_per_bar=100)

        bars = sampler.sample(tick_data)

        # With 100 ticks per bar, should have 1 bar
        assert len(bars) == 1

        bar = bars.row(0, named=True)

        # Check OHLCV relationships
        assert bar["open"] == tick_data["price"][0]
        assert bar["close"] == tick_data["price"][-1]
        assert bar["high"] == tick_data["price"].max()
        assert bar["low"] == tick_data["price"].min()
        assert bar["volume"] == tick_data["volume"].sum()


# =============================================================================
# Volume Bar Tests
# =============================================================================


class TestVolumeBarSampler:
    """Tests for VolumeBarSampler."""

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        # Valid
        sampler = VolumeBarSampler(volume_per_bar=10000)
        assert sampler.volume_per_bar == 10000

        # Invalid (non-positive)
        with pytest.raises(ValueError, match="must be positive"):
            VolumeBarSampler(volume_per_bar=0)

        with pytest.raises(ValueError, match="must be positive"):
            VolumeBarSampler(volume_per_bar=-100)

    def test_requires_side_column(self) -> None:
        """Test that volume bars require 'side' column."""
        tick_data = generate_tick_data(n_ticks=100)
        tick_data = tick_data.drop("side")  # Remove side column

        sampler = VolumeBarSampler(volume_per_bar=10000)

        with pytest.raises(DataValidationError, match="require 'side' column"):
            sampler.sample(tick_data)

    def test_basic_sampling(self) -> None:
        """Test basic volume bar sampling."""
        tick_data = generate_tick_data(n_ticks=500, seed=42)
        total_volume = tick_data["volume"].sum()

        # Sample with volume_per_bar = total_volume / 5
        volume_per_bar = total_volume / 5
        sampler = VolumeBarSampler(volume_per_bar=volume_per_bar)

        bars = sampler.sample(tick_data, include_incomplete=False)

        # Should have approximately 5 bars
        assert 3 <= len(bars) <= 7  # Allow some variance

        # Check bar structure
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns

        # Buy + sell should equal total volume per bar
        for i in range(len(bars)):
            total = bars["buy_volume"][i] + bars["sell_volume"][i]
            assert abs(total - bars["volume"][i]) < 1e-6

    def test_volume_threshold(self) -> None:
        """Test that bars form when volume threshold is reached."""
        tick_data = generate_tick_data(n_ticks=100, seed=42)
        sampler = VolumeBarSampler(volume_per_bar=5000)

        bars = sampler.sample(tick_data)

        # Each bar should have volume >= threshold (approximately)
        for vol in bars["volume"]:
            assert vol >= 4000  # Allow small tolerance

    def test_buy_sell_volume_split(self) -> None:
        """Test buy/sell volume calculation."""
        # Create data with known buy/sell split
        tick_data = pl.DataFrame(
            {
                "timestamp": [datetime.now()] * 10,
                "price": [100.0] * 10,
                "volume": [100.0] * 10,  # Total volume = 1000
                "side": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],  # 5 buys, 5 sells
            }
        )

        sampler = VolumeBarSampler(volume_per_bar=1000)
        bars = sampler.sample(tick_data)

        assert len(bars) == 1
        assert bars["buy_volume"][0] == 500.0
        assert bars["sell_volume"][0] == 500.0


# =============================================================================
# Dollar Bar Tests
# =============================================================================


class TestDollarBarSampler:
    """Tests for DollarBarSampler."""

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        # Valid
        sampler = DollarBarSampler(dollars_per_bar=100000)
        assert sampler.dollars_per_bar == 100000

        # Invalid (non-positive)
        with pytest.raises(ValueError, match="must be positive"):
            DollarBarSampler(dollars_per_bar=0)

        with pytest.raises(ValueError, match="must be positive"):
            DollarBarSampler(dollars_per_bar=-1000)

    def test_basic_sampling(self) -> None:
        """Test basic dollar bar sampling."""
        tick_data = generate_tick_data(n_ticks=500, seed=42)

        # Calculate total dollar volume
        total_dollars = (tick_data["price"] * tick_data["volume"]).sum()

        # Sample with dollars_per_bar = total / 5
        dollars_per_bar = total_dollars / 5
        sampler = DollarBarSampler(dollars_per_bar=dollars_per_bar)

        bars = sampler.sample(tick_data, include_incomplete=False)

        # Should have approximately 5 bars
        assert 3 <= len(bars) <= 7

        # Check bar structure
        assert "dollar_volume" in bars.columns
        assert "vwap" in bars.columns

    def test_vwap_calculation(self) -> None:
        """Test VWAP calculation."""
        # Create simple data for VWAP verification
        tick_data = pl.DataFrame(
            {
                "timestamp": [datetime.now()] * 4,
                "price": [100.0, 101.0, 99.0, 100.0],
                "volume": [10.0, 20.0, 30.0, 40.0],
            }
        )

        # Threshold set low enough that bar forms from first 3 ticks
        # Ticks 1-3: 100*10 + 101*20 + 99*30 = 5990 > 5000, bar forms
        # VWAP for first 3: 5990 / 60 = 99.833...
        expected_vwap_bar = (100 * 10 + 101 * 20 + 99 * 30) / (10 + 20 + 30)

        sampler = DollarBarSampler(dollars_per_bar=5000)
        bars = sampler.sample(tick_data)

        assert len(bars) == 1
        # VWAP should match the bar's ticks (first 3 ticks until threshold crossed)
        assert abs(bars["vwap"][0] - expected_vwap_bar) < 1e-6

    def test_dollar_threshold(self) -> None:
        """Test that bars form when dollar threshold is reached."""
        tick_data = generate_tick_data(n_ticks=200, base_price=100, seed=42)
        sampler = DollarBarSampler(dollars_per_bar=50000)

        bars = sampler.sample(tick_data)

        # Each bar should have dollar_volume >= threshold (approximately)
        for dv in bars["dollar_volume"]:
            assert dv >= 40000  # Allow tolerance


# =============================================================================
# Imbalance Bar Tests
# =============================================================================


class TestImbalanceBarSampler:
    """Tests for ImbalanceBarSampler."""

    def test_init_validation(self) -> None:
        """Test initialization validation."""
        # Valid
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=100,
            initial_expectation=1000,
            alpha=0.1,
        )
        assert sampler.expected_ticks_per_bar == 100
        assert sampler.initial_expectation == 1000
        assert sampler.alpha == 0.1

        # Invalid expected_ticks_per_bar
        with pytest.raises(ValueError, match="must be positive"):
            ImbalanceBarSampler(expected_ticks_per_bar=0)

        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ImbalanceBarSampler(expected_ticks_per_bar=100, alpha=1.5)

    def test_requires_side_column(self) -> None:
        """Test that imbalance bars require 'side' column."""
        tick_data = generate_tick_data(n_ticks=100)
        tick_data = tick_data.drop("side")

        sampler = ImbalanceBarSampler(expected_ticks_per_bar=50)

        with pytest.raises(DataValidationError, match="require 'side' column"):
            sampler.sample(tick_data)

    def test_basic_sampling(self) -> None:
        """Test basic imbalance bar sampling."""
        tick_data = generate_tick_data(n_ticks=500, seed=42)
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=100)

        bars = sampler.sample(tick_data, include_incomplete=False)

        # Should have some bars
        assert len(bars) > 0

        # Check bar structure
        assert "buy_volume" in bars.columns
        assert "sell_volume" in bars.columns
        assert "imbalance" in bars.columns
        assert "cumulative_theta" in bars.columns
        assert "expected_imbalance" in bars.columns

    def test_imbalance_calculation(self) -> None:
        """Test imbalance = buy_volume - sell_volume."""
        tick_data = generate_tick_data(n_ticks=200, seed=42)
        sampler = ImbalanceBarSampler(expected_ticks_per_bar=50)

        bars = sampler.sample(tick_data)

        for i in range(len(bars)):
            buy_vol = bars["buy_volume"][i]
            sell_vol = bars["sell_volume"][i]
            imbalance = bars["imbalance"][i]

            assert abs(imbalance - (buy_vol - sell_vol)) < 1e-6

    def test_adaptive_threshold(self) -> None:
        """Test that expected imbalance adapts over time."""
        tick_data = generate_tick_data(n_ticks=1000, seed=42)
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=50,
            initial_expectation=1000,
            alpha=0.1,
        )

        bars = sampler.sample(tick_data)

        # Expected imbalance should vary (adaptive)
        if len(bars) > 2:
            expectations = bars["expected_imbalance"].to_numpy()
            # Check that expectations change
            assert np.std(expectations) > 0

    def test_initial_expectation_estimation(self) -> None:
        """Test automatic estimation of initial expectation."""
        tick_data = generate_tick_data(n_ticks=500, seed=42)

        # Don't provide initial_expectation
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=100,
            initial_expectation=None,  # Will be estimated
        )

        bars = sampler.sample(tick_data)

        # Should still produce bars
        assert len(bars) > 0

        # NOTE: initial_expectation is now computed dynamically per AFML methodology
        # and is not stored on the sampler instance. The threshold adapts based on
        # the data, so we just verify bars are produced successfully.


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestBarSamplerValidation:
    """Tests for data validation across all samplers."""

    def test_non_numeric_price(self) -> None:
        """Test that non-numeric price raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "price": ["not_a_number"],  # Invalid
                "volume": [100.0],
            }
        )

        sampler = TickBarSampler(ticks_per_bar=10)

        with pytest.raises(DataValidationError, match="must be numeric"):
            sampler.sample(df)

    def test_non_numeric_volume(self) -> None:
        """Test that non-numeric volume raises error."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "price": [100.0],
                "volume": ["invalid"],  # Invalid
            }
        )

        sampler = TickBarSampler(ticks_per_bar=10)

        with pytest.raises(DataValidationError, match="must be numeric"):
            sampler.sample(df)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBarSamplerIntegration:
    """Integration tests comparing different bar types."""

    def test_all_samplers_produce_bars(self) -> None:
        """Test that all samplers can process the same tick data."""
        tick_data = generate_tick_data(n_ticks=1000, seed=42)

        # Tick bars
        tick_sampler = TickBarSampler(ticks_per_bar=100)
        tick_bars = tick_sampler.sample(tick_data)
        assert len(tick_bars) > 0

        # Volume bars
        volume_sampler = VolumeBarSampler(volume_per_bar=50000)
        volume_bars = volume_sampler.sample(tick_data)
        assert len(volume_bars) > 0

        # Dollar bars
        dollar_sampler = DollarBarSampler(dollars_per_bar=500000)
        dollar_bars = dollar_sampler.sample(tick_data)
        assert len(dollar_bars) > 0

        # Imbalance bars
        imbalance_sampler = ImbalanceBarSampler(expected_ticks_per_bar=100)
        imbalance_bars = imbalance_sampler.sample(tick_data)
        assert len(imbalance_bars) > 0

    def test_bars_capture_market_activity(self) -> None:
        """Test that information bars adapt to changing market conditions."""
        # Create data with changing volatility
        low_vol_data = generate_tick_data(n_ticks=500, volatility=0.5, seed=42)
        high_vol_data = generate_tick_data(n_ticks=500, volatility=2.0, seed=43)

        # Volume bars should form faster during high activity
        volume_sampler = VolumeBarSampler(volume_per_bar=50000)

        low_vol_bars = volume_sampler.sample(low_vol_data)
        high_vol_bars = volume_sampler.sample(high_vol_data)

        # Both should produce bars (basic smoke test)
        assert len(low_vol_bars) > 0
        assert len(high_vol_bars) > 0
