"""Tests for market microstructure features."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.microstructure import (
    amihud_illiquidity,
    effective_tick_rule,
    kyle_lambda,
    order_flow_imbalance,
    price_impact_ratio,
    quote_stuffing_indicator,
    realized_spread,
    roll_spread_estimator,
    trade_intensity,
    volume_at_price_ratio,
    volume_synchronicity,
    volume_weighted_price_momentum,
)


class TestMicrostructureFeatures:
    """Test microstructure feature calculations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        np.random.seed(42)
        n = 100

        # Create realistic price movements
        returns = np.random.normal(0, 0.001, n)
        close_prices = 100 * np.exp(np.cumsum(returns))

        # Create OHLC data
        daily_range = np.abs(np.random.normal(0, 0.002, n))
        high_prices = close_prices + daily_range * close_prices * 0.5
        low_prices = close_prices - daily_range * close_prices * 0.5

        # Volume with some patterns
        base_volume = 1_000_000
        volume = base_volume + base_volume * 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
        volume = np.abs(volume) + np.random.normal(0, base_volume * 0.1, n)
        volume = np.maximum(volume, 1000)  # Ensure positive

        # Number of trades (for quote stuffing)
        num_trades = volume / 100 + np.random.normal(0, 100, n)
        num_trades = np.maximum(num_trades, 10)

        df = pl.DataFrame(
            {
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
                "returns": returns,
                "price": close_prices,  # For dollar volume
                "num_trades": num_trades,
            },
        )

        return df

    def test_amihud_illiquidity(self, sample_data):
        """Test Amihud illiquidity measure."""
        result = sample_data.select(
            amihud_illiquidity("returns", "volume", "price", period=20).alias("amihud"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Amihud should be positive (absolute returns / dollar volume)
        amihud_values = result["amihud"].drop_nulls()
        assert (amihud_values >= 0).all()

        # Check reasonable range (depends on scaling)
        assert amihud_values.mean() > 0
        assert amihud_values.mean() < 100  # Reasonable upper bound

    def test_kyle_lambda(self, sample_data):
        """Test Kyle's lambda calculation."""
        # Test ratio method
        result_ratio = sample_data.select(
            kyle_lambda("returns", "volume", period=20, method="ratio").alias(
                "kyle_ratio",
            ),
        )

        # Verify results are positive
        kyle_ratio = result_ratio["kyle_ratio"].drop_nulls()
        assert (kyle_ratio >= 0).all()

        # Note: regression method is not yet implemented and will raise NotImplementedError

    def test_roll_spread_estimator(self, sample_data):
        """Test Roll spread estimation."""
        result = sample_data.select(
            roll_spread_estimator("close", period=20).alias("roll_spread"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Spread should be non-negative (it's 2*sqrt(max(-cov, 0)))
        spreads = result["roll_spread"].drop_nulls()
        assert (spreads >= 0).all()

        # Should be a small percentage of price
        avg_spread_pct = spreads.mean() / sample_data["close"].mean()
        assert avg_spread_pct < 0.01  # Less than 1%

    def test_effective_tick_rule(self, sample_data):
        """Test effective tick rule for trade classification."""
        result = sample_data.select(effective_tick_rule("close").alias("tick_rule"))

        # Check output shape
        assert len(result) == len(sample_data)

        # Values should be -1, 0, or 1
        tick_values = result["tick_rule"]
        unique_values = set(tick_values.unique().to_list())
        assert unique_values.issubset({-1, 0, 1})

        # Should have mix of buy and sell signals
        assert (tick_values == 1).sum() > 0  # Some buys
        assert (tick_values == -1).sum() > 0  # Some sells

    def test_volume_weighted_price_momentum(self, sample_data):
        """Test volume-weighted price momentum."""
        result = sample_data.select(
            volume_weighted_price_momentum("close", "volume", period=20).alias("vwpm"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # VWPM should be close to regular momentum but volume weighted
        vwpm_values = result["vwpm"].drop_nulls()

        # Should oscillate around 0
        assert -0.1 < vwpm_values.mean() < 0.1

    def test_order_flow_imbalance(self, sample_data):
        """Test order flow imbalance calculation."""
        result = sample_data.select(
            order_flow_imbalance("volume", "close", use_tick_rule=True).alias("ofi"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # OFI should be between -1 and 1
        ofi_values = result["ofi"].drop_nulls()
        assert (ofi_values >= -1).all()
        assert (ofi_values <= 1).all()

        # Should have both positive and negative values
        assert (ofi_values > 0).sum() > 0
        assert (ofi_values < 0).sum() > 0

    def test_volume_at_price_ratio(self, sample_data):
        """Test volume concentration at price levels."""
        result = sample_data.select(
            volume_at_price_ratio(
                "high",
                "low",
                "close",
                "volume",
                n_bins=10,
                period=20,
            ).alias("vap_ratio"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Ratio should be between 0 and 1
        vap_values = result["vap_ratio"].drop_nulls()
        assert (vap_values >= 0).all()
        assert (vap_values <= 1).all()

    def test_price_impact_ratio(self, sample_data):
        """Test price impact ratio calculation."""
        result = sample_data.select(
            price_impact_ratio(
                "returns",
                "volume",
                period=20,
                impact_threshold=0.001,
            ).alias("impact_ratio"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Ratio should be non-negative
        impact_values = result["impact_ratio"].drop_nulls()
        assert (impact_values >= 0).all()

    def test_realized_spread(self, sample_data):
        """Test realized spread calculation."""
        result = sample_data.select(
            realized_spread("high", "low", "close", period=20).alias("real_spread"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Spread should be positive
        spread_values = result["real_spread"].drop_nulls()
        assert (spread_values >= 0).all()

        # Should be small percentage
        avg_spread_pct = spread_values.mean()
        assert avg_spread_pct < 0.02  # Less than 2%

    def test_volume_synchronicity(self, sample_data):
        """Test volume-return synchronicity."""
        result = sample_data.select(
            volume_synchronicity("volume", "returns", period=20).alias("sync"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Synchronicity should be between -1 and 1 (correlation)
        sync_values = result["sync"].drop_nulls()
        assert (sync_values >= -1).all()
        assert (sync_values <= 1).all()

    def test_trade_intensity(self, sample_data):
        """Test trade intensity calculation."""
        result = sample_data.select(
            trade_intensity("volume", time_interval=1, period=20).alias("intensity"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Intensity should be positive (normalized volume)
        intensity_values = result["intensity"].drop_nulls()
        assert (intensity_values > 0).all()

        # Should average around 1 (normalized)
        assert 0.5 < intensity_values.mean() < 2.0

    def test_quote_stuffing_indicator(self, sample_data):
        """Test quote stuffing detection."""
        # Test with trade count
        result_with_trades = sample_data.select(
            quote_stuffing_indicator("volume", "num_trades", period=5).alias("stuffing"),
        )

        # Test without trade count
        result_no_trades = sample_data.select(
            quote_stuffing_indicator("volume", None, period=5).alias("stuffing_vol"),
        )

        # Both should return 0 or 1
        stuff_values = result_with_trades["stuffing"].drop_nulls()
        assert set(stuff_values.unique().to_list()).issubset({0.0, 1.0})

        stuff_vol_values = result_no_trades["stuffing_vol"].drop_nulls()
        assert set(stuff_vol_values.unique().to_list()).issubset({0.0, 1.0})

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        # Test with zero volume
        df_zero_vol = sample_data.with_columns(pl.lit(0).alias("zero_volume"))

        result = df_zero_vol.select(
            amihud_illiquidity("returns", "zero_volume", "price", period=5).alias(
                "amihud",
            ),
        )
        # Should handle division by zero gracefully
        assert len(result) == len(df_zero_vol)

        # Test with constant prices
        df_const_price = sample_data.with_columns(pl.lit(100.0).alias("const_price"))

        result_const = df_const_price.select(
            effective_tick_rule("const_price").alias("tick"),
        )
        # Should return mostly zeros for no price change
        tick_values = result_const["tick"]
        assert (tick_values == 0).sum() > len(tick_values) * 0.8

    def test_with_nulls(self):
        """Test handling of null values."""
        df_with_nulls = pl.DataFrame(
            {
                "high": [101, 102, None, 104, 105],
                "low": [99, 100, None, 102, 103],
                "close": [100, 101, None, 103, 104],
                "volume": [1000, None, 1500, 2000, 1800],
                "returns": [0.01, 0.01, None, 0.02, 0.01],
                "price": [100, 101, None, 103, 104],
            },
        )

        # Should handle nulls gracefully
        result = df_with_nulls.select(
            amihud_illiquidity("returns", "volume", "price", period=3).alias("amihud"),
        )
        assert len(result) == len(df_with_nulls)

        # Some values should be null due to missing data
        assert result["amihud"].null_count() > 0

    def test_realistic_values(self, sample_data):
        """Test that values are in realistic ranges."""
        # Calculate multiple features
        result = sample_data.select(
            [
                amihud_illiquidity("returns", "volume", "price", period=20).alias(
                    "amihud",
                ),
                kyle_lambda("returns", "volume", period=20).alias("kyle"),
                roll_spread_estimator("close", period=20).alias("roll_spread"),
                realized_spread("high", "low", "close", period=20).alias("real_spread"),
            ],
        )

        # Check that liquidity measures are correlated
        # (They all measure aspects of liquidity/market impact)
        df_clean = result.drop_nulls()
        if len(df_clean) > 30:
            # Amihud and Kyle should be positively correlated
            corr_matrix = df_clean.select(
                [
                    pl.corr("amihud", "kyle").alias("amihud_kyle_corr"),
                ],
            )
            assert corr_matrix["amihud_kyle_corr"][0] > -0.5  # Not strongly negative
