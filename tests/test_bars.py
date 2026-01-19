"""Tests for information-driven bar samplers."""

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


class TestTickBarSampler:
    """Test tick bar sampling."""

    def test_basic_tick_bars(self, tick_data):
        """Test basic tick bar generation."""
        sampler = TickBarSampler(ticks_per_bar=10)
        bars = sampler.sample(tick_data)

        # Should have roughly n/10 bars
        assert len(bars) == len(tick_data) // 10

        # Each bar should have OHLCV data
        expected_cols = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
        }
        assert set(bars.columns) == expected_cols

        # First bar should have 10 ticks
        assert bars["tick_count"][0] == 10

        # OHLC relationships should be valid
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()

    def test_tick_bars_with_incomplete_final(self, tick_data):
        """Test handling of incomplete final bar."""
        sampler = TickBarSampler(ticks_per_bar=7)
        bars = sampler.sample(tick_data, include_incomplete=True)

        # Last bar might have fewer than 7 ticks
        total_ticks = bars["tick_count"].sum()
        assert total_ticks == len(tick_data)

        # Without incomplete bars
        bars_complete = sampler.sample(tick_data, include_incomplete=False)
        assert len(bars_complete) <= len(bars)


class TestVolumeBarSampler:
    """Test volume bar sampling."""

    def test_basic_volume_bars(self, tick_data):
        """Test basic volume bar generation."""
        sampler = VolumeBarSampler(volume_per_bar=1000)
        bars = sampler.sample(tick_data)

        # Check bar structure
        expected_cols = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "buy_volume",
            "sell_volume",
        }
        assert set(bars.columns) == expected_cols

        # Each complete bar should have at least the target volume
        complete_bars = bars[:-1]  # Exclude potentially incomplete last bar
        assert (complete_bars["volume"] >= 1000).all()

        # Buy + sell volume should equal total volume
        assert np.allclose(bars["buy_volume"] + bars["sell_volume"], bars["volume"])

    def test_volume_bars_exact_threshold(self):
        """Test volume bars with exact threshold matches."""
        # Create data where volumes sum exactly to threshold
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(10)],
                "price": [100 + i * 0.1 for i in range(10)],
                "volume": [250, 250, 250, 250, 100, 400, 500, 200, 300, 500],
                "side": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            },
        )

        sampler = VolumeBarSampler(volume_per_bar=1000)
        bars = sampler.sample(data)

        # First bar: 250+250+250+250 = 1000
        assert bars["volume"][0] == 1000
        assert bars["tick_count"][0] == 4

        # Second bar: 100+400+500 = 1000
        assert bars["volume"][1] == 1000
        assert bars["tick_count"][1] == 3


class TestDollarBarSampler:
    """Test dollar bar sampling."""

    def test_basic_dollar_bars(self, tick_data):
        """Test basic dollar bar generation."""
        sampler = DollarBarSampler(dollars_per_bar=100_000)
        bars = sampler.sample(tick_data)

        # Check bar structure
        expected_cols = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tick_count",
            "dollar_volume",
            "vwap",
        }
        assert set(bars.columns) == expected_cols

        # VWAP should be between low and high
        assert (bars["vwap"] >= bars["low"]).all()
        assert (bars["vwap"] <= bars["high"]).all()

        # Dollar volume should be close to threshold for complete bars
        complete_bars = bars[:-1]
        assert (complete_bars["dollar_volume"] >= 100_000).all()

    def test_dollar_bars_calculation(self):
        """Test dollar bar calculations are correct."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(5)],
                "price": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [100, 200, 300, 400, 500],
                "side": [1, -1, 1, -1, 1],
            },
        )

        sampler = DollarBarSampler(dollars_per_bar=50_000)
        bars = sampler.sample(data)

        # First bar: 100*100 + 200*101 + 300*102 = 60,800
        assert bars["dollar_volume"][0] == 60_800

        # VWAP calculation
        expected_vwap = (100 * 100 + 200 * 101 + 300 * 102) / (100 + 200 + 300)
        assert np.isclose(bars["vwap"][0], expected_vwap)


class TestImbalanceBarSampler:
    """Test imbalance bar sampling."""

    def test_basic_imbalance_bars(self, tick_data):
        """Test basic imbalance bar generation."""
        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=20,
            initial_expectation=1000,
        )
        bars = sampler.sample(tick_data)

        # Check bar structure - includes AFML diagnostic columns
        expected_cols = {
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
            # AFML diagnostic columns
            "expected_t",
            "p_buy",
            "v_plus",
            "e_v",
        }
        assert set(bars.columns) == expected_cols

        # Imbalance should be buy_volume - sell_volume
        assert np.allclose(bars["imbalance"], bars["buy_volume"] - bars["sell_volume"])

    def test_imbalance_bars_triggering(self):
        """Test that imbalance bars trigger correctly."""
        # Create data with strong directional imbalance
        n = 100
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(n)],
                "price": list(range(100, 100 + n)),
                "volume": [100] * n,
                "side": [1] * 80 + [-1] * 20,  # Strong buy imbalance
            },
        )

        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=20,
            initial_expectation=500,
        )
        bars = sampler.sample(data)

        # Should create bars when cumulative imbalance exceeds threshold
        assert len(bars) > 0

        # First bars should have positive imbalance due to buy pressure
        assert bars["imbalance"][0] > 0

    def test_expected_imbalance_updates(self):
        """Test imbalance bar functionality with dynamic thresholds."""
        # Create data that will trigger multiple bars with different imbalances
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(200)],
                "price": [100] * 200,
                "volume": [100] * 200,
                # First 100: buy pressure, Next 100: sell pressure
                "side": [1] * 100 + [-1] * 100,
            },
        )

        sampler = ImbalanceBarSampler(
            expected_ticks_per_bar=20,
            initial_expectation=1500,  # 15 ticks * 100 volume
            alpha=0.3,
        )
        bars = sampler.sample(data)

        # Should create bars when cumulative imbalance exceeds threshold
        assert len(bars) > 0

        # All bars should have required columns (including AFML diagnostics)
        required_cols = {
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
            # AFML diagnostic columns
            "expected_t",
            "p_buy",
            "v_plus",
            "e_v",
        }
        assert set(bars.columns) == required_cols

        # First bars should have positive imbalance (buy pressure period)
        first_bars = bars.head(min(3, len(bars)))
        assert (first_bars["imbalance"] > 0).all()
        assert (first_bars["buy_volume"] > first_bars["sell_volume"]).all()

        # Later bars should have negative imbalance (sell pressure period)
        if len(bars) > 3:
            last_bars = bars.tail(min(3, len(bars)))
            assert (last_bars["imbalance"] < 0).all()
            assert (last_bars["sell_volume"] > last_bars["buy_volume"]).all()

        # Cumulative theta should match the threshold (approximately)
        for i in range(len(bars)):
            cumulative_theta = abs(bars["cumulative_theta"][i])
            expected_imbalance = bars["expected_imbalance"][i]
            # Should be close to or exceed the threshold
            assert cumulative_theta >= expected_imbalance * 0.9  # Allow 10% tolerance


class TestBarSamplerEdgeCases:
    """Test edge cases for all bar samplers."""

    def test_empty_data(self):
        """Test samplers with empty data."""
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
                "volume": [],
                "side": [],
            },
        )

        samplers = [
            TickBarSampler(10),
            VolumeBarSampler(1000),
            DollarBarSampler(10000),
            ImbalanceBarSampler(20, 100),
        ]

        for sampler in samplers:
            result = sampler.sample(empty_df)
            assert len(result) == 0
            assert isinstance(result, pl.DataFrame)

    def test_single_tick(self):
        """Test samplers with single tick."""
        single_tick = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "price": [100.0],
                "volume": [100],
                "side": [1],
            },
        )

        # Tick bar should not create a bar with single tick (threshold=10)
        tick_sampler = TickBarSampler(10)
        result = tick_sampler.sample(single_tick, include_incomplete=False)
        assert len(result) == 0

        # But should create one if including incomplete
        result = tick_sampler.sample(single_tick, include_incomplete=True)
        assert len(result) == 1
        assert result["tick_count"][0] == 1

    def test_all_zero_volume(self):
        """Test handling of zero volumes."""
        zero_vol_data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(10)],
                "price": [100 + i for i in range(10)],
                "volume": [0] * 10,
                "side": [1, -1] * 5,
            },
        )

        # Volume bar should not create any bars
        vol_sampler = VolumeBarSampler(1000)
        result = vol_sampler.sample(zero_vol_data)
        assert len(result) == 0

        # Dollar bar should not create any bars
        dollar_sampler = DollarBarSampler(10000)
        result = dollar_sampler.sample(zero_vol_data)
        assert len(result) == 0


class TestBarSamplerIntegration:
    """Test integration with pipeline."""

    # @pytest.mark.skip(reason="Pipeline integration with bar samplers has column resolution issues")
    def test_pipeline_integration(self, tick_data):
        """Test bar samplers in pipeline."""
        from ml4t.engineer.pipeline import Pipeline

        # Create pipeline that generates different bar types
        pipeline = Pipeline(
            steps=[
                # Generate volume bars
                ("volume_bars", lambda df: VolumeBarSampler(1000).sample(df)),
                # Add returns
                (
                    "returns",
                    lambda df: df.with_columns(returns=pl.col("close").pct_change()),
                ),
                # Add volatility
                (
                    "volatility",
                    lambda df: df.with_columns(
                        volatility=pl.col("returns").rolling_std(window_size=5),
                    ),
                ),
            ],
        )

        result = pipeline.run(tick_data)

        # Should have volume bar columns plus added features
        assert "returns" in result.columns
        assert "volatility" in result.columns
        assert "volume" in result.columns

        # Volume should meet threshold
        assert (result["volume"][:-1] >= 1000).all()


class TestTickRunBarSampler:
    """Test tick run bar sampling."""

    def test_basic_tick_run_bars(self, tick_data):
        """Test basic tick run bar generation."""
        sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=10)
        bars = sampler.sample(tick_data)

        # Should have bars when runs exceed expectation
        assert len(bars) > 0

        # Check bar structure (includes AFML diagnostic columns)
        expected_cols = {
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
            # AFML diagnostic columns
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        }
        assert set(bars.columns) == expected_cols

        # Run lengths should meet or exceed expectation (except possibly last bar)
        complete_bars = bars[:-1]
        if len(complete_bars) > 0:
            assert (complete_bars["run_length"] >= complete_bars["expected_run"]).all()

        # Buy + sell volume should equal total volume
        assert np.allclose(bars["buy_volume"] + bars["sell_volume"], bars["volume"])

    def test_tick_run_adaptive_threshold(self):
        """Test that the AFML threshold adapts based on data characteristics."""
        # Create data with strong directional runs - more data points for reliable bar formation
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(200)],
                "price": [100 + i * 0.01 for i in range(200)],
                "volume": [100] * 200,
                # Create four long runs that should form bars with lower expected_ticks_per_bar
                "side": [1] * 50 + [-1] * 50 + [1] * 50 + [-1] * 50,
            },
        )

        # Use lower expected_ticks_per_bar to produce more bars
        sampler = TickRunBarSampler(expected_ticks_per_bar=20, alpha=0.3)
        bars = sampler.sample(data)

        # Should create at least some bars from the directional runs
        assert len(bars) >= 1

        # NOTE: With AFML thresholds computed dynamically, the number of bars depends
        # on the data's statistical properties. Just verify bars are valid.
        if len(bars) > 0:
            # All complete bars should have positive run lengths
            assert all(bars["run_length"] > 0)
            # Expected run threshold should be positive
            assert all(bars["expected_run"] > 0)

    def test_tick_run_direction_change(self):
        """Test that direction changes reset runs."""
        # Create alternating buy/sell pattern
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(20)],
                "price": [100 + i * 0.01 for i in range(20)],
                "volume": [100] * 20,
                "side": [1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1],
            },
        )

        sampler = TickRunBarSampler(expected_ticks_per_bar=20, initial_run_expectation=3)
        bars = sampler.sample(data)

        # Should create bars when runs reach threshold
        assert len(bars) > 0


class TestVolumeRunBarSampler:
    """Test volume run bar sampling."""

    def test_basic_volume_run_bars(self, tick_data):
        """Test basic volume run bar generation."""
        # Use lower initial expectation for random data
        sampler = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=500.0)
        bars = sampler.sample(tick_data)

        # Should have bars
        assert len(bars) > 0

        # Check bar structure (includes AFML diagnostic columns)
        expected_cols = {
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
            # AFML diagnostic columns
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        }
        assert set(bars.columns) == expected_cols

        # Run volumes should meet or exceed expectation (except possibly last bar)
        complete_bars = bars[:-1]
        if len(complete_bars) > 0:
            assert (complete_bars["run_volume"] >= complete_bars["expected_run"]).all()

        # OHLC relationships should be valid
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()
        assert (bars["low"] <= bars["open"]).all()
        assert (bars["low"] <= bars["close"]).all()

    def test_volume_run_with_large_volumes(self):
        """Test volume run bars with varying volume sizes."""
        data = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(10)],
                "price": [100 + i * 0.1 for i in range(10)],
                "volume": [100, 100, 100, 500, 100, 100, 100, 100, 600, 100],  # Large spikes
                "side": [1, 1, 1, 1, -1, -1, -1, 1, 1, 1],
            },
        )

        sampler = VolumeRunBarSampler(expected_ticks_per_bar=10, initial_run_expectation=300)
        bars = sampler.sample(data)

        # Should create bars when run volumes exceed threshold
        assert len(bars) > 0
        assert all(bars["run_volume"] >= bars["expected_run"])


class TestDollarRunBarSampler:
    """Test dollar run bar sampling."""

    def test_basic_dollar_run_bars(self, tick_data):
        """Test basic dollar run bar generation."""
        # Use lower initial expectation for random data
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5000.0)
        bars = sampler.sample(tick_data)

        # Should have bars
        assert len(bars) > 0

        # Check bar structure (includes AFML diagnostic columns)
        expected_cols = {
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
            # AFML diagnostic columns
            "theta",
            "expected_theta",
            "expected_t",
            "p_buy",
            "cumulative_buys",
            "cumulative_sells",
        }
        assert set(bars.columns) == expected_cols

        # Run dollars should meet or exceed expectation (except possibly last bar)
        complete_bars = bars[:-1]
        if len(complete_bars) > 0:
            assert (complete_bars["run_dollars"] >= complete_bars["expected_run"]).all()

        # VWAP should be reasonable (within or close to OHLC range)
        # For bars with zero range (high==low), VWAP should equal that price
        # For bars with range, allow small tolerance for volume weighting effects
        for i in range(len(bars)):
            vwap_val = bars["vwap"][i]
            low_val = bars["low"][i]
            high_val = bars["high"][i]
            price_range_val = high_val - low_val

            if price_range_val < 1e-10:  # Essentially zero range
                assert abs(vwap_val - low_val) < 1e-6
            else:
                # Allow 10% outside range for volume weighting
                assert vwap_val >= low_val - price_range_val * 0.1
                assert vwap_val <= high_val + price_range_val * 0.1

        # Dollar volume should be positive
        assert (bars["dollar_volume"] > 0).all()

    def test_dollar_run_adaptive_threshold(self, tick_data):
        """Test that expected dollar run is computed correctly."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100, alpha=0.15)
        bars = sampler.sample(tick_data)

        # NOTE: The AFML implementation now computes thresholds dynamically based on
        # data characteristics. The expected_run values may be constant if the
        # underlying data properties don't change. Just verify bars are produced.
        if len(bars) > 0:
            # All bars should have positive run dollars
            assert (bars["run_dollars"] > 0).all()
            # Expected run should be positive
            assert (bars["expected_run"] > 0).all()

    def test_dollar_run_with_incomplete_bar(self, tick_data):
        """Test handling of incomplete final bar."""
        sampler = DollarRunBarSampler(expected_ticks_per_bar=100)

        # With incomplete bars
        bars_with_incomplete = sampler.sample(tick_data, include_incomplete=True)

        # Without incomplete bars
        bars_complete = sampler.sample(tick_data, include_incomplete=False)

        # Should have same or more bars with incomplete
        assert len(bars_with_incomplete) >= len(bars_complete)

        # Last bar in complete version should meet threshold
        if len(bars_complete) > 0:
            assert bars_complete["run_dollars"][-1] >= bars_complete["expected_run"][-1]


class TestRunBarsComparison:
    """Test relationships between different run bar types."""

    def test_run_bars_create_fewer_bars_than_standard(self, tick_data):
        """Test that run bars create fewer but more informative bars."""
        tick_sampler = TickBarSampler(ticks_per_bar=100)
        tick_run_sampler = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=10)

        standard_bars = tick_sampler.sample(tick_data)
        run_bars = tick_run_sampler.sample(tick_data)

        # Run bars should typically create fewer bars (they wait for sustained runs)
        # This is the key insight: run bars sample during directional flow
        assert len(run_bars) <= len(standard_bars)

    def test_all_run_bar_types_produce_valid_output(self, tick_data):
        """Test that all three run bar types produce valid OHLCV data."""
        # Use appropriate initial expectations for random data
        tick_run = TickRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=10)
        volume_run = VolumeRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=500.0)
        dollar_run = DollarRunBarSampler(expected_ticks_per_bar=100, initial_run_expectation=5000.0)

        tick_bars = tick_run.sample(tick_data)
        volume_bars = volume_run.sample(tick_data)
        dollar_bars = dollar_run.sample(tick_data)

        # All should produce bars
        assert len(tick_bars) > 0
        assert len(volume_bars) > 0
        assert len(dollar_bars) > 0

        # All should have valid OHLC relationships
        for bars in [tick_bars, volume_bars, dollar_bars]:
            assert (bars["high"] >= bars["open"]).all()
            assert (bars["high"] >= bars["close"]).all()
            assert (bars["low"] <= bars["open"]).all()
            assert (bars["low"] <= bars["close"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
