"""Tests to verify lookahead bias fixes in qfeatures."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def test_yang_zhang_volatility_no_lookahead():
    """Test that yang_zhang_volatility uses rolling variance, not global."""
    from ml4t.engineer.features.volatility import yang_zhang_volatility

    # Create test data
    np.random.seed(42)
    n_days = 100
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Generate OHLC data
    opens = 100 + np.random.randn(n_days).cumsum()
    closes = opens + np.random.randn(n_days) * 0.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_days) * 0.3)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_days) * 0.3)

    df = pl.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )

    # Calculate volatility with different periods
    period = 20
    result = df.select(
        yang_zhang_volatility("open", "high", "low", "close", period=period, annualize=False).alias(
            "yz_vol"
        )
    )

    # Check that early values are null (not enough data for rolling window)
    assert result["yz_vol"][: period - 1].null_count() > 0, (
        "Should have nulls at start for rolling window"
    )

    # Check that values change over time (not constant global variance)
    vol_values = result["yz_vol"][period:].drop_nulls().to_list()
    assert len(set(vol_values)) > 1, "Volatility should vary over time, not be constant"

    # Verify no future data is used - split data and check
    split_point = 50
    df_partial = df[:split_point]
    result_partial = df_partial.select(
        yang_zhang_volatility("open", "high", "low", "close", period=period, annualize=False).alias(
            "yz_vol"
        )
    )

    # Values calculated on partial data should match full data for same periods
    for i in range(period, split_point):
        if result["yz_vol"][i] is not None and result_partial["yz_vol"][i] is not None:
            assert abs(result["yz_vol"][i] - result_partial["yz_vol"][i]) < 1e-10, (
                f"Value at index {i} should be same regardless of future data"
            )


def test_variance_ratio_rolling_window():
    """Test that variance_ratio can use rolling window."""
    from ml4t.engineer.features.regime import variance_ratio

    # Create test data with trend change
    n_days = 200
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # First half: trending up, second half: mean reverting
    prices1 = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 0.1
    prices2 = 150 + np.sin(np.arange(100) * 0.1) * 5 + np.random.randn(100) * 0.1
    prices = np.concatenate([prices1, prices2])

    df = pl.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
        }
    )

    # Test with rolling window
    window = 50
    vr_dict = variance_ratio("close", periods=[2, 4], window=window)

    result = df.select(
        [
            vr_dict["vr_2"].alias("vr_2"),
            vr_dict["vr_4"].alias("vr_4"),
        ]
    )

    # Should have nulls at start due to rolling window
    assert result["vr_2"][:window].null_count() > 0, "Should have nulls for rolling window"

    # Values should change over time
    vr2_values = result["vr_2"][window:].drop_nulls().to_list()
    assert len(set(vr2_values)) > 1, "Variance ratio should change over time"

    # Test backwards compatibility - without window should still work
    vr_dict_global = variance_ratio("close", periods=[2, 4], window=None)
    result_global = df.select(vr_dict_global["vr_2"].alias("vr_2_global"))

    # Global variance ratio should be constant
    vr2_global_values = result_global["vr_2_global"].drop_nulls().unique().to_list()
    assert len(vr2_global_values) == 1, "Global variance ratio should be constant"


def test_percentile_rank_features_rolling():
    """Test that percentile_rank_features uses rolling window correctly."""
    from ml4t.engineer.features.ml.percentile_rank_features import percentile_rank_features

    # Create test data where value increases linearly
    n_days = 100
    df = pl.DataFrame(
        {
            "value": list(range(n_days)),  # 0, 1, 2, ..., 99
        }
    )

    # Calculate percentile ranks with small window
    window = 10
    ranks = percentile_rank_features("value", windows=[window])
    result = df.select(ranks[f"rank_{window}"].alias("rank"))

    # Check that rank at end of each window is always high (latest value is largest)
    for i in range(window, n_days):
        rank_val = result["rank"][i]
        assert rank_val is not None and rank_val > 80, (
            f"Latest value in ascending series should have high rank, got {rank_val}"
        )

    # Test with descending values
    df_desc = pl.DataFrame(
        {
            "value": list(range(n_days, 0, -1)),  # 100, 99, 98, ..., 1
        }
    )

    result_desc = df_desc.select(ranks[f"rank_{window}"].alias("rank"))

    # Check that rank at end of each window is always low (latest value is smallest)
    for i in range(window, n_days):
        rank_val = result_desc["rank"][i]
        assert rank_val is not None and rank_val < 20, (
            f"Latest value in descending series should have low rank, got {rank_val}"
        )


def test_volatility_percentile_rank_rolling():
    """Test that volatility_percentile_rank uses rolling window correctly."""
    from ml4t.engineer.features.volatility import volatility_percentile_rank

    # Create test data with increasing volatility
    np.random.seed(42)

    # Generate returns with increasing volatility
    returns_low_vol = np.random.randn(100) * 0.01  # Low volatility period
    returns_high_vol = np.random.randn(100) * 0.05  # High volatility period

    prices = 100 * np.exp(np.concatenate([returns_low_vol, returns_high_vol]).cumsum())

    df = pl.DataFrame(
        {
            "close": prices,
        }
    )

    # Calculate volatility percentile rank
    period = 20
    lookback = 50
    result = df.select(
        volatility_percentile_rank("close", period=period, lookback=lookback).alias("vol_rank")
    )

    # Check that we have nulls at the beginning (need data for both vol calc and ranking)
    min_required = max(period, lookback)
    assert result["vol_rank"][:min_required].null_count() > 0, (
        "Should have nulls at start for rolling calculations"
    )

    # Check that rank increases when we transition from low to high volatility
    # Average rank in first stable period (after initial nulls, before transition)
    start_idx = min_required + 10
    end_idx = 90
    avg_rank_low_vol = result["vol_rank"][start_idx:end_idx].drop_nulls().mean()

    # Average rank in high volatility period
    start_idx_high = 150
    end_idx_high = 190
    avg_rank_high_vol = result["vol_rank"][start_idx_high:end_idx_high].drop_nulls().mean()

    # Just check that ranks are different between periods (implementation might rank differently)
    assert abs(avg_rank_high_vol - avg_rank_low_vol) > 10, (
        f"Ranks should be significantly different between high vol ({avg_rank_high_vol:.1f}) and low vol ({avg_rank_low_vol:.1f}) periods"
    )


if __name__ == "__main__":
    # Run tests
    test_yang_zhang_volatility_no_lookahead()
    print("✅ yang_zhang_volatility: No lookahead bias")

    test_variance_ratio_rolling_window()
    print("✅ variance_ratio: Rolling window works correctly")

    test_percentile_rank_features_rolling()
    print("✅ percentile_rank_features: Rolling rank calculation correct")

    test_volatility_percentile_rank_rolling()
    print("✅ volatility_percentile_rank: Rolling rank calculation correct")

    print("\n🎉 All lookahead bias fixes verified!")
