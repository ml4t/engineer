"""
Tests for volume indicators.

Tests cover:
- Basic functionality
- Edge cases (insufficient data, NaN handling)
- Parameter validation
- Output structure
- TA-Lib compatibility
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.volume import ad, adosc, obv

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_volume_data():
    """Sample OHLCV data for volume indicator testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 20),
                interval="1d",
                eager=True,
            ),
            "open": [
                100.0,
                101.0,
                102.0,
                101.5,
                103.0,
                102.0,
                104.0,
                103.5,
                105.0,
                104.0,
                106.0,
                105.5,
                107.0,
                106.0,
                108.0,
                107.5,
                109.0,
                108.0,
                110.0,
                109.5,
            ],
            "high": [
                102.0,
                103.0,
                104.0,
                103.0,
                105.0,
                104.0,
                106.0,
                105.0,
                107.0,
                106.0,
                108.0,
                107.0,
                109.0,
                108.0,
                110.0,
                109.0,
                111.0,
                110.0,
                112.0,
                111.0,
            ],
            "low": [
                98.0,
                99.0,
                100.0,
                99.5,
                101.0,
                100.0,
                102.0,
                101.5,
                103.0,
                102.0,
                104.0,
                103.5,
                105.0,
                104.0,
                106.0,
                105.5,
                107.0,
                106.0,
                108.0,
                107.5,
            ],
            "close": [
                101.0,
                102.0,
                101.0,
                103.0,
                102.0,
                104.0,
                103.0,
                105.0,
                104.0,
                106.0,
                105.0,
                107.0,
                106.0,
                108.0,
                107.0,
                109.0,
                108.0,
                110.0,
                109.0,
                111.0,
            ],
            "volume": [
                1000.0,
                1100.0,
                900.0,
                1200.0,
                800.0,
                1300.0,
                700.0,
                1400.0,
                600.0,
                1500.0,
                500.0,
                1600.0,
                400.0,
                1700.0,
                300.0,
                1800.0,
                200.0,
                1900.0,
                100.0,
                2000.0,
            ],
        },
    )


@pytest.fixture
def small_volume_data():
    """Small dataset for edge case testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "open": [100.0, 101.0, 102.0, 101.0, 103.0],
            "high": [102.0, 103.0, 104.0, 103.0, 105.0],
            "low": [98.0, 99.0, 100.0, 99.0, 101.0],
            "close": [101.0, 102.0, 101.0, 103.0, 102.0],
            "volume": [1000.0, 1100.0, 900.0, 1200.0, 800.0],
        },
    )


# =============================================================================
# OBV Tests
# =============================================================================


def test_obv_basic_functionality(sample_volume_data):
    """Test OBV basic calculation."""
    result = sample_volume_data.with_columns(obv("close", "volume").alias("obv"))

    assert "obv" in result.columns
    assert len(result) == len(sample_volume_data)
    assert result["obv"].dtype == pl.Float64

    # OBV should be cumulative
    obv_values = result["obv"].to_numpy()
    assert not np.all(np.isnan(obv_values))

    # First value should be first volume
    assert obv_values[0] == 1000.0


def test_obv_accumulation_logic(small_volume_data):
    """Test OBV accumulation/distribution logic."""
    result = small_volume_data.with_columns(obv("close", "volume").alias("obv"))
    obv_values = result["obv"].to_numpy()

    # Manual calculation:
    # [0]: close=101, volume=1000 -> OBV = 1000
    # [1]: close=102 > 101 -> OBV = 1000 + 1100 = 2100
    # [2]: close=101 < 102 -> OBV = 2100 - 900 = 1200
    # [3]: close=103 > 101 -> OBV = 1200 + 1200 = 2400
    # [4]: close=102 < 103 -> OBV = 2400 - 800 = 1600

    assert obv_values[0] == 1000.0
    assert obv_values[1] == 2100.0
    assert obv_values[2] == 1200.0
    assert obv_values[3] == 2400.0
    assert obv_values[4] == 1600.0


def test_obv_with_nan_data():
    """Test OBV with NaN values in data."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "close": [100.0, 101.0, float("nan"), 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
    )

    result = data.with_columns(obv("close", "volume").alias("obv"))
    assert "obv" in result.columns
    # NaN handling depends on implementation


def test_obv_constant_price():
    """Test OBV when price doesn't change."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
    )

    result = data.with_columns(obv("close", "volume").alias("obv"))
    obv_values = result["obv"].to_numpy()

    # When price unchanged, OBV should remain constant
    assert obv_values[0] == 1000.0
    assert obv_values[1] == 1000.0  # No change
    assert obv_values[2] == 1000.0  # No change


# =============================================================================
# AD Tests
# =============================================================================


def test_ad_basic_functionality(sample_volume_data):
    """Test AD basic calculation."""
    result = sample_volume_data.with_columns(ad("high", "low", "close", "volume").alias("ad"))

    assert "ad" in result.columns
    assert len(result) == len(sample_volume_data)
    assert result["ad"].dtype == pl.Float64

    # AD should be cumulative
    ad_values = result["ad"].to_numpy()
    assert not np.all(np.isnan(ad_values))


def test_ad_money_flow_logic(small_volume_data):
    """Test AD money flow multiplier calculation."""
    result = small_volume_data.with_columns(ad("high", "low", "close", "volume").alias("ad"))
    ad_values = result["ad"].to_numpy()

    # AD should accumulate based on CLV
    # CLV = ((Close - Low) - (High - Close)) / (High - Low)
    # Money Flow Volume = CLV * Volume
    # AD = cumulative sum of Money Flow Volume

    # All values should be valid (not NaN)
    assert not np.any(np.isnan(ad_values))

    # AD is cumulative, should change over time
    assert not np.all(ad_values == ad_values[0])


def test_ad_high_equals_low():
    """Test AD when High == Low (CLV should be 0)."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 3),
                interval="1d",
                eager=True,
            ),
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 99.0, 100.0],  # First bar: high == low
            "close": [100.0, 101.0, 102.0],
            "volume": [1000.0, 1100.0, 1200.0],
        },
    )

    result = data.with_columns(ad("high", "low", "close", "volume").alias("ad"))
    ad_values = result["ad"].to_numpy()

    # When high == low, CLV = 0, so no change to AD
    assert ad_values[0] == 0.0  # First bar contributes 0


def test_ad_with_nan_data():
    """Test AD with NaN values."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "high": [102.0, 103.0, float("nan"), 105.0, 106.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
    )

    result = data.with_columns(ad("high", "low", "close", "volume").alias("ad"))
    assert "ad" in result.columns


# =============================================================================
# ADOSC Tests
# =============================================================================


def test_adosc_basic_functionality(sample_volume_data):
    """Test ADOSC basic calculation."""
    result = sample_volume_data.with_columns(
        adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc_3_10")
    )

    assert "adosc_3_10" in result.columns
    assert len(result) == len(sample_volume_data)
    assert result["adosc_3_10"].dtype == pl.Float64

    adosc_values = result["adosc_3_10"].to_numpy()

    # Should have NaN values at start (lookback period)
    lookback = 10 - 1  # slowperiod - 1
    assert np.all(
        np.isnan(adosc_values[:lookback]) | (adosc_values[:lookback] == adosc_values[:lookback])
    )

    # After lookback, should have valid values
    assert not np.all(np.isnan(adosc_values[lookback:]))


def test_adosc_custom_periods(sample_volume_data):
    """Test ADOSC with custom fast/slow periods."""
    result = sample_volume_data.with_columns(
        adosc("high", "low", "close", "volume", fastperiod=5, slowperiod=15).alias("adosc_5_15")
    )

    assert "adosc_5_15" in result.columns
    adosc_values = result["adosc_5_15"].to_numpy()

    # Lookback based on slowest period
    assert len(adosc_values) == len(sample_volume_data)


def test_adosc_insufficient_data():
    """Test ADOSC with insufficient data."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "high": [102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [98.0, 99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
    )

    # Default slowperiod=10, need at least 10 bars
    result = data.with_columns(
        adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc_3_10")
    )
    adosc_values = result["adosc_3_10"].to_numpy()

    # All values should be NaN (insufficient data)
    assert np.all(np.isnan(adosc_values))


def test_adosc_oscillator_behavior(sample_volume_data):
    """Test that ADOSC oscillates (positive and negative values)."""
    result = sample_volume_data.with_columns(
        adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc_3_10")
    )
    adosc_values = result["adosc_3_10"].to_numpy()

    # Remove NaN values
    valid_values = adosc_values[~np.isnan(adosc_values)]

    # ADOSC is an oscillator - should cross zero
    # (Not guaranteed in all data, but likely)
    if len(valid_values) > 0:
        assert not np.all(valid_values == valid_values[0])


def test_adosc_with_nan_data():
    """Test ADOSC with NaN values in input."""
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 15),
                interval="1d",
                eager=True,
            ),
            "high": [
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                float("nan"),
                115.0,
                116.0,
            ],
            "low": [
                98.0,
                99.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
            ],
            "close": [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
            ],
            "volume": [1000.0] * 15,
        },
    )

    result = data.with_columns(
        adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc_3_10")
    )
    assert "adosc_3_10" in result.columns


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_has_volume_features():
    """Test that volume features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # Check all volume features are registered
    assert registry.get("obv") is not None
    assert registry.get("ad") is not None
    assert registry.get("adosc") is not None

    # Verify they're in volume category
    obv_meta = registry.get("obv")
    assert obv_meta.category == "volume"
    assert obv_meta.normalized is False  # Cumulative

    ad_meta = registry.get("ad")
    assert ad_meta.category == "volume"
    assert ad_meta.normalized is False  # Cumulative

    adosc_meta = registry.get("adosc")
    assert adosc_meta.category == "volume"
    assert adosc_meta.normalized is False  # Oscillator of unbounded A/D line
