"""Validation tests for QuantLab-TA against reference implementations.

This module tests our Polars-based technical indicators against established
libraries like TA-Lib to ensure mathematical accuracy.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

# Try to import reference libraries
try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

import importlib.util

HAS_PANDAS = importlib.util.find_spec("pandas") is not None

from ml4t.engineer.features.momentum import macd, macd_signal, rsi, stochastic
from ml4t.engineer.features.trend import ema, sma
from ml4t.engineer.features.utils.arithmetic import returns
from ml4t.engineer.features.utils.helpers import add_indicators
from ml4t.engineer.features.volatility import atr, bollinger_bands


def create_test_price_data():
    """Create realistic test price data for validation."""
    np.random.seed(42)  # For reproducible results

    # Generate 250 days of price data (1 trading year)
    n_days = 250

    # Create realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility

    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate OHLC from close prices
    closes = np.array(prices)

    # High/Low with some spread
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, n_days)))

    # Opens (previous close + gap)
    opens = np.roll(closes, 1) * (1 + np.random.normal(0, 0.005, n_days))
    opens[0] = closes[0]

    # Volume
    volumes = np.random.lognormal(12, 0.5, n_days).astype(int)

    return pl.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
    )


class TestOverlapStudies:
    """Test overlap study indicators against reference implementations."""

    def setup_method(self):
        """Set up test data."""
        self.data = create_test_price_data()
        self.closes = self.data["close"].to_numpy()
        self.highs = self.data["high"].to_numpy()
        self.lows = self.data["low"].to_numpy()

    @pytest.mark.validation
    def test_sma_vs_talib(self):
        """Test SMA against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 20

        # Our implementation
        result = self.data.with_columns([sma("close", window).alias("sma")])
        our_sma = result["sma"].to_numpy()

        # TA-Lib reference
        talib_sma = talib.SMA(self.closes, timeperiod=window)

        # Compare (skip NaN values)
        mask = ~np.isnan(talib_sma)
        np.testing.assert_allclose(
            our_sma[mask],
            talib_sma[mask],
            rtol=1e-10,
            atol=1e-10,
            err_msg="SMA values don't match TA-Lib",
        )

    @pytest.mark.validation
    def test_ema_vs_talib(self):
        """Test EMA against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 20

        # Our implementation
        result = self.data.with_columns([ema("close", window).alias("ema")])
        our_ema = result["ema"].to_numpy()

        # TA-Lib reference
        talib_ema = talib.EMA(self.closes, timeperiod=window)

        # Compare (skip NaN values and allow for initialization differences)
        mask = ~np.isnan(talib_ema)
        # Skip first few values after initialization to focus on steady-state behavior
        steady_state_mask = mask.copy()
        first_valid_idx = np.where(mask)[0][0]
        steady_state_mask[: first_valid_idx + 10] = False  # Skip first 10 after initialization

        if np.any(steady_state_mask):
            np.testing.assert_allclose(
                our_ema[steady_state_mask],
                talib_ema[steady_state_mask],
                rtol=1e-2,  # Allow 1% difference due to different initialization methods
                err_msg="EMA steady-state values don't match TA-Lib within tolerance",
            )

    @pytest.mark.validation
    def test_bollinger_bands_vs_talib(self):
        """Test Bollinger Bands against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 20
        std_dev = 2.0

        # Our implementation
        result = self.data.with_columns(
            [bollinger_bands("close", window, std_dev).alias("bb")],
        ).with_columns(
            [
                pl.col("bb").struct.field("upper").alias("bb_upper"),
                pl.col("bb").struct.field("middle").alias("bb_middle"),
                pl.col("bb").struct.field("lower").alias("bb_lower"),
            ],
        )

        our_upper = result["bb_upper"].to_numpy()
        our_middle = result["bb_middle"].to_numpy()
        our_lower = result["bb_lower"].to_numpy()

        # TA-Lib reference
        talib_upper, talib_middle, talib_lower = talib.BBANDS(
            self.closes,
            timeperiod=window,
            nbdevup=std_dev,
            nbdevdn=std_dev,
        )

        # Compare each band
        mask = ~np.isnan(talib_middle)
        if np.any(mask):
            np.testing.assert_allclose(
                our_middle[mask],
                talib_middle[mask],
                rtol=1e-10,
                err_msg="Bollinger middle band doesn't match TA-Lib",
            )

            np.testing.assert_allclose(
                our_upper[mask],
                talib_upper[mask],
                rtol=1e-10,
                err_msg="Bollinger upper band doesn't match TA-Lib",
            )

            np.testing.assert_allclose(
                our_lower[mask],
                talib_lower[mask],
                rtol=1e-10,
                err_msg="Bollinger lower band doesn't match TA-Lib",
            )


class TestMomentumIndicators:
    """Test momentum indicators against reference implementations."""

    def setup_method(self):
        """Set up test data."""
        self.data = create_test_price_data()
        self.closes = self.data["close"].to_numpy()
        self.highs = self.data["high"].to_numpy()
        self.lows = self.data["low"].to_numpy()

    @pytest.mark.validation
    def test_rsi_vs_talib(self):
        """Test RSI against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 14

        # Our implementation
        result = self.data.with_columns([rsi("close", window).alias("rsi")])
        our_rsi = result["rsi"].to_numpy()

        # TA-Lib reference
        talib_rsi = talib.RSI(self.closes, timeperiod=window)

        # Compare (skip NaN values and first few values due to initialization differences)
        mask = ~np.isnan(talib_rsi)
        # Skip first 30 values to account for initialization differences
        mask[:30] = False

        if np.any(mask):
            np.testing.assert_allclose(
                our_rsi[mask],
                talib_rsi[mask],
                rtol=1e-2,  # Allow 1% difference due to different smoothing initialization
                err_msg="RSI values don't match TA-Lib within tolerance",
            )

    @pytest.mark.validation
    def test_macd_vs_talib(self):
        """Test MACD against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        fast = 12
        slow = 26
        signal = 9

        # Our implementation
        result = self.data.with_columns(
            [
                macd("close", fast, slow).alias("macd"),
                macd_signal("close", fast, slow, signal).alias("macd_signal"),
            ],
        )

        our_macd = result["macd"].to_numpy()
        our_signal = result["macd_signal"].to_numpy()

        # TA-Lib reference
        talib_macd, talib_signal, _ = talib.MACD(
            self.closes,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal,
        )

        # Compare MACD line (skip NaN values)
        mask = ~np.isnan(talib_macd)
        if np.any(mask):
            np.testing.assert_allclose(
                our_macd[mask],
                talib_macd[mask],
                rtol=1e-6,
                err_msg="MACD line doesn't match TA-Lib",
            )

        # Compare signal line
        # Note: TA-Lib's signal line calculation is affected by the unstable period
        # which causes initialization differences. We'll check that the values
        # converge after the initialization period.
        talib_mask = ~np.isnan(talib_signal)
        our_mask = ~np.isnan(our_signal)

        # Skip the first 20 values after signal starts to allow for initialization differences
        if np.any(our_mask):
            first_our = np.where(our_mask)[0][0]
            our_mask[: first_our + 20] = False

        # Find the overlap where both have values
        overlap_mask = talib_mask & our_mask

        if np.any(overlap_mask):
            np.testing.assert_allclose(
                our_signal[overlap_mask],
                talib_signal[overlap_mask],
                rtol=1e-2,  # Allow 1% difference due to initialization
                err_msg="MACD signal line doesn't converge with TA-Lib after initialization",
            )

    @pytest.mark.validation
    def test_stochastic_vs_talib(self):
        """Test Stochastic Oscillator against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 14

        # Our implementation
        result = self.data.with_columns(
            [
                stochastic(
                    "high",
                    "low",
                    "close",
                    fastk_period=window,
                    slowk_period=1,
                    slowd_period=1,
                ).alias("stoch"),
            ],
        )
        our_stoch = result["stoch"].to_numpy()

        # TA-Lib reference
        talib_stoch_k, _ = talib.STOCH(
            self.highs,
            self.lows,
            self.closes,
            fastk_period=window,
            slowk_period=1,
            slowd_period=1,
        )

        # Compare (skip NaN values)
        mask = ~np.isnan(talib_stoch_k)
        if np.any(mask):
            np.testing.assert_allclose(
                our_stoch[mask],
                talib_stoch_k[mask],
                rtol=1e-10,
                err_msg="Stochastic %K doesn't match TA-Lib",
            )


class TestVolatilityIndicators:
    """Test volatility indicators against reference implementations."""

    def setup_method(self):
        """Set up test data."""
        self.data = create_test_price_data()
        self.closes = self.data["close"].to_numpy()
        self.highs = self.data["high"].to_numpy()
        self.lows = self.data["low"].to_numpy()

    @pytest.mark.validation
    def test_atr_vs_talib(self):
        """Test ATR against TA-Lib implementation."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        window = 14

        # Our implementation
        result = self.data.with_columns(
            [atr("high", "low", "close", window).alias("atr")],
        )
        our_atr = result["atr"].to_numpy()

        # TA-Lib reference
        talib_atr = talib.ATR(self.highs, self.lows, self.closes, timeperiod=window)

        # Compare (skip NaN values and allow for initialization differences)
        mask = ~np.isnan(talib_atr)
        mask[:30] = False  # Skip first 30 values

        if np.any(mask):
            np.testing.assert_allclose(
                our_atr[mask],
                talib_atr[mask],
                rtol=1e-2,  # Allow 1% difference due to Wilder's smoothing initialization
                err_msg="ATR values don't match TA-Lib within tolerance",
            )


class TestArithmeticOperations:
    """Test arithmetic operations."""

    def setup_method(self):
        """Set up test data."""
        self.data = create_test_price_data()
        self.closes = self.data["close"].to_numpy()

    def test_returns_calculation(self):
        """Test returns calculation against manual computation."""
        # Our implementation
        result = self.data.with_columns([returns("close", 1).alias("returns")])
        our_returns = result["returns"].to_numpy()

        # Manual calculation
        manual_returns = np.full(len(self.closes), np.nan)
        manual_returns[1:] = (self.closes[1:] / self.closes[:-1]) - 1

        # Compare (skip NaN values)
        mask = ~np.isnan(manual_returns)
        np.testing.assert_allclose(
            our_returns[mask],
            manual_returns[mask],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Returns calculation is incorrect",
        )

    def test_multiperiod_returns(self):
        """Test multi-period returns calculation."""
        periods = 5

        # Our implementation
        result = self.data.with_columns(
            [returns("close", periods).alias("returns")],
        )
        our_returns = result["returns"].to_numpy()

        # Manual calculation
        manual_returns = np.full(len(self.closes), np.nan)
        manual_returns[periods:] = (self.closes[periods:] / self.closes[:-periods]) - 1

        # Compare (skip NaN values)
        mask = ~np.isnan(manual_returns)
        np.testing.assert_allclose(
            our_returns[mask],
            manual_returns[mask],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Multi-period returns calculation is incorrect",
        )


class TestUtilities:
    """Test utility functions."""

    def setup_method(self):
        """Set up test data."""
        self.data = create_test_price_data()

    def test_add_indicators(self):
        """Test batch indicator addition."""
        indicators = {
            "sma_20": sma("close", 20),
            "rsi_14": rsi("close", 14),
            "macd": macd("close", 12, 26),
        }

        result = add_indicators(self.data, indicators)

        # Check that all indicator columns were added
        for name in indicators:
            assert name in result.columns, f"Column {name} not found in result"

        # Check that data shape is correct
        assert result.shape[0] == self.data.shape[0]
        assert result.shape[1] == self.data.shape[1] + len(indicators)
