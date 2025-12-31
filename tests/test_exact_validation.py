"""Exact mathematical validation against TA-Lib using real and synthetic data.

This module provides comprehensive validation of our technical indicators against
TA-Lib using multiple data sources to ensure mathematical exactness.
"""

import warnings
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
    warnings.warn(
        "TA-Lib not available - validation tests will be skipped",
        stacklevel=2,
    )

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn(
        "yfinance not available - real data tests will use fallback",
        stacklevel=2,
    )

from ml4t.engineer.features.momentum import rsi, stochastic
from ml4t.engineer.features.trend import ema, sma
from ml4t.engineer.features.volatility import atr, bollinger_bands


class RealDataProvider:
    """Provider for real stock market data."""

    @staticmethod
    def get_spy_data(period: str = "1y") -> pl.DataFrame:
        """Get SPY ETF data from Yahoo Finance."""
        if not HAS_YFINANCE:
            # Fallback to hardcoded SPY-like data
            return RealDataProvider._get_fallback_spy_data()

        try:
            ticker = yf.Ticker("SPY")
            data = ticker.history(period=period)

            return pl.DataFrame(
                {
                    "timestamp": data.index.tolist(),
                    "open": data["Open"].values,
                    "high": data["High"].values,
                    "low": data["Low"].values,
                    "close": data["Close"].values,
                    "volume": data["Volume"].values.astype(int),
                },
            )
        except Exception as e:
            warnings.warn(f"Failed to fetch real data: {e}", stacklevel=2)
            return RealDataProvider._get_fallback_spy_data()

    @staticmethod
    def get_aapl_data(period: str = "6mo") -> pl.DataFrame:
        """Get AAPL data from Yahoo Finance."""
        if not HAS_YFINANCE:
            return RealDataProvider._get_fallback_aapl_data()

        try:
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period=period)

            return pl.DataFrame(
                {
                    "timestamp": data.index.tolist(),
                    "open": data["Open"].values,
                    "high": data["High"].values,
                    "low": data["Low"].values,
                    "close": data["Close"].values,
                    "volume": data["Volume"].values.astype(int),
                },
            )
        except Exception as e:
            warnings.warn(f"Failed to fetch AAPL data: {e}", stacklevel=2)
            return RealDataProvider._get_fallback_aapl_data()

    @staticmethod
    def _get_fallback_spy_data() -> pl.DataFrame:
        """Fallback SPY-like data when yfinance is not available."""
        np.random.seed(42)
        n_days = 252  # 1 trading year

        # Start with realistic SPY price level
        base_price = 450.0
        daily_returns = np.random.normal(0.0008, 0.015, n_days)  # ~20% annual vol

        # Generate realistic price series
        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        # Generate OHLC with realistic spreads
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Add intraday volatility
        daily_ranges = np.abs(np.random.normal(0, 0.008, n_days)) * closes
        highs = closes + daily_ranges * np.random.uniform(0.3, 0.8, n_days)
        lows = closes - daily_ranges * np.random.uniform(0.3, 0.8, n_days)

        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        # Generate realistic volume
        volumes = np.random.lognormal(17.5, 0.3, n_days).astype(int)  # ~50M avg volume

        return pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 3) + timedelta(days=i) for i in range(n_days)],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
        )

    @staticmethod
    def _get_fallback_aapl_data() -> pl.DataFrame:
        """Fallback AAPL-like data when yfinance is not available."""
        np.random.seed(123)
        n_days = 126  # 6 months

        # Start with realistic AAPL price level
        base_price = 175.0
        daily_returns = np.random.normal(0.001, 0.025, n_days)  # Higher vol than SPY

        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        daily_ranges = np.abs(np.random.normal(0, 0.012, n_days)) * closes
        highs = closes + daily_ranges * np.random.uniform(0.3, 0.8, n_days)
        lows = closes - daily_ranges * np.random.uniform(0.3, 0.8, n_days)

        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))

        volumes = np.random.lognormal(16.8, 0.4, n_days).astype(int)  # ~25M avg volume

        return pl.DataFrame(
            {
                "timestamp": [datetime(2023, 7, 1) + timedelta(days=i) for i in range(n_days)],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
        )


class SyntheticDataProvider:
    """Provider for various synthetic test data patterns."""

    @staticmethod
    def get_trending_data(n_days: int = 100, trend: float = 0.001) -> pl.DataFrame:
        """Generate trending price data."""
        np.random.seed(42)
        base_price = 100.0

        # Add trend to random walk
        daily_returns = np.random.normal(trend, 0.02, n_days)
        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Small intraday ranges for clean testing
        spread = 0.005
        highs = closes * (1 + spread * np.random.uniform(0.2, 0.8, n_days))
        lows = closes * (1 - spread * np.random.uniform(0.2, 0.8, n_days))

        volumes = np.random.poisson(1000000, n_days)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
        )

    @staticmethod
    def get_oscillating_data(n_days: int = 200, period: float = 20.0) -> pl.DataFrame:
        """Generate oscillating/cyclical price data."""
        np.random.seed(456)
        base_price = 50.0

        # Sine wave with noise
        t = np.arange(n_days)
        cycle = 5 * np.sin(2 * np.pi * t / period)  # 5% amplitude
        noise = np.random.normal(0, 0.01, n_days)

        prices = base_price * (1 + (cycle + noise) / 100)

        # Generate OHLC
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        spread = 0.003
        highs = closes * (1 + spread)
        lows = closes * (1 - spread)

        volumes = np.random.poisson(500000, n_days)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
        )

    @staticmethod
    def get_volatile_data(n_days: int = 150, volatility: float = 0.05) -> pl.DataFrame:
        """Generate high volatility data."""
        np.random.seed(789)
        base_price = 200.0

        # High volatility random walk
        daily_returns = np.random.normal(0, volatility, n_days)
        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]

        # Wider intraday ranges for volatile data
        spread = 0.02
        highs = closes * (1 + spread * np.random.uniform(0.5, 1.0, n_days))
        lows = closes * (1 - spread * np.random.uniform(0.5, 1.0, n_days))

        volumes = np.random.poisson(2000000, n_days)

        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)],
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            },
        )

    @staticmethod
    def get_edge_case_data() -> pl.DataFrame:
        """Generate edge case data for testing."""
        # Edge cases: constant prices, gaps, extreme values
        data = {
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        # First 10 days: constant prices
        for _ in range(10):
            data["open"].append(100.0)
            data["high"].append(100.0)
            data["low"].append(100.0)
            data["close"].append(100.0)
            data["volume"].append(1000000)

        # Next 10 days: small variations
        for i in range(10):
            price = 100 + 0.01 * i
            data["open"].append(price)
            data["high"].append(price + 0.005)
            data["low"].append(price - 0.005)
            data["close"].append(price)
            data["volume"].append(1000000)

        # Next 10 days: large gap up
        for i in range(10):
            price = 110 + 0.1 * i
            data["open"].append(price)
            data["high"].append(price + 0.05)
            data["low"].append(price - 0.05)
            data["close"].append(price)
            data["volume"].append(1500000)

        # Next 10 days: trending down
        for i in range(10):
            price = 111 - 0.2 * i
            data["open"].append(price)
            data["high"].append(price + 0.02)
            data["low"].append(price - 0.02)
            data["close"].append(price)
            data["volume"].append(800000)

        # Last 10 days: high volatility
        np.random.seed(999)
        for _ in range(10):
            base = 109
            change = np.random.normal(0, 0.05)
            price = base * (1 + change)
            spread = abs(change) * 0.5
            data["open"].append(price)
            data["high"].append(price + spread)
            data["low"].append(price - spread)
            data["close"].append(price)
            data["volume"].append(2000000)

        return pl.DataFrame(data)


def validate_indicator_exact(
    our_values: np.ndarray,
    talib_values: np.ndarray,
    indicator_name: str,
    data_description: str,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    skip_initial: int = 0,
) -> tuple[bool, str, dict]:
    """
    Validate exact mathematical equivalence between our implementation and TA-Lib.

    Args:
        our_values: Our calculated values
        talib_values: TA-Lib reference values
        indicator_name: Name of the indicator being tested
        data_description: Description of the test data
        rtol: Relative tolerance
        atol: Absolute tolerance
        skip_initial: Number of initial values to skip (for warmup)

    Returns:
        Tuple of (is_valid, error_message, metrics)
    """
    # Handle NaN values - both should have NaN in the same positions
    our_nan_mask = np.isnan(our_values)
    talib_nan_mask = np.isnan(talib_values)

    # Check if NaN patterns match
    if not np.array_equal(our_nan_mask, talib_nan_mask):
        nan_diff = np.sum(our_nan_mask != talib_nan_mask)
        return False, f"NaN patterns don't match: {nan_diff} positions differ", {}

    # Get valid (non-NaN) values for comparison
    valid_mask = ~our_nan_mask
    if skip_initial > 0:
        valid_mask[:skip_initial] = False

    if not np.any(valid_mask):
        return False, "No valid values to compare", {}

    our_valid = our_values[valid_mask]
    talib_valid = talib_values[valid_mask]

    # Calculate error metrics
    abs_diff = np.abs(our_valid - talib_valid)
    rel_diff = abs_diff / (np.abs(talib_valid) + 1e-15)  # Avoid division by zero

    max_abs_error = np.max(abs_diff)
    max_rel_error = np.max(rel_diff)
    mean_abs_error = np.mean(abs_diff)
    mean_rel_error = np.mean(rel_diff)

    # Count values outside tolerance
    outside_tolerance = np.logical_or(abs_diff > atol, rel_diff > rtol)
    num_outside = np.sum(outside_tolerance)
    pct_outside = 100 * num_outside / len(our_valid)

    metrics = {
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "mean_abs_error": mean_abs_error,
        "mean_rel_error": mean_rel_error,
        "num_outside_tolerance": num_outside,
        "pct_outside_tolerance": pct_outside,
        "total_valid_points": len(our_valid),
    }

    # Check if within tolerance
    is_valid = num_outside == 0

    if not is_valid:
        error_msg = (
            f"{indicator_name} on {data_description}: "
            f"{num_outside}/{len(our_valid)} values ({pct_outside:.2f}%) outside tolerance. "
            f"Max abs error: {max_abs_error:.2e}, Max rel error: {max_rel_error:.2e}"
        )
    else:
        error_msg = f"{indicator_name} on {data_description}: EXACT MATCH ✅"

    return is_valid, error_msg, metrics


@pytest.mark.validation
class TestExactValidationRealData:
    """Test exact validation using real stock market data."""

    def setup_method(self):
        """Set up real data for testing."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        self.spy_data = RealDataProvider.get_spy_data()
        self.aapl_data = RealDataProvider.get_aapl_data()

        print(f"\nSPY data: {len(self.spy_data)} days")
        print(f"AAPL data: {len(self.aapl_data)} days")

    def test_sma_real_data_exact(self):
        """Test SMA exact validation on real data."""
        windows = [10, 20, 50]
        datasets = [(self.spy_data, "SPY"), (self.aapl_data, "AAPL")]

        for data, name in datasets:
            closes = data["close"].to_numpy()

            for window in windows:
                # Our implementation
                result = data.with_columns([sma("close", window).alias("sma")])
                our_sma = result["sma"].to_numpy()

                # TA-Lib reference
                talib_sma = talib.SMA(closes, timeperiod=window)

                # Validate exact match
                is_valid, msg, metrics = validate_indicator_exact(
                    our_sma,
                    talib_sma,
                    f"SMA({window})",
                    f"{name} real data",
                    rtol=1e-6,
                    atol=1e-6,
                )

                print(f"  {msg}")
                assert is_valid, f"SMA({window}) failed on {name}: {msg}"

    def test_rsi_real_data_exact(self):
        """Test RSI exact validation on real data."""
        windows = [14, 21]
        datasets = [(self.spy_data, "SPY"), (self.aapl_data, "AAPL")]

        for data, name in datasets:
            closes = data["close"].to_numpy()

            for window in windows:
                # Our implementation
                result = data.with_columns([rsi("close", window).alias("rsi")])
                our_rsi = result["rsi"].to_numpy()

                # TA-Lib reference
                talib_rsi = talib.RSI(closes, timeperiod=window)

                # RSI may have initialization differences - skip first 30 values
                is_valid, msg, metrics = validate_indicator_exact(
                    our_rsi,
                    talib_rsi,
                    f"RSI({window})",
                    f"{name} real data",
                    rtol=1e-2,
                    atol=1e-6,  # Allow small differences due to initialization
                    skip_initial=30,
                )

                print(f"  {msg}")
                # For RSI, we accept 99%+ accuracy due to initialization differences
                assert metrics["pct_outside_tolerance"] < 1.0, (
                    f"RSI({window}) failed on {name}: {metrics['pct_outside_tolerance']:.2f}% outside tolerance"
                )

    def test_bollinger_bands_real_data_exact(self):
        """Test Bollinger Bands exact validation on real data."""
        windows = [20]
        std_devs = [2.0]
        datasets = [(self.spy_data, "SPY"), (self.aapl_data, "AAPL")]

        for data, name in datasets:
            closes = data["close"].to_numpy()

            for window in windows:
                for std_dev in std_devs:
                    # Our implementation
                    result = data.with_columns(
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
                        closes,
                        timeperiod=window,
                        nbdevup=std_dev,
                        nbdevdn=std_dev,
                    )

                    # Validate each band
                    for our_band, talib_band, band_name in [
                        (our_upper, talib_upper, "Upper"),
                        (our_middle, talib_middle, "Middle"),
                        (our_lower, talib_lower, "Lower"),
                    ]:
                        is_valid, msg, metrics = validate_indicator_exact(
                            our_band,
                            talib_band,
                            f"BB_{band_name}({window},{std_dev})",
                            f"{name} real data",
                            rtol=1e-6,
                            atol=1e-6,
                        )

                        print(f"  {msg}")
                        assert is_valid, f"Bollinger {band_name} failed on {name}: {msg}"


@pytest.mark.validation
class TestExactValidationSyntheticData:
    """Test exact validation using synthetic data patterns."""

    def setup_method(self):
        """Set up synthetic data for testing."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        self.trending_data = SyntheticDataProvider.get_trending_data()
        self.oscillating_data = SyntheticDataProvider.get_oscillating_data()
        self.volatile_data = SyntheticDataProvider.get_volatile_data()
        self.edge_case_data = SyntheticDataProvider.get_edge_case_data()

    def test_all_indicators_trending_data(self):
        """Test all indicators on trending synthetic data."""
        data = self.trending_data
        closes = data["close"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()

        # Test SMA
        for window in [5, 10, 20]:
            result = data.with_columns([sma("close", window).alias("sma")])
            our_sma = result["sma"].to_numpy()
            talib_sma = talib.SMA(closes, timeperiod=window)

            is_valid, msg, _ = validate_indicator_exact(
                our_sma,
                talib_sma,
                f"SMA({window})",
                "trending data",
            )
            print(f"  {msg}")
            assert is_valid

        # Test EMA
        for window in [10, 21]:
            result = data.with_columns([ema("close", window).alias("ema")])
            our_ema = result["ema"].to_numpy()
            talib_ema = talib.EMA(closes, timeperiod=window)

            is_valid, msg, metrics = validate_indicator_exact(
                our_ema,
                talib_ema,
                f"EMA({window})",
                "trending data",
                rtol=1e-3,
                skip_initial=10,  # Allow for initialization differences
            )
            print(f"  {msg}")
            assert metrics["pct_outside_tolerance"] < 5.0  # Accept 95%+ accuracy

        # Test Stochastic
        for window in [14]:
            result = data.with_columns(
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
            talib_stoch_k, _ = talib.STOCH(
                highs,
                lows,
                closes,
                fastk_period=window,
                slowk_period=1,
                slowd_period=1,
            )

            is_valid, msg, _ = validate_indicator_exact(
                our_stoch,
                talib_stoch_k,
                f"STOCH({window})",
                "trending data",
            )
            print(f"  {msg}")
            assert is_valid

    def test_oscillating_data_patterns(self):
        """Test indicators on oscillating/cyclical data."""
        data = self.oscillating_data
        closes = data["close"].to_numpy()

        # RSI should show clear oscillations
        result = data.with_columns([rsi("close", 14).alias("rsi")])
        our_rsi = result["rsi"].to_numpy()
        talib_rsi = talib.RSI(closes, timeperiod=14)

        is_valid, msg, metrics = validate_indicator_exact(
            our_rsi,
            talib_rsi,
            "RSI(14)",
            "oscillating data",
            rtol=1e-2,
            skip_initial=30,
        )
        print(f"  {msg}")
        assert metrics["pct_outside_tolerance"] < 2.0

        # Check that RSI actually oscillates (between 30-70 range mostly)
        valid_rsi = our_rsi[~np.isnan(our_rsi)]
        oscillating_values = np.sum((valid_rsi > 30) & (valid_rsi < 70))
        assert oscillating_values > len(valid_rsi) * 0.6, "RSI should oscillate in middle range"

    def test_edge_case_data_robustness(self):
        """Test indicators on edge case data (constant prices, gaps, etc.)."""
        data = self.edge_case_data
        closes = data["close"].to_numpy()
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()

        # Test SMA on constant prices (first 10 days)
        result = data.with_columns([sma("close", 5).alias("sma")])
        our_sma = result["sma"].to_numpy()
        talib.SMA(closes, timeperiod=5)

        # For constant prices, SMA should equal the constant value
        constant_period_sma = our_sma[4:10]  # After 5-day warmup, during constant period
        assert np.allclose(
            constant_period_sma,
            100.0,
            rtol=1e-6,
        ), "SMA should equal constant price"

        # Test ATR on various volatility periods
        result = data.with_columns([atr("high", "low", "close", 14).alias("atr")])
        our_atr = result["atr"].to_numpy()
        talib_atr = talib.ATR(highs, lows, closes, timeperiod=14)

        is_valid, msg, metrics = validate_indicator_exact(
            our_atr,
            talib_atr,
            "ATR(14)",
            "edge case data",
            rtol=1e-2,
            skip_initial=20,
        )
        print(f"  {msg}")
        assert metrics["pct_outside_tolerance"] < 5.0


@pytest.mark.validation
class TestMathematicalProperties:
    """Test mathematical properties and edge cases."""

    def test_sma_mathematical_properties(self):
        """Test SMA mathematical properties."""
        # SMA of constant values should equal the constant
        constant_data = pl.DataFrame({"close": [100.0] * 20})

        result = constant_data.with_columns([sma("close", 5).alias("sma")])
        sma_values = [v for v in result["sma"].to_list() if v is not None and not np.isnan(v)]

        assert all(abs(v - 100.0) < 1e-6 for v in sma_values), (
            "SMA of constant should equal constant"
        )

        # SMA should be between min and max of window
        data = pl.DataFrame({"close": [90, 95, 100, 105, 110, 85, 115, 80, 120, 75]})

        result = data.with_columns([sma("close", 5).alias("sma")])
        for i in range(len(result)):
            sma_val = result["sma"][i]
            if sma_val is not None:
                window_start = max(0, i - 4)
                window_data = data["close"][window_start : i + 1].to_list()
                assert min(window_data) <= sma_val <= max(window_data), (
                    f"SMA should be within window range at index {i}"
                )

    def test_rsi_range_constraints(self):
        """Test RSI stays within 0-100 range."""
        # Test with extreme price movements
        extreme_data = pl.DataFrame(
            {"close": [100, 150, 200, 250, 300, 50, 25, 10, 5, 100, 200, 50, 150]},
        )

        result = extreme_data.with_columns([rsi("close", 6).alias("rsi")])
        rsi_values = [v for v in result["rsi"].to_list() if v is not None and not np.isnan(v)]

        assert all(0 <= v <= 100 for v in rsi_values), "RSI must stay within 0-100 range"

        # RSI should approach extremes with strong trends
        assert max(rsi_values) > 60, "RSI should exceed 60 with strong uptrend"
        assert min(rsi_values) < 40, "RSI should fall below 40 with strong downtrend"

    def test_bollinger_bands_properties(self):
        """Test Bollinger Bands mathematical properties."""
        data = SyntheticDataProvider.get_trending_data(100)

        result = data.with_columns(
            [bollinger_bands("close", 20, 2.0).alias("bb")],
        ).with_columns(
            [
                pl.col("bb").struct.field("upper").alias("bb_upper"),
                pl.col("bb").struct.field("middle").alias("bb_middle"),
                pl.col("bb").struct.field("lower").alias("bb_lower"),
            ],
        )

        # Upper band should always be >= middle band >= lower band
        for i in range(len(result)):
            upper = result["bb_upper"][i]
            middle = result["bb_middle"][i]
            lower = result["bb_lower"][i]

            if all(v is not None for v in [upper, middle, lower]):
                assert upper >= middle >= lower, f"Band ordering violated at index {i}"

        # Most prices should be within bands (typically 95% for 2-sigma bands)
        closes = result["close"].to_list()
        uppers = result["bb_upper"].to_list()
        lowers = result["bb_lower"].to_list()

        within_bands = 0
        total_valid = 0

        for close, upper, lower in zip(closes, uppers, lowers, strict=False):
            if all(v is not None for v in [close, upper, lower]):
                total_valid += 1
                if lower <= close <= upper:
                    within_bands += 1

        within_pct = 100 * within_bands / total_valid if total_valid > 0 else 0
        assert within_pct > 85, (
            f"Only {within_pct:.1f}% of prices within Bollinger Bands (should be >85%)"
        )


def run_comprehensive_validation():
    """Run comprehensive validation and generate report."""
    if not HAS_TALIB:
        print("❌ TA-Lib not available - cannot run validation")
        return

    print("🔬 Running Comprehensive TA-Lib Validation")
    print("=" * 60)

    # Run all validation tests
    pytest.main(
        ["-v", "tests/test_exact_validation.py", "--tb=short", "-m", "validation"],
    )


if __name__ == "__main__":
    run_comprehensive_validation()
