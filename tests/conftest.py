"""Shared test fixtures for all tests."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.core import registry as registry_module


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test to prevent memory leaks."""
    yield
    # Close all figures after test completes
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass  # matplotlib not installed


@pytest.fixture(scope="session")
def crypto_data():
    """
    Load actual crypto data for performance testing.

    Uses real BTC spot data from the parent directory.
    Returns 100K rows for consistent performance testing.
    """
    # Try to find crypto data relative to project root
    # Support both development and CI environments
    possible_paths = [
        Path.cwd() / "data" / "crypto" / "spot" / "BTC.parquet",
        Path.cwd().parent / "data" / "crypto" / "spot" / "BTC.parquet",
        Path.home() / "quantlab" / "data" / "crypto" / "spot" / "BTC.parquet",
        Path("/tmp") / "test_data" / "BTC.parquet",  # CI environment
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if not data_path:
        # Try ETH as fallback
        for base_path in [p.parent for p in possible_paths if p.parent.exists()]:
            eth_path = base_path / "ETH.parquet"
            if eth_path.exists():
                data_path = eth_path
                break

    if not data_path:
        # If no data files exist, generate synthetic data as fallback
        print("Warning: No crypto data files found, using synthetic data")
        return _generate_synthetic_crypto_data()

    # Load the data
    df = pl.read_parquet(data_path)

    # Ensure we have the expected columns (rename if necessary)
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]

    # Common column mappings
    col_mappings = {
        "date": "timestamp",
        "datetime": "timestamp",
        "date_time": "timestamp",
        "time": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vol": "volume",
    }

    # Rename columns if needed
    for old_col, new_col in col_mappings.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename({old_col: new_col})

    # Select only the columns we need
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df.select(available_cols)

    # Ensure volume is float64 for TA-Lib compatibility
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Float64))

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort("timestamp")

    # Return exactly 100K rows (or all if less)
    if len(df) > 100_000:
        return df.head(100_000)
    print(f"Warning: Only {len(df)} rows available in crypto data")
    return df


@pytest.fixture(scope="session")
def crypto_data_small(crypto_data):
    """Smaller crypto dataset for quick tests (10K rows)."""
    return crypto_data.head(10_000)


def _generate_synthetic_crypto_data():
    """Generate synthetic crypto data as fallback."""
    np.random.seed(42)
    n = 100_000

    # Generate realistic crypto returns
    returns = np.random.standard_t(df=3, size=n) * 0.0005
    trend = 0.00001
    returns = returns + trend

    # Generate price series
    base_price = 30000.0  # BTC-like price
    price_series = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    closes = price_series
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # Add gaps
    gaps = np.random.normal(0, 0.0001, n)
    opens = opens * (1 + gaps)

    # High/Low with wicks
    wick_size = np.abs(np.random.standard_t(df=4, size=n)) * 0.002 + 0.001
    highs = np.maximum(opens, closes) * (1 + wick_size * np.random.uniform(0.5, 1.5, n))
    lows = np.minimum(opens, closes) * (1 - wick_size * np.random.uniform(0.5, 1.5, n))

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Generate volume
    base_volume = np.random.lognormal(np.log(100), 0.5, n)
    price_changes = np.abs(np.diff(np.concatenate([[closes[0]], closes])))
    volatility_factor = price_changes / np.mean(price_changes)
    volumes = base_volume * (1 + volatility_factor * 2)

    return pl.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes.astype(float),
        },
    )


@pytest.fixture
def performance_threshold():
    """
    Dynamic performance threshold calculator based on indicator complexity.

    Returns a function that calculates appropriate threshold based on:
    - Indicator type
    - Complexity (simple, rolling, recursive)
    - Expected Python/C performance ratio
    """

    def get_threshold(indicator_type: str) -> int:
        """
        Get performance threshold multiplier for different indicator types.

        Parameters
        ----------
        indicator_type : str
            Type of indicator (e.g., 'simple', 'rolling', 'ema', 'wilders', 'complex')

        Returns
        -------
        int
            Maximum allowed slowdown factor vs TA-Lib
        """
        # REALISTIC PERFORMANCE THRESHOLDS
        # TA-Lib is pure C - Python+Numba will be slightly slower
        # Goal: Within 2x of TA-Lib (competitive performance)
        thresholds = {
            # Basic math operations (MAX, MIN, SUM) - 20x threshold
            # These are the most basic operations where TA-Lib's hand-optimized
            # C code has maximum advantage. NumPy's sliding_window_view is
            # already near-optimal, achieving 7-15x vs TA-Lib's C is excellent.
            "math_basic": 20.0,
            # Simple arithmetic (OHLC transforms) - 2x threshold
            "simple": 2.0,
            # Basic rolling calculations (SMA, STDDEV) - 2x threshold
            "rolling": 2.0,
            # EMA-based (single pass with state) - 2x threshold
            "ema": 2.0,
            # Wilder's smoothing (ATR, ADX, RSI) - 2x threshold
            "wilders": 2.0,
            # Complex multi-step (MACD, STOCH, Bollinger) - 3x threshold
            "complex": 3.0,
            # Multi-window indicators (ULTOSC with 3 separate rolling windows) - 7x threshold
            # ULTOSC calculates 3 independent rolling windows (7, 14, 28 periods)
            # TA-Lib's pure C has significant advantage for these multiple passes
            "multi_window": 7.0,
            # Very complex indicators - 3x threshold
            "very_complex": 3.0,
            # Standard threshold for most indicators - 2x threshold
            "standard": 2.0,
        }

        return thresholds.get(indicator_type, 2.0)  # Default to 2.0x (realistic)

    return get_threshold


@pytest.fixture
def tick_data():
    """Generate sample tick data for bar sampler tests."""
    from datetime import datetime, timedelta

    n = 1000
    base_time = datetime(2024, 1, 1)

    # Generate realistic tick data
    np.random.seed(42)
    prices = 100 + np.random.randn(n).cumsum() * 0.1
    volumes = np.random.poisson(100, n)

    return pl.DataFrame(
        {
            "timestamp": [base_time + timedelta(seconds=i) for i in range(n)],
            "price": prices,
            "volume": volumes,
            "side": np.random.choice([1, -1], n),  # Buy/sell indicator
        },
    )


@pytest.fixture
def warmup_jit():
    """
    Warmup JIT compiled functions before performance testing.

    Returns a function that runs an indicator with small data to trigger JIT compilation.
    """

    def warmup(func, *sample_args):
        """
        Warmup a function with sample data.

        Parameters
        ----------
        func : callable
            The indicator function to warmup
        sample_args : tuple
            Sample arguments matching the function signature
        """
        import contextlib

        with contextlib.suppress(Exception):
            # Run once to trigger JIT compilation
            _ = func(*sample_args)

    return warmup


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """
    Automatically add markers to tests based on their characteristics.

    This hook runs after test collection and adds appropriate markers:
    - performance: Tests using 'performance_threshold' fixture or named '*_performance'
    - integration: Tests in test_all_features_integration.py or test_api.py
    - property: Tests using hypothesis strategies
    """
    for item in items:
        # Mark performance tests
        if "performance_threshold" in item.fixturenames or "_performance" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.benchmark)

        # Mark integration tests
        if "integration" in str(item.fspath) or "test_api.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark property-based tests (already have @given decorator, but add marker too)
        if "hypothesis" in str(item.obj):
            item.add_marker(pytest.mark.property)


@pytest.fixture(autouse=True)
def preserve_registry_state():
    """Preserve global registry state between tests.

    This fixture ensures test isolation by saving the registry state before each test
    and restoring it afterwards. This prevents tests from polluting the global registry
    which causes integration tests to fail with unexpected feature counts.
    """
    # Save the current registry state
    original_features = registry_module._global_registry._features.copy()

    # Let the test run
    yield

    # Restore the original state
    registry_module._global_registry._features = original_features
