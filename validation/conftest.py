"""Shared fixtures for validation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Standard tolerance for numerical comparisons
RTOL = 1e-6
ATOL = 1e-10


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    # Generate realistic price series
    returns = np.random.randn(n) * 0.02
    close = 100.0 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = (high + low) / 2 + np.random.randn(n) * 0.5

    # Ensure OHLC constraints
    high = np.maximum.reduce([open_, close, high])
    low = np.minimum.reduce([open_, close, low])

    volume = np.random.randint(100000, 1000000, n).astype(np.float64)

    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def sample_ohlcv_pandas(sample_ohlcv: pl.DataFrame):
    """Convert sample data to pandas for reference implementations."""
    return sample_ohlcv.to_pandas().set_index("timestamp")


@pytest.fixture
def sample_ohlcv_numpy(sample_ohlcv: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
    """Convert sample data to numpy arrays for TA-Lib."""
    return {
        "open": sample_ohlcv["open"].to_numpy().astype(np.float64),
        "high": sample_ohlcv["high"].to_numpy().astype(np.float64),
        "low": sample_ohlcv["low"].to_numpy().astype(np.float64),
        "close": sample_ohlcv["close"].to_numpy().astype(np.float64),
        "volume": sample_ohlcv["volume"].to_numpy().astype(np.float64),
    }


@pytest.fixture
def spy_data() -> pl.DataFrame:
    """Fetch real SPY data for validation."""
    try:
        import yfinance as yf

        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="2y")

        return pl.DataFrame(
            {
                "timestamp": pl.Series(hist.index.to_pydatetime()),
                "open": hist["Open"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "close": hist["Close"].values,
                "volume": hist["Volume"].values.astype(np.float64),
            }
        )
    except Exception:
        pytest.skip("Could not fetch SPY data")


@pytest.fixture
def constant_series() -> pl.DataFrame:
    """Series with constant values (edge case)."""
    n = 100
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000000.0] * n,
        }
    )


@pytest.fixture
def trending_series() -> pl.DataFrame:
    """Monotonically increasing series (edge case)."""
    n = 100
    close = np.linspace(100, 200, n)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000000.0] * n,
        }
    )


@pytest.fixture
def alternating_series() -> pl.DataFrame:
    """Alternating up/down series (edge case)."""
    n = 100
    close = np.array([100.0 + (i % 2) * 10 for i in range(n)])
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": [1000000.0] * n,
        }
    )


def assert_arrays_close(
    actual: NDArray[np.float64] | pl.Series,
    expected: NDArray[np.float64] | pl.Series,
    rtol: float = RTOL,
    atol: float = ATOL,
    name: str = "values",
) -> None:
    """Assert two arrays are close, with helpful error messages."""
    if isinstance(actual, pl.Series):
        actual = actual.to_numpy()
    if isinstance(expected, pl.Series):
        expected = expected.to_numpy()

    # Handle NaN values
    nan_mask = np.isnan(actual) | np.isnan(expected)
    if nan_mask.any():
        # Check NaN positions match
        np.testing.assert_array_equal(
            np.isnan(actual),
            np.isnan(expected),
            err_msg=f"{name}: NaN positions don't match",
        )
        # Compare non-NaN values
        actual = actual[~nan_mask]
        expected = expected[~nan_mask]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name}: values don't match within tolerance",
    )
