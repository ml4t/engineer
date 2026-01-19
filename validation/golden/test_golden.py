"""
Golden File Regression Tests

Compare current feature calculations against stored reference outputs.
If any test fails, it means the calculation has changed from the golden reference.

To update golden files after intentional changes:
    python validation/golden/generate_golden.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.trend import sma, ema, wma
from ml4t.engineer.features.momentum import rsi, roc, mom
from ml4t.engineer.features.volatility import atr, natr
from ml4t.engineer.features.statistics import stddev
from ml4t.engineer.features.volume import obv, ad


GOLDEN_DATA_DIR = Path(__file__).parent / "data"

# Check if golden files exist
GOLDEN_FILES_EXIST = GOLDEN_DATA_DIR.exists() and len(list(GOLDEN_DATA_DIR.glob("*.parquet"))) > 0


# Reference data (must match generate_golden.py exactly)
np.random.seed(42)
N = 500
RETURNS = np.random.randn(N) * 0.02
CLOSE = 100.0 * np.exp(np.cumsum(RETURNS))
HIGH = CLOSE * (1 + np.abs(np.random.randn(N)) * 0.01)
LOW = CLOSE * (1 - np.abs(np.random.randn(N)) * 0.01)
OPEN = (HIGH + LOW) / 2
HIGH = np.maximum.reduce([OPEN, CLOSE, HIGH])
LOW = np.minimum.reduce([OPEN, CLOSE, LOW])
VOLUME = np.random.randint(100000, 1000000, N).astype(np.float64)


def load_golden(name: str) -> pl.DataFrame:
    """Load a golden reference file."""
    path = GOLDEN_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"Golden file not found: {path}")
    return pl.read_parquet(path)


def assert_golden_match(current: np.ndarray, golden: np.ndarray, name: str) -> None:
    """Assert current calculation matches golden reference exactly."""
    # Handle NaN positions
    current_nan = np.isnan(current)
    golden_nan = np.isnan(golden)

    # NaN positions must match
    np.testing.assert_array_equal(
        current_nan,
        golden_nan,
        err_msg=f"{name}: NaN positions changed",
    )

    # Non-NaN values must match exactly
    current_valid = current[~current_nan]
    golden_valid = golden[~golden_nan]

    np.testing.assert_array_almost_equal(
        current_valid,
        golden_valid,
        decimal=10,
        err_msg=f"{name}: Values changed from golden reference",
    )


@pytest.mark.skipif(not GOLDEN_FILES_EXIST, reason="Golden files not generated")
class TestTrendGolden:
    """Golden tests for trend indicators."""

    def test_sma_20(self) -> None:
        """SMA-20 matches golden reference."""
        golden = load_golden("sma_20")
        current = sma(CLOSE, period=20)
        assert_golden_match(current, golden["feature"].to_numpy(), "SMA-20")

    def test_sma_50(self) -> None:
        """SMA-50 matches golden reference."""
        golden = load_golden("sma_50")
        current = sma(CLOSE, period=50)
        assert_golden_match(current, golden["feature"].to_numpy(), "SMA-50")

    def test_ema_20(self) -> None:
        """EMA-20 matches golden reference."""
        golden = load_golden("ema_20")
        current = ema(CLOSE, period=20)
        assert_golden_match(current, golden["feature"].to_numpy(), "EMA-20")

    def test_ema_50(self) -> None:
        """EMA-50 matches golden reference."""
        golden = load_golden("ema_50")
        current = ema(CLOSE, period=50)
        assert_golden_match(current, golden["feature"].to_numpy(), "EMA-50")

    def test_wma_20(self) -> None:
        """WMA-20 matches golden reference."""
        golden = load_golden("wma_20")
        current = wma(CLOSE, period=20)
        assert_golden_match(current, golden["feature"].to_numpy(), "WMA-20")


@pytest.mark.skipif(not GOLDEN_FILES_EXIST, reason="Golden files not generated")
class TestMomentumGolden:
    """Golden tests for momentum indicators."""

    def test_rsi_14(self) -> None:
        """RSI-14 matches golden reference."""
        golden = load_golden("rsi_14")
        current = rsi(CLOSE, period=14)
        assert_golden_match(current, golden["feature"].to_numpy(), "RSI-14")

    def test_roc_10(self) -> None:
        """ROC-10 matches golden reference."""
        golden = load_golden("roc_10")
        current = roc(CLOSE, period=10)
        assert_golden_match(current, golden["feature"].to_numpy(), "ROC-10")

    def test_mom_10(self) -> None:
        """MOM-10 matches golden reference."""
        golden = load_golden("mom_10")
        current = mom(CLOSE, period=10)
        assert_golden_match(current, golden["feature"].to_numpy(), "MOM-10")


@pytest.mark.skipif(not GOLDEN_FILES_EXIST, reason="Golden files not generated")
class TestVolatilityGolden:
    """Golden tests for volatility indicators."""

    def test_atr_14(self) -> None:
        """ATR-14 matches golden reference."""
        golden = load_golden("atr_14")
        current = atr(HIGH, LOW, CLOSE, period=14)
        assert_golden_match(current, golden["feature"].to_numpy(), "ATR-14")

    def test_natr_14(self) -> None:
        """NATR-14 matches golden reference."""
        golden = load_golden("natr_14")
        current = natr(HIGH, LOW, CLOSE, period=14)
        assert_golden_match(current, golden["feature"].to_numpy(), "NATR-14")

    def test_stddev_20(self) -> None:
        """STDDEV-20 matches golden reference."""
        golden = load_golden("stddev_20")
        current = stddev(CLOSE, period=20)
        assert_golden_match(current, golden["feature"].to_numpy(), "STDDEV-20")


@pytest.mark.skipif(not GOLDEN_FILES_EXIST, reason="Golden files not generated")
class TestVolumeGolden:
    """Golden tests for volume indicators."""

    def test_obv(self) -> None:
        """OBV matches golden reference."""
        golden = load_golden("obv")
        current = obv(CLOSE, VOLUME)
        assert_golden_match(current, golden["feature"].to_numpy(), "OBV")

    def test_ad(self) -> None:
        """A/D matches golden reference."""
        golden = load_golden("ad")
        current = ad(HIGH, LOW, CLOSE, VOLUME)
        assert_golden_match(current, golden["feature"].to_numpy(), "A/D")
