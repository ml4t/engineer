"""A missing observation must never produce a silently wrong indicator value.

Every feature here is computed twice on the same series, once clean and once with a
single observation removed, and the two are compared. A gap may cost the indicator a
warmup, and it may make the rest of the series NaN, but it may not return a finite
number that differs from the number the clean series produced. Before this was
enforced, one null in the input of ``momentum.rsi`` returned exactly ``100.0`` for
87.5% of the remaining series, and nineteen other feature-column pairs returned a
finite number that was simply wrong.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features import momentum, volatility, volume

N = 2000
HOLE = 500
TAIL = slice(-200, None)

# Wilder's average has infinite memory with geometric decay, so a re-seeded
# recursion converges towards the clean series rather than reaching it exactly.
# A silently wrong value differs by order 1 to 1e4, so this separates the two.
RTOL = 1e-6

# feature name -> (expression builder, the columns it is driven by)
FEATURES: dict[str, tuple[Callable[[], pl.Expr], list[str]]] = {
    "rsi": (lambda: momentum.rsi("close", period=14), ["close"]),
    "cmo": (lambda: momentum.cmo("close", timeperiod=14), ["close"]),
    "adx": (lambda: momentum.adx("high", "low", "close", period=14), ["high", "low", "close"]),
    "dx": (lambda: momentum.dx("high", "low", "close", timeperiod=14), ["high", "low", "close"]),
    "plus_di": (
        lambda: momentum.plus_di("high", "low", "close", timeperiod=14),
        ["high", "low", "close"],
    ),
    "minus_di": (
        lambda: momentum.minus_di("high", "low", "close", timeperiod=14),
        ["high", "low", "close"],
    ),
    "mfi": (
        lambda: momentum.mfi("high", "low", "close", "volume", period=14),
        ["high", "low", "close", "volume"],
    ),
    "cci": (lambda: momentum.cci("high", "low", "close", period=20), ["high", "low", "close"]),
    "ultosc": (lambda: momentum.ultosc("high", "low", "close"), ["high", "low", "close"]),
    "macd": (lambda: momentum.macd("close"), ["close"]),
    "stochrsi": (lambda: momentum.stochrsi("close"), ["close"]),
    "stochastic": (lambda: momentum.stochastic("high", "low", "close"), ["high", "low", "close"]),
    "willr": (lambda: momentum.willr("high", "low", "close", period=14), ["high", "low", "close"]),
    "atr": (lambda: volatility.atr("high", "low", "close", period=14), ["high", "low", "close"]),
    "natr": (lambda: volatility.natr("high", "low", "close", period=14), ["high", "low", "close"]),
    "obv": (lambda: volume.obv("close", "volume"), ["close", "volume"]),
    "ad": (
        lambda: volume.ad("high", "low", "close", "volume"),
        ["high", "low", "close", "volume"],
    ),
    "adosc": (
        lambda: volume.adosc("high", "low", "close", "volume"),
        ["high", "low", "close", "volume"],
    ),
}

CASES = [(name, driver) for name, (_, drivers) in FEATURES.items() for driver in drivers]


def _panel() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.01, 1.0, N))
    return pl.DataFrame(
        {
            "close": close,
            "high": close + np.abs(rng.normal(0.5, 0.2, N)),
            "low": close - np.abs(rng.normal(0.5, 0.2, N)),
            "open": close + rng.normal(0.0, 0.3, N),
            "volume": rng.integers(1_000, 100_000, N).astype(float),
        }
    )


def _with_hole(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.int_range(pl.len()) == HOLE).then(None).otherwise(pl.col(column)).alias(column)
    )


def _values(frame: pl.DataFrame, expr: pl.Expr) -> np.ndarray:
    out = frame.select(expr)
    series = out.to_series(0)
    if series.dtype == pl.Struct:
        series = out.unnest(out.columns[0]).to_series(0)
    return series.to_numpy().astype(float)


@pytest.mark.parametrize(("feature", "driver"), CASES, ids=[f"{f}-{d}" for f, d in CASES])
def test_one_missing_bar_is_never_silently_absorbed(feature: str, driver: str) -> None:
    frame = _panel()
    build, _ = FEATURES[feature]

    clean = _values(frame, build())
    holed = _values(_with_hole(frame, driver), build())

    finite_in_both = np.isfinite(clean) & np.isfinite(holed)
    disagree = finite_in_both & ~np.isclose(clean, holed, rtol=RTOL, atol=1e-9)

    # A gap may cost the indicator a warmup, so allow disagreement while it
    # re-seeds, but never at the end of the series.
    tail_disagree = np.flatnonzero(disagree[TAIL])
    assert tail_disagree.size == 0, (
        f"{feature} returns a finite but different value at the end of the series when "
        f"one {driver} bar is missing: "
        f"clean {clean[TAIL][tail_disagree][:3]} vs holed {holed[TAIL][tail_disagree][:3]}"
    )


def test_rsi_does_not_saturate_after_a_gap() -> None:
    """The exact failure this suite exists for: one null, then 100.0 forever."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.01, 1.0, 4000))
    holed = close.copy()
    holed[500] = np.nan

    clean_out = momentum.rsi(close, period=9)
    holed_out = momentum.rsi(holed, period=9)

    after_gap = holed_out[501:]
    finite = after_gap[np.isfinite(after_gap)]
    assert finite.size > 3000
    assert np.mean(finite == 100.0) == 0.0, "RSI saturated at 100 after a gap"

    # The gap costs exactly `period` observations: the missing bar and its warmup.
    assert np.all(np.isnan(holed_out[500:510]))
    assert np.isfinite(holed_out[510])

    # Everything before the gap is untouched, and the recursion re-converges after it.
    np.testing.assert_allclose(clean_out[:500], holed_out[:500])
    np.testing.assert_allclose(clean_out[-500:], holed_out[-500:], rtol=1e-9, atol=1e-9)
