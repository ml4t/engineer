"""Public price expressions must not discard exact integer ticks before division."""

import math

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.engineer import compute_features
from ml4t.engineer.core.exceptions import InvalidParameterError
from ml4t.engineer.features.utils.arithmetic import (
    log_returns,
    percentage_change,
    returns,
)

INTEGER_CASES = [
    (pl.Int64, [2**53, 2**53 + 1, 2**53 + 2]),
    (pl.Int64, [2**63 - 1, 2**63 - 2, 2**63 - 3]),
    (pl.UInt64, [2**63, 2**63 + 1, 2**63 + 2]),
    (pl.UInt64, [2**64 - 1, 2**64 - 2, 2**64 - 3]),
    (pl.Int64, [-(2**63), 2**63 - 1, -(2**63)]),
    (pl.UInt64, [2**64 - 1, 0, 1, 2**64 - 1]),
    (pl.UInt64, [2**64 - 1, 1, 2]),
    (pl.Int128, [2**127 - 1, 2**127 - 2, 2**127 - 3]),
    (pl.Int128, [-(2**127), 2**127 - 1, -(2**127)]),
    (pl.Int8, [-128, 127, -128]),
    (pl.UInt8, [250, 5, 10]),
    (pl.Int64, [100, 101, None, 99, 103]),
    (pl.Int64, [0, 0, 10, 0, -10]),
    (pl.Int64, [-100, -101, -99, -102]),
    (pl.Int64, []),
    (pl.Int64, [2**53 + 1]),
]
METHODS = [returns, log_returns, percentage_change]


def _expected(prices, periods, method):
    result = []
    for index, current in enumerate(prices):
        previous = prices[index - periods] if index >= periods else None
        if current is None or previous is None:
            result.append(None)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            change = (current - previous) / previous if previous else np.divide(current, previous)
            if method is log_returns:
                value = (
                    np.log1p(change) if abs(change) < 0.5 else np.log(np.divide(current, previous))
                )
            else:
                value = change * (100 if method is percentage_change else 1)
        result.append(float(value))
    return result


def _assert_values(actual, expected):
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        if wanted is None:
            assert observed is None
        elif math.isnan(wanted):
            assert math.isnan(observed)
        else:
            assert observed == pytest.approx(wanted, rel=2e-15, abs=0)


@pytest.mark.parametrize("dtype,prices", INTEGER_CASES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("periods", [1, 2])
@pytest.mark.parametrize("lazy", [False, True])
def test_exact_integer_prices_preserve_ticks_limits_and_missingness(
    dtype, prices, method, periods, lazy
):
    frame = pl.DataFrame({"close": pl.Series(prices, dtype=dtype)})
    before = frame.clone()
    data = frame.lazy() if lazy else frame
    produced = data.select(method("close", periods).alias("return"))
    if lazy:
        assert isinstance(produced, pl.LazyFrame)
        assert produced.collect_schema()["return"] == pl.Float64
        produced = produced.collect()
    _assert_values(produced["return"].to_list(), _expected(prices, periods, method))
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("lazy", [False, True])
def test_float_prices_keep_native_dtype_domain_and_missing_conventions(method, dtype, lazy):
    frame = pl.DataFrame(
        {"close": pl.Series([100.5, 101.25, None, float("nan"), 0, -1], dtype=dtype)}
    )
    source = pl.col("close")
    ratio = source / source.shift(1)
    expected = ratio.log() if method is log_returns else (ratio - 1)
    if method is percentage_change:
        expected = expected * 100
    data = frame.lazy() if lazy else frame
    produced = data.select(method("close").alias("return"))
    control = data.select(expected.alias("return"))
    if lazy:
        produced, control = produced.collect(), control.collect()
    assert_frame_equal(produced, control)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("lazy", [False, True])
def test_public_price_validation_then_grouped_returns_does_not_mix_assets(method, lazy):
    frame = pl.DataFrame(
        {
            "asset": ["A", "B", "A", "B", "A", "B"],
            "close": [2**53, 100, 2**53 + 1, 101, 2**53 + 2, 99],
        }
    ).with_columns(
        pl.col("close").alias("open"),
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
        pl.lit(100).alias("volume"),
    )
    before = frame.clone()
    source = frame.lazy() if lazy else frame
    validated = compute_features(source, [], group_col="asset", calendar=None)
    produced = validated.with_columns(method("close").over("asset").alias("return"))
    if lazy:
        produced = produced.collect(engine="streaming")
    for asset, prices in (("A", [2**53, 2**53 + 1, 2**53 + 2]), ("B", [100, 101, 99])):
        _assert_values(
            produced.filter(pl.col("asset") == asset)["return"].to_list(),
            _expected(prices, 1, method),
        )
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("periods", [0, -1])
def test_invalid_lag_fails_before_input_or_lazy_plan_is_changed(method, periods):
    frame = pl.DataFrame({"close": [100, 101]})
    before = frame.clone()
    with pytest.raises(InvalidParameterError, match="periods must be >= 1"):
        frame.lazy().select(method("close", periods))
    assert_frame_equal(frame, before)
