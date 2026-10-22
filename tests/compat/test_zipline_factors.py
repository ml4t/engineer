"""Finite Pipeline definitions must not silently use recursive or sample methods."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry


def prices(close: list[float | None], volume: list[float | None] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open": close,
            "high": [None if x is None else x + 2 for x in close],
            "low": [None if x is None else x / 2 for x in close],
            "close": close,
            "volume": volume if volume is not None else [10.0] * len(close),
        },
        schema_overrides=dict.fromkeys(("open", "high", "low", "close", "volume"), pl.Float64),
    )


def calculate(frame: pl.DataFrame, name: str, **params: object) -> pl.DataFrame:
    return compute_features(
        frame, [{"name": name, "params": params, "output": "factor"}], calendar=None
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_cutler_uses_finite_changes_instead_of_wilder_recursion(lazy: bool) -> None:
    frame = prices([10.0, 13.0, 12.0, 14.0, 13.0, 17.0, 16.0])
    result = compute_features(
        frame.lazy() if lazy else frame,
        [{"name": "rsi_cutler", "params": {"period": 3}}],
        calendar=None,
    )
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    actual = result.collect() if lazy else result
    np.testing.assert_allclose(
        actual["rsi_cutler"].to_numpy(),
        [np.nan, np.nan, np.nan, 100 * 5 / 6, 50, 100 * 6 / 7, 100 * 4 / 6],
        rtol=1e-14,
        equal_nan=True,
    )
    native = calculate(frame, "rsi", period=3)["factor"].to_numpy()
    assert not np.allclose(native[4:], actual["rsi_cutler"].to_numpy()[4:])


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_cutler_omits_missing_changes_without_filling_quotes(missing: float | None) -> None:
    frame = prices([1.0, 2.0, missing, 4.0, 5.0, 5.0, 3.0])
    result = calculate(frame, "rsi_cutler", period=3)
    np.testing.assert_allclose(
        result["factor"].to_numpy(),
        [np.nan, np.nan, np.nan, 100, 100, 100, 100 / 3],
        equal_nan=True,
    )
    assert_frame_equal(result.select(frame.columns), frame)


@pytest.mark.parametrize(
    ("name", "close", "volume", "params", "expected"),
    [
        ("finite_mean", [1.0, np.nan, 5.0], None, {}, 3.0),
        ("finite_weighted_mean", [1.0, np.nan, 5.0], [1.0, 2.0, 3.0], {}, 16 / 6),
        ("finite_weighted_mean", [1.0, 2.0, 5.0], [0.0, 0.0, 0.0], {}, np.nan),
        ("average_dollar_volume", [1.0, np.nan, 5.0], [1.0, 2.0, 3.0], {}, 16 / 3),
        ("average_dollar_volume", [1.0, np.nan, 5.0], [0.0, 0.0, 0.0], {}, 0.0),
        ("linear_weighted_mean", [1.0, np.nan, 5.0], None, {}, 16 / 6),
        ("finite_ewma", [1.0, 3.0, 9.0], None, {"decay_rate": 0.5}, 43 / 7),
        ("finite_ewma", [1.0, 3.0, 9.0], None, {"decay_rate": 1.0}, 13 / 3),
        ("finite_ewma", [1.0, np.nan, 9.0], None, {"decay_rate": 0.5}, np.nan),
        ("finite_ewm_std", [1.0, 3.0, 9.0], None, {"decay_rate": 1.0}, math.sqrt(52 / 3)),
        ("finite_ewm_std", [1.0, 3.0, 9.0], None, {"decay_rate": 0.5}, math.sqrt(138 / 7)),
        ("finite_ewm_std", [1.0, np.nan, 9.0], None, {"decay_rate": 0.5}, np.nan),
        (
            "annualized_volatility",
            [1.0, 3.0, 9.0],
            None,
            {"annualization_factor": 9.0},
            math.sqrt(104),
        ),
        ("finite_max_drawdown", [100.0, 50.0, 60.0], None, {}, 1.0),
        ("finite_max_drawdown", [100.0, 60.0, 50.0], None, {}, 1.0),
    ],
)
def test_finite_methods_preserve_nan_denominators_weights_and_scale(
    name: str,
    close: list[float],
    volume: list[float] | None,
    params: dict[str, object],
    expected: float,
) -> None:
    frame = prices(close, volume)
    if name == "annualized_volatility":
        frame = frame.with_columns(pl.col("close").alias("returns"))
    actual = calculate(frame, name, period=3, **params)["factor"].to_numpy()
    np.testing.assert_allclose(actual, [np.nan, np.nan, expected], equal_nan=True, rtol=1e-14)


@pytest.mark.parametrize("name", ["finite_ewma", "finite_ewm_std"])
def test_finite_weights_do_not_overflow_or_underflow_with_legacy_common_power(name: str) -> None:
    frame = prices([4.0, 4.0, 4.0, 4.0])
    result = calculate(frame, name, period=4, decay_rate=1e-100)["factor"][-1]
    assert result == (4.0 if name == "finite_ewma" else 0.0)


@pytest.mark.parametrize(
    "name", ["finite_mean", "linear_weighted_mean", "finite_ewma", "rsi_cutler"]
)
def test_finite_window_never_crosses_asset_boundaries_and_lazy_matches_eager(name: str) -> None:
    first = prices([10.0, 13.0, 12.0, 14.0]).with_columns(pl.lit("A").alias("asset"))
    second = prices([100.0, 130.0, 120.0, 140.0]).with_columns(pl.lit("B").alias("asset"))
    frame = pl.concat([first, second]).sort("close")
    spec = [{"name": name, "params": {"period": 3}}]
    result = compute_features(frame, spec, group_col="asset", calendar=None)
    lazy = compute_features(frame.lazy(), spec, group_col="asset", calendar=None).collect()
    assert_frame_equal(result, lazy)
    for asset in ("A", "B"):
        expected = compute_features(frame.filter(pl.col("asset") == asset), spec, calendar=None)
        assert_frame_equal(result.filter(pl.col("asset") == asset), expected)


@pytest.mark.parametrize(
    "name",
    [
        "finite_mean",
        "finite_weighted_mean",
        "average_dollar_volume",
        "linear_weighted_mean",
        "finite_ewma",
        "finite_ewm_std",
        "annualized_volatility",
        "finite_max_drawdown",
        "rsi_cutler",
    ],
)
@pytest.mark.parametrize("period", [0, -1, True, 1.5])
def test_invalid_windows_fail_without_modifying_input(name: str, period: object) -> None:
    frame = prices([1.0, 2.0, 3.0]).with_columns(pl.lit(0.1).alias("returns"))
    before = frame.clone()
    with pytest.raises((TypeError, ValueError), match="period"):
        calculate(frame, name, period=period)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("decay", [0.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_decay_fails_before_output(decay: float) -> None:
    with pytest.raises((TypeError, ValueError), match="decay_rate"):
        calculate(prices([1.0, 2.0, 3.0]), "finite_ewma", period=3, decay_rate=decay)


def test_cutler_registry_metadata_declares_price_count_and_keeps_native_defaults() -> None:
    cutler = get_registry().get("rsi_cutler")
    native = get_registry().get("rsi")
    assert cutler is not None and native is not None
    assert cutler.lookback(period=5) == 5
    assert cutler.value_range == (0.0, 100.0)
    assert cutler.normalized and not cutler.ta_lib_compatible
    assert native.parameters["period"] == 14
    assert native.ta_lib_compatible


def test_finite_bands_use_population_std_and_nan_omission() -> None:
    actual = calculate(prices([1.0, 3.0, 9.0]), "finite_bollinger_bands", period=3, k=2)
    middle = 13 / 3
    distance = 2 * math.sqrt(104 / 9)
    assert actual["factor_lower"][-1] == pytest.approx(middle - distance)
    assert actual["factor_middle"][-1] == pytest.approx(middle)
    assert actual["factor_upper"][-1] == pytest.approx(middle + distance)
    missing = calculate(prices([1.0, np.nan, 9.0]), "finite_bollinger_bands", period=3, k=1)
    assert missing.select("factor_lower", "factor_middle", "factor_upper").row(-1) == (1, 5, 9)


def test_aroon_first_extreme_tie_is_not_native_latest_tie() -> None:
    frame = prices([3.5, 2.0, 4.0]).with_columns(
        pl.Series("high", [4.0, 5.0, 5.0]), pl.Series("low", [3.0, 1.0, 1.0])
    )
    actual = calculate(frame, "finite_aroon", period=3)
    assert actual["factor_down"][-1] == 50
    assert actual["factor_up"][-1] == 50
    assert (
        actual["factor_up"].head(2).is_null().all()
        or np.isnan(actual["factor_up"].head(2).to_numpy()).all()
    )


def test_fast_stochastic_has_no_smoothing_or_zero_range_replacement() -> None:
    frame = prices([3.5, 2.0, 4.0]).with_columns(
        pl.Series("high", [4.0, 5.0, 5.0]), pl.Series("low", [3.0, 1.0, 1.0])
    )
    assert calculate(frame, "fast_stochastic", period=3)["factor"][-1] == 75
    flat = pl.DataFrame({name: [3.0] * 3 for name in ("open", "high", "low", "close", "volume")})
    assert math.isnan(calculate(flat, "fast_stochastic", period=3)["factor"][-1])


def test_ichimoku_fields_use_unshifted_spans_and_position_within_window() -> None:
    frame = prices([10.0, 14.0, 11.0, 20.0, 16.0, 14.0]).with_columns(
        pl.Series("high", [12.0, 18.0, 14.0, 22.0, 20.0, 16.0]),
        pl.Series("low", [8.0, 10.0, 9.0, 16.0, 13.0, 12.0]),
    )
    actual = calculate(
        frame, "finite_ichimoku", period=6, tenkan_length=2, kijun_length=3, chikou_position=2
    )
    assert actual.select(pl.col("^factor_.*$")).row(-1) == (16, 17, 16.5, 15, 11)
    assert all(
        actual[c].head(5).null_count() == 5 for c in actual.columns if c.startswith("factor_")
    )


def test_finite_macd_signal_uses_normalized_windows_not_recursive_ema() -> None:
    actual = calculate(
        prices([1.0, 2.0, 4.0, 8.0, 16.0]),
        "finite_macd_signal",
        fast_period=2,
        slow_period=3,
        signal_period=2,
    )
    np.testing.assert_allclose(
        actual["factor"].to_numpy(), [np.nan] * 3 + [0.875, 1.75], equal_nan=True
    )


def test_single_observation_macd_means_are_defined_instead_of_zero_weight_division() -> None:
    actual = calculate(
        prices([1.0, 3.0, 9.0]),
        "finite_macd_signal",
        fast_period=1,
        slow_period=2,
        signal_period=1,
    )
    np.testing.assert_allclose(actual["factor"].to_numpy(), [np.nan, 0.5, 1.5], equal_nan=True)


def test_finite_true_range_uses_two_price_rows_and_nanmax_components() -> None:
    frame = prices([10.0, 12.0, 13.0]).with_columns(
        pl.Series("high", [12.0, 15.0, 14.0]), pl.Series("low", [8.0, 11.0, 12.0])
    )
    np.testing.assert_allclose(
        calculate(frame, "finite_true_range")["factor"].to_numpy(), [np.nan, 5, 2], equal_nan=True
    )


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("finite_aroon", {"period": 1}),
        ("finite_bollinger_bands", {"k": float("nan")}),
        ("finite_ichimoku", {"period": 6, "tenkan_length": 7}),
        ("finite_ichimoku", {"period": 6, "chikou_position": 6}),
        ("finite_macd_signal", {"fast_period": 3, "slow_period": 3}),
        ("finite_macd_signal", {"signal_period": 0}),
    ],
)
def test_technical_options_reject_undefined_windows(name: str, params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        calculate(prices([10.0] * 60), name, **params)


@pytest.mark.parametrize("dtype,base", [(pl.Int64, 2**53), (pl.UInt64, 2**63), (pl.Int128, 2**126)])
def test_cutler_preserves_integer_tick_differences_before_float_conversion(dtype, base) -> None:
    frame = pl.DataFrame({"close": pl.Series([base, base + 3, base + 2, base + 4], dtype=dtype)})
    from ml4t.engineer.features.rolling import rsi_cutler

    assert frame.select(rsi_cutler("close", period=3)).item(-1, 0) == pytest.approx(100 * 5 / 6)


@pytest.mark.parametrize(
    "name",
    [
        "finite_mean",
        "finite_weighted_mean",
        "average_dollar_volume",
        "linear_weighted_mean",
        "finite_ewma",
        "finite_ewm_std",
        "rsi_cutler",
        "finite_max_drawdown",
    ],
)
def test_all_missing_window_does_not_create_an_observed_quote(name: str) -> None:
    actual = calculate(prices([None] * 4), name, period=3)["factor"].to_numpy()
    expected = (
        0
        if name in {"average_dollar_volume", "linear_weighted_mean", "finite_weighted_mean"}
        else np.nan
    )
    np.testing.assert_allclose(actual[-1], expected, equal_nan=True)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf"), True])
def test_population_volatility_requires_explicit_positive_scale(factor: float) -> None:
    with pytest.raises(ValueError, match="annualization_factor"):
        calculate(
            prices([10.0] * 4).with_columns(pl.lit(0.1).alias("returns")),
            "annualized_volatility",
            period=3,
            annualization_factor=factor,
        )


@pytest.mark.parametrize("dtype,base", [(pl.Int64, 2**53), (pl.UInt64, 2**63), (pl.Int128, 2**126)])
def test_finite_weighted_std_preserves_tiny_integer_price_variation(dtype, base) -> None:
    frame = pl.DataFrame({"close": pl.Series([base, base + 1, base + 2], dtype=dtype)})
    frame = frame.with_columns(
        *(pl.col("close").alias(c) for c in ("open", "high", "low")), pl.lit(1.0).alias("volume")
    )
    assert calculate(frame, "finite_ewm_std", period=3, decay_rate=1.0)["factor"][-1] == 1.0


def test_local_variance_does_not_lose_ticks_after_an_earlier_price_level_change() -> None:
    frame = pl.DataFrame({"close": pl.Series([1, 2**63, 2**63 + 1, 2**63 + 2], dtype=pl.UInt64)})
    from ml4t.engineer.features.rolling import finite_ewm_std

    assert frame.select(finite_ewm_std("close", period=3, decay_rate=1.0)).item(-1, 0) == 1.0


def test_finite_weighted_moments_do_not_overflow_representable_float_results() -> None:
    from ml4t.engineer.features.rolling import finite_ewm_std, finite_ewma

    constant = pl.DataFrame({"close": [1e308] * 3})
    assert constant.select(finite_ewma("close", period=3, decay_rate=1)).item(
        -1, 0
    ) == pytest.approx(1e308)
    spread = pl.DataFrame({"close": [-1e300, 1e300]})
    assert spread.select(finite_ewm_std("close", period=2, decay_rate=1)).item(
        -1, 0
    ) == pytest.approx(math.sqrt(2) * 1e300)


@pytest.mark.parametrize("name", ["finite_true_range", "fast_stochastic", "finite_aroon"])
def test_normalized_and_range_methods_do_not_collapse_integer_price_ticks(name: str) -> None:
    base = 2**63
    frame = pl.DataFrame(
        {
            "close": pl.Series([base + 2, base + 3, base + 4], dtype=pl.UInt64),
            "high": pl.Series([base + 3, base + 4, base + 5], dtype=pl.UInt64),
            "low": pl.Series([base, base + 1, base + 2], dtype=pl.UInt64),
        }
    ).with_columns(pl.col("close").alias("open"), pl.lit(1.0).alias("volume"))
    result = calculate(frame, name, **({} if name == "finite_true_range" else {"period": 3}))
    if name == "finite_true_range":
        np.testing.assert_allclose(result["factor"].to_numpy(), [np.nan, 3, 3], equal_nan=True)
    elif name == "fast_stochastic":
        assert result["factor"][-1] == 80
    else:
        assert result["factor_up"][-1] == 100
        assert result["factor_down"][-1] == 0


@pytest.mark.parametrize("decay", [1e-10, 1e-50, 1e-100])
def test_two_observation_weighted_std_retains_exact_degrees_of_freedom(decay: float) -> None:
    # For any two positive weights, the unbiased variance is (x1-x0)**2/2.
    actual = calculate(prices([1.0, 5.0]), "finite_ewm_std", period=2, decay_rate=decay)
    assert actual["factor"][-1] == pytest.approx(math.sqrt(8), rel=1e-14, abs=0)
