"""Thin legacy-name recipes must preserve methods, input ordering and output masks."""

import importlib
import math
from datetime import UTC

import numpy as np
import polars as pl
import pytest


def recipe(name: str, **options):
    return importlib.import_module("ml4t.engineer.compat.zipline").factor_expression(
        name, **options
    )


@pytest.mark.parametrize(
    "name,window,expected",
    [
        ("Returns", 3, [np.nan, np.nan, 0.21, 0.21]),
        ("DailyReturns", None, [np.nan, 0.1, 0.1, 0.1]),
        ("DailyReturns", 3, [np.nan, np.nan, 0.21, 0.21]),
        ("RateOfChangePercentage", 3, [np.nan, np.nan, 21, 21]),
    ],
)
def test_return_recipes_translate_price_count_to_changes(name, window, expected) -> None:
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]})
    options = {"window_length": window} if window is not None else {}
    if name == "RateOfChangePercentage":
        options["inputs"] = ["close"]
    actual = frame.lazy().select(recipe(name, **options).alias("result")).collect()
    np.testing.assert_allclose(
        actual["result"].to_numpy(), expected, rtol=1e-14, atol=0, equal_nan=True
    )


def test_absolute_percent_change_does_not_use_signed_denominator_or_percent_scale() -> None:
    frame = pl.DataFrame({"value": [-100.0, -90.0, -110.0]})
    actual = frame.select(
        recipe("PercentChange", inputs=["value"], window_length=2).alias("result")
    )
    np.testing.assert_allclose(
        actual["result"].to_numpy(), [np.nan, 0.1, -2 / 9], rtol=1e-14, equal_nan=True
    )
    zero = pl.DataFrame({"value": [-0.0, 1.0, -0.0, 2.0]})
    np.testing.assert_allclose(
        zero.select(recipe("PercentChange", inputs=["value"], window_length=2))
        .to_series()
        .to_numpy(),
        [np.nan, np.inf, -1, np.inf],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "name",
    ["EWMA", "ExponentialWeightedMovingAverage", "EWMSTD", "ExponentialWeightedMovingStdDev"],
)
@pytest.mark.parametrize(
    "constructor,params",
    [
        ("from_span", {"span": 3.0}),
        ("from_halflife", {"halflife": 1.0}),
        ("from_center_of_mass", {"center_of_mass": 1.0}),
    ],
)
def test_exponential_factories_select_the_same_declared_finite_weights(
    name, constructor, params
) -> None:
    frame = pl.DataFrame({"close": [1.0, 3.0, 9.0]})
    actual = frame.select(
        recipe(f"{name}.{constructor}", inputs=["close"], window_length=3, **params)
    ).item(-1, 0)
    expected = (
        math.sqrt(138 / 7) if name in {"EWMSTD", "ExponentialWeightedMovingStdDev"} else 43 / 7
    )
    assert actual == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "name",
    [
        "SimpleMovingAverage",
        "WeightedAverageValue",
        "VWAP",
        "AverageDollarVolume",
        "LinearWeightedMovingAverage",
        "MaxDrawdown",
        "RSI",
        "BollingerBands",
        "Aroon",
        "FastStochasticOscillator",
        "IchimokuKinkoHyo",
        "TrueRange",
        "AnnualizedVolatility",
        "MACDSignal",
        "MovingAverageConvergenceDivergenceSignal",
    ],
)
def test_numeric_recipes_produce_lazy_expressions_without_importing_legacy_runtime(name) -> None:
    import sys

    frame = pl.DataFrame(
        {"close": np.linspace(10.0, 20.0, 60), "volume": np.arange(1.0, 61.0)}
    ).with_columns(
        (pl.col("close") + 2).alias("high"),
        (pl.col("close") - 2).alias("low"),
    )
    options = {}
    if name in {"SimpleMovingAverage", "MaxDrawdown", "LinearWeightedMovingAverage"}:
        options.update(inputs=["close"], window_length=3)
    elif name == "WeightedAverageValue":
        options.update(inputs=["close", "volume"], window_length=3)
    elif name in {"VWAP", "AverageDollarVolume", "BollingerBands", "Aroon"}:
        options["window_length"] = 3
    output = recipe(name, **options)
    if isinstance(output, dict):
        expressions = [value.alias(key) for key, value in output.items()]
    else:
        assert isinstance(output, pl.Expr)
        expressions = [output.alias("result")]
    result = frame.lazy().select(expressions).collect()
    assert result.height == frame.height
    assert not any(key == "zipline" or key.startswith("zipline.") for key in sys.modules)


def test_aroon_recipe_translates_legacy_low_high_input_order() -> None:
    frame = pl.DataFrame({"l": [3.0, 1.0, 1.0], "h": [4.0, 5.0, 5.0]})
    output = recipe("Aroon", inputs=["l", "h"], window_length=3)
    result = frame.select([v.alias(k) for k, v in output.items()])
    assert result.row(-1) == (50.0, 50.0)


def test_output_mask_does_not_erase_prior_input_history() -> None:
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0], "mask": [False, True, False, True]})
    result = frame.with_columns(
        recipe("SimpleMovingAverage", inputs=["close"], window_length=3, mask="mask").alias(
            "result"
        )
    )
    assert result["result"].to_list() == [None, None, None, 3.0]
    assert result.height == 4
    assert result["close"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_clip_recipe_delegates_bounds_and_preserves_missing_values() -> None:
    frame = pl.DataFrame({"value": [float("nan"), -3.0, -1.0, 4.0]})
    result = frame.select(recipe("Clip", inputs=["value"], min_bound=-2.0, max_bound=2.0))
    np.testing.assert_allclose(result.to_series().to_numpy(), [np.nan, -2, -1, 2], equal_nan=True)


def test_default_annualized_recipe_keeps_price_to_returns_dependency_warmup() -> None:
    frame = pl.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]})
    result = frame.select(recipe("AnnualizedVolatility", window_length=3).alias("result"))
    initial = result["result"].head(3)
    assert (initial.is_null() | initial.is_nan()).all()
    assert result["result"][-1] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize(
    "name,options",
    [
        ("RSI", {"regression_length": 3}),
        ("VWAP", {"correlation_length": 3}),
        ("VWAP", {"window_length": 3, "inputs": {"close", "volume"}}),
        (
            "BusinessDaysSincePreviousEvent",
            {
                "inputs": ["event"],
                "window_length": False,
                "reference_date_col": "session",
                "known_at_col": "known",
                "decision_time_col": "decision",
            },
        ),
    ],
)
def test_options_cannot_change_unrelated_factor_meaning_or_input_order(name, options) -> None:
    with pytest.raises((TypeError, ValueError)):
        recipe(name, **options)


@pytest.mark.parametrize(
    "name,length,expected",
    [
        ("RollingPearson", "correlation_length", 9 / math.sqrt(95)),
        ("RollingSpearman", "correlation_length", 3 / math.sqrt(10)),
    ],
)
def test_paired_recipes_preserve_actual_paired_series(name, length, expected) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 2.0, 4.0]})
    result = frame.select(recipe(name, inputs=["y", "x"], **{length: 4}))
    assert result.item(-1, 0) == pytest.approx(expected, rel=1e-14, abs=0)


def test_regression_recipe_exposes_actual_named_fields() -> None:
    pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 2.0, 4.0]})
    result = frame.select(
        recipe("RollingLinearRegression", inputs=["y", "x"], regression_length=4).alias(
            "regression"
        )
    ).unnest("regression")
    np.testing.assert_allclose(
        result.row(-1),
        [0, 0.9, 9 / math.sqrt(95), 1 - 9 / math.sqrt(95), math.sqrt(0.07)],
        rtol=1e-13,
        atol=1e-15,
    )


def test_beta_recipe_preserves_allowed_missing_fraction_without_stats_extra() -> None:
    gapped = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, None, 5.0, 7.0]})
    result = gapped.select(recipe("SimpleBeta", inputs=["y", "x"], regression_length=4))
    assert result.item(-1, 0) == pytest.approx(2.0, rel=1e-14)


@pytest.mark.parametrize(
    "values,dtype", [(["A", "B"], pl.String), ([True, False], pl.Boolean), ([1, 2], pl.Int64)]
)
def test_latest_recipe_preserves_nonprice_input_dtypes(values, dtype) -> None:
    frame = pl.DataFrame({"value": pl.Series(values, dtype=dtype)})
    result = frame.select(recipe("Latest", inputs=["value"]).alias("value"))
    assert result.equals(frame)


@pytest.mark.parametrize(
    "name,expected", [("BusinessDaysSincePreviousEvent", 1.0), ("BusinessDaysUntilNextEvent", -1.0)]
)
def test_event_recipes_use_explicit_date_and_knowledge_roles(name, expected) -> None:
    from datetime import date, datetime

    frame = pl.DataFrame(
        {
            "event": [date(2025, 7, 4)],
            "session": [date(2025, 7, 7)],
            "known": [datetime(2025, 7, 1, tzinfo=UTC)],
            "decision": [datetime(2025, 7, 7, tzinfo=UTC)],
        }
    )
    result = frame.select(
        recipe(
            name,
            inputs=["event"],
            reference_date_col="session",
            known_at_col="known",
            decision_time_col="decision",
        )
    )
    assert result.item() == expected


@pytest.mark.parametrize(
    "name,options",
    [
        ("RSI", {"unknown": 1}),
        ("RSI", {"window_length": True}),
        ("RSI", {"inputs": "close"}),
        ("VWAP", {"inputs": ["close"], "window_length": 3}),
        ("EWMA", {"inputs": ["close"], "window_length": 3}),
        ("EWMA", {"inputs": ["close"], "window_length": 3, "span": 3, "decay_rate": 0.5}),
        ("SimpleMovingAverage", {"window_length": 3}),
        ("SimpleMovingAverage", {"inputs": ["close"]}),
        ("TrueRange", {"window_length": 3}),
        ("RSI.from_span", {"span": 3}),
        ("MACDSignal", {"window_length": 34}),
        ("Factor", {}),
    ],
)
def test_recipe_rejects_ambiguous_or_nonfunctional_requests_before_evaluation(
    name, options
) -> None:
    with pytest.raises((TypeError, ValueError)):
        recipe(name, **options)
