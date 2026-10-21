"""Dated eligible cross-sections preserve stable ties and original missingness."""

import importlib
from datetime import date

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal


def selection():
    return importlib.import_module("ml4t.engineer.features.selection")


def panel():
    return pl.DataFrame(
        {
            "timestamp": [date(2025, 7, 7)] * 6,
            "asset": ["D", "B", "A", "C", "E", "F"],
            "value": [3.0, 1.0, 1.0, 2.0, float("nan"), 999.0],
            "sector": ["x", "x", "x", "y", "x", "x"],
            "eligible": [True] * 5 + [False],
        }
    )


def calculate(frame, method, **options):
    return frame.with_columns(
        selection()
        .cross_section("value", method=method, groupby="sector", mask="eligible", **options)
        .alias("result")
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "ties,expected",
    [
        ("ordinal", [3.0, 2.0, 1.0, 1.0]),
        ("min", [3.0, 1.0, 1.0, 1.0]),
        ("max", [3.0, 2.0, 2.0, 1.0]),
        ("dense", [2.0, 1.0, 1.0, 1.0]),
        ("average", [3.0, 1.5, 1.5, 1.0]),
    ],
)
def test_rank_ties_use_stable_ids_inside_the_actual_eligible_group(lazy, ties, expected):
    frame = panel()
    actual = calculate(frame.lazy() if lazy else frame, "rank", ties=ties)
    assert isinstance(actual, pl.LazyFrame if lazy else pl.DataFrame)
    actual = actual.collect() if lazy else actual
    np.testing.assert_allclose(actual["result"].head(4), expected)
    assert actual["result"].tail(2).null_count() == 2
    assert_frame_equal(actual.select(frame.columns), frame)
    shuffled = calculate(frame.reverse(), "rank", ties=ties).sort("asset")
    assert_frame_equal(shuffled, actual.sort("asset"))


@pytest.mark.parametrize(
    "method,expected",
    [
        ("demean", [4 / 3, -2 / 3, -2 / 3, 0.0]),
        ("zscore", [np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2), np.nan]),
    ],
)
def test_group_transforms_use_only_contemporary_eligible_values(method, expected):
    output = calculate(panel(), method)
    np.testing.assert_allclose(output["result"].head(4), expected, rtol=1e-14, equal_nan=True)
    assert output["result"].tail(2).null_count() == 2
    future = panel().with_columns(
        pl.lit(date(2025, 7, 8)).alias("timestamp"), (pl.col("value") * 100).alias("value")
    )
    combined = calculate(pl.concat([panel(), future]), method).head(6)
    assert_frame_equal(output, combined)


def test_undefined_numeric_classification_is_not_a_real_contemporary_group():
    frame = (
        panel()
        .with_columns(pl.when(pl.col("sector") == "x").then(0.0).otherwise(1.0).alias("sector"))
        .with_columns((pl.col("sector") / pl.col("sector")).alias("sector"))
    )
    result = calculate(frame, "rank")
    assert result["result"].to_list() == [None, None, None, 1.0, None, None]


@pytest.mark.parametrize(
    "method,expected",
    [
        ("top", [True, False, False, True, False, False]),
        ("bottom", [False, False, True, True, False, False]),
    ],
)
def test_top_bottom_use_exact_cardinality_and_stable_tie_selection(method, expected):
    output = calculate(panel(), method, n=1)
    assert output["result"].to_list() == expected
    assert calculate(panel(), method, n=0)["result"].to_list() == [False] * 6


def test_negative_group_maximum_uses_numeric_order_not_float_storage_bits():
    frame = panel().with_columns((-pl.col("value")).alias("value"))
    result = calculate(frame, "top", n=1)
    assert result["result"].to_list() == [False, False, True, True, False, False]


def test_winsorization_uses_legacy_order_statistic_cutoffs_not_linear_clipping():
    frame = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, 7)] * 5,
            "asset": list("ABCDE"),
            "value": [0.0, 1.0, 2.0, 100.0, np.nan],
        }
    )
    result = frame.select(
        selection().cross_section(
            "value", method="winsorize", min_percentile=0.25, max_percentile=0.75
        )
    )
    np.testing.assert_allclose(result.to_series(), [1.0, 1.0, 2.0, 2.0, np.nan], equal_nan=True)


def test_quantiles_use_linear_edges_and_percentile_filter_includes_both_bounds():
    frame = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, 7)] * 6,
            "asset": list("ABCDEF"),
            "value": [0.0, 1.0, 2.0, 3.0, np.nan, None],
        }
    )
    result = frame.with_columns(
        selection().cross_section("value", method="quantiles", bins=2).alias("quantile"),
        selection()
        .cross_section(
            "value", method="percentile_between", min_percentile=0.25, max_percentile=0.75
        )
        .alias("between"),
    )
    assert result["quantile"].to_list() == [0, 0, 1, 1, None, None]
    assert result["between"].to_list() == [False, True, True, False, False, False]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "values,bins,selected",
    [
        ([-1e308, 0.0, 1e308], [0, 0, 1], [False, True, False]),
        ([-1e308, 1e308], [0, 1], [False, False]),
        ([-1e100, 0.0, 1e-100, 1e100], [0, 0, 1, 1], [False, True, True, False]),
    ],
)
def test_linear_quantile_selection_preserves_finite_opposite_extremes_and_local_scale(
    lazy, values, bins, selected
):
    frame = pl.DataFrame(
        {"timestamp": [1] * len(values), "asset": range(len(values)), "value": values}
    )
    result = (frame.lazy() if lazy else frame).with_columns(
        selection().cross_section("value", method="quantiles", bins=2).alias("bin"),
        selection()
        .cross_section(
            "value", method="percentile_between", min_percentile=0.25, max_percentile=0.75
        )
        .alias("selected"),
    )
    result = result.collect() if lazy else result
    assert result["bin"].to_list() == bins
    assert result["selected"].to_list() == selected


def test_integer_percentile_bounds_do_not_truncate_fractional_value_edges():
    frame = pl.DataFrame({"timestamp": [1] * 3, "asset": [1, 2, 3], "value": [-0.5, 0.25, 0.75]})
    expression = selection().cross_section(
        "value", method="percentile_between", min_percentile=0, max_percentile=1
    )
    assert frame.select(expression).to_series().to_list() == [True, True, True]
    assert frame.clear().select(expression).schema["value"] == pl.Boolean


@pytest.mark.parametrize("method,options", [("quantiles", {"bins": 2}), ("percentile_between", {})])
def test_nonfinite_quantile_inputs_fail_explicitly_without_partial_selection(method, options):
    frame = pl.DataFrame({"timestamp": [1, 1], "asset": [1, 2], "value": [1.0, float("inf")]})
    with pytest.raises(ValueError, match="finite"):
        frame.with_columns(selection().cross_section("value", method=method, **options))
    tied = frame.with_columns(pl.lit(1.0).alias("value"))
    with pytest.raises((ValueError, pl.exceptions.DuplicateError)):
        tied.select(selection().cross_section("value", method="quantiles", bins=4))


def test_peer_counts_keep_supplied_classifier_nulls_and_exclude_ineligible_members():
    frame = panel().with_columns(
        pl.when(pl.col("asset") == "E").then(None).otherwise(pl.col("sector")).alias("sector")
    )
    output = frame.select(selection().cross_section("sector", method="peer_count", mask="eligible"))
    np.testing.assert_allclose(
        output.to_series(), [3.0, 3.0, 3.0, 1.0, np.nan, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("method", ["demean", "zscore"])
def test_adjacent_large_integer_cross_sections_do_not_lose_tick_dispersion(method):
    frame = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, 7)] * 3,
            "asset": list("ABC"),
            "value": pl.Series([2**63, 2**63 + 1, 2**63 + 2], dtype=pl.UInt64),
        }
    )
    result = frame.select(selection().cross_section("value", method=method)).to_series()
    expected = [-1.0, 0.0, 1.0] if method == "demean" else [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("method", ["any", "all", "at_least"])
def test_rolling_boolean_filters_keep_full_window_and_explicit_missing_false(method):
    frame = pl.DataFrame({"value": [True, None, False, True, True]})
    kwargs = {"minimum": 2} if method == "at_least" else {}
    result = frame.select(
        selection().rolling_filter("value", period=3, method=method, **kwargs)
    ).to_series()
    expected = (
        [None, None, True, True, True]
        if method == "any"
        else (
            [None, None, False, False, False]
            if method == "all"
            else [None, None, False, False, True]
        )
    )
    assert result.to_list() == expected


@pytest.mark.parametrize(
    "options",
    [
        {"method": "unknown"},
        {"method": "rank", "ties": "random"},
        {"method": "top", "n": -1},
        {"method": "top", "n": True},
        {"method": "quantiles", "bins": 0},
        {"method": "quantiles", "bins": True},
        {"method": "winsorize", "min_percentile": 0.8, "max_percentile": 0.2},
    ],
)
def test_invalid_selection_contracts_fail_before_output(options):
    with pytest.raises((ValueError, TypeError)):
        panel().select(selection().cross_section("value", **options))


@pytest.mark.parametrize("scale", [np.ldexp(1.0, -1000), np.ldexp(1.0, 1000)])
def test_representable_group_zscores_do_not_underflow_or_overflow_in_variance(scale):
    frame = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, 7)] * 3,
            "asset": list("ABC"),
            "value": [scale, 2 * scale, 3 * scale],
        }
    )
    actual = frame.select(selection().cross_section("value", method="zscore")).to_series()
    np.testing.assert_allclose(actual, [-np.sqrt(1.5), 0.0, np.sqrt(1.5)], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("demean", [-1e308, 1e308]),
        ("zscore", [-1.0, 1.0]),
    ],
)
def test_opposite_extreme_group_values_do_not_overflow_a_representable_transform(method, expected):
    frame = pl.DataFrame(
        {"timestamp": [date(2025, 7, 7)] * 2, "asset": list("AB"), "value": [-1e308, 1e308]}
    )
    actual = frame.select(selection().cross_section("value", method=method)).to_series()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        (pl.Float64, [1.0, np.nan, None, np.inf], [False, True, True, False]),
        (pl.Int64, [1, -1, None, 0], [False, False, True, False]),
        (pl.String, ["a", "", None, "-1"], [False, False, True, False]),
    ],
)
def test_native_missing_predicate_does_not_invent_classifier_sentinels(dtype, values, expected):
    actual = pl.DataFrame({"x": pl.Series(values, dtype=dtype)}).select(selection().is_missing("x"))
    assert actual.to_series().to_list() == expected


@pytest.mark.parametrize(
    "method",
    ["rank", "demean", "zscore", "winsorize", "quantiles", "percentile_between", "peer_count"],
)
def test_empty_all_missing_and_singleton_cross_sections_keep_defined_shapes(method):
    kwargs = {"bins": 1} if method == "quantiles" else {}
    frame = pl.DataFrame(
        {"timestamp": [date(2025, 7, 7)], "asset": ["A"], "value": [None]},
        schema_overrides={"value": pl.Float64},
    )
    for source in (frame, frame.clear()):
        result = source.select(selection().cross_section("value", method=method, **kwargs))
        assert result.height == source.height
        if source.height:
            assert (
                result.item() is None if method != "percentile_between" else result.item() is False
            )
    present = frame.with_columns(pl.lit(3.0).alias("value"))
    result = present.select(selection().cross_section("value", method=method, **kwargs)).item()
    if method == "zscore":
        assert np.isnan(result)
    else:
        assert (
            result
            == {
                "rank": 1.0,
                "demean": 0.0,
                "winsorize": 3.0,
                "quantiles": 0,
                "percentile_between": True,
                "peer_count": 1.0,
            }[method]
        )


@pytest.mark.parametrize(
    "options",
    [
        {"method": "demean", "n": 1},
        {"method": "demean", "bins": 2},
        {"method": "top", "n": 1, "ties": "dense"},
        {"method": "rank", "ascending": 1},
        {"method": "zscore", "ddof": -1},
        {"method": "rank", "ddof": 1},
        {"method": "rank", "min_percentile": 0.1},
        {"method": "winsorize", "max_percentile": np.nan},
        {"method": "rank", "entity_col": []},
        {"method": "rank", "groupby": [""]},
        {"method": "rank", "mask": 42},
    ],
)
def test_inapplicable_or_invalid_options_are_not_silently_ignored(options):
    with pytest.raises((ValueError, TypeError)):
        panel().select(selection().cross_section("value", **options))


@pytest.mark.parametrize(
    "options",
    [
        {"period": 0, "method": "any"},
        {"period": 2, "method": "unknown"},
        {"period": 2, "method": "all", "minimum": 1},
        {"period": 2, "method": "at_least", "minimum": 3},
        {"period": 2, "method": "at_least", "minimum": True},
    ],
)
def test_invalid_boolean_window_contracts_fail_explicitly(options):
    with pytest.raises(ValueError):
        pl.DataFrame({"x": [True, False]}).select(selection().rolling_filter("x", **options))


def test_boolean_window_rejects_numeric_predicates_and_respects_asset_groups():
    with pytest.raises(ValueError, match="boolean"):
        pl.DataFrame({"x": [1, 0]}).select(selection().rolling_filter("x", period=2, method="any"))
    frame = pl.DataFrame({"asset": ["A", "B"] * 3, "x": [True, False, True, False, True, False]})
    result = frame.select(selection().rolling_filter("x", period=2, method="all").over("asset"))
    assert result.to_series().to_list() == [None, None, True, False, True, False]
