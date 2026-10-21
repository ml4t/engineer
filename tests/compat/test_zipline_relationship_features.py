"""Paired rolling factors must use aligned target rows, not time-trend regression."""

from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal


def paired():
    return importlib.import_module("ml4t.engineer.features.paired")


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta"])
@pytest.mark.parametrize("lazy", [False, True])
def test_paired_methods_match_independent_complete_window_oracles(method: str, lazy: bool) -> None:
    frame = pl.DataFrame({"x": [1.0, 3.0, 2.0, 4.0, 6.0], "y": [2.0, 8.0, 7.0, 9.0, 10.0]})
    expression = getattr(paired(), f"rolling_{method}")("y", "x", period=3).alias("result")
    result = (frame.lazy() if lazy else frame).select(expression)
    if lazy:
        assert isinstance(result, pl.LazyFrame)
        result = result.collect()
    expected = [np.nan, np.nan]
    for end in range(2, len(frame)):
        x, y = frame["x"].to_numpy()[end - 2 : end + 1], frame["y"].to_numpy()[end - 2 : end + 1]
        if method == "pearson":
            expected.append(np.corrcoef(x, y)[0, 1])
        elif method == "spearman":
            expected.append(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
        else:
            expected.append(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    np.testing.assert_allclose(result["result"].to_numpy(), expected, atol=1e-14, equal_nan=True)


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta"])
def test_missing_pair_is_removed_from_both_sides_and_full_row_window_remains(method: str) -> None:
    frame = pl.DataFrame({"x": [1.0, 90.0, 3.0, 4.0], "y": [2.0, None, 6.0, 8.0]})
    function = getattr(paired(), f"rolling_{method}")
    actual = frame.select(function("y", "x", period=4, allowed_missing=1).alias("result"))
    np.testing.assert_allclose(
        actual["result"].to_numpy(), [np.nan] * 3 + [2 if method == "beta" else 1], equal_nan=True
    )
    strict = frame.select(function("y", "x", period=4).alias("result")).item(-1, 0)
    assert strict is None or np.isnan(strict)
    other_side = pl.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [2.0, 90.0, 6.0, 8.0]})
    assert other_side.select(function("y", "x", period=4, allowed_missing=1)).item(
        -1, 0
    ) == pytest.approx(2 if method == "beta" else 1)


def test_spearman_recomputes_average_tie_ranks_in_each_window() -> None:
    frame = pl.DataFrame({"x": [1.0, 1.0, 4.0, 2.0, 3.0], "y": [1.0, 2.0, 2.0, 4.0, 1.0]})
    actual = frame.select(paired().rolling_spearman("y", "x", period=3))["y"].to_numpy()
    np.testing.assert_allclose(actual[2:], [0.5, 0.0, -0.5], atol=1e-14)


@pytest.mark.parametrize("lazy", [False, True])
def test_rolling_regression_returns_all_five_actual_paired_fields(lazy: bool) -> None:
    stats = pytest.importorskip("scipy.stats", reason="Regression p-values require stats extra")
    frame = pl.DataFrame({"x": [1.0, 3.0, 2.0, 4.0, 6.0], "y": [2.0, 8.0, 7.0, 9.0, 10.0]})
    result = (
        (frame.lazy() if lazy else frame)
        .select(paired().rolling_regression("y", "x", period=3).alias("regression"))
        .unnest("regression")
    )
    if lazy:
        result = result.collect()
    assert result.columns == ["alpha", "beta", "r_value", "p_value", "stderr"]
    assert all(result[c].head(2).is_nan().all() for c in result.columns)
    for end in range(2, frame.height):
        reference = stats.linregress(frame["x"].slice(end - 2, 3), frame["y"].slice(end - 2, 3))
        np.testing.assert_allclose(
            result.row(end),
            [
                reference.intercept,
                reference.slope,
                reference.rvalue,
                reference.pvalue,
                reference.stderr,
            ],
            rtol=1e-13,
            atol=0,
        )


def test_regression_missing_pairs_and_constant_target_are_explicit() -> None:
    pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    frame = pl.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [3.0, 100.0, 7.0, 9.0]})
    result = frame.select(
        paired().rolling_regression("y", "x", period=4, allowed_missing=1).alias("r")
    ).unnest("r")
    assert result["alpha"][-1] == pytest.approx(1)
    assert result["beta"][-1] == pytest.approx(2)
    constant = frame.with_columns(pl.lit(1.0).alias("x"))
    result = constant.select(paired().rolling_regression("y", "x", period=3).alias("r")).unnest("r")
    assert all(np.isnan(result[c][-1]) for c in result.columns)


def test_regression_has_independent_slope_intercept_pvalue_and_stderr_oracle() -> None:
    pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 2.0, 4.0]})
    actual = frame.select(paired().rolling_regression("y", "x", period=4).alias("r")).unnest("r")
    # Sxx=5, Sxy=4.5, Syy=4.75, residual SS=.7, df=2.
    # The two-sided Student-t df=2 tail is 1-|t|/sqrt(t*t+2).
    expected = [0.0, 0.9, 9 / np.sqrt(95), 1 - 9 / np.sqrt(95), np.sqrt(0.07)]
    np.testing.assert_allclose(actual.row(-1), expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta", "regression"])
def test_empty_and_singleton_pair_series_remain_empty_or_undefined(method: str) -> None:
    if method == "regression":
        pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    function = getattr(paired(), f"rolling_{method}")
    for count in (0, 1):
        frame = pl.DataFrame({"x": [1.0] * count, "y": [2.0] * count})
        actual = frame.select(function("y", "x", period=3).alias("r"))
        assert actual.height == count
        if count:
            if method == "regression":
                assert all(np.isnan(v) for v in actual["r"][0].values())
            else:
                assert actual["r"][0] is None or np.isnan(actual["r"][0])


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta", "regression"])
def test_paired_methods_do_not_cross_assets_or_mutate_inputs(method: str) -> None:
    if method == "regression":
        pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    frame = pl.DataFrame(
        {
            "asset": ["A", "B"] * 4,
            "x": [1.0, 10.0, 3.0, 20.0, 2.0, 40.0, 4.0, 30.0],
            "y": [2.0, 30.0, 8.0, 10.0, 7.0, 50.0, 9.0, 20.0],
        }
    )
    before = frame.clone()
    expression = getattr(paired(), f"rolling_{method}")("y", "x", period=3)
    combined = frame.with_columns(expression.over("asset").alias("result"))
    for asset in ("A", "B"):
        separate = frame.filter(pl.col("asset") == asset).with_columns(expression.alias("result"))
        assert_frame_equal(combined.filter(pl.col("asset") == asset), separate)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta", "regression"])
@pytest.mark.parametrize(
    "params",
    [
        {"period": 0},
        {"period": 1},
        {"period": True},
        {"period": 3, "allowed_missing": -1},
        {"period": 3, "allowed_missing": 4},
        {"period": 3, "allowed_missing": True},
    ],
)
def test_invalid_pair_windows_fail_before_evaluation(
    method: str, params: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        getattr(paired(), f"rolling_{method}")("y", "x", **params)
