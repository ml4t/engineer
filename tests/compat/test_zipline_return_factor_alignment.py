"""Actual price->target return->rolling factor preparation preserves keys and clocks."""

import importlib
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

DAYS = [date(2025, 7, day) for day in (7, 8, 9, 10, 11, 14, 15, 16)]


def feed(**options) -> FeedSpec:
    defaults = {
        "time_contract_version": 2,
        "calendar": None,
        "entity_col": "asset",
        "data_frequency": "1d",
        "bar_type": "time",
        "timestamp_semantics": "session_label",
    }
    return FeedSpec(**{**defaults, **options})


def run(frame, **options):
    function = importlib.import_module("ml4t.engineer.features.paired").target_return_features
    defaults = {
        "target": {"asset": "B"},
        "period": 3,
        "method": "beta",
        "price_basis": "raw",
        "feed_spec": feed(),
        "output": "market",
    }
    return function(frame, **{**defaults, **options})


def panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": DAYS[:4] * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "close": [100.0, 120.0, 80.0, 112.0, 100.0, 110.0, 99.0, 118.8],
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_actual_price_pipeline_preserves_exact_target_keys_and_independent_beta(lazy: bool) -> None:
    frame = panel().sort("timestamp", "asset", descending=[False, True])
    result = run(frame.lazy() if lazy else frame)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if lazy:
        result = result.collect()
    assert_frame_equal(result.select(frame.columns), frame)
    a = result.filter(pl.col("asset") == "A")
    np.testing.assert_allclose(
        a["market_asset_return"].to_numpy(),
        [np.nan, 0.2, -1 / 3, 0.4],
        rtol=1e-14,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        a["market_target_return"].to_numpy(),
        [np.nan, 0.1, -0.1, 0.2],
        rtol=1e-14,
        atol=0,
        equal_nan=True,
    )
    assert a["market_beta"][-1] == pytest.approx(52 / 21, rel=1e-13)
    assert result.filter(pl.col("asset") == "B")["market_beta"][-1] == pytest.approx(1.0, rel=1e-13)
    assert result["market_available_at"].null_count() == result.height
    assert result["market_price_basis"].unique().to_list() == ["raw"]


def test_common_clock_is_not_compressed_before_missing_tolerant_statistics() -> None:
    close = [100.0, 110.0, 121.0, 108.9, 130.68, 143.748, 129.3732, 155.24784]
    b = pl.DataFrame({"timestamp": DAYS, "asset": "B", "close": close})
    a = b.filter(~pl.col("timestamp").is_in(DAYS[3:5])).with_columns(pl.lit("A").alias("asset"))
    frame = pl.concat([a, b]).sort("timestamp", "asset")
    common = run(frame, period=4, allowed_missing=1, index_mode="common_grid")
    per_asset = run(frame, period=4, allowed_missing=1, index_mode="per_asset")
    at = (pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[6])
    value = common.filter(at)["market_beta"][0]
    assert value is None or np.isnan(value)
    assert np.isfinite(per_asset.filter(at)["market_beta"][0])
    missing_exit = common.filter((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[5]))
    assert missing_exit["market_asset_return"][0] is None
    assert per_asset.filter((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[5]))[
        "market_asset_return"
    ][0] == pytest.approx(0.188, rel=1e-14)
    assert common.height == frame.height


def test_named_grid_fails_before_a_gap_can_become_a_one_bar_return() -> None:
    frame = panel().filter(pl.col("timestamp") != DAYS[1])
    before = frame.clone()
    with pytest.raises(ValueError, match="missing scheduled session"):
        run(frame, calendar="NYSE")
    assert_frame_equal(frame, before)


def test_feature_availability_uses_both_assets_and_the_actual_price_dependencies() -> None:
    frame = (
        panel()
        .with_columns(
            (pl.col("timestamp").cast(pl.Datetime("ns", "UTC")) + timedelta(hours=16)).alias(
                "known"
            )
        )
        .with_columns(
            pl.when((pl.col("asset") == "B") & (pl.col("timestamp") == DAYS[3]))
            .then(pl.col("known") + timedelta(hours=4))
            .otherwise(pl.col("known"))
            .alias("known")
        )
    )
    result = run(frame, available_at_col="known")
    assert result.filter(pl.col("asset") == "A")["market_available_at"][-1] == datetime(
        2025, 7, 10, 20, tzinfo=UTC
    )
    earlier = run(frame.filter(pl.col("timestamp") <= DAYS[2]), method="pearson")
    assert earlier["market_available_at"].null_count() == earlier.height


def test_custom_roles_compound_identity_and_regression_fields_survive_the_pipeline() -> None:
    pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    frame = (
        panel()
        .with_columns(
            pl.struct(pl.lit("bundle").alias("namespace"), pl.col("asset").alias("sid")).alias(
                "asset"
            )
        )
        .rename({"timestamp": "when", "asset": "identity", "close": "quote"})
    )
    contract = FeedSpec(
        time_contract_version=2,
        calendar=None,
        entity_col="identity",
        timestamp_col="when",
        price_col="quote",
        timestamp_semantics="session_label",
        data_frequency="1d",
    )
    output = run(
        frame,
        feed_spec=contract,
        target={"identity": {"namespace": "bundle", "sid": "B"}},
        method="regression",
    )
    assert output.schema["identity"] == frame.schema["identity"]
    row = output.filter(pl.col("identity").struct.field("sid") == "A")["market_regression"][-1]
    assert row["beta"] == pytest.approx(52 / 21, rel=1e-13)
    assert row["alpha"] == pytest.approx(-8 / 105, rel=1e-13)


@pytest.mark.parametrize(
    "options",
    [
        {"target": {"asset": "missing"}},
        {"target": {}},
        {"period": True},
        {"method": "time_trend"},
        {"price_basis": ""},
        {"calendar": "NYSE", "feed_spec": FeedSpec()},
        {"entity_col": "another"},
        {"timestamp_col": "another"},
        {"price_col": "another"},
        {"index_mode": "ffill"},
        {"window_unit": "elapsed"},
    ],
)
def test_target_contract_errors_are_atomic(options) -> None:
    frame = panel()
    before = frame.clone()
    with pytest.raises((TypeError, ValueError)):
        run(frame, **options)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("method", ["pearson", "spearman", "beta", "regression"])
def test_empty_frame_retains_declared_added_schema(method: str) -> None:
    if method == "regression":
        pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    for frame in (panel().clear(), panel().clear().lazy()):
        result = run(frame, method=method)
        result = result.collect() if isinstance(result, pl.LazyFrame) else result
        assert result.height == 0
        assert result.schema["market_available_at"] == pl.Datetime("ns", "UTC")
        assert result.schema["market_" + method].base_type() == (
            pl.Struct if method == "regression" else pl.Float64
        )


def test_appended_future_prices_do_not_change_prior_prepared_values() -> None:
    before = panel()
    later = pl.DataFrame({"timestamp": [DAYS[4]] * 2, "asset": ["A", "B"], "close": [1e9, 0.001]})
    expected = run(before)
    actual = run(pl.concat([before, later])).filter(pl.col("timestamp") <= DAYS[3])
    assert_frame_equal(actual, expected)
    # Lazy filtering must not remove warmup or the target before preparation.
    selected = (
        run(before.lazy())
        .filter((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[3]))
        .collect()
    )
    assert selected["market_beta"][0] == pytest.approx(52 / 21)


def test_adjacent_uint64_ticks_survive_actual_price_to_factor_preparation() -> None:
    frame = panel().with_columns(
        pl.Series("close", [2**63 + v for v in (0, 2, 6, 12, 0, 1, 3, 6)], dtype=pl.UInt64)
    )
    result = run(frame)
    a = result.filter(pl.col("asset") == "A")
    np.testing.assert_allclose(
        a["market_asset_return"].tail(3), [2 / 2**63, 4 / (2**63 + 2), 6 / (2**63 + 6)], rtol=1e-14
    )
    assert a["market_beta"][-1] == pytest.approx(2.0, rel=1e-13)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_supplied_prices_raise_or_become_explicit_missing_pairs(bad) -> None:
    frame = panel().with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[1]))
        .then(pl.lit(bad))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(ValueError, match="finite, positive"):
        run(frame)
    actual = run(frame, missing_price_policy="null")
    a = actual.filter(pl.col("asset") == "A")
    assert a["market_asset_return"][1] is None
    assert a["market_asset_return"][2] is None
    assert not np.isfinite(a["market_beta"][-1]) if a["market_beta"][-1] is not None else True


def test_multicolumn_identity_does_not_join_reused_tickers_across_namespaces() -> None:
    frame = pl.concat(
        [
            panel().with_columns(pl.lit("one").alias("venue")),
            panel().with_columns(pl.lit("two").alias("venue")),
        ]
    )
    contract = FeedSpec(time_contract_version=2, calendar=None, entity_col=["venue", "asset"])
    output = run(frame, feed_spec=contract, target={"venue": "one", "asset": "B"})
    assert_frame_equal(output.select(frame.columns), frame)
    assert output.filter((pl.col("asset") == "A") & (pl.col("venue") == "two"))["market_beta"][
        -1
    ] == pytest.approx(52 / 21)


@pytest.mark.parametrize(
    "options",
    [
        {"returns_periods": 0},
        {"period": 2**64},
        {"returns_periods": 2**64},
        {"output": ""},
        {"missing_price_policy": "ffill"},
        {"entity_col": []},
        {"target": {"asset": None}},
        {"target": {"asset": "B", "other": 1}},
        {"window_unit": "expected_bar"},
        {"allowed_missing": -1},
        {"feed_spec": {"timestamp_col": "timestamp", "time_col": "other"}},
        {"feed_spec": None, "entity_col": []},
        {"feed_spec": None, "price_col": "timestamp"},
    ],
)
def test_invalid_preparation_options_fail_before_mutating_input(options) -> None:
    before = panel()
    with pytest.raises((TypeError, ValueError)):
        run(before, **options)
    assert_frame_equal(before, panel())


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "null_key",
        "missing_column",
        "string_time",
        "string_price",
        "collision",
        "internal",
    ],
)
def test_invalid_physical_frames_fail_with_explicit_errors(kind: str) -> None:
    frame = panel()
    if kind == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif kind == "null_key":
        frame = frame.with_columns(pl.lit(None).alias("asset"))
    elif kind == "missing_column":
        frame = frame.drop("close")
    elif kind == "string_time":
        frame = frame.with_columns(pl.col("timestamp").cast(pl.String))
    elif kind == "string_price":
        frame = frame.with_columns(pl.col("close").cast(pl.String))
    else:
        frame = frame.with_columns(
            pl.lit(1).alias("market_beta" if kind == "collision" else "__ml4t_target_quote")
        )
    with pytest.raises(ValueError):
        run(frame)


def test_unknown_and_out_of_range_publication_instants_are_not_fabricated_or_wrapped() -> None:
    frame = panel().with_columns(pl.lit(None, pl.Datetime("ns", "UTC")).alias("known"))
    assert run(frame, available_at_col="known")["market_available_at"].null_count() == frame.height
    naive = frame.with_columns(pl.lit(datetime(2025, 7, 7)).alias("known"))
    with pytest.raises(ValueError, match="timezone-aware"):
        run(naive, available_at_col="known")
    wide = frame.with_columns(pl.lit(datetime(2500, 1, 1, tzinfo=UTC)).alias("known"))
    with pytest.raises(ValueError, match="range"):
        run(wide, available_at_col="known")


def test_named_calendar_and_explicit_none_precedence_reuse_the_actual_grid_validator() -> None:
    frame = panel()
    contract = feed(calendar="NYSE")
    actual = run(frame, feed_spec=contract)
    assert actual["market_beta"][-1] == pytest.approx(1.0)
    sparse = frame.filter(pl.col("timestamp") != DAYS[1])
    with pytest.raises(ValueError, match="missing scheduled session"):
        run(sparse, feed_spec=contract)
    result = run(sparse, feed_spec=contract, calendar=None)
    assert result.height == sparse.height
    result = run(frame, feed_spec=None, calendar=None, entity_col="asset")
    assert result.filter(pl.col("asset") == "A")["market_beta"][-1] == pytest.approx(52 / 21)


@pytest.mark.parametrize(
    "name,method,length",
    [
        ("RollingPearsonOfReturns", "pearson", "correlation_length"),
        ("RollingSpearmanOfReturns", "spearman", "correlation_length"),
        ("RollingLinearRegressionOfReturns", "regression", "regression_length"),
        ("SimpleBeta", "beta", "regression_length"),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_return_factor_recipe_uses_native_keyed_preparation_without_masking_target(
    name, method, length, lazy
) -> None:
    if method == "regression":
        pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    function = importlib.import_module("ml4t.engineer.compat.zipline").return_factor_features
    frame = panel()
    output = function(
        frame.lazy() if lazy else frame,
        name,
        target={"asset": "B"},
        price_basis="raw",
        feed_spec=feed(),
        output="market",
        mask=pl.col("asset") == "A",
        **{length: 3},
    )
    assert isinstance(output, pl.LazyFrame if lazy else pl.DataFrame)
    output = output.collect() if lazy else output
    expected = run(frame, method=method)
    assert_frame_equal(
        output.filter(pl.col("asset") == "A"), expected.filter(pl.col("asset") == "A")
    )
    assert output.filter(pl.col("asset") == "B")["market_" + method].null_count() == 4
    assert_frame_equal(output.select(frame.columns), frame)


def test_return_recipe_translates_price_lookback_and_retains_structural_warmup() -> None:
    function = importlib.import_module("ml4t.engineer.compat.zipline").return_factor_features
    close = [100.0, 110.0, 121.0, 108.9, 130.68, 143.748, 129.3732, 155.24784]
    frame = pl.DataFrame(
        {"timestamp": DAYS * 2, "asset": ["A"] * 8 + ["B"] * 8, "close": close + close}
    )
    result = function(
        frame,
        "RollingPearsonOfReturns",
        target={"asset": "B"},
        returns_length=3,
        correlation_length=3,
        price_basis="raw",
        feed_spec=feed(),
        output="market",
    )
    a = result.filter(pl.col("asset") == "A")
    assert a["market_asset_return"][2] == pytest.approx(0.21)
    assert a["market_pearson"].head(4).is_null().all()
    np.testing.assert_allclose(a["market_pearson"].tail(4), 1.0, atol=1e-14)
    beta = function(
        frame,
        "SimpleBeta",
        target={"asset": "B"},
        regression_length=4,
        allowed_missing_percentage=1.0,
        price_basis="raw",
        feed_spec=feed(),
    )
    assert beta.filter(pl.col("asset") == "A")["target_beta"].head(4).is_null().all()


@pytest.mark.parametrize(
    "name,options",
    [
        ("unknown", {}),
        ("SimpleBeta", {"regression_length": 3, "returns_length": 3}),
        ("SimpleBeta", {"regression_length": 2}),
        ("SimpleBeta", {"regression_length": 3, "allowed_missing_percentage": -1}),
        ("RollingPearsonOfReturns", {"correlation_length": 3, "returns_length": 1}),
        ("RollingPearsonOfReturns", {"regression_length": 3}),
        ("RollingSpearmanOfReturns", {"correlation_length": 3, "allowed_missing_percentage": 0.2}),
        ("RollingLinearRegressionOfReturns", {"regression_length": 3, "correlation_length": 3}),
        ("SimpleBeta", {"regression_length": 3, "mask": 17}),
    ],
)
def test_invalid_legacy_return_options_fail_explicitly(name, options) -> None:
    function = importlib.import_module("ml4t.engineer.compat.zipline").return_factor_features
    with pytest.raises((TypeError, ValueError)):
        function(
            panel(), name, target={"asset": "B"}, price_basis="raw", feed_spec=feed(), **options
        )
