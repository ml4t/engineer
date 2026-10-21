"""Actual optional inputs preserve their units, coverage and decision information set."""

import importlib
from datetime import UTC, date, datetime

import polars as pl
import pytest
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

DAYS = [date(2025, 7, d) for d in (7, 8, 9, 10, 11)]


def inputs():
    return importlib.import_module("ml4t.engineer.features.inputs")


def feed(**options):
    return FeedSpec(
        **{"time_contract_version": 2, "calendar": None, "entity_col": "asset", **options}
    )


def market():
    return pl.DataFrame(
        {
            "timestamp": DAYS * 2,
            "asset": [17] * 5 + [23] * 5,
            "close": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 50.0, 25.0, 12.5, 6.25],
            "volume": [100.0, 0.0, None, 20.0, 10.0, 10.0, 20.0, 40.0, 80.0, 160.0],
            "available_at": [datetime(2025, 7, d.day, 20, tzinfo=UTC) for d in DAYS] * 2,
            "decision_time": [datetime(2025, 7, d.day, 13, tzinfo=UTC) for d in DAYS] * 2,
            "currency": ["USD"] * 10,
            "multiplier": [1.0] * 10,
            "volume_unit": ["shares"] * 10,
        }
    )


def prepare_market(frame, **options):
    return inputs().prepare_market_inputs(
        frame,
        **{
            "feed_spec": feed(),
            "price_basis": "raw",
            "volume_basis": "raw",
            "purpose": "causal",
            "liquidity_window": 2,
            "completed_at_col": "available_at",
            **options,
        },
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_lagged_liquidity_uses_completed_price_volume_products_once_not_current_price(lazy):
    original = market().sort("timestamp", "asset", descending=[False, True])
    result = prepare_market(original.lazy() if lazy else original)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    result = result.collect() if lazy else result
    assert_frame_equal(result.select(original.columns), original)
    a, b = (result.filter(pl.col("asset") == asset) for asset in (17, 23))
    assert a["notional_volume"].to_list() == [None] * 5
    assert a["lagged_mean_notional_volume"].to_list() == [None, None, 500.0, None, None]
    assert b["lagged_mean_notional_volume"].to_list() == [None, None, 1000.0, 1000.0, 1000.0]
    assert a["liquidity_covered"].to_list() == [False, False, True, False, False]
    assert a["liquidity_available_at"][2] == datetime(2025, 7, 8, 20, tzinfo=UTC)
    assert a["liquidity_lag"].to_list() == [1] * 5
    assert_frame_equal(original, market().sort("timestamp", "asset", descending=[False, True]))


def test_retrospective_raw_products_distinguish_real_zero_from_uncovered_volume():
    result = prepare_market(market(), purpose="retrospective")
    assert result.filter(pl.col("asset") == 17)["notional_volume"].to_list() == [
        1000.0,
        0.0,
        None,
        800.0,
        500.0,
    ]
    assert result["preparation_purpose"].unique().to_list() == ["retrospective"]
    no_volume = prepare_market(market().drop("volume"))
    assert no_volume["lagged_mean_notional_volume"].null_count() == no_volume.height
    assert_frame_equal(
        no_volume.select("timestamp", "asset", "close"),
        market().select("timestamp", "asset", "close"),
    )


def test_explicit_contract_multiplier_and_split_compatible_basis_preserve_notional_units():
    contracts = market().with_columns(
        pl.lit(50.0).alias("multiplier"), pl.lit("contracts").alias("volume_unit")
    )
    result = prepare_market(contracts)
    assert result.filter(pl.col("asset") == 23)["lagged_mean_notional_volume"][2] == 50000.0
    split = market().with_columns(
        (pl.col("close") / 2).alias("close"), (pl.col("volume") * 2).alias("volume")
    )
    result = prepare_market(split, price_basis="split_adjusted", volume_basis="split_adjusted")
    assert result.filter(pl.col("asset") == 17)["lagged_mean_notional_volume"][2] == 500.0
    with pytest.raises(ValueError, match="basis"):
        prepare_market(split, price_basis="split_adjusted", volume_basis="raw")


def test_delayed_publication_or_uncompleted_bar_cannot_supply_prior_causal_liquidity():
    delayed = market().with_columns(
        pl.when((pl.col("asset") == 17) & (pl.col("timestamp") == DAYS[1]))
        .then(pl.lit(datetime(2025, 7, 10, tzinfo=UTC)))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    result = prepare_market(delayed)
    assert result.filter(pl.col("asset") == 17)["lagged_mean_notional_volume"][2] is None
    provisional = market().with_columns(pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("finished"))
    assert not prepare_market(provisional, completed_at_col="finished")["liquidity_covered"].any()


def test_native_market_validation_without_derived_features_preserves_the_input_exactly():
    result = prepare_market(market(), liquidity_window=None)
    assert_frame_equal(result, market())


def requests():
    return pl.DataFrame(
        {
            "timestamp": [DAYS[1], DAYS[2], DAYS[3]] * 2,
            "asset": [17] * 3 + [23] * 3,
            "decision_time": [datetime(2025, 7, d.day, 13, tzinfo=UTC) for d in DAYS[1:4]] * 2,
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_loading_coverage_uses_requested_factors_and_actual_publication_not_zero_fill(lazy):
    facts = pl.DataFrame(
        {
            "timestamp": [DAYS[1], DAYS[2], DAYS[3]],
            "asset": [17] * 3,
            "factor": ["MKT"] * 3,
            "loading": [1.0, 3.0, 2.0],
            "available_at": [datetime(2025, 7, d, 12, tzinfo=UTC) for d in (8, 10, 10)],
        }
    )
    asked = requests().lazy() if lazy else requests()
    result = inputs().prepare_loading_inputs(
        facts, requests=asked, factors=["MKT", "SIZE"], feed_spec=feed(), purpose="causal"
    )
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    result = result.collect() if lazy else result
    assert result.height == 12
    a = result.filter((pl.col("asset") == 17) & (pl.col("factor") == "MKT"))
    assert a["loading"].to_list() == [1.0, None, 2.0]
    assert a["loading_covered"].to_list() == [True, False, True]
    assert a["loading_known_at_decision"].to_list() == [True, False, True]
    assert result.filter(pl.col("factor") == "SIZE")["loading"].null_count() == 6
    assert result.filter(pl.col("asset") == 23)["loading"].null_count() == 6
    retrospective = inputs().prepare_loading_inputs(
        facts, requests=requests(), factors=["MKT"], feed_spec=feed(), purpose="retrospective"
    )
    a = retrospective.filter(pl.col("asset") == 17)
    assert a["loading"].to_list() == [1.0, 3.0, 2.0]
    assert a["loading_known_at_decision"].to_list() == [True, False, True]


def test_dated_classifications_never_carry_expired_or_future_published_categories():
    facts = pl.DataFrame(
        {
            "asset": [17, 17],
            "effective_from": [DAYS[0], DAYS[2]],
            "effective_to": pl.Series([DAYS[2], None], dtype=pl.Date),
            "available_at": [datetime(2025, 7, d, 12, tzinfo=UTC) for d in (7, 10)],
            "sector": ["tech", "bank"],
        }
    )
    result = inputs().prepare_classification_inputs(
        facts, requests=requests(), feed_spec=feed(), purpose="causal"
    )
    a = result.filter(pl.col("asset") == 17)
    assert a["sector"].to_list() == ["tech", None, "bank"]
    assert a["sector_covered"].to_list() == [True, False, True]
    assert result.filter(pl.col("asset") == 23)["sector"].null_count() == 3
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        inputs().prepare_classification_inputs(
            pl.concat([facts, facts.head(1)]),
            requests=requests(),
            feed_spec=feed(),
            purpose="causal",
        )


def test_static_sector_maps_require_explicit_static_and_causal_knowledge_declarations():
    with pytest.raises(ValueError, match="static"):
        inputs().prepare_classification_inputs(
            {17: "tech"}, requests=requests(), feed_spec=feed(), purpose="causal"
        )
    with pytest.raises(ValueError, match="known"):
        inputs().prepare_classification_inputs(
            {17: "tech"}, requests=requests(), feed_spec=feed(), purpose="causal", static=True
        )
    result = inputs().prepare_classification_inputs(
        {17: "tech"},
        requests=requests(),
        feed_spec=feed(),
        purpose="causal",
        static=True,
        static_known_at=datetime(2025, 7, 1, tzinfo=UTC),
    )
    assert result.filter(pl.col("asset") == 17)["sector"].to_list() == ["tech"] * 3
    assert result.filter(pl.col("asset") == 23)["sector_covered"].to_list() == [False] * 3


def test_legacy_market_conversion_preserves_integer_ticks_adjacent_instants_and_missing_volume():
    import pandas as pd

    stamps = pd.to_datetime([1_750_000_000_000_000_001, 1_750_000_000_000_000_002], utc=True)
    legacy = pd.DataFrame(
        {2**63 + 17: pd.array([2**53, 0, 2**53 + 1, None], dtype="UInt64")},
        index=pd.MultiIndex.from_product([stamps, ["price", "volume"]]),
    )
    result = inputs().market_to_long(legacy, feed_spec=feed())
    assert result["timestamp"].cast(pl.Int64).to_list() == stamps.asi8.tolist()
    assert result["asset"].dtype == pl.UInt64
    assert result["asset"].to_list() == [2**63 + 17] * 2
    assert result["close"].to_list() == [2**53, 2**53 + 1]
    assert result["volume"].to_list() == [0, None]
    assert "available_at" not in result.columns


def test_legacy_loading_conversion_retains_explicit_missing_cells_without_filling_dates():
    import pandas as pd

    legacy = pd.DataFrame(
        {"MKT": [1.0, None], "SIZE": [0.0, 2.0]},
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2025-07-08", "2025-07-10"], utc=True), [17, 23]]
        ),
    )
    result = inputs().loadings_to_long(legacy, feed_spec=feed())
    assert result.height == 4
    assert result["asset"].to_list() == [17, 23, 17, 23]
    assert result["factor"].to_list() == ["MKT", "MKT", "SIZE", "SIZE"]
    assert result["loading"].to_list() == [1.0, None, 0.0, 2.0]
    assert "available_at" not in result.columns


@pytest.mark.parametrize("lazy", [False, True])
def test_absent_optional_facts_retain_request_coverage_without_touching_price_features(lazy):
    original = requests().with_columns(pl.lit(123.0).alias("price_feature"))
    asked = original.lazy() if lazy else original
    loadings = inputs().prepare_loading_inputs(
        None, requests=asked, factors=["MKT", "SIZE"], feed_spec=feed(), purpose="causal"
    )
    sectors = inputs().prepare_classification_inputs(
        None, requests=asked, feed_spec=feed(), purpose="causal"
    )
    loadings = loadings.collect() if lazy else loadings
    sectors = sectors.collect() if lazy else sectors
    assert loadings.height == 12
    assert loadings["loading"].null_count() == 12
    assert not loadings["loading_covered"].any()
    assert sectors["sector"].null_count() == 6
    assert not sectors["sector_covered"].any()
    assert_frame_equal(sectors.select(original.columns), original)
    assert loadings["price_feature"].to_list() == [123.0] * 12


@pytest.mark.parametrize("lazy", [False, True])
def test_future_loading_revisions_cannot_change_earlier_causal_values_or_infer_assets(lazy):
    facts = pl.DataFrame(
        {
            "timestamp": [DAYS[1]],
            "asset": [17],
            "factor": ["MKT"],
            "loading": [1.0],
            "available_at": [datetime(2025, 7, 8, 12, tzinfo=UTC)],
        }
    )

    def prepare(frame):
        result = inputs().prepare_loading_inputs(
            frame.lazy() if lazy else frame,
            requests=requests().lazy() if lazy else requests(),
            factors=["MKT"],
            feed_spec=feed(),
            purpose="causal",
        )
        return result.collect() if lazy else result

    expected = prepare(facts)
    future = facts.with_columns(
        pl.lit(999.0).alias("loading"),
        pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"),
    )
    unknown = facts.with_columns(pl.lit(999, pl.Int64).alias("asset"))
    assert_frame_equal(prepare(pl.concat([facts, future, unknown])), expected)
    with pytest.raises(ValueError, match="Ambiguous"):
        prepare(pl.concat([facts, facts]))


def test_legacy_asset_objects_require_explicit_compound_stable_identity_mapping():
    import pandas as pd

    opaque = object()
    original = pd.DataFrame(
        {opaque: [10.0, 20.0]},
        index=pd.MultiIndex.from_product(
            [pd.to_datetime(["2025-07-08"], utc=True), ["price", "volume"]]
        ),
    )
    with pytest.raises((ValueError, TypeError, pl.exceptions.ComputeError)):
        inputs().market_to_long(original, feed_spec=feed())
    result = inputs().market_to_long(
        original,
        feed_spec=feed(entity_col=("venue", "sid")),
        entity_map={opaque: ("XNYS", 17)},
    )
    assert result.select("venue", "sid", "close", "volume").rows() == [("XNYS", 17, 10.0, 20.0)]


@pytest.mark.parametrize(
    "converter,required",
    [
        ("market_to_long", {"close": [10.0, 11.0]}),
        ("loadings_to_long", {"factor": ["MKT", "MKT"], "loading": [1.0, 2.0]}),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_native_conversion_preserves_order_dtypes_and_laziness_but_rejects_duplicate_keys(
    converter, required, lazy
):
    original = pl.DataFrame({"timestamp": [DAYS[2], DAYS[1]], "asset": [17, 17], **required})
    convert = getattr(inputs(), converter)
    result = convert(original.lazy() if lazy else original, feed_spec=feed())
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert_frame_equal(result.collect() if lazy else result, original)
    duplicated = pl.concat([original, original.head(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        invalid = convert(duplicated.lazy() if lazy else duplicated, feed_spec=feed())
        if lazy:
            invalid.filter(pl.col("timestamp") == DAYS[1]).collect()
