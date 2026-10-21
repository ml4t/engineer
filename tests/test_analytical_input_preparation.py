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


@pytest.mark.parametrize("purpose", ["causal", "retrospective"])
def test_assets_with_nonoverlapping_histories_cannot_inherit_another_assets_liquidity_warmup(
    purpose,
):
    original = (
        market()
        .with_columns(
            pl.when(pl.col("asset") == 23)
            .then(pl.col(c).dt.offset_by("7d"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("timestamp", "available_at", "decision_time")
        )
        .sort("timestamp", descending=True)
    )
    result = prepare_market(original.lazy(), purpose=purpose).collect()
    assert_frame_equal(result.select(original.columns), original)
    b = result.filter(pl.col("asset") == 23).sort("timestamp")
    assert b["lagged_mean_notional_volume"].to_list() == [None, None, 1000.0, 1000.0, 1000.0]
    assert b["liquidity_covered"].to_list() == [False, False, True, True, True]


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


@pytest.mark.parametrize("lazy", [False, True])
def test_liquidity_cannot_average_different_quote_currencies_for_one_stable_entity(lazy):
    mixed = market().with_columns(
        pl.when((pl.col("asset") == 23) & (pl.col("timestamp") >= DAYS[1]))
        .then(pl.lit("EUR"))
        .otherwise("currency")
        .alias("currency")
    )
    saved = mixed.clone()
    assert_frame_equal(prepare_market(mixed, liquidity_window=None), mixed)
    with pytest.raises(ValueError, match="currency"):
        result = prepare_market(mixed.lazy() if lazy else mixed)
        if lazy:
            result.collect()
    assert_frame_equal(mixed, saved)
    distinct = market().with_columns(
        pl.when(pl.col("asset") == 23).then(pl.lit("EUR")).otherwise("currency").alias("currency")
    )
    valid = prepare_market(distinct)
    assert valid.filter(pl.col("asset") == 23)["lagged_mean_notional_volume"][2] == 1000.0


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


@pytest.mark.parametrize("timezone", [None, "Europe/Berlin"])
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_legacy_decimal_cells_and_datetime_labels_preserve_values_units_and_unknown_keys(
    timezone, unit
):
    from decimal import Decimal

    import pandas as pd

    stamps = pd.date_range("2025-07-07", periods=2, tz=timezone).as_unit(unit)
    original = pd.DataFrame(
        {
            "stable17": [
                Decimal("9007199254740992.01"),
                Decimal(0),
                Decimal("9007199254740992.02"),
                None,
            ]
        },
        index=pd.MultiIndex.from_product([stamps, ["price", "volume"]]),
    )
    result = inputs().market_to_long(original, feed_spec=feed())
    assert result["close"].to_list() == [
        Decimal("9007199254740992.01"),
        Decimal("9007199254740992.02"),
    ]
    assert result["volume"].to_list() == [Decimal(0), None]
    expected_unit = "ms" if unit == "s" else unit
    assert result["timestamp"].dtype == pl.Datetime(expected_unit, timezone)
    assert (
        result["timestamp"].cast(pl.Int64).to_list() == stamps.as_unit(expected_unit).asi8.tolist()
    )
    unknown = original.copy()
    unknown.index = pd.MultiIndex.from_arrays(
        [[pd.NaT, stamps[0], stamps[1], stamps[1]], original.index.get_level_values(1)]
    )
    with pytest.raises(ValueError, match="Duplicate|unknown"):
        inputs().market_to_long(unknown, feed_spec=feed())


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


@pytest.mark.parametrize(
    "column,value",
    [
        ("volume", -1.0),
        ("volume", float("inf")),
        ("multiplier", 0.0),
        ("multiplier", -50.0),
        ("close", float("inf")),
        ("currency", ""),
        ("volume_unit", "  "),
        ("volume", "unknown"),
        ("multiplier", "1 USD"),
        ("currency", 840),
        ("timestamp", "2025-07-07"),
        ("asset", None),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_invalid_market_keys_numbers_and_units_fail_atomically(column, value, lazy):
    bad = market().with_columns(pl.lit(value).alias(column))
    saved = bad.clone()
    with pytest.raises(ValueError):
        result = prepare_market(bad.lazy() if lazy else bad)
        if lazy:
            result.collect()
    assert_frame_equal(bad, saved)


@pytest.mark.parametrize(
    "options",
    [
        {"purpose": "guess"},
        {"price_basis": ""},
        {"volume_basis": None},
        {"completed_at_col": None},
        {"currency_col": "close"},
        {"feed_spec": {}},
        {"liquidity_window": 0},
        {"completed_lag": 0},
    ],
)
def test_incomplete_or_ambiguous_market_declarations_are_not_inferred(options):
    with pytest.raises((ValueError, TypeError)):
        prepare_market(market(), **options)


def test_market_missing_clocks_overflow_and_double_lag_fail_without_overwriting_input():
    with pytest.raises(ValueError, match="Missing"):
        prepare_market(market().drop("currency"))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_market(pl.concat([market(), market().head(1)]))
    with pytest.raises(ValueError, match="reserved"):
        prepare_market(market().with_columns(pl.lit(1).alias("__ml4t_inputs_user")))
    with pytest.raises(ValueError, match="representable"):
        prepare_market(market().with_columns(pl.lit(1e308).alias("close")))
    with pytest.raises(ValueError, match="known aware"):
        prepare_market(market().with_columns(pl.lit(None).alias("decision_time")))
    result = prepare_market(market())
    with pytest.raises(ValueError, match="double lag"):
        prepare_market(result)
    assert_frame_equal(prepare_market(result, liquidity_window=None), result)
    retrospective = prepare_market(
        market().drop("available_at", "decision_time"), purpose="retrospective"
    )
    assert retrospective["liquidity_available_at"].null_count() == 10
    assert retrospective["lagged_mean_notional_volume"][2] == 500.0


def test_market_future_price_shocks_and_unknown_units_do_not_become_causal_past_liquidity():
    original = market()
    future = original.with_columns(
        pl.when(pl.col("timestamp") > DAYS[2]).then(999.0).otherwise("close").alias("close")
    )
    expected = prepare_market(original).filter(pl.col("timestamp") <= DAYS[2])
    assert_frame_equal(prepare_market(future).filter(pl.col("timestamp") <= DAYS[2]), expected)
    for column in ("currency", "multiplier", "volume_unit"):
        result = prepare_market(original.with_columns(pl.lit(None).alias(column)))
        assert result["lagged_mean_notional_volume"].null_count() == result.height
        assert not result["liquidity_covered"].any()
    empty = prepare_market(original.clear())
    assert empty.height == 0
    assert empty.schema == prepare_market(original).schema


def loading_facts():
    return pl.DataFrame(
        {
            "timestamp": [DAYS[1]],
            "asset": [17],
            "factor": ["MKT"],
            "loading": [1.0],
            "available_at": [datetime(2025, 7, 8, 12, tzinfo=UTC)],
        }
    )


def loading_prepare(facts=None, **options):
    return inputs().prepare_loading_inputs(
        loading_facts() if facts is None else facts,
        **{
            "requests": requests(),
            "factors": ["MKT"],
            "feed_spec": feed(),
            "purpose": "causal",
            **options,
        },
    )


@pytest.mark.parametrize(
    "options",
    [
        {"purpose": "guess"},
        {"output": ""},
        {"factors": []},
        {"factors": "MKT"},
        {"factors": ["MKT", "MKT"]},
        {"factors": [""]},
        {"factors": [1]},
        {"output": "asset"},
        {"output": "factor"},
        {"requests": requests().drop("decision_time")},
        {"requests": pl.concat([requests(), requests().head(1)])},
        {"requests": requests().with_columns(pl.lit(None).alias("timestamp"))},
        {"requests": requests().with_columns(pl.lit(17, pl.UInt64).alias("asset"))},
        {"requests": requests().with_columns(pl.lit("MKT").alias("factor"))},
        {"requests": requests().with_columns(pl.lit(1).alias("__ml4t_inputs_x"))},
    ],
)
def test_ambiguous_loading_requests_fail_instead_of_silently_coercing_or_expanding(options):
    with pytest.raises(ValueError):
        loading_prepare(**options)


@pytest.mark.parametrize(
    "column,value",
    [
        ("asset", None),
        ("factor", None),
        ("factor", 1),
        ("loading", "unknown"),
        ("loading", float("inf")),
        ("timestamp", None),
    ],
)
def test_invalid_selected_loading_facts_are_not_zero_filled(column, value):
    with pytest.raises(ValueError):
        loading_prepare(loading_facts().with_columns(pl.lit(value).alias(column)))


def test_loading_metadata_output_names_cannot_overwrite_declared_factor_keys():
    facts = loading_facts().rename({"factor": "loading_available_at"})
    with pytest.raises(ValueError, match="factor"):
        loading_prepare(facts, factor_col="loading_available_at")


@pytest.mark.parametrize("factor_col", [None, ""])
def test_loading_preparation_cannot_disable_factor_identity_with_an_absent_column_name(factor_col):
    with pytest.raises(ValueError, match="factor"):
        loading_prepare(factor_col=factor_col, factors=["MKT", "SIZE"])


@pytest.mark.parametrize("key", [("XNYS",), ("XNYS", 17, "extra"), "XNYS"])
def test_compound_static_classification_keys_must_match_the_declared_identity_columns(key):
    asked = requests().rename({"asset": "sid"}).with_columns(pl.lit("XNYS").alias("venue"))
    options = {
        "requests": asked,
        "feed_spec": feed(entity_col=("venue", "sid")),
        "purpose": "causal",
        "static": True,
        "static_known_at": datetime(2025, 7, 1, tzinfo=UTC),
    }
    with pytest.raises(ValueError, match="identit"):
        inputs().prepare_classification_inputs({key: "bank"}, **options)
    result = inputs().prepare_classification_inputs({("XNYS", 17): "bank"}, **options)
    assert result.filter(pl.col("sid") == 17)["sector"].to_list() == ["bank"] * 3
    assert result.filter(pl.col("sid") == 23)["sector"].null_count() == 3


def test_dated_inputs_preserve_adjacent_ns_publication_cutoffs_and_exact_effective_keys():
    base = 1_750_000_000_000_000_000
    stamps = pl.Series("timestamp", [base + 1] * 3, dtype=pl.Datetime("ns", "UTC"))
    asked = pl.DataFrame(
        {
            "timestamp": stamps,
            "asset": [17] * 3,
            "decision_time": pl.Series(
                [base + n for n in (3, 4, 5)], dtype=pl.Datetime("ns", "UTC")
            ),
        }
    )
    facts = pl.DataFrame(
        {
            "timestamp": stamps.head(1),
            "asset": [17],
            "factor": ["MKT"],
            "loading": [2.0],
            "available_at": pl.Series([base + 4], dtype=pl.Datetime("ns", "UTC")),
        }
    )
    result = loading_prepare(facts, requests=asked)
    assert result["loading"].to_list() == [None, 2.0, 2.0]
    assert result["loading_available_at"].cast(pl.Int64).to_list() == [None, base + 4, base + 4]
    adjacent = facts.with_columns(
        pl.Series("timestamp", [base + 2], dtype=pl.Datetime("ns", "UTC"))
    )
    assert loading_prepare(adjacent, requests=asked)["loading"].null_count() == 3


def test_loading_effective_instants_need_timezone_or_aware_clocks_not_date_flooring():
    fact = loading_facts().with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    asked = requests().with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    with pytest.raises(ValueError, match="timezone"):
        loading_prepare(fact, requests=asked)
    result = loading_prepare(fact, requests=asked, feed_spec=feed(timezone="America/New_York"))
    assert result["loading"].to_list() == [1.0, None, None, None, None, None]
    with pytest.raises(ValueError, match="clock kind"):
        loading_prepare(fact)
    with pytest.raises(ValueError, match="Date labels"):
        loading_prepare(
            loading_facts().with_columns(pl.col("timestamp").cast(pl.String)),
            requests=requests().with_columns(pl.col("timestamp").cast(pl.String)),
        )


def test_dated_session_labels_do_not_silently_compare_naive_wall_time_with_aware_instants():
    fact = loading_facts().with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
    asked = requests().with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns")).dt.replace_time_zone("America/New_York")
    )
    with pytest.raises(ValueError, match="session.label"):
        loading_prepare(fact, requests=asked, feed_spec=feed(timestamp_semantics="session_label"))
    result = loading_prepare(
        fact,
        requests=asked.with_columns(pl.col("timestamp").dt.replace_time_zone(None)),
        feed_spec=feed(timestamp_semantics="session_label"),
    )
    assert result["loading"].to_list() == [1.0, None, None, None, None, None]


def test_expected_bar_liquidity_reuses_declared_calendar_and_explicit_none_is_row_mode():
    pytest.importorskip("pandas_market_calendars", reason="Named schedules require calendar extra")
    spec = feed(
        calendar="NYSE",
        timestamp_semantics="session_label",
        bar_type="time",
        data_frequency="1d",
    )
    result = prepare_market(market(), feed_spec=spec)
    assert result["liquidity_window_unit"].unique().to_list() == ["expected_bar"]
    sparse = market().filter(pl.col("timestamp") != DAYS[1])
    with pytest.raises(ValueError):
        prepare_market(sparse, feed_spec=spec)
    row = prepare_market(sparse, feed_spec=spec, calendar=None)
    assert row["liquidity_window_unit"].unique().to_list() == ["observation"]
    assert row.height == sparse.height


@pytest.mark.parametrize("converter", ["market_to_long", "loadings_to_long"])
def test_conversion_rejects_unknown_native_keys_and_missing_required_columns(converter):
    convert = getattr(inputs(), converter)
    with pytest.raises(ValueError, match="Missing"):
        convert(pl.DataFrame({"close": [10.0]}), feed_spec=feed())
    fields = {"close": [10.0], "factor": ["MKT"], "loading": [1.0]}
    with pytest.raises(ValueError, match="unknown"):
        convert(pl.DataFrame({"timestamp": [DAYS[0]], "asset": [None], **fields}), feed_spec=feed())


def test_legacy_conversion_rejects_invalid_index_shapes_and_ambiguous_cells():
    import pandas as pd

    convert = inputs().market_to_long
    with pytest.raises(TypeError):
        convert([[10.0]], feed_spec=feed())
    bad_indices = [
        pd.Index([0]),
        pd.MultiIndex.from_tuples([(0, "price")]),
        pd.MultiIndex.from_tuples([(pd.Timestamp("2025-07-07"), "price", "extra")]),
    ]
    for index in bad_indices:
        with pytest.raises(ValueError):
            convert(pd.DataFrame({17: [10.0]}, index=index), feed_spec=feed())
    legacy = pd.DataFrame(
        {17: [10.0]},
        index=pd.MultiIndex.from_product([pd.to_datetime(["2025-07-07"], utc=True), ["price"]]),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        convert(pd.concat([legacy, legacy]), feed_spec=feed())
    with pytest.raises(ValueError, match="distinct"):
        convert(legacy, feed_spec=feed(), volume_field="price")
    with pytest.raises(ValueError, match="price field"):
        convert(legacy, feed_spec=feed(), price_field="adjusted")
    with pytest.raises(ValueError, match="identity"):
        convert(legacy, feed_spec=feed(), entity_map={})
    with pytest.raises(ValueError, match="Compound"):
        convert(legacy, feed_spec=feed(entity_col=("venue", "sid")), entity_map={17: 17})
    with pytest.raises(ValueError, match="unknown"):
        convert(legacy, feed_spec=feed(), entity_map={17: None})
    with pytest.raises(ValueError, match="stable"):
        convert(legacy, feed_spec=feed(), entity_map={17: 17.5})
    result = convert(legacy, feed_spec=feed(), entity_map={17: "stable17"})
    assert result["asset"].to_list() == ["stable17"]
    assert result["volume"].to_list() == [None]
    empty = convert(legacy.iloc[:, :0], feed_spec=feed())
    assert empty.height == 0
    assert empty.columns == ["timestamp", "asset", "close", "volume"]


def test_legacy_factor_names_and_mixed_value_dtypes_require_explicit_normalization():
    import pandas as pd

    index = pd.MultiIndex.from_arrays([pd.to_datetime(["2025-07-07"], utc=True), [17]])
    for column in (1, ""):
        with pytest.raises(ValueError, match="factor columns"):
            inputs().loadings_to_long(pd.DataFrame({column: [1.0]}, index=index), feed_spec=feed())
    mixed = pd.DataFrame({"MKT": pd.array([1], dtype="Int64"), "SIZE": [1.0]}, index=index)
    with pytest.raises(ValueError, match="dtypes"):
        inputs().loadings_to_long(mixed, feed_spec=feed())
    assert inputs().loadings_to_long(mixed.iloc[:, :0], feed_spec=feed()).height == 0
    market_index = pd.MultiIndex.from_product([pd.to_datetime(["2025-07-07"], utc=True), ["price"]])
    mixed_market = pd.DataFrame(
        {17: pd.array([2**53 + 1], dtype="Int64"), 23: [1.0]}, index=market_index
    )
    with pytest.raises(ValueError, match="dtypes"):
        inputs().market_to_long(mixed_market, feed_spec=feed())


def test_classification_unknown_interval_end_and_static_flags_preserve_explicit_scope():
    facts = pl.DataFrame(
        {
            "asset": [17],
            "sector": ["tech"],
            "effective_from": [DAYS[0]],
            "effective_to": [None],
            "available_at": [datetime(2025, 7, 7, tzinfo=UTC)],
        }
    )
    result = (
        inputs()
        .prepare_classification_inputs(
            facts.lazy(), requests=requests(), feed_spec=feed(), purpose="causal"
        )
        .collect()
    )
    assert result.filter(pl.col("asset") == 17)["sector"].to_list() == ["tech"] * 3
    with pytest.raises(ValueError, match="nonempty"):
        inputs().prepare_classification_inputs(
            facts.with_columns(pl.col("effective_from").alias("effective_to")),
            requests=requests(),
            feed_spec=feed(),
            purpose="causal",
        )
    with pytest.raises(ValueError, match="clock kind"):
        inputs().prepare_classification_inputs(
            facts.with_columns(
                pl.col("effective_from").cast(pl.Datetime("ns")).alias("effective_to")
            ),
            requests=requests(),
            feed_spec=feed(),
            purpose="causal",
        )
    with pytest.raises(ValueError, match="mapping"):
        inputs().prepare_classification_inputs(
            facts,
            requests=requests(),
            feed_spec=feed(),
            purpose="causal",
            static=True,
        )
    with pytest.raises(ValueError, match="mapping"):
        inputs().prepare_classification_inputs(
            None,
            requests=requests(),
            feed_spec=feed(),
            purpose="causal",
            static=True,
        )
    retro = inputs().prepare_classification_inputs(
        {},
        requests=requests(),
        feed_spec=feed(),
        purpose="retrospective",
        static=True,
    )
    assert retro["sector"].null_count() == retro.height
    assert retro["sector_static"].all()


def test_publication_outside_ns_range_fails_explicitly_instead_of_wrapping():
    with pytest.raises(ValueError, match="range"):
        loading_prepare(
            loading_facts().with_columns(
                pl.lit(datetime(3000, 1, 1, tzinfo=UTC)).alias("available_at")
            )
        )
