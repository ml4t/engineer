"""Actual prepared histories become causal features without revision/adjustment selection."""

import importlib
import json
from datetime import UTC, date, datetime

import polars as pl
import pytest
import yaml
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

from ml4t.engineer.artifacts import FeatureSpec
from ml4t.engineer.features.paired import rolling_regression
from ml4t.engineer.features.rolling import finite_mean

DAYS = [date(2025, 7, d) for d in (7, 8, 9, 10)]


def prepare(data, **options):
    function = importlib.import_module("ml4t.engineer.features.preparation").prepare_feature_panel
    defaults = {
        "features": {"average": finite_mean("close", period=3)},
        "lookbacks": {"average": 3},
        "input_columns": ["close"],
        "decisions": decisions(),
        "universe": universe(),
        "price_basis": "raw",
        "feed_spec": FeedSpec(
            time_contract_version=2,
            calendar=None,
            entity_col="asset",
            timestamp_semantics="session_label",
        ),
    }
    return function(data, **{**defaults, **options})


def history():
    return pl.DataFrame(
        {
            "timestamp": DAYS * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "close": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "available_at": [datetime(2025, 7, day.day, 16, tzinfo=UTC) for day in DAYS] * 2,
        }
    )


def decisions():
    return pl.DataFrame(
        {
            "timestamp": DAYS[2:],
            "decision_time": [datetime(2025, 7, d.day, 17, tzinfo=UTC) for d in DAYS[2:]],
            "effective_session": DAYS[2:],
        }
    )


def universe():
    return pl.DataFrame(
        {
            "asset": ["A", "B"],
            "effective_from": [DAYS[2]] * 2,
            "effective_to": pl.Series([None, None], dtype=pl.Date),
            "available_at": [datetime(2025, 7, 8, 12, tzinfo=UTC)] * 2,
            "member": [True, False],
            "sector": ["tech", "bank"],
        }
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_dated_membership_keeps_preentry_warmup_and_original_feature_clock(lazy):
    frame = history()
    result = prepare(frame.lazy() if lazy else frame)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    result = result.collect() if lazy else result
    assert result.select("timestamp", "asset").rows() == [
        (DAYS[2], "A"),
        (DAYS[2], "B"),
        (DAYS[3], "A"),
        (DAYS[3], "B"),
    ]
    assert result["average"].to_list() == [2.0, 20.0, 3.0, 30.0]
    assert result["eligible"].to_list() == [True, False, True, False]
    assert result["member"].to_list() == [True, False, True, False]
    assert result["sector"].to_list() == ["tech", "bank", "tech", "bank"]
    assert (
        result["available_at"].to_list()
        == [datetime(2025, 7, 9, 16, tzinfo=UTC)] * 2 + [datetime(2025, 7, 10, 16, tzinfo=UTC)] * 2
    )
    assert_frame_equal(frame, history())


def test_next_session_decision_uses_current_membership_not_the_prior_price_session():
    requested = pl.DataFrame(
        {
            "timestamp": [DAYS[2]],
            "decision_time": [datetime(2025, 7, 10, 13, tzinfo=UTC)],
            "effective_session": [DAYS[3]],
        }
    )
    facts = universe().with_columns(
        pl.when(pl.col("asset") == "A")
        .then(pl.lit(DAYS[3]))
        .otherwise(pl.col("effective_to"))
        .alias("effective_to"),
        pl.when(pl.col("asset") == "B")
        .then(pl.lit(DAYS[3]))
        .otherwise(pl.col("effective_from"))
        .alias("effective_from"),
        pl.lit(True).alias("member"),
    )
    result = prepare(history(), decisions=requested, universe=facts)
    assert result["eligible"].to_list() == [False, True]
    assert result["average"].to_list() == [2.0, 20.0]
    assert result["effective_session"].to_list() == [DAYS[3], DAYS[3]]


def test_explicit_instant_universe_clock_preserves_timezone_equivalence_and_exclusive_ns_end():
    base = int(
        pl.Series([datetime(2025, 7, 10, 13, tzinfo=UTC)])
        .cast(pl.Datetime("ns", "UTC"))
        .cast(pl.Int64)[0]
    )
    requested = pl.DataFrame(
        {
            "timestamp": [DAYS[2]] * 2,
            "decision_time": pl.Series([base + 2, base + 3], dtype=pl.Datetime("ns", "UTC")),
        }
    )
    facts = pl.DataFrame(
        {
            "asset": ["A"],
            "effective_from": pl.Series([base + 1], dtype=pl.Datetime("ns", "America/New_York")),
            "effective_to": pl.Series([base + 3], dtype=pl.Datetime("ns", "UTC")),
            "available_at": [datetime(2025, 7, 9, tzinfo=UTC)],
            "member": [True],
        }
    )
    result = prepare(
        history(), decisions=requested, universe=facts, universe_time_col="decision_time"
    )
    a = result.filter(pl.col("asset") == "A")
    assert a["decision_time"].cast(pl.Int64).to_list() == [base + 2, base + 3]
    assert a["eligible"].to_list() == [True, False]


def test_missing_universe_decision_key_is_not_inferred_from_the_price_or_utc_date():
    with pytest.raises(ValueError, match="effective_session"):
        prepare(history(), decisions=decisions().drop("effective_session"))
    assert prepare(history(), universe=universe().with_columns(pl.lit(None).alias("effective_to")))[
        "eligible"
    ].to_list() == [True, False, True, False]


def test_future_publication_cannot_fill_or_compress_a_required_window():
    delayed = history().with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[1]))
        .then(pl.lit(datetime(2025, 7, 10, 18, tzinfo=UTC)))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    result = prepare(delayed)
    a = result.filter(pl.col("asset") == "A")
    assert a["average"].null_count() == 2
    assert a["eligible"].to_list() == [False, False]
    assert a["available_at"].null_count() == 2


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_known_missing_profile_uses_kernel_definition_without_changing_rows(lazy):
    frame = history().with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[1]))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    strict = prepare(frame)
    result = prepare(frame.lazy() if lazy else frame, missing_input_policy="known_missing")
    result = result.collect() if lazy else result
    assert result.select("timestamp", "asset").rows() == strict.select("timestamp", "asset").rows()
    a = result.filter(pl.col("asset") == "A")
    assert strict.filter(pl.col("asset") == "A")["eligible"].to_list() == [False, False]
    assert a["average"].to_list() == [2.0, 3.5]
    assert a["eligible"].to_list() == [True, True]
    assert a["available_at"].to_list() == [
        datetime(2025, 7, 9, 16, tzinfo=UTC),
        datetime(2025, 7, 10, 16, tzinfo=UTC),
    ]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_known_missing_profile_never_omits_unknown_future_or_invalid_infinite_inputs(value):
    frame = history().with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[1]))
        .then(pl.lit(value, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    if value is None or value != value:
        frame = frame.with_columns(
            pl.when((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[1]))
            .then(None)
            .otherwise(pl.col("available_at"))
            .alias("available_at")
        )
    result = prepare(frame, missing_input_policy="known_missing")
    assert result.filter(pl.col("asset") == "A")["eligible"].to_list() == [False, False]
    future = history().with_columns(
        pl.when(pl.col("timestamp") == DAYS[1])
        .then(pl.lit(datetime(2025, 7, 10, 18, tzinfo=UTC)))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    assert not prepare(future, missing_input_policy="known_missing")["eligible"].any()


def test_known_missing_profile_preserves_masks_full_window_and_undefined_outputs():
    result = prepare(
        history(), input_mask=pl.col("close") > 1, missing_input_policy="known_missing"
    )
    assert result.filter(pl.col("asset") == "A")["average"].to_list() == [2.5, 3.0]
    missing = history().with_columns(pl.lit(float("nan")).alias("close"))
    result = prepare(missing, missing_input_policy="known_missing")
    assert not result["eligible"].any()
    short = history().filter(pl.col("timestamp") >= DAYS[1])
    result = prepare(short, missing_input_policy="known_missing")
    a = result.filter(pl.col("asset") == "A")
    assert a["average"].to_list() == [None, 3.0]
    assert a["eligible"].to_list() == [False, True]


def test_custom_finite_expression_is_evaluated_per_entity_not_over_the_sorted_panel():
    result = prepare(
        history(),
        features={"tail_sum": pl.col("close").tail(3).sum()},
        lookbacks={"tail_sum": 3},
    )
    assert result["tail_sum"].to_list() == [6.0, 60.0, 9.0, 90.0]


def test_changing_only_final_screen_does_not_change_warmup_or_group_values():
    before = prepare(history())
    screened = prepare(history(), screen=pl.col("asset") == "A")
    assert_frame_equal(screened, before.filter(pl.col("asset") == "A"))
    masked = prepare(history(), input_mask=pl.col("close") > 1.0)
    assert masked.filter((pl.col("timestamp") == DAYS[2]) & (pl.col("asset") == "A"))[
        "eligible"
    ].to_list() == [False]


def test_unavailable_and_expired_universe_facts_do_not_default_to_membership():
    facts = universe().with_columns(
        pl.when(pl.col("asset") == "A")
        .then(pl.lit(datetime(2025, 7, 10, 18, tzinfo=UTC)))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    result = prepare(history(), universe=facts)
    assert result.filter(pl.col("asset") == "A")["member"].null_count() == 2
    assert not result["eligible"].any()
    expired = universe().with_columns(pl.lit(DAYS[3]).alias("effective_to"))
    result = prepare(history(), universe=expired)
    assert result.filter(pl.col("timestamp") == DAYS[3])["member"].null_count() == 2


def test_appended_future_quote_and_unknown_universe_change_do_not_revise_prior_features():
    expected = prepare(history())
    future = (
        history()
        .with_columns(
            pl.lit(date(2025, 7, 11)).alias("timestamp"),
            pl.lit(1e9).alias("close"),
            pl.lit(datetime(2025, 7, 11, 16, tzinfo=UTC)).alias("available_at"),
        )
        .unique(subset=["asset", "timestamp"])
    )
    future_fact = universe().with_columns(
        pl.lit(date(2025, 7, 11)).alias("effective_from"),
        pl.lit(datetime(2025, 7, 11, 12, tzinfo=UTC)).alias("available_at"),
        pl.lit("future").alias("sector"),
    )
    actual = prepare(pl.concat([history(), future]), universe=pl.concat([universe(), future_fact]))
    assert_frame_equal(actual, expected)


def test_decision_specific_Data_snapshots_are_consumed_without_selecting_revisions():
    early = history().with_columns(
        pl.lit(datetime(2025, 7, 9, 17, tzinfo=UTC)).alias("snapshot_at")
    )
    late = history().with_columns(
        (pl.col("close") * 2).alias("close"),
        pl.lit(datetime(2025, 7, 10, 17, tzinfo=UTC)).alias("snapshot_at"),
    )
    result = prepare(pl.concat([early, late]), history_decision_time_col="snapshot_at")
    assert result["average"].to_list() == [2.0, 20.0, 6.0, 60.0]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        prepare(pl.concat([early, late]))


def test_ambiguous_overlapping_universe_records_require_upstream_resolution():
    facts = pl.concat(
        [
            universe(),
            universe().filter(pl.col("asset") == "A").with_columns(pl.lit("other").alias("sector")),
        ]
    )
    with pytest.raises(ValueError, match="[Aa]mbiguous|[Dd]uplicate"):
        prepare(history(), universe=facts)


@pytest.mark.parametrize("calendar", ["omitted", None, "NYSE"])
def test_feature_artifact_public_roundtrip_preserves_required_feed_and_eligibility_declarations(
    calendar,
):
    feed = FeedSpec(
        time_contract_version=2, **({} if calendar == "omitted" else {"calendar": calendar})
    )
    payload = {
        "artifact_id": "causal-features",
        "schema": {
            "timestamp_col": "when",
            "entity_col": "identity",
            "feature_columns": ["average"],
            "available_at_col": "known",
            "decision_time_col": "decision",
            "eligible_col": "eligible",
            "universe_time_col": "effective_session",
        },
        "definition": {
            "family": "financial",
            "feed_spec": feed.to_dict(),
            "window_unit": "observation" if calendar in (None, "omitted") else "expected_bar",
            "price_basis": "split_adjusted",
            "lookbacks": {"average": 3},
            "missing_input_policy": "known_missing",
        },
    }
    spec = FeatureSpec.from_mapping(json.loads(json.dumps(payload)))
    output = json.loads(json.dumps(spec.to_dict()))
    for name, value in payload["schema"].items():
        assert output["schema"][name] == value
    for name, value in payload["definition"].items():
        assert output["definition"][name] == value


def test_adjacent_ns_observation_decision_and_publication_keys_do_not_round():
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series([0, 1, 2], dtype=pl.Datetime("ns", "UTC")),
            "asset": ["A"] * 3,
            "close": [1.0, 2.0, 3.0],
            "available_at": pl.Series([1, 2, 3], dtype=pl.Datetime("ns", "UTC")),
        }
    )
    requested = pl.DataFrame(
        {
            "timestamp": pl.Series([1, 2], dtype=pl.Datetime("ns", "UTC")),
            "decision_time": pl.Series([2, 3], dtype=pl.Datetime("ns", "UTC")),
        }
    )
    facts = pl.DataFrame(
        {
            "asset": ["A"],
            "effective_from": pl.Series([0], dtype=pl.Datetime("ns", "UTC")),
            "effective_to": pl.Series([None], dtype=pl.Datetime("ns", "UTC")),
            "available_at": pl.Series([0], dtype=pl.Datetime("ns", "UTC")),
            "member": [True],
        }
    )
    result = prepare(
        frame,
        features={"average": finite_mean("close", period=2)},
        lookbacks={"average": 2},
        decisions=requested,
        universe=facts,
        universe_time_col="decision_time",
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar=None,
            entity_col="asset",
            timestamp_semantics="event_time",
        ),
    )
    assert result["timestamp"].cast(pl.Int64).to_list() == [1, 2]
    assert result["decision_time"].cast(pl.Int64).to_list() == [2, 3]
    assert result["available_at"].cast(pl.Int64).to_list() == [2, 3]
    assert result["average"].to_list() == [1.5, 2.5]


def test_instant_observation_cutoff_cannot_follow_the_actual_decision():
    frame = history().with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC")).dt.offset_by("20h")
    )
    requested = decisions().with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC")).dt.offset_by("20h")
    )
    facts = universe().with_columns(
        pl.col("effective_from").cast(pl.Datetime("ns", "UTC")),
        pl.col("effective_to").cast(pl.Datetime("ns", "UTC")),
    )
    with pytest.raises(ValueError, match="cutoff|decision"):
        prepare(
            frame,
            decisions=requested,
            universe=facts,
            universe_time_col="decision_time",
            feed_spec=FeedSpec(
                time_contract_version=2, calendar=None, timestamp_semantics="event_time"
            ),
        )


def test_custom_roles_compound_ids_and_lazy_output_filter_preserve_the_real_clock():
    frame = (
        history()
        .with_columns(
            pl.struct(pl.lit("bundle").alias("namespace"), pl.col("asset").alias("sid")).alias(
                "asset"
            )
        )
        .rename({"timestamp": "when", "asset": "identity"})
    )
    facts = (
        universe()
        .with_columns(
            pl.struct(pl.lit("bundle").alias("namespace"), pl.col("asset").alias("sid")).alias(
                "asset"
            )
        )
        .rename({"asset": "identity"})
    )
    requested = decisions().rename({"timestamp": "when"})
    result = prepare(
        frame.lazy(),
        decisions=requested.lazy(),
        universe=facts.lazy(),
        feed_spec=FeedSpec(
            time_contract_version=2, calendar=None, timestamp_col="when", entity_col="identity"
        ),
    )
    selected = result.filter(
        (pl.col("identity").struct.field("sid") == "A") & (pl.col("when") == DAYS[3])
    ).collect()
    assert selected["average"].to_list() == [3.0]
    assert selected.schema["identity"] == frame.schema["identity"]


def test_missing_endpoint_or_market_coverage_does_not_carry_stale_features_forward():
    result = prepare(
        history().filter(~((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[3])))
    )
    a = result.filter((pl.col("asset") == "A") & (pl.col("timestamp") == DAYS[3]))
    assert a["member"][0] is True
    assert a["average"][0] is None
    assert a["eligible"][0] is False
    empty = prepare(history().clear())
    assert empty.height == 4
    assert not empty["eligible"].any()
    assert prepare(history(), decisions=decisions().clear()).is_empty()


@pytest.mark.parametrize(
    "options",
    [
        {"features": {}},
        {"lookbacks": {"other": 3}},
        {"lookbacks": {"average": True}},
        {"input_columns": "close"},
        {"input_columns": ["close", "close"]},
        {"input_columns": ["asset"]},
        {"price_basis": ""},
        {"missing_input_policy": "ignore"},
        {"input_mask": 1},
        {"screen": 1},
        {"timestamp_col": "other"},
        {"entity_col": "other"},
        {"entity_col": []},
        {"window_unit": "elapsed"},
        {"lookbacks": {"average": 2**64}},
        {"feed_spec": {"timestamp_col": "timestamp", "time_col": "other"}},
        {"features": {"average": "close"}},
        {"features": {"eligible": pl.col("close")}, "lookbacks": {"eligible": 1}},
    ],
)
def test_invalid_feature_contracts_fail_atomically(options):
    before = history()
    with pytest.raises((TypeError, ValueError)):
        prepare(before, **options)
    assert_frame_equal(before, history())


@pytest.mark.parametrize(
    "kind",
    [
        "missing_history",
        "missing_decision",
        "missing_universe",
        "bad_time",
        "null_key",
        "duplicate_key",
        "duplicate_decision",
        "bad_member",
        "bad_interval",
        "null_effective",
        "classification_collision",
        "reserved",
        "naive_known",
        "null_decision",
        "range",
        "missing_snapshot",
    ],
)
def test_invalid_physical_inputs_do_not_produce_partial_panels(kind):
    frame, requested, facts = history(), decisions(), universe()
    options = {}
    if kind == "missing_history":
        frame = frame.drop("close")
    elif kind == "missing_decision":
        requested = requested.drop("decision_time")
    elif kind == "missing_universe":
        facts = facts.drop("member")
    elif kind == "bad_time":
        frame = frame.with_columns(pl.col("timestamp").cast(pl.String))
    elif kind == "null_key":
        frame = frame.with_columns(pl.lit(None).alias("asset"))
    elif kind == "duplicate_key":
        frame = pl.concat([frame, frame.head(1)])
    elif kind == "duplicate_decision":
        requested = pl.concat([requested, requested.head(1)])
    elif kind == "bad_member":
        facts = facts.with_columns(pl.col("member").cast(pl.Int64))
    elif kind == "bad_interval":
        facts = facts.with_columns(pl.col("effective_from").alias("effective_to"))
    elif kind == "null_effective":
        facts = facts.with_columns(pl.lit(None, pl.Date).alias("effective_from"))
    elif kind == "classification_collision":
        facts = facts.with_columns(pl.lit(1).alias("average"))
    elif kind == "reserved":
        frame = frame.with_columns(pl.lit(1).alias("__ml4t_feature_bad"))
    elif kind == "naive_known":
        frame = frame.with_columns(pl.col("available_at").dt.replace_time_zone(None))
    elif kind == "null_decision":
        requested = requested.with_columns(
            pl.lit(None, pl.Datetime("ns", "UTC")).alias("decision_time")
        )
    elif kind == "range":
        requested = requested.with_columns(
            pl.lit(datetime(2500, 1, 1, tzinfo=UTC)).alias("decision_time")
        )
    else:
        frame = frame.with_columns(pl.lit(datetime(2025, 7, 11, tzinfo=UTC)).alias("snapshot_at"))
        options["history_decision_time_col"] = "snapshot_at"
    before = frame.clone()
    with pytest.raises((TypeError, ValueError)):
        prepare(frame, decisions=requested, universe=facts, **options)
    assert_frame_equal(frame, before)


def test_named_schedule_checks_original_rows_and_explicit_none_remains_row_based():
    frame = history().filter(pl.col("timestamp") != DAYS[1])
    contract = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        timestamp_semantics="session_label",
        data_frequency="1d",
        bar_type="time",
    )
    with pytest.raises(ValueError, match="missing scheduled session"):
        prepare(frame, feed_spec=contract)
    row = prepare(frame, feed_spec=contract, calendar=None)
    a = row.filter(pl.col("asset") == "A")
    assert a["average"][0] is None
    assert a["average"][1] == pytest.approx(8 / 3)


def test_unknown_availability_and_undefined_struct_outputs_cannot_become_eligible():
    unknown = history().with_columns(pl.lit(None).alias("available_at"))
    result = prepare(unknown)
    assert not result["eligible"].any()
    assert result["available_at"].null_count() == result.height


def test_actual_undefined_regression_struct_fields_cannot_become_eligible():
    pytest.importorskip("scipy", reason="Regression p-values require stats extra")
    expression = rolling_regression("close", pl.lit(1.0), period=2)
    result = prepare(history(), features={"estimate": expression}, lookbacks={"estimate": 2})
    assert not result["eligible"].any()


def test_empty_history_does_not_waive_an_undeclared_default_calendar_grid():
    with pytest.raises(ValueError, match="declared bar grid"):
        prepare(history().clear(), feed_spec=FeedSpec())


@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_naive_observation_instants_require_and_use_declared_timezone(timezone):
    frame = history().with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.offset_by("12h"))
    requested = (
        decisions()
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")).dt.offset_by("12h"))
        .with_columns(pl.col("timestamp").dt.offset_by("1h").alias("effective_at"))
    )
    facts = universe().with_columns(
        pl.col("effective_from").cast(pl.Datetime("us")),
        pl.col("effective_to").cast(pl.Datetime("us")),
    )
    options = {
        "decisions": requested,
        "universe": facts,
        "universe_time_col": "effective_at",
        "feed_spec": FeedSpec(
            time_contract_version=2,
            calendar=None,
            timestamp_semantics="event_time",
            timezone=timezone,
        ),
    }
    if timezone is None:
        with pytest.raises(ValueError, match="timezone"):
            prepare(frame, **options)
    else:
        assert prepare(frame, **options)["eligible"].to_list() == [True, False, True, False]


@pytest.mark.parametrize(
    "definition",
    [
        {"lookbacks": {"average": 0}},
        {"lookbacks": {"average": True}},
        {"lookbacks": {"": 3}},
        {"lookbacks": {"average": "3"}},
        {"price_basis": " "},
        {"window_unit": "median_spacing"},
        {"missing_input_policy": "ignore"},
    ],
)
def test_invalid_artifact_declarations_do_not_survive_public_loading(definition):
    with pytest.raises((TypeError, ValueError)):
        FeatureSpec.from_mapping({"artifact_id": "invalid", "definition": definition})


def test_minimal_artifacts_keep_old_wire_keys_and_source_lineage():
    spec = FeatureSpec.from_mapping(
        {
            "artifact_id": "old",
            "definition": {
                "source_artifacts": ["Data:snapshot:abc"],
                "join_keys": ["timestamp", "venue", "asset"],
            },
        }
    )
    output = spec.to_dict()
    assert set(output["schema"]) == {"timestamp_col", "entity_col", "feature_columns"}
    assert set(output["definition"]) == {"family", "join_keys", "source_artifacts"}
    assert output["definition"]["source_artifacts"] == ["Data:snapshot:abc"]


def test_persisted_compound_entity_columns_keep_canonical_schema_and_physical_ns(tmp_path):
    payload = {
        "artifact_id": "compound-features",
        "schema": {"entity_col": ["venue", "asset"], "feature_columns": ["average"]},
        "definition": {
            "feed_spec": FeedSpec(
                time_contract_version=2, calendar=None, entity_col=["venue", "asset"]
            ).to_dict(),
            "price_basis": "raw",
            "missing_input_policy": "complete",
        },
    }
    original = history().with_columns(pl.lit("XNYS").alias("venue"))
    facts = universe().with_columns(pl.lit("XNYS").alias("venue"))
    prepared = prepare(
        original,
        universe=facts,
        feed_spec=FeedSpec.from_mapping(payload["definition"]["feed_spec"]),
    )
    prepared = prepared.with_columns(
        (pl.col("available_at").cast(pl.Int64) + 1)
        .cast(pl.Datetime("ns", "UTC"))
        .alias("available_at")
    )
    path = tmp_path / "features.parquet"
    prepared.write_parquet(path)
    restored = pl.read_parquet(path)
    assert_frame_equal(restored, prepared)
    for encoding in (json.loads(json.dumps(payload)), yaml.safe_load(yaml.safe_dump(payload))):
        output = FeatureSpec.from_mapping(encoding).to_dict()
        assert output["schema"]["entity_col"] == ["venue", "asset"]
        feed = FeedSpec.from_mapping(output["definition"]["feed_spec"])
        assert feed.resolve_calendar() is None
        assert (
            restored.select(feed.timestamp_col, *feed.entity_col).rows()
            == prepared.select("timestamp", "venue", "asset").rows()
        )


@pytest.mark.parametrize("columns", [[], [""], ["asset", "asset"], [1]])
def test_invalid_compound_entity_schema_fails_before_persistence(columns):
    with pytest.raises(ValueError, match="entity_col"):
        FeatureSpec.from_mapping({"artifact_id": "invalid", "schema": {"entity_col": columns}})


def test_explicit_pandas_presentation_keeps_effective_dates_exact_instants_and_stable_ids():
    pytest.importorskip("pyarrow", reason="Explicit pandas presentation requires PyArrow")
    frame = prepare(history()).with_columns(
        pl.when(pl.col("asset") == "A")
        .then(2**63 + 1)
        .otherwise(2**63 + 2)
        .cast(pl.UInt64)
        .alias("asset")
    )
    presentation = (
        frame.select("effective_session", "asset", "average")
        .to_pandas(date_as_object=True)
        .set_index(["effective_session", "asset"])
    )
    assert list(presentation.index) == [
        (DAYS[2], 2**63 + 1),
        (DAYS[2], 2**63 + 2),
        (DAYS[3], 2**63 + 1),
        (DAYS[3], 2**63 + 2),
    ]
    frame = frame.with_columns(
        pl.Series("decision_time", [1, 2, 3, 4], dtype=pl.Datetime("ns", "UTC"))
    )
    presentation = (
        frame.select("decision_time", "asset", "average")
        .to_pandas()
        .set_index(["decision_time", "asset"])
    )
    assert presentation.index.get_level_values("decision_time").asi8.tolist() == [1, 2, 3, 4]
