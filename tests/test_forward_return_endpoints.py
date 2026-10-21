"""Independent endpoint and arithmetic contracts for supplied-price outcomes."""

from fractions import Fraction

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.engineer import labeling
from ml4t.engineer.labeling.utils import get_future_price_at_time


def _prices(prices=(100, 110, 121), offsets=None, *, asset="A", unit="h"):
    offsets = list(range(len(prices))) if offsets is None else offsets
    opened = pd.Timestamp("2025-07-03T00:00:00Z").value
    step = pd.Timedelta("1" + unit).value
    return pl.DataFrame(
        {
            "timestamp": pl.Series([opened + step * i for i in offsets], dtype=pl.Int64).cast(
                pl.Datetime("ns", "UTC")
            ),
            "asset": asset,
            "close": prices,
        }
    ).with_columns((pl.col("timestamp") + pl.duration(nanoseconds=7)).alias("known_at"))


@pytest.mark.parametrize("duration", ["1ns", "500ns", "1ms", "1h30m"])
def test_public_future_price_lookup_never_truncates_declared_fixed_duration(duration):
    frame = _prices(unit=duration[1:] if duration in ("1ns", "1ms") else "ns")
    if duration == "500ns":
        frame = _prices(offsets=[0, 500, 1000], unit="ns")
    if duration == "1h30m":
        frame = _prices(offsets=[0, 90, 180], unit="m")
    future, valid = get_future_price_at_time(
        frame,
        pd.Timedelta(duration),
        price_col="close",
        timestamp_col="timestamp",
        group_cols=["asset"],
    )
    assert future.to_list() == [110, 121, None]
    assert valid.to_list() == [True, True, False]


def _run(frame, *, lazy=False, **kwargs):
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"two": 2},
        calendar=None,
        price_basis="raw",
        available_at_col="known_at",
        **kwargs,
    )
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    return result.collect() if lazy else result


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("method,expected", [("cumulative", 0.21), ("exit_bar", 0.1)])
def test_horizon_two_distinguishes_entry_return_from_exit_bar_only(lazy, method, expected):
    frame = _prices()
    output = _run(frame, lazy=lazy, endpoint_method=method)
    assert output.height == 3
    assert output["forward_return"].to_list()[1:] == [None, None]
    assert output["forward_return"][0] == pytest.approx(expected, abs=1e-15)
    assert output["timestamp"].equals(frame["timestamp"])
    assert output["exit_time"][0] == frame["timestamp"][2]
    assert output["outcome_available_at"].cast(pl.Int64)[0] == frame["known_at"].cast(pl.Int64)[2]
    assert output["outcome_status"].to_list() == ["ok", "insufficient_tail", "insufficient_tail"]
    assert output["endpoint_method"].unique().to_list() == [method]
    assert output["window_unit"].unique().to_list() == ["observation"]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("index_mode,expected", [("per_asset", 0.21), ("common_grid", None)])
def test_common_clock_missing_row_is_not_an_asset_shift_or_filled_quote(lazy, index_mode, expected):
    frame = pl.concat([_prices(offsets=[0, 2, 3]), _prices((50, 51, 52, 53), asset="B")])
    output = _run(frame, lazy=lazy, index_mode=index_mode)
    first = output.filter((pl.col("asset") == "A") & (pl.col("timestamp") == frame["timestamp"][0]))
    # Two global-clock positions is hour2, not the second following A observation at hour3.
    if index_mode == "common_grid":
        assert first["forward_return"][0] == pytest.approx(0.1)
        one = labeling.forward_returns(
            frame, horizons={"one": 1}, index_mode=index_mode, calendar=None, price_basis="raw"
        ).filter((pl.col("asset") == "A") & (pl.col("timestamp") == frame["timestamp"][0]))
        assert one["forward_return"][0] is expected
        assert one["outcome_status"][0] == "missing_endpoint"
    else:
        assert first["forward_return"][0] == pytest.approx(expected)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("duration", ["1ns", "500ns", "1ms", "30m", "1h30m", "1d", "1d1h1ns"])
def test_elapsed_targets_and_selected_exit_preserve_exact_declared_duration(lazy, duration):
    step = (
        (25 * 3600 * 10**9 + 1)
        if duration == "1d1h1ns"
        else pd.Timedelta("1D" if duration == "1d" else duration).value
    )
    frame = _prices(offsets=[0, step, 2 * step], unit="ns")
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"future": duration},
        window_unit="elapsed",
        calendar=None,
        price_basis="raw",
    )
    result = result.collect() if lazy else result
    assert result["forward_return"].to_list()[:2] == pytest.approx([0.1, 0.1])
    assert (
        result["requested_exit_time"].cast(pl.Int64)[0] - result["timestamp"].cast(pl.Int64)[0]
        == step
    )
    assert (
        result["exit_time"].cast(pl.Int64)[:2].to_list()
        == frame["timestamp"].cast(pl.Int64)[1:].to_list()
    )


@pytest.mark.parametrize("dtype,start", [(pl.Int64, 2**53), (pl.UInt64, 2**63)])
@pytest.mark.parametrize("method", ["returns", "log_returns"])
def test_outcome_integer_ticks_use_shared_precise_return_kernel(dtype, start, method):
    frame = _prices((start, start + 1, start + 2)).with_columns(pl.col("close").cast(dtype))
    result = _run(frame, return_method=method)
    expected = float(Fraction(2, start))
    if method == "log_returns":
        import math

        expected = math.log1p(expected)
    assert result["forward_return"][0] == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("policy", ["raise", "null"])
def test_null_price_policy_never_fills_a_present_missing_quote(policy):
    frame = _prices((100, None, 121))
    if policy == "raise":
        with pytest.raises(ValueError, match="price"):
            _run(frame, missing_price_policy=policy)
    else:
        output = _run(frame, missing_price_policy=policy, endpoint_method="exit_bar")
        assert output["forward_return"].to_list() == [None, None, None]
        assert output["outcome_status"][0] == "missing_price"
        assert output["exit_time"][0] == frame["timestamp"][2]


def test_elapsed_tolerance_does_not_relabel_a_late_match_as_exact_target():
    frame = _prices(offsets=[0, 2, 3])
    result = labeling.forward_returns(
        frame,
        horizons={"one": "1h"},
        window_unit="elapsed",
        tolerance="30m",
        calendar=None,
        price_basis="raw",
    )
    assert result["requested_exit_time"][0] < frame["timestamp"][1]
    assert result["exit_time"][0] is None
    assert result["outcome_status"][0] == "missing_endpoint"


@pytest.mark.parametrize("policy", ["null", "drop", "raise"])
def test_insufficient_tail_policy_is_explicit_and_atomic(policy):
    frame = _prices()
    before = frame.clone()
    if policy == "raise":
        with pytest.raises(ValueError, match="tail"):
            _run(frame, tail_policy=policy)
    else:
        result = _run(frame, tail_policy=policy)
        assert result.height == (1 if policy == "drop" else 3)
    assert frame.equals(before)


def test_sparse_explicit_entry_keys_cannot_be_backshifted_by_array_position():
    frame = pl.concat([_prices(), _prices((200, 220, 242), asset="B")])
    entries = frame.select("timestamp", "asset").gather([4, 0])
    result = _run(frame, entries=entries)
    assert result.select("timestamp", "asset").equals(entries)
    assert result["forward_return"].to_list() == [None, pytest.approx(0.21)]


def test_unknown_price_availability_remains_unknown_not_the_event_time():
    result = labeling.forward_returns(
        _prices(), horizons={"two": 2}, calendar=None, price_basis="raw"
    )
    assert result["outcome_available_at"].null_count() == result.height


def test_expected_bars_validate_real_schedule_gaps_before_deriving_outcomes():
    frame = _prices(offsets=[0, 2, 3])
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    with pytest.raises(ValueError, match="missing scheduled bar"):
        labeling.forward_returns(
            frame,
            horizons={"two": 2},
            feed_spec=feed,
            window_unit="expected_bar",
            price_basis="raw",
        )
    assert _run(frame).height == frame.height


@pytest.mark.parametrize("lazy", [False, True])
def test_multiple_horizons_keep_entry_order_and_explicit_units(lazy):
    frame = _prices()
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"one": 1, "two": 2},
        calendar=None,
        price_basis="split_adjusted",
    )
    result = result.collect() if lazy else result
    assert result["horizon"].to_list() == ["one", "two"] * 3
    assert result["horizon_value"].to_list() == ["1", "2"] * 3
    assert result["forward_return"].to_list() == [
        pytest.approx(0.1),
        pytest.approx(0.21),
        pytest.approx(0.1),
        None,
        None,
        None,
    ]
    assert result["price_basis"].unique().to_list() == ["split_adjusted"]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("unit,horizon", [("observation", 1), ("elapsed", "1ns"), ("session", 1)])
def test_empty_schema_retains_all_outcome_roles(lazy, unit, horizon):
    frame = _prices().head(0)
    kwargs = {"calendar": None}
    if unit == "session":
        kwargs = {
            "feed_spec": FeedSpec(
                time_contract_version=2,
                calendar="24/7",
                entity_col="asset",
                data_frequency="1h",
                timestamp_semantics="event_time",
                bar_type="time",
                bar_grid_anchor="session_open",
                bar_label="left",
                partial_bar_policy="exclude",
            )
        }
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"future": horizon},
        window_unit=unit,
        price_basis="raw",
        **kwargs,
    )
    result = result.collect() if lazy else result
    assert result.is_empty()
    assert result.schema["exit_time"] == pl.Datetime("ns", "UTC")
    assert result.schema["forward_return"] == pl.Float64
    assert result.schema["outcome_available_at"] == pl.Datetime("ns", "UTC")


@pytest.mark.parametrize("side,expected_hour", [("left", 16), ("right", 17)])
@pytest.mark.parametrize("lazy", [False, True])
def test_session_endpoint_uses_nyse_early_close_and_not_elapsed_24_hours(side, expected_hour, lazy):
    # Last June2 bar -> July3 early close, with no July4/July5/July6 observations.
    times = [
        "2025-07-02T19:00:00Z",
        "2025-07-03T13:30:00Z",
        "2025-07-03T14:30:00Z",
        "2025-07-03T15:30:00Z",
        "2025-07-03T16:30:00Z",
        "2025-07-07T13:30:00Z",
    ]
    if side == "right":
        times = [
            "2025-07-02T20:00:00Z",
            "2025-07-03T14:30:00Z",
            "2025-07-03T15:30:00Z",
            "2025-07-03T16:30:00Z",
            "2025-07-03T17:00:00Z",
            "2025-07-07T14:30:00Z",
        ]
    # July2's session-open-aligned last left label is19:30, not19:00.
    times[0] = "2025-07-02T19:30:00Z" if side == "left" else "2025-07-02T20:00:00Z"
    frame = pl.DataFrame(
        {
            "timestamp": [pd.Timestamp(t).value for t in times],
            "asset": "A",
            "close": [100, 101, 102, 103, 110, 120],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label=side,
        partial_bar_policy="include",
    )
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"next_session": 1},
        window_unit="session",
        feed_spec=feed,
        price_basis="raw",
    )
    result = result.collect() if lazy else result
    assert result["forward_return"][0] == pytest.approx(0.1)
    assert result["exit_time"][0].hour == expected_hour
    assert result["requested_exit_time"][0] == result["exit_time"][0]
    assert result["outcome_status"][-1] == "insufficient_tail"


def test_daily_session_label_outcomes_preserve_civil_labels_and_unknown_availability():
    frame = pl.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(t).date() for t in ["2025-07-03", "2025-07-07", "2025-07-08"]
            ],
            "asset": "A",
            "close": [100, 110, 121],
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        data_frequency="1d",
        timestamp_semantics="session_label",
        bar_type="time",
    )
    result = labeling.forward_returns(
        frame,
        horizons={"next_session": 1},
        window_unit="session",
        feed_spec=feed,
        price_basis="raw",
    )
    assert result["forward_return"].to_list() == [pytest.approx(0.1), pytest.approx(0.1), None]
    assert result.schema["exit_time"] == pl.Date
    assert result["exit_time"][0] == frame["timestamp"][1]
    assert result["outcome_available_at"].null_count() == 3


def test_naive_elapsed_targets_use_physical_declared_timezone_through_dst():
    times = [
        pd.Timestamp(t).to_pydatetime()
        for t in ["2025-03-09T01:00", "2025-03-09T03:00", "2025-03-09T04:00"]
    ]
    frame = pl.DataFrame({"timestamp": times, "asset": "A", "close": [100, 110, 121]})
    feed = FeedSpec(
        time_contract_version=2, calendar=None, entity_col="asset", timezone="America/New_York"
    )
    result = labeling.forward_returns(
        frame, horizons={"hour": "1h"}, window_unit="elapsed", feed_spec=feed, price_basis="raw"
    )
    assert result["forward_return"].to_list() == [pytest.approx(0.1), pytest.approx(0.1), None]
    assert (
        result["exit_time"].cast(pl.Int64)[0] - result["timestamp"].cast(pl.Int64)[0]
    ) == 3600 * 10**9


@pytest.mark.parametrize("compound", [False, True])
def test_custom_column_mapping_and_stable_compound_entity_ids_are_preserved(compound):
    frame = _prices().rename({"timestamp": "event", "asset": "instrument", "close": "px"})
    if compound:
        frame = frame.with_columns(
            pl.struct(pl.lit("provider").alias("namespace"), pl.lit(7).alias("sid")).alias(
                "instrument"
            )
        )
    feed = FeedSpec(
        time_contract_version=2,
        calendar=None,
        timestamp_col="event",
        entity_col="instrument",
        price_col="px",
    )
    result = labeling.forward_returns(frame, horizons={"two": 2}, feed_spec=feed, price_basis="raw")
    assert result["instrument"].equals(frame["instrument"])
    assert result["forward_return"][0] == pytest.approx(0.21)


def test_delayed_entry_quote_controls_outcome_availability_even_if_exit_known_earlier():
    frame = _prices().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("known_at").max() + pl.duration(hours=5))
        .otherwise(pl.col("known_at"))
        .alias("known_at")
    )
    result = _run(frame)
    assert result["outcome_available_at"][0] == frame["known_at"][0]


@pytest.mark.parametrize(
    "option,value",
    [
        ("endpoint_method", "compound"),
        ("index_mode", "infer"),
        ("return_method", "binary"),
        ("tail_policy", "fill"),
        ("missing_price_policy", "ffill"),
    ],
)
def test_unknown_options_fail_before_any_input_mutation(option, value):
    frame = _prices()
    before = frame.clone()
    with pytest.raises(ValueError, match=option):
        _run(frame, **{option: value})
    assert frame.equals(before)


@pytest.mark.parametrize("tolerance,expected", [("1ns", None), ("2ns", 0.1)])
def test_submicrosecond_endpoint_tolerance_does_not_round_to_zero(tolerance, expected):
    result = labeling.forward_returns(
        _prices(offsets=[0, 3, 4], unit="ns"),
        horizons={"tick": "1ns"},
        window_unit="elapsed",
        tolerance=tolerance,
        calendar=None,
        price_basis="raw",
    )
    assert result["forward_return"][0] == (
        pytest.approx(expected) if expected is not None else None
    )


@pytest.mark.parametrize("time_based", [False, True])
def test_existing_fixed_time_labels_do_not_wrap_declining_unsigned_prices(time_based):
    frame = _prices((100, 90, 81)).with_columns(pl.col("close").cast(pl.UInt64))
    result = labeling.fixed_time_horizon_labels(
        frame, horizon="1h" if time_based else 1, timestamp_col="timestamp"
    )
    column = "label_return_1h" if time_based else "label_return_1p"
    assert result[column].to_list() == [pytest.approx(-0.1), pytest.approx(-0.1), None]


@pytest.mark.parametrize("lazy", [False, True])
def test_positive_elapsed_horizon_never_wraps_into_past_at_ns_range_limit(lazy):
    upper = 2**63 - 1
    frame = _prices().with_columns(
        pl.Series("timestamp", [upper - 2, upper - 1, upper], dtype=pl.Int64).cast(
            pl.Datetime("ns", "UTC")
        )
    )
    before = frame.clone()
    with pytest.raises(ValueError, match="range|overflow"):
        result = labeling.forward_returns(
            frame.lazy() if lazy else frame,
            horizons={"future": "10ns"},
            window_unit="elapsed",
            calendar=None,
            price_basis="raw",
        )
        if lazy:
            result.collect()
    assert frame.equals(before)


def test_input_outside_nanosecond_range_is_not_silently_moved_to_another_century():
    from datetime import UTC, datetime

    frame = _prices().with_columns(
        pl.Series("timestamp", [datetime(1500, 1, day, tzinfo=UTC) for day in [1, 2, 3]])
    )
    before = frame.clone()
    with pytest.raises(ValueError, match="range"):
        _run(frame)
    assert frame.equals(before)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"horizons": {}}, "horizons"),
        ({"horizons": None}, "horizons"),
        ({"horizons": {"": 1}}, "names"),
        ({"horizons": {7: 1}}, "names"),
        ({"price_basis": ""}, "price_basis"),
        ({"window_unit": "days"}, "days"),
        ({"horizons": {"bad": True}}, "positive integers"),
        ({"horizons": {"bad": 0}}, "positive integers"),
        ({"horizons": {"bad": 1.5}}, "positive integers"),
        ({"horizons": {"bad": 1}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "0ns"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "-1ns"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "NaT"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "banana"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "1M"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"horizons": {"bad": "999999d"}, "window_unit": "elapsed"}, "fixed durations"),
        ({"tolerance": "1h"}, "only to elapsed"),
        ({"window_unit": "session"}, "named calendar"),
        ({"window_unit": "expected_bar"}, "named calendar"),
        ({"entity_col": "horizon"}, "key names"),
        ({"entity_col": "_out_private"}, "key names"),
        ({"entity_col": "timestamp"}, "key names"),
    ],
)
def test_invalid_horizon_and_boundary_options_fail_atomically(options, match):
    frame = _prices()
    before = frame.clone()
    kwargs = {"horizons": {"one": 1}, "calendar": None, "price_basis": "raw", **options}
    with pytest.raises(ValueError, match=match):
        labeling.forward_returns(frame, **kwargs)
    assert frame.equals(before)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_price",
        "null_key",
        "duplicate_prices",
        "text_price",
        "string_time",
        "naive_time",
        "duplicate_entries",
        "absent_entries",
        "wrong_session_type",
    ],
)
def test_bad_public_tables_are_rejected_without_mutation(kind):
    frame = _prices()
    options = {}
    match = "."
    if kind == "missing_price":
        frame = frame.drop("close")
    elif kind == "null_key":
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("asset"))
    elif kind == "duplicate_prices":
        frame = pl.concat([frame, frame.head(1)])
    elif kind == "text_price":
        frame = frame.with_columns(pl.col("close").cast(pl.String))
    elif kind == "string_time":
        frame = frame.with_columns(pl.col("timestamp").cast(pl.String))
    elif kind == "naive_time":
        frame = frame.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    elif kind == "duplicate_entries":
        options = {"entries": pl.concat([frame.select("timestamp", "asset")] * 2)}
    elif kind == "absent_entries":
        options = {
            "entries": frame.select("timestamp", "asset").with_columns(
                pl.lit("missing").alias("asset")
            )
        }
    else:
        options = {
            "feed_spec": FeedSpec(
                time_contract_version=2, calendar=None, timestamp_semantics="session_label"
            )
        }
    before = frame.clone()
    with pytest.raises(ValueError, match=match):
        _run(frame, **options)
    assert frame.equals(before)


@pytest.mark.parametrize(
    "column,value", [("timestamp_col", "event"), ("entity_col", "sid"), ("price_col", "px")]
)
def test_explicit_column_names_cannot_contradict_declared_feed_roles(column, value):
    feed = FeedSpec(time_contract_version=2, calendar=None, entity_col="asset")
    with pytest.raises(ValueError, match="conflict"):
        _run(_prices(), feed_spec=feed, **{column: value})


def test_daily_session_labels_cannot_be_interpreted_as_elapsed_instants():
    feed = FeedSpec(time_contract_version=2, calendar=None, timestamp_semantics="session_label")
    frame = _prices().with_columns(pl.col("timestamp").dt.date())
    with pytest.raises(ValueError, match="Session labels"):
        labeling.forward_returns(
            frame, horizons={"day": "1d"}, window_unit="elapsed", feed_spec=feed, price_basis="raw"
        )


@pytest.mark.parametrize("unit", ["observation", "session"])
def test_beyond_coverage_arbitrary_precision_count_produces_tail_not_native_integer_overflow(unit):
    frame = _prices()
    kwargs = {"calendar": None}
    if unit == "session":
        kwargs = {
            "feed_spec": FeedSpec(
                time_contract_version=2,
                calendar="24/7",
                entity_col="asset",
                data_frequency="1h",
                timestamp_semantics="event_time",
                bar_type="time",
                bar_grid_anchor="session_open",
                bar_label="left",
                partial_bar_policy="exclude",
            )
        }
    result = labeling.forward_returns(
        frame, horizons={"very_large": 10**100}, window_unit=unit, price_basis="raw", **kwargs
    )
    assert result["forward_return"].to_list() == [None] * 3
    assert result["outcome_status"].to_list() == ["insufficient_tail"] * 3


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_external_research_price_view_retains_original_quote_and_row_masks(lazy):
    original = _prices((100, None, 121))
    view = original.with_columns(
        pl.lit(True).alias("original_row"),
        pl.col("close").is_not_null().alias("original_quote"),
        pl.col("close").forward_fill(),
    )
    before = view.clone()
    result = labeling.forward_returns(
        view.lazy() if lazy else view,
        horizons={"one": 1},
        calendar=None,
        price_basis="raw",
        original_row_col="original_row",
        original_price_col="original_quote",
    )
    result = result.collect() if lazy else result
    assert result["forward_return"].to_list() == [0.0, pytest.approx(0.21), None]
    assert result["entry_price_observed"].to_list() == [True, False, True]
    assert result["exit_price_observed"].to_list() == [False, True, None]
    assert result["reference_price_observed"].to_list() == [True, False, True]
    assert result["entry_row_observed"].to_list() == [True] * 3
    assert view.equals(before)


def test_research_fill_rows_cannot_conceal_a_missing_scheduled_bar():
    sparse = _prices(offsets=[0, 2, 3])
    # A caller-owned research view inserts the missing clock row and fills its quote.
    view = (
        _prices((100, None, 110, 121))
        .join(
            sparse.select("timestamp").with_columns(pl.lit(True).alias("original_row")),
            on="timestamp",
            how="left",
        )
        .with_columns(
            pl.col("original_row").fill_null(False),
            pl.col("close").is_not_null().alias("original_quote"),
            pl.col("close").forward_fill(),
        )
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    with pytest.raises(ValueError, match="missing scheduled bar"):
        labeling.forward_returns(
            view,
            horizons={"one": 1},
            feed_spec=feed,
            price_basis="raw",
            window_unit="expected_bar",
            original_row_col="original_row",
            original_price_col="original_quote",
        )
    row_mode = labeling.forward_returns(
        view,
        horizons={"one": 1},
        calendar=None,
        price_basis="raw",
        original_row_col="original_row",
        original_price_col="original_quote",
    )
    assert row_mode["exit_row_observed"].to_list() == [False, True, True, None]
    assert row_mode["entry_row_observed"].to_list() == [True, False, True, True]


@pytest.mark.parametrize(
    "bad",
    ["missing_pair", "not_bool", "unknown_row", "contradictory", "unknown_price", "not_bool_price"],
)
def test_external_research_view_requires_unambiguous_original_masks(bad):
    view = _prices().with_columns(
        pl.lit(True).alias("original_row"), pl.lit(True).alias("original_quote")
    )
    options = {"original_row_col": "original_row", "original_price_col": "original_quote"}
    if bad == "missing_pair":
        del options["original_row_col"]
    elif bad == "not_bool":
        view = view.with_columns(pl.col("original_row").cast(pl.Int8))
    elif bad == "unknown_row":
        view = view.with_columns(pl.lit(None, dtype=pl.Boolean).alias("original_row"))
    elif bad == "unknown_price":
        view = view.with_columns(pl.lit(None, dtype=pl.Boolean).alias("original_quote"))
    elif bad == "not_bool_price":
        view = view.with_columns(pl.col("original_quote").cast(pl.Int8))
    else:
        view = view.with_columns(pl.lit(False).alias("original_row"))
    with pytest.raises(ValueError, match="original"):
        _run(view, **options)


def test_named_grid_requires_actual_original_coverage_for_each_research_asset():
    view = _prices().with_columns(
        pl.lit(False).alias("original_row"), pl.lit(False).alias("original_quote")
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    with pytest.raises(ValueError, match="original row coverage"):
        labeling.forward_returns(
            view,
            horizons={"one": 1},
            feed_spec=feed,
            price_basis="raw",
            original_row_col="original_row",
            original_price_col="original_quote",
        )


def test_input_role_cannot_shadow_an_internal_outcome_working_column():
    frame = _prices().rename({"close": "_out_available"})
    before = frame.clone()
    with pytest.raises(ValueError, match="reserved _out_"):
        _run(frame, price_col="_out_available")
    assert frame.equals(before)


@pytest.mark.parametrize("lazy", [False, True])
def test_session_with_no_full_bar_is_not_skipped_or_misreported_as_outside_tail(lazy):
    times = [
        pd.Timestamp(stamp).value for stamp in ["2025-07-02T13:30:00Z", "2025-07-07T13:30:00Z"]
    ]
    frame = pl.DataFrame({"timestamp": times, "asset": "A", "close": [100, 121]}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        data_frequency="4h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    output = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"one_session": 1, "two_sessions": 2},
        window_unit="session",
        price_basis="raw",
        feed_spec=feed,
        entries=frame.select("timestamp", "asset").head(1),
    )
    output = output.collect() if lazy else output
    # July3 is a real3.5-hour session, with no full4-hour bar under exclusion.
    assert output["outcome_status"].to_list() == ["missing_endpoint", "ok"]
    assert output["requested_exit_time"].to_list() == [None, frame["timestamp"][1]]
    assert output["forward_return"].to_list() == [None, pytest.approx(0.21)]
    with pytest.raises(ValueError, match="missing scheduled bar"):
        labeling.forward_returns(
            frame,
            horizons={"one": 1},
            window_unit="session",
            price_basis="raw",
            feed_spec=feed.with_overrides(partial_bar_policy="include"),
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_overnight_session_is_selected_by_exchange_open_not_utc_civil_date(lazy):
    times = [pd.Timestamp("2025-01-06T21:00:00Z").value]
    times += pd.date_range("2025-01-06T23:00:00Z", periods=23, freq="h").as_unit("ns").asi8.tolist()
    frame = pl.DataFrame(
        {"timestamp": times, "asset": "A", "close": [100 + i for i in range(24)]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="CME_Equity",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    result = labeling.forward_returns(
        frame.lazy() if lazy else frame,
        horizons={"next_session": 1},
        feed_spec=feed,
        window_unit="session",
        price_basis="raw",
    )
    result = result.collect() if lazy else result
    assert result["exit_time"][0] == pd.Timestamp("2025-01-07T21:00:00Z")
    assert result["forward_return"][0] == pytest.approx(0.23)
    # 23UTC on Jan6 already belongs to the Jan7 session, not the Jan6 session.
    assert result["outcome_status"][1] == "insufficient_tail"


@pytest.mark.parametrize(
    "unit,horizon,exit_stamp,expected",
    [
        ("session", 1, "2025-03-10T19:30:00Z", 0.07),
        ("expected_bar", 1, "2025-03-10T13:30:00Z", 0.01),
        ("elapsed", "1h", "2025-03-10T13:30:00Z", 0.01),
    ],
)
def test_real_nyse_dst_weekend_distinguishes_bar_session_and_elapsed_targets(
    unit, horizon, exit_stamp, expected
):
    times = [pd.Timestamp("2025-03-07T20:30:00Z").value]
    times += pd.date_range("2025-03-10T13:30:00Z", periods=7, freq="h").as_unit("ns").asi8.tolist()
    frame = pl.DataFrame(
        {"timestamp": times, "asset": "A", "close": [100 + i for i in range(8)]}
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    result = labeling.forward_returns(
        frame, horizons={"future": horizon}, window_unit=unit, feed_spec=feed, price_basis="raw"
    )
    assert result["exit_time"][0] == pd.Timestamp(exit_stamp)
    assert result["forward_return"][0] == pytest.approx(expected)
    if unit == "elapsed":
        assert result["requested_exit_time"][0] == pd.Timestamp("2025-03-07T21:30:00Z")
    if unit == "session":
        assert (
            result["exit_time"].cast(pl.Int64)[0] - result["timestamp"].cast(pl.Int64)[0]
            == 71 * 3600 * 10**9
        )


@pytest.mark.parametrize("selector", ["omitted", None, "24/7", "override", "disable"])
def test_calendar_precedence_is_applied_to_actual_outcomes_without_losing_presence(selector):
    from ml4t.specs import UNSET_CALENDAR

    times = [pd.Timestamp("2025-07-03T13:30:00Z").value + i * 1800 * 10**9 for i in range(3)]
    frame = _prices().with_columns(
        pl.Series("timestamp", times, dtype=pl.Int64).cast(pl.Datetime("ns", "UTC"))
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar=UNSET_CALENDAR
        if selector == "omitted"
        else (None if selector is None else "24/7"),
        entity_col="asset",
        data_frequency="30m",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    kwargs = (
        {"calendar": "NYSE"}
        if selector == "override"
        else ({"calendar": None} if selector == "disable" else {})
    )
    before = feed.to_dict()
    result = labeling.forward_returns(
        frame, horizons={"two": 2}, feed_spec=feed, price_basis="raw", **kwargs
    )
    assert result["forward_return"][0] == pytest.approx(0.21)
    assert result["calendar"].unique().to_list() == [
        "NYSE"
        if selector in {"omitted", "override"}
        else (None if selector in {None, "disable"} else "24/7")
    ]
    assert result["calendar_declared"].unique().to_list() == [selector != "omitted"]
    assert feed.to_dict() == before


def test_no_calendar_observation_returns_are_invariant_to_timestamp_spacing():
    regular = _prices()
    sparse = _prices(offsets=[0, 24, 24 * 30])
    assert _run(regular)["forward_return"].equals(_run(sparse)["forward_return"])
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        entity_col="asset",
        data_frequency="1h",
        timestamp_semantics="event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    with pytest.raises(ValueError, match="missing scheduled bar"):
        labeling.forward_returns(sparse, horizons={"two": 2}, feed_spec=feed, price_basis="raw")
    assert _run(sparse, feed_spec=feed)["forward_return"].equals(_run(regular)["forward_return"])


@pytest.mark.parametrize("route", ["native_outcome", "fixed_label", "price_expression"])
@pytest.mark.parametrize("method", ["returns", "log_returns"])
@pytest.mark.parametrize(
    "start,tick", [("9007199254740992", "1"), ("12345678901234.12345678901234", "0.00000000000001")]
)
def test_decimal_price_scale_is_preserved_before_fractional_return_conversion(
    route, method, start, tick
):
    import math
    from decimal import Decimal, localcontext

    from ml4t.engineer.features.utils import arithmetic

    with localcontext() as context:
        context.prec = 50
        initial, change = Decimal(start), Decimal(tick)
        prices = [initial + change * index for index in range(3)]
        expected = float(Fraction(prices[2] - prices[0]) / Fraction(prices[0]))
    if method == "log_returns":
        expected = math.log1p(expected)
    frame = _prices(prices)
    before = frame.clone()
    if route == "native_outcome":
        values = _run(frame, return_method=method)["forward_return"]
        actual = values[0]
    elif route == "fixed_label":
        output = labeling.fixed_time_horizon_labels(
            frame, horizon=2, method=method, timestamp_col="timestamp"
        )
        values = output["label_log_return_2p" if method == "log_returns" else "label_return_2p"]
        actual = values[0]
    else:
        values = frame.select(getattr(arithmetic, method)("close", periods=2)).to_series()
        actual = values[2]
    assert float(actual) == pytest.approx(expected, rel=2e-15, abs=0)
    assert values.dtype == pl.Float64
    assert frame.equals(before)


@pytest.mark.parametrize("positive_first", [False, True])
@pytest.mark.parametrize("method", ["returns", "log_returns"])
def test_decimal_opposite_sign_limits_do_not_overflow_before_domain_selection(
    positive_first, method
):
    import math
    from decimal import Decimal

    from ml4t.engineer.features.utils import arithmetic

    maximum = Decimal("9" * 38)
    prices = (
        [maximum, maximum.copy_negate()] if positive_first else [maximum.copy_negate(), maximum]
    )
    values = (
        pl.DataFrame({"price": prices}).select(getattr(arithmetic, method)("price")).to_series()
    )
    assert values.dtype == pl.Float64
    assert values[0] is None
    if method == "returns":
        assert values[1] == -2.0
    else:
        assert math.isnan(values[1])


@pytest.mark.parametrize("duration,step", [(pd.Timedelta("1ns"), 1), (pd.Timedelta("1us"), 1000)])
def test_actual_persisted_horizon_value_can_replay_the_native_outcome_without_an_extra_parser(
    duration, step
):
    import json

    from ml4t.engineer.artifacts import LabelSpec

    frame = _prices(offsets=[0, step, 2 * step], unit="ns")
    feed = FeedSpec(time_contract_version=2, calendar=None, entity_col="asset")
    output = labeling.forward_returns(
        frame,
        horizons={"future": duration},
        window_unit="elapsed",
        feed_spec=feed,
        price_basis="raw",
    )
    spec = LabelSpec.from_mapping(
        {
            "artifact_id": "actual-outcome-replay",
            "definition": {
                "horizon": output["horizon_value"][0],
                "window_unit": "elapsed",
                "endpoint_method": "cumulative",
                "return_method": "returns",
                "missing_price_policy": "raise",
                "price_basis": "raw",
                "feed_spec": feed.to_dict(),
            },
        }
    )
    restored = LabelSpec.from_mapping(json.loads(json.dumps(spec.to_dict())))
    replayed = labeling.forward_returns(
        frame,
        horizons={"future": restored.definition.horizon},
        window_unit=restored.definition.window_unit,
        feed_spec=restored.definition.feed_spec,
        price_basis=restored.definition.price_basis,
    )
    assert replayed.equals(output)
    assert output["horizon_value"].unique().to_list() == [f"{step}ns"]
