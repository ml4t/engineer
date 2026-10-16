from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from ml4t.specs import (
    UNSET_CALENDAR,
    FeedSpec,
    MarketDataSchema,
    MarketDataSemantics,
    MarketDataSpec,
    TimestampSemantics,
    WindowUnit,
    read_spec_payload,
    write_spec_payload,
)

from ml4t.engineer import compute_features
from ml4t.engineer.core.exceptions import DataSchemaError


def prices(times):
    close = [100.0 + i * 3 for i in range(len(times))]
    return pl.DataFrame(
        {
            "timestamp": times,
            "symbol": "TEST",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def returns(frame):
    return frame.with_columns(pl.col("close").log().diff().over("symbol").alias("returns"))


REQUEST = [{"name": "realized_volatility", "params": {"period": 3, "annualize": False}}]


def feed(calendar):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        entity_col="symbol",
        data_frequency="1h",
        timezone="UTC",
        bar_type="time",
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )


def run(frame, lazy, **kwargs):
    result = compute_features(frame.lazy() if lazy else frame, REQUEST, **kwargs)
    return result.collect() if isinstance(result, pl.LazyFrame) else result


@pytest.mark.parametrize("lazy", [False, True])
def test_undeclared_grid_does_not_interpret_outage_return_as_one_bar(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in [0, 1, 2, 50, 51, 52]]))
    original = frame.clone()
    with pytest.raises(ValueError, match="declared.*grid"):
        run(frame, lazy)
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_continuous_hourly_outage_is_rejected_without_mutating_prices_or_returns(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in [0, 1, 2, 50, 51, 52]]))
    original = frame.clone()
    with pytest.raises(ValueError, match="missing.*2025-07-11.*03:00"):
        run(frame, lazy, calendar="24/7", feed_spec=feed("24/7"))
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_complete_continuous_hourly_prices_keep_row_count_and_expected_volatility(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    result = run(frame, lazy, calendar="24/7", feed_spec=feed("24/7"))
    expected = frame.select(pl.col("returns").rolling_std(3)).to_series()
    assert result.height == frame.height
    assert result["realized_volatility"].equals(expected.rename("realized_volatility"))


def nyse_times():
    zone = ZoneInfo("America/New_York")
    july_3 = [datetime(2025, 7, 3, h, 30, tzinfo=zone) for h in [10, 11, 12]]
    july_3.append(datetime(2025, 7, 3, 13, tzinfo=zone))
    july_7 = [datetime(2025, 7, 7, h, 30, tzinfo=zone) for h in range(10, 16)]
    july_7.append(datetime(2025, 7, 7, 16, tzinfo=zone))
    return july_3 + july_7


@pytest.mark.parametrize("lazy", [False, True])
def test_nyse_holiday_weekend_and_early_close_are_not_missing_hourly_bars(lazy):
    frame = returns(prices(nyse_times()))
    result = run(frame, lazy, calendar="NYSE", feed_spec=feed("NYSE"))
    assert result.height == frame.height
    assert result["timestamp"].equals(frame["timestamp"])


@pytest.mark.parametrize("lazy", [False, True])
def test_nyse_actual_missing_scheduled_bar_is_not_reclassified_as_a_closure(lazy):
    times = nyse_times()
    del times[5]  # Delete the price observation before deriving the returns.
    frame = returns(prices(times))
    original = frame.clone()
    with pytest.raises(ValueError, match="missing.*2025-07-07.*15:30"):
        run(frame, lazy, calendar="NYSE", feed_spec=feed("NYSE"))
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_none_is_row_invariant_on_regular_and_irregular_timestamps(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    results = []
    for hours in [range(8), [0, 1, 3, 73, 74, 200, 201, 300]]:
        frame = returns(prices([start + timedelta(hours=h) for h in hours]))
        result = run(frame, lazy, calendar=None, feed_spec=feed("NYSE"))
        assert result.height == frame.height
        results.append(result["realized_volatility"])
    assert results[0].equals(results[1])


@pytest.mark.parametrize("lazy", [False, True])
def test_row_mode_does_not_reorder_values_when_timestamp_order_changes(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    outputs = []
    for hours in [range(8), [200, 0, 73, 1, 2, 300, 7, 5]]:
        data = returns(prices([start + timedelta(hours=h) for h in hours]))
        output = run(data, lazy, calendar=None)
        assert output["timestamp"].equals(data["timestamp"])
        outputs.append(output["realized_volatility"])
    assert outputs[0].equals(outputs[1])


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
@pytest.mark.parametrize("calendar", ["NYSE", None, UNSET_CALENDAR])
def test_serialized_artifact_calendar_intent_reaches_real_feature_workflow(
    tmp_path, suffix, calendar
):
    declaration = feed("NYSE").to_dict()
    declaration["calendar"] = calendar
    if calendar is UNSET_CALENDAR:
        del declaration["calendar"]
    semantics = MarketDataSemantics.from_mapping(declaration)
    artifact = MarketDataSpec(
        artifact_id="prices",
        schema=MarketDataSchema(entity_col="symbol"),
        semantics=semantics,
    )
    path = tmp_path / f"prices{suffix}"
    write_spec_payload(artifact, path)
    metadata = read_spec_payload(path)
    assert ("calendar" in metadata["semantics"]) is (calendar is not UNSET_CALENDAR)
    data = returns(prices(nyse_times()))
    output = run(data, False, feed_spec=metadata)
    assert output.height == data.height
    if calendar is None:
        assert output["timestamp"].equals(data["timestamp"])
    else:
        incomplete = returns(prices(nyse_times()[:5] + nyse_times()[6:]))
        with pytest.raises(ValueError, match="missing"):
            run(incomplete, False, feed_spec=metadata)


def test_explicit_calendar_wins_over_metadata_including_none():
    data = returns(prices(nyse_times()))
    assert run(data, False, calendar="NYSE", feed_spec=feed("24/7")).height == data.height
    with pytest.raises(ValueError, match="missing|off-grid"):
        run(data, False, calendar="24/7", feed_spec=feed("NYSE"))
    assert run(data, False, calendar=None, feed_spec=feed("NYSE")).height == data.height
    disabled = feed("NYSE").with_overrides(calendar=None)
    assert run(data, False, feed_spec=disabled).height == data.height


@pytest.mark.parametrize("lazy", [False, True])
def test_price_grid_is_validated_before_deriving_returns_without_changing_rows(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    complete = prices([start + timedelta(hours=h) for h in range(8)])
    data = complete.lazy() if lazy else complete
    checked = compute_features(data, [], feed_spec=feed("24/7"))
    checked = checked.collect() if isinstance(checked, pl.LazyFrame) else checked
    assert checked.equals(complete)
    result = run(returns(checked), lazy, feed_spec=feed("24/7"))
    assert result.height == complete.height
    missing_prices = complete.filter(pl.col("timestamp") != start + timedelta(hours=3))
    with pytest.raises(ValueError, match="missing"):
        compute_features(
            missing_prices.lazy() if lazy else missing_prices,
            [],
            feed_spec=feed("24/7"),
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_reindexing_already_derived_returns_does_not_hide_a_cross_gap_return(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    raw = prices([start + timedelta(hours=h) for h in [0, 1, 2, 5, 6, 7]])
    unsafe = (
        returns(raw)
        .upsample("timestamp", every="1h")
        .with_columns(pl.col("symbol").fill_null("TEST"))
    )
    before = unsafe.clone()
    with pytest.raises(ValueError, match="Return adjacent to a missing price"):
        run(unsafe, lazy, feed_spec=feed("24/7"))
    assert unsafe.equals(before)
    safe_prices = raw.upsample("timestamp", every="1h").with_columns(
        pl.col("symbol").fill_null("TEST")
    )
    safe = run(returns(safe_prices), lazy, feed_spec=feed("24/7"))
    assert safe.height == 8
    assert safe["realized_volatility"][5] is None


@pytest.mark.parametrize("lazy", [False, True])
def test_invalid_grid_never_executes_any_feature(monkeypatch, lazy):
    import ml4t.engineer.api as api

    executed = []

    def unexpected(*_args, **_kwargs):
        executed.append(True)
        raise AssertionError("Feature execution must follow complete grid validation")

    monkeypatch.setattr(api, "_execute_feature", unexpected)
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start, start + timedelta(hours=2)]))
    with pytest.raises(ValueError, match="missing"):
        run(data, lazy, feed_spec=feed("24/7"))
    assert executed == []


@pytest.mark.parametrize("unit", [WindowUnit.SESSION, WindowUnit.ELAPSED, WindowUnit.OBSERVATION])
def test_calendar_grid_does_not_reinterpret_other_window_units_as_expected_bars(unit):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    with pytest.raises(ValueError, match="window_unit='expected_bar'"):
        run(data, False, feed_spec=feed("24/7"), window_unit=unit)
    if unit is WindowUnit.OBSERVATION:
        assert run(data, False, calendar=None, window_unit=unit).height == 8
    else:
        with pytest.raises(ValueError, match="window_unit='observation'"):
            run(data, False, calendar=None, window_unit=unit)


@pytest.mark.parametrize(
    "field",
    [
        "data_frequency",
        "bar_grid_anchor",
        "bar_label",
        "partial_bar_policy",
        "timestamp_semantics",
        "bar_type",
    ],
)
def test_missing_shared_grid_declaration_fails_instead_of_guessing(field):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    with pytest.raises(ValueError, match="declared bar grid"):
        run(data, False, feed_spec=replace(feed("24/7"), **{field: None}))


def test_naive_bars_need_declared_timezone_but_aware_bars_already_declare_it():
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    no_timezone = replace(feed("24/7"), timezone=None)
    assert run(data, False, feed_spec=no_timezone).height == 8
    naive = data.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    with pytest.raises(ValueError, match="timezone metadata"):
        run(naive, False, feed_spec=no_timezone)


def test_legacy_v1_metadata_is_not_silently_reinterpreted_as_explicit_disable():
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    legacy = FeedSpec(calendar=None, entity_col="symbol")
    assert legacy.calendar is None
    assert legacy.resolve_calendar() == "NYSE"
    with pytest.raises(ValueError, match="time_contract_version=2"):
        run(data, False, feed_spec=legacy)
    assert run(data, False, feed_spec=legacy, calendar=None).height == 8


@pytest.mark.parametrize("frequency", ["NaT", "1ns", "1d", "1month", 3600, "0h"])
def test_unsupported_frequency_is_not_inferred_from_observation_spacing(frequency):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    with pytest.raises(ValueError, match="frequency"):
        run(data, False, feed_spec=replace(feed("24/7"), data_frequency=frequency))


@pytest.mark.parametrize("lazy", [False, True])
def test_metadata_column_names_and_calendar_sorting_apply_before_features(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)])).rename(
        {"timestamp": "when", "symbol": "instrument"}
    )
    contract = replace(feed("24/7"), timestamp_col="when", entity_col="instrument")
    output = run(data.reverse(), lazy, feed_spec=contract)
    assert output["when"].equals(data["when"])
    assert output["realized_volatility"][3] is not None


@pytest.mark.parametrize("lazy", [False, True])
def test_missing_group_identifier_fails_before_feature_evaluation(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)])).with_columns(
        pl.when(pl.col("timestamp") == start).then(None).otherwise(pl.col("symbol")).alias("symbol")
    )
    with pytest.raises(DataSchemaError, match="null"):
        run(data, lazy, feed_spec=feed("24/7"))


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("lazy", [False, True])
def test_empty_and_singleton_declared_grids_keep_the_input_shape(count, lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start])).head(count)
    output = run(data, lazy, feed_spec=feed("24/7"))
    assert output.height == count
    assert output.select(data.columns).equals(data)


def test_row_mode_does_not_load_optional_schedule_provider(monkeypatch):
    import builtins

    original = builtins.__import__

    def no_provider(name, *args, **kwargs):
        if name == "pandas_market_calendars":
            raise ImportError("Provider unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_provider)
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    assert run(data, False, calendar=None).height == 8
    with pytest.raises(ImportError, match="required"):
        run(data, False, feed_spec=feed("24/7"))


@pytest.mark.parametrize("label, expected", [("left", [9, 10, 11]), ("right", [10, 11, 12])])
def test_declared_event_bar_side_and_partial_policy_do_not_guess_from_midnight(label, expected):
    zone = ZoneInfo("America/New_York")
    data = returns(prices([datetime(2025, 7, 3, h, 30, tzinfo=zone) for h in expected]))
    contract = replace(
        feed("NYSE"),
        timestamp_semantics="event_time",
        bar_label=label,
        partial_bar_policy="exclude",
    )
    assert run(data, False, feed_spec=contract).height == 3


@pytest.mark.parametrize(
    "field", ["open_col", "high_col", "low_col", "close_col", "volume_col", "price_col"]
)
def test_unsupported_price_column_mapping_is_rejected_instead_of_using_other_prices(field):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    contract = replace(feed("24/7"), **{field: "other_prices"})
    with pytest.raises(ValueError, match="canonical|price_col"):
        run(data, False, feed_spec=contract)


def test_shared_single_entity_sequence_is_not_flattened_into_a_column_name():
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    assert run(data, False, feed_spec=replace(feed("24/7"), entity_col=["symbol"])).height == 8


@pytest.mark.timeout(5)
@pytest.mark.parametrize("lazy", [False, True])
def test_sparse_fine_grid_rejects_first_missing_bar_without_materializing_billions(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start, start + timedelta(hours=1)]))
    contract = replace(feed("24/7"), data_frequency="1us")
    with pytest.raises(ValueError, match=r"missing.*2025-07-11T00:00:00\.000001"):
        run(data, lazy, feed_spec=contract)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("nanoseconds", [0, 1])
def test_nanosecond_input_does_not_silently_truncate_off_grid_timestamps(lazy, nanoseconds):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    data = returns(prices([start + timedelta(hours=h) for h in range(8)])).with_columns(
        pl.col("timestamp").dt.cast_time_unit("ns") + pl.duration(nanoseconds=nanoseconds)
    )
    if nanoseconds:
        with pytest.raises(ValueError, match="off-grid.*precision"):
            run(data, lazy, feed_spec=feed("24/7"))
    else:
        output = run(data, lazy, feed_spec=feed("24/7"))
        assert output["timestamp"].equals(data["timestamp"])
