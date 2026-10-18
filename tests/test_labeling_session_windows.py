from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import calendar_aware_labels, triple_barrier_labels


def _feed(**kwargs):
    return FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timestamp_semantics="session_label",
        data_frequency="1d",
        bar_type="time",
        **kwargs,
    )


def _config(horizon=1, **kwargs):
    values = {
        "upper_barrier": None,
        "lower_barrier": None,
        "max_holding_period": horizon,
        "window_unit": "session",
    }
    values.update(kwargs)
    return LabelingConfig.triple_barrier(**values)


@pytest.mark.parametrize("representation", ["date", "naive", "UTC", "Asia/Tokyo"])
@pytest.mark.parametrize("sessions,exits", [(1, [1, 2, 3, 4, 4]), (2, [2, 3, 4, 4, 4])])
def test_session_labels_count_scheduled_sessions_not_elapsed_holiday_days(
    representation, sessions, exits
):
    days = [date(2025, 7, day) for day in (2, 3, 7, 8, 9)]
    timestamps = days
    if representation != "date":
        zone = None if representation == "naive" else ZoneInfo(representation)
        timestamps = [datetime.combine(day, time(0, 15), tzinfo=zone) for day in days]
    prices = pl.DataFrame({"timestamp": timestamps, "close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    buffer = BytesIO()
    prices.write_parquet(buffer)
    buffer.seek(0)
    restored = pl.read_parquet(buffer)
    result = calendar_aware_labels(restored.reverse(), _config(sessions), feed_spec=_feed())
    assert result.select(prices.columns).equals(prices)
    assert result["label_time"].to_list() == [timestamps[index] for index in exits]
    assert result["label_bars"].to_list() == [exit - entry for entry, exit in enumerate(exits)]
    assert result["label_return"].to_list() == pytest.approx(
        [prices["close"][exit] / prices["close"][entry] - 1 for entry, exit in enumerate(exits)]
    )
    assert result["barrier_hit"].to_list() == ["time"] * 5


@pytest.mark.parametrize("unit", ["expected_bar", "observation"])
def test_daily_count_horizon_crosses_valid_sessions_instead_of_zero_length_segments(unit):
    timestamps = [date(2025, 7, day) for day in (3, 7, 8)]
    prices = pl.DataFrame({"timestamp": timestamps, "close": [100.0, 101.0, 103.0]})
    result = calendar_aware_labels(prices, _config(window_unit=unit), feed_spec=_feed())
    assert result["label_return"].to_list() == pytest.approx([0.01, 2 / 101, 0.0])
    assert result["label_time"].to_list() == [timestamps[1], timestamps[2], timestamps[2]]


@pytest.mark.parametrize("representation", ["date", "naive", "UTC", "Asia/Tokyo"])
@pytest.mark.parametrize("unit", ["observation", "expected_bar", "session"])
@pytest.mark.parametrize("empty", [False, True])
def test_session_identifiers_do_not_fabricate_physical_elapsed_durations(
    representation, unit, empty
):
    days = [date(2025, 7, day) for day in (3, 7)]
    timestamps = days
    if representation != "date":
        zone = None if representation == "naive" else ZoneInfo(representation)
        timestamps = [datetime.combine(day, time(0, 15), tzinfo=zone) for day in days]
    prices = pl.DataFrame({"timestamp": timestamps, "close": [100.0, 101.0]})
    if empty:
        prices = prices.filter(pl.col("close") > 1000)
    result = calendar_aware_labels(prices, _config(window_unit=unit), feed_spec=_feed())
    assert result.select(prices.columns).equals(prices)
    assert result["label_duration"].is_null().all()
    assert result.schema["label_duration"] != pl.Object
    if not empty:
        assert result["label_time"].to_list() == [timestamps[1], timestamps[1]]
        assert result["label_bars"].to_list() == [1, 0]
        assert result["label_return"].to_list() == pytest.approx([0.01, 0.0])
    buffer = BytesIO()
    result.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).equals(result)


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_serialized_session_identifiers_keep_elapsed_duration_unset(tmp_path, suffix):
    days = [date(2025, 7, day) for day in (3, 7)]
    first = pl.DataFrame(
        {"timestamp": days, "close": [100.0, 101.0], "symbol": "A", "event_time": [days[0], None]}
    )
    second = first.with_columns((pl.col("close") * 2).alias("close"), pl.lit("B").alias("symbol"))
    prices = pl.concat([second, first]).sort("timestamp", "symbol")
    config = _config(feed_spec=_feed(entity_col="symbol"))
    path = tmp_path / f"labels.{suffix}"
    getattr(config, f"to_{suffix}")(path)
    restored = getattr(LabelingConfig, f"from_{suffix}")(path)
    result = calendar_aware_labels(prices, restored)
    assert result["label_duration"].is_null().all()
    for symbol in ("A", "B"):
        group = result.filter(pl.col("symbol") == symbol)
        assert group["label_bars"].to_list() == [1, None]
        assert group["label_return"].to_list() == pytest.approx([0.01, None])


def test_standalone_date_labels_keep_legacy_calendar_day_differences():
    prices = pl.DataFrame(
        {"timestamp": [date(2025, 7, day) for day in (3, 7)], "close": [100.0, 101.0]}
    )
    config = LabelingConfig.triple_barrier(
        upper_barrier=None, lower_barrier=None, max_holding_period=1
    )
    result = triple_barrier_labels(prices, config)
    assert result["label_duration"][0] == timedelta(days=4)
    assert result["label_return"][0] == pytest.approx(0.01)


def _intraday():
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("NYSE"),
            datetime(2025, 7, 2, tzinfo=UTC),
            datetime(2025, 7, 9, tzinfo=UTC),
            timedelta(hours=1),
            "right",
            True,
        )
    )
    prices = pl.DataFrame(
        {"timestamp": timestamps, "close": [100.0 + i for i in range(len(timestamps))]}
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timestamp_semantics="bar_close",
        data_frequency="1h",
        timezone="UTC",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )
    return prices, feed


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("observation", timedelta(hours=1)),
        ("expected_bar", timedelta(hours=1)),
        ("session", timedelta(hours=26, minutes=30)),
    ],
)
def test_instant_labeling_keeps_real_elapsed_durations(unit, expected):
    prices, feed = _intraday()
    result = calendar_aware_labels(prices, _config(window_unit=unit), feed_spec=feed)
    assert result["label_duration"][0] == expected


@pytest.mark.parametrize("representation", ["UTC", "America/New_York", "naive"])
@pytest.mark.parametrize("unit", ["observation", "expected_bar", "elapsed", "session"])
@pytest.mark.parametrize("sparse", [False, True])
def test_declared_local_clocks_use_physical_elapsed_time_through_spring_dst(
    representation, unit, sparse
):
    instants = [datetime(2025, 3, 9, 5, tzinfo=UTC) + timedelta(hours=i) for i in range(6)]
    timestamps = pl.Series("timestamp", instants)
    if representation != "UTC":
        timestamps = timestamps.dt.convert_time_zone("America/New_York")
    if representation == "naive":
        timestamps = timestamps.dt.replace_time_zone(None)
    prices = pl.DataFrame({"timestamp": timestamps, "close": [100.0 + i for i in range(6)]})
    if sparse:
        prices = prices.with_columns(
            pl.when(pl.col("close") == 101).then(pl.col("timestamp")).alias("event_time")
        )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="America/New_York",
        timestamp_semantics="event_time",
        data_frequency="1h",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    config = _config("2h" if unit == "elapsed" else 1 if unit == "session" else 2, window_unit=unit)
    result = calendar_aware_labels(prices, config, feed_spec=feed)
    assert result.select(prices.columns).equals(prices)
    assert result.schema["label_time"] == timestamps.dtype
    exits = [5] * 6 if unit == "session" else [2, 3, 4, 5, 5, 5]
    for entry, exit_index in enumerate(exits):
        if sparse and entry != 1:
            assert result["label_time"][entry] is None
            assert result["label_duration"][entry] is None
            continue
        assert result["label_time"][entry] == timestamps[exit_index]
        assert result["label_bars"][entry] == exit_index - entry
        assert result["label_return"][entry] == pytest.approx(
            (100 + exit_index) / (100 + entry) - 1
        )
        assert result["label_duration"][entry] == instants[exit_index] - instants[entry]


@pytest.mark.parametrize("suffix", ["json", "yaml"])
@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("empty", [False, True])
def test_serialized_local_clock_labels_preserve_panel_columns_and_timestamp_precision(
    tmp_path, suffix, time_unit, empty
):
    instants = [datetime(2025, 3, 9, 5, tzinfo=UTC) + timedelta(hours=i) for i in range(6)]
    timestamps = (
        pl.Series("ts", instants, dtype=pl.Datetime(time_unit, "UTC"))
        .dt.convert_time_zone("America/New_York")
        .dt.replace_time_zone(None)
    )
    first = pl.DataFrame({"ts": timestamps, "price": [100.0 + i for i in range(6)], "asset": "A"})
    second = first.with_columns((pl.col("price") * 2).alias("price"), pl.lit("B").alias("asset"))
    prices = pl.concat([first, second]).sort("ts", "asset")
    if empty:
        prices = prices.filter(pl.col("price") > 1000)
    config = _config(
        "2h",
        window_unit="elapsed",
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="24/7",
            timezone="America/New_York",
            timestamp_semantics="event_time",
            timestamp_col="ts",
            price_col="price",
            entity_col="asset",
        ),
    )
    path = tmp_path / f"labels.{suffix}"
    getattr(config, f"to_{suffix}")(path)
    restored = getattr(LabelingConfig, f"from_{suffix}")(path)
    result = calendar_aware_labels(prices, restored)
    assert result.select(prices.columns).equals(prices.sort("asset", "ts"))
    assert result.schema["label_time"] == pl.Datetime(time_unit)
    assert result.schema["label_duration"] == pl.Duration(time_unit)
    if not empty:
        for asset in ("A", "B"):
            group = result.filter(pl.col("asset") == asset)
            assert group["label_time"][1] == datetime(2025, 3, 9, 4)
            assert group["label_duration"][0] == timedelta(hours=2)
            assert group["label_return"][1] == pytest.approx(2 / 101)
    buffer = BytesIO()
    result.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).equals(result)


def test_intraday_session_horizon_reaches_next_session_close_not_lunch_or_overnight_break():
    prices, feed = _intraday()
    result = calendar_aware_labels(prices, _config(), feed_spec=feed)
    # July 3 closes at 17:00 UTC; July 4 is closed. July 7 closes at 20:00 UTC.
    july2 = result.filter(pl.col("timestamp").dt.date() == date(2025, 7, 2))
    july3 = result.filter(pl.col("timestamp").dt.date() == date(2025, 7, 3))
    assert july2["label_time"].to_list() == [datetime(2025, 7, 3, 17, tzinfo=UTC)] * 7
    assert july3["label_time"].to_list() == [datetime(2025, 7, 7, 20, tzinfo=UTC)] * 4
    assert july3["label_bars"].to_list() == [10, 9, 8, 7]
    assert result["label_return"][0] == pytest.approx(110 / 100 - 1)


def test_missing_entire_scheduled_session_is_rejected_before_any_label_or_input_change():
    prices, feed = _intraday()
    incomplete = prices.filter(pl.col("timestamp").dt.date() != date(2025, 7, 3))
    before = incomplete.clone()
    with pytest.raises(ValueError, match="missing scheduled bar.*2025-07-03"):
        calendar_aware_labels(incomplete, _config(), feed_spec=feed)
    assert incomplete.equals(before)
    daily = pl.DataFrame(
        {"timestamp": [date(2025, 7, 2), date(2025, 7, 7)], "close": [100.0, 103.0]}
    )
    with pytest.raises(ValueError, match="missing scheduled session.*2025-07-03"):
        calendar_aware_labels(daily, _config(), feed_spec=_feed())


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_session_config_handoff_retains_panel_sparse_events_and_existing_barrier_kernel(
    tmp_path, suffix
):
    days = [date(2025, 7, day) for day in (2, 3, 7, 8)]
    first = pl.DataFrame(
        {
            "timestamp": days,
            "close": [100.0, 101.0, 104.0, 103.0],
            "symbol": "A",
            "event_time": [days[0], None, days[2], None],
        }
    )
    second = first.with_columns((pl.col("close") * 2).alias("close"), pl.lit("B").alias("symbol"))
    config = _config(2, upper_barrier=0.02, feed_spec=_feed())
    path = tmp_path / f"session.{suffix}"
    getattr(config, f"to_{suffix}")(path)
    loaded = getattr(LabelingConfig, f"from_{suffix}")(path)
    result = calendar_aware_labels(pl.concat([first, second]).reverse(), loaded)
    for symbol in ("A", "B"):
        asset = result.filter(pl.col("symbol") == symbol)
        assert asset["label"].to_list() == [1, None, 0, None]
        assert asset["label_time"].to_list() == [days[2], None, days[3], None]
        assert asset["label_bars"].to_list() == [2, None, 1, None]
        assert asset["barrier_hit"].to_list() == ["upper", None, "time", None]
    assert loaded.to_dict()["window_unit"] == "session"


@pytest.mark.parametrize("unit", ["session", "elapsed"])
def test_none_never_becomes_a_continuous_calendar_to_satisfy_temporal_labels(unit):
    prices = pl.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="observation"):
        calendar_aware_labels(prices, _config(), calendar=None, window_unit=unit)


def test_elapsed_horizon_rejects_session_identifiers_instead_of_inventing_midnight_instants():
    prices = pl.DataFrame(
        {"timestamp": [date(2025, 7, 3), date(2025, 7, 7)], "close": [100.0, 101.0]}
    )
    config = LabelingConfig.triple_barrier(max_holding_period="1d", window_unit="elapsed")
    with pytest.raises(ValueError, match="session_label.*elapsed"):
        calendar_aware_labels(prices, config, feed_spec=_feed())


@pytest.mark.parametrize("counts", [[0, 1, 1], [1.5, 1.0, 1.0], [float("nan"), 1.0, 1.0]])
def test_session_count_columns_reject_non_positive_non_integral_or_missing_counts(counts):
    prices = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, day) for day in (3, 7, 8)],
            "close": [100.0, 101.0, 103.0],
            "horizon": counts,
        }
    )
    with pytest.raises(ValueError, match="finite positive integral"):
        calendar_aware_labels(prices, _config("horizon"), feed_spec=_feed())


def test_per_event_session_counts_and_temporary_column_collisions_preserve_original_data():
    prices = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, day) for day in (2, 3, 7, 8)],
            "close": [100.0, 101.0, 103.0, 105.0],
            "horizon": [2, 1, 1, 1],
            "__ml4t_session_holding_bars": [19] * 4,
        }
    )
    result = calendar_aware_labels(prices, _config("horizon"), feed_spec=_feed())
    assert result["label_bars"].to_list() == [2, 1, 1, 0]
    assert result.select(prices.columns).equals(prices)


def test_overnight_cme_daily_session_labels_do_not_shift_to_the_prior_civil_date():
    days = [date(2025, 7, day) for day in (7, 8, 9)]
    prices = pl.DataFrame({"timestamp": days, "close": [100.0, 101.0, 103.0]})
    result = calendar_aware_labels(prices, _config(), feed_spec=_feed(), calendar="CME_Equity")
    assert result["label_time"].to_list() == [days[1], days[2], days[2]]
    assert result["label_return"].to_list() == pytest.approx([0.01, 2 / 101, 0.0])


def test_explicit_continuous_session_counts_include_weekends_without_becoming_row_mode():
    from dataclasses import replace

    days = [date(2025, 7, day) for day in (4, 5, 6, 7)]
    prices = pl.DataFrame({"timestamp": days, "close": [100.0, 101.0, 103.0, 105.0]})
    result = calendar_aware_labels(prices, _config(2), feed_spec=replace(_feed(), calendar="24/7"))
    assert result["label_time"].to_list() == [days[2], days[3], days[3], days[3]]
    with pytest.raises(ValueError, match="missing scheduled session"):
        calendar_aware_labels(
            prices.filter(pl.col("timestamp") != days[1]),
            _config(),
            calendar="24/7",
            feed_spec=_feed(),
        )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("horizon,first_exit", [(1, 0), (2, 1)])
def test_barless_short_session_never_pushes_a_session_deadline_into_a_later_session(
    side, sparse, horizon, first_exit
):
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("NYSE"),
            datetime(2025, 7, 2, tzinfo=UTC),
            datetime(2025, 7, 9, tzinfo=UTC),
            timedelta(hours=5),
            side,
            False,
        )
    )
    assert [stamp.date() for stamp in timestamps] == [date(2025, 7, d) for d in (2, 7, 8)]
    prices = pl.DataFrame({"timestamp": timestamps, "close": [100.0, 101.0, 102.0]})
    if sparse:
        prices = prices.with_columns(pl.Series("event_time", [timestamps[0], None, timestamps[2]]))
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="UTC",
        data_frequency="5h",
        timestamp_semantics="bar_close" if side == "right" else "event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label=side,
        partial_bar_policy="exclude",
    )
    result = calendar_aware_labels(prices, _config(horizon), feed_spec=feed)
    exits = [first_exit, None if sparse else 2, 2]
    assert result["label_time"].to_list() == [
        timestamps[index] if index is not None else None for index in exits
    ]
    assert result["label_bars"].to_list() == [
        index - entry if index is not None else None for entry, index in enumerate(exits)
    ]
    assert result["label_return"][0] == pytest.approx(first_exit / 100)


@pytest.mark.parametrize("representation", ["UTC", "America/New_York", "naive"])
def test_session_horizons_retain_actual_exchange_close_through_spring_dst(representation):
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("NYSE"),
            datetime(2025, 3, 7, tzinfo=UTC),
            datetime(2025, 3, 12, tzinfo=UTC),
            timedelta(hours=1),
            "right",
            True,
        )
    )
    series = pl.Series("timestamp", timestamps)
    if representation != "UTC":
        series = series.dt.convert_time_zone("America/New_York")
    if representation == "naive":
        series = series.dt.replace_time_zone(None)
    prices = pl.DataFrame(
        {"timestamp": series, "close": [100.0 + i for i in range(len(timestamps))]}
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="America/New_York",
        timestamp_semantics="bar_close",
        data_frequency="1h",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )
    result = calendar_aware_labels(prices, _config(), feed_spec=feed)
    expected = datetime(2025, 3, 10, 20, tzinfo=UTC)
    actual = result["label_time"]
    if representation == "naive":
        actual = actual.dt.replace_time_zone("America/New_York")
    assert actual.dt.convert_time_zone("UTC")[0] == expected
    assert result["label_bars"][0] == 13
    assert result["label_return"][0] == pytest.approx(0.13)
    assert result["label_duration"][0] == timedelta(hours=76, minutes=30)
    assert result.select(prices.columns).equals(prices)


def test_overnight_cme_session_horizon_counts_exchange_labels_not_utc_civil_dates():
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("CME_Equity"),
            datetime(2025, 7, 6, tzinfo=UTC),
            datetime(2025, 7, 10, tzinfo=UTC),
            timedelta(hours=1),
            "right",
            True,
        )
    )
    assert timestamps[0] == datetime(2025, 7, 6, 23, tzinfo=UTC)
    prices = pl.DataFrame(
        {"timestamp": timestamps, "close": [100.0 + i for i in range(len(timestamps))]}
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="CME_Equity",
        timezone="UTC",
        timestamp_semantics="bar_close",
        data_frequency="1h",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )
    result = calendar_aware_labels(prices, _config(), feed_spec=feed)
    # Sunday evening belongs to Monday's exchange session. One following session ends Tuesday.
    assert result["label_time"][0] == datetime(2025, 7, 8, 21, tzinfo=UTC)
    assert result["label_bars"][0] == 45
    assert result["label_return"][0] == pytest.approx(0.45)
