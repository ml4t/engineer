from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from io import BytesIO

import polars as pl
import pytest
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling import calendar_aware_labels, triple_barrier_labels
from ml4t.engineer.labeling.calendar import PandasMarketCalendar


def _config(**kwargs):
    return LabelingConfig.triple_barrier(
        upper_barrier=0.5, lower_barrier=0.5, max_holding_period=1, **kwargs
    )


def _prices(timestamps=None):
    values = {"row": [0, 1, 2, 3], "close": [100.0, 101.0, 99.0, 102.0]}
    if timestamps is not None:
        values["timestamp"] = timestamps
    return pl.DataFrame(values)


@pytest.mark.parametrize(
    "timestamps",
    [
        None,
        [datetime(2025, 7, 4) + timedelta(hours=i) for i in range(4)],
        [datetime(2025, 7, day) for day in (14, 4, 31, 7)],
    ],
)
def test_explicit_none_labels_ordered_observations_without_loading_calendar(
    monkeypatch, timestamps
):
    import sys

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    result = calendar_aware_labels(_prices(timestamps), _config(), calendar=None)
    assert result["row"].to_list() == [0, 1, 2, 3]
    assert result["label_return"].to_list() == pytest.approx([0.01, -2 / 101, 3 / 99, 0.0])
    assert result["label_bars"].to_list() == [1, 1, 1, 0]


def test_row_mode_keeps_interleaved_asset_order_and_scans_each_asset_independently():
    data = pl.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "close": [100.0, 200.0, 101.0, 198.0],
            "timestamp": [datetime(2025, 7, d) for d in (14, 11, 4, 7)],
            "row": [0, 1, 2, 3],
        }
    )
    result = calendar_aware_labels(data, _config(), calendar=None)
    assert result["row"].to_list() == [0, 1, 2, 3]
    assert result["label_return"].to_list() == pytest.approx([0.01, -0.01, 0.0, 0.0])


@pytest.mark.parametrize("suffix", ["json", "yaml"])
@pytest.mark.parametrize(
    "declaration,expected",
    [({}, "NYSE"), ({"calendar": None}, None), ({"calendar": "24/7"}, "24/7")],
)
def test_real_config_roundtrip_preserves_omitted_named_and_disabled_calendar(
    tmp_path, suffix, declaration, expected
):
    config = _config(**declaration)
    path = tmp_path / f"labeling.{suffix}"
    getattr(config, f"to_{suffix}")(path)
    loaded = getattr(LabelingConfig, f"from_{suffix}")(path)
    copied = deepcopy(loaded).model_copy()
    assert ("calendar" in copied.to_dict()) == ("calendar" in declaration)
    saturday = pl.DataFrame({"timestamp": [datetime(2025, 7, 5, 14, tzinfo=UTC)], "close": [100.0]})
    if expected == "NYSE":
        with pytest.raises(DataValidationError, match="outside"):
            calendar_aware_labels(saturday, copied, window_unit="observation")
    else:
        assert calendar_aware_labels(saturday, copied, window_unit="observation").height == 1


def test_feed_metadata_precedence_and_declared_null_survive_nested_json_handoff(tmp_path):
    feed = FeedSpec(
        time_contract_version=2,
        calendar=None,
        price_col="mid",
        timestamp_col="time",
        entity_col=None,
    )
    config = _config(feed_spec=feed)
    path = tmp_path / "metadata.json"
    config.to_json(path)
    loaded = LabelingConfig.from_json(path)
    data = pl.DataFrame(
        {"mid": [100.0, 101.0], "time": [datetime(2025, 7, 5), datetime(2025, 7, 4)]}
    )
    result = calendar_aware_labels(data, loaded)
    assert result["label_return"].to_list() == pytest.approx([0.01, 0.0])
    assert loaded.to_dict()["feed_spec"]["calendar"] is None
    with pytest.raises(DataValidationError, match="outside"):
        calendar_aware_labels(
            data,
            loaded,
            calendar="NYSE",
            feed_spec=FeedSpec(
                time_contract_version=2,
                timezone="UTC",
                timestamp_semantics="event_time",
                price_col="mid",
                timestamp_col="time",
                entity_col=None,
            ),
            window_unit="observation",
        )


def test_explicit_argument_overrides_config_and_metadata_without_mutating_either():
    data = _prices([datetime(2025, 7, 5) + timedelta(hours=i) for i in range(4)])
    config = _config(calendar="NYSE", feed_spec=FeedSpec(time_contract_version=2, calendar="LSE"))
    before = config.to_dict()
    result = calendar_aware_labels(data, config, calendar=None)
    assert result.height == 4
    assert config.to_dict() == before
    assert_frame_equal(data, _prices([datetime(2025, 7, 5) + timedelta(hours=i) for i in range(4)]))


def test_named_schedule_uses_declared_naive_timezone_instead_of_legacy_utc():
    data = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 7, 7, 9, 30), datetime(2025, 7, 7, 10)],
            "close": [100.0, 101.0],
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="America/New_York",
        timestamp_semantics="event_time",
    )
    result = calendar_aware_labels(data, _config(), feed_spec=feed, window_unit="observation")
    assert result["label_return"].to_list() == pytest.approx([0.01, 0.0])
    assert_frame_equal(result.select(data.columns), data)


@pytest.mark.parametrize("unit", ["session", "elapsed", "expected_bar"])
def test_disabled_calendar_rejects_non_observation_units(unit):
    with pytest.raises(ValueError, match="observation"):
        calendar_aware_labels(_prices(), _config(), calendar=None, window_unit=unit)


def test_expected_bar_labels_reject_price_outage_before_horizon_interpretation():
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="UTC",
        data_frequency="1h",
        timestamp_semantics="bar_close",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )
    complete = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 7, 7, i, tzinfo=UTC) for i in range(1, 5)],
            "close": [100.0, 101.0, 99.0, 102.0],
        }
    )
    assert (
        calendar_aware_labels(
            complete, _config(), feed_spec=feed, window_unit="expected_bar"
        ).height
        == 4
    )
    outage = complete.filter(pl.col("timestamp").dt.hour() != 2)
    with pytest.raises(ValueError, match="missing scheduled bar"):
        calendar_aware_labels(outage, _config(), feed_spec=feed, window_unit="expected_bar")


@pytest.mark.parametrize(
    "declaration,error",
    [
        ({"timestamp_semantics": "session_label"}, "declared bar grid"),
        ({"timestamp_semantics": "event_time"}, "timezone"),
    ],
)
def test_named_schedule_rejects_ambiguous_timestamp_declaration_atomically(declaration, error):
    data = _prices([datetime(2025, 7, 7, 14) + timedelta(minutes=i) for i in range(4)])
    original = data.clone()
    with pytest.raises(ValueError, match=error):
        calendar_aware_labels(
            data, _config(), feed_spec=FeedSpec(time_contract_version=2, **declaration)
        )
    assert_frame_equal(data, original)


def test_close_labeled_price_at_lunch_break_is_valid_and_does_not_cross_the_break():
    data = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 7, 7, h, m, tzinfo=UTC) for h, m in [(3, 30), (4, 0), (5, 30)]
            ],
            "close": [100.0, 101.0, 110.0],
        }
    )
    feed = FeedSpec(time_contract_version=2, calendar="HKEX", timestamp_semantics="bar_close")
    result = calendar_aware_labels(data, _config(), feed_spec=feed, window_unit="observation")
    assert result["label_return"].to_list() == pytest.approx([0.01, 0.0, 0.0])
    assert_frame_equal(result.select(data.columns), data)


def test_named_feed_does_not_infer_timestamp_meaning_from_aware_values():
    data = _prices([datetime(2025, 7, 7, 14, tzinfo=UTC) + timedelta(minutes=i) for i in range(4)])
    with pytest.raises(ValueError, match="timestamp_semantics"):
        calendar_aware_labels(data, _config(), feed_spec=FeedSpec(time_contract_version=2))


def test_legacy_kernel_does_not_silently_ignore_a_declared_calendar_opt_out():
    data = _prices([datetime(2025, 7, day) for day in (14, 4, 31, 7)])
    with pytest.raises(DataValidationError, match="calendar_aware_labels"):
        triple_barrier_labels(data, _config(calendar=None))


def test_new_default_does_not_treat_an_undeclared_intraday_grid_as_complete():
    data = _prices([datetime(2025, 7, 7, 14, tzinfo=UTC) + timedelta(hours=i) for i in range(4)])
    with pytest.raises(ValueError, match="declared bar grid"):
        calendar_aware_labels(data, _config())


def test_config_equality_does_not_collapse_disabled_and_undeclared_calendars():
    inherited = _config()
    disabled = inherited.model_copy(update={"calendar": None})
    assert disabled != inherited
    assert disabled == deepcopy(disabled)
    assert disabled.to_dict()["calendar"] is None
    assert "calendar" not in inherited.to_dict()


@pytest.mark.parametrize("mode", ["python", "json"])
def test_declared_null_calendar_does_not_leak_into_explicit_serialization_projection(mode):
    config = _config(calendar=None)
    assert config.model_dump(mode=mode, include={"method"}) == {"method": "triple_barrier"}
    assert "calendar" not in config.model_dump(mode=mode, exclude={"calendar"}, exclude_none=True)
    assert config.to_dict(mode=mode, exclude_none=True)["calendar"] is None


def test_schedule_adapter_canonical_calendar_preserves_legacy_selector_and_rejects_conflicts():
    assert PandasMarketCalendar(calendar="LSE").calendar_name == "LSE"
    assert PandasMarketCalendar(calendar_name="LSE", calendar="LSE").calendar == "LSE"
    assert PandasMarketCalendar().calendar == "NYSE"
    with pytest.raises(ValueError, match="conflict"):
        PandasMarketCalendar(calendar_name="NYSE", calendar="LSE")
    with pytest.raises(ValueError, match="named calendar"):
        PandasMarketCalendar(calendar=None)


@pytest.mark.parametrize("precision", ["us", "ns"])
@pytest.mark.parametrize("workflow", ["calendar", "row", "standalone"])
def test_asset_without_selected_events_keeps_compatible_null_label_schema(precision, workflow):
    data = pl.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "timestamp": [datetime(2025, 7, 7, hour, tzinfo=UTC) for hour in (14, 14, 15, 15)],
            "close": [100.0, 200.0, 101.0, 201.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(precision, "UTC")))
    selected = data.with_columns(
        pl.when(pl.col("symbol") == "A")
        .then(pl.col("timestamp"))
        .otherwise(None)
        .alias("event_time")
    )
    if workflow == "standalone":
        result = triple_barrier_labels(selected, _config(), calculate_uniqueness=True)
        assert result.filter(pl.col("symbol") == "B")["sample_weight"].is_null().all()
    else:
        result = calendar_aware_labels(
            selected, _config(), calendar=None if workflow == "row" else "NYSE"
        )
    assert result.height == data.height
    inactive = result.filter(pl.col("symbol") == "B")
    assert inactive["label_time"].is_null().all()
    assert inactive["label_duration"].is_null().all()
    assert result.filter(pl.col("symbol") == "A")["label_return"].to_list() == pytest.approx(
        [0.01, 0.0]
    )
    assert result["label_time"].dtype == (
        pl.Int64 if workflow == "row" else data["timestamp"].dtype
    )


def test_legacy_date_labels_report_calendar_day_difference_without_numpy_resolution_error():
    data = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, day) for day in (3, 7, 8)],
            "close": [100.0, 101.0, 99.0],
        }
    )
    result = triple_barrier_labels(data, _config())
    assert result["label_time"].dtype == pl.Date
    assert result["label_time"].to_list() == [date(2025, 7, 7), date(2025, 7, 8), date(2025, 7, 8)]
    assert result["label_duration"].to_list() == [
        timedelta(days=4),
        timedelta(days=1),
        timedelta(0),
    ]
    assert result["label_bars"].to_list() == [1, 1, 0]


def test_row_labels_persist_without_untyped_python_object_duration_metadata():
    result = calendar_aware_labels(_prices(), _config(), calendar=None)
    target = BytesIO()
    result.write_parquet(target)
    target.seek(0)
    assert_frame_equal(pl.read_parquet(target), result)
    assert result["label_duration"].is_null().all()
