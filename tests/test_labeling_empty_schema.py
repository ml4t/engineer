from datetime import UTC, date, datetime
from io import BytesIO

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import calendar_aware_labels, triple_barrier_labels


@pytest.mark.parametrize("kind", ["date", "instant"])
@pytest.mark.parametrize("mode", ["session", "expected_bar", "observation", "standalone"])
@pytest.mark.parametrize("sparse_events", [False, True])
def test_filtering_all_prices_still_produces_a_portable_typed_label_table(
    kind, mode, sparse_events
):
    days = [date(2025, 7, day) for day in (3, 7)]
    if kind == "instant":
        days = [datetime(2025, 7, day, tzinfo=UTC) for day in (3, 7)]
    prices = pl.DataFrame({"timestamp": days, "close": [100.0, 101.0], "symbol": "A"})
    if sparse_events:
        prices = prices.with_columns(pl.col("timestamp").alias("event_time"))
    empty = prices.filter(pl.col("close") > 1000)
    config = LabelingConfig.triple_barrier(max_holding_period=1)
    if mode == "standalone":
        result = triple_barrier_labels(empty, config, timestamp_col="timestamp")
    else:
        feed = FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            timestamp_semantics="session_label",
            data_frequency="1d",
            bar_type="time",
            entity_col="symbol",
        )
        result = calendar_aware_labels(empty, config, feed_spec=feed, window_unit=mode)
    assert result.height == 0
    assert result.select(empty.columns).schema == empty.schema
    assert result.schema["barrier_hit"] == pl.String
    assert result.schema["label_time"] == prices.schema["timestamp"]
    buffer = BytesIO()
    result.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).schema == result.schema


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("events", ["dense", "sparse", "none"])
def test_timestamp_free_legacy_labels_keep_unknown_duration_portable_without_changing_holding_bars(
    empty, grouped, events
):
    prices = pl.DataFrame({"close": [100.0, 101.0, 102.0], "symbol": "A"})
    if not grouped:
        prices = prices.drop("symbol")
    if events != "dense":
        prices = prices.with_row_index("event_time").with_columns(
            pl.when(pl.col("event_time") % 2 == 0 if events == "sparse" else pl.col("close") > 1000)
            .then(pl.col("event_time"))
            .otherwise(None)
            .alias("event_time")
        )
    if empty:
        prices = prices.filter(pl.col("close") > 1000)
    config = LabelingConfig.triple_barrier(
        upper_barrier=None, lower_barrier=None, max_holding_period=1
    )
    result = triple_barrier_labels(prices, config)
    assert result.select(prices.columns).equals(prices)
    assert result.schema["label_duration"] == pl.Null
    assert result["label_duration"].is_null().all()
    if not empty:
        expected = {"dense": [1, 1, 0], "sparse": [1, None, 0], "none": [None] * 3}
        assert result["label_bars"].to_list() == expected[events]
        if events != "none":
            assert result["label_return"][0] == pytest.approx(0.01)
    buffer = BytesIO()
    result.write_parquet(buffer)
    buffer.seek(0)
    assert pl.read_parquet(buffer).equals(result)
