"""The public declared-grid producer must preserve and validate submicrosecond bars."""

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.engineer import compute_features


def _prices(offsets, *, calendar="24/7"):
    opened = pd.Timestamp(
        "2025-07-03T13:30:00Z" if calendar == "NYSE" else "2025-07-03T00:00:00Z"
    ).value
    values = [100 + index for index in range(len(offsets))]
    return pl.DataFrame(
        {
            "timestamp": pl.Series([opened + offset for offset in offsets], dtype=pl.Int64).cast(
                pl.Datetime("ns", "UTC")
            ),
            "asset": "A",
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 1,
        }
    )


def _feed(calendar="24/7", side="left", frequency="1ns"):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        entity_col="asset",
        timezone="UTC",
        data_frequency=frequency,
        bar_type="time",
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label=side,
        partial_bar_policy="exclude",
    )


@pytest.mark.parametrize("calendar", ["24/7", "NYSE"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("lazy", [False, True])
def test_complete_nanosecond_bars_remain_distinct_through_public_grid_validation(
    calendar, side, lazy
):
    start = 0 if side == "left" else 1
    frame = _prices([start, start + 1, start + 2], calendar=calendar)
    before = frame.clone()
    output = compute_features(frame.lazy() if lazy else frame, [], feed_spec=_feed(calendar, side))
    if lazy:
        assert isinstance(output, pl.LazyFrame)
        output = output.collect()
    assert output.height == 3
    assert (
        output["timestamp"].cast(pl.Int64).to_list() == before["timestamp"].cast(pl.Int64).to_list()
    )
    assert output.equals(before)
    assert frame.equals(before)


@pytest.mark.parametrize("lazy", [False, True])
def test_one_missing_nanosecond_bar_fails_before_any_return_is_derived(lazy):
    frame = _prices([0, 2, 3])
    before = frame.clone()
    with pytest.raises(ValueError, match="missing scheduled bar"):
        compute_features(frame.lazy() if lazy else frame, [], feed_spec=_feed())
    assert frame.equals(before)


@pytest.mark.parametrize("frequency", ["0ns", "-1ns"])
def test_nonpositive_declared_ns_frequency_fails_atomically(frequency):
    frame = _prices([0, 1])
    before = frame.clone()
    with pytest.raises(ValueError, match="frequency"):
        compute_features(frame, [], feed_spec=_feed(frequency=frequency))
    assert frame.equals(before)


def test_submicrosecond_off_grid_timestamp_is_not_rounded_into_valid_bar():
    frame = _prices([0, 3, 4])
    with pytest.raises(ValueError, match="missing scheduled bar|off-grid bar"):
        compute_features(frame, [], feed_spec=_feed(frequency="2ns"))


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_declared_grid_preserves_compound_asset_identity_without_hashing_or_stringifying(
    daily, lazy
):
    frame = _prices(
        [0, (4 * 86400 if daily else 3600) * 10**9], calendar="NYSE" if daily else "24/7"
    )
    frame = frame.with_columns(
        pl.struct(pl.lit("source_a").alias("namespace"), pl.lit(7).alias("sid")).alias("asset")
    )
    feed = _feed(frequency="1h")
    if daily:
        frame = frame.with_columns(pl.col("timestamp").dt.date())
        feed = FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            entity_col="asset",
            data_frequency="1d",
            bar_type="time",
            timestamp_semantics="session_label",
        )
    output = compute_features(frame.lazy() if lazy else frame, [], feed_spec=feed)
    output = output.collect() if lazy else output
    assert output.equals(frame)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "frequency,period,offsets,expected",
    [
        ("500ns", pd.Timedelta("1us"), [0, 500, 1000, 1500], [None, None, 101.5, 102.5]),
        ("1ns", pd.Timedelta("1ns"), [0, 1, 2], [None, 101.0, 102.0]),
    ],
)
def test_elapsed_feature_uses_exact_ns_open_left_window_not_rounded_microseconds(
    lazy, frequency, period, offsets, expected
):
    frame = _prices(offsets)
    output = compute_features(
        frame.lazy() if lazy else frame,
        [{"name": "sma", "params": {"period": period}}],
        feed_spec=_feed(frequency=frequency),
        window_unit="elapsed",
    )
    output = output.collect() if lazy else output
    assert output["sma"].to_list() == expected
