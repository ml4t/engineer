"""Valid public grids remain valid in pandas2/NumPy2 legacy-coexistence environments."""

from datetime import date

import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.engineer import compute_features


def bars(timestamp: pl.Series) -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": timestamp, "asset": ["A"] * 4, "close": [100.0, 110.0, 120.0, 130.0]}
    ).with_columns(
        pl.col("close").alias("open"),
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
        pl.lit(100).alias("volume"),
    )


def test_daily_public_feature_grid_does_not_misclassify_a_dependency_warning_as_bad_frequency():
    frame = bars(pl.Series([date(2025, 7, d) for d in (7, 8, 9, 10)]))
    result = compute_features(
        frame.lazy(),
        [{"name": "sma", "params": {"period": 2}}],
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            entity_col="asset",
            timestamp_semantics="session_label",
            bar_type="time",
            data_frequency="1d",
        ),
    ).collect()
    assert result["sma"].to_list() == [None, 105.0, 115.0, 125.0]
    assert result["timestamp"].equals(frame["timestamp"])


@pytest.mark.parametrize(
    "frequency,step",
    [("1h", 3_600_000_000_000), ("30m", 1_800_000_000_000), ("1ms", 1_000_000), ("1ns", 1)],
)
def test_intraday_public_declared_grid_keeps_exact_frequency_and_rows_across_dependency_versions(
    frequency, step
):
    opened = 1751846400000000000
    frame = bars(pl.Series([opened + i * step for i in range(4)], dtype=pl.Datetime("ns", "UTC")))
    contract = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        entity_col="asset",
        timezone="UTC",
        timestamp_semantics="event_time",
        bar_type="time",
        data_frequency=frequency,
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="exclude",
    )
    result = compute_features(frame, [{"name": "sma", "params": {"period": 2}}], feed_spec=contract)
    assert result["sma"].to_list() == [None, 105.0, 115.0, 125.0]
    assert result["timestamp"].equals(frame["timestamp"])


@pytest.mark.parametrize("frequency", ["NaT", "0h", "1month", "1M", "1Y", "1T", "1H"])
def test_invalid_or_deprecated_ambiguous_frequencies_still_fail_before_calculation(frequency):
    step = 60_000_000_000 if frequency == "1T" else 3_600_000_000_000
    frame = bars(
        pl.Series(
            [1751846400000000000 + i * step for i in range(4)], dtype=pl.Datetime("ns", "UTC")
        )
    )
    with pytest.raises(ValueError, match="frequency"):
        compute_features(
            frame,
            ["sma"],
            feed_spec=FeedSpec(
                time_contract_version=2,
                calendar="24/7",
                timestamp_semantics="event_time",
                timezone="UTC",
                bar_grid_anchor="session_open",
                bar_label="left",
                partial_bar_policy="exclude",
                bar_type="time",
                data_frequency=frequency,
            ),
        )
