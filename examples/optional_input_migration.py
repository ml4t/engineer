"""Prepare supplied legacy market/loading containers without taking over analysis."""

import json
from datetime import UTC, datetime

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ml4t.engineer.features.inputs import (
    loadings_to_long,
    market_to_long,
    prepare_classification_inputs,
    prepare_loading_inputs,
    prepare_market_inputs,
)


def main() -> None:
    sessions = pd.date_range("2025-07-07", periods=5, tz="UTC")
    legacy = pd.DataFrame(
        {17: [10.0, 100.0, 20.0, 0.0, 30.0, None, 40.0, 20.0, 50.0, 10.0]},
        index=pd.MultiIndex.from_product([sessions, ["price", "volume"]]),
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar=None,
        entity_col="asset",
        timestamp_semantics="session_label",
    )
    raw = market_to_long(legacy, feed_spec=feed)
    assert isinstance(raw, pl.DataFrame)
    # These capture facts are supplied by this synthetic source, not inferred by Engineer.
    supplied = raw.with_columns(
        pl.col("timestamp").dt.offset_by("20h").alias("available_at"),
        pl.col("timestamp").dt.offset_by("20h").alias("completed_at"),
        pl.col("timestamp").dt.offset_by("13h").alias("decision_time"),
        pl.lit("USD").alias("currency"),
        pl.lit(1.0).alias("multiplier"),
        pl.lit("shares").alias("volume_unit"),
    )
    prepared = prepare_market_inputs(
        supplied.lazy(),
        feed_spec=feed,
        price_basis="raw",
        volume_basis="raw",
        purpose="causal",
        liquidity_window=2,
        completed_at_col="completed_at",
    ).collect()
    assert prepared["lagged_mean_notional_volume"].to_list() == [None, None, 500.0, None, None]
    assert prepared["close"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert prepared["volume"].to_list() == [100.0, 0.0, None, 20.0, 10.0]

    legacy_loadings = pd.DataFrame(
        {"MKT": [1.0, 2.0]},
        index=pd.MultiIndex.from_arrays([sessions[[1, 3]], [17, 17]]),
    )
    loadings = loadings_to_long(legacy_loadings, feed_spec=feed)
    assert isinstance(loadings, pl.DataFrame)
    loadings = loadings.with_columns(pl.col("timestamp").dt.offset_by("12h").alias("available_at"))
    requests = supplied.select("timestamp", "asset", "decision_time")
    selected = prepare_loading_inputs(
        loadings,
        requests=requests,
        factors=["MKT", "SIZE"],
        feed_spec=feed,
        purpose="causal",
    )
    assert selected.filter(pl.col("factor") == "MKT")["loading"].to_list() == [
        None,
        1.0,
        None,
        2.0,
        None,
    ]
    assert selected.filter(pl.col("factor") == "SIZE")["loading"].null_count() == 5
    sectors = prepare_classification_inputs(
        {17: "technology"},
        requests=requests,
        feed_spec=feed,
        purpose="causal",
        static=True,
        static_known_at=datetime(2025, 7, 1, tzinfo=UTC),
    )
    assert sectors["sector_static"].all()
    print(
        json.dumps(
            {
                "liquidity": prepared["lagged_mean_notional_volume"].to_list(),
                "loading_rows": selected.height,
                "static_sector_rows": sectors.height,
            }
        )
    )


if __name__ == "__main__":
    main()
