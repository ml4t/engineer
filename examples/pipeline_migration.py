"""Port finite custom factors, classifications and screens without a legacy graph runtime."""

import json
from datetime import UTC, date, datetime

import polars as pl
from ml4t.specs import FeedSpec

from ml4t.engineer.compat.zipline import factor_expression
from ml4t.engineer.features.preparation import prepare_feature_panel
from ml4t.engineer.features.selection import cross_section


def momentum_screen(
    prices: pl.DataFrame | pl.LazyFrame,
    decisions: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    feed_spec: FeedSpec,
    price_basis: str,
    window_length: int = 15,
    n: int = 3,
) -> pl.DataFrame | pl.LazyFrame:
    """Return supplied-member RSI selections plus a finite custom mean/classifier."""
    rsi = factor_expression("RSI", inputs=[feed_spec.price_col], window_length=window_length)
    assert isinstance(rsi, pl.Expr)
    panel = prepare_feature_panel(
        prices,
        features={
            "rsi": rsi,
            "custom_mean": pl.col(feed_spec.price_col).rolling_mean(3, min_samples=1),
        },
        lookbacks={"rsi": window_length, "custom_mean": 3},
        input_columns=[feed_spec.price_col],
        decisions=decisions,
        universe=universe,
        price_basis=price_basis,
        feed_spec=feed_spec,
        missing_input_policy="known_missing",
    )
    return panel.with_columns(
        cross_section(
            "rsi", method="top", n=n, timestamp_col="decision_time", mask="eligible"
        ).alias("longs"),
        cross_section(
            "rsi", method="bottom", n=n, timestamp_col="decision_time", mask="eligible"
        ).alias("shorts"),
        pl.col("sector").replace_strict({"tech": 1, "bank": 2}, default=None).alias("sector_code"),
        (pl.col("custom_mean") > 0).fill_null(False).alias("custom_filter"),
    ).filter(pl.col("member").fill_null(False))


def sample() -> pl.DataFrame:
    days = [date(2025, 7, day) for day in (7, 8, 9)]
    prices = pl.DataFrame(
        {
            "timestamp": days * 3,
            "asset": [17] * 3 + [23] * 3 + [31] * 3,
            "close": [10.0, 11.0, 10.0, 10.0, 12.0, 13.0, 10.0, 9.0, 8.0],
            "available_at": [datetime(2025, 7, day.day, 20, tzinfo=UTC) for day in days] * 3,
        }
    )
    decisions = pl.DataFrame(
        {
            "timestamp": [days[-1]],
            "decision_time": [datetime(2025, 7, 10, 13, tzinfo=UTC)],
            "effective_session": [date(2025, 7, 10)],
        }
    )
    universe = pl.DataFrame(
        {
            "asset": [17, 23, 31],
            "effective_from": [date(2025, 7, 10)] * 3,
            "effective_to": pl.Series([None] * 3, dtype=pl.Date),
            "available_at": [datetime(2025, 7, 1, tzinfo=UTC)] * 3,
            "member": [True] * 3,
            "sector": ["tech", "bank", None],
        }
    )
    result = momentum_screen(
        prices.lazy(),
        decisions,
        universe,
        feed_spec=FeedSpec(time_contract_version=2, calendar=None, entity_col="asset"),
        price_basis="raw",
        window_length=3,
        n=1,
    )
    assert isinstance(result, pl.LazyFrame)
    return result.collect()


if __name__ == "__main__":
    result = sample()
    assert result["rsi"].to_list() == [50.0, 100.0, 0.0]
    assert result.filter("longs")["asset"].to_list() == [23]
    assert result.filter("shorts")["asset"].to_list() == [31]
    assert result["sector_code"].to_list() == [1, 2, None]
    print(json.dumps(result.to_dicts(), ensure_ascii=True, default=str))
