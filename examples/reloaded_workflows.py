"""Native migration recipes, without importing a legacy graph or analytics package."""

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from ml4t.engineer.features.inputs import market_to_long, prepare_market_inputs
from ml4t.engineer.features.preparation import prepare_feature_panel
from ml4t.engineer.features.rolling import annualized_volatility, rsi_cutler
from ml4t.engineer.features.utils.arithmetic import returns


def precise_wide_returns(wide: pd.DataFrame) -> pl.DataFrame:
    """Keep the original leading row, unlike the legacy dropped-first-row presentation."""
    feed = FeedSpec(time_contract_version=2, calendar=None, entity_col="asset")
    legacy = pd.concat({"price": wide}, names=["field"]).swaplevel()
    long = market_to_long(legacy, feed_spec=feed)
    assert isinstance(long, pl.DataFrame)
    return (
        long.sort("asset", "timestamp")
        .with_columns(returns("close").over("asset").alias("return"))
        .select("timestamp", "asset", "close", "return")
    )


def snapshot_screen(
    prices: pl.DataFrame,
    decisions: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    feed_spec: FeedSpec,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
) -> pl.DataFrame:
    """Combine independently prepared price features and optional monetary liquidity."""
    identity = feed_spec.entity_col
    entities = [identity] if isinstance(identity, str) else list(identity or ["asset"])
    keys = [feed_spec.timestamp_col, *entities]
    market = prepare_market_inputs(
        prices.lazy(),
        feed_spec=feed_spec,
        calendar=calendar,
        price_basis="split_adjusted",
        volume_basis="split_adjusted",
        purpose="causal",
        liquidity_window=2,
        completed_at_col="available_at",
    ).collect()
    features = prepare_feature_panel(
        prices.lazy(),
        features={
            "momentum": returns(feed_spec.price_col, periods=2),
            "volatility": annualized_volatility(returns(feed_spec.price_col), period=2),
        },
        lookbacks={"momentum": 3, "volatility": 3},
        input_columns=[feed_spec.price_col],
        decisions=decisions,
        universe=universe,
        feed_spec=feed_spec,
        calendar=calendar,
        price_basis="split_adjusted",
    ).collect()
    # Separate preparation keeps missing optional volume out of the price-only input set.
    result = features.join(
        market.select(
            *keys,
            "lagged_mean_notional_volume",
            "liquidity_covered",
            "liquidity_available_at",
        ),
        on=keys,
        how="left",
        validate="1:1",
    )
    return result.with_columns(
        (
            pl.col("eligible")
            & pl.col("liquidity_covered").fill_null(False)
            & (pl.col("momentum") > 0)
            & (pl.col("volatility") < 0.5)
            & (pl.col("lagged_mean_notional_volume") > 100000)
            & (pl.col("sector") == "bank")
        )
        .fill_null(False)
        .alias("screen")
    )


def custom_bands(
    prices: pl.DataFrame,
    decisions: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    feed_spec: FeedSpec,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
) -> pl.DataFrame:
    """Port a multi-output CustomFactor and retain its own required-input eligibility."""
    value = pl.col(feed_spec.price_col)
    return prepare_feature_panel(
        prices.lazy(),
        features={
            "bands": pl.struct(
                value.rolling_mean(3).alias("mid"),
                value.rolling_std(3, ddof=0).alias("width"),
            )
        },
        lookbacks={"bands": 3},
        input_columns=[feed_spec.price_col],
        decisions=decisions,
        universe=universe,
        feed_spec=feed_spec,
        calendar=calendar,
        price_basis="split_adjusted",
    ).collect()


def sampled_values(frame: pl.DataFrame, sampling_sessions: Sequence[date]) -> pl.DataFrame:
    """Carry explicit sampled decision rows, including missing samples, not price observations.

    sampling_sessions comes from the declared full schedule,including a preceding
    sample when output starts mid-period. Do not derive it from observed gaps.
    Keep pre-sample history for feature warmup before selecting these rows.
    """
    keys = ["effective_session", "asset"]
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Sampled feature rows require unique decision-session/entity keys")
    sampled = frame.filter(pl.col("effective_session").is_in(sampling_sessions)).select(
        *keys,
        pl.col("effective_session").alias("sample_session"),
        pl.col("available_at").alias("sample_available_at"),
        pl.when(pl.col("eligible")).then(pl.col("rsi")).alias("sampled_rsi"),
    )
    result = (
        frame.with_row_index("_recipe_order")
        .sort("asset", "effective_session")
        .join_asof(
            sampled.sort("asset", "effective_session"),
            by="asset",
            on="effective_session",
            strategy="backward",
            check_sortedness=False,
        )
    )
    if result.filter(pl.col("sample_available_at") > pl.col("decision_time")).height:
        raise ValueError("A sampled feature was not known at the receiving decision")
    return result.sort("_recipe_order").drop("_recipe_order")


def captured_snapshot() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, FeedSpec]:
    """Retain a small Data-produced split/rename capture and supplied universe facts.

    Market values match Data's dated_history_migration.py at d48d849f77d3ff75d05383cbaaed9b42f6ca95b7,
    explicitly requesting split_adjusted rather than its total-return default.
    The as-of is July7 at21UTC. These later-adjusted historical values are not
    claimed to have been tradable on earlier observation dates.
    """
    days = [date(2025, 7, d) for d in (1, 2, 3, 7)]
    cutoff = datetime(2025, 7, 7, 21, tzinfo=UTC)
    prices = pl.DataFrame(
        {
            "timestamp": days,
            "asset": [17] * 4,
            "ticker": ["OLD", "OLD", "OLD", "NEW"],
            "adj_close": [99.0, 100.0, 98.0, 101.0],
            "adj_volume": [2000.0, 2000.0, 7.0, 9.0],
            "available_at": [
                datetime(2025, 7, d.day, 17 if d.day == 3 else 20, tzinfo=UTC) for d in days
            ],
            "decision_time": [cutoff] * 4,
            "currency": ["USD"] * 4,
            "multiplier": [1.0] * 4,
            "volume_unit": ["shares"] * 4,
        }
    )
    decisions = pl.DataFrame(
        {
            "timestamp": [days[-1]],
            "decision_time": [cutoff],
            "effective_session": [date(2025, 7, 8)],
        }
    )
    universe = pl.DataFrame(
        {
            "asset": [17],
            "effective_from": [date(2025, 7, 3)],
            "effective_to": pl.Series([None], dtype=pl.Date),
            "available_at": [datetime(2025, 7, 3, 12, tzinfo=UTC)],
            "member": [True],
            "sector": ["bank"],
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        entity_col="asset",
        price_col="adj_close",
        volume_col="adj_volume",
        timestamp_semantics="session_label",
        timezone="America/New_York",
        data_frequency="1d",
        bar_type="time",
    )
    return prices, decisions, universe, feed


def cadence_example() -> pl.DataFrame:
    """Prepare actual price features before carrying explicitly chosen decision samples."""
    days = [date(2025, 7, day) for day in (1, 2, 3, 7, 8, 9, 10, 11)]
    effective = [date(2025, 7, day) for day in (7, 8, 9, 14)]
    prices = pl.DataFrame(
        {
            "timestamp": days,
            "asset": [17] * 8,
            "close": [100.0, 101.0, 92.0, 94.25, 89.0, 90.0, 91.0, None],
            "available_at": [
                datetime(2025, 7, day.day, 17 if day.day == 3 else 20, tzinfo=UTC) for day in days
            ],
        }
    )
    decisions = pl.DataFrame(
        {
            "timestamp": [days[i] for i in (2, 3, 4, 7)],
            "effective_session": effective,
            "decision_time": [datetime(2025, 7, day.day, 13, tzinfo=UTC) for day in effective],
        }
    )
    universe = pl.DataFrame(
        {
            "asset": [17],
            "effective_from": [days[0]],
            "effective_to": pl.Series([None], dtype=pl.Date),
            "available_at": [datetime(2025, 6, 1, tzinfo=UTC)],
            "member": [True],
        }
    )
    panel = prepare_feature_panel(
        prices.lazy(),
        features={"rsi": rsi_cutler("close", period=2)},
        lookbacks={"rsi": 3},
        input_columns=["close"],
        decisions=decisions,
        universe=universe,
        price_basis="raw",
        feed_spec=FeedSpec(time_contract_version=2, calendar=None, entity_col="asset"),
    ).collect()
    # These are supplied NYSE week-start decisions in this captured July interval.
    return sampled_values(panel, [date(2025, 7, 7), date(2025, 7, 14)])


def main() -> None:
    stamps = pd.date_range("2025-07-07", periods=3, tz="UTC")
    wide = pd.DataFrame({17: pd.array([2**63, 2**63 + 1, 2**63 + 2], dtype="UInt64")}, index=stamps)
    precise = precise_wide_returns(wide)
    assert precise["return"][0] is None
    assert precise["return"][1] == 1 / 2**63
    prices, decisions, universe, feed = captured_snapshot()
    # Core-only demonstration deliberately counts rows. The named NYSE profile
    # is checked separately with the real schedule dependency,not approximated.
    screen = snapshot_screen(prices, decisions, universe, feed_spec=feed, calendar=None)
    assert screen["screen"].to_list() == [True]
    assert screen["lagged_mean_notional_volume"].to_list() == [100343.0]
    bands = custom_bands(prices, decisions, universe, feed_spec=feed, calendar=None)
    assert bands["eligible"].to_list() == [True]
    cadence = cadence_example()
    np.testing.assert_allclose(
        cadence["sampled_rsi"].to_numpy(),
        [10.0, 10.0, 10.0, np.nan],
        rtol=1e-12,
        atol=0,
        equal_nan=True,
    )
    print(
        json.dumps(
            {
                "precise_returns": precise["return"].to_list(),
                "screened_assets": screen.filter("screen")["asset"].to_list(),
                "custom_mid": bands["bands"].struct.field("mid").to_list(),
                "sampled_rsi": cadence["sampled_rsi"].to_list(),
            }
        )
    )


if __name__ == "__main__":
    main()
