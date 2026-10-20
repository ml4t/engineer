"""Known event weekday counts must not infer exchange closures or future knowledge."""

import importlib
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest


def expression(direction: str = "since") -> pl.Expr:
    module = importlib.import_module("ml4t.engineer.features.events")
    return module.event_business_days(
        "event",
        reference_date_col="session",
        known_at_col="known",
        decision_time_col="decision",
        direction=direction,
    ).alias("days")


@pytest.mark.parametrize("direction", ["since", "until"])
@pytest.mark.parametrize("lazy", [False, True])
def test_event_counts_use_weekdays_including_nyse_holidays(direction: str, lazy: bool) -> None:
    frame = pl.DataFrame(
        {
            "event": [date(2025, 7, 4), date(2025, 7, 5), None, date(2025, 7, 7)],
            "session": [date(2025, 7, 7)] * 4,
            "known": [datetime(2025, 7, 1, tzinfo=UTC)] * 4,
            "decision": [datetime(2025, 7, 7, 14, tzinfo=UTC)] * 4,
        }
    )
    output = (frame.lazy() if lazy else frame).select(expression(direction))
    if lazy:
        output = output.collect()
    events = np.array(["2025-07-04", "2025-07-05", "NaT", "2025-07-07"], dtype="datetime64[D]")
    valid = ~np.isnat(events)
    expected = np.full(4, np.nan)
    reference = np.array("2025-07-07", dtype="datetime64[D]")
    expected[valid] = (
        np.busday_count(events[valid], reference)
        if direction == "since"
        else np.busday_count(reference, events[valid])
    )
    np.testing.assert_allclose(output["days"].to_numpy(), expected, equal_nan=True)
    assert abs(output["days"][0]) == 1


def test_event_dates_are_ineligible_before_announcement_at_exact_ns_cutoff() -> None:
    instant = 1751896800000000000
    frame = pl.DataFrame(
        {
            "event": [date(2025, 7, 10)] * 4,
            "session": [date(2025, 7, 7)] * 4,
            "known": pl.Series([instant - 1, instant, instant + 1, None], dtype=pl.Int64).cast(
                pl.Datetime("ns", "UTC")
            ),
            "decision": pl.Series([instant] * 4).cast(pl.Datetime("ns", "UTC")),
        }
    )
    actual = frame.select(expression("until"))["days"].to_list()
    assert actual == [3.0, 3.0, None, None]


def test_event_knowledge_comparison_preserves_timezone_and_out_of_ns_instants() -> None:
    frame = pl.DataFrame(
        {
            "event": [date(2500, 1, 2)],
            "session": [date(2500, 1, 5)],
            "known": [datetime(2499, 12, 1, tzinfo=UTC)],
            "decision": [datetime(2500, 1, 5, tzinfo=UTC)],
        }
    )
    original = frame.select(expression())
    changed = frame.with_columns(pl.col("known").dt.convert_time_zone("America/New_York")).select(
        expression()
    )
    assert original.equals(changed)
    assert original["days"][0] == np.busday_count("2500-01-02", "2500-01-05")


@pytest.mark.parametrize(
    "invalid", ["reference_datetime", "naive_knowledge", "naive_decision", "direction"]
)
def test_event_contract_rejects_ambiguous_time_roles(invalid: str) -> None:
    frame = pl.DataFrame(
        {
            "event": [date(2025, 7, 4)],
            "session": [date(2025, 7, 7)],
            "known": [datetime(2025, 7, 1, tzinfo=UTC)],
            "decision": [datetime(2025, 7, 7, tzinfo=UTC)],
        }
    )
    if invalid == "reference_datetime":
        frame = frame.with_columns(pl.col("session").cast(pl.Datetime("us", "UTC")))
    elif invalid == "naive_knowledge":
        frame = frame.with_columns(pl.col("known").dt.replace_time_zone(None))
    elif invalid == "naive_decision":
        frame = frame.with_columns(pl.col("decision").dt.replace_time_zone(None))
    with pytest.raises(ValueError):
        frame.select(expression("invalid" if invalid == "direction" else "since"))
