"""Predictive weekday counts from supplied event dates and publication instants."""

from typing import Literal

import polars as pl


def _date(column: str) -> pl.Expr:
    def check(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if dtype != pl.Date:
            raise ValueError(
                f"{column!r} must be a civil/session Date, not an inferred instant date"
            )
        return value

    return pl.col(column).pipe_with_dtype(check)


def _instant_ticks(column: str) -> pl.Expr:
    def check(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise ValueError(
                f"{column!r} must declare a timezone-aware publication/decision instant"
            )
        scale = {"ms": 1_000_000, "us": 1_000, "ns": 1}[dtype.time_unit]
        # Compare physical instants without truncating ns or overflowing a
        # Datetime(ns) conversion for otherwise valid wider-range us/ms dates.
        return value.cast(pl.Int64).cast(pl.Int128) * scale

    return pl.col(column).pipe_with_dtype(check)


def event_business_days(
    event_date_col: str,
    *,
    reference_date_col: str,
    known_at_col: str,
    decision_time_col: str,
    direction: Literal["since", "until"] = "since",
) -> pl.Expr:
    """Weekdays since/until a supplied event known by the decision instant.

    Event/reference columns are civil or session Date labels. Publication and
    decision columns are separate timezone-aware instants. Monday through Friday
    count, including exchange holidays; the end date is excluded. No exchange
    calendar, date provider, latest-event lookup or observation-spacing inference
    is used. Missing or not-yet-known event dates yield null. Signed negative
    counts remain possible if the caller supplies a date on the opposite side.
    """
    if direction not in {"since", "until"}:
        raise ValueError("direction must be 'since' or 'until'")
    event, reference = _date(event_date_col), _date(reference_date_col)
    start, end = (event, reference) if direction == "since" else (reference, event)
    eligible = _instant_ticks(known_at_col) <= _instant_ticks(decision_time_col)
    return pl.when(eligible).then(pl.business_day_count(start, end).cast(pl.Float64))
