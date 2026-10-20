"""Native keyed realized outcomes, without analytical cleaning or price filling."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec, TimestampSemantics, WindowUnit

from ml4t.engineer.core.bar_grid import (
    _expected_bar_label_ticks,
    scheduled_session_indices,
    validate_declared_grid,
)
from ml4t.engineer.features.utils.arithmetic import _price_return
from ml4t.engineer.labeling.utils import _get_future_price_lookup


def _fixed_duration(value: str | timedelta, *, zero_allowed: bool = False) -> timedelta:
    import pandas as pd

    try:
        if isinstance(value, str):
            parts = list(re.finditer(r"(\d+)(ns|us|ms|w|d|h|m|s)", value))
            if not parts or "".join(part[0] for part in parts) != value:
                raise ValueError
            scales = {
                "w": 604800 * 10**9,
                "d": 86400 * 10**9,
                "h": 3600 * 10**9,
                "m": 60 * 10**9,
                "s": 10**9,
                "ms": 10**6,
                "us": 1000,
                "ns": 1,
            }
            duration = pd.Timedelta(
                sum(int(part[1]) * scales[part[2]] for part in parts), unit="ns"
            )
        else:
            duration = pd.Timedelta(value)
        if not isinstance(duration, timedelta) or duration < timedelta(0):
            raise ValueError
        if not zero_allowed and duration == timedelta(0):
            raise ValueError
        # Validate the ns range before building a Polars expression.
        _ = duration.value
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Elapsed horizons/tolerance require finite fixed durations") from exc
    return duration


def _utc_instants(frame: pl.DataFrame, name: str, timezone: str | None) -> pl.DataFrame:
    dtype = frame.schema[name]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"{name!r} must contain datetime instants")
    value = pl.col(name).cast(pl.Datetime("ns", dtype.time_zone))
    if frame.select(value.cast(dtype).ne_missing(pl.col(name)).any()).item():
        raise ValueError(f"{name!r} is outside the supported nanosecond timestamp range")
    if dtype.time_zone is None:
        if timezone is None:
            raise ValueError(f"Naive {name!r} requires explicit FeedSpec.timezone")
        value = value.dt.replace_time_zone(timezone, ambiguous="raise", non_existent="raise")
    return frame.with_columns(value.dt.convert_time_zone("UTC"))


def _session_requests(
    frame: pl.DataFrame, count: int, feed: FeedSpec, calendar: str, timestamp_col: str
) -> pl.Series:
    import pandas as pd

    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    dtype = frame.schema[timestamp_col]
    if frame.is_empty():
        return pl.Series("requested_exit_time", [], dtype=dtype)
    ids = scheduled_session_indices(
        frame, feed_spec=feed, calendar=calendar, timestamp_col=timestamp_col
    )
    values = frame[timestamp_col]
    days = values.dt.date().to_list()
    adapter = PandasMarketCalendar(calendar)
    schedule = adapter._calendar.schedule(
        start_date=min(days) - timedelta(days=7), end_date=max(days) + timedelta(days=7)
    )
    endpoints = []
    for index, row in schedule.iterrows():
        assert isinstance(index, datetime)
        if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
            endpoints.append(index.date())
        else:
            assert feed.bar_label in {"left", "right"}
            closed = row["market_close"].value
            frequency = pd.Timedelta("1D" if feed.data_frequency == "1d" else feed.data_frequency)
            assert isinstance(frequency, timedelta)
            step = pd.Timedelta(frequency).value
            labels = list(
                _expected_bar_label_ticks(
                    adapter,
                    closed - 2 * step,
                    closed,
                    frequency,
                    feed.bar_label,
                    feed.partial_bar_policy == "include",
                )
            )
            endpoints.append(labels[-1] if labels else None)
    requested = [
        endpoints[int(index) + count] if int(index) + count < len(endpoints) else None
        for index in ids
    ]
    if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
        return pl.Series("requested_exit_time", requested, dtype=dtype)
    return pl.Series("requested_exit_time", requested, dtype=pl.Int64).cast(dtype)


def forward_returns(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    horizons: Mapping[str, int | str | timedelta],
    price_basis: str,
    entries: pl.DataFrame | None = None,
    feed_spec: FeedSpec | None = None,
    timestamp_col: str | None = None,
    entity_col: str | list[str] | None = None,
    price_col: str | None = None,
    calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    window_unit: WindowUnit | str = WindowUnit.OBSERVATION,
    endpoint_method: Literal["cumulative", "exit_bar"] = "cumulative",
    return_method: Literal["returns", "log_returns"] = "returns",
    index_mode: Literal["per_asset", "common_grid"] = "per_asset",
    tolerance: str | timedelta | None = None,
    missing_price_policy: Literal["raise", "null"] = "raise",
    tail_policy: Literal["null", "drop", "raise"] = "null",
    available_at_col: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Produce one long outcome row per exact entry key and named horizon.

    Count horizons use chronological observations or validated expected bars.
    Session counts select the final scheduled bar of the Nth following session.
    Elapsed horizons select the first timestamp at/after the exact physical target.
    ``common_grid`` uses the supplied union clock, including absent asset quotes;
    ``per_asset`` uses each asset's own observations. Neither mode fills prices.

    A LazyFrame remains lazy. Instants normalize to UTC nanoseconds; declared daily
    session labels remain dates. Inputs are never modified. Missing availability
    stays unknown, rather than being fabricated from an event/session timestamp.
    Future labels must not be used as contemporaneously available features.
    """
    if not isinstance(horizons, Mapping) or not horizons:
        raise ValueError("horizons must be a nonempty mapping of names to values")
    if any(not isinstance(name, str) or not name for name in horizons):
        raise ValueError("horizon names must be nonempty strings")
    unit = WindowUnit(window_unit)
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("price_basis must explicitly declare the supplied price basis")
    for name, value, allowed in (
        ("endpoint_method", endpoint_method, {"cumulative", "exit_bar"}),
        ("return_method", return_method, {"returns", "log_returns"}),
        ("index_mode", index_mode, {"per_asset", "common_grid"}),
        ("missing_price_policy", missing_price_policy, {"raise", "null"}),
        ("tail_policy", tail_policy, {"null", "drop", "raise"}),
    ):
        if value not in allowed:
            raise ValueError(f"Invalid {name}: {value!r}")
    parsed = {}
    for name, value in horizons.items():
        if unit == WindowUnit.ELAPSED:
            if not isinstance(value, str | timedelta):
                raise ValueError("Elapsed horizons require fixed durations, not row counts")
            parsed[name] = _fixed_duration(value)
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Count horizons must be positive integers")
            parsed[name] = value
    if tolerance is not None and unit != WindowUnit.ELAPSED:
        raise ValueError("tolerance applies only to elapsed horizons")
    parsed_tolerance = (
        _fixed_duration(tolerance, zero_allowed=True) if tolerance is not None else None
    )
    feed = feed_spec or FeedSpec(time_contract_version=2)
    ts = timestamp_col or feed.timestamp_col
    entity = entity_col if entity_col is not None else feed.entity_col
    entities = [entity] if isinstance(entity, str) else list(entity or ["asset"])
    price = price_col or feed.price_col
    if feed_spec is not None and (
        ts != feed.timestamp_col
        or (
            feed.entity_col is not None
            and entities
            != ([feed.entity_col] if isinstance(feed.entity_col, str) else list(feed.entity_col))
        )
        or price != feed.price_col
    ):
        raise ValueError("Explicit columns conflict with feed_spec declarations")
    selected_calendar = feed.resolve_calendar(calendar=calendar)
    if unit in {WindowUnit.EXPECTED_BAR, WindowUnit.SESSION} and selected_calendar is None:
        raise ValueError("Expected-bar/session windows require a named calendar and declared grid")
    if unit == WindowUnit.ELAPSED and feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
        raise ValueError("Session labels cannot define elapsed horizons")
    keys = [*entities, ts]
    output_names = {
        "horizon",
        "horizon_value",
        "window_unit",
        "endpoint_method",
        "return_method",
        "index_mode",
        "price_basis",
        "calendar",
        "calendar_declared",
        "requested_exit_time",
        "exit_time",
        "reference_time",
        "outcome_available_at",
        "forward_return",
        "outcome_status",
    }
    if (
        len(set(keys)) != len(keys)
        or set(keys) & output_names
        or any(k.startswith("_out_") for k in keys)
    ):
        raise ValueError("Entry key names must be distinct from outcome metadata columns")
    if isinstance(data, pl.LazyFrame):
        kwargs: dict[str, Any] = {
            "horizons": horizons,
            "price_basis": price_basis,
            "entries": entries,
            "feed_spec": feed_spec,
            "timestamp_col": timestamp_col,
            "entity_col": entity_col,
            "price_col": price_col,
            "calendar": calendar,
            "window_unit": window_unit,
            "endpoint_method": endpoint_method,
            "return_method": return_method,
            "index_mode": index_mode,
            "tolerance": tolerance,
            "missing_price_policy": missing_price_policy,
            "tail_policy": tail_policy,
            "available_at_col": available_at_col,
        }
        kwargs["entries"] = entries.head(0) if entries is not None else None
        template = forward_returns(pl.DataFrame(schema=data.collect_schema()), **kwargs)
        kwargs["entries"] = entries
        assert isinstance(template, pl.DataFrame)

        def apply(frame: pl.DataFrame) -> pl.DataFrame:
            result = forward_returns(frame, **kwargs)
            assert isinstance(result, pl.DataFrame)
            return result

        return data.map_batches(
            apply,
            schema=template.schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    required = [*keys, price, *([available_at_col] if available_at_col is not None else [])]
    absent = set(required) - set(data.columns)
    if absent:
        raise ValueError(f"Missing outcome input columns: {sorted(absent)}")
    frame = data.select(list(dict.fromkeys(required)))
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError("Outcome entry keys cannot be null")
    if frame.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Outcome price keys must be unique")
    if not frame[price].dtype.is_numeric():
        raise ValueError("Outcome prices must be numeric")
    invalid = ~pl.col(price).cast(pl.Float64).is_finite().fill_null(False) | (pl.col(price) <= 0)
    if missing_price_policy == "raise" and frame.select(invalid.any()).item():
        raise ValueError("Outcome price contains null, nonfinite or nonpositive values")
    frame = frame.with_columns(pl.when(invalid).then(None).otherwise(pl.col(price)).alias(price))
    labels = feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL
    if labels:
        if frame.schema[ts] != pl.Date:
            raise ValueError("Declared session labels require Date identifiers")
    else:
        frame = _utc_instants(frame, ts, feed.timezone)
    if available_at_col is not None:
        frame = _utc_instants(frame, available_at_col, feed.timezone)
    requested_entries = entries if entries is not None else frame.select(keys)
    if entries is not None and not labels:
        requested_entries = _utc_instants(entries, ts, feed.timezone)
    requested_entries = requested_entries.select(keys).with_row_index("_out_order")
    if requested_entries.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError("Explicit entry keys must be unique")
    if requested_entries.join(frame.select(keys), on=keys, how="anti").height:
        raise ValueError("Every explicit entry key must match an exact supplied price key")
    if selected_calendar is not None:
        validate_declared_grid(
            frame, feed_spec=feed, calendar=selected_calendar, timestamp_col=ts, group_cols=entities
        )
    availability = (
        pl.col(available_at_col)
        if available_at_col is not None
        else pl.lit(None, dtype=pl.Datetime("ns", "UTC"))
    )
    frame = frame.with_columns(availability.alias("_out_available"))
    clock = frame.select(ts).unique().sort(ts)
    groups = frame.partition_by(entities, maintain_order=True) or [frame]
    results = []
    for group in groups:
        observed = group.sort(ts)
        working = observed.select(
            ts, pl.col(price).alias("_out_price"), "_out_available"
        ).with_columns(pl.lit(True).alias("_out_present"))
        if index_mode == "common_grid":
            working = clock.join(working, on=ts, how="left", validate="1:1").sort(ts)
        for horizon_index, (name, horizon) in enumerate(parsed.items()):
            if isinstance(horizon, int) and unit != WindowUnit.SESSION:
                # A beyond-coverage Python count must not overflow a native shift argument.
                shift = min(horizon, working.height + 1)
                reference_shift = 0 if endpoint_method == "cumulative" else shift - 1
                matched = working.with_columns(
                    pl.col(ts).shift(-shift).alias("requested_exit_time"),
                    pl.col(ts).shift(-shift).alias("exit_time"),
                    pl.col("_out_price").shift(-shift).alias("_out_exit_price"),
                    pl.col("_out_present").shift(-shift).alias("_out_exit_present"),
                    pl.col("_out_available").shift(-shift).alias("_out_exit_available"),
                    pl.col("_out_price").shift(-reference_shift).alias("_out_reference_price"),
                    pl.col("_out_available")
                    .shift(-reference_shift)
                    .alias("_out_reference_available"),
                    pl.col(ts).shift(-reference_shift).alias("reference_time"),
                )
                within = pl.col("requested_exit_time").is_not_null()
            else:
                lookup = working.with_columns(
                    pl.col("_out_price").shift(1).alias("_out_reference_price"),
                    pl.col("_out_available").shift(1).alias("_out_reference_available"),
                    pl.col(ts).shift(1).alias("reference_time"),
                )
                reference = lookup.with_row_index("_future_row_index").select(
                    "_future_row_index",
                    "_out_reference_price",
                    "_out_reference_available",
                    "reference_time",
                    pl.col("_out_available").alias("_out_exit_available"),
                    pl.col("_out_present").alias("_out_exit_present"),
                )
                if unit == WindowUnit.SESSION:
                    assert isinstance(horizon, int) and selected_calendar is not None
                    targets = working.with_columns(
                        _session_requests(working, horizon, feed, selected_calendar, ts)
                    )
                    matched = targets.join(
                        lookup.with_row_index("_future_row_index").select(
                            pl.col(ts).alias("exit_time"),
                            pl.col("_out_price").alias("_out_exit_price"),
                            "_future_row_index",
                        ),
                        left_on="requested_exit_time",
                        right_on="exit_time",
                        how="left",
                        coalesce=False,
                    )
                else:
                    assert isinstance(horizon, timedelta)
                    matched = _get_future_price_lookup(
                        working, horizon, "_out_price", ts, parsed_tolerance, None
                    ).rename(
                        {
                            "_target_ts": "requested_exit_time",
                            "_lookup_ts": "exit_time",
                            "_future_price": "_out_exit_price",
                        }
                    )
                matched = matched.join(reference, on="_future_row_index", how="left")
                if endpoint_method == "cumulative":
                    matched = matched.with_columns(
                        pl.col("_out_price").alias("_out_reference_price"),
                        pl.col("_out_available").alias("_out_reference_available"),
                        pl.col(ts).alias("reference_time"),
                    )
                within = pl.col("requested_exit_time").is_not_null() & (
                    pl.col("requested_exit_time") <= pl.col(ts).max()
                )
            status = (
                pl.when(~within.fill_null(False))
                .then(pl.lit("insufficient_tail"))
                .when(~pl.col("_out_exit_present").fill_null(False))
                .then(pl.lit("missing_endpoint"))
                .when(
                    pl.col("_out_exit_price").is_null() | pl.col("_out_reference_price").is_null()
                )
                .then(pl.lit("missing_price"))
                .otherwise(pl.lit("ok"))
            )
            matched = matched.with_columns(status.alias("outcome_status"))
            numeric = _price_return(
                "_out_exit_price",
                1,
                logarithmic=return_method == "log_returns",
                reference=pl.col("_out_reference_price"),
            )
            known = (
                pl.col("_out_exit_available").is_not_null()
                & pl.col("_out_reference_available").is_not_null()
            )
            result = matched.select(
                ts,
                "requested_exit_time",
                "reference_time",
                pl.when(pl.col("_out_exit_present").fill_null(False))
                .then(pl.col("exit_time"))
                .otherwise(None)
                .alias("exit_time"),
                pl.when((pl.col("outcome_status") == "ok") & known)
                .then(pl.max_horizontal("_out_exit_available", "_out_reference_available"))
                .otherwise(None)
                .alias("outcome_available_at"),
                pl.when(pl.col("outcome_status") == "ok")
                .then(numeric)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("forward_return"),
                "outcome_status",
                pl.lit(name).alias("horizon"),
                pl.lit(str(horizons[name])).alias("horizon_value"),
                pl.lit(unit.value).alias("window_unit"),
                pl.lit(endpoint_method).alias("endpoint_method"),
                pl.lit(return_method).alias("return_method"),
                pl.lit(index_mode).alias("index_mode"),
                pl.lit(price_basis).alias("price_basis"),
                pl.lit(selected_calendar, dtype=pl.String).alias("calendar"),
                pl.lit(calendar is not UNSET_CALENDAR or "calendar" in feed.to_dict()).alias(
                    "calendar_declared"
                ),
                pl.lit(horizon_index, dtype=pl.UInt32).alias("_out_horizon_order"),
            ).join(observed.select(keys), on=ts, how="inner", validate="1:1")
            results.append(result)
    output = (
        requested_entries.join(pl.concat(results), on=keys, how="left")
        .sort("_out_order", "_out_horizon_order")
        .drop("_out_order", "_out_horizon_order")
    )
    if (
        tail_policy == "raise"
        and output.filter(pl.col("outcome_status") == "insufficient_tail").height
    ):
        raise ValueError("Insufficient price tail for requested outcome horizon")
    return (
        output.filter(pl.col("outcome_status") != "insufficient_tail")
        if tail_policy == "drop"
        else output
    )
