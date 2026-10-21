"""Optional prepared market and dated inputs, without analysis or source selection."""

from typing import Any, Literal

import polars as pl
from ml4t.specs import UNSET_CALENDAR, FeedSpec, WindowUnit

from ml4t.engineer.core.bar_grid import validate_declared_grid
from ml4t.engineer.features.events import _instant_ticks
from ml4t.engineer.features.rolling import _period, finite_ewma


def _roles(feed_spec: FeedSpec) -> tuple[str, list[str]]:
    if not isinstance(feed_spec, FeedSpec):
        raise TypeError("feed_spec must be a canonical FeedSpec instance")
    identity = feed_spec.entity_col
    entities = [identity] if isinstance(identity, str) else list(identity or ["asset"])
    keys = [feed_spec.timestamp_col, *entities]
    if not entities or len(set(keys)) != len(keys) or any(not c for c in keys):
        raise ValueError("Declare distinct nonempty timestamp/entity columns")
    return feed_spec.timestamp_col, entities


def _clock(
    frame: pl.DataFrame, column: str, output: str, *, required: bool = False
) -> pl.DataFrame:
    expression = (
        pl.lit(None, pl.Int128) if frame.schema[column] == pl.Null else _instant_ticks(column)
    )
    result = frame.with_columns(expression.alias(output))
    if required and result[output].null_count():
        raise ValueError(f"{column} requires known aware instants")
    if result.filter((pl.col(output) < -(2**63)) | (pl.col(output) > 2**63 - 1)).height:
        raise ValueError(f"{column} exceeds Datetime(ns) output range")
    return result


def prepare_market_inputs(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    feed_spec: FeedSpec,
    price_basis: str,
    volume_basis: str,
    purpose: Literal["causal", "retrospective"],
    liquidity_window: int | None = None,
    completed_lag: int = 1,
    calendar: Any = UNSET_CALENDAR,
    available_at_col: str = "available_at",
    completed_at_col: str | None = None,
    decision_time_col: str = "decision_time",
    currency_col: str = "currency",
    multiplier_col: str = "multiplier",
    volume_unit_col: str = "volume_unit",
) -> pl.DataFrame | pl.LazyFrame:
    """Validate native market inputs and optionally append lagged notional volume.

    Data/callers supply compatible prepared price/share bases, quote currency,
    volume unit and actual monetary multiplier. No equity-unit default, FX,
    source revision, adjustment or holdings normalization is inferred. With no
    liquidity_window the original table is returned unchanged after validation.

    The optional feature is mean(price*volume*multiplier).shift(completed_lag),
    over complete original row windows. It is not current price times lagged
    volume. Missing volume remains uncovered; genuine zero remains zero.
    Causal preparation requires actual publication, completion and decision
    instants. completed_at_col names a supplied completion clock, not a guess
    from publication or observation spacing. Map it explicitly to publication
    only when the source contract establishes those records are completed bars.
    Raw columns remain untouched; derived causal values appear only when known.
    Retrospective preparation is named in the output and is not a signal claim.
    """
    timestamp, entities = _roles(feed_spec)
    if purpose not in {"causal", "retrospective"}:
        raise ValueError("purpose must be causal or retrospective")
    if any(not isinstance(b, str) or not b.strip() for b in (price_basis, volume_basis)):
        raise ValueError("Declare nonempty price_basis and volume_basis")
    _period(completed_lag)
    if liquidity_window is not None:
        _period(liquidity_window)
        if price_basis != volume_basis:
            raise ValueError("Prepare compatible price/volume basis upstream before liquidity")
        if purpose == "causal" and completed_at_col is None:
            raise ValueError("Causal liquidity requires a declared completed_at_col")
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    price, volume = feed_spec.price_col, feed_spec.volume_col
    keys = [timestamp, *entities]
    roles = [*keys, price, volume, currency_col, multiplier_col, volume_unit_col]
    if len(set(roles)) != len(roles):
        raise ValueError("Market value,unit and key roles must be distinct")
    required = [*keys, price, currency_col, multiplier_col, volume_unit_col]
    if liquidity_window is not None and purpose == "causal":
        required += [available_at_col, decision_time_col]
        if completed_at_col is not None:
            required.append(completed_at_col)
    if set(required) - set(schema):
        raise ValueError(
            f"Missing market declarations/columns: {sorted(set(required) - set(schema))}"
        )
    if schema[timestamp].base_type() not in {pl.Date, pl.Datetime}:
        raise ValueError(
            "Market observation keys must be Date labels or declared Datetime instants"
        )
    for column in (price, volume, multiplier_col):
        if column in schema and schema[column] != pl.Null and not schema[column].is_numeric():
            raise ValueError(f"{column} requires numeric values or explicit missing observations")
    for column in (currency_col, volume_unit_col):
        if schema[column].base_type() not in {pl.String, pl.Categorical, pl.Enum, pl.Null}:
            raise ValueError(f"{column} requires explicit string unit declarations")
    resolved = feed_spec.resolve_calendar(calendar=calendar)
    prefix = "__ml4t_inputs_"
    if any(c.startswith(prefix) for c in schema):
        raise ValueError("Input columns collide with reserved preparation names")
    added = {
        "notional_volume": pl.Float64,
        "lagged_mean_notional_volume": pl.Float64,
        "liquidity_available_at": pl.Datetime("ns", "UTC"),
        "liquidity_covered": pl.Boolean,
        "liquidity_window": pl.Int64,
        "liquidity_lag": pl.Int64,
        "liquidity_window_unit": pl.String,
        "liquidity_price_basis": pl.String,
        "liquidity_volume_basis": pl.String,
        "preparation_purpose": pl.String,
    }
    if liquidity_window is not None and set(added) & set(schema):
        raise ValueError("Already prepared liquidity columns; do not double lag prepared inputs")

    def calculate(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.select(pl.any_horizontal(pl.col(c).is_null() for c in keys).any()).item():
            raise ValueError("Market observation/entity keys cannot be unknown")
        if frame.select(pl.struct(keys).is_duplicated().any()).item():
            raise ValueError("Duplicate market keys; select source revisions upstream")
        for column in (price, volume, multiplier_col):
            if column not in frame.columns:
                continue
            numeric = pl.col(column).cast(pl.Float64)
            invalid = numeric.is_infinite()
            if column == volume:
                invalid |= numeric < 0
            if column == multiplier_col:
                invalid |= numeric <= 0
            if frame.filter(invalid).height:
                raise ValueError(
                    f"Invalid numeric {column}; volume is nonnegative,multiplier positive"
                )
        for column in (currency_col, volume_unit_col):
            if frame.filter(pl.col(column).cast(pl.String).str.strip_chars() == "").height:
                raise ValueError(f"{column} cannot be an empty unit declaration")
        if resolved is not None:
            validate_declared_grid(
                frame,
                feed_spec=feed_spec,
                calendar=resolved,
                timestamp_col=timestamp,
                group_cols=entities,
            )
        if liquidity_window is None:
            return frame
        order, product, mean, known, completed, decision, contributing = (
            prefix + name
            for name in (
                "order",
                "product",
                "mean",
                "known",
                "completed",
                "decision",
                "contributing",
            )
        )
        frame = frame.with_row_index(order).sort([*entities, timestamp])
        volume_expression = (
            pl.col(volume).cast(pl.Float64) if volume in schema else pl.lit(None, pl.Float64)
        )
        valid = pl.all_horizontal(pl.col(c).is_not_null() for c in (currency_col, volume_unit_col))
        product_expression = (
            pl.col(price).cast(pl.Float64)
            * volume_expression
            * pl.col(multiplier_col).cast(pl.Float64)
        )
        frame = frame.with_columns(
            pl.when(valid).then(product_expression).fill_nan(None).alias(product)
        )
        if frame.filter(pl.col(product).is_infinite()).height:
            raise ValueError(
                "Notional volume product is not representable; supply compatible units"
            )
        for source, target in ((available_at_col, known), (completed_at_col, completed)):
            frame = (
                _clock(frame, source, target)
                if source in frame.columns
                else frame.with_columns(pl.lit(None, pl.Int128).alias(target))
            )
        if purpose == "causal":
            frame = _clock(frame, decision_time_col, decision, required=True)
        else:
            frame = frame.with_columns(pl.lit(None, pl.Int128).alias(decision))
        available = pl.when(pl.col(known).is_not_null() & pl.col(completed).is_not_null()).then(
            pl.max_horizontal(known, completed).cast(pl.Int64)
        )
        full = (
            available.is_not_null().cast(pl.Int64).rolling_sum(liquidity_window) == liquidity_window
        )
        frame = frame.with_columns(
            finite_ewma(product, period=liquidity_window, decay_rate=1.0)
            .shift(completed_lag)
            .over(entities)
            .alias(mean),
            pl.when(full)
            .then(available.rolling_max(liquidity_window))
            .shift(completed_lag)
            .over(entities)
            .alias(contributing),
        )
        covered = pl.col(mean).is_finite().fill_null(False)
        current_known = (
            pl.col(known).is_not_null()
            & (pl.col(known) <= pl.col(decision))
            & (pl.col(completed) <= pl.col(decision))
        )
        if purpose == "causal":
            covered &= pl.col(contributing).is_not_null() & (
                pl.col(contributing) <= pl.col(decision)
            )
        frame = frame.with_columns(
            pl.when(current_known if purpose == "causal" else pl.lit(True))
            .then(pl.col(product))
            .alias("notional_volume"),
            pl.when(covered).then(pl.col(mean)).alias("lagged_mean_notional_volume"),
            pl.col(contributing).cast(pl.Datetime("ns", "UTC")).alias("liquidity_available_at"),
            covered.alias("liquidity_covered"),
            pl.lit(liquidity_window, pl.Int64).alias("liquidity_window"),
            pl.lit(completed_lag, pl.Int64).alias("liquidity_lag"),
            pl.lit(
                (WindowUnit.OBSERVATION if resolved is None else WindowUnit.EXPECTED_BAR).value
            ).alias("liquidity_window_unit"),
            pl.lit(price_basis).alias("liquidity_price_basis"),
            pl.lit(volume_basis).alias("liquidity_volume_basis"),
            pl.lit(purpose).alias("preparation_purpose"),
        )
        return frame.sort(order).select(*schema, *added)

    if isinstance(data, pl.LazyFrame):
        return data.map_batches(
            calculate,
            schema={**schema, **(added if liquidity_window is not None else {})},
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    return calculate(data)
