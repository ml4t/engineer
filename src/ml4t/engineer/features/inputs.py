"""Optional prepared market and dated inputs, without analysis or source selection."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

import numpy as np
import polars as pl
from ml4t.specs import UNSET_CALENDAR, FeedSpec, TimestampSemantics, WindowUnit

from ml4t.engineer.core.bar_grid import validate_declared_grid
from ml4t.engineer.features.events import _instant_ticks
from ml4t.engineer.features.preparation import _usable_value
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


def _native_table(
    data: pl.DataFrame | pl.LazyFrame, required: Sequence[str], keys: Sequence[str]
) -> pl.DataFrame | pl.LazyFrame:
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    if set(required) - set(schema):
        raise ValueError(f"Missing input columns: {sorted(set(required) - set(schema))}")

    def validate(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.select(pl.any_horizontal(pl.col(c).is_null() for c in keys).any()).item():
            raise ValueError("Input identity keys cannot be unknown")
        if frame.select(pl.struct(keys).is_duplicated().any()).item():
            raise ValueError("Duplicate input keys; select source revisions upstream")
        return frame

    if isinstance(data, pl.LazyFrame):
        return data.map_batches(
            validate,
            schema=schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    return validate(data)


def _legacy_table(data: Any) -> Any:
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Supply a native Polars table or a pandas MultiIndex DataFrame")
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
        raise ValueError("Legacy input requires a two-level date/field or date/asset index")
    if not isinstance(data.index.get_level_values(0), pd.DatetimeIndex):
        raise ValueError("Legacy first index level must contain declared Datetime keys")
    if data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError("Duplicate legacy input keys/columns; select revisions upstream")
    return data


def _pandas_values(values: Any, name: str) -> pl.Series:
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        index = pd.DatetimeIndex(values)
        unit = index.unit
        if unit == "s":
            index = index.as_unit("ms")
            unit = "ms"
        result = pl.Series(name, index.asi8).cast(
            pl.Datetime(unit, str(index.tz) if index.tz is not None else None)
        )
        return result.set(pl.Series(index.isna()), None)
    dtype = getattr(values.dtype, "numpy_dtype", values.dtype)
    declared = (
        pl.Series(np.empty(0, dtype=dtype)).dtype
        if pd.api.types.is_numeric_dtype(dtype)
        else pl.String
        if len(values) == 0
        else None
    )
    return pl.Series(name, [None if pd.isna(value) else value for value in values], dtype=declared)


def _identities(labels: Any, entities: Sequence[str], entity_map: Mapping[Any, Any] | None):
    import pandas as pd

    if entity_map is None and len(entities) == 1:
        values = _pandas_values(pd.Series(labels), entities[0])
        result = pl.DataFrame(values)
    else:
        if entity_map is None or any(key not in entity_map for key in labels):
            raise ValueError("Declare every legacy asset's stable identity in entity_map")
        values = [entity_map[key] for key in labels]
        rows = [(key,) for key in values] if len(entities) == 1 else values
        if any(not isinstance(key, tuple) or len(key) != len(entities) for key in rows):
            raise ValueError("Compound stable identities require matching tuple values")
        frame = pd.DataFrame(rows, columns=list(entities))
        result = pl.DataFrame({_name: _pandas_values(frame[_name], _name) for _name in entities})
    if result.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Stable entity identities cannot be unknown")
    if any(
        dtype.base_type() not in {pl.String, pl.Categorical, pl.Enum} and not dtype.is_integer()
        for dtype in result.schema.values()
    ):
        raise ValueError("Use stable integer/string IDs, not floats or opaque asset objects")
    return result


def market_to_long(
    data: Any,
    *,
    feed_spec: FeedSpec,
    price_field: str = "price",
    volume_field: str = "volume",
    entity_map: Mapping[Any, Any] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Convert legacy (date,field)-by-asset market cells without financial adjustments.

    Native tables retain their columns, dtypes, order and lazy behavior. Legacy
    conversion preserves supplied prices/volumes, zeros and missing cells. An
    opaque legacy asset needs an explicit stable entity_map; no stringification,
    completion/publication clock, units, currency or adjustment is invented.
    Numeric nullable arrays and timestamp keys retain their physical dtypes
    without requiring Arrow or converting precise integers through floats.
    """
    timestamp, entities = _roles(feed_spec)
    price, volume = feed_spec.price_col, feed_spec.volume_col
    keys = [timestamp, *entities]
    if len({*keys, price, volume}) != len(keys) + 2:
        raise ValueError("Market key/price/volume roles must be distinct")
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        return _native_table(data, [*keys, price], keys)
    import pandas as pd

    legacy = _legacy_table(data)
    if not price_field or not volume_field or price_field == volume_field:
        raise ValueError("Declare distinct nonempty legacy price/volume fields")
    if price_field not in legacy.index.get_level_values(1):
        raise ValueError("Legacy input has no declared price field")
    identities = _identities(legacy.columns, entities, entity_map)

    def field_frame(field: str, output: str) -> pl.DataFrame:
        wide = legacy.xs(field, level=1)
        pieces = []
        for i, column in enumerate(wide.columns):
            frame = pl.DataFrame(
                {
                    timestamp: _pandas_values(pd.Series(wide.index), timestamp),
                    output: _pandas_values(wide[column], output),
                }
            )
            pieces.append(
                frame.with_columns(
                    identities[c].slice(i, 1).repeat_by(frame.height).explode().alias(c)
                    for c in entities
                )
            )
        if not pieces:
            return pl.DataFrame(
                schema={
                    timestamp: _pandas_values(pd.Series(wide.index), timestamp).dtype,
                    **identities.schema,
                    output: pl.Float64,
                }
            )
        try:
            return pl.concat(pieces)
        except pl.exceptions.SchemaError as exc:
            raise ValueError(
                "Normalize differing asset value dtypes explicitly before conversion"
            ) from exc

    prices = field_frame(price_field, price)
    if volume_field in legacy.index.get_level_values(1):
        result = prices.join(field_frame(volume_field, volume), on=keys, how="full", coalesce=True)
    else:
        result = prices.with_columns(pl.lit(None, pl.Float64).alias(volume))
    return _native_table(result.select(*keys, price, volume), [*keys, price], keys)


def loadings_to_long(
    data: Any,
    *,
    feed_spec: FeedSpec,
    factor_col: str = "factor",
    value_col: str = "loading",
    entity_map: Mapping[Any, Any] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Convert supplied (date,asset)-by-factor loadings without filling or inferring history.

    This only changes containers. Causal preparation still requires actual
    publication times from Data/the caller and explicitly requested factors.
    """
    timestamp, entities = _roles(feed_spec)
    keys = [timestamp, *entities, factor_col]
    if len({*keys, value_col}) != len(keys) + 1 or not factor_col or not value_col:
        raise ValueError("Loading value/factor/key roles must be distinct and nonempty")
    if isinstance(data, pl.DataFrame | pl.LazyFrame):
        return _native_table(data, [*keys, value_col], keys)
    import pandas as pd

    legacy = _legacy_table(data)
    if any(not isinstance(column, str) or not column for column in legacy.columns):
        raise ValueError("Legacy factor columns must be distinct nonempty strings")
    identities = _identities(legacy.index.get_level_values(1), entities, entity_map)
    frame = identities.with_columns(
        _pandas_values(pd.Series(legacy.index.get_level_values(0)), timestamp)
    )
    pieces = [
        frame.with_columns(
            pl.lit(column).alias(factor_col),
            _pandas_values(legacy[column], value_col),
        )
        for column in legacy.columns
    ]
    try:
        result = (
            pl.concat(pieces)
            if pieces
            else frame.clear().with_columns(
                pl.lit(None, pl.String).alias(factor_col), pl.lit(None, pl.Float64).alias(value_col)
            )
        )
    except pl.exceptions.SchemaError as exc:
        raise ValueError("Normalize differing loading dtypes explicitly before conversion") from exc
    return _native_table(result.select(*keys, value_col), [*keys, value_col], keys)


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
        currency_count = prefix + "currency_count"
        if (
            frame.group_by(entities)
            .agg(pl.col(currency_col).drop_nulls().n_unique().alias(currency_count))
            .filter(pl.col(currency_count) > 1)
            .height
        ):
            raise ValueError(
                "Each stable entity needs one liquidity currency; normalize currency changes upstream"
            )
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


def _effective(frame: pl.DataFrame, column: str, target: str, feed: FeedSpec) -> pl.DataFrame:
    dtype = frame.schema[column]
    value = pl.col(column)
    if dtype == pl.Date:
        return frame.with_columns(value.cast(pl.Int128).alias(target))
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("Effective keys require Date labels or declared Datetime instants")
    if dtype.time_zone is None:
        if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
            scale = {"ns": 1, "us": 1000, "ms": 1_000_000}[dtype.time_unit]
            return frame.with_columns((value.cast(pl.Int128) * scale).alias(target))
        if feed.timezone is None:
            raise ValueError("Naive effective instants require a declared timezone")
        frame = frame.with_columns(value.dt.replace_time_zone(feed.timezone).alias(target))
        return frame.with_columns(_instant_ticks(target).alias(target))
    return frame.with_columns(_instant_ticks(column).alias(target))


def _dated_inputs(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    requests: pl.DataFrame | pl.LazyFrame,
    feed_spec: FeedSpec,
    purpose: str,
    value_col: str,
    output: str,
    effective_time_col: str | None,
    available_at_col: str,
    decision_time_col: str,
    factor_col: str | None = None,
    factors: Sequence[str] | None = None,
    intervals: bool = False,
    static: bool = False,
    effective_from_col: str = "effective_from",
    effective_to_col: str = "effective_to",
) -> pl.DataFrame | pl.LazyFrame:
    timestamp, entities = _roles(feed_spec)
    effective = effective_time_col or timestamp
    if purpose not in {"causal", "retrospective"}:
        raise ValueError("purpose must be causal or retrospective")
    if not output or not value_col:
        raise ValueError("Declare nonempty value/output column names")
    source_schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    request_schema = (
        requests.collect_schema() if isinstance(requests, pl.LazyFrame) else requests.schema
    )
    for label, schema, required in (
        ("requests", request_schema, [timestamp, *entities, effective, decision_time_col]),
        (
            "facts",
            source_schema,
            [
                *entities,
                value_col,
                available_at_col,
                *([factor_col] if factor_col else []),
                *(
                    []
                    if static
                    else [effective_from_col, effective_to_col]
                    if intervals
                    else [timestamp]
                ),
            ],
        ),
    ):
        if set(required) - set(schema):
            raise ValueError(f"Missing {label} columns: {sorted(set(required) - set(schema))}")
    if any(source_schema[c] != request_schema[c] for c in entities):
        raise ValueError("Fact/request stable entity dtypes must match without coercion")
    if factor_col:
        if (
            isinstance(factors, str)
            or not factors
            or any(not isinstance(f, str) or not f for f in factors)
            or len(set(factors)) != len(factors)
        ):
            raise ValueError("Declare distinct nonempty requested factors")
        if factor_col in request_schema or source_schema[factor_col].base_type() not in {
            pl.String,
            pl.Categorical,
            pl.Enum,
        }:
            raise ValueError(
                "Factor names must be declared string keys,not duplicate request columns"
            )
        if source_schema[value_col] != pl.Null and not source_schema[value_col].is_numeric():
            raise ValueError("Factor loadings must be numeric or explicitly missing")
    chosen_dtype = request_schema[effective]
    from_col = effective_from_col if intervals else timestamp
    from_dtype = chosen_dtype if static else source_schema[from_col]
    to_dtype = source_schema[effective_to_col] if intervals else from_dtype
    if not static and from_dtype.base_type() != chosen_dtype.base_type():
        raise ValueError("Fact/request effective keys must have the same declared clock kind")
    if intervals and to_dtype != pl.Null and to_dtype.base_type() != chosen_dtype.base_type():
        raise ValueError("Effective interval ends must match the declared clock kind")
    if (
        feed_spec.timestamp_semantics == TimestampSemantics.SESSION_LABEL
        and isinstance(chosen_dtype, pl.Datetime)
        and any(
            isinstance(dtype, pl.Datetime) and dtype.time_zone != chosen_dtype.time_zone
            for dtype in (from_dtype, to_dtype)
        )
    ):
        raise ValueError("Fact/request session-label timezone declarations must match")
    if to_dtype == pl.Null:
        to_dtype = from_dtype
    added = {
        output: source_schema[value_col],
        output + "_available_at": pl.Datetime("ns", "UTC"),
        output + "_covered": pl.Boolean,
        output + "_known_at_decision": pl.Boolean,
        output + "_effective_from": from_dtype,
        output + "_effective_to": to_dtype,
        output + "_purpose": pl.String,
        output + "_static": pl.Boolean,
    }
    if set(added) & set(request_schema):
        raise ValueError("Prepared output names collide with request columns")
    if factor_col in added:
        raise ValueError("Prepared output names cannot overwrite the declared factor identity")
    prefix = "__ml4t_inputs_"
    if any(c.startswith(prefix) for c in [*source_schema, *request_schema]):
        raise ValueError("Columns collide with reserved preparation names")
    schema = {**request_schema, **({factor_col: pl.String} if factor_col else {}), **added}
    requested_factors = list(factors or ())
    row, target, decision, known, start, end, value, display_start, display_end = (
        prefix + name
        for name in (
            "row",
            "target",
            "decision",
            "known",
            "start",
            "end",
            "value",
            "display_start",
            "display_end",
        )
    )

    def calculate(asked: pl.DataFrame) -> pl.DataFrame:
        facts = data.collect() if isinstance(data, pl.LazyFrame) else data.clone()
        if asked.select(
            pl.any_horizontal(pl.col(c).is_null() for c in [timestamp, *entities, effective]).any()
        ).item():
            raise ValueError("Request identity/effective keys cannot be unknown")
        asked = _clock(asked, decision_time_col, decision, required=True)
        asked = _effective(asked, effective, target, feed_spec)
        unique = list(dict.fromkeys([timestamp, *entities, target, decision]))
        if asked.select(pl.struct(unique).is_duplicated().any()).item():
            raise ValueError("Duplicate requested decision keys")
        if facts.select(pl.any_horizontal(pl.col(c).is_null() for c in entities).any()).item():
            raise ValueError("Fact stable entity keys cannot be unknown")
        facts = _clock(facts, available_at_col, known)
        if factor_col:
            facts = facts.with_columns(pl.col(factor_col).cast(pl.String))
            if facts[factor_col].null_count():
                raise ValueError("Fact factor names cannot be unknown")
            asked = asked.join(pl.DataFrame({factor_col: requested_factors}), how="cross")
        asked = asked.with_row_index(row)
        if static:
            facts = facts.with_columns(
                pl.lit(None, chosen_dtype).alias(display_start),
                pl.lit(None, chosen_dtype).alias(display_end),
            )
        else:
            if facts[from_col].null_count():
                raise ValueError("Fact effective keys cannot be unknown")
            facts = _effective(facts, from_col, start, feed_spec)
            facts = facts.with_columns(pl.col(from_col).alias(display_start))
            if intervals:
                if facts.schema[effective_to_col] == pl.Null:
                    facts = facts.with_columns(pl.col(effective_to_col).cast(from_dtype))
                facts = _effective(facts, effective_to_col, end, feed_spec)
                if facts.filter(pl.col(end) <= pl.col(start)).height:
                    raise ValueError("Effective intervals must be nonempty,end-exclusive")
                facts = facts.with_columns(pl.col(effective_to_col).alias(display_end))
            else:
                facts = facts.with_columns(pl.lit(None, from_dtype).alias(display_end))
        join_keys = [*entities, *([factor_col] if factor_col else [])]
        columns = [
            *join_keys,
            pl.col(value_col).alias(value),
            known,
            display_start,
            display_end,
            *([] if static else [start, *([end] if intervals else [])]),
        ]
        if not static and not intervals:
            candidates = asked.join(
                facts.select(columns),
                left_on=[*join_keys, target],
                right_on=[*join_keys, start],
                how="inner",
            )
        else:
            candidates = asked.join(facts.select(columns), on=join_keys, how="inner")
        valid = pl.lit(True)
        if not static:
            valid &= (
                (
                    (pl.col(start) <= pl.col(target))
                    & (pl.col(end).is_null() | (pl.col(end) > pl.col(target)))
                )
                if intervals
                else pl.lit(True)
            )
        if purpose == "causal":
            valid &= pl.col(known).is_not_null() & (pl.col(known) <= pl.col(decision))
        candidates = candidates.filter(valid)
        if factor_col and candidates.filter(pl.col(value).cast(pl.Float64).is_infinite()).height:
            raise ValueError("Selected factor loadings must be finite or explicitly missing")
        if candidates.select(pl.col(row).is_duplicated().any()).item():
            raise ValueError("Ambiguous active facts;select source revisions upstream")
        selected = candidates.select(row, value, known, display_start, display_end)
        result = asked.join(selected, on=row, how="left", validate="1:1")
        known_now = (pl.col(known).is_not_null() & (pl.col(known) <= pl.col(decision))).fill_null(
            False
        )
        result = result.with_columns(
            pl.col(value).alias(output),
            pl.col(known).cast(pl.Datetime("ns", "UTC")).alias(output + "_available_at"),
            _usable_value(pl.col(value), source_schema[value_col])
            .fill_null(False)
            .alias(output + "_covered"),
            known_now.alias(output + "_known_at_decision"),
            pl.col(display_start).alias(output + "_effective_from"),
            pl.col(display_end).alias(output + "_effective_to"),
            pl.lit(purpose).alias(output + "_purpose"),
            pl.lit(static).alias(output + "_static"),
        )
        return result.sort(row).select(*schema)

    if isinstance(requests, pl.LazyFrame) or isinstance(data, pl.LazyFrame):
        asked = requests if isinstance(requests, pl.LazyFrame) else requests.lazy()
        return asked.map_batches(
            calculate,
            schema=schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    return calculate(requests)


def prepare_loading_inputs(
    data: pl.DataFrame | pl.LazyFrame | None,
    *,
    requests: pl.DataFrame | pl.LazyFrame,
    factors: Sequence[str],
    feed_spec: FeedSpec,
    purpose: Literal["causal", "retrospective"],
    value_col: str = "loading",
    factor_col: str = "factor",
    output: str = "loading",
    effective_time_col: str | None = None,
    available_at_col: str = "available_at",
    decision_time_col: str = "decision_time",
) -> pl.DataFrame | pl.LazyFrame:
    """Select exact dated supplied loadings,retaining requested missing factor coverage.

    Explicit factors and requested stable IDs define the output grid. No prior
    loading is carried to an absent date and no missing loading becomes zero.
    Data/callers resolve revisions;multiple active known candidates are ambiguous.
    effective_time_col selects the request's applicable key,which can differ from
    its price observation cutoff. No effective date is inferred from UTC dates.
    Retrospective output retains actual publication and known-at-decision flags.
    Lazy requests/facts are evaluated on collection,not silently made eager.
    """
    if not isinstance(factor_col, str) or not factor_col:
        raise ValueError("Declare a nonempty factor identity column name")
    if data is None:
        timestamp, entities = _roles(feed_spec)
        request_schema = (
            requests.collect_schema() if isinstance(requests, pl.LazyFrame) else requests.schema
        )
        data = pl.DataFrame(
            schema={
                **{c: request_schema[c] for c in entities},
                timestamp: request_schema[effective_time_col or timestamp],
                factor_col: pl.String,
                value_col: pl.Float64,
                available_at_col: pl.Datetime("ns", "UTC"),
            }
        )
    return _dated_inputs(
        data,
        requests=requests,
        feed_spec=feed_spec,
        purpose=purpose,
        value_col=value_col,
        output=output,
        effective_time_col=effective_time_col,
        available_at_col=available_at_col,
        decision_time_col=decision_time_col,
        factor_col=factor_col,
        factors=factors,
    )


def prepare_classification_inputs(
    data: pl.DataFrame | pl.LazyFrame | Mapping[Any, Any] | None,
    *,
    requests: pl.DataFrame | pl.LazyFrame,
    feed_spec: FeedSpec,
    purpose: Literal["causal", "retrospective"],
    value_col: str = "sector",
    output: str = "sector",
    effective_time_col: str | None = None,
    available_at_col: str = "available_at",
    decision_time_col: str = "decision_time",
    effective_from_col: str = "effective_from",
    effective_to_col: str = "effective_to",
    static: bool = False,
    static_known_at: datetime | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Prepare known date-effective classifications or an explicitly static map.

    Intervals are start-inclusive/end-exclusive. Unknown or expired categories
    remain uncovered,not a default sector. A static mapping is not historical
    truth;declare static=True,and a real static_known_at instant for causal use.
    This joins supplied facts without creating a provider,universe or revision
    selector. It does not normalize holdings or compute sector exposures.
    """
    timestamp, entities = _roles(feed_spec)
    if data is None:
        if static:
            raise ValueError("static=True requires an explicit mapping,not absent facts")
        request_schema = (
            requests.collect_schema() if isinstance(requests, pl.LazyFrame) else requests.schema
        )
        effective_dtype = request_schema[effective_time_col or timestamp]
        data = pl.DataFrame(
            schema={
                **{c: request_schema[c] for c in entities},
                effective_from_col: effective_dtype,
                effective_to_col: effective_dtype,
                value_col: pl.String,
                available_at_col: pl.Datetime("ns", "UTC"),
            }
        )
    if isinstance(data, Mapping):
        if not static:
            raise ValueError("Declare static=True for a static classification map")
        if purpose == "causal" and static_known_at is None:
            raise ValueError("A causal static map requires static_known_at")
        request_schema = (
            requests.collect_schema() if isinstance(requests, pl.LazyFrame) else requests.schema
        )
        identities = list(data)
        if len(entities) > 1 and any(
            not isinstance(key, tuple) or len(key) != len(entities) for key in identities
        ):
            raise ValueError("Compound static identities must match the declared entity columns")
        fields = {}
        for i, column in enumerate(entities):
            values = identities if len(entities) == 1 else [key[i] for key in identities]
            fields[column] = pl.Series(column, values, dtype=request_schema[column])
        fields[value_col] = pl.Series(
            value_col, list(data.values()), dtype=pl.String if not data else None
        )
        fields[available_at_col] = pl.Series(
            available_at_col,
            [static_known_at] * len(data),
            dtype=pl.Datetime("ns", "UTC") if static_known_at is None else None,
        )
        data = pl.DataFrame(fields)
    elif static:
        raise ValueError("static=True requires an explicit mapping,not dated facts")
    return _dated_inputs(
        data,
        requests=requests,
        feed_spec=feed_spec,
        purpose=purpose,
        value_col=value_col,
        output=output,
        effective_time_col=effective_time_col,
        available_at_col=available_at_col,
        decision_time_col=decision_time_col,
        intervals=not static,
        static=static,
        effective_from_col=effective_from_col,
        effective_to_col=effective_to_col,
    )
