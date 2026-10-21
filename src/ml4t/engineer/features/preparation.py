"""Decision-time feature eligibility over caller/Data-prepared native histories."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import polars as pl
from ml4t.specs import UNSET_CALENDAR, FeedSpec, TimestampSemantics, WindowUnit

from ml4t.engineer.core.bar_grid import validate_declared_grid
from ml4t.engineer.features.events import _instant_ticks
from ml4t.engineer.features.rolling import _period


def _usable_value(value: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    valid = value.is_not_null()
    if dtype.is_float():
        valid &= value.is_finite()
    elif isinstance(dtype, pl.Struct):
        valid &= pl.all_horizontal(
            _usable_value(
                value.struct.field(field.name),
                field.dtype if isinstance(field.dtype, pl.DataType) else field.dtype(),
            )
            for field in dtype.fields
        )
    return valid


def prepare_feature_panel(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    features: Mapping[str, pl.Expr],
    lookbacks: Mapping[str, int],
    input_columns: Sequence[str],
    decisions: pl.DataFrame | pl.LazyFrame,
    universe: pl.DataFrame | pl.LazyFrame,
    price_basis: str,
    feed_spec: FeedSpec | None = None,
    timestamp_col: str | None = None,
    entity_col: str | list[str] | None = None,
    calendar: Any = UNSET_CALENDAR,
    window_unit: str | WindowUnit | None = None,
    available_at_col: str = "available_at",
    decision_time_col: str = "decision_time",
    universe_time_col: str = "effective_session",
    history_decision_time_col: str | None = None,
    missing_input_policy: Literal["complete", "known_missing"] = "complete",
    input_mask: str | pl.Expr | None = None,
    screen: str | pl.Expr | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calculate finite per-asset features with explicit causal eligibility.

    decisions supplies an observation cutoff, an aware decision instant and the
    universe's decision-effective key (effective_session by default). These need
    not identify the same session. universe_time_col can select an explicit
    instant column for instant-effective facts; no session is inferred from dates.
    universe supplies stable IDs, effective_from/inclusive, effective_to/exclusive
    (null means open), available_at and nullable boolean member, plus optional
    classifications. Known active overlaps are ambiguous and raise. Missing facts
    never imply membership. Pre-entry price history is retained for warmup.

    Data/callers select revisions and adjustment perspective before this boundary.
    For different prepared histories at different decisions, label those snapshots
    with history_decision_time_col; no snapshot/revision is implicitly selected or
    carried forward. The declared price_basis is retained, not calculated.

    Supply bare per-asset expressions, their finite row lookbacks, and every input
    column they require. Full required windows must contain known, usable inputs
    by default. known_missing permits confirmed null/NaN or input-masked values
    to follow the supplied kernel's missing-value definition. All original rows
    must still have known publication times; infinite inputs remain invalid.
    Unavailable/masked rows stay in history with null inputs, never compressed or
    filled. A final screen runs only after all values and eligibility are computed.
    Cross-sectional transforms operate on the resulting eligible population.
    """
    if not features or set(features) != set(lookbacks):
        raise ValueError("Declare the same nonempty feature names and lookbacks")
    if any(
        not isinstance(name, str) or not name or not isinstance(expr, pl.Expr)
        for name, expr in features.items()
    ):
        raise ValueError("features must map nonempty names to native expressions")
    for count in lookbacks.values():
        _period(count)
        if count > 2**63 - 1:
            raise ValueError("Feature lookbacks must fit signed 64-bit row counts")
    if (
        isinstance(input_columns, str)
        or not input_columns
        or len(set(input_columns)) != len(input_columns)
    ):
        raise ValueError("Declare distinct input_columns for the supplied expressions")
    if not isinstance(price_basis, str) or not price_basis.strip():
        raise ValueError("Declare a nonempty price_basis")
    if missing_input_policy not in {"complete", "known_missing"}:
        raise ValueError("missing_input_policy must be complete or known_missing")
    if any(
        value is not None and not isinstance(value, str | pl.Expr) for value in (input_mask, screen)
    ):
        raise TypeError("input_mask and screen must be column names or expressions")
    if feed_spec is not None and not isinstance(feed_spec, FeedSpec):
        raise TypeError("feed_spec must be a canonical FeedSpec instance, not an unresolved config")
    contract = FeedSpec(time_contract_version=2) if feed_spec is None else feed_spec
    for name, explicit, declared in (
        ("timestamp_col", timestamp_col, contract.timestamp_col),
        ("entity_col", entity_col, contract.entity_col),
    ):
        if (
            explicit is not None
            and declared is not None
            and feed_spec is not None
            and ([explicit] if isinstance(explicit, str) else list(explicit))
            != ([declared] if isinstance(declared, str) else list(declared))
        ):
            raise ValueError(f"Conflicting {name} and feed_spec declaration")
    timestamp = timestamp_col or contract.timestamp_col
    identity = entity_col if entity_col is not None else contract.entity_col
    entities = (
        [identity]
        if isinstance(identity, str)
        else list(identity)
        if identity is not None
        else ["asset"]
    )
    if not entities or len(set(entities)) != len(entities):
        raise ValueError("Declare distinct stable entity columns")
    keys = [timestamp, *entities]
    if len(set(keys)) != len(keys) or set(input_columns) & set(keys):
        raise ValueError("Input columns and timestamp/entity roles must be distinct")
    resolved = contract.resolve_calendar(calendar=calendar)
    unit = (
        WindowUnit(window_unit)
        if window_unit is not None
        else (WindowUnit.OBSERVATION if resolved is None else WindowUnit.EXPECTED_BAR)
    )
    if unit != (WindowUnit.OBSERVATION if resolved is None else WindowUnit.EXPECTED_BAR):
        raise ValueError(
            "Finite feature preparation requires observation or declared expected_bar rows"
        )
    decision_frame = (
        decisions.collect() if isinstance(decisions, pl.LazyFrame) else decisions.clone()
    )
    universe_frame = universe.collect() if isinstance(universe, pl.LazyFrame) else universe.clone()
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    for label, columns, required in (
        (
            "history",
            schema,
            [
                *keys,
                *input_columns,
                available_at_col,
                *([history_decision_time_col] if history_decision_time_col else []),
            ],
        ),
        ("decisions", decision_frame.schema, [timestamp, decision_time_col, universe_time_col]),
        (
            "universe",
            universe_frame.schema,
            [*entities, "effective_from", "effective_to", available_at_col, "member"],
        ),
    ):
        if set(required) - set(columns):
            raise ValueError(f"Missing {label} columns: {sorted(set(required) - set(columns))}")
    if schema[timestamp].base_type() not in {pl.Date, pl.Datetime}:
        raise ValueError("Observation keys require Date labels or declared Datetime instants")
    if resolved is not None:
        validate_declared_grid(
            pl.DataFrame(schema=schema),
            feed_spec=contract,
            calendar=resolved,
            timestamp_col=timestamp,
            group_cols=entities,
        )
    if schema[timestamp] != decision_frame.schema[timestamp]:
        raise ValueError("Decision observation cutoffs must preserve the input timestamp dtype")
    universe_dtype = decision_frame.schema[universe_time_col]
    if universe_frame.schema["effective_to"] == pl.Null:
        universe_frame = universe_frame.with_columns(pl.col("effective_to").cast(universe_dtype))
    if universe_dtype.base_type() not in {pl.Date, pl.Datetime} or any(
        universe_frame.schema[c].base_type() != universe_dtype.base_type()
        for c in ("effective_from", "effective_to")
    ):
        raise ValueError(
            "Universe effective intervals must match the declared decision-effective clock"
        )
    if universe_frame.schema["member"] != pl.Boolean:
        raise ValueError("Universe member must be nullable boolean, not an inferred classifier")
    roles = [*keys, decision_time_col, available_at_col, "member", "eligible", "price_basis"]
    if len(set(roles)) != len(roles) or set(features) & set(roles):
        raise ValueError("Feature and output roles collide")
    if universe_time_col in {
        *entities,
        available_at_col,
        "member",
        "eligible",
        "price_basis",
        *features,
    }:
        raise ValueError("The universe clock collides with a different output role")
    if universe_time_col not in roles:
        roles.append(universe_time_col)
    extra = [
        c
        for c in universe_frame.columns
        if c not in [*entities, "effective_from", "effective_to", available_at_col, "member"]
    ]
    if set(extra) & (set(roles) | set(features)):
        raise ValueError("Universe classification names collide with feature/output roles")
    prefix = "__ml4t_feature_"
    if any(
        c.startswith(prefix)
        for c in [*schema, *universe_frame.columns, *decision_frame.columns, *features]
    ):
        raise ValueError("Input columns collide with reserved feature preparation names")
    known, cutoff, source_usable, source_ready, feature_known = (
        prefix + c for c in ("known", "cutoff", "usable", "ready", "time")
    )

    def clock(
        frame: pl.DataFrame,
        column: str,
        output: str,
        *,
        required: bool = False,
        range_guard: bool = True,
    ) -> pl.DataFrame:
        expr = (
            pl.lit(None, pl.Int128) if frame.schema[column] == pl.Null else _instant_ticks(column)
        )
        result = frame.with_columns(expr.alias(output))
        if required and result[output].null_count():
            raise ValueError(f"{column} must contain known decision/snapshot instants")
        if (
            range_guard
            and result.filter((pl.col(output) < -(2**63)) | (pl.col(output) > 2**63 - 1)).height
        ):
            raise ValueError(f"{column} exceeds Datetime(ns) output range")
        return result

    decision_frame = clock(decision_frame, decision_time_col, cutoff, required=True)
    timestamp_dtype = schema[timestamp]
    if (
        isinstance(timestamp_dtype, pl.Datetime)
        and contract.timestamp_semantics != TimestampSemantics.SESSION_LABEL
    ):
        observation = pl.col(timestamp)
        if timestamp_dtype.time_zone is None:
            if contract.timezone is None:
                raise ValueError("Naive observation instants require a declared timezone")
            observation = observation.dt.replace_time_zone(contract.timezone)
        decision_frame = decision_frame.with_columns(observation.alias(prefix + "observation"))
        decision_frame = clock(
            decision_frame, prefix + "observation", prefix + "observation_ticks", required=True
        )
        if decision_frame.filter(pl.col(prefix + "observation_ticks") > pl.col(cutoff)).height:
            raise ValueError("Observation cutoff cannot follow its decision instant")
    universe_frame = clock(universe_frame, available_at_col, known)
    universe_from, universe_to, universe_target = (
        "effective_from",
        "effective_to",
        universe_time_col,
    )
    if isinstance(universe_dtype, pl.Datetime):

        def effective_clock(frame: pl.DataFrame, column: str, target: str) -> pl.DataFrame:
            dtype = frame.schema[column]
            assert isinstance(dtype, pl.Datetime)
            value = pl.col(column)
            if dtype.time_zone is None:
                if contract.timezone is None:
                    raise ValueError("Naive universe instants require a declared timezone")
                value = value.dt.replace_time_zone(contract.timezone)
            temporary = target + "_aware"
            return clock(
                frame.with_columns(value.alias(temporary)), temporary, target, range_guard=False
            ).drop(temporary)

        universe_from, universe_to, universe_target = (
            prefix + name for name in ("from", "to", "target")
        )
        universe_frame = effective_clock(universe_frame, "effective_from", universe_from)
        universe_frame = effective_clock(universe_frame, "effective_to", universe_to)
        decision_frame = effective_clock(decision_frame, universe_time_col, universe_target)
    if decision_frame[universe_target].null_count():
        raise ValueError("Decision-effective universe keys cannot be unknown")
    if (
        decision_frame[timestamp].null_count()
        or decision_frame.select(
            pl.struct(list(dict.fromkeys([timestamp, cutoff, universe_target])))
            .is_duplicated()
            .any()
        ).item()
    ):
        raise ValueError("Decision observation/instant keys must be nonnull and unique")
    if universe_frame.select(
        pl.any_horizontal(pl.col(c).is_null() for c in [*entities, "effective_from"]).any()
    ).item():
        raise ValueError("Universe identity/effective start cannot be unknown")
    if universe_frame.filter(pl.col(universe_to) <= pl.col(universe_from)).height:
        raise ValueError("Universe effective intervals must be nonempty and end-exclusive")
    template = (
        pl.DataFrame(schema=schema)
        .select(*keys, *input_columns)
        .lazy()
        .with_columns(expr.over(entities).alias(name) for name, expr in features.items())
        .collect_schema()
    )
    result_schema = {
        timestamp: schema[timestamp],
        **{c: schema[c] for c in entities},
        decision_time_col: pl.Datetime("ns", "UTC"),
        **{name: template[name] for name in features},
        available_at_col: pl.Datetime("ns", "UTC"),
        "member": pl.Boolean,
        "eligible": pl.Boolean,
        "price_basis": pl.String,
        **{c: universe_frame.schema[c] for c in extra},
    }
    if universe_time_col not in result_schema:
        result_schema[universe_time_col] = decision_frame.schema[universe_time_col]

    def calculate(frame: pl.DataFrame) -> pl.DataFrame:
        source = clock(frame, available_at_col, known)
        unique_keys = [*keys]
        if history_decision_time_col:
            source = clock(source, history_decision_time_col, prefix + "snapshot", required=True)
            unique_keys.append(prefix + "snapshot")
        if source.select(pl.any_horizontal(pl.col(c).is_null() for c in keys).any()).item():
            raise ValueError("History timestamp/entity keys cannot be unknown")
        if source.select(pl.struct(unique_keys).is_duplicated().any()).item():
            raise ValueError(
                "Duplicate history keys; select source revisions/perspectives upstream"
            )
        output = []
        for i in range(decision_frame.height):
            decision = decision_frame.slice(i, 1)
            bound = pl.lit(decision[timestamp]).first()
            instant = decision[cutoff].item()
            effective = (
                pl.lit(decision[universe_target].item(), dtype=pl.Int128)
                if isinstance(universe_dtype, pl.Datetime)
                else pl.lit(decision[universe_time_col]).first()
            )
            history = source
            if history_decision_time_col:
                history = history.filter(pl.col(prefix + "snapshot") == instant)
                if history.is_empty():
                    raise ValueError("Missing prepared history snapshot for a requested decision")
            history = history.filter(pl.col(timestamp) <= bound).sort([*entities, timestamp])
            facts = universe_frame.filter(
                (pl.col(universe_from) <= effective)
                & (pl.col(universe_to).is_null() | (pl.col(universe_to) > effective))
                & (pl.col(known) <= instant)
            )
            if facts.select(pl.struct(entities).is_duplicated().any()).item():
                raise ValueError("Ambiguous overlapping known universe facts; resolve upstream")
            if resolved is not None and not history.is_empty():
                validate_declared_grid(
                    history,
                    feed_spec=contract,
                    calendar=resolved,
                    timestamp_col=timestamp,
                    group_cols=entities,
                )
            usable = pl.col(known).is_not_null() & (pl.col(known) <= instant)
            confirmed = usable
            for column in input_columns:
                value = pl.col(column)
                valid = _usable_value(value, schema[column])
                missing = value.is_null()
                if schema[column].is_float():
                    missing |= value.is_nan()
                confirmed &= valid | missing
                usable &= valid
            if input_mask is not None:
                usable &= (
                    pl.col(input_mask) if isinstance(input_mask, str) else input_mask
                ).fill_null(False)
            history = history.with_columns(
                usable.alias(source_usable),
                (usable if missing_input_policy == "complete" else confirmed).alias(source_ready),
            )
            assets = pl.concat(
                [history.filter(pl.col(known) <= instant).select(entities), facts.select(entities)]
            ).unique()
            history = history.with_columns(
                pl.when(pl.col(source_usable)).then(pl.col(c)).alias(c) for c in input_columns
            )
            available = []
            for name in features:
                count = lookbacks[name]
                full = pl.col(source_ready).cast(pl.Int64).rolling_sum(count) == count
                available.append(
                    pl.when(full)
                    .then(pl.col(known).cast(pl.Int64).rolling_max(count))
                    .over(entities)
                    .alias(prefix + name)
                )
            history = (
                history.with_columns(available)
                .select(*keys, *input_columns, *[prefix + name for name in features])
                .with_columns(expr.over(entities).alias(name) for name, expr in features.items())
            )
            history = history.with_columns(
                pl.when(pl.col(prefix + name).is_not_null()).then(pl.col(name)).alias(name)
                for name in features
            )
            knowledge = pl.when(
                pl.all_horizontal(pl.col(prefix + name).is_not_null() for name in features)
            ).then(pl.max_horizontal(pl.col(prefix + name) for name in features))
            latest = history.filter(pl.col(timestamp) == bound).select(
                *entities, *features, knowledge.alias(feature_known)
            )
            result = assets.join(latest, on=entities, how="left", validate="1:1").join(
                facts.select(
                    *entities, "member", *extra, pl.col(known).cast(pl.Int64).alias(known)
                ),
                on=entities,
                how="left",
                validate="1:1",
            )
            result = result.with_columns(
                bound.alias(timestamp),
                pl.lit(instant).cast(pl.Datetime("ns", "UTC")).alias(decision_time_col),
                pl.when(pl.col(feature_known).is_not_null())
                .then(pl.max_horizontal(feature_known, known))
                .cast(pl.Datetime("ns", "UTC"))
                .alias(available_at_col),
                (
                    pl.col("member").fill_null(False)
                    & pl.col(feature_known).is_not_null()
                    & pl.all_horizontal(
                        _usable_value(pl.col(name), template[name]) for name in features
                    )
                ).alias("eligible"),
                pl.lit(price_basis).alias("price_basis"),
            )
            if universe_time_col not in {*keys, decision_time_col}:
                result = result.with_columns(
                    pl.lit(decision[universe_time_col]).first().alias(universe_time_col)
                )
            output.append(result.select(*result_schema))
        result = pl.concat(output) if output else pl.DataFrame(schema=result_schema)
        result = result.sort(
            list(dict.fromkeys([decision_time_col, timestamp, universe_time_col, *entities]))
        )
        if screen is not None:
            result = result.filter(pl.col(screen) if isinstance(screen, str) else screen)
        return result

    if isinstance(data, pl.LazyFrame):
        return data.map_batches(
            calculate,
            schema=result_schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    return calculate(data)
