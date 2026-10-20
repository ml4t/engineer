"""Label artifact contracts owned by ml4t-engineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml4t.specs import FeedSpec, WindowUnit
from ml4t.specs.base import (
    ArtifactKind,
    ArtifactProvenance,
    ArtifactSpec,
    ArtifactStorage,
    optional_str,
)


@dataclass(frozen=True, slots=True)
class LabelSchema:
    """Column layout for a label artifact."""

    timestamp_col: str = "timestamp"
    entity_col: str = "asset"
    label_col: str = "label_value"
    entry_time_col: str | None = None
    requested_exit_time_col: str | None = None
    exit_time_col: str | None = None
    outcome_available_at_col: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> LabelSchema:
        if mapping is None:
            return cls()
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=str(mapping.get("entity_col", "asset")),
            label_col=str(mapping.get("label_col", "label_value")),
            entry_time_col=optional_str(mapping.get("entry_time_col")),
            requested_exit_time_col=optional_str(mapping.get("requested_exit_time_col")),
            exit_time_col=optional_str(mapping.get("exit_time_col")),
            outcome_available_at_col=optional_str(mapping.get("outcome_available_at_col")),
        )


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """Definition metadata for a label artifact."""

    family: str = "forward_return"
    task_type: str = "regression"
    horizon: str | None = None
    buffer: str | None = None
    source_artifact: str | None = None
    reference_field: str | None = None
    execution_delay: str | None = None
    session_bounded: bool = False
    window_unit: WindowUnit | None = None
    endpoint_method: str | None = None
    return_method: str | None = None
    missing_price_policy: str | None = None
    price_basis: str | None = None
    feed_spec: FeedSpec | None = None
    index_mode: str | None = None
    tail_policy: str | None = None
    endpoint_tolerance: str | None = None
    original_row_col: str | None = None
    original_price_col: str | None = None

    def __post_init__(self) -> None:
        if (self.original_row_col is None) != (self.original_price_col is None):
            raise ValueError("Declare both original row and price mask columns")
        if self.window_unit is not None:
            try:
                object.__setattr__(self, "window_unit", WindowUnit(self.window_unit))
            except ValueError as exc:
                raise ValueError(f"Invalid window_unit: {self.window_unit!r}") from exc
        for name, allowed in (
            ("endpoint_method", {"cumulative", "exit_bar"}),
            ("return_method", {"returns", "log_returns"}),
            ("missing_price_policy", {"raise", "null"}),
            ("index_mode", {"per_asset", "common_grid"}),
            ("tail_policy", {"null", "drop", "raise"}),
        ):
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(f"Invalid {name}: {value!r}; expected one of {sorted(allowed)}")
        if self.feed_spec is not None:
            object.__setattr__(self, "feed_spec", FeedSpec.from_any(self.feed_spec))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> LabelDefinition:
        if mapping is None:
            return cls()
        return cls(
            family=str(mapping.get("family", "forward_return")),
            task_type=str(mapping.get("task_type", "regression")),
            horizon=optional_str(mapping.get("horizon")),
            buffer=optional_str(mapping.get("buffer")),
            source_artifact=optional_str(mapping.get("source_artifact")),
            reference_field=optional_str(mapping.get("reference_field")),
            execution_delay=optional_str(mapping.get("execution_delay")),
            session_bounded=bool(mapping.get("session_bounded", False)),
            window_unit=mapping.get("window_unit"),
            endpoint_method=optional_str(mapping.get("endpoint_method")),
            return_method=optional_str(mapping.get("return_method")),
            missing_price_policy=optional_str(mapping.get("missing_price_policy")),
            price_basis=optional_str(mapping.get("price_basis")),
            feed_spec=FeedSpec.from_any(mapping["feed_spec"])
            if mapping.get("feed_spec") is not None
            else None,
            index_mode=optional_str(mapping.get("index_mode")),
            tail_policy=optional_str(mapping.get("tail_policy")),
            endpoint_tolerance=optional_str(mapping.get("endpoint_tolerance")),
            original_row_col=optional_str(mapping.get("original_row_col")),
            original_price_col=optional_str(mapping.get("original_price_col")),
        )


@dataclass(frozen=True, slots=True)
class LabelSpec(ArtifactSpec):
    """Shared specification for persisted label artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.LABELS, init=False)
    schema: LabelSchema = field(default_factory=LabelSchema)
    definition: LabelDefinition = field(default_factory=LabelDefinition)

    def to_dict(self) -> dict[str, Any]:
        payload = ArtifactSpec.to_dict(self)
        for name in (
            "entry_time_col",
            "requested_exit_time_col",
            "exit_time_col",
            "outcome_available_at_col",
        ):
            if payload["schema"][name] is None:
                del payload["schema"][name]
        for name in (
            "window_unit",
            "endpoint_method",
            "return_method",
            "missing_price_policy",
            "price_basis",
            "feed_spec",
            "index_mode",
            "tail_policy",
            "endpoint_tolerance",
            "original_row_col",
            "original_price_col",
        ):
            if payload["definition"][name] is None:
                del payload["definition"][name]
        if self.definition.feed_spec is not None:
            # Shared serialization, not dataclass asdict, owns calendar omission/null.
            payload["definition"]["feed_spec"] = self.definition.feed_spec.to_dict()
        return payload

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> LabelSpec:
        return cls(
            artifact_id=str(mapping["artifact_id"]),
            version=int(mapping.get("version", 1)),
            storage=ArtifactStorage.from_mapping(mapping.get("storage")),
            provenance=ArtifactProvenance.from_mapping(mapping.get("provenance")),
            schema=LabelSchema.from_mapping(mapping.get("schema")),
            definition=LabelDefinition.from_mapping(mapping.get("definition")),
        )


__all__ = ["LabelDefinition", "LabelSchema", "LabelSpec"]
