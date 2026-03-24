"""Label artifact contracts owned by ml4t-engineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml4t.data.artifacts.base import (
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
    entity_col: str = "symbol"
    label_col: str = "label"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> LabelSchema:
        if mapping is None:
            return cls()
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=str(mapping.get("entity_col", "symbol")),
            label_col=str(mapping.get("label_col", "label")),
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
        )


@dataclass(frozen=True, slots=True)
class LabelSpec(ArtifactSpec):
    """Shared specification for persisted label artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.LABELS, init=False)
    schema: LabelSchema = field(default_factory=LabelSchema)
    definition: LabelDefinition = field(default_factory=LabelDefinition)

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
