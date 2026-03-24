"""Feature artifact contracts owned by ml4t-engineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml4t.data.artifacts.base import ArtifactKind, ArtifactProvenance, ArtifactSpec, ArtifactStorage


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Column layout for a feature artifact."""

    timestamp_col: str = "timestamp"
    entity_col: str = "symbol"
    feature_columns: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> FeatureSchema:
        if mapping is None:
            return cls()
        feature_columns = mapping.get("feature_columns", ())
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=str(mapping.get("entity_col", "symbol")),
            feature_columns=tuple(str(item) for item in feature_columns),
        )


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Definition metadata for a feature artifact."""

    family: str = "financial"
    join_keys: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> FeatureDefinition:
        if mapping is None:
            return cls()
        join_keys = mapping.get("join_keys", ())
        source_artifacts = mapping.get("source_artifacts", ())
        return cls(
            family=str(mapping.get("family", "financial")),
            join_keys=tuple(str(item) for item in join_keys),
            source_artifacts=tuple(str(item) for item in source_artifacts),
        )


@dataclass(frozen=True, slots=True)
class FeatureSpec(ArtifactSpec):
    """Shared specification for persisted feature artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.FEATURES, init=False)
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    definition: FeatureDefinition = field(default_factory=FeatureDefinition)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> FeatureSpec:
        return cls(
            artifact_id=str(mapping["artifact_id"]),
            version=int(mapping.get("version", 1)),
            storage=ArtifactStorage.from_mapping(mapping.get("storage")),
            provenance=ArtifactProvenance.from_mapping(mapping.get("provenance")),
            schema=FeatureSchema.from_mapping(mapping.get("schema")),
            definition=FeatureDefinition.from_mapping(mapping.get("definition")),
        )


__all__ = ["FeatureDefinition", "FeatureSchema", "FeatureSpec"]
