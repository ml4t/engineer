"""Prediction artifact contracts owned by ml4t-engineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml4t.specs.base import (
    ArtifactKind,
    ArtifactProvenance,
    ArtifactSpec,
    ArtifactStorage,
    optional_str,
)


@dataclass(frozen=True, slots=True)
class PredictionSchema:
    """Column layout for a prediction artifact."""

    timestamp_col: str = "timestamp"
    entity_col: str = "asset"
    prediction_col: str = "prediction_value"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> PredictionSchema:
        if mapping is None:
            return cls()
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=str(mapping.get("entity_col", "asset")),
            prediction_col=str(mapping.get("prediction_col", "prediction_value")),
        )


@dataclass(frozen=True, slots=True)
class PredictionDefinition:
    """Definition metadata for a prediction artifact."""

    split_protocol: str = "walk_forward_oos"
    label_artifact: str | None = None
    feature_artifacts: tuple[str, ...] = ()
    training_hash: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> PredictionDefinition:
        if mapping is None:
            return cls()
        feature_artifacts = mapping.get("feature_artifacts", ())
        if isinstance(feature_artifacts, str):
            feature_artifacts = (feature_artifacts,)
        return cls(
            split_protocol=str(mapping.get("split_protocol", "walk_forward_oos")),
            label_artifact=optional_str(mapping.get("label_artifact")),
            feature_artifacts=tuple(str(item) for item in feature_artifacts),
            training_hash=optional_str(mapping.get("training_hash")),
        )


@dataclass(frozen=True, slots=True)
class PredictionSpec(ArtifactSpec):
    """Shared specification for persisted prediction artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.PREDICTIONS, init=False)
    schema: PredictionSchema = field(default_factory=PredictionSchema)
    definition: PredictionDefinition = field(default_factory=PredictionDefinition)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> PredictionSpec:
        return cls(
            artifact_id=str(mapping["artifact_id"]),
            version=int(mapping.get("version", 1)),
            storage=ArtifactStorage.from_mapping(mapping.get("storage")),
            provenance=ArtifactProvenance.from_mapping(mapping.get("provenance")),
            schema=PredictionSchema.from_mapping(mapping.get("schema")),
            definition=PredictionDefinition.from_mapping(mapping.get("definition")),
        )


__all__ = ["PredictionDefinition", "PredictionSchema", "PredictionSpec"]
