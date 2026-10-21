"""Feature artifact contracts owned by ml4t-engineer."""

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
class FeatureSchema:
    """Column layout for a feature artifact."""

    timestamp_col: str = "timestamp"
    entity_col: str | tuple[str, ...] = "asset"
    feature_columns: tuple[str, ...] = ()
    available_at_col: str | None = None
    decision_time_col: str | None = None
    eligible_col: str | None = None
    universe_time_col: str | None = None

    def __post_init__(self) -> None:
        columns = [self.entity_col] if isinstance(self.entity_col, str) else list(self.entity_col)
        if not columns or any(not isinstance(c, str) or not c for c in columns):
            raise ValueError("entity_col must declare nonempty stable entity columns")
        if len(set(columns)) != len(columns):
            raise ValueError("entity_col columns must be distinct")
        if not isinstance(self.entity_col, str):
            object.__setattr__(self, "entity_col", tuple(columns))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> FeatureSchema:
        if mapping is None:
            return cls()
        feature_columns = mapping.get("feature_columns", ())
        return cls(
            timestamp_col=str(mapping.get("timestamp_col", "timestamp")),
            entity_col=mapping.get("entity_col", "asset"),
            feature_columns=tuple(str(item) for item in feature_columns),
            available_at_col=optional_str(mapping.get("available_at_col")),
            decision_time_col=optional_str(mapping.get("decision_time_col")),
            eligible_col=optional_str(mapping.get("eligible_col")),
            universe_time_col=optional_str(mapping.get("universe_time_col")),
        )


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Definition metadata for a feature artifact."""

    family: str = "financial"
    join_keys: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()
    feed_spec: FeedSpec | None = None
    window_unit: WindowUnit | None = None
    price_basis: str | None = None
    lookbacks: dict[str, int] | None = None
    missing_input_policy: str | None = None

    def __post_init__(self) -> None:
        if self.feed_spec is not None:
            object.__setattr__(self, "feed_spec", FeedSpec.from_any(self.feed_spec))
        if self.window_unit is not None:
            object.__setattr__(self, "window_unit", WindowUnit(self.window_unit))
        if self.price_basis is not None and not self.price_basis.strip():
            raise ValueError("price_basis must be a nonempty declaration")
        if self.missing_input_policy not in {None, "complete", "known_missing"}:
            raise ValueError("missing_input_policy must be complete or known_missing")
        if self.lookbacks is not None:
            if any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for name, value in self.lookbacks.items()
            ):
                raise ValueError("lookbacks must map feature names to positive integer row counts")
            object.__setattr__(self, "lookbacks", dict(self.lookbacks))

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
            feed_spec=FeedSpec.from_any(mapping["feed_spec"])
            if mapping.get("feed_spec") is not None
            else None,
            window_unit=mapping.get("window_unit"),
            price_basis=optional_str(mapping.get("price_basis")),
            lookbacks=mapping.get("lookbacks"),
            missing_input_policy=optional_str(mapping.get("missing_input_policy")),
        )


@dataclass(frozen=True, slots=True)
class FeatureSpec(ArtifactSpec):
    """Shared specification for persisted feature artifacts."""

    kind: ArtifactKind = field(default=ArtifactKind.FEATURES, init=False)
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    definition: FeatureDefinition = field(default_factory=FeatureDefinition)

    def to_dict(self) -> dict[str, Any]:
        payload = ArtifactSpec.to_dict(self)
        for name in ("available_at_col", "decision_time_col", "eligible_col", "universe_time_col"):
            if payload["schema"][name] is None:
                del payload["schema"][name]
        for name in (
            "feed_spec",
            "window_unit",
            "price_basis",
            "lookbacks",
            "missing_input_policy",
        ):
            if payload["definition"][name] is None:
                del payload["definition"][name]
        if self.definition.feed_spec is not None:
            payload["definition"]["feed_spec"] = self.definition.feed_spec.to_dict()
        return payload

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
