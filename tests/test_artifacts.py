"""Behavioral tests for persisted artifact contracts."""

from ml4t.engineer.artifacts import (
    FeatureDefinition,
    FeatureSchema,
    FeatureSpec,
    LabelDefinition,
    LabelSchema,
    LabelSpec,
    PredictionDefinition,
    PredictionSchema,
    PredictionSpec,
)


def test_artifact_component_defaults() -> None:
    assert FeatureSchema.from_mapping(None) == FeatureSchema()
    assert FeatureDefinition.from_mapping(None) == FeatureDefinition()
    assert LabelSchema.from_mapping(None) == LabelSchema()
    assert LabelDefinition.from_mapping(None) == LabelDefinition()
    assert PredictionSchema.from_mapping(None) == PredictionSchema()
    assert PredictionDefinition.from_mapping(None) == PredictionDefinition()


def test_feature_spec_converts_nested_mapping_values() -> None:
    spec = FeatureSpec.from_mapping(
        {
            "artifact_id": 42,
            "version": "3",
            "storage": {
                "path": "features.parquet",
                "format": "parquet",
                "partition_by": ["date", "asset"],
            },
            "provenance": {
                "source_artifacts": ["bars-v1"],
                "content_hash": "sha256:abc",
                "created_by": "pipeline",
            },
            "schema": {
                "timestamp_col": "event_time",
                "entity_col": "symbol",
                "feature_columns": ["rsi", 7],
            },
            "definition": {
                "family": "technical",
                "join_keys": ["event_time", "symbol"],
                "source_artifacts": ["bars-v1"],
            },
        }
    )

    assert spec.artifact_id == "42"
    assert spec.version == 3
    assert spec.kind.value == "features"
    assert spec.storage.partition_by == ("date", "asset")
    assert spec.provenance.source_artifacts == ("bars-v1",)
    assert spec.schema == FeatureSchema("event_time", "symbol", ("rsi", "7"))
    assert spec.definition == FeatureDefinition(
        "technical",
        ("event_time", "symbol"),
        ("bars-v1",),
    )


def test_label_spec_converts_optional_definition_values() -> None:
    spec = LabelSpec.from_mapping(
        {
            "artifact_id": "labels-v2",
            "schema": {
                "timestamp_col": "event_time",
                "entity_col": "symbol",
                "label_col": "target",
            },
            "definition": {
                "family": "triple_barrier",
                "task_type": "classification",
                "horizon": 20,
                "buffer": "1h",
                "source_artifact": "bars-v1",
                "reference_field": "close",
                "execution_delay": 1,
                "session_bounded": 1,
            },
        }
    )

    assert spec.kind.value == "labels"
    assert spec.schema == LabelSchema("event_time", "symbol", "target")
    assert spec.definition == LabelDefinition(
        family="triple_barrier",
        task_type="classification",
        horizon="20",
        buffer="1h",
        source_artifact="bars-v1",
        reference_field="close",
        execution_delay="1",
        session_bounded=True,
    )


def test_prediction_spec_accepts_one_or_many_feature_artifacts() -> None:
    common = {
        "artifact_id": "predictions-v1",
        "schema": {
            "timestamp_col": "event_time",
            "entity_col": "symbol",
            "prediction_col": "score",
        },
        "definition": {
            "split_protocol": "purged_walk_forward",
            "label_artifact": "labels-v2",
            "training_hash": 123,
        },
    }

    one = PredictionSpec.from_mapping(
        {
            **common,
            "definition": {**common["definition"], "feature_artifacts": "features-v1"},
        }
    )
    many = PredictionSpec.from_mapping(
        {
            **common,
            "definition": {
                **common["definition"],
                "feature_artifacts": ["features-v1", 2],
            },
        }
    )

    assert one.kind.value == "predictions"
    assert one.schema == PredictionSchema("event_time", "symbol", "score")
    assert one.definition == PredictionDefinition(
        split_protocol="purged_walk_forward",
        label_artifact="labels-v2",
        feature_artifacts=("features-v1",),
        training_hash="123",
    )
    assert many.definition.feature_artifacts == ("features-v1", "2")
