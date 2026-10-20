"""Persisted native outcomes must retain the declaration used to select endpoints."""

import json

import polars as pl
import pytest
import yaml
from ml4t.specs import FeedSpec

from ml4t.engineer.artifacts import LabelSpec


def test_legacy_label_spec_wire_keys_are_unchanged_without_new_declarations():
    wire = LabelSpec(artifact_id="legacy").to_dict()
    assert wire["schema"] == {
        "timestamp_col": "timestamp",
        "entity_col": "asset",
        "label_col": "label_value",
    }
    assert set(wire["definition"]) == {
        "family",
        "task_type",
        "horizon",
        "buffer",
        "source_artifact",
        "reference_field",
        "execution_delay",
        "session_bounded",
    }


def _wire_definition(feed):
    return {
        "family": "forward_return",
        "horizon": "1ns",
        "window_unit": "elapsed",
        "endpoint_method": "cumulative",
        "return_method": "returns",
        "missing_price_policy": "null",
        "price_basis": "split_adjusted",
        "feed_spec": feed.to_dict(),
        "index_mode": "common_grid",
        "tail_policy": "null",
        "endpoint_tolerance": "1ns",
        "original_row_col": "original_row",
        "original_price_col": "original_quote",
    }


@pytest.mark.parametrize("selector", ["omitted", None, "24/7"])
@pytest.mark.parametrize("format", ["json", "yaml"])
def test_outcome_spec_roundtrip_keeps_calendar_presence_and_endpoint_definition(selector, format):
    feed = FeedSpec(time_contract_version=2, entity_col="asset", data_frequency="1ns")
    if selector != "omitted":
        feed = feed.with_overrides(calendar=selector)
    definition = _wire_definition(feed)
    source = {
        "artifact_id": "outcomes",
        "definition": definition,
        "schema": {
            "label_col": "fwd_ns",
            "entry_time_col": "entry",
            "requested_exit_time_col": "target",
            "exit_time_col": "exit",
            "outcome_available_at_col": "available",
        },
    }
    produced = LabelSpec.from_mapping(source).to_dict()
    for key, value in definition.items():
        assert produced["definition"][key] == value
    for key, value in source["schema"].items():
        assert produced["schema"][key] == value
    stored = json.dumps(produced) if format == "json" else yaml.safe_dump(produced)
    restored = json.loads(stored) if format == "json" else yaml.safe_load(stored)
    assert LabelSpec.from_mapping(restored).to_dict() == produced
    assert ("calendar" in restored["definition"]["feed_spec"]) == (selector != "omitted")
    assert FeedSpec.from_any(restored["definition"]["feed_spec"]).resolve_calendar() == (
        "NYSE" if selector == "omitted" else selector
    )


def test_outcome_frame_and_spec_keep_distinct_nanosecond_endpoints_on_disk(tmp_path):
    stamp = 1735689600000000000
    frame = pl.DataFrame(
        {
            "timestamp": pl.Series([stamp], dtype=pl.Int64).cast(pl.Datetime("ns", "UTC")),
            "asset": ["A"],
            "fwd_ns": [0.1],
            "exit": pl.Series([stamp + 1], dtype=pl.Int64).cast(pl.Datetime("ns", "UTC")),
        }
    )
    feed = FeedSpec(time_contract_version=2, calendar=None, data_frequency="1ns")
    spec = LabelSpec.from_mapping(
        {
            "artifact_id": "outcomes",
            "schema": {"label_col": "fwd_ns", "exit_time_col": "exit"},
            "definition": _wire_definition(feed),
        }
    )
    frame.write_parquet(tmp_path / "outcomes.parquet")
    (tmp_path / "outcomes.json").write_text(json.dumps(spec.to_dict()))
    restored = pl.read_parquet(tmp_path / "outcomes.parquet")
    restored_spec = LabelSpec.from_mapping(json.loads((tmp_path / "outcomes.json").read_text()))
    assert (
        restored[restored_spec.schema.exit_time_col].cast(pl.Int64)[0]
        - restored["timestamp"].cast(pl.Int64)[0]
        == 1
    )
    assert restored_spec.to_dict()["definition"] == spec.to_dict()["definition"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("window_unit", "days"),
        ("endpoint_method", "closest"),
        ("return_method", "percent"),
        ("missing_price_policy", "fill"),
        ("index_mode", "infer"),
        ("tail_policy", "fill"),
    ],
)
def test_outcome_definition_rejects_unknown_method_instead_of_silently_dropping_it(field, value):
    definition = _wire_definition(FeedSpec(time_contract_version=2, calendar=None))
    definition[field] = value
    with pytest.raises(ValueError, match=field):
        LabelSpec.from_mapping({"artifact_id": "invalid-outcomes", "definition": definition})


def test_outcome_definition_cannot_drop_one_original_mask_role():
    definition = _wire_definition(FeedSpec(time_contract_version=2, calendar=None))
    definition.pop("original_row_col")
    with pytest.raises(ValueError, match="original row and price"):
        LabelSpec.from_mapping({"artifact_id": "incomplete-mask-roles", "definition": definition})
