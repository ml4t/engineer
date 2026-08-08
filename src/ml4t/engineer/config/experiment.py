"""Experiment configuration loading utilities.

This module provides helpers for loading complete experiment configurations
from YAML files, returning typed config objects for reproducible ML pipelines.

Examples
--------
>>> from ml4t.engineer.config import load_experiment_config
>>>
>>> # Load all configs from a single YAML file
>>> configs = load_experiment_config("experiment.yaml")
>>>
>>> # Access typed configs
>>> label_config = configs.labeling  # LabelingConfig
>>> prep_config = configs.preprocessing  # PreprocessingConfig
>>> feature_specs = configs.features  # list[dict]
>>>
>>> # Use with compute_features and labeling functions
>>> df = compute_features(df, feature_specs)
>>> labeled = triple_barrier_labels(df, config=label_config)
>>> scaler = prep_config.create_scaler()
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ml4t.engineer.config.base import (
    _decode_portable_timedelta,
    _encode_portable_timedelta,
)
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.config.preprocessing_config import PreprocessingConfig

_SCHEMA_VERSION = 1
_RECOGNIZED_SECTIONS = frozenset({"schema_version", "features", "labeling", "preprocessing"})


class _FeatureConfiguration(BaseModel):
    """Validated feature request stored in an experiment document."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    output: str | None = Field(default=None, min_length=1)


class _ExperimentDocument(BaseModel):
    """Typed schema for recognized experiment sections."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal[1] = _SCHEMA_VERSION
    features: list[_FeatureConfiguration] = Field(default_factory=list)
    labeling: LabelingConfig | None = None
    preprocessing: PreprocessingConfig | None = None


@dataclass
class ExperimentConfig:
    """Container for experiment configuration components.

    Holds typed configuration objects for all experiment components,
    loaded from a single YAML file.

    Attributes
    ----------
    features : list[dict]
        Feature specifications for compute_features()
    labeling : LabelingConfig | None
        Labeling configuration (triple barrier, ATR, etc.)
    preprocessing : PreprocessingConfig | None
        Preprocessing/scaler configuration
    raw : dict
        Raw YAML content for any custom sections
    """

    features: list[dict[str, Any]] = field(default_factory=list)
    labeling: LabelingConfig | None = None
    preprocessing: PreprocessingConfig | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def load_experiment_config(
    path: str | Path,
    *,
    validate: bool = True,
) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Parses a YAML file containing feature, labeling, and preprocessing
    configurations, returning typed Pydantic config objects.

    Parameters
    ----------
    path : str | Path
        Path to YAML configuration file
    validate : bool, default True
        Validate recognized field values using Pydantic. Root and section shapes,
        schema versions, and portable value encodings are always validated.

    Returns
    -------
    ExperimentConfig
        Container with typed configuration objects:
        - features: list[dict] for compute_features()
        - labeling: LabelingConfig for labeling functions
        - preprocessing: PreprocessingConfig for scalers
        - raw: dict with full YAML content

    Examples
    --------
    >>> # experiment.yaml:
    >>> # features:
    >>> #   - name: rsi
    >>> #     params: {period: 14}
    >>> #   - name: macd
    >>> # labeling:
    >>> #   method: triple_barrier
    >>> #   upper_barrier: 0.02
    >>> #   lower_barrier: 0.01
    >>> # preprocessing:
    >>> #   scaler: robust
    >>>
    >>> configs = load_experiment_config("experiment.yaml")
    >>> df = compute_features(df, configs.features)
    >>> labeled = triple_barrier_labels(df, config=configs.labeling)
    >>> scaler = configs.preprocessing.create_scaler()

    Notes
    -----
    The YAML file can contain any subset of sections. Missing sections
    will have None values in the returned ExperimentConfig. Unknown top-level
    sections are preserved in ``raw``.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist
    yaml.YAMLError
        If the YAML is malformed
    pydantic.ValidationError
        If validate=True and config values are invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    raw = _require_document_mapping(raw)
    _validate_document_structure(raw)
    decoded = _decode_document_values(raw)

    if validate:
        document = _ExperimentDocument.model_validate(decoded)
        labeling = document.labeling
        preprocessing = document.preprocessing
    else:
        labeling_raw = decoded.get("labeling")
        labeling = (
            LabelingConfig.model_construct(**labeling_raw) if labeling_raw is not None else None
        )
        preprocessing_raw = decoded.get("preprocessing")
        preprocessing = (
            PreprocessingConfig.model_construct(**preprocessing_raw)
            if preprocessing_raw is not None
            else None
        )

    return ExperimentConfig(
        features=deepcopy(raw.get("features", [])),
        labeling=labeling,
        preprocessing=preprocessing,
        raw=deepcopy(raw),
    )


def save_experiment_config(
    config: ExperimentConfig,
    path: str | Path,
    *,
    include_defaults: bool = False,
) -> None:
    """Save experiment configuration to YAML file.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to save
    path : str | Path
        Output file path
    include_defaults : bool, default False
        Include fields with default values in output

    Notes
    -----
    Typed ``features``, ``labeling``, and ``preprocessing`` values are authoritative.
    Other top-level sections from ``config.raw`` are preserved.

    Examples
    --------
    >>> config = ExperimentConfig(
    ...     features=[{"name": "rsi", "params": {"period": 14}}],
    ...     labeling=LabelingConfig.triple_barrier(upper_barrier=0.02),
    ...     preprocessing=PreprocessingConfig.robust(),
    ... )
    >>> save_experiment_config(config, "experiment.yaml")
    """
    path = Path(path)

    if not isinstance(config.raw, dict):
        raise ValueError("ExperimentConfig.raw must be a mapping")

    custom = {
        key: deepcopy(value) for key, value in config.raw.items() if key not in _RECOGNIZED_SECTIONS
    }
    output: dict[str, Any] = {"schema_version": _SCHEMA_VERSION, **custom}

    if config.features:
        output["features"] = deepcopy(config.features)

    if config.labeling is not None:
        labeling = config.labeling.model_dump(
            exclude_defaults=not include_defaults,
            exclude_none=True,
        )
        output["labeling"] = _encode_labeling_values(labeling)

    if config.preprocessing is not None:
        output["preprocessing"] = config.preprocessing.model_dump(
            exclude_defaults=not include_defaults,
            exclude_none=True,
            mode="json",
        )

    output = _tuples_to_lists(output)
    _validate_document_structure(output)
    _ExperimentDocument.model_validate(_decode_document_values(output))

    serialized = yaml.safe_dump(output, default_flow_style=False, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _require_document_mapping(raw: Any) -> dict[str, Any]:
    """Require the YAML root to be a string-keyed mapping."""
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment document root must be a mapping, got {type(raw).__name__}")
    non_string_keys = [key for key in raw if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(f"Experiment section names must be strings, got: {non_string_keys!r}")
    return raw


def _validate_document_structure(raw: dict[str, Any]) -> None:
    """Validate format structure even when field-value validation is disabled."""
    version = raw.get("schema_version", _SCHEMA_VERSION)
    if version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported experiment schema_version {version!r}; expected {_SCHEMA_VERSION}"
        )

    for section in raw:
        if section in _RECOGNIZED_SECTIONS:
            continue
        suggestion = get_close_matches(
            section,
            _RECOGNIZED_SECTIONS,
            n=1,
            cutoff=0.8,
        )
        if suggestion:
            raise ValueError(
                f"Unknown experiment section {section!r}; did you mean {suggestion[0]!r}?"
            )

    features = raw.get("features", [])
    if not isinstance(features, list):
        raise ValueError(
            f"Experiment section 'features' must be a list, got {type(features).__name__}"
        )
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(
                f"Experiment section 'features.{index}' must be a mapping, "
                f"got {type(feature).__name__}"
            )

    for section in ("labeling", "preprocessing"):
        value = raw.get(section)
        if value is not None and not isinstance(value, dict):
            raise ValueError(
                f"Experiment section {section!r} must be a mapping, got {type(value).__name__}"
            )


def _encode_labeling_values(labeling: dict[str, Any]) -> dict[str, Any]:
    """Encode non-primitive labeling values in the versioned YAML schema."""
    if "max_holding_period" in labeling:
        labeling["max_holding_period"] = _encode_portable_timedelta(labeling["max_holding_period"])
    return labeling


def _decode_document_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Decode versioned values before typed validation."""
    decoded = deepcopy(raw)
    labeling = decoded.get("labeling")
    if not isinstance(labeling, dict):
        return decoded

    holding_period = labeling.get("max_holding_period")
    if not isinstance(holding_period, dict):
        return decoded
    labeling["max_holding_period"] = _decode_portable_timedelta(
        holding_period,
        field_name="labeling.max_holding_period",
    )
    return decoded


def _tuples_to_lists(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML-safe serialization."""
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [_tuples_to_lists(item) for item in obj]
    return obj


__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
]
