# mypy: disable-error-code="misc,no-any-return"
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.config.preprocessing_config import PreprocessingConfig


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
        Validate config values using Pydantic validation

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
    will have None values in the returned ExperimentConfig.

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

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Extract features section
    features: list[dict[str, Any]] = []
    if "features" in raw:
        features_raw = raw["features"]
        if isinstance(features_raw, list):
            features = features_raw

    # Extract and parse labeling section
    labeling: LabelingConfig | None = None
    if "labeling" in raw:
        labeling_raw = raw["labeling"]
        if isinstance(labeling_raw, dict):
            if validate:
                labeling = LabelingConfig(**labeling_raw)
            else:
                labeling = LabelingConfig.model_construct(**labeling_raw)

    # Extract and parse preprocessing section
    preprocessing: PreprocessingConfig | None = None
    if "preprocessing" in raw:
        preprocessing_raw = raw["preprocessing"]
        if isinstance(preprocessing_raw, dict):
            if validate:
                preprocessing = PreprocessingConfig(**preprocessing_raw)
            else:
                preprocessing = PreprocessingConfig.model_construct(**preprocessing_raw)

    return ExperimentConfig(
        features=features,
        labeling=labeling,
        preprocessing=preprocessing,
        raw=raw,
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

    output: dict[str, Any] = {}

    if config.features:
        output["features"] = config.features

    if config.labeling is not None:
        output["labeling"] = config.labeling.model_dump(
            exclude_defaults=not include_defaults,
            exclude_none=True,
        )

    if config.preprocessing is not None:
        output["preprocessing"] = config.preprocessing.model_dump(
            exclude_defaults=not include_defaults,
            exclude_none=True,
        )

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
]
