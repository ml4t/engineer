"""ML4T Engineer Configuration System.

This module provides Pydantic v2 configuration schemas for feature engineering:

- **Feature Diagnostics**: Stationarity, ACF, volatility, distribution
- **Feature-Outcome Analysis**: IC, classification, thresholds, ML diagnostics

Examples:
    Quick start with defaults:

    >>> from ml4t.engineer.config import ModuleCConfig
    >>> config = ModuleCConfig()

    Custom configuration:

    >>> config = ModuleCConfig(
    ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21])
    ... )

    Load from YAML:

    >>> config = ModuleCConfig.from_yaml("config.yaml")
"""

from ml4t.engineer.config.base import BaseConfig, ComputationalConfig, StatisticalTestConfig
from ml4t.engineer.config.feature_config import (
    ACFConfig,
    BinaryClassificationConfig,
    ClusteringConfig,
    CorrelationConfig,
    DistributionConfig,
    FeatureEvaluatorConfig,
    ICConfig,
    MLDiagnosticsConfig,
    ModuleAConfig,
    ModuleBConfig,
    ModuleCConfig,
    PCAConfig,
    RedundancyConfig,
    StationarityConfig,
    ThresholdAnalysisConfig,
    VolatilityConfig,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "StatisticalTestConfig",
    "ComputationalConfig",
    # Feature evaluation
    "FeatureEvaluatorConfig",
    "ModuleAConfig",
    "ModuleBConfig",
    "ModuleCConfig",
    "StationarityConfig",
    "ACFConfig",
    "VolatilityConfig",
    "DistributionConfig",
    "CorrelationConfig",
    "PCAConfig",
    "ClusteringConfig",
    "RedundancyConfig",
    "ICConfig",
    "BinaryClassificationConfig",
    "ThresholdAnalysisConfig",
    "MLDiagnosticsConfig",
]
