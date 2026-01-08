"""ML4T Engineer Configuration System.

This module provides Pydantic v2 configuration schemas for feature engineering:

- **Labeling**: Triple barrier, ATR barrier, fixed horizon, trend scanning
- **Preprocessing**: Standard, MinMax, Robust scalers with create_scaler()
- **Feature Diagnostics**: Stationarity, ACF, volatility, distribution
- **Cross-Feature Analysis**: Correlation, PCA, clustering, redundancy
- **Feature-Outcome Analysis**: IC, classification, thresholds, ML diagnostics

D06 Pattern Support:
    This module supports the D06 configuration pattern with single-level nesting:
    - Primary configs use `*Config` naming (e.g., `EngineerConfig`)
    - Nested settings use `*Settings` naming (e.g., `StationaritySettings`)
    - Both patterns are fully supported via aliases

Examples:
    Quick start with defaults:

    >>> from ml4t.engineer.config import EngineerConfig
    >>> config = EngineerConfig()

    Use presets:

    >>> config = EngineerConfig.for_quick_analysis()
    >>> config = EngineerConfig.for_research()
    >>> config = EngineerConfig.for_production()

    Custom configuration:

    >>> config = EngineerConfig(
    ...     module_a=ModuleAConfig(
    ...         stationarity=StationaritySettings(significance_level=0.01)
    ...     )
    ... )

    Load from YAML:

    >>> config = EngineerConfig.from_yaml("config.yaml")
"""

from ml4t.engineer.config.base import (
    BaseConfig,
    ComputationalConfig,
    StatisticalTestConfig,
)
from ml4t.engineer.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
    save_experiment_config,
)
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
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.config.preprocessing_config import PreprocessingConfig

# =============================================================================
# D06 Pattern Aliases - Settings Classes
# =============================================================================
# These aliases provide compatibility with the diagnostic library's D06 pattern
# where nested configuration classes use *Settings naming.

StationaritySettings = StationarityConfig
ACFSettings = ACFConfig
VolatilitySettings = VolatilityConfig
DistributionSettings = DistributionConfig
CorrelationSettings = CorrelationConfig
PCASettings = PCAConfig
ClusteringSettings = ClusteringConfig
RedundancySettings = RedundancyConfig
ICSettings = ICConfig
BinaryClassificationSettings = BinaryClassificationConfig
ThresholdAnalysisSettings = ThresholdAnalysisConfig
MLDiagnosticsSettings = MLDiagnosticsConfig

# =============================================================================
# D06 Pattern Aliases - Top-Level Configs
# =============================================================================

# Primary alias - EngineerConfig is the D06-style name
EngineerConfig = FeatureEvaluatorConfig

# DiagnosticConfig alias for symmetry with ml4t.diagnostic
DiagnosticConfig = FeatureEvaluatorConfig

# RuntimeConfig is the D06-style name for computational settings
RuntimeConfig = ComputationalConfig

__all__ = [
    # Base configs
    "BaseConfig",
    "StatisticalTestConfig",
    "RuntimeConfig",
    "ComputationalConfig",  # Backward compatibility
    # Labeling and preprocessing configs
    "LabelingConfig",
    "PreprocessingConfig",
    # Experiment config loading
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    # Primary config (D06 pattern)
    "EngineerConfig",
    "DiagnosticConfig",  # Alias for symmetry
    # Feature evaluation (original names)
    "FeatureEvaluatorConfig",
    "ModuleAConfig",
    "ModuleBConfig",
    "ModuleCConfig",
    # Settings classes (D06 pattern - *Settings naming)
    "StationaritySettings",
    "ACFSettings",
    "VolatilitySettings",
    "DistributionSettings",
    "CorrelationSettings",
    "PCASettings",
    "ClusteringSettings",
    "RedundancySettings",
    "ICSettings",
    "BinaryClassificationSettings",
    "ThresholdAnalysisSettings",
    "MLDiagnosticsSettings",
    # Original config names (backward compatibility)
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
