"""ML4T Engineer Configuration System.

This module provides Pydantic v2 configuration schemas for feature engineering:

- **Labeling**: Triple barrier, ATR barrier, fixed horizon, trend scanning
- **Preprocessing**: Standard, MinMax, Robust scalers with create_scaler()
- **Data Contract**: Schema validation for input data
- **Experiment**: Experiment configuration and serialization

Note:
    Feature evaluation configs (StationarityConfig, ACFConfig, etc.) have moved
    to ``ml4t-diagnostic``. Install with: ``pip install ml4t-diagnostic``
"""

from ml4t.engineer.config.base import (
    BaseConfig,
    ComputationalConfig,
    StatisticalTestConfig,
)
from ml4t.engineer.config.data_contract import DataContractConfig
from ml4t.engineer.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
    save_experiment_config,
)
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.config.preprocessing_config import PreprocessingConfig

__all__ = [
    # Base configs
    "BaseConfig",
    "StatisticalTestConfig",
    "ComputationalConfig",
    # Labeling and preprocessing configs
    "LabelingConfig",
    "DataContractConfig",
    "PreprocessingConfig",
    # Experiment config loading
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
]
