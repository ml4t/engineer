"""ml4t-engineer - A Financial Machine Learning Feature Engineering Library.

ml4t-engineer is a comprehensive FML stack designed for correctness, reproducibility,
and performance. It provides tools for feature engineering, labeling, and validation
of financial machine learning models.
"""

__version__ = "0.3.0"

from . import (
    core,
    dataset,
    discovery,
    features,
    labeling,
    outcome,
    pipeline,
    preprocessing,
    relationships,
    store,
    validation,
    visualization,
)
from .api import compute_features
from .dataset import (
    DatasetInfo,
    FoldResult,
    MLDatasetBuilder,
    create_dataset_builder,
)
from .discovery import FeatureCatalog
from .discovery.catalog import features as feature_catalog
from .preprocessing import (
    BaseScaler,
    MinMaxScaler,
    NotFittedError,
    PreprocessingPipeline,
    Preprocessor,
    RobustScaler,
    StandardScaler,
    TransformType,
)

__all__ = [
    # Main API
    "compute_features",
    # Feature Discovery (discoverability API)
    "FeatureCatalog",
    "feature_catalog",
    # Dataset builder (leakage-safe train/test preparation)
    "MLDatasetBuilder",
    "create_dataset_builder",
    "FoldResult",
    "DatasetInfo",
    # Preprocessing (leakage-safe scalers)
    "Preprocessor",
    "PreprocessingPipeline",
    "TransformType",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "BaseScaler",
    "NotFittedError",
    # Submodules
    "core",
    "dataset",
    "discovery",
    "features",
    "labeling",
    "outcome",
    "pipeline",
    "preprocessing",
    "relationships",
    "store",
    "validation",
    "visualization",
]
