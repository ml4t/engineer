"""ml4t-engineer - A Financial Machine Learning Feature Engineering Library.

ml4t-engineer is a comprehensive FML stack designed for correctness, reproducibility,
and performance. It provides tools for feature engineering, labeling, and validation
of financial machine learning models.

Core API
--------
compute_features : Compute features from configuration
list_features : List available features
list_categories : List feature categories
describe_feature : Get feature metadata

Modules
-------
features : Technical indicators and feature engineering
labeling : Triple-barrier and other labeling methods
bars : Alternative bar sampling (volume, tick, imbalance)
pipeline : DAG-based feature orchestration
preprocessing : Scalers and preprocessing utilities
dataset : ML dataset building utilities
store : Feature storage (DuckDB)
validation : Cross-validation utilities

Example
-------
>>> import polars as pl
>>> from ml4t.engineer import compute_features
>>>
>>> # Load OHLCV data
>>> df = pl.DataFrame({
...     "open": [100.0, 101.0, 102.0],
...     "high": [102.0, 103.0, 104.0],
...     "low": [99.0, 100.0, 101.0],
...     "close": [101.0, 102.0, 103.0],
...     "volume": [1000, 1100, 1200],
... })
>>>
>>> # Compute features
>>> result = compute_features(df, ["rsi_14", "sma_20"])
"""

__version__ = "0.1.0"

from ml4t.engineer.api import (
    compute_features,
    describe_feature,
    list_categories,
    list_features,
)
from ml4t.engineer.core import (
    # Registry
    FeatureMetadata,
    FeatureRegistry,
    feature,
    get_registry,
    # Exceptions
    ComputationError,
    DataSchemaError,
    DataValidationError,
    ImplementationNotAvailableError,
    IndicatorError,
    InsufficientDataError,
    InvalidArgumentError,
    InvalidParameterError,
    ML4TEngineerError,
    ValidationError,
    # Schemas
    OHLCV_SCHEMA,
    validate_ohlcv_schema,
    # Types
    FeatureArray,
    FeatureValue,
    Implementation,
    # Validation
    validate_period,
    validate_window,
)
from ml4t.engineer.dataset import (
    DatasetInfo,
    FoldResult,
    MLDatasetBuilder,
    create_dataset_builder,
)
from ml4t.engineer.preprocessing import (
    BaseScaler,
    MinMaxScaler,
    NotFittedError,
    Preprocessor,
    RobustScaler,
    StandardScaler,
)

# Import submodules
from ml4t.engineer import (
    bars,
    config,
    core,
    dataset,
    features,
    labeling,
    logging,
    outcome,
    pipeline,
    preprocessing,
    relationships,
    selection,
    store,
    validation,
    visualization,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "compute_features",
    "list_features",
    "list_categories",
    "describe_feature",
    # Registry
    "feature",
    "FeatureMetadata",
    "FeatureRegistry",
    "get_registry",
    # Dataset builder
    "MLDatasetBuilder",
    "create_dataset_builder",
    "FoldResult",
    "DatasetInfo",
    # Preprocessing
    "Preprocessor",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "BaseScaler",
    "NotFittedError",
    # Exceptions
    "ComputationError",
    "DataSchemaError",
    "DataValidationError",
    "ImplementationNotAvailableError",
    "IndicatorError",
    "InsufficientDataError",
    "InvalidArgumentError",
    "InvalidParameterError",
    "ML4TEngineerError",
    "ValidationError",
    # Schemas
    "OHLCV_SCHEMA",
    "validate_ohlcv_schema",
    # Types
    "FeatureArray",
    "FeatureValue",
    "Implementation",
    # Validation
    "validate_period",
    "validate_window",
    # Submodules
    "bars",
    "config",
    "core",
    "dataset",
    "features",
    "labeling",
    "logging",
    "outcome",
    "pipeline",
    "preprocessing",
    "relationships",
    "selection",
    "store",
    "validation",
    "visualization",
]
