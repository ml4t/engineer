"""Core module for ml4t-engineer.

Contains base types, exceptions, schemas, and registry.
"""

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import (
    ComputationError,
    DataSchemaError,
    DataValidationError,
    ImplementationNotAvailableError,
    IndicatorError,
    InsufficientDataError,
    InvalidArgumentError,
    InvalidParameterError,
    ML4TEngineerError,
    TechnicalAnalysisError,
    ValidationError,
)
from ml4t.engineer.core.registry import (
    FeatureMetadata,
    FeatureRegistry,
    get_registry,
)
from ml4t.engineer.core.schemas import (
    EXTENDED_OHLCV_SCHEMA,
    FEATURE_SCHEMA,
    LABELED_DATA_SCHEMA,
    OHLCV_SCHEMA,
    validate_ohlcv_schema,
    validate_schema,
)
from ml4t.engineer.core.types import (
    AssetId,
    FeatureArray,
    FeatureValue,
    Frequency,
    Implementation,
    OrderId,
    Price,
    Quantity,
    StepConfig,
    StepName,
    Symbol,
    TimeUnit,
    Timestamp,
)
from ml4t.engineer.core.validation import (
    validate_column_exists,
    validate_lag,
    validate_list_length,
    validate_numeric_column,
    validate_percentage,
    validate_period,
    validate_positive,
    validate_probability,
    validate_threshold,
    validate_window,
)

__all__ = [
    # Registry
    "feature",
    "FeatureMetadata",
    "FeatureRegistry",
    "get_registry",
    # Schemas
    "EXTENDED_OHLCV_SCHEMA",
    "FEATURE_SCHEMA",
    "LABELED_DATA_SCHEMA",
    "OHLCV_SCHEMA",
    "validate_ohlcv_schema",
    "validate_schema",
    # Types
    "AssetId",
    "FeatureArray",
    "FeatureValue",
    "Frequency",
    "Implementation",
    "OrderId",
    "Price",
    "Quantity",
    "StepConfig",
    "StepName",
    "Symbol",
    "TimeUnit",
    "Timestamp",
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
    "TechnicalAnalysisError",
    "ValidationError",
    # Validation
    "validate_column_exists",
    "validate_lag",
    "validate_list_length",
    "validate_numeric_column",
    "validate_percentage",
    "validate_period",
    "validate_positive",
    "validate_probability",
    "validate_threshold",
    "validate_window",
]
