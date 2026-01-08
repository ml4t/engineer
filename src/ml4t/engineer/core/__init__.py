"""Core module for ml4t-engineer.

Contains base types, exceptions, schemas, calendar utilities, and registry.
"""

from ml4t.engineer.core.calendars import (
    CryptoCalendar,
    EquityCalendar,
    ExchangeCalendar,
)
from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import (
    ComputationError,
    # First-level exceptions (flat hierarchy)
    ConfigurationError,
    DataError,
    DataSchemaError,
    DataValidationError,
    ImplementationNotAvailableError,
    IndicatorError,  # Deprecated
    InsufficientDataError,
    IntegrationError,
    InvalidArgumentError,
    InvalidParameterError,
    # Base exception
    QuantLabTAError,
    # Backward compatibility aliases
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
    Timestamp,
    TimeUnit,
)
from ml4t.engineer.core.validation import (
    validate_lag,
    validate_list_length,
    validate_period,
    validate_positive,
    validate_probability,
    validate_threshold,
    validate_window,
)

__all__ = [
    # Schemas
    "EXTENDED_OHLCV_SCHEMA",
    "FEATURE_SCHEMA",
    "LABELED_DATA_SCHEMA",
    "OHLCV_SCHEMA",
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
    "Timestamp",
    "TimeUnit",
    # Calendars
    "CryptoCalendar",
    "EquityCalendar",
    "ExchangeCalendar",
    # Registry
    "feature",
    "FeatureMetadata",
    "FeatureRegistry",
    "get_registry",
    # Exceptions - Base
    "QuantLabTAError",
    # Exceptions - First-level (flat hierarchy)
    "ConfigurationError",
    "ValidationError",
    "InvalidParameterError",
    "DataValidationError",
    "DataSchemaError",
    "InsufficientDataError",
    "ComputationError",
    "DataError",
    "IntegrationError",
    # Exceptions - Backward compatibility aliases
    "TechnicalAnalysisError",
    "IndicatorError",  # Deprecated
    "InvalidArgumentError",
    "ImplementationNotAvailableError",
    # Validation
    "validate_lag",
    "validate_list_length",
    "validate_period",
    "validate_positive",
    "validate_probability",
    "validate_threshold",
    "validate_window",
]
