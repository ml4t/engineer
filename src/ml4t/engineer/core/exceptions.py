"""Custom exceptions for ML4T Engineer.

Provides a comprehensive exception hierarchy for systematic error handling
across the ML4T Engineer library. All exceptions preserve context information and
provide actionable error messages.

Exception Hierarchy (Flat - D06 Pattern):
    QuantLabTAError (base)
    ├── ConfigurationError      # Configuration and setup errors
    ├── ValidationError         # Input validation failures (also inherits ValueError)
    ├── InvalidParameterError   # Invalid parameters to indicators
    ├── DataValidationError     # Data validation failures
    ├── DataSchemaError         # Schema validation failures
    ├── InsufficientDataError   # Insufficient data for calculation
    ├── ComputationError        # Calculation and numerical errors
    ├── DataError               # Data access and format errors
    └── IntegrationError        # External library integration errors

Example:
    >>> from ml4t.engineer.core.exceptions import ValidationError
    >>> try:
    ...     validate_data(data)
    ... except ValidationError as e:
    ...     print(f"Validation failed: {e}")
    ...     print(f"Context: {e.context}")
"""

from __future__ import annotations

import warnings
from typing import Any


class QuantLabTAError(Exception):
    """
    Base exception class for all ML4T Engineer errors.

    All exceptions inherit from this base class, providing
    consistent error handling and context preservation.

    Attributes:
        message: Human-readable error description
        context: Additional error context (dict)
        cause: Original exception if error was wrapped

    Example:
        >>> raise QuantLabTAError(
        ...     "Operation failed",
        ...     context={"operation": "compute_rsi", "reason": "insufficient_data"}
        ... )
    """

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize error.

        Args:
            message: Error description
            context: Additional error context
            cause: Original exception (for error chaining)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """Format error message with context."""
        parts = [self.message]

        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.cause:
            parts.append(f"\nCaused by: {type(self.cause).__name__}: {self.cause}")

        return "".join(parts)

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"context={self.context!r}, "
            f"cause={self.cause!r})"
        )


# =============================================================================
# First-Level Exception Classes (Flat Hierarchy - D06 Pattern)
# =============================================================================


class ConfigurationError(QuantLabTAError):
    """
    Configuration and setup errors.

    Raised when:
    - Invalid configuration values
    - Missing required configuration
    - Incompatible settings
    - Setup/initialization failures

    Example:
        >>> from ml4t.engineer.core.exceptions import ConfigurationError
        >>> raise ConfigurationError(
        ...     "Invalid pipeline configuration",
        ...     context={"setting": "lookback", "value": -1}
        ... )
    """

    pass


class ValidationError(QuantLabTAError, ValueError):
    """
    Input validation failures.

    Inherits from both QuantLabTAError (for library-specific catching)
    and ValueError (for standard Python parameter error handling).

    Raised when:
    - Required columns missing
    - Data type mismatches
    - Value constraints violated
    - Schema validation failures

    Example:
        >>> from ml4t.engineer.core.exceptions import ValidationError
        >>> try:
        ...     validate_ohlcv(data)
        ... except ValidationError as e:
        ...     print(f"Validation failed: {e}")
    """

    pass


class InvalidParameterError(QuantLabTAError, ValueError):
    """
    Invalid parameters provided to indicators.

    Raised when indicator parameters fail validation:
    - Period out of valid range
    - Invalid multiplier values
    - Incompatible parameter combinations

    Note:
        Also inherits from ValueError for compatibility with parameter
        validation libraries and pytest.raises(ValueError) patterns.

    Example:
        >>> from ml4t.engineer.core.exceptions import InvalidParameterError
        >>> raise InvalidParameterError(
        ...     "period must be >= 1",
        ...     context={"parameter": "period", "value": -1}
        ... )
    """

    pass


class DataValidationError(QuantLabTAError):
    """
    Data validation failures.

    Raised when input data fails validation:
    - Missing required columns
    - Invalid data types in columns
    - Empty or insufficient data
    - Data integrity issues

    Example:
        >>> from ml4t.engineer.core.exceptions import DataValidationError
        >>> raise DataValidationError(
        ...     "Column 'close' not found in data",
        ...     context={"required": "close", "available": ["open", "high", "low"]}
        ... )
    """

    pass


class DataSchemaError(QuantLabTAError):
    """
    Schema validation failures.

    Raised when data doesn't match expected schema:
    - Wrong column types
    - Missing required columns
    - Invalid column names
    - Schema mismatch

    Example:
        >>> from ml4t.engineer.core.exceptions import DataSchemaError
        >>> raise DataSchemaError(
        ...     "Expected Float64 for 'close', got String",
        ...     context={"column": "close", "expected": "Float64", "actual": "String"}
        ... )
    """

    pass


class InsufficientDataError(QuantLabTAError):
    """
    Insufficient data for calculation.

    Raised when:
    - Not enough rows for lookback period
    - Empty data provided
    - Data shorter than minimum required window

    Example:
        >>> from ml4t.engineer.core.exceptions import InsufficientDataError
        >>> raise InsufficientDataError(
        ...     "RSI requires at least 14 rows, got 10",
        ...     context={"required": 14, "actual": 10, "indicator": "RSI"}
        ... )
    """

    pass


class ComputationError(QuantLabTAError):
    """
    Calculation and numerical errors.

    Raised when:
    - Numerical instability (division by zero, overflow)
    - Algorithm convergence failures
    - Invalid mathematical operations
    - Numerical precision issues

    Example:
        >>> from ml4t.engineer.core.exceptions import ComputationError
        >>> raise ComputationError(
        ...     "Division by zero in standard deviation",
        ...     context={"operation": "stddev", "variance": 0.0}
        ... )
    """

    pass


class DataError(QuantLabTAError):
    """
    Data access and format errors.

    Raised when:
    - Data cannot be loaded
    - Unexpected data format
    - Missing expected data
    - Data corruption

    Example:
        >>> from ml4t.engineer.core.exceptions import DataError
        >>> raise DataError(
        ...     "Failed to load feature data",
        ...     context={"path": "features.parquet", "error": "file not found"}
        ... )
    """

    pass


class IntegrationError(QuantLabTAError):
    """
    External library integration errors.

    Raised when:
    - TA-Lib integration fails
    - Numba compilation errors
    - External API errors
    - Version compatibility issues

    Example:
        >>> from ml4t.engineer.core.exceptions import IntegrationError
        >>> raise IntegrationError(
        ...     "TA-Lib not available",
        ...     context={"library": "talib", "required_for": "RSI validation"}
        ... )
    """

    pass


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


# Primary alias for historical naming
TechnicalAnalysisError = QuantLabTAError


class _DeprecatedIndicatorError(QuantLabTAError):
    """Deprecated intermediate class - use QuantLabTAError or specific errors."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        warnings.warn(
            "IndicatorError is deprecated. Use QuantLabTAError or a specific "
            "exception type (InsufficientDataError, ComputationError) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(message, context, cause)


# IndicatorError was an intermediate class - deprecated but kept for backward compat
IndicatorError = _DeprecatedIndicatorError

# InvalidArgumentError is an alias for InvalidParameterError
InvalidArgumentError = InvalidParameterError

# ImplementationNotAvailableError maps to IntegrationError
ImplementationNotAvailableError = IntegrationError


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base exception
    "QuantLabTAError",
    # First-level exceptions (flat hierarchy)
    "ConfigurationError",
    "ValidationError",
    "InvalidParameterError",
    "DataValidationError",
    "DataSchemaError",
    "InsufficientDataError",
    "ComputationError",
    "DataError",
    "IntegrationError",
    # Backward compatibility aliases
    "TechnicalAnalysisError",
    "IndicatorError",  # Deprecated
    "InvalidArgumentError",
    "ImplementationNotAvailableError",
]
