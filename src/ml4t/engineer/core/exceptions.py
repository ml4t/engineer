"""Custom exceptions for ML4T Engineer."""

from typing import Any


class ML4TEngineerError(Exception):
    """
    Base exception class for all ML4T Engineer errors.

    All exceptions inherit from this base class, providing
    consistent error handling and context preservation.

    Attributes:
        message: Human-readable error description
        context: Additional error context (dict)
        cause: Original exception if error was wrapped
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


class ValidationError(ML4TEngineerError, ValueError):
    """
    Raised when input validation fails.

    Inherits from both ML4TEngineerError (for library-specific catching)
    and ValueError (for standard Python parameter error handling).

    Raised when:
    - Required columns missing
    - Data type mismatches
    - Value constraints violated
    - Schema validation failures
    """

    pass


class IndicatorError(ML4TEngineerError):
    """Raised when indicator calculation fails."""

    pass


class InsufficientDataError(IndicatorError):
    """Raised when insufficient data is provided for calculation."""

    pass


class InvalidParameterError(ValidationError):
    """Raised when invalid parameters are provided to indicators."""

    pass


class DataValidationError(ValidationError):
    """Raised when data validation fails."""

    pass


class DataSchemaError(ValidationError):
    """Raised when data doesn't match expected schema."""

    pass


class ImplementationNotAvailableError(ML4TEngineerError):
    """Raised when a specific implementation is not available."""

    pass


class ComputationError(ML4TEngineerError):
    """
    Raised when computation fails during diagnostic tests.

    Raised when:
    - Numerical instability (division by zero, overflow)
    - Insufficient data for calculation
    - Algorithm convergence failures
    - Invalid mathematical operations
    """

    pass


# Aliases for backward compatibility
TechnicalAnalysisError = ML4TEngineerError
InvalidArgumentError = InvalidParameterError
