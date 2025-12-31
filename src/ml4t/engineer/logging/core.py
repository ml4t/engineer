# mypy: disable-error-code="arg-type,call-arg,return-value,assignment,operator"
"""Core logging functionality for ml4t.engineer.

Provides structured logging for feature engineering operations, performance tracking,
data quality monitoring, and error reporting with rich context information.
"""

import functools
import logging
import sys
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

# Configure logging format
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class QFeaturesFormatter(logging.Formatter):
    """Custom formatter for ml4t.engineer logging with enhanced context."""

    def __init__(self) -> None:
        super().__init__(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        # Add ml4t.engineer-specific context
        if hasattr(record, "feature_name"):
            record.msg = f"[{record.feature_name}] {record.msg}"

        if hasattr(record, "data_shape"):
            record.msg = f"{record.msg} (shape: {record.data_shape})"

        if hasattr(record, "performance_ms"):
            record.msg = f"{record.msg} ({record.performance_ms:.2f}ms)"

        return super().format(record)


class FeatureLogger:
    """Structured logger for feature engineering operations.

    Provides contextual logging for feature calculations, data validation,
    performance tracking, and error reporting.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize feature logger.

        Parameters
        ----------
        name : str
            Logger name (typically feature module name)
        level : int
            Logging level (default: INFO)
        """
        self.logger = logging.getLogger(f"mlquant.features.{name}")
        self.logger.setLevel(level)

        # Add custom formatter if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(QFeaturesFormatter())
            self.logger.addHandler(handler)

        self.name = name

    def log_feature_start(
        self,
        feature_name: str,
        data_shape: tuple[int, ...],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log the start of feature calculation.

        Parameters
        ----------
        feature_name : str
            Name of the feature being calculated
        data_shape : tuple
            Shape of input data
        params : dict, optional
            Feature parameters
        """
        msg = "Starting calculation"
        extra = {
            "feature_name": feature_name,
            "data_shape": data_shape,
        }

        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            msg = f"{msg} with parameters: {param_str}"

        self.logger.info(msg, extra=extra)

    def log_feature_complete(
        self,
        feature_name: str,
        output_shape: tuple[int, ...],
        execution_time_ms: float,
        nan_count: int | None = None,
    ) -> None:
        """Log completion of feature calculation.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        output_shape : tuple
            Shape of output data
        execution_time_ms : float
            Execution time in milliseconds
        nan_count : int, optional
            Number of NaN values in output
        """
        msg = "Completed calculation"
        extra = {
            "feature_name": feature_name,
            "data_shape": output_shape,
            "performance_ms": execution_time_ms,
        }

        if nan_count is not None:
            msg = f"{msg}, NaN values: {nan_count}"

        self.logger.info(msg, extra=extra)

    def log_data_quality(
        self,
        feature_name: str,
        data_shape: tuple[int, ...],
        issues: dict[str, Any],
    ) -> None:
        """Log data quality issues.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        data_shape : tuple
            Shape of data
        issues : dict
            Dictionary of data quality issues
        """
        extra = {
            "feature_name": feature_name,
            "data_shape": data_shape,
        }

        for issue_type, details in issues.items():
            if issue_type == "nan_percentage" and details > 0.1:  # > 10% NaN
                self.logger.warning(f"High NaN percentage: {details:.1%}", extra=extra)
            elif issue_type == "infinite_values" and details > 0:
                self.logger.warning(f"Found {details} infinite values", extra=extra)
            elif issue_type == "constant_values" and details:
                self.logger.warning("Data contains constant values", extra=extra)
            elif issue_type == "extreme_outliers" and details > 0:
                self.logger.warning(
                    f"Found {details} extreme outliers (>5 std)",
                    extra=extra,
                )

    def log_validation_error(
        self,
        feature_name: str,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log validation or calculation errors.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        error : Exception
            The error that occurred
        context : dict, optional
            Additional context information
        """
        extra = {"feature_name": feature_name}

        msg = f"Error in calculation: {error}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            msg = f"{msg} (context: {context_str})"

        self.logger.error(msg, extra=extra, exc_info=True)

    def log_performance_warning(
        self,
        feature_name: str,
        execution_time_ms: float,
        threshold_ms: float = 1000.0,
    ) -> None:
        """Log performance warnings for slow operations.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        execution_time_ms : float
            Execution time in milliseconds
        threshold_ms : float
            Warning threshold in milliseconds
        """
        if execution_time_ms > threshold_ms:
            extra = {
                "feature_name": feature_name,
                "performance_ms": execution_time_ms,
            }

            self.logger.warning(f"Slow execution (>{threshold_ms}ms)", extra=extra)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration information.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        config_str = ", ".join(f"{k}={v}" for k, v in config.items())
        self.logger.info(f"Configuration: {config_str}")


class PerformanceTracker:
    """Performance tracking context manager for feature calculations."""

    def __init__(
        self,
        logger: FeatureLogger,
        feature_name: str,
        data_shape: tuple[int, ...] | None = None,
        warn_threshold_ms: float = 1000.0,
    ) -> None:
        """Initialize performance tracker.

        Parameters
        ----------
        logger : FeatureLogger
            Logger instance
        feature_name : str
            Name of the feature being tracked
        data_shape : tuple, optional
            Shape of input data
        warn_threshold_ms : float
            Warning threshold for slow operations
        """
        self.logger = logger
        self.feature_name = feature_name
        self.data_shape = data_shape
        self.warn_threshold_ms = warn_threshold_ms
        self.start_time = None
        self.end_time = None

    def __enter__(self) -> "PerformanceTracker":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """End timing and log results."""
        self.end_time = time.perf_counter()
        # start_time is always set by __enter__ before __exit__ is called
        assert self.start_time is not None
        execution_time_ms = (self.end_time - self.start_time) * 1000

        if exc_type is None:
            # Success - log completion
            self.logger.log_feature_complete(
                self.feature_name,
                self.data_shape or (0,),
                execution_time_ms,
            )

            # Check for performance issues
            self.logger.log_performance_warning(
                self.feature_name,
                execution_time_ms,
                self.warn_threshold_ms,
            )
        elif exc_val is not None and isinstance(exc_val, Exception):
            # Error occurred (exc_val is guaranteed non-None when exc_type is not None)
            self.logger.log_validation_error(
                self.feature_name,
                exc_val,
                {"execution_time_ms": execution_time_ms},
            )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0

        end_time = self.end_time or time.perf_counter()
        return (end_time - self.start_time) * 1000


def logged_feature(
    feature_name: str | None = None,
    warn_threshold_ms: float = 1000.0,
    log_data_quality: bool = False,
) -> Any:
    """Decorator to add structured logging to feature functions.

    Parameters
    ----------
    feature_name : str, optional
        Name of the feature (defaults to function name)
    warn_threshold_ms : float
        Performance warning threshold
    log_data_quality : bool
        Whether to analyze and log data quality metrics

    Returns
    -------
    Decorated function with logging
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get feature name
            name = feature_name or func.__name__

            # Get logger
            module_name = (
                func.__module__.split(".")[-1] if hasattr(func, "__module__") else "unknown"
            )
            logger = get_logger(module_name)

            # Extract data shape if possible
            data_shape = None
            if args:
                first_arg = args[0]
                if hasattr(first_arg, "shape"):
                    data_shape = first_arg.shape
                elif hasattr(first_arg, "__len__"):
                    with suppress(Exception):
                        data_shape = (len(first_arg),)

            # Log start
            logger.log_feature_start(name, data_shape or (0,), kwargs)

            # Track performance
            with PerformanceTracker(logger, name, data_shape, warn_threshold_ms):
                result = func(*args, **kwargs)

                # Log data quality if requested
                if log_data_quality and result is not None:
                    try:
                        issues = _analyze_data_quality(result)
                        if issues:
                            logger.log_data_quality(
                                name,
                                getattr(result, "shape", (0,)),
                                issues,
                            )
                    except Exception as e:
                        logger.logger.debug(f"Could not analyze data quality: {e}")

                return result

        return wrapper

    return decorator


def _analyze_data_quality(
    data: npt.NDArray[np.float64] | pl.Series | pl.DataFrame,
) -> dict[str, Any]:
    """Analyze data quality and return issues."""
    issues = {}

    try:
        if isinstance(data, pl.DataFrame):
            # Analyze each column
            for col in data.columns:
                col_data = data[col]
                if col_data.dtype in [pl.Float32, pl.Float64]:
                    col_issues = _analyze_numeric_series(col_data)
                    if col_issues:
                        issues[col] = col_issues
        elif isinstance(data, pl.Series):
            if data.dtype in [pl.Float32, pl.Float64]:
                issues = _analyze_numeric_series(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            issues = _analyze_numeric_array(data)

    except Exception:
        # Don't let data quality analysis break the main function
        pass

    return issues


def _analyze_numeric_series(series: pl.Series) -> dict[str, Any]:
    """Analyze numeric Polars series for quality issues."""
    issues = {}

    # NaN percentage
    nan_count = series.null_count()
    if nan_count > 0:
        issues["nan_percentage"] = nan_count / len(series)

    # Infinite values
    if series.dtype in (pl.Float64, pl.Float32):
        inf_count = series.is_infinite().sum()
        if inf_count > 0:
            issues["infinite_values"] = inf_count

    # Constant values
    non_null = series.drop_nulls()
    if len(non_null) > 1 and non_null.min() == non_null.max():
        issues["constant_values"] = True

    return issues


def _analyze_numeric_array(arr: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Analyze numeric numpy array for quality issues."""
    issues = {}

    # NaN percentage
    nan_count = np.isnan(arr).sum()
    if nan_count > 0:
        issues["nan_percentage"] = nan_count / arr.size

    # Infinite values
    inf_count = np.isinf(arr).sum()
    if inf_count > 0:
        issues["infinite_values"] = inf_count

    # Constant values
    valid_data = arr[~(np.isnan(arr) | np.isinf(arr))]
    if len(valid_data) > 1:
        if np.min(valid_data) == np.max(valid_data):
            issues["constant_values"] = True

        # Extreme outliers (>5 standard deviations)
        if len(valid_data) > 10:  # Need reasonable sample size
            std = np.std(valid_data)
            mean = np.mean(valid_data)
            if std > 0:
                outliers = np.abs(valid_data - mean) > 5 * std
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    issues["extreme_outliers"] = outlier_count

    return issues


def setup_logging(
    level: int | str = logging.INFO,
    format_string: str | None = None,
    include_timestamp: bool = True,  # noqa: ARG001 - reserved for future use
) -> None:
    """Setup global logging configuration for ml4t.engineer.

    Parameters
    ----------
    level : int or str
        Logging level
    format_string : str, optional
        Custom format string
    include_timestamp : bool
        Whether to include timestamps
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Setup root ml4t.engineer logger
    engineer_logger = logging.getLogger("ml4t.engineer")
    engineer_logger.setLevel(level)

    # Remove existing handlers
    for handler in engineer_logger.handlers[:]:
        engineer_logger.removeHandler(handler)

    # Create new handler
    handler = logging.StreamHandler(sys.stdout)

    # Use custom formatter
    formatter = logging.Formatter(format_string) if format_string else QFeaturesFormatter()

    handler.setFormatter(formatter)
    engineer_logger.addHandler(handler)

    # Prevent propagation to root logger
    engineer_logger.propagate = False


def get_logger(name: str, level: int | None = None) -> FeatureLogger:
    """Get or create a feature logger.

    Parameters
    ----------
    name : str
        Logger name
    level : int, optional
        Logging level (defaults to current ml4t.engineer level)

    Returns
    -------
    FeatureLogger
        Configured logger instance
    """
    if level is None:
        # Get level from parent ml4t.engineer logger
        parent_logger = logging.getLogger("ml4t.engineer")
        level = parent_logger.level or logging.INFO

    return FeatureLogger(name, level)


@contextmanager
def suppress_warnings() -> Iterator[None]:
    """Context manager to suppress warnings during feature calculation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Global logger instances for common modules
_LOGGERS = {}


def _get_module_logger(module_name: str) -> FeatureLogger:
    """Get cached logger for module."""
    if module_name not in _LOGGERS:
        _LOGGERS[module_name] = get_logger(module_name)
    return _LOGGERS[module_name]


# Convenience functions for common modules
def get_ta_logger() -> FeatureLogger:
    """Get logger for technical analysis features."""
    return _get_module_logger("ta")


def get_ml_logger() -> FeatureLogger:
    """Get logger for ML features."""
    return _get_module_logger("ml_features")


def get_microstructure_logger() -> FeatureLogger:
    """Get logger for microstructure features."""
    return _get_module_logger("microstructure")


def get_volatility_logger() -> FeatureLogger:
    """Get logger for volatility features."""
    return _get_module_logger("volatility_advanced")


def get_regime_logger() -> FeatureLogger:
    """Get logger for regime features."""
    return _get_module_logger("regime")


def get_cross_asset_logger() -> FeatureLogger:
    """Get logger for cross-asset features."""
    return _get_module_logger("cross_asset")


def get_risk_logger() -> FeatureLogger:
    """Get logger for risk features."""
    return _get_module_logger("risk")
