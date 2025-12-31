"""Structured logging functionality for ml4t.engineer.

This module provides comprehensive logging capabilities for the ml4t.engineer library,
including performance tracking, data quality monitoring, and error reporting.
"""

from ml4t.engineer.logging.config import LoggingConfig, configure_logging
from ml4t.engineer.logging.core import (
    FeatureLogger,
    PerformanceTracker,
    get_logger,
    logged_feature,
    setup_logging,
)

__all__ = [
    "FeatureLogger",
    "LoggingConfig",
    "PerformanceTracker",
    "configure_logging",
    "get_logger",
    "logged_feature",
    "setup_logging",
]
