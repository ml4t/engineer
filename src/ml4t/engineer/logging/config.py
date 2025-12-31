# mypy: disable-error-code="arg-type,dict-item"
"""Logging configuration utilities for ml4t.engineer.

Provides configuration options and presets for different logging scenarios.
"""

import logging
import os
from typing import Any


class LoggingConfig:
    """Configuration class for ml4t.engineer logging."""

    # Default configurations
    PRESETS = {
        "development": {
            "level": logging.DEBUG,
            "performance_warnings": True,
            "data_quality_checks": True,
            "warn_threshold_ms": 500.0,
        },
        "production": {
            "level": logging.WARNING,
            "performance_warnings": True,
            "data_quality_checks": False,
            "warn_threshold_ms": 2000.0,
        },
        "performance": {
            "level": logging.INFO,
            "performance_warnings": True,
            "data_quality_checks": False,
            "warn_threshold_ms": 100.0,
        },
        "quiet": {
            "level": logging.ERROR,
            "performance_warnings": False,
            "data_quality_checks": False,
            "warn_threshold_ms": 10000.0,
        },
    }

    def __init__(
        self,
        level: int | str = logging.INFO,
        performance_warnings: bool = True,
        data_quality_checks: bool = True,
        warn_threshold_ms: float = 1000.0,
        format_string: str | None = None,
        include_timestamp: bool = True,
    ):
        """Initialize logging configuration.

        Parameters
        ----------
        level : int or str
            Logging level
        performance_warnings : bool
            Enable performance warnings
        data_quality_checks : bool
            Enable data quality analysis
        warn_threshold_ms : float
            Performance warning threshold in milliseconds
        format_string : str, optional
            Custom log format string
        include_timestamp : bool
            Include timestamps in log messages
        """
        self.level = level
        self.performance_warnings = performance_warnings
        self.data_quality_checks = data_quality_checks
        self.warn_threshold_ms = warn_threshold_ms
        self.format_string = format_string
        self.include_timestamp = include_timestamp

    @classmethod
    def from_preset(cls, preset: str) -> "LoggingConfig":
        """Create configuration from preset.

        Parameters
        ----------
        preset : str
            Preset name ('development', 'production', 'performance', 'quiet')

        Returns
        -------
        LoggingConfig
            Configuration instance
        """
        if preset not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(cls.PRESETS.keys())}",
            )

        return cls(**cls.PRESETS[preset])

    @classmethod
    def from_environment(cls) -> "LoggingConfig":
        """Create configuration from environment variables.

        Environment variables:
        - QFEATURES_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        - QFEATURES_LOG_PRESET: Preset name
        - QFEATURES_PERFORMANCE_WARNINGS: Enable performance warnings (true/false)
        - QFEATURES_DATA_QUALITY_CHECKS: Enable data quality checks (true/false)
        - QFEATURES_WARN_THRESHOLD_MS: Performance warning threshold

        Returns
        -------
        LoggingConfig
            Configuration instance
        """
        # Check for preset first
        preset = os.getenv("QFEATURES_LOG_PRESET")
        if preset:
            config = cls.from_preset(preset)

            # Override with specific environment variables
            level_str = os.getenv("QFEATURES_LOG_LEVEL")
            if level_str:
                config.level = getattr(logging, level_str.upper())

            return config

        # Build configuration from individual variables
        level_str = os.getenv("QFEATURES_LOG_LEVEL", "INFO")
        level = getattr(logging, level_str.upper())

        performance_warnings = os.getenv("QFEATURES_PERFORMANCE_WARNINGS", "true").lower() == "true"
        data_quality_checks = os.getenv("QFEATURES_DATA_QUALITY_CHECKS", "true").lower() == "true"

        warn_threshold_str = os.getenv("QFEATURES_WARN_THRESHOLD_MS", "1000.0")
        warn_threshold_ms = float(warn_threshold_str)

        return cls(
            level=level,
            performance_warnings=performance_warnings,
            data_quality_checks=data_quality_checks,
            warn_threshold_ms=warn_threshold_ms,
        )

    def to_dict(self) -> dict[str, int | bool | float | str | None]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            "level": self.level,
            "performance_warnings": self.performance_warnings,
            "data_quality_checks": self.data_quality_checks,
            "warn_threshold_ms": self.warn_threshold_ms,
            "format_string": self.format_string,
            "include_timestamp": self.include_timestamp,
        }


def configure_logging(
    config: LoggingConfig | str | dict[str, Any] | None = None,
) -> LoggingConfig:
    """Configure ml4t.engineer logging.

    Parameters
    ----------
    config : LoggingConfig, str, dict, or None
        Configuration to apply. Can be:
        - LoggingConfig instance
        - Preset name (str)
        - Configuration dictionary
        - None (use environment variables or defaults)

    Returns
    -------
    LoggingConfig
        Applied configuration
    """
    from ml4t.engineer.logging.core import setup_logging

    # Determine configuration
    if config is None:
        # Try environment variables, fall back to defaults
        try:
            config = LoggingConfig.from_environment()
        except Exception:
            config = LoggingConfig()
    elif isinstance(config, str):
        # Preset name
        config = LoggingConfig.from_preset(config)
    elif isinstance(config, dict):
        # Dictionary
        config = LoggingConfig(**config)
    elif not isinstance(config, LoggingConfig):
        raise TypeError("config must be LoggingConfig, str, dict, or None")

    # Apply configuration
    setup_logging(
        level=config.level,
        format_string=config.format_string,
        include_timestamp=config.include_timestamp,
    )

    return config
