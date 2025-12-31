"""Tests for structured logging functionality."""

import logging

import numpy as np
import pytest

from ml4t.engineer.logging import (
    FeatureLogger,
    LoggingConfig,
    PerformanceTracker,
    configure_logging,
    get_logger,
    logged_feature,
    setup_logging,
)


class TestFeatureLogger:
    """Test FeatureLogger functionality."""

    def test_logger_creation(self):
        """Test logger creation and basic functionality."""
        logger = FeatureLogger("test_module")
        assert logger.name == "test_module"
        assert logger.logger.name == "mlquant.features.test_module"

    def test_log_feature_lifecycle(self, caplog):
        """Test logging feature calculation lifecycle."""
        logger = FeatureLogger("test", level=logging.INFO)

        with caplog.at_level(logging.INFO):
            # Log start
            logger.log_feature_start("test_feature", (100, 5), {"param1": 14})

            # Log completion
            logger.log_feature_complete("test_feature", (100,), 125.5, nan_count=3)

        assert len(caplog.records) == 2
        assert "Starting calculation" in caplog.records[0].message
        assert "test_feature" in caplog.records[0].message
        assert "param1=14" in caplog.records[0].message

        assert "Completed calculation" in caplog.records[1].message
        assert "NaN values: 3" in caplog.records[1].message

    def test_log_data_quality_issues(self, caplog):
        """Test data quality issue logging."""
        logger = FeatureLogger("test", level=logging.WARNING)

        issues = {
            "nan_percentage": 0.15,  # 15% - should warn
            "infinite_values": 5,  # Should warn
            "constant_values": True,  # Should warn
            "extreme_outliers": 2,  # Should warn
        }

        with caplog.at_level(logging.WARNING):
            logger.log_data_quality("test_feature", (100,), issues)

        # Should have 4 warning messages
        assert len(caplog.records) == 4
        assert any("High NaN percentage" in record.message for record in caplog.records)
        assert any("infinite values" in record.message for record in caplog.records)
        assert any("constant values" in record.message for record in caplog.records)
        assert any("extreme outliers" in record.message for record in caplog.records)

    def test_log_validation_error(self, caplog):
        """Test error logging."""
        logger = FeatureLogger("test", level=logging.ERROR)

        error = ValueError("Invalid parameter")
        context = {"param": "invalid_value", "shape": (50,)}

        with caplog.at_level(logging.ERROR):
            logger.log_validation_error("test_feature", error, context)

        assert len(caplog.records) == 1
        assert "Error in calculation" in caplog.records[0].message
        assert "Invalid parameter" in caplog.records[0].message
        assert "param=invalid_value" in caplog.records[0].message

    def test_performance_warning(self, caplog):
        """Test performance warning logging."""
        logger = FeatureLogger("test", level=logging.WARNING)

        with caplog.at_level(logging.WARNING):
            # Fast execution - no warning
            logger.log_performance_warning("fast_feature", 100.0, threshold_ms=500.0)

            # Slow execution - should warn
            logger.log_performance_warning("slow_feature", 1500.0, threshold_ms=500.0)

        assert len(caplog.records) == 1
        assert "Slow execution" in caplog.records[0].message
        assert "slow_feature" in caplog.records[0].message


class TestPerformanceTracker:
    """Test PerformanceTracker functionality."""

    def test_performance_tracking_success(self, caplog):
        """Test successful performance tracking."""
        logger = FeatureLogger("test", level=logging.INFO)

        with (
            caplog.at_level(logging.INFO),
            PerformanceTracker(logger, "test_feature", (100,), warn_threshold_ms=50.0),
        ):
            # Simulate some work
            import time

            time.sleep(0.001)  # 1ms

        assert len(caplog.records) == 1
        assert "Completed calculation" in caplog.records[0].message
        assert "test_feature" in caplog.records[0].message

    def test_performance_tracking_with_error(self, caplog):
        """Test performance tracking when error occurs."""
        logger = FeatureLogger("test", level=logging.ERROR)

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(ValueError),
            PerformanceTracker(logger, "error_feature", (100,)),
        ):
            raise ValueError("Test error")

        assert len(caplog.records) == 1
        assert "Error in calculation" in caplog.records[0].message
        assert "Test error" in caplog.records[0].message

    def test_elapsed_time_property(self):
        """Test elapsed time property."""
        logger = FeatureLogger("test")

        tracker = PerformanceTracker(logger, "test_feature")

        # Before starting
        assert tracker.elapsed_ms == 0.0

        # During execution
        with tracker:
            import time

            time.sleep(0.001)
            assert tracker.elapsed_ms > 0.0


class TestLoggedFeatureDecorator:
    """Test logged_feature decorator."""

    def test_decorator_basic_functionality(self, caplog):
        """Test basic decorator functionality."""

        @logged_feature("test_function", warn_threshold_ms=100.0)
        def test_function(data):
            return data * 2

        test_data = np.array([1, 2, 3, 4, 5])

        with caplog.at_level(logging.INFO):
            result = test_function(test_data)

        assert np.array_equal(result, test_data * 2)
        assert len(caplog.records) == 2  # Start and complete
        assert "Starting calculation" in caplog.records[0].message
        assert "Completed calculation" in caplog.records[1].message

    def test_decorator_with_data_quality(self, caplog):
        """Test decorator with data quality analysis."""

        @logged_feature("test_function", log_data_quality=True)
        def test_function_with_nans(data):
            result = data.copy()
            result[0] = np.nan  # Introduce NaN
            return result

        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with caplog.at_level(logging.DEBUG):
            result = test_function_with_nans(test_data)

        assert np.isnan(result[0])
        # Should have logged start and completion
        assert len(caplog.records) >= 2

    def test_decorator_error_handling(self, caplog):
        """Test decorator error handling."""

        @logged_feature("failing_function")
        def failing_function(data):  # noqa: ARG001
            raise ValueError("Intentional error")

        test_data = np.array([1, 2, 3])

        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            failing_function(test_data)

        # Should have start log and error log
        assert len(caplog.records) >= 1
        assert any("Error in calculation" in record.message for record in caplog.records)


class TestLoggingConfiguration:
    """Test logging configuration functionality."""

    def test_config_from_environment_with_preset(self, monkeypatch):
        """Test configuration from environment variables with preset."""
        monkeypatch.setenv("QFEATURES_LOG_PRESET", "development")
        monkeypatch.setenv("QFEATURES_LOG_LEVEL", "WARNING")

        config = LoggingConfig.from_environment()

        # Should use preset but override level
        assert config.level == logging.WARNING
        assert config.performance_warnings is True
        assert config.data_quality_checks is True

    def test_config_from_environment_without_preset(self, monkeypatch):
        """Test configuration from environment variables without preset."""
        # Clear any preset
        monkeypatch.delenv("QFEATURES_LOG_PRESET", raising=False)
        monkeypatch.setenv("QFEATURES_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("QFEATURES_PERFORMANCE_WARNINGS", "false")
        monkeypatch.setenv("QFEATURES_DATA_QUALITY_CHECKS", "false")
        monkeypatch.setenv("QFEATURES_WARN_THRESHOLD_MS", "2500.0")

        config = LoggingConfig.from_environment()

        assert config.level == logging.DEBUG
        assert config.performance_warnings is False
        assert config.data_quality_checks is False
        assert config.warn_threshold_ms == 2500.0

    def test_config_from_environment_defaults(self, monkeypatch):
        """Test configuration from environment with defaults."""
        # Clear all environment variables
        monkeypatch.delenv("QFEATURES_LOG_PRESET", raising=False)
        monkeypatch.delenv("QFEATURES_LOG_LEVEL", raising=False)
        monkeypatch.delenv("QFEATURES_PERFORMANCE_WARNINGS", raising=False)
        monkeypatch.delenv("QFEATURES_DATA_QUALITY_CHECKS", raising=False)
        monkeypatch.delenv("QFEATURES_WARN_THRESHOLD_MS", raising=False)

        config = LoggingConfig.from_environment()

        # Should use defaults
        assert config.level == logging.INFO
        assert config.performance_warnings is True
        assert config.data_quality_checks is True
        assert config.warn_threshold_ms == 1000.0

    def test_configure_logging_none_uses_environment(self, monkeypatch):
        """Test configure_logging with None uses environment variables."""
        monkeypatch.setenv("QFEATURES_LOG_PRESET", "quiet")

        config = configure_logging(None)

        assert config.level == logging.ERROR
        assert config.performance_warnings is False

    def test_configure_logging_none_falls_back_to_defaults(self, monkeypatch):
        """Test configure_logging with None falls back to defaults on error."""
        # Set invalid preset to trigger exception
        monkeypatch.setenv("QFEATURES_LOG_PRESET", "invalid_preset_name")

        config = configure_logging(None)

        # Should fall back to default LoggingConfig
        assert config is not None
        assert isinstance(config, LoggingConfig)

    def test_configure_logging_invalid_type(self):
        """Test configure_logging with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="config must be"):
            configure_logging(12345)  # type: ignore

        with pytest.raises(TypeError, match="config must be"):
            configure_logging([1, 2, 3])  # type: ignore

    def test_logging_config_creation(self):
        """Test LoggingConfig creation."""
        config = LoggingConfig(
            level=logging.DEBUG,
            performance_warnings=True,
            data_quality_checks=False,
            warn_threshold_ms=200.0,
        )

        assert config.level == logging.DEBUG
        assert config.performance_warnings is True
        assert config.data_quality_checks is False
        assert config.warn_threshold_ms == 200.0

    def test_config_from_preset(self):
        """Test configuration from preset."""
        dev_config = LoggingConfig.from_preset("development")

        assert dev_config.level == logging.DEBUG
        assert dev_config.performance_warnings is True
        assert dev_config.data_quality_checks is True
        assert dev_config.warn_threshold_ms == 500.0

        prod_config = LoggingConfig.from_preset("production")
        assert prod_config.level == logging.WARNING
        assert prod_config.data_quality_checks is False

    def test_invalid_preset(self):
        """Test invalid preset handling."""
        with pytest.raises(ValueError, match="Unknown preset"):
            LoggingConfig.from_preset("invalid_preset")

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = LoggingConfig(level=logging.INFO, performance_warnings=False)
        config_dict = config.to_dict()

        assert config_dict["level"] == logging.INFO
        assert config_dict["performance_warnings"] is False
        assert "warn_threshold_ms" in config_dict

    def test_configure_logging_with_preset(self):
        """Test configure_logging with preset."""
        config = configure_logging("quiet")

        assert config.level == logging.ERROR
        assert config.performance_warnings is False

        # Verify logger level was set
        ml4t_logger = logging.getLogger("ml4t.engineer")
        assert ml4t_logger.level == logging.ERROR

    def test_configure_logging_with_dict(self):
        """Test configure_logging with dictionary."""
        config_dict = {
            "level": logging.WARNING,
            "performance_warnings": True,
            "warn_threshold_ms": 750.0,
        }

        config = configure_logging(config_dict)

        assert config.level == logging.WARNING
        assert config.warn_threshold_ms == 750.0


class TestLoggerUtilities:
    """Test logger utility functions."""

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")

        assert isinstance(logger, FeatureLogger)
        assert logger.name == "test_module"

    def test_setup_logging(self):
        """Test setup_logging function."""
        setup_logging(level=logging.WARNING)

        ml4t_logger = logging.getLogger("ml4t.engineer")
        assert ml4t_logger.level == logging.WARNING

    def test_module_specific_loggers(self):
        """Test module-specific logger getters."""
        from ml4t.engineer.logging.core import (
            get_cross_asset_logger,
            get_ml_logger,
            get_ta_logger,
        )

        ta_logger = get_ta_logger()
        ml_logger = get_ml_logger()
        cross_asset_logger = get_cross_asset_logger()

        assert ta_logger.name == "ta"
        assert ml_logger.name == "ml_features"
        assert cross_asset_logger.name == "cross_asset"


class TestDataQualityAnalysis:
    """Test data quality analysis functionality."""

    def test_numpy_array_analysis(self):
        """Test data quality analysis for numpy arrays."""
        from ml4t.engineer.logging.core import _analyze_data_quality

        # Array with NaNs and outliers
        data = np.array([1.0, 2.0, np.nan, 4.0, 100.0])  # 100.0 is outlier
        issues = _analyze_data_quality(data)

        assert "nan_percentage" in issues
        assert issues["nan_percentage"] == 0.2  # 1/5 = 20%

    def test_constant_values_detection(self):
        """Test detection of constant values."""
        from ml4t.engineer.logging.core import _analyze_numeric_array

        # Constant array
        data = np.array([5.0, 5.0, 5.0, 5.0])
        issues = _analyze_numeric_array(data)

        assert "constant_values" in issues
        assert issues["constant_values"] is True

    def test_infinite_values_detection(self):
        """Test detection of infinite values."""
        from ml4t.engineer.logging.core import _analyze_numeric_array

        # Array with infinites
        data = np.array([1.0, 2.0, np.inf, 4.0, -np.inf])
        issues = _analyze_numeric_array(data)

        assert "infinite_values" in issues
        assert issues["infinite_values"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
