"""Tests for config system (BaseConfig, validation, serialization)."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import Field, ValidationError

from ml4t.engineer.config.base import BaseConfig, ComputationalConfig, StatisticalTestConfig

# =============================================================================
# Test Models
# =============================================================================


class SimpleTestConfig(BaseConfig):
    """Simple config for testing."""

    value: int = 42
    name: str = "test"
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class NestedTestConfig(BaseConfig):
    """Nested config for testing."""

    simple: SimpleTestConfig = Field(default_factory=SimpleTestConfig)
    multiplier: float = 2.0


# =============================================================================
# BaseConfig Tests
# =============================================================================


class TestBaseConfigSerialization:
    """Tests for BaseConfig serialization methods."""

    def test_to_dict_basic(self) -> None:
        """Test basic dictionary conversion."""
        config = SimpleTestConfig(value=100, name="demo")
        d = config.to_dict()

        assert d["value"] == 100
        assert d["name"] == "demo"
        assert d["threshold"] == 0.5

    def test_to_dict_exclude_none(self) -> None:
        """Test dictionary conversion excluding None values."""

        class ConfigWithOptional(BaseConfig):
            required: int = 42
            optional: int | None = None

        config = ConfigWithOptional(required=100)
        d = config.to_dict(exclude_none=True)

        assert "required" in d
        assert "optional" not in d

    def test_to_dict_json_mode(self) -> None:
        """Test dictionary conversion in JSON mode."""
        config = SimpleTestConfig()
        d = config.to_dict(mode="json")

        # JSON mode should produce JSON-serializable types
        assert isinstance(d["value"], int)
        assert isinstance(d["name"], str)

    def test_json_roundtrip(self) -> None:
        """Test JSON save and load roundtrip."""
        config = SimpleTestConfig(value=999, name="roundtrip", threshold=0.75)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            # Save
            config.to_json(path)
            assert path.exists()

            # Load
            loaded = SimpleTestConfig.from_json(path)
            assert loaded.value == 999
            assert loaded.name == "roundtrip"
            assert loaded.threshold == 0.75

    def test_json_creates_directory(self) -> None:
        """Test that to_json creates parent directories."""
        config = SimpleTestConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.json"

            config.to_json(path)
            assert path.exists()

    def test_json_file_not_found(self) -> None:
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            SimpleTestConfig.from_json("/nonexistent/path/config.json")

    def test_json_invalid_content(self) -> None:
        """Test loading from invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            path.write_text("not valid json {{{")

            with pytest.raises((json.JSONDecodeError, ValueError)):
                SimpleTestConfig.from_json(path)

    def test_yaml_roundtrip(self) -> None:
        """Test YAML save and load roundtrip."""
        config = SimpleTestConfig(value=888, name="yaml_test", threshold=0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"

            # Save
            config.to_yaml(path)
            assert path.exists()

            # Load
            loaded = SimpleTestConfig.from_yaml(path)
            assert loaded.value == 888
            assert loaded.name == "yaml_test"
            assert loaded.threshold == 0.9

    def test_yaml_creates_directory(self) -> None:
        """Test that to_yaml creates parent directories."""
        config = SimpleTestConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "test.yaml"

            config.to_yaml(path)
            assert path.exists()

    def test_yaml_file_not_found(self) -> None:
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            SimpleTestConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_yaml_invalid_content(self) -> None:
        """Test loading from invalid YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.yaml"
            path.write_text("invalid: yaml: content:\n  - broken")

            with pytest.raises((yaml.YAMLError, ValidationError)):
                SimpleTestConfig.from_yaml(path)


class TestBaseConfigFromDict:
    """Tests for from_dict method."""

    def test_from_dict_basic(self) -> None:
        """Test creating config from dictionary."""
        data = {"value": 777, "name": "from_dict"}
        config = SimpleTestConfig.from_dict(data)

        assert config.value == 777
        assert config.name == "from_dict"

    def test_from_dict_validation_error(self) -> None:
        """Test that from_dict validates input."""
        data = {"value": "not_an_int"}  # Wrong type

        with pytest.raises(ValidationError):
            SimpleTestConfig.from_dict(data)

    def test_from_dict_extra_fields(self) -> None:
        """Test that extra fields are rejected."""
        data = {"value": 42, "name": "test", "extra_field": "should_fail"}

        with pytest.raises(ValidationError):
            SimpleTestConfig.from_dict(data)


class TestBaseConfigFromFile:
    """Tests for auto-detecting file type."""

    def test_from_file_json(self) -> None:
        """Test loading from JSON file via from_file."""
        config = SimpleTestConfig(value=111)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            config.to_json(path)

            loaded = SimpleTestConfig.from_file(path)
            assert loaded.value == 111

    def test_from_file_yaml(self) -> None:
        """Test loading from YAML file via from_file."""
        config = SimpleTestConfig(value=222)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            config.to_yaml(path)

            loaded = SimpleTestConfig.from_file(path)
            assert loaded.value == 222

    def test_from_file_yml_extension(self) -> None:
        """Test loading from .yml file."""
        config = SimpleTestConfig(value=333)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yml"
            config.to_yaml(path)

            loaded = SimpleTestConfig.from_file(path)
            assert loaded.value == 333

    def test_from_file_unsupported_extension(self) -> None:
        """Test that unsupported file extensions raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("some content")

            with pytest.raises(ValueError, match="Unsupported file type"):
                SimpleTestConfig.from_file(path)

    def test_from_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            SimpleTestConfig.from_file("/nonexistent/config.yaml")


class TestBaseConfigValidation:
    """Tests for validation methods."""

    def test_validate_fully_success(self) -> None:
        """Test validate_fully with valid config."""
        config = SimpleTestConfig(value=50)
        errors = config.validate_fully()

        assert errors == []

    def test_validate_fully_invalid(self) -> None:
        """Test validate_fully detects validation errors."""
        # Create config with valid data first
        config = SimpleTestConfig(value=50, threshold=0.5)

        # Manually modify to invalid state (bypass validation)
        config.__dict__["threshold"] = 2.0  # Out of range [0, 1]

        errors = config.validate_fully()
        # Should detect threshold out of range
        assert len(errors) > 0

    def test_pydantic_validation_on_assignment(self) -> None:
        """Test that assignment validation is enabled."""
        config = SimpleTestConfig()

        with pytest.raises(ValidationError):
            config.threshold = 2.0  # Out of range

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            SimpleTestConfig(value=42, unknown_field="should_fail")


class TestBaseConfigComparison:
    """Tests for diff method."""

    def test_diff_no_differences(self) -> None:
        """Test diff with identical configs."""
        config1 = SimpleTestConfig(value=100)
        config2 = SimpleTestConfig(value=100)

        diff = config1.diff(config2)
        assert diff == {}

    def test_diff_single_field(self) -> None:
        """Test diff with one differing field."""
        config1 = SimpleTestConfig(value=100, name="test1")
        config2 = SimpleTestConfig(value=200, name="test1")

        diff = config1.diff(config2)
        assert diff == {"value": (100, 200)}

    def test_diff_multiple_fields(self) -> None:
        """Test diff with multiple differing fields."""
        config1 = SimpleTestConfig(value=100, name="a", threshold=0.3)
        config2 = SimpleTestConfig(value=200, name="b", threshold=0.7)

        diff = config1.diff(config2)
        assert "value" in diff
        assert "name" in diff
        assert "threshold" in diff

    def test_diff_nested(self) -> None:
        """Test diff with nested configs."""
        config1 = NestedTestConfig(simple=SimpleTestConfig(value=10), multiplier=1.0)
        config2 = NestedTestConfig(simple=SimpleTestConfig(value=20), multiplier=1.0)

        diff = config1.diff(config2)
        assert "simple.value" in diff

    def test_diff_type_mismatch(self) -> None:
        """Test diff rejects comparing different types."""
        config1 = SimpleTestConfig()
        config2 = NestedTestConfig()

        with pytest.raises(TypeError):
            config1.diff(config2)  # type: ignore[arg-type]


# =============================================================================
# StatisticalTestConfig Tests
# =============================================================================


class TestStatisticalTestConfig:
    """Tests for StatisticalTestConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = StatisticalTestConfig()

        assert config.enabled is True
        assert config.significance_level == 0.05

    def test_custom_significance_level(self) -> None:
        """Test custom significance levels."""
        config = StatisticalTestConfig(significance_level=0.01)
        assert config.significance_level == 0.01

        config = StatisticalTestConfig(significance_level=0.10)
        assert config.significance_level == 0.10

    def test_significance_level_validation(self) -> None:
        """Test significance level range validation."""
        # Too low
        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=0.0001)

        # Too high
        with pytest.raises(ValidationError):
            StatisticalTestConfig(significance_level=0.5)

    def test_enabled_flag(self) -> None:
        """Test enabled flag."""
        config = StatisticalTestConfig(enabled=False)
        assert config.enabled is False


# =============================================================================
# ComputationalConfig Tests
# =============================================================================


class TestComputationalConfig:
    """Tests for ComputationalConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ComputationalConfig()

        assert config.n_jobs == -1  # All cores
        assert config.cache_enabled is True
        assert config.cache_ttl is None
        assert config.verbose is False

    def test_cache_directory_created(self) -> None:
        """Test that cache directory is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "custom_cache"

            ComputationalConfig(cache_dir=cache_dir)
            assert cache_dir.exists()

    def test_cache_directory_not_created_when_disabled(self) -> None:
        """Test cache directory not created when caching disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "should_not_exist"

            ComputationalConfig(cache_enabled=False, cache_dir=cache_dir)
            assert not cache_dir.exists()

    def test_n_jobs_validation(self) -> None:
        """Test n_jobs validation."""
        # Valid values
        config = ComputationalConfig(n_jobs=1)
        assert config.n_jobs == 1

        config = ComputationalConfig(n_jobs=4)
        assert config.n_jobs == 4

        config = ComputationalConfig(n_jobs=-1)
        assert config.n_jobs == -1

        # Invalid value
        with pytest.raises(ValidationError):
            ComputationalConfig(n_jobs=-2)

    def test_cache_ttl_validation(self) -> None:
        """Test cache TTL validation."""
        # Valid values
        config = ComputationalConfig(cache_ttl=3600)
        assert config.cache_ttl == 3600

        config = ComputationalConfig(cache_ttl=None)
        assert config.cache_ttl is None

        # Invalid (negative)
        with pytest.raises(ValidationError):
            ComputationalConfig(cache_ttl=-100)

    def test_verbose_flag(self) -> None:
        """Test verbose flag."""
        config = ComputationalConfig(verbose=True)
        assert config.verbose is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeatureConfigValidators:
    """Tests for feature_config.py validators."""

    def test_stationarity_config_at_least_one_test(self) -> None:
        """Test that StationarityConfig requires at least one test enabled."""
        from ml4t.engineer.config.feature_config import StationarityConfig

        # Valid: at least one test enabled
        config = StationarityConfig(adf_enabled=True, kpss_enabled=False, pp_enabled=False)
        assert config.adf_enabled is True

        config = StationarityConfig(adf_enabled=False, kpss_enabled=True, pp_enabled=False)
        assert config.kpss_enabled is True

        config = StationarityConfig(adf_enabled=False, kpss_enabled=False, pp_enabled=True)
        assert config.pp_enabled is True

    def test_stationarity_config_none_enabled_raises(self) -> None:
        """Test that StationarityConfig raises if no tests enabled."""
        from ml4t.engineer.config.feature_config import StationarityConfig

        with pytest.raises(ValidationError, match="At least one stationarity test"):
            StationarityConfig(adf_enabled=False, kpss_enabled=False, pp_enabled=False)

    def test_volatility_config_window_sizes_validation(self) -> None:
        """Test VolatilityConfig window_sizes validator."""
        from ml4t.engineer.config.feature_config import VolatilityConfig

        # Valid window sizes
        config = VolatilityConfig(window_sizes=[5, 10, 20])
        assert config.window_sizes == [5, 10, 20]

        # Window sizes get sorted
        config = VolatilityConfig(window_sizes=[20, 5, 10])
        assert config.window_sizes == [5, 10, 20]

    def test_volatility_config_empty_window_sizes_raises(self) -> None:
        """Test that empty window_sizes raises error."""
        from ml4t.engineer.config.feature_config import VolatilityConfig

        with pytest.raises(ValidationError, match="at least one window size"):
            VolatilityConfig(window_sizes=[])

    def test_volatility_config_invalid_window_size_raises(self) -> None:
        """Test that window size < 2 raises error."""
        from ml4t.engineer.config.feature_config import VolatilityConfig

        with pytest.raises(ValidationError, match="must be >= 2"):
            VolatilityConfig(window_sizes=[1, 5, 10])


class TestConfigIntegration:
    """Integration tests for config system."""

    def test_complex_config_roundtrip(self) -> None:
        """Test roundtrip of complex nested config."""
        config = NestedTestConfig(
            simple=SimpleTestConfig(value=999, name="nested", threshold=0.8),
            multiplier=3.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "complex.yaml"
            json_path = Path(tmpdir) / "complex.json"

            # Save to both formats
            config.to_yaml(yaml_path)
            config.to_json(json_path)

            # Load from YAML
            yaml_loaded = NestedTestConfig.from_yaml(yaml_path)
            assert yaml_loaded.simple.value == 999
            assert yaml_loaded.multiplier == 3.5

            # Load from JSON
            json_loaded = NestedTestConfig.from_json(json_path)
            assert json_loaded.simple.value == 999
            assert json_loaded.multiplier == 3.5

    def test_config_modification_and_save(self) -> None:
        """Test modifying config and saving."""
        config = SimpleTestConfig(value=100)

        # Modify
        config.value = 200
        config.name = "modified"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "modified.yaml"
            config.to_yaml(path)

            loaded = SimpleTestConfig.from_yaml(path)
            assert loaded.value == 200
            assert loaded.name == "modified"

    def test_equality_comparison(self) -> None:
        """Test config equality."""
        config1 = SimpleTestConfig(value=100, name="test")
        config2 = SimpleTestConfig(value=100, name="test")
        config3 = SimpleTestConfig(value=200, name="test")

        assert config1 == config2
        assert config1 != config3
