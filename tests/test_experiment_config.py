"""Tests for config/experiment.py - ExperimentConfig load/save/roundtrip."""

import pytest
import yaml

from ml4t.engineer.config.experiment import (
    ExperimentConfig,
    load_experiment_config,
    save_experiment_config,
)
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.config.preprocessing_config import PreprocessingConfig


@pytest.fixture
def yaml_file(tmp_path):
    """Create a YAML config file for testing."""
    config = {
        "features": [
            {"name": "rsi", "params": {"period": 14}},
            {"name": "macd"},
        ],
        "labeling": {
            "method": "triple_barrier",
            "upper_barrier": 0.02,
            "lower_barrier": 0.01,
            "max_holding_period": 20,
        },
        "preprocessing": {
            "scaler": "robust",
            "quantile_range": [10.0, 90.0],
        },
    }
    path = tmp_path / "experiment.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def minimal_yaml(tmp_path):
    """Create a minimal YAML with only features."""
    config = {"features": [{"name": "rsi"}]}
    path = tmp_path / "minimal.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


class TestLoadExperimentConfig:
    """Tests for load_experiment_config()."""

    def test_load_full_config(self, yaml_file):
        config = load_experiment_config(yaml_file)
        assert isinstance(config, ExperimentConfig)
        assert len(config.features) == 2
        assert config.features[0]["name"] == "rsi"
        assert config.features[0]["params"]["period"] == 14

    def test_load_labeling_config(self, yaml_file):
        config = load_experiment_config(yaml_file)
        assert isinstance(config.labeling, LabelingConfig)
        assert config.labeling.method == "triple_barrier"
        assert config.labeling.upper_barrier == 0.02
        assert config.labeling.lower_barrier == 0.01
        assert config.labeling.max_holding_period == 20

    def test_load_preprocessing_config(self, yaml_file):
        config = load_experiment_config(yaml_file)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert config.preprocessing.scaler == "robust"
        assert config.preprocessing.quantile_range == (10.0, 90.0)

    def test_load_raw_preserved(self, yaml_file):
        config = load_experiment_config(yaml_file)
        assert "features" in config.raw
        assert "labeling" in config.raw
        assert "preprocessing" in config.raw

    def test_load_minimal(self, minimal_yaml):
        config = load_experiment_config(minimal_yaml)
        assert len(config.features) == 1
        assert config.labeling is None
        assert config.preprocessing is None

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_experiment_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_yaml(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        config = load_experiment_config(path)
        assert config.features == []
        assert config.labeling is None
        assert config.preprocessing is None

    def test_load_no_validate(self, yaml_file):
        config = load_experiment_config(yaml_file, validate=False)
        assert config.labeling is not None
        assert config.preprocessing is not None

    def test_load_string_path(self, yaml_file):
        config = load_experiment_config(str(yaml_file))
        assert len(config.features) == 2


class TestSaveExperimentConfig:
    """Tests for save_experiment_config()."""

    def test_save_full_config(self, tmp_path):
        config = ExperimentConfig(
            features=[{"name": "rsi", "params": {"period": 14}}],
            labeling=LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.01),
            preprocessing=PreprocessingConfig.robust(),
        )
        path = tmp_path / "output.yaml"
        save_experiment_config(config, path)
        assert path.exists()

        with open(path) as f:
            raw = yaml.safe_load(f)
        assert "features" in raw
        assert "labeling" in raw
        assert "preprocessing" in raw

    def test_save_minimal(self, tmp_path):
        config = ExperimentConfig(features=[{"name": "sma"}])
        path = tmp_path / "minimal.yaml"
        save_experiment_config(config, path)

        with open(path) as f:
            raw = yaml.safe_load(f)
        assert "features" in raw
        assert "labeling" not in raw
        assert "preprocessing" not in raw

    def test_save_empty(self, tmp_path):
        config = ExperimentConfig()
        path = tmp_path / "empty.yaml"
        save_experiment_config(config, path)
        assert path.exists()

    def test_save_include_defaults(self, tmp_path):
        config = ExperimentConfig(
            preprocessing=PreprocessingConfig.standard(),
        )
        path = tmp_path / "with_defaults.yaml"
        save_experiment_config(config, path, include_defaults=True)

        with open(path) as f:
            raw = yaml.safe_load(f)
        # With include_defaults=True, default fields should appear
        assert "preprocessing" in raw
        assert raw["preprocessing"]["scaler"] == "standard"


class TestRoundtrip:
    """Tests for save → load roundtrip fidelity."""

    def test_roundtrip_features(self, tmp_path):
        original = ExperimentConfig(
            features=[
                {"name": "rsi", "params": {"period": 14}},
                {"name": "macd", "params": {"fast": 12, "slow": 26}},
            ],
        )
        path = tmp_path / "roundtrip.yaml"
        save_experiment_config(original, path)
        loaded = load_experiment_config(path)
        assert loaded.features == original.features

    def test_roundtrip_labeling(self, tmp_path):
        original = ExperimentConfig(
            labeling=LabelingConfig.triple_barrier(
                upper_barrier=0.03,
                lower_barrier=0.015,
                max_holding_period=30,
            ),
        )
        path = tmp_path / "roundtrip.yaml"
        save_experiment_config(original, path, include_defaults=True)
        loaded = load_experiment_config(path)
        assert loaded.labeling is not None
        assert loaded.labeling.upper_barrier == 0.03
        assert loaded.labeling.lower_barrier == 0.015
        assert loaded.labeling.max_holding_period == 30

    def test_roundtrip_preprocessing(self, tmp_path):
        original = ExperimentConfig(
            preprocessing=PreprocessingConfig.robust(quantile_range=(5.0, 95.0)),
        )
        path = tmp_path / "roundtrip.yaml"
        save_experiment_config(original, path, include_defaults=True)
        loaded = load_experiment_config(path)
        assert loaded.preprocessing is not None
        assert loaded.preprocessing.scaler == "robust"
        assert loaded.preprocessing.quantile_range == (5.0, 95.0)

    def test_roundtrip_full(self, tmp_path):
        original = ExperimentConfig(
            features=[{"name": "rsi", "params": {"period": 20}}],
            labeling=LabelingConfig.triple_barrier(upper_barrier=0.02),
            preprocessing=PreprocessingConfig.standard(),
        )
        path = tmp_path / "full.yaml"
        save_experiment_config(original, path, include_defaults=True)
        loaded = load_experiment_config(path)

        assert loaded.features == original.features
        assert loaded.labeling is not None
        assert loaded.labeling.upper_barrier == 0.02
        assert loaded.preprocessing is not None
        assert loaded.preprocessing.scaler == "standard"


class TestExperimentConfigDataclass:
    """Tests for ExperimentConfig dataclass."""

    def test_defaults(self):
        config = ExperimentConfig()
        assert config.features == []
        assert config.labeling is None
        assert config.preprocessing is None
        assert config.raw == {}

    def test_with_values(self):
        config = ExperimentConfig(
            features=[{"name": "rsi"}],
            labeling=LabelingConfig.triple_barrier(upper_barrier=0.02),
        )
        assert len(config.features) == 1
        assert config.labeling is not None
        assert config.preprocessing is None
