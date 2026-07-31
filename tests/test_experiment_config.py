"""Tests for config/experiment.py - ExperimentConfig load/save/roundtrip."""

from datetime import timedelta

import pytest
import yaml

from ml4t.engineer.config.data_contract import DataContractConfig
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

    @pytest.mark.parametrize(
        ("document", "match"),
        [
            (["not", "a", "mapping"], "mapping"),
            ({"features": {"name": "sma"}}, "features"),
            ({"labeling": "triple_barrier"}, "labeling"),
            ({"preprocessing": "robust"}, "preprocessing"),
            ({"features": [{"name": "sma"}, 7]}, "features"),
            ({"features": [{"name": "sma", "param": {"period": 2}}]}, "param"),
            ({"preprocesssing": {"scaler": "robust"}}, "preprocessing"),
        ],
    )
    def test_validation_rejects_malformed_recognized_sections(
        self,
        tmp_path,
        document,
        match,
    ):
        path = tmp_path / "malformed.yaml"
        path.write_text(yaml.safe_dump(document))

        with pytest.raises(ValueError, match=match):
            load_experiment_config(path, validate=True)

    def test_mixed_valid_and_invalid_sections_fail_atomically(self, tmp_path):
        path = tmp_path / "mixed.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "features": [{"name": "rsi"}],
                    "labeling": {"method": "triple_barrier"},
                    "preprocessing": {"scaleer": "robust"},
                }
            )
        )

        with pytest.raises(ValueError, match="scaleer"):
            load_experiment_config(path, validate=True)

    def test_validate_false_still_rejects_wrong_section_shapes(self, tmp_path):
        path = tmp_path / "malformed.yaml"
        path.write_text(yaml.safe_dump({"labeling": "triple_barrier"}))

        with pytest.raises(ValueError, match="labeling"):
            load_experiment_config(path, validate=False)

    @pytest.mark.parametrize(
        "holding_period",
        [
            {"__ml4t_type__": "timedelta", "microseconds": 1.5},
            {
                "__ml4t_type__": "timedelta",
                "microseconds": 1,
                "unexpected": True,
            },
        ],
    )
    def test_malformed_duration_encodings_are_rejected(self, tmp_path, holding_period):
        path = tmp_path / "duration.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "labeling": {"max_holding_period": holding_period},
                }
            )
        )

        with pytest.raises(ValueError, match="max_holding_period"):
            load_experiment_config(path)

    def test_unknown_schema_version_is_rejected_even_without_value_validation(self, tmp_path):
        path = tmp_path / "future.yaml"
        path.write_text(yaml.safe_dump({"schema_version": 2, "features": []}))

        with pytest.raises(ValueError, match="schema_version"):
            load_experiment_config(path, validate=False)


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

    def test_invalid_config_does_not_replace_existing_file(self, tmp_path):
        path = tmp_path / "existing.yaml"
        path.write_text("existing: true\n")
        config = ExperimentConfig(features=[{"name": "sma", "param": {"period": 2}}])

        with pytest.raises(ValueError, match="param"):
            save_experiment_config(config, path)

        assert path.read_text() == "existing: true\n"


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

    def test_roundtrip_timedelta_uses_safe_portable_yaml(self, tmp_path):
        original = ExperimentConfig(
            labeling=LabelingConfig.triple_barrier(
                max_holding_period=timedelta(hours=4, microseconds=500),
            )
        )
        path = tmp_path / "duration.yaml"

        save_experiment_config(original, path, include_defaults=True)
        raw = yaml.safe_load(path.read_text())
        loaded = load_experiment_config(path)

        assert raw["schema_version"] == 1
        assert loaded.labeling is not None
        assert loaded.labeling.max_holding_period == timedelta(hours=4, microseconds=500)
        assert isinstance(loaded.labeling.max_holding_period, timedelta)

    def test_custom_sections_survive_typed_section_edits(self, tmp_path):
        path = tmp_path / "custom.yaml"
        custom = {
            "features": [{"name": "rsi"}],
            "execution": {
                "seed": 7,
                "venue": "paper",
                "nested": {"retries": [1, 2, 3]},
            },
        }
        path.write_text(yaml.safe_dump(custom))
        config = load_experiment_config(path)
        config.features = [{"name": "sma", "params": {"period": 20}}]

        save_experiment_config(config, path)
        saved = yaml.safe_load(path.read_text())

        assert saved["execution"] == custom["execution"]
        assert saved["features"] == config.features

    @pytest.mark.parametrize("holding_period", [20, "4h", timedelta(hours=4)])
    def test_every_holding_period_form_retains_its_type(self, tmp_path, holding_period):
        path = tmp_path / "holding-period.yaml"
        original = ExperimentConfig(
            labeling=LabelingConfig.triple_barrier(max_holding_period=holding_period)
        )

        save_experiment_config(original, path, include_defaults=True)
        loaded = load_experiment_config(path)

        assert loaded.labeling is not None
        assert loaded.labeling.max_holding_period == holding_period
        assert type(loaded.labeling.max_holding_period) is type(holding_period)

    def test_all_typed_fields_roundtrip_together(self, tmp_path):
        path = tmp_path / "complete.yaml"
        labeling = LabelingConfig(
            method="triple_barrier",
            price_col="settlement",
            timestamp_col="event_time",
            group_col=["venue", "symbol"],
            data_contract=DataContractConfig(
                timestamp_col="event_time",
                symbol_col=["venue", "symbol"],
                price_col="settlement",
            ),
            upper_barrier="take_profit",
            lower_barrier="stop_loss",
            max_holding_period=timedelta(days=2, seconds=3, microseconds=4),
            side="trade_side",
            trailing_stop="trailing_distance",
            weight_scheme="returns",
            weight_decay_rate=0.25,
            atr_tp_multiple=3.0,
            atr_sl_multiple=1.5,
            atr_period=21,
            horizon=15,
            return_method="binary",
            threshold=0.001,
            min_horizon=4,
            max_horizon=12,
            t_value_threshold=3.0,
            percentile_window=100,
            n_bins=5,
        )
        preprocessing = PreprocessingConfig(
            scaler="robust",
            columns=["rsi", "atr"],
            with_mean=False,
            with_std=False,
            feature_range=(-1.0, 1.0),
            with_centering=False,
            with_scaling=False,
            quantile_range=(10.0, 90.0),
        )
        original = ExperimentConfig(
            features=[
                {
                    "name": "sma",
                    "params": {"period": 20},
                    "output": "sma_20",
                }
            ],
            labeling=labeling,
            preprocessing=preprocessing,
        )

        save_experiment_config(original, path, include_defaults=True)
        loaded = load_experiment_config(path)

        assert loaded.features == original.features
        assert loaded.labeling == labeling
        assert loaded.preprocessing == preprocessing


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
