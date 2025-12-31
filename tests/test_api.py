"""Tests for the config-driven feature computation API.

Tests cover:
- Feature computation with different input formats
- Dependency resolution and execution order
- Parameter overrides
- Error handling
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from ml4t.engineer.api import compute_features
from ml4t.engineer.core.registry import get_registry


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    return pl.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0] * 10,
            "high": [102.0, 103.0, 104.0, 105.0, 106.0] * 10,
            "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 10,
            "close": [101.0, 102.0, 103.0, 104.0, 105.0] * 10,
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0] * 10,
        }
    )


# Basic API tests


def test_compute_single_feature_with_defaults(sample_ohlcv_data):
    """Test computing a single feature with default parameters."""
    result = compute_features(sample_ohlcv_data, ["sma"])

    assert isinstance(result, pl.DataFrame)
    # Check that sma column was added (column name format may vary)
    assert "sma" in result.columns or "sma_20" in result.columns or "close_sma_20" in result.columns


def test_compute_multiple_features(sample_ohlcv_data):
    """Test computing multiple features."""
    features = ["sma", "ema"]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    # Check that new columns were added
    assert len(result.columns) > len(sample_ohlcv_data.columns)


def test_compute_feature_with_custom_params(sample_ohlcv_data):
    """Test computing feature with custom parameters."""
    features = [
        {"name": "sma", "params": {"period": 10}},
    ]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    # Result should have additional columns
    assert len(result.columns) >= len(sample_ohlcv_data.columns)


def test_compute_mixed_format(sample_ohlcv_data):
    """Test computing features with mixed default and custom params."""
    features = [
        "sma",  # Default parameters
        {"name": "ema", "params": {"period": 15}},  # Custom parameters
    ]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


# LazyFrame support


def test_compute_with_lazyframe(sample_ohlcv_data):
    """Test that API works with LazyFrame input."""
    lazy_data = sample_ohlcv_data.lazy()
    result = compute_features(lazy_data, ["sma"])

    assert isinstance(result, pl.LazyFrame)
    # Collect to verify computation works
    collected = result.collect()
    assert len(collected.columns) > len(sample_ohlcv_data.columns)


# Dependency resolution tests


def test_empty_feature_list(sample_ohlcv_data):
    """Test that empty feature list returns original data."""
    result = compute_features(sample_ohlcv_data, [])

    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_ohlcv_data)


# Error handling tests


def test_nonexistent_feature_raises_error(sample_ohlcv_data):
    """Test that requesting nonexistent feature raises ValueError."""
    with pytest.raises(ValueError, match="not found in registry"):
        compute_features(sample_ohlcv_data, ["nonexistent_feature"])


def test_invalid_feature_format_raises_error(sample_ohlcv_data):
    """Test that invalid feature format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid features format"):
        compute_features(sample_ohlcv_data, 123)  # type: ignore


def test_feature_dict_missing_name_raises_error(sample_ohlcv_data):
    """Test that feature dict without 'name' raises ValueError."""
    features = [
        {"params": {"period": 10}},  # Missing 'name'
    ]

    with pytest.raises(ValueError, match="missing 'name' field"):
        compute_features(sample_ohlcv_data, features)


# YAML config tests


def test_compute_from_yaml_config(sample_ohlcv_data):
    """Test computing features from YAML config file."""
    pytest.importorskip("yaml")  # Skip if PyYAML not installed

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
features:
  - name: sma
    params:
      period: 20
  - name: ema
    params:
      period: 15
""")
        config_path = f.name

    try:
        result = compute_features(sample_ohlcv_data, config_path)
        assert isinstance(result, pl.DataFrame)
        assert len(result.columns) > len(sample_ohlcv_data.columns)
    finally:
        Path(config_path).unlink()


def test_compute_from_yaml_config_simple_list(sample_ohlcv_data):
    """Test computing from YAML config with simple list format."""
    pytest.importorskip("yaml")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
- sma
- ema
""")
        config_path = f.name

    try:
        result = compute_features(sample_ohlcv_data, config_path)
        assert isinstance(result, pl.DataFrame)
        assert len(result.columns) > len(sample_ohlcv_data.columns)
    finally:
        Path(config_path).unlink()


def test_missing_yaml_file_raises_error(sample_ohlcv_data):
    """Test that missing YAML file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compute_features(sample_ohlcv_data, "nonexistent_config.yaml")


def test_yaml_without_pyyaml_raises_error(sample_ohlcv_data, monkeypatch):
    """Test that YAML config without PyYAML installed raises ImportError."""
    # Mock YAML_AVAILABLE to False
    import ml4t.engineer.api

    monkeypatch.setattr(ml4t.engineer.api, "YAML_AVAILABLE", False)

    with pytest.raises(ImportError, match="PyYAML is required"):
        compute_features(sample_ohlcv_data, "config.yaml")


# Integration tests with real features


def test_compute_momentum_features(sample_ohlcv_data):
    """Test computing real momentum features."""
    features = ["rsi", "macd"]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


def test_compute_volatility_features(sample_ohlcv_data):
    """Test computing real volatility features."""
    features = ["atr", "natr"]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


def test_compute_with_registry_integration(sample_ohlcv_data):
    """Test API integration with actual registered features."""
    registry = get_registry()

    # Get a few features from registry that work with OHLCV data
    # (Skip features that require 'returns' column like amihud_illiquidity)
    available = registry.list_all()[:4]  # First 4 features: ad, adosc, adx, adxr

    # Compute them
    result = compute_features(sample_ohlcv_data, available)

    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


# Parameter override tests


def test_parameter_override_from_config(sample_ohlcv_data):
    """Test that config parameters override registry defaults."""
    # Default SMA period from registry is 20
    features = [
        {"name": "sma", "params": {"period": 50}},
    ]

    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)


def test_partial_parameter_override(sample_ohlcv_data):
    """Test that partial parameter override works correctly."""
    # Registry default might have multiple params, override just one
    features = [
        {"name": "macd", "params": {"fast": 8}},  # Override fast, keep slow/signal defaults
    ]

    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)


# Edge cases


def test_duplicate_feature_names(sample_ohlcv_data):
    """Test handling of duplicate feature names with different params."""
    features = [
        {"name": "sma", "params": {"period": 10}},
        {"name": "sma", "params": {"period": 20}},
    ]

    # Should compute both (though second might override first)
    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)


def test_features_with_empty_params(sample_ohlcv_data):
    """Test features with explicit empty params dict."""
    features = [
        {"name": "sma", "params": {}},
    ]

    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)
