"""Tests for the feature registry system.

Tests cover:
- Feature registration and retrieval
- Query methods (by category, stationarity, TA-Lib compatibility)
- Dependency tracking
- Error handling and edge cases
- Global registry singleton pattern
"""

import polars as pl
import pytest

from ml4t.engineer.core.registry import FeatureMetadata, FeatureRegistry, get_registry

# Test fixtures


@pytest.fixture(autouse=True)
def preserve_global_registry():
    """Preserve and restore global registry state for each test.

    This fixture automatically runs for every test in this file to prevent
    test isolation issues. Tests that call .clear() on the global registry
    will have it restored after the test completes.
    """
    global_reg = get_registry()

    # Save current state (shallow copy of the internal dict)
    saved_features = global_reg._features.copy()

    yield  # Run the test

    # Restore state after test
    global_reg._features.clear()
    global_reg._features.update(saved_features)


def dummy_feature_func(df: pl.DataFrame) -> pl.DataFrame:
    """Dummy feature function for testing."""
    return df.with_columns(pl.lit(1.0).alias("test_feature"))


def dependent_feature_func(df: pl.DataFrame) -> pl.DataFrame:
    """Dummy dependent feature function for testing."""
    return df.with_columns(pl.lit(2.0).alias("dependent_feature"))


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    reg = FeatureRegistry()
    return reg


@pytest.fixture
def sample_metadata():
    """Create sample feature metadata."""
    return FeatureMetadata(
        name="test_rsi",
        func=dummy_feature_func,
        category="momentum",
        description="Test RSI indicator",
        formula="RSI = 100 - (100 / (1 + RS))",
        normalized=True,
        lookback=14,
        ta_lib_compatible=True,
        input_type="close",
        output_type="indicator",
        parameters={"period": 14},
        dependencies=[],
        references=["Wilder (1978)"],
        tags=["oscillator", "momentum"],
    )


@pytest.fixture
def populated_registry(registry):
    """Create a registry with several test features."""
    # Momentum features
    registry.register(
        FeatureMetadata(
            name="rsi",
            func=dummy_feature_func,
            category="momentum",
            description="Relative Strength Index",
            normalized=True,
            lookback=14,
            ta_lib_compatible=True,
            parameters={"period": 14},
            tags=["oscillator"],
        )
    )

    registry.register(
        FeatureMetadata(
            name="macd",
            func=dummy_feature_func,
            category="momentum",
            description="MACD indicator",
            normalized=False,
            lookback=26,
            ta_lib_compatible=True,
            parameters={"fast": 12, "slow": 26, "signal": 9},
            tags=["trend"],
        )
    )

    # Volatility features
    registry.register(
        FeatureMetadata(
            name="atr",
            func=dummy_feature_func,
            category="volatility",
            description="Average True Range",
            normalized=False,
            lookback=14,
            ta_lib_compatible=True,
            input_type="OHLC",
            parameters={"period": 14},
            tags=["range"],
        )
    )

    registry.register(
        FeatureMetadata(
            name="garch_forecast",
            func=dummy_feature_func,
            category="volatility",
            description="GARCH volatility forecast",
            normalized=False,
            lookback=100,
            ta_lib_compatible=False,
            input_type="returns",
            parameters={"omega": 0.00001, "alpha": 0.1, "beta": 0.85},
            tags=["academic", "forecast"],
        )
    )

    # Feature with dependencies
    registry.register(
        FeatureMetadata(
            name="rsi_divergence",
            func=dependent_feature_func,
            category="ml",
            description="RSI divergence signal",
            normalized=True,
            lookback=50,
            ta_lib_compatible=False,
            dependencies=["rsi"],
            tags=["divergence", "signal"],
        )
    )

    return registry


# Basic registration tests


def test_registry_initialization(registry):
    """Test registry starts empty."""
    assert len(registry) == 0
    assert registry.list_all() == []


def test_register_feature(registry, sample_metadata):
    """Test registering a single feature."""
    registry.register(sample_metadata)

    assert len(registry) == 1
    assert "test_rsi" in registry
    assert registry.get("test_rsi") == sample_metadata


def test_register_duplicate_name_raises_error(registry, sample_metadata):
    """Test registering duplicate feature name raises ValueError."""
    registry.register(sample_metadata)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(sample_metadata)


def test_register_multiple_features(registry):
    """Test registering multiple features."""
    for i in range(5):
        registry.register(
            FeatureMetadata(
                name=f"feature_{i}",
                func=dummy_feature_func,
                category="test",
                description=f"Test feature {i}",
            )
        )

    assert len(registry) == 5
    assert registry.list_all() == ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]


# Retrieval tests


def test_get_existing_feature(populated_registry):
    """Test retrieving existing feature metadata."""
    metadata = populated_registry.get("rsi")

    assert metadata is not None
    assert metadata.name == "rsi"
    assert metadata.category == "momentum"
    assert metadata.normalized is True


def test_get_nonexistent_feature(populated_registry):
    """Test retrieving nonexistent feature returns None."""
    assert populated_registry.get("nonexistent") is None


def test_contains_operator(populated_registry):
    """Test __contains__ operator (in keyword)."""
    assert "rsi" in populated_registry
    assert "macd" in populated_registry
    assert "nonexistent" not in populated_registry


def test_len_operator(populated_registry):
    """Test __len__ operator."""
    assert len(populated_registry) == 5  # 5 features registered in fixture


# List all tests


def test_list_all_empty(registry):
    """Test list_all on empty registry."""
    assert registry.list_all() == []


def test_list_all_returns_sorted(populated_registry):
    """Test list_all returns sorted feature names."""
    features = populated_registry.list_all()

    assert features == sorted(features)
    assert "atr" in features
    assert "rsi" in features
    assert "macd" in features


# Category queries


def test_list_by_category_momentum(populated_registry):
    """Test listing momentum features."""
    momentum = populated_registry.list_by_category("momentum")

    assert len(momentum) == 2
    assert "rsi" in momentum
    assert "macd" in momentum
    assert "atr" not in momentum


def test_list_by_category_volatility(populated_registry):
    """Test listing volatility features."""
    volatility = populated_registry.list_by_category("volatility")

    assert len(volatility) == 2
    assert "atr" in volatility
    assert "garch_forecast" in volatility


def test_list_by_category_empty(populated_registry):
    """Test listing features in nonexistent category."""
    assert populated_registry.list_by_category("nonexistent") == []


def test_list_by_category_returns_sorted(populated_registry):
    """Test list_by_category returns sorted names."""
    volatility = populated_registry.list_by_category("volatility")
    assert volatility == sorted(volatility)


# Stationarity queries


def test_list_normalized_features(populated_registry):
    """Test listing stationary features."""
    stationary = populated_registry.list_normalized()

    assert len(stationary) == 2
    assert "rsi" in stationary
    assert "rsi_divergence" in stationary
    assert "macd" not in stationary
    assert "atr" not in stationary


def test_list_normalized_empty(registry):
    """Test list_normalized on registry with no stationary features."""
    registry.register(
        FeatureMetadata(
            name="non_stationary",
            func=dummy_feature_func,
            category="test",
            description="Non-stationary test",
            normalized=False,
        )
    )

    assert registry.list_normalized() == []


# TA-Lib compatibility queries


def test_list_ta_lib_compatible(populated_registry):
    """Test listing TA-Lib compatible features."""
    compatible = populated_registry.list_ta_lib_compatible()

    assert len(compatible) == 3
    assert "rsi" in compatible
    assert "macd" in compatible
    assert "atr" in compatible
    assert "garch_forecast" not in compatible


def test_list_ta_lib_compatible_empty(registry):
    """Test list_ta_lib_compatible with no compatible features."""
    registry.register(
        FeatureMetadata(
            name="custom_feature",
            func=dummy_feature_func,
            category="test",
            description="Custom feature",
            ta_lib_compatible=False,
        )
    )

    assert registry.list_ta_lib_compatible() == []


# Dependency queries


def test_get_dependencies_no_dependencies(populated_registry):
    """Test getting dependencies for feature with none."""
    deps = populated_registry.get_dependencies("rsi")

    assert deps == []


def test_get_dependencies_with_dependencies(populated_registry):
    """Test getting dependencies for dependent feature."""
    deps = populated_registry.get_dependencies("rsi_divergence")

    assert len(deps) == 1
    assert "rsi" in deps


def test_get_dependencies_nonexistent_feature(populated_registry):
    """Test getting dependencies for nonexistent feature raises KeyError."""
    with pytest.raises(KeyError, match="not found in registry"):
        populated_registry.get_dependencies("nonexistent")


def test_get_dependencies_returns_copy(populated_registry):
    """Test get_dependencies returns a copy, not original list."""
    deps1 = populated_registry.get_dependencies("rsi_divergence")
    deps2 = populated_registry.get_dependencies("rsi_divergence")

    # Modifying one shouldn't affect the other
    deps1.append("fake_dependency")
    assert deps2 == ["rsi"]


# Clear and reset tests


def test_clear_registry(populated_registry):
    """Test clearing all features from registry."""
    assert len(populated_registry) > 0

    populated_registry.clear()

    assert len(populated_registry) == 0
    assert populated_registry.list_all() == []


def test_clear_empty_registry(registry):
    """Test clearing already empty registry."""
    registry.clear()
    assert len(registry) == 0


# String representation


def test_repr(populated_registry):
    """Test __repr__ method."""
    repr_str = repr(populated_registry)

    assert "FeatureRegistry" in repr_str
    assert "features=5" in repr_str


# Global registry tests


def test_get_global_registry():
    """Test getting global registry instance."""
    reg1 = get_registry()
    reg2 = get_registry()

    # Should return same instance (singleton)
    assert reg1 is reg2


def test_global_registry_persistence():
    """Test global registry persists across get_registry calls."""
    reg = get_registry()

    # Clear to start fresh
    reg.clear()

    # Register a feature
    reg.register(
        FeatureMetadata(
            name="global_test",
            func=dummy_feature_func,
            category="test",
            description="Global test feature",
        )
    )

    # Get registry again and check feature is there
    reg2 = get_registry()
    assert "global_test" in reg2

    # Clean up
    reg.clear()


# Edge cases and validation


def test_metadata_with_minimal_fields():
    """Test creating metadata with only required fields."""
    metadata = FeatureMetadata(
        name="minimal",
        func=dummy_feature_func,
        category="test",
        description="Minimal metadata",
    )

    # Check defaults are applied
    assert metadata.formula == ""
    assert metadata.normalized is False
    assert callable(metadata.lookback)
    assert metadata.lookback() == 0  # Default lookback returns 0
    assert metadata.ta_lib_compatible is False
    assert metadata.parameters == {}
    assert metadata.dependencies == []
    assert metadata.references == []
    assert metadata.tags == []


def test_metadata_with_all_fields(sample_metadata):
    """Test creating metadata with all fields populated."""
    assert sample_metadata.name == "test_rsi"
    assert sample_metadata.func == dummy_feature_func
    assert sample_metadata.category == "momentum"
    assert sample_metadata.description == "Test RSI indicator"
    assert sample_metadata.formula == "RSI = 100 - (100 / (1 + RS))"
    assert sample_metadata.normalized is True
    # Lookback can be int or callable - handle both
    if callable(sample_metadata.lookback):
        assert sample_metadata.lookback() == 14
    else:
        assert sample_metadata.lookback == 14
    assert sample_metadata.ta_lib_compatible is True
    assert sample_metadata.input_type == "close"
    assert sample_metadata.output_type == "indicator"
    assert sample_metadata.parameters == {"period": 14}
    assert sample_metadata.dependencies == []
    assert sample_metadata.references == ["Wilder (1978)"]
    assert sample_metadata.tags == ["oscillator", "momentum"]


def test_registry_with_real_features():
    """Test registry integration with real registered features."""

    registry = get_registry()

    # Should have all registered features (count may vary as features are added)
    # As of v0.3.0: 107 features registered
    assert len(registry) >= 100  # At least 100 features should be registered

    # Test specific features exist
    assert "rsi" in registry
    assert "macd" in registry
    assert "atr" in registry
    assert "kyle_lambda" in registry

    # Test category queries work
    momentum = registry.list_by_category("momentum")
    assert len(momentum) == 31

    volatility = registry.list_by_category("volatility")
    assert len(volatility) == 15

    # Test stationarity queries
    stationary = registry.list_normalized()
    assert len(stationary) >= 9  # Based on actual registration (v0.3.0: 9)
    assert "rsi" in stationary

    # Test TA-Lib compatibility
    compatible = registry.list_ta_lib_compatible()
    assert len(compatible) >= 59  # Based on actual registration (v0.3.0: 59)
    assert "rsi" in compatible


def test_query_methods_return_sorted_lists(populated_registry):
    """Test all query methods return sorted lists."""
    # All list methods should return sorted results
    assert populated_registry.list_all() == sorted(populated_registry.list_all())
    assert populated_registry.list_normalized() == sorted(populated_registry.list_normalized())
    assert populated_registry.list_ta_lib_compatible() == sorted(
        populated_registry.list_ta_lib_compatible()
    )
    assert populated_registry.list_by_category("momentum") == sorted(
        populated_registry.list_by_category("momentum")
    )
