"""Tests for the config-driven feature computation API.

Tests cover:
- Feature computation with different input formats
- Dependency resolution and execution order
- Parameter overrides
- Error handling
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import ml4t.engineer.api as api
from ml4t.engineer.api import compute_features
from ml4t.engineer.core.exceptions import DataSchemaError
from ml4t.engineer.core.registry import FeatureMetadata, FeatureRegistry, get_registry


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(50)],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0] * 10,
            "high": [102.0, 103.0, 104.0, 105.0, 106.0] * 10,
            "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 10,
            "close": [101.0, 102.0, 103.0, 104.0, 105.0] * 10,
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0] * 10,
        }
    )


# Basic API tests


def test_compute_single_feature_with_defaults(sample_ohlcv_data):
    """Test computing a single feature using registry default parameters."""
    # RSI has parameters={"period": 14} in its @feature() registration
    result = compute_features(sample_ohlcv_data, ["rsi"])

    assert isinstance(result, pl.DataFrame)
    assert "rsi" in result.columns


def test_compute_multiple_features(sample_ohlcv_data):
    """Test computing multiple features."""
    features = [
        {"name": "sma", "params": {"period": 20}},
        {"name": "ema", "params": {"period": 10}},
    ]
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
        "rsi",  # Has parameters={"period": 14} in registration
        {"name": "ema", "params": {"period": 15}},  # Custom parameters
    ]
    result = compute_features(sample_ohlcv_data, features)

    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) > len(sample_ohlcv_data.columns)


# LazyFrame support


def test_compute_with_lazyframe(sample_ohlcv_data):
    """Test that API works with LazyFrame input."""
    lazy_data = sample_ohlcv_data.lazy()
    result = compute_features(lazy_data, [{"name": "sma", "params": {"period": 20}}])

    assert isinstance(result, pl.LazyFrame)
    # Collect to verify computation works
    collected = result.collect()
    assert len(collected.columns) > len(sample_ohlcv_data.columns)


# Dependency resolution tests


def test_empty_feature_list(sample_ohlcv_data):
    """Test that empty feature list returns original data."""
    result = compute_features(sample_ohlcv_data, [])
    no_time_data = sample_ohlcv_data.drop("timestamp")
    no_time_result = compute_features(no_time_data, [])

    assert isinstance(result, pl.DataFrame)
    assert result.equals(sample_ohlcv_data)
    assert no_time_result.equals(no_time_data)


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

    # Use features that have default parameters in their registration
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
- rsi
- macd
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
    # Override one function parameter and retain the other function default.
    features = [
        {"name": "macd", "params": {"fast_period": 8}},
    ]

    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)


# Edge cases


def test_duplicate_feature_names(sample_ohlcv_data):
    """Duplicate outputs fail instead of silently discarding a request."""
    features = [
        {"name": "sma", "params": {"period": 10}},
        {"name": "sma", "params": {"period": 20}},
    ]

    with pytest.raises(ValueError, match="Duplicate feature output 'sma'"):
        compute_features(sample_ohlcv_data, features)


def test_duplicate_features_require_explicit_distinct_outputs(sample_ohlcv_data):
    """Multiple configurations of one feature retain both requested outputs."""
    result = compute_features(
        sample_ohlcv_data,
        [
            {"name": "sma", "params": {"period": 10}, "output": "sma_10"},
            {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
        ],
    )

    expected_10 = sample_ohlcv_data.select(pl.col("close").rolling_mean(10)).to_series()
    expected_20 = sample_ohlcv_data.select(pl.col("close").rolling_mean(20)).to_series()
    assert result["sma_10"].equals(expected_10)
    assert result["sma_20"].equals(expected_20)


def test_unknown_feature_parameter_fails_before_execution(sample_ohlcv_data):
    """A misspelled parameter never falls back to the registered default."""
    with pytest.raises(ValueError, match=r"Feature 'sma'.*unknown parameter.*perod"):
        compute_features(
            sample_ohlcv_data,
            [
                {"name": "ema", "params": {"period": 5}},
                {"name": "sma", "params": {"perod": 2}},
            ],
        )


def test_output_cannot_replace_input_column(sample_ohlcv_data):
    """Feature output names cannot silently overwrite source data."""
    with pytest.raises(ValueError, match="conflicts with an input column"):
        compute_features(
            sample_ohlcv_data,
            [{"name": "sma", "params": {"period": 2}, "output": "close"}],
        )


def test_compute_requires_time_or_explicit_order_assumption(sample_ohlcv_data):
    """Row-order semantics must be explicit when no time column exists."""
    data = sample_ohlcv_data.drop("timestamp")

    with pytest.raises(DataSchemaError, match="No time column found"):
        compute_features(data, [{"name": "sma", "params": {"period": 2}}])

    result = compute_features(
        data,
        [{"name": "sma", "params": {"period": 2}}],
        assume_sorted=True,
    )
    assert "sma" in result.columns


def test_compute_isolates_assets_and_sorts_each_history():
    """Interleaved, unsorted panel rows match independent per-asset calculations."""
    start = datetime(2024, 1, 1)
    data = pl.DataFrame(
        {
            "timestamp": [
                start + timedelta(days=1),
                start + timedelta(days=1),
                start,
                start,
            ],
            "asset_id": ["A", "B", "A", "B"],
            "open": [12.0, 110.0, 10.0, 100.0],
            "high": [12.0, 110.0, 10.0, 100.0],
            "low": [12.0, 110.0, 10.0, 100.0],
            "close": [12.0, 110.0, 10.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = compute_features(data, [{"name": "sma", "params": {"period": 2}}])

    assert result.select("asset_id", "timestamp").rows() == [
        ("A", start),
        ("A", start + timedelta(days=1)),
        ("B", start),
        ("B", start + timedelta(days=1)),
    ]
    assert result["sma"].to_list() == [None, 11.0, None, 105.0]


def test_compute_cannot_disable_detected_asset_grouping():
    """A recognized panel cannot be routed through one shared history."""
    start = datetime(2024, 1, 1)
    data = pl.DataFrame(
        {
            "timestamp": [start, start],
            "asset_id": ["A", "B"],
            "open": [10.0, 100.0],
            "high": [10.0, 100.0],
            "low": [10.0, 100.0],
            "close": [10.0, 100.0],
            "volume": [1.0, 1.0],
        }
    )

    with pytest.raises(DataSchemaError, match="Grouping cannot be disabled"):
        compute_features(data, ["sma"], group_col=[])


def test_lazy_panel_matches_eager_panel():
    """Lazy and eager panel execution use the same grouping and ordering contract."""
    start = datetime(2024, 1, 1)
    data = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i // 2) for i in range(8)],
            "asset_id": ["A", "B"] * 4,
            "open": [10.0, 100.0, 11.0, 102.0, 12.0, 104.0, 13.0, 106.0],
            "high": [10.0, 100.0, 11.0, 102.0, 12.0, 104.0, 13.0, 106.0],
            "low": [10.0, 100.0, 11.0, 102.0, 12.0, 104.0, 13.0, 106.0],
            "close": [10.0, 100.0, 11.0, 102.0, 12.0, 104.0, 13.0, 106.0],
            "volume": [1.0] * 8,
        }
    )
    features = [
        {"name": "sma", "params": {"period": 2}},
        {"name": "roc", "params": {"period": 1}},
    ]

    eager = compute_features(data, features)
    lazy = compute_features(data.lazy(), features)

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(eager)


def test_features_with_empty_params(sample_ohlcv_data):
    """Test features with explicit empty params dict uses registration defaults."""
    # RSI has parameters={"period": 14} in registration, so empty params works
    features = [
        {"name": "rsi", "params": {}},
    ]

    result = compute_features(sample_ohlcv_data, features)
    assert isinstance(result, pl.DataFrame)


def test_features_without_params_raises_clear_error(sample_ohlcv_data):
    """Test that unregistered features raise a clear error."""
    with pytest.raises(ValueError, match="not found in registry"):
        compute_features(sample_ohlcv_data, ["nonexistent_feature_xyz"])


@pytest.mark.parametrize(
    ("feature_spec", "match"),
    [
        ([1], "every list item"),
        ([{"name": "sma", "unknown": True}], "Unknown feature spec fields"),
        ([{"name": ""}], "non-empty string"),
        ([{"name": "sma", "params": []}], "params must be a mapping"),
        ([{"name": "sma", "output": 1}], "output must be a non-empty string"),
    ],
)
def test_malformed_feature_specs_are_rejected_before_execution(
    sample_ohlcv_data: pl.DataFrame,
    feature_spec: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_features(sample_ohlcv_data, feature_spec)  # type: ignore[arg-type]


def test_yaml_document_requires_a_feature_list(
    sample_ohlcv_data: pl.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("metadata: {}\n")

    with pytest.raises(ValueError, match="Invalid YAML format"):
        compute_features(sample_ohlcv_data, path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"group_col": "missing"}, "Grouping columns not found"),
        ({"timestamp_col": "missing"}, "Timestamp column 'missing' not found"),
    ],
)
def test_explicit_ordering_columns_must_exist(
    sample_ohlcv_data: pl.DataFrame,
    kwargs: dict[str, str],
    match: str,
) -> None:
    with pytest.raises(DataSchemaError, match=match):
        compute_features(sample_ohlcv_data, ["sma"], **kwargs)  # type: ignore[arg-type]


def test_explicit_group_and_timestamp_columns_are_honored() -> None:
    data = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 2), datetime(2024, 1, 1)],
            "instrument": ["A", "A"],
            "open": [2.0, 1.0],
            "high": [2.0, 1.0],
            "low": [2.0, 1.0],
            "close": [2.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )

    result = compute_features(
        data,
        [{"name": "sma", "params": {"period": 2}}],
        group_col="instrument",
        timestamp_col="event_time",
    )

    assert result["event_time"].to_list() == sorted(data["event_time"].to_list())
    assert result["sma"].to_list() == [None, 1.5]


def test_ordering_columns_require_temporal_dtype_and_non_null_values(
    sample_ohlcv_data: pl.DataFrame,
) -> None:
    invalid_dtype = sample_ohlcv_data.with_columns(pl.col("timestamp").cast(pl.String))
    with pytest.raises(DataSchemaError, match="must be .*Date"):
        compute_features(invalid_dtype, ["sma"])

    null_group = sample_ohlcv_data.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.lit("A")).alias("asset_id")
    )
    with pytest.raises(DataSchemaError, match="cannot contain nulls"):
        compute_features(null_group, ["sma"])


def _registry_with_feature(
    func: object,
    *,
    name: str = "test_feature",
    parameters: dict[str, object] | None = None,
    dependencies: list[str] | None = None,
) -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register(
        FeatureMetadata(
            name=name,
            func=func,  # type: ignore[arg-type]
            category="ml",
            description="API dispatch contract fixture",
            parameters=parameters or {},
            dependencies=dependencies or [],
        )
    )
    return registry


def test_dependency_resolution_rejects_unknown_and_cyclic_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FeatureRegistry()
    monkeypatch.setattr(api, "get_registry", lambda: registry)
    with pytest.raises(ValueError, match="not found in registry"):
        api._resolve_dependencies([{"name": "missing", "params": {}, "output": "missing"}])

    first = _registry_with_feature(
        lambda close: pl.col(close),
        name="first",
        dependencies=["second"],
    )
    first.register(
        FeatureMetadata(
            name="second",
            func=lambda close: pl.col(close),
            category="ml",
            description="Second cyclic feature",
            dependencies=["first"],
        )
    )
    monkeypatch.setattr(api, "get_registry", lambda: first)
    with pytest.raises(ValueError, match="Circular dependency"):
        api._resolve_dependencies(
            [
                {"name": "first", "params": {}, "output": "first"},
                {"name": "second", "params": {}, "output": "second"},
            ]
        )


def test_feature_dispatch_passes_columns_defaults_and_keyword_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, str, int, str]] = []

    def expression(
        close: str,
        high: str = "high",
        period: int = 2,
        *,
        implementation: str = "polars",
    ) -> pl.Expr:
        received.append((close, high, period, implementation))
        return pl.col(close).rolling_mean(period)

    registry = _registry_with_feature(
        expression,
        parameters={"period": 3, "implementation": "native"},
    )
    monkeypatch.setattr(api, "get_registry", lambda: registry)
    data = pl.DataFrame({"asset": ["A"] * 4, "close": [1.0, 2.0, 3.0, 4.0]})

    result = api._execute_feature(data, "test_feature", {}, "mean", ["asset"])

    assert received == [("close", "high", 3, "native")]
    assert result["mean"].to_list() == [None, None, 2.0, 3.0]


def test_feature_dispatch_reports_missing_parameters_and_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_parameter(close: str, required: int) -> pl.Expr:
        return pl.col(close) + required

    registry = _registry_with_feature(missing_parameter)
    monkeypatch.setattr(api, "get_registry", lambda: registry)
    data = pl.DataFrame({"close": [1.0]})
    with pytest.raises(ValueError, match="requires parameter 'required'"):
        api._execute_feature(data, "test_feature", {}, "result", [])
    with pytest.raises(ValueError, match="not found in registry"):
        api._execute_feature(data, "missing", {}, "result", [])

    def raises_type_error(close: str) -> pl.Expr:
        raise TypeError(f"cannot process {close}")

    registry = _registry_with_feature(raises_type_error)
    monkeypatch.setattr(api, "get_registry", lambda: registry)
    with pytest.raises(ValueError, match="Failed to execute feature"):
        api._execute_feature(data, "test_feature", {}, "result", [])


@pytest.mark.parametrize(
    ("result", "match"),
    [
        ({"invalid": 1}, "dict without Expr"),
        ([1, "invalid"], "tuple/list without Expr"),
        (1, "unexpected type"),
    ],
)
def test_feature_dispatch_rejects_unsupported_results(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
    match: str,
) -> None:
    registry = _registry_with_feature(lambda close: result if close else result)
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    with pytest.raises((TypeError, ValueError), match=match):
        api._execute_feature(
            pl.DataFrame({"close": [1.0]}),
            "test_feature",
            {},
            "result",
            [],
        )


@pytest.mark.parametrize("as_mapping", [True, False])
def test_feature_dispatch_keeps_only_expression_outputs(
    monkeypatch: pytest.MonkeyPatch,
    as_mapping: bool,
) -> None:
    result: object
    if as_mapping:
        result = {"valid": pl.col("close") + 1, "ignored": 1}
    else:
        result = [pl.col("close") + 1, 1]
    registry = _registry_with_feature(lambda close: result if close else result)
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    output = api._execute_feature(
        pl.DataFrame({"close": [1.0]}),
        "test_feature",
        {},
        "result",
        [],
    )

    expected_column = "result_valid" if as_mapping else "result_0"
    assert output[expected_column].to_list() == [2.0]


def test_appending_feature_expressions_rejects_conflicts_and_duplicates() -> None:
    data = pl.DataFrame({"close": [1.0]})
    with pytest.raises(ValueError, match="conflict with existing columns"):
        api._append_feature_expressions(data, [("close", pl.lit(2.0))])
    with pytest.raises(ValueError, match="duplicate output columns"):
        api._append_feature_expressions(
            data,
            [("result", pl.lit(2.0)), ("result", pl.lit(3.0))],
        )
