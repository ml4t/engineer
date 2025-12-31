"""Integration test to verify ALL 107 registered features work end-to-end.

This test prevents regression by ensuring every feature in the registry can be
successfully computed via the compute_features() API.

Created: 2025-11-03
Purpose: Prevent the kind of fundamental breakage that occurred where features
         were registered but couldn't be called due to API-feature signature mismatches.
"""

import polars as pl
import pytest

from ml4t.engineer.api import compute_features
from ml4t.engineer.core.registry import get_registry


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data with sufficient rows for all features."""
    return pl.DataFrame(
        {
            "open": list(range(99, 149)),
            "high": list(range(102, 152)),
            "low": list(range(99, 149)),
            "close": list(range(100, 150)),
            "volume": list(range(1000, 1050)),
            # Order book columns for microstructure features
            "bid_price": [x - 0.5 for x in range(100, 150)],
            "ask_price": [x + 0.5 for x in range(100, 150)],
            "bid_size": list(range(500, 550)),
            "ask_size": list(range(450, 500)),
        }
    ).with_columns(
        [
            (pl.col("close").pct_change()).alias("returns"),
        ]
    )


def test_all_features_registered():
    """Verify that features are actually registered."""
    registry = get_registry()
    assert len(registry) == 120, f"Expected 120 features, got {len(registry)}"


def test_all_features_compute_individually(sample_ohlcv_data):
    """Test that EVERY registered feature can be computed successfully.

    This is the critical integration test that would have caught the original
    API-feature signature mismatch issue.
    """
    registry = get_registry()
    all_features = registry.list_all()

    failures = []

    for feature_name in sorted(all_features):
        try:
            result = compute_features(sample_ohlcv_data, [feature_name])

            # Verify result is valid
            assert result is not None, f"{feature_name} returned None"
            assert len(result) == len(sample_ohlcv_data), f"{feature_name} changed row count"

        except Exception as e:
            failures.append((feature_name, str(e)))

    # Report all failures at once
    if failures:
        failure_msg = "\n".join([f"  {name}: {error[:100]}" for name, error in failures])
        pytest.fail(f"\n{len(failures)}/{len(all_features)} features failed:\n{failure_msg}")


def test_all_features_compute_together(sample_ohlcv_data):
    """Test that all features can be computed together in one call."""
    registry = get_registry()
    all_features = registry.list_all()

    # Compute all features at once
    result = compute_features(sample_ohlcv_data, all_features)

    # Verify result
    assert result is not None
    assert len(result) == len(sample_ohlcv_data)

    # Verify we have more columns than we started with
    original_cols = set(sample_ohlcv_data.columns)
    result_cols = set(result.columns)
    new_cols = result_cols - original_cols

    # Should have created feature columns (exact count may vary due to multi-output features)
    assert len(new_cols) >= len(all_features), (
        f"Expected at least {len(all_features)} new columns, got {len(new_cols)}"
    )


def test_compute_features_with_parameters(sample_ohlcv_data):
    """Test that parameter overrides work correctly."""
    # Test with list of dicts format (parameter overrides)
    features = [
        {"name": "rsi", "params": {"period": 20}},
        {"name": "sma", "params": {"period": 50}},
        {"name": "ema", "params": {"period": 30}},
    ]

    result = compute_features(sample_ohlcv_data, features)

    assert result is not None
    assert len(result) == len(sample_ohlcv_data)


def test_compute_features_with_dependencies(sample_ohlcv_data):
    """Test that features with dependencies are resolved correctly."""
    # These features might have dependencies on each other
    features = ["rsi", "macd", "bollinger_bands", "atr"]

    result = compute_features(sample_ohlcv_data, features)

    assert result is not None
    assert len(result) == len(sample_ohlcv_data)


@pytest.mark.parametrize(
    "category",
    [
        "momentum",
        "trend",
        "volatility",
        "volume",
        "statistics",
        "math",
        "price_transform",
        "microstructure",
        "ml",
    ],
)
def test_features_by_category(sample_ohlcv_data, category):
    """Test all features in each category."""
    registry = get_registry()
    features = registry.list_by_category(category)

    if not features:
        pytest.skip(f"No features in category '{category}'")

    failures = []

    for feature_name in features:
        try:
            result = compute_features(sample_ohlcv_data, [feature_name])
            assert result is not None
        except Exception as e:
            failures.append((feature_name, str(e)))

    if failures:
        failure_msg = "\n".join([f"  {name}: {error[:100]}" for name, error in failures])
        pytest.fail(
            f"\n{len(failures)}/{len(features)} features in '{category}' failed:\n{failure_msg}"
        )


def test_registry_metadata_completeness():
    """Verify all registered features have complete metadata."""
    registry = get_registry()

    for feature_name in registry.list_all():
        metadata = registry.get(feature_name)

        # Check required fields are present
        assert metadata.name == feature_name
        assert metadata.func is not None
        assert metadata.category is not None
        assert metadata.description is not None
        assert metadata.input_type is not None
        assert metadata.output_type is not None
        assert isinstance(metadata.parameters, dict)
        assert isinstance(metadata.dependencies, list)
        assert isinstance(metadata.references, list)
        assert isinstance(metadata.tags, list)
        # Lookback can be int (0 for no lookback) or callable (for dynamic lookback based on params)
        assert isinstance(metadata.lookback, int | type(lambda: None))
        assert isinstance(metadata.normalized, bool)
        assert isinstance(metadata.ta_lib_compatible, bool)


def test_no_duplicate_registrations():
    """Verify no features are registered multiple times."""
    registry = get_registry()
    all_features = registry.list_all()

    # Check for duplicates
    assert len(all_features) == len(set(all_features)), "Duplicate feature names found in registry"


if __name__ == "__main__":
    # Allow running this test file directly for quick verification
    import sys

    df = pl.DataFrame(
        {
            "open": list(range(99, 149)),
            "high": list(range(102, 152)),
            "low": list(range(99, 149)),
            "close": list(range(100, 150)),
            "volume": list(range(1000, 1050)),
            # Order book columns for microstructure features
            "bid_price": [x - 0.5 for x in range(100, 150)],
            "ask_price": [x + 0.5 for x in range(100, 150)],
            "bid_size": list(range(500, 550)),
            "ask_size": list(range(450, 500)),
        }
    ).with_columns(
        [
            (pl.col("close").pct_change()).alias("returns"),
        ]
    )

    print("Testing all features...")
    registry = get_registry()
    all_features = registry.list_all()

    working = []
    failing = []

    for feature_name in sorted(all_features):
        try:
            result = compute_features(df, [feature_name])
            working.append(feature_name)
            print(f"✓ {feature_name}")
        except Exception as e:
            failing.append((feature_name, str(e)))
            print(f"✗ {feature_name}: {str(e)[:80]}")

    print(f"\n{'=' * 60}")
    print(
        f"✓ Working: {len(working)}/{len(all_features)} ({100 * len(working) // len(all_features)}%)"
    )
    print(f"✗ Failing: {len(failing)}/{len(all_features)}")

    if failing:
        print("\nFailing features:")
        for name, error in failing:
            print(f"  {name}: {error[:100]}")
        sys.exit(1)
    else:
        print("\n🎉 ALL FEATURES WORKING!")
        sys.exit(0)
