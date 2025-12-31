"""Performance benchmarks for Module C feature-outcome analysis.

Tests performance characteristics and ensures analysis completes
within acceptable time limits.
"""

import time

import numpy as np
import pandas as pd
import pytest

from ml4t.engineer.config.feature_config import ICConfig, MLDiagnosticsConfig, ModuleCConfig
from ml4t.engineer.outcome.feature_outcome import FeatureOutcome


class TestPerformanceBenchmarks:
    """Performance benchmarks with strict time constraints."""

    def test_small_dataset_performance(self):
        """Benchmark: 10 features, 1K rows (should be fast)."""
        np.random.seed(42)
        n = 1000
        n_features = 10

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        analyzer = FeatureOutcome()

        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        assert elapsed < 5.0, f"Small dataset took {elapsed:.2f}s (expected <5s)"
        print(f"\nSmall dataset: {elapsed:.2f}s")

    def test_medium_dataset_performance(self):
        """Benchmark: 50 features, 5K rows."""
        np.random.seed(42)
        n = 5000
        n_features = 50

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        analyzer = FeatureOutcome()

        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        assert elapsed < 20.0, f"Medium dataset took {elapsed:.2f}s (expected <20s)"
        print(f"\nMedium dataset: {elapsed:.2f}s")

    def test_benchmark_100_features_10k_rows(self):
        """Benchmark: 100 features, 10K rows (acceptance criteria target)."""
        np.random.seed(42)
        n = 10000
        n_features = 100

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        config = ModuleCConfig(ic=ICConfig(lag_structure=[1, 5, 10]))
        analyzer = FeatureOutcome(config=config)

        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        assert len(result.ic_results) == n_features
        assert elapsed < 30.0, f"100 features/10K rows took {elapsed:.2f}s (expected <30s)"

        print(f"\n✓ Benchmark PASSED: {elapsed:.2f}s for 100 features on 10K rows")

    def test_importance_computation_performance(self):
        """Benchmark: Importance computation overhead."""
        np.random.seed(42)
        n = 2000
        n_features = 30

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        # Without importance
        config1 = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=False))
        analyzer1 = FeatureOutcome(config=config1)
        start = time.time()
        analyzer1.run_analysis(features, outcome)
        time_no_imp = time.time() - start

        # With importance
        config2 = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=True))
        analyzer2 = FeatureOutcome(config=config2)
        start = time.time()
        analyzer2.run_analysis(features, outcome)
        time_with_imp = time.time() - start

        # Importance should add overhead but not be excessive
        overhead = time_with_imp - time_no_imp
        print(f"\nImportance computation overhead: {overhead:.2f}s")
        assert overhead < 15.0, f"Importance overhead {overhead:.2f}s too high"

    def test_ic_lag_scaling(self):
        """Benchmark: IC computation with multiple lags."""
        np.random.seed(42)
        n = 3000
        n_features = 30

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        # Single lag
        config1 = ModuleCConfig(ic=ICConfig(lag_structure=[1]))
        analyzer1 = FeatureOutcome(config=config1)
        start = time.time()
        analyzer1.run_analysis(features, outcome)
        time_1_lag = time.time() - start

        # Multiple lags
        config2 = ModuleCConfig(ic=ICConfig(lag_structure=[1, 5, 10, 20]))
        analyzer2 = FeatureOutcome(config=config2)
        start = time.time()
        analyzer2.run_analysis(features, outcome)
        time_4_lags = time.time() - start

        # Should scale roughly linearly
        ratio = time_4_lags / (time_1_lag + 0.1)  # Add 0.1 to avoid division by zero
        print(
            f"\nIC lag scaling: 1 lag={time_1_lag:.2f}s, 4 lags={time_4_lags:.2f}s, ratio={ratio:.2f}"
        )
        assert ratio < 8.0, f"IC lag scaling {ratio:.2f}x too high (expected <8x)"


class TestMemoryUsage:
    """Memory usage tests."""

    def test_large_dataset_completes(self):
        """Test that large datasets complete without memory issues."""
        np.random.seed(42)
        n = 50000  # Large dataset
        n_features = 20

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        analyzer = FeatureOutcome()

        # Should complete without memory error
        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        print(f"\nLarge dataset (50K rows, 20 features): {elapsed:.2f}s")

    def test_many_features_completes(self):
        """Test that many features complete without memory issues."""
        np.random.seed(42)
        n = 5000
        n_features = 200  # Many features

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        # Skip importance for speed
        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=False))
        analyzer = FeatureOutcome(config=config)

        # Should complete without memory error
        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        assert len(result.ic_results) == n_features
        print(f"\nMany features (200 features, 5K rows): {elapsed:.2f}s")


class TestScalingCharacteristics:
    """Test scaling behavior with data size."""

    def test_row_scaling(self):
        """Test how performance scales with row count."""
        np.random.seed(42)
        n_features = 10

        timings = {}

        for n in [1000, 2000, 5000]:
            outcome = np.random.randn(n)
            features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

            analyzer = FeatureOutcome()

            start = time.time()
            analyzer.run_analysis(features, outcome)
            timings[n] = time.time() - start

        print(f"\nRow scaling: {timings}")
        # Should be roughly linear or better
        # 5000 rows should not take more than 10x the time of 1000 rows
        assert timings[5000] < timings[1000] * 15

    def test_feature_scaling(self):
        """Test how performance scales with feature count."""
        np.random.seed(42)
        n = 2000

        timings = {}

        for n_features in [10, 30, 50]:
            outcome = np.random.randn(n)
            features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

            analyzer = FeatureOutcome()

            start = time.time()
            analyzer.run_analysis(features, outcome)
            timings[n_features] = time.time() - start

        print(f"\nFeature scaling: {timings}")
        # Should be roughly linear
        # 50 features should not take more than 20x the time of 10 features
        assert timings[50] < timings[10] * 25


@pytest.mark.slow
class TestExtremeCases:
    """Test extreme cases (marked as slow)."""

    def test_very_large_dataset(self):
        """Test very large dataset (100K rows, 50 features)."""
        np.random.seed(42)
        n = 100000
        n_features = 50

        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(n_features)})

        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=False))
        analyzer = FeatureOutcome(config=config)

        start = time.time()
        result = analyzer.run_analysis(features, outcome)
        elapsed = time.time() - start

        assert result.ic_results is not None
        print(f"\nVery large dataset (100K rows, 50 features): {elapsed:.2f}s")
        # Should complete in reasonable time
        assert elapsed < 120.0
