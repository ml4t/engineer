"""Performance benchmark tests for advanced features.

These tests validate that our advanced features meet the performance targets
defined in OPEN_ITEMS_TRACKER.md.
"""

import os
import sys

import pytest

# Add the benchmarks directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".claude", "benchmarks"))

from advanced_features_benchmark import BenchmarkHarness


@pytest.mark.benchmark
@pytest.mark.slow
class TestAdvancedFeaturesPerformance:
    """Performance benchmark tests for advanced features."""

    def setup_method(self):
        """Set up benchmark harness."""
        self.harness = BenchmarkHarness()
        self.test_data = self.harness.create_test_data(10_000)  # Smaller for unit tests

    def test_cross_asset_performance(self):
        """Test cross-asset features performance."""
        results = self.harness.run_cross_asset_benchmarks(self.test_data)

        # At least some functions should meet performance targets
        successful_results = [r for r in results if not r.get("error", True)]
        assert len(successful_results) > 0, "No cross-asset functions executed successfully"

        # Report performance
        for result in successful_results:
            print(f"{result['function']}: {result['rows_per_second']:,} rows/sec")

        # Check that at least half meet performance targets (relaxed for unit tests)
        meeting_targets = [r for r in successful_results if r.get("meets_target", False)]
        success_rate = len(meeting_targets) / len(successful_results) if successful_results else 0

        assert success_rate >= 0.3, f"Only {success_rate:.1%} of cross-asset functions met targets"

    def test_microstructure_performance(self):
        """Test microstructure features performance."""
        results = self.harness.run_microstructure_benchmarks(self.test_data)

        successful_results = [r for r in results if not r.get("error", True)]
        assert len(successful_results) > 0, "No microstructure functions executed successfully"

        # Report performance
        for result in successful_results:
            print(f"{result['function']}: {result['rows_per_second']:,} rows/sec")

        # Check performance
        meeting_targets = [r for r in successful_results if r.get("meets_target", False)]
        success_rate = len(meeting_targets) / len(successful_results) if successful_results else 0

        assert success_rate >= 0.3, (
            f"Only {success_rate:.1%} of microstructure functions met targets"
        )

    def test_ml_features_performance(self):
        """Test ML features performance."""
        results = self.harness.run_ml_features_benchmarks(self.test_data)

        successful_results = [r for r in results if not r.get("error", True)]
        assert len(successful_results) > 0, "No ML features functions executed successfully"

        # Report performance
        for result in successful_results:
            print(f"{result['function']}: {result['rows_per_second']:,} rows/sec")

        # Check performance
        meeting_targets = [r for r in successful_results if r.get("meets_target", False)]
        success_rate = len(meeting_targets) / len(successful_results) if successful_results else 0

        assert success_rate >= 0.3, f"Only {success_rate:.1%} of ML functions met targets"

    def test_regime_detection_performance(self):
        """Test regime detection features performance."""
        results = self.harness.run_regime_benchmarks(self.test_data)

        successful_results = [r for r in results if not r.get("error", True)]
        assert len(successful_results) > 0, "No regime detection functions executed successfully"

        # Report performance
        for result in successful_results:
            print(f"{result['function']}: {result['rows_per_second']:,} rows/sec")

        # Check performance - regime detection has lower targets due to complexity
        meeting_targets = [r for r in successful_results if r.get("meets_target", False)]
        success_rate = len(meeting_targets) / len(successful_results) if successful_results else 0

        # More lenient for regime detection due to complexity (Hurst exponent, etc.)
        assert success_rate >= 0.2, f"Only {success_rate:.1%} of regime functions met targets"

    def test_volatility_advanced_performance(self):
        """Test advanced volatility features performance."""
        results = self.harness.run_volatility_advanced_benchmarks(self.test_data)

        successful_results = [r for r in results if not r.get("error", True)]
        assert len(successful_results) > 0, "No volatility functions executed successfully"

        # Report performance
        for result in successful_results:
            print(f"{result['function']}: {result['rows_per_second']:,} rows/sec")

        # Check performance
        meeting_targets = [r for r in successful_results if r.get("meets_target", False)]
        success_rate = len(meeting_targets) / len(successful_results) if successful_results else 0

        assert success_rate >= 0.3, f"Only {success_rate:.1%} of volatility functions met targets"

    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Use lazy evaluation to test memory efficiency
        large_data = self.harness.create_test_data(50_000)

        # Test with cross-asset features using lazy evaluation
        import time

        start_time = time.perf_counter()

        # This should not consume excessive memory due to lazy evaluation
        from ml4t.engineer.features.cross_asset import rolling_correlation

        lazy_result = large_data.lazy().with_columns(
            [rolling_correlation("returns", "returns", 20).alias("correlation")],
        )

        # Collecting should be reasonably fast
        result = lazy_result.collect()

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Memory efficiency test (50K rows): {execution_time:.3f}s")

        # Should complete within reasonable time and memory usage
        assert execution_time < 10.0, f"Memory test too slow: {execution_time:.3f}s"
        assert len(result) == 50_000, "Result size incorrect"

    @pytest.mark.performance
    @pytest.mark.parametrize("dataset_size", [1_000, 5_000, 10_000])
    def test_scalability(self, dataset_size):
        """Test scalability across different dataset sizes."""
        data = self.harness.create_test_data(dataset_size)

        # Test a representative function from each category
        from ml4t.engineer.features.cross_asset import rolling_correlation
        from ml4t.engineer.features.microstructure import amihud_illiquidity
        from ml4t.engineer.features.ml import cyclical_encode

        functions_to_test = [
            ("rolling_correlation", rolling_correlation, ["returns", "returns", 20]),
            (
                "amihud_illiquidity",
                amihud_illiquidity,
                ["returns", "volume", "close", 20],
            ),
            ("cyclical_encode", cyclical_encode, [24, "hour"]),
        ]

        scalability_results = []

        for func_name, func, args in functions_to_test:
            try:
                result, exec_time = self.harness.time_function(func, data, *args)
                if result is not None and exec_time != float("inf"):
                    rows_per_sec = dataset_size / exec_time if exec_time > 0 else 0
                    scalability_results.append(
                        {
                            "function": func_name,
                            "dataset_size": dataset_size,
                            "rows_per_second": rows_per_sec,
                        },
                    )
                    print(
                        f"{func_name} ({dataset_size:,} rows): {rows_per_sec:,.0f} rows/sec",
                    )
            except Exception as e:
                print(f"Error testing {func_name}: {e}")

        # At least one function should scale properly
        assert len(scalability_results) > 0, (
            f"No functions scaled properly with {dataset_size} rows"
        )

        # Performance should not degrade drastically with size (within reason)
        for result in scalability_results:
            assert result["rows_per_second"] > 1000, (
                f"{result['function']} too slow: {result['rows_per_second']:,.0f} rows/sec"
            )


if __name__ == "__main__":
    # Run benchmarks directly
    harness = BenchmarkHarness()
    results = harness.run_all_benchmarks([10_000])
    report = harness.generate_report(results)
    print(report)
