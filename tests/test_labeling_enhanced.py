"""
Tests for enhanced triple barrier labeling with uniqueness weighting and sequential bootstrap.
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling.triple_barrier import triple_barrier_labels
from ml4t.engineer.labeling.uniqueness import (
    _build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
    sequential_bootstrap,
)


class TestConcurrencyCalculation:
    """Test the concurrency calculation logic."""

    def test_non_overlapping_labels(self):
        """Test concurrency when labels don't overlap."""
        # Labels: [0-2], [3-5], [6-8]
        starts = np.array([0, 3, 6])
        ends = np.array([2, 5, 8])
        n_bars = 10

        concurrency = _build_concurrency(n_bars, starts, ends)

        # Expected: 1 active label at each timestamp
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        assert_array_equal(concurrency, expected)

    def test_fully_overlapping_labels(self):
        """Test concurrency when all labels overlap."""
        # All labels span [0-5]
        starts = np.array([0, 0, 0])
        ends = np.array([5, 5, 5])
        n_bars = 10

        concurrency = _build_concurrency(n_bars, starts, ends)

        # Expected: 3 active labels for bars 0-5
        expected = np.array([3, 3, 3, 3, 3, 3, 0, 0, 0, 0])
        assert_array_equal(concurrency, expected)

    def test_partially_overlapping_labels(self):
        """Test concurrency with partial overlaps."""
        # Labels: [0-3], [2-5], [4-7]
        starts = np.array([0, 2, 4])
        ends = np.array([3, 5, 7])
        n_bars = 10

        concurrency = _build_concurrency(n_bars, starts, ends)

        # Expected concurrency pattern
        expected = np.array([1, 1, 2, 2, 2, 2, 1, 1, 0, 0])
        assert_array_equal(concurrency, expected)

    def test_minute_bar_example(self):
        """Test the 60-minute holding period example from the document."""
        # Simulate 60 overlapping labels (each 60 bars long)
        n_labels = 60
        starts = np.arange(n_labels)  # Start at bars 0, 1, 2, ..., 59
        ends = starts + 59  # Each lasts 60 bars
        n_bars = 120

        concurrency = _build_concurrency(n_bars, starts, ends)

        # At bar 59, all 60 labels are active
        assert concurrency[59] == 60
        # At bar 60, label 0 has ended, so 59 are active
        assert concurrency[60] == 59
        # At bar 118, only the last label is active
        assert concurrency[118] == 1


class TestUniquenessCalculation:
    """Test label uniqueness calculation."""

    def test_non_overlapping_uniqueness(self):
        """Non-overlapping labels should have maximum uniqueness."""
        starts = np.array([0, 5, 10])
        ends = np.array([2, 7, 12])
        n_bars = 15

        uniqueness = calculate_label_uniqueness(starts, ends, n_bars)

        # All labels are unique (no overlap)
        assert_array_almost_equal(uniqueness, [1.0, 1.0, 1.0])

    def test_fully_overlapping_uniqueness(self):
        """Fully overlapping labels should have low uniqueness."""
        starts = np.array([0, 0, 0])
        ends = np.array([5, 5, 5])
        n_bars = 10

        uniqueness = calculate_label_uniqueness(starts, ends, n_bars)

        # All labels overlap completely: u_i = 1/3 for all
        expected_uniqueness = 1.0 / 3.0
        assert_array_almost_equal(uniqueness, [expected_uniqueness] * 3)

    def test_early_exit_vs_full_horizon(self):
        """Test that early exits have lower uniqueness than full-horizon labels.

        Key insight: For this to work, all labels must start at the same time.
        Early exits then only experience the crowded period (high concurrency),
        while full-horizon labels also experience the quiet period after others exit
        (low concurrency), giving them higher average uniqueness.
        """
        # All labels start together at bar 0
        # Some exit early (only see crowded period)
        # Some go full horizon (see crowded period + quiet period after exits)
        starts = np.array([0, 0, 0, 0, 0])  # All start at same time
        ends = np.array(
            [
                10,  # Label 0: exits early - only sees 5-label concurrency
                15,  # Label 1: exits mid-way - sees 5→4-label concurrency
                50,  # Label 2: longer hold - sees transition to 3→2-label periods
                59,  # Label 3: full horizon - includes quiet 2-label period at end
                59,  # Label 4: full horizon
            ]
        )
        n_bars = 60

        uniqueness = calculate_label_uniqueness(starts, ends, n_bars)

        # Early exits should have lower uniqueness (stuck in crowded period)
        # Full-horizon labels should have higher uniqueness (experience quiet period)
        assert uniqueness[0] < uniqueness[2], (
            f"Early exit uniqueness {uniqueness[0]:.3f} should be < mid-horizon {uniqueness[2]:.3f}"
        )
        assert uniqueness[0] < uniqueness[3], (
            f"Early exit uniqueness {uniqueness[0]:.3f} should be < full horizon {uniqueness[3]:.3f}"
        )
        assert uniqueness[1] < uniqueness[3], (
            f"Mid-exit uniqueness {uniqueness[1]:.3f} should be < full horizon {uniqueness[3]:.3f}"
        )

        # Verify the ordering makes sense: shorter holds → lower uniqueness
        assert uniqueness[0] < uniqueness[1] < uniqueness[2] < uniqueness[3], (
            f"Uniqueness should increase with holding period: {uniqueness[:4]}"
        )

    def test_uniqueness_calculation_example(self):
        """Test specific uniqueness calculation."""
        # Simple case: 2 labels with partial overlap
        starts = np.array([0, 2])
        ends = np.array([4, 6])
        n_bars = 10

        uniqueness = calculate_label_uniqueness(starts, ends, n_bars)

        # Label 0 [0-4]: bars 0,1 have c=1, bars 2,3,4 have c=2
        # u_0 = (1/1 + 1/1 + 1/2 + 1/2 + 1/2) / 5 = 3.5 / 5 = 0.7

        # Label 1 [2-6]: bars 2,3,4 have c=2, bars 5,6 have c=1
        # u_1 = (1/2 + 1/2 + 1/2 + 1/1 + 1/1) / 5 = 3.5 / 5 = 0.7

        assert_array_almost_equal(uniqueness, [0.7, 0.7], decimal=5)


class TestSampleWeights:
    """Test sample weight calculation."""

    def test_returns_uniqueness_weighting(self):
        """Test De Prado's recommended weighting scheme."""
        uniqueness = np.array([0.5, 0.8, 0.3])
        returns = np.array([0.02, -0.01, 0.03])

        weights = calculate_sample_weights(uniqueness, returns, "returns_uniqueness")

        # Expected: u_i * |r_i|, normalized
        expected_raw = np.array([0.5 * 0.02, 0.8 * 0.01, 0.3 * 0.03])
        expected = expected_raw * 3 / expected_raw.sum()  # Normalize to sum to n

        assert_array_almost_equal(weights, expected)

    def test_uniqueness_only_weighting(self):
        """Test uniqueness-only weighting."""
        uniqueness = np.array([0.5, 0.8, 0.3])
        returns = np.array([0.02, -0.01, 0.03])

        weights = calculate_sample_weights(uniqueness, returns, "uniqueness_only")

        # Expected: just uniqueness, normalized
        expected = uniqueness * 3 / uniqueness.sum()

        assert_array_almost_equal(weights, expected)

    def test_equal_weighting(self):
        """Test equal weighting."""
        uniqueness = np.array([0.5, 0.8, 0.3])
        returns = np.array([0.02, -0.01, 0.03])

        weights = calculate_sample_weights(uniqueness, returns, "equal")

        # All weights should be 1.0
        assert_array_almost_equal(weights, [1.0, 1.0, 1.0])

    def test_economic_significance(self):
        """Test that large returns get higher weight in returns_uniqueness scheme."""
        uniqueness = np.array([0.5, 0.5, 0.5])  # Same uniqueness
        returns = np.array([0.01, 0.05, 0.001])  # Different returns

        weights = calculate_sample_weights(uniqueness, returns, "returns_uniqueness")

        # Larger return should get higher weight
        assert weights[1] > weights[0] > weights[2]


class TestSequentialBootstrap:
    """Test sequential bootstrap sampling."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap functionality."""
        starts = np.array([0, 5, 10])
        ends = np.array([4, 9, 14])
        n_bars = 15
        n_draws = 3

        order = sequential_bootstrap(
            starts, ends, n_bars, n_draws, with_replacement=False, random_state=42
        )

        # Should return 3 indices
        assert len(order) == n_draws
        # All indices should be valid
        assert all(0 <= idx < 3 for idx in order)
        # Without replacement: no duplicates
        assert len(np.unique(order)) == n_draws

    def test_bootstrap_favors_uniqueness(self):
        """Test that bootstrap favors unique labels."""
        # Create scenario where label 2 doesn't overlap with others
        starts = np.array([0, 1, 10])  # Label 2 starts later
        ends = np.array([5, 6, 15])  # Label 2 doesn't overlap
        n_bars = 20
        n_draws = 100  # Many draws to see the pattern

        order = sequential_bootstrap(
            starts, ends, n_bars, n_draws, with_replacement=True, random_state=42
        )

        # Count how often each label is selected
        counts = np.bincount(order, minlength=3)

        # Label 2 (non-overlapping) should be selected more often
        # than labels 0 and 1 (which overlap)
        assert counts[2] > counts[0], f"Non-overlapping label should be favored: {counts}"
        assert counts[2] > counts[1], f"Non-overlapping label should be favored: {counts}"

    def test_bootstrap_with_replacement(self):
        """Test bootstrap with replacement."""
        starts = np.array([0, 5, 10])
        ends = np.array([4, 9, 14])
        n_bars = 15
        n_draws = 10  # More draws than labels

        order = sequential_bootstrap(
            starts, ends, n_bars, n_draws, with_replacement=True, random_state=42
        )

        # Should return 10 indices
        assert len(order) == n_draws
        # All indices should be valid
        assert all(0 <= idx < 3 for idx in order)
        # With replacement: duplicates allowed
        # (but not guaranteed - depends on probabilities)

    def test_bootstrap_reproducibility(self):
        """Test that random_state makes results reproducible."""
        starts = np.array([0, 5, 10])
        ends = np.array([4, 9, 14])

        order1 = sequential_bootstrap(starts, ends, random_state=42)
        order2 = sequential_bootstrap(starts, ends, random_state=42)
        order3 = sequential_bootstrap(starts, ends, random_state=123)

        # Same seed should give same results
        assert_array_equal(order1, order2)
        # Different seed should (usually) give different results
        assert not np.array_equal(order1, order3)


class TestEnhancedTripleBarrier:
    """Test the enhanced triple barrier labeling function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n = 100
        returns = np.random.randn(n) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        return pl.DataFrame(
            {
                "close": prices,
                "datetime": pl.datetime_range(
                    start=pl.datetime(2024, 1, 1),
                    end=pl.datetime(2024, 1, 1, 23, 59),
                    interval="1m",
                    eager=True,
                )[:n],
            }
        )

    def test_enhanced_labeling_basic(self, sample_data):
        """Test basic enhanced triple barrier labeling."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            calculate_uniqueness=True,
            uniqueness_weight_scheme="returns_uniqueness",
        )

        # Check that all required columns are present
        required_cols = ["label", "label_return", "label_uniqueness", "sample_weight"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check data types
        assert result["label"].dtype == pl.Int32
        assert result["label_return"].dtype == pl.Float64
        assert result["label_uniqueness"].dtype == pl.Float64
        assert result["sample_weight"].dtype == pl.Float64

        # Check value ranges
        labels = result["label"].to_numpy()
        assert all(label in [-1, 0, 1] for label in labels if label is not None)

        uniqueness = result["label_uniqueness"].to_numpy()
        assert all(0 <= u <= 1 for u in uniqueness if u is not None)

        weights = result["sample_weight"].to_numpy()
        assert all(w >= 0 for w in weights if w is not None)

    def test_uniqueness_affects_weights(self, sample_data):
        """Test that uniqueness actually affects sample weights."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,  # Longer periods for more overlap
        )

        # Compare two weight schemes
        result_with = triple_barrier_labels(
            sample_data,
            config,
            calculate_uniqueness=True,
            uniqueness_weight_scheme="returns_uniqueness",
        )

        result_without = triple_barrier_labels(
            sample_data,
            config,
            calculate_uniqueness=True,
            uniqueness_weight_scheme="returns_only",
        )

        weights_with = result_with["sample_weight"].to_numpy()
        weights_without = result_without["sample_weight"].to_numpy()

        # Weights should be different when uniqueness is included
        assert not np.allclose(weights_with, weights_without)

    def test_weight_schemes(self, sample_data):
        """Test different weight schemes produce different results."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
        )

        schemes = ["returns_uniqueness", "uniqueness_only", "returns_only", "equal"]
        results = {}

        for scheme in schemes:
            result = triple_barrier_labels(
                sample_data,
                config,
                calculate_uniqueness=True,
                uniqueness_weight_scheme=scheme,
            )
            results[scheme] = result["sample_weight"].to_numpy()

        # Each scheme should produce different weights (except possibly by chance)
        # At minimum, equal weights should differ from others
        assert not np.allclose(results["equal"], results["returns_uniqueness"])
        assert not np.allclose(results["equal"], results["returns_only"])

    def test_no_uniqueness_calculation(self, sample_data):
        """Test that uniqueness calculation can be skipped."""
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=10,
        )

        result = triple_barrier_labels(
            sample_data,
            config,
            calculate_uniqueness=False,
        )

        # Uniqueness columns should not be present when not calculated
        assert "label_uniqueness" not in result.columns
        assert "sample_weight" not in result.columns

        # But standard labeling columns should be present
        assert "label" in result.columns
        assert "label_return" in result.columns


def test_integration_example():
    """Test the complete workflow with sequential bootstrap."""
    # Create synthetic price data
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pl.DataFrame(
        {
            "close": prices,
        }
    )

    # Step 1: Apply triple barrier labeling
    config = LabelingConfig.triple_barrier(
        upper_barrier=0.015,
        lower_barrier=0.01,
        max_holding_period=20,
    )

    labeled = triple_barrier_labels(
        df,
        config,
        calculate_uniqueness=True,
        uniqueness_weight_scheme="returns_uniqueness",
    )

    # Step 2: Extract necessary arrays
    event_indices = np.arange(len(labeled))
    label_indices = event_indices + labeled["label_bars"].to_numpy()
    label_indices = np.clip(label_indices, 0, len(labeled) - 1)

    # Step 3: Apply sequential bootstrap
    n_train = 150
    bootstrap_order = sequential_bootstrap(
        starts=event_indices,
        ends=label_indices,
        n_bars=len(labeled),
        n_draws=n_train,
        with_replacement=True,
        random_state=42,
    )

    # Step 4: Create training set
    train_indices = bootstrap_order
    X_train = labeled[train_indices].select(["close"])  # Features
    y_train = labeled[train_indices]["label"]  # Labels
    weights_train = labeled[train_indices]["sample_weight"]  # Weights

    # Verify the training set
    assert len(X_train) == n_train
    assert len(y_train) == n_train
    assert len(weights_train) == n_train

    # Check that bootstrap selected diverse samples
    unique_samples = len(np.unique(bootstrap_order))
    assert unique_samples > n_train * 0.5, (
        f"Bootstrap should select diverse samples, got {unique_samples}/{n_train}"
    )

    print(f"Integration test passed: {unique_samples} unique samples in {n_train} draws")


if __name__ == "__main__":
    # Run a simple demonstration
    test_integration_example()
    print("\nAll integration tests passed!")

    # Run more detailed tests
    pytest.main([__file__, "-v"])
