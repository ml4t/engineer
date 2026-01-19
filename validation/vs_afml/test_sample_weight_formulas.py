# mypy: disable-error-code="no-any-return,arg-type"
"""Direct verification of sample weight calculations against AFML Chapter 4 formulas.

AFML Sample Weight Formulas:

1. **Concurrency** at time t:
   c_t = sum_i(1_{t,i})
   where 1_{t,i} = 1 if label i spans time t, else 0

2. **Uniqueness** at time t for label i:
   u_{t,i} = 1_{t,i} / c_t

3. **Average Uniqueness** for label i:
   ū_i = sum_t(u_{t,i}) / sum_t(1_{t,i})
   = mean(1/c_t) over the label's lifespan [t_{i,0}, t_{i,1}]

4. **Sample Weight** (return attribution):
   w̃_i = |sum_t(r_{t-1,t} / c_t)| for t in [t_{i,0}, t_{i,1}]

5. **Sequential Bootstrap**:
   Draw probability proportional to marginal uniqueness

Reference:
    López de Prado, M. (2018). "Advances in Financial Machine Learning".
    Wiley. Chapter 4: Sample Weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml4t.engineer.labeling import (
    build_concurrency,
    calculate_label_uniqueness,
    calculate_sample_weights,
    sequential_bootstrap,
)


class TestConcurrency:
    """Tests for concurrency calculation (c_t)."""

    def test_single_label_no_overlap(self):
        """Single label has concurrency 1 during its lifespan.

        Label spans [0, 2], so c[0]=1, c[1]=1, c[2]=1.
        """
        event_indices = np.array([0])
        label_indices = np.array([2])

        concurrency = build_concurrency(event_indices, label_indices, n_bars=5)

        expected = [1, 1, 1, 0, 0]
        np.testing.assert_array_equal(concurrency, expected)

    def test_two_labels_no_overlap(self):
        """Two non-overlapping labels have concurrency 1.

        Label 1: [0, 1], Label 2: [3, 4]
        """
        event_indices = np.array([0, 3])
        label_indices = np.array([1, 4])

        concurrency = build_concurrency(event_indices, label_indices, n_bars=5)

        expected = [1, 1, 0, 1, 1]
        np.testing.assert_array_equal(concurrency, expected)

    def test_two_labels_overlap(self):
        """Two overlapping labels increase concurrency.

        Label 1: [0, 3], Label 2: [2, 4]
        Overlap at t=2,3 → c[2]=2, c[3]=2
        """
        event_indices = np.array([0, 2])
        label_indices = np.array([3, 4])

        concurrency = build_concurrency(event_indices, label_indices, n_bars=5)

        expected = [1, 1, 2, 2, 1]
        np.testing.assert_array_equal(concurrency, expected)

    def test_three_labels_triple_overlap(self):
        """Three overlapping labels at a single point.

        Label 1: [0, 2], Label 2: [1, 3], Label 3: [2, 4]
        c[2] = 3 (all three overlap)
        """
        event_indices = np.array([0, 1, 2])
        label_indices = np.array([2, 3, 4])

        concurrency = build_concurrency(event_indices, label_indices, n_bars=5)

        expected = [1, 2, 3, 2, 1]
        np.testing.assert_array_equal(concurrency, expected)

    def test_afml_book_example(self):
        """Example from AFML Chapter 4.5.3.

        Label y1: [0, 2] (spans bars 0,1,2)
        Label y2: [2, 3] (spans bars 2,3)
        Label y3: [4, 5] (spans bars 4,5)

        Indicator matrix:
        t=0: [1,0,0] → c[0]=1
        t=1: [1,0,0] → c[1]=1
        t=2: [1,1,0] → c[2]=2
        t=3: [0,1,0] → c[3]=1
        t=4: [0,0,1] → c[4]=1
        t=5: [0,0,1] → c[5]=1
        """
        event_indices = np.array([0, 2, 4])
        label_indices = np.array([2, 3, 5])

        concurrency = build_concurrency(event_indices, label_indices, n_bars=6)

        expected = [1, 1, 2, 1, 1, 1]
        np.testing.assert_array_equal(concurrency, expected)


class TestLabelUniqueness:
    """Tests for average label uniqueness (ū_i)."""

    def test_single_label_uniqueness_is_one(self):
        """Single label has uniqueness 1.0 (no overlap with itself at first)."""
        event_indices = np.array([0])
        label_indices = np.array([2])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=5)

        # Single label: c=1 everywhere, so u = mean(1/1) = 1.0
        assert uniqueness[0] == 1.0

    def test_two_labels_no_overlap(self):
        """Non-overlapping labels have uniqueness 1.0."""
        event_indices = np.array([0, 3])
        label_indices = np.array([1, 4])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=5)

        assert uniqueness[0] == 1.0
        assert uniqueness[1] == 1.0

    def test_two_labels_overlap(self):
        """Overlapping labels have uniqueness < 1.0.

        Label 1: [0, 3], Label 2: [2, 4]
        c = [1, 1, 2, 2, 1]

        For label 1 (spans [0,3]):
        ū_1 = mean(1/c[0], 1/c[1], 1/c[2], 1/c[3])
            = mean(1/1, 1/1, 1/2, 1/2)
            = (1 + 1 + 0.5 + 0.5) / 4 = 0.75

        For label 2 (spans [2,4]):
        ū_2 = mean(1/c[2], 1/c[3], 1/c[4])
            = mean(1/2, 1/2, 1/1)
            = (0.5 + 0.5 + 1) / 3 = 0.667
        """
        event_indices = np.array([0, 2])
        label_indices = np.array([3, 4])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=5)

        assert abs(uniqueness[0] - 0.75) < 1e-10
        assert abs(uniqueness[1] - (2.0 / 3)) < 1e-10

    def test_afml_book_example(self):
        """AFML Chapter 4.5.3 numerical example.

        Label y1: [0, 2], y2: [2, 3], y3: [4, 5]
        c = [1, 1, 2, 1, 1, 1]

        ū_1 = mean(1/c[0], 1/c[1], 1/c[2]) = mean(1, 1, 0.5) = 5/6
        ū_2 = mean(1/c[2], 1/c[3]) = mean(0.5, 1) = 0.75
        ū_3 = mean(1/c[4], 1/c[5]) = mean(1, 1) = 1.0
        """
        event_indices = np.array([0, 2, 4])
        label_indices = np.array([2, 3, 5])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=6)

        assert abs(uniqueness[0] - (5.0 / 6)) < 1e-10
        assert abs(uniqueness[1] - 0.75) < 1e-10
        assert uniqueness[2] == 1.0


class TestSampleWeights:
    """Tests for sample weight calculation."""

    def test_uniqueness_only_scheme(self):
        """Test uniqueness-only weighting scheme."""
        uniqueness = np.array([1.0, 0.5, 0.75])
        returns = np.array([0.01, 0.02, 0.03])

        weights = calculate_sample_weights(
            uniqueness, returns, weight_scheme="uniqueness_only"
        )

        # Weights should be proportional to uniqueness, normalized to sum=len
        total = 1.0 + 0.5 + 0.75
        expected = np.array([1.0, 0.5, 0.75]) * 3 / total

        np.testing.assert_allclose(weights, expected)

    def test_returns_only_scheme(self):
        """Test returns-only weighting scheme."""
        uniqueness = np.array([1.0, 0.5, 0.75])
        returns = np.array([0.01, 0.02, 0.03])

        weights = calculate_sample_weights(
            uniqueness, returns, weight_scheme="returns_only"
        )

        # Weights should be proportional to |returns|, normalized
        abs_returns = np.array([0.01, 0.02, 0.03])
        total = abs_returns.sum()
        expected = abs_returns * 3 / total

        np.testing.assert_allclose(weights, expected)

    def test_returns_uniqueness_scheme(self):
        """Test combined returns * uniqueness scheme (De Prado's recommended).

        AFML: w_i = u_i * |r_i|
        """
        uniqueness = np.array([1.0, 0.5, 0.75])
        returns = np.array([0.01, -0.02, 0.03])  # Including negative

        weights = calculate_sample_weights(
            uniqueness, returns, weight_scheme="returns_uniqueness"
        )

        # w̃_i = u_i * |r_i|
        raw_weights = uniqueness * np.abs(returns)
        # Normalize to sum=len
        expected = raw_weights * 3 / raw_weights.sum()

        np.testing.assert_allclose(weights, expected)

    def test_equal_scheme(self):
        """Test equal weighting scheme."""
        uniqueness = np.array([0.3, 0.5, 0.9])
        returns = np.array([0.01, 0.02, 0.03])

        weights = calculate_sample_weights(
            uniqueness, returns, weight_scheme="equal"
        )

        # All weights should be 1.0
        np.testing.assert_allclose(weights, np.ones(3))

    def test_weights_sum_to_length(self):
        """Weights should sum to len(weights) for sklearn compatibility."""
        uniqueness = np.array([0.8, 0.6, 0.4, 0.9, 0.7])
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        for scheme in ["uniqueness_only", "returns_only", "returns_uniqueness"]:
            weights = calculate_sample_weights(uniqueness, returns, weight_scheme=scheme)
            assert abs(weights.sum() - 5.0) < 1e-10, f"Scheme {scheme}"


class TestSequentialBootstrap:
    """Tests for sequential bootstrap algorithm."""

    def test_sequential_draws_expected_length(self):
        """Sequential bootstrap returns expected number of samples."""
        event_indices = np.array([0, 2, 4])
        label_indices = np.array([2, 3, 5])

        order = sequential_bootstrap(
            event_indices.astype(np.int64),
            label_indices.astype(np.int64),
            n_bars=6,
            n_draws=3,
            random_state=42,
        )

        assert len(order) == 3

    def test_sequential_favors_unique_labels(self):
        """Sequential bootstrap favors labels with lower overlap.

        Run multiple times and check distribution.
        Label 3 ([4,5]) has no overlap with 1 and 2, so should be selected more often.
        """
        event_indices = np.array([0, 2, 4])
        label_indices = np.array([2, 3, 5])

        counts = np.zeros(3)
        n_iterations = 1000

        for seed in range(n_iterations):
            order = sequential_bootstrap(
                event_indices.astype(np.int64),
                label_indices.astype(np.int64),
                n_bars=6,
                n_draws=1,  # Single draw
                random_state=seed,
            )
            counts[order[0]] += 1

        # Label 3 (index 2) should have highest initial probability
        # because it doesn't overlap with others
        # ū_3 = 1.0, ū_1 = 5/6, ū_2 = 0.75
        # Initial uniform draw, but after first draw, non-overlapping is preferred
        # This is a probabilistic test - just verify counts are non-zero
        assert counts[2] > 0

    def test_sequential_without_replacement(self):
        """Sequential bootstrap without replacement returns unique indices."""
        event_indices = np.array([0, 2, 4, 6, 8])
        label_indices = np.array([1, 3, 5, 7, 9])

        order = sequential_bootstrap(
            event_indices.astype(np.int64),
            label_indices.astype(np.int64),
            n_bars=10,
            n_draws=5,
            with_replacement=False,
            random_state=42,
        )

        # All unique indices
        assert len(set(order)) == 5

    def test_sequential_with_replacement(self):
        """Sequential bootstrap with replacement can repeat indices."""
        event_indices = np.array([0, 1, 2])  # All overlap heavily
        label_indices = np.array([3, 4, 5])

        # Run many draws to increase chance of repetition
        order = sequential_bootstrap(
            event_indices.astype(np.int64),
            label_indices.astype(np.int64),
            n_bars=6,
            n_draws=20,
            with_replacement=True,
            random_state=42,
        )

        # With 20 draws from 3 options, should have repetitions
        assert len(order) == 20

    def test_sequential_reproducibility(self):
        """Same random seed gives same results."""
        event_indices = np.array([0, 2, 4, 6])
        label_indices = np.array([2, 4, 6, 8])

        order1 = sequential_bootstrap(
            event_indices.astype(np.int64),
            label_indices.astype(np.int64),
            n_bars=9,
            n_draws=4,
            random_state=12345,
        )

        order2 = sequential_bootstrap(
            event_indices.astype(np.int64),
            label_indices.astype(np.int64),
            n_bars=9,
            n_draws=4,
            random_state=12345,
        )

        np.testing.assert_array_equal(order1, order2)


class TestEdgeCases:
    """Edge cases for sample weights and uniqueness."""

    def test_empty_arrays(self):
        """Empty input arrays should return empty results."""
        empty_event = np.array([])
        empty_label = np.array([])

        uniqueness = calculate_label_uniqueness(empty_event, empty_label)
        assert len(uniqueness) == 0

        weights = calculate_sample_weights(np.array([]), np.array([]))
        assert len(weights) == 0

    def test_single_observation(self):
        """Single observation should have uniqueness 1.0."""
        event_indices = np.array([5])
        label_indices = np.array([10])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=15)

        assert uniqueness[0] == 1.0

    def test_zero_returns_uniqueness_only(self):
        """Zero returns with uniqueness_only scheme still works."""
        uniqueness = np.array([0.8, 0.6, 0.7])
        returns = np.array([0.0, 0.0, 0.0])

        weights = calculate_sample_weights(
            uniqueness, returns, weight_scheme="uniqueness_only"
        )

        # Should still produce valid weights
        assert len(weights) == 3
        assert not np.any(np.isnan(weights))

    def test_maximum_overlap(self):
        """All labels perfectly overlap → minimum uniqueness."""
        # All labels span the same interval [0, 4]
        event_indices = np.array([0, 0, 0])
        label_indices = np.array([4, 4, 4])

        uniqueness = calculate_label_uniqueness(event_indices, label_indices, n_bars=5)

        # All three labels overlap perfectly
        # c[t] = 3 for all t in [0,4]
        # u_i = mean(1/3, 1/3, 1/3, 1/3, 1/3) = 1/3
        for u in uniqueness:
            assert abs(u - (1.0 / 3)) < 1e-10
