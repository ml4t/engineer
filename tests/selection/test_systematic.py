"""Tests for systematic feature selection.

These tests require ml4t-diagnostic to be installed for the test fixtures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

# Feature selection tests require ml4t-diagnostic for the outcome data classes
try:
    from ml4t.diagnostic.evaluation.drift import (
        DriftSummaryResult,
        FeatureDriftResult,
        PSIResult,
    )
    from ml4t.diagnostic.evaluation.feature_outcome import (
        FeatureICResults,
        FeatureImportanceResults,
        FeatureOutcomeResult,
    )

    HAS_DIAGNOSTIC = True
except ImportError:
    HAS_DIAGNOSTIC = False

from ml4t.engineer.selection import FeatureSelector
from ml4t.engineer.selection.systematic import SelectionReport, SelectionStep

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.skipif(
    not HAS_DIAGNOSTIC, reason="ml4t-diagnostic required for feature selection tests"
)


@pytest.fixture
def sample_outcome_results() -> FeatureOutcomeResult:
    """Create sample feature-outcome results for testing."""
    features = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]

    # IC results with varying quality
    ic_results = {
        "feature_a": FeatureICResults(
            feature="feature_a",
            ic_mean=0.05,  # Strong IC
            ic_std=0.01,
            ic_ir=5.0,
            t_stat=10.0,
            p_value=0.001,
            ic_by_lag={0: 0.04, 1: 0.05, 5: 0.06},
            n_observations=100,
        ),
        "feature_b": FeatureICResults(
            feature="feature_b",
            ic_mean=0.03,  # Good IC
            ic_std=0.01,
            ic_ir=3.0,
            t_stat=6.0,
            p_value=0.01,
            ic_by_lag={0: 0.02, 1: 0.03, 5: 0.04},
            n_observations=100,
        ),
        "feature_c": FeatureICResults(
            feature="feature_c",
            ic_mean=0.01,  # Weak IC
            ic_std=0.01,
            ic_ir=1.0,
            t_stat=2.0,
            p_value=0.05,
            ic_by_lag={0: 0.005, 1: 0.01, 5: 0.015},
            n_observations=100,
        ),
        "feature_d": FeatureICResults(
            feature="feature_d",
            ic_mean=0.025,  # Medium IC
            ic_std=0.01,
            ic_ir=2.5,
            t_stat=5.0,
            p_value=0.02,
            ic_by_lag={0: 0.02, 1: 0.025, 5: 0.03},
            n_observations=100,
        ),
        "feature_e": FeatureICResults(
            feature="feature_e",
            ic_mean=0.002,  # Very weak IC
            ic_std=0.01,
            ic_ir=0.2,
            t_stat=0.4,
            p_value=0.7,
            ic_by_lag={0: 0.001, 1: 0.002, 5: 0.003},
            n_observations=100,
        ),
    }

    # Importance results
    importance_results = {
        "feature_a": FeatureImportanceResults(
            feature="feature_a",
            mdi_importance=0.30,  # High importance
            permutation_importance=0.25,
            permutation_std=0.02,
            shap_mean=0.28,
            shap_std=0.03,
            rank_mdi=1,
            rank_permutation=1,
        ),
        "feature_b": FeatureImportanceResults(
            feature="feature_b",
            mdi_importance=0.25,  # Good importance
            permutation_importance=0.22,
            permutation_std=0.02,
            shap_mean=0.23,
            shap_std=0.03,
            rank_mdi=2,
            rank_permutation=2,
        ),
        "feature_c": FeatureImportanceResults(
            feature="feature_c",
            mdi_importance=0.05,  # Low importance
            permutation_importance=0.03,
            permutation_std=0.01,
            shap_mean=0.04,
            shap_std=0.02,
            rank_mdi=5,
            rank_permutation=5,
        ),
        "feature_d": FeatureImportanceResults(
            feature="feature_d",
            mdi_importance=0.20,  # Medium importance
            permutation_importance=0.18,
            permutation_std=0.02,
            shap_mean=0.19,
            shap_std=0.03,
            rank_mdi=3,
            rank_permutation=3,
        ),
        "feature_e": FeatureImportanceResults(
            feature="feature_e",
            mdi_importance=0.15,  # Medium-low importance
            permutation_importance=0.12,
            permutation_std=0.02,
            shap_mean=0.13,
            shap_std=0.02,
            rank_mdi=4,
            rank_permutation=4,
        ),
    }

    # Drift results
    drift_feature_results = [
        FeatureDriftResult(
            feature="feature_a",
            psi_result=PSIResult(
                psi=0.05,  # No drift
                bin_psi=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
                bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                reference_counts=np.array([20, 20, 20, 20, 20]),
                test_counts=np.array([19, 21, 20, 20, 20]),
                reference_percents=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                test_percents=np.array([0.19, 0.21, 0.2, 0.2, 0.2]),
                n_bins=5,
                is_categorical=False,
                alert_level="green",
                interpretation="No drift",
            ),
            drifted=False,
            n_methods_run=1,
            n_methods_detected=0,
            drift_probability=0.0,
        ),
        FeatureDriftResult(
            feature="feature_b",
            psi_result=PSIResult(
                psi=0.15,  # Small drift
                bin_psi=np.array([0.03, 0.03, 0.03, 0.03, 0.03]),
                bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                reference_counts=np.array([20, 20, 20, 20, 20]),
                test_counts=np.array([15, 25, 20, 20, 20]),
                reference_percents=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                test_percents=np.array([0.15, 0.25, 0.2, 0.2, 0.2]),
                n_bins=5,
                is_categorical=False,
                alert_level="yellow",
                interpretation="Small drift",
            ),
            drifted=False,
            n_methods_run=1,
            n_methods_detected=0,
            drift_probability=0.0,
        ),
        FeatureDriftResult(
            feature="feature_c",
            psi_result=PSIResult(
                psi=0.25,  # Significant drift
                bin_psi=np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
                bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                reference_counts=np.array([20, 20, 20, 20, 20]),
                test_counts=np.array([10, 30, 20, 20, 20]),
                reference_percents=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                test_percents=np.array([0.1, 0.3, 0.2, 0.2, 0.2]),
                n_bins=5,
                is_categorical=False,
                alert_level="red",
                interpretation="Significant drift",
            ),
            drifted=True,
            n_methods_run=1,
            n_methods_detected=1,
            drift_probability=1.0,
        ),
        FeatureDriftResult(
            feature="feature_d",
            psi_result=PSIResult(
                psi=0.08,  # No drift
                bin_psi=np.array([0.016, 0.016, 0.016, 0.016, 0.016]),
                bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                reference_counts=np.array([20, 20, 20, 20, 20]),
                test_counts=np.array([18, 22, 20, 20, 20]),
                reference_percents=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                test_percents=np.array([0.18, 0.22, 0.2, 0.2, 0.2]),
                n_bins=5,
                is_categorical=False,
                alert_level="green",
                interpretation="No drift",
            ),
            drifted=False,
            n_methods_run=1,
            n_methods_detected=0,
            drift_probability=0.0,
        ),
        FeatureDriftResult(
            feature="feature_e",
            psi_result=PSIResult(
                psi=0.30,  # Strong drift
                bin_psi=np.array([0.06, 0.06, 0.06, 0.06, 0.06]),
                bin_edges=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
                reference_counts=np.array([20, 20, 20, 20, 20]),
                test_counts=np.array([5, 35, 20, 20, 20]),
                reference_percents=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                test_percents=np.array([0.05, 0.35, 0.2, 0.2, 0.2]),
                n_bins=5,
                is_categorical=False,
                alert_level="red",
                interpretation="Strong drift",
            ),
            drifted=True,
            n_methods_run=1,
            n_methods_detected=1,
            drift_probability=1.0,
        ),
    ]

    drift_results = DriftSummaryResult(
        feature_results=drift_feature_results,
        n_features=5,
        n_features_drifted=2,
        drifted_features=["feature_c", "feature_e"],
        overall_drifted=True,
    )

    return FeatureOutcomeResult(
        features=features,
        ic_results=ic_results,
        importance_results=importance_results,
        drift_results=drift_results,
    )


@pytest.fixture
def sample_correlation_matrix() -> pl.DataFrame:
    """Create sample correlation matrix with some highly correlated features."""
    # feature_b and feature_d are highly correlated (0.85)
    # Others have moderate correlation
    # Create as polars DataFrame directly
    corr_data = {
        "feature": ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"],
        "feature_a": [1.0, 0.3, 0.2, 0.25, 0.1],
        "feature_b": [0.3, 1.0, 0.4, 0.85, 0.35],  # High correlation with feature_d
        "feature_c": [0.2, 0.4, 1.0, 0.45, 0.5],
        "feature_d": [0.25, 0.85, 0.45, 1.0, 0.4],  # High correlation with feature_b
        "feature_e": [0.1, 0.35, 0.5, 0.4, 1.0],
    }
    return pl.DataFrame(corr_data)


class TestFeatureSelector:
    """Test suite for FeatureSelector class."""

    def test_initialization(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test selector initialization."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        assert len(selector.initial_features) == 5
        assert len(selector.selected_features) == 5
        assert len(selector.removed_features) == 0
        assert len(selector.selection_steps) == 0

    def test_initialization_with_custom_features(
        self, sample_outcome_results: FeatureOutcomeResult
    ):
        """Test initialization with custom feature list."""
        initial = ["feature_a", "feature_b", "feature_c"]
        selector = FeatureSelector(sample_outcome_results, initial_features=initial)

        assert len(selector.initial_features) == 3
        assert len(selector.selected_features) == 3

    def test_filter_by_ic_threshold(self, sample_outcome_results: FeatureOutcomeResult):
        """Test IC filtering with threshold."""
        selector = FeatureSelector(sample_outcome_results)

        # Filter with threshold 0.02 (should keep a, b, d)
        selector.filter_by_ic(threshold=0.02)

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # feature_a (0.05), feature_b (0.03), feature_d (0.025) should pass
        # feature_c (0.01), feature_e (0.002) should be removed
        assert "feature_a" in selected
        assert "feature_b" in selected
        assert "feature_d" in selected
        assert "feature_c" in removed
        assert "feature_e" in removed

    def test_filter_by_ic_with_lag(self, sample_outcome_results: FeatureOutcomeResult):
        """Test IC filtering with specific lag."""
        selector = FeatureSelector(sample_outcome_results)

        # Filter using lag=5 with threshold 0.03
        selector.filter_by_ic(threshold=0.03, lag=5)

        selected = selector.get_selected_features()

        # At lag=5: feature_a (0.06), feature_b (0.04), feature_d (0.03) should pass
        assert "feature_a" in selected
        assert "feature_b" in selected
        assert "feature_d" in selected

    def test_filter_by_ic_min_periods(self, sample_outcome_results: FeatureOutcomeResult):
        """Test IC filtering respects min_periods."""
        # Modify one feature to have insufficient observations
        sample_outcome_results.ic_results["feature_a"].n_observations = 10

        selector = FeatureSelector(sample_outcome_results)
        selector.filter_by_ic(threshold=0.01, min_periods=20)

        removed = selector.get_removed_features()

        # feature_a should be removed due to insufficient observations
        assert "feature_a" in removed

    def test_filter_by_importance_mdi(self, sample_outcome_results: FeatureOutcomeResult):
        """Test importance filtering with MDI."""
        selector = FeatureSelector(sample_outcome_results)

        # Filter with MDI threshold 0.15 (should keep a, b, d, e)
        selector.filter_by_importance(threshold=0.15, method="mdi")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # feature_c (0.05) should be removed
        assert "feature_c" in removed
        assert len(selected) == 4

    def test_filter_by_importance_permutation(self, sample_outcome_results: FeatureOutcomeResult):
        """Test importance filtering with permutation."""
        selector = FeatureSelector(sample_outcome_results)

        selector.filter_by_importance(threshold=0.15, method="permutation")

        selected = selector.get_selected_features()

        # Based on permutation_importance values
        assert "feature_a" in selected  # 0.25
        assert "feature_b" in selected  # 0.22
        assert "feature_d" in selected  # 0.18

    def test_filter_by_importance_shap(self, sample_outcome_results: FeatureOutcomeResult):
        """Test importance filtering with SHAP."""
        selector = FeatureSelector(sample_outcome_results)

        selector.filter_by_importance(threshold=0.15, method="shap")

        selected = selector.get_selected_features()

        # Based on shap_mean values
        assert "feature_a" in selected  # 0.28
        assert "feature_b" in selected  # 0.23
        assert "feature_d" in selected  # 0.19

    def test_filter_by_importance_top_k(self, sample_outcome_results: FeatureOutcomeResult):
        """Test importance filtering with top-K selection."""
        selector = FeatureSelector(sample_outcome_results)

        # Keep only top 3 features
        selector.filter_by_importance(threshold=0, method="mdi", top_k=3)

        selected = selector.get_selected_features()

        assert len(selected) == 3
        # Should keep feature_a, feature_b, feature_d (highest MDI)
        assert "feature_a" in selected
        assert "feature_b" in selected
        assert "feature_d" in selected

    def test_filter_by_correlation(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test correlation filtering."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        # Filter with threshold 0.8 (should remove one of feature_b/feature_d)
        selector.filter_by_correlation(threshold=0.8, keep_strategy="higher_ic")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # feature_b and feature_d have correlation 0.85
        # feature_b has IC 0.03, feature_d has IC 0.025
        # Should keep feature_b (higher IC) and remove feature_d
        assert "feature_b" in selected
        assert "feature_d" in removed

    def test_filter_by_correlation_higher_importance(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test correlation filtering with importance strategy."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        selector.filter_by_correlation(threshold=0.8, keep_strategy="higher_importance")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # feature_b has MDI 0.25, feature_d has MDI 0.20
        # Should keep feature_b and remove feature_d
        assert "feature_b" in selected
        assert "feature_d" in removed

    def test_filter_by_correlation_first_strategy(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test correlation filtering with first strategy."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        selector.filter_by_correlation(threshold=0.8, keep_strategy="first")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # Should keep feature_b (alphabetically first) and remove feature_d
        assert "feature_b" in selected
        assert "feature_d" in removed

    def test_filter_by_correlation_no_matrix(self, sample_outcome_results: FeatureOutcomeResult):
        """Test correlation filtering raises error without correlation matrix."""
        selector = FeatureSelector(sample_outcome_results, correlation_matrix=None)

        with pytest.raises(ValueError, match="Correlation matrix required"):
            selector.filter_by_correlation(threshold=0.8)

    def test_filter_by_drift_psi(self, sample_outcome_results: FeatureOutcomeResult):
        """Test drift filtering with PSI method."""
        selector = FeatureSelector(sample_outcome_results)

        # Filter features with red alert (PSI >= 0.2)
        selector.filter_by_drift(threshold=0.2, method="psi")

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        # feature_c and feature_e have red alert
        assert "feature_c" in removed
        assert "feature_e" in removed
        assert "feature_a" in selected
        assert "feature_b" in selected
        assert "feature_d" in selected

    def test_filter_by_drift_consensus(self, sample_outcome_results: FeatureOutcomeResult):
        """Test drift filtering with consensus method."""
        selector = FeatureSelector(sample_outcome_results)

        # Filter features with drift_probability >= 0.5
        selector.filter_by_drift(threshold=0.5, method="consensus")

        removed = selector.get_removed_features()

        # feature_c and feature_e have drift_probability = 1.0
        assert "feature_c" in removed
        assert "feature_e" in removed

    def test_filter_by_drift_no_results(self, sample_outcome_results: FeatureOutcomeResult):
        """Test drift filtering raises error without drift results."""
        # Remove drift results
        sample_outcome_results.drift_results = None

        selector = FeatureSelector(sample_outcome_results)

        with pytest.raises(ValueError, match="Drift results not available"):
            selector.filter_by_drift(threshold=0.2)

    def test_run_pipeline(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test running complete pipeline."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        selector.run_pipeline(
            [
                ("ic", {"threshold": 0.02, "min_periods": 20}),
                ("correlation", {"threshold": 0.8}),
                ("importance", {"threshold": 0.15, "method": "mdi"}),
            ]
        )

        selected = selector.get_selected_features()

        # After IC filter: a, b, d remain (c, e removed)
        # After correlation filter: a, b remain (d removed due to correlation with b)
        # After importance filter: a, b remain (both > 0.15 MDI)
        assert len(selected) == 2
        assert "feature_a" in selected
        assert "feature_b" in selected

    def test_run_pipeline_with_drift(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test pipeline including drift filtering."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        selector.run_pipeline(
            [
                ("drift", {"threshold": 0.2, "method": "psi"}),
                ("ic", {"threshold": 0.02}),
                ("importance", {"threshold": 0.15, "method": "mdi"}),
            ]
        )

        selected = selector.get_selected_features()

        # After drift: a, b, d remain (c, e removed)
        # After IC: a, b, d remain
        # After importance: a, b, d remain (all > 0.15)
        assert "feature_a" in selected
        assert "feature_b" in selected
        assert "feature_d" in selected

    def test_run_pipeline_invalid_filter(self, sample_outcome_results: FeatureOutcomeResult):
        """Test pipeline raises error for invalid filter."""
        selector = FeatureSelector(sample_outcome_results)

        with pytest.raises(ValueError, match="Unknown filter"):
            selector.run_pipeline([("invalid_filter", {})])

    def test_method_chaining(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test method chaining for fluent API."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        # Chain multiple filters
        result = (
            selector.filter_by_ic(threshold=0.02)
            .filter_by_correlation(threshold=0.8)
            .filter_by_importance(threshold=0.15, method="mdi")
        )

        # Should return self
        assert result is selector

        selected = selector.get_selected_features()
        assert len(selected) > 0

    def test_get_selection_report(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test selection report generation."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        selector.filter_by_ic(threshold=0.02)
        selector.filter_by_importance(threshold=0.15, method="mdi")

        report = selector.get_selection_report()

        assert len(report.initial_features) == 5
        assert len(report.final_features) < 5
        assert len(report.steps) == 2
        assert report.total_removed > 0
        assert report.removal_rate > 0

        # Test summary
        summary = report.summary()
        assert "Feature Selection Report" in summary
        assert "Initial Features:" in summary
        assert "Final Features:" in summary

    def test_selection_step_summary(self, sample_outcome_results: FeatureOutcomeResult):
        """Test selection step summary generation."""
        selector = FeatureSelector(sample_outcome_results)

        selector.filter_by_ic(threshold=0.02)

        step = selector.selection_steps[0]
        summary = step.summary()

        assert "IC Filtering" in summary
        assert "threshold" in summary
        assert "removed" in summary.lower()

    def test_reset(self, sample_outcome_results: FeatureOutcomeResult):
        """Test resetting selector to initial state."""
        selector = FeatureSelector(sample_outcome_results)

        # Apply some filters
        selector.filter_by_ic(threshold=0.02)
        selector.filter_by_importance(threshold=0.15, method="mdi")

        # Verify features were removed
        assert len(selector.selected_features) < len(selector.initial_features)
        assert len(selector.selection_steps) > 0

        # Reset
        selector.reset()

        # Verify reset
        assert len(selector.selected_features) == len(selector.initial_features)
        assert len(selector.removed_features) == 0
        assert len(selector.selection_steps) == 0

    def test_edge_case_no_features_pass(self, sample_outcome_results: FeatureOutcomeResult):
        """Test handling when no features pass filter."""
        selector = FeatureSelector(sample_outcome_results)

        # Very high threshold - no features should pass
        selector.filter_by_ic(threshold=1.0)

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        assert len(selected) == 0
        assert len(removed) == 5

    def test_edge_case_all_features_pass(self, sample_outcome_results: FeatureOutcomeResult):
        """Test handling when all features pass filter."""
        selector = FeatureSelector(sample_outcome_results)

        # Very low threshold - all features should pass
        selector.filter_by_ic(threshold=0.0)

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        assert len(selected) == 5
        assert len(removed) == 0

    def test_edge_case_insufficient_features_for_correlation(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test correlation filtering with insufficient features."""
        # Start with only one feature
        selector = FeatureSelector(
            sample_outcome_results,
            sample_correlation_matrix,
            initial_features=["feature_a"],
        )

        # Should not crash
        selector.filter_by_correlation(threshold=0.8)

        # Should have same feature
        assert len(selector.selected_features) == 1
        assert "feature_a" in selector.selected_features


class TestSelectionReport:
    """Test suite for SelectionReport class."""

    def test_report_post_init(self):
        """Test report post-initialization calculation."""
        report = SelectionReport(
            initial_features=["a", "b", "c", "d", "e"],
            final_features=["a", "b"],
        )

        assert report.total_removed == 3
        assert report.removal_rate == 60.0

    def test_report_summary_formatting(self):
        """Test report summary formatting."""
        step = SelectionStep(
            step_name="Test Filter",
            parameters={"threshold": 0.5},
            features_before=5,
            features_after=3,
            features_removed=["c", "d"],
            features_kept=["a", "b", "e"],
            reasoning="Test reasoning",
        )

        report = SelectionReport(
            initial_features=["a", "b", "c", "d", "e"],
            final_features=["a", "b", "e"],
            steps=[step],
        )

        summary = report.summary()

        assert "Feature Selection Report" in summary
        assert "Initial Features: 5" in summary
        assert "Final Features: 3" in summary
        assert "Removed: 2 (40.0%)" in summary
        assert "Test Filter" in summary
        assert "✓ a" in summary
        assert "✓ b" in summary
        assert "✓ e" in summary


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_selection_workflow(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test complete realistic selection workflow."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        # Realistic pipeline
        selector.run_pipeline(
            [
                # First remove drifted features
                ("drift", {"threshold": 0.2, "method": "psi"}),
                # Then filter by predictive power
                ("ic", {"threshold": 0.02, "min_periods": 20}),
                # Remove redundant features
                ("correlation", {"threshold": 0.8, "keep_strategy": "higher_ic"}),
                # Finally select most important features
                ("importance", {"threshold": 0.15, "method": "mdi"}),
            ]
        )

        selected = selector.get_selected_features()
        report = selector.get_selection_report()

        # Verify we have a reasonable selection
        assert len(selected) > 0
        assert len(selected) < len(sample_outcome_results.features)

        # Verify report has all steps
        assert len(report.steps) == 4

        # Print report for manual inspection
        print("\n" + report.summary())

    def test_top_k_selection_workflow(
        self, sample_outcome_results: FeatureOutcomeResult, sample_correlation_matrix: pl.DataFrame
    ):
        """Test top-K feature selection workflow."""
        selector = FeatureSelector(sample_outcome_results, sample_correlation_matrix)

        # First remove problematic features, then select top K
        selector.run_pipeline(
            [
                ("drift", {"threshold": 0.2, "method": "psi"}),
                ("correlation", {"threshold": 0.8}),
                ("importance", {"threshold": 0, "method": "mdi", "top_k": 2}),
            ]
        )

        selected = selector.get_selected_features()

        # Should have exactly 2 features
        assert len(selected) == 2

        # Should be the top 2 by MDI among non-drifted features
        assert "feature_a" in selected
        assert "feature_b" in selected
