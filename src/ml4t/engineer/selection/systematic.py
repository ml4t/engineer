"""Systematic feature selection for ML pipelines.

This module provides a comprehensive feature selection workflow that combines
multiple filtering criteria:

- **IC Filtering**: Select features with strong information coefficient
- **Importance Filtering**: Select features based on MDI/permutation/SHAP importance
- **Correlation Filtering**: Remove redundant highly correlated features
- **Drift Filtering**: Remove features with unstable distributions

The FeatureSelector class supports both individual filters and automated pipelines
that execute multiple filters in sequence.

Example - Basic Usage:
    >>> from ml4t.engineer.selection import FeatureSelector
    >>> from ml4t.engineer.outcome import FeatureOutcome
    >>> from ml4t.engineer.relationships import compute_correlation_matrix
    >>>
    >>> # Run feature-outcome analysis
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, returns_df)
    >>>
    >>> # Compute correlation matrix
    >>> corr_matrix = compute_correlation_matrix(features_df)
    >>>
    >>> # Create selector
    >>> selector = FeatureSelector(
    ...     outcome_results=results,
    ...     correlation_matrix=corr_matrix
    ... )
    >>>
    >>> # Apply individual filters
    >>> selector.filter_by_ic(threshold=0.02)
    >>> selector.filter_by_correlation(threshold=0.8)
    >>> selected = selector.get_selected_features()
    >>> print(f"Selected {len(selected)} features")

Example - Pipeline:
    >>> # Run automated pipeline
    >>> selector = FeatureSelector(results, corr_matrix)
    >>> selector.run_pipeline([
    ...     ("ic", {"threshold": 0.02, "min_periods": 20}),
    ...     ("correlation", {"threshold": 0.8}),
    ...     ("importance", {"threshold": 0.01, "method": "mdi"})
    ... ])
    >>> report = selector.get_selection_report()
    >>> print(report)

Example - With Drift:
    >>> # Include drift filtering
    >>> selector = FeatureSelector(results, corr_matrix)
    >>> selector.filter_by_drift(threshold=0.2)  # PSI threshold
    >>> selector.filter_by_importance(threshold=0.05, method="shap")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from ml4t.engineer.outcome.feature_outcome import FeatureOutcomeResult


@dataclass
class SelectionStep:
    """Record of a single selection step.

    Attributes:
        step_name: Name of the filter applied
        parameters: Parameters used for the filter
        features_before: Number of features before filter
        features_after: Number of features after filter
        features_removed: List of features removed in this step
        features_kept: List of features kept after this step
        reasoning: Explanation of why features were removed
    """

    step_name: str
    parameters: dict[str, Any]
    features_before: int
    features_after: int
    features_removed: list[str]
    features_kept: list[str]
    reasoning: str

    def summary(self) -> str:
        """Generate summary of this selection step."""
        pct_removed = 100 * len(self.features_removed) / max(1, self.features_before)
        return (
            f"{self.step_name}: {self.features_before} → {self.features_after} "
            f"({len(self.features_removed)} removed, {pct_removed:.1f}%)\n"
            f"  Parameters: {self.parameters}\n"
            f"  Reasoning: {self.reasoning}"
        )


@dataclass
class SelectionReport:
    """Complete feature selection report.

    Attributes:
        initial_features: Features at start of selection
        final_features: Features after all filters
        steps: List of selection steps applied
        total_removed: Total number of features removed
        removal_rate: Percentage of features removed
    """

    initial_features: list[str]
    final_features: list[str]
    steps: list[SelectionStep] = field(default_factory=list)
    total_removed: int = field(init=False)
    removal_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.total_removed = len(self.initial_features) - len(self.final_features)
        self.removal_rate = 100 * self.total_removed / max(1, len(self.initial_features))

    def summary(self) -> str:
        """Generate comprehensive selection report."""
        lines = [
            "=" * 70,
            "Feature Selection Report",
            "=" * 70,
            f"Initial Features: {len(self.initial_features)}",
            f"Final Features: {len(self.final_features)}",
            f"Removed: {self.total_removed} ({self.removal_rate:.1f}%)",
            "",
            "Selection Pipeline:",
            "-" * 70,
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(f"\nStep {i}: {step.summary()}")

        lines.extend(
            [
                "",
                "-" * 70,
                "Final Selected Features:",
                "-" * 70,
            ]
        )
        for feature in sorted(self.final_features):
            lines.append(f"  ✓ {feature}")

        lines.append("=" * 70)

        return "\n".join(lines)


class FeatureSelector:
    """Systematic feature selection with multiple filtering criteria.

    This class provides a comprehensive feature selection workflow that combines
    IC analysis, importance scoring, correlation filtering, and drift detection.

    Parameters
    ----------
    outcome_results : FeatureOutcomeResult
        Results from feature-outcome analysis (IC, importance, drift)
    correlation_matrix : pl.DataFrame, optional
        Feature correlation matrix from compute_correlation_matrix()
    initial_features : list[str], optional
        Initial set of features to select from.
        If None, uses all features from outcome_results.

    Attributes
    ----------
    selected_features : set[str]
        Current set of selected features (updated by filters)
    removed_features : set[str]
        Features removed by filters
    selection_steps : list[SelectionStep]
        History of selection steps applied
    """

    def __init__(
        self,
        outcome_results: FeatureOutcomeResult,
        correlation_matrix: pl.DataFrame | None = None,
        initial_features: list[str] | None = None,
    ):
        """Initialize feature selector.

        Args:
            outcome_results: Feature-outcome analysis results
            correlation_matrix: Optional correlation matrix for correlation filtering
            initial_features: Optional initial feature set (defaults to all features)
        """
        self.outcome_results = outcome_results
        self.correlation_matrix = correlation_matrix

        # Initialize feature sets
        if initial_features is not None:
            self.initial_features = set(initial_features)
        else:
            self.initial_features = set(outcome_results.features)

        self.selected_features = self.initial_features.copy()
        self.removed_features: set[str] = set()
        self.selection_steps: list[SelectionStep] = []

    def filter_by_ic(
        self,
        threshold: float,
        min_periods: int = 1,
        lag: int | None = None,
    ) -> FeatureSelector:
        """Filter features by Information Coefficient.

        Keeps features with |IC| > threshold. IC measures the predictive power
        of a feature for the outcome variable.

        Parameters
        ----------
        threshold : float
            Minimum absolute IC value to keep a feature.
            Typical values: 0.01-0.05 (1-5% correlation)
        min_periods : int, default 1
            Minimum number of observations required for IC calculation
        lag : int | None, default None
            Specific forward lag to use for filtering.
            If None, uses mean IC across all lags.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining

        Examples
        --------
        >>> selector.filter_by_ic(threshold=0.02, min_periods=20)
        >>> # Keep only features with |IC| > 0.02
        """
        features_before = len(self.selected_features)
        features_to_remove = []

        for feature in self.selected_features:
            if feature not in self.outcome_results.ic_results:
                continue

            ic_result = self.outcome_results.ic_results[feature]

            # Check min_periods
            if ic_result.n_observations < min_periods:
                features_to_remove.append(feature)
                continue

            # Get IC value
            if lag is not None:
                # Use specific lag
                if lag not in ic_result.ic_by_lag:
                    features_to_remove.append(feature)
                    continue
                ic_value = abs(ic_result.ic_by_lag[lag])
            else:
                # Use mean IC
                ic_value = abs(ic_result.ic_mean)

            # Filter by threshold
            if ic_value < threshold:
                features_to_remove.append(feature)

        # Update selected features
        self.selected_features -= set(features_to_remove)
        self.removed_features |= set(features_to_remove)

        # Record step
        step = SelectionStep(
            step_name="IC Filtering",
            parameters={
                "threshold": threshold,
                "min_periods": min_periods,
                "lag": lag,
            },
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=f"Removed features with |IC| < {threshold}",
        )
        self.selection_steps.append(step)

        return self

    def filter_by_importance(
        self,
        threshold: float,
        method: Literal["mdi", "permutation", "shap"] = "mdi",
        top_k: int | None = None,
    ) -> FeatureSelector:
        """Filter features by ML importance scores.

        Keeps features with importance > threshold or top K most important features.

        Parameters
        ----------
        threshold : float
            Minimum importance value to keep a feature.
            Set to 0 if using top_k instead.
        method : {"mdi", "permutation", "shap"}, default "mdi"
            Importance method to use:
            - "mdi": Mean Decrease in Impurity (tree-based)
            - "permutation": Permutation importance
            - "shap": SHAP values (if available)
        top_k : int | None, default None
            If provided, keeps only the top K most important features
            regardless of threshold.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining

        Examples
        --------
        >>> # Threshold-based filtering
        >>> selector.filter_by_importance(threshold=0.01, method="mdi")
        >>>
        >>> # Top-K filtering
        >>> selector.filter_by_importance(threshold=0, method="shap", top_k=20)
        """
        features_before = len(self.selected_features)

        # Get importance values for all selected features
        feature_importance = []
        for feature in self.selected_features:
            if feature not in self.outcome_results.importance_results:
                continue

            imp_result = self.outcome_results.importance_results[feature]

            # Get importance based on method
            if method == "mdi":
                importance = imp_result.mdi_importance
            elif method == "permutation":
                importance = imp_result.permutation_importance
            elif method == "shap":
                if imp_result.shap_mean is None:
                    continue  # Skip features without SHAP
                importance = imp_result.shap_mean
            else:
                raise ValueError(
                    f"Unknown importance method: {method}. Choose from 'mdi', 'permutation', 'shap'"
                )

            feature_importance.append((feature, importance))

        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Determine features to keep
        if top_k is not None:
            # Keep top K
            features_to_keep = [f for f, _ in feature_importance[:top_k]]
            reasoning = f"Kept top {top_k} features by {method} importance"
        else:
            # Keep features above threshold
            features_to_keep = [f for f, imp in feature_importance if imp >= threshold]
            reasoning = f"Removed features with {method} importance < {threshold}"

        # Update selected features
        features_to_remove = [f for f in self.selected_features if f not in features_to_keep]
        self.selected_features = set(features_to_keep)
        self.removed_features |= set(features_to_remove)

        # Record step
        step = SelectionStep(
            step_name=f"Importance Filtering ({method.upper()})",
            parameters={
                "threshold": threshold,
                "method": method,
                "top_k": top_k,
            },
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=reasoning,
        )
        self.selection_steps.append(step)

        return self

    def filter_by_correlation(
        self,
        threshold: float,
        keep_strategy: Literal["higher_ic", "higher_importance", "first"] = "higher_ic",
    ) -> FeatureSelector:
        """Remove highly correlated features to reduce redundancy.

        When two features have correlation > threshold, keeps one based on
        the keep_strategy.

        Parameters
        ----------
        threshold : float
            Maximum absolute correlation allowed between features.
            Typical values: 0.7-0.9
        keep_strategy : {"higher_ic", "higher_importance", "first"}, default "higher_ic"
            Strategy for choosing which feature to keep:
            - "higher_ic": Keep feature with higher |IC|
            - "higher_importance": Keep feature with higher MDI importance
            - "first": Keep feature that appears first alphabetically

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining

        Raises
        ------
        ValueError
            If correlation_matrix was not provided during initialization

        Examples
        --------
        >>> selector.filter_by_correlation(threshold=0.8, keep_strategy="higher_ic")
        >>> # Removes one feature from each pair with correlation > 0.8
        """
        if self.correlation_matrix is None:
            raise ValueError(
                "Correlation matrix required for correlation filtering. "
                "Provide correlation_matrix during FeatureSelector initialization."
            )

        features_before = len(self.selected_features)
        features_to_remove = set()

        # Convert to pandas for easier manipulation

        # Check if 'feature' column exists (indexed format)
        if "feature" in self.correlation_matrix.columns:
            # Convert from Polars with 'feature' column to pandas with index
            corr_df = self.correlation_matrix.to_pandas()
            corr_df = corr_df.set_index("feature")
        else:
            # Already in proper format
            corr_df = self.correlation_matrix.to_pandas()

        # Filter correlation matrix to selected features only
        selected_list = sorted(self.selected_features)
        if not all(f in corr_df.index for f in selected_list):
            # Some features missing from correlation matrix - skip them
            selected_list = [f for f in selected_list if f in corr_df.index]

        if len(selected_list) < 2:
            # Not enough features to check correlation
            step = SelectionStep(
                step_name="Correlation Filtering",
                parameters={"threshold": threshold, "keep_strategy": keep_strategy},
                features_before=features_before,
                features_after=features_before,
                features_removed=[],
                features_kept=list(self.selected_features),
                reasoning="Insufficient features for correlation filtering",
            )
            self.selection_steps.append(step)
            return self

        corr_subset = corr_df.loc[selected_list, selected_list]

        # Find pairs above threshold
        for i, feat1 in enumerate(selected_list):
            if feat1 in features_to_remove:
                continue

            for feat2 in selected_list[i + 1 :]:
                if feat2 in features_to_remove:
                    continue

                corr_value = abs(corr_subset.loc[feat1, feat2])

                if corr_value > threshold:
                    # Decide which feature to remove
                    if keep_strategy == "higher_ic":
                        ic1 = abs(
                            self.outcome_results.ic_results.get(
                                feat1, type("", (), {"ic_mean": 0})()
                            ).ic_mean
                        )
                        ic2 = abs(
                            self.outcome_results.ic_results.get(
                                feat2, type("", (), {"ic_mean": 0})()
                            ).ic_mean
                        )
                        to_remove = feat2 if ic1 > ic2 else feat1

                    elif keep_strategy == "higher_importance":
                        imp1 = self.outcome_results.importance_results.get(
                            feat1, type("", (), {"mdi_importance": 0})()
                        ).mdi_importance
                        imp2 = self.outcome_results.importance_results.get(
                            feat2, type("", (), {"mdi_importance": 0})()
                        ).mdi_importance
                        to_remove = feat2 if imp1 > imp2 else feat1

                    else:  # "first"
                        to_remove = feat2  # Keep first alphabetically

                    features_to_remove.add(to_remove)

        # Update selected features
        self.selected_features -= features_to_remove
        self.removed_features |= features_to_remove

        # Record step
        step = SelectionStep(
            step_name="Correlation Filtering",
            parameters={"threshold": threshold, "keep_strategy": keep_strategy},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=list(features_to_remove),
            features_kept=list(self.selected_features),
            reasoning=f"Removed features with correlation > {threshold} using {keep_strategy} strategy",
        )
        self.selection_steps.append(step)

        return self

    def filter_by_drift(
        self,
        threshold: float = 0.2,
        method: Literal["psi", "consensus"] = "psi",
    ) -> FeatureSelector:
        """Remove features with unstable distributions (drift).

        Features with distribution drift may not generalize well to new data.

        Parameters
        ----------
        threshold : float, default 0.2
            Drift threshold:
            - For PSI: PSI >= 0.2 indicates significant drift
            - For consensus: drift_probability >= threshold
        method : {"psi", "consensus"}, default "psi"
            Drift detection method:
            - "psi": Use PSI alert level (red = drifted)
            - "consensus": Use consensus probability from multiple methods

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining

        Raises
        ------
        ValueError
            If drift_results not available in outcome_results

        Examples
        --------
        >>> selector.filter_by_drift(threshold=0.2, method="psi")
        >>> # Removes features with PSI >= 0.2 (red alert)
        """
        if self.outcome_results.drift_results is None:
            raise ValueError(
                "Drift results not available. Run outcome analysis with drift_detection=True."
            )

        features_before = len(self.selected_features)
        features_to_remove = []

        drift_results = self.outcome_results.drift_results

        for feature_result in drift_results.feature_results:
            feature = feature_result.feature

            if feature not in self.selected_features:
                continue

            # Determine if feature has drifted
            if method == "psi":
                # Use PSI alert level - red alert indicates drift
                if (
                    feature_result.psi_result is not None
                    and feature_result.psi_result.alert_level == "red"
                ):
                    features_to_remove.append(feature)

            elif method == "consensus":
                # Use consensus drift probability
                if feature_result.drift_probability >= threshold:
                    features_to_remove.append(feature)

            else:
                raise ValueError(f"Unknown drift method: {method}. Choose from 'psi', 'consensus'")

        # Update selected features
        self.selected_features -= set(features_to_remove)
        self.removed_features |= set(features_to_remove)

        # Record step
        step = SelectionStep(
            step_name="Drift Filtering",
            parameters={"threshold": threshold, "method": method},
            features_before=features_before,
            features_after=len(self.selected_features),
            features_removed=features_to_remove,
            features_kept=list(self.selected_features),
            reasoning=f"Removed features with {method} drift >= {threshold}",
        )
        self.selection_steps.append(step)

        return self

    def run_pipeline(
        self,
        steps: list[tuple[str, dict[str, Any]]],
    ) -> FeatureSelector:
        """Execute multiple selection filters in sequence.

        Provides a convenient way to run a complete selection pipeline
        with multiple filtering criteria.

        Parameters
        ----------
        steps : list[tuple[str, dict]]
            List of (filter_name, parameters) tuples.
            Valid filter names:
            - "ic": filter_by_ic()
            - "importance": filter_by_importance()
            - "correlation": filter_by_correlation()
            - "drift": filter_by_drift()

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining

        Examples
        --------
        >>> selector.run_pipeline([
        ...     ("ic", {"threshold": 0.02, "min_periods": 20}),
        ...     ("correlation", {"threshold": 0.8}),
        ...     ("importance", {"threshold": 0.01, "method": "mdi"}),
        ...     ("drift", {"threshold": 0.2})
        ... ])
        >>> print(selector.get_selection_report().summary())
        """
        for filter_name, params in steps:
            if filter_name == "ic":
                self.filter_by_ic(**params)
            elif filter_name == "importance":
                self.filter_by_importance(**params)
            elif filter_name == "correlation":
                self.filter_by_correlation(**params)
            elif filter_name == "drift":
                self.filter_by_drift(**params)
            else:
                raise ValueError(
                    f"Unknown filter: {filter_name}. "
                    "Valid filters: ic, importance, correlation, drift"
                )

        return self

    def get_selected_features(self) -> list[str]:
        """Get current list of selected features.

        Returns
        -------
        list[str]
            Sorted list of selected feature names
        """
        return sorted(self.selected_features)

    def get_removed_features(self) -> list[str]:
        """Get list of features that were removed.

        Returns
        -------
        list[str]
            Sorted list of removed feature names
        """
        return sorted(self.removed_features)

    def get_selection_report(self) -> SelectionReport:
        """Generate comprehensive selection report.

        Returns
        -------
        SelectionReport
            Report with selection steps and final features
        """
        return SelectionReport(
            initial_features=sorted(self.initial_features),
            final_features=self.get_selected_features(),
            steps=self.selection_steps,
        )

    def reset(self) -> FeatureSelector:
        """Reset selector to initial feature set.

        Clears all filters and selection steps.

        Returns
        -------
        self : FeatureSelector
            Returns self for method chaining
        """
        self.selected_features = self.initial_features.copy()
        self.removed_features = set()
        self.selection_steps = []
        return self
