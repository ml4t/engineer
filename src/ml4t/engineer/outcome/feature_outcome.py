# mypy: disable-error-code="call-arg,misc,dict-item,assignment,no-any-return"
# ruff: noqa: UP006, UP045, ARG002, SIM102
"""Feature-outcome relationship analysis (Module C).

Exports:
    FeatureOutcome - Main class for feature-outcome analysis
        .run_analysis(features, returns, ...) -> FeatureOutcomeResult
        Orchestrates IC, classification, threshold, and ML diagnostics.

    compute_shap_importance(model, X, ...) -> DataFrame
        SHAP-based feature importance with optional interactions.

    Classes:
        FeatureICResults - IC analysis results per feature
        FeatureImportanceResults - ML feature importance results
        FeatureOutcomeResult - Complete analysis results with recommendations

This module provides comprehensive analysis of how features relate to outcomes:
- **IC Analysis**: Information Coefficient for predictive power
- **Binary Classification**: Precision, recall, lift for signal quality
- **Threshold Optimization**: Find optimal thresholds for signals
- **ML Diagnostics**: Feature importance, SHAP, interactions
- **Drift Detection**: Monitor feature distribution stability

The FeatureOutcome class orchestrates all analyses into a unified workflow.

Example:
    >>> from ml4t.engineer.outcome.feature_outcome import FeatureOutcome
    >>> from ml4t.engineer.config.feature_config import ModuleCConfig
    >>>
    >>> # Basic usage
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, returns_df)
    >>> print(results.summary)
    >>>
    >>> # Custom configuration
    >>> config = ModuleCConfig(
    ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21]),
    ...     ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True)
    ... )
    >>> analyzer = FeatureOutcome(config=config)
    >>> results = analyzer.run_analysis(features_df, returns_df, verbose=True)
    >>>
    >>> # Get recommendations
    >>> for rec in results.get_recommendations():
    ...     print(f"• {rec}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Import TYPE_CHECKING for type hints without runtime import
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # noqa: UP035

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from ml4t.engineer.config.feature_config import ModuleCConfig
from ml4t.engineer.outcome.drift import DriftSummaryResult, analyze_drift
from ml4t.engineer.utils.dependencies import DEPS, warn_if_missing

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FeatureICResults:
    """IC analysis results for a single feature.

    Attributes:
        feature: Feature name
        ic_mean: Mean IC across time
        ic_std: Standard deviation of IC
        ic_ir: IC Information Ratio (mean/std)
        t_stat: T-statistic for IC
        p_value: P-value for IC significance
        ic_by_lag: IC values by forward horizon
        hac_adjusted: Whether HAC adjustment was applied
        n_observations: Number of observations used
    """

    feature: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    t_stat: float = 0.0
    p_value: float = 1.0
    ic_by_lag: Dict[int, float] = field(default_factory=dict)
    hac_adjusted: bool = False
    n_observations: int = 0


@dataclass
class FeatureImportanceResults:
    """ML feature importance results.

    Attributes:
        feature: Feature name
        mdi_importance: Mean Decrease in Impurity (tree-based)
        permutation_importance: Permutation-based importance
        permutation_std: Standard deviation of permutation importance
        shap_mean: Mean absolute SHAP value (if computed)
        shap_std: Standard deviation of SHAP values (if computed)
        rank_mdi: Rank by MDI importance
        rank_permutation: Rank by permutation importance
    """

    feature: str
    mdi_importance: float = 0.0
    permutation_importance: float = 0.0
    permutation_std: float = 0.0
    shap_mean: Optional[float] = None
    shap_std: Optional[float] = None
    rank_mdi: int = 0
    rank_permutation: int = 0


@dataclass
class FeatureOutcomeResult:
    """Comprehensive feature-outcome analysis results.

    This aggregates all Module C analyses into a single result object.

    Attributes:
        features: List of features analyzed
        ic_results: IC analysis per feature
        importance_results: ML importance per feature
        drift_results: Drift detection results (if enabled)
        interaction_matrix: H-statistic interaction matrix (if computed)
        summary: High-level summary DataFrame
        recommendations: Actionable recommendations
        config: Configuration used
        metadata: Analysis metadata (runtime, samples, etc.)
        errors: Dict of features that failed analysis
    """

    features: List[str]
    ic_results: Dict[str, FeatureICResults] = field(default_factory=dict)
    importance_results: Dict[str, FeatureImportanceResults] = field(default_factory=dict)
    drift_results: Optional[DriftSummaryResult] = None
    interaction_matrix: Optional[pd.DataFrame] = None
    summary: Optional[pd.DataFrame] = None
    recommendations: List[str] = field(default_factory=list)
    config: Optional[ModuleCConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export summary as DataFrame.

        Returns:
            DataFrame with one row per feature, columns for all metrics
        """
        if self.summary is not None:
            return self.summary

        # Build summary from individual results
        rows = []
        for feature in self.features:
            row = {"feature": feature}

            # IC metrics (with defaults for missing)
            if feature in self.ic_results:
                ic = self.ic_results[feature]
                row.update(
                    {
                        "ic_mean": ic.ic_mean,
                        "ic_std": ic.ic_std,
                        "ic_ir": ic.ic_ir,
                        "ic_pvalue": ic.p_value,
                        "ic_significant": ic.p_value < 0.05,
                    }
                )
            else:
                # Add NaN placeholders if IC not computed
                row.update(
                    {
                        "ic_mean": np.nan,
                        "ic_std": np.nan,
                        "ic_ir": np.nan,
                        "ic_pvalue": np.nan,
                        "ic_significant": False,
                    }
                )

            # Importance metrics (with defaults for missing)
            if feature in self.importance_results:
                imp = self.importance_results[feature]
                row.update(
                    {
                        "mdi_importance": imp.mdi_importance,
                        "permutation_importance": imp.permutation_importance,
                        "shap_mean": imp.shap_mean if imp.shap_mean is not None else np.nan,
                        "shap_std": imp.shap_std if imp.shap_std is not None else np.nan,
                        "rank_mdi": imp.rank_mdi,
                        "rank_permutation": imp.rank_permutation,
                    }
                )
            else:
                row.update(
                    {
                        "mdi_importance": np.nan,
                        "permutation_importance": np.nan,
                        "shap_mean": np.nan,
                        "shap_std": np.nan,
                        "rank_mdi": np.nan,
                        "rank_permutation": np.nan,
                    }
                )

            # Drift metrics
            if self.drift_results is not None:
                drift_df = self.drift_results.to_dataframe()
                # Convert to pandas if polars
                if isinstance(drift_df, pl.DataFrame):
                    drift_df = drift_df.to_pandas()

                feature_drift = drift_df[drift_df["feature"] == feature]
                if len(feature_drift) > 0:
                    row["drifted"] = feature_drift["drifted"].iloc[0]
                    if "psi" in feature_drift.columns:
                        row["psi"] = feature_drift["psi"].iloc[0]
                else:
                    row["drifted"] = False
            else:
                row["drifted"] = False

            # Error status
            row["error"] = feature in self.errors

            rows.append(row)

        return pd.DataFrame(rows)

    def get_top_features(
        self, n: int = 10, by: str = "ic_ir", ascending: bool = False
    ) -> List[str]:
        """Get top N features by specified metric.

        Args:
            n: Number of features to return
            by: Metric to sort by ('ic_ir', 'ic_mean', 'mdi_importance', etc.)
            ascending: Sort in ascending order (default: descending)

        Returns:
            List of top feature names

        Example:
            >>> # Get 5 features with highest IC IR
            >>> top_features = results.get_top_features(n=5, by='ic_ir')
        """
        df = self.to_dataframe()

        if by not in df.columns:
            available = [c for c in df.columns if c != "feature"]
            raise ValueError(f"Metric '{by}' not found. Available: {available}")

        # Remove features with errors or NaN values
        df = df[~df["error"]]
        df = df.dropna(subset=[by])

        # Sort and return top N
        df = df.sort_values(by=by, ascending=ascending)
        return df.head(n)["feature"].tolist()

    def get_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis.

        Returns:
            List of recommendation strings

        Example:
            >>> for rec in results.get_recommendations():
            ...     print(f"• {rec}")
        """
        if self.recommendations:
            return self.recommendations

        # Generate recommendations from results
        recommendations = []
        df = self.to_dataframe()

        # Strong signals (high IC IR, no drift)
        strong = df[(df["ic_ir"] > 2.0) & (~df.get("drifted", False))]
        if len(strong) > 0:
            for _, row in strong.iterrows():
                recommendations.append(
                    f"{row['feature']}: Strong predictive power (IC IR={row['ic_ir']:.2f}), "
                    f"stable distribution"
                )

        # Weak signals (low IC)
        weak = df[df["ic_ir"].abs() < 0.5]
        if len(weak) > 0:
            features = ", ".join(weak["feature"].tolist()[:5])
            more = f" (+{len(weak) - 5} more)" if len(weak) > 5 else ""
            recommendations.append(f"Consider removing weak signals: {features}{more}")

        # Drifted features
        if "drifted" in df.columns:
            drifted = df[df["drifted"] == True]  # noqa: E712
            if len(drifted) > 0:
                for _, row in drifted.iterrows():
                    recommendations.append(
                        f"{row['feature']}: Distribution drift detected - "
                        f"consider retraining or investigation"
                    )

        # Features with errors
        if len(self.errors) > 0:
            error_features = ", ".join(list(self.errors.keys())[:3])
            more = f" (+{len(self.errors) - 3} more)" if len(self.errors) > 3 else ""
            recommendations.append(f"Analysis failed for: {error_features}{more}")

        return recommendations


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame | pl.DataFrame | NDArray[Any],
    feature_names: list[str] | None = None,
    check_additivity: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Compute SHAP (SHapley Additive exPlanations) values and aggregate to feature importance.

    SHAP values provide a unified measure of feature importance based on Shapley values
    from cooperative game theory. Each feature's contribution to a prediction is
    calculated by considering all possible feature coalitions.

    **Key advantages over MDI**:
    - Theoretically sound (based on game theory)
    - Consistent (removing a feature always decreases its importance)
    - Local explanations (per-prediction feature contributions)
    - Interaction-aware (accounts for feature interactions)
    - Unbiased (no bias toward high-cardinality features)

    Parameters
    ----------
    model : Any
        Fitted tree-based model compatible with shap.TreeExplainer
        (LightGBM, XGBoost, or sklearn tree models)
    X : pd.DataFrame | pl.DataFrame | npt.NDArray[np.float64]
        Feature matrix for SHAP computation (typically test/validation set)
        Shape: (n_samples, n_features)
    feature_names : list[str] | None, default None
        Feature names for labeling. If None, uses column names from DataFrame
        or generates numeric names for arrays
    check_additivity : bool, default False
        Verify that SHAP values sum to model predictions (sanity check).
        Disable for speed (recommended for large datasets).
    max_samples : int | None, default None
        Maximum number of samples to compute SHAP values for (for speed).
        If None, uses all samples. Recommended: 1000-5000 for large datasets.

    Returns
    -------
    dict[str, Any]
        Dictionary with SHAP importance results:
        - shap_values: SHAP values array, shape (n_samples, n_features)
        - importances: Mean absolute SHAP values per feature (sorted descending)
        - feature_names: Feature labels (sorted by importance)
        - base_value: Expected model output (average prediction)
        - n_features: Number of features
        - n_samples: Number of samples used for SHAP computation
        - model_type: Type of model used
        - additivity_verified: Whether additivity check passed

    Raises
    ------
    ImportError
        If shap library not installed
    ValueError
        If model is not supported by TreeExplainer
    RuntimeError
        If SHAP computation fails

    Examples
    --------
    >>> import lightgbm as lgb
    >>> import numpy as np
    >>> from ml4t.engineer.outcome.feature_outcome import compute_shap_importance
    >>>
    >>> # Train model
    >>> X_train = np.random.randn(1000, 10)
    >>> y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    >>> model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Compute SHAP importance on test set
    >>> X_test = np.random.randn(200, 10)
    >>> result = compute_shap_importance(model, X_test, max_samples=200)
    >>>
    >>> # View top features
    >>> for feat, imp in zip(result['feature_names'][:5], result['importances'][:5]):
    ...     print(f"{feat}: {imp:.4f}")

    Notes
    -----
    **Performance Considerations**:
    - TreeExplainer is fast (~1-10ms per sample) but scales with dataset size
    - For large datasets (>10K samples), use max_samples parameter
    - SHAP is typically 10-100x slower than MDI but provides better insights
    - Memory usage: O(n_samples * n_features) for storing SHAP values

    **Global Importance Aggregation**:
    Feature importance is computed as:
        importance[j] = mean(|shap_values[:, j]|)

    This measures the average magnitude of feature j's contribution across
    all predictions. Unlike MDI, this is unbiased and reflects actual
    predictive power on the dataset.

    References
    ----------
    - Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting
      Model Predictions". NeurIPS. https://arxiv.org/abs/1705.07874
    - Lundberg, S. M. et al. (2020). "From local explanations to global understanding
      with explainable AI for trees". Nature Machine Intelligence.
    """
    # Check if shap is installed
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "SHAP library is not installed. Install with: uv pip install shap>=0.43.0"
        ) from e

    # Convert X to appropriate format
    if isinstance(X, pl.DataFrame):
        X_array = X.to_numpy()
        if feature_names is None:
            feature_names = list(X.columns)
    elif isinstance(X, pd.DataFrame):
        X_array = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_array = np.asarray(X)

    # Validate shape
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X_array.shape}")

    # Set default feature names if needed
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    # Ensure feature_names is a list
    feature_names = list(feature_names)

    n_samples_full, n_features = X_array.shape

    # Subsample if requested (for performance on large datasets)
    if max_samples is not None and n_samples_full > max_samples:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_samples_full, size=max_samples, replace=False)
        X_array = X_array[sample_idx]
        n_samples = max_samples
    else:
        n_samples = n_samples_full

    # Validate feature names length
    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) does not match "
            f"number of features in X ({n_features})"
        )

    # Create TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise ValueError(
            f"Failed to create TreeExplainer for model type {type(model).__name__}. "
            f"Ensure model is a supported tree-based model (LightGBM, XGBoost, sklearn trees). "
            f"Error: {e}"
        ) from e

    # Compute SHAP values
    try:
        shap_values_raw = explainer.shap_values(X_array, check_additivity=check_additivity)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compute SHAP values. This may indicate model compatibility issues. "
            f"Error: {e}"
        ) from e

    # Handle binary classification (returns list of arrays OR 3D array)
    if isinstance(shap_values_raw, list):
        # Binary classification (older SHAP versions) - use positive class, else first class
        shap_values = shap_values_raw[1] if len(shap_values_raw) == 2 else shap_values_raw[0]
    else:
        shap_values = shap_values_raw
        # Handle 3D array for binary/multiclass (newer SHAP versions)
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 2:
                # Binary classification: take positive class (index 1)
                shap_values = shap_values[:, :, 1]
            else:
                # Multiclass: aggregate across classes (mean absolute)
                shap_values = np.mean(np.abs(shap_values), axis=2)

    # Validate SHAP values shape
    if shap_values.shape != (n_samples, n_features):
        raise RuntimeError(
            f"Unexpected SHAP values shape: {shap_values.shape}, expected ({n_samples}, {n_features})"
        )

    # Compute feature importance as mean absolute SHAP value
    importances = np.mean(np.abs(shap_values), axis=0)

    # Sort by importance (descending)
    sorted_idx = np.argsort(importances)[::-1]

    # Get base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, list | np.ndarray):
        # For binary/multiclass, take positive class or first class
        base_value = base_value[1] if len(base_value) == 2 else base_value[0]

    # Determine model type
    model_type = f"{type(model).__module__}.{type(model).__name__}"

    return {
        "shap_values": shap_values,
        "importances": importances[sorted_idx],
        "feature_names": [feature_names[i] for i in sorted_idx],
        "base_value": float(base_value) if base_value is not None else 0.0,
        "n_features": n_features,
        "n_samples": n_samples,
        "model_type": model_type,
        "additivity_verified": check_additivity,
    }


class FeatureOutcome:
    """Main orchestration class for feature-outcome analysis (Module C).

    Coordinates comprehensive analysis of feature-outcome relationships:
    - IC analysis (Information Coefficient for predictive power)
    - Binary classification metrics (precision, recall, lift)
    - Threshold optimization
    - ML feature importance (MDI, permutation, SHAP)
    - Feature interactions (H-statistic)
    - Drift detection

    This class provides a unified interface for all Module C analyses,
    handling configuration, execution, and result aggregation.

    Examples:
        >>> # Basic usage with defaults
        >>> analyzer = FeatureOutcome()
        >>> results = analyzer.run_analysis(features_df, returns_df)
        >>> print(results.summary)
        >>>
        >>> # Custom configuration
        >>> config = ModuleCConfig(
        ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21]),
        ...     ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True)
        ... )
        >>> analyzer = FeatureOutcome(config=config)
        >>> results = analyzer.run_analysis(features_df, returns_df, verbose=True)
        >>>
        >>> # Select specific features
        >>> results = analyzer.run_analysis(
        ...     features_df,
        ...     returns_df,
        ...     feature_names=['momentum', 'volume', 'volatility']
        ... )
        >>>
        >>> # Get actionable insights
        >>> top_features = results.get_top_features(n=10, by='ic_ir')
        >>> recommendations = results.get_recommendations()
    """

    def __init__(self, config: Optional[ModuleCConfig] = None):
        """Initialize FeatureOutcome analyzer.

        Args:
            config: Module C configuration. Uses defaults if None.

        Example:
            >>> # Default configuration
            >>> analyzer = FeatureOutcome()
            >>>
            >>> # Custom configuration
            >>> config = ModuleCConfig(
            ...     ic=ICConfig(hac_adjustment=True),
            ...     ml_diagnostics=MLDiagnosticsConfig(drift_detection=True)
            ... )
            >>> analyzer = FeatureOutcome(config=config)
        """
        self.config = config or ModuleCConfig()

    def run_analysis(
        self,
        features: pd.DataFrame | pl.DataFrame,
        outcomes: pd.DataFrame | pl.DataFrame | pd.Series | npt.NDArray[np.float64],
        feature_names: Optional[List[str]] = None,
        date_col: Optional[str] = None,
        verbose: bool = False,
    ) -> FeatureOutcomeResult:
        """Run comprehensive feature-outcome analysis.

        Executes all enabled analyses in Module C configuration:
        1. IC analysis (if ic.enabled)
        2. ML feature importance (if ml_diagnostics.enabled)
        3. Feature interactions (if ml_diagnostics.enabled)
        4. Drift detection (if ml_diagnostics.drift_detection)

        Args:
            features: Feature DataFrame (T x N) with date index or date column
            outcomes: Outcome/returns DataFrame, Series, or array (T x 1 or T)
            feature_names: Specific features to analyze (None = all numeric)
            date_col: Date column name if not in index
            verbose: Print progress messages

        Returns:
            FeatureOutcomeResult with all analyses

        Raises:
            ValueError: If inputs are invalid or incompatible

        Example:
            >>> # Basic usage
            >>> results = analyzer.run_analysis(features_df, returns_df)
            >>>
            >>> # With progress tracking
            >>> results = analyzer.run_analysis(
            ...     features_df, returns_df, verbose=True
            ... )
            >>> # Output:
            >>> # Analyzing 10 features...
            >>> # [1/10] feature1: IC=0.15, importance=0.25
            >>> # [2/10] feature2: IC=0.08, importance=0.12
            >>> # ...
            >>> # Analysis complete in 12.3s
        """
        start_time = time.time()

        # ===================================================================
        # 0. Configuration and Dependency Validation
        # ===================================================================
        if verbose:
            print("Validating configuration and dependencies...")

        # Check dependencies based on configuration
        missing_deps = []
        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.feature_importance:
            if not DEPS.check("lightgbm"):
                missing_deps.append("lightgbm")
                if verbose:
                    print("  ⚠️  LightGBM not available - feature importance will be skipped")
                    print(f"      Install with: {DEPS.lightgbm.install_cmd}")

        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.shap_analysis:
            if not DEPS.check("shap"):
                missing_deps.append("shap")
                if verbose:
                    print("  ⚠️  SHAP not available - SHAP analysis will be skipped")
                    print(f"      Install with: {DEPS.shap.install_cmd}")

        # Log what features are available
        if verbose and not missing_deps:
            print("  ✓ All required dependencies available")

        # ===================================================================
        # 1. Input Validation and Preprocessing
        # ===================================================================
        if verbose:
            print("Validating inputs...")

        # Convert to pandas for consistency
        if isinstance(features, pl.DataFrame):
            features = features.to_pandas()
        if isinstance(outcomes, pl.DataFrame):
            outcomes = outcomes.to_pandas()
        if isinstance(outcomes, pl.Series):
            outcomes = outcomes.to_pandas()

        # Handle outcomes format
        if isinstance(outcomes, pd.Series):
            outcomes_series = outcomes
        elif isinstance(outcomes, np.ndarray):
            if outcomes.ndim == 1:
                outcomes_series = pd.Series(outcomes, index=features.index)
            else:
                # Take first column
                outcomes_series = pd.Series(outcomes[:, 0], index=features.index)
        elif isinstance(outcomes, pd.DataFrame):
            # Take first column
            outcomes_series = outcomes.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported outcomes type: {type(outcomes)}")

        # Validate alignment
        if len(features) != len(outcomes_series):
            raise ValueError(
                f"Features ({len(features)}) and outcomes ({len(outcomes_series)}) "
                f"must have same length"
            )

        # ===================================================================
        # 2. Determine Features to Analyze
        # ===================================================================
        if feature_names is None:
            # Use all numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_names = numeric_cols
        else:
            # Validate specified features exist
            missing = set(feature_names) - set(features.columns)
            if missing:
                raise ValueError(f"Features not found in DataFrame: {missing}")

        if not feature_names:
            raise ValueError("No features to analyze")

        if verbose:
            print(f"Analyzing {len(feature_names)} features...")

        # ===================================================================
        # 3. Initialize Results Storage
        # ===================================================================
        ic_results = {}
        importance_results = {}
        errors = {}

        # ===================================================================
        # 4. Run IC Analysis (if enabled)
        # ===================================================================
        if self.config.ic.enabled:
            if verbose:
                print("Running IC analysis...")

            for i, feature in enumerate(feature_names, 1):
                try:
                    feature_data = features[feature].values
                    outcome_data = outcomes_series.values

                    # Compute IC by lag using lag_structure
                    from scipy.stats import spearmanr

                    ic_by_lag = {}
                    ic_values = []

                    for lag in self.config.ic.lag_structure:
                        # Shift outcome forward by lag periods
                        if lag == 0:
                            # Contemporaneous correlation
                            lagged_outcome = outcome_data
                            feature_aligned = feature_data
                        else:
                            # Forward-looking correlation
                            # feature[t] vs outcome[t+lag]
                            lagged_outcome = outcome_data[lag:]
                            feature_aligned = feature_data[:-lag]

                        # Remove NaN pairs
                        mask = ~(np.isnan(feature_aligned) | np.isnan(lagged_outcome))
                        feature_clean = feature_aligned[mask]
                        outcome_clean = lagged_outcome[mask]

                        if len(feature_clean) < 10:
                            continue

                        # Compute IC for this lag
                        ic_lag, _ = spearmanr(feature_clean, outcome_clean)
                        ic_by_lag[lag] = float(ic_lag)
                        ic_values.append(ic_lag)

                    if not ic_values:
                        errors[feature] = "Insufficient non-NaN samples for all lags"
                        continue

                    # Aggregate IC statistics across lags
                    ic_mean = np.mean(ic_values)
                    ic_std = np.std(ic_values)
                    ic_ir = ic_mean / (ic_std + 1e-10)

                    # Get p-value from lag 0 (or first available lag)
                    first_lag = self.config.ic.lag_structure[0]
                    lagged_outcome = outcome_data[first_lag:] if first_lag > 0 else outcome_data
                    feature_aligned = feature_data[:-first_lag] if first_lag > 0 else feature_data
                    mask = ~(np.isnan(feature_aligned) | np.isnan(lagged_outcome))
                    _, p_value = spearmanr(feature_aligned[mask], lagged_outcome[mask])

                    ic_results[feature] = FeatureICResults(
                        feature=feature,
                        ic_mean=ic_mean,
                        ic_std=ic_std,
                        ic_ir=ic_ir,
                        p_value=p_value,
                        ic_by_lag=ic_by_lag,
                        n_observations=len(feature_aligned[mask]),
                    )

                    if verbose and i % max(1, len(feature_names) // 10) == 0:
                        print(f"  [{i}/{len(feature_names)}] {feature}: IC={ic_mean:.3f}")

                except Exception as e:
                    errors[feature] = str(e)
                    if verbose:
                        print(f"  [{i}/{len(feature_names)}] {feature}: ERROR - {e}")

        # ===================================================================
        # 5. Run ML Diagnostics (if enabled)
        # ===================================================================
        if self.config.ml_diagnostics.enabled and self.config.ml_diagnostics.feature_importance:
            if verbose:
                print("Running feature importance analysis...")

            try:
                # Check if LightGBM is available
                if warn_if_missing("lightgbm", "feature importance", "skipping analysis"):
                    import lightgbm as lgb

                    # Prepare data
                    X = features[feature_names].values
                    y = outcomes_series.values

                    # Remove NaN rows
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    X_clean = X[mask]
                    y_clean = y[mask]

                    if len(X_clean) >= 100:
                        # Train simple model for importance
                        model = lgb.LGBMRegressor(
                            n_estimators=100, max_depth=3, random_state=42, verbose=-1
                        )
                        model.fit(X_clean, y_clean)

                        # Get MDI importance
                        mdi_importances = model.feature_importances_

                        # Rank features
                        ranks = np.argsort(mdi_importances)[::-1]

                        for idx, feature in enumerate(feature_names):
                            if feature not in errors:
                                rank = int(np.where(ranks == idx)[0][0]) + 1
                                importance_results[feature] = FeatureImportanceResults(
                                    feature=feature,
                                    mdi_importance=float(mdi_importances[idx]),
                                    rank_mdi=rank,
                                )

                        if verbose:
                            top_feature = feature_names[ranks[0]]
                            print(
                                f"  Top feature by MDI: {top_feature} "
                                f"(importance={mdi_importances[ranks[0]]:.3f})"
                            )
                    else:
                        if verbose:
                            print(
                                f"  Insufficient clean samples for importance analysis ({len(X_clean)}/100)"
                            )
                else:
                    if verbose:
                        print("  Feature importance skipped (LightGBM not available)")

            except Exception as e:
                if verbose:
                    print(f"  Feature importance failed: {e}")

        # ===================================================================
        # 5b. Run SHAP Analysis (if enabled)
        # ===================================================================
        if (
            self.config.ml_diagnostics.enabled
            and self.config.ml_diagnostics.shap_analysis
            and "model" in locals()
        ):
            if verbose:
                print("Running SHAP importance analysis...")

            try:
                # Check if SHAP is available
                if warn_if_missing("shap", "SHAP analysis", "skipping"):
                    # Compute SHAP importance using the trained model
                    # Use max_samples for performance (5000 samples is good balance)
                    max_shap_samples = min(5000, len(X_clean))

                    if verbose:
                        print(f"  Computing SHAP values for {max_shap_samples} samples...")

                    shap_result = compute_shap_importance(
                        model=model,
                        X=X_clean,
                        feature_names=feature_names,
                        check_additivity=False,  # Disable for speed
                        max_samples=max_shap_samples,
                    )

                    # Update importance results with SHAP values
                    shap_importances = shap_result["importances"]
                    shap_feature_names = shap_result["feature_names"]

                    # Compute standard deviation of absolute SHAP values per feature
                    shap_values = shap_result["shap_values"]
                    shap_stds = np.std(np.abs(shap_values), axis=0)

                    # Create a mapping from feature name to SHAP importance
                    shap_map = {}
                    shap_std_map = {}
                    for fname, imp, _std_val in zip(
                        shap_feature_names, shap_importances, shap_stds
                    ):
                        # Find original index
                        orig_idx = feature_names.index(fname)
                        shap_map[fname] = float(imp)
                        shap_std_map[fname] = float(shap_stds[orig_idx])

                    # Update existing importance results with SHAP values
                    for feature in feature_names:
                        if feature not in errors:
                            if feature in importance_results:
                                # Update existing result
                                importance_results[feature].shap_mean = shap_map.get(feature)
                                importance_results[feature].shap_std = shap_std_map.get(feature)
                            else:
                                # Create new result (shouldn't happen if MDI ran, but just in case)
                                importance_results[feature] = FeatureImportanceResults(
                                    feature=feature,
                                    shap_mean=shap_map.get(feature),
                                    shap_std=shap_std_map.get(feature),
                                )

                    if verbose:
                        # Show top feature by SHAP
                        top_shap_feature = shap_feature_names[0]
                        top_shap_importance = shap_importances[0]
                        print(
                            f"  Top feature by SHAP: {top_shap_feature} "
                            f"(importance={top_shap_importance:.4f})"
                        )
                        print(f"  SHAP computation completed in {shap_result['n_samples']} samples")
                else:
                    if verbose:
                        print("  SHAP analysis skipped (SHAP library not available)")

            except Exception as e:
                if verbose:
                    print(f"  SHAP analysis failed: {e}")
                # Don't fail the entire analysis if SHAP fails
                # This is optional analysis, so we can continue

        # ===================================================================
        # 6. Run Drift Detection (if enabled)
        # ===================================================================
        drift_results = None
        if self.config.ml_diagnostics.drift_detection:
            if verbose:
                print("Running drift detection...")

            try:
                # Split data into reference (first half) and test (second half)
                split_idx = len(features) // 2
                reference = features[feature_names].iloc[:split_idx]
                test = features[feature_names].iloc[split_idx:]

                drift_results = analyze_drift(
                    reference,
                    test,
                    features=feature_names,
                    methods=["psi", "wasserstein"],  # Fast methods only
                )

                if verbose:
                    n_drifted = drift_results.n_features_drifted
                    print(f"  Drift detected in {n_drifted}/{len(feature_names)} features")

            except Exception as e:
                if verbose:
                    print(f"  Drift detection failed: {e}")

        # ===================================================================
        # 7. Build Summary and Generate Recommendations
        # ===================================================================
        result = FeatureOutcomeResult(
            features=feature_names,
            ic_results=ic_results,
            importance_results=importance_results,
            drift_results=drift_results,
            config=self.config,
            errors=errors,
            metadata={
                "n_features": len(feature_names),
                "n_observations": len(features),
                "n_errors": len(errors),
                "computation_time": time.time() - start_time,
                "ic_enabled": self.config.ic.enabled,
                "ml_diagnostics_enabled": self.config.ml_diagnostics.enabled,
                "shap_analysis_enabled": self.config.ml_diagnostics.shap_analysis,
                "drift_detection_enabled": self.config.ml_diagnostics.drift_detection,
            },
        )

        # Build summary DataFrame
        result.summary = result.to_dataframe()

        # Generate recommendations
        result.recommendations = result.get_recommendations()

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nAnalysis complete in {elapsed:.1f}s")
            print(f"  Features analyzed: {len(feature_names)}")
            print(f"  Errors: {len(errors)}")
            if result.recommendations:
                print(f"  Recommendations: {len(result.recommendations)}")

        return result


# Re-export for convenience
__all__ = [
    "FeatureICResults",
    "FeatureImportanceResults",
    "FeatureOutcomeResult",
    "FeatureOutcome",
]
