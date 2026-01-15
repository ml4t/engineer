# mypy: disable-error-code="call-arg,arg-type,assignment,no-untyped-def,return-value"
"""Distribution drift detection for feature monitoring.

Exports:
    analyze_drift(reference, test, methods=["psi", "wasserstein"], ...) -> DriftSummaryResult
        Multi-method drift analysis with consensus-based flagging. Main API.

    compute_psi(reference, test, n_bins=10, ...) -> PSIResult
        Population Stability Index for univariate drift detection.

    compute_wasserstein_distance(reference, test, ...) -> WassersteinResult
        Optimal transport metric for continuous feature drift.

    compute_domain_classifier_drift(reference, test, ...) -> DomainClassifierResult
        ML-based multivariate drift detection with feature importance.

    Classes:
        PSIResult - PSI computation results with alert level
        WassersteinResult - Wasserstein distance with calibrated threshold
        DomainClassifierResult - Domain classifier AUC and feature importance
        FeatureDriftResult - Per-feature drift results across methods
        DriftSummaryResult - Comprehensive drift analysis results

This module provides comprehensive drift detection with three complementary methods
and a unified analysis interface:

**Individual Methods**:
- **PSI (Population Stability Index)**: Bin-based distribution comparison
- **Wasserstein Distance**: Optimal transport metric for continuous features
- **Domain Classifier**: ML-based multivariate drift detection with feature importance

**Unified Interface**:
- **analyze_drift()**: Multi-method drift analysis with consensus-based flagging

Distribution drift is critical for ML model monitoring:
- Feature distributions change over time (concept drift)
- Model performance degrades when test distribution differs from training
- Early detection allows proactive model retraining
- Multi-method consensus increases confidence in drift detection

PSI Interpretation:
    - PSI < 0.1: No significant change (green)
    - 0.1 ≤ PSI < 0.2: Small change, monitor (yellow)
    - PSI ≥ 0.2: Significant change, investigate (red)

Wasserstein Distance Interpretation:
    - W = 0: Identical distributions
    - W > 0: Distribution drift detected
    - Larger values indicate greater drift magnitude
    - Threshold calibrated via permutation testing

Domain Classifier Interpretation:
    - AUC ≈ 0.5: No drift (random guess between reference and test)
    - AUC = 0.6: Weak drift
    - AUC = 0.7-0.8: Moderate drift
    - AUC > 0.9: Strong drift
    - Feature importance identifies which features drifted

When to Use:
    - **PSI**: Categorical features or when binning is acceptable
    - **Wasserstein**: Continuous features, more sensitive to small shifts
    - **Domain Classifier**: Multivariate drift, interaction detection
    - **analyze_drift()**: Comprehensive analysis with multiple methods
    - Model monitoring: Compare production data to training data
    - Temporal drift: Compare recent data to historical baseline
    - Segmentation drift: Compare distributions across segments

References:
    - Yurdakul, B. (2018). Statistical Properties of Population Stability Index.
      https://scholarship.richmond.edu/honors-theses/1131/
    - Webb, G. I., et al. (2016). Characterizing concept drift.
      Data Mining and Knowledge Discovery, 30(4), 964-994.
    - Villani, C. (2009). Optimal Transport: Old and New. Springer.
    - Ramdas, A., et al. (2017). On Wasserstein Two-Sample Testing and Related
      Families of Nonparametric Tests. Entropy, 19(2), 47.
    - Lopez-Paz, D., & Oquab, M. (2017). Revisiting Classifier Two-Sample Tests.
      ICLR 2017.
    - Rabanser, S., et al. (2019). Failing Loudly: An Empirical Study of Methods
      for Detecting Dataset Shift. NeurIPS 2019.

Example - Individual Methods:
    >>> import numpy as np
    >>> from ml4t.engineer.outcome.drift import (
    ...     compute_psi, compute_wasserstein_distance, compute_domain_classifier_drift
    ... )
    >>>
    >>> # PSI for univariate drift
    >>> reference = np.random.normal(0, 1, 1000)
    >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted
    >>> psi_result = compute_psi(reference, test, n_bins=10)
    >>> print(f"PSI: {psi_result.psi:.4f}, Alert: {psi_result.alert_level}")
    >>>
    >>> # Wasserstein for continuous features
    >>> ws_result = compute_wasserstein_distance(reference, test)
    >>> print(f"Wasserstein: {ws_result.distance:.4f}, Drifted: {ws_result.drifted}")

Example - Unified Analysis:
    >>> import pandas as pd
    >>> from ml4t.engineer.outcome.drift import analyze_drift
    >>>
    >>> # Create reference and test datasets
    >>> reference = pd.DataFrame({
    ...     'feature1': np.random.normal(0, 1, 1000),
    ...     'feature2': np.random.normal(0, 1, 1000),
    ... })
    >>> test = pd.DataFrame({
    ...     'feature1': np.random.normal(0.5, 1, 1000),  # Drifted
    ...     'feature2': np.random.normal(0, 1, 1000),    # Stable
    ... })
    >>>
    >>> # Comprehensive drift analysis with all methods
    >>> result = analyze_drift(reference, test)
    >>> print(result.summary())
    >>> print(f"Drifted features: {result.drifted_features}")
    >>>
    >>> # Get detailed results as DataFrame
    >>> df = result.to_dataframe()
    >>> print(df)
    >>>
    >>> # Use specific methods only
    >>> result = analyze_drift(reference, test, methods=['psi', 'wasserstein'])
    >>>
    >>> # Customize consensus threshold (default: 0.5)
    >>> result = analyze_drift(reference, test, consensus_threshold=0.66)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from scipy.stats import wasserstein_distance

# Check optional ML dependencies
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PSIResult:
    """Result of Population Stability Index calculation.

    Attributes:
        psi: Overall PSI value (sum of bin-level PSI contributions)
        bin_psi: PSI contribution per bin
        bin_edges: Bin boundaries (continuous) or category labels (categorical)
        reference_counts: Number of samples per bin in reference distribution
        test_counts: Number of samples per bin in test distribution
        reference_percents: Percentage of samples per bin in reference
        test_percents: Percentage of samples per bin in test
        n_bins: Number of bins used
        is_categorical: Whether feature is categorical
        alert_level: Alert level based on PSI thresholds
            - "green": PSI < 0.1 (no significant change)
            - "yellow": 0.1 ≤ PSI < 0.2 (small change, monitor)
            - "red": PSI ≥ 0.2 (significant change, investigate)
        interpretation: Human-readable interpretation
    """

    psi: float
    bin_psi: npt.NDArray[np.float64]
    bin_edges: npt.NDArray[np.float64] | list[str]
    reference_counts: npt.NDArray[np.intp]
    test_counts: npt.NDArray[np.intp]
    reference_percents: npt.NDArray[np.float64]
    test_percents: npt.NDArray[np.float64]
    n_bins: int
    is_categorical: bool
    alert_level: Literal["green", "yellow", "red"]
    interpretation: str

    def summary(self) -> str:
        """Return formatted summary of PSI results."""
        lines = [
            "Population Stability Index (PSI) Report",
            "=" * 50,
            f"PSI Value: {self.psi:.4f}",
            f"Alert Level: {self.alert_level.upper()}",
            f"Feature Type: {'Categorical' if self.is_categorical else 'Continuous'}",
            f"Number of Bins: {self.n_bins}",
            "",
            f"Interpretation: {self.interpretation}",
            "",
            "Bin-Level Analysis:",
            "-" * 50,
        ]

        # Add bin-level details
        for i in range(self.n_bins):
            if self.is_categorical:
                bin_label = self.bin_edges[i]
            else:
                if i == 0:
                    bin_label = f"(-inf, {self.bin_edges[i + 1]:.3f}]"
                elif i == self.n_bins - 1:
                    bin_label = f"({self.bin_edges[i]:.3f}, +inf)"
                else:
                    bin_label = f"({self.bin_edges[i]:.3f}, {self.bin_edges[i + 1]:.3f}]"

            lines.append(
                f"Bin {i + 1:2d} {bin_label:20s}: "
                f"Ref={self.reference_percents[i]:6.2%} "
                f"Test={self.test_percents[i]:6.2%} "
                f"PSI={self.bin_psi[i]:.4f}"
            )

        return "\n".join(lines)


def compute_psi(
    reference: npt.NDArray[np.float64] | pl.Series,
    test: npt.NDArray[np.float64] | pl.Series,
    n_bins: int = 10,
    is_categorical: bool = False,
    missing_category_handling: Literal["ignore", "separate", "error"] = "separate",
    psi_threshold_yellow: float = 0.1,
    psi_threshold_red: float = 0.2,
) -> PSIResult:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI measures the distribution shift between a reference dataset (e.g., training)
    and a test dataset (e.g., production). It quantifies how much the distribution
    has changed.

    Formula:
        PSI = Σ (test_% - ref_%) × ln(test_% / ref_%)

    For each bin i:
        PSI_i = (P_test[i] - P_ref[i]) × ln(P_test[i] / P_ref[i])

    Args:
        reference: Reference distribution (e.g., training data)
        test: Test distribution (e.g., production data)
        n_bins: Number of quantile bins for continuous features (default: 10)
        is_categorical: Whether feature is categorical (default: False)
        missing_category_handling: How to handle categories in test not in reference:
            - "ignore": Skip missing categories (not recommended)
            - "separate": Create separate bin for missing categories (default)
            - "error": Raise error if new categories found
        psi_threshold_yellow: Threshold for yellow alert (default: 0.1)
        psi_threshold_red: Threshold for red alert (default: 0.2)

    Returns:
        PSIResult with overall PSI, bin-level contributions, and interpretation

    Raises:
        ValueError: If inputs are invalid or missing categories found with "error" handling

    Example:
        >>> # Continuous feature
        >>> ref = np.random.normal(0, 1, 1000)
        >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted
        >>> result = compute_psi(ref, test, n_bins=10)
        >>> print(result.summary())
        >>>
        >>> # Categorical feature
        >>> ref_cat = np.array(['A', 'B', 'C'] * 100)
        >>> test_cat = np.array(['A', 'A', 'B'] * 100)  # Distribution changed
        >>> result = compute_psi(ref_cat, test_cat, is_categorical=True)
        >>> print(f"PSI: {result.psi:.4f}, Alert: {result.alert_level}")
    """
    # Convert to numpy arrays
    if isinstance(reference, pl.Series):
        reference = reference.to_numpy()
    if isinstance(test, pl.Series):
        test = test.to_numpy()

    reference = np.asarray(reference)
    test = np.asarray(test)

    # Validate inputs
    if len(reference) == 0 or len(test) == 0:
        raise ValueError("Reference and test arrays must not be empty")

    if not is_categorical:
        # Continuous feature: quantile binning
        bin_edges, ref_counts, test_counts = _bin_continuous(reference, test, n_bins)
        bin_labels = bin_edges  # Will be formatted in summary()
        # Update n_bins to reflect actual number after unique edge collapsing
        n_bins = len(bin_edges) - 1 if len(bin_edges) > 1 else 1
    else:
        # Categorical feature: category-based binning
        bin_labels, ref_counts, test_counts = _bin_categorical(
            reference, test, missing_category_handling
        )
        bin_edges = bin_labels
        n_bins = len(bin_labels)

    # Convert counts to percentages
    ref_percents = ref_counts / ref_counts.sum()
    test_percents = test_counts / test_counts.sum()

    # Compute PSI per bin with numerical stability
    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    ref_percents_safe = np.maximum(ref_percents, epsilon)
    test_percents_safe = np.maximum(test_percents, epsilon)

    # PSI formula: (test% - ref%) * ln(test% / ref%)
    bin_psi = (test_percents_safe - ref_percents_safe) * np.log(
        test_percents_safe / ref_percents_safe
    )

    # Total PSI is sum of bin contributions
    psi = float(np.sum(bin_psi))

    # Determine alert level
    if psi < psi_threshold_yellow:
        alert_level = "green"
        interpretation = (
            f"No significant distribution change detected (PSI={psi:.4f} < {psi_threshold_yellow}). "
            "Feature distribution is stable."
        )
    elif psi < psi_threshold_red:
        alert_level = "yellow"
        interpretation = (
            f"Small distribution change detected ({psi_threshold_yellow} ≤ PSI={psi:.4f} < {psi_threshold_red}). "
            "Monitor feature closely but no immediate action required."
        )
    else:
        alert_level = "red"
        interpretation = (
            f"Significant distribution change detected (PSI={psi:.4f} ≥ {psi_threshold_red}). "
            "Investigate cause and consider model retraining."
        )

    return PSIResult(
        psi=psi,
        bin_psi=bin_psi,
        bin_edges=bin_edges,
        reference_counts=ref_counts,
        test_counts=test_counts,
        reference_percents=ref_percents,
        test_percents=test_percents,
        n_bins=n_bins,
        is_categorical=is_categorical,
        alert_level=alert_level,
        interpretation=interpretation,
    )


def _bin_continuous(
    reference: npt.NDArray[np.float64], test: npt.NDArray[np.float64], n_bins: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Bin continuous features using quantiles from reference distribution.

    Uses quantile binning to ensure roughly equal-sized bins in reference distribution.
    Test distribution is binned using same bin edges.

    Args:
        reference: Reference data (used to compute quantiles)
        test: Test data (binned using reference quantiles)
        n_bins: Number of bins

    Returns:
        Tuple of (bin_edges, reference_counts, test_counts)
    """
    # Compute quantiles from reference distribution
    # Use (n_bins + 1) to get n_bins bins with n_bins + 1 edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)

    # Ensure edges are unique (handle constant features)
    bin_edges = np.unique(bin_edges)

    # If all values are the same, create a single bin
    if len(bin_edges) == 1:
        return bin_edges, np.array([len(reference)]), np.array([len(test)])

    # Bin both distributions using same edges
    # Use digitize for open-interval binning
    ref_bins = np.digitize(reference, bin_edges[1:-1])
    test_bins = np.digitize(test, bin_edges[1:-1])

    # Count samples per bin
    ref_counts = np.bincount(ref_bins, minlength=len(bin_edges) - 1)
    test_counts = np.bincount(test_bins, minlength=len(bin_edges) - 1)

    return bin_edges, ref_counts, test_counts


def _bin_categorical(
    reference: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    missing_handling: Literal["ignore", "separate", "error"],
) -> tuple[list[str], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Bin categorical features by category labels.

    Args:
        reference: Reference categories
        test: Test categories
        missing_handling: How to handle new categories in test

    Returns:
        Tuple of (category_labels, reference_counts, test_counts)

    Raises:
        ValueError: If new categories found and missing_handling="error"
    """
    # Get unique categories from reference
    ref_categories = sorted(set(reference))
    test_categories = set(test)

    # Check for new categories in test
    new_categories = test_categories - set(ref_categories)

    if new_categories:
        if missing_handling == "error":
            raise ValueError(
                f"New categories found in test set: {new_categories}. "
                "These categories were not present in reference distribution."
            )
        elif missing_handling == "separate":
            # Add new categories to the end
            ref_categories.extend(sorted(new_categories))
        # else "ignore": new categories will be dropped

    # Count occurrences per category
    ref_counts = np.array([np.sum(reference == cat) for cat in ref_categories], dtype=np.intp)
    test_counts = np.array([np.sum(test == cat) for cat in ref_categories], dtype=np.intp)

    return ref_categories, ref_counts, test_counts


@dataclass
class WassersteinResult:
    """Result of Wasserstein distance calculation.

    The Wasserstein distance (also called Earth Mover's Distance) measures the
    minimum "cost" to transform one distribution into another. It's a true metric
    and doesn't require binning, making it ideal for continuous features.

    Attributes:
        distance: Wasserstein distance value (W_p)
        p: Order of Wasserstein distance (1 or 2)
        threshold: Calibrated threshold from permutation test (if calibrated)
        p_value: Statistical significance p-value (if calibrated)
        drifted: Whether drift was detected (distance > threshold)
        n_reference: Number of samples in reference distribution
        n_test: Number of samples in test distribution
        reference_stats: Summary statistics of reference distribution
        test_stats: Summary statistics of test distribution
        threshold_calibration_config: Configuration used for threshold calibration
        interpretation: Human-readable interpretation
        computation_time: Time taken to compute (seconds)
    """

    distance: float
    p: int
    threshold: float | None
    p_value: float | None
    drifted: bool
    n_reference: int
    n_test: int
    reference_stats: dict[str, float]
    test_stats: dict[str, float]
    threshold_calibration_config: dict[str, Any] | None
    interpretation: str
    computation_time: float

    def summary(self) -> str:
        """Return formatted summary of Wasserstein distance results."""
        lines = [
            "Wasserstein Distance Drift Detection Report",
            "=" * 60,
            f"Wasserstein-{self.p} Distance: {self.distance:.6f}",
            f"Drift Detected: {'YES' if self.drifted else 'NO'}",
            "",
            "Sample Sizes:",
            f"  Reference: {self.n_reference:,}",
            f"  Test: {self.n_test:,}",
            "",
        ]

        if self.threshold is not None:
            lines.extend(
                [
                    "Threshold Calibration:",
                    f"  Threshold: {self.threshold:.6f}",
                    f"  P-value: {self.p_value:.4f}" if self.p_value else "  P-value: N/A",
                    f"  Config: {self.threshold_calibration_config}",
                    "",
                ]
            )

        lines.extend(
            [
                "Distribution Statistics:",
                "-" * 60,
                f"Reference: Mean={self.reference_stats['mean']:.4f}, "
                f"Std={self.reference_stats['std']:.4f}, "
                f"Min={self.reference_stats['min']:.4f}, "
                f"Max={self.reference_stats['max']:.4f}",
                f"Test:      Mean={self.test_stats['mean']:.4f}, "
                f"Std={self.test_stats['std']:.4f}, "
                f"Min={self.test_stats['min']:.4f}, "
                f"Max={self.test_stats['max']:.4f}",
                "",
                f"Interpretation: {self.interpretation}",
                "",
                f"Computation Time: {self.computation_time:.3f}s",
            ]
        )

        return "\n".join(lines)


def compute_wasserstein_distance(
    reference: npt.NDArray[np.float64] | pl.Series,
    test: npt.NDArray[np.float64] | pl.Series,
    p: int = 1,
    threshold_calibration: bool = True,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    n_samples: int | None = None,
    random_state: int | None = None,
) -> WassersteinResult:
    """Compute Wasserstein distance between reference and test distributions.

    The Wasserstein distance (Earth Mover's Distance) measures the minimum cost
    to transform one probability distribution into another. Unlike PSI, it doesn't
    require binning and provides a true metric with desirable properties:
    - Metric properties: non-negative, symmetric, triangle inequality
    - More sensitive to small shifts than PSI
    - Natural interpretation as "transport cost"
    - No binning artifacts

    The p-Wasserstein distance is defined as:
        W_p(P, Q) = (∫|F_P^{-1}(u) - F_Q^{-1}(u)|^p du)^{1/p}

    For empirical distributions with sorted samples x_1 ≤ ... ≤ x_n:
        W_1(P, Q) = (1/n) Σ|x_i^P - x_i^Q|

    Threshold calibration uses a permutation test:
        H0: reference and test come from the same distribution
        H1: distributions differ

    Args:
        reference: Reference distribution (e.g., training data)
        test: Test distribution (e.g., production data)
        p: Order of Wasserstein distance (1 or 2). Default: 1
            - p=1: More robust, easier to interpret
            - p=2: More sensitive to tail differences
        threshold_calibration: Whether to calibrate threshold via permutation test
        n_permutations: Number of permutations for threshold calibration
        alpha: Significance level for threshold (default: 0.05)
        n_samples: Subsample to this many samples if provided (for large datasets)
        random_state: Random seed for reproducibility

    Returns:
        WassersteinResult with distance, threshold, p-value, and interpretation

    Raises:
        ValueError: If inputs are invalid or p not in {1, 2}

    Example:
        >>> # Detect mean shift
        >>> ref = np.random.normal(0, 1, 1000)
        >>> test = np.random.normal(0.5, 1, 1000)  # Mean shifted by 0.5
        >>> result = compute_wasserstein_distance(ref, test)
        >>> print(result.summary())
        >>>
        >>> # Detect variance shift
        >>> test_var = np.random.normal(0, 2, 1000)  # Variance doubled
        >>> result = compute_wasserstein_distance(ref, test_var)
        >>> print(f"Distance: {result.distance:.4f}, Drifted: {result.drifted}")
        >>>
        >>> # Without threshold calibration (faster)
        >>> result = compute_wasserstein_distance(
        ...     ref, test, threshold_calibration=False
        ... )

    References:
        - Villani, C. (2009). Optimal Transport: Old and New. Springer.
        - Ramdas, A., et al. (2017). On Wasserstein Two-Sample Testing.
          Entropy, 19(2), 47.
    """
    start_time = time.time()

    # Convert to numpy arrays
    if isinstance(reference, pl.Series):
        reference = reference.to_numpy()
    if isinstance(test, pl.Series):
        test = test.to_numpy()

    reference = np.asarray(reference, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)

    # Validate inputs
    if len(reference) == 0 or len(test) == 0:
        raise ValueError("Reference and test arrays must not be empty")

    if p not in [1, 2]:
        raise ValueError(f"p must be 1 or 2, got {p}")

    # Create local RNG to avoid global state pollution
    rng = np.random.default_rng(random_state)

    # Subsample if requested
    if n_samples is not None and len(reference) > n_samples:
        indices_ref = rng.choice(len(reference), n_samples, replace=False)
        reference = reference[indices_ref]
    if n_samples is not None and len(test) > n_samples:
        indices_test = rng.choice(len(test), n_samples, replace=False)
        test = test[indices_test]

    n_reference = len(reference)
    n_test = len(test)

    # Compute distribution statistics
    reference_stats = {
        "mean": float(np.mean(reference)),
        "std": float(np.std(reference)),
        "min": float(np.min(reference)),
        "max": float(np.max(reference)),
        "median": float(np.median(reference)),
        "q25": float(np.percentile(reference, 25)),
        "q75": float(np.percentile(reference, 75)),
    }

    test_stats = {
        "mean": float(np.mean(test)),
        "std": float(np.std(test)),
        "min": float(np.min(test)),
        "max": float(np.max(test)),
        "median": float(np.median(test)),
        "q25": float(np.percentile(test, 25)),
        "q75": float(np.percentile(test, 75)),
    }

    # Compute Wasserstein distance
    if p == 1:
        distance = float(wasserstein_distance(reference, test))
    else:  # p == 2
        # scipy's wasserstein_distance computes W_1
        # For W_2, we need to compute it manually
        distance = _wasserstein_2(reference, test)

    # Threshold calibration via permutation test
    threshold = None
    p_value = None
    calibration_config = None

    if threshold_calibration:
        threshold, p_value = _calibrate_wasserstein_threshold(
            reference, test, distance, n_permutations, alpha, p, rng
        )
        calibration_config = {
            "n_permutations": n_permutations,
            "alpha": alpha,
            "method": "permutation",
        }

    # Determine drift status
    if threshold is not None:
        drifted = distance > threshold
    else:
        # Without calibration, use heuristic based on distribution statistics
        # Drift if distance > 0.5 * std of reference
        drifted = distance > 0.5 * reference_stats["std"]
        threshold = 0.5 * reference_stats["std"]

    # Generate interpretation
    if drifted:
        if p_value is not None:
            interpretation = (
                f"Distribution drift detected (W_{p}={distance:.6f} > {threshold:.6f}, "
                f"p={p_value:.4f}). The test distribution differs significantly from "
                f"the reference distribution."
            )
        else:
            interpretation = (
                f"Distribution drift detected (W_{p}={distance:.6f} > {threshold:.6f}). "
                f"The test distribution differs from the reference distribution."
            )
    else:
        if p_value is not None:
            interpretation = (
                f"No significant drift detected (W_{p}={distance:.6f} ≤ {threshold:.6f}, "
                f"p={p_value:.4f}). Distributions are consistent."
            )
        else:
            interpretation = (
                f"No significant drift detected (W_{p}={distance:.6f} ≤ {threshold:.6f}). "
                f"Distributions are consistent."
            )

    computation_time = time.time() - start_time

    return WassersteinResult(
        distance=distance,
        p=p,
        threshold=threshold,
        p_value=p_value,
        drifted=drifted,
        n_reference=n_reference,
        n_test=n_test,
        reference_stats=reference_stats,
        test_stats=test_stats,
        threshold_calibration_config=calibration_config,
        interpretation=interpretation,
        computation_time=computation_time,
    )


def _wasserstein_2(u_values: npt.NDArray[np.float64], v_values: npt.NDArray[np.float64]) -> float:
    """Compute Wasserstein-2 distance between two 1D distributions.

    W_2(P, Q) = sqrt(∫|F_P^{-1}(u) - F_Q^{-1}(u)|^2 du)

    For empirical distributions, this is computed as:
    W_2 = sqrt((1/n) Σ(x_i - y_i)^2) where x, y are sorted samples

    Args:
        u_values: First distribution samples
        v_values: Second distribution samples

    Returns:
        Wasserstein-2 distance
    """
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)

    # Align to same length via CDF interpolation
    # Use linear interpolation between sorted samples
    n = min(len(u_sorted), len(v_sorted))
    u_quantiles = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_quantiles = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)

    # Compute L2 distance
    return float(np.sqrt(np.mean((u_quantiles - v_quantiles) ** 2)))


def _calibrate_wasserstein_threshold(
    reference: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    observed_distance: float,
    n_permutations: int,
    alpha: float,
    p: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Calibrate Wasserstein distance threshold via permutation test.

    Tests the null hypothesis that reference and test come from the same
    distribution by computing the null distribution of Wasserstein distances
    under random permutations.

    H0: P_ref = P_test (no drift)
    H1: P_ref ≠ P_test (drift detected)

    Args:
        reference: Reference distribution samples
        test: Test distribution samples
        observed_distance: Observed Wasserstein distance
        n_permutations: Number of permutations
        alpha: Significance level
        p: Order of Wasserstein distance

    Returns:
        Tuple of (threshold, p_value)
            - threshold: (1-alpha) quantile of null distribution
            - p_value: Fraction of null distances >= observed
    """
    # Pool all samples
    pooled = np.concatenate([reference, test])
    n_ref = len(reference)

    # Create local RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Compute null distribution
    null_distances = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Random permutation
        rng.shuffle(pooled)

        # Split into two groups
        ref_perm = pooled[:n_ref]
        test_perm = pooled[n_ref:]

        # Compute distance
        if p == 1:
            null_distances[i] = wasserstein_distance(ref_perm, test_perm)
        else:  # p == 2
            null_distances[i] = _wasserstein_2(ref_perm, test_perm)

    # Compute threshold as (1-alpha) quantile
    threshold = float(np.percentile(null_distances, (1 - alpha) * 100))

    # Compute p-value
    p_value = float(np.mean(null_distances >= observed_distance))

    return threshold, p_value


@dataclass
class DomainClassifierResult:
    """Result of domain classifier drift detection.

    Domain classifier trains a binary model to distinguish reference (label=0)
    from test (label=1) samples. AUC indicates drift magnitude, feature importances
    show which features drifted.

    Attributes:
        auc: AUC-ROC score (0.5 = no drift, 1.0 = complete distribution shift)
        drifted: Whether drift was detected (auc > threshold)
        feature_importances: DataFrame with feature, importance, rank columns
        threshold: AUC threshold used for drift detection
        n_reference: Number of samples in reference distribution
        n_test: Number of samples in test distribution
        n_features: Number of features used
        model_type: Type of classifier used (lightgbm, xgboost, sklearn)
        cv_auc_mean: Mean AUC from cross-validation
        cv_auc_std: Std of AUC from cross-validation
        interpretation: Human-readable interpretation
        computation_time: Time taken to compute (seconds)
        metadata: Additional metadata
    """

    auc: float
    drifted: bool
    feature_importances: pl.DataFrame
    threshold: float
    n_reference: int
    n_test: int
    n_features: int
    model_type: str
    cv_auc_mean: float
    cv_auc_std: float
    interpretation: str
    computation_time: float
    metadata: dict[str, Any]

    def summary(self) -> str:
        """Return formatted summary of domain classifier results."""
        lines = [
            "Domain Classifier Drift Detection Report",
            "=" * 60,
            f"AUC-ROC: {self.auc:.4f} (CV: {self.cv_auc_mean:.4f} ± {self.cv_auc_std:.4f})",
            f"Drift Detected: {'YES' if self.drifted else 'NO'}",
            f"Threshold: {self.threshold:.4f}",
            "",
            "Sample Sizes:",
            f"  Reference: {self.n_reference:,}",
            f"  Test: {self.n_test:,}",
            "",
            f"Model: {self.model_type}",
            f"Features: {self.n_features}",
            "",
            "Top 5 Most Drifted Features:",
            "-" * 60,
        ]

        # Show top 5 features
        top_features = self.feature_importances.head(5)
        for row in top_features.iter_rows(named=True):
            lines.append(
                f"  {row['rank']:2d}. {row['feature']:30s} (importance: {row['importance']:.4f})"
            )

        lines.extend(
            [
                "",
                f"Interpretation: {self.interpretation}",
                "",
                f"Computation Time: {self.computation_time:.3f}s",
            ]
        )

        return "\n".join(lines)


def compute_domain_classifier_drift(
    reference: npt.NDArray[np.float64] | pd.DataFrame | pl.DataFrame,
    test: npt.NDArray[np.float64] | pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
    *,
    model_type: str = "lightgbm",
    n_estimators: int = 100,
    max_depth: int = 5,
    threshold: float = 0.6,
    cv_folds: int = 5,
    random_state: int = 42,
) -> DomainClassifierResult:
    """Detect distribution drift using domain classifier.

    Trains a binary classifier to distinguish reference (label=0) from test (label=1)
    samples. AUC-ROC indicates drift magnitude, feature importance shows which features
    drifted most.

    The domain classifier approach detects multivariate drift by testing whether
    a classifier can distinguish between two distributions. If AUC ≈ 0.5, the
    distributions are indistinguishable (no drift). If AUC → 1.0, the distributions
    are completely separated (strong drift).

    **Advantages**:
        - Detects multivariate drift and feature interactions
        - Non-parametric (no distributional assumptions)
        - Interpretable via feature importance
        - Sensitive to subtle multivariate shifts

    **AUC Interpretation**:
        - AUC ≈ 0.5: No drift (random guess)
        - AUC = 0.6: Weak drift
        - AUC = 0.7-0.8: Moderate drift
        - AUC > 0.9: Strong drift

    Args:
        reference: Reference distribution (e.g., training data).
            Can be numpy array, pandas DataFrame, or polars DataFrame.
        test: Test distribution (e.g., production data).
            Can be numpy array, pandas DataFrame, or polars DataFrame.
        features: List of feature names to use. If None, uses all numeric columns.
            Only applicable for DataFrame inputs.
        model_type: Classifier type. Options:
            - "lightgbm": LightGBM (default, fastest)
            - "xgboost": XGBoost
            - "sklearn": sklearn RandomForestClassifier (always available)
        n_estimators: Number of trees/estimators (default: 100)
        max_depth: Maximum tree depth (default: 5)
        threshold: AUC threshold for flagging drift (default: 0.6)
        cv_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        DomainClassifierResult with AUC, feature importances, drift flag, etc.

    Raises:
        ValueError: If inputs are invalid or model_type is unknown
        ImportError: If required ML library is not installed

    Example:
        >>> import numpy as np
        >>> import polars as pl
        >>> from ml4t.engineer.outcome.drift import compute_domain_classifier_drift
        >>>
        >>> # No drift (identical distributions)
        >>> np.random.seed(42)
        >>> ref = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500),
        >>> })
        >>> test = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test)
        >>> print(f"AUC: {result.auc:.4f}, Drifted: {result.drifted}")
        AUC: 0.5123, Drifted: False
        >>>
        >>> # Strong drift (mean shift)
        >>> test_shifted = pl.DataFrame({
        ...     "x1": np.random.normal(2, 1, 500),
        ...     "x2": np.random.normal(2, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test_shifted)
        >>> print(f"AUC: {result.auc:.4f}, Drifted: {result.drifted}")
        AUC: 0.9876, Drifted: True
        >>> print(result.summary())
        >>>
        >>> # Interaction-based drift
        >>> test_corr = pl.DataFrame({
        ...     "x1": np.random.normal(0, 1, 500),
        ...     "x2": np.random.normal(0, 1, 500) + 0.8 * np.random.normal(0, 1, 500),
        >>> })
        >>> result = compute_domain_classifier_drift(ref, test_corr)
        >>> # Will detect correlation change via feature interactions

    References:
        - Lopez-Paz, D., & Oquab, M. (2017). Revisiting Classifier Two-Sample Tests.
          ICLR 2017.
        - Rabanser, S., et al. (2019). Failing Loudly: An Empirical Study of Methods
          for Detecting Dataset Shift. NeurIPS 2019.
    """
    start_time = time.time()

    # Prepare data
    X, y, feature_names = _prepare_domain_classification_data(reference, test, features)

    # Train classifier with cross-validation
    model, cv_scores = _train_domain_classifier(
        X,
        y,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Extract feature importances
    importances_df = _extract_feature_importances(model, feature_names)

    # Compute CV AUC statistics for drift decision
    # Using cross-validated AUC avoids optimistic bias from training AUC
    cv_auc_mean = float(np.mean(cv_scores))
    cv_auc_std = float(np.std(cv_scores))

    # Determine drift status using cross-validated AUC (unbiased estimate)
    drifted = cv_auc_mean > threshold

    # Generate interpretation
    if drifted:
        if cv_auc_mean > 0.9:
            severity = "strong"
        elif cv_auc_mean > 0.7:
            severity = "moderate"
        else:
            severity = "weak"

        interpretation = (
            f"{severity.capitalize()} distribution drift detected "
            f"(CV AUC={cv_auc_mean:.4f} ± {cv_auc_std:.4f} > {threshold:.4f}). "
            f"The classifier can distinguish reference from test distributions. "
            f"Top drifted feature: {importances_df['feature'][0]}."
        )
    else:
        interpretation = (
            f"No significant drift detected (CV AUC={cv_auc_mean:.4f} ± {cv_auc_std:.4f} ≤ {threshold:.4f}). "
            f"Distributions are indistinguishable by the classifier."
        )

    computation_time = time.time() - start_time

    return DomainClassifierResult(
        auc=cv_auc_mean,  # Use CV AUC as main metric (unbiased estimate)
        drifted=drifted,
        feature_importances=importances_df,
        threshold=threshold,
        n_reference=int(np.sum(y == 0)),
        n_test=int(np.sum(y == 1)),
        n_features=len(feature_names),
        model_type=model_type,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        interpretation=interpretation,
        computation_time=computation_time,
        metadata={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "cv_folds": cv_folds,
            "random_state": random_state,
        },
    )


def _prepare_domain_classification_data(
    reference: npt.NDArray[np.float64] | pd.DataFrame | pl.DataFrame,
    test: npt.NDArray[np.float64] | pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
    """Prepare labeled dataset for domain classification.

    Args:
        reference: Reference distribution
        test: Test distribution
        features: Feature names to use (for DataFrames)

    Returns:
        Tuple of (X, y, feature_names):
            - X: Feature matrix (reference + test concatenated)
            - y: Labels (0 for reference, 1 for test)
            - feature_names: List of feature names

    Raises:
        ValueError: If inputs are invalid or incompatible
    """
    # Convert to numpy arrays
    if isinstance(reference, pl.DataFrame):
        if features is None:
            # Use all numeric columns
            features = [
                c
                for c in reference.columns
                if reference[c].dtype
                in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
            ]
        X_ref = reference[features].to_numpy()
        feature_names = features

    elif isinstance(reference, pd.DataFrame):
        if features is None:
            # Use all numeric columns
            features = list(reference.select_dtypes(include=[np.number]).columns)
        X_ref = reference[features].to_numpy()
        feature_names = features

    elif isinstance(reference, np.ndarray):
        X_ref = reference
        if features is None:
            # Generate default feature names
            if X_ref.ndim == 1:
                X_ref = X_ref.reshape(-1, 1)
            feature_names = [f"feature_{i}" for i in range(X_ref.shape[1])]
        else:
            feature_names = features

    else:
        raise ValueError(
            f"Unsupported reference type: {type(reference)}. "
            "Must be numpy array, pandas DataFrame, or polars DataFrame."
        )

    # Process test data
    if isinstance(test, (pl.DataFrame, pd.DataFrame)):
        X_test = test[feature_names].to_numpy()
    elif isinstance(test, np.ndarray):
        X_test = test
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
    else:
        raise ValueError(
            f"Unsupported test type: {type(test)}. "
            "Must be numpy array, pandas DataFrame, or polars DataFrame."
        )

    # Validate shapes
    if X_ref.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch: reference has {X_ref.shape[1]} features, "
            f"test has {X_test.shape[1]} features."
        )

    # Concatenate and create labels
    X = np.vstack([X_ref, X_test])
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_test))])

    return X, y, feature_names


def _train_domain_classifier(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    model_type: str = "lightgbm",
    n_estimators: int = 100,
    max_depth: int = 5,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[Any, npt.NDArray[np.float64]]:
    """Train binary classifier for domain classification.

    Args:
        X: Feature matrix
        y: Labels (0=reference, 1=test)
        model_type: Classifier type
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        cv_folds: Cross-validation folds
        random_state: Random seed

    Returns:
        Tuple of (trained_model, cv_auc_scores)

    Raises:
        ValueError: If model_type is unknown
        ImportError: If required library is not installed
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Select and configure model
    if model_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM required for domain classifier drift detection. "
                "Install with: pip install ml4t-eval[ml] or pip install lightgbm"
            )

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbose=-1,
            force_col_wise=True,  # Suppress warning
        )

    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost required for domain classifier drift detection. "
                "Install with: pip install xgboost"
            )

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
        )

    elif model_type == "sklearn":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Must be 'lightgbm', 'xgboost', or 'sklearn'."
        )

    # Cross-validation for AUC using stratified folds
    # StratifiedKFold ensures balanced class distribution in each fold
    # (critical since data is [zeros, ones] concatenated)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    # Train on full data
    model.fit(X, y)

    return model, cv_scores


def _extract_feature_importances(model: Any, feature_names: list[str]) -> pl.DataFrame:
    """Extract and rank feature importances.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        Polars DataFrame with columns: feature, importance, rank

    Raises:
        ValueError: If model doesn't have feature importances
    """
    # Get importances (works for LightGBM, XGBoost, sklearn)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError(f"Model type {type(model)} does not have feature_importances_ attribute")

    # Create DataFrame
    df = pl.DataFrame({"feature": feature_names, "importance": importances})

    # Sort by importance (descending)
    df = df.sort("importance", descending=True)

    # Add rank
    df = df.with_columns(pl.arange(1, len(df) + 1).alias("rank"))

    return df


@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature across multiple methods.

    Attributes:
        feature: Feature name
        psi_result: PSI drift detection result (if method was run)
        wasserstein_result: Wasserstein drift detection result (if method was run)
        drifted: Consensus drift flag (based on multiple methods)
        n_methods_run: Number of methods that were run on this feature
        n_methods_detected: Number of methods that detected drift
        drift_probability: Fraction of methods that detected drift
        interpretation: Human-readable interpretation
    """

    feature: str
    psi_result: PSIResult | None = None
    wasserstein_result: WassersteinResult | None = None
    drifted: bool = False
    n_methods_run: int = 0
    n_methods_detected: int = 0
    drift_probability: float = 0.0
    interpretation: str = ""

    def summary(self) -> str:
        """Generate summary string for this feature's drift analysis."""
        lines = [f"Feature: {self.feature}"]
        lines.append(
            f"  Drifted: {self.drifted} ({self.n_methods_detected}/{self.n_methods_run} methods)"
        )
        lines.append(f"  Drift Probability: {self.drift_probability:.2%}")

        if self.psi_result is not None:
            lines.append(f"  PSI: {self.psi_result.psi:.4f} ({self.psi_result.alert_level})")

        if self.wasserstein_result is not None:
            drifted_str = "drifted" if self.wasserstein_result.drifted else "no drift"
            lines.append(f"  Wasserstein: {self.wasserstein_result.distance:.4f} ({drifted_str})")

        return "\n".join(lines)


@dataclass
class DriftSummaryResult:
    """Summary of multi-method drift analysis across features.

    This result aggregates drift detection across multiple methods (PSI,
    Wasserstein, Domain Classifier) to provide a comprehensive drift assessment.

    Attributes:
        feature_results: Per-feature drift results (PSI + Wasserstein)
        domain_classifier_result: Multivariate drift result (if domain classifier was run)
        n_features: Total number of features analyzed
        n_features_drifted: Number of features flagged as drifted
        drifted_features: List of feature names that drifted
        overall_drifted: Overall drift flag (True if any feature drifted or domain classifier detected drift)
        consensus_threshold: Minimum fraction of methods that must agree to flag drift
        methods_used: List of drift detection methods used
        univariate_methods: Methods run on individual features
        multivariate_methods: Methods run on all features jointly
        interpretation: Human-readable interpretation
        computation_time: Total time taken for all methods (seconds)
    """

    feature_results: list[FeatureDriftResult]
    domain_classifier_result: DomainClassifierResult | None = None
    n_features: int = 0
    n_features_drifted: int = 0
    drifted_features: list[str] | None = None
    overall_drifted: bool = False
    consensus_threshold: float = 0.5
    methods_used: list[str] | None = None
    univariate_methods: list[str] | None = None
    multivariate_methods: list[str] | None = None
    interpretation: str = ""
    computation_time: float = 0.0

    def __post_init__(self):
        """Initialize computed fields."""
        if self.drifted_features is None:
            self.drifted_features = []
        if self.methods_used is None:
            self.methods_used = []
        if self.univariate_methods is None:
            self.univariate_methods = []
        if self.multivariate_methods is None:
            self.multivariate_methods = []

    def get_drifted_features(
        self, severity: Literal["all", "high", "medium", "low"] = "all"
    ) -> list[str]:
        """Get list of drifted features filtered by severity.

        Severity is determined by drift probability:
        - High: drift_probability >= 0.8 (strong consensus)
        - Medium: 0.5 <= drift_probability < 0.8 (moderate consensus)
        - Low: drift_probability < 0.5 but drifted=True (weak signal)

        Args:
            severity: Severity filter. Options:
                - "all": All drifted features (default)
                - "high": Only features with strong drift consensus
                - "medium": Only features with moderate drift consensus
                - "low": Only features with weak drift signal

        Returns:
            List of feature names matching severity criteria

        Example:
            >>> result = analyze_drift(reference, test)
            >>> # Get all drifted features
            >>> all_drifted = result.get_drifted_features()
            >>> # Get only high-severity drift
            >>> high_severity = result.get_drifted_features(severity="high")
            >>> print(f"High severity drift: {high_severity}")
        """
        filtered = []

        for feature_result in self.feature_results:
            if not feature_result.drifted:
                continue

            prob = feature_result.drift_probability

            if (
                severity == "all"
                or severity == "high"
                and prob >= 0.8
                or severity == "medium"
                and 0.5 <= prob < 0.8
                or severity == "low"
                and prob < 0.5
            ):
                filtered.append(feature_result.feature)

        return filtered

    def summary(self) -> str:
        """Generate comprehensive summary of drift analysis with visual indicators."""
        lines = ["=" * 60]
        lines.append("Drift Analysis Summary")
        lines.append("=" * 60)
        lines.append(f"Methods Used: {', '.join(self.methods_used or [])}")
        lines.append(f"Consensus Threshold: {self.consensus_threshold:.0%}")
        lines.append(f"Total Features: {self.n_features}")
        lines.append(
            f"Drifted Features: {self.n_features_drifted} ({self.n_features_drifted / max(1, self.n_features):.0%})"
        )

        # Add visual indicator for overall drift status
        drift_icon = "WARNING" if self.overall_drifted else "OK"
        lines.append(f"Overall Drift Detected: {drift_icon} {self.overall_drifted}")
        lines.append("")

        if self.drifted_features:
            # Categorize by severity
            high_severity = self.get_drifted_features(severity="high")
            medium_severity = self.get_drifted_features(severity="medium")
            low_severity = self.get_drifted_features(severity="low")

            if high_severity:
                lines.append("HIGH SEVERITY Drift (strong consensus):")
                for feature in high_severity:
                    # Find the feature result to get drift probability
                    for fr in self.feature_results:
                        if fr.feature == feature:
                            lines.append(
                                f"  WARNING {feature} (drift prob: {fr.drift_probability:.0%})"
                            )
                            break
                lines.append("")

            if medium_severity:
                lines.append("MEDIUM SEVERITY Drift (moderate consensus):")
                for feature in medium_severity:
                    for fr in self.feature_results:
                        if fr.feature == feature:
                            lines.append(
                                f"  CAUTION {feature} (drift prob: {fr.drift_probability:.0%})"
                            )
                            break
                lines.append("")

            if low_severity:
                lines.append("LOW SEVERITY Drift (weak signal):")
                for feature in low_severity:
                    for fr in self.feature_results:
                        if fr.feature == feature:
                            lines.append(
                                f"  NOTICE {feature} (drift prob: {fr.drift_probability:.0%})"
                            )
                            break
                lines.append("")

        if self.domain_classifier_result is not None:
            lines.append("Multivariate Drift (Domain Classifier):")
            dc_icon = "WARNING" if self.domain_classifier_result.drifted else "OK"
            lines.append(f"  {dc_icon} AUC: {self.domain_classifier_result.auc:.4f}")
            lines.append(f"  Drifted: {self.domain_classifier_result.drifted}")
            lines.append("")

        lines.append(f"Computation Time: {self.computation_time:.2f}s")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert feature-level results to a DataFrame.

        Returns:
            Polars DataFrame with per-feature drift analysis results
        """
        data = []
        for result in self.feature_results:
            row = {
                "feature": result.feature,
                "drifted": result.drifted,
                "drift_probability": result.drift_probability,
                "n_methods_detected": result.n_methods_detected,
                "n_methods_run": result.n_methods_run,
            }

            if result.psi_result is not None:
                row["psi"] = result.psi_result.psi
                row["psi_alert"] = result.psi_result.alert_level

            if result.wasserstein_result is not None:
                row["wasserstein_distance"] = result.wasserstein_result.distance
                row["wasserstein_drifted"] = result.wasserstein_result.drifted
                if result.wasserstein_result.p_value is not None:
                    row["wasserstein_pvalue"] = result.wasserstein_result.p_value

            data.append(row)

        return pl.DataFrame(data)


# Threshold presets for drift detection
DRIFT_THRESHOLDS = {
    "strict": {
        "psi_threshold_yellow": 0.05,
        "psi_threshold_red": 0.1,
        "consensus_threshold": 0.66,  # Require 2/3 methods to agree
        "domain_classifier_threshold": 0.55,
    },
    "moderate": {
        "psi_threshold_yellow": 0.1,
        "psi_threshold_red": 0.2,
        "consensus_threshold": 0.5,  # Require 1/2 methods to agree
        "domain_classifier_threshold": 0.6,
    },
    "lenient": {
        "psi_threshold_yellow": 0.15,
        "psi_threshold_red": 0.25,
        "consensus_threshold": 0.34,  # Require 1/3 methods to agree
        "domain_classifier_threshold": 0.7,
    },
}


def analyze_drift(
    reference: pd.DataFrame | pl.DataFrame,
    test: pd.DataFrame | pl.DataFrame,
    features: list[str] | None = None,
    *,
    methods: list[str] | None = None,
    consensus_threshold: float = 0.5,
    threshold_preset: Literal["strict", "moderate", "lenient"] | None = None,
    # PSI parameters
    psi_config: dict[str, Any] | None = None,
    # Wasserstein parameters
    wasserstein_config: dict[str, Any] | None = None,
    # Domain classifier parameters
    domain_classifier_config: dict[str, Any] | None = None,
) -> DriftSummaryResult:
    """Comprehensive drift analysis using multiple detection methods.

    This function provides a unified interface for drift detection across multiple
    methods (PSI, Wasserstein, Domain Classifier). It runs univariate methods on
    each feature and optionally multivariate methods on all features jointly.

    **Univariate Methods** (run per feature):
        - PSI: Population Stability Index (binning-based)
        - Wasserstein: Earth Mover's Distance (metric-based)

    **Multivariate Methods** (run on all features):
        - Domain Classifier: ML-based drift detection with feature importance

    **Consensus Logic**:
        A feature is flagged as drifted if the fraction of methods detecting drift
        exceeds the consensus_threshold. For example, with threshold=0.5:
        - If 2/3 methods detect drift → flagged as drifted
        - If 1/3 methods detect drift → not flagged as drifted

    **Threshold Presets**:
        Use threshold_preset for convenient configuration:
        - "strict": Low tolerance (PSI red=0.1, consensus=66%)
        - "moderate": Balanced (PSI red=0.2, consensus=50%) - Default
        - "lenient": High tolerance (PSI red=0.25, consensus=34%)

    Args:
        reference: Reference distribution (e.g., training data)
            Can be pandas or polars DataFrame
        test: Test distribution (e.g., production data)
            Can be pandas or polars DataFrame
        features: List of feature names to analyze. If None, uses all numeric columns
        methods: List of methods to use. Options: ["psi", "wasserstein", "domain_classifier"]
            Default: ["psi", "wasserstein", "domain_classifier"]
        consensus_threshold: Minimum fraction of methods that must detect drift
            to flag a feature as drifted (default: 0.5). Overridden by threshold_preset.
        threshold_preset: Use predefined threshold configuration. Options:
            - "strict": Sensitive to small drifts
            - "moderate": Balanced sensitivity (default)
            - "lenient": Only flags significant drifts
            If None, uses individual config parameters.
        psi_config: Configuration dict for PSI. Keys:
            - n_bins: int (default: 10)
            - is_categorical: bool (default: False)
            - psi_threshold_yellow: float (default: 0.1)
            - psi_threshold_red: float (default: 0.2)
        wasserstein_config: Configuration dict for Wasserstein. Keys:
            - p: int (default: 1)
            - threshold_calibration: bool (default: True)
            - n_permutations: int (default: 1000)
            - alpha: float (default: 0.05)
        domain_classifier_config: Configuration dict for domain classifier. Keys:
            - model_type: str (default: "lightgbm")
            - n_estimators: int (default: 100)
            - max_depth: int (default: 5)
            - threshold: float (default: 0.6)
            - cv_folds: int (default: 5)

    Returns:
        DriftSummaryResult with per-feature results, multivariate results,
        and overall drift assessment

    Raises:
        ValueError: If inputs are invalid or methods list is empty

    Example:
        >>> import pandas as pd
        >>> from ml4t.engineer.outcome.drift import analyze_drift
        >>>
        >>> # Create reference and test data
        >>> reference = pd.DataFrame({
        ...     'feature1': np.random.normal(0, 1, 1000),
        ...     'feature2': np.random.normal(0, 1, 1000)
        ... })
        >>> test = pd.DataFrame({
        ...     'feature1': np.random.normal(0.5, 1, 1000),  # Mean shifted
        ...     'feature2': np.random.normal(0, 1, 1000)      # No shift
        ... })
        >>>
        >>> # Run drift analysis with default settings
        >>> result = analyze_drift(reference, test)
        >>> print(result.summary())
        >>>
        >>> # Use strict preset for production monitoring
        >>> result = analyze_drift(reference, test, threshold_preset="strict")
        >>> high_severity = result.get_drifted_features(severity="high")
        >>> print(f"High severity drift: {high_severity}")
        >>>
        >>> # Get per-feature details
        >>> df = result.to_dataframe()
        >>> print(df)
    """
    start_time = time.time()

    # Input validation
    if reference is None or test is None:
        raise ValueError("reference and test must not be None")

    # Apply threshold preset if specified
    if threshold_preset is not None:
        if threshold_preset not in DRIFT_THRESHOLDS:
            raise ValueError(
                f"Invalid threshold_preset: '{threshold_preset}'. "
                f"Valid options: {list(DRIFT_THRESHOLDS.keys())}"
            )

        preset = DRIFT_THRESHOLDS[threshold_preset]

        # Override consensus threshold
        consensus_threshold = preset["consensus_threshold"]

        # Set PSI thresholds if not explicitly configured
        if psi_config is None:
            psi_config = {}
        if "psi_threshold_yellow" not in psi_config:
            psi_config["psi_threshold_yellow"] = preset["psi_threshold_yellow"]
        if "psi_threshold_red" not in psi_config:
            psi_config["psi_threshold_red"] = preset["psi_threshold_red"]

        # Set domain classifier threshold if not explicitly configured
        if domain_classifier_config is None:
            domain_classifier_config = {}
        if "threshold" not in domain_classifier_config:
            domain_classifier_config["threshold"] = preset["domain_classifier_threshold"]

    # Convert to pandas for easier processing
    if isinstance(reference, pl.DataFrame):
        reference = reference.to_pandas()
    if isinstance(test, pl.DataFrame):
        test = test.to_pandas()

    # Determine features to analyze
    if features is None:
        # Use all numeric columns
        numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_cols
    else:
        # Validate features exist
        missing_in_ref = set(features) - set(reference.columns)
        missing_in_test = set(features) - set(test.columns)
        if missing_in_ref or missing_in_test:
            raise ValueError(
                f"Features not found - reference: {missing_in_ref}, test: {missing_in_test}"
            )

    if not features:
        raise ValueError("No features to analyze")

    # Determine methods to use
    if methods is None:
        methods = ["psi", "wasserstein", "domain_classifier"]

    valid_methods = ["psi", "wasserstein", "domain_classifier"]
    invalid_methods = set(methods) - set(valid_methods)
    if invalid_methods:
        raise ValueError(f"Invalid methods: {invalid_methods}. Valid: {valid_methods}")

    # Separate univariate and multivariate methods
    univariate_methods = [m for m in methods if m in ["psi", "wasserstein"]]
    multivariate_methods = [m for m in methods if m == "domain_classifier"]

    # Set default configs
    if psi_config is None:
        psi_config = {}
    if wasserstein_config is None:
        wasserstein_config = {}
    if domain_classifier_config is None:
        domain_classifier_config = {}

    # Run univariate methods on each feature
    feature_results = []
    for feature in features:
        ref_values = reference[feature].values
        test_values = test[feature].values

        psi_result = None
        wasserstein_result = None
        n_methods_run = 0
        n_methods_detected = 0

        # PSI
        if "psi" in methods:
            try:
                psi_result = compute_psi(ref_values, test_values, **psi_config)
                n_methods_run += 1
                if psi_result.alert_level in ["yellow", "red"]:
                    n_methods_detected += 1
            except Exception as e:
                # Log warning but continue
                logger.warning("PSI failed for feature %s: %s", feature, e)

        # Wasserstein
        if "wasserstein" in methods:
            try:
                wasserstein_result = compute_wasserstein_distance(
                    ref_values, test_values, **wasserstein_config
                )
                n_methods_run += 1
                if wasserstein_result.drifted:
                    n_methods_detected += 1
            except Exception as e:
                # Log warning but continue
                logger.warning("Wasserstein failed for feature %s: %s", feature, e)

        # Consensus drift flag
        drift_probability = n_methods_detected / max(1, n_methods_run)
        drifted = drift_probability >= consensus_threshold

        # Interpretation
        if drifted:
            interpretation = (
                f"{n_methods_detected}/{n_methods_run} methods detected drift "
                f"(probability: {drift_probability:.0%})"
            )
        else:
            interpretation = (
                f"No consensus drift ({n_methods_detected}/{n_methods_run} methods, "
                f"threshold: {consensus_threshold:.0%})"
            )

        feature_results.append(
            FeatureDriftResult(
                feature=feature,
                psi_result=psi_result,
                wasserstein_result=wasserstein_result,
                drifted=drifted,
                n_methods_run=n_methods_run,
                n_methods_detected=n_methods_detected,
                drift_probability=drift_probability,
                interpretation=interpretation,
            )
        )

    # Run multivariate domain classifier if requested
    domain_classifier_result = None
    if "domain_classifier" in methods:
        try:
            domain_classifier_result = compute_domain_classifier_drift(
                reference[features], test[features], **domain_classifier_config
            )
        except Exception as e:
            # Log warning but continue
            logger.warning("Domain classifier failed: %s", e)

    # Aggregate results
    n_features = len(features)
    n_features_drifted = sum(r.drifted for r in feature_results)
    drifted_features = [r.feature for r in feature_results if r.drifted]

    # Overall drift flag
    overall_drifted = n_features_drifted > 0
    if domain_classifier_result is not None and domain_classifier_result.drifted:
        overall_drifted = True

    # Interpretation
    if overall_drifted:
        interpretation = (
            f"Drift detected in {n_features_drifted}/{n_features} features "
            f"({n_features_drifted / max(1, n_features):.0%})"
        )
    else:
        interpretation = f"No drift detected across {n_features} features"

    computation_time = time.time() - start_time

    return DriftSummaryResult(
        feature_results=feature_results,
        domain_classifier_result=domain_classifier_result,
        n_features=n_features,
        n_features_drifted=n_features_drifted,
        drifted_features=drifted_features,
        overall_drifted=overall_drifted,
        consensus_threshold=consensus_threshold,
        methods_used=methods,
        univariate_methods=univariate_methods,
        multivariate_methods=multivariate_methods,
        interpretation=interpretation,
        computation_time=computation_time,
    )


# Re-export for convenience
__all__ = [
    "compute_psi",
    "PSIResult",
    "compute_wasserstein_distance",
    "WassersteinResult",
    "compute_domain_classifier_drift",
    "DomainClassifierResult",
    "analyze_drift",
    "FeatureDriftResult",
    "DriftSummaryResult",
    "DRIFT_THRESHOLDS",
]
