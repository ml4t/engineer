# mypy: disable-error-code="misc,call-arg,arg-type"
# ruff: noqa: UP006, UP045
"""Feature evaluation configuration (Modules A, B, C).

Exports:
    Module A (Feature Diagnostics):
        StationarityConfig - ADF, KPSS, PP test settings
        ACFConfig - Autocorrelation analysis settings
        VolatilityConfig - Volatility clustering detection
        DistributionConfig - Distribution analysis settings
        ModuleAConfig - Combined Module A configuration

    Module B (Cross-Feature Analysis):
        CorrelationConfig - Correlation matrix settings
        PCAConfig - Principal component analysis
        ClusteringConfig - Feature clustering
        RedundancyConfig - Redundancy detection
        ModuleBConfig - Combined Module B configuration

    Module C (Feature-Outcome):
        ICConfig - Information coefficient analysis
        BinaryClassificationConfig - Classification metrics
        ThresholdAnalysisConfig - Threshold optimization
        MLDiagnosticsConfig - SHAP and importance settings
        ModuleCConfig - Combined Module C configuration

    Main:
        FeatureEvaluatorConfig - Master configuration for all modules

This module defines configuration for:
- **Module A**: Feature diagnostics (stationarity, ACF, volatility, distribution)
- **Module B**: Cross-feature analysis (correlation, PCA, clustering, redundancy)
- **Module C**: Feature-outcome relationships (IC, classification, thresholds, ML diagnostics)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ml4t.engineer.config.base import BaseConfig, StatisticalTestConfig
from ml4t.engineer.config.validation import (
    ClusteringMethod,
    CorrelationMethod,
    DistanceMetric,
    DriftDetectionMethod,
    LinkageMethod,
    NonNegativeInt,
    NormalityTest,
    OutlierMethod,
    PositiveFloat,
    PositiveInt,
    Probability,
    RegressionType,
    ThresholdOptimizationTarget,
    VolatilityClusterMethod,
    validate_min_max_range,
)

# =============================================================================
# Module A: Feature Diagnostics
# =============================================================================


class StationarityConfig(StatisticalTestConfig):
    """Configuration for stationarity testing.

    Tests whether time series are stationary (constant mean/variance over time).
    Non-stationary series should typically be differenced or detrended before use
    in forecasting models.

    Attributes:
        enabled: Run stationarity tests
        adf_enabled: Run Augmented Dickey-Fuller test
        kpss_enabled: Run KPSS test
        pp_enabled: Run Phillips-Perron test
        adf_regression: Regression type for ADF ("c", "ct", "ctt", "n")
        kpss_regression: Regression type for KPSS ("c", "ct")
        pp_regression: Regression type for PP ("c", "ct", "ctt", "n")
        max_lag: Maximum lag for tests ("auto" or positive int)
        significance_level: Significance level for hypothesis tests

    Examples:
        >>> # Default: ADF + KPSS at 5% significance
        >>> config = StationarityConfig()

        >>> # Custom: Only ADF at 1% significance
        >>> config = StationarityConfig(
        ...     adf_enabled=True,
        ...     kpss_enabled=False,
        ...     pp_enabled=False,
        ...     significance_level=0.01
        ... )

    References:
        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the estimators for
          autoregressive time series with a unit root." JASA.
        - Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
          against the alternative of a unit root." Journal of Econometrics.
    """

    adf_enabled: bool = Field(True, description="Run Augmented Dickey-Fuller test")
    kpss_enabled: bool = Field(True, description="Run KPSS test")
    pp_enabled: bool = Field(False, description="Run Phillips-Perron test (similar to ADF)")
    adf_regression: RegressionType = Field(
        RegressionType.CONSTANT, description="ADF regression type: c, ct, ctt, or n"
    )
    kpss_regression: Literal["c", "ct"] = Field("c", description="KPSS regression type: c or ct")
    pp_regression: RegressionType = Field(
        RegressionType.CONSTANT, description="PP regression type: c, ct, ctt, or n"
    )
    max_lag: Literal["auto"] | PositiveInt = Field(
        "auto", description="Maximum lag for tests (auto or positive int)"
    )

    @model_validator(mode="after")
    def check_at_least_one_test(self) -> StationarityConfig:
        """Ensure at least one test is enabled."""
        if not (self.adf_enabled or self.kpss_enabled or self.pp_enabled):
            raise ValueError("At least one stationarity test must be enabled")
        return self


class ACFConfig(BaseConfig):
    """Configuration for autocorrelation function (ACF) and partial ACF (PACF) analysis.

    Analyzes temporal dependencies in time series to detect:
    - Serial correlation (autocorrelation)
    - Lag structure for AR/MA models
    - Periodicity and cycles

    Attributes:
        enabled: Run ACF/PACF analysis
        n_lags: Number of lags to compute (auto or positive int)
        alpha: Significance level for confidence bands
        compute_pacf: Also compute partial autocorrelation
        pacf_method: Method for PACF ("yw", "ols", "mle")
        use_fft: Use FFT for ACF computation (faster for long series)

    Examples:
        >>> # Default: 40 lags with 95% confidence
        >>> config = ACFConfig()

        >>> # Custom: 100 lags with 99% confidence
        >>> config = ACFConfig(n_lags=100, alpha=0.01)
    """

    enabled: bool = Field(True, description="Run ACF/PACF analysis")
    n_lags: Literal["auto"] | PositiveInt = Field(
        40, description="Number of lags (auto = min(10*log10(n), n//2))"
    )
    alpha: Probability = Field(0.05, description="Significance level for confidence bands")
    compute_pacf: bool = Field(True, description="Also compute partial autocorrelation")
    pacf_method: Literal["yw", "ols", "mle"] = Field(
        "yw", description="PACF method: yw (Yule-Walker), ols, or mle"
    )
    use_fft: bool = Field(True, description="Use FFT for ACF computation (faster)")


class VolatilityConfig(BaseConfig):
    """Configuration for volatility analysis.

    Analyzes volatility patterns to detect:
    - Volatility clustering (GARCH effects)
    - Heteroscedasticity
    - Regime changes

    Attributes:
        enabled: Run volatility analysis
        window_sizes: Rolling window sizes for volatility estimation
        detect_clustering: Test for volatility clustering (GARCH effects)
        cluster_method: Method for cluster detection ("ljung_box", "engle_arch")
        significance_level: Significance level for clustering tests
        compute_rolling_vol: Compute rolling volatility estimates

    Examples:
        >>> # Default: 21-day rolling with cluster detection
        >>> config = VolatilityConfig()

        >>> # Custom: Multiple windows without clustering
        >>> config = VolatilityConfig(
        ...     window_sizes=[10, 21, 63],
        ...     detect_clustering=False
        ... )
    """

    enabled: bool = Field(True, description="Run volatility analysis")
    window_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [21], description="Rolling window sizes for volatility"
    )
    detect_clustering: bool = Field(
        True, description="Test for volatility clustering (GARCH effects)"
    )
    cluster_method: VolatilityClusterMethod = Field(
        VolatilityClusterMethod.LJUNG_BOX,
        description="Clustering detection method: ljung_box or engle_arch",
    )
    significance_level: Probability = Field(
        0.05, description="Significance level for clustering tests"
    )
    compute_rolling_vol: bool = Field(True, description="Compute rolling volatility estimates")

    @field_validator("window_sizes")
    @classmethod
    def check_window_sizes(cls, v: list[int]) -> list[int]:
        """Ensure window sizes are positive and reasonable."""
        if not v:
            raise ValueError("Must specify at least one window size")
        if any(w < 2 for w in v):
            raise ValueError("Window sizes must be >= 2")
        return sorted(v)  # Sort for consistent ordering


class DistributionConfig(BaseConfig):
    """Configuration for distribution analysis.

    Analyzes distributional properties:
    - Normality (critical for many statistical tests)
    - Moments (mean, std, skew, kurtosis)
    - Outliers

    Attributes:
        enabled: Run distribution analysis
        test_normality: Test for normality
        normality_tests: Which normality tests to run
        compute_moments: Compute higher moments (skew, kurtosis)
        detect_outliers: Detect outliers
        outlier_method: Outlier detection method
        outlier_threshold: Z-score threshold for outlier detection

    Examples:
        >>> # Default: Normality + moments
        >>> config = DistributionConfig()

        >>> # Custom: Full analysis with outlier detection
        >>> config = DistributionConfig(
        ...     normality_tests=[NormalityTest.SHAPIRO, NormalityTest.JARQUE_BERA],
        ...     detect_outliers=True,
        ...     outlier_method=OutlierMethod.ISOLATION_FOREST
        ... )
    """

    enabled: bool = Field(True, description="Run distribution analysis")
    test_normality: bool = Field(True, description="Test for normality")
    normality_tests: list[NormalityTest] = Field(
        default_factory=lambda: [NormalityTest.JARQUE_BERA],
        description="Normality tests to run",
    )
    compute_moments: bool = Field(True, description="Compute higher moments (skew, kurtosis)")
    detect_outliers: bool = Field(False, description="Detect outliers (can be expensive)")
    outlier_method: OutlierMethod = Field(
        OutlierMethod.ZSCORE, description="Outlier detection method"
    )
    outlier_threshold: PositiveFloat = Field(
        3.0, description="Z-score threshold for outlier detection"
    )


class ModuleAConfig(BaseConfig):
    """Configuration for Module A: Feature Diagnostics.

    Analyzes individual feature properties:
    - Stationarity (unit roots, trend)
    - Temporal structure (autocorrelation)
    - Volatility (clustering, heteroscedasticity)
    - Distribution (normality, outliers)

    Examples:
        >>> # Default: All diagnostics enabled
        >>> config = ModuleAConfig()

        >>> # Custom: Only stationarity and ACF
        >>> config = ModuleAConfig(
        ...     stationarity=StationarityConfig(),
        ...     acf=ACFConfig(),
        ...     volatility=VolatilityConfig(enabled=False),
        ...     distribution=DistributionConfig(enabled=False)
        ... )
    """

    stationarity: StationarityConfig = Field(
        default_factory=StationarityConfig, description="Stationarity testing configuration"
    )
    acf: ACFConfig = Field(default_factory=ACFConfig, description="ACF/PACF configuration")
    volatility: VolatilityConfig = Field(
        default_factory=VolatilityConfig, description="Volatility analysis configuration"
    )
    distribution: DistributionConfig = Field(
        default_factory=DistributionConfig, description="Distribution analysis configuration"
    )


# =============================================================================
# Module B: Cross-Feature Analysis
# =============================================================================


class CorrelationConfig(BaseConfig):
    """Configuration for correlation analysis.

    Analyzes relationships between features to detect:
    - Linear relationships (Pearson)
    - Monotonic relationships (Spearman)
    - General dependence (Kendall)
    - Lagged relationships

    Attributes:
        enabled: Run correlation analysis
        methods: Correlation methods to use
        compute_pairwise: Compute all pairwise correlations (vs just with outcome)
        min_periods: Minimum observations for correlation
        lag_correlations: Compute lagged cross-correlations
        max_lag: Maximum lag for cross-correlations

    Examples:
        >>> # Default: Pearson only
        >>> config = CorrelationConfig()

        >>> # Custom: Multiple methods with lags
        >>> config = CorrelationConfig(
        ...     methods=[CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN],
        ...     lag_correlations=True,
        ...     max_lag=10
        ... )
    """

    enabled: bool = Field(True, description="Run correlation analysis")
    methods: list[CorrelationMethod] = Field(
        default_factory=lambda: [CorrelationMethod.PEARSON],
        description="Correlation methods: pearson, spearman, kendall",
    )
    compute_pairwise: bool = Field(
        True, description="Compute all pairwise correlations (vs just with outcome)"
    )
    min_periods: PositiveInt = Field(30, description="Minimum observations for correlation")
    lag_correlations: bool = Field(False, description="Compute lagged cross-correlations")
    max_lag: PositiveInt = Field(10, description="Maximum lag for cross-correlations")

    @field_validator("methods")
    @classmethod
    def check_methods(cls, v: list[CorrelationMethod]) -> list[CorrelationMethod]:
        """Ensure at least one method specified."""
        if not v:
            raise ValueError("Must specify at least one correlation method")
        return v


class PCAConfig(BaseConfig):
    """Configuration for Principal Component Analysis (PCA).

    Dimensionality reduction and feature redundancy analysis:
    - Identify principal components
    - Measure explained variance
    - Detect redundancy

    Attributes:
        enabled: Run PCA
        n_components: Number of components (int, float for variance %, or "auto")
        variance_threshold: Cumulative variance to explain (for n_components="auto")
        standardize: Standardize features before PCA (recommended)
        rotation: Optional rotation for interpretability ("varimax", "quartimax")

    Examples:
        >>> # Default: Disabled (opt-in)
        >>> config = PCAConfig()

        >>> # Custom: Explain 95% of variance
        >>> config = PCAConfig(
        ...     enabled=True,
        ...     n_components="auto",
        ...     variance_threshold=0.95
        ... )

        >>> # Custom: Exactly 5 components
        >>> config = PCAConfig(enabled=True, n_components=5)
    """

    enabled: bool = Field(False, description="Run PCA (opt-in, can be expensive)")
    n_components: PositiveInt | Probability | Literal["auto"] = Field(
        "auto", description="Number of components: int (exact), float (variance %), or auto"
    )
    variance_threshold: Probability = Field(
        0.95, description="Cumulative variance to explain (for n_components='auto')"
    )
    standardize: bool = Field(
        True, description="Standardize features before PCA (strongly recommended)"
    )
    rotation: Literal["varimax", "quartimax"] | None = Field(
        None, description="Optional rotation for interpretability"
    )

    @model_validator(mode="after")
    def check_n_components_config(self) -> PCAConfig:
        """Validate n_components configuration."""
        if not self.enabled:
            return self

        if self.n_components == "auto" and not (0 < self.variance_threshold < 1):
            raise ValueError("variance_threshold must be in (0, 1) when n_components='auto'")

        return self


class ClusteringConfig(BaseConfig):
    """Configuration for feature clustering.

    Groups similar features to detect:
    - Redundant features
    - Feature families
    - Latent structure

    Attributes:
        enabled: Run clustering
        method: Clustering algorithm
        n_clusters: Number of clusters (int or "auto")
        linkage: Linkage method for hierarchical clustering
        distance_metric: Distance metric
        min_cluster_size: Minimum cluster size (for DBSCAN)
        eps: DBSCAN epsilon parameter

    Examples:
        >>> # Default: Disabled (opt-in)
        >>> config = ClusteringConfig()

        >>> # Custom: Hierarchical with auto clusters
        >>> config = ClusteringConfig(
        ...     enabled=True,
        ...     method=ClusteringMethod.HIERARCHICAL,
        ...     n_clusters="auto",
        ...     linkage=LinkageMethod.WARD
        ... )
    """

    enabled: bool = Field(False, description="Run clustering (opt-in)")
    method: ClusteringMethod = Field(ClusteringMethod.HIERARCHICAL, description="Clustering method")
    n_clusters: PositiveInt | Literal["auto"] = Field(
        "auto", description="Number of clusters (auto uses elbow method)"
    )
    linkage: LinkageMethod = Field(
        LinkageMethod.WARD, description="Linkage method for hierarchical clustering"
    )
    distance_metric: DistanceMetric = Field(DistanceMetric.EUCLIDEAN, description="Distance metric")
    min_cluster_size: PositiveInt = Field(5, description="Minimum cluster size (DBSCAN)")
    eps: PositiveFloat = Field(0.5, description="DBSCAN epsilon parameter")


class RedundancyConfig(BaseConfig):
    """Configuration for feature redundancy detection.

    Identifies redundant features using:
    - Pairwise correlation thresholds
    - Variance Inflation Factor (VIF)

    Attributes:
        enabled: Run redundancy detection
        correlation_threshold: Correlation threshold for redundancy
        compute_vif: Compute Variance Inflation Factor
        vif_threshold: VIF threshold for multicollinearity
        keep_strategy: Which feature to keep when redundant ("first", "last", "highest_ic")

    Examples:
        >>> # Default: correlation > 0.95
        >>> config = RedundancyConfig()

        >>> # Custom: VIF-based with 0.90 threshold
        >>> config = RedundancyConfig(
        ...     correlation_threshold=0.90,
        ...     compute_vif=True,
        ...     vif_threshold=5.0
        ... )
    """

    enabled: bool = Field(True, description="Run redundancy detection")
    correlation_threshold: Probability = Field(
        0.95, description="Correlation threshold for redundancy"
    )
    compute_vif: bool = Field(False, description="Compute VIF (can be slow for many features)")
    vif_threshold: PositiveFloat = Field(10.0, description="VIF threshold for multicollinearity")
    keep_strategy: Literal["first", "last", "highest_ic"] = Field(
        "highest_ic", description="Which feature to keep when redundant"
    )


class ModuleBConfig(BaseConfig):
    """Configuration for Module B: Cross-Feature Analysis.

    Analyzes relationships between features:
    - Correlation (linear, monotonic, lagged)
    - PCA (dimensionality reduction)
    - Clustering (feature grouping)
    - Redundancy (multicollinearity)

    Examples:
        >>> # Default: Correlation + redundancy only
        >>> config = ModuleBConfig()

        >>> # Custom: Full analysis with PCA
        >>> config = ModuleBConfig(
        ...     correlation=CorrelationConfig(lag_correlations=True),
        ...     pca=PCAConfig(enabled=True),
        ...     clustering=ClusteringConfig(enabled=True)
        ... )
    """

    correlation: CorrelationConfig = Field(
        default_factory=CorrelationConfig, description="Correlation analysis configuration"
    )
    pca: PCAConfig = Field(default_factory=PCAConfig, description="PCA configuration")
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="Clustering configuration"
    )
    redundancy: RedundancyConfig = Field(
        default_factory=RedundancyConfig, description="Redundancy detection configuration"
    )


# =============================================================================
# Module C: Feature-Outcome Relationships
# =============================================================================


class ICConfig(BaseConfig):
    """Configuration for Information Coefficient (IC) analysis.

    Measures predictive power of features:
    - Contemporaneous IC (lag 0)
    - Forward-looking IC (lag > 0)
    - HAC-adjusted IC (autocorrelation correction)
    - IC decay over time

    Attributes:
        enabled: Run IC analysis
        method: Correlation method for IC
        lag_structure: Lags to analyze (e.g., [0, 1, 5, 10, 21])
        hac_adjustment: Apply Newey-West HAC adjustment
        max_lag_hac: Maximum lag for HAC (auto = int(4*(n/100)^(2/9)))
        compute_t_stats: Compute t-statistics for IC
        compute_decay: Analyze IC decay over time

    Examples:
        >>> # Default: Pearson IC at lags 0, 1, 5
        >>> config = ICConfig()

        >>> # Custom: Spearman IC with HAC adjustment
        >>> config = ICConfig(
        ...     method=CorrelationMethod.SPEARMAN,
        ...     lag_structure=[0, 1, 5, 10, 21],
        ...     hac_adjustment=True
        ... )

    References:
        - Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
          Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    """

    enabled: bool = Field(True, description="Run IC analysis")
    method: CorrelationMethod = Field(
        CorrelationMethod.PEARSON, description="Correlation method for IC"
    )
    lag_structure: list[NonNegativeInt] = Field(
        default_factory=lambda: [0, 1, 5], description="Lags to analyze forward returns"
    )
    hac_adjustment: bool = Field(False, description="Apply Newey-West HAC adjustment (expensive)")
    max_lag_hac: PositiveInt | Literal["auto"] = Field(
        "auto", description="Maximum lag for HAC adjustment"
    )
    compute_t_stats: bool = Field(True, description="Compute t-statistics for IC")
    compute_decay: bool = Field(False, description="Analyze IC decay over time (expensive)")

    @field_validator("lag_structure")
    @classmethod
    def check_lag_structure(cls, v: list[int]) -> list[int]:
        """Ensure lag structure is valid."""
        if not v:
            raise ValueError("Must specify at least one lag")
        if any(lag < 0 for lag in v):
            raise ValueError("Lags must be non-negative")
        return sorted(v)


class BinaryClassificationConfig(BaseConfig):
    """Configuration for binary classification metrics.

    Evaluates signals as binary predictions:
    - Precision: % of predicted positives that are correct
    - Recall: % of actual positives that are detected
    - F1: Harmonic mean of precision and recall
    - Lift: Improvement over random
    - Coverage: % of universe with signals

    Attributes:
        enabled: Run binary classification analysis
        thresholds: Thresholds for converting scores to binary predictions
        metrics: Metrics to compute
        positive_class: What constitutes a "positive" signal
        compute_confusion_matrix: Compute confusion matrix
        compute_roc_curve: Compute ROC curve and AUC

    Examples:
        >>> # Default: Disabled (requires threshold selection)
        >>> config = BinaryClassificationConfig()

        >>> # Custom: Multiple thresholds
        >>> config = BinaryClassificationConfig(
        ...     enabled=True,
        ...     thresholds=[0.0, 0.5, 1.0],
        ...     metrics=["precision", "recall", "f1", "lift"]
        ... )
    """

    enabled: bool = Field(False, description="Run binary classification analysis (opt-in)")
    thresholds: list[float] = Field(
        default_factory=lambda: [0.0], description="Thresholds for binary conversion"
    )
    metrics: list[Literal["precision", "recall", "f1", "lift", "coverage"]] = Field(
        default_factory=lambda: ["precision", "recall", "f1"],
        description="Metrics to compute",
    )
    positive_class: int | str = Field(1, description="Positive class label")
    compute_confusion_matrix: bool = Field(True, description="Compute confusion matrix")
    compute_roc_curve: bool = Field(False, description="Compute ROC curve (expensive)")


class ThresholdAnalysisConfig(BaseConfig):
    """Configuration for threshold optimization and sensitivity analysis.

    Sweeps thresholds to find optimal values:
    - Maximize Sharpe, precision, recall, etc.
    - Subject to constraints (e.g., coverage >= 30%)
    - Sensitivity analysis

    Attributes:
        enabled: Run threshold analysis
        sweep_range: (min, max) threshold range to sweep
        n_points: Number of points in sweep
        optimization_target: What to optimize
        constraint_metric: Optional constraint metric
        constraint_value: Constraint threshold
        constraint_type: Constraint type (">=", "<=", "==")

    Examples:
        >>> # Default: Disabled (expensive)
        >>> config = ThresholdAnalysisConfig()

        >>> # Custom: Maximize Sharpe with 30% coverage
        >>> config = ThresholdAnalysisConfig(
        ...     enabled=True,
        ...     sweep_range=(-2.0, 2.0),
        ...     n_points=100,
        ...     optimization_target=ThresholdOptimizationTarget.SHARPE,
        ...     constraint_metric="coverage",
        ...     constraint_value=0.30,
        ...     constraint_type=">="
        ... )
    """

    enabled: bool = Field(False, description="Run threshold analysis (expensive, opt-in)")
    sweep_range: tuple[float, float] = Field((-2.0, 2.0), description="(min, max) threshold range")
    n_points: PositiveInt = Field(50, description="Number of points in sweep")
    optimization_target: ThresholdOptimizationTarget = Field(
        ThresholdOptimizationTarget.SHARPE, description="Optimization objective"
    )
    constraint_metric: str | None = Field(None, description="Optional constraint metric")
    constraint_value: float | None = Field(None, description="Constraint threshold")
    constraint_type: Literal[">=", "<=", "=="] = Field(">=", description="Constraint type")

    @model_validator(mode="after")
    def validate_sweep_range(self) -> ThresholdAnalysisConfig:
        """Validate sweep range."""
        if self.enabled:
            validate_min_max_range(self.sweep_range[0], self.sweep_range[1], "sweep_range")
        return self

    @model_validator(mode="after")
    def validate_constraint(self) -> ThresholdAnalysisConfig:
        """Validate constraint configuration."""
        has_metric = self.constraint_metric is not None
        has_value = self.constraint_value is not None

        if has_metric != has_value:
            raise ValueError(
                "Both constraint_metric and constraint_value must be set (or both None)"
            )

        return self


class MLDiagnosticsConfig(BaseConfig):
    """Configuration for ML model diagnostics.

    Advanced feature analysis using ML:
    - Feature importance (tree-based, permutation)
    - SHAP values (Shapley Additive Explanations)
    - Feature drift detection
    - Interaction detection

    Attributes:
        enabled: Run ML diagnostics
        feature_importance: Compute feature importance
        importance_method: Importance method ("tree", "permutation")
        shap_analysis: Compute SHAP values (very expensive)
        shap_sample_size: Subsample size for SHAP (None for full)
        drift_detection: Detect feature drift over time
        drift_method: Drift detection method
        drift_window: Rolling window for drift detection

    Examples:
        >>> # Default: Feature importance only
        >>> config = MLDiagnosticsConfig()

        >>> # Custom: Full analysis with SHAP
        >>> config = MLDiagnosticsConfig(
        ...     shap_analysis=True,
        ...     shap_sample_size=1000,
        ...     drift_detection=True
        ... )

    Warning:
        SHAP analysis can be very slow for large datasets. Use shap_sample_size
        to limit computation time.
    """

    enabled: bool = Field(True, description="Run ML diagnostics")
    feature_importance: bool = Field(True, description="Compute feature importance")
    importance_method: Literal["tree", "permutation"] = Field(
        "tree", description="Importance method: tree-based or permutation"
    )
    shap_analysis: bool = Field(False, description="Compute SHAP values (very expensive)")
    shap_sample_size: PositiveInt | None = Field(
        None, description="Subsample for SHAP (None = all data)"
    )
    drift_detection: bool = Field(False, description="Detect feature drift over time")
    drift_method: DriftDetectionMethod = Field(
        DriftDetectionMethod.KOLMOGOROV_SMIRNOV, description="Drift detection method"
    )
    drift_window: PositiveInt = Field(63, description="Rolling window for drift detection (days)")


class ModuleCConfig(BaseConfig):
    """Configuration for Module C: Feature-Outcome Relationships.

    Analyzes how features relate to outcomes:
    - IC analysis (predictive power)
    - Binary classification (signal quality)
    - Threshold optimization
    - ML diagnostics (importance, drift)

    Examples:
        >>> # Default: IC + ML diagnostics
        >>> config = ModuleCConfig()

        >>> # Custom: Full analysis
        >>> config = ModuleCConfig(
        ...     ic=ICConfig(lag_structure=[0, 1, 5, 10, 21]),
        ...     binary_classification=BinaryClassificationConfig(enabled=True),
        ...     threshold_analysis=ThresholdAnalysisConfig(enabled=True),
        ...     ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True)
        ... )
    """

    ic: ICConfig = Field(default_factory=ICConfig, description="IC analysis configuration")
    binary_classification: BinaryClassificationConfig = Field(
        default_factory=BinaryClassificationConfig,
        description="Binary classification configuration",
    )
    threshold_analysis: ThresholdAnalysisConfig = Field(
        default_factory=ThresholdAnalysisConfig, description="Threshold analysis configuration"
    )
    ml_diagnostics: MLDiagnosticsConfig = Field(
        default_factory=MLDiagnosticsConfig, description="ML diagnostics configuration"
    )


# =============================================================================
# Top-Level Feature Evaluator Configuration
# =============================================================================


class FeatureEvaluatorConfig(BaseConfig):
    """Top-level configuration for feature evaluation (Modules A, B, C).

    Orchestrates comprehensive feature analysis:
    - **Module A**: Individual feature diagnostics
    - **Module B**: Cross-feature relationships
    - **Module C**: Feature-outcome relationships

    Attributes:
        module_a: Feature diagnostics configuration
        module_b: Cross-feature analysis configuration
        module_c: Feature-outcome configuration
        export_recommendations: Export preprocessing recommendations
        export_to_qfeatures: Export in qfeatures-compatible format
        return_dataframes: Return metrics as DataFrames
        n_jobs: Parallel processing (-1 for all cores)
        cache_enabled: Enable caching of expensive computations
        cache_dir: Cache directory
        verbose: Enable verbose output

    Examples:
        >>> # Quick start with defaults
        >>> config = FeatureEvaluatorConfig()
        >>> evaluator = FeatureEvaluator(config)
        >>> results = evaluator.evaluate(features_df, outcomes_df)

        >>> # Load from YAML
        >>> config = FeatureEvaluatorConfig.from_yaml("feature_config.yaml")

        >>> # Use preset
        >>> config = FeatureEvaluatorConfig.for_quick_analysis()

        >>> # Custom configuration
        >>> config = FeatureEvaluatorConfig(
        ...     module_a=ModuleAConfig(
        ...         stationarity=StationarityConfig(significance_level=0.01)
        ...     ),
        ...     module_c=ModuleCConfig(
        ...         ic=ICConfig(lag_structure=[0, 1, 5, 10, 21])
        ...     ),
        ...     n_jobs=-1
        ... )
    """

    module_a: ModuleAConfig = Field(
        default_factory=ModuleAConfig, description="Feature diagnostics (Module A)"
    )
    module_b: ModuleBConfig = Field(
        default_factory=ModuleBConfig, description="Cross-feature analysis (Module B)"
    )
    module_c: ModuleCConfig = Field(
        default_factory=ModuleCConfig, description="Feature-outcome analysis (Module C)"
    )

    # Integration settings
    export_recommendations: bool = Field(True, description="Export preprocessing recommendations")
    export_to_qfeatures: bool = Field(False, description="Export in qfeatures-compatible format")
    return_dataframes: bool = Field(True, description="Return metrics as DataFrames")

    # Computational settings
    n_jobs: int = Field(-1, ge=-1, description="Parallel jobs (-1 = all cores)")
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "qeval" / "features",
        description="Cache directory",
    )
    verbose: bool = Field(False, description="Verbose output")

    @classmethod
    def for_quick_analysis(cls) -> FeatureEvaluatorConfig:
        """Preset for quick exploratory analysis (fast, essential diagnostics only).

        Returns:
            Config optimized for speed
        """
        return cls(
            module_a=ModuleAConfig(
                stationarity=StationarityConfig(pp_enabled=False),
                volatility=VolatilityConfig(detect_clustering=False),
                distribution=DistributionConfig(detect_outliers=False),
            ),
            module_b=ModuleBConfig(
                correlation=CorrelationConfig(lag_correlations=False),
                pca=PCAConfig(enabled=False),
                clustering=ClusteringConfig(enabled=False),
            ),
            module_c=ModuleCConfig(
                ic=ICConfig(hac_adjustment=False, compute_decay=False),
                ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False, drift_detection=False),
            ),
        )

    @classmethod
    def for_research(cls) -> FeatureEvaluatorConfig:
        """Preset for academic research (comprehensive, expensive analyses enabled).

        Returns:
            Config with all analyses enabled
        """
        return cls(
            module_a=ModuleAConfig(
                stationarity=StationarityConfig(pp_enabled=True),
                volatility=VolatilityConfig(window_sizes=[10, 21, 63]),
                distribution=DistributionConfig(
                    detect_outliers=True,
                    normality_tests=[
                        NormalityTest.JARQUE_BERA,
                        NormalityTest.SHAPIRO,
                        NormalityTest.ANDERSON,
                    ],
                ),
            ),
            module_b=ModuleBConfig(
                correlation=CorrelationConfig(
                    methods=[
                        CorrelationMethod.PEARSON,
                        CorrelationMethod.SPEARMAN,
                        CorrelationMethod.KENDALL,
                    ],
                    lag_correlations=True,
                ),
                pca=PCAConfig(enabled=True),
                clustering=ClusteringConfig(enabled=True),
            ),
            module_c=ModuleCConfig(
                ic=ICConfig(
                    lag_structure=[0, 1, 5, 10, 21],
                    hac_adjustment=True,
                    compute_decay=True,
                ),
                binary_classification=BinaryClassificationConfig(enabled=True),
                threshold_analysis=ThresholdAnalysisConfig(enabled=True),
                ml_diagnostics=MLDiagnosticsConfig(
                    shap_analysis=True,
                    drift_detection=True,
                ),
            ),
        )

    @classmethod
    def for_production(cls) -> FeatureEvaluatorConfig:
        """Preset for production monitoring (fast, focused on drift and degradation).

        Returns:
            Config optimized for production monitoring
        """
        return cls(
            module_a=ModuleAConfig(
                stationarity=StationarityConfig(pp_enabled=False),
                acf=ACFConfig(enabled=False),
                volatility=VolatilityConfig(enabled=False),
                distribution=DistributionConfig(test_normality=False, compute_moments=True),
            ),
            module_b=ModuleBConfig(
                correlation=CorrelationConfig(lag_correlations=False),
                pca=PCAConfig(enabled=False),
                clustering=ClusteringConfig(enabled=False),
            ),
            module_c=ModuleCConfig(
                ic=ICConfig(compute_decay=False),
                ml_diagnostics=MLDiagnosticsConfig(
                    feature_importance=True,
                    drift_detection=True,
                    drift_window=21,  # Faster detection
                ),
            ),
        )
