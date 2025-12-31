"""Complete ML4T Engineer Workflow Example.

This example demonstrates the end-to-end feature engineering workflow:
1. Diagnostics - Analyze feature quality
2. Relationships - Compute correlation matrices
3. Outcome - Analyze feature-outcome relationships
4. Selection - Systematic feature filtering
5. Visualization - Generate comprehensive plots

This is the recommended workflow for production ML research.
"""

# ruff: noqa: E402
from __future__ import annotations

import numpy as np
import polars as pl

# ============================================================================
# Step 0: Generate Sample Data
# ============================================================================

print("=" * 80)
print("ML4T ENGINEER - Complete Workflow Example")
print("=" * 80)
print()

# Create realistic sample data (in production, load from your data source)
np.random.seed(42)
n_rows = 10_000

# Generate outcome variable (e.g., forward returns)
outcome = np.random.randn(n_rows) * 0.02  # 2% daily volatility

# Generate features with varying quality
features = pl.DataFrame(
    {
        # High quality features (predictive)
        "momentum_5d": outcome + np.random.randn(n_rows) * 0.01,
        "momentum_20d": np.convolve(outcome, np.ones(5) / 5, mode="same"),
        "volatility_10d": np.abs(outcome) + np.random.randn(n_rows) * 0.005,
        # Medium quality features
        "volume_trend": np.cumsum(np.random.randn(n_rows) * 0.1),
        "rsi_14": 50 + 20 * np.random.randn(n_rows),
        # Low quality features (noise)
        "noise_1": np.random.randn(n_rows),
        "noise_2": np.random.randn(n_rows),
        # Highly correlated pair (for redundancy testing)
        "signal_a": outcome * 0.8 + np.random.randn(n_rows) * 0.05,
        "signal_b": outcome * 0.8 + np.random.randn(n_rows) * 0.06,
        # Non-stationary feature (should be filtered)
        "trend": np.cumsum(np.random.randn(n_rows)),
    }
)

outcomes = pl.DataFrame({"forward_returns_1d": outcome})

print(f"✓ Generated sample data: {n_rows} rows, {len(features.columns)} features")
print()

# ============================================================================
# Step 1: Feature Quality Diagnostics
# ============================================================================

print("=" * 80)
print("STEP 1: Feature Quality Diagnostics")
print("=" * 80)
print()

from ml4t.engineer.diagnostics import diagnose_features

# Diagnose each feature individually
print("Analyzing feature quality (stationarity, autocorrelation, volatility, distribution)...")
diagnostic_results = {}
for col in features.columns:
    result = diagnose_features(features[col].to_pandas())
    diagnostic_results[col] = result

    # Print summary for non-stationary features
    if result.stationarity and result.stationarity.consensus in [
        "strong_nonstationary",
        "likely_nonstationary",
    ]:
        print(f"  ⚠️  {col}: Non-stationary ({result.stationarity.consensus})")

print(f"\n✓ Completed diagnostics for {len(diagnostic_results)} features")
print()

# ============================================================================
# Step 2: Feature Relationships Analysis
# ============================================================================

print("=" * 80)
print("STEP 2: Feature Relationships Analysis")
print("=" * 80)
print()

from ml4t.engineer.relationships import compute_correlation_matrix

# Compute correlation matrix
print("Computing feature correlation matrix...")
corr_matrix = compute_correlation_matrix(features, method="pearson")

# Identify highly correlated features
print("\nHighly correlated feature pairs (>0.7):")
for i, feat_a in enumerate(features.columns):
    row = corr_matrix.filter(pl.col("feature") == feat_a)
    for feat_b in features.columns[i + 1 :]:
        corr = abs(row.select(feat_b).item())
        if corr > 0.7:
            print(f"  • {feat_a} ↔ {feat_b}: {corr:.3f}")

print(f"\n✓ Correlation matrix computed ({len(features.columns)}x{len(features.columns)})")
print()

# ============================================================================
# Step 3: Feature-Outcome Analysis
# ============================================================================

print("=" * 80)
print("STEP 3: Feature-Outcome Analysis")
print("=" * 80)
print()

from ml4t.engineer.config.feature_config import ICConfig, MLDiagnosticsConfig, ModuleCConfig
from ml4t.engineer.outcome import FeatureOutcome

# Configure analysis (skip SHAP for speed in example)
config = ModuleCConfig(
    ic=ICConfig(lag_structure=[1, 5, 10]),  # Test 1, 5, 10-day forward IC
    ml_diagnostics=MLDiagnosticsConfig(
        shap_analysis=False,  # Set to True for SHAP importance (slower)
    ),
)

# Run analysis
print("Analyzing feature-outcome relationships...")
print("  - Computing Information Coefficient (IC)")
print("  - Computing feature importance (MDI, permutation)")
print("  - Analyzing predictive power...")

analyzer = FeatureOutcome(config=config)
outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])

# Print top features by IC
print("\nTop 5 features by Information Coefficient (1-day forward):")
ic_by_feature = [
    (feat, result.ic_by_lag.get(1, 0.0)) for feat, result in outcome_result.ic_results.items()
]
ic_by_feature.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, ic in ic_by_feature[:5]:
    print(f"  {feat:20s}: IC = {ic:+.4f}")

# Print top features by importance
print("\nTop 5 features by MDI importance:")
importance_by_feature = [
    (feat, result.mdi_importance or 0.0)
    for feat, result in outcome_result.importance_results.items()
]
importance_by_feature.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importance_by_feature[:5]:
    print(f"  {feat:20s}: Importance = {imp:.4f}")

print("\n✓ Feature-outcome analysis complete")
print()

# ============================================================================
# Step 4: Systematic Feature Selection
# ============================================================================

print("=" * 80)
print("STEP 4: Systematic Feature Selection")
print("=" * 80)
print()

from ml4t.engineer.selection import FeatureSelector

# Create selector
selector = FeatureSelector(outcome_result, corr_matrix)

print(f"Starting with {len(selector.selected_features)} features")
print()

# Apply multi-stage filtering
print("Applying systematic filters...")

# Filter 1: IC threshold
print("  1. IC threshold (|IC| > 0.01)...")
selector.filter_by_ic(threshold=0.01, min_periods=100)
print(f"     → {len(selector.selected_features)} features remaining")

# Filter 2: Correlation (remove redundant features)
print("  2. Correlation filter (remove if corr > 0.75, keep higher IC)...")
selector.filter_by_correlation(threshold=0.75, keep_strategy="higher_ic")
print(f"     → {len(selector.selected_features)} features remaining")

# Filter 3: Importance (keep top features)
print("  3. Importance filter (keep top 5 by MDI)...")
selector.filter_by_importance(threshold=0, method="mdi", top_k=5)
print(f"     → {len(selector.selected_features)} features remaining")

# Get selection report
report = selector.get_selection_report()

print("\n✓ Feature selection complete")
print(f"  Initial features: {report.initial_features}")
print(f"  Final features: {report.final_features}")
print(f"  Selected features: {', '.join(report.final_features)}")
print()

# ============================================================================
# Step 5: Visualization
# ============================================================================

print("=" * 80)
print("STEP 5: Visualization & Export")
print("=" * 80)
print()

from ml4t.engineer.visualization import export_plot, plot_feature_analysis_summary

# Generate comprehensive summary plot
print("Generating 3-panel summary visualization...")
print("  - Panel 1: Feature importance (MDI)")
print("  - Panel 2: Information Coefficient by forward horizon")
print("  - Panel 3: Feature correlation heatmap")

fig = plot_feature_analysis_summary(
    outcome_result,
    corr_matrix,
    top_n=5,
    importance_type="mdi",
    title="ML4T Engineer - Feature Analysis Summary",
)

# Export to file
output_file = "feature_analysis_summary.png"
export_plot(fig, output_file, dpi=150)
print(f"\n✓ Visualization saved to: {output_file}")
print()

# ============================================================================
# Step 6: Final Recommendations
# ============================================================================

print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

print("Selected Features for ML Model:")
for i, feat in enumerate(report.final_features, 1):
    ic_result = outcome_result.ic_results[feat]
    imp_result = outcome_result.importance_results[feat]
    ic_1d = ic_result.ic_by_lag.get(1, 0.0)
    importance = imp_result.mdi_importance or 0.0

    print(f"  {i}. {feat}")
    print(f"     IC (1-day): {ic_1d:+.4f}")
    print(f"     Importance: {importance:.4f}")
    print()

print("Next Steps:")
print("  1. Use selected features in your ML model")
print("  2. Monitor feature drift over time (use analyze_drift)")
print("  3. Retrain periodically to maintain performance")
print("  4. Consider ensemble methods to combine multiple features")
print()

print("=" * 80)
print("Workflow complete! ✓")
print("=" * 80)
