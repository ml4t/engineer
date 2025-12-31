"""Unified visualization for feature analysis.

This module provides comprehensive summary visualizations combining multiple
analysis types:

- Feature importance (MDI, permutation, SHAP)
- Information Coefficient (IC) analysis
- Feature correlation analysis

All in publication-ready formats.

Key Functions
-------------
plot_feature_analysis_summary : Create 3-panel comprehensive summary
export_plot : Export figure to PNG/PDF with quality control

Examples
--------
>>> from ml4t.engineer.outcome import FeatureOutcome
>>> from ml4t.engineer.relationships import compute_correlation_matrix
>>> from ml4t.engineer.visualization import plot_feature_analysis_summary, export_plot
>>>
>>> # Run analysis
>>> analyzer = FeatureOutcome()
>>> results = analyzer.run_analysis(features_df, outcomes_df)
>>> corr = compute_correlation_matrix(features_df)
>>>
>>> # Create and export comprehensive summary
>>> fig = plot_feature_analysis_summary(results, corr, top_n=15)
>>> export_plot(fig, "feature_analysis.png", dpi=300)
"""

from ml4t.engineer.visualization.summary import export_plot, plot_feature_analysis_summary

__all__ = [
    "plot_feature_analysis_summary",
    "export_plot",
]
