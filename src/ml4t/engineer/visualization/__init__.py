"""Unified visualization for feature analysis.

.. deprecated:: 0.1.0a8
    Feature outcome visualizations have moved to ``ml4t-diagnostic``.

This module provides basic plot export utilities. For feature analysis
visualizations (importance, IC, drift), use ``ml4t.diagnostic.visualization``.

Key Functions
-------------
export_plot : Export figure to PNG/PDF with quality control
plot_feature_analysis_summary : DEPRECATED - use ml4t.diagnostic

Examples
--------
>>> # For feature analysis visualizations, use ml4t-diagnostic:
>>> from ml4t.diagnostic.evaluation import FeatureOutcome
>>> from ml4t.diagnostic.visualization import plot_importance_summary
>>>
>>> analyzer = FeatureOutcome()
>>> results = analyzer.run_analysis(features_df, outcomes_df)
>>> fig = plot_importance_summary(results)
>>>
>>> # Use export_plot for saving any matplotlib figure:
>>> from ml4t.engineer.visualization import export_plot
>>> export_plot(fig, "analysis.png", dpi=300)
"""

from ml4t.engineer.visualization.summary import export_plot, plot_feature_analysis_summary

__all__ = [
    "plot_feature_analysis_summary",
    "export_plot",
]
