# Book Guide

This guide maps `ml4t-engineer` tasks to public notebooks from *Machine Learning for
Trading, Third Edition*. Every link uses companion commit
[`d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb`](https://github.com/stefan-jansen/machine-learning-for-trading/tree/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb).
All paths and the paired Python sources were checked at that commit.

The relationship column distinguishes three cases:

- **Calls Engineer**: the notebook imports and runs the named `ml4t.engineer` API.
- **Teaches manually**: the notebook implements the method for instruction and does not
  use Engineer for that task.
- **Related workflow**: the notebook shows where the task fits, but its broader workflow
  is not an Engineer API example.

## Feature computation and discovery

| Book notebook | Relationship | Engineer API | Task guide |
|---|---|---|---|
| [The ml4t Library Ecosystem](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) | Calls Engineer to inspect registry metadata and run `compute_features()` with names and parameter dictionaries | `compute_features`, `get_registry` | [Features](../user-guide/features.md), [Feature Discovery](../user-guide/discovery.md) |
| [Price and Volume Feature Families](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/01_price_volume_features.ipynb) | Calls Engineer for registry features, volatility, regime, risk, and fractional-differencing functions; also derives selected features manually | `compute_features` and feature modules | [Features](../user-guide/features.md), [ML Readiness](../user-guide/ml-readiness.md) |
| [Microstructure Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/02_microstructure_features.ipynb) | Calls Engineer for the tick rule and liquidity estimators, then builds a wider teaching workflow | `ml4t.engineer.features.microstructure` | [Features](../user-guide/features.md) |
| [Structural and Cross-Instrument Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/03_structural_cross_instrument_features.ipynb) | Calls Engineer for market beta; teaches carry and options features manually | `beta_to_market` | [Features](../user-guide/features.md) |
| [Slow Features and Context](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/08_financial_features/04_fundamentals_macro_calendar.ipynb) | Calls Engineer for calendar encoding; teaches point-in-time joins and slow features manually | `cyclical_encode` | [Features](../user-guide/features.md) |
| [Panel Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/09_model_based_features/14_panel_features.ipynb) | Calls Engineer for cross-asset features and compares them with manual statistical work | `ml4t.engineer.features.cross_asset` | [Features](../user-guide/features.md) |
| [ETFs: Feature Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/03_financial_features.ipynb) | Calls individual Engineer feature functions inside a full case-study pipeline | momentum, trend, volatility, volume, and regime feature modules | [Features](../user-guide/features.md) |

The chapter notebooks use book datasets and plotting dependencies. The
[Quickstart](../getting-started/quickstart.md) provides an offline synthetic path for
the same released computation API.

## Labeling

| Book notebook | Relationship | Engineer API | Task guide |
|---|---|---|---|
| [Label Engineering Methods](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/03_label_methods.ipynb) | Calls Engineer for fixed-horizon, percentile, triple-barrier, ATR-barrier, trend-scanning, meta-labeling, and sample-weighting workflows | `ml4t.engineer.labeling`, `LabelingConfig` | [Labeling](../user-guide/labeling.md) |
| [ETFs: Label Engineering](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/02_labels.ipynb) | Related workflow that constructs and audits case-study labels without calling Engineer | no direct Engineer call | [Labeling](../user-guide/labeling.md) |

The second notebook is useful for the artifact and timing workflow. It is not evidence
that the case study uses Engineer's labeling functions.

## Alternative bars

| Book notebook | Relationship | Engineer API | Task guide |
|---|---|---|---|
| [ITCH Bar Sampling](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/03_market_microstructure/14_itch_bar_sampling.ipynb) | Calls Engineer for tick, volume, dollar, imbalance, and run bars on ITCH trades | bar sampler classes | [Alternative Bars](../user-guide/bars.md) |
| [Information-Bar Formulas and Parameters](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/03_market_microstructure/16_itch_information_bars.ipynb) | Calls Engineer and compares manual formulas with adaptive, fixed, and window samplers | imbalance-bar sampler classes | [Alternative Bars](../user-guide/bars.md) |
| [Databento Bar Calibration](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/03_market_microstructure/17_databento_bar_sampling.ipynb) | Calls Engineer in a multi-day calibration workflow that requires Databento data | bar sampler classes | [Alternative Bars](../user-guide/bars.md) |

The first two notebooks require book data. The Databento notebook also requires the
vendor dataset. The task guide and `examples/bars_example.py` provide an offline,
synthetic verification path.

## Preprocessing and fractional differencing

| Book notebook | Relationship | Engineer API | Task guide |
|---|---|---|---|
| [Preprocessing Pipeline](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/07_defining_the_learning_task/02_preprocessing_pipeline.ipynb) | Calls Engineer's `StandardScaler` for train-only fitting; teaches the broader cleaning pipeline manually | `StandardScaler` | [Preprocessing](../user-guide/preprocessing.md) |
| [Fractional Differencing](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/09_model_based_features/03_fractional_differencing.ipynb) | Calls Engineer's fractional-differencing helpers while teaching the statistical method | `ffdiff`, `find_optimal_d`, `fdiff_diagnostics` | [Fractional Differencing](../user-guide/fractional-differencing.md) |
| [ETFs: Model-Based Features](https://github.com/stefan-jansen/machine-learning-for-trading/blob/d2edec54b1c7a6a9d7a97d8129eb05db4491e1eb/case_studies/etfs/04_model_based_features.ipynb) | Calls `ffdiff` inside a walk-forward case-study workflow; its HMM and GARCH work is outside Engineer's fractional-differencing API | `ffdiff` | [Fractional Differencing](../user-guide/fractional-differencing.md) |

No checked book notebook at this revision calls `create_dataset_builder`. Use the
[Dataset Builder guide](../user-guide/dataset-builder.md) and the repository's
`examples/complete_workflow_example.py` for that workflow.

## Alphalens migration scope

Engineer overlaps with Alphalens only before factor analysis: it can compute factor
values with `compute_features()` and fit preprocessing state on training data. Engineer
does not replace Alphalens tearsheets, information-coefficient analysis, quantile-return
analysis, turnover analysis, or event studies. Use
[ML4T Diagnostic](https://www.ml4trading.io/docs/diagnostic/) for those evaluation
tasks. The book's feature notebooks above show factor construction; topical similarity
does not make them Alphalens replacements.

## Supported and experimental boundaries

- The principal documented workflows are feature computation and discovery, labeling,
  alternative bars, preprocessing, dataset building, and fractional differencing.
- Cross-asset functions are supported advanced APIs. Validate asset ordering and
  point-in-time alignment before use.
- Adaptive imbalance bars require calibration. Fixed-threshold samplers provide the
  bounded production path described in the [Alternative Bars guide](../user-guide/bars.md).
- `fdiff_diagnostics()` and `find_optimal_d()` require the `stats` extra. `ffdiff()` is
  available from the core installation.
- The optional DuckDB store is experimental and has no verified book adoption path.
- `transfer_entropy()` is not implemented for production use. It is not part of the
  principal documented workflow.

## Run a workflow first

- [Quickstart](../getting-started/quickstart.md) computes features from synthetic data.
- [Features](../user-guide/features.md) covers configuration and input contracts.
- [Labeling](../user-guide/labeling.md) covers target construction and timing.
- [Alternative Bars](../user-guide/bars.md) covers sampler choice and calibration.
- [Dataset Builder](../user-guide/dataset-builder.md) covers leakage-safe splits.
- [API Reference](../api/index.md) provides exact signatures.
