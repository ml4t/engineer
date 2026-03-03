# Book Integration Audit: ml4t-engineer

*Audit date: 2026-03-03 | Library version: v0.1.0a11 | Book: Machine Learning for Trading, 3rd Edition*

## Executive Summary

ml4t-engineer's core value proposition is validated by heavy book usage: 120 features, 7 labeling methods, and 11 bar samplers are all exercised in chapters 3, 7-9 and all 9 case studies. Several high-quality modules (MLDatasetBuilder, FeatureCatalog.search()) previously had zero book exposure but are now showcased in Ch7 NB10. The pipeline engine and DuckDB store are honestly low-value.

---

## Usage Matrix: Chapter x Module

### Feature Computation (`compute_features`)

| Chapter / Case Study | Notebook | Features Used | Notes |
|---------------------|----------|---------------|-------|
| Ch7 | `10_ml4t_library_ecosystem.py` | rsi, sma, ema, atr, macd, bollinger_bands | Registry tour, 3 input formats, MLDatasetBuilder |
| Ch8 | `01_price_volume_features.py` | momentum (31), trend (10), volatility (15) | Core feature teaching |
| Ch8 | `02_microstructure_features.py` | microstructure (12) | Kyle Lambda, VPIN, Amihud |
| Ch8 | `03_structural_cross_instrument_features.py` | cross-asset (10) | beta_to_market, correlations |
| Ch8 | `04_fundamentals_macro_calendar.py` | ML features, calendar | Lag, encodings, macro features |
| Ch9 | `02_structural_breaks.py` | statistics | Structural break detection |
| Ch9 | `03_fractional_differencing.py` | ffdiff (4) | Fractional differencing + ADF |
| Ch9 | `05_spectral_features.py` | ML features | Spectral, FFT |
| Ch9 | `08_garch_volatility.py` | volatility (15) | GARCH, EWMA, realized vol |
| Ch9 | `09_har_rough_volatility.py` | volatility | HAR model features |
| Ch9 | `11_hmm_regimes.py` | regime (6) | Hurst, HMM state probabilities |
| Ch9 | `13_regime_as_feature.py` | regime (6) | Regime encoding as features |
| Ch9 | `14_panel_features.py` | cross-asset (10) | Cross-sectional panel features |
| ETFs | `03_features.py`, `04_temporal.py` | momentum, volatility, volume, ffdiff | Full pipeline |
| US Equities Panel | `03_features.py`, `04_temporal.py` | momentum, volatility, ffdiff | Full pipeline |
| CME Futures | `02_labels.py`, `03_features.py` | momentum, volatility, atr | Futures-specific |

### Labeling Methods

| Chapter / Case Study | Notebook | Method | Config Style |
|---------------------|----------|--------|--------------|
| Ch7 | `03_label_methods.py` | triple_barrier_labels | LabelingConfig.triple_barrier() |
| Ch7 | `03_label_methods.py` | rolling_percentile_binary_labels | Direct call |
| Ch7 | `03_label_methods.py` | trend_scanning_labels | Direct call |
| Ch7 | `03_label_methods.py` | meta_labels + compute_bet_size | Meta-labeling workflow |
| Ch7 | `03_label_methods.py` | sequential_bootstrap | Sample weighting |
| CME Futures | `02_labels.py` | atr_triple_barrier_labels | LabelingConfig.atr_barrier() |
| ETFs | `02_labels.py` | rolling_percentile_binary_labels | Direct call |
| US Equities Panel | `02_labels.py` | triple_barrier_labels | LabelingConfig |
| All case studies | `02_labels.py` | fixed_time_horizon_labels | Direct call |

### Alternative Bar Sampling

| Chapter | Notebook | Sampler | Notes |
|---------|----------|---------|-------|
| Ch3 | `08_itch_bar_sampling.py` | TickBarSampler, VolumeBarSampler, DollarBarSampler | ITCH tick data |
| Ch3 | `10_itch_information_bars.py` | TickImbalanceBarSampler, FixedTickImbalanceBarSampler | Information-driven bars |
| Ch3 | `13_databento_bar_sampling.py` | Bar sampling on Databento data | Alternative data source |

### Feature Discovery & Registry

| Chapter | Notebook | API Used |
|---------|----------|----------|
| Ch7 | `10_ml4t_library_ecosystem.py` | get_registry(), list_all(), get(), list_by_category() |
| Ch7 | `10_ml4t_library_ecosystem.py` | feature_catalog.search(), feature_catalog.list(), describe() |
| Ch7 | `10_ml4t_library_ecosystem.py` | compute_features (3 formats: list, dict, YAML) |

### MLDatasetBuilder & Preprocessing

| Chapter | Notebook | API Used |
|---------|----------|----------|
| Ch7 | `10_ml4t_library_ecosystem.py` | create_dataset_builder, train_test_split, scaler="robust" |
| Ch7 | `10_ml4t_library_ecosystem.py` | LabelingConfig.to_yaml(), from_yaml() |
| Ch7 | `02_preprocessing_pipeline.py` | StandardScaler, split-aware preprocessing |

---

## Book Chapter Structure (Actual)

| Chapter | Directory | Notebooks | Primary ml4t-engineer Usage |
|---------|-----------|-----------|----------------------------|
| Ch3 | `03_market_microstructure/` | 17 | bars module |
| Ch7 | `07_defining_learning_task/` | 10 | labeling, registry, dataset builder |
| Ch8 | `08_feature_engineering/` | 8 + meta | features (all categories) |
| Ch9 | `09_time_series_analysis/` | 14 + meta | volatility, regime, ffdiff, cross-asset |

### Case Study Structure (Standard Pattern)

All 9 case studies follow the same 18-file pattern:

| Step | File | ml4t-engineer Usage |
|------|------|---------------------|
| Setup | `01_setup.py` | — |
| Labels | `02_labels.py` | `atr_triple_barrier_labels`, `rolling_percentile_binary_labels`, `fixed_time_horizon_labels` |
| Features | `03_features.py` | `compute_features`, individual feature functions |
| Temporal | `04_temporal.py` | `ffdiff`, walk-forward CV |
| Evaluation | `05_evaluation.py` | — (ml4t-diagnostic) |
| Models | `06-13_*.py` | — |
| Backtest | `14_backtest.py` | — (ml4t-backtest) |

---

## Feature Triage

### Heavily Used (Core Value)

| Module | Lines | Book Coverage | Confidence | Action |
|--------|-------|---------------|------------|--------|
| 120 features (10 categories) | ~8,000 | Ch8 (8 notebooks), Ch9 (14 notebooks), 9 case studies | 59 TA-Lib validated | Keep, document well |
| 7 labeling methods | ~2,000 | Ch7 NB03, all 9 case study `02_labels.py` | AFML validated | Keep, document well |
| 11 bar samplers | ~2,000 | Ch3 (3 notebooks) | Production-ready | Keep, document well |
| ffdiff module | 383 | Ch9 NB03, ETFs/Equities `04_temporal.py` | Unique value | Keep, dedicated guide |
| LabelingConfig | 467 | Ch7 NB03, all case studies | API surface | Keep, document well |
| Registry/Catalog | ~650 | Ch7 NB10 | Discovery | Keep, dedicated guide |
| MLDatasetBuilder | 638 | Ch7 NB10 (newly added) | Leakage-safe prep | Keep, dedicated guide |

### Honestly Low-Value

| Module | Lines | Assessment | Recommended Action |
|--------|-------|------------|-------------------|
| Pipeline engine | ~300 | `compute_features` already handles dependency ordering. Thin DAG wrapper adds little. | Label "Advanced" |
| Store (DuckDB) | ~500 | No adoption path, no book usage, no clear user need. | Label "Experimental" |
| FeatureSelector | stub | Correctly moved to ml4t-diagnostic. Stub remains as migration aid. | Keep stub, document redirect |

---

## Case Studies NOT Using ml4t-engineer

These case studies implement features manually. This is **correct** in most cases:

| Case Study | Reason for Manual Implementation | Library Overlap |
|-----------|----------------------------------|-----------------|
| Crypto Perps Funding | Domain-specific funding rate features | None — inline appropriate |
| S&P 500 Options / Option Analytics | Greeks, IV surfaces — specialized derivatives analytics | None — out of scope |
| US Firm Characteristics | Accounting ratios from financial statements | None — out of scope |
| NASDAQ-100 Microstructure | Kyle's Lambda, Amihud, VPIN implemented manually for pedagogy | **High** — all in library (callout added) |
| FX Pairs | Garman-Klass volatility, momentum features | **Partial** — some in library (callout added) |

---

## Cross-Reference: Book Notebooks Using ml4t.engineer

### Direct imports (`from ml4t.engineer`)

| File | Imports | Status |
|------|---------|--------|
| `07_defining_learning_task/code/10_ml4t_library_ecosystem.py` | compute_features, get_registry, feature_catalog, create_dataset_builder, LabelingConfig | Working |
| `07_defining_learning_task/code/03_label_methods.py` | LabelingConfig, 7 labeling functions | Working (migrated from BarrierConfig, un-skipped) |
| `07_defining_learning_task/code/04_minimum_favorable_adverse_excursion.py` | LabelingConfig | Working (migrated from BarrierConfig, un-skipped) |
| `08_feature_engineering/code/01_price_volume_features.py` | ml4t.engineer.features.volatility, momentum, trend | Working |
| `09_time_series_analysis/code/08_garch_volatility.py` | ml4t.engineer.features.volatility (6 functions) | Working |
| All case study `02_labels.py` | ml4t.engineer.labeling (atr_triple_barrier_labels etc.) | Working |
| All case study `03_features.py` | ml4t.engineer.features (momentum, volatility, regime, trend) | Working |

### Indirect usage (via `utils/label_functions.py`)

Some case study `02_labels.py` files use standalone label utility wrappers that mirror the ml4t.engineer API. These are isolated from API changes but are less idiomatic.

---

## Documentation Coverage

| User Guide Page | Lines | Book Reference | Status |
|----------------|-------|----------------|--------|
| `labeling.md` | 522 | Ch7 `03_label_methods.py`, CME `02_labels.py`, ETFs `02_labels.py` | Complete |
| `features.md` | 388 | Ch8 NB01-04, Ch9 NB08-14, ETFs/Equities/CME `03_features.py` | Complete |
| `bars.md` | 405 | Ch3 `08_itch_bar_sampling.py`, `10_itch_information_bars.py`, `13_databento_bar_sampling.py` | Complete |
| `ml-readiness.md` | 178 | Ch8 `01_price_volume_features.py` | Complete |
| `discovery.md` | 162 | Ch7 `10_ml4t_library_ecosystem.py` | Complete |
| `fractional-differencing.md` | 188 | Ch9 `03_fractional_differencing.py`, ETFs/Equities `04_temporal.py` | Complete |
| `preprocessing.md` | 171 | Ch7 `02_preprocessing_pipeline.py` | Complete |
| `dataset-builder.md` | 201 | Ch7 `10_ml4t_library_ecosystem.py` | Complete |

---

## Value Assessment

### What ml4t-engineer does well

1. **Feature computation is the clear winner**: 120 features, validated, fast, config-driven. Used in 30+ notebooks across 8 chapters and 9 case studies.
2. **Labeling methods are comprehensive**: All 7 AFML methods implemented, validated, calendar-aware. Used in every case study.
3. **Bar sampling is uniquely valuable**: No other Python library provides production-quality imbalance bars with threshold spiral warnings.
4. **Registry/discovery is elegant**: Metadata-driven feature selection with TA-Lib compatibility flags and normalization status.
5. **MLDatasetBuilder fills a real gap**: Leakage-safe dataset prep with CV integration — now demonstrated in Ch7 NB10.

### What should be scoped honestly

1. **Pipeline engine**: `compute_features` already does dependency ordering. The Pipeline class adds a thin DAG wrapper that few users need. Document as "Advanced".
2. **DuckDB Store**: No user demand, no book usage. Keep but label experimental.
3. **Cross-asset features (8 of 10 unused in book)**: Strong implementations but limited coverage. Only `beta_to_market` and `rolling_correlation` are commonly needed.

---

*This audit was used to drive the user guide expansion and book notebook updates for v0.1.0a11.*
