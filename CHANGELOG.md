# Changelog

All notable changes to ml4t-engineer are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0b3] - 2026-04-01

### Added
- Public documentation overhaul with a workflow-first landing page, Book Guide,
  and tighter API/user-guide routing
- Expanded `AGENTS.md` coverage across the package, including secondary modules
  and wheel artifacts

### Changed
- Shared feed and artifact schema definitions now come from `ml4t-specs`
- `data_contract_from_market_data_spec()` now normalizes inputs through
  `ml4t.specs.FeedSpec`
- Artifact schema defaults align with shared spec naming:
  `asset`, `label_value`, and `prediction_value`

### Removed
- `DataContractConfig.from_ml4t_data()` in favor of shared spec-driven contracts

## [0.1.0b2] - 2026-03-23

### Added
- Engineered artifact spec dataclasses for feature, label, and prediction outputs

### Changed
- Shared market data specs can now be bridged directly into engineer config

## [0.1.0b1] - 2026-03-03

### Removed
- Dead modules: `selection/`, `validation/`, `visualization/`, `pipeline/`
- Diagnostic config classes (`feature_config.py`) — moved to ml4t-diagnostic
- Deprecation machinery (`core/deprecation.py`, deprecated params in bar samplers and `mom()`)
- Backward-compatibility shims in labeling module
- `[tool.mypy]` config (migrated to ty)

### Added
- Comprehensive volatility tests (58 new tests covering all 11 non-TA-Lib estimators)
- `perf` pytest marker — performance benchmarks excluded from default runs, available via `pytest -m perf`

### Changed
- TA-Lib moved from dev dependency group to `[ta]` optional extra (fixes CI for lint/typecheck jobs)
- `mom()` parameter renamed: `timeperiod` → `period` (consistency with other indicators)
- Bar sampler constructors: removed `initial_expectation` / `initial_run_expectation` params

## [0.1.0a11] - 2026-03-03

### Changed
- API hardening and correctness fixes for beta preparation
- Labeling leakage gap closed: data sorted chronologically before all label computations
- Public API aligned with documentation

## [0.1.0a10] - 2026-02-28

### Fixed
- Labeling leakage gap: ensured chronological sorting in all labeling functions
- Public API documentation alignment

## [0.1.0a9] - 2026-02-28

### Fixed
- `__version__` sourced from generated version metadata instead of hardcoded string

## [0.1.0a8] - 2026-02-27

### Added
- GitHub Actions CI workflow (lint, typecheck, test matrix, build)
- Release workflow with OIDC trusted publishing
- Ecosystem diagrams in README

### Changed
- Removed outcome module (migrated to ml4t-diagnostic)
- Feature count: 120 features across 10 categories
- Standardized labeling API on `LabelingConfig`-first pattern

### Fixed
- Normalized metadata for 4 features (33 → 37 normalized)
- ty type checking rules and CI configuration
- Numba cleanup crash workaround for Python 3.13

## [0.1.0a7] - 2026-01-20

### Added
- Time-based duration strings for labeling horizons (`"1h"`, `"4h"`, `"1d"`)
- `fixed_time_horizon_labels()` accepts `horizon="1h"`
- `triple_barrier_labels()` accepts `max_holding_period="1h"`
- `rolling_percentile_binary_labels()` accepts time-based horizon/lookback
- 51 new tests for time-based horizons

### Fixed
- Chronological sorting in `triple_barrier_labels`, `trend_scanning_labels`,
  `fixed_time_horizon_labels`, and `rolling_percentile_binary_labels`
- dtype-based timestamp detection (replaces name matching)

## [0.1.0a6] - 2026-01-18

### Added
- Validation infrastructure with AFML and mlfinpy reference tests
- 86 validation tests (AFML formulas + mlfinpy comparison)
- Triple barrier, meta-labeling, sample weights validated at 1e-10 tolerance

### Fixed
- Triple barrier edge cases
- Multiple drift detection bugs
- Tuple syntax for isinstance type checks

## [0.1.0a5] - 2026-01-14

### Added
- `get_agent_docs()` for AI agent discoverability
- Hierarchical AGENTS.md navigation files
- AGENTS.md files included in wheel builds

### Fixed
- `variance_ratio` Int64 bug

## [0.1.0a4] - 2026-01-08

### Fixed
- Synced missing modules from development workspace

## [0.1.0a3] - 2026-01-04

Initial public alpha release.

### Added
- 120 feature functions across 10 categories (momentum, trend, volatility,
  volume, microstructure, ML, risk, cycle, pattern, statistics)
- 60 indicators validated against TA-Lib at 1e-6 tolerance
- Triple-barrier labeling system (De Prado AFML)
- ATR-adjusted barriers, fixed horizon, trend scanning, percentile labels
- Meta-labeling and sample uniqueness (sequential bootstrap)
- Alternative bar types: volume, dollar, tick, imbalance, run bars
- Polars-native with Numba JIT compilation
- `compute_features()` pipeline with dependency resolution
- `FeatureCatalog` for feature discovery and metadata
- `LabelingConfig` with Pydantic v2 serialization
- `MLDatasetBuilder` for dataset construction
- `PreprocessingPipeline` for feature transformation

[Unreleased]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0b3...HEAD
[0.1.0b3]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0b2...v0.1.0b3
[0.1.0b2]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a11...v0.1.0b1
[0.1.0a11]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a10...v0.1.0a11
[0.1.0a10]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a9...v0.1.0a10
[0.1.0a9]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a8...v0.1.0a9
[0.1.0a8]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a7...v0.1.0a8
[0.1.0a7]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/stefan-jansen/ml4t-engineer/releases/tag/v0.1.0a3
