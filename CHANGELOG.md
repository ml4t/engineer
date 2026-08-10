# Changelog

All notable changes to ml4t-engineer are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`momentum.rsi` no longer saturates at 100 after a null in the input.** Wilder's
  average is recursive and had no notion of a gap, and the kernel was compiled with
  `fastmath=True`, which licenses the compiler to assume no NaN. One null returned
  exactly `100.0` for the rest of the series. The input is now split on NaN and each
  gap-free run is seeded on its own, so a gap costs the missing observation plus
  `period` warmup rows and the oscillator then resumes.

  Two published datasets carried this in their output. A crypto perpetual-funding
  premium index with 537 nulls emitted `100.0` on 79.0% of 100,812 rows against a
  hand-computed median of 49.68. A CME crude-oil series carried a **single** null,
  on 2020-04-20, the day the contract settled negative and the caller's own
  non-positive-price gate nulled it - 12.6% of that product's `rsi_14`, every one of
  them after the null, running to the end of the series five years later.

- **A missing observation no longer produces a silently wrong value anywhere in the
  feature surface.** `fastmath=True` was the general mechanism, not an RSI detail.
  Feeding one null into every momentum, volatility and volume feature, 20 of 91
  feature-column pairs returned a finite number that differed from the correct one,
  by up to 91.9 for `mfi` and 1.6e4 for the A/D line. `fastmath` is removed from the
  feature kernels; `volume.obv` no longer freezes its level forever when both
  comparisons against a NaN previous close are false; and `volume.ad`'s Polars path
  no longer uses `cum_sum`, which skips nulls and carries the level on as if the bar
  had been observed while the Numba path propagates. Every feature now either
  recovers from a gap or returns NaN.

  Measured cost over 5M rows: `rsi` 42.7 -> 53.8 ms, `atr` 51.4 -> 90.3 ms, `adx`
  unchanged. No TA-Lib exactness result changed.

- **`microstructure.amihud_illiquidity` averages over the periods that traded.**
  Amihud (2002) divides by the number of days the asset traded; nulling untraded bars
  and taking `rolling_mean(period)` propagates the null across the following window,
  which instead required every bar in the window to have traded. On a NASDAQ-100
  minute panel, 0.64% untraded bars produced a 9.08% null share.

- **A rolling kernel is compiled before its window runs, not inside it.**
  `polars.Expr.rolling_map` does not advance its window while the callback is being
  compiled, so the first evaluation of a feature whose callback calls a lazily-jitted
  kernel returned the first window's value for the entire series, near-constant, with
  nothing raised. `regime.hurst_exponent`, `statistics.coefficient_of_variation`,
  `statistics.rolling_cv_zscore`, `statistics.rolling_drift`, `ml.rolling_entropy_lz`
  and `ml.rolling_entropy_plugin` each returned two distinct values on their first
  evaluation in a process and the full set on every later one. Affects any fresh
  environment - a new virtualenv, a CI job, a first run - and not a repeat run, which
  loads the kernel already compiled.

## [0.1.0b9] - 2026-06-21

### Added
- `FixedTickRunBarSampler` for stable, fixed-threshold tick run bars

### Fixed
- Public package metadata links now point to the `ml4t/engineer` repository and
  published documentation
- Added the missing MIT license file to the source distribution
- Synced lockfile metadata with relaxed dependency bounds
- Reduced pytest warning noise for performance markers and cyclical feature
  metadata

## [0.1.0b8] - 2026-05-05

### Fixed
- Dropped the Python 3.14-specific dependency split pins for `pandas`,
  `pyarrow`, `scipy`, `scikit-learn`, `statsmodels`, and `numba`, keeping the
  source-level minimums as unconditional requirements so resolvers can choose
  the newest compatible builds for each platform

## [0.1.0b7] - 2026-05-05

### Fixed
- Lowered the Python 3.14 `pandas` floor from `>=3.0.0` to `>=2.3.3`, which
  already publishes `cp314` wheels and avoids unnecessary dependency conflicts
  downstream

## [0.1.0b6] - 2026-05-05

### Fixed
- Lowered the Python 3.14 `scikit-learn` floor from `>=1.8.0` to `>=1.7.2`,
  which already publishes `cp314` wheels and avoids unnecessary dependency
  conflicts downstream

## [0.1.0b5] - 2026-05-05

### Fixed
- Restored actual Python 3.14 installability by routing compiled scientific
  dependencies onto release lines that publish `cp314` wheels

## [0.1.0b4] - 2026-04-02

### Changed
- Vendored the shared ML4T docs theme into `docs/overrides/` using MkDocs
  Material `custom_dir`, removing the external theme-package requirement
- Public `engineer` workflows now resolve `ml4t-specs` from PyPI instead of a
  local sibling checkout or editable path override

### Fixed
- GitHub Docs workflow now builds and deploys from the public repo without
  depending on private adjacent repositories

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

[Unreleased]: https://github.com/ml4t/engineer/compare/v0.1.0b9...HEAD
[0.1.0b9]: https://github.com/ml4t/engineer/compare/v0.1.0b8...v0.1.0b9
[0.1.0b8]: https://github.com/ml4t/engineer/compare/v0.1.0b7...v0.1.0b8
[0.1.0b7]: https://github.com/ml4t/engineer/compare/v0.1.0b6...v0.1.0b7
[0.1.0b6]: https://github.com/ml4t/engineer/compare/v0.1.0b5...v0.1.0b6
[0.1.0b5]: https://github.com/ml4t/engineer/compare/v0.1.0b4...v0.1.0b5
[0.1.0b4]: https://github.com/stefan-jansen/ml4t-engineer/compare/v0.1.0b3...v0.1.0b4
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
