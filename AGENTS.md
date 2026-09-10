# ml4t-engineer

`ml4t-engineer` provides Polars-first feature engineering, labeling, alternative
bars, and leakage-safe dataset preparation for financial machine learning.

## Source orientation

Runtime code lives under `src/ml4t/engineer/`.

- `api.py`, `features/`, and `discovery/` implement registry-backed feature
  computation and discovery.
- `labeling/` implements path-dependent, fixed-horizon, percentile, meta-label,
  and sample-weighting workflows.
- `bars/` converts trade data into tick, volume, dollar, imbalance, and run bars.
- `dataset.py` and `preprocessing.py` implement train/test preparation and
  train-only transformations.
- `core/` and `config/` contain registry metadata, validation, schemas, and
  reusable configuration models.
- `artifacts/`, `relationships/`, `store/`, `logging/`, and `utils/` provide
  supporting services.

Tests are under `tests/`. Executable examples and repository checks are under
`examples/` and `scripts/`.

## Public workflows

Stable entry points include:

- `ml4t.engineer.compute_features`
- `ml4t.engineer.feature_catalog`
- `ml4t.engineer.create_dataset_builder`
- labeling functions under `ml4t.engineer.labeling`
- bar samplers under `ml4t.engineer.bars`

Start with `README.md` and `docs/getting-started/`. Detailed workflow guidance
is under `docs/user-guide/`, and the generated API reference is under
`docs/api/`.

## Engineering constraints

- Python 3.12, 3.13, and 3.14 are supported.
- Feature computation must preserve the documented DataFrame or LazyFrame
  return behavior.
- Changes to registered features must keep implementation, registry metadata,
  dependencies, normalization metadata, and lookback behavior consistent.
- Fit preprocessing state only on training data.
- Keep optional dependencies isolated from the base import.
- Shared data-contract changes originate in `ml4t-specs`.

## Quality commands

Run from the repository root:

```bash
uv sync --dev --extra docs --extra ta --extra store --extra viz
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q
uv build
uv run python -c "import ml4t.engineer"
uv run mkdocs build --strict
```
