# ml4t.engineer package

These instructions apply within `src/ml4t/engineer/`. Use the repository-root
`AGENTS.md` for setup and quality commands.

## Internal structure

- `core/registry.py` and `core/decorators.py` own feature registration and
  metadata.
- `api.py` resolves registered features, their dependencies, and execution
  order.
- `features/` contains registered indicators and standalone transforms.
- `labeling/`, `bars/`, `dataset.py`, and `preprocessing.py` implement the
  primary downstream workflows.
- `config/` bridges reusable configuration and shared `ml4t-specs` contracts.

## Local rules

- A registered feature change must keep its decorator metadata and first usable
  row consistent with its implementation.
- Preserve DataFrame and LazyFrame behavior where the public workflow supports
  both.
- Label calculations must not introduce future information before the declared
  horizon.
- Preprocessing state is fitted on training observations only.
- Guard optional integrations through the existing dependency helpers so
  `import ml4t.engineer` works with base dependencies.

Use the scoped guides in `features/AGENTS.md`, `labeling/AGENTS.md`, and
`bars/AGENTS.md` when working in those subsystems.
