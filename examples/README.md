# ML4T Engineer Examples

This directory contains runnable example scripts for the main `ml4t-engineer`
workflows.

## End-to-End Workflows

### 1. **complete_workflow_example.py** - End-to-End Workflow

Shows the main workflow in one place:

- feature computation
- labeling
- preprocessing and dataset preparation
- leakage-safe train/test construction

```bash
python examples/complete_workflow_example.py
```

### 2. **fml_features_demo.py** - Financial ML Feature Demo

Focused demonstration of the finance-specific feature stack:

- information-driven bars
- fractional differencing
- triple-barrier labeling
- public API composition

## Individual Feature Examples

### 3. **bars_example.py** - Information-Driven Bars

Demonstrates different bar sampling methods:

- Tick bars, volume bars, dollar bars, imbalance bars
- Statistical comparison of bar properties
- Contract checks and sampling statistics

### 4. **fdiff_example.py** - Fractional Differencing

Shows how to achieve stationarity while preserving memory:

- Finding optimal differencing parameter
- Diagnostics for the selected transform

### 5. **labeling_example.py** - Triple-Barrier Labeling

Demonstrates advanced labeling for ML:

- Fixed and dynamic barriers
- Volatility-based barriers
- Trailing stops
- OHLC execution semantics

## API Composition Example

### 6. **pipeline_example.py** - Feature and Preprocessing APIs

Shows how to compose supported transformations:

- Configured feature computation
- Multiple technical indicators
- Train-only data preprocessing

## Data Requirements

Every example creates deterministic synthetic data and can run without local
datasets.

## Running Examples

Each example can be run directly:

```bash
# Run the full workflow
python examples/complete_workflow_example.py

# Run the financial-ML feature demo
python examples/fml_features_demo.py

# Run specific feature example
python examples/bars_example.py
```

## Next Steps

After running these examples:

1. Experiment with different parameters
2. Apply to your own data
3. Integrate into your trading strategies
4. Combine with ML models for alpha generation
