# ML4T Engineer Examples

This directory contains example scripts and notebooks demonstrating the Financial Machine Learning (FML) features of the ml4t-engineer library.

## Core FML Examples

### 1. **fml_features_demo.py** - Complete FML Features Demonstration

Comprehensive example showing all FML features working together:

- Information-driven bars (tick, volume, dollar, imbalance)
- Fractional differencing for stationarity
- Triple-barrier labeling for ML targets
- Pipeline integration
- Visualizations and statistics

```bash
python examples/fml_features_demo.py
```

### 2. **fml_features_notebook.py** - Interactive Jupyter Notebook

Interactive exploration of FML features with detailed explanations:

- Step-by-step walkthrough of each component
- Statistical analysis and visualizations
- Example ML model training
- Convert to notebook: `jupyter nbconvert --to notebook --execute fml_features_notebook.py`

## Individual Feature Examples

### 3. **bars_example.py** - Information-Driven Bars

Demonstrates different bar sampling methods:

- Tick bars, volume bars, dollar bars, imbalance bars
- Statistical comparison of bar properties
- Visualization of sampling differences

### 4. **fdiff_example.py** - Fractional Differencing

Shows how to achieve stationarity while preserving memory:

- Finding optimal differencing parameter
- Comparing with integer differencing
- Weight function visualization

### 5. **labeling_example.py** - Triple-Barrier Labeling

Demonstrates advanced labeling for ML:

- Fixed and dynamic barriers
- Volatility-based barriers
- Trailing stops
- Pipeline integration

### 6. **labeling_simple.py** - Simple Labeling Example

Quick introduction to triple-barrier labeling with minimal code.

## Pipeline Examples

### 7. **pipeline_example.py** - Pipeline API

Shows how to chain transformations:

- Feature engineering pipelines
- Combining technical indicators
- Data preprocessing workflows

## Technical Analysis Examples

### 8. **indicator_comparison.py** - TA Indicator Comparison

Compare ml4t-engineer indicators with TA-Lib for validation.

### 9. **performance_comparison.py** - Performance Benchmarks

Benchmark performance against TA-Lib implementations.

### 10. **multi_asset_example.py** - Multi-Asset Analysis

Demonstrate multi-asset capabilities using the Pipeline API.

## Data Requirements

Examples will use real data if available, otherwise create synthetic data:

- BTC futures: `data/crypto/futures/BTC.parquet`
- SPY tick data: `data/equities/SPY/`

## Running Examples

Each example can be run directly:

```bash
# Run complete FML demo
python examples/fml_features_demo.py

# Run specific feature example
python examples/bars_example.py

# Convert notebook to Jupyter format
jupyter nbconvert --to notebook --execute fml_features_notebook.py
```

## Output

Examples generate visualizations in the `examples/` directory:

- `fml_features_demo.png` - Overview of all FML features
- `bars_comparison.png` - Bar sampling comparison
- `fdiff_results.png` - Fractional differencing analysis
- `labeling_results.png` - Labeling visualization

## Next Steps

After running these examples:

1. Experiment with different parameters
2. Apply to your own data
3. Integrate into your trading strategies
4. Combine with ML models for alpha generation
