# config/ - Reusable Configuration Models

Pydantic-based configuration objects for labeling, preprocessing, experiments, and
data contracts.

## Main Files

| File | Purpose | Key Objects |
|------|---------|-------------|
| `labeling.py` | barrier and horizon labeling configs | `LabelingConfig` |
| `preprocessing_config.py` | scaler and preprocessing configs | `PreprocessingConfig` |
| `data_contract.py` | schema and column-contract mapping | `DataContractConfig` |
| `experiment.py` | experiment persistence and loading | `ExperimentConfig` |
| `spec_bridge.py` | bridge from market-data specs into contracts | `data_contract_from_market_data_spec()` |
| `base.py` | shared config primitives | `BaseConfig`, `ComputationalConfig` |

## Typical Usage

```python
from ml4t.engineer.config import LabelingConfig, PreprocessingConfig

label_cfg = LabelingConfig.triple_barrier(
    upper_barrier=0.02,
    lower_barrier=0.01,
    max_holding_period=20,
)
prep_cfg = PreprocessingConfig.robust()
```
