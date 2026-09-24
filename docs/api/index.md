# API Reference

Use this page when you already know the workflow and need exact objects, method
signatures, and module locations. If you are still deciding which workflow to use,
start with the [User Guide](../index.md) or the [Book Guide](../book-guide/index.md).

## Core Functions

::: ml4t.engineer.api
    options:
      show_root_heading: true
      members:
        - compute_features

## Feature Discovery

::: ml4t.engineer.discovery.catalog
    options:
      show_root_heading: true
      members:
        - FeatureCatalog

## Labeling

::: ml4t.engineer.labeling
    options:
      show_root_heading: true

## Dataset Builder

::: ml4t.engineer.dataset.MLDatasetBuilder
    options:
      show_root_heading: true
      signature_crossrefs: false
      members:
        - set_scaler
        - info
        - split
        - train_test_split
        - to_numpy
        - to_pandas

::: ml4t.engineer.dataset.create_dataset_builder
    options:
      show_root_heading: true
      signature_crossrefs: false

::: ml4t.engineer.dataset.FoldResult
    options:
      show_root_heading: true
      signature_crossrefs: false

::: ml4t.engineer.dataset.DatasetInfo
    options:
      show_root_heading: true
      signature_crossrefs: false

## Preprocessing

::: ml4t.engineer.preprocessing
    options:
      show_root_heading: true
      signature_crossrefs: false
      members:
        - StandardScaler
        - MinMaxScaler
        - RobustScaler
        - PreprocessingPipeline
        - TransformType
        - NotFittedError

## Configuration

::: ml4t.engineer.config
    options:
      show_root_heading: true
      signature_crossrefs: false
      members:
        - LabelingConfig
        - DataContractConfig
        - PreprocessingConfig
        - ExperimentConfig

## Alternative Bars

::: ml4t.engineer.bars
    options:
      show_root_heading: true

## Fractional Differencing

::: ml4t.engineer.features.fdiff
    options:
      show_root_heading: true
      signature_crossrefs: false
      members:
        - get_ffd_weights
        - ffdiff
        - find_optimal_d
        - fdiff_diagnostics

## Next Steps

- Read [Features](../user-guide/features.md) for the main computation workflow.
- Read [Labeling](../user-guide/labeling.md) for supervised target construction.
- Read [Alternative Bars](../user-guide/bars.md) for information-driven sampling.
- Use the [Book Guide](../book-guide/index.md) to map these APIs back to the book and
  case studies.
