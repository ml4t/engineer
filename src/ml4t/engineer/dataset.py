"""ML Dataset Builder for leakage-free train/test preparation.

This module provides the MLDatasetBuilder class that orchestrates:
1. Feature/label management
2. Train-only preprocessing (scaling, imputation)
3. Integration with cross-validation splitters
4. sklearn compatibility

The key insight is that preprocessing statistics (mean, std, quantiles) must
be computed ONLY on training data, then applied to both train and test sets.
This prevents lookahead bias, a critical issue in time-series ML.

Example:
    >>> from ml4t.engineer.dataset import MLDatasetBuilder
    >>> from ml4t.diagnostic.splitters import PurgedWalkForwardCV
    >>>
    >>> # Build dataset
    >>> builder = MLDatasetBuilder(features_df, labels_series)
    >>> builder.set_scaler(StandardScaler())
    >>>
    >>> # Use with cross-validation
    >>> cv = PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)
    >>> for X_train, X_test, y_train, y_test in builder.split(cv):
    ...     model.fit(X_train, y_train)
    ...     preds = model.predict(X_test)

Reference:
    López de Prado (2018). "Advances in Financial Machine Learning", Chapter 7.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl

from ml4t.engineer.preprocessing import BaseScaler, StandardScaler

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


class SplitterProtocol(Protocol):
    """Protocol for cross-validation splitters.

    Compatible with ml4t.diagnostic.splitters.BaseSplitter and
    sklearn-style cross-validation splitters.
    """

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate train/test indices."""
        ...


@dataclass
class FoldResult:
    """Result from a single cross-validation fold.

    Attributes
    ----------
    X_train : pl.DataFrame
        Preprocessed training features.
    X_test : pl.DataFrame
        Preprocessed test features (using train statistics).
    y_train : pl.Series
        Training labels.
    y_test : pl.Series
        Test labels.
    train_indices : NDArray[np.intp]
        Original indices of training samples.
    test_indices : NDArray[np.intp]
        Original indices of test samples.
    fold_number : int
        Zero-indexed fold number.
    scaler : BaseScaler | None
        Fitted scaler used for this fold (None if no scaling).
    """

    X_train: pl.DataFrame
    X_test: pl.DataFrame
    y_train: pl.Series
    y_test: pl.Series
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    fold_number: int
    scaler: BaseScaler | None = None

    def to_numpy(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Convert to numpy arrays for sklearn compatibility.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray, NDArray]
            (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        return (
            self.X_train.to_numpy(),
            self.X_test.to_numpy(),
            self.y_train.to_numpy(),
            self.y_test.to_numpy(),
        )


@dataclass
class DatasetInfo:
    """Information about the dataset.

    Attributes
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    feature_names : list[str]
        Feature column names.
    label_name : str
        Label column name.
    has_dates : bool
        Whether dates are provided.
    """

    n_samples: int
    n_features: int
    feature_names: list[str]
    label_name: str
    has_dates: bool


@dataclass
class MLDatasetBuilder:
    """Build train/test datasets with proper leakage prevention.

    This class provides a unified interface for:
    1. Managing features and labels
    2. Applying train-only preprocessing
    3. Integrating with cross-validation splitters
    4. Converting to sklearn-compatible formats

    Parameters
    ----------
    features : pl.DataFrame
        Feature matrix with named columns.
    labels : pl.Series | pl.DataFrame
        Target variable(s). If DataFrame, first column is used.
    dates : pl.Series | None, optional
        Date/time index for time-series ordering.

    Attributes
    ----------
    features : pl.DataFrame
        Feature matrix.
    labels : pl.Series
        Target labels.
    dates : pl.Series | None
        Date/time index.
    scaler : BaseScaler | None
        Scaler to apply to features.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.dataset import MLDatasetBuilder
    >>> from ml4t.engineer.preprocessing import StandardScaler
    >>>
    >>> # Create synthetic data
    >>> features = pl.DataFrame({
    ...     "momentum": [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5],
    ...     "volatility": [0.01, 0.02, 0.015, 0.025, 0.02, 0.03, 0.028, 0.035],
    ... })
    >>> labels = pl.Series("target", [0, 1, 0, 1, 0, 1, 1, 1])
    >>>
    >>> # Build dataset with scaling
    >>> builder = MLDatasetBuilder(features, labels)
    >>> builder.set_scaler(StandardScaler())
    >>>
    >>> # Manual train/test split
    >>> X_train, X_test, y_train, y_test = builder.train_test_split(
    ...     train_size=0.75
    ... )

    Notes
    -----
    The key design principle is that ALL statistics (mean, std, quantiles, etc.)
    are computed from training data ONLY. This prevents information leakage
    from future data into predictions.
    """

    features: pl.DataFrame
    labels: pl.Series
    dates: pl.Series | None = None
    _scaler: BaseScaler | None = field(default=None, repr=False)
    _feature_columns: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validate and setup after initialization."""
        # Handle labels
        if isinstance(self.labels, pl.DataFrame):
            self.labels = self.labels.to_series(0)

        # Validate shapes match
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features and labels must have same length. "
                f"Got features={len(self.features)}, labels={len(self.labels)}"
            )

        # Validate dates if provided
        if self.dates is not None and len(self.dates) != len(self.features):
            raise ValueError(
                f"Dates must have same length as features. "
                f"Got dates={len(self.dates)}, features={len(self.features)}"
            )

        # Store feature column names
        self._feature_columns = list(self.features.columns)

    @property
    def scaler(self) -> BaseScaler | None:
        """Get the current scaler."""
        return self._scaler

    def set_scaler(self, scaler: BaseScaler | None) -> MLDatasetBuilder:
        """Set the scaler for preprocessing.

        Parameters
        ----------
        scaler : BaseScaler | None
            Scaler to use. Pass None to disable scaling.

        Returns
        -------
        self
            Returns self for method chaining.

        Examples
        --------
        >>> builder.set_scaler(StandardScaler())
        >>> builder.set_scaler(MinMaxScaler(feature_range=(-1, 1)))
        >>> builder.set_scaler(None)  # Disable scaling
        """
        self._scaler = scaler
        return self

    @property
    def info(self) -> DatasetInfo:
        """Get dataset information.

        Returns
        -------
        DatasetInfo
            Summary of dataset properties.
        """
        return DatasetInfo(
            n_samples=len(self.features),
            n_features=self.features.width,
            feature_names=self._feature_columns,
            label_name=self.labels.name or "label",
            has_dates=self.dates is not None,
        )

    def split(
        self,
        cv: SplitterProtocol,
        groups: pl.Series | None = None,
    ) -> Iterator[FoldResult]:
        """Generate train/test splits with proper preprocessing.

        Parameters
        ----------
        cv : SplitterProtocol
            Cross-validation splitter (from ml4t.diagnostic.splitters or sklearn).
        groups : pl.Series | None, optional
            Group labels for group-based splitting.

        Yields
        ------
        FoldResult
            Result object containing preprocessed train/test data.

        Examples
        --------
        >>> from ml4t.diagnostic.splitters import PurgedWalkForwardCV
        >>>
        >>> cv = PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)
        >>> for fold in builder.split(cv):
        ...     model.fit(fold.X_train, fold.y_train)
        ...     preds = model.predict(fold.X_test)
        ...     print(f"Fold {fold.fold_number}: {len(fold.train_indices)} train, "
        ...           f"{len(fold.test_indices)} test")

        Notes
        -----
        For each fold:
        1. Training indices are extracted from the splitter
        2. Scaler (if any) is fit on training data ONLY
        3. Both train and test features are transformed using train statistics
        4. Labels are sliced without transformation
        """
        # Prepare data for splitter (numpy arrays for compatibility)
        X_for_split = self.features
        y_for_split = self.labels

        # Handle groups
        groups_arr = groups.to_numpy() if groups is not None else None

        # Iterate through folds
        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X_for_split, y_for_split, groups_arr)
        ):
            # Extract train/test features and labels
            X_train_raw = self.features[train_idx]
            X_test_raw = self.features[test_idx]
            y_train = self.labels.gather(train_idx)
            y_test = self.labels.gather(test_idx)

            # Apply preprocessing (train-only fit)
            if self._scaler is not None:
                # Clone scaler for this fold (fresh fit)
                fold_scaler = type(self._scaler)(
                    columns=self._scaler._columns,
                )
                # Copy other attributes if they exist
                for attr in ["with_mean", "with_std", "ddof"]:  # StandardScaler
                    if hasattr(self._scaler, attr):
                        setattr(fold_scaler, attr, getattr(self._scaler, attr))
                for attr in ["feature_range"]:  # MinMaxScaler
                    if hasattr(self._scaler, attr):
                        setattr(fold_scaler, attr, getattr(self._scaler, attr))
                for attr in ["with_centering", "with_scaling", "quantile_range"]:  # RobustScaler
                    if hasattr(self._scaler, attr):
                        setattr(fold_scaler, attr, getattr(self._scaler, attr))

                # Fit on train, transform both
                X_train = fold_scaler.fit_transform(X_train_raw)
                X_test = fold_scaler.transform(X_test_raw)
            else:
                fold_scaler = None
                X_train = X_train_raw
                X_test = X_test_raw

            yield FoldResult(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_indices=train_idx,
                test_indices=test_idx,
                fold_number=fold_idx,
                scaler=fold_scaler,
            )

    def train_test_split(
        self,
        train_size: float = 0.8,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """Simple train/test split with preprocessing.

        Parameters
        ----------
        train_size : float, default 0.8
            Proportion of data for training (0.0 to 1.0).
        shuffle : bool, default False
            Whether to shuffle before splitting. For time-series, keep False.
        random_state : int | None, optional
            Random seed for reproducibility when shuffling.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]
            (X_train, X_test, y_train, y_test) with preprocessing applied.

        Examples
        --------
        >>> X_train, X_test, y_train, y_test = builder.train_test_split(
        ...     train_size=0.7
        ... )

        Notes
        -----
        For time-series data, set shuffle=False to preserve temporal ordering.
        The split point is based on row position, not dates.
        """
        n_samples = len(self.features)
        n_train = int(n_samples * train_size)

        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(n_samples)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
        else:
            train_idx = np.arange(n_train)
            test_idx = np.arange(n_train, n_samples)

        # Extract train/test
        X_train_raw = self.features[train_idx]
        X_test_raw = self.features[test_idx]
        y_train = self.labels.gather(train_idx)
        y_test = self.labels.gather(test_idx)

        # Apply preprocessing
        if self._scaler is not None:
            X_train = self._scaler.fit_transform(X_train_raw)
            X_test = self._scaler.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test = X_test_raw

        return X_train, X_test, y_train, y_test

    def to_numpy(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """Convert full dataset to numpy arrays.

        Returns
        -------
        tuple[NDArray, NDArray]
            (features, labels) as numpy arrays.

        Notes
        -----
        This does NOT apply scaling. Use for raw data access only.
        For sklearn compatibility with scaling, use split() or train_test_split().
        """
        return self.features.to_numpy(), self.labels.to_numpy()

    def to_pandas(self) -> tuple[pd.DataFrame, pd.Series]:
        """Convert full dataset to pandas.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (features, labels) as pandas objects.

        Notes
        -----
        This does NOT apply scaling. Use for raw data access only.
        """
        return self.features.to_pandas(), self.labels.to_pandas()

    def get_feature_percentiles(
        self,
        train_indices: NDArray[np.intp],
        quantiles: list[float] | None = None,
    ) -> dict[str, dict[float, float]]:
        """Compute percentile cutoffs from training data only.

        Parameters
        ----------
        train_indices : NDArray[np.intp]
            Indices of training samples.
        quantiles : list[float], default [0.1, 0.25, 0.5, 0.75, 0.9]
            Quantiles to compute (between 0 and 1).

        Returns
        -------
        dict[str, dict[float, float]]
            Mapping of feature name to quantile value mapping.

        Examples
        --------
        >>> train_idx = np.arange(1000)  # First 1000 samples for training
        >>> cutoffs = builder.get_feature_percentiles(train_idx)
        >>> # Use cutoffs to bin test data
        >>> momentum_q50 = cutoffs["momentum"][0.5]

        Notes
        -----
        These cutoffs should be used to bin TEST data, ensuring
        no lookahead bias. Never compute quantiles on test data.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        train_features = self.features[train_indices]

        result: dict[str, dict[float, float]] = {}
        for col in self._feature_columns:
            series = train_features[col].drop_nulls()
            # Note: quantile on numeric columns returns numeric types
            result[col] = {q: float(series.quantile(q)) for q in quantiles}

        return result

    def compute_label_percentiles(
        self,
        train_indices: NDArray[np.intp],
        n_quantiles: int = 5,
    ) -> list[float]:
        """Compute label quantile cutoffs from training data.

        Parameters
        ----------
        train_indices : NDArray[np.intp]
            Indices of training samples.
        n_quantiles : int, default 5
            Number of quantile bins.

        Returns
        -------
        list[float]
            Quantile boundaries for binning labels.

        Examples
        --------
        >>> train_idx = np.arange(1000)
        >>> cutoffs = builder.compute_label_percentiles(train_idx, n_quantiles=5)
        >>> # cutoffs = [threshold_20, threshold_40, threshold_60, threshold_80]
        >>> # Use to bin test labels into quintiles

        Notes
        -----
        Returns n_quantiles - 1 cutoff points. Use with pl.cut() or np.digitize()
        to assign quantile labels to test data.
        """
        train_labels = self.labels.gather(train_indices).drop_nulls()

        quantiles = [i / n_quantiles for i in range(1, n_quantiles)]
        # Note: quantile on numeric columns returns numeric types
        return [float(train_labels.quantile(q)) for q in quantiles]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.features)

    def __repr__(self) -> str:
        """Return string representation."""
        info = self.info
        scaler_str = type(self._scaler).__name__ if self._scaler else "None"
        return (
            f"MLDatasetBuilder("
            f"n_samples={info.n_samples}, "
            f"n_features={info.n_features}, "
            f"scaler={scaler_str})"
        )


def create_dataset_builder(
    features: pl.DataFrame,
    labels: pl.Series | pl.DataFrame,
    dates: pl.Series | None = None,
    scaler: BaseScaler | str | None = "standard",
) -> MLDatasetBuilder:
    """Convenience function to create MLDatasetBuilder with common defaults.

    Parameters
    ----------
    features : pl.DataFrame
        Feature matrix.
    labels : pl.Series | pl.DataFrame
        Target variable.
    dates : pl.Series | None, optional
        Date/time index.
    scaler : BaseScaler | str | None, default "standard"
        Scaler to use. Options:
        - "standard": StandardScaler (z-score)
        - "minmax": MinMaxScaler ([0, 1])
        - "robust": RobustScaler (median/IQR)
        - BaseScaler instance: Use provided scaler
        - None: No scaling

    Returns
    -------
    MLDatasetBuilder
        Configured dataset builder.

    Examples
    --------
    >>> builder = create_dataset_builder(features, labels, scaler="robust")
    """
    from ml4t.engineer.preprocessing import MinMaxScaler, RobustScaler

    # Convert DataFrame labels to Series if needed (MLDatasetBuilder.__post_init__ also handles this)
    labels_series = labels.to_series(0) if isinstance(labels, pl.DataFrame) else labels
    builder = MLDatasetBuilder(features=features, labels=labels_series, dates=dates)

    if scaler is None:
        pass
    elif isinstance(scaler, str):
        scaler_map = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if scaler.lower() not in scaler_map:
            raise ValueError(
                f"Unknown scaler: {scaler}. Options: {list(scaler_map.keys())} or None"
            )
        builder.set_scaler(scaler_map[scaler.lower()]())
    elif isinstance(scaler, BaseScaler):
        builder.set_scaler(scaler)
    else:
        raise TypeError(f"scaler must be str, BaseScaler, or None. Got {type(scaler)}")

    return builder
