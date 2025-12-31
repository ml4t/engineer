"""Preprocessing utilities for feature standardization with train-only fitting.

This module provides sklearn-like preprocessing transformers that maintain
strict separation between training and test data statistics, preventing
lookahead bias in ML pipelines.

Key Concepts:
- Fit on training data only, transform both train and test
- Polars-native implementation for performance
- Immutable after fit (statistics locked)
- Serializable for production deployment

Example:
    >>> import polars as pl
    >>> from ml4t.engineer.preprocessing import StandardScaler

    >>> # Create train/test split
    >>> train_df = pl.DataFrame({"feature_a": [1.0, 2.0, 3.0], "feature_b": [10.0, 20.0, 30.0]})
    >>> test_df = pl.DataFrame({"feature_a": [4.0, 5.0], "feature_b": [40.0, 50.0]})

    >>> # Fit on train only, transform both
    >>> scaler = StandardScaler()
    >>> train_scaled = scaler.fit_transform(train_df)
    >>> test_scaled = scaler.transform(test_df)  # Uses train statistics

Reference:
    Based on sklearn StandardScaler API with Polars implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import polars as pl


class ScalerMethod(str, Enum):
    """Scaling method options."""

    STANDARD = "standard"  # Z-score: (x - mean) / std
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Median/IQR based (outlier resistant)


class NotFittedError(Exception):
    """Raised when transform is called before fit."""

    pass


class BaseScaler(ABC):
    """Abstract base class for all scalers.

    All scalers follow the sklearn-like API:
    - fit(X) - Compute statistics from training data
    - transform(X) - Transform using fitted statistics
    - fit_transform(X) - Fit and transform in one step
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        """Initialize scaler.

        Parameters
        ----------
        columns : list[str] | None
            Columns to scale. If None, all numeric columns are scaled.
        """
        self._columns: list[str] | None = columns
        self._fitted_columns: list[str] = []
        self._statistics: dict[str, dict[str, float]] = {}
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the scaler has been fitted."""
        return self._is_fitted

    @property
    def fitted_columns(self) -> list[str]:
        """Return list of fitted column names."""
        return self._fitted_columns.copy()

    @property
    def statistics(self) -> dict[str, dict[str, float]]:
        """Return fitted statistics per column."""
        if not self._is_fitted:
            raise NotFittedError("Scaler has not been fitted. Call fit() first.")
        return self._statistics.copy()

    def _get_numeric_columns(self, X: pl.DataFrame) -> list[str]:
        """Get numeric columns from DataFrame."""
        return [col for col in X.columns if X[col].dtype.is_numeric()]

    def _validate_columns(self, X: pl.DataFrame) -> list[str]:
        """Validate and return columns to process."""
        if self._columns is not None:
            missing = set(self._columns) - set(X.columns)
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return self._columns

        return self._get_numeric_columns(X)

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._is_fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() or fit_transform() first."
            )

    def _check_transform_columns(self, X: pl.DataFrame) -> None:
        """Verify transform DataFrame has fitted columns."""
        missing = set(self._fitted_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Transform data missing fitted columns: {missing}")

    @abstractmethod
    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute statistics from training data."""
        pass

    @abstractmethod
    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply transformation using fitted statistics."""
        pass

    def fit(self, X: pl.DataFrame) -> BaseScaler:
        """Compute statistics from training data.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.

        Returns
        -------
        self
            Fitted scaler instance.
        """
        columns = self._validate_columns(X)
        self._statistics = self._compute_statistics(X, columns)
        self._fitted_columns = columns
        self._is_fitted = True
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform data using fitted statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform.

        Returns
        -------
        pl.DataFrame
            Transformed data.
        """
        self._check_fitted()
        self._check_transform_columns(X)
        return self._apply_transform(X)

    def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        X : pl.DataFrame
            Training data.

        Returns
        -------
        pl.DataFrame
            Transformed training data.
        """
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler to dictionary.

        Returns
        -------
        dict
            Serialized scaler state.
        """
        self._check_fitted()
        return {
            "class": self.__class__.__name__,
            "columns": self._fitted_columns,
            "statistics": self._statistics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseScaler:
        """Deserialize scaler from dictionary.

        Parameters
        ----------
        data : dict
            Serialized scaler state.

        Returns
        -------
        BaseScaler
            Reconstructed scaler instance.
        """
        scaler = cls(columns=data["columns"])
        scaler._statistics = data["statistics"]
        scaler._fitted_columns = data["columns"]
        scaler._is_fitted = True
        return scaler


class StandardScaler(BaseScaler):
    """Z-score normalization: (x - mean) / std.

    Transforms features to have mean=0 and std=1 using training data statistics.

    Parameters
    ----------
    columns : list[str] | None
        Columns to scale. If None, all numeric columns are scaled.
    with_mean : bool, default True
        Center data by subtracting mean.
    with_std : bool, default True
        Scale data by dividing by std.
    ddof : int, default 1
        Delta degrees of freedom for std calculation.

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> train_scaled = scaler.fit_transform(train_df)
    >>> test_scaled = scaler.transform(test_df)  # Uses train mean/std
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        with_mean: bool = True,
        with_std: bool = True,
        ddof: int = 1,
    ) -> None:
        super().__init__(columns)
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute mean and std for each column."""
        stats = {}
        for col in columns:
            series = X[col].drop_nulls()

            # Handle empty series
            if len(series) == 0:
                mean_val = 0.0
                std_val = 1.0
            else:
                mean_raw = series.mean()
                std_raw = series.std(ddof=self.ddof) if len(series) > 1 else None

                # Convert to float, handling None/NaN
                # Note: mean/std on numeric columns return numeric types, but Polars
                # type signature includes non-numeric possibilities for mixed dtype Series
                mean_val = float(mean_raw) if mean_raw is not None else 0.0
                std_val = float(std_raw) if std_raw is not None else 1.0

                # Apply with_mean/with_std settings
                if not self.with_mean:
                    mean_val = 0.0
                if not self.with_std:
                    std_val = 1.0

            # Handle zero std (constant column) or NaN
            if std_val == 0.0 or std_val != std_val:  # NaN check
                std_val = 1.0

            stats[col] = {"mean": mean_val, "std": std_val}
        return stats

    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply z-score normalization."""
        exprs = []
        for col in self._fitted_columns:
            mean_val = self._statistics[col]["mean"]
            std_val = self._statistics[col]["std"]

            if self.with_mean and self.with_std:
                expr = ((pl.col(col) - mean_val) / std_val).alias(col)
            elif self.with_mean:
                expr = (pl.col(col) - mean_val).alias(col)
            elif self.with_std:
                expr = (pl.col(col) / std_val).alias(col)
            else:
                expr = pl.col(col)

            exprs.append(expr)

        # Keep non-fitted columns unchanged
        other_cols = [pl.col(c) for c in X.columns if c not in self._fitted_columns]

        return X.select(exprs + other_cols)


class MinMaxScaler(BaseScaler):
    """Scale features to [0, 1] range using min/max from training data.

    Parameters
    ----------
    columns : list[str] | None
        Columns to scale. If None, all numeric columns are scaled.
    feature_range : tuple[float, float], default (0.0, 1.0)
        Desired range of transformed data.

    Examples
    --------
    >>> scaler = MinMaxScaler()
    >>> train_scaled = scaler.fit_transform(train_df)  # [0, 1] range
    >>> test_scaled = scaler.transform(test_df)  # May exceed [0, 1]
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__(columns)
        self.feature_range = feature_range

    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute min and max for each column."""
        stats = {}
        for col in columns:
            series = X[col].drop_nulls()
            # Note: min/max on numeric columns return numeric types
            min_val = float(series.min())
            max_val = float(series.max())

            # Handle constant column (min == max)
            range_val = max_val - min_val
            if range_val == 0.0:
                range_val = 1.0

            stats[col] = {"min": min_val, "max": max_val, "range": range_val}
        return stats

    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply min-max scaling."""
        target_min, target_max = self.feature_range
        target_range = target_max - target_min

        exprs = []
        for col in self._fitted_columns:
            min_val = self._statistics[col]["min"]
            range_val = self._statistics[col]["range"]

            # Scale to [0, 1] then to target range
            expr = (((pl.col(col) - min_val) / range_val) * target_range + target_min).alias(col)
            exprs.append(expr)

        # Keep non-fitted columns unchanged
        other_cols = [pl.col(c) for c in X.columns if c not in self._fitted_columns]

        return X.select(exprs + other_cols)


class RobustScaler(BaseScaler):
    """Scale using median and IQR (robust to outliers).

    Uses median instead of mean, and interquartile range (IQR) instead of std.

    Parameters
    ----------
    columns : list[str] | None
        Columns to scale. If None, all numeric columns are scaled.
    with_centering : bool, default True
        Center data by subtracting median.
    with_scaling : bool, default True
        Scale data by dividing by IQR.
    quantile_range : tuple[float, float], default (25.0, 75.0)
        Quantile range for IQR calculation.

    Examples
    --------
    >>> scaler = RobustScaler()
    >>> train_scaled = scaler.fit_transform(train_df)
    >>> test_scaled = scaler.transform(test_df)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__(columns)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range

    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute median and IQR for each column."""
        q_low, q_high = self.quantile_range[0] / 100.0, self.quantile_range[1] / 100.0

        stats = {}
        for col in columns:
            series = X[col].drop_nulls()

            # Note: median/quantile on numeric columns return numeric types
            median_val = float(series.median()) if self.with_centering else 0.0
            if self.with_scaling:
                q1 = float(series.quantile(q_low))
                q3 = float(series.quantile(q_high))
                iqr_val = q3 - q1
                if iqr_val == 0.0:
                    iqr_val = 1.0
            else:
                iqr_val = 1.0

            stats[col] = {"median": median_val, "iqr": iqr_val}
        return stats

    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply robust scaling."""
        exprs = []
        for col in self._fitted_columns:
            median_val = self._statistics[col]["median"]
            iqr_val = self._statistics[col]["iqr"]

            if self.with_centering and self.with_scaling:
                expr = ((pl.col(col) - median_val) / iqr_val).alias(col)
            elif self.with_centering:
                expr = (pl.col(col) - median_val).alias(col)
            elif self.with_scaling:
                expr = (pl.col(col) / iqr_val).alias(col)
            else:
                expr = pl.col(col)

            exprs.append(expr)

        # Keep non-fitted columns unchanged
        other_cols = [pl.col(c) for c in X.columns if c not in self._fitted_columns]

        return X.select(exprs + other_cols)


# Convenience alias
Preprocessor = StandardScaler
