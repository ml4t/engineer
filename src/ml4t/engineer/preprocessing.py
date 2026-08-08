"""Preprocessing utilities for feature standardization with train-only fitting.

This module provides sklearn-like preprocessing transformers that maintain
strict separation between training and test data statistics, preventing
lookahead bias in ML pipelines.

Exports:
    StandardScaler - Z-score normalization (mean=0, std=1)
    MinMaxScaler - Scale to [0, 1] range
    RobustScaler - IQR-based scaling (outlier resistant)
    PreprocessingPipeline - Chain multiple transformers

    ScalerMethod - Enum: STANDARD, MINMAX, ROBUST
    TransformType - Enum: SCALE, CLIP, WINSORIZE

Key Concepts:
- Fit on training data only, transform both train and test
- Polars-native implementation for performance
- Immutable after fit (statistics locked)
- Serializable for production deployment

Example:
    >>> from ml4t.engineer.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> train_scaled = scaler.fit_transform(train_df)
    >>> test_scaled = scaler.transform(test_df)  # Uses train statistics
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from decimal import Decimal
from enum import StrEnum
from numbers import Real
from typing import Any

import polars as pl


class ScalerMethod(StrEnum):
    """Scaling method options."""

    STANDARD = "standard"  # Z-score: (x - mean) / std
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Median/IQR based (outlier resistant)


class NotFittedError(Exception):
    """Raised when transform is called before fit."""

    pass


_SCALER_SCHEMA_VERSION = 1


def _validate_columns_config(columns: list[str] | None) -> list[str] | None:
    if columns is None:
        return None
    if not isinstance(columns, list) or not all(
        isinstance(column, str) and column for column in columns
    ):
        raise ValueError("columns must be None or a list of non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError("columns must not contain duplicates")
    return columns.copy()


def _validate_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return result


def _numeric_statistic(
    value: object,
    *,
    context: str,
    default: float | None = None,
) -> float:
    """Convert a numeric Polars aggregation result to float."""
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, Real | Decimal):
        raise ValueError(f"{context} requires at least one numeric value")
    return float(value)


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
        self._columns = _validate_columns_config(columns)
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

    @staticmethod
    def _validate_finite(X: pl.DataFrame, columns: list[str]) -> None:
        invalid = []
        for column in columns:
            values = X[column].drop_nulls()
            if values.is_nan().any() or values.is_infinite().any():
                invalid.append(column)
        if invalid:
            raise ValueError(f"Fitting data contains NaN or infinity in columns: {invalid}")

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
        self._validate_finite(X, columns)
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
        self._validate_finite(X, self._fitted_columns)
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

    def clone(self) -> BaseScaler:
        """Create an unfitted copy of this scaler with the same parameters.

        Returns
        -------
        BaseScaler
            New scaler instance with the same configuration but no fitted state.
        """
        import copy

        new = copy.copy(self)
        new._fitted_columns = []
        new._statistics = {}
        new._is_fitted = False
        return new

    def _get_config(self) -> dict[str, Any]:
        return {"columns": copy.deepcopy(self._columns)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler to dictionary.

        Returns
        -------
        dict
            Serialized scaler state.
        """
        self._check_fitted()
        return {
            "schema_version": _SCALER_SCHEMA_VERSION,
            "class": self.__class__.__name__,
            "config": self._get_config(),
            "columns": copy.deepcopy(self._fitted_columns),
            "statistics": copy.deepcopy(self._statistics),
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
        if not isinstance(data, dict):
            raise ValueError("Serialized scaler must be a dictionary")

        required = {"schema_version", "class", "config", "columns", "statistics"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Serialized scaler is missing fields: {sorted(missing)}")
        if data["schema_version"] != _SCALER_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported scaler schema version: {data['schema_version']!r}; "
                f"expected {_SCALER_SCHEMA_VERSION}"
            )
        if data["class"] != cls.__name__:
            raise ValueError(
                f"Serialized scaler class is {data['class']!r}, expected {cls.__name__!r}"
            )
        if not isinstance(data["config"], dict):
            raise ValueError("Serialized scaler config must be a dictionary")

        fitted_columns = _validate_columns_config(data["columns"])
        if fitted_columns is None:
            raise ValueError("Serialized fitted columns must be a list")
        statistics = data["statistics"]
        if not isinstance(statistics, dict) or set(statistics) != set(fitted_columns):
            raise ValueError("Serialized statistics must match the fitted columns")
        if not all(isinstance(value, dict) for value in statistics.values()):
            raise ValueError("Serialized column statistics must be dictionaries")

        try:
            scaler = cls(**copy.deepcopy(data["config"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid serialized scaler config: {exc}") from exc
        scaler._statistics = copy.deepcopy(statistics)
        scaler._fitted_columns = fitted_columns
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
        if not isinstance(with_mean, bool):
            raise ValueError("with_mean must be a boolean")
        if not isinstance(with_std, bool):
            raise ValueError("with_std must be a boolean")
        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer")
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof

    def _get_config(self) -> dict[str, Any]:
        return {
            **super()._get_config(),
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "ddof": self.ddof,
        }

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

                mean_val = _numeric_statistic(mean_raw, context=f"mean for '{col}'", default=0.0)
                std_val = _numeric_statistic(std_raw, context=f"std for '{col}'", default=1.0)

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
        fitted_set = set(self._fitted_columns)
        exprs = []
        for col in X.columns:
            if col in fitted_set:
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
            else:
                expr = pl.col(col)

            exprs.append(expr)

        return X.select(exprs)


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
        if not isinstance(feature_range, list | tuple) or len(feature_range) != 2:
            raise ValueError("feature_range must contain exactly two finite numbers")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in feature_range):
            raise ValueError("feature_range must contain exactly two finite numbers")
        validated_range = (float(feature_range[0]), float(feature_range[1]))
        if not all(math.isfinite(value) for value in validated_range):
            raise ValueError("feature_range must contain exactly two finite numbers")
        if validated_range[0] >= validated_range[1]:
            raise ValueError("feature_range minimum must be less than maximum")
        self.feature_range = validated_range

    def _get_config(self) -> dict[str, Any]:
        return {
            **super()._get_config(),
            "feature_range": list(self.feature_range),
        }

    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute min and max for each column."""
        stats = {}
        for col in columns:
            series = X[col].drop_nulls()

            # Handle empty series (all nulls)
            if len(series) == 0:
                min_val = 0.0
                max_val = 0.0
            else:
                min_val = _numeric_statistic(series.min(), context=f"minimum for '{col}'")
                max_val = _numeric_statistic(series.max(), context=f"maximum for '{col}'")

            # Handle constant column (min == max) or empty
            range_val = max_val - min_val
            if range_val == 0.0:
                range_val = 1.0

            stats[col] = {"min": min_val, "max": max_val, "range": range_val}
        return stats

    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply min-max scaling."""
        target_min, target_max = self.feature_range
        target_range = target_max - target_min
        fitted_set = set(self._fitted_columns)

        exprs = []
        for col in X.columns:
            if col in fitted_set:
                min_val = self._statistics[col]["min"]
                range_val = self._statistics[col]["range"]
                expr = (((pl.col(col) - min_val) / range_val) * target_range + target_min).alias(
                    col
                )
            else:
                expr = pl.col(col)
            exprs.append(expr)

        return X.select(exprs)


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
        if not isinstance(with_centering, bool):
            raise ValueError("with_centering must be a boolean")
        if not isinstance(with_scaling, bool):
            raise ValueError("with_scaling must be a boolean")
        if not isinstance(quantile_range, list | tuple) or len(quantile_range) != 2:
            raise ValueError("quantile_range must contain exactly two finite percentages")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in quantile_range):
            raise ValueError("quantile_range must contain exactly two finite percentages")
        validated_range = (float(quantile_range[0]), float(quantile_range[1]))
        if not all(math.isfinite(value) for value in validated_range):
            raise ValueError("quantile_range must contain exactly two finite percentages")
        if not 0.0 <= validated_range[0] < validated_range[1] <= 100.0:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 100")
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = validated_range

    def _get_config(self) -> dict[str, Any]:
        return {
            **super()._get_config(),
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "quantile_range": list(self.quantile_range),
        }

    def _compute_statistics(
        self, X: pl.DataFrame, columns: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute median and IQR for each column."""
        q_low, q_high = self.quantile_range[0] / 100.0, self.quantile_range[1] / 100.0

        stats = {}
        for col in columns:
            series = X[col].drop_nulls()

            # Handle empty series (all nulls)
            if len(series) == 0:
                median_val = 0.0
                iqr_val = 1.0
            else:
                median_val = (
                    _numeric_statistic(series.median(), context=f"median for '{col}'")
                    if self.with_centering
                    else 0.0
                )
                if self.with_scaling:
                    q1 = _numeric_statistic(
                        series.quantile(q_low), context=f"lower quantile for '{col}'"
                    )
                    q3 = _numeric_statistic(
                        series.quantile(q_high), context=f"upper quantile for '{col}'"
                    )
                    iqr_val = q3 - q1
                    if iqr_val == 0.0:
                        iqr_val = 1.0
                else:
                    iqr_val = 1.0

            stats[col] = {"median": median_val, "iqr": iqr_val}
        return stats

    def _apply_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply robust scaling."""
        fitted_set = set(self._fitted_columns)
        exprs = []
        for col in X.columns:
            if col in fitted_set:
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
            else:
                expr = pl.col(col)

            exprs.append(expr)

        return X.select(exprs)


# =============================================================================
# PreprocessingPipeline - Bidirectional Integration with ML4T Diagnostic
# =============================================================================


class TransformType(StrEnum):
    """Transform types supported by PreprocessingPipeline.

    These align with ml4t.diagnostic.integration.engineer_contract.TransformType.
    """

    NONE = "none"  # No transformation
    LOG = "log"  # Natural log (for right-skewed data)
    SQRT = "sqrt"  # Square root (milder than log)
    STANDARDIZE = "standardize"  # Z-score normalization
    NORMALIZE = "normalize"  # Min-max to [0, 1]
    WINSORIZE = "winsorize"  # Cap outliers at percentiles
    DIFF = "diff"  # First difference (for non-stationary)


class PreprocessingPipeline:
    """Apply preprocessing recommendations from ML4T Diagnostic.

    This class enables bidirectional integration between ML4T Diagnostic and
    ML4T Engineer. After diagnostic evaluates features, it can recommend
    transforms which this pipeline applies with proper train/test separation.

    The pipeline follows sklearn conventions:
    - fit(X): Learn statistics from training data only
    - transform(X): Apply transforms using fitted statistics
    - fit_transform(X): Combined fit and transform

    Parameters
    ----------
    recommendations : dict | None
        Feature recommendations from FeatureEvaluatorConfig (ml4t-diagnostic).
        Format: {"feature_name": {"transform": "standardize", "confidence": 0.9}}
    min_confidence : float, default 0.0
        Minimum confidence threshold for applying recommendations.
        Recommendations below this threshold default to NONE.
    winsorize_limits : tuple[float, float], default (0.01, 0.99)
        Percentile limits for winsorization.

    Examples
    --------
    >>> # From ML4T Diagnostic recommendations
    >>> recommendations = {
    ...     "rsi_14": {"transform": "standardize", "confidence": 0.9},
    ...     "returns": {"transform": "winsorize", "confidence": 0.85},
    ...     "volume": {"transform": "log", "confidence": 0.8}
    ... }
    >>> pipeline = PreprocessingPipeline.from_recommendations(recommendations)
    >>> train_transformed = pipeline.fit_transform(train_df)
    >>> test_transformed = pipeline.transform(test_df)

    >>> # Serialize for production
    >>> pipeline_dict = pipeline.to_dict()
    >>> # ... save to disk ...
    >>> loaded_pipeline = PreprocessingPipeline.from_dict(pipeline_dict)
    """

    def __init__(
        self,
        recommendations: dict[str, dict[str, Any]] | None = None,
        min_confidence: float = 0.0,
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        """Initialize pipeline with recommendations."""
        self._recommendations = self._validate_recommendations(recommendations)
        self._min_confidence = _validate_probability(
            min_confidence,
            name="min_confidence",
        )
        if not isinstance(winsorize_limits, list | tuple) or len(winsorize_limits) != 2:
            raise ValueError("winsorize_limits must contain exactly two probabilities")
        lower = _validate_probability(winsorize_limits[0], name="winsorize_limits[0]")
        upper = _validate_probability(winsorize_limits[1], name="winsorize_limits[1]")
        if lower >= upper:
            raise ValueError("winsorize_limits must satisfy lower < upper")
        self._winsorize_limits = (lower, upper)
        self._is_fitted = False
        self._statistics: dict[str, dict[str, Any]] = {}
        self._fitted_features: list[str] = []

    @staticmethod
    def _validate_recommendations(
        recommendations: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if recommendations is None:
            return {}
        if not isinstance(recommendations, dict):
            raise ValueError("recommendations must be a dictionary")

        validated: dict[str, dict[str, Any]] = {}
        accepted = ", ".join(transform.value for transform in TransformType)
        for feature, recommendation in recommendations.items():
            if not isinstance(feature, str) or not feature:
                raise ValueError("recommendation feature names must be non-empty strings")
            if not isinstance(recommendation, dict):
                raise ValueError(f"Recommendation for '{feature}' must be a dictionary")

            missing = {"transform", "confidence"} - set(recommendation)
            if missing:
                raise ValueError(
                    f"Recommendation for '{feature}' is missing fields: {sorted(missing)}"
                )
            transform_value = recommendation["transform"]
            try:
                transform = TransformType(transform_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unknown transform {transform_value!r} for feature '{feature}'. "
                    f"Accepted values: {accepted}"
                ) from exc

            confidence = _validate_probability(
                recommendation["confidence"],
                name=f"confidence for feature '{feature}'",
            )
            validated[feature] = copy.deepcopy(recommendation)
            validated[feature]["transform"] = transform.value
            validated[feature]["confidence"] = confidence
        return validated

    @classmethod
    def from_recommendations(
        cls,
        recommendations: dict[str, dict[str, Any]],
        min_confidence: float = 0.0,
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
    ) -> PreprocessingPipeline:
        """Create pipeline from diagnostic recommendations.

        Parameters
        ----------
        recommendations : dict
            Output from FeatureEvaluatorConfig (ml4t-diagnostic) or similar format.
            Expected structure: {"feature": {"transform": "...", "confidence": ...}}
        min_confidence : float, default 0.0
            Minimum confidence threshold.
        winsorize_limits : tuple, default (0.01, 0.99)
            Percentile limits for winsorization.

        Returns
        -------
        PreprocessingPipeline
            Configured pipeline ready for fitting.
        """
        return cls(
            recommendations=recommendations,
            min_confidence=min_confidence,
            winsorize_limits=winsorize_limits,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether pipeline has been fitted."""
        return self._is_fitted

    def _get_transform_type(self, feature: str) -> TransformType:
        """Get transform type for a feature, respecting confidence threshold."""
        if feature not in self._recommendations:
            return TransformType.NONE

        rec = self._recommendations[feature]
        confidence = rec["confidence"]

        if confidence < self._min_confidence:
            return TransformType.NONE

        return TransformType(rec["transform"])

    def _compute_statistics(
        self, X: pl.DataFrame, feature: str, transform: TransformType
    ) -> dict[str, Any]:
        """Compute statistics needed for transform."""
        series = X[feature].drop_nulls()

        if transform == TransformType.STANDARDIZE:
            mean_val = _numeric_statistic(
                series.mean(), context=f"mean for '{feature}'", default=0.0
            )
            std_val = _numeric_statistic(series.std(), context=f"std for '{feature}'", default=1.0)
            if std_val == 0.0:
                std_val = 1.0
            return {"mean": mean_val, "std": std_val}

        elif transform == TransformType.NORMALIZE:
            min_val = _numeric_statistic(series.min(), context=f"minimum for '{feature}'")
            max_val = _numeric_statistic(series.max(), context=f"maximum for '{feature}'")
            range_val = max_val - min_val
            if range_val == 0.0:
                range_val = 1.0
            return {"min": min_val, "max": max_val, "range": range_val}

        elif transform == TransformType.WINSORIZE:
            q_low, q_high = self._winsorize_limits
            lower = _numeric_statistic(
                series.quantile(q_low), context=f"lower quantile for '{feature}'"
            )
            upper = _numeric_statistic(
                series.quantile(q_high), context=f"upper quantile for '{feature}'"
            )
            return {"lower": lower, "upper": upper}

        elif transform == TransformType.LOG:
            # For log, we need to handle non-positive values
            min_val = _numeric_statistic(series.min(), context=f"minimum for '{feature}'")
            # Offset to ensure positive values
            offset = max(0.0, -min_val + 1e-10)
            return {"offset": offset}

        elif transform == TransformType.DIFF:
            # Store last value for potential inverse
            last_val = float(series.tail(1).item())
            return {"last_value": last_val}

        return {}

    def _apply_transform(
        self,
        _X: pl.DataFrame,
        feature: str,
        transform: TransformType,
        *,
        use_training_boundary: bool,
    ) -> pl.Expr:
        """Apply transform to a feature column."""
        col = pl.col(feature)
        stats = self._statistics.get(feature, {})

        if transform == TransformType.NONE:
            return col.alias(feature)

        elif transform == TransformType.STANDARDIZE:
            mean_val = stats["mean"]
            std_val = stats["std"]
            return ((col - mean_val) / std_val).alias(feature)

        elif transform == TransformType.NORMALIZE:
            min_val = stats["min"]
            range_val = stats["range"]
            return ((col - min_val) / range_val).alias(feature)

        elif transform == TransformType.WINSORIZE:
            lower = stats["lower"]
            upper = stats["upper"]
            return col.clip(lower, upper).alias(feature)

        elif transform == TransformType.LOG:
            offset = stats["offset"]
            return (col + offset).log().alias(feature)

        elif transform == TransformType.SQRT:
            # SQRT doesn't need fitted statistics, but handle negatives
            return col.abs().sqrt().alias(feature)

        elif transform == TransformType.DIFF:
            if use_training_boundary:
                last_value = stats["last_value"]
                row_number = pl.int_range(0, pl.len())
                return (
                    pl.when(row_number == 0)
                    .then(col - last_value)
                    .otherwise(col.diff())
                    .alias(feature)
                )
            return col.diff().alias(feature)

        return col.alias(feature)

    def fit(self, X: pl.DataFrame) -> PreprocessingPipeline:
        """Fit pipeline on training data.

        Computes statistics needed for each transform from training data only.

        Parameters
        ----------
        X : pl.DataFrame
            Training data with feature columns.

        Returns
        -------
        self
            Fitted pipeline.
        """
        missing = set(self._recommendations) - set(X.columns)
        if missing:
            raise ValueError(f"Recommended features missing from fitting data: {sorted(missing)}")

        statistics: dict[str, dict[str, Any]] = {}
        fitted_features: list[str] = []

        for feature in X.columns:
            if feature in self._recommendations:
                transform = self._get_transform_type(feature)
                if transform != TransformType.NONE:
                    if not X[feature].dtype.is_numeric():
                        raise ValueError(
                            f"Recommended feature '{feature}' must be numeric for {transform.value}"
                        )
                    values = X[feature].drop_nulls()
                    if len(values) == 0:
                        raise ValueError(
                            f"Recommended feature '{feature}' requires at least one numeric value"
                        )
                    if values.is_nan().any() or values.is_infinite().any():
                        raise ValueError(
                            f"Recommended feature '{feature}' contains NaN or infinity"
                        )
                statistics[feature] = self._compute_statistics(X, feature, transform)
                fitted_features.append(feature)

        self._statistics = statistics
        self._fitted_features = fitted_features
        self._is_fitted = True
        return self

    def _transform(self, X: pl.DataFrame, *, use_training_boundary: bool) -> pl.DataFrame:
        if not self._is_fitted:
            raise NotFittedError("Pipeline has not been fitted. Call fit() first.")

        missing = set(self._fitted_features) - set(X.columns)
        if missing:
            raise ValueError(f"Transform data missing fitted features: {sorted(missing)}")

        exprs = []
        for feature in X.columns:
            if feature in self._recommendations:
                transform = self._get_transform_type(feature)
                exprs.append(
                    self._apply_transform(
                        X,
                        feature,
                        transform,
                        use_training_boundary=use_training_boundary,
                    )
                )
            else:
                exprs.append(pl.col(feature))

        return X.select(exprs)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform new data using fitted training statistics.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform.

        Returns
        -------
        pl.DataFrame
            Transformed data.

        Raises
        ------
        NotFittedError
            If pipeline has not been fitted.
        """
        return self._transform(X, use_training_boundary=True)

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
        self.fit(X)
        return self._transform(X, use_training_boundary=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline state for persistence.

        Returns
        -------
        dict
            Serializable representation of fitted pipeline.
        """
        if not self._is_fitted:
            raise NotFittedError("Pipeline has not been fitted. Call fit() first.")

        return {
            "recommendations": copy.deepcopy(self._recommendations),
            "min_confidence": self._min_confidence,
            "winsorize_limits": list(self._winsorize_limits),
            "statistics": copy.deepcopy(self._statistics),
            "fitted_features": self._fitted_features.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingPipeline:
        """Load fitted pipeline from serialized state.

        Parameters
        ----------
        data : dict
            Output from to_dict().

        Returns
        -------
        PreprocessingPipeline
            Reconstructed fitted pipeline.
        """
        if not isinstance(data, dict):
            raise ValueError("Serialized pipeline must be a dictionary")
        required = {"recommendations", "statistics", "fitted_features"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Serialized pipeline is missing fields: {sorted(missing)}")

        statistics = data["statistics"]
        if not isinstance(statistics, dict):
            raise ValueError("Serialized pipeline statistics must be a dictionary")
        fitted_features = data["fitted_features"]
        if (
            not isinstance(fitted_features, list)
            or not all(isinstance(feature, str) and feature for feature in fitted_features)
            or len(fitted_features) != len(set(fitted_features))
        ):
            raise ValueError(
                "Serialized pipeline fitted_features must be a list of unique non-empty strings"
            )

        pipeline = cls(
            recommendations=copy.deepcopy(data["recommendations"]),
            min_confidence=data.get("min_confidence", 0.0),
            winsorize_limits=tuple(data.get("winsorize_limits", (0.01, 0.99))),
        )
        if set(fitted_features) != set(pipeline._recommendations):
            raise ValueError(
                "Serialized pipeline fitted_features must match recommendation features"
            )
        if set(statistics) != set(fitted_features) or not all(
            isinstance(value, dict) for value in statistics.values()
        ):
            raise ValueError(
                "Serialized pipeline statistics must match fitted features and contain dictionaries"
            )

        required_statistics = {
            TransformType.STANDARDIZE: {"mean", "std"},
            TransformType.NORMALIZE: {"min", "max", "range"},
            TransformType.WINSORIZE: {"lower", "upper"},
            TransformType.LOG: {"offset"},
            TransformType.DIFF: {"last_value"},
        }
        for feature in fitted_features:
            transform = pipeline._get_transform_type(feature)
            feature_statistics = statistics[feature]
            expected = required_statistics.get(transform, set())
            if not expected <= set(feature_statistics):
                raise ValueError(
                    f"Serialized pipeline statistics for '{feature}' are missing "
                    f"fields: {sorted(expected - set(feature_statistics))}"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real | Decimal)
                or not math.isfinite(float(value))
                for value in feature_statistics.values()
            ):
                raise ValueError(
                    f"Serialized pipeline statistics for '{feature}' must be finite numbers"
                )

        pipeline._statistics = copy.deepcopy(statistics)
        pipeline._fitted_features = fitted_features.copy()
        pipeline._is_fitted = True
        return pipeline

    def get_transform_summary(self) -> dict[str, str]:
        """Get summary of transforms to be applied.

        Returns
        -------
        dict
            Mapping of feature names to transform types.
        """
        return {
            feature: self._get_transform_type(feature).value for feature in self._recommendations
        }

    def __repr__(self) -> str:
        """Return string representation."""
        n_recs = len(self._recommendations)
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"PreprocessingPipeline(features={n_recs}, {fitted_str})"


# Convenience alias
Preprocessor = StandardScaler
