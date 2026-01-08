# mypy: disable-error-code="misc,no-any-return,call-arg"
"""Preprocessing configuration with Pydantic v2 serialization.

This module provides config classes for preprocessing (scalers) that extend BaseConfig,
enabling JSON/YAML serialization for reproducible ML pipelines.

Examples
--------
>>> from ml4t.engineer.config import PreprocessingConfig
>>>
>>> # Create and serialize config
>>> config = PreprocessingConfig(scaler="standard")
>>> config.to_yaml("preprocessing_config.yaml")
>>>
>>> # Load and create scaler
>>> config = PreprocessingConfig.from_yaml("preprocessing_config.yaml")
>>> scaler = config.create_scaler()
>>> train_scaled = scaler.fit_transform(train_features)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ml4t.engineer.config.base import BaseConfig

if TYPE_CHECKING:
    from ml4t.engineer.preprocessing import BaseScaler


class PreprocessingConfig(BaseConfig):
    """Configuration for preprocessing (feature scaling).

    Extends BaseConfig for full JSON/YAML serialization support.
    Use `create_scaler()` to instantiate the configured scaler.

    Attributes
    ----------
    scaler : str | None
        Scaler type: "standard", "minmax", "robust", or None (no scaling)
    columns : list[str] | None
        Specific columns to scale (None = all numeric columns)

    Standard Scaler Parameters
    --------------------------
    with_mean : bool
        Center features by removing the mean
    with_std : bool
        Scale features to unit variance

    MinMax Scaler Parameters
    ------------------------
    feature_range : tuple[float, float]
        Target range for scaling (default: (0.0, 1.0))

    Robust Scaler Parameters
    ------------------------
    with_centering : bool
        Center features using median
    with_scaling : bool
        Scale features using IQR
    quantile_range : tuple[float, float]
        Quantile range for IQR (default: (25.0, 75.0))

    Examples
    --------
    >>> # Standard scaling (z-score normalization)
    >>> config = PreprocessingConfig(scaler="standard")
    >>> scaler = config.create_scaler()
    >>> train_scaled = scaler.fit_transform(train_features)
    >>> test_scaled = scaler.transform(test_features)  # Uses train statistics
    >>>
    >>> # Robust scaling (outlier-resistant)
    >>> config = PreprocessingConfig.robust(quantile_range=(10.0, 90.0))
    >>> scaler = config.create_scaler()
    >>>
    >>> # Serialize for reproducibility
    >>> config.to_yaml("preprocessing.yaml")
    """

    # Scaler type
    scaler: Literal["standard", "minmax", "robust"] | None = Field(
        "standard",
        description="Scaler type: 'standard', 'minmax', 'robust', or None",
    )

    # Column selection (None = all numeric)
    columns: list[str] | None = Field(
        None,
        description="Columns to scale (None = all numeric columns)",
    )

    # StandardScaler parameters
    with_mean: bool = Field(
        True,
        description="Center features by removing the mean (StandardScaler)",
    )
    with_std: bool = Field(
        True,
        description="Scale features to unit variance (StandardScaler)",
    )

    # MinMaxScaler parameters
    feature_range: tuple[float, float] = Field(
        (0.0, 1.0),
        description="Target range for scaling (MinMaxScaler)",
    )

    # RobustScaler parameters
    with_centering: bool = Field(
        True,
        description="Center features using median (RobustScaler)",
    )
    with_scaling: bool = Field(
        True,
        description="Scale features using IQR (RobustScaler)",
    )
    quantile_range: tuple[float, float] = Field(
        (25.0, 75.0),
        description="Quantile range for IQR calculation (RobustScaler)",
    )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def standard(
        cls,
        with_mean: bool = True,
        with_std: bool = True,
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> PreprocessingConfig:
        """Create StandardScaler config.

        Z-score normalization: (x - mean) / std

        Parameters
        ----------
        with_mean : bool
            Center features by removing the mean
        with_std : bool
            Scale features to unit variance
        columns : list[str] | None
            Columns to scale (None = all)

        Returns
        -------
        PreprocessingConfig
            Configured for StandardScaler

        Examples
        --------
        >>> config = PreprocessingConfig.standard()
        >>> scaler = config.create_scaler()
        """
        return cls(
            scaler="standard",
            with_mean=with_mean,
            with_std=with_std,
            columns=columns,
            **kwargs,
        )

    @classmethod
    def minmax(
        cls,
        feature_range: tuple[float, float] = (0.0, 1.0),
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> PreprocessingConfig:
        """Create MinMaxScaler config.

        Scales features to [min, max] range.

        Parameters
        ----------
        feature_range : tuple[float, float]
            Target range for scaling (default: (0.0, 1.0))
        columns : list[str] | None
            Columns to scale (None = all)

        Returns
        -------
        PreprocessingConfig
            Configured for MinMaxScaler

        Examples
        --------
        >>> config = PreprocessingConfig.minmax(feature_range=(-1.0, 1.0))
        >>> scaler = config.create_scaler()
        """
        return cls(
            scaler="minmax",
            feature_range=feature_range,
            columns=columns,
            **kwargs,
        )

    @classmethod
    def robust(
        cls,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        columns: list[str] | None = None,
        **kwargs: Any,
    ) -> PreprocessingConfig:
        """Create RobustScaler config.

        Uses median and IQR, making it robust to outliers.

        Parameters
        ----------
        with_centering : bool
            Center features using median
        with_scaling : bool
            Scale features using IQR
        quantile_range : tuple[float, float]
            Quantile range for IQR (default: (25.0, 75.0))
        columns : list[str] | None
            Columns to scale (None = all)

        Returns
        -------
        PreprocessingConfig
            Configured for RobustScaler

        Examples
        --------
        >>> config = PreprocessingConfig.robust(quantile_range=(10.0, 90.0))
        >>> scaler = config.create_scaler()
        """
        return cls(
            scaler="robust",
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            columns=columns,
            **kwargs,
        )

    @classmethod
    def none(cls) -> PreprocessingConfig:
        """Create config with no scaling.

        Returns
        -------
        PreprocessingConfig
            Configured for no scaling

        Examples
        --------
        >>> config = PreprocessingConfig.none()
        >>> assert config.create_scaler() is None
        """
        return cls(scaler=None)

    # =========================================================================
    # Scaler Creation
    # =========================================================================

    def create_scaler(self) -> BaseScaler | None:
        """Create scaler instance from stored parameters.

        Returns
        -------
        BaseScaler | None
            Configured scaler, or None if scaler="none"

        Examples
        --------
        >>> config = PreprocessingConfig(scaler="standard")
        >>> scaler = config.create_scaler()
        >>> train_scaled = scaler.fit_transform(train_features)
        >>> test_scaled = scaler.transform(test_features)
        """
        if self.scaler is None:
            return None

        from ml4t.engineer.preprocessing import (
            MinMaxScaler,
            RobustScaler,
            StandardScaler,
        )

        if self.scaler == "standard":
            return StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std,
            )
        elif self.scaler == "minmax":
            return MinMaxScaler(
                feature_range=self.feature_range,
            )
        elif self.scaler == "robust":
            return RobustScaler(
                with_centering=self.with_centering,
                with_scaling=self.with_scaling,
                quantile_range=self.quantile_range,
            )
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler}")


__all__ = ["PreprocessingConfig"]
