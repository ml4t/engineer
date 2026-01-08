"""Simple feature metadata registry for QFeatures.

This module provides a lightweight registry system for tracking feature metadata
without the overhead of complex class hierarchies.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class FeatureMetadata:
    """Metadata for a feature function.

    Attributes
    ----------
    name : str
        Unique feature identifier (e.g., "rsi", "macd")
    func : Callable
        The feature computation function
    category : str
        Feature category (e.g., "momentum", "volatility", "ml")
    description : str
        Brief description of what the feature computes
    formula : str
        Mathematical formula or algorithm description
    normalized : bool | None
        Whether the feature is scale-invariant and ML-ready without preprocessing
        - True: Normalized oscillators (RSI, Stochastic), percentages (ROC, returns)
                Can be used directly in ML models (e.g., gradient boosting)
                Value range independent of underlying asset price level
        - False: Price-scale features (SMA, EMA), unbounded (OBV, volume)
                 Requires preprocessing: returns, normalization, or z-score
        - None: Depends on parameters or input

        Note: This indicates ML-readiness, NOT statistical stationarity.
              For proper stationarity testing (ADF/KPSS), use the
              ml4t.engineer.diagnostics.stationarity module.
    lookback : Callable[[dict[str, Any]], int]
        Function that returns minimum lookback period given parameters
        For example: lambda **kwargs: kwargs.get("period", 14)
    ta_lib_compatible : bool
        Whether feature matches TA-Lib output
    input_type : str
        Expected input data type (e.g., "OHLCV", "close", "returns")
    output_type : str
        Output data type (e.g., "indicator", "signal", "label")
    parameters : dict[str, Any]
        Default parameters for the feature function
    dependencies : list[str]
        List of feature names this feature depends on
    references : list[str]
        Academic papers or documentation references
    tags : list[str]
        Additional searchable tags
    """

    name: str
    func: Callable[..., pl.DataFrame | pl.LazyFrame]
    category: str
    description: str
    formula: str = ""
    normalized: bool | None = False
    lookback: Callable[..., int] = lambda **_kwargs: 0
    ta_lib_compatible: bool = False
    input_type: str = "OHLCV"
    output_type: str = "indicator"
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    value_range: tuple[float, float] | None = None


class FeatureRegistry:
    """Simple dict-based registry for feature metadata.

    This registry uses a straightforward dictionary for storage with
    methods for common query patterns. No complex DAG engine or
    class hierarchies - just clean metadata lookup.

    Examples
    --------
    >>> from ml4t.engineer.core.registry import FeatureRegistry, FeatureMetadata
    >>> from ml4t.engineer.features.momentum.rsi import rsi
    >>>
    >>> # Create registry instance
    >>> registry = FeatureRegistry()
    >>>
    >>> # Register a feature
    >>> registry.register(FeatureMetadata(
    ...     name="rsi",
    ...     func=rsi,
    ...     category="momentum",
    ...     description="Relative Strength Index",
    ...     formula="RSI = 100 - (100 / (1 + RS))",
    ...     normalized=True,
    ...     lookback=14,
    ...     ta_lib_compatible=True,
    ... ))
    >>>
    >>> # Query features
    >>> metadata = registry.get("rsi")
    >>> momentum_features = registry.list_by_category("momentum")
    >>> ml_ready_features = registry.list_normalized()
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._features: dict[str, FeatureMetadata] = {}

    def register(self, metadata: FeatureMetadata) -> None:
        """Register a feature with its metadata.

        Parameters
        ----------
        metadata : FeatureMetadata
            Feature metadata to register

        Raises
        ------
        ValueError
            If feature name already registered
        """
        if metadata.name in self._features:
            raise ValueError(f"Feature '{metadata.name}' already registered")
        self._features[metadata.name] = metadata

    def get(self, name: str) -> FeatureMetadata | None:
        """Get metadata for a specific feature.

        Parameters
        ----------
        name : str
            Feature name to retrieve

        Returns
        -------
        FeatureMetadata | None
            Feature metadata if found, None otherwise
        """
        return self._features.get(name)

    def list_all(self) -> list[str]:
        """List all registered feature names.

        Returns
        -------
        list[str]
            Sorted list of all feature names
        """
        return sorted(self._features.keys())

    def list_normalized(self) -> list[str]:
        """List all normalized (ML-ready) features.

        Returns features that can be used directly in ML models without
        preprocessing. These are scale-invariant and won't have out-of-distribution
        values when the underlying asset price changes.

        Returns
        -------
        list[str]
            Sorted list of normalized (ML-ready) feature names
        """
        return sorted(name for name, meta in self._features.items() if meta.normalized)

    def list_by_category(self, category: str) -> list[str]:
        """List all features in a specific category.

        Parameters
        ----------
        category : str
            Category name (e.g., "momentum", "volatility")

        Returns
        -------
        list[str]
            Sorted list of feature names in the category
        """
        return sorted(name for name, meta in self._features.items() if meta.category == category)

    def list_ta_lib_compatible(self) -> list[str]:
        """List all TA-Lib compatible features.

        Returns
        -------
        list[str]
            Sorted list of TA-Lib compatible feature names
        """
        return sorted(name for name, meta in self._features.items() if meta.ta_lib_compatible)

    def get_dependencies(self, name: str) -> list[str]:
        """Get dependency list for a feature.

        Parameters
        ----------
        name : str
            Feature name

        Returns
        -------
        list[str]
            List of feature names this feature depends on (empty if none)

        Raises
        ------
        KeyError
            If feature not found in registry
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        return self._features[name].dependencies.copy()

    def clear(self) -> None:
        """Clear all registered features.

        Useful for testing or resetting the registry state.
        """
        self._features.clear()

    def __len__(self) -> int:
        """Return number of registered features."""
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        """Check if feature is registered."""
        return name in self._features

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return f"FeatureRegistry(features={len(self._features)})"


# Global registry instance
_global_registry = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance.

    Returns
    -------
    FeatureRegistry
        The global registry instance
    """
    return _global_registry
