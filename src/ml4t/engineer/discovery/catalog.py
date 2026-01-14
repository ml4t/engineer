"""Feature catalog for enhanced discoverability.

Exports:
    FeatureCatalog(registry) - Feature discovery interface
        .list(category=None, normalized=None, ...) -> list[FeatureMetadata]
        .search(query, ...) -> list[FeatureMetadata]
        .describe(name) -> str - Rich feature description
        .categories() -> list[str] - Available categories
        .tags() -> list[str] - Available tags

    Module-level API (via proxy):
        from ml4t.engineer import features
        features.list(category="momentum")
        features.search("volatility")
        features.describe("rsi")

Provides filtering, search, and description capabilities for the feature registry.

Examples
--------
>>> from ml4t.engineer import features
>>>
>>> # List all momentum features
>>> features.list(category="momentum")
>>>
>>> # Find normalized features for ML
>>> features.list(normalized=True, limit=10)
>>>
>>> # Search for volatility-related features
>>> features.search("volatility")
>>>
>>> # Get detailed description
>>> features.describe("rsi")
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

# Type alias to avoid shadowing by list() method
_list = builtins.list

if TYPE_CHECKING:
    from ml4t.engineer.core.registry import FeatureMetadata, FeatureRegistry


class FeatureCatalog:
    """Enhanced feature discovery interface.

    Wraps the FeatureRegistry to provide filtering, search, and
    rich description capabilities for feature discovery.

    Parameters
    ----------
    registry : FeatureRegistry | None
        Registry to wrap. If None, uses the global registry.

    Examples
    --------
    >>> from ml4t.engineer.discovery import FeatureCatalog
    >>> catalog = FeatureCatalog()
    >>>
    >>> # Multi-criteria filtering
    >>> catalog.list(category="momentum", normalized=True, ta_lib_compatible=True)
    >>>
    >>> # Full-text search
    >>> results = catalog.search("moving average")
    >>> for name, score in results:
    ...     print(f"{name}: {score:.2f}")
    >>>
    >>> # Rich description
    >>> info = catalog.describe("rsi")
    >>> print(info["formula"])
    """

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        """Initialize catalog with registry.

        Parameters
        ----------
        registry : FeatureRegistry | None
            Registry to wrap. If None, uses the global registry.
        """
        if registry is None:
            from ml4t.engineer.core.registry import get_registry

            registry = get_registry()
        self._registry = registry

    def list(
        self,
        category: str | None = None,
        normalized: bool | None = None,
        ta_lib_compatible: bool | None = None,
        tags: _list[str] | None = None,
        input_type: str | None = None,
        output_type: str | None = None,
        has_dependencies: bool | None = None,
        limit: int | None = None,
    ) -> _list[str]:
        """List features matching specified criteria.

        All criteria are combined with AND logic. If no criteria specified,
        returns all registered features.

        Parameters
        ----------
        category : str | None
            Filter by category (e.g., "momentum", "volatility", "ml")
        normalized : bool | None
            Filter by ML-ready status (True = scale-invariant)
        ta_lib_compatible : bool | None
            Filter by TA-Lib validation status
        tags : _list[str] | None
            Filter by tags (AND matching - must have ALL specified tags)
        input_type : str | None
            Filter by input data requirements (e.g., "OHLCV", "close")
        output_type : str | None
            Filter by output type (e.g., "indicator", "signal", "label")
        has_dependencies : bool | None
            Filter by whether feature has dependencies
        limit : int | None
            Maximum number of results to return

        Returns
        -------
        _list[str]
            Sorted list of feature names matching all criteria

        Examples
        --------
        >>> # All momentum indicators
        >>> features.list(category="momentum")
        >>>
        >>> # ML-ready volatility features
        >>> features.list(category="volatility", normalized=True)
        >>>
        >>> # Features that only need close price
        >>> features.list(input_type="close")
        """
        results: _list[str] = []

        for name, meta in self._registry._features.items():
            # Apply filters
            if category is not None and meta.category != category:
                continue
            if normalized is not None and meta.normalized != normalized:
                continue
            if ta_lib_compatible is not None and meta.ta_lib_compatible != ta_lib_compatible:
                continue
            if input_type is not None and meta.input_type != input_type:
                continue
            if output_type is not None and meta.output_type != output_type:
                continue
            if has_dependencies is not None:
                has_deps = len(meta.dependencies) > 0
                if has_deps != has_dependencies:
                    continue
            # AND matching - must have ALL specified tags
            if tags is not None and not all(tag in meta.tags for tag in tags):
                continue

            results.append(name)

        # Sort results
        results.sort()

        # Apply limit
        if limit is not None:
            results = results[:limit]

        return results

    def describe(self, name: str) -> dict[str, Any]:
        """Get rich metadata for a single feature.

        Parameters
        ----------
        name : str
            Feature name to describe

        Returns
        -------
        dict[str, Any]
            Full metadata as dictionary with computed properties:
            - name, category, description, formula
            - normalized, ta_lib_compatible
            - input_type, output_type
            - parameters (default values)
            - dependencies, references, tags
            - value_range (if defined)
            - lookback_period (computed from default params)

        Raises
        ------
        KeyError
            If feature not found in registry

        Examples
        --------
        >>> info = features.describe("rsi")
        >>> print(info["description"])
        'Relative Strength Index'
        >>> print(info["parameters"])
        {'period': 14}
        """
        meta = self._registry.get(name)
        if meta is None:
            available = self._registry.list_all()[:10]
            raise KeyError(
                f"Feature '{name}' not found in registry. "
                f"Available features: {available}{'...' if len(self._registry) > 10 else ''}"
            )

        # Build description dict
        result: dict[str, Any] = {
            "name": meta.name,
            "category": meta.category,
            "description": meta.description,
            "formula": meta.formula,
            "normalized": meta.normalized,
            "ta_lib_compatible": meta.ta_lib_compatible,
            "input_type": meta.input_type,
            "output_type": meta.output_type,
            "parameters": dict(meta.parameters),
            "dependencies": list(meta.dependencies),
            "references": list(meta.references),
            "tags": list(meta.tags),
            "value_range": meta.value_range,
        }

        # Compute lookback period from default parameters
        try:
            result["lookback_period"] = meta.lookback(**meta.parameters)
        except Exception:
            result["lookback_period"] = None

        return result

    def search(
        self,
        query: str,
        search_fields: _list[str] | None = None,
        max_results: int = 10,
    ) -> _list[tuple[str, float]]:
        """Full-text search across feature metadata.

        Searches name, description, formula, and tags by default.
        Returns results sorted by relevance score (higher = better match).

        Parameters
        ----------
        query : str
            Search query (case-insensitive substring matching)
        search_fields : _list[str] | None
            Fields to search. Default: ["name", "description", "formula", "tags"]
            Available: name, description, formula, category, tags, references
        max_results : int
            Maximum number of results to return (default 10)

        Returns
        -------
        _list[tuple[str, float]]
            List of (feature_name, relevance_score) tuples, sorted by score.
            Score is 0.0-1.0, with 1.0 being exact name match.

        Examples
        --------
        >>> # Search for volatility features
        >>> results = features.search("volatility")
        >>> for name, score in results[:5]:
        ...     print(f"{name}: {score:.2f}")
        >>>
        >>> # Search only in names and tags
        >>> results = features.search("momentum", search_fields=["name", "tags"])
        """
        if search_fields is None:
            search_fields = ["name", "description", "formula", "tags"]

        # Early return for empty query
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        query_terms = query_lower.split()
        scored_results: _list[tuple[str, float]] = []

        for name, meta in self._registry._features.items():
            score = 0.0

            # Score each field
            for field in search_fields:
                field_value = self._get_field_text(meta, field)
                if not field_value:
                    continue

                field_lower = field_value.lower()

                # Exact match in name gets highest score
                if field == "name" and field_lower == query_lower:
                    score += 1.0
                # Name contains query
                elif field == "name" and query_lower in field_lower:
                    score += 0.8
                # Other fields contain full query
                elif query_lower in field_lower:
                    score += 0.5
                # Individual term matching
                else:
                    term_matches = sum(1 for term in query_terms if term in field_lower)
                    if term_matches > 0:
                        score += 0.3 * (term_matches / len(query_terms))

            if score > 0:
                scored_results.append((name, score))

        # Sort by score descending, then by name
        scored_results.sort(key=lambda x: (-x[1], x[0]))

        return scored_results[:max_results]

    def by_input_type(self, input_type: str) -> _list[str]:
        """Get features that accept a specific input type.

        Parameters
        ----------
        input_type : str
            Input type to filter by (e.g., "OHLCV", "close", "returns")

        Returns
        -------
        _list[str]
            Sorted list of feature names requiring this input type

        Examples
        --------
        >>> # Features that only need close prices (simpler data requirements)
        >>> features.by_input_type("close")
        >>>
        >>> # Features that need full OHLCV data
        >>> features.by_input_type("OHLCV")
        """
        return self.list(input_type=input_type)

    def by_lookback(self, max_lookback: int) -> _list[str]:
        """Get features with lookback period at or below threshold.

        Useful for real-time applications with limited history.

        Parameters
        ----------
        max_lookback : int
            Maximum acceptable lookback period (in bars)

        Returns
        -------
        _list[str]
            Sorted list of feature names with lookback <= max_lookback

        Examples
        --------
        >>> # Features usable with only 20 bars of history
        >>> features.by_lookback(20)
        """
        results: _list[str] = []

        for name, meta in self._registry._features.items():
            try:
                lookback = meta.lookback(**meta.parameters)
                if lookback <= max_lookback:
                    results.append(name)
            except Exception:
                # Skip features where lookback can't be computed
                continue

        return sorted(results)

    def categories(self) -> _list[str]:
        """Get all unique feature categories.

        Returns
        -------
        _list[str]
            Sorted list of unique category names

        Examples
        --------
        >>> features.categories()
        ['math', 'microstructure', 'ml', 'momentum', 'price_transform', ...]
        """
        categories = {meta.category for meta in self._registry._features.values()}
        return sorted(categories)

    def input_types(self) -> _list[str]:
        """Get all unique input types across features.

        Returns
        -------
        _list[str]
            Sorted list of unique input types

        Examples
        --------
        >>> features.input_types()
        ['OHLCV', 'close', 'returns']
        """
        types = {meta.input_type for meta in self._registry._features.values()}
        return sorted(types)

    def stats(self) -> dict[str, Any]:
        """Get summary statistics about registered features.

        Returns
        -------
        dict[str, Any]
            Statistics including:
            - total: Total number of features
            - by_category: Count per category
            - normalized: Count of ML-ready features
            - ta_lib_compatible: Count of TA-Lib validated features
            - by_input_type: Count per input type

        Examples
        --------
        >>> stats = features.stats()
        >>> print(f"Total features: {stats['total']}")
        >>> print(f"Momentum: {stats['by_category'].get('momentum', 0)}")
        """
        total = len(self._registry)
        by_category: dict[str, int] = {}
        by_input_type: dict[str, int] = {}
        normalized_count = 0
        ta_lib_count = 0

        for meta in self._registry._features.values():
            # Count by category
            by_category[meta.category] = by_category.get(meta.category, 0) + 1

            # Count by input type
            by_input_type[meta.input_type] = by_input_type.get(meta.input_type, 0) + 1

            # Count normalized
            if meta.normalized:
                normalized_count += 1

            # Count TA-Lib compatible
            if meta.ta_lib_compatible:
                ta_lib_count += 1

        return {
            "total": total,
            "by_category": dict(sorted(by_category.items())),
            "by_input_type": dict(sorted(by_input_type.items())),
            "normalized": normalized_count,
            "ta_lib_compatible": ta_lib_count,
        }

    def _get_field_text(self, meta: FeatureMetadata, field: str) -> str:
        """Extract text content from a metadata field for searching."""
        if field == "name":
            return meta.name
        elif field == "description":
            return meta.description
        elif field == "formula":
            return meta.formula
        elif field == "category":
            return meta.category
        elif field == "tags":
            return " ".join(meta.tags)
        elif field == "references":
            return " ".join(meta.references)
        else:
            return ""

    def __len__(self) -> int:
        """Return number of registered features."""
        return len(self._registry)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"FeatureCatalog(features={len(self)})"


# Module-level convenience instance
# Lazily initialized to avoid circular imports
_features: FeatureCatalog | None = None


def _get_features() -> FeatureCatalog:
    """Get or create the global FeatureCatalog instance."""
    global _features
    if _features is None:
        _features = FeatureCatalog()
    return _features


class _FeatureCatalogProxy:
    """Proxy class to enable module-level attribute access to FeatureCatalog.

    This allows `from ml4t.engineer import features` followed by
    `features.list()` without needing to call a function.
    """

    def list(self, **kwargs: Any) -> _list[str]:
        return _get_features().list(**kwargs)

    def describe(self, name: str) -> dict[str, Any]:
        return _get_features().describe(name)

    def search(
        self, query: str, search_fields: _list[str] | None = None, max_results: int = 10
    ) -> _list[tuple[str, float]]:
        return _get_features().search(query, search_fields, max_results)

    def by_input_type(self, input_type: str) -> _list[str]:
        return _get_features().by_input_type(input_type)

    def by_lookback(self, max_lookback: int) -> _list[str]:
        return _get_features().by_lookback(max_lookback)

    def categories(self) -> _list[str]:
        return _get_features().categories()

    def input_types(self) -> _list[str]:
        return _get_features().input_types()

    def stats(self) -> dict[str, Any]:
        return _get_features().stats()

    def __len__(self) -> int:
        return len(_get_features())

    def __repr__(self) -> str:
        return repr(_get_features())


# Global proxy instance for convenience access
features = _FeatureCatalogProxy()

__all__ = ["FeatureCatalog", "features"]
