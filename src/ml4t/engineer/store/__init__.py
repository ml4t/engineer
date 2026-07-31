"""Feature storage layer for ml4t.engineer.

Provides offline feature store capabilities using DuckDB with Arrow
integration for zero-copy performance.
"""

from .offline import FeatureStoreError, OfflineFeatureStore

__all__ = ["FeatureStoreError", "OfflineFeatureStore"]
