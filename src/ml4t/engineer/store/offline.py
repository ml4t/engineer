# mypy: disable-error-code="arg-type,call-arg,assignment,return-value"
"""Offline feature store using DuckDB with Arrow integration.

Exports:
    OfflineFeatureStore(path) - DuckDB-based feature store
        .save(features, name, ...) - Save feature DataFrame
        .load(name, columns=None, ...) -> DataFrame - Load features
        .list_features() -> list[str] - List stored features
        .delete(name) - Delete stored features
        .point_in_time_join(features, labels, ...) -> DataFrame

    FeatureStoreError - Exception for store operations

This module provides a DuckDB-based offline feature store that enables:
- Zero-copy integration with Polars via Arrow
- Efficient storage and retrieval of computed features
- Point-in-time correct feature joins
- Partitioned storage for large datasets

Design Philosophy:
1. Zero-Copy: Arrow integration eliminates data copying
2. Performance: DuckDB provides fast analytics on cached features
3. Simplicity: Simple API for save/load operations
4. Correctness: Point-in-time joins prevent data leakage
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None  # type: ignore[assignment]

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import duckdb
    import polars as pl


class FeatureStoreError(Exception):
    """Raised when feature store operations fail."""


class OfflineFeatureStore:
    """DuckDB-based offline feature store with Arrow integration.

    Provides efficient storage and retrieval of computed features using
    DuckDB's columnar storage and Arrow zero-copy integration.

    Features:
        - Zero-copy Polars ↔ DuckDB via Arrow
        - Point-in-time correct feature retrieval
        - Partitioned storage for large datasets
        - SQL query support for filtering

    Example:
        >>> store = OfflineFeatureStore("features.duckdb")
        >>> store.save_features(features_df, "rsi_macd_features")
        >>> loaded = store.load_features("rsi_macd_features")
        >>> store.close()

        >>> # Or use context manager
        >>> with OfflineFeatureStore("features.duckdb") as store:
        ...     store.save_features(df, "my_features")
    """

    def __init__(
        self,
        path: str | Path | None = None,
        read_only: bool = False,
    ):
        """Initialize offline feature store.

        Args:
            path: Path to DuckDB database file. If None, creates in-memory DB.
            read_only: Whether to open database in read-only mode

        Raises:
            FeatureStoreError: If DuckDB not installed or connection fails

        Example:
            >>> # Persistent storage
            >>> store = OfflineFeatureStore("features.duckdb")

            >>> # In-memory (for testing)
            >>> store = OfflineFeatureStore()

            >>> # Read-only mode
            >>> store = OfflineFeatureStore("features.duckdb", read_only=True)
        """
        if not HAS_DUCKDB:
            raise FeatureStoreError("DuckDB not installed. Install with: pip install duckdb")

        if not HAS_POLARS:
            raise FeatureStoreError("Polars not installed. Install with: pip install polars")

        self.path = Path(path) if path else None
        self.read_only = read_only
        self._connection: duckdb.DuckDBPyConnection | None = None

        # Initialize connection
        self._connect()

    def _connect(self) -> None:
        """Establish DuckDB connection with Arrow support.

        Raises:
            FeatureStoreError: If connection or Arrow extension fails
        """
        try:
            # Connect to database (or in-memory)
            if self.path:
                # Ensure parent directory exists
                self.path.parent.mkdir(parents=True, exist_ok=True)

                # Connect with appropriate mode
                if self.read_only:
                    self._connection = duckdb.connect(str(self.path), read_only=True)
                else:
                    self._connection = duckdb.connect(str(self.path))
            else:
                # In-memory database
                self._connection = duckdb.connect(":memory:")

            # Note: Arrow integration is built into DuckDB 1.0+ core
            # No need to load any extension - Polars ↔ DuckDB works out of the box
            # Zero-copy operations are automatically enabled when available

        except Exception as e:
            raise FeatureStoreError(f"Failed to connect to DuckDB: {e}") from e

    @property
    def connection(self) -> "duckdb.DuckDBPyConnection":
        """Get active DuckDB connection.

        Returns:
            DuckDB connection object

        Raises:
            FeatureStoreError: If connection is closed
        """
        if self._connection is None:
            raise FeatureStoreError("Connection is closed. Call connect() first.")

        return self._connection

    def close(self) -> None:
        """Close DuckDB connection and release resources.

        Safe to call multiple times. After closing, the store cannot be used
        unless reconnected.

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> # ... use store ...
            >>> store.close()
        """
        if hasattr(self, "_connection") and self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                warnings.warn(f"Error closing connection: {e}", UserWarning, stacklevel=2)
            finally:
                self._connection = None

    def __enter__(self) -> "OfflineFeatureStore":
        """Context manager entry.

        Returns:
            Self for context manager use
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit - ensures connection is closed."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures connection is closed."""
        self.close()

    def is_connected(self) -> bool:
        """Check if connection is active.

        Returns:
            True if connected, False otherwise

        Example:
            >>> store = OfflineFeatureStore()
            >>> store.is_connected()
            True
            >>> store.close()
            >>> store.is_connected()
            False
        """
        return self._connection is not None

    def list_tables(self) -> list[str]:
        """List all tables in the feature store.

        Returns:
            List of table names

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> store.save_features(df, "my_features")
            >>> store.list_tables()
            ['my_features']
        """
        result = self.connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()

        return [row[0] for row in result]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the store.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists, False otherwise

        Example:
            >>> store = OfflineFeatureStore()
            >>> store.table_exists("my_features")
            False
            >>> store.save_features(df, "my_features")
            >>> store.table_exists("my_features")
            True
        """
        return table_name in self.list_tables()

    def execute(self, query: str) -> "duckdb.DuckDBPyRelation":
        """Execute raw SQL query on the store.

        Args:
            query: SQL query to execute

        Returns:
            DuckDB relation with query results

        Example:
            >>> result = store.execute("SELECT COUNT(*) FROM my_features")
            >>> count = result.fetchone()[0]
        """
        return self.connection.sql(query)

    def save_features(
        self,
        df: "pl.DataFrame",
        table_name: str,
        mode: str = "replace",
    ) -> None:
        """Save features to DuckDB with zero-copy Arrow integration.

        Args:
            df: Polars DataFrame with features to save
            table_name: Name of table to create/update
            mode: Write mode - "replace" (default), "append", or "fail"
                - replace: Drop and recreate table
                - append: Add rows to existing table
                - fail: Raise error if table exists

        Raises:
            FeatureStoreError: If mode is invalid or table exists with mode="fail"
            ValueError: If df is empty or table_name is invalid

        Example:
            >>> store = OfflineFeatureStore("features.duckdb")
            >>> store.save_features(df, "rsi_features")
            >>> store.save_features(df2, "rsi_features", mode="append")
        """
        # Validate inputs
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a Polars DataFrame, got {type(df).__name__}")

        if df.is_empty():
            raise ValueError("Cannot save empty DataFrame")

        if not table_name or not isinstance(table_name, str):
            raise ValueError("table_name must be a non-empty string")

        if mode not in ("replace", "append", "fail"):
            raise ValueError(f"mode must be 'replace', 'append', or 'fail', got '{mode}'")

        # Check table existence
        exists = self.table_exists(table_name)

        # Handle mode="fail"
        if mode == "fail" and exists:
            raise FeatureStoreError(f"Table '{table_name}' already exists and mode='fail'")

        # Drop table if mode="replace"
        if mode == "replace" and exists:
            self.connection.execute(f"DROP TABLE {table_name}")

        # Save using Arrow zero-copy
        # DuckDB can read directly from Arrow without copying
        try:
            if mode == "append" and exists:
                # Insert into existing table
                self.connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            else:
                # Create new table from DataFrame
                self.connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        except Exception as e:
            raise FeatureStoreError(f"Failed to save features to '{table_name}': {e}") from e

    def load_features(
        self,
        table_name: str,
        columns: list[str] | None = None,
        filter_expr: str | None = None,
        limit: int | None = None,
    ) -> "pl.DataFrame":
        """Load features from DuckDB with zero-copy Arrow integration.

        Args:
            table_name: Name of table to load
            columns: Optional list of columns to load (loads all if None)
            filter_expr: Optional SQL WHERE clause (without "WHERE" keyword)
                Example: "timestamp >= '2024-01-01'"
            limit: Optional row limit for result set

        Returns:
            Polars DataFrame with requested features

        Raises:
            FeatureStoreError: If table doesn't exist or query fails
            ValueError: If columns list is empty

        Example:
            >>> # Load all features
            >>> df = store.load_features("rsi_features")

            >>> # Load specific columns
            >>> df = store.load_features("rsi_features", columns=["timestamp", "rsi_14"])

            >>> # Load with filter
            >>> df = store.load_features("rsi_features", filter_expr="rsi_14 > 70")

            >>> # Load recent data with limit
            >>> df = store.load_features("rsi_features", limit=1000)
        """
        # Validate table exists
        if not self.table_exists(table_name):
            raise FeatureStoreError(
                f"Table '{table_name}' does not exist. Available tables: {self.list_tables()}"
            )

        # Validate columns
        if columns is not None:
            if not isinstance(columns, list):
                raise TypeError("columns must be a list of strings")
            if len(columns) == 0:
                raise ValueError("columns list cannot be empty")
            if not all(isinstance(col, str) for col in columns):
                raise TypeError("all columns must be strings")

        # Build SQL query
        select_clause = "*" if columns is None else ", ".join(columns)
        query = f"SELECT {select_clause} FROM {table_name}"

        # Add WHERE clause if provided
        if filter_expr:
            if not isinstance(filter_expr, str):
                raise TypeError("filter_expr must be a string")
            query += f" WHERE {filter_expr}"

        # Add LIMIT clause if provided
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            query += f" LIMIT {limit}"

        # Execute query and convert to Polars via Arrow (zero-copy)
        try:
            result = self.connection.execute(query)
            # Use .pl() method for zero-copy Arrow → Polars conversion
            return result.pl()
        except Exception as e:
            raise FeatureStoreError(f"Failed to load features from '{table_name}': {e}") from e

    def point_in_time_join(
        self,
        labels: "pl.DataFrame",
        features_table: str,
        timestamp_col: str = "timestamp",
        join_keys: list[str] | None = None,
        tolerance: str | None = None,
    ) -> "pl.DataFrame":
        """Perform point-in-time correct join to prevent data leakage.

        Joins labels with features, ensuring each label only uses features
        that were available at or before the label's timestamp. This prevents
        look-ahead bias in ML models.

        Args:
            labels: DataFrame with labels and timestamps
            features_table: Name of features table to join
            timestamp_col: Name of timestamp column (default: "timestamp")
            join_keys: Optional list of additional join keys (e.g., ["symbol"])
                If None, joins only on time
            tolerance: Optional time tolerance (e.g., "1h", "1d")
                Maximum time difference allowed for a match

        Returns:
            Polars DataFrame with labels and point-in-time correct features

        Raises:
            FeatureStoreError: If features table doesn't exist or query fails
            ValueError: If labels DataFrame is invalid or missing timestamp column

        Example:
            >>> # Simple time-based join
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features"
            ... )

            >>> # Join with additional keys (e.g., per-symbol)
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features",
            ...     join_keys=["symbol"]
            ... )

            >>> # Join with time tolerance (use features within 1 hour)
            >>> result = store.point_in_time_join(
            ...     labels=labels_df,
            ...     features_table="rsi_features",
            ...     tolerance="1h"
            ... )

        Notes:
            For each label row, this joins the most recent feature row where:
            - feature.timestamp <= label.timestamp (no look-ahead)
            - Additional join keys match (if specified)
            - Within tolerance window (if specified)

            This is critical for backtesting to avoid data leakage.
        """
        # Validate labels DataFrame
        if not isinstance(labels, pl.DataFrame):
            raise TypeError(f"labels must be a Polars DataFrame, got {type(labels).__name__}")

        if labels.is_empty():
            raise ValueError("labels DataFrame cannot be empty")

        # Validate timestamp column exists in labels
        if timestamp_col not in labels.columns:
            raise ValueError(
                f"timestamp column '{timestamp_col}' not found in labels. "
                f"Available columns: {labels.columns}"
            )

        # Validate features table exists
        if not self.table_exists(features_table):
            raise FeatureStoreError(
                f"Features table '{features_table}' does not exist. "
                f"Available tables: {self.list_tables()}"
            )

        # Validate join_keys exist in labels if provided
        if join_keys:
            if not isinstance(join_keys, list):
                raise TypeError("join_keys must be a list of strings")
            missing_keys = [key for key in join_keys if key not in labels.columns]
            if missing_keys:
                raise ValueError(
                    f"join_keys {missing_keys} not found in labels. "
                    f"Available columns: {labels.columns}"
                )

        # Load features from DuckDB and use Polars native join_asof
        # This is simpler, more readable, and leverages Polars' optimized temporal join
        try:
            # Load features from DuckDB table
            features_df = self.load_features(features_table)

            # Ensure both DataFrames are sorted by join_keys and timestamp for join_asof
            # When using 'by' parameter, we need to sort by both join_keys and timestamp
            # to avoid sortedness warnings and ensure correct join behavior
            sort_cols = join_keys + [timestamp_col] if join_keys else [timestamp_col]
            labels_sorted = labels.sort(sort_cols)
            features_sorted = features_df.sort(sort_cols)

            # Perform point-in-time join using Polars' join_asof
            # Strategy: backward (take most recent feature <= label timestamp)
            # Note: Polars warns about sortedness verification when using 'by' parameter
            # but we explicitly sorted above, so we suppress this warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sortedness of columns cannot be checked when 'by' groups provided",
                    category=UserWarning,
                )
                result = labels_sorted.join_asof(
                    features_sorted,
                    on=timestamp_col,
                    by=join_keys,  # Additional join keys (e.g., symbol)
                    strategy="backward",  # Most recent feature <= label timestamp
                    tolerance=tolerance,  # Optional time window
                )

            return result

        except Exception as e:
            raise FeatureStoreError(
                f"Failed to perform point-in-time join with '{features_table}': {e}"
            ) from e

    def __repr__(self) -> str:
        """String representation of feature store."""
        location = f"path={self.path}" if self.path else "in-memory"

        status = "connected" if self.is_connected() else "closed"
        mode = "read-only" if self.read_only else "read-write"

        return f"OfflineFeatureStore({location}, {mode}, {status})"
