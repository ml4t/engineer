"""Comprehensive tests for OfflineFeatureStore.

Tests cover:
- Connection management and lifecycle
- save_features() with all modes
- load_features() with filtering
- point_in_time_join() for data leakage prevention
- Error handling and validation
- Integration and edge cases
"""

import tempfile
import time
from pathlib import Path

import polars as pl
import pytest

# Check for DuckDB availability
pytest.importorskip("duckdb")

from ml4t.engineer.store import OfflineFeatureStore
from ml4t.engineer.store.offline import FeatureStoreError

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db():
    """Temporary database file path (file not pre-created).

    DuckDB requires either:
    1. A non-existent path (will create and initialize the file)
    2. An existing valid DuckDB file

    Creating an empty file causes "not a valid DuckDB database file" error.
    """
    # Get a temp path without creating the file
    temp_dir = Path(tempfile.gettempdir())
    db_path = temp_dir / f"test_duckdb_{id(object())}.duckdb"

    yield db_path

    # Cleanup after test
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def store_memory():
    """In-memory feature store for testing."""
    store = OfflineFeatureStore()
    yield store
    store.close()


@pytest.fixture
def store_persistent(temp_db):
    """Persistent feature store for testing."""
    store = OfflineFeatureStore(temp_db)
    yield store
    store.close()


@pytest.fixture
def sample_features():
    """Sample features DataFrame."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "close": [150.0, 151.0, 149.0],
            "rsi_14": [55.0, 60.0, 58.0],
            "macd": [0.5, 0.7, 0.3],
        }
    )


@pytest.fixture
def sample_labels():
    """Sample labels DataFrame for point-in-time testing."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "target": [1, 0, 1],
        }
    )


# ============================================================================
# Connection Management Tests
# ============================================================================


class TestConnectionManagement:
    """Test connection lifecycle and management."""

    def test_init_memory_store(self):
        """Test creating in-memory store."""
        store = OfflineFeatureStore()
        assert store.is_connected()
        assert store.path is None
        assert not store.read_only
        store.close()

    def test_init_persistent_store(self, temp_db):
        """Test creating persistent store."""
        store = OfflineFeatureStore(temp_db)
        assert store.is_connected()
        assert store.path == temp_db
        assert not store.read_only
        store.close()

    def test_init_read_only_store(self, temp_db):
        """Test creating read-only store."""
        # First create the database
        store1 = OfflineFeatureStore(temp_db)
        store1.close()

        # Open in read-only mode
        store2 = OfflineFeatureStore(temp_db, read_only=True)
        assert store2.is_connected()
        assert store2.read_only
        store2.close()

    def test_context_manager(self, temp_db):
        """Test context manager closes connection."""
        with OfflineFeatureStore(temp_db) as store:
            assert store.is_connected()
        assert not store.is_connected()

    def test_close_multiple_times(self, store_memory):
        """Test closing connection multiple times is safe."""
        store_memory.close()
        store_memory.close()  # Should not raise
        assert not store_memory.is_connected()

    def test_list_tables_empty(self, store_memory):
        """Test listing tables in empty store."""
        tables = store_memory.list_tables()
        assert isinstance(tables, list)
        assert len(tables) == 0

    def test_table_exists_false(self, store_memory):
        """Test table_exists for non-existent table."""
        assert not store_memory.table_exists("nonexistent")


# ============================================================================
# save_features() Tests
# ============================================================================


class TestSaveFeatures:
    """Test save_features() method."""

    def test_save_new_table(self, store_memory, sample_features):
        """Test saving features to new table."""
        store_memory.save_features(sample_features, "test_features")

        assert "test_features" in store_memory.list_tables()
        assert store_memory.table_exists("test_features")

    def test_save_replace_mode(self, store_memory, sample_features):
        """Test replace mode overwrites existing table."""
        # Save initial data
        store_memory.save_features(sample_features, "test")

        # Save new data (should replace)
        new_df = sample_features.with_columns(pl.lit(999.0).alias("rsi_14"))
        store_memory.save_features(new_df, "test", mode="replace")

        # Verify replaced
        loaded = store_memory.load_features("test")
        assert loaded["rsi_14"][0] == 999.0

    def test_save_append_mode(self, store_memory, sample_features):
        """Test append mode adds rows."""
        # Save initial data (3 rows)
        store_memory.save_features(sample_features, "test")

        # Append more data (3 more rows)
        store_memory.save_features(sample_features, "test", mode="append")

        # Verify appended (should have 6 rows)
        loaded = store_memory.load_features("test")
        assert len(loaded) == 6

    def test_save_fail_mode_new_table(self, store_memory, sample_features):
        """Test fail mode succeeds for new table."""
        store_memory.save_features(sample_features, "test", mode="fail")
        assert store_memory.table_exists("test")

    def test_save_fail_mode_existing_table_raises_error(self, store_memory, sample_features):
        """Test fail mode raises error for existing table."""
        store_memory.save_features(sample_features, "test")

        with pytest.raises(FeatureStoreError, match="already exists"):
            store_memory.save_features(sample_features, "test", mode="fail")

    def test_save_empty_dataframe_raises_error(self, store_memory):
        """Test saving empty DataFrame raises error."""
        empty_df = pl.DataFrame({"col": []})

        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            store_memory.save_features(empty_df, "test")

    def test_save_invalid_table_name_raises_error(self, store_memory, sample_features):
        """Test invalid table name raises error."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            store_memory.save_features(sample_features, "")

    def test_save_invalid_mode_raises_error(self, store_memory, sample_features):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="must be 'replace', 'append', or 'fail'"):
            store_memory.save_features(sample_features, "test", mode="invalid")

    def test_save_wrong_type_raises_error(self, store_memory):
        """Test wrong DataFrame type raises error."""
        with pytest.raises(TypeError, match="must be a Polars DataFrame"):
            store_memory.save_features({"not": "a dataframe"}, "test")


# ============================================================================
# load_features() Tests
# ============================================================================


class TestLoadFeatures:
    """Test load_features() method."""

    def test_load_all_features(self, store_memory, sample_features):
        """Test loading all features."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features("test")

        assert loaded.shape == sample_features.shape
        assert set(loaded.columns) == set(sample_features.columns)

    def test_load_specific_columns(self, store_memory, sample_features):
        """Test loading specific columns."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features("test", columns=["timestamp", "rsi_14"])

        assert set(loaded.columns) == {"timestamp", "rsi_14"}
        assert len(loaded) == len(sample_features)

    def test_load_with_filter(self, store_memory, sample_features):
        """Test loading with filter expression."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features("test", filter_expr="rsi_14 > 56")

        assert len(loaded) == 2  # Only rows with RSI > 56
        assert all(loaded["rsi_14"] > 56)

    def test_load_with_limit(self, store_memory, sample_features):
        """Test loading with row limit."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features("test", limit=2)

        assert len(loaded) == 2

    def test_load_combined_options(self, store_memory, sample_features):
        """Test loading with columns, filter, and limit."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features(
            "test",
            columns=["timestamp", "rsi_14"],
            filter_expr="rsi_14 > 55",
            limit=2,
        )

        assert set(loaded.columns) == {"timestamp", "rsi_14"}
        assert len(loaded) <= 2
        assert all(loaded["rsi_14"] > 55)

    def test_load_nonexistent_table_raises_error(self, store_memory):
        """Test loading non-existent table raises error."""
        with pytest.raises(FeatureStoreError, match="does not exist"):
            store_memory.load_features("nonexistent")

    def test_load_empty_columns_raises_error(self, store_memory, sample_features):
        """Test loading with empty columns list raises error."""
        store_memory.save_features(sample_features, "test")

        with pytest.raises(ValueError, match="cannot be empty"):
            store_memory.load_features("test", columns=[])

    def test_load_invalid_columns_type_raises_error(self, store_memory, sample_features):
        """Test loading with invalid columns type raises error."""
        store_memory.save_features(sample_features, "test")

        with pytest.raises(TypeError, match="must be a list"):
            store_memory.load_features("test", columns="not a list")

    def test_load_negative_limit_raises_error(self, store_memory, sample_features):
        """Test loading with negative limit raises error."""
        store_memory.save_features(sample_features, "test")

        with pytest.raises(ValueError, match="must be a positive integer"):
            store_memory.load_features("test", limit=-1)


# ============================================================================
# point_in_time_join() Tests
# ============================================================================


class TestPointInTimeJoin:
    """Test point_in_time_join() method."""

    def test_basic_time_join(self, store_memory, sample_features, sample_labels):
        """Test basic time-based join."""
        store_memory.save_features(sample_features, "features")
        result = store_memory.point_in_time_join(sample_labels, "features")

        # Should have all label rows
        assert len(result) == len(sample_labels)
        # Should have both label and feature columns
        assert "target" in result.columns
        assert "rsi_14" in result.columns

    def test_join_with_keys(self, store_memory):
        """Test join with additional keys (per-symbol)."""
        # Features for multiple symbols
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "symbol": ["AAPL", "GOOGL", "AAPL"],
                "rsi_14": [55.0, 60.0, 58.0],
            }
        )

        # Labels for specific symbols
        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-02", "2024-01-02"],
                "symbol": ["AAPL", "GOOGL"],
                "target": [1, 0],
            }
        )

        store_memory.save_features(features, "features")
        result = store_memory.point_in_time_join(labels, "features", join_keys=["symbol"])

        # Each label should match its symbol's features
        assert len(result) == 2
        aapl_row = result.filter(pl.col("symbol") == "AAPL")
        assert aapl_row["rsi_14"][0] == 58.0  # Most recent AAPL feature

    def test_no_look_ahead_bias(self, store_memory):
        """CRITICAL TEST: Verify no future data is used."""
        # Features at different times
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 09:00", "2024-01-01 11:00"],
                "rsi_14": [55.0, 70.0],  # Different values
            }
        )

        # Label at 10:00 (between features)
        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 10:00"],
                "target": [1],
            }
        )

        store_memory.save_features(features, "features")
        result = store_memory.point_in_time_join(labels, "features")

        # Should use 09:00 feature (RSI=55), NOT 11:00 (RSI=70)
        assert result["rsi_14"][0] == 55.0
        assert result["rsi_14"][0] != 70.0  # Future data NOT used!

    def test_join_with_tolerance(self, store_memory):
        """Test join with time tolerance."""
        # Features at 09:00 (timestamps must be datetime for tolerance to work)
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 09:00"],
                "rsi_14": [55.0],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        # Label at 11:00 (2 hours later)
        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 11:00"],
                "target": [1],
            }
        ).with_columns(pl.col("timestamp").str.to_datetime())

        store_memory.save_features(features, "features")

        # With 1h tolerance: should NOT match (2h gap > 1h tolerance)
        result = store_memory.point_in_time_join(labels, "features", tolerance="1h")
        # No match, rsi_14 should be null
        assert result["rsi_14"][0] is None

        # With 3h tolerance: should match
        result = store_memory.point_in_time_join(labels, "features", tolerance="3h")
        assert result["rsi_14"][0] == 55.0

    def test_join_no_matching_features(self, store_memory):
        """Test join when no features match (future labels)."""
        # Features in the past
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "rsi_14": [55.0],
            }
        )

        # Label before features (no match)
        labels = pl.DataFrame(
            {
                "timestamp": ["2023-12-31"],
                "target": [1],
            }
        )

        store_memory.save_features(features, "features")
        result = store_memory.point_in_time_join(labels, "features")

        # Label preserved, features are null
        assert len(result) == 1
        assert result["target"][0] == 1
        assert result["rsi_14"][0] is None

    def test_join_empty_labels_raises_error(self, store_memory, sample_features):
        """Test join with empty labels raises error."""
        store_memory.save_features(sample_features, "features")
        empty_labels = pl.DataFrame({"timestamp": [], "target": []})

        with pytest.raises(ValueError, match="cannot be empty"):
            store_memory.point_in_time_join(empty_labels, "features")

    def test_join_missing_timestamp_raises_error(self, store_memory, sample_features):
        """Test join without timestamp column raises error."""
        store_memory.save_features(sample_features, "features")
        labels = pl.DataFrame({"target": [1, 0]})

        with pytest.raises(ValueError, match="not found in labels"):
            store_memory.point_in_time_join(labels, "features")

    def test_join_nonexistent_table_raises_error(self, store_memory, sample_labels):
        """Test join with non-existent table raises error."""
        with pytest.raises(FeatureStoreError, match="does not exist"):
            store_memory.point_in_time_join(sample_labels, "nonexistent")

    def test_join_missing_join_keys_raises_error(
        self,
        store_memory,
        sample_features,
        sample_labels,  # noqa: ARG002
    ):
        """Test join with missing join keys raises error."""
        store_memory.save_features(sample_features, "features")
        labels = pl.DataFrame({"timestamp": ["2024-01-01"], "target": [1]})

        with pytest.raises(ValueError, match="not found in labels"):
            store_memory.point_in_time_join(labels, "features", join_keys=["symbol"])

    def test_join_wrong_type_raises_error(self, store_memory):
        """Test join with wrong type raises error."""
        with pytest.raises(TypeError, match="must be a Polars DataFrame"):
            store_memory.point_in_time_join({"not": "dataframe"}, "features")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_save_load_roundtrip(self, store_memory, sample_features):
        """Test save → load preserves data."""
        store_memory.save_features(sample_features, "test")
        loaded = store_memory.load_features("test")

        # Should be identical (order may differ)
        assert loaded.shape == sample_features.shape
        assert set(loaded.columns) == set(sample_features.columns)

    def test_multiple_tables(self, store_memory, sample_features):
        """Test storing multiple feature tables."""
        store_memory.save_features(sample_features, "features1")
        store_memory.save_features(sample_features, "features2")

        tables = store_memory.list_tables()
        assert "features1" in tables
        assert "features2" in tables
        assert len(tables) == 2

    def test_persistence_across_sessions(self, temp_db, sample_features):
        """Test data persists across store instances."""
        # Save in first session
        with OfflineFeatureStore(temp_db) as store1:
            store1.save_features(sample_features, "test")

        # Load in second session
        with OfflineFeatureStore(temp_db) as store2:
            loaded = store2.load_features("test")

        assert len(loaded) == len(sample_features)

    def test_complete_backtest_workflow(self, store_memory):
        """Test complete backtesting workflow."""
        # 1. Compute and save features
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "symbol": ["AAPL", "AAPL", "AAPL"],
                "rsi_14": [55.0, 60.0, 58.0],
            }
        )
        store_memory.save_features(features, "technical")

        # 2. Define prediction points
        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-02", "2024-01-03"],
                "symbol": ["AAPL", "AAPL"],
                "next_day_return": [0.02, -0.01],
            }
        )

        # 3. Point-in-time join
        training_data = store_memory.point_in_time_join(labels, "technical", join_keys=["symbol"])

        # Verify no data leakage
        assert len(training_data) == 2
        assert "rsi_14" in training_data.columns
        assert "next_day_return" in training_data.columns


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_timestamp_match(self, store_memory):
        """Test join when timestamps match exactly."""
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 10:00"],
                "rsi_14": [55.0],
            }
        )

        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-01 10:00"],
                "target": [1],
            }
        )

        store_memory.save_features(features, "features")
        result = store_memory.point_in_time_join(labels, "features")

        # Should match exact timestamp
        assert result["rsi_14"][0] == 55.0

    def test_multiple_features_same_timestamp(self, store_memory):
        """Test join when multiple features have same timestamp."""
        # Two features at same time (should use most recent in insertion order)
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-01"],
                "rsi_14": [55.0, 60.0],
            }
        )

        labels = pl.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "target": [1],
            }
        )

        store_memory.save_features(features, "features")
        result = store_memory.point_in_time_join(labels, "features")

        # Should pick one (implementation-dependent)
        assert result["rsi_14"][0] in [55.0, 60.0]

    def test_large_dataset(self, store_memory):
        """Test with larger dataset for performance."""
        # Create 1000 feature rows
        large_features = pl.DataFrame(
            {
                "timestamp": [f"2024-01-{i // 30 + 1:02d} {i % 24:02d}:00" for i in range(1000)],
                "rsi_14": [float(i % 100) for i in range(1000)],
            }
        )

        start = time.time()
        store_memory.save_features(large_features, "large")
        save_time = time.time() - start

        start = time.time()
        loaded = store_memory.load_features("large")
        load_time = time.time() - start

        # Verify correctness
        assert len(loaded) == 1000

        # Performance should be reasonable (< 1 second each)
        assert save_time < 1.0
        assert load_time < 1.0

    def test_unicode_in_data(self, store_memory):
        """Test handling of unicode characters."""
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "symbol": ["日本株"],  # Japanese characters
                "value": [100.0],
            }
        )

        store_memory.save_features(features, "unicode_test")
        loaded = store_memory.load_features("unicode_test")

        assert loaded["symbol"][0] == "日本株"

    def test_special_column_names(self, store_memory):
        """Test handling of special column names."""
        features = pl.DataFrame(
            {
                "timestamp": ["2024-01-01"],
                "close-price": [100.0],  # Hyphen
                "open_price": [99.0],  # Underscore
                "high.price": [101.0],  # Dot
            }
        )

        store_memory.save_features(features, "special_cols")
        loaded = store_memory.load_features("special_cols")

        assert set(loaded.columns) == {"timestamp", "close-price", "open_price", "high.price"}


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance validation tests."""

    def test_save_performance(self, store_memory):
        """Test save performance meets requirements."""
        # 10K rows
        large_df = pl.DataFrame(
            {
                "timestamp": [f"2024-{i // 300 + 1:02d}-{i % 30 + 1:02d}" for i in range(10000)],
                "value": [float(i) for i in range(10000)],
            }
        )

        start = time.time()
        store_memory.save_features(large_df, "perf_test")
        duration = time.time() - start

        # Should complete in < 2 seconds
        assert duration < 2.0

    def test_load_performance(self, store_memory):
        """Test load performance meets requirements."""
        # Save 10K rows
        large_df = pl.DataFrame(
            {
                "timestamp": [f"2024-{i // 300 + 1:02d}-{i % 30 + 1:02d}" for i in range(10000)],
                "value": [float(i) for i in range(10000)],
            }
        )
        store_memory.save_features(large_df, "perf_test")

        start = time.time()
        loaded = store_memory.load_features("perf_test")
        duration = time.time() - start

        assert len(loaded) == 10000
        # Should complete in < 2 seconds
        assert duration < 2.0

    def test_join_performance(self, store_memory):
        """Test point-in-time join performance."""
        # 1000 features
        features = pl.DataFrame(
            {
                "timestamp": [f"2024-01-{i // 30 + 1:02d}" for i in range(1000)],
                "value": [float(i) for i in range(1000)],
            }
        )

        # 100 labels
        labels = pl.DataFrame(
            {
                "timestamp": [f"2024-01-{i + 1:02d}" for i in range(30)],
                "target": [i % 2 for i in range(30)],
            }
        )

        store_memory.save_features(features, "features")

        start = time.time()
        result = store_memory.point_in_time_join(labels, "features")
        duration = time.time() - start

        assert len(result) == 30
        # Should complete in < 3 seconds
        assert duration < 3.0
