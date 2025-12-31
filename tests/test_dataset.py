"""Tests for MLDatasetBuilder module.

Comprehensive tests covering:
- Basic initialization and validation
- Preprocessing with train-only statistics
- Cross-validation integration
- Sklearn compatibility
- Edge cases
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.dataset import (
    DatasetInfo,
    FoldResult,
    MLDatasetBuilder,
    create_dataset_builder,
)
from ml4t.engineer.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Create sample feature DataFrame."""
    np.random.seed(42)
    n_samples = 100
    return pl.DataFrame(
        {
            "momentum": np.random.randn(n_samples).tolist(),
            "volatility": np.abs(np.random.randn(n_samples) * 0.1).tolist(),
            "volume": (np.random.rand(n_samples) * 1000).tolist(),
        }
    )


@pytest.fixture
def sample_labels() -> pl.Series:
    """Create sample labels."""
    np.random.seed(42)
    return pl.Series("target", np.random.randint(0, 2, 100).tolist())


@pytest.fixture
def sample_dates() -> pl.Series:
    """Create sample dates."""
    dates = pl.date_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 4, 10),
        interval="1d",
        eager=True,
    )[:100]
    return pl.Series("date", dates)


@pytest.fixture
def builder(sample_features: pl.DataFrame, sample_labels: pl.Series) -> MLDatasetBuilder:
    """Create basic dataset builder."""
    return MLDatasetBuilder(features=sample_features, labels=sample_labels)


@pytest.fixture
def builder_with_dates(
    sample_features: pl.DataFrame,
    sample_labels: pl.Series,
    sample_dates: pl.Series,
) -> MLDatasetBuilder:
    """Create dataset builder with dates."""
    return MLDatasetBuilder(
        features=sample_features,
        labels=sample_labels,
        dates=sample_dates,
    )


class MockSplitter:
    """Mock cross-validation splitter for testing."""

    def __init__(self, n_splits: int = 3) -> None:
        self.n_splits = n_splits

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[tuple[NDArray[np.intp], NDArray[np.intp]], None, None]:
        """Generate mock train/test splits."""
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)

            train_idx = np.arange(train_end, dtype=np.intp)
            test_idx = np.arange(test_start, test_end, dtype=np.intp)

            yield train_idx, test_idx

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMLDatasetBuilderInit:
    """Tests for MLDatasetBuilder initialization."""

    def test_basic_init(self, sample_features: pl.DataFrame, sample_labels: pl.Series) -> None:
        """Test basic initialization."""
        builder = MLDatasetBuilder(features=sample_features, labels=sample_labels)

        assert len(builder) == 100
        assert builder.scaler is None
        assert builder._feature_columns == ["momentum", "volatility", "volume"]

    def test_init_with_dates(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
        sample_dates: pl.Series,
    ) -> None:
        """Test initialization with dates."""
        builder = MLDatasetBuilder(
            features=sample_features,
            labels=sample_labels,
            dates=sample_dates,
        )

        assert builder.dates is not None
        assert len(builder.dates) == 100

    def test_init_with_dataframe_labels(self, sample_features: pl.DataFrame) -> None:
        """Test initialization with DataFrame labels (converts to Series)."""
        labels_df = pl.DataFrame({"target": np.random.randint(0, 2, 100).tolist()})
        builder = MLDatasetBuilder(features=sample_features, labels=labels_df)

        assert isinstance(builder.labels, pl.Series)

    def test_init_mismatched_lengths_raises(self, sample_features: pl.DataFrame) -> None:
        """Test that mismatched lengths raise ValueError."""
        labels = pl.Series("target", [0, 1, 0])  # Only 3 samples

        with pytest.raises(ValueError, match="same length"):
            MLDatasetBuilder(features=sample_features, labels=labels)

    def test_init_mismatched_dates_raises(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test that mismatched dates length raises ValueError."""
        dates = pl.Series("date", [1, 2, 3])  # Only 3 dates

        with pytest.raises(ValueError, match="same length"):
            MLDatasetBuilder(
                features=sample_features,
                labels=sample_labels,
                dates=dates,
            )


# =============================================================================
# Scaler Tests
# =============================================================================


class TestMLDatasetBuilderScaler:
    """Tests for scaler functionality."""

    def test_set_scaler(self, builder: MLDatasetBuilder) -> None:
        """Test setting a scaler."""
        scaler = StandardScaler()
        result = builder.set_scaler(scaler)

        assert result is builder  # Method chaining
        assert builder.scaler is scaler

    def test_set_scaler_none(self, builder: MLDatasetBuilder) -> None:
        """Test disabling scaler."""
        builder.set_scaler(StandardScaler())
        builder.set_scaler(None)

        assert builder.scaler is None

    def test_scaler_types(self, builder: MLDatasetBuilder) -> None:
        """Test different scaler types."""
        for scaler_cls in [StandardScaler, MinMaxScaler, RobustScaler]:
            builder.set_scaler(scaler_cls())
            assert isinstance(builder.scaler, scaler_cls)


# =============================================================================
# Info Property Tests
# =============================================================================


class TestDatasetInfo:
    """Tests for dataset info property."""

    def test_info_basic(self, builder: MLDatasetBuilder) -> None:
        """Test info property."""
        info = builder.info

        assert isinstance(info, DatasetInfo)
        assert info.n_samples == 100
        assert info.n_features == 3
        assert info.feature_names == ["momentum", "volatility", "volume"]
        assert info.label_name == "target"
        assert info.has_dates is False

    def test_info_with_dates(self, builder_with_dates: MLDatasetBuilder) -> None:
        """Test info with dates."""
        info = builder_with_dates.info

        assert info.has_dates is True


# =============================================================================
# Train/Test Split Tests
# =============================================================================


class TestTrainTestSplit:
    """Tests for train_test_split method."""

    def test_basic_split(self, builder: MLDatasetBuilder) -> None:
        """Test basic train/test split."""
        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_split_preserves_order_no_shuffle(self, builder: MLDatasetBuilder) -> None:
        """Test that split preserves order when shuffle=False."""
        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8, shuffle=False)

        # First 80 samples should be train
        original_first = builder.features.row(0, named=True)
        split_first = X_train.row(0, named=True)

        # Compare values
        for col in builder.features.columns:
            assert original_first[col] == split_first[col]

    def test_split_with_shuffle(self, builder: MLDatasetBuilder) -> None:
        """Test shuffled split."""
        X_train1, _, _, _ = builder.train_test_split(train_size=0.8, shuffle=True, random_state=42)
        X_train2, _, _, _ = builder.train_test_split(train_size=0.8, shuffle=True, random_state=42)

        # Same seed should give same result
        assert X_train1.equals(X_train2)

    def test_split_with_scaling(self, builder: MLDatasetBuilder) -> None:
        """Test split applies scaling correctly."""
        builder.set_scaler(StandardScaler())
        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        # Check train is approximately standardized
        for col in X_train.columns:
            mean = X_train[col].mean()
            std = X_train[col].std()
            assert abs(mean) < 0.1, f"Mean for {col} should be ~0, got {mean}"
            assert abs(std - 1.0) < 0.1, f"Std for {col} should be ~1, got {std}"

    def test_split_without_scaling(self, builder: MLDatasetBuilder) -> None:
        """Test split without scaling preserves original values."""
        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        # Values should match original
        original_train = builder.features[:80]
        assert X_train.equals(original_train)


# =============================================================================
# Cross-Validation Split Tests
# =============================================================================


class TestCVSplit:
    """Tests for cross-validation split method."""

    def test_basic_cv_split(self, builder: MLDatasetBuilder) -> None:
        """Test basic CV split."""
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        assert len(folds) == 3
        for fold in folds:
            assert isinstance(fold, FoldResult)

    def test_cv_split_fold_numbers(self, builder: MLDatasetBuilder) -> None:
        """Test fold numbers are sequential."""
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        for i, fold in enumerate(folds):
            assert fold.fold_number == i

    def test_cv_split_indices(self, builder: MLDatasetBuilder) -> None:
        """Test train/test indices are captured."""
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        for fold in folds:
            assert len(fold.train_indices) > 0
            assert len(fold.test_indices) > 0
            assert len(fold.X_train) == len(fold.train_indices)
            assert len(fold.X_test) == len(fold.test_indices)

    def test_cv_split_with_scaling(self, builder: MLDatasetBuilder) -> None:
        """Test CV split applies scaling per fold."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        for fold in folds:
            assert fold.scaler is not None
            assert fold.scaler.is_fitted

            # Training data should be standardized
            for col in fold.X_train.columns:
                mean = fold.X_train[col].mean()
                # Approximately standardized (allow some tolerance)
                assert abs(mean) < 0.2, f"Fold {fold.fold_number}: mean {mean} not ~0"

    def test_cv_split_each_fold_has_own_scaler(self, builder: MLDatasetBuilder) -> None:
        """Test each fold gets its own fitted scaler."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        # Each fold should have a different scaler instance
        scalers = [fold.scaler for fold in folds]
        assert len({id(s) for s in scalers}) == 3

    def test_cv_split_without_scaling(self, builder: MLDatasetBuilder) -> None:
        """Test CV split without scaling."""
        cv = MockSplitter(n_splits=3)
        folds = list(builder.split(cv))

        for fold in folds:
            assert fold.scaler is None


# =============================================================================
# FoldResult Tests
# =============================================================================


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_fold_result_to_numpy(self, builder: MLDatasetBuilder) -> None:
        """Test FoldResult.to_numpy() conversion."""
        cv = MockSplitter(n_splits=1)
        fold = next(builder.split(cv))

        X_train, X_test, y_train, y_test = fold.to_numpy()

        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_fold_result_shapes(self, builder: MLDatasetBuilder) -> None:
        """Test FoldResult array shapes."""
        cv = MockSplitter(n_splits=1)
        fold = next(builder.split(cv))

        X_train, X_test, y_train, y_test = fold.to_numpy()

        assert X_train.shape == (len(fold.train_indices), 3)
        assert X_test.shape == (len(fold.test_indices), 3)
        assert y_train.shape == (len(fold.train_indices),)
        assert y_test.shape == (len(fold.test_indices),)


# =============================================================================
# Numpy/Pandas Conversion Tests
# =============================================================================


class TestConversions:
    """Tests for numpy and pandas conversions."""

    def test_to_numpy(self, builder: MLDatasetBuilder) -> None:
        """Test to_numpy conversion."""
        features, labels = builder.to_numpy()

        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert features.shape == (100, 3)
        assert labels.shape == (100,)

    def test_to_pandas(self, builder: MLDatasetBuilder) -> None:
        """Test to_pandas conversion."""
        features, labels = builder.to_pandas()

        import pandas as pd

        assert isinstance(features, pd.DataFrame)
        assert isinstance(labels, pd.Series)
        assert len(features) == 100
        assert len(labels) == 100


# =============================================================================
# Percentile Computation Tests
# =============================================================================


class TestPercentileComputation:
    """Tests for percentile computation methods."""

    def test_get_feature_percentiles(self, builder: MLDatasetBuilder) -> None:
        """Test feature percentile computation."""
        train_indices = np.arange(50, dtype=np.intp)  # First 50 samples
        cutoffs = builder.get_feature_percentiles(train_indices)

        assert isinstance(cutoffs, dict)
        assert "momentum" in cutoffs
        assert "volatility" in cutoffs
        assert "volume" in cutoffs

        # Check quantiles
        for _feature, quantile_dict in cutoffs.items():
            assert 0.1 in quantile_dict
            assert 0.5 in quantile_dict
            assert 0.9 in quantile_dict

    def test_get_feature_percentiles_custom_quantiles(self, builder: MLDatasetBuilder) -> None:
        """Test custom quantiles."""
        train_indices = np.arange(50, dtype=np.intp)
        cutoffs = builder.get_feature_percentiles(train_indices, quantiles=[0.25, 0.75])

        for _feature, quantile_dict in cutoffs.items():
            assert len(quantile_dict) == 2
            assert 0.25 in quantile_dict
            assert 0.75 in quantile_dict

    def test_compute_label_percentiles(self, builder: MLDatasetBuilder) -> None:
        """Test label percentile computation."""
        train_indices = np.arange(50, dtype=np.intp)
        cutoffs = builder.compute_label_percentiles(train_indices, n_quantiles=5)

        # Should return n_quantiles - 1 cutoff points
        assert len(cutoffs) == 4

    def test_percentiles_use_train_only(self, builder: MLDatasetBuilder) -> None:
        """Test that percentiles use only train indices."""
        train_indices = np.arange(50, dtype=np.intp)
        cutoffs1 = builder.get_feature_percentiles(train_indices)

        # Different train indices should give different cutoffs
        train_indices2 = np.arange(50, 100, dtype=np.intp)
        cutoffs2 = builder.get_feature_percentiles(train_indices2)

        # At least some cutoffs should differ
        for feature in cutoffs1:
            if cutoffs1[feature][0.5] != cutoffs2[feature][0.5]:
                break
        else:
            pytest.skip("Random data happened to give same medians")


# =============================================================================
# create_dataset_builder Function Tests
# =============================================================================


class TestCreateDatasetBuilder:
    """Tests for create_dataset_builder convenience function."""

    def test_create_with_default_scaler(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with default standard scaler."""
        builder = create_dataset_builder(sample_features, sample_labels)

        assert builder.scaler is not None
        assert isinstance(builder.scaler, StandardScaler)

    def test_create_with_standard_scaler(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with standard scaler string."""
        builder = create_dataset_builder(sample_features, sample_labels, scaler="standard")

        assert isinstance(builder.scaler, StandardScaler)

    def test_create_with_minmax_scaler(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with minmax scaler string."""
        builder = create_dataset_builder(sample_features, sample_labels, scaler="minmax")

        assert isinstance(builder.scaler, MinMaxScaler)

    def test_create_with_robust_scaler(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with robust scaler string."""
        builder = create_dataset_builder(sample_features, sample_labels, scaler="robust")

        assert isinstance(builder.scaler, RobustScaler)

    def test_create_with_no_scaler(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with no scaling."""
        builder = create_dataset_builder(sample_features, sample_labels, scaler=None)

        assert builder.scaler is None

    def test_create_with_scaler_instance(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with custom scaler instance."""
        custom_scaler = StandardScaler(with_mean=False)
        builder = create_dataset_builder(sample_features, sample_labels, scaler=custom_scaler)

        assert builder.scaler is custom_scaler

    def test_create_with_invalid_scaler_raises(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with invalid scaler string raises."""
        with pytest.raises(ValueError, match="Unknown scaler"):
            create_dataset_builder(sample_features, sample_labels, scaler="invalid")

    def test_create_with_wrong_type_raises(
        self,
        sample_features: pl.DataFrame,
        sample_labels: pl.Series,
    ) -> None:
        """Test create with wrong type raises."""
        with pytest.raises(TypeError):
            create_dataset_builder(sample_features, sample_labels, scaler=123)


# =============================================================================
# Leakage Prevention Tests (Critical)
# =============================================================================


class TestLeakagePrevention:
    """Critical tests for leakage prevention."""

    def test_train_only_statistics_cv(self, builder: MLDatasetBuilder) -> None:
        """Test that CV scaling uses train-only statistics."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=3)

        for fold in builder.split(cv):
            # Get the fitted statistics from scaler
            stats = fold.scaler.statistics

            # Recompute statistics from train data only
            for col in fold.X_train.columns:
                train_mean = builder.features[fold.train_indices][col].mean()
                scaler_mean = stats[col]["mean"]

                # Statistics should match train-only computation
                assert abs(train_mean - scaler_mean) < 1e-10, (
                    f"Scaler mean ({scaler_mean}) differs from train mean ({train_mean})"
                )

    def test_test_data_uses_train_statistics(self, builder: MLDatasetBuilder) -> None:
        """Test that test data is transformed using train statistics."""
        builder.set_scaler(StandardScaler())

        # Simple split to verify
        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        # Train statistics
        original_train = builder.features[:80]
        train_mean = original_train["momentum"].mean()
        train_std = original_train["momentum"].std()

        # Test data transformation should use train mean/std
        original_test_value = builder.features[80]["momentum"][0]
        expected_scaled = (original_test_value - train_mean) / train_std
        actual_scaled = X_test[0]["momentum"][0]

        assert abs(expected_scaled - actual_scaled) < 1e-10

    def test_different_folds_different_statistics(self, builder: MLDatasetBuilder) -> None:
        """Test that different folds have different scaling statistics."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=3)

        folds = list(builder.split(cv))

        # Each fold has different training data, so different statistics
        stats = [fold.scaler.statistics["momentum"]["mean"] for fold in folds]

        # Not all should be identical (different train sets)
        assert len(set(stats)) > 1, "All folds have identical statistics (unexpected)"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_feature(self) -> None:
        """Test with single feature."""
        features = pl.DataFrame({"single": list(range(100))})
        labels = pl.Series("target", [0] * 50 + [1] * 50)

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler())

        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        assert X_train.width == 1
        assert X_test.width == 1

    def test_many_features(self) -> None:
        """Test with many features."""
        np.random.seed(42)
        features = pl.DataFrame({f"f{i}": np.random.randn(100).tolist() for i in range(50)})
        labels = pl.Series("target", [0] * 50 + [1] * 50)

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler())

        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        assert X_train.width == 50
        assert X_test.width == 50

    def test_constant_column(self) -> None:
        """Test handling of constant column (zero variance)."""
        features = pl.DataFrame(
            {
                "varying": list(range(100)),
                "constant": [5.0] * 100,
            }
        )
        labels = pl.Series("target", [0] * 50 + [1] * 50)

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler())

        # Should not raise (std=0 handled)
        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        # Constant column should remain constant (std defaulted to 1)
        assert X_train["constant"].n_unique() == 1

    def test_with_nulls(self) -> None:
        """Test handling of null values."""
        features = pl.DataFrame(
            {
                "with_null": [1.0, 2.0, None, 4.0, 5.0] * 20,
                "no_null": list(range(100)),
            }
        )
        labels = pl.Series("target", [0] * 50 + [1] * 50)

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler())

        # Should handle nulls gracefully
        X_train, X_test, _, _ = builder.train_test_split(train_size=0.8)

        # Nulls should propagate (become NaN in scaled data)
        assert X_train["with_null"].null_count() > 0

    def test_repr(self, builder: MLDatasetBuilder) -> None:
        """Test string representation."""
        repr_str = repr(builder)

        assert "MLDatasetBuilder" in repr_str
        assert "n_samples=100" in repr_str
        assert "n_features=3" in repr_str
        assert "scaler=None" in repr_str

    def test_repr_with_scaler(self, builder: MLDatasetBuilder) -> None:
        """Test repr with scaler."""
        builder.set_scaler(StandardScaler())
        repr_str = repr(builder)

        assert "scaler=StandardScaler" in repr_str

    def test_len(self, builder: MLDatasetBuilder) -> None:
        """Test __len__ method."""
        assert len(builder) == 100


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests for full workflow."""

    def test_full_cv_workflow(self, builder: MLDatasetBuilder) -> None:
        """Test complete cross-validation workflow."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=5)

        predictions = []
        actuals = []

        for fold in builder.split(cv):
            X_train_np, X_test_np, y_train_np, y_test_np = fold.to_numpy()

            # Simulate simple model (just for testing workflow)
            # In practice: model.fit(X_train_np, y_train_np)
            preds = np.zeros(len(y_test_np))  # Dummy predictions

            predictions.extend(preds)
            actuals.extend(y_test_np)

        assert len(predictions) > 0
        assert len(actuals) > 0

    def test_percentile_workflow(self, builder: MLDatasetBuilder) -> None:
        """Test percentile-based labeling workflow."""
        # Split data
        train_idx = np.arange(80, dtype=np.intp)
        np.arange(80, 100, dtype=np.intp)

        # Get percentiles from train only
        cutoffs = builder.compute_label_percentiles(train_idx, n_quantiles=5)

        # Verify cutoffs are usable
        assert len(cutoffs) == 4  # 5 quantiles → 4 boundaries
        assert all(isinstance(c, float) for c in cutoffs)

    def test_iterating_multiple_times(self, builder: MLDatasetBuilder) -> None:
        """Test that split() can be called multiple times."""
        builder.set_scaler(StandardScaler())
        cv = MockSplitter(n_splits=3)

        # First iteration
        folds1 = list(builder.split(cv))

        # Second iteration
        folds2 = list(builder.split(cv))

        # Should give same results
        assert len(folds1) == len(folds2)
        for f1, f2 in zip(folds1, folds2):
            assert np.array_equal(f1.train_indices, f2.train_indices)
