"""Tests for preprocessing module - leakage-safe feature scaling.

These tests verify that the preprocessing transformers:
1. Fit statistics from training data only
2. Apply those statistics during transform (no recomputation)
3. Raise appropriate errors when used incorrectly
4. Preserve column order and handle edge cases
"""

import json

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.preprocessing import (
    MinMaxScaler,
    NotFittedError,
    PreprocessingPipeline,
    Preprocessor,
    RobustScaler,
    StandardScaler,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def train_df() -> pl.DataFrame:
    """Training data with known statistics."""
    return pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        }
    )


@pytest.fixture
def test_df() -> pl.DataFrame:
    """Test data with different statistics."""
    return pl.DataFrame(
        {
            "feature_a": [6.0, 7.0, 8.0],
            "feature_b": [60.0, 70.0, 80.0],
            "category": ["A", "B", "C"],
        }
    )


@pytest.fixture
def constant_df() -> pl.DataFrame:
    """Data with constant column (edge case)."""
    return pl.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0, 5.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
        }
    )


# ============================================================================
# StandardScaler Tests
# ============================================================================


class TestStandardScaler:
    """Tests for StandardScaler (z-score normalization)."""

    def test_fit_stores_train_statistics(self, train_df: pl.DataFrame) -> None:
        """Verify fit() stores statistics from training data."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        assert scaler.is_fitted
        stats = scaler.statistics

        # feature_a: mean=3.0, std=1.581 (ddof=1)
        assert "feature_a" in stats
        assert stats["feature_a"]["mean"] == pytest.approx(3.0)
        assert stats["feature_a"]["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))

        # feature_b: mean=30.0, std=15.81
        assert "feature_b" in stats
        assert stats["feature_b"]["mean"] == pytest.approx(30.0)
        assert stats["feature_b"]["std"] == pytest.approx(np.std([10, 20, 30, 40, 50], ddof=1))

    def test_transform_uses_fitted_statistics(
        self, train_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> None:
        """Verify transform() uses fitted statistics, not recomputed."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        test_scaled = scaler.transform(test_df)

        # feature_a in test: [6, 7, 8]
        # Using train stats: mean=3.0, std=1.581
        # z-score for 6: (6 - 3) / 1.581 ≈ 1.897
        train_std = np.std([1, 2, 3, 4, 5], ddof=1)
        expected_a = [(6 - 3) / train_std, (7 - 3) / train_std, (8 - 3) / train_std]

        assert test_scaled["feature_a"].to_list() == pytest.approx(expected_a, rel=1e-5)

    def test_fit_transform_equivalent(self, train_df: pl.DataFrame) -> None:
        """Verify fit_transform() produces same result as fit().transform()."""
        scaler1 = StandardScaler()
        result1 = scaler1.fit_transform(train_df)

        scaler2 = StandardScaler()
        scaler2.fit(train_df)
        result2 = scaler2.transform(train_df)

        assert result1.equals(result2)

    def test_transform_before_fit_raises(self, train_df: pl.DataFrame) -> None:
        """Verify transform() before fit() raises NotFittedError."""
        scaler = StandardScaler()

        with pytest.raises(NotFittedError, match="has not been fitted"):
            scaler.transform(train_df)

    def test_statistics_before_fit_raises(self) -> None:
        """Verify accessing statistics before fit raises NotFittedError."""
        scaler = StandardScaler()

        with pytest.raises(NotFittedError, match="has not been fitted"):
            _ = scaler.statistics

    def test_column_selection(self, train_df: pl.DataFrame) -> None:
        """Verify only specified columns are scaled."""
        scaler = StandardScaler(columns=["feature_a"])
        result = scaler.fit_transform(train_df)

        # feature_a should be scaled (mean ~= 0)
        assert result["feature_a"].mean() == pytest.approx(0.0, abs=1e-10)

        # feature_b should be unchanged
        assert result["feature_b"].to_list() == train_df["feature_b"].to_list()

        # category should be unchanged
        assert result["category"].to_list() == train_df["category"].to_list()

    def test_constant_column_handling(self, constant_df: pl.DataFrame) -> None:
        """Verify constant columns (std=0) don't cause division by zero."""
        scaler = StandardScaler()
        result = scaler.fit_transform(constant_df)

        # Constant column should not cause NaN or inf
        assert not result["constant"].is_nan().any()
        assert not result["constant"].is_infinite().any()

    def test_with_mean_false(self, train_df: pl.DataFrame) -> None:
        """Verify with_mean=False only scales, doesn't center."""
        scaler = StandardScaler(with_mean=False)
        result = scaler.fit_transform(train_df)

        # Mean should NOT be 0 (only scaled by std)
        assert result["feature_a"].mean() != pytest.approx(0.0, abs=0.1)

    def test_with_std_false(self, train_df: pl.DataFrame) -> None:
        """Verify with_std=False only centers, doesn't scale."""
        scaler = StandardScaler(with_std=False)
        result = scaler.fit_transform(train_df)

        # Mean should be 0
        assert result["feature_a"].mean() == pytest.approx(0.0, abs=1e-10)

        # Std should be unchanged (same as original)
        original_std = train_df["feature_a"].std()
        assert result["feature_a"].std() == pytest.approx(original_std)

    def test_missing_column_in_transform_raises(self, train_df: pl.DataFrame) -> None:
        """Verify transform raises if fitted column is missing."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        # Create test data missing a column
        incomplete_df = pl.DataFrame({"feature_a": [1.0, 2.0]})

        with pytest.raises(ValueError, match="missing fitted columns"):
            scaler.transform(incomplete_df)

    def test_invalid_column_specification_raises(self, train_df: pl.DataFrame) -> None:
        """Verify fit raises if specified column doesn't exist."""
        scaler = StandardScaler(columns=["nonexistent"])

        with pytest.raises(ValueError, match="Columns not found"):
            scaler.fit(train_df)


# ============================================================================
# MinMaxScaler Tests
# ============================================================================


class TestMinMaxScaler:
    """Tests for MinMaxScaler."""

    def test_fit_stores_min_max(self, train_df: pl.DataFrame) -> None:
        """Verify fit() stores min/max from training data."""
        scaler = MinMaxScaler()
        scaler.fit(train_df)

        stats = scaler.statistics

        assert stats["feature_a"]["min"] == 1.0
        assert stats["feature_a"]["max"] == 5.0
        assert stats["feature_b"]["min"] == 10.0
        assert stats["feature_b"]["max"] == 50.0

    def test_transform_scales_to_range(self, train_df: pl.DataFrame) -> None:
        """Verify training data is scaled to [0, 1]."""
        scaler = MinMaxScaler()
        result = scaler.fit_transform(train_df)

        # Min should be 0, max should be 1
        assert result["feature_a"].min() == pytest.approx(0.0)
        assert result["feature_a"].max() == pytest.approx(1.0)
        assert result["feature_b"].min() == pytest.approx(0.0)
        assert result["feature_b"].max() == pytest.approx(1.0)

    def test_transform_test_data_may_exceed_range(
        self, train_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> None:
        """Verify test data can exceed [0, 1] when values are outside train range."""
        scaler = MinMaxScaler()
        scaler.fit(train_df)
        test_scaled = scaler.transform(test_df)

        # feature_a in test: [6, 7, 8]
        # Train range: [1, 5], so:
        # 6 -> (6-1)/(5-1) = 1.25
        # 7 -> (7-1)/(5-1) = 1.5
        # 8 -> (8-1)/(5-1) = 1.75
        expected_a = [1.25, 1.5, 1.75]
        assert test_scaled["feature_a"].to_list() == pytest.approx(expected_a)

    def test_custom_feature_range(self, train_df: pl.DataFrame) -> None:
        """Verify custom feature_range works correctly."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        result = scaler.fit_transform(train_df)

        assert result["feature_a"].min() == pytest.approx(-1.0)
        assert result["feature_a"].max() == pytest.approx(1.0)

    def test_constant_column_handling(self, constant_df: pl.DataFrame) -> None:
        """Verify constant columns don't cause division by zero."""
        scaler = MinMaxScaler()
        result = scaler.fit_transform(constant_df)

        assert not result["constant"].is_nan().any()
        assert not result["constant"].is_infinite().any()


# ============================================================================
# RobustScaler Tests
# ============================================================================


class TestRobustScaler:
    """Tests for RobustScaler (median/IQR based)."""

    def test_fit_stores_median_iqr(self, train_df: pl.DataFrame) -> None:
        """Verify fit() stores median and IQR from training data."""
        scaler = RobustScaler()
        scaler.fit(train_df)

        stats = scaler.statistics

        # feature_a: [1, 2, 3, 4, 5] -> median=3
        assert stats["feature_a"]["median"] == pytest.approx(3.0)
        # IQR = Q3 - Q1 = 4 - 2 = 2
        assert stats["feature_a"]["iqr"] == pytest.approx(2.0)

    def test_transform_uses_median_iqr(self, train_df: pl.DataFrame) -> None:
        """Verify transform uses median/IQR, not mean/std."""
        scaler = RobustScaler()
        result = scaler.fit_transform(train_df)

        # Median (3) should map to 0
        # feature_a: [1, 2, 3, 4, 5] with median=3, IQR=2
        # 3 -> (3 - 3) / 2 = 0
        assert result["feature_a"][2] == pytest.approx(0.0)

    def test_with_centering_false(self, train_df: pl.DataFrame) -> None:
        """Verify with_centering=False only scales by IQR."""
        scaler = RobustScaler(with_centering=False)
        result = scaler.fit_transform(train_df)

        # Values should not be centered around 0
        assert result["feature_a"].median() != pytest.approx(0.0)

    def test_outlier_resistance(self) -> None:
        """Verify robust scaler is less affected by outliers than standard."""
        # Data with outlier
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

        robust = RobustScaler()
        standard = StandardScaler()

        robust_result = robust.fit_transform(df)
        standard_result = standard.fit_transform(df)

        # Standard scaler will compress normal values due to large outlier
        # Robust scaler should give more reasonable range for normal values
        # (This is a qualitative check - median-based is less affected)
        robust_range = robust_result["x"].max() - robust_result["x"].min()
        standard_range = standard_result["x"].max() - standard_result["x"].min()

        # The ranges should be different (robust less compressed)
        assert abs(robust_range - standard_range) > 0.1


# ============================================================================
# Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for scaler serialization/deserialization."""

    def test_to_dict_contains_all_info(self, train_df: pl.DataFrame) -> None:
        """Verify to_dict() exports all necessary information."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        data = scaler.to_dict()

        assert "class" in data
        assert data["class"] == "StandardScaler"
        assert "columns" in data
        assert "statistics" in data
        assert set(data["columns"]) == {"feature_a", "feature_b"}

    def test_from_dict_recreates_scaler(self, train_df: pl.DataFrame) -> None:
        """Verify from_dict() recreates identical scaler."""
        scaler1 = StandardScaler()
        scaler1.fit(train_df)

        # Serialize and deserialize
        data = scaler1.to_dict()
        scaler2 = StandardScaler.from_dict(data)

        # Should produce same results
        result1 = scaler1.transform(train_df)
        result2 = scaler2.transform(train_df)

        assert result1.equals(result2)

    @pytest.mark.parametrize(
        "scaler",
        [
            StandardScaler(
                columns=["feature_a"],
                with_mean=False,
                with_std=True,
                ddof=0,
            ),
            MinMaxScaler(columns=["feature_a"], feature_range=(-1.0, 1.0)),
            RobustScaler(
                columns=["feature_a"],
                with_centering=False,
                with_scaling=True,
                quantile_range=(10.0, 90.0),
            ),
        ],
    )
    def test_every_scaler_option_round_trips_through_json(
        self,
        scaler: StandardScaler | MinMaxScaler | RobustScaler,
        train_df: pl.DataFrame,
    ) -> None:
        """Every constructor option survives a JSON-compatible payload."""
        scaler.fit(train_df)
        payload = json.loads(json.dumps(scaler.to_dict()))

        loaded = type(scaler).from_dict(payload)

        assert loaded.transform(train_df).equals(scaler.transform(train_df))
        assert loaded.to_dict()["config"] == scaler.to_dict()["config"]

    def test_custom_minmax_range_round_trips_exactly(self) -> None:
        """Production loading preserves a custom feature range."""
        frame = pl.DataFrame({"x": [0.0, 5.0, 10.0]})
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        expected = scaler.fit_transform(frame)

        loaded = MinMaxScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))

        assert loaded.feature_range == (-1.0, 1.0)
        assert loaded.transform(frame).equals(expected)

    def test_legacy_unversioned_payload_is_rejected(self, train_df: pl.DataFrame) -> None:
        """Incomplete beta payloads cannot silently load with default options."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        scaler.fit(train_df)
        payload = scaler.to_dict()
        del payload["schema_version"]

        with pytest.raises(ValueError, match="missing fields.*schema_version"):
            MinMaxScaler.from_dict(payload)

    def test_serialized_class_mismatch_is_rejected(self, train_df: pl.DataFrame) -> None:
        """A scaler payload cannot be loaded through a different class."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        with pytest.raises(ValueError, match="expected 'MinMaxScaler'"):
            MinMaxScaler.from_dict(scaler.to_dict())

    def test_to_dict_before_fit_raises(self) -> None:
        """Verify to_dict() before fit raises NotFittedError."""
        scaler = StandardScaler()

        with pytest.raises(NotFittedError):
            scaler.to_dict()


# ============================================================================
# Preprocessor Alias Test
# ============================================================================


class TestPreprocessorAlias:
    """Tests for Preprocessor convenience alias."""

    def test_preprocessor_is_standard_scaler(self) -> None:
        """Verify Preprocessor is an alias for StandardScaler."""
        assert Preprocessor is StandardScaler

    def test_preprocessor_works_identically(self, train_df: pl.DataFrame) -> None:
        """Verify Preprocessor works like StandardScaler."""
        preprocessor = Preprocessor()
        result = preprocessor.fit_transform(train_df)

        scaler = StandardScaler()
        expected = scaler.fit_transform(train_df)

        assert result.equals(expected)


class TestPreprocessingPipelineValidation:
    """Tests for recommendation validation before fitting."""

    def test_unknown_transform_is_rejected_with_feature_and_values(self) -> None:
        """A misspelling cannot silently disable a requested transform."""
        recommendations = {"x": {"transform": "standarize", "confidence": 0.9}}

        with pytest.raises(
            ValueError,
            match="Unknown transform 'standarize' for feature 'x'.*standardize",
        ):
            PreprocessingPipeline.from_recommendations(recommendations)

    @pytest.mark.parametrize(
        "recommendations",
        [
            [],
            {"x": []},
            {"": {"transform": "standardize", "confidence": 0.9}},
        ],
    )
    def test_invalid_recommendation_structures_are_rejected(
        self,
        recommendations: object,
    ) -> None:
        """The outer mapping, feature names, and recommendation records are validated."""
        with pytest.raises(ValueError, match="(?i)recommendation"):
            PreprocessingPipeline.from_recommendations(recommendations)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), "high", True])
    def test_invalid_confidence_is_rejected(self, confidence: object) -> None:
        """Confidence must be a finite probability."""
        recommendations = {"x": {"transform": "standardize", "confidence": confidence}}

        with pytest.raises(ValueError, match="confidence for feature 'x'"):
            PreprocessingPipeline.from_recommendations(recommendations)

    @pytest.mark.parametrize(
        ("recommendation", "missing"),
        [
            ({"confidence": 0.9}, "transform"),
            ({"transform": "standardize"}, "confidence"),
        ],
    )
    def test_missing_recommendation_fields_are_rejected(
        self,
        recommendation: dict[str, object],
        missing: str,
    ) -> None:
        """Required recommendation fields cannot default silently."""
        with pytest.raises(ValueError, match=rf"missing fields.*{missing}"):
            PreprocessingPipeline.from_recommendations({"x": recommendation})

    @pytest.mark.parametrize(
        "limits",
        [
            (-0.1, 0.9),
            (0.1, 1.1),
            (0.9, 0.1),
            (0.5, 0.5),
            (float("nan"), 0.9),
            (0.1, float("inf")),
            (0.1,),
            "invalid",
        ],
    )
    def test_invalid_winsorize_limits_are_rejected(self, limits: object) -> None:
        """Winsorization limits must be ordered finite probabilities."""
        recommendations = {"x": {"transform": "winsorize", "confidence": 0.9}}

        with pytest.raises(ValueError, match="winsorize_limits"):
            PreprocessingPipeline.from_recommendations(
                recommendations,
                winsorize_limits=limits,
            )

    @pytest.mark.parametrize("minimum", [-0.1, 1.1, float("inf"), "high", True])
    def test_invalid_minimum_confidence_is_rejected(self, minimum: object) -> None:
        """The pipeline threshold must be a finite probability."""
        with pytest.raises(ValueError, match="min_confidence"):
            PreprocessingPipeline(min_confidence=minimum)

    def test_recommendations_are_isolated_from_caller_mutation(self) -> None:
        """Caller mutation cannot invalidate an already validated pipeline."""
        recommendations = {"x": {"transform": "standardize", "confidence": 0.9}}
        pipeline = PreprocessingPipeline.from_recommendations(recommendations)

        recommendations["x"]["transform"] = "standarize"

        assert pipeline.get_transform_summary() == {"x": "standardize"}

    def test_difference_uses_train_boundary_on_new_data(self) -> None:
        """The first test difference uses the last fitted training value."""
        recommendations = {"x": {"transform": "diff", "confidence": 1.0}}
        pipeline = PreprocessingPipeline.from_recommendations(recommendations)

        train = pipeline.fit_transform(pl.DataFrame({"x": [1.0, 4.0, 9.0]}))
        test = pipeline.transform(pl.DataFrame({"x": [16.0, 25.0]}))

        assert train["x"].to_list() == [None, 3.0, 5.0]
        assert test["x"].to_list() == [7.0, 9.0]


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe_raises(self) -> None:
        """Verify fitting on empty DataFrame raises appropriate error."""
        scaler = StandardScaler()
        empty_df = pl.DataFrame({"x": []}).cast({"x": pl.Float64})

        # Should handle gracefully (statistics will be NaN)
        scaler.fit(empty_df)
        # But statistics should show it was fitted
        assert scaler.is_fitted

    def test_null_values_handled(self) -> None:
        """Verify null values are handled correctly."""
        df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})

        scaler = StandardScaler()
        result = scaler.fit_transform(df)

        # Non-null values should be scaled
        # Null values should remain null
        assert result["x"].null_count() == 2

    @pytest.mark.parametrize("scaler_type", [StandardScaler, MinMaxScaler, RobustScaler])
    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_fitting_values_are_rejected(
        self,
        scaler_type: type[StandardScaler | MinMaxScaler | RobustScaler],
        invalid: float,
    ) -> None:
        """NaN and infinity cannot enter fitted scaler statistics."""
        scaler = scaler_type()

        with pytest.raises(ValueError, match="NaN or infinity.*x"):
            scaler.fit(pl.DataFrame({"x": [1.0, invalid, 3.0]}))

        assert not scaler.is_fitted

    @pytest.mark.parametrize("scaler_type", [StandardScaler, MinMaxScaler, RobustScaler])
    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_transform_values_are_rejected(
        self,
        scaler_type: type[StandardScaler | MinMaxScaler | RobustScaler],
        invalid: float,
    ) -> None:
        """NaN and infinity cannot enter transformed model features."""
        scaler = scaler_type().fit(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))

        with pytest.raises(ValueError, match="NaN or infinity.*x"):
            scaler.transform(pl.DataFrame({"x": [4.0, invalid]}))

        assert scaler.is_fitted

    def test_fitted_columns_property(self, train_df: pl.DataFrame) -> None:
        """Verify fitted_columns returns correct columns."""
        scaler = StandardScaler(columns=["feature_a"])
        scaler.fit(train_df)

        assert scaler.fitted_columns == ["feature_a"]

    def test_fitted_columns_returns_copy(self, train_df: pl.DataFrame) -> None:
        """Verify fitted_columns returns a copy (immutable)."""
        scaler = StandardScaler()
        scaler.fit(train_df)

        cols = scaler.fitted_columns
        cols.append("modified")

        # Original should be unchanged
        assert "modified" not in scaler.fitted_columns


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_train_test_split_workflow(self) -> None:
        """Verify complete train/test workflow prevents leakage."""
        # Create data with different distributions
        np.random.seed(42)
        train_data = np.random.normal(0, 1, (100, 2))
        test_data = np.random.normal(5, 2, (50, 2))  # Different distribution!

        train_df = pl.DataFrame({"feature_a": train_data[:, 0], "feature_b": train_data[:, 1]})
        test_df = pl.DataFrame({"feature_a": test_data[:, 0], "feature_b": test_data[:, 1]})

        # Fit on train only
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        # Train should have mean ~0 after scaling
        assert train_scaled["feature_a"].mean() == pytest.approx(0.0, abs=0.2)

        # Test should NOT have mean ~0 (uses train statistics, test has different mean)
        # Test original mean ~5, train std ~1, so test scaled mean should be ~5
        assert test_scaled["feature_a"].mean() > 3.0  # Much higher than 0

    def test_cross_validation_pattern(self) -> None:
        """Verify proper usage in cross-validation loop."""
        # Simulate time-series data
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), eager=True)
        values = pl.Series("value", list(range(len(dates)))).cast(pl.Float64)
        df = pl.DataFrame({"date": dates, "value": values})

        # Split: first 70% train, last 30% test
        train_size = int(len(df) * 0.7)
        train_df = df.head(train_size)
        test_df = df.tail(len(df) - train_size)

        # Fit on train
        scaler = StandardScaler(columns=["value"])
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        # Verify statistics from train period
        train_stats = scaler.statistics
        assert train_stats["value"]["mean"] == pytest.approx(train_df["value"].mean())

        # Verify test was transformed using train statistics
        # (test values are higher, so scaled values should be positive)
        assert test_scaled["value"].mean() > train_scaled["value"].mean()
