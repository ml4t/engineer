"""Tests for the pipeline engine DAG functionality."""

import polars as pl
import pytest

from ml4t.engineer.pipeline.engine import Pipeline, PipelineStep


class TestPipelineEngine:
    """Test the DAG-based pipeline engine."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "price": [
                    100.0,
                    102.0,
                    101.0,
                    103.0,
                    104.0,
                    102.0,
                    105.0,
                    106.0,
                    104.0,
                    107.0,
                ],
                "volume": [1000, 1200, 800, 1500, 1100, 900, 1300, 1400, 1000, 1600],
            },
        )

    def test_simple_pipeline_no_dependencies(self, sample_data: pl.DataFrame) -> None:
        """Test pipeline with steps that have no dependencies."""

        def add_returns(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(returns=pl.col("price").pct_change())

        def add_log_returns(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(log_returns=pl.col("price").log().diff())

        pipeline = Pipeline(
            [("returns", add_returns), ("log_returns", add_log_returns)],
        )

        result = pipeline.run(sample_data)

        # Check that both columns were added
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert len(result) == len(sample_data)

    def test_pipeline_with_dependencies(self, sample_data: pl.DataFrame) -> None:
        """Test pipeline with step dependencies."""

        def add_returns(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(returns=pl.col("price").pct_change())

        def add_volatility(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                volatility=pl.col("returns").rolling_std(window_size=3),
            )

        def add_sharpe(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                sharpe=pl.col("returns").mean() / pl.col("volatility"),
            )

        # Create pipeline with dependencies
        pipeline = Pipeline(
            [
                PipelineStep("returns", add_returns),
                PipelineStep("volatility", add_volatility, dependencies=["returns"]),
                PipelineStep(
                    "sharpe",
                    add_sharpe,
                    dependencies=["returns", "volatility"],
                ),
            ],
        )

        result = pipeline.run(sample_data)

        # Verify all columns exist
        assert "returns" in result.columns
        assert "volatility" in result.columns
        assert "sharpe" in result.columns

    def test_execution_order_respects_dependencies(
        self,
        sample_data: pl.DataFrame,
    ) -> None:
        """Test that execution order respects dependencies."""
        execution_log = []

        def step_a(df: pl.DataFrame) -> pl.DataFrame:
            execution_log.append("A")
            return df.with_columns(col_a=pl.lit(1))

        def step_b(df: pl.DataFrame) -> pl.DataFrame:
            execution_log.append("B")
            return df.with_columns(col_b=pl.lit(2))

        def step_c(df: pl.DataFrame) -> pl.DataFrame:
            execution_log.append("C")
            return df.with_columns(col_c=pl.lit(3))

        def step_d(df: pl.DataFrame) -> pl.DataFrame:
            execution_log.append("D")
            return df.with_columns(col_d=pl.lit(4))

        # Create pipeline: D depends on C and B, C depends on A, B depends on A
        # Expected order: A, then B and C (in either order), then D
        pipeline = Pipeline(
            [
                PipelineStep("D", step_d, dependencies=["C", "B"]),
                PipelineStep("B", step_b, dependencies=["A"]),
                PipelineStep("A", step_a),
                PipelineStep("C", step_c, dependencies=["A"]),
            ],
        )

        pipeline.run(sample_data)

        # A must come first
        assert execution_log[0] == "A"
        # D must come last
        assert execution_log[-1] == "D"
        # B and C must come after A but before D
        assert execution_log.index("B") > execution_log.index("A")
        assert execution_log.index("C") > execution_log.index("A")
        assert execution_log.index("B") < execution_log.index("D")
        assert execution_log.index("C") < execution_log.index("D")

    def test_cycle_detection(self) -> None:
        """Test that cycles are properly detected."""

        def dummy_step(df: pl.DataFrame) -> pl.DataFrame:
            return df

        # Create a cycle: A -> B -> C -> A
        with pytest.raises(ValueError, match="Cycle detected in pipeline"):
            Pipeline(
                [
                    PipelineStep("A", dummy_step, dependencies=["C"]),
                    PipelineStep("B", dummy_step, dependencies=["A"]),
                    PipelineStep("C", dummy_step, dependencies=["B"]),
                ],
            )

    def test_self_dependency_cycle(self) -> None:
        """Test detection of self-dependency cycles."""

        def dummy_step(df: pl.DataFrame) -> pl.DataFrame:
            return df

        with pytest.raises(ValueError, match="Cycle detected in pipeline"):
            Pipeline([PipelineStep("A", dummy_step, dependencies=["A"])])

    def test_unknown_dependency(self) -> None:
        """Test error when depending on unknown step."""

        def dummy_step(df: pl.DataFrame) -> pl.DataFrame:
            return df

        with pytest.raises(ValueError, match="depends on unknown step"):
            Pipeline([PipelineStep("A", dummy_step, dependencies=["unknown_step"])])

    def test_complex_dag_ordering(self, sample_data: pl.DataFrame) -> None:
        """Test complex DAG with multiple dependency levels."""
        execution_log = []

        def make_step(name: str):
            def step(df: pl.DataFrame) -> pl.DataFrame:
                execution_log.append(name)
                return df.with_columns(**{f"col_{name.lower()}": pl.lit(1)})

            return step

        # Complex DAG:
        #     A
        #   /   \
        #  B     C
        #  |   / | \
        #  D  E  F  G
        #   \ |  | /
        #     H  I
        #      \ |
        #        J

        pipeline = Pipeline(
            [
                PipelineStep("J", make_step("J"), dependencies=["H", "I"]),
                PipelineStep("I", make_step("I"), dependencies=["F", "G"]),
                PipelineStep("H", make_step("H"), dependencies=["D", "E"]),
                PipelineStep("G", make_step("G"), dependencies=["C"]),
                PipelineStep("F", make_step("F"), dependencies=["C"]),
                PipelineStep("E", make_step("E"), dependencies=["C"]),
                PipelineStep("D", make_step("D"), dependencies=["B"]),
                PipelineStep("C", make_step("C"), dependencies=["A"]),
                PipelineStep("B", make_step("B"), dependencies=["A"]),
                PipelineStep("A", make_step("A")),
            ],
        )

        result = pipeline.run(sample_data)

        # Check topological constraints
        def get_position(name: str) -> int:
            return execution_log.index(name)

        # A must come before all others
        assert get_position("A") < get_position("B")
        assert get_position("A") < get_position("C")

        # B must come before D
        assert get_position("B") < get_position("D")

        # C must come before E, F, G
        assert get_position("C") < get_position("E")
        assert get_position("C") < get_position("F")
        assert get_position("C") < get_position("G")

        # D and E must come before H
        assert get_position("D") < get_position("H")
        assert get_position("E") < get_position("H")

        # F and G must come before I
        assert get_position("F") < get_position("I")
        assert get_position("G") < get_position("I")

        # H and I must come before J
        assert get_position("H") < get_position("J")
        assert get_position("I") < get_position("J")

        # Verify all columns were created
        expected_cols = [f"col_{name.lower()}" for name in "ABCDEFGHIJ"]
        for col in expected_cols:
            assert col in result.columns

    def test_add_step_method(self, sample_data: pl.DataFrame) -> None:
        """Test dynamically adding steps to pipeline."""

        def add_returns(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(returns=pl.col("price").pct_change())

        def add_volatility(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                volatility=pl.col("returns").rolling_std(window_size=3),
            )

        # Start with empty pipeline
        pipeline = Pipeline([])

        # Add steps dynamically
        pipeline.add_step(("returns", add_returns))
        pipeline.add_step(
            PipelineStep("volatility", add_volatility, dependencies=["returns"]),
        )

        result = pipeline.run(sample_data)

        assert "returns" in result.columns
        assert "volatility" in result.columns

    def test_get_intermediate_result(self, sample_data: pl.DataFrame) -> None:
        """Test accessing intermediate results."""

        def add_returns(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(returns=pl.col("price").pct_change())

        def add_volatility(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                volatility=pl.col("returns").rolling_std(window_size=3),
            )

        pipeline = Pipeline(
            [
                ("returns", add_returns),
                PipelineStep("volatility", add_volatility, dependencies=["returns"]),
            ],
        )

        final_result = pipeline.run(sample_data)

        # Get intermediate result after returns step
        returns_result = pipeline.get_intermediate_result("returns")
        assert returns_result is not None
        assert "returns" in returns_result.columns
        assert "volatility" not in returns_result.columns

        # Get result after volatility step
        volatility_result = pipeline.get_intermediate_result("volatility")
        assert volatility_result is not None
        assert "returns" in volatility_result.columns
        assert "volatility" in volatility_result.columns

        # Check that final result matches the last intermediate result
        assert final_result.equals(volatility_result)

    def test_step_with_parameters(self, sample_data: pl.DataFrame) -> None:
        """Test pipeline steps with parameters."""

        def add_rolling_mean(
            df: pl.DataFrame,
            window: int = 5,
            column: str = "price",
        ) -> pl.DataFrame:
            return df.with_columns(
                **{
                    f"rolling_mean_{window}": pl.col(column).rolling_mean(
                        window_size=window,
                    ),
                },
            )

        pipeline = Pipeline(
            [
                PipelineStep(
                    "rolling_mean_3",
                    add_rolling_mean,
                    params={"window": 3, "column": "price"},
                ),
                PipelineStep(
                    "rolling_mean_5",
                    add_rolling_mean,
                    params={"window": 5, "column": "price"},
                ),
            ],
        )

        result = pipeline.run(sample_data)

        assert "rolling_mean_3" in result.columns
        assert "rolling_mean_5" in result.columns

    def test_deterministic_ordering(self, sample_data: pl.DataFrame) -> None:
        """Test that pipeline execution is deterministic when multiple orderings are valid."""
        execution_logs = []

        def make_step(name: str):
            def step(df: pl.DataFrame) -> pl.DataFrame:
                execution_logs[-1].append(name)
                return df.with_columns(**{f"col_{name.lower()}": pl.lit(1)})

            return step

        # Create pipeline where B and C can both execute after A
        # but the algorithm should produce consistent ordering
        for _ in range(5):  # Run multiple times
            execution_logs.append([])

            pipeline = Pipeline(
                [
                    PipelineStep("C", make_step("C"), dependencies=["A"]),
                    PipelineStep("A", make_step("A")),
                    PipelineStep("B", make_step("B"), dependencies=["A"]),
                ],
            )

            pipeline.run(sample_data)

        # All runs should have the same execution order (deterministic)
        first_order = execution_logs[0]
        for log in execution_logs[1:]:
            assert log == first_order, f"Non-deterministic ordering: {log} != {first_order}"
