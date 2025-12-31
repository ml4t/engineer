"""Pipeline engine for ml4t.engineer.

Provides the DAG-based execution engine for feature engineering pipelines.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class PipelineStep:
    """Represents a single step in a pipeline.

    Parameters
    ----------
    name : str
        The name of this step
    func : Callable
        The function to execute
    params : dict
        Parameters to pass to the function
    dependencies : list[str]
        Names of steps this step depends on
    """

    name: str
    func: Callable[..., Any]
    params: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


class Pipeline:
    """DAG-based pipeline for feature engineering.

    The pipeline executes a series of transformations on data in dependency order.
    Each step can depend on the output of previous steps.

    Parameters
    ----------
    steps : list[tuple[str, Callable] | PipelineStep]
        List of pipeline steps. Each step can be either:
        - A tuple of (name, function)
        - A PipelineStep instance

    Examples
    --------
    >>> import ml4t_features as qf
    >>> import polars as pl
    >>>
    >>> # Create a simple pipeline
    >>> pipeline = qf.pipeline.Pipeline(steps=[
    ...     ("returns", lambda df: df.with_columns(
    ...         returns=pl.col("close").pct_change()
    ...     )),
    ...     ("volatility", lambda df: df.with_columns(
    ...         volatility=pl.col("returns").rolling_std(20)
    ...     ))
    ... ])
    >>>
    >>> # Run the pipeline
    >>> result = pipeline.run(data)
    """

    def __init__(
        self,
        steps: list[tuple[str, Callable[..., Any]] | PipelineStep],
    ):
        """Initialize the pipeline."""
        self.steps: list[PipelineStep] = []
        self._results: dict[str, pl.DataFrame] = {}

        # Convert tuples to PipelineStep objects
        for step in steps:
            if isinstance(step, tuple):
                name: str = step[0]
                func: Callable[..., Any] = step[1]
                self.steps.append(PipelineStep(name=name, func=func))
            elif isinstance(step, PipelineStep):
                self.steps.append(step)
            else:
                raise ValueError(
                    f"Step must be tuple or PipelineStep, got {type(step)}",
                )

        # Validate DAG (check for cycles)
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Validate that the pipeline forms a valid DAG (no cycles)."""
        # Check that dependencies exist
        step_names = {step.name for step in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(
                        f"Step '{step.name}' depends on unknown step '{dep}'",
                    )

        # Check for cycles using DFS
        self._detect_cycles()

    def _detect_cycles(self) -> None:
        """Detect cycles in the dependency graph using DFS.

        Raises
        ------
        ValueError
            If a cycle is detected in the dependency graph
        """
        # Build adjacency list for dependency graph
        graph: dict[str, list[str]] = {step.name: step.dependencies for step in self.steps}

        # Track visit states: 0=unvisited, 1=visiting, 2=visited
        visit_state: dict[str, int] = {step.name: 0 for step in self.steps}

        def dfs_visit(node: str, path: list[str]) -> None:
            if visit_state[node] == 1:  # Currently visiting - cycle detected
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise ValueError(f"Cycle detected in pipeline: {' -> '.join(cycle)}")

            if visit_state[node] == 2:  # Already visited
                return

            visit_state[node] = 1  # Mark as visiting
            path.append(node)

            for dependency in graph[node]:
                dfs_visit(dependency, path)

            path.pop()
            visit_state[node] = 2  # Mark as visited

        # Check each unvisited node
        for step_name in graph:
            if visit_state[step_name] == 0:
                dfs_visit(step_name, [])

    def _get_execution_order(self) -> list[PipelineStep]:
        """Get the order in which steps should be executed.

        Uses Kahn's algorithm for topological sorting to determine execution order.
        Steps with no dependencies are executed first, followed by steps whose
        dependencies have all been satisfied.

        IMPORTANT: Preserves original step order for steps with equal priority
        (i.e., when no dependency constraints exist).

        Returns
        -------
        list[PipelineStep]
            Steps ordered by their dependencies (topologically sorted)
        """
        # Create step lookup and index mapping for original order
        step_lookup = {step.name: step for step in self.steps}
        step_index = {step.name: i for i, step in enumerate(self.steps)}

        # Build in-degree count for each step
        in_degree = {step.name: len(step.dependencies) for step in self.steps}

        # Initialize queue with steps that have no dependencies (preserve original order)
        queue = [step_name for step_name, count in in_degree.items() if count == 0]
        result = []

        # Process steps in topological order
        while queue:
            # Sort by original order to preserve insertion sequence
            queue.sort(key=lambda name: step_index[name])
            current_step_name = queue.pop(0)
            result.append(step_lookup[current_step_name])

            # For each step that depends on the current step, reduce its in-degree
            for step in self.steps:
                if current_step_name in step.dependencies:
                    in_degree[step.name] -= 1
                    if in_degree[step.name] == 0:
                        queue.append(step.name)

        # Verify all steps were processed (should never happen after cycle detection)
        if len(result) != len(self.steps):
            remaining = [name for name, count in in_degree.items() if count > 0]
            raise ValueError(f"Unable to resolve dependencies for steps: {remaining}")

        return result

    def run(self, data: pl.DataFrame) -> pl.DataFrame:
        """Execute the pipeline on the input data.

        Parameters
        ----------
        data : pl.DataFrame
            The input data

        Returns
        -------
        pl.DataFrame
            The transformed data after all pipeline steps
        """
        result = data
        self._results = {"input": data}

        # Execute steps in order
        for step in self._get_execution_order():
            # Apply the transformation
            result = step.func(result, **step.params) if step.params else step.func(result)

            # Store intermediate result
            self._results[step.name] = result

        return result

    def get_intermediate_result(self, step_name: str) -> pl.DataFrame | None:
        """Get the result after a specific step.

        Parameters
        ----------
        step_name : str
            The name of the step

        Returns
        -------
        pl.DataFrame or None
            The data after the specified step, or None if not found
        """
        return self._results.get(step_name)

    def add_step(
        self,
        step: tuple[str, Callable[..., Any]] | PipelineStep,
    ) -> "Pipeline":
        """Add a step to the pipeline.

        Parameters
        ----------
        step : tuple or PipelineStep
            The step to add

        Returns
        -------
        Pipeline
            Self for method chaining
        """
        if isinstance(step, tuple):
            step_name: str = step[0]
            step_func: Callable[..., Any] = step[1]
            self.steps.append(PipelineStep(name=step_name, func=step_func))
        elif isinstance(step, PipelineStep):
            self.steps.append(step)
        else:
            raise ValueError(f"Step must be tuple or PipelineStep, got {type(step)}")

        self._validate_dag()
        return self


__all__ = ["Pipeline", "PipelineStep"]
