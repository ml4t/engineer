"""Config-driven feature computation API for ML4T Engineer.

This module provides the main public API for computing features from configurations.
"""

from pathlib import Path
from typing import Any

import polars as pl

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ml4t.engineer.core.registry import get_registry


def compute_features(
    data: pl.DataFrame | pl.LazyFrame,
    features: list[str] | list[dict[str, Any]] | Path | str,
) -> pl.DataFrame | pl.LazyFrame:
    """Compute features from a configuration.

    This is the main public API for ml4t-engineer. It accepts feature specifications
    in multiple formats and computes them in dependency order.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Input data (typically OHLCV)
    features : list[str] | list[dict] | Path | str
        Feature specification in one of three formats:

        1. List of feature names (use default parameters):
           ```python
           ["rsi", "macd", "bollinger_bands"]
           ```

        2. List of dicts with parameters:
           ```python
           [
               {"name": "rsi", "params": {"period": 14}},
               {"name": "macd", "params": {"fast": 12, "slow": 26}},
           ]
           ```

        3. Path to YAML config file:
           ```python
           Path("features.yaml")
           # or string path
           "config/features.yaml"
           ```

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Input data with computed feature columns added

    Raises
    ------
    ValueError
        If feature not found in registry or circular dependency detected
    ImportError
        If YAML config provided but PyYAML not installed
    FileNotFoundError
        If config file path doesn't exist

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer import compute_features
    >>>
    >>> # Load OHLCV data
    >>> df = pl.DataFrame({
    ...     "open": [100.0, 101.0, 102.0],
    ...     "high": [102.0, 103.0, 104.0],
    ...     "low": [99.0, 100.0, 101.0],
    ...     "close": [101.0, 102.0, 103.0],
    ...     "volume": [1000, 1100, 1200],
    ... })
    >>>
    >>> # Compute features with default parameters
    >>> result = compute_features(df, ["rsi", "sma"])
    >>>
    >>> # Compute features with custom parameters
    >>> result = compute_features(df, [
    ...     {"name": "rsi", "params": {"period": 20}},
    ...     {"name": "sma", "params": {"period": 50}},
    ... ])
    >>>
    >>> # Compute from YAML config
    >>> result = compute_features(df, "features.yaml")

    Notes
    -----
    - Features are computed in dependency order using topological sort
    - Circular dependencies are detected and raise ValueError
    - Parameters in config override default parameters from registry
    """
    # Parse input to standardized format
    feature_specs = _parse_feature_input(features)

    # Resolve dependencies and get execution order
    execution_order = _resolve_dependencies(feature_specs)

    # Execute features in order
    result = data
    for feature_name, params in execution_order:
        result = _execute_feature(result, feature_name, params)

    return result


def list_features(category: str | None = None) -> list[str]:
    """List available features.

    Parameters
    ----------
    category : str, optional
        Filter by category (momentum, trend, volatility, etc.)

    Returns
    -------
    list[str]
        Sorted list of feature names
    """
    registry = get_registry()
    if category:
        return registry.list_by_category(category)
    return registry.list_all()


def list_categories() -> list[str]:
    """List available feature categories.

    Returns
    -------
    list[str]
        Sorted list of category names
    """
    registry = get_registry()
    categories = set()
    for name in registry.list_all():
        meta = registry.get(name)
        if meta:
            categories.add(meta.category)
    return sorted(categories)


def describe_feature(name: str) -> dict[str, Any]:
    """Get detailed information about a feature.

    Parameters
    ----------
    name : str
        Feature name

    Returns
    -------
    dict[str, Any]
        Feature metadata including description, parameters, etc.

    Raises
    ------
    ValueError
        If feature not found
    """
    registry = get_registry()
    meta = registry.get(name)
    if meta is None:
        available = registry.list_all()[:10]
        raise ValueError(
            f"Feature '{name}' not found. "
            f"Available features: {', '.join(available)}..."
        )
    return {
        "name": meta.name,
        "category": meta.category,
        "description": meta.description,
        "formula": meta.formula,
        "normalized": meta.normalized,
        "value_range": meta.value_range,
        "ta_lib_compatible": meta.ta_lib_compatible,
        "input_type": meta.input_type,
        "output_type": meta.output_type,
        "parameters": meta.parameters,
        "dependencies": meta.dependencies,
        "references": meta.references,
        "tags": meta.tags,
    }


def _parse_feature_input(
    features: list[str] | list[dict[str, Any]] | Path | str,
) -> list[dict[str, Any]]:
    """Parse feature input to standardized dict format."""
    # Handle YAML config file
    if isinstance(features, Path | str):
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML configs. Install with: pip install pyyaml"
            )

        config_path = Path(features)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract features list from YAML
        if isinstance(config, dict) and "features" in config:
            features = config["features"]
        elif isinstance(config, list):
            features = config
        else:
            raise ValueError(
                f"Invalid YAML format. Expected list or dict with 'features' key, got: {type(config)}"
            )

    # Handle list of strings (feature names only)
    if isinstance(features, list) and all(isinstance(f, str) for f in features):
        return [{"name": name, "params": {}} for name in features]

    # Handle list of dicts
    if isinstance(features, list) and all(isinstance(f, dict) for f in features):
        result = []
        for spec_item in features:
            spec = spec_item if isinstance(spec_item, dict) else {}
            if "name" not in spec:
                raise ValueError(f"Feature spec missing 'name' field: {spec}")
            result.append({"name": spec["name"], "params": spec.get("params", {})})
        return result

    # Handle mixed list of strings and dicts
    if isinstance(features, list) and all(isinstance(f, str | dict) for f in features):
        result = []
        for item in features:
            if isinstance(item, str):
                result.append({"name": item, "params": {}})
            elif isinstance(item, dict):
                if "name" not in item:
                    raise ValueError(f"Feature spec missing 'name' field: {item}")
                result.append({"name": item["name"], "params": item.get("params", {})})
        return result

    raise ValueError(
        f"Invalid features format. Expected list[str], list[dict], or Path, got: {type(features)}"
    )


def _resolve_dependencies(feature_specs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Resolve feature dependencies using topological sort (Kahn's algorithm)."""
    registry = get_registry()

    # Build dependency graph
    feature_map = {spec["name"]: spec["params"] for spec in feature_specs}
    in_degree = dict.fromkeys(feature_map, 0)
    dependencies = {}

    for name in feature_map:
        metadata = registry.get(name)
        if metadata is None:
            raise ValueError(
                f"Feature '{name}' not found in registry. "
                f"Available features: {', '.join(registry.list_all())}"
            )

        dependencies[name] = metadata.dependencies
        for dep in metadata.dependencies:
            if dep in feature_map:
                in_degree[name] += 1

    # Kahn's algorithm for topological sort
    queue = [name for name in feature_map if in_degree[name] == 0]
    result = []

    while queue:
        queue.sort()
        current = queue.pop(0)
        result.append((current, feature_map[current]))

        for name in feature_map:
            if current in dependencies[name]:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)

    # Check for circular dependencies
    if len(result) != len(feature_map):
        unresolved = [name for name in feature_map if name not in dict(result)]
        raise ValueError(
            f"Circular dependency detected. Unresolved features: {', '.join(unresolved)}"
        )

    return result


def _execute_feature(
    data: pl.DataFrame | pl.LazyFrame,
    feature_name: str,
    params: dict[str, Any],
) -> pl.DataFrame | pl.LazyFrame:
    """Execute a single feature computation."""
    import inspect

    registry = get_registry()
    metadata = registry.get(feature_name)

    if metadata is None:
        raise ValueError(f"Feature '{feature_name}' not found in registry")

    # Get function signature
    sig = inspect.signature(metadata.func)
    func_params = sig.parameters

    # Map of expected column argument names to DataFrame column names
    COLUMN_ARG_MAP = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "returns": "returns",
        "price": "close",
        "value": "close",
        "feature": "close",
        "features": ["close"],
        "volatility": "close",
        "regime": "close",
    }

    KEYWORD_ONLY_PARAMS = {"implementation"}

    # Merge default parameters with overrides
    final_params = {**metadata.parameters, **params}

    # Separate column arguments from keyword parameters
    column_args = []
    keyword_params = {}

    for param_name, param_obj in func_params.items():
        if param_name in KEYWORD_ONLY_PARAMS:
            continue

        if param_name in COLUMN_ARG_MAP:
            if param_obj.default is inspect.Parameter.empty:
                column_args.append(COLUMN_ARG_MAP[param_name])
        elif param_name in final_params:
            keyword_params[param_name] = final_params[param_name]
        elif param_obj.default is not inspect.Parameter.empty:
            pass
        else:
            common_defaults = {
                "period": 14,
                "window": 20,
                "lookback": 20,
                "lag": 1,
                "lags": [1],
                "n": 5,
                "bins": 10,
                "features": ["close"],
                "windows": [5, 10, 20],
            }

            if param_name in common_defaults:
                keyword_params[param_name] = common_defaults[param_name]
            else:
                raise ValueError(
                    f"Feature '{feature_name}' requires parameter '{param_name}' but it's not "
                    f"provided. Call with explicit parameters: "
                    f'compute_features(df, [{{"name": "{feature_name}", "{param_name}": value}}])'
                )

    # Call the feature function
    try:
        result = metadata.func(*column_args, **keyword_params)
    except TypeError as e:
        raise ValueError(
            f"Failed to execute feature '{feature_name}': {e}\n"
            f"Function signature: {sig}\n"
            f"Attempted call with column_args={column_args}, keyword_params={keyword_params}\n"
            f"Available metadata: input_type='{metadata.input_type}', "
            f"parameters={metadata.parameters}"
        ) from e

    # Handle different return types
    if isinstance(result, pl.Expr):
        return data.with_columns(result.alias(feature_name))
    elif isinstance(result, dict):
        exprs = []
        for key, expr in result.items():
            if isinstance(expr, pl.Expr):
                exprs.append(expr.alias(f"{feature_name}_{key}"))
        if exprs:
            return data.with_columns(exprs)
        else:
            raise ValueError(f"Feature '{feature_name}' returned dict without Expr values")
    elif isinstance(result, tuple | list):
        exprs = []
        for i, expr in enumerate(result):
            if isinstance(expr, pl.Expr):
                exprs.append(expr.alias(f"{feature_name}_{i}"))
        if exprs:
            return data.with_columns(exprs)
        else:
            raise ValueError(f"Feature '{feature_name}' returned tuple/list without Expr values")
    else:
        raise TypeError(
            f"Feature '{feature_name}' returned unexpected type: {type(result)}\n"
            f"Expected pl.Expr, dict, or tuple, got {type(result).__name__}"
        )
