"""Config-driven feature computation API for ml4t.engineer.

This module provides the main public API for computing features from configurations.

Exports:
    compute_features(data, features, column_map=None) -> DataFrame
        Main API for computing technical indicators on OHLCV data.

    Constants:
        COLUMN_ARG_MAP: dict - Maps function params to DataFrame columns
        INPUT_TYPE_COLUMNS: dict - Maps input_type metadata to required columns

Internal:
    _parse_feature_input() - Parse feature specifications
    _resolve_dependencies() - Topological sort of features
    _execute_feature() - Execute single feature computation
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

# =============================================================================
# Column Mapping Configuration
# =============================================================================
# These mappings translate function parameter names to DataFrame column names.
# Moved to module level to avoid recreation on every feature execution.
# =============================================================================

# Map of function parameter names to DataFrame column names
# After V3 standardization, most parameters match column names directly.
# Only legacy aliases and special cases need explicit mapping.
COLUMN_ARG_MAP: dict[str, str | list[str]] = {
    # Standard OHLCV columns - direct mapping (parameter name = column name)
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "returns": "returns",
    # Legacy parameter names (some older features use these)
    "price": "close",  # Microstructure features often use "price" parameter
    "value": "close",  # ML features may use "value" for generic input
    # Meta-feature defaults (features that operate on other features)
    "feature": "close",  # Single-feature input defaults to close
    "features": ["close"],  # Multi-feature input defaults to close only
    "volatility": "close",  # Volatility features compute from close
    "regime": "close",  # Regime detection features use close
}

# Map input_type metadata to required DataFrame columns
# This enables deriving column requirements from FeatureMetadata.input_type
INPUT_TYPE_COLUMNS: dict[str, list[str]] = {
    "OHLCV": ["open", "high", "low", "close", "volume"],
    "OHLC": ["open", "high", "low", "close"],
    "HLC": ["high", "low", "close"],
    "HL": ["high", "low"],
    "close": ["close"],
    "returns": ["returns"],
    "volume": ["volume"],
}

# Parameters that should always be passed as kwargs, never as positional
# These typically have defaults and shouldn't be treated as column inputs
KEYWORD_ONLY_PARAMS: frozenset[str] = frozenset(
    {
        "implementation",  # Always has default, selects algorithm variant
    }
)

# Common default values for required parameters missing from metadata
# Used as fallback when feature registration is incomplete
COMMON_PARAM_DEFAULTS: dict[str, Any] = {
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


def compute_features(
    data: pl.DataFrame | pl.LazyFrame,
    features: list[str] | list[dict[str, Any]] | Path | str,
) -> pl.DataFrame | pl.LazyFrame:
    """Compute features from a configuration.

    This is the main public API for QFeatures. It accepts feature specifications
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
    >>> from ml4t.engineer.api import compute_features
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


def _parse_feature_input(
    features: list[str] | list[dict[str, Any]] | Path | str,
) -> list[dict[str, Any]]:
    """Parse feature input to standardized dict format.

    Parameters
    ----------
    features : list[str] | list[dict] | Path | str
        Feature specification in any supported format

    Returns
    -------
    list[dict[str, Any]]
        Standardized format: [{"name": str, "params": dict}, ...]

    Raises
    ------
    ImportError
        If YAML config provided but PyYAML not installed
    FileNotFoundError
        If config file doesn't exist
    """
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
        # Standardize format
        result = []
        for spec_item in features:
            # Type narrowing: we know this is a dict from the isinstance check above
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
    """Resolve feature dependencies using topological sort (Kahn's algorithm).

    Parameters
    ----------
    feature_specs : list[dict[str, Any]]
        List of feature specifications

    Returns
    -------
    list[tuple[str, dict]]
        Features in execution order: [(name, params), ...]

    Raises
    ------
    ValueError
        If feature not in registry or circular dependency detected
    """
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
        # Sort queue for deterministic ordering
        queue.sort()
        current = queue.pop(0)
        result.append((current, feature_map[current]))

        # Update in-degrees
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
    """Execute a single feature computation using signature-aware dispatch.

    This function introspects the feature's actual signature to determine
    which columns and parameters to pass. Column mappings are configured
    at module level in COLUMN_ARG_MAP.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Input data
    feature_name : str
        Feature name from registry
    params : dict[str, Any]
        Parameters to override defaults

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Data with feature column added

    Raises
    ------
    ValueError
        If feature not in registry or if function signature cannot be matched
    """
    import inspect

    registry = get_registry()
    metadata = registry.get(feature_name)

    if metadata is None:
        raise ValueError(f"Feature '{feature_name}' not found in registry")

    # Get function signature
    sig = inspect.signature(metadata.func)
    func_params = sig.parameters

    # Merge default parameters with overrides
    final_params = {**metadata.parameters, **params}

    # Separate column arguments from keyword parameters
    # We need to maintain order for positional arguments
    column_args: list[str | list[str]] = []
    keyword_params: dict[str, Any] = {}

    for param_name, param_obj in func_params.items():
        # Skip parameters that should always be kwargs
        if param_name in KEYWORD_ONLY_PARAMS:
            continue

        # Check if this parameter name matches a known column argument
        if param_name in COLUMN_ARG_MAP:
            # Only add as positional argument if it's required (no default)
            if param_obj.default is inspect.Parameter.empty:
                # Required column argument - pass the column name as string
                column_args.append(COLUMN_ARG_MAP[param_name])
            # else: Has default, will use None or default value, don't pass
        elif param_name in final_params:
            # It's a configurable parameter - add to kwargs
            keyword_params[param_name] = final_params[param_name]
        elif param_obj.default is not inspect.Parameter.empty:
            # Has a default in function signature - use it (don't need to pass explicitly)
            pass
        else:
            # Required parameter with no default and not in COLUMN_ARG_MAP
            # This indicates incomplete metadata - try common defaults
            if param_name in COMMON_PARAM_DEFAULTS:
                keyword_params[param_name] = COMMON_PARAM_DEFAULTS[param_name]
            else:
                # Cannot proceed - need user to provide this parameter explicitly
                raise ValueError(
                    f"Feature '{feature_name}' requires parameter '{param_name}' but it's not "
                    f"provided. Call with explicit parameters: "
                    f'compute_features(df, [{{"name": "{feature_name}", "{param_name}": value}}])'
                )

    # Call the feature function
    try:
        result = metadata.func(*column_args, **keyword_params)
    except TypeError as e:
        # Provide detailed error message for debugging
        raise ValueError(
            f"Failed to execute feature '{feature_name}': {e}\n"
            f"Function signature: {sig}\n"
            f"Attempted call with column_args={column_args}, keyword_params={keyword_params}\n"
            f"Available metadata: input_type='{metadata.input_type}', "
            f"parameters={metadata.parameters}"
        ) from e

    # Handle different return types
    if isinstance(result, pl.Expr):
        # Single expression - add it directly
        return data.with_columns(result.alias(feature_name))
    elif isinstance(result, dict):
        # Multiple expressions - add all with prefixed names
        exprs = []
        for key, expr in result.items():
            if isinstance(expr, pl.Expr):
                exprs.append(expr.alias(f"{feature_name}_{key}"))
        if exprs:
            return data.with_columns(exprs)
        else:
            raise ValueError(f"Feature '{feature_name}' returned dict without Expr values")
    elif isinstance(result, tuple | list):
        # Multiple expressions as tuple/list - add all
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
