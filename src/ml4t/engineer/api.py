"""Config-driven feature computation API for ml4t.engineer.

This module provides the main public API for computing features from configurations.

Exports:
    compute_features(data, features, *, group_col=None, timestamp_col=None) -> DataFrame
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
from typing import Any, TypedDict

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

_GROUP_COLUMN_CANDIDATES = ("asset_id", "symbol", "ticker", "product", "asset")
_TIME_COLUMN_CANDIDATES = ("timestamp", "event_time", "date", "datetime")


class _FeatureSpec(TypedDict):
    name: str
    params: dict[str, Any]
    output: str


def compute_features(
    data: pl.DataFrame | pl.LazyFrame,
    features: list[str | dict[str, Any]] | Path | str,
    *,
    group_col: str | list[str] | None = None,
    timestamp_col: str | None = None,
    assume_sorted: bool = False,
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
               {"name": "macd", "params": {"fast_period": 12, "slow_period": 26}},
           ]
           ```

        3. Path to YAML config file:
           ```python
           Path("features.yaml")
           # or string path
           "config/features.yaml"
           ```
    group_col : str | list[str] | None, default None
        Asset grouping column or columns. Common asset columns are detected when
        omitted. Rolling and lagged features are computed independently per group.
    timestamp_col : str | None, default None
        Temporal ordering column. Common datetime columns are detected when omitted.
        Input is sorted by group and timestamp before feature computation.
    assume_sorted : bool, default False
        Permit row-order execution when no timestamp column exists. This is an
        explicit assertion by the caller that rows are already in the required
        temporal order.

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
    >>> from datetime import datetime, timedelta
    >>> import polars as pl
    >>> from ml4t.engineer.api import compute_features
    >>>
    >>> # Load OHLCV data
    >>> df = pl.DataFrame({
    ...     "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)],
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
    >>> # Multiple configurations require explicit output names
    >>> result = compute_features(df, [
    ...     {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
    ...     {"name": "sma", "params": {"period": 50}, "output": "sma_50"},
    ... ])
    >>>
    >>> # Compute from YAML config
    >>> result = compute_features(df, "features.yaml")

    Notes
    -----
    - Features are computed in dependency order using topological sort.
    - Circular dependencies are detected and raise ValueError.
    - Parameters in config override default parameters from the function signature.
    - Recognized asset columns isolate rolling and lagged calculations.
    - Input is sorted by group and timestamp. Without a timestamp, callers must set
      ``assume_sorted=True`` explicitly.
    """
    from ml4t.engineer.core.schemas import validate_ohlcv_schema

    # Validate input schema before parsing or execution.
    validate_ohlcv_schema(data, require_asset_id=False, allow_flexible_time=True)
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema

    # Parse input to standardized format
    feature_specs = _parse_feature_input(features)
    _validate_feature_specs(feature_specs, set(schema.names()))
    if not feature_specs:
        return data

    group_cols, resolved_timestamp_col = _resolve_ordering_columns(
        data=data,
        group_col=group_col,
        timestamp_col=timestamp_col,
        assume_sorted=assume_sorted,
    )

    # Resolve dependencies and get execution order
    execution_order = _resolve_dependencies(feature_specs)

    sort_cols = [*group_cols]
    if resolved_timestamp_col is not None:
        sort_cols.append(resolved_timestamp_col)

    # Execute features in order
    result = data.sort(sort_cols) if sort_cols else data
    for spec in execution_order:
        result = _execute_feature(
            result,
            feature_name=spec["name"],
            params=spec["params"],
            output_name=spec["output"],
            group_cols=group_cols,
        )

    return result


def _parse_feature_input(
    features: list[str | dict[str, Any]] | Path | str,
) -> list[_FeatureSpec]:
    """Parse feature input to standardized dict format.

    Parameters
    ----------
    features : list[str] | list[dict] | Path | str
        Feature specification in any supported format

    Returns
    -------
    list[_FeatureSpec]
        Validated normalized feature requests.

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

    if not isinstance(features, list):
        raise ValueError(
            f"Invalid features format. Expected list[str], list[dict], or Path, "
            f"got: {type(features)}"
        )

    result: list[_FeatureSpec] = []
    for item in features:
        if isinstance(item, str):
            result.append({"name": item, "params": {}, "output": item})
        elif isinstance(item, dict):
            result.append(_normalize_feature_spec(item))
        else:
            raise ValueError(
                "Invalid features format. Expected every list item to be a string "
                f"or mapping, got: {type(item)}"
            )
    return result


def _normalize_feature_spec(spec: dict[str, Any]) -> _FeatureSpec:
    """Validate and normalize one mapping feature request."""
    allowed_keys = {"name", "params", "output"}
    unknown_keys = set(spec) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown feature spec fields: {sorted(unknown_keys)}. "
            f"Accepted fields: {sorted(allowed_keys)}"
        )
    if "name" not in spec:
        raise ValueError(f"Feature spec missing 'name' field: {spec}")

    name = spec["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"Feature spec 'name' must be a non-empty string, got: {name!r}")

    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"Feature '{name}' params must be a mapping, got: {type(params).__name__}")

    output = spec.get("output", name)
    if not isinstance(output, str) or not output:
        raise ValueError(f"Feature '{name}' output must be a non-empty string, got: {output!r}")

    return {"name": name, "params": params, "output": output}


def _validate_feature_specs(feature_specs: list[_FeatureSpec], input_columns: set[str]) -> None:
    """Validate every request before any feature executes."""
    import inspect

    registry = get_registry()
    seen_outputs: set[str] = set()

    for spec in feature_specs:
        name = spec["name"]
        output = spec["output"]
        metadata = registry.get(name)
        if metadata is None:
            raise ValueError(
                f"Feature '{name}' not found in registry. "
                f"Available features: {', '.join(registry.list_all())}"
            )

        if output in seen_outputs:
            raise ValueError(
                f"Duplicate feature output '{output}'. "
                "Set a distinct 'output' for each feature configuration."
            )
        if output in input_columns:
            raise ValueError(
                f"Feature output '{output}' conflicts with an input column. "
                "Choose a distinct 'output' name."
            )
        seen_outputs.add(output)

        signature = inspect.signature(metadata.func)
        configurable = {
            param_name for param_name in signature.parameters if param_name not in COLUMN_ARG_MAP
        }
        unknown_params = set(spec["params"]) - configurable
        if unknown_params:
            accepted = sorted(configurable)
            qualifier = "parameter" if len(unknown_params) == 1 else "parameters"
            raise ValueError(
                f"Feature '{name}' has unknown {qualifier} {sorted(unknown_params)}. "
                f"Accepted parameters: {accepted}"
            )


def _resolve_ordering_columns(
    data: pl.DataFrame | pl.LazyFrame,
    group_col: str | list[str] | None,
    timestamp_col: str | None,
    assume_sorted: bool,
) -> tuple[list[str], str | None]:
    """Resolve and validate grouping and temporal ordering columns."""
    from ml4t.engineer.core.exceptions import DataSchemaError

    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    columns = set(schema.names())

    if group_col is None:
        group_cols = [name for name in _GROUP_COLUMN_CANDIDATES if name in columns][:1]
    elif isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)
        detected_group_cols = [name for name in _GROUP_COLUMN_CANDIDATES if name in columns]
        if not group_cols and detected_group_cols:
            raise DataSchemaError(
                "Grouping cannot be disabled while a recognized asset column is present. "
                f"Detected: {detected_group_cols}"
            )

    missing_group_cols = set(group_cols) - columns
    if missing_group_cols:
        raise DataSchemaError(
            f"Grouping columns not found: {sorted(missing_group_cols)}. "
            f"Available columns: {sorted(columns)}"
        )

    if timestamp_col is None:
        resolved_timestamp_col = next(
            (name for name in _TIME_COLUMN_CANDIDATES if name in columns),
            None,
        )
    else:
        resolved_timestamp_col = timestamp_col
        if timestamp_col not in columns:
            raise DataSchemaError(
                f"Timestamp column '{timestamp_col}' not found. "
                f"Available columns: {sorted(columns)}"
            )

    if resolved_timestamp_col is None and not assume_sorted:
        raise DataSchemaError(
            "No time column found. Provide timestamp_col, use one of "
            f"{list(_TIME_COLUMN_CANDIDATES)}, or set assume_sorted=True to assert "
            "that row order is already temporal."
        )

    if resolved_timestamp_col is not None:
        dtype = schema[resolved_timestamp_col]
        if dtype.base_type() not in {pl.Date, pl.Datetime}:
            raise DataSchemaError(
                f"Timestamp column '{resolved_timestamp_col}' must be Date or Datetime, got {dtype}"
            )

    if isinstance(data, pl.DataFrame):
        checked_columns = [*group_cols]
        if resolved_timestamp_col is not None:
            checked_columns.append(resolved_timestamp_col)
        null_columns = [name for name in checked_columns if data[name].null_count() > 0]
        if null_columns:
            raise DataSchemaError(
                f"Grouping and ordering columns cannot contain nulls: {null_columns}"
            )

    return group_cols, resolved_timestamp_col


def _resolve_dependencies(feature_specs: list[_FeatureSpec]) -> list[_FeatureSpec]:
    """Resolve feature dependencies using topological sort (Kahn's algorithm).

    Parameters
    ----------
    feature_specs : list[dict[str, Any]]
        List of feature specifications

    Returns
    -------
    list[_FeatureSpec]
        Feature requests in dependency order.

    Raises
    ------
    ValueError
        If feature not in registry or circular dependency detected
    """
    registry = get_registry()
    requested_names = {spec["name"] for spec in feature_specs}
    completed_names: set[str] = set()
    remaining = list(feature_specs)
    result: list[_FeatureSpec] = []

    while remaining:
        ready = []
        for spec in remaining:
            metadata = registry.get(spec["name"])
            if metadata is None:
                raise ValueError(f"Feature '{spec['name']}' not found in registry")
            requested_dependencies = set(metadata.dependencies) & requested_names
            if requested_dependencies <= completed_names:
                ready.append(spec)

        if not ready:
            unresolved = [spec["name"] for spec in remaining]
            raise ValueError(
                f"Circular dependency detected. Unresolved features: {', '.join(unresolved)}"
            )

        for spec in ready:
            result.append(spec)
            completed_names.add(spec["name"])
            remaining.remove(spec)

    if len(result) != len(feature_specs):
        unresolved = [spec["name"] for spec in feature_specs if spec not in result]
        raise ValueError(
            f"Circular dependency detected. Unresolved features: {', '.join(unresolved)}"
        )

    return result


def _execute_feature(
    data: pl.DataFrame | pl.LazyFrame,
    feature_name: str,
    params: dict[str, Any],
    output_name: str,
    group_cols: list[str],
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
    output_name : str
        Base output name selected by the feature request
    group_cols : list[str]
        Columns that isolate rolling and lagged computations

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
            if param_name in final_params:
                keyword_params[param_name] = final_params[param_name]
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
            raise ValueError(
                f"Feature '{feature_name}' requires parameter '{param_name}' but it's not "
                f"provided in metadata.parameters or call params. "
                f'Use: compute_features(df, [{{"name": "{feature_name}", '
                f'"params": {{"{param_name}": value}}}}])'
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
        expr = result.over(group_cols) if group_cols else result
        return _append_feature_expressions(data, [(output_name, expr)])
    elif isinstance(result, dict):
        # Multiple expressions - add all with prefixed names
        exprs: list[tuple[str, pl.Expr]] = []
        for key, expr in result.items():
            if isinstance(expr, pl.Expr):
                grouped_expr = expr.over(group_cols) if group_cols else expr
                exprs.append((f"{output_name}_{key}", grouped_expr))
        if exprs:
            return _append_feature_expressions(data, exprs)
        else:
            raise ValueError(f"Feature '{feature_name}' returned dict without Expr values")
    elif isinstance(result, tuple | list):
        # Multiple expressions as tuple/list - add all
        exprs = []
        for i, expr in enumerate(result):
            if isinstance(expr, pl.Expr):
                grouped_expr = expr.over(group_cols) if group_cols else expr
                exprs.append((f"{output_name}_{i}", grouped_expr))
        if exprs:
            return _append_feature_expressions(data, exprs)
        else:
            raise ValueError(f"Feature '{feature_name}' returned tuple/list without Expr values")
    else:
        raise TypeError(
            f"Feature '{feature_name}' returned unexpected type: {type(result)}\n"
            f"Expected pl.Expr, dict, or tuple, got {type(result).__name__}"
        )


def _append_feature_expressions(
    data: pl.DataFrame | pl.LazyFrame,
    named_expressions: list[tuple[str, pl.Expr]],
) -> pl.DataFrame | pl.LazyFrame:
    """Add feature expressions without replacing existing columns."""
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    existing_columns = set(schema.names())
    output_names = [name for name, _expr in named_expressions]
    conflicts = existing_columns & set(output_names)
    if conflicts:
        raise ValueError(
            f"Feature outputs conflict with existing columns: {sorted(conflicts)}. "
            "Choose distinct 'output' names."
        )
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"Feature produced duplicate output columns: {output_names}")
    return data.with_columns([expr.alias(name) for name, expr in named_expressions])
