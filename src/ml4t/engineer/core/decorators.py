"""Feature registration decorators.

Simple decorator-based registration for features with zero overhead.
Metadata is attached at import time, computation has no wrapper overhead.
"""

import inspect
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from ml4t.engineer.core.dispatch import COLUMN_ARG_MAP
from ml4t.engineer.core.lookbacks import bind_feature_lookback
from ml4t.engineer.core.registry import FeatureMetadata, get_registry

# Type variable for function preservation
F = TypeVar("F", bound=Callable[..., Any])


def feature(
    *,
    name: str,
    category: Literal[
        "momentum",
        "trend",
        "volatility",
        "volume",
        "statistics",
        "math",
        "price_transform",
        "microstructure",
        "ml",
        "risk",
        "regime",
    ],
    description: str,
    lookback: int | str | Callable[..., int] | None = None,
    normalized: bool = False,
    value_range: tuple[float, float] | None = None,
    formula: str = "",
    ta_lib_compatible: bool = False,
    input_type: str = "close",
    output_type: str = "indicator",
    parameters: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
    references: list[str] | None = None,
    tags: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to register a feature with metadata.

    This decorator attaches metadata to a feature function and registers it
    in the global registry. The original function is returned unchanged, so
    there is zero runtime overhead.

    Parameters
    ----------
    name : str
        Unique feature identifier (e.g., "rsi", "macd")
    category : str
        Feature category (momentum, trend, volatility, etc.)
    description : str
        Brief description of what the feature computes
    lookback : int, str, callable, or None
        Lookback calculation for third-party features. Built-in features use
        the authoritative calculations in ``core.lookbacks``.
    normalized : bool, default False
        Whether feature is stationary (range-bound or returns-based)
        - True: Bounded oscillators (RSI 0-100), returns, ratios
        - False: Price-following (SMA, EMA), cumulative (OBV)
    value_range : tuple[float, float] or None
        Expected output range if bounded (e.g., (0, 100) for RSI)
    formula : str, default ""
        Mathematical formula or algorithm description
    ta_lib_compatible : bool, default False
        Whether feature matches TA-Lib output
    input_type : str, default "close"
        Expected input data type (OHLCV, close, returns, etc.)
    output_type : str, default "indicator"
        Output data type (indicator, signal, label)
    parameters : dict, optional
        Default parameters for the feature function
    dependencies : list[str], optional
        List of feature names this feature depends on
    references : list[str], optional
        Academic papers or documentation references
    tags : list[str], optional
        Additional searchable tags

    Returns
    -------
    Callable
        Decorator function that registers the feature

    Examples
    --------
    >>> @feature(
    ...     name="price_change",
    ...     category="momentum",
    ...     description="One-period price change",
    ...     lookback=1,
    ...     normalized=True,
    ...     formula="change = close - close.shift(1)",
    ... )
    ... def price_change(close):
    ...     return close.diff()

    Notes
    -----
    Classification Guidelines:

    **normalized = True**:
    - Bounded oscillators: RSI (0-100), Stochastic (0-100), Williams %R (-100, 0)
    - Returns: ROC, momentum, percent change
    - Ratios: MFI, CCI (approximately bounded)
    - Normalized: Any feature explicitly normalized to [0, 1] or [-1, 1]

    **normalized = False**:
    - Price-following: SMA, EMA, Bollinger Bands (follow price level)
    - Cumulative: OBV, A/D Line (accumulate over time)
    - Volatility in price units: ATR (scales with price)

    **lookback Guidelines for third-party features**:
    - The value is the zero-based index of the first structurally usable output.
    - Fixed period: Use int (e.g., 0 for an instantaneous feature).
    - Parameter-dependent: Use a parameter name or callable.

    **value_range Guidelines**:
    - Strict bounds: (0, 100) for RSI, Stochastic
    - Symmetric: (-1, 1) for correlations, (-100, 0) for Williams %R
    - Theoretical bounds: (0, float('inf')) for positive-only metrics
    - None: For unbounded indicators
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        declared_parameters = dict(parameters or {})
        unknown_parameters = set(declared_parameters) - set(signature.parameters)
        if unknown_parameters:
            raise TypeError(
                f"Feature '{name}' declares unknown parameters: {sorted(unknown_parameters)}"
            )

        default_parameters: dict[str, Any] = {}
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in COLUMN_ARG_MAP:
                continue
            if parameter.default is not inspect.Parameter.empty:
                default_parameters[parameter_name] = parameter.default

        for parameter_name, value in declared_parameters.items():
            parameter = signature.parameters[parameter_name]
            if parameter.default is not inspect.Parameter.empty and parameter.default != value:
                raise TypeError(
                    f"Feature '{name}' metadata default for '{parameter_name}' is {value!r}, "
                    f"but the function default is {parameter.default!r}"
                )
            default_parameters[parameter_name] = value

        # Validate stationary features should have value_range for ML users
        if normalized and value_range is None:
            import warnings

            warnings.warn(
                f"Feature '{name}': normalized=True but value_range is None. "
                f"ML users need value ranges for normalization. Consider specifying value_range.",
                UserWarning,
                stacklevel=3,
            )

        # --- End Validation ---

        lookback_fn = bind_feature_lookback(name, default_parameters, lookback)
        default_lookback = lookback_fn()
        if (
            isinstance(default_lookback, bool)
            or not isinstance(default_lookback, int)
            or default_lookback < 0
        ):
            raise TypeError(
                f"Feature '{name}' produced invalid default lookback {default_lookback!r}"
            )

        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            func=func,
            category=category,
            description=description,
            formula=formula,
            normalized=normalized,
            lookback=lookback_fn,
            ta_lib_compatible=ta_lib_compatible,
            input_type=input_type,
            output_type=output_type,
            parameters=default_parameters,
            dependencies=dependencies or [],
            references=references or [],
            tags=tags or [],
            value_range=value_range,
        )

        # Bidirectional validation: bounded features should be stationary
        if value_range is not None and not normalized:
            import warnings

            min_val, max_val = value_range
            is_bounded = min_val != float("-inf") and max_val != float("inf")

            if is_bounded:
                warnings.warn(
                    f"Feature '{name}': has bounded value_range {value_range} but normalized=False. "
                    f"Bounded features should typically be marked as stationary for ML compatibility.",
                    UserWarning,
                    stacklevel=3,
                )

        # Register feature
        get_registry().register(metadata)

        # Return original function unchanged - ZERO overhead
        return func

    return decorator


__all__ = ["feature"]
