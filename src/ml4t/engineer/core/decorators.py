# mypy: disable-error-code="misc,arg-type,assignment"
"""Feature registration decorators.

Simple decorator-based registration for features with zero overhead.
Metadata is attached at import time, computation has no wrapper overhead.
"""

from collections.abc import Callable
from typing import Any, Literal, TypeVar

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
    lookback: int | str,
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
    lookback : int or str
        Lookback period. Use int for fixed (e.g., 14), or str for
        parameter-dependent (e.g., "period", "fast_period")
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
    ...     name="rsi",
    ...     category="momentum",
    ...     description="Relative Strength Index",
    ...     lookback=14,
    ...     normalized=True,
    ...     value_range=(0, 100),
    ...     formula="RSI = 100 - (100 / (1 + RS))",
    ...     ta_lib_compatible=True,
    ... )
    ... def rsi(values, period=14):
    ...     # Implementation
    ...     return result

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

    **lookback Guidelines**:
    - Fixed period: Use int (e.g., 14 for RSI)
    - Parameter-dependent: Use parameter name (e.g., "period", "window")
    - Multiple parameters: Use primary parameter or "max(fast, slow)"
    - No lookback: Use 0

    **value_range Guidelines**:
    - Strict bounds: (0, 100) for RSI, Stochastic
    - Symmetric: (-1, 1) for correlations, (-100, 0) for Williams %R
    - Theoretical bounds: (0, float('inf')) for positive-only metrics
    - None: For unbounded indicators
    """

    def decorator(func: F) -> F:
        # --- Validation: Enforce metadata completeness ---

        # Validate lookback is meaningful
        if isinstance(lookback, str):
            # Parameter-dependent lookback must be non-empty
            if not lookback.strip():
                raise TypeError(
                    f"Feature '{name}': 'lookback' cannot be empty string. "
                    f"Use int for fixed period, parameter name for dynamic, or 1 for minimal lookback."
                )
        elif (
            isinstance(lookback, int)
            and lookback == 0
            and parameters
            and any(k in parameters for k in ["period", "window", "lookback", "windows"])
        ):
            # Integer lookback can be 0 only for instantaneous features (log returns, etc.)
            # but should be 1+ for any rolling window features
            # We allow 0 but warn if parameters suggest a window is used
            import warnings

            warnings.warn(
                f"Feature '{name}': lookback=0 but has period/window parameter. "
                f"Consider using lookback='period' or specifying the actual lookback.",
                UserWarning,
                stacklevel=3,
            )

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

        # Convert lookback to callable for consistent interface
        if isinstance(lookback, int):
            # Fixed lookback: always return the same value (capture by value)
            def lookback_fn(val: int = lookback, **_kwargs: Any) -> int:
                return val
        elif isinstance(lookback, str):
            # Parameter-dependent: extract from kwargs or use default from parameters
            param_name = lookback
            default_value = (parameters or {}).get(param_name, 1) if parameters else 1

            def lookback_fn(
                pname: str = param_name, default: int = default_value, **kwargs: Any
            ) -> int:
                return int(kwargs.get(pname, default))
        else:
            # Already callable, use as-is
            lookback_fn = lookback

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
            parameters=parameters or {},
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
