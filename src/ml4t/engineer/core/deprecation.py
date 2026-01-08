"""Parameter deprecation utilities for ML4T Engineer.

Provides utilities for deprecating parameters with clear migration paths.
"""

from __future__ import annotations

import warnings
from typing import TypeVar

T = TypeVar("T")


def resolve_period_parameter(
    *,
    period: int | None,
    timeperiod: int | None = None,
    window: int | None = None,
    lookback: int | None = None,
    default: int,
    func_name: str = "",
) -> int:
    """
    Resolve period parameter from multiple possible sources.

    This function handles backward compatibility for the standardization
    from `timeperiod`, `window`, and `lookback` to `period`.

    Parameters
    ----------
    period : int | None
        The preferred parameter name (new standard)
    timeperiod : int | None
        Deprecated TA-Lib style parameter
    window : int | None
        Deprecated rolling window style parameter
    lookback : int | None
        Deprecated lookback style parameter
    default : int
        Default value if none provided
    func_name : str
        Name of the function for warning messages

    Returns
    -------
    int
        The resolved period value

    Raises
    ------
    ValueError
        If multiple conflicting values are provided

    Examples
    --------
    >>> period = resolve_period_parameter(
    ...     period=None, timeperiod=14, default=10, func_name="rsi"
    ... )
    >>> # Issues deprecation warning and returns 14
    """
    # Collect all provided values
    provided: dict[str, int] = {}
    deprecated_params: dict[str, int] = {}

    if period is not None:
        provided["period"] = period
    if timeperiod is not None:
        deprecated_params["timeperiod"] = timeperiod
    if window is not None:
        deprecated_params["window"] = window
    if lookback is not None:
        deprecated_params["lookback"] = lookback

    # If period is provided, use it (ignore deprecated)
    if period is not None:
        if deprecated_params:
            # Warn about redundant deprecated params
            deprecated_names = ", ".join(f"'{k}'" for k in deprecated_params)
            warnings.warn(
                f"Both 'period' and deprecated parameter(s) {deprecated_names} provided "
                f"to {func_name}. Using 'period={period}'.",
                DeprecationWarning,
                stacklevel=3,
            )
        return period

    # Check deprecated parameters
    if deprecated_params:
        if len(deprecated_params) > 1:
            raise ValueError(
                f"Multiple deprecated period parameters provided to {func_name}: "
                f"{list(deprecated_params.keys())}. Use 'period' instead."
            )

        # Get the single deprecated param
        param_name, param_value = next(iter(deprecated_params.items()))
        warnings.warn(
            f"'{param_name}' parameter is deprecated, use 'period' instead in {func_name}.",
            DeprecationWarning,
            stacklevel=3,
        )
        return param_value

    # No period provided, use default
    return default


def deprecated_parameter(
    old_name: str,
    new_name: str = "period",
    *,
    func_name: str = "",
) -> None:
    """
    Issue a deprecation warning for a renamed parameter.

    Parameters
    ----------
    old_name : str
        The deprecated parameter name
    new_name : str
        The new parameter name (default: "period")
    func_name : str
        Name of the function for the warning message
    """
    warnings.warn(
        f"'{old_name}' parameter is deprecated, use '{new_name}' instead"
        + (f" in {func_name}" if func_name else "")
        + ".",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = ["resolve_period_parameter", "deprecated_parameter"]
