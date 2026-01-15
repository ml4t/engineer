"""Input validation utilities for ml4t-engineer.

This module provides common validation patterns for feature engineering functions.
"""

from typing import Any

import polars as pl


def validate_window(window: int, min_window: int = 1, name: str = "window") -> None:
    """Validate rolling window parameter.

    Parameters
    ----------
    window : int
        Window size to validate
    min_window : int, default 1
        Minimum allowed window size
    name : str, default "window"
        Parameter name for error messages

    Raises
    ------
    ValueError
        If window is not positive or less than min_window
    TypeError
        If window is not an integer
    """
    if not isinstance(window, int):
        raise TypeError(f"{name} must be an integer, got {type(window).__name__}")
    if window < min_window:
        raise ValueError(f"{name} must be at least {min_window}, got {window}")


def validate_period(period: int, min_period: int = 1, name: str = "period") -> None:
    """Validate period parameter (alias for validate_window).

    Parameters
    ----------
    period : int
        Period to validate
    min_period : int, default 1
        Minimum allowed period
    name : str, default "period"
        Parameter name for error messages
    """
    validate_window(period, min_period, name)


def validate_threshold(
    threshold: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    name: str = "threshold",
) -> None:
    """Validate threshold parameter.

    Parameters
    ----------
    threshold : float
        Threshold value to validate
    min_val : float, default 0.0
        Minimum allowed value
    max_val : float, default 1.0
        Maximum allowed value
    name : str, default "threshold"
        Parameter name for error messages

    Raises
    ------
    ValueError
        If threshold is outside allowed range
    TypeError
        If threshold is not numeric
    """
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(threshold).__name__}")
    if not min_val <= threshold <= max_val:
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got {threshold}",
        )


def validate_lag(lag: int, max_lag: int | None = None, name: str = "lag") -> None:
    """Validate lag parameter.

    Parameters
    ----------
    lag : int
        Lag value to validate
    max_lag : int, optional
        Maximum allowed lag
    name : str, default "lag"
        Parameter name for error messages

    Raises
    ------
    ValueError
        If lag is negative or exceeds max_lag
    TypeError
        If lag is not an integer
    """
    if not isinstance(lag, int):
        raise TypeError(f"{name} must be an integer, got {type(lag).__name__}")
    if lag < 0:
        raise ValueError(f"{name} must be non-negative, got {lag}")
    if max_lag is not None and lag > max_lag:
        raise ValueError(f"{name} must not exceed {max_lag}, got {lag}")


def validate_list_length(
    lst: list[Any],
    min_length: int = 1,
    max_length: int | None = None,
    name: str = "list",
) -> None:
    """Validate list length.

    Parameters
    ----------
    lst : list
        List to validate
    min_length : int, default 1
        Minimum required length
    max_length : int, optional
        Maximum allowed length
    name : str, default "list"
        Parameter name for error messages

    Raises
    ------
    ValueError
        If list length is invalid
    TypeError
        If input is not a list
    """
    if not isinstance(lst, list):
        raise TypeError(f"{name} must be a list, got {type(lst).__name__}")
    if len(lst) < min_length:
        raise ValueError(
            f"{name} must have at least {min_length} elements, got {len(lst)}",
        )
    if max_length is not None and len(lst) > max_length:
        raise ValueError(
            f"{name} must have at most {max_length} elements, got {len(lst)}",
        )


def validate_column_exists(df: pl.DataFrame, column: str) -> None:
    """Validate that column exists in DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to check
    column : str
        Column name to validate

    Raises
    ------
    ValueError
        If column does not exist
    """
    if column not in df.columns:
        available = ", ".join(df.columns[:5])
        if len(df.columns) > 5:
            available += f", ... ({len(df.columns)} total)"
        raise ValueError(f"Column '{column}' not found. Available columns: {available}")


def validate_numeric_column(expr: pl.Expr, name: str = "column") -> None:
    """Validate that expression is numeric.

    Parameters
    ----------
    expr : pl.Expr
        Expression to validate
    name : str, default "column"
        Parameter name for error messages

    Note
    ----
    This is a placeholder as runtime type checking of expressions
    is limited. Consider adding schema validation at the DataFrame level.
    """
    # Limited validation possible at expression level


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that value is positive.

    Parameters
    ----------
    value : int or float
        Value to validate
    name : str
        Parameter name for error messages

    Raises
    ------
    ValueError
        If value is not positive
    TypeError
        If value is not numeric
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_probability(prob: float, name: str = "probability") -> None:
    """Validate probability value (0 to 1).

    Parameters
    ----------
    prob : float
        Probability to validate
    name : str, default "probability"
        Parameter name for error messages
    """
    validate_threshold(prob, 0.0, 1.0, name)


def validate_percentage(pct: float, name: str = "percentage") -> None:
    """Validate percentage value (0 to 100).

    Parameters
    ----------
    pct : float
        Percentage to validate
    name : str, default "percentage"
        Parameter name for error messages
    """
    validate_threshold(pct, 0.0, 100.0, name)
