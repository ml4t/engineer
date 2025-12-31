# mypy: disable-error-code="misc"
import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_positive,
)
from ml4t.engineer.logging import logged_feature


@logged_feature("cyclical_encode", warn_threshold_ms=100.0)
@feature(
    name="cyclical_encode",
    category="ml",
    description="Cyclical Encoding - sin/cos encoding for cyclical features",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def cyclical_encode(
    value: pl.Expr | str,
    period: float,
    name_prefix: str = "cyclical",
) -> dict[str, pl.Expr]:
    """Encode cyclical features using sine and cosine transformation.

    Converts cyclical features (time of day, day of week, etc.) into
    continuous representations that preserve cyclical relationships.

    Parameters
    ----------
    value : pl.Expr | str
        Column with cyclical close (e.g., hour, day, month)
    period : int | float
        Period of the cycle (e.g., 24 for hours, 7 for days)
    name_prefix : str, default "cyclical"
        Prefix for output column names

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary with sin and cos transformations

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not numeric or name_prefix is not a string

    Examples
    --------
    >>> # Encode hour of day
    >>> hour_features = cyclical_encode("hour", 24, "hour")
    >>> df.with_columns([
    ...     hour_features["hour_sin"],
    ...     hour_features["hour_cos"]
    ... ])
    """
    # Validate inputs
    validate_positive(period, name="period")
    if not isinstance(name_prefix, str):
        raise TypeError(
            f"name_prefix must be a string, got {type(name_prefix).__name__}",
        )

    value = pl.col(value) if isinstance(value, str) else value

    # Convert to radians
    radians = 2 * np.pi * value / period

    return {f"{name_prefix}_sin": radians.sin(), f"{name_prefix}_cos": radians.cos()}
