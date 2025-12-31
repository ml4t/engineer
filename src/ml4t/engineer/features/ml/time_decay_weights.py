import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="time_decay_weights",
    category="ml",
    description="Time Decay Weights - exponentially decaying weights",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def time_decay_weights(
    lookback: int,
    decay_type: str = "exponential",
    half_life: int | None = None,
) -> pl.Expr:
    """Create time decay weights for weighted features.

    Generates weights that decay over time, useful for giving more
    importance to recent observations.

    Parameters
    ----------
    lookback : int
        Number of periods to look back
    decay_type : str, default "exponential"
        Type of decay: "exponential", "linear", "sqrt"
    half_life : int, optional
        Half-life for exponential decay (default: lookback/3)

    Returns
    -------
    pl.Expr
        Weight expression

    Raises
    ------
    ValueError
        If lookback is not positive, decay_type is invalid, or half_life is not positive
    TypeError
        If lookback or half_life are not integers, or decay_type is not a string
    """
    # Validate inputs
    validate_window(lookback, min_window=1, name="lookback")
    if not isinstance(decay_type, str):
        raise TypeError(f"decay_type must be a string, got {type(decay_type).__name__}")
    if decay_type not in ["exponential", "linear", "sqrt"]:
        raise ValueError(
            f"Unknown decay_type: {decay_type}. Supported types: ['exponential', 'linear', 'sqrt']",
        )

    if half_life is not None:
        validate_window(half_life, min_window=1, name="half_life")

    if decay_type == "exponential":
        if half_life is None:
            half_life_val: int = lookback // 3
        else:
            half_life_val = half_life
        # Create exponential weights (older = lower weight, recent = higher weight)
        alpha = np.log(2) / half_life_val
        # Reverse order so index 0 (oldest) has lowest weight and index -1 (newest) has highest
        weights = np.exp(-alpha * (lookback - 1 - np.arange(lookback)))
    elif decay_type == "linear":
        # Linear decay: older observations get lower weights, recent get higher weights
        weights = np.linspace(0, 1, lookback)
    elif decay_type == "sqrt":
        # Square root decay: older observations get lower weights, recent get higher weights
        weights = np.sqrt(np.linspace(1, lookback, lookback) / lookback)
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    # Normalize weights
    weights = weights / weights.sum()

    # Convert to Polars expression
    return pl.lit(weights.tolist())
